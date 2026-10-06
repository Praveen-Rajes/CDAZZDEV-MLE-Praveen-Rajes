# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'QLoRA training with TRL SFTTrainer, token stats, label-mask check, memory probe, overfit test and LR sweep per plan Phase 5', Date: 2026-10-06
"""QLoRA training for the PDPA triage student (TRL 1.x, PEFT, bitsandbytes).

Importable (the notebook calls these functions) and a CLI:
    python -m src.train --stage token-stats|label-mask|probe|overfit|sweep|all
    python -m src.train --smoke        # CPU plumbing test with SmolLM2-135M (see scripts/smoke_cpu_train.py)

Design notes (each is defended in the README):
- 4-bit NF4 base with double quantisation, fp16 compute (T4 has no native bf16).
- prepare_model_for_kbit_training + get_peft_model are called here, explicitly, so the trainable
  parameter count is printed before training and TRL does not repeat the k-bit preparation.
- Conversational prompt-completion dataset: prompt = [system, user], completion = [assistant].
  TRL then computes the loss on the completion only. No formatting_func is passed.
- Pad token is <|finetune_right_pad_id|>, never eos, so the model still learns to emit <|eot_id|>.
- warmup_steps is computed here from warmup_fraction (we do not rely on warmup_ratio).
- Finished runs write run_summary.json next to their adapter; re-running the notebook reuses them
  instead of retraining (and a disconnect resumes from the last epoch checkpoint on Drive).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import Config, load_config
from .json_utils import extract_first_json
from .schemas import validate_answer
from .utils import (chat_ids, dtype_kwarg, free_memory, read_json, read_jsonl, set_seed, setup_tokenizer,
                    write_json)


# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------
def gpu_check() -> Dict[str, Any]:
    """nvidia-smi, device name and bf16 support (expect bf16 unsupported on a T4)."""
    import torch

    info: Dict[str, Any] = {"cuda_available": torch.cuda.is_available()}
    try:
        info["nvidia_smi"] = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["nvidia_smi"] = "nvidia-smi not available"
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
        info["total_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    return info


# ---------------------------------------------------------------------------
# Tokenizer, model, LoRA
# ---------------------------------------------------------------------------
def load_tokenizer(cfg: Config, model_id: Optional[str] = None, padding_side: str = "right"):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id or cfg.student.model_id)
    return setup_tokenizer(tok, cfg.student.pad_token, cfg.student.pin_chat_template_date, padding_side)


def bnb_config(cfg: Config):
    import torch
    from transformers import BitsAndBytesConfig

    q = cfg.quantization
    return BitsAndBytesConfig(
        load_in_4bit=q.load_in_4bit,
        bnb_4bit_quant_type=q.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=q.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=getattr(torch, q.bnb_4bit_compute_dtype),
    )


def load_base_4bit(cfg: Config, tokenizer: Any, model_id: Optional[str] = None):
    """Base model in 4-bit NF4 on GPU 0."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id or cfg.student.model_id,
        quantization_config=bnb_config(cfg),
        device_map={"": 0},
        attn_implementation=cfg.student.attn_implementation,
        **dtype_kwarg(torch.float16),
    )
    model.config.use_cache = False               # incompatible with gradient checkpointing during training
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


def lora_config(cfg: Config, r: Optional[int] = None, alpha: Optional[int] = None, dropout: Optional[float] = None):
    from peft import LoraConfig

    lc = cfg.lora
    return LoraConfig(
        r=r or lc.r,
        lora_alpha=alpha or lc.alpha,
        lora_dropout=lc.dropout if dropout is None else dropout,
        target_modules=list(lc.target_modules),
        bias=lc.bias,
        task_type=lc.task_type,
    )


def build_peft_model(model: Any, lcfg: Any, quantized: bool = True, gradient_checkpointing: bool = True):
    from peft import get_peft_model, prepare_model_for_kbit_training

    if quantized:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    model = get_peft_model(model, lcfg)
    return model


def trainable_parameters(model: Any) -> Dict[str, Any]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": int(trainable), "total": int(total), "percent": round(100 * trainable / total, 4)}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_records(cfg: Config, split: str, n: Optional[int] = None, allow_fixture: bool = False) -> List[Dict[str, Any]]:
    """Chat records for a split. With allow_fixture, falls back to the test fixtures (smoke runs only)."""
    rows = read_jsonl(cfg.split_path(split))
    if not rows and allow_fixture:
        from .split_format import student_system_prompt, to_chat_record

        fixture = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sample_items.jsonl"
        items = read_jsonl(fixture)
        sp = student_system_prompt(cfg)
        rows = [to_chat_record(it, sp) for it in items]
        print(f"[data] {split} split not found; using {len(rows)} fixture records (smoke only)")
    if n is not None:
        rows = (rows * (n // max(len(rows), 1) + 1))[:n] if len(rows) < n and allow_fixture else rows[:n]
    return rows


def to_prompt_completion(record: Dict[str, Any]) -> Dict[str, Any]:
    msgs = record["messages"]
    return {"prompt": msgs[:2], "completion": [msgs[2]]}


def make_dataset(records: List[Dict[str, Any]]):
    from datasets import Dataset

    return Dataset.from_list([to_prompt_completion(r) for r in records])


def token_stats(cfg: Config, tokenizer: Any, splits: Tuple[str, ...] = ("train", "val"),
                records_by_split: Optional[Dict[str, List[Dict[str, Any]]]] = None,
                multiple: Optional[int] = None) -> Dict[str, Any]:
    """Full-sequence and completion token lengths with the real chat template.
    max_length = ceil(longest full sequence / multiple) * multiple, so nothing is truncated."""
    import numpy as np

    multiple = multiple or cfg.training.max_length_multiple
    out: Dict[str, Any] = {"tokenizer": getattr(tokenizer, "name_or_path", ""), "splits": {}}
    longest = 0
    for s in splits:
        recs = records_by_split[s] if records_by_split else read_jsonl(cfg.split_path(s))
        full = np.array([len(chat_ids(tokenizer, r["messages"], add_generation_prompt=False)) for r in recs])
        prompt = np.array([len(chat_ids(tokenizer, r["messages"][:2], add_generation_prompt=True)) for r in recs])
        comp = full - prompt
        out["splits"][s] = {
            "n": int(len(recs)),
            "full_max": int(full.max()), "full_p99": float(np.percentile(full, 99)), "full_mean": round(float(full.mean()), 1),
            "completion_max": int(comp.max()), "completion_p99": float(np.percentile(comp, 99)),
            "completion_mean": round(float(comp.mean()), 1), "prompt_max": int(prompt.max()),
        }
        longest = max(longest, int(full.max()))
    max_length = cfg.training.max_length or int(math.ceil(longest / multiple) * multiple)
    out["max_length"] = max_length
    out["max_length_rule"] = f"ceil(longest full sequence {longest} / {multiple}) * {multiple}"
    out["truncated_examples"] = 0
    for s in splits:
        recs = records_by_split[s] if records_by_split else read_jsonl(cfg.split_path(s))
        out["truncated_examples"] += sum(
            1 for r in recs if len(chat_ids(tokenizer, r["messages"], add_generation_prompt=False)) > max_length)
    assert out["truncated_examples"] == 0, "some examples would be truncated; raise max_length"
    return out


def steps_per_epoch(n_examples: int, batch_size: int, grad_accum: int) -> int:
    batches = math.ceil(n_examples / batch_size)
    return max(1, math.ceil(batches / grad_accum))


def warmup_steps_for(n_examples: int, batch_size: int, grad_accum: int, epochs: int, fraction: float) -> int:
    total = steps_per_epoch(n_examples, batch_size, grad_accum) * epochs
    return max(1, math.ceil(fraction * total)) if fraction > 0 else 0


# ---------------------------------------------------------------------------
# SFTConfig / SFTTrainer
# ---------------------------------------------------------------------------
def build_sft_config(cfg: Config, output_dir: Path, max_length: int, learning_rate: float, epochs: int,
                     warmup_steps: int, run_name: str, overrides: Optional[Dict[str, Any]] = None):
    """SFTConfig from config.yaml. Handles TRL/transformers renames and prints anything not accepted."""
    from trl import SFTConfig

    t = cfg.training
    kwargs: Dict[str, Any] = dict(
        output_dir=str(output_dir),
        run_name=run_name,
        learning_rate=learning_rate,
        lr_scheduler_type=t.lr_scheduler_type,
        warmup_steps=warmup_steps,
        num_train_epochs=epochs,
        per_device_train_batch_size=t.per_device_train_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        max_length=max_length,
        completion_only_loss=t.completion_only_loss,
        packing=t.packing,
        optim=t.optim,
        weight_decay=t.weight_decay,
        max_grad_norm=t.max_grad_norm,
        gradient_checkpointing=t.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=t.fp16,
        bf16=t.bf16,
        eval_strategy=t.eval_strategy,
        save_strategy=t.save_strategy,
        load_best_model_at_end=t.load_best_model_at_end,
        metric_for_best_model=t.metric_for_best_model,
        greater_is_better=False,
        save_total_limit=t.save_total_limit,
        logging_steps=t.logging_steps,
        seed=t.seed,
        data_seed=t.data_seed,
        group_by_length=t.group_by_length,
        torch_compile=t.torch_compile,
        report_to=t.report_to,
    )
    kwargs.update(overrides or {})
    fields = set(getattr(SFTConfig, "__dataclass_fields__", {}))
    if "max_length" not in fields and "max_seq_length" in fields:           # TRL < 0.20
        kwargs["max_seq_length"] = kwargs.pop("max_length")
    if "eval_strategy" not in fields and "evaluation_strategy" in fields:     # transformers < 4.41
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    dropped = sorted(k for k in kwargs if fields and k not in fields)
    if dropped:
        print(f"[compat] SFTConfig in this TRL version does not accept {dropped}; they are not passed")
        for k in dropped:
            kwargs.pop(k)
    return SFTConfig(**kwargs)


def build_trainer(model: Any, tokenizer: Any, args: Any, train_ds: Any, eval_ds: Any = None):
    from trl import SFTTrainer

    return SFTTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds, processing_class=tokenizer)


def _release(trainer: Any = None) -> None:
    if trainer is not None:
        acc = getattr(trainer, "accelerator", None)
        if acc is not None and hasattr(acc, "free_memory"):
            acc.free_memory()
    free_memory()


# ---------------------------------------------------------------------------
# Label-mask check (proves completion-only loss)
# ---------------------------------------------------------------------------
def label_mask_check(trainer: Any, tokenizer: Any) -> Dict[str, Any]:
    """Decode one training batch: full input and only the supervised (labels != -100) positions.
    The supervised text must be exactly the assistant JSON followed by the end-of-turn token."""
    batch = next(iter(trainer.get_train_dataloader()))
    ids = batch["input_ids"][0]
    labels = batch["labels"][0]
    attn = batch.get("attention_mask")
    full_ids = ids[attn[0].bool()] if attn is not None else ids
    sup_ids = ids[labels != -100]
    full_text = tokenizer.decode(full_ids, skip_special_tokens=False)
    sup_text = tokenizer.decode(sup_ids, skip_special_tokens=False)
    eot = "<|eot_id|>" if "<|eot_id|>" in tokenizer.get_vocab() else tokenizer.eos_token
    body = sup_text[: -len(eot)] if sup_text.endswith(eot) else sup_text
    parsed = extract_first_json(body)
    ok_json = parsed is not None and validate_answer(parsed, mode="strict")[0] is not None
    return {
        "full_text": full_text,
        "supervised_text": sup_text,
        "ends_with_eot": sup_text.endswith(eot),
        "supervised_is_valid_answer_json": ok_json,
        "supervised_starts_with_brace": body.lstrip().startswith("{"),
        "n_tokens": int(len(full_ids)),
        "n_supervised_tokens": int(len(sup_ids)),
        "passed": bool(sup_text.endswith(eot) and ok_json and body.lstrip().startswith("{")),
    }


# ---------------------------------------------------------------------------
# Per-epoch table and plots
# ---------------------------------------------------------------------------
def per_epoch_table(log_history: List[Dict[str, Any]], step0: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """Rows: epoch, mean train loss over that epoch's logged steps, validation loss. Epoch 0 = before training."""
    train_losses: Dict[int, List[float]] = defaultdict(list)
    val: Dict[int, float] = {}
    for e in log_history:
        if "epoch" not in e:
            continue
        if "loss" in e:
            train_losses[max(1, math.ceil(e["epoch"] - 1e-6))].append(float(e["loss"]))
        if "eval_loss" in e:
            val[int(round(e["epoch"]))] = float(e["eval_loss"])
    rows = []
    if step0:
        rows.append({"epoch": 0, "train_loss": step0.get("train_loss"), "val_loss": step0.get("eval_loss")})
    for ep in sorted(set(train_losses) | set(val)):
        if ep == 0:
            continue
        tl = train_losses.get(ep)
        rows.append({"epoch": ep, "train_loss": round(sum(tl) / len(tl), 4) if tl else None,
                     "val_loss": round(val[ep], 4) if ep in val else None})
    return rows


def is_strictly_decreasing(values: List[Optional[float]]) -> bool:
    vals = [v for v in values if v is not None]
    return len(vals) >= 2 and all(b < a for a, b in zip(vals, vals[1:]))


def select_run(runs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Choose the run whose validation loss decreases at every epoch and ends lowest. None if no run qualifies."""
    ok = [r for r in runs if r.get("monotonic")]
    return min(ok, key=lambda r: r["final_val_loss"]) if ok else None


def plot_loss_curves(table: List[Dict[str, Any]], path: Path, title: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ep_t = [r["epoch"] for r in table if r.get("train_loss") is not None]
    ax.plot(ep_t, [r["train_loss"] for r in table if r.get("train_loss") is not None], "o-", label="train loss (epoch mean)")
    ep_v = [r["epoch"] for r in table if r.get("val_loss") is not None]
    ax.plot(ep_v, [r["val_loss"] for r in table if r.get("val_loss") is not None], "s-", label="validation loss")
    ax.set_xlabel("epoch (0 = before training)")
    ax.set_ylabel("completion-only cross-entropy")
    ax.set_xticks(sorted(set(ep_t) | set(ep_v)))
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def inspect_label_mask(cfg: Config, tokenizer: Any, max_length: int, out_dir: Path) -> Dict[str, Any]:
    """Load the 4-bit model with fresh LoRA, build the real trainer and run label_mask_check."""
    train = make_dataset(load_records(cfg, "train"))
    model = build_peft_model(load_base_4bit(cfg, tokenizer), lora_config(cfg))
    args = build_sft_config(cfg, out_dir / "label_mask_check", max_length, cfg.training.learning_rate, 1, 0,
                            "label_mask_check", overrides={"eval_strategy": "no", "save_strategy": "no",
                                                           "load_best_model_at_end": False, "report_to": "none"})
    trainer = build_trainer(model, tokenizer, args, train)
    result = label_mask_check(trainer, tokenizer)
    _release(trainer)
    del trainer, model
    free_memory()
    return result


def memory_probe(cfg: Config, tokenizer: Any, max_length: int, out_dir: Path) -> Dict[str, Any]:
    """Two short runs on the longest training examples:
    A) batch 16, gradient checkpointing off (expected to OOM on a T4);
    B) the chosen config (batch 4, grad accumulation 4, checkpointing on).
    Results go to reports/oom_log.md, whatever happens."""
    import torch

    recs = load_records(cfg, "train")
    longest = sorted(recs, key=lambda r: -len(chat_ids(tokenizer, r["messages"], False)))
    results = []
    probes = [
        ("A_large_batch_no_checkpointing", cfg.memory_probe.per_device_train_batch_size, 1, cfg.memory_probe.gradient_checkpointing),
        ("B_chosen_config", cfg.training.per_device_train_batch_size, cfg.training.gradient_accumulation_steps,
         cfg.training.gradient_checkpointing),
    ]
    for name, bs, ga, gc_on in probes:
        n = bs * ga * cfg.memory_probe.steps
        ds = make_dataset(longest[:n])
        free_memory()
        torch.cuda.reset_peak_memory_stats()
        rec: Dict[str, Any] = {"probe": name, "per_device_train_batch_size": bs, "gradient_accumulation_steps": ga,
                               "gradient_checkpointing": gc_on, "max_length": max_length, "steps": cfg.memory_probe.steps}
        trainer = model = None
        t0 = time.time()
        try:
            model = build_peft_model(load_base_4bit(cfg, tokenizer), lora_config(cfg), gradient_checkpointing=gc_on)
            args = build_sft_config(cfg, out_dir / f"probe_{name}", max_length, cfg.training.learning_rate, 1, 0,
                                    f"probe_{name}", overrides={
                                        "per_device_train_batch_size": bs, "gradient_accumulation_steps": ga,
                                        "gradient_checkpointing": gc_on, "max_steps": cfg.memory_probe.steps,
                                        "eval_strategy": "no", "save_strategy": "no", "load_best_model_at_end": False,
                                        "logging_steps": 1, "report_to": "none"})
            trainer = build_trainer(model, tokenizer, args, ds)
            trainer.train()
            rec["outcome"] = "fits (no OOM)"
        except torch.cuda.OutOfMemoryError as e:
            rec["outcome"] = "CUDA out of memory"
            rec["error"] = str(e).split("\n")[0][:400]
        rec["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        rec["total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        rec["seconds"] = round(time.time() - t0, 1)
        _release(trainer)
        del trainer, model
        free_memory()
        shutil.rmtree(out_dir / f"probe_{name}", ignore_errors=True)
        results.append(rec)
        print(f"[probe] {rec}")
    write_oom_log(cfg, results)
    return {"probes": results}


def write_oom_log(cfg: Config, results: List[Dict[str, Any]]) -> None:
    a = results[0]
    lines = [
        "# Memory probe and OOM log",
        "",
        "Real experiment on the Colab GPU, run by `src.train.memory_probe` on the longest training examples. "
        "Recorded as it happened.",
        "",
        "| Probe | Batch | Grad accum | Grad checkpointing | max_length | Outcome | Peak allocated (GB) | GPU total (GB) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['probe']} | {r['per_device_train_batch_size']} | {r['gradient_accumulation_steps']} | "
                     f"{r['gradient_checkpointing']} | {r['max_length']} | {r['outcome']} | {r['peak_allocated_gb']} | {r['total_gb']} |")
    lines += [""]
    if a.get("error"):
        lines += ["Error message from probe A:", "", "```", a["error"], "```", ""]
    if a["outcome"].startswith("CUDA"):
        lines += ["Fix applied: per-device batch 4, gradient accumulation 4 (same effective batch of 16), "
                  "gradient checkpointing on (non-reentrant). Probe B shows the peak memory of that configuration."]
    else:
        lines += ["Probe A did not run out of memory on this GPU. The peak memory is recorded above. The training "
                  "config still uses batch 4 x accumulation 4 with checkpointing for headroom on longer batches."]
    path = cfg.report_path("oom_log.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(cfg.report_path("memory_probe.json"), results)


def overfit_test(cfg: Config, tokenizer: Any, max_length: int, out_dir: Path) -> Dict[str, Any]:
    """8 training examples, 60 steps, no eval. The loss must fall below target_loss or the pipeline is broken."""
    oc = cfg.overfit
    set_seed(cfg.project.seed)
    ds = make_dataset(load_records(cfg, "train", n=oc.n_examples))
    model = build_peft_model(load_base_4bit(cfg, tokenizer), lora_config(cfg, dropout=oc.lora_dropout))
    args = build_sft_config(cfg, out_dir / "overfit", max_length, oc.learning_rate, 1, 0, "overfit_test", overrides={
        "max_steps": oc.max_steps, "per_device_train_batch_size": 4, "gradient_accumulation_steps": 1,
        "eval_strategy": "no", "save_strategy": "no", "load_best_model_at_end": False, "logging_steps": 1,
        "lr_scheduler_type": "constant", "report_to": "none"})
    trainer = build_trainer(model, tokenizer, args, ds)
    trainer.train()
    losses = [(e["step"], float(e["loss"])) for e in trainer.state.log_history if "loss" in e]
    final = sum(l for _, l in losses[-5:]) / min(5, len(losses))
    result = {"n_examples": oc.n_examples, "steps": oc.max_steps, "learning_rate": oc.learning_rate,
              "lora_dropout": oc.lora_dropout, "lr_scheduler": "constant", "losses": losses,
              "final_loss_mean_last5": round(final, 4), "target": oc.target_loss, "passed": final < oc.target_loss}
    _release(trainer)
    del trainer, model
    free_memory()
    shutil.rmtree(out_dir / "overfit", ignore_errors=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot([s for s, _ in losses], [l for _, l in losses], "-", color="#4C72B0")
    ax.axhline(oc.target_loss, color="red", linestyle="--", label=f"target {oc.target_loss}")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("train loss (log scale)")
    ax.set_title(f"Overfit test: {oc.n_examples} examples, {oc.max_steps} steps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.figure_path("overfit_test.png"), dpi=130)
    plt.close(fig)
    write_json(cfg.report_path("overfit_test.json"), result)
    return result


def run_name_for(lr: float, epochs: int, dropout: float) -> str:
    return f"lr{lr:g}_ep{epochs}_do{dropout:g}"


def train_run(cfg: Config, tokenizer: Any, max_length: int, out_root: Path, learning_rate: float,
              epochs: Optional[int] = None, dropout: Optional[float] = None, reuse: bool = True) -> Dict[str, Any]:
    """One full QLoRA run with step-0 eval, per-epoch eval/save and a final adapter.
    If the run already finished (run_summary.json exists) and reuse=True, the saved summary is returned."""
    import torch

    epochs = epochs or cfg.training.num_train_epochs
    dropout = cfg.lora.dropout if dropout is None else dropout
    name = run_name_for(learning_rate, epochs, dropout)
    run_dir = out_root / name
    summary_path = run_dir / "run_summary.json"
    if reuse and summary_path.exists():
        print(f"[train] {name}: finished earlier, reusing {summary_path}")
        return read_json(summary_path)

    set_seed(cfg.project.seed)
    train_recs, val_recs = load_records(cfg, "train"), load_records(cfg, "val")
    train_ds, val_ds = make_dataset(train_recs), make_dataset(val_recs)
    t = cfg.training
    warmup = warmup_steps_for(len(train_recs), t.per_device_train_batch_size, t.gradient_accumulation_steps,
                              epochs, t.warmup_fraction)
    model = build_peft_model(load_base_4bit(cfg, tokenizer), lora_config(cfg, dropout=dropout),
                             gradient_checkpointing=t.gradient_checkpointing)
    params = trainable_parameters(model)
    print(f"[train] {name}: trainable params {params['trainable']:,} / {params['total']:,} ({params['percent']}%)")
    args = build_sft_config(cfg, run_dir, max_length, learning_rate, epochs, warmup, name)
    trainer = build_trainer(model, tokenizer, args, train_ds, val_ds)

    step0_path = run_dir / "step0.json"
    if step0_path.exists():
        step0 = read_json(step0_path)
    else:
        step0 = {}
        if t.step0_eval:
            step0["eval_loss"] = float(trainer.evaluate()["eval_loss"])
        if t.step0_train_eval:
            # trainer.train_dataset is the tokenized copy TRL built; the raw Dataset cannot be evaluated directly.
            step0["train_loss"] = float(trainer.evaluate(eval_dataset=trainer.train_dataset,
                                                         metric_key_prefix="train0")["train0_loss"])
        write_json(step0_path, step0)
    print(f"[train] {name}: epoch 0 {step0}")

    resume = any(run_dir.glob("checkpoint-*"))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train(resume_from_checkpoint=True if resume else None)
    wall = time.time() - t0
    peak = round(torch.cuda.max_memory_allocated() / 1024**3, 2)

    table = per_epoch_table(trainer.state.log_history, step0)
    adapter_dir = run_dir / "final_adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    with open(run_dir / "log_history.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=1)
    vals = [r["val_loss"] for r in table]
    summary = {
        "run_name": name, "learning_rate": learning_rate, "epochs": epochs, "lora_dropout": dropout,
        "warmup_steps": warmup, "max_length": max_length, "n_train": len(train_recs), "n_val": len(val_recs),
        "steps_per_epoch": steps_per_epoch(len(train_recs), t.per_device_train_batch_size, t.gradient_accumulation_steps),
        "trainable_params": params, "table": table, "monotonic": is_strictly_decreasing(vals),
        "final_val_loss": vals[-1], "best_val_loss": min(v for v in vals if v is not None),
        "best_checkpoint": trainer.state.best_model_checkpoint, "adapter_dir": str(adapter_dir),
        "wall_time_s": round(wall, 1), "peak_gpu_gb": peak, "gpu": torch.cuda.get_device_name(0),
    }
    write_json(summary_path, summary)
    _release(trainer)
    del trainer, model
    free_memory()
    return summary


def write_sweep_csv(cfg: Config, runs: List[Dict[str, Any]]) -> None:
    rows = []
    for r in runs:
        for row in r["table"]:
            rows.append({"run_name": r["run_name"], "learning_rate": r["learning_rate"], "epochs": r["epochs"],
                         "lora_dropout": r["lora_dropout"], "epoch": row["epoch"], "train_loss": row["train_loss"],
                         "val_loss": row["val_loss"], "monotonic_run": r["monotonic"],
                         "wall_time_s": r["wall_time_s"], "peak_gpu_gb": r["peak_gpu_gb"]})
    write_csv(cfg.report_path("sweep.csv"), rows)


def lr_sweep(cfg: Config, tokenizer: Any, max_length: int, out_root: Path) -> Dict[str, Any]:
    """LR sweep on validation loss only; fallbacks (2 epochs, then dropout 0.15) if no run decreases monotonically.
    Every run, including failures, is written to reports/sweep.csv."""
    runs = [train_run(cfg, tokenizer, max_length, out_root, lr) for lr in cfg.sweep.learning_rates]
    write_sweep_csv(cfg, runs)
    best = select_run(runs)
    notes = []
    if best is None:
        lr = min(runs, key=lambda r: r["final_val_loss"])["learning_rate"]
        notes.append(f"No sweep run decreased at every epoch; fallback 1: {cfg.sweep.fallback_epochs} epochs at lr {lr:g}")
        runs.append(train_run(cfg, tokenizer, max_length, out_root, lr, epochs=cfg.sweep.fallback_epochs))
        write_sweep_csv(cfg, runs)
        best = select_run(runs[-1:])
        if best is None:
            notes.append(f"Fallback 1 not monotonic; fallback 2: lora_dropout {cfg.sweep.fallback_dropout} at lr {lr:g}")
            runs.append(train_run(cfg, tokenizer, max_length, out_root, lr, dropout=cfg.sweep.fallback_dropout))
            write_sweep_csv(cfg, runs)
            best = select_run(runs[-1:])
    if best is None:
        best = min(runs, key=lambda r: r["final_val_loss"])
        notes.append("No run decreased at every epoch. Using the lowest final validation loss and reporting this plainly.")
    finalise_training(cfg, best, runs, notes)
    return {"runs": runs, "chosen": best, "notes": notes}


def finalise_training(cfg: Config, best: Dict[str, Any], runs: List[Dict[str, Any]], notes: List[str]) -> None:
    """The chosen sweep run IS the final run (no retraining for show): write its table, plot and summary."""
    write_csv(cfg.report_path("training_log.csv"), best["table"])
    plot_loss_curves(best["table"], cfg.figure_path("loss_curves.png"),
                     f"QLoRA {best['run_name']}: train vs validation loss")
    write_json(cfg.report_path("training_summary.json"), {
        "chosen_run": best["run_name"], "selection_rule": "strictly decreasing validation loss at every epoch, "
        "then lowest final validation loss (validation set only)",
        "notes": notes, "chosen": {k: v for k, v in best.items() if k != "table"}, "table": best["table"],
        "all_runs": [{k: r[k] for k in ("run_name", "learning_rate", "epochs", "lora_dropout", "monotonic",
                                        "final_val_loss", "wall_time_s", "peak_gpu_gb")} for r in runs],
    })


# ---------------------------------------------------------------------------
# Smoke run (CPU, tiny model, no quantisation)
# ---------------------------------------------------------------------------
def smoke_train(cfg: Config, out_dir: Path) -> Dict[str, Any]:
    """Same code path as train_run on CPU: SmolLM2-135M, no 4-bit (bitsandbytes needs CUDA), LoRA r=8,
    16 examples, 1 epoch, max_length 512. Quality is irrelevant; this catches plumbing bugs."""
    import torch
    from transformers import AutoModelForCausalLM

    sm = cfg.smoke
    set_seed(cfg.project.seed)
    tok = load_tokenizer(cfg, sm.model_id)
    train_recs = load_records(cfg, "train", n=sm.n_train, allow_fixture=True)
    val_recs = load_records(cfg, "val", n=sm.n_eval, allow_fixture=True)
    model = AutoModelForCausalLM.from_pretrained(sm.model_id, **dtype_kwarg(torch.float32))
    model.config.pad_token_id = tok.pad_token_id
    model = build_peft_model(model, lora_config(cfg, r=sm.lora_r, alpha=2 * sm.lora_r), quantized=False)
    params = trainable_parameters(model)
    args = build_sft_config(cfg, out_dir, sm.max_length, cfg.training.learning_rate, sm.epochs, 0, "smoke", overrides={
        "optim": "adamw_torch", "fp16": False, "bf16": False, "gradient_checkpointing": False,
        "per_device_train_batch_size": 4, "gradient_accumulation_steps": 1, "per_device_eval_batch_size": 4,
        "save_strategy": "no", "load_best_model_at_end": False, "logging_steps": 1, "report_to": "none",
        "use_cpu": not torch.cuda.is_available()})
    trainer = build_trainer(model, tok, args, make_dataset(train_recs), make_dataset(val_recs))
    mask = label_mask_check(trainer, tok)
    step0 = {"eval_loss": float(trainer.evaluate()["eval_loss"])}
    t0 = time.time()
    trainer.train()
    table = per_epoch_table(trainer.state.log_history, step0)
    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    result = {"model": sm.model_id, "trainable_params": params, "table": table, "wall_time_s": round(time.time() - t0, 1),
              "label_mask_supervised_tokens": mask["n_supervised_tokens"], "label_mask_ends_with_eos": mask["ends_with_eot"],
              "adapter_dir": str(adapter_dir)}
    _release(trainer)
    del trainer, model
    free_memory()
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="QLoRA training")
    ap.add_argument("--stage", choices=["token-stats", "label-mask", "probe", "overfit", "sweep", "all"], default="all")
    ap.add_argument("--smoke", action="store_true", help="CPU plumbing test (tiny model, no quantisation)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_root = cfg.work_dir() / "checkpoints"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        res = smoke_train(cfg, out_root / "smoke")
        print(json.dumps(res, indent=2))
        return

    tok = load_tokenizer(cfg)
    stats = token_stats(cfg, tok)
    write_json(cfg.report_path("token_stats.json"), stats)
    print(f"[train] token stats: {json.dumps(stats['splits'])}; max_length={stats['max_length']}")
    ml = stats["max_length"]
    if args.stage in ("label-mask", "all"):
        mask = inspect_label_mask(cfg, tok, ml, out_root)
        print(mask["supervised_text"])
        print(f"[train] label-mask check passed: {mask['passed']}")
    if args.stage in ("probe", "all"):
        memory_probe(cfg, tok, ml, out_root)
    if args.stage in ("overfit", "all"):
        res = overfit_test(cfg, tok, ml, out_root)
        print(f"[train] overfit test: final loss {res['final_loss_mean_last5']} passed={res['passed']}")
        if not res["passed"]:
            raise SystemExit("Overfit test failed: debug the pipeline before the sweep.")
    if args.stage in ("sweep", "all"):
        res = lr_sweep(cfg, tok, ml, out_root / "sweep")
        print(json.dumps({k: v for k, v in res["chosen"].items() if k != "table"}, indent=2))


if __name__ == "__main__":
    main()
