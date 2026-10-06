# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'merge_and_unload in fp16, parity check, Hub push and model card per plan Phase 6 and Appendix H', Date: 2026-10-06
"""Merge the LoRA adapter into fp16 base weights, check parity, push to the Hub.

1. Free the training model.
2. Load the base in fp16 (not quantised) on the GPU (~6.5 GB for 3B).
3. PeftModel.from_pretrained(base, adapter_dir) -> merge_and_unload().
4. save_pretrained(out_dir, max_shard_size="2GB") (+ safe_serialization where the API still takes it);
   tokenizer saved with the pad token and pinned template.
5. Parity: validation loss and greedy outputs on 5 validation prompts for (a) 4-bit base + adapter
   and (b) merged fp16. The adapter was trained against NF4-dequantised weights but is merged into
   the original fp16 weights, so small drift is expected and reported.
6. Push merged model (public) with a model card; reload fresh from the Hub and run one example.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config, load_config, load_text
from .utils import (chat_ids, dtype_kwarg, free_memory, read_json, read_jsonl, setup_tokenizer, supported_kwargs,
                    write_json)


# ---------------------------------------------------------------------------
# Hub helpers
# ---------------------------------------------------------------------------
def hub_user(cfg: Config) -> str:
    if cfg.hub.username:
        return cfg.hub.username
    from huggingface_hub import HfApi

    return HfApi().whoami()["name"]


def repo_ids(cfg: Config) -> Dict[str, str]:
    user = hub_user(cfg)
    return {"adapter": f"{user}/{cfg.hub.adapter_repo}", "merged": f"{user}/{cfg.hub.merged_repo}"}


def upload_folder(folder: Path, repo_id: str, private: bool, message: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="model", commit_message=message)
    return f"https://huggingface.co/{repo_id}"


def push_adapter(cfg: Config, adapter_dir: Path) -> str:
    write_model_card(cfg, adapter_dir, kind="adapter")
    return upload_folder(adapter_dir, repo_ids(cfg)["adapter"], cfg.hub.private, "Upload LoRA adapter")


# ---------------------------------------------------------------------------
# Loss and greedy outputs (parity check)
# ---------------------------------------------------------------------------
def completion_loss(model: Any, tokenizer: Any, records: List[Dict[str, Any]]) -> float:
    """Token-weighted mean cross-entropy over the assistant tokens (prompt positions masked with -100)."""
    import torch

    total, count = 0.0, 0
    model.eval()
    for r in records:
        prompt = chat_ids(tokenizer, r["messages"][:2], add_generation_prompt=True)
        full = chat_ids(tokenizer, r["messages"], add_generation_prompt=False)
        ids = torch.tensor([full], device=model.device)
        labels = ids.clone()
        labels[0, : len(prompt)] = -100
        with torch.no_grad():
            loss = model(input_ids=ids, labels=labels).loss
        n = len(full) - len(prompt)
        total += float(loss) * n
        count += n
    return total / max(count, 1)


def greedy_outputs(cfg: Config, model: Any, tokenizer: Any, records: List[Dict[str, Any]]) -> List[str]:
    from .infer import generate_batch

    msgs = [r["messages"][:2] for r in records]
    return [o["raw_text"] for o in generate_batch(model, tokenizer, msgs, cfg.inference.max_new_tokens)]


def _verdicts(texts: List[str]) -> List[Optional[str]]:
    from .json_utils import extract_first_json

    out = []
    for t in texts:
        p = extract_first_json(t)
        out.append(p.get("verdict") if isinstance(p, dict) else None)
    return out


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
def load_tokenizer_for(cfg: Config, path: str, padding_side: str = "left"):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    return setup_tokenizer(tok, cfg.student.pad_token, cfg.student.pin_chat_template_date, padding_side)


def adapter_side_of_parity(cfg: Config, adapter_dir: Path, n_prompts: int = 5) -> Dict[str, Any]:
    """(a) 4-bit base + adapter: validation loss and greedy outputs. Frees GPU memory afterwards."""
    from peft import PeftModel

    from .train import load_base_4bit

    tok = load_tokenizer_for(cfg, str(adapter_dir))
    val = read_jsonl(cfg.split_path("val"))
    base = load_base_4bit(cfg, tok)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.config.use_cache = True
    model.eval()
    res = {"val_loss": completion_loss(model, tok, val), "outputs": greedy_outputs(cfg, model, tok, val[:n_prompts])}
    del model, base
    free_memory()
    return res


def merge_adapter(cfg: Config, adapter_dir: Path, out_dir: Path) -> Path:
    """fp16 base on GPU + adapter -> merge_and_unload -> save (sharded safetensors)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    free_memory()
    tok = load_tokenizer_for(cfg, str(adapter_dir))
    base = AutoModelForCausalLM.from_pretrained(cfg.student.model_id, device_map={"": 0},
                                                attn_implementation=cfg.student.attn_implementation,
                                                **dtype_kwarg(torch.float16))
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged.config.pad_token_id = tok.pad_token_id
    merged.generation_config.pad_token_id = tok.pad_token_id
    out_dir.mkdir(parents=True, exist_ok=True)
    save_kwargs = supported_kwargs(merged.save_pretrained, {"safe_serialization": True,
                                                            "max_shard_size": cfg.hub.max_shard_size})
    merged.save_pretrained(str(out_dir), **save_kwargs)
    tok.padding_side = "left"
    tok.save_pretrained(str(out_dir))
    print(f"[merge] merged model saved to {out_dir}: {sorted(p.name for p in out_dir.iterdir())}")
    del merged, model, base
    free_memory()
    return out_dir


def parity_check(cfg: Config, adapter_dir: Path, merged_dir: Path, n_prompts: int = 5) -> Dict[str, Any]:
    from .infer import load_model_fp16

    a = adapter_side_of_parity(cfg, adapter_dir, n_prompts)
    tok = load_tokenizer_for(cfg, str(merged_dir))
    model = load_model_fp16(str(merged_dir), cfg, tok)
    val = read_jsonl(cfg.split_path("val"))
    b = {"val_loss": completion_loss(model, tok, val), "outputs": greedy_outputs(cfg, model, tok, val[:n_prompts])}
    del model
    free_memory()
    va, vb = _verdicts(a["outputs"]), _verdicts(b["outputs"])
    res = {
        "val_loss_4bit_plus_adapter": round(a["val_loss"], 5),
        "val_loss_merged_fp16": round(b["val_loss"], 5),
        "val_loss_abs_diff": round(abs(a["val_loss"] - b["val_loss"]), 5),
        "n_prompts": n_prompts,
        "verdicts_4bit_plus_adapter": va,
        "verdicts_merged_fp16": vb,
        "verdict_agreement": sum(x == y for x, y in zip(va, vb)),
        "identical_text": sum(x == y for x, y in zip(a["outputs"], b["outputs"])),
        "ids": [r["id"] for r in val[:n_prompts]],
        "outputs_4bit_plus_adapter": a["outputs"],
        "outputs_merged_fp16": b["outputs"],
        "note": "LoRA was trained against NF4-dequantised weights and merged into the original fp16 weights, "
                "so a small loss difference and occasional wording changes are expected.",
    }
    write_json(cfg.report_path("merge_parity.json"), res)
    return res


def push_merged(cfg: Config, merged_dir: Path) -> str:
    write_model_card(cfg, merged_dir, kind="merged")
    return upload_folder(merged_dir, repo_ids(cfg)["merged"], cfg.hub.private, "Upload merged fp16 model")


def fresh_reload_check(cfg: Config, n: int = 1) -> List[Dict[str, Any]]:
    """Reload the merged model from the Hub (not the local folder) and run test examples: evaluate what you ship."""
    from .infer import load_generation_tokenizer, load_model_fp16, predict_records

    repo = repo_ids(cfg)["merged"]
    tok = load_generation_tokenizer(cfg, repo)
    model = load_model_fp16(repo, cfg, tok)
    rows = predict_records(cfg, model, tok, read_jsonl(cfg.split_path("test"))[:n], "ft_hub_reload", progress=False)
    del model
    free_memory()
    return rows


# ---------------------------------------------------------------------------
# Model card (Appendix H)
# ---------------------------------------------------------------------------
def _maybe(path: Path) -> Optional[Any]:
    return read_json(path) if path.exists() else None


def write_model_card(cfg: Config, out_dir: Path, kind: str = "merged") -> str:
    from .metrics import comparison_table

    sp = load_text(cfg.prompt_path("student_system_prompt.md")).strip()
    split = _maybe(cfg.report_path("split_sizes.json"))
    train_sum = _maybe(cfg.report_path("training_summary.json"))
    evalr = _maybe(cfg.report_path("eval_results.json"))
    review = _maybe(cfg.report_path("manual_review_summary.json"))
    try:
        ids = repo_ids(cfg)
    except Exception:
        ids = {"adapter": f"<hf_user>/{cfg.hub.adapter_repo}", "merged": f"<hf_user>/{cfg.hub.merged_repo}"}
    repo = ids["merged"] if kind == "merged" else ids["adapter"]
    t, lc, q = cfg.training, cfg.lora, cfg.quantization

    data_line = (f"{split['total']} synthetic examples (train {split['sizes']['train']}, val {split['sizes']['val']}, "
                 f"test {split['sizes']['test']}), stratified by verdict." if split else "See the data card in the GitHub repo.")
    chosen = train_sum["chosen"] if train_sum else None
    hp_rows = [
        ("Quantisation", f"4-bit {q.bnb_4bit_quant_type.upper()}, double quant {q.bnb_4bit_use_double_quant}, compute {q.bnb_4bit_compute_dtype}"),
        ("LoRA", f"r={lc.r}, alpha={lc.alpha}, dropout={chosen['lora_dropout'] if chosen else lc.dropout}, targets {', '.join(lc.target_modules)}"),
        ("Learning rate", f"{chosen['learning_rate'] if chosen else t.learning_rate} ({t.lr_scheduler_type}, warmup {chosen['warmup_steps'] if chosen else 'computed'} steps)"),
        ("Epochs", str(chosen["epochs"] if chosen else t.num_train_epochs)),
        ("Batch", f"{t.per_device_train_batch_size} x grad accum {t.gradient_accumulation_steps} = {t.per_device_train_batch_size * t.gradient_accumulation_steps}"),
        ("Max length", str(chosen["max_length"] if chosen else "data-driven")),
        ("Optimiser", f"{t.optim}, weight decay {t.weight_decay}, max grad norm {t.max_grad_norm}"),
        ("Precision", f"fp16={t.fp16}, bf16={t.bf16} (Colab T4)"),
        ("Loss", "completion only (prompt tokens masked)"),
    ]
    lines = [
        "---",
        "license: llama3.2",
        f"base_model: {cfg.student.official_model_id}",
        "language:\n- en",
        "library_name: " + ("transformers" if kind == "merged" else "peft"),
        "pipeline_tag: text-generation",
        "tags:\n- llama\n- qlora\n- data-protection\n- compliance\n- synthetic-data\n- sri-lanka",
        "---",
        "",
        f"# {repo.split('/')[-1]}",
        "",
        "**Built with Llama.**",
        "",
        "## Model summary",
        "",
        f"{'Merged fp16 model' if kind == 'merged' else 'LoRA adapter'} fine-tuned with QLoRA from "
        f"`{cfg.student.official_model_id}` (weights loaded from the ungated mirror `{cfg.student.model_id}`, identical weights). "
        "Task: PDPA compliance triage. Given a short internal request from an employee of a fictional Sri Lankan bank "
        "(\"the Bank\"), the model returns one JSON object with a rationale, the policy principles involved (P01 to P16), "
        "issues, missing information, a verdict (COMPLIANT, NON_COMPLIANT, NEEDS_MORE_INFO), a risk level and required actions.",
        "",
        "## Intended use and out-of-scope use",
        "",
        "- Intended: a research and assessment demo of domain fine-tuning and honest evaluation.",
        "- **Not legal advice.** The rules come from a fictional, simplified policy inspired by the structure of Sri Lanka's "
        "Personal Data Protection Act No. 9 of 2022 (as amended). It is not a statement of the law.",
        "- Out of scope: real customer decisions, real compliance sign-off, any use without human review.",
        "",
        "## Training data",
        "",
        f"Trained on synthetic data only. Teacher model `{cfg.teacher.model}` generated scenarios and answers from stratified seeds; "
        "items passed strict schema and consistency validation, leakage, forbidden-content and near-duplicate filters, "
        f"and a 30-item human audit. {data_line}",
        "",
        "## Training procedure",
        "",
        "| Setting | Value |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in hp_rows],
        "",
    ]
    if train_sum:
        lines += ["Per-epoch losses (epoch 0 = before training):", "", "| Epoch | Train loss | Validation loss |", "|---|---|---|"]
        lines += [f"| {r['epoch']} | {r['train_loss']} | {r['val_loss']} |" for r in train_sum["table"]]
        lines += [""]
    lines += ["## Evaluation", ""]
    if evalr:
        lines += [f"Test set: {evalr['test_size']} held-out synthetic items. Greedy decoding. "
                  f"Judge: `{cfg.judge.model}` (different model family from teacher and student).", "",
                  comparison_table(evalr), ""]
    else:
        lines += ["Results are added after evaluation (see the GitHub repo).", ""]
    if review:
        ft = review.get("systems", {}).get("ft")
        if ft and ft["hallucination_rate"]["rate"] is not None:
            h = ft["hallucination_rate"]
            lines += [f"Manual review: {ft['n_reviewed']} fine-tuned outputs labelled blind; hallucination rate "
                      f"{100 * h['rate']:.1f}% (Wilson 95% CI {100 * h['wilson95_low']:.1f} to {100 * h['wilson95_high']:.1f}%).", ""]
    lines += [
        "## Limitations and biases",
        "",
        "- Labels come from a single teacher model; the test references share its biases.",
        "- Synthetic distribution: real requests are messier.",
        "- Small test set, so confidence intervals are wide.",
        "- English only. Fictional policy, not law.",
        "",
        "## Licence",
        "",
        "Llama 3.2 Community License. Built with Llama.",
        "",
        "## How to use",
        "",
        "```python",
        "import torch",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        f'repo = "{ids["merged"]}"',
        "tok = AutoTokenizer.from_pretrained(repo)",
        'model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float16, device_map="auto")',
        f"system = {json.dumps(sp)}",
        'scenario = "Subject: ..."  # the internal request',
        'messages = [{"role": "system", "content": system}, {"role": "user", "content": "Internal request:\\n" + scenario}]',
        "text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)",
        "inputs = tok(text, return_tensors='pt', add_special_tokens=False).to(model.device)",
        "out = model.generate(**inputs, max_new_tokens=512, do_sample=False)",
        "print(tok.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True))",
        "```",
        "",
    ]
    card = "\n".join(lines)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text(card, encoding="utf-8")
    return card


def push_model_card(cfg: Config, kind: str = "merged") -> str:
    """Re-render the card with the latest results and upload README.md only."""
    import tempfile

    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory() as d:
        write_model_card(cfg, Path(d), kind=kind)
        repo = repo_ids(cfg)[kind]
        HfApi().upload_file(path_or_fileobj=str(Path(d) / "README.md"), path_in_repo="README.md", repo_id=repo,
                            commit_message="Update model card with evaluation results")
    return f"https://huggingface.co/{repo}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge, parity check, push")
    ap.add_argument("--adapter-dir", required=False, help="default: chosen run from reports/training_summary.json")
    ap.add_argument("--skip-push", action="store_true")
    ap.add_argument("--card-only", action="store_true", help="re-render and upload the model card only")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.card_only:
        print(push_model_card(cfg))
        return
    adapter_dir = Path(args.adapter_dir or read_json(cfg.report_path("training_summary.json"))["chosen"]["adapter_dir"])
    merged_dir = cfg.work_dir() / "merged" / cfg.hub.merged_repo
    merge_adapter(cfg, adapter_dir, merged_dir)
    print(json.dumps({k: v for k, v in parity_check(cfg, adapter_dir, merged_dir).items() if not k.startswith("outputs")}, indent=2))
    if not args.skip_push:
        print(push_adapter(cfg, adapter_dir))
        print(push_merged(cfg, merged_dir))
        print(fresh_reload_check(cfg)[0]["raw_text"])


if __name__ == "__main__":
    main()
