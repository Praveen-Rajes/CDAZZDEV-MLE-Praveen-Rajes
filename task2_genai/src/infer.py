# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'Batched greedy generation with per-token logprobs and perplexity per plan Phase 7', Date: 2026-10-06
"""Batched greedy generation for every evaluated system.

Per example we save: id, system, raw text, parsed JSON (or null), parse_ok, schema_ok,
generated token count, mean token log-prob and perplexity (compute_transition_scores with
output_scores=True, return_dict_in_generate=True), finish reason and latency.

CLI:
    python -m src.infer --system base_zeroshot [--split test]
    python -m src.infer --system base_fewshot
    python -m src.infer --system ft --model <hf_user>/Llama-3.2-3B-PDPA-Triage
"""
from __future__ import annotations

import argparse
import copy
import math
import time
from typing import Any, Callable, Dict, List, Optional

from .config import Config, load_config
from .json_utils import extract_first_json
from .schemas import Verdict, strict_violations, validate_answer
from .utils import chat_text, dtype_kwarg, read_jsonl, setup_tokenizer, write_json, write_jsonl

SYSTEMS = ("base_zeroshot", "base_fewshot", "ft")
FEWSHOT_ORDER = (Verdict.NON_COMPLIANT.value, Verdict.COMPLIANT.value, Verdict.NEEDS_MORE_INFO.value)


def predictions_path(cfg: Config, system: str, split: str = "test"):
    suffix = "" if split == "test" else f"_{split}"
    return cfg.report_path(f"predictions_{system}{suffix}.jsonl")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_generation_tokenizer(cfg: Config, model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    return setup_tokenizer(tok, cfg.student.pad_token, cfg.student.pin_chat_template_date, padding_side="left")


def load_model_fp16(model_id: str, cfg: Config, tokenizer: Any):
    """fp16 weights on GPU 0 (fp32 on CPU), eval mode, KV cache on."""
    import torch
    from transformers import AutoModelForCausalLM

    on_gpu = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": 0} if on_gpu else None,
        attn_implementation=cfg.student.attn_implementation,
        **dtype_kwarg(torch.float16 if on_gpu else torch.float32),
    )
    model.eval()
    model.config.use_cache = True
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
def select_fewshot(records: List[Dict[str, Any]], per_verdict: int = 1) -> List[Dict[str, Any]]:
    """Fixed few-shot examples: per verdict, prefer 'straightforward' items, then the shortest scenario, then id."""
    chosen = []
    for verdict in FEWSHOT_ORDER:
        pool = [r for r in records if r["meta"]["verdict"] == verdict]
        pool.sort(key=lambda r: (r["meta"].get("difficulty") != "straightforward",
                                 len(r["messages"][1]["content"].split()), r["id"]))
        chosen += pool[:per_verdict]
    return chosen


def fewshot_turns(fewshot: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    turns: List[Dict[str, str]] = []
    for r in fewshot:
        turns += [r["messages"][1], r["messages"][2]]
    return turns


def build_messages(record: Dict[str, Any], fewshot: Optional[List[Dict[str, str]]] = None,
                   user_content: Optional[str] = None) -> List[Dict[str, str]]:
    """[system] + optional prior user/assistant turns + [user]."""
    system = record["messages"][0]
    user = {"role": "user", "content": user_content if user_content is not None else record["messages"][1]["content"]}
    return [system, *(fewshot or []), user]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _eos_ids(model: Any, tokenizer: Any) -> List[int]:
    eos = getattr(model.generation_config, "eos_token_id", None)
    ids = list(eos) if isinstance(eos, (list, tuple)) else ([eos] if eos is not None else [])
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ids:
        ids.append(tokenizer.eos_token_id)
    return ids


def generate_batch(model: Any, tokenizer: Any, message_lists: List[List[Dict[str, str]]],
                   max_new_tokens: int) -> List[Dict[str, Any]]:
    """Greedy decoding with left padding. Returns text plus token log-prob statistics per sequence."""
    import torch

    tokenizer.padding_side = "left"
    texts = [chat_text(tokenizer, m, add_generation_prompt=True) for m in message_lists]
    enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    eos_ids = _eos_ids(model, tokenizer)
    gen_cfg = copy.deepcopy(model.generation_config)
    gen_cfg.do_sample = False
    gen_cfg.temperature = 1.0      # neutral values: greedy ignores them; avoids sampling-parameter warnings
    gen_cfg.top_p = 1.0
    gen_cfg.max_new_tokens = max_new_tokens
    gen_cfg.pad_token_id = tokenizer.pad_token_id
    gen_cfg.eos_token_id = eos_ids
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc, generation_config=gen_cfg, output_scores=True, return_dict_in_generate=True)
    latency = time.perf_counter() - t0
    # Log-prob of each chosen token under the model (normalize_logits=True applies log-softmax).
    scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True).float().cpu()
    gen = out.sequences[:, enc["input_ids"].shape[1]:].cpu()
    results = []
    for i in range(gen.shape[0]):
        toks = gen[i].tolist()
        end = next((j for j, t in enumerate(toks) if t in eos_ids), None)
        n_valid = (end + 1) if end is not None else len(toks)
        # Positions after the end of the sequence are padding with meaningless scores: excluded.
        lp = scores[i, :n_valid]
        lp = lp[torch.isfinite(lp)]
        mean_lp = float(lp.mean()) if lp.numel() else float("nan")
        results.append({
            "raw_text": tokenizer.decode(toks[: (end if end is not None else len(toks))], skip_special_tokens=True).strip(),
            "generated_tokens": int(n_valid),
            "finish_reason": "eos" if end is not None else "length",
            "mean_logprob": round(mean_lp, 5),
            "perplexity": round(math.exp(-mean_lp), 5) if math.isfinite(mean_lp) else None,
            "latency_s": round(latency / gen.shape[0], 3),
        })
    del out, scores, enc
    return results


def parse_prediction(raw_text: str, rationale_max_words: int = 90) -> Dict[str, Any]:
    parsed = extract_first_json(raw_text)
    model, err = validate_answer(parsed, mode="scoring") if parsed is not None else (None, "no_json")
    viol = strict_violations(parsed, rationale_max_words=rationale_max_words) if model is not None else ["unparseable_or_invalid_schema"]
    return {
        "parsed": model.model_dump(mode="json") if model is not None else parsed,
        "parse_ok": parsed is not None,
        "schema_ok": model is not None,
        "schema_error": err,
        "strict_violations": viol,
    }


def predict_records(cfg: Config, model: Any, tokenizer: Any, records: List[Dict[str, Any]], system: str,
                    fewshot: Optional[List[Dict[str, str]]] = None,
                    user_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
                    batch_size: Optional[int] = None, max_new_tokens: Optional[int] = None,
                    progress: bool = True) -> List[Dict[str, Any]]:
    batch_size = batch_size or cfg.inference.batch_size
    max_new_tokens = max_new_tokens or cfg.inference.max_new_tokens
    rows: List[Dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        msgs = [build_messages(r, fewshot, user_fn(r) if user_fn else None) for r in chunk]
        outs = generate_batch(model, tokenizer, msgs, max_new_tokens)
        for r, o in zip(chunk, outs):
            row = {"id": r["id"], "system": system, **o}
            row.update(parse_prediction(o["raw_text"], cfg.metrics.rationale_max_words_scoring))
            rows.append(row)
        if progress:
            print(f"[infer] {system}: {min(start + batch_size, len(records))}/{len(records)}")
    return rows


def run_system(cfg: Config, system: str, model: Any, tokenizer: Any, split: str = "test",
               force: bool = False) -> List[Dict[str, Any]]:
    """Predict a whole split for one system and save it. Reuses the saved file if it is complete."""
    records = read_jsonl(cfg.split_path(split))
    path = predictions_path(cfg, system, split)
    existing = read_jsonl(path)
    if not force and existing and {r["id"] for r in existing} == {r["id"] for r in records}:
        print(f"[infer] {system}/{split}: reusing {path} ({len(existing)} rows)")
        return existing
    fewshot = None
    if system == "base_fewshot":
        shots = select_fewshot(read_jsonl(cfg.split_path("train")), cfg.inference.fewshot_per_verdict)
        write_json(cfg.report_path("fewshot_ids.json"), {"ids": [r["id"] for r in shots],
                                                         "verdicts": [r["meta"]["verdict"] for r in shots]})
        print(f"[infer] few-shot examples: {[r['id'] for r in shots]}")
        fewshot = fewshot_turns(shots)
    rows = predict_records(cfg, model, tokenizer, records, system, fewshot=fewshot)
    write_jsonl(path, rows)
    ok = sum(r["parse_ok"] for r in rows)
    print(f"[infer] {system}/{split}: saved {len(rows)} rows, JSON parse ok {ok}/{len(rows)} -> {path}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Batched generation for one system")
    ap.add_argument("--system", choices=SYSTEMS, required=True)
    ap.add_argument("--model", default=None, help="model id or path (default: base for base_*, required for ft)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    model_id = args.model or cfg.student.model_id
    if args.system == "ft" and not args.model:
        raise SystemExit("--model is required for --system ft (Hub id or merged folder)")
    tok = load_generation_tokenizer(cfg, model_id)
    model = load_model_fp16(model_id, cfg, tok)
    run_system(cfg, args.system, model, tok, split=args.split, force=args.force)


if __name__ == "__main__":
    main()
