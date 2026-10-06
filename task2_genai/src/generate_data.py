# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'Resumable teacher data generation CLI with per-item validation per plan Phase 2', Date: 2026-10-06
"""Teacher generation (resumable).

CLI:
    python -m src.generate_data --model openai/gpt-oss-120b --batch-size 4 --max-calls 120
    python -m src.generate_data --render-only            # write the rendered teacher prompt, count tokens
    python -m src.generate_data --summary                # print the generation log summary, no API calls

- Static system message: rendered teacher prompt (Appendix C + rulebook). Byte-identical on
  every call so the provider caches the prefix.
- Dynamic user message: teacher user template with the batch's seeds as JSON.
- Each returned item is validated at once (strict TriageAnswer + consistency + seed checks).
  Pass -> raw/generations.jsonl, fail -> raw/rejected.jsonl with the reason. Append-only,
  flushed after every call, so a crash or quota stop loses nothing.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import ValidationError

from .config import Config, load_config, load_text
from .schemas import TeacherBatch, TeacherItem, _short_error, to_strict_json_schema, word_count
from .utils import append_jsonl, read_jsonl


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------
def render_teacher_system_prompt(cfg: Config) -> str:
    template = load_text(cfg.prompt_path("teacher_system_prompt.md"))
    rulebook = load_text(cfg.path("rulebook")).strip()
    return template.replace("{RULEBOOK}", rulebook)


def save_rendered_teacher_prompt(cfg: Config) -> str:
    text = render_teacher_system_prompt(cfg)
    out = cfg.prompt_path("teacher_system_prompt.rendered.md")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return str(out)


def count_tokens(text: str) -> Tuple[int, str]:
    """Token count with tiktoken o200k_base (the gpt-oss tokenizer family). Falls back to a word estimate."""
    try:
        import tiktoken

        return len(tiktoken.get_encoding("o200k_base").encode(text)), "tiktoken o200k_base"
    except Exception:  # tiktoken missing or encoding download blocked
        return int(round(len(text.split()) * 1.33)), "estimate (words x 1.33)"


def build_user_message(cfg: Config, seeds: List[Dict[str, str]]) -> str:
    template = load_text(cfg.prompt_path("teacher_user_template.md"))
    seeds_json = "[\n" + ",\n".join(json.dumps(s, ensure_ascii=False) for s in seeds) + "\n]"
    return template.replace("{SEEDS_JSON}", seeds_json).strip()


# ---------------------------------------------------------------------------
# Per-item checks
# ---------------------------------------------------------------------------
def length_bounds(cfg: Config, bucket: str) -> Tuple[float, float]:
    lo, hi = cfg.seeds.length_words[bucket]
    tol = cfg.seeds.length_tolerance
    return lo * (1 - tol), hi * (1 + tol)


def check_item(item: Any, seed: Dict[str, str], cfg: Config) -> Tuple[bool, str, Optional[TeacherItem]]:
    """Validate one teacher item against the schema, the consistency rules and its seed."""
    if not isinstance(item, dict):
        return False, "schema: item is not an object", None
    try:
        parsed = TeacherItem.model_validate(item, context={"mode": "strict"})
    except ValidationError as e:
        msg = _short_error(e)
        kind = "consistency" if "Value error" in msg else "schema"
        return False, f"{kind}: {msg}", None
    if parsed.seed_id != seed["seed_id"]:
        return False, f"seed_id_mismatch: got {parsed.seed_id}", None
    if parsed.answer.verdict.value != seed["target_verdict"]:
        return False, f"verdict_mismatch: got {parsed.answer.verdict.value}, want {seed['target_verdict']}", None
    lo, hi = length_bounds(cfg, seed["length_bucket"])
    n_words = word_count(parsed.scenario)
    if not (lo <= n_words <= hi):
        return False, f"length_out_of_bucket: {n_words} words, {seed['length_bucket']} allows {lo:.0f}-{hi:.0f}", None
    return True, "ok", parsed


def reason_code(reason: str) -> str:
    """Short category for summaries: text before the first colon."""
    return reason.split(":", 1)[0].strip()


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------
def processed_seed_ids(cfg: Config, retry_rejected: bool = False) -> Set[str]:
    accepted = {r["seed_id"] for r in read_jsonl(cfg.path("generations"))}
    if retry_rejected:
        return accepted
    rejected = {r["seed_id"] for r in read_jsonl(cfg.path("rejected"))}
    return accepted | rejected


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def generation_log_summary(cfg: Config) -> Dict[str, Any]:
    """Summary of everything generated so far, read from the files only (no API calls)."""
    acc = read_jsonl(cfg.path("generations"))
    rej = read_jsonl(cfg.path("rejected"))
    calls = read_jsonl(cfg.path("call_log"))
    teacher_calls = [c for c in calls if c.get("purpose") == "teacher"]
    ok_calls = [c for c in teacher_calls if c.get("status") in ("ok", "parse_error")]
    prompt = sum(c.get("prompt_tokens", 0) for c in ok_calls)
    cached = sum(c.get("cached_tokens", 0) for c in ok_calls)
    return {
        "accepted": len(acc),
        "rejected": len(rej),
        "accepted_by_teacher": dict(Counter(r.get("teacher_model") for r in acc)),
        "rejected_by_reason": dict(Counter(reason_code(r.get("reason", "")) for r in rej).most_common()),
        "api_calls_total": len(teacher_calls),
        "api_calls_by_status": dict(Counter(c.get("status") for c in teacher_calls)),
        "calls_with_cached_tokens": sum(1 for c in ok_calls if c.get("cached_tokens", 0) > 0),
        "successful_calls": len(ok_calls),
        "cache_hit_rate": round(cached / prompt, 3) if prompt else None,
        "uncached_tokens_total": sum(c.get("uncached_tokens", 0) for c in teacher_calls),
        "completion_tokens_total": sum(c.get("completion_tokens", 0) for c in teacher_calls),
    }


def _print_running(stats: Counter, reasons: Counter, usage: Dict[str, int], calls: int) -> None:
    hit = usage["cached"] / usage["prompt"] if usage["prompt"] else 0.0
    print(f"[gen] calls={calls} accepted={stats['accepted']} rejected={stats['rejected']} "
          f"failed_calls={stats['failed_calls']} uncached_tokens={usage['uncached']} cache_hit_rate={hit:.2f}")
    if reasons:
        print(f"      rejected by reason: {dict(reasons.most_common())}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_generation(
    cfg: Config,
    model: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_calls: Optional[int] = None,
    retry_rejected: bool = False,
    seed_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate for pending seeds until max_calls, the daily budget, or the fallback cap stops it."""
    from .llm_client import BudgetExhausted, LLMClient

    model = model or cfg.teacher.model
    batch_size = batch_size or cfg.generation.batch_size
    max_calls = cfg.generation.max_calls if max_calls is None else max_calls
    is_fallback = model == cfg.teacher_fallback.model and model != cfg.teacher.model
    role = cfg.teacher_fallback if is_fallback else cfg.teacher

    seeds = read_jsonl(cfg.path("seeds"))
    if not seeds:
        raise FileNotFoundError(f"No seeds at {cfg.path('seeds')}. Run: python -m src.seeds")
    done = processed_seed_ids(cfg, retry_rejected=retry_rejected)
    pending = [s for s in seeds if s["seed_id"] not in done]
    if seed_ids is not None:
        wanted = set(seed_ids)
        pending = [s for s in pending if s["seed_id"] in wanted]

    cap = None
    if is_fallback:
        cap = math.floor(cfg.teacher_fallback.max_share * len(seeds))
        already = sum(1 for r in read_jsonl(cfg.path("generations")) if r.get("teacher_model") == model)
        print(f"[gen] fallback teacher {model}: {already} accepted so far, cap {cap} ({cfg.teacher_fallback.max_share:.0%} of seeds)")
        if already >= cap:
            print("[gen] fallback cap reached; nothing to do")
            return {"calls": 0, "accepted": 0, "rejected": 0, "stopped": "fallback_cap"}
        cap -= already

    system_prompt = render_teacher_system_prompt(cfg)   # identical bytes every call -> cached prefix
    schema = to_strict_json_schema(TeacherBatch)
    client = LLMClient(cfg, role, purpose="teacher")

    print(f"[gen] model={model} pending_seeds={len(pending)} batch_size={batch_size} max_calls={max_calls} "
          f"used_today={client.tokens_used_today(model)} budget={role.rate_limits.daily_token_budget}")

    stats: Counter = Counter()
    reasons: Counter = Counter()
    usage = {"prompt": 0, "cached": 0, "uncached": 0}
    calls = 0
    stopped = "no_pending_seeds"
    for start in range(0, len(pending), batch_size):
        if calls >= max_calls:
            stopped = "max_calls"
            break
        if cap is not None and stats["accepted"] >= cap:
            stopped = "fallback_cap"
            break
        batch = pending[start : start + batch_size]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_message(cfg, batch)},
        ]
        try:
            res = client.chat_json(messages, schema, "teacher_batch", model=model)
        except BudgetExhausted as e:
            print(f"[gen] STOP: {e}")
            stopped = "daily_budget"
            break
        calls += 1
        usage["prompt"] += res.usage.get("prompt_tokens", 0)
        usage["cached"] += res.usage.get("cached_tokens", 0)
        usage["uncached"] += res.usage.get("uncached_tokens", 0)
        if not res.ok:
            # Seeds stay pending (not written to rejected) and are retried on the next run.
            stats["failed_calls"] += 1
            print(f"[gen] call {res.call_id} failed: {res.status} {res.error}")
        else:
            items = res.data.get("items", []) if isinstance(res.data, dict) else []
            by_id = {it.get("seed_id"): it for it in items if isinstance(it, dict)}
            for seed in batch:
                item = by_id.get(seed["seed_id"])
                if item is None:
                    ok, reason, parsed = False, "missing_in_batch: teacher returned no item for this seed", None
                else:
                    ok, reason, parsed = check_item(item, seed, cfg)
                if ok and parsed is not None:
                    append_jsonl(cfg.path("generations"), {
                        "seed_id": seed["seed_id"],
                        "seed": seed,
                        "scenario": parsed.scenario,
                        "answer": parsed.answer.model_dump(mode="json"),
                        "teacher_model": model,
                        "call_id": res.call_id,
                        "timestamp": _now(),
                    })
                    stats["accepted"] += 1
                else:
                    append_jsonl(cfg.path("rejected"), {
                        "seed_id": seed["seed_id"],
                        "seed": seed,
                        "reason": reason,
                        "item": item,
                        "teacher_model": model,
                        "call_id": res.call_id,
                        "timestamp": _now(),
                    })
                    stats["rejected"] += 1
                    reasons[reason_code(reason)] += 1
        if calls % cfg.generation.summary_every == 0:
            _print_running(stats, reasons, usage, calls)

    _print_running(stats, reasons, usage, calls)
    if calls >= 2 and usage["prompt"] and usage["cached"] == 0:
        print("[gen] WARNING: cached_tokens is zero on every call. The system prefix is not being cached; "
              "check that the system prompt bytes are stable before spending more quota.")
    return {"calls": calls, "accepted": stats["accepted"], "rejected": stats["rejected"],
            "failed_calls": stats["failed_calls"], "rejected_by_reason": dict(reasons),
            "uncached_tokens": usage["uncached"], "cache_hit_rate": (usage["cached"] / usage["prompt"]) if usage["prompt"] else None,
            "stopped": stopped}


def items_table(rows: List[Dict[str, Any]], max_chars: int = 400) -> List[Dict[str, Any]]:
    """Readable rows for gate G2 (the notebook shows them as a DataFrame)."""
    out = []
    for r in rows:
        a = r["answer"]
        out.append({
            "seed_id": r["seed_id"],
            "target": r["seed"]["target_verdict"],
            "difficulty": r["seed"]["difficulty"],
            "verdict": a["verdict"],
            "risk": a["risk_level"],
            "principles": ",".join(a["principles"]),
            "words": word_count(r["scenario"]),
            "scenario": r["scenario"][:max_chars],
            "rationale": a["rationale"],
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Teacher data generation (resumable)")
    ap.add_argument("--model", default=None, help="teacher model id (default: config teacher.model)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--retry-rejected", action="store_true", help="re-attempt seeds that were rejected earlier")
    ap.add_argument("--render-only", action="store_true", help="write the rendered teacher prompt and exit")
    ap.add_argument("--summary", action="store_true", help="print the generation summary and exit")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    if args.render_only:
        path = save_rendered_teacher_prompt(cfg)
        n, method = count_tokens(render_teacher_system_prompt(cfg))
        print(f"Rendered teacher system prompt -> {path}")
        print(f"Token count ({method}): {n}")
        return
    if args.summary:
        print(json.dumps(generation_log_summary(cfg), indent=2))
        return
    result = run_generation(cfg, model=args.model, batch_size=args.batch_size, max_calls=args.max_calls,
                            retry_rejected=args.retry_rejected)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
