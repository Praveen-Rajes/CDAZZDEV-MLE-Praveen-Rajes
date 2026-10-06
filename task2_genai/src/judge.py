# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'LLM-as-judge with strict schema and Pydantic validation, blind and pointwise, per plan Phase 7', Date: 2026-10-06
"""LLM-as-judge (qwen3.8-27b on Groq by default; a third model family).

- Pointwise and blind: one candidate per call, never the system name; items shuffled with a fixed seed.
- temperature 0, reasoning off, strict JSON schema JudgeScore, validated with Pydantic.
  Invalid replies are retried twice, then logged as excluded and counted.
- Resumable: (system, id) pairs already in reports/judge_scores.jsonl are skipped.
- Systems are judged in config order (base_zeroshot and ft first, few-shot only if quota remains).

CLI: python -m src.judge [--systems base_zeroshot ft] [--max-calls N]
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from .config import Config, load_config, load_text
from .schemas import JUDGE_CRITERIA, JudgeScore, _short_error, to_strict_json_schema
from .utils import append_jsonl, read_jsonl


def judge_messages(cfg: Config, scenario: str, reference: str, candidate: str) -> List[Dict[str, str]]:
    system = load_text(cfg.prompt_path("judge_system_prompt.md")).strip()
    user = (load_text(cfg.prompt_path("judge_user_template.md")).strip()
            .replace("{SCENARIO}", scenario).replace("{REFERENCE}", reference).replace("{CANDIDATE}", candidate))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _done_pairs(cfg: Config) -> set:
    return {(r["system"], r["id"]) for r in read_jsonl(cfg.report_path("judge_scores.jsonl"))}


def build_queue(cfg: Config, systems: List[str]) -> List[Dict[str, Any]]:
    """Work items in priority groups: the first two systems together (shuffled), then the rest."""
    from .infer import predictions_path
    from .split_format import scenario_of

    test = {r["id"]: r for r in read_jsonl(cfg.split_path("test"))}
    done = _done_pairs(cfg)
    rng = random.Random(cfg.judge.shuffle_seed)
    groups = [systems[:2], systems[2:]]
    queue: List[Dict[str, Any]] = []
    for group in groups:
        items = []
        for s in group:
            for p in read_jsonl(predictions_path(cfg, s)):
                if (s, p["id"]) in done or p["id"] not in test:
                    continue
                rec = test[p["id"]]
                items.append({"id": p["id"], "system": s, "scenario": scenario_of(rec),
                              "reference": rec["messages"][2]["content"], "candidate": p.get("raw_text", "")})
        rng.shuffle(items)
        queue += items
    return queue


def run_judge(cfg: Config, systems: Optional[List[str]] = None, max_calls: Optional[int] = None) -> Dict[str, Any]:
    from .llm_client import BudgetExhausted, LLMClient

    systems = systems or list(cfg.judge.systems)
    queue = build_queue(cfg, systems)
    if not queue:
        print("[judge] nothing to judge (all pairs done or no predictions)")
        return {"judged": 0}
    client = LLMClient(cfg, cfg.judge, purpose="judge")
    schema = to_strict_json_schema(JudgeScore)
    out_path = cfg.report_path("judge_scores.jsonl")
    print(f"[judge] model={cfg.judge.model} queue={len(queue)} used_today={client.tokens_used_today(cfg.judge.model)}")
    stats = {"ok": 0, "excluded": 0, "calls": 0}
    for k, item in enumerate(queue, 1):
        if max_calls is not None and stats["calls"] >= max_calls:
            break
        msgs = judge_messages(cfg, item["scenario"], item["reference"], item["candidate"])
        score, attempts, last_err, call_id = None, 0, None, None
        try:
            for attempts in range(1, cfg.judge.invalid_retries + 2):      # 1 try + N retries
                res = client.chat_json(msgs, schema, "judge_score")
                stats["calls"] += 1
                call_id = res.call_id
                if not res.ok:
                    last_err = f"{res.status}: {res.error}"
                    continue
                try:
                    score = JudgeScore.model_validate(res.data)
                    break
                except ValidationError as e:
                    last_err = f"validation: {_short_error(e)}"
        except BudgetExhausted as e:
            print(f"[judge] STOP: {e}")
            break
        row = {"id": item["id"], "system": item["system"], "judge_model": cfg.judge.model, "call_id": call_id,
               "attempts": attempts, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if score is not None:
            s = score.model_dump()
            row.update({"status": "ok", "scores": s,
                        "overall": round(sum(s[c] for c in JUDGE_CRITERIA) / len(JUDGE_CRITERIA), 3)})
            stats["ok"] += 1
        else:
            row.update({"status": "excluded", "error": last_err})
            stats["excluded"] += 1
        append_jsonl(out_path, row)
        if k % 10 == 0:
            print(f"[judge] {k}/{len(queue)} ok={stats['ok']} excluded={stats['excluded']}")
    print(f"[judge] done: {stats}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-as-judge over saved predictions")
    ap.add_argument("--systems", nargs="*", default=None)
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    print(json.dumps(run_judge(cfg, args.systems, args.max_calls)))


if __name__ == "__main__":
    main()
