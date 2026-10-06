# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'Provider check: list models, tiny strict-schema call to teacher and judge, print usage and rate-limit headers per plan Phase 0', Date: 2026-10-06
"""Gate G1: confirm the provider, models, strict JSON schema and reasoning settings work.

- Lists models from /models and confirms the teacher and judge ids are served.
- Sends one tiny strict-schema request to the teacher and the judge with the reasoning settings
  from config.yaml, prints the parsed JSON, usage and the x-ratelimit-* headers.
- Sends one JudgeScore-schema request to the judge (strict mode on non-gpt-oss models is the most
  likely parameter to be rejected). If any call returns a 400, fix config.yaml, not the code.

Usage: python scripts/provider_check.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT))

from src.config import load_config  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402
from src.schemas import JudgeScore, to_strict_json_schema  # noqa: E402

TINY_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}, "echo": {"type": "string"}},
    "required": ["ok", "echo"],
    "additionalProperties": False,
}


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(TASK_ROOT.parent / ".env")
        load_dotenv()
    except ImportError:
        pass
    cfg = load_config()
    failures = 0

    from openai import OpenAI

    key_env = cfg.provider.api_key_env
    if not os.environ.get(key_env):
        print(f"{key_env} is not set. Add it to .env (local) or Colab Secrets, then re-run.")
        return 1
    client = OpenAI(base_url=cfg.provider.base_url, api_key=os.environ[key_env])
    served = sorted(m.id for m in client.models.list().data)
    print(f"Models served at {cfg.provider.base_url} ({len(served)}):")
    for m in served:
        print(f"  {m}")
    for role_name, role in (("teacher", cfg.teacher), ("judge", cfg.judge)):
        if cfg.role_base_url(role) == cfg.provider.base_url:
            status = "available" if role.model in served else "NOT SERVED - pick a fallback in config.yaml"
            print(f"{role_name}: {role.model} -> {status}")

    checks = [("teacher", cfg.teacher, TINY_SCHEMA, "provider_check_tiny"),
              ("judge", cfg.judge, TINY_SCHEMA, "provider_check_tiny"),
              ("judge (JudgeScore schema)", cfg.judge, to_strict_json_schema(JudgeScore), "judge_score")]
    for label, role, schema, name in checks:
        llm = LLMClient(cfg, role, purpose="provider_check")
        if name == "judge_score":
            msgs = [{"role": "system", "content": "Score the candidate from 1 to 5 on each criterion. Return JSON only."},
                    {"role": "user", "content": "REQUEST: test\nREFERENCE ANSWER: {}\nCANDIDATE ANSWER: {}"}]
        else:
            msgs = [{"role": "user", "content": 'Return JSON with ok=true and echo="ping".'}]
        res = llm.chat_json(msgs, schema, name, max_tokens=400 if name == "judge_score" else 200)
        print(f"\n== {label}: model={role.model} extra_body={json.dumps(role.extra_body)} strict={role.strict_schema}")
        print(f"   status={res.status} latency={res.latency_s:.2f}s")
        print(f"   parsed={json.dumps(res.data)}")
        print(f"   usage={json.dumps(res.usage)}")
        print(f"   rate-limit headers={json.dumps(res.headers)}")
        if not res.ok:
            failures += 1
            print(f"   ERROR: {res.error}")
    print("\nAll checks passed." if failures == 0 else f"\n{failures} check(s) failed. Adjust config.yaml (model ids, "
          "strict_schema, extra_body reasoning settings) following the provider's docs, then re-run.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
