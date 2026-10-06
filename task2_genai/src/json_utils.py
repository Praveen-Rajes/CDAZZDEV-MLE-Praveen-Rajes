# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'Robust first-JSON-object extraction from model text per plan Phase 1', Date: 2026-10-06
"""Extract the first JSON object from free model text.

Handles code fences, leading chatter, trailing text and nested braces with a
balanced-brace scan that respects strings and escapes, then `json.loads`.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional


def _balanced_candidates(text: str):
    """Yield every substring that starts at a '{' and ends at its matching '}'."""
    n = len(text)
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, n):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : i + 1]
                    break
        start = text.find("{", start + 1)


def extract_first_json(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return the first parseable JSON object in `text`, or None.

    Tries the whole (fence-stripped) text first, then each balanced-brace candidate
    from left to right. Only dicts are returned; a top-level list is ignored.
    """
    if not text:
        return None
    stripped = text.strip()
    # Strip a surrounding ``` or ```json fence if present.
    if stripped.startswith("```"):
        first_nl = stripped.find("\n")
        stripped = stripped[first_nl + 1 :] if first_nl != -1 else stripped[3:]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
        stripped = stripped.strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    for cand in _balanced_candidates(text):
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None
