# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'OpenAI-compatible JSON client with pacing, daily budget guard, retries and call logging per plan Phase 0', Date: 2026-10-06
"""OpenAI-compatible chat client for the teacher and the judge.

- Strict JSON schema via response_format (constrained decoding where supported).
- Rolling-minute pacing per model (RPM and TPM from config).
- Daily budget guard on uncached tokens, read back from the call log so it survives restarts.
- 429: honour retry-after; exponential backoff with jitter (tenacity), max 6 tries.
- 400: logged and returned as a failure object; the run never crashes on one bad call.
- Every call is logged (tokens, cached tokens, latency, status). Keys and prompts never are.
"""
from __future__ import annotations

import json
import os
import random
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from .config import Config, LLMRoleCfg
from .json_utils import extract_first_json
from .utils import append_jsonl, read_jsonl

MAX_TRIES = 6
DEFAULT_TOKEN_ESTIMATE = 2500   # used for pacing until the first real call reports usage


class BudgetExhausted(RuntimeError):
    """Raised before a call when today's uncached-token budget would be exceeded."""


@dataclass
class CallResult:
    ok: bool
    status: str                       # ok | api_error | parse_error
    call_id: str
    data: Optional[Dict[str, Any]] = None
    raw_text: Optional[str] = None
    error: Optional[str] = None
    latency_s: float = 0.0
    usage: Dict[str, int] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)


class RateLimiter:
    """Rolling 60-second window over requests and tokens."""

    def __init__(self, rpm: int, tpm: int):
        self.rpm = rpm
        self.tpm = tpm
        self.events: Deque[Tuple[float, int]] = deque()

    def _prune(self, now: float) -> None:
        while self.events and now - self.events[0][0] >= 60.0:
            self.events.popleft()

    def wait(self, est_tokens: int) -> float:
        """Sleep until one more request of `est_tokens` fits in the window. Returns seconds slept."""
        slept = 0.0
        while True:
            now = time.monotonic()
            self._prune(now)
            used = sum(t for _, t in self.events)
            if len(self.events) < self.rpm and (used + est_tokens <= self.tpm or not self.events):
                return slept
            pause = 60.0 - (now - self.events[0][0]) + 0.25
            time.sleep(max(pause, 0.25))
            slept += max(pause, 0.25)

    def record(self, tokens: int) -> None:
        self.events.append((time.monotonic(), tokens))


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _usage_dict(usage: Any) -> Dict[str, int]:
    if usage is None:
        return {}
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    cached = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = int(getattr(details, "cached_tokens", 0) or 0)
    reasoning = 0
    cdetails = getattr(usage, "completion_tokens_details", None)
    if cdetails is not None:
        reasoning = int(getattr(cdetails, "reasoning_tokens", 0) or 0)
    return {
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "uncached_tokens": prompt - cached + completion,
    }


class LLMClient:
    def __init__(self, cfg: Config, role: LLMRoleCfg, log_path: Optional[Path] = None, purpose: str = "teacher"):
        from openai import OpenAI

        self.cfg = cfg
        self.role = role
        self.purpose = purpose
        self.log_path = Path(log_path) if log_path else cfg.path("call_log")
        key_env = cfg.role_api_key_env(role)
        api_key = os.environ.get(key_env)
        if not api_key:
            raise RuntimeError(f"Environment variable {key_env} is not set. Add it to .env (local) or Colab Secrets.")
        # max_retries=0: retries are handled here so 429s honour retry-after and get logged.
        self.client = OpenAI(base_url=cfg.role_base_url(role), api_key=api_key,
                             timeout=cfg.provider.timeout_s, max_retries=0)
        self.limiters: Dict[str, RateLimiter] = {}
        self._used_today: Dict[str, int] = {}
        self._last_tokens: Dict[str, int] = {}

    # ---- budget -----------------------------------------------------------
    def tokens_used_today(self, model: str) -> int:
        """Uncached tokens used today (UTC) for `model`, read from the call log once, then tracked."""
        if model not in self._used_today:
            today = _today_utc()
            total = 0
            for row in read_jsonl(self.log_path):
                if row.get("model") == model and str(row.get("timestamp", "")).startswith(today):
                    total += int(row.get("uncached_tokens", 0) or 0)
            self._used_today[model] = total
        return self._used_today[model]

    def _limiter(self, model: str) -> RateLimiter:
        if model not in self.limiters:
            rl = self.role.rate_limits
            self.limiters[model] = RateLimiter(rl.rpm, rl.tpm)
        return self.limiters[model]

    # ---- logging ------------------------------------------------------------
    def _log(self, model: str, call_id: str, usage: Dict[str, int], latency: float, status: str,
             error: Optional[str], headers: Dict[str, str], attempt: int) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "purpose": self.purpose,
            "model": model,
            "call_id": call_id,
            "attempt": attempt,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "cached_tokens": usage.get("cached_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "reasoning_tokens": usage.get("reasoning_tokens", 0),
            "uncached_tokens": usage.get("uncached_tokens", 0),
            "latency_s": round(latency, 2),
            "status": status,
            "error": (error or "")[:300] or None,
            "ratelimit_remaining_tokens": headers.get("x-ratelimit-remaining-tokens"),
            "ratelimit_remaining_requests": headers.get("x-ratelimit-remaining-requests"),
        }
        append_jsonl(self.log_path, row)

    # ---- main call -----------------------------------------------------------
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        schema_name: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> CallResult:
        """One JSON-schema call. Returns parsed JSON plus usage; never raises on API errors
        except BudgetExhausted (raised before the call so the caller can stop cleanly)."""
        import openai

        model = model or self.role.model
        temperature = self.role.temperature if temperature is None else temperature
        max_tokens = max_tokens or self.role.max_tokens
        strict = self.role.strict_schema if strict is None else strict
        body = dict(self.role.extra_body)
        body.update(extra_body or {})
        call_id = uuid.uuid4().hex[:12]

        est = self._last_tokens.get(model, DEFAULT_TOKEN_ESTIMATE)
        budget = self.role.rate_limits.daily_token_budget
        if self.tokens_used_today(model) + est > budget:
            raise BudgetExhausted(
                f"{model}: {self.tokens_used_today(model)} uncached tokens used today; budget {budget}. Resume tomorrow.")

        request = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=self.role.top_p,
            max_tokens=max_tokens,
            response_format={"type": "json_schema",
                             "json_schema": {"name": schema_name, "strict": strict, "schema": schema}},
            extra_body=body,
        )

        limiter = self._limiter(model)
        transient = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError,
                     openai.InternalServerError)
        attempt_no = {"n": 0}

        def one_attempt() -> CallResult:
            """One HTTP call. Transient errors are logged then re-raised for tenacity to retry."""
            attempt_no["n"] += 1
            attempt = attempt_no["n"]
            limiter.wait(est)
            t0 = time.perf_counter()
            headers: Dict[str, str] = {}
            try:
                raw = self.client.chat.completions.with_raw_response.create(**request)
                latency = time.perf_counter() - t0
                headers = {k.lower(): v for k, v in raw.headers.items() if k.lower().startswith("x-ratelimit")}
                resp = raw.parse()
            except openai.BadRequestError as e:          # 400: schema or parameter problem, do not retry
                latency = time.perf_counter() - t0
                limiter.record(est)
                msg = _error_text(e)
                self._log(model, call_id, {}, latency, "api_error_400", msg, headers, attempt)
                return CallResult(ok=False, status="api_error", call_id=call_id, error=msg, latency_s=latency)
            except transient as e:
                latency = time.perf_counter() - t0
                limiter.record(est)
                status = "rate_limited_429" if isinstance(e, openai.RateLimitError) else "transient_error"
                self._log(model, call_id, {}, latency, status, _error_text(e), headers, attempt)
                raise
            except openai.APIStatusError as e:            # other 4xx: log and give up on this call
                latency = time.perf_counter() - t0
                limiter.record(est)
                msg = _error_text(e)
                self._log(model, call_id, {}, latency, f"api_error_{e.status_code}", msg, headers, attempt)
                return CallResult(ok=False, status="api_error", call_id=call_id, error=msg, latency_s=latency)

            usage = _usage_dict(resp.usage)
            limiter.record(usage.get("uncached_tokens", est))
            self._used_today[model] = self.tokens_used_today(model) + usage.get("uncached_tokens", 0)
            self._last_tokens[model] = max(usage.get("uncached_tokens", est), 500)
            text = resp.choices[0].message.content if resp.choices else None
            data = None
            if text:
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    data = extract_first_json(text)
            status = "ok" if isinstance(data, dict) else "parse_error"
            finish = resp.choices[0].finish_reason if resp.choices else None
            err = None if status == "ok" else f"unparseable reply (finish_reason={finish})"
            self._log(model, call_id, usage, latency, status, err, headers, attempt)
            return CallResult(ok=status == "ok", status=status, call_id=call_id, data=data if status == "ok" else None,
                              raw_text=text, error=err, latency_s=latency, usage=usage, headers=headers)

        retrying = Retrying(stop=stop_after_attempt(MAX_TRIES), wait=_wait_retry_after_or_backoff,
                            retry=retry_if_exception_type(transient), reraise=True)
        try:
            return retrying(one_attempt)
        except transient as e:
            return CallResult(ok=False, status="api_error", call_id=call_id,
                              error=f"gave up after {MAX_TRIES} tries: {_error_text(e)}")


def _error_text(e: Exception) -> str:
    """Short error description. The SDK never includes the API key in error messages."""
    return str(e)[:300]


_exp_jitter = wait_random_exponential(multiplier=1, max=60)


def _wait_retry_after_or_backoff(retry_state: Any) -> float:
    """Honour retry-after if the server sent one, else exponential backoff with jitter (cap 60 s)."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    resp = getattr(exc, "response", None)
    if resp is not None:
        ra = resp.headers.get("retry-after")
        if ra:
            try:
                return float(ra) + random.uniform(0, 1.0)
            except ValueError:
                pass
    return _exp_jitter(retry_state)
