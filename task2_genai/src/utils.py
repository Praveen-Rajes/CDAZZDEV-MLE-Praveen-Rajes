# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'Seeding, JSONL IO, timers, GPU memory report and transformers version helpers per plan Section 4', Date: 2026-10-06
"""Small shared helpers. Heavy libraries (torch, transformers) are imported lazily."""
from __future__ import annotations

import inspect
import json
import logging
import os
import random
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if installed) torch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# JSON / JSONL IO
# ---------------------------------------------------------------------------
def read_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path | str, rows: Iterable[Dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: Path | str, row: Dict[str, Any]) -> None:
    """Append one row and flush immediately, so a crash loses nothing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
        f.write("\n")


def read_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")


# ---------------------------------------------------------------------------
# Timers and GPU memory
# ---------------------------------------------------------------------------
@contextmanager
def timer(label: str, store: Optional[Dict[str, float]] = None) -> Iterator[None]:
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    if store is not None:
        store[label] = dt
    print(f"[timer] {label}: {dt:.1f} s")


def gpu_memory_report(reset_peak: bool = False) -> Dict[str, float]:
    """Current and peak CUDA memory in GB (empty dict on CPU)."""
    try:
        import torch
    except ImportError:
        return {}
    if not torch.cuda.is_available():
        return {}
    gb = 1024**3
    rep = {
        "allocated_gb": round(torch.cuda.memory_allocated() / gb, 2),
        "reserved_gb": round(torch.cuda.memory_reserved() / gb, 2),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / gb, 2),
        "total_gb": round(torch.cuda.get_device_properties(0).total_memory / gb, 2),
    }
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()
    return rep


def free_memory() -> None:
    """Collect garbage and empty the CUDA cache. Callers `del` their own references first."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# transformers version traps (plan Section 5.1)
# ---------------------------------------------------------------------------
def dtype_kwarg(dtype: Any) -> Dict[str, Any]:
    """Newer transformers uses `dtype=` in from_pretrained; older versions use `torch_dtype=`."""
    import transformers
    from packaging.version import Version

    if Version(transformers.__version__) >= Version("4.56.0"):
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


def cast_float_params(model: Any, from_dtype: Any, to_dtype: Any, trainable_only: bool = False) -> int:
    """Cast parameters of `from_dtype` (None = any non-fp32 float) to `to_dtype`. 4-bit weights (uint8) are untouched."""
    import torch

    n = 0
    for p in model.parameters():
        if trainable_only and not p.requires_grad:
            continue
        if not torch.is_floating_point(p):
            continue
        if (from_dtype is None and p.dtype != to_dtype) or p.dtype == from_dtype:
            p.data = p.data.to(to_dtype)
            n += 1
    return n


def supported_kwargs(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs that `fn` accepts by name. Prints anything dropped (no silent defaults)."""
    params = inspect.signature(fn).parameters
    kept = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(kept))
    if dropped:
        print(f"[compat] {getattr(fn, '__qualname__', fn)} does not accept {dropped}; not passed")
    return kept


def library_versions() -> Dict[str, str]:
    """Versions of the libraries that matter for reproducibility."""
    from importlib.metadata import PackageNotFoundError, version

    names = ["torch", "transformers", "trl", "peft", "bitsandbytes", "accelerate", "datasets",
             "rouge_score", "bert_score", "chromadb", "sentence-transformers", "openai", "pydantic"]
    out = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            out[n] = "not installed"
    return out


# ---------------------------------------------------------------------------
# Tokenizer helpers shared by train, infer and merge
# ---------------------------------------------------------------------------
def pin_chat_template_date(tokenizer: Any, date_string: Optional[str]) -> bool:
    """Replace the Llama 3.2 template's strftime_now call with a fixed date.

    The stock template prints "Today Date: <today>" in every system header, so a prompt
    built on training day differs from one built on evaluation day. Pinning the date makes
    train and inference prompts byte-identical. Returns True if the template was changed.
    """
    if not date_string or not getattr(tokenizer, "chat_template", None):
        return False
    tpl = tokenizer.chat_template
    if not isinstance(tpl, str):
        return False
    pinned, n = re.subn(r"""strftime_now\(\s*["']%d %b %Y["']\s*\)""", json.dumps(date_string), tpl)
    if n:
        tokenizer.chat_template = pinned
    return n > 0


def setup_tokenizer(tokenizer: Any, pad_token: Optional[str], date_string: Optional[str], padding_side: str = "right") -> Any:
    """Set a pad token that is NOT eos, pin the template date and set the padding side."""
    if pad_token and pad_token in tokenizer.get_vocab():
        tokenizer.pad_token = pad_token
    elif tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        # Only for non-Llama models (for example the SmolLM2 smoke test): use unk if it exists.
        if tokenizer.unk_token is not None and tokenizer.unk_token_id != tokenizer.eos_token_id:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            print("[tokenizer] WARNING: no dedicated pad token; falling back to eos (smoke runs only)")
            tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        print("[tokenizer] WARNING: pad token equals eos")
    pin_chat_template_date(tokenizer, date_string)
    tokenizer.padding_side = padding_side
    return tokenizer


def chat_text(tokenizer: Any, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    """Render messages to a string. tokenize=False keeps behaviour identical across
    transformers v4 and v5 (v5 returns a dict when tokenize=True)."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def chat_ids(tokenizer: Any, messages: List[Dict[str, str]], add_generation_prompt: bool) -> List[int]:
    """Token ids of the rendered chat. The template already contains BOS, so no special tokens are added."""
    text = chat_text(tokenizer, messages, add_generation_prompt)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False
