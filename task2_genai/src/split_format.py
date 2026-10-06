# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'Stratified 80/10/10 split, cross-split leakage check and chat JSONL per plan Phase 3', Date: 2026-10-06
"""Split the clean data and write chat-format JSONL.

- Stratified by verdict, 80/10/10, seed 42.
- Cross-split leakage: any val/test scenario with cosine >= 0.90 to any train scenario moves to train.
- One object per line: {"id", "messages": [system, user, assistant], "meta": {...}}.
  The assistant content is the canonical answer JSON (keys in schema order), used everywhere:
  training targets, references and metrics.

CLI: python -m src.split_format
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence

from .config import Config, load_config, load_text
from .schemas import canonical_answer_json
from .utils import read_jsonl, write_json, write_jsonl

USER_PREFIX = "Internal request:\n"
SPLITS = ("train", "val", "test")


def user_message(scenario: str) -> str:
    return f"{USER_PREFIX}{scenario}"


def student_system_prompt(cfg: Config) -> str:
    return load_text(cfg.prompt_path("student_system_prompt.md")).strip()


def stratified_split(items: List[Dict[str, Any]], train: float, val: float, test: float, seed: int,
                     key: Callable[[Dict[str, Any]], str] = lambda it: it["answer"]["verdict"]) -> Dict[str, List[Dict[str, Any]]]:
    """Per class: shuffle (fixed seed), take round(n*test) for test, round(n*val) for val, rest train."""
    assert abs(train + val + test - 1.0) < 1e-6, "split fractions must sum to 1"
    by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for it in sorted(items, key=lambda x: x["id"]):          # sort first: independent of file order
        by_class[key(it)].append(it)
    out: Dict[str, List[Dict[str, Any]]] = {s: [] for s in SPLITS}
    for cls in sorted(by_class):
        group = by_class[cls][:]
        random.Random(f"{seed}-{cls}").shuffle(group)
        n_test = int(round(len(group) * test))
        n_val = int(round(len(group) * val))
        out["test"] += group[:n_test]
        out["val"] += group[n_test : n_test + n_val]
        out["train"] += group[n_test + n_val :]
    for s in SPLITS:
        out[s].sort(key=lambda x: x["id"])
    return out


def move_leaky_to_train(splits: Dict[str, List[Dict[str, Any]]], threshold: float,
                        embed_fn: Callable[[Sequence[str]], Any]) -> Dict[str, Any]:
    """Move val/test items whose cosine to ANY train item is >= threshold into train (in place)."""
    import numpy as np

    train_embs = np.asarray(embed_fn([it["scenario"] for it in splits["train"]]))
    moved: Dict[str, List[Dict[str, Any]]] = {"val": [], "test": []}
    for s in ("val", "test"):
        if not splits[s]:
            continue
        embs = np.asarray(embed_fn([it["scenario"] for it in splits[s]]))
        max_sim = (embs @ train_embs.T).max(axis=1) if len(train_embs) else np.zeros(len(embs))
        keep = []
        for it, sim in zip(splits[s], max_sim):
            if float(sim) >= threshold:
                moved[s].append({"id": it["id"], "max_cosine_to_train": round(float(sim), 4)})
                splits["train"].append(it)
            else:
                keep.append(it)
        splits[s] = keep
    splits["train"].sort(key=lambda x: x["id"])
    return {"threshold": threshold, "moved_from_val": moved["val"], "moved_from_test": moved["test"],
            "n_moved": len(moved["val"]) + len(moved["test"])}


def to_chat_record(item: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
    seed = item.get("seed", {})
    return {
        "id": item["id"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message(item["scenario"])},
            {"role": "assistant", "content": canonical_answer_json(item["answer"])},
        ],
        "meta": {
            "verdict": item["answer"]["verdict"],
            "risk_level": item["answer"]["risk_level"],
            "principles": item["answer"]["principles"],
            "teacher_model": item.get("teacher_model"),
            "business_unit": seed.get("business_unit"),
            "data_category": seed.get("data_category"),
            "processing_activity": seed.get("processing_activity"),
            "artifact_type": seed.get("artifact_type"),
            "difficulty": seed.get("difficulty"),
            "length_bucket": seed.get("length_bucket"),
        },
    }


def scenario_of(record: Dict[str, Any]) -> str:
    """Recover the scenario from a chat record's user turn."""
    content = record["messages"][1]["content"]
    return content[len(USER_PREFIX):] if content.startswith(USER_PREFIX) else content


def run(cfg: Config, embed_fn: Optional[Callable[[Sequence[str]], Any]] = None) -> Dict[str, Any]:
    from .validate_data import embed_texts

    items = read_jsonl(cfg.path("clean"))
    if not items:
        raise FileNotFoundError(f"No clean data at {cfg.path('clean')}. Run: python -m src.validate_data")
    sc = cfg.split
    splits = stratified_split(items, sc.train, sc.val, sc.test, sc.seed)
    sizes_before = {s: len(v) for s, v in splits.items()}
    embed_fn = embed_fn or (lambda t: embed_texts(t, cfg.filters.embed_model))
    leak = move_leaky_to_train(splits, sc.cross_split_threshold, embed_fn)
    system_prompt = student_system_prompt(cfg)
    for s in SPLITS:
        write_jsonl(cfg.split_path(s), [to_chat_record(it, system_prompt) for it in splits[s]])
    sizes = {s: len(splits[s]) for s in SPLITS}
    report = {
        "total": sum(sizes.values()),
        "sizes": sizes,
        "sizes_before_leakage_check": sizes_before,
        "fractions": {s: round(sizes[s] / max(sum(sizes.values()), 1), 3) for s in SPLITS},
        "verdict_by_split": {s: dict(Counter(it["answer"]["verdict"] for it in splits[s])) for s in SPLITS},
        "difficulty_by_split": {s: dict(Counter(it["seed"]["difficulty"] for it in splits[s])) for s in SPLITS},
        "leakage_check": leak,
        "seed": sc.seed,
    }
    write_json(cfg.report_path("split_sizes.json"), report)
    print(f"[split] sizes={sizes} (moved to train by leakage check: {leak['n_moved']})")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Stratified split + chat JSONL")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
