# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'Balanced stratified seed sampler per plan Phase 2', Date: 2026-10-06
"""Balanced stratified seed sampler.

Categorical dimensions: cycle the values to length N, shuffle each independently with a
fixed seed, then zip. Weighted dimensions: exact quotas (largest remainder rounding),
shuffled. This guarantees even coverage and avoids near-identical variations.

CLI: python -m src.seeds [--n 400]
"""
from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from typing import Dict, List

from .config import Config, load_config
from .utils import write_jsonl

CATEGORICAL_DIMS = ["business_unit", "data_category", "processing_activity", "artifact_type"]
WEIGHTED_DIMS = {  # seed field -> config attribute holding the quota
    "target_verdict": "verdict_quota",
    "difficulty": "difficulty_quota",
    "length_bucket": "length_quota",
}
SEED_FIELDS = ["seed_id", *CATEGORICAL_DIMS, *WEIGHTED_DIMS.keys()]


def quota_counts(quota: Dict[str, float], n: int) -> Dict[str, int]:
    """Exact integer counts summing to n (largest remainder method, ties broken by key order)."""
    total = sum(quota.values())
    raw = {k: n * v / total for k, v in quota.items()}
    counts = {k: math.floor(x) for k, x in raw.items()}
    short = n - sum(counts.values())
    order = sorted(quota.keys(), key=lambda k: (-(raw[k] - counts[k]), list(quota).index(k)))
    for k in order[:short]:
        counts[k] += 1
    return counts


def cycle_to(values: List[str], n: int) -> List[str]:
    return [values[i % len(values)] for i in range(n)]


def build_seeds(cfg: Config, n: int | None = None, seed: int | None = None) -> List[Dict[str, str]]:
    n = n or cfg.seeds.n
    base_seed = cfg.project.seed if seed is None else seed
    columns: Dict[str, List[str]] = {}
    for i, dim in enumerate(CATEGORICAL_DIMS):
        col = cycle_to(list(getattr(cfg.seeds, dim)), n)
        random.Random(base_seed + 1 + i).shuffle(col)          # independent stream per dimension
        columns[dim] = col
    for j, (dim, attr) in enumerate(WEIGHTED_DIMS.items()):
        counts = quota_counts(getattr(cfg.seeds, attr), n)
        col = [k for k, c in counts.items() for _ in range(c)]
        random.Random(base_seed + 101 + j).shuffle(col)
        columns[dim] = col
    seeds = []
    for idx in range(n):
        s = {"seed_id": f"S{idx + 1:04d}"}
        for dim in [*CATEGORICAL_DIMS, *WEIGHTED_DIMS.keys()]:
            s[dim] = columns[dim][idx]
        seeds.append(s)
    return seeds


def coverage(seeds: List[Dict[str, str]]) -> Dict[str, Dict[str, int]]:
    return {dim: dict(Counter(s[dim] for s in seeds)) for dim in [*CATEGORICAL_DIMS, *WEIGHTED_DIMS.keys()]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build stratified seeds")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = build_seeds(cfg, n=args.n)
    out = cfg.path("seeds")
    write_jsonl(out, seeds)
    print(f"Wrote {len(seeds)} seeds to {out}")
    for dim, counts in coverage(seeds).items():
        print(f"  {dim}: min={min(counts.values())} max={max(counts.values())} {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
