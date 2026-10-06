# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'Tests for the balanced stratified seed sampler', Date: 2026-10-06
from collections import Counter

from src.seeds import CATEGORICAL_DIMS, SEED_FIELDS, build_seeds, quota_counts


def test_quota_counts_exact_and_sum():
    assert quota_counts({"a": 0.30, "b": 0.45, "c": 0.25}, 400) == {"a": 120, "b": 180, "c": 100}
    q = quota_counts({"a": 0.30, "b": 0.45, "c": 0.25}, 7)
    assert sum(q.values()) == 7
    assert quota_counts({"x": 1, "y": 1, "z": 1}, 10) == {"x": 4, "y": 3, "z": 3}


def test_build_seeds_shape_ids_and_order(cfg):
    seeds = build_seeds(cfg, n=400)
    assert len(seeds) == 400
    assert [s["seed_id"] for s in seeds[:3]] == ["S0001", "S0002", "S0003"]
    assert len({s["seed_id"] for s in seeds}) == 400
    assert list(seeds[0].keys()) == SEED_FIELDS


def test_categorical_dims_are_balanced(cfg):
    seeds = build_seeds(cfg, n=400)
    for dim in CATEGORICAL_DIMS:
        counts = Counter(s[dim] for s in seeds)
        assert set(counts) == set(getattr(cfg.seeds, dim))
        assert max(counts.values()) - min(counts.values()) <= 1


def test_weighted_dims_follow_exact_quotas(cfg):
    seeds = build_seeds(cfg, n=400)
    assert Counter(s["target_verdict"] for s in seeds) == {"COMPLIANT": 120, "NON_COMPLIANT": 180, "NEEDS_MORE_INFO": 100}
    assert Counter(s["difficulty"] for s in seeds) == {"straightforward": 200, "multi_principle": 120, "tricky": 80}
    assert Counter(s["length_bucket"] for s in seeds) == {"short": 120, "medium": 180, "long": 100}


def test_deterministic_and_seed_sensitive(cfg):
    assert build_seeds(cfg, n=50) == build_seeds(cfg, n=50)
    assert build_seeds(cfg, n=50) != build_seeds(cfg, n=50, seed=7)


def test_no_combination_dominates(cfg):
    seeds = build_seeds(cfg, n=400)
    combos = Counter((s["business_unit"], s["data_category"], s["processing_activity"]) for s in seeds)
    assert max(combos.values()) <= 3
