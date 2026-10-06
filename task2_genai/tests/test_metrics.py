# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'Tests for task metrics, paired bootstrap, Wilson interval, kappa and citation regexes', Date: 2026-10-06
import copy

import numpy as np
import pytest

from src.manual_review import cohen_kappa, wilson
from src.metrics import (comparison_table, component_matrix, confusion, example_components, paired_bootstrap,
                         point_metrics, principle_counts)
from src.validate_data import statute_hits


def _pred(answer, parse_ok=True):
    return {"parsed": answer if parse_ok else None, "parse_ok": parse_ok, "schema_ok": parse_ok,
            "raw_text": "garbage" if not parse_ok else "{}", "strict_violations": [] if parse_ok else ["x"],
            "generated_tokens": 100, "perplexity": 1.5}


def test_perfect_prediction_components(answers):
    ref = answers["NON_COMPLIANT"]
    c = example_components(_pred(copy.deepcopy(ref)), ref)
    assert c["verdict_correct"] == 1 and c["risk_correct"] == 1
    assert (c["p_tp"], c["p_fp"], c["p_fn"]) == (2, 0, 0)
    assert c["invented"] == 0 and c["strict_ok"] == 1


def test_unparseable_prediction_scores_zero(answers):
    ref = answers["COMPLIANT"]
    c = example_components(_pred(None, parse_ok=False), ref)
    assert c["verdict_correct"] == 0 and c["parse_ok"] == 0 and c["p_jaccard"] == 0
    assert c["v_fn_COMPLIANT"] == 1


def test_invented_principle_and_statute_flagged(answers):
    ref = answers["NON_COMPLIANT"]
    p = copy.deepcopy(ref)
    p["principles"] = ["P11", "P21"]
    p["rationale"] = "This breaches section 12 of the Act. It is serious."
    c = example_components(_pred(p), ref)
    assert c["invalid_id"] == 1 and c["statute"] == 1 and c["invented"] == 1


def test_principle_counts():
    assert principle_counts(["P01", "P02"], ["P02", "P03"]) == (1, 1, 1, 1 / 3)
    assert principle_counts([], []) == (0, 0, 0, 1.0)


def test_point_metrics_micro_and_macro(answers):
    refs = [answers["COMPLIANT"], answers["NON_COMPLIANT"], answers["NEEDS_MORE_INFO"]]
    preds = [copy.deepcopy(refs[0]), copy.deepcopy(refs[1]), copy.deepcopy(refs[1])]   # last one wrong verdict
    comp = component_matrix([example_components(_pred(p), r) for p, r in zip(preds, refs)])
    m = point_metrics(comp)
    assert m["verdict_accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    # per-class F1: COMPLIANT 1.0, NON_COMPLIANT 2/3, NEEDS_MORE_INFO 0.0
    assert m["verdict_macro_f1"] == pytest.approx((1 + 2 / 3 + 0) / 3, abs=1e-4)
    assert m["json_valid"] == 1.0


def test_majority_components_only_have_verdict_metrics(answers):
    ref = answers["NEEDS_MORE_INFO"]
    comp = component_matrix([example_components(None, ref, pred_verdict="NON_COMPLIANT")])
    m = point_metrics(comp)
    assert m["verdict_accuracy"] == 0.0 and "json_valid" not in m


def test_bootstrap_identical_systems_gives_zero(answers):
    refs = [answers[v] for v in ("COMPLIANT", "NON_COMPLIANT", "NEEDS_MORE_INFO")] * 4
    comp = component_matrix([example_components(_pred(copy.deepcopy(r)), r) for r in refs])
    out = paired_bootstrap(comp, comp, 500, 42)
    assert out["verdict_accuracy"]["diff"] == 0 and out["verdict_accuracy"]["ci_low"] == 0
    assert out["principles_micro_f1"]["ci_high"] == 0


def test_bootstrap_better_system_positive(answers):
    refs = [answers[v] for v in ("COMPLIANT", "NON_COMPLIANT", "NEEDS_MORE_INFO")] * 10
    good = component_matrix([example_components(_pred(copy.deepcopy(r)), r) for r in refs])
    bad = component_matrix([example_components(_pred(None, parse_ok=False), r) for r in refs])
    out = paired_bootstrap(good, bad, 1000, 42)
    assert out["verdict_accuracy"]["diff"] == 1.0
    assert out["json_valid"]["ci_low"] > 0


def test_statute_regex():
    assert statute_hits("Under Section 26 the Bank must act.")
    assert statute_hits("A Rs. 50,000 fine applies.")
    assert statute_hits("Penalties may follow.")
    assert statute_hits("see Act No. 9 of 2022")
    assert not statute_hits("Charge a Rs. 500 fee for the card.")
    assert not statute_hits("Customers' records were reviewed.")


def test_wilson_interval_known_values():
    p, lo, hi = wilson(10, 20)
    assert p == 0.5 and lo == pytest.approx(0.2993, abs=1e-3) and hi == pytest.approx(0.7007, abs=1e-3)
    p, lo, hi = wilson(0, 20)
    assert p == 0 and lo == 0 and hi == pytest.approx(0.1611, abs=1e-3)
    assert wilson(0, 0) == (None, None, None)


def test_cohen_kappa():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0
    assert cohen_kappa([], []) is None
    assert cohen_kappa([True, True], [True, True]) is None      # no variation: kappa undefined


def test_confusion_counts_invalid(answers):
    preds = [_pred(answers["COMPLIANT"]), _pred(None, parse_ok=False)]
    c = confusion(preds, ["COMPLIANT", "NEEDS_MORE_INFO"])
    m = np.array(c["matrix"])
    assert m[0, 0] == 1 and m[2, 3] == 1 and m.sum() == 2


def test_comparison_table_handles_missing_values():
    res = {"systems": {"ft": {"rougeL_full": 0.5, "json_valid": 1.0}}, "comparisons": {
        "ft_minus_base_zeroshot": {"rougeL_full": {"diff": 0.1, "ci_low": 0.05, "ci_high": 0.15}}}}
    table = comparison_table(res)
    assert "| ROUGE-L F1 (full output) | n/a | n/a | n/a | 0.500 | +0.100 [+0.050, +0.150] | n/a |" in table
    assert "100.0%" in table


def test_rouge_l_identical_is_one():
    pytest.importorskip("rouge_score")
    from src.metrics import rouge_l

    assert rouge_l(["the cat sat"], ["the cat sat"])[0] == pytest.approx(1.0)
