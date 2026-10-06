# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'Tests for schemas: every consistency rule, strict vs scoring mode, strict JSON schema export', Date: 2026-10-06
import json

import pytest

from src.schemas import (ANSWER_KEYS, JudgeScore, TeacherBatch, TeacherItem, TriageAnswer, canonical_answer_json,
                         sentence_count, strict_violations, to_strict_json_schema, validate_answer)


def test_fixture_answers_are_valid(items):
    for it in items:
        TeacherItem.model_validate({"seed_id": it["id"], "scenario": it["scenario"], "answer": it["answer"]})


@pytest.mark.parametrize("verdict,mutate,code", [
    ("COMPLIANT", lambda a: a.update(issues=["x happened"]), "compliant_with_issues"),
    ("COMPLIANT", lambda a: a.update(missing_information=["what?"]), "compliant_with_missing_information"),
    ("COMPLIANT", lambda a: a.update(risk_level="MEDIUM"), "compliant_risk_not_low"),
    ("NON_COMPLIANT", lambda a: a.update(issues=[]), "non_compliant_without_issues"),
    ("NON_COMPLIANT", lambda a: a.update(missing_information=["what?"]), "non_compliant_with_missing_information"),
    ("NEEDS_MORE_INFO", lambda a: a.update(missing_information=[]), "needs_more_info_without_missing_information"),
])
def test_each_consistency_rule(answers, verdict, mutate, code):
    a = answers[verdict]
    assert strict_violations(a) == []
    mutate(a)
    assert code in strict_violations(a)
    model, err = validate_answer(a, mode="strict")
    assert model is None and code in err


@pytest.mark.parametrize("mutate,prefix", [
    (lambda a: a.update(principles=[]), "principles_count"),
    (lambda a: a.update(principles=["P01", "P02", "P03", "P04"]), "principles_count"),
    (lambda a: a.update(principles=["P01", "P01"]), "principles_duplicate"),
    (lambda a: a.update(principles=["P17"]), "principles_invalid"),
    (lambda a: a.update(required_actions=[]), "required_actions_count"),
    (lambda a: a.update(required_actions=["Do " + "very " * 30 + "much."]), "required_actions_too_long"),
    (lambda a: a.update(rationale="One sentence only."), "rationale_sentences"),
    (lambda a: a.update(rationale="A. B. C. D. E."), "rationale_sentences"),
    (lambda a: a.update(rationale=("word " * 85).strip() + ". Second sentence."), "rationale_words"),
    (lambda a: a.update(issues=["a", "b", "c", "d"]), "issues_count"),
])
def test_field_rules(answers, mutate, prefix):
    a = answers["NON_COMPLIANT"]
    mutate(a)
    assert any(v.startswith(prefix) for v in strict_violations(a))


def test_scoring_mode_accepts_rule_breaks_but_not_type_errors(answers):
    a = answers["NON_COMPLIANT"]
    a["issues"] = []                     # consistency break
    a["principles"] = ["P99"]            # invalid id
    assert validate_answer(a, mode="scoring")[0] is not None
    assert validate_answer(a, mode="strict")[0] is None
    bad = dict(a, verdict="MAYBE")
    assert validate_answer(bad, mode="scoring")[0] is None
    extra = dict(a, extra_key=1)
    assert validate_answer(extra, mode="scoring")[0] is None
    assert validate_answer(["not", "a", "dict"], mode="scoring")[1] == "not_a_json_object"


def test_sentence_count_ignores_abbreviations():
    assert sentence_count("Fees are Rs. 500 per month. The form has no notice.") == 2
    assert sentence_count("Use e.g. masked data. Then delete it! Is it logged?") == 3


def test_canonical_json_key_order_and_format(answers):
    a = answers["COMPLIANT"]
    shuffled = {k: a[k] for k in reversed(ANSWER_KEYS)}
    s = canonical_answer_json(shuffled)
    assert list(json.loads(s).keys()) == ANSWER_KEYS
    assert '", "' in s or '": ' in s
    assert s == canonical_answer_json(a)


def test_strict_json_schema_export():
    s = to_strict_json_schema(TeacherBatch)
    text = json.dumps(s)
    assert "$ref" not in text and "$defs" not in text
    for banned in ("minItems", "maxItems", "minLength", "maxLength", "minimum", "maximum"):
        assert banned not in text
    assert s["additionalProperties"] is False and s["required"] == ["items"]
    item = s["properties"]["items"]["items"]
    assert item["required"] == ["seed_id", "scenario", "answer"] and item["additionalProperties"] is False
    ans = item["properties"]["answer"]
    assert ans["required"] == ANSWER_KEYS and ans["additionalProperties"] is False
    assert ans["properties"]["verdict"]["enum"] == ["COMPLIANT", "NON_COMPLIANT", "NEEDS_MORE_INFO"]
    j = to_strict_json_schema(JudgeScore)
    assert j["properties"]["faithfulness"]["enum"] == [1, 2, 3, 4, 5]
    assert j["properties"]["hallucination_detected"]["type"] == "boolean"
    assert j["required"][-1] == "justification"


def test_judge_score_bounds():
    ok = dict(verdict_correctness=5, principle_grounding=3, faithfulness=1, actionability=4, format_compliance=5,
              hallucination_detected=False, justification="fine")
    JudgeScore.model_validate(ok)
    with pytest.raises(Exception):
        JudgeScore.model_validate(dict(ok, faithfulness=6))
    with pytest.raises(Exception):
        TriageAnswer.model_validate({"rationale": "x"})
