# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'Pydantic v2 schemas, consistency validators and strict JSON schema export per plan Appendix G', Date: 2026-10-06
"""Answer, teacher and judge schemas.

Two validation modes for TriageAnswer:
- "strict" (default, training data): every field rule and the consistency rule.
- "scoring" (model outputs): type and enum checks only, so a slightly long rationale
  is still scored rather than discarded. Rule violations become metrics instead
  (see `strict_violations`).

Select the mode with a validation context:
    TriageAnswer.model_validate(obj, context={"mode": "scoring"})
"""
from __future__ import annotations

import copy
import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, model_validator

PRINCIPLE_IDS = {f"P{i:02d}" for i in range(1, 17)}
ANSWER_KEYS = [
    "rationale",
    "principles",
    "issues",
    "missing_information",
    "verdict",
    "risk_level",
    "required_actions",
]

# Strict-mode limits (plan Appendix C answer rules and Appendix G comments).
RATIONALE_MIN_SENTENCES = 2
RATIONALE_MAX_SENTENCES = 4
RATIONALE_MAX_WORDS = 80
PRINCIPLES_MIN, PRINCIPLES_MAX = 1, 3
LIST_MAX = 3                     # issues and missing_information: 0 to 3
ACTIONS_MIN, ACTIONS_MAX = 1, 3
ACTION_MAX_WORDS = 25

# Abbreviations that end with a full stop but do not end a sentence.
_ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "Rs.", "No.", "Mr.", "Mrs.", "Ms.", "Dr.", "vs.", "approx.", "Ltd.", "Pvt.")


class Verdict(str, Enum):
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    NEEDS_MORE_INFO = "NEEDS_MORE_INFO"


class Risk(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


def word_count(text: str) -> int:
    return len(text.split())


def sentence_count(text: str) -> int:
    """Count sentences by terminal punctuation, ignoring common abbreviations."""
    masked = text
    for abbr in _ABBREVIATIONS:
        masked = masked.replace(abbr, abbr.replace(".", "<dot>"))
    parts = [p for p in re.split(r"(?<=[.!?])\s+", masked.strip()) if p.strip()]
    return len(parts)


def strict_violations(answer: Dict[str, Any], rationale_max_words: int = RATIONALE_MAX_WORDS) -> List[str]:
    """Every strict-mode rule the answer breaks, as short codes. Empty list means valid.

    Expects a dict with the seven answer keys (enum values as plain strings).
    """
    v: List[str] = []
    rationale = answer.get("rationale", "") or ""
    principles = answer.get("principles", []) or []
    issues = answer.get("issues", []) or []
    missing = answer.get("missing_information", []) or []
    actions = answer.get("required_actions", []) or []
    verdict = answer.get("verdict")
    risk = answer.get("risk_level")
    verdict = verdict.value if isinstance(verdict, Enum) else verdict
    risk = risk.value if isinstance(risk, Enum) else risk

    # Field-level checks.
    n_sent = sentence_count(rationale)
    if not (RATIONALE_MIN_SENTENCES <= n_sent <= RATIONALE_MAX_SENTENCES):
        v.append(f"rationale_sentences={n_sent}")
    if word_count(rationale) > rationale_max_words:
        v.append(f"rationale_words={word_count(rationale)}")
    if not (PRINCIPLES_MIN <= len(principles) <= PRINCIPLES_MAX):
        v.append(f"principles_count={len(principles)}")
    if len(set(principles)) != len(principles):
        v.append("principles_duplicate")
    bad_ids = [p for p in principles if p not in PRINCIPLE_IDS]
    if bad_ids:
        v.append(f"principles_invalid={bad_ids}")
    if len(issues) > LIST_MAX:
        v.append(f"issues_count={len(issues)}")
    if len(missing) > LIST_MAX:
        v.append(f"missing_information_count={len(missing)}")
    if not (ACTIONS_MIN <= len(actions) <= ACTIONS_MAX):
        v.append(f"required_actions_count={len(actions)}")
    long_actions = [i for i, a in enumerate(actions) if word_count(a) > ACTION_MAX_WORDS]
    if long_actions:
        v.append(f"required_actions_too_long={long_actions}")
    for name, items in (("issues", issues), ("missing_information", missing), ("required_actions", actions)):
        if any(not str(x).strip() for x in items):
            v.append(f"{name}_empty_item")

    # Consistency between verdict, lists and risk.
    if verdict == Verdict.COMPLIANT.value:
        if issues:
            v.append("compliant_with_issues")
        if missing:
            v.append("compliant_with_missing_information")
        if risk != Risk.LOW.value:
            v.append("compliant_risk_not_low")
    elif verdict == Verdict.NON_COMPLIANT.value:
        if len(issues) < 1:
            v.append("non_compliant_without_issues")
        if missing:
            v.append("non_compliant_with_missing_information")
    elif verdict == Verdict.NEEDS_MORE_INFO.value:
        if len(missing) < 1:
            v.append("needs_more_info_without_missing_information")
    return v


def _mode(info: ValidationInfo) -> str:
    ctx = info.context or {}
    return ctx.get("mode", "strict")


class TriageAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rationale: str
    principles: List[str]
    issues: List[str]
    missing_information: List[str]
    verdict: Verdict
    risk_level: Risk
    required_actions: List[str]

    @model_validator(mode="after")
    def consistency(self, info: ValidationInfo):
        # COMPLIANT       -> issues == [] and missing_information == [] and risk_level == LOW
        # NON_COMPLIANT   -> len(issues) >= 1 and missing_information == []
        # NEEDS_MORE_INFO -> len(missing_information) >= 1
        # plus counts, lengths and valid unique principle IDs (see strict_violations).
        if _mode(info) == "strict":
            problems = strict_violations(self.model_dump(mode="json"))
            if problems:
                raise ValueError("; ".join(problems))
        return self


class TeacherItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed_id: str
    scenario: str
    answer: TriageAnswer


class TeacherBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: List[TeacherItem]


class JudgeScore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict_correctness: int = Field(ge=1, le=5)
    principle_grounding: int = Field(ge=1, le=5)
    faithfulness: int = Field(ge=1, le=5)
    actionability: int = Field(ge=1, le=5)
    format_compliance: int = Field(ge=1, le=5)
    hallucination_detected: bool
    justification: str


JUDGE_CRITERIA = ["verdict_correctness", "principle_grounding", "faithfulness", "actionability", "format_compliance"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def canonical_answer_json(answer: Dict[str, Any]) -> str:
    """The one canonical serialisation: keys in schema order, compact, UTF-8."""
    ordered = {}
    for k in ANSWER_KEYS:
        val = answer[k]
        ordered[k] = val.value if isinstance(val, Enum) else val
    return json.dumps(ordered, ensure_ascii=False, separators=(", ", ": "))


def validate_answer(obj: Any, mode: str = "strict") -> Tuple[Optional[TriageAnswer], Optional[str]]:
    """Validate an answer dict. Returns (model, None) or (None, error message)."""
    if not isinstance(obj, dict):
        return None, "not_a_json_object"
    try:
        return TriageAnswer.model_validate(obj, context={"mode": mode}), None
    except ValidationError as e:
        return None, _short_error(e)


def _short_error(e: ValidationError) -> str:
    parts = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()))
        msg = err.get("msg", "")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return " | ".join(parts)[:500]


_DROP_KEYS = {
    "title", "default", "description", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "minItems", "maxItems", "minLength", "maxLength", "pattern", "format",
}


def to_strict_json_schema(model: type[BaseModel]) -> Dict[str, Any]:
    """Export a strict-mode API schema.

    Every property is required, every object has additionalProperties: false, $refs are
    inlined, enums stay string enums, bounded integers become integer enums (judge 1 to 5),
    and count/length constraints are dropped (they are enforced by Pydantic after the call).
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, list):
            return [resolve(x) for x in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return resolve(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
        if "allOf" in node and len(node["allOf"]) == 1:
            return resolve(node["allOf"][0])
        out: Dict[str, Any] = {}
        for k, val in node.items():
            if k in _DROP_KEYS or k == "$defs":
                continue
            if k == "properties":
                out[k] = {name: resolve(sub) for name, sub in val.items()}
            else:
                out[k] = resolve(val)
        if out.get("type") == "integer" and "minimum" in node and "maximum" in node:
            out["enum"] = list(range(int(node["minimum"]), int(node["maximum"]) + 1))
        if out.get("type") == "object" and "properties" in out:
            out["required"] = list(out["properties"].keys())
            out["additionalProperties"] = False
        return out

    return resolve(schema)
