# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'Tests for per-epoch loss table, run selection, warmup maths, prompts and RAG chunking', Date: 2026-10-06
from pathlib import Path

import pytest

from src.config import REPO_ROOT, load_text
from src.rag_fallback import format_chunks, rulebook_chunks
from src.train import is_strictly_decreasing, per_epoch_table, select_run, steps_per_epoch, warmup_steps_for


def test_per_epoch_table():
    log = [
        {"epoch": 0.25, "loss": 2.0}, {"epoch": 0.5, "loss": 1.6}, {"epoch": 1.0, "loss": 1.2},
        {"epoch": 1.0, "eval_loss": 1.1},
        {"epoch": 1.5, "loss": 0.9}, {"epoch": 2.0, "loss": 0.7}, {"epoch": 2.0, "eval_loss": 0.8},
        {"epoch": 2.0, "train_loss": 1.2, "train_runtime": 10},
    ]
    t = per_epoch_table(log, {"eval_loss": 2.5, "train_loss": 2.4})
    assert t[0] == {"epoch": 0, "train_loss": 2.4, "val_loss": 2.5}
    assert t[1] == {"epoch": 1, "train_loss": round((2.0 + 1.6 + 1.2) / 3, 4), "val_loss": 1.1}
    assert t[2] == {"epoch": 2, "train_loss": 0.8, "val_loss": 0.8}


def test_monotonic_and_selection():
    assert is_strictly_decreasing([2.5, 1.1, 0.9, 0.85])
    assert not is_strictly_decreasing([2.5, 1.1, 1.2])
    assert not is_strictly_decreasing([1.0])
    runs = [{"run_name": "a", "monotonic": True, "final_val_loss": 0.9},
            {"run_name": "b", "monotonic": False, "final_val_loss": 0.7},
            {"run_name": "c", "monotonic": True, "final_val_loss": 0.8}]
    assert select_run(runs)["run_name"] == "c"
    assert select_run([runs[1]]) is None


def test_steps_and_warmup():
    assert steps_per_epoch(290, 4, 4) == 19          # 73 batches -> 19 optimiser steps
    assert warmup_steps_for(290, 4, 4, 3, 0.05) == 3  # ceil(0.05 * 57)
    assert warmup_steps_for(8, 4, 1, 1, 0.0) == 0


def test_pin_chat_template_date():
    from src.utils import pin_chat_template_date

    class Tok:
        chat_template = '{%- set date_string = strftime_now("%d %b %Y") %}Today Date: {{ date_string }}'

    t = Tok()
    assert pin_chat_template_date(t, "26 Jul 2024")
    assert t.chat_template == '{%- set date_string = "26 Jul 2024" %}Today Date: {{ date_string }}'
    assert not pin_chat_template_date(t, "26 Jul 2024")       # already pinned: no change


def test_rulebook_chunks():
    chunks = rulebook_chunks(load_text(Path(__file__).resolve().parents[1] / "knowledge" / "pdpa_policy_rulebook.md"))
    assert len(chunks) == 16 * 4 + 2
    ids = {c["metadata"]["principle_id"] for c in chunks}
    assert ids == {f"P{i:02d}" for i in range(1, 17)} | {"GUIDE"}
    p12 = [c for c in chunks if c["id"] == "P12-bank_standard"][0]
    assert p12["text"].startswith("P12 Cross-border data transfers. Bank standard: Every transfer")
    text = format_chunks([{"text": p12["text"], "principle_id": "P12", "subsection": "Bank standard"}])
    assert text.startswith("[1] (P12, Bank standard) Every transfer")


def test_student_prompt_lists_all_principles(cfg):
    text = load_text(cfg.prompt_path("student_system_prompt.md"))
    for i in range(1, 17):
        assert f"P{i:02d} " in text


def _plan_block(plan: str, start_marker: str) -> str:
    start = plan.index(start_marker)
    fence = plan.index("```", start)
    body_start = plan.index("\n", fence) + 1
    body_end = plan.index("\n```", body_start)
    return plan[body_start:body_end]


@pytest.mark.parametrize("marker,filename", [
    ("## Appendix A: Rulebook", "knowledge/pdpa_policy_rulebook.md"),
    ("## Appendix B: Student system prompt", "prompts/student_system_prompt.md"),
    ("### C1. `prompts/teacher_system_prompt.md`", "prompts/teacher_system_prompt.md"),
    ("### C2. `prompts/teacher_user_template.md`", "prompts/teacher_user_template.md"),
    ("### D1. `prompts/judge_system_prompt.md`", "prompts/judge_system_prompt.md"),
    ("### D2. `prompts/judge_user_template.md`", "prompts/judge_user_template.md"),
    ("## Appendix E: RAG user template", "prompts/rag_user_template.md"),
])
def test_prompt_files_match_plan_verbatim(marker, filename):
    plan_path = REPO_ROOT / "plan.md"
    if not plan_path.exists():
        pytest.skip("plan.md not present")
    plan = plan_path.read_text(encoding="utf-8")
    expected = _plan_block(plan, marker)
    actual = (Path(__file__).resolve().parents[1] / filename).read_text(encoding="utf-8").rstrip("\n")
    assert actual == expected
