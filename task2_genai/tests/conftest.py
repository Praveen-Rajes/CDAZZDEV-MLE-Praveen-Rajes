# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'Shared pytest fixtures for task2_genai tests', Date: 2026-10-06
import copy
import sys
from pathlib import Path

import pytest

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT))

from src.config import load_config  # noqa: E402
from src.utils import read_jsonl  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture
def items():
    """Six hand-written items (two per verdict); deep-copied so tests can mutate them."""
    return copy.deepcopy(read_jsonl(TASK_ROOT / "tests" / "fixtures" / "sample_items.jsonl"))


@pytest.fixture
def answers(items):
    return {it["answer"]["verdict"]: copy.deepcopy(it["answer"]) for it in items}
