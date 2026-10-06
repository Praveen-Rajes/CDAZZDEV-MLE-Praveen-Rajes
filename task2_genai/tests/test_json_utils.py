# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'Tests for robust JSON extraction edge cases', Date: 2026-10-06
from src.json_utils import extract_first_json


def test_plain_object():
    assert extract_first_json('{"a": 1}') == {"a": 1}


def test_code_fence_with_language():
    assert extract_first_json('```json\n{"a": {"b": [1, 2]}}\n```') == {"a": {"b": [1, 2]}}


def test_code_fence_without_language():
    assert extract_first_json('```\n{"a": 2}\n```') == {"a": 2}


def test_leading_chatter_and_trailing_text():
    text = 'Sure! Here is the triage:\n{"verdict": "COMPLIANT", "x": {"y": 1}}\nLet me know if you need more.'
    assert extract_first_json(text) == {"verdict": "COMPLIANT", "x": {"y": 1}}


def test_braces_inside_strings_and_escapes():
    text = 'note {"r": "use {curly} and \\"quotes\\" here", "n": 1} end'
    assert extract_first_json(text) == {"r": 'use {curly} and "quotes" here', "n": 1}


def test_skips_invalid_first_candidate():
    text = '{not json} then {"ok": true}'
    assert extract_first_json(text) == {"ok": True}


def test_returns_first_of_two_objects():
    assert extract_first_json('{"a": 1} {"b": 2}') == {"a": 1}


def test_none_cases():
    assert extract_first_json(None) is None
    assert extract_first_json("") is None
    assert extract_first_json("no json here") is None
    assert extract_first_json('{"unterminated": ') is None


def test_top_level_list_is_not_returned_but_inner_object_is():
    assert extract_first_json('[1, 2, 3]') is None
    assert extract_first_json('[{"a": 1}]') == {"a": 1}
