# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'Tests for stratified split, cross-split leakage and chat formatting', Date: 2026-10-06
import json
import random

import numpy as np

from src.schemas import ANSWER_KEYS
from src.split_format import USER_PREFIX, move_leaky_to_train, scenario_of, stratified_split, to_chat_record


def _fake_items(counts):
    items, k = [], 0
    for verdict, n in counts.items():
        for _ in range(n):
            k += 1
            items.append({"id": f"S{k:04d}", "scenario": f"scenario {k}", "answer": {"verdict": verdict}})
    return items


def test_sizes_and_stratification():
    items = _fake_items({"COMPLIANT": 90, "NON_COMPLIANT": 135, "NEEDS_MORE_INFO": 75})
    s = stratified_split(items, 0.8, 0.1, 0.1, 42)
    # Per class: round(n * 0.1) for test and for val (Python rounds 13.5 -> 14 and 7.5 -> 8).
    for split in ("val", "test"):
        verdicts = [it["answer"]["verdict"] for it in s[split]]
        assert {v: verdicts.count(v) for v in set(verdicts)} == {"COMPLIANT": 9, "NON_COMPLIANT": 14, "NEEDS_MORE_INFO": 8}
    assert {k: len(v) for k, v in s.items()} == {"train": 238, "val": 31, "test": 31}
    ids = [it["id"] for v in s.values() for it in v]
    assert len(ids) == len(set(ids)) == 300


def test_split_deterministic_and_order_independent():
    items = _fake_items({"COMPLIANT": 20, "NON_COMPLIANT": 30, "NEEDS_MORE_INFO": 15})
    a = stratified_split(items, 0.8, 0.1, 0.1, 42)
    shuffled = items[:]
    random.Random(1).shuffle(shuffled)
    b = stratified_split(shuffled, 0.8, 0.1, 0.1, 42)
    assert {k: [i["id"] for i in v] for k, v in a.items()} == {k: [i["id"] for i in v] for k, v in b.items()}
    c = stratified_split(items, 0.8, 0.1, 0.1, 7)
    assert [i["id"] for i in a["test"]] != [i["id"] for i in c["test"]]


def test_leaky_items_move_to_train():
    items = _fake_items({"COMPLIANT": 10})
    splits = {"train": items[:8], "val": [items[8]], "test": [items[9]]}
    splits["test"][0]["scenario"] = splits["train"][0]["scenario"]          # exact copy of a train scenario

    def embed(texts):
        vecs = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.normal(size=16)
            vecs.append(v / np.linalg.norm(v))
        return np.array(vecs)

    report = move_leaky_to_train(splits, 0.90, embed)
    assert report["n_moved"] == 1 and report["moved_from_test"][0]["id"] == "S0010"
    assert len(splits["train"]) == 9 and splits["test"] == [] and len(splits["val"]) == 1


def test_chat_record_format(items):
    rec = to_chat_record(items[0], "SYSTEM PROMPT")
    assert [m["role"] for m in rec["messages"]] == ["system", "user", "assistant"]
    assert rec["messages"][1]["content"] == USER_PREFIX + items[0]["scenario"]
    assert scenario_of(rec) == items[0]["scenario"]
    answer = json.loads(rec["messages"][2]["content"])
    assert list(answer.keys()) == ANSWER_KEYS
    assert rec["messages"][2]["content"].startswith('{"rationale": ')
    assert rec["meta"]["verdict"] == items[0]["answer"]["verdict"]
    assert rec["meta"]["difficulty"] == items[0]["seed"]["difficulty"]
