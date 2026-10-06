# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'Tests for per-item teacher checks and the cleaning filters', Date: 2026-10-06
import copy

import numpy as np

from src.generate_data import build_user_message, check_item, length_bounds, render_teacher_system_prompt
from src.validate_data import forbidden_hits, leakage_hits, run_filters, text_hash


def _item(it):
    return {"seed_id": it["id"], "scenario": it["scenario"], "answer": it["answer"]}


def test_check_item_accepts_fixtures(cfg, items):
    for it in items:
        ok, reason, parsed = check_item(_item(it), it["seed"], cfg)
        assert ok, reason


def test_check_item_rejections(cfg, items):
    it = items[0]
    seed = dict(it["seed"], target_verdict="COMPLIANT")
    assert check_item(_item(it), seed, cfg)[1].startswith("verdict_mismatch")
    seed = dict(it["seed"], length_bucket="long")
    assert check_item(_item(it), seed, cfg)[1].startswith("length_out_of_bucket")
    bad = _item(it)
    bad["seed_id"] = "S9999"
    assert check_item(bad, it["seed"], cfg)[1].startswith("seed_id_mismatch")
    bad = _item(it)
    bad["answer"] = dict(bad["answer"], issues=[])
    assert check_item(bad, it["seed"], cfg)[1].startswith("consistency")
    bad = _item(it)
    bad["answer"] = dict(bad["answer"], verdict="PROBABLY")
    assert check_item(bad, it["seed"], cfg)[1].startswith("schema")


def test_length_bounds(cfg):
    assert length_bounds(cfg, "short") == (50 * 0.85, 90 * 1.15)


def test_leakage():
    assert "P12" in leakage_hits("As per P12 this is fine")
    assert leakage_hits("This looks compliant to me")
    assert leakage_hits("Clearly non-compliant")
    assert leakage_hits("Give me your verdict")
    assert not leakage_hits("Please send this to the compliance team for review.")


def test_forbidden_content():
    ans = {"rationale": "ok. ok.", "issues": [], "missing_information": [], "required_actions": ["Do it."]}
    assert any(h.startswith("bank_name") for h in forbidden_hits("We moved from Hatton National Bank last year.", ans))
    assert not forbidden_hits("Sampath from Treasury asked about the transfer.", ans)
    assert any("full_nic" in h for h in forbidden_hits("Customer NIC 856123456V was shared.", ans))
    assert not forbidden_hits("Customer NIC ending 4521V was shared.", ans)
    assert any(h.startswith("statute") for h in forbidden_hits("fine", dict(ans, rationale="See section 5. Done.")))


def test_text_hash_normalises():
    assert text_hash("Hello,   World!") == text_hash("hello world")


def test_run_filters_drops_exact_and_near_duplicates(cfg, items):
    rows = [{"seed_id": it["id"], "seed": it["seed"], "scenario": it["scenario"], "answer": it["answer"],
             "teacher_model": "t", "call_id": "c"} for it in items]
    dup = copy.deepcopy(rows[1])
    dup["seed_id"] = dup["seed"]["seed_id"] = "F101"
    dup["scenario"] = dup["scenario"].upper()                      # exact duplicate after normalisation
    near = copy.deepcopy(rows[2])
    near["seed_id"] = near["seed"]["seed_id"] = "F102"
    near["scenario"] = near["scenario"] + " Thanks."                # near duplicate
    leak = copy.deepcopy(rows[3])
    leak["seed_id"] = leak["seed"]["seed_id"] = "F103"
    leak["scenario"] = leak["scenario"].replace("Hi Roshan", "Hi Roshan, is this compliant")
    rows += [dup, near, leak]

    def embed(texts):
        vecs = []
        for t in texts:
            base = t.replace(" Thanks.", "")
            rng = np.random.default_rng(int(text_hash(base)[:8], 16))
            v = rng.normal(size=32)
            vecs.append(v / np.linalg.norm(v))
        return np.array(vecs)

    clean, report = run_filters(cfg, rows, embed_fn=embed)
    assert [c["id"] for c in clean] == [it["id"] for it in items]
    stages = {s["stage"]: s["dropped"] for s in report["stages"]}
    assert stages == {"1_schema_consistency": 0, "2_leakage": 1, "3_forbidden_content": 0,
                      "4_exact_duplicates": 1, "5_near_duplicates": 1}
    assert clean[0]["answer_json"].startswith('{"rationale": ')


def test_teacher_prompt_rendering(cfg):
    text = render_teacher_system_prompt(cfg)
    assert "{RULEBOOK}" not in text
    assert "## P16 Accountability, records and DPO consultation" in text
    assert text == render_teacher_system_prompt(cfg)                       # byte-stable -> cacheable prefix
    msg = build_user_message(cfg, [{"seed_id": "S0001", "business_unit": "x"}])
    assert msg.startswith("Seeds:\n[\n{") and '"seed_id": "S0001"' in msg
