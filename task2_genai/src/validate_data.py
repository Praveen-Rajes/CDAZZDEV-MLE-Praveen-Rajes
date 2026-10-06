# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'Schema, leakage, forbidden-content, dedup filters and audit cards per plan Phase 3', Date: 2026-10-06
"""Clean the accepted teacher items. Filters run in this order and each logs its counts:

1. Schema and consistency (re-check, including seed checks).
2. Leakage: the scenario must not contain a principle ID or the verdict words.
3. Forbidden content: statute/fine citations in answers; real bank names; full NIC numbers.
4. Exact duplicates (normalised text hash).
5. Near duplicates (all-MiniLM-L6-v2 cosine >= 0.92; the later item is dropped).

CLI:
    python -m src.validate_data                 # run filters -> data/clean/all.jsonl, reports/filter_report.json
    python -m src.validate_data --audit 30      # write data/audit_log.md (30 cards for JP, gate G3)
    python -m src.validate_data --audit-summary # count JP's audit answers -> reports/audit_summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import random
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .config import Config, load_config
from .generate_data import check_item
from .schemas import ANSWER_KEYS, canonical_answer_json
from .utils import read_jsonl, write_json, write_jsonl

# 2. Leakage in the scenario.
LEAK_PRINCIPLE_RE = re.compile(r"\bP\d{2}\b")
LEAK_WORDS_RE = re.compile(r"\b(non-compliant|compliant|verdict)\b|NEEDS_MORE_INFO", re.IGNORECASE)

# 3. Forbidden content. Statute, fine and penalty citations (checked in answers and model outputs).
STATUTE_PATTERNS = [
    re.compile(r"\bsection\s+\d+", re.IGNORECASE),
    re.compile(r"\bs\.\s?\d+", re.IGNORECASE),
    re.compile(r"\bAct\s+No\b", re.IGNORECASE),
    re.compile(r"\bRs\.?\s?\d[\d,]*.*\bfine[sd]?\b", re.IGNORECASE),
    re.compile(r"\bpenalt(?:y|ies)\b", re.IGNORECASE),
]

# Licensed commercial banks in Sri Lanka (Central Bank of Sri Lanka list) and common short forms.
# Matched case-sensitively on word boundaries so "a licensed commercial bank" is not a hit.
# "Sampath" alone is left out on purpose: it is also a common Sinhala first name.
REAL_BANK_NAMES = [
    "Amana Bank", "Bank of Ceylon", "BOC", "Bank of China", "Cargills Bank", "Citibank", "Citi Bank",
    "Commercial Bank of Ceylon", "Commercial Bank", "ComBank", "Deutsche Bank", "DFCC Bank", "DFCC",
    "Habib Bank", "Hatton National Bank", "HNB", "Indian Bank", "Indian Overseas Bank", "MCB Bank",
    "National Development Bank", "NDB", "Nations Trust Bank", "NTB", "Pan Asia Banking Corporation",
    "Pan Asia Bank", "People's Bank", "Peoples Bank", "Public Bank Berhad", "Public Bank", "Sampath Bank",
    "Seylan Bank", "Seylan", "Standard Chartered", "State Bank of India", "HSBC",
    "Hongkong and Shanghai Banking Corporation", "Union Bank of Colombo", "Union Bank", "ICICI Bank", "Axis Bank",
]
BANK_RE = re.compile(r"\b(" + "|".join(re.escape(n) for n in sorted(REAL_BANK_NAMES, key=len, reverse=True)) + r")\b")

# Full-length NIC numbers (old 9 digits + letter, new 12 digits).
NIC_PATTERNS = [re.compile(r"\b\d{9}[VvXx]\b"), re.compile(r"\b\d{12}\b")]


def answer_text(answer: Dict[str, Any]) -> str:
    """All free-text fields of an answer joined (what a reader would see)."""
    parts = [str(answer.get("rationale", ""))]
    for k in ("issues", "missing_information", "required_actions"):
        parts.extend(str(x) for x in (answer.get(k) or []))
    return "\n".join(parts)


def statute_hits(text: str) -> List[str]:
    return [p.pattern for p in STATUTE_PATTERNS if p.search(text)]


def leakage_hits(scenario: str) -> List[str]:
    hits = LEAK_PRINCIPLE_RE.findall(scenario)
    hits += [m.group(0) for m in LEAK_WORDS_RE.finditer(scenario)]
    return hits


def forbidden_hits(scenario: str, answer: Dict[str, Any]) -> List[str]:
    hits = [f"statute:{p}" for p in statute_hits(answer_text(answer))]
    for text, where in ((scenario, "scenario"), (answer_text(answer), "answer")):
        hits += [f"bank_name:{m.group(0)}@{where}" for m in BANK_RE.finditer(text)]
        for p in NIC_PATTERNS:
            if p.search(text):
                hits.append(f"full_nic@{where}")
    return hits


def normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalise(text).encode("utf-8")).hexdigest()


def embed_texts(texts: Sequence[str], model_name: str):
    """L2-normalised sentence embeddings (numpy array)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(list(texts), batch_size=64, show_progress_bar=False, normalize_embeddings=True)


def near_duplicate_drop(embs, threshold: float) -> Tuple[List[int], List[Tuple[int, int, float]]]:
    """Greedy in input order: keep an item unless its cosine to an already kept item >= threshold.
    Returns (kept indices, [(dropped, matched_kept, cosine)])."""
    import numpy as np

    kept: List[int] = []
    dropped: List[Tuple[int, int, float]] = []
    for i in range(len(embs)):
        if kept:
            sims = embs[kept] @ embs[i]
            j = int(np.argmax(sims))
            if float(sims[j]) >= threshold:
                dropped.append((i, kept[j], float(sims[j])))
                continue
        kept.append(i)
    return kept, dropped


def to_clean_record(row: Dict[str, Any]) -> Dict[str, Any]:
    answer = {k: row["answer"][k] for k in ANSWER_KEYS}
    return {
        "id": row["seed_id"],
        "scenario": row["scenario"],
        "answer": answer,
        "answer_json": canonical_answer_json(answer),
        "seed": row["seed"],
        "teacher_model": row["teacher_model"],
        "call_id": row.get("call_id"),
    }


def run_filters(cfg: Config, rows: List[Dict[str, Any]],
                embed_fn: Optional[Callable[[Sequence[str]], Any]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply filters 1 to 5 in order. Returns (clean records, report)."""
    report: Dict[str, Any] = {"input": len(rows), "stages": []}
    dropped_log: List[Dict[str, Any]] = []

    def stage(name: str, before: int, after: int, reasons: Counter) -> None:
        report["stages"].append({"stage": name, "in": before, "dropped": before - after, "out": after,
                                 "reasons": dict(reasons.most_common())})
        print(f"[filter] {name}: in={before} dropped={before - after} out={after}")

    # 1. Schema and consistency.
    keep, reasons = [], Counter()
    for r in rows:
        item = {"seed_id": r["seed_id"], "scenario": r["scenario"], "answer": r["answer"]}
        ok, reason, _ = check_item(item, r["seed"], cfg)
        if ok:
            keep.append(r)
        else:
            reasons[reason.split(":", 1)[0]] += 1
            dropped_log.append({"id": r["seed_id"], "stage": "schema_consistency", "reason": reason})
    stage("1_schema_consistency", len(rows), len(keep), reasons)
    rows = keep

    # 2. Leakage.
    keep, reasons = [], Counter()
    for r in rows:
        hits = leakage_hits(r["scenario"])
        if hits:
            reasons.update(h.lower() for h in hits)
            dropped_log.append({"id": r["seed_id"], "stage": "leakage", "reason": ",".join(hits)})
        else:
            keep.append(r)
    stage("2_leakage", len(rows), len(keep), reasons)
    rows = keep

    # 3. Forbidden content.
    keep, reasons = [], Counter()
    for r in rows:
        hits = forbidden_hits(r["scenario"], r["answer"])
        if hits:
            reasons.update(h.split(":")[0].split("@")[0] for h in hits)
            dropped_log.append({"id": r["seed_id"], "stage": "forbidden", "reason": ",".join(hits)})
        else:
            keep.append(r)
    stage("3_forbidden_content", len(rows), len(keep), reasons)
    rows = keep

    # 4. Exact duplicates.
    keep, seen, reasons = [], {}, Counter()
    for r in rows:
        h = text_hash(r["scenario"])
        if h in seen:
            reasons["exact_duplicate"] += 1
            dropped_log.append({"id": r["seed_id"], "stage": "exact_dup", "reason": f"same as {seen[h]}"})
        else:
            seen[h] = r["seed_id"]
            keep.append(r)
    stage("4_exact_duplicates", len(rows), len(keep), reasons)
    rows = keep

    # 5. Near duplicates.
    if rows:
        embed_fn = embed_fn or (lambda texts: embed_texts(texts, cfg.filters.embed_model))
        embs = embed_fn([r["scenario"] for r in rows])
        kept_idx, dropped = near_duplicate_drop(embs, cfg.filters.near_dup_threshold)
        for i, j, sim in dropped:
            dropped_log.append({"id": rows[i]["seed_id"], "stage": "near_dup",
                                "reason": f"cosine {sim:.3f} to {rows[j]['seed_id']}"})
        reasons = Counter({"near_duplicate": len(dropped)}) if dropped else Counter()
        stage("5_near_duplicates", len(rows), len(kept_idx), reasons)
        rows = [rows[i] for i in kept_idx]
    else:
        stage("5_near_duplicates", 0, 0, Counter())

    clean = [to_clean_record(r) for r in rows]
    report["output"] = len(clean)
    report["output_by_verdict"] = dict(Counter(c["answer"]["verdict"] for c in clean))
    report["output_by_teacher"] = dict(Counter(c["teacher_model"] for c in clean))
    report["dropped"] = dropped_log
    return clean, report


# ---------------------------------------------------------------------------
# Audit (gate G3)
# ---------------------------------------------------------------------------
AUDIT_FIELDS = ("Label agrees? (yes/no)", "Scenario realistic? (yes/no)", "Errors (none / describe)", "JP verdict / notes")


def write_audit(cfg: Config, n: int, force: bool = False) -> str:
    """Write n random clean items as readable cards with empty answer lines for JP."""
    out = cfg.path("audit_log")
    if out.exists() and not force:
        raise FileExistsError(f"{out} exists and may hold JP's notes. Pass force=True (--force) to overwrite.")
    clean = read_jsonl(cfg.path("clean"))
    rng = random.Random(cfg.project.seed)
    sample = rng.sample(clean, min(n, len(clean)))
    lines = [
        "# Data audit log (gate G3)",
        "",
        f"{len(sample)} items sampled at random (seed {cfg.project.seed}) from `data/clean/all.jsonl`.",
        "JP: read every card and fill the four lines under it. If more than 3 of 30 labels are wrong,",
        "fix the teacher prompt and regenerate the affected slice. Then run",
        "`python -m src.validate_data --audit-summary` to count the answers.",
        "",
    ]
    for k, it in enumerate(sample, 1):
        a, s = it["answer"], it["seed"]
        lines += [
            f"## {k}. {it['id']}",
            "",
            f"Seed: {s['business_unit']} | {s['data_category']} | {s['processing_activity']} | "
            f"{s['artifact_type']} | target {s['target_verdict']} | {s['difficulty']} | {s['length_bucket']}",
            "",
            "**Scenario**",
            "",
            *[f"> {line}" if line else ">" for line in it["scenario"].splitlines()],
            "",
            f"**Verdict:** {a['verdict']} | **Risk:** {a['risk_level']} | **Principles:** {', '.join(a['principles'])}",
            "",
            f"**Rationale:** {a['rationale']}",
            "",
            "**Issues:** " + ("; ".join(a["issues"]) or "(none)"),
            "",
            "**Missing information:** " + ("; ".join(a["missing_information"]) or "(none)"),
            "",
            "**Required actions:** " + "; ".join(a["required_actions"]),
            "",
            *[f"- {f}: " for f in AUDIT_FIELDS],
            "",
            "---",
            "",
        ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    return str(out)


def audit_summary(cfg: Config) -> Dict[str, Any]:
    """Count JP's yes/no answers in the audit log. Unanswered lines are counted as blank."""
    text = cfg.path("audit_log").read_text(encoding="utf-8")
    counts: Dict[str, Counter] = {f: Counter() for f in AUDIT_FIELDS[:2]}
    for f in AUDIT_FIELDS[:2]:
        for m in re.finditer(r"^- " + re.escape(f) + r":[ \t]*(.*)$", text, flags=re.MULTILINE):
            ans = m.group(1).strip().lower()
            counts[f]["yes" if ans.startswith("y") else "no" if ans.startswith("n") else "blank"] += 1
    errors = [m.group(1).strip() for m in re.finditer(r"^- " + re.escape(AUDIT_FIELDS[2]) + r":[ \t]*(.*)$", text, re.MULTILINE)]
    n_items = len(re.findall(r"^## \d+\. ", text, flags=re.MULTILINE))
    summary = {
        "items": n_items,
        "label_agrees": dict(counts[AUDIT_FIELDS[0]]),
        "scenario_realistic": dict(counts[AUDIT_FIELDS[1]]),
        "items_with_errors_noted": sum(1 for e in errors if e and e.lower() not in ("none", "no", "-")),
        "complete": counts[AUDIT_FIELDS[0]].get("blank", 0) == 0 and n_items > 0,
    }
    summary["wrong_labels"] = counts[AUDIT_FIELDS[0]].get("no", 0)
    summary["gate_passed"] = summary["complete"] and summary["wrong_labels"] <= 3
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean, audit and summarise the generated data")
    ap.add_argument("--audit", type=int, default=None, help="write N audit cards to data/audit_log.md")
    ap.add_argument("--force", action="store_true", help="overwrite an existing audit log")
    ap.add_argument("--audit-summary", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.audit is not None:
        print(f"Wrote audit cards -> {write_audit(cfg, args.audit, force=args.force)}")
        return
    if args.audit_summary:
        s = audit_summary(cfg)
        write_json(cfg.report_path("audit_summary.json"), s)
        print(s)
        return
    rows = read_jsonl(cfg.path("generations"))
    clean, report = run_filters(cfg, rows)
    write_jsonl(cfg.path("clean"), clean)
    write_json(cfg.report_path("filter_report.json"), report)
    print(f"Clean items: {len(clean)} -> {cfg.path('clean')}")


if __name__ == "__main__":
    main()
