# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'Blind manual review sheet, hallucination rate with Wilson CI, judge agreement and evidence pack per plan Phases 8-9', Date: 2026-10-06
"""Manual review (human gate G5) and the evidence pack for the qualitative analysis.

    python -m src.manual_review --build      # reports/manual_review.csv (+ key file), labels left EMPTY
    python -m src.manual_review --score      # after JP labels: rates, Wilson 95% CI, kappa vs judge
    python -m src.manual_review --evidence   # reports/evidence_pack.md for JP's two paragraphs

Labels (Appendix F): correct | partially_correct | hallucinated.
error_type (semicolon separated): none; wrong_verdict; wrong_risk; missed_principle; irrelevant_principle;
invented_fact; invented_id_or_law; vague_actions; format_error; reference_wrong.
This module never fills or suggests labels.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .config import Config, load_config
from .seeds import quota_counts
from .utils import read_jsonl, write_json

LABELS = ("correct", "partially_correct", "hallucinated")
ERROR_TYPES = ("none", "wrong_verdict", "wrong_risk", "missed_principle", "irrelevant_principle", "invented_fact",
               "invented_id_or_law", "vague_actions", "format_error", "reference_wrong")
SHEET_COLUMNS = ["row_id", "example_id", "system", "scenario", "reference_json", "prediction_json",
                 "label", "error_type", "notes"]


def sheet_path(cfg: Config):
    return cfg.report_path("manual_review.csv")


def key_path(cfg: Config):
    return cfg.report_path("manual_review_key.csv")


def _read_csv(path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:          # utf-8-sig opens cleanly in Excel
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def pick_items(test: List[Dict[str, Any]], n: int, seed: int) -> List[str]:
    """n test ids stratified by verdict (proportional, largest remainder), sampled with a fixed seed."""
    by_v: Dict[str, List[str]] = defaultdict(list)
    for r in sorted(test, key=lambda x: x["id"]):
        by_v[r["meta"]["verdict"]].append(r["id"])
    quota = quota_counts({v: len(ids) for v, ids in sorted(by_v.items())}, min(n, len(test)))
    rng = random.Random(seed)
    chosen = []
    for v in sorted(by_v):
        chosen += rng.sample(by_v[v], min(quota[v], len(by_v[v])))
    return sorted(chosen)


def build_sheet(cfg: Config, force: bool = False) -> Tuple[str, str]:
    from .infer import predictions_path
    from .split_format import scenario_of

    out = sheet_path(cfg)
    if out.exists() and not force:
        existing = _read_csv(out)
        if any((r.get("label") or "").strip() for r in existing):
            raise FileExistsError(f"{out} already has labels. Refusing to overwrite human work (use force=True).")
    test = read_jsonl(cfg.split_path("test"))
    by_id = {r["id"]: r for r in test}
    mr = cfg.manual_review
    ids = pick_items(test, mr.n_items, mr.seed)
    rows = []
    for system in mr.systems:
        preds = {p["id"]: p for p in read_jsonl(predictions_path(cfg, system))}
        missing = [i for i in ids if i not in preds]
        if missing:
            raise FileNotFoundError(f"predictions for {system} missing ids {missing[:5]}; run src.infer first")
        for i in ids:
            rec = by_id[i]
            rows.append({"example_id": i, "system": system, "scenario": scenario_of(rec),
                         "reference_json": rec["messages"][2]["content"], "prediction_json": preds[i].get("raw_text", "")})
    random.Random(mr.seed).shuffle(rows)
    sheet, key = [], []
    for k, r in enumerate(rows, 1):
        row_id = f"R{k:03d}"
        key.append({"row_id": row_id, "example_id": r["example_id"], "system": r["system"]})
        sheet.append({"row_id": row_id, "example_id": r["example_id"], "system": "hidden",
                      "scenario": r["scenario"], "reference_json": r["reference_json"],
                      "prediction_json": r["prediction_json"], "label": "", "error_type": "", "notes": ""})
    _write_csv(out, sheet, SHEET_COLUMNS)
    _write_csv(key_path(cfg), key, ["row_id", "example_id", "system"])
    print(f"[review] wrote {len(sheet)} rows ({len(ids)} items x {len(mr.systems)} systems) -> {out}")
    print(f"[review] system key -> {key_path(cfg)} (do not open it until labelling is finished)")
    return str(out), str(key_path(cfg))


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if n == 0:
        return None, None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(p, 4), round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def cohen_kappa(a: List[bool], b: List[bool]) -> Optional[float]:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe == 1:
        return None
    return round((po - pe) / (1 - pe), 4)


def parse_error_types(cell: str) -> List[str]:
    return [t.strip() for t in (cell or "").replace(",", ";").split(";") if t.strip()]


def score_sheet(cfg: Config) -> Dict[str, Any]:
    sheet = _read_csv(sheet_path(cfg))
    key = {r["row_id"]: r for r in _read_csv(key_path(cfg))}
    problems = []
    rows = []
    for r in sheet:
        label = (r.get("label") or "").strip().lower()
        if not label:
            problems.append(f"{r['row_id']}: no label")
            continue
        if label not in LABELS:
            problems.append(f"{r['row_id']}: unknown label '{label}'")
            continue
        ets = parse_error_types(r.get("error_type", ""))
        bad = [e for e in ets if e not in ERROR_TYPES]
        if bad:
            problems.append(f"{r['row_id']}: unknown error_type {bad}")
        rows.append({"row_id": r["row_id"], "example_id": key[r["row_id"]]["example_id"],
                     "system": key[r["row_id"]]["system"], "label": label, "error_types": ets})
    summary: Dict[str, Any] = {"rows_in_sheet": len(sheet), "rows_labelled": len(rows),
                               "complete": len(rows) == len(sheet), "problems": problems, "systems": {}}
    judge = {(j["system"], j["id"]): j for j in read_jsonl(cfg.report_path("judge_scores.jsonl")) if j.get("status") == "ok"}
    for system in sorted({r["system"] for r in rows}):
        sub = [r for r in rows if r["system"] == system]
        n = len(sub)
        counts = Counter(r["label"] for r in sub)
        s: Dict[str, Any] = {"n_reviewed": n, "counts": {lbl: counts.get(lbl, 0) for lbl in LABELS}}
        for lbl in LABELS:
            p, lo, hi = wilson(counts.get(lbl, 0), n)
            s[f"{lbl}_rate"] = {"rate": p, "wilson95_low": lo, "wilson95_high": hi}
        s["hallucination_rate"] = s["hallucinated_rate"]
        ets = Counter(e for r in sub for e in r["error_types"])
        s["error_type_counts"] = dict(ets.most_common())
        s["wrong_verdict_count"] = ets.get("wrong_verdict", 0)
        s["reference_wrong_count"] = ets.get("reference_wrong", 0)
        pairs = [(r["label"] == "hallucinated", bool(judge[(system, r["example_id"])]["scores"]["hallucination_detected"]))
                 for r in sub if (system, r["example_id"]) in judge]
        s["judge_agreement"] = {"n_pairs": len(pairs),
                                "cohen_kappa_hallucination": cohen_kappa([a for a, _ in pairs], [b for _, b in pairs]),
                                "raw_agreement": round(sum(a == b for a, b in pairs) / len(pairs), 4) if pairs else None}
        summary["systems"][system] = s
    allpairs = []
    for r in rows:
        j = judge.get((r["system"], r["example_id"]))
        if j:
            allpairs.append((r["label"] == "hallucinated", bool(j["scores"]["hallucination_detected"])))
    summary["judge_agreement_all"] = {"n_pairs": len(allpairs),
                                      "cohen_kappa_hallucination": cohen_kappa([a for a, _ in allpairs], [b for _, b in allpairs])}
    unique_items = {r["example_id"] for r in rows}
    ref_wrong_items = {r["example_id"] for r in rows if "reference_wrong" in r["error_types"]}
    summary["teacher_label_noise_estimate"] = {"items_reviewed": len(unique_items),
                                               "items_marked_reference_wrong": len(ref_wrong_items)}
    write_json(cfg.report_path("manual_review_summary.json"), summary)
    return summary


def summary_table(summary: Dict[str, Any]) -> str:
    lines = ["| System | Reviewed | Correct | Partially correct | Hallucinated (rate, Wilson 95% CI) | Wrong verdicts | Judge kappa (hallucination) |",
             "|---|---|---|---|---|---|---|"]
    for system, s in summary.get("systems", {}).items():
        h = s["hallucination_rate"]
        ci = f"{100 * h['rate']:.1f}% [{100 * h['wilson95_low']:.1f}, {100 * h['wilson95_high']:.1f}]" if h["rate"] is not None else "n/a"
        k = s["judge_agreement"]["cohen_kappa_hallucination"]
        lines.append(f"| {system} | {s['n_reviewed']} | {s['counts']['correct']} | {s['counts']['partially_correct']} | "
                     f"{s['counts']['hallucinated']} ({ci}) | {s['wrong_verdict_count']} | {k if k is not None else 'n/a'} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evidence pack (Phase 9) - prepares evidence only; JP writes the paragraphs.
# ---------------------------------------------------------------------------
def _verdict(pred: Dict[str, Any]) -> Optional[str]:
    p = pred.get("parsed")
    return p.get("verdict") if pred.get("parse_ok") and isinstance(p, dict) else None


def evidence_pack(cfg: Config, max_items: int = 5) -> str:
    from .infer import predictions_path
    from .split_format import scenario_of

    test = {r["id"]: r for r in read_jsonl(cfg.split_path("test"))}
    base = {p["id"]: p for p in read_jsonl(predictions_path(cfg, "base_zeroshot"))}
    ft = {p["id"]: p for p in read_jsonl(predictions_path(cfg, "ft"))}
    ids = sorted(i for i in test if i in base and i in ft)
    ref_v = {i: test[i]["meta"]["verdict"] for i in ids}
    lines = ["# Evidence pack for the qualitative analysis", "",
             "Prepared by `src.manual_review --evidence`. JP writes the two paragraphs; this file only collects evidence.", ""]

    fixed = [i for i in ids if _verdict(base[i]) != ref_v[i] and _verdict(ft[i]) == ref_v[i]]
    lines += [f"## Base wrong, fine-tuned right ({len(fixed)} items; first {min(max_items, len(fixed))} shown)", ""]
    for i in fixed[:max_items]:
        lines += [f"### {i} (reference {ref_v[i]}, difficulty {test[i]['meta'].get('difficulty')})", "",
                  "Scenario:", "", *[f"> {l}" for l in scenario_of(test[i]).splitlines()], "",
                  f"Base zero-shot (verdict {_verdict(base[i])}):", "", "```", base[i].get("raw_text", "")[:1500], "```", "",
                  f"Fine-tuned (verdict {_verdict(ft[i])}):", "", "```", ft[i].get("raw_text", "")[:1500], "```", ""]

    wrong = [i for i in ids if _verdict(ft[i]) != ref_v[i]]
    labels = {}
    if sheet_path(cfg).exists() and key_path(cfg).exists():
        key = {r["row_id"]: r for r in _read_csv(key_path(cfg))}
        for r in _read_csv(sheet_path(cfg)):
            k = key.get(r["row_id"])
            if k and k["system"] == "ft" and (r.get("error_type") or "").strip():
                labels[k["example_id"]] = ";".join(parse_error_types(r["error_type"]))
    groups: Dict[str, List[str]] = defaultdict(list)
    for i in wrong:
        g = labels.get(i) or f"verdict {ref_v[i]} -> {_verdict(ft[i]) or 'INVALID'}"
        groups[g].append(i)
    lines += [f"## Fine-tuned still wrong ({len(wrong)} items), grouped by error type (manual labels) or confusion", ""]
    shown = 0
    for g, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        lines += [f"### {g}: {len(members)} items ({', '.join(members)})", ""]
        for i in members:
            if shown >= max_items:
                break
            shown += 1
            lines += [f"#### {i} (reference {ref_v[i]}, difficulty {test[i]['meta'].get('difficulty')})", "",
                      *[f"> {l}" for l in scenario_of(test[i]).splitlines()], "",
                      "Reference:", "", "```", test[i]["messages"][2]["content"], "```", "",
                      "Fine-tuned:", "", "```", ft[i].get("raw_text", "")[:1500], "```", ""]

    lines += ["## Verdict accuracy by reference verdict and by difficulty", "",
              "| Group | n | Base zero-shot | Fine-tuned |", "|---|---|---|---|"]
    for field in ("verdict", "difficulty"):
        values = sorted({test[i]["meta"].get(field) for i in ids if test[i]["meta"].get(field)})
        for v in values:
            sub = [i for i in ids if test[i]["meta"].get(field) == v]
            b = sum(_verdict(base[i]) == ref_v[i] for i in sub) / len(sub)
            f = sum(_verdict(ft[i]) == ref_v[i] for i in sub) / len(sub)
            lines.append(f"| {field}={v} | {len(sub)} | {b:.3f} | {f:.3f} |")
    text = "\n".join(lines) + "\n"
    out = cfg.report_path("evidence_pack.md")
    out.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description="Manual review tooling")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true")
    g.add_argument("--score", action="store_true")
    g.add_argument("--evidence", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.build:
        build_sheet(cfg, force=args.force)
    elif args.score:
        s = score_sheet(cfg)
        print(summary_table(s))
        if s["problems"]:
            print("Problems:", *s["problems"], sep="\n  ")
    else:
        print(evidence_pack(cfg))


if __name__ == "__main__":
    main()
