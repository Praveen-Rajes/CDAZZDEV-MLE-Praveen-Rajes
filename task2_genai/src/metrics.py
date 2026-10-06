# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'ROUGE-L, BERTScore, task metrics, paired bootstrap CIs, comparison table and figures per plan Phase 7', Date: 2026-10-06
"""Evaluation metrics for every system on the same test split.

- ROUGE-L F1 (rouge_score, use_stemmer=True) on the raw generated text vs the reference
  assistant text (the rubric metric), plus on the rationale only and on canonical
  re-serialised JSON (shows the gain is content, not formatting).
- BERTScore F1 (lang="en", rescale_with_baseline=True) on rationale + issues + actions.
  Unparseable outputs use the raw text.
- Task metrics: JSON parse rate, schema-valid rate, verdict accuracy and macro-F1, risk accuracy,
  principles micro-F1 and Jaccard, invalid principle ID rate, statute/fine citation rate.
- Paired bootstrap (10,000 resamples, seed 42): FT minus base_zeroshot and FT minus base_fewshot.

Every metric is a function of per-example component sums, so the bootstrap recomputes
ratio metrics (micro-F1, macro-F1) exactly on each resample, vectorised with numpy.

CLI: python -m src.metrics            (after src.infer; adds judge results if judge_scores.jsonl exists)
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import Config, load_config
from .schemas import JUDGE_CRITERIA, PRINCIPLE_IDS, canonical_answer_json
from .utils import read_jsonl, write_json

VERDICTS = ["COMPLIANT", "NON_COMPLIANT", "NEEDS_MORE_INFO"]
SYSTEM_ORDER = ["majority", "base_zeroshot", "base_fewshot", "ft"]
SYSTEM_LABELS = {"majority": "Majority", "base_zeroshot": "Base zero-shot", "base_fewshot": "Base 3-shot",
                 "ft": "Fine-tuned"}


# ---------------------------------------------------------------------------
# Text views of a prediction
# ---------------------------------------------------------------------------
def _schema_ok_answer(pred: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    p = pred.get("parsed")
    return p if pred.get("schema_ok") and isinstance(p, dict) else None


def pred_rationale(pred: Dict[str, Any]) -> str:
    p = pred.get("parsed")
    if isinstance(p, dict) and isinstance(p.get("rationale"), str):
        return p["rationale"]
    return pred.get("raw_text", "")


def pred_canonical(pred: Dict[str, Any]) -> str:
    a = _schema_ok_answer(pred)
    return canonical_answer_json(a) if a else pred.get("raw_text", "")


def bertscore_text_answer(a: Dict[str, Any]) -> str:
    parts = [str(a.get("rationale", ""))]
    for k in ("issues", "required_actions"):
        v = a.get(k)
        if isinstance(v, list):
            parts += [str(x) for x in v]
    return " ".join(parts).strip()


def pred_bertscore_text(pred: Dict[str, Any]) -> str:
    p = pred.get("parsed")
    if isinstance(p, dict) and pred.get("parse_ok"):
        txt = bertscore_text_answer(p)
        if txt:
            return txt
    return pred.get("raw_text", "") or " "


def pred_output_text(pred: Dict[str, Any]) -> str:
    """Text a reader sees, for citation checks: free-text fields if parsed, else the raw output."""
    from .validate_data import answer_text

    p = pred.get("parsed")
    return answer_text(p) if isinstance(p, dict) else pred.get("raw_text", "")


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------
def rouge_l(preds: Sequence[str], refs: Sequence[str]) -> List[float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]


def bertscore_f1(preds: Sequence[str], refs: Sequence[str], lang: str = "en", rescale: bool = True) -> List[float]:
    from bert_score import score

    _, _, f1 = score(list(preds), list(refs), lang=lang, rescale_with_baseline=rescale, verbose=False)
    return [float(x) for x in f1]


# ---------------------------------------------------------------------------
# Per-example components
# ---------------------------------------------------------------------------
def principle_counts(pred_ids: Sequence[str], ref_ids: Sequence[str]) -> Tuple[int, int, int, float]:
    p, r = set(pred_ids), set(ref_ids)
    tp, fp, fn = len(p & r), len(p - r), len(r - p)
    jac = tp / len(p | r) if (p | r) else 1.0
    return tp, fp, fn, jac


def example_components(pred: Optional[Dict[str, Any]], ref: Dict[str, Any], pred_verdict: Optional[str] = None) -> Dict[str, float]:
    """Numbers whose sums define every task metric. `pred` is a prediction row (None for majority,
    which passes only pred_verdict)."""
    from .validate_data import statute_hits

    c: Dict[str, float] = {"one": 1.0}
    if pred is not None:
        parsed = pred.get("parsed") if isinstance(pred.get("parsed"), dict) else {}
        c["parse_ok"] = float(bool(pred.get("parse_ok")))
        c["schema_ok"] = float(bool(pred.get("schema_ok")))
        pred_verdict = parsed.get("verdict") if pred.get("parse_ok") else None
        c["risk_correct"] = float(pred.get("parse_ok", False) and parsed.get("risk_level") == ref["risk_level"])
        pids = parsed.get("principles") if isinstance(parsed.get("principles"), list) else []
        pids = [str(x) for x in pids]
        tp, fp, fn, jac = principle_counts(pids, ref["principles"])
        c.update({"p_tp": tp, "p_fp": fp, "p_fn": fn, "p_jaccard": jac if pred.get("parse_ok") else 0.0})
        invalid = any(p not in PRINCIPLE_IDS for p in pids)
        statute = bool(statute_hits(pred_output_text(pred)))
        c.update({"invalid_id": float(invalid), "statute": float(statute), "invented": float(invalid or statute),
                  "strict_ok": float(not pred.get("strict_violations")),
                  "gen_tokens": float(pred.get("generated_tokens") or 0)})
        ppl = pred.get("perplexity")
        c["ppl"] = float(ppl) if ppl is not None and math.isfinite(ppl) else float("nan")
    c["verdict_correct"] = float(pred_verdict == ref["verdict"])
    for v in VERDICTS:
        c[f"v_tp_{v}"] = float(pred_verdict == v and ref["verdict"] == v)
        c[f"v_fp_{v}"] = float(pred_verdict == v and ref["verdict"] != v)
        c[f"v_fn_{v}"] = float(pred_verdict != v and ref["verdict"] == v)
    return c


# ---------------------------------------------------------------------------
# Metric definitions over component sums (work on scalars or (B,) arrays)
# ---------------------------------------------------------------------------
def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.asarray(b) > 0, np.asarray(a, dtype=float) / np.where(np.asarray(b) > 0, b, 1), np.nan)


def _macro_f1(S: Dict[str, Any]):
    f1s = []
    for v in VERDICTS:
        tp, fp, fn = S[f"v_tp_{v}"], S[f"v_fp_{v}"], S[f"v_fn_{v}"]
        f1s.append(_div(2 * tp, 2 * tp + fp + fn))   # class absent from refs and preds -> nan, skipped
    return np.nanmean(np.vstack([np.atleast_1d(f) for f in f1s]), axis=0)


def _mean(key: str) -> Callable[[Dict[str, Any]], Any]:
    return lambda S: _div(S[key], S["one"])


METRICS: Dict[str, Tuple[str, Callable[[Dict[str, Any]], Any]]] = {
    # name: (component that must exist, function)
    "rougeL_full": ("rougeL_full", _mean("rougeL_full")),
    "rougeL_rationale": ("rougeL_rationale", _mean("rougeL_rationale")),
    "rougeL_canonical": ("rougeL_canonical", _mean("rougeL_canonical")),
    "bertscore_f1": ("bertscore_f1", _mean("bertscore_f1")),
    "json_valid": ("parse_ok", _mean("parse_ok")),
    "schema_valid": ("schema_ok", _mean("schema_ok")),
    "verdict_accuracy": ("verdict_correct", _mean("verdict_correct")),
    "verdict_macro_f1": ("v_tp_COMPLIANT", _macro_f1),
    "risk_accuracy": ("risk_correct", _mean("risk_correct")),
    "principles_micro_f1": ("p_tp", lambda S: _div(2 * S["p_tp"], 2 * S["p_tp"] + S["p_fp"] + S["p_fn"])),
    "principles_jaccard": ("p_jaccard", _mean("p_jaccard")),
    "invalid_principle_rate": ("invalid_id", _mean("invalid_id")),
    "statute_citation_rate": ("statute", _mean("statute")),
    "invented_id_or_statute_rate": ("invented", _mean("invented")),
    "strict_rule_ok_rate": ("strict_ok", _mean("strict_ok")),
    "mean_generated_tokens": ("gen_tokens", _mean("gen_tokens")),
}


def component_matrix(rows: List[Dict[str, float]]) -> Dict[str, np.ndarray]:
    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows], dtype=float) for k in keys}


def point_metrics(comp: Dict[str, np.ndarray]) -> Dict[str, Optional[float]]:
    S = {k: v.sum() for k, v in comp.items() if k != "ppl"}
    out: Dict[str, Optional[float]] = {}
    for name, (need, fn) in METRICS.items():
        if need in comp:
            val = float(np.asarray(fn(S)).ravel()[0])
            out[name] = None if math.isnan(val) else round(val, 4)
    if "ppl" in comp:
        ppl = comp["ppl"][np.isfinite(comp["ppl"])]
        out["mean_perplexity"] = round(float(ppl.mean()), 4) if ppl.size else None
    return out


def paired_bootstrap(comp_a: Dict[str, np.ndarray], comp_b: Dict[str, np.ndarray], n_resamples: int,
                     seed: int) -> Dict[str, Dict[str, Optional[float]]]:
    """Mean difference (a minus b) and 95% percentile CI for every metric both systems have.
    Same resampled indices for both systems (paired)."""
    n = len(comp_a["one"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    keys = [k for k in comp_a if k in comp_b and k != "ppl"]       # ppl may hold NaN; not a summed metric
    Sa = {k: comp_a[k][idx].sum(axis=1) for k in keys}
    Sb = {k: comp_b[k][idx].sum(axis=1) for k in keys}
    full_a = {k: comp_a[k].sum() for k in keys}
    full_b = {k: comp_b[k].sum() for k in keys}
    out = {}
    for name, (need, fn) in METRICS.items():
        if need not in keys:
            continue
        diffs = np.asarray(fn(Sa), dtype=float) - np.asarray(fn(Sb), dtype=float)
        point = float(np.asarray(fn(full_a)).ravel()[0] - np.asarray(fn(full_b)).ravel()[0])
        lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
        out[name] = {"diff": None if math.isnan(point) else round(point, 4),
                     "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
                     "boot_mean_diff": round(float(np.nanmean(diffs)), 4)}
    return out


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------
def load_references(cfg: Config, split: str = "test") -> List[Dict[str, Any]]:
    import json

    recs = read_jsonl(cfg.split_path(split))
    for r in recs:
        r["ref_answer"] = json.loads(r["messages"][2]["content"])
        r["ref_text"] = r["messages"][2]["content"]
    return recs


def evaluate_systems(cfg: Config, systems: Optional[List[str]] = None, with_bertscore: bool = True) -> Dict[str, Any]:
    from .infer import predictions_path

    refs = load_references(cfg, "test")
    ids = [r["id"] for r in refs]
    by_id = {r["id"]: r for r in refs}
    systems = systems or ["base_zeroshot", "base_fewshot", "ft"]
    results: Dict[str, Any] = {"test_size": len(refs), "systems": {}, "comparisons": {}, "confusion": {},
                               "per_verdict_accuracy": {}, "per_difficulty_accuracy": {}}
    comps: Dict[str, Dict[str, np.ndarray]] = {}
    preds_by_system: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # Majority baseline: the most common train verdict for every test item.
    train_verdicts = Counter(r["meta"]["verdict"] for r in read_jsonl(cfg.split_path("train")))
    majority = train_verdicts.most_common(1)[0][0]
    comps["majority"] = component_matrix([example_components(None, by_id[i]["ref_answer"], pred_verdict=majority) for i in ids])
    results["majority_verdict"] = majority

    for system in systems:
        rows = {r["id"]: r for r in read_jsonl(predictions_path(cfg, system))}
        if set(rows) != set(ids):
            print(f"[metrics] {system}: predictions missing or incomplete ({len(rows)}/{len(ids)}); skipped")
            continue
        preds_by_system[system] = rows
        ordered = [rows[i] for i in ids]
        comp_rows = [example_components(p, by_id[i]["ref_answer"]) for p, i in zip(ordered, ids)]
        rf = rouge_l([p.get("raw_text", "") for p in ordered], [by_id[i]["ref_text"] for i in ids])
        rr = rouge_l([pred_rationale(p) for p in ordered], [by_id[i]["ref_answer"]["rationale"] for i in ids])
        rc = rouge_l([pred_canonical(p) for p in ordered], [by_id[i]["ref_text"] for i in ids])
        for c, a, b, d in zip(comp_rows, rf, rr, rc):
            c.update({"rougeL_full": a, "rougeL_rationale": b, "rougeL_canonical": d})
        if with_bertscore:
            bs = bertscore_f1([pred_bertscore_text(p) for p in ordered],
                              [bertscore_text_answer(by_id[i]["ref_answer"]) for i in ids],
                              cfg.metrics.bertscore_lang, cfg.metrics.bertscore_rescale)
            for c, v in zip(comp_rows, bs):
                c["bertscore_f1"] = v
        comps[system] = component_matrix(comp_rows)
        results["confusion"][system] = confusion(ordered, [by_id[i]["ref_answer"]["verdict"] for i in ids])
        results["per_verdict_accuracy"][system] = group_accuracy(comps[system], [by_id[i]["meta"]["verdict"] for i in ids])
        results["per_difficulty_accuracy"][system] = group_accuracy(comps[system], [by_id[i]["meta"].get("difficulty") for i in ids])

    for system, comp in comps.items():
        results["systems"][system] = point_metrics(comp)
    B, seed = cfg.metrics.bootstrap_resamples, cfg.metrics.bootstrap_seed
    if "ft" in comps:
        for other in ("base_zeroshot", "base_fewshot"):
            if other in comps:
                results["comparisons"][f"ft_minus_{other}"] = paired_bootstrap(comps["ft"], comps[other], B, seed)
    results["settings"] = {"bootstrap_resamples": B, "bootstrap_seed": seed, "rouge": "rougeL F1, use_stemmer=True",
                           "bertscore": f"lang={cfg.metrics.bertscore_lang}, rescale_with_baseline={cfg.metrics.bertscore_rescale}",
                           "decoding": f"greedy, max_new_tokens={cfg.inference.max_new_tokens}"}
    return results


def confusion(preds: List[Dict[str, Any]], ref_verdicts: List[str]) -> Dict[str, Any]:
    labels = VERDICTS + ["INVALID"]
    m = np.zeros((3, 4), dtype=int)
    for p, rv in zip(preds, ref_verdicts):
        parsed = p.get("parsed") if isinstance(p.get("parsed"), dict) else {}
        pv = parsed.get("verdict") if p.get("parse_ok") else None
        col = labels.index(pv) if pv in VERDICTS else 3
        m[VERDICTS.index(rv), col] += 1
    return {"rows_true": VERDICTS, "cols_pred": labels, "matrix": m.tolist()}


def group_accuracy(comp: Dict[str, np.ndarray], groups: List[Optional[str]]) -> Dict[str, Dict[str, float]]:
    out = {}
    for g in sorted({x for x in groups if x is not None}):
        mask = np.array([x == g for x in groups])
        out[g] = {"n": int(mask.sum()), "accuracy": round(float(comp["verdict_correct"][mask].mean()), 4)}
    return out


# ---------------------------------------------------------------------------
# Judge results (added after src.judge runs)
# ---------------------------------------------------------------------------
def judge_results(cfg: Config) -> Dict[str, Any]:
    rows = [r for r in read_jsonl(cfg.report_path("judge_scores.jsonl"))]
    if not rows:
        return {}
    out: Dict[str, Any] = {"model": cfg.judge.model, "systems": {}, "comparisons": {}}
    by_sys: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        by_sys.setdefault(r["system"], {})[r["id"]] = r
    for system, items in by_sys.items():
        ok = [r for r in items.values() if r.get("status") == "ok"]
        excl = [r for r in items.values() if r.get("status") != "ok"]
        s: Dict[str, Any] = {"n_judged": len(ok), "n_excluded": len(excl)}
        if ok:
            for c in JUDGE_CRITERIA:
                s[c] = round(float(np.mean([r["scores"][c] for r in ok])), 3)
            s["overall"] = round(float(np.mean([r["overall"] for r in ok])), 3)
            s["hallucination_detected_rate"] = round(float(np.mean([r["scores"]["hallucination_detected"] for r in ok])), 4)
        out["systems"][system] = s
    for other in ("base_zeroshot", "base_fewshot"):
        if "ft" in by_sys and other in by_sys:
            common = sorted(i for i in by_sys["ft"] if i in by_sys[other]
                            and by_sys["ft"][i].get("status") == "ok" and by_sys[other][i].get("status") == "ok")
            if len(common) < 2:
                continue
            def comp(sys_name: str) -> Dict[str, np.ndarray]:
                return {"one": np.ones(len(common)),
                        "overall": np.array([by_sys[sys_name][i]["overall"] for i in common], dtype=float),
                        "halluc": np.array([float(by_sys[sys_name][i]["scores"]["hallucination_detected"]) for i in common])}
            a, b = comp("ft"), comp(other)
            rng = np.random.default_rng(cfg.metrics.bootstrap_seed)
            idx = rng.integers(0, len(common), size=(cfg.metrics.bootstrap_resamples, len(common)))
            res = {"n_common": len(common)}
            for key in ("overall", "halluc"):
                d = a[key][idx].mean(1) - b[key][idx].mean(1)
                lo, hi = np.percentile(d, [2.5, 97.5])
                res[key] = {"diff": round(float(a[key].mean() - b[key].mean()), 4),
                            "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4)}
            out["comparisons"][f"ft_minus_{other}"] = res
    return out


# ---------------------------------------------------------------------------
# Table and figures
# ---------------------------------------------------------------------------
TABLE_ROWS = [
    ("ROUGE-L F1 (full output)", "rougeL_full", "f3"),
    ("ROUGE-L F1 (rationale)", "rougeL_rationale", "f3"),
    ("ROUGE-L F1 (canonical JSON)", "rougeL_canonical", "f3"),
    ("BERTScore F1 (rescaled)", "bertscore_f1", "f3"),
    ("Judge overall (1 to 5)", "judge_overall", "f2"),
    ("JSON valid %", "json_valid", "pct"),
    ("Schema valid %", "schema_valid", "pct"),
    ("Verdict accuracy", "verdict_accuracy", "f3"),
    ("Verdict macro-F1", "verdict_macro_f1", "f3"),
    ("Risk accuracy", "risk_accuracy", "f3"),
    ("Principles micro-F1", "principles_micro_f1", "f3"),
    ("Principles Jaccard", "principles_jaccard", "f3"),
    ("Invented ID or statute %", "invented_id_or_statute_rate", "pct"),
]


def _fmt(v: Optional[float], kind: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if kind == "pct":
        return f"{100 * v:.1f}%"
    if kind == "f2":
        return f"{v:.2f}"
    return f"{v:.3f}"


def _fmt_ci(c: Optional[Dict[str, Any]], kind: str) -> str:
    if not c or c.get("diff") is None:
        return "n/a"
    scale, suffix, nd = (100.0, " pp", 1) if kind == "pct" else (1.0, "", 3 if kind == "f3" else 2)

    def s(x: float) -> str:
        return f"{round(x * scale, nd) + 0.0:+.{nd}f}"        # + 0.0 turns -0.0 into 0.0 (no "-0.000")

    return f"{s(c['diff'])}{suffix} [{s(c['ci_low'])}, {s(c['ci_high'])}]"


def comparison_table(res: Dict[str, Any]) -> str:
    """Markdown comparison table rendered from eval_results.json."""
    judge = res.get("judge", {})
    systems = res.get("systems", {})
    header = ("| Metric | Majority | Base zero-shot | Base 3-shot | Fine-tuned | "
              "FT minus zero-shot (95% CI) | FT minus 3-shot (95% CI) |")
    lines = [header, "|---|---|---|---|---|---|---|"]
    for label, key, kind in TABLE_ROWS:
        cells = []
        for s in SYSTEM_ORDER:
            if key == "judge_overall":
                v = judge.get("systems", {}).get(s, {}).get("overall")
            else:
                v = systems.get(s, {}).get(key)
            cells.append(_fmt(v, kind))
        cis = []
        for other in ("base_zeroshot", "base_fewshot"):
            if key == "judge_overall":
                cis.append(_fmt_ci(judge.get("comparisons", {}).get(f"ft_minus_{other}", {}).get("overall"), kind))
            else:
                cis.append(_fmt_ci(res.get("comparisons", {}).get(f"ft_minus_{other}", {}).get(key), kind))
        lines.append(f"| {label} | " + " | ".join(cells) + " | " + " | ".join(cis) + " |")
    return "\n".join(lines)


def save_figures(cfg: Config, res: Dict[str, Any]) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    keys = [("rougeL_full", "ROUGE-L full"), ("rougeL_rationale", "ROUGE-L rationale"), ("bertscore_f1", "BERTScore"),
            ("json_valid", "JSON valid"), ("verdict_accuracy", "Verdict acc"), ("verdict_macro_f1", "Verdict macro-F1"),
            ("principles_micro_f1", "Principles F1")]
    systems = [s for s in SYSTEM_ORDER if s in res["systems"]]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    width = 0.8 / max(len(systems), 1)
    x = np.arange(len(keys))
    colors = {"majority": "#BBBBBB", "base_zeroshot": "#8172B3", "base_fewshot": "#64B5CD", "ft": "#C44E52"}
    for j, s in enumerate(systems):
        vals = [res["systems"][s].get(k) for k, _ in keys]
        ax.bar(x + j * width - 0.4 + width / 2, [v if v is not None else 0 for v in vals], width,
               label=SYSTEM_LABELS[s], color=colors[s])
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in keys])
    ax.set_ylim(min(0, ax.get_ylim()[0]), 1.0)
    ax.set_title(f"Test set (n = {res['test_size']}): key metrics by system")
    ax.legend(fontsize=8, ncol=len(systems), loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = cfg.figure_path("metrics_bar.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(str(p))

    conf_systems = [s for s in ("base_zeroshot", "base_fewshot", "ft") if s in res["confusion"]]
    if conf_systems:
        fig, axes = plt.subplots(1, len(conf_systems), figsize=(4.6 * len(conf_systems), 3.9))
        for ax, s in zip(np.atleast_1d(axes), conf_systems):
            c = res["confusion"][s]
            m = np.array(c["matrix"])
            ax.imshow(m, cmap="Blues")
            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    ax.text(j, i, str(m[i, j]), ha="center", va="center",
                            color="white" if m[i, j] > m.max() / 2 else "black")
            ax.set_xticks(range(4))
            ax.set_xticklabels(["COMP", "NON", "NMI", "INVALID"], fontsize=8)
            ax.set_yticks(range(3))
            ax.set_yticklabels(["COMP", "NON", "NMI"], fontsize=8)
            ax.set_xlabel("predicted")
            ax.set_ylabel("reference")
            ax.set_title(SYSTEM_LABELS[s])
        fig.tight_layout()
        p = cfg.figure_path("confusion_matrices.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(str(p))
    return paths


def run(cfg: Config, with_bertscore: bool = True) -> Dict[str, Any]:
    res = evaluate_systems(cfg, with_bertscore=with_bertscore)
    j = judge_results(cfg)
    if j:
        res["judge"] = j
    save_figures(cfg, res)
    write_json(cfg.report_path("eval_results.json"), res)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute eval_results.json from saved predictions")
    ap.add_argument("--no-bertscore", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = run(cfg, with_bertscore=not args.no_bertscore)
    print(comparison_table(res))


if __name__ == "__main__":
    main()
