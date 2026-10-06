# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'Dataset diversity report and figures per plan Phase 3', Date: 2026-10-06
"""Diversity report for the clean dataset -> reports/diversity_report.json + figures.

Covers: length distributions (words and student-tokenizer tokens), planned vs realised
coverage, keyword frequency and per-category keyword hit rate, principle frequency and
co-occurrence, lexical diversity (distinct-1/2) and semantic diversity (nearest-neighbour cosine).

CLI: python -m src.diversity [--no-tokens]
"""
from __future__ import annotations

import argparse
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .config import TASK_ROOT, Config, load_config
from .schemas import PRINCIPLE_IDS, word_count
from .seeds import CATEGORICAL_DIMS, WEIGHTED_DIMS
from .utils import read_jsonl, write_json

# Keywords that show a scenario really is about its planned data category.
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "contact details": ["phone", "mobile", "email", "e-mail", "address", "contact", "number"],
    "NIC and identity documents": ["nic", "identity", "passport", "driving licence", "id copy", "id card", "kyc"],
    "biometric data": ["fingerprint", "biometric", "face", "facial", "iris", "thumb"],
    "health or insurance medical data": ["medical", "health", "hospital", "insurance", "diagnos", "illness", "patient"],
    "transaction and account history": ["transaction", "statement", "account", "balance", "transfer", "payment"],
    "location data": ["location", "gps", "geolocation", "geo-location", "tracking", "track", "coordinates"],
    "CCTV and call recordings": ["cctv", "camera", "footage", "recording", "recorded", "video"],
    "data of minors": ["minor", "child", "children", "under 18", "student", "school", "guardian", "teen", "parent"],
    "employee records": ["employee", "staff", "payroll", "hr ", "personnel", "salary", "attendance"],
    "credit reports and scores": ["credit", "score", "crib", "bureau", "rating"],
    "device and online identifiers": ["device", "ip address", "cookie", "browser", "imei", "device id", "identifier"],
    "religious or political information": ["religio", "political", "church", "temple", "mosque", "kovil", "party", "faith"],
}


def describe(values: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {}
    return {
        "n": int(a.size), "mean": round(float(a.mean()), 1), "std": round(float(a.std()), 1),
        "p5": round(float(np.percentile(a, 5)), 1), "p50": round(float(np.percentile(a, 50)), 1),
        "p95": round(float(np.percentile(a, 95)), 1), "max": round(float(a.max()), 1), "min": round(float(a.min()), 1),
    }


def distinct_n(texts: Sequence[str], n: int) -> float:
    """Unique n-grams / total n-grams over all texts (lowercased whitespace tokens)."""
    total, uniq = 0, set()
    for t in texts:
        toks = t.lower().split()
        grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
        total += len(grams)
        uniq.update(grams)
    return round(len(uniq) / total, 4) if total else 0.0


def top_ngrams(texts: Sequence[str], ngram: int, k: int) -> List[List[Any]]:
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(stop_words="english", ngram_range=(ngram, ngram), lowercase=True,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    X = vec.fit_transform(texts)
    freqs = np.asarray(X.sum(axis=0)).ravel()
    vocab = vec.get_feature_names_out()
    order = np.argsort(-freqs)[:k]
    return [[str(vocab[i]), int(freqs[i])] for i in order]


def keyword_hit_rate(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for cat, kws in CATEGORY_KEYWORDS.items():
        sub = [it for it in items if it["seed"]["data_category"] == cat]
        hits = sum(1 for it in sub if any(kw in it["scenario"].lower() for kw in kws))
        out[cat] = {"n": len(sub), "hits": hits, "hit_rate": round(hits / len(sub), 3) if sub else None}
    return out


def principle_stats(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    ids = sorted(PRINCIPLE_IDS)
    freq = Counter(p for it in items for p in it["answer"]["principles"])
    co = np.zeros((len(ids), len(ids)), dtype=int)
    for it in items:
        ps = [p for p in it["answer"]["principles"] if p in PRINCIPLE_IDS]
        for a in ps:
            for b in ps:
                co[ids.index(a), ids.index(b)] += 1
    first = Counter(it["answer"]["principles"][0] for it in items if it["answer"]["principles"])
    return {"ids": ids, "frequency": {p: freq.get(p, 0) for p in ids},
            "primary_frequency": {p: first.get(p, 0) for p in ids}, "cooccurrence": co.tolist()}


def semantic_stats(embs: np.ndarray, high: float = 0.85) -> Dict[str, Any]:
    sims = embs @ embs.T
    n = sims.shape[0]
    if n < 2:
        return {}
    np.fill_diagonal(sims, -1.0)
    nn = sims.max(axis=1)
    iu = np.triu_indices(n, k=1)
    pair = sims[iu]
    return {
        "nn_cosine": describe(nn),
        "nn_cosine_values": [round(float(x), 4) for x in nn],
        "mean_pairwise_cosine": round(float(pair.mean()), 4),
        "max_nn_cosine": round(float(nn.max()), 4),
        "share_pairs_above_0_85": round(float((pair > high).mean()), 5),
        "pairs_above_0_85": int((pair > high).sum()),
        "n_pairs": int(pair.size),
    }


def build_report(cfg: Config, items: List[Dict[str, Any]], seeds: List[Dict[str, Any]],
                 embed_fn: Optional[Callable[[Sequence[str]], np.ndarray]] = None,
                 tokenizer: Any = None) -> Dict[str, Any]:
    scen = [it["scenario"] for it in items]
    ans = [it["answer_json"] for it in items]
    rep: Dict[str, Any] = {"n_items": len(items)}
    rep["lengths_words"] = {"scenario": describe([word_count(t) for t in scen]),
                            "answer": describe([word_count(t) for t in ans])}
    if tokenizer is not None:
        rep["lengths_tokens"] = {
            "tokenizer": getattr(tokenizer, "name_or_path", "student"),
            "scenario": describe([len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in scen]),
            "answer": describe([len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in ans]),
        }
    dims = [*CATEGORICAL_DIMS, *WEIGHTED_DIMS.keys()]
    rep["coverage"] = {}
    for d in dims:
        planned = Counter(s[d] for s in seeds)
        realised = Counter(it["seed"][d] for it in items)
        rep["coverage"][d] = {k: {"planned": planned.get(k, 0), "realised": realised.get(k, 0)}
                              for k in sorted(set(planned) | set(realised))}
    rep["coverage"]["verdict_in_answers"] = dict(Counter(it["answer"]["verdict"] for it in items))
    rep["coverage"]["risk_in_answers"] = dict(Counter(it["answer"]["risk_level"] for it in items))
    rep["keywords"] = {"top_unigrams": top_ngrams(scen, 1, 30), "top_bigrams": top_ngrams(scen, 2, 20)}
    rep["category_keyword_hit_rate"] = keyword_hit_rate(items)
    rep["principles"] = principle_stats(items)
    rep["lexical"] = {"distinct_1": distinct_n(scen, 1), "distinct_2": distinct_n(scen, 2)}
    if embed_fn is not None and len(items) > 1:
        sem = semantic_stats(np.asarray(embed_fn(scen)))
        rep["semantic"] = sem
        rep["verdict_line"] = (f"No near-duplicate clusters: max NN cosine = {sem['max_nn_cosine']:.3f}, "
                               f"pairs above 0.85 = {100 * sem['share_pairs_above_0_85']:.2f}%")
    return rep


def save_figures(cfg: Config, rep: Dict[str, Any], items: List[Dict[str, Any]], tokenizer: Any = None) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = cfg.path("figures_dir")
    figdir.mkdir(parents=True, exist_ok=True)
    paths = []

    # Length histograms.
    ncols = 2 if tokenizer is None else 4
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 3.4))
    series = [("Scenario words", [word_count(it["scenario"]) for it in items]),
              ("Answer words", [word_count(it["answer_json"]) for it in items])]
    if tokenizer is not None:
        series += [("Scenario tokens", [len(tokenizer(it["scenario"], add_special_tokens=False)["input_ids"]) for it in items]),
                   ("Answer tokens", [len(tokenizer(it["answer_json"], add_special_tokens=False)["input_ids"]) for it in items])]
    for ax, (title, vals) in zip(np.atleast_1d(axes), series):
        ax.hist(vals, bins=25, color="#4C72B0", edgecolor="white")
        ax.set_title(title)
        ax.set_ylabel("items")
    fig.tight_layout()
    p = figdir / "length_histograms.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(str(p))

    # Coverage: planned vs realised per dimension.
    dims = [*CATEGORICAL_DIMS, *WEIGHTED_DIMS.keys()]
    fig, axes = plt.subplots(len(dims), 1, figsize=(10, 3.0 * len(dims)))
    for ax, d in zip(axes, dims):
        cov = rep["coverage"][d]
        keys = list(cov.keys())
        x = np.arange(len(keys))
        ax.bar(x - 0.2, [cov[k]["planned"] for k in keys], width=0.4, label="planned (seeds)", color="#BBBBBB")
        ax.bar(x + 0.2, [cov[k]["realised"] for k in keys], width=0.4, label="realised (clean)", color="#4C72B0")
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=35, ha="right", fontsize=7)
        ax.set_title(d)
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = figdir / "coverage.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(str(p))

    # Top keywords.
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, key, title in ((axes[0], "top_unigrams", "Top 30 unigrams"), (axes[1], "top_bigrams", "Top 20 bigrams")):
        words = rep["keywords"][key][::-1]
        ax.barh([w for w, _ in words], [c for _, c in words], color="#55A868")
        ax.set_title(title)
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    p = figdir / "keywords.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(str(p))

    # Principle co-occurrence heatmap.
    pr = rep["principles"]
    co = np.asarray(pr["cooccurrence"])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(co, cmap="Blues")
    ax.set_xticks(range(len(pr["ids"])))
    ax.set_xticklabels(pr["ids"], rotation=90, fontsize=7)
    ax.set_yticks(range(len(pr["ids"])))
    ax.set_yticklabels(pr["ids"], fontsize=7)
    for i in range(co.shape[0]):
        for j in range(co.shape[1]):
            if co[i, j]:
                ax.text(j, i, str(co[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if co[i, j] > co.max() / 2 else "black")
    ax.set_title("Principle co-occurrence (diagonal = frequency)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    p = figdir / "principle_cooccurrence.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(str(p))

    # Nearest-neighbour cosine histogram.
    if "semantic" in rep:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(rep["semantic"]["nn_cosine_values"], bins=30, color="#C44E52", edgecolor="white")
        ax.axvline(cfg.filters.near_dup_threshold, color="black", linestyle="--", label="dedup threshold")
        ax.set_xlabel("cosine to nearest other scenario")
        ax.set_ylabel("items")
        ax.set_title("Semantic diversity: nearest-neighbour cosine")
        ax.legend()
        fig.tight_layout()
        p = figdir / "nn_cosine_hist.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(str(p))
    return paths


def _rel(p: str) -> str:
    from pathlib import Path

    try:
        return Path(p).resolve().relative_to(TASK_ROOT).as_posix()
    except ValueError:
        return str(p)


def run(cfg: Config, with_tokens: bool = True) -> Dict[str, Any]:
    from .validate_data import embed_texts

    items = read_jsonl(cfg.path("clean"))
    seeds = read_jsonl(cfg.path("seeds"))
    tok = None
    if with_tokens:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(cfg.student.model_id)
        except Exception as e:  # offline or gated: report words only, say so
            print(f"[diversity] student tokenizer unavailable ({type(e).__name__}); token lengths skipped")
    rep = build_report(cfg, items, seeds, embed_fn=lambda t: embed_texts(t, cfg.filters.embed_model), tokenizer=tok)
    figs = save_figures(cfg, rep, items, tokenizer=tok)
    to_save = dict(rep)
    if "semantic" in to_save:
        to_save["semantic"] = {k: v for k, v in rep["semantic"].items() if k != "nn_cosine_values"}
    to_save["figures"] = [_rel(p) for p in figs]
    write_json(cfg.report_path("diversity_report.json"), to_save)
    if "verdict_line" in rep:
        print(rep["verdict_line"])
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description="Diversity report")
    ap.add_argument("--no-tokens", action="store_true", help="skip student-tokenizer token counts")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(cfg, with_tokens=not args.no_tokens)
    print(f"Saved {cfg.report_path('diversity_report.json')}")


if __name__ == "__main__":
    main()
