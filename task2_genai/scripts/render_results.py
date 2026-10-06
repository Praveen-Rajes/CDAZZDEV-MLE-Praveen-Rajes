# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'Render README result tables from reports/*.json between markers per plan Phase 11', Date: 2026-10-06
"""Render every number in the READMEs from JSON/CSV files in task2_genai/reports/.

Writes between markers:
- task2_genai/README.md:       <!-- RESULTS:START --> ... <!-- RESULTS:END -->
- task2_genai/data/README.md:  <!-- DATACARD:START --> ... <!-- DATACARD:END -->

Sections whose source file does not exist yet are rendered as "Pending: <what to run>".

Usage: python scripts/render_results.py [--print]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Optional

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT))

from src.config import load_config  # noqa: E402
from src.utils import read_json, write_json  # noqa: E402

cfg = load_config()
R = cfg.path("reports_dir")


def _load(name: str) -> Optional[Any]:
    p = R / name
    return read_json(p) if p.exists() else None


def _pending(what: str) -> str:
    return f"_Pending: {what}._\n"


def _pct(x: Optional[float], nd: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{nd}f}%"


def _img(rel: str, alt: str) -> str:
    return f"![{alt}](reports/figures/{rel})" if (R / "figures" / rel).exists() else ""


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def generation_section() -> str:
    from src.generate_data import generation_log_summary

    if cfg.path("generations").exists():
        write_json(R / "generation_summary.json", generation_log_summary(cfg))
    g = _load("generation_summary.json")
    if not g:
        return _pending("teacher generation (notebook Section 2A.4)")
    lines = ["| Item | Value |", "|---|---|",
             f"| Accepted teacher items | {g['accepted']} |",
             f"| Rejected teacher items | {g['rejected']} |",
             f"| Accepted by teacher model | {', '.join(f'{k}: {v}' for k, v in g['accepted_by_teacher'].items())} |",
             f"| Teacher API calls (all statuses) | {g['api_calls_total']} |",
             f"| Calls with cached prompt tokens | {g['calls_with_cached_tokens']} of {g['successful_calls']} successful |",
             f"| Prompt cache hit rate (cached / prompt tokens) | {_pct(g['cache_hit_rate'])} |",
             f"| Uncached tokens spent | {g['uncached_tokens_total']:,} |"]
    if g["rejected_by_reason"]:
        lines += ["", "Rejections by reason: " + "; ".join(f"{k} {v}" for k, v in g["rejected_by_reason"].items()) + "."]
    return "\n".join(lines) + "\n"


def filter_section() -> str:
    f = _load("filter_report.json")
    if not f:
        return _pending("`python -m src.validate_data` (notebook Section 2A.5)")
    lines = ["| Filter (in order) | In | Dropped | Out | Reasons |", "|---|---|---|---|---|"]
    for s in f["stages"]:
        reasons = ", ".join(f"{k} {v}" for k, v in s["reasons"].items()) or "-"
        lines.append(f"| {s['stage']} | {s['in']} | {s['dropped']} | {s['out']} | {reasons} |")
    lines += ["", f"Clean items: **{f['output']}** ("
              + ", ".join(f"{k} {v}" for k, v in sorted(f["output_by_verdict"].items())) + ")."]
    return "\n".join(lines) + "\n"


def split_section() -> str:
    s = _load("split_sizes.json")
    if not s:
        return _pending("`python -m src.split_format` (notebook Section 2A.8)")
    lines = ["| Split | Size | Share | COMPLIANT | NON_COMPLIANT | NEEDS_MORE_INFO |", "|---|---|---|---|---|---|"]
    for k in ("train", "val", "test"):
        v = s["verdict_by_split"][k]
        lines.append(f"| {k} | {s['sizes'][k]} | {_pct(s['fractions'][k])} | {v.get('COMPLIANT', 0)} | "
                     f"{v.get('NON_COMPLIANT', 0)} | {v.get('NEEDS_MORE_INFO', 0)} |")
    lines += ["", f"Total {s['total']}. Stratified by verdict, seed {s['seed']}. Cross-split leakage check "
              f"(cosine >= {s['leakage_check']['threshold']}): {s['leakage_check']['n_moved']} items moved to train."]
    return "\n".join(lines) + "\n"


def audit_section() -> str:
    a = _load("audit_summary.json")
    if not a:
        return _pending("JP's 30-item audit (gate G3), then `python -m src.validate_data --audit-summary`")
    return (f"Hand audit of {a['items']} random clean items: labels agreed on {a['label_agrees'].get('yes', 0)}, "
            f"disagreed on {a['label_agrees'].get('no', 0)}; scenarios realistic {a['scenario_realistic'].get('yes', 0)} "
            f"of {a['items']}; items with errors noted: {a['items_with_errors_noted']}. "
            f"Gate (at most 3 wrong labels) passed: {a['gate_passed']}.\n")


def diversity_section() -> str:
    d = _load("diversity_report.json")
    if not d:
        return _pending("`python -m src.diversity` (notebook Section 2A.7)")
    lw = d["lengths_words"]
    lines = ["| Length | Mean | Std | p5 | p50 | p95 | Max |", "|---|---|---|---|---|---|---|"]
    for name, st in (("Scenario words", lw["scenario"]), ("Answer words", lw["answer"])):
        lines.append(f"| {name} | {st['mean']} | {st['std']} | {st['p5']} | {st['p50']} | {st['p95']} | {st['max']} |")
    if "lengths_tokens" in d:
        for name, key in (("Scenario tokens", "scenario"), ("Answer tokens", "answer")):
            st = d["lengths_tokens"][key]
            lines.append(f"| {name} | {st['mean']} | {st['std']} | {st['p5']} | {st['p50']} | {st['p95']} | {st['max']} |")
    lines += ["", f"Lexical diversity: distinct-1 = {d['lexical']['distinct_1']}, distinct-2 = {d['lexical']['distinct_2']}."]
    if "verdict_line" in d:
        sem = d["semantic"]
        lines += [f"Semantic diversity: mean pairwise cosine = {sem['mean_pairwise_cosine']}, "
                  f"median nearest-neighbour cosine = {sem['nn_cosine']['p50']}. **{d['verdict_line']}.**"]
    hr = d["category_keyword_hit_rate"]
    lines += ["", "Does the text match the plan? Share of scenarios per data category that mention a category keyword:", "",
              "| Data category | n | Keyword hit rate |", "|---|---|---|"]
    for cat, v in hr.items():
        lines.append(f"| {cat} | {v['n']} | {_pct(v['hit_rate'], 0)} |")
    lines += ["", "Top unigrams: " + ", ".join(f"{w} ({c})" for w, c in d["keywords"]["top_unigrams"][:15]) + "."]
    for rel, alt in (("length_histograms.png", "Length histograms"), ("coverage.png", "Planned vs realised coverage"),
                     ("keywords.png", "Top keywords"), ("principle_cooccurrence.png", "Principle co-occurrence"),
                     ("nn_cosine_hist.png", "Nearest-neighbour cosine")):
        img = _img(rel, alt)
        if img:
            lines += ["", img]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def training_section() -> str:
    out = []
    ts = _load("token_stats.json")
    if ts:
        tr = ts["splits"]["train"]
        out.append(f"Token stats (real Llama 3.2 chat template): longest train sequence {tr['full_max']} tokens, "
                   f"p99 {tr['full_p99']:.0f}; longest completion {tr['completion_max']}. "
                   f"max_length = {ts['max_length']} ({ts['max_length_rule']}); truncated examples: {ts['truncated_examples']}.\n")
    ov = _load("overfit_test.json")
    if ov:
        out.append(f"Overfit test: {ov['n_examples']} examples, {ov['steps']} steps, LR {ov['learning_rate']}: "
                   f"final loss (mean of last 5 steps) {ov['final_loss_mean_last5']} vs target {ov['target']} -> "
                   f"**{'passed' if ov['passed'] else 'FAILED'}**.\n")
        img = _img("overfit_test.png", "Overfit test")
        if img:
            out.append(img + "\n")
    mp = _load("memory_probe.json")
    if mp:
        out.append("Memory probe (full log: [reports/oom_log.md](reports/oom_log.md)):\n")
        out.append("| Probe | Batch x accum | Checkpointing | Outcome | Peak GB |\n|---|---|---|---|---|")
        for r in mp:
            out.append(f"| {r['probe']} | {r['per_device_train_batch_size']} x {r['gradient_accumulation_steps']} | "
                       f"{r['gradient_checkpointing']} | {r['outcome']} | {r['peak_allocated_gb']} |")
        out.append("")
    t = _load("training_summary.json")
    if not t:
        out.append(_pending("LR sweep and final run (notebook cells 2B.7 and 2B.8)"))
        return "\n".join(out) + "\n"
    out.append("LR sweep (validation loss only; every run listed, including any that failed the rule):\n")
    out.append("| Run | LR | Epochs | Dropout | Val loss decreases every epoch | Final val loss | Wall time (min) | Peak GPU GB |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in t["all_runs"]:
        out.append(f"| {r['run_name']} | {r['learning_rate']:g} | {r['epochs']} | {r['lora_dropout']} | {r['monotonic']} | "
                   f"{r['final_val_loss']:.4f} | {r['wall_time_s'] / 60:.1f} | {r['peak_gpu_gb']} |")
    c = t["chosen"]
    out += ["", f"Chosen run: **{t['chosen_run']}** ({t['selection_rule']}). Warmup {c['warmup_steps']} steps, "
            f"{c['steps_per_epoch']} optimiser steps per epoch, trainable parameters {c['trainable_params']['trainable']:,} "
            f"({c['trainable_params']['percent']}% of {c['trainable_params']['total']:,}).", ""]
    for note in t.get("notes", []):
        out.append(f"- {note}")
    out += ["", "| Epoch | Train loss (epoch mean) | Validation loss |", "|---|---|---|"]
    for row in t["table"]:
        tl = "n/a" if row["train_loss"] is None else f"{row['train_loss']:.4f}"
        vl = "n/a" if row["val_loss"] is None else f"{row['val_loss']:.4f}"
        out.append(f"| {row['epoch']} | {tl} | {vl} |")
    img = _img("loss_curves.png", "Loss curves")
    if img:
        out += ["", img]
    mpar = _load("merge_parity.json")
    if mpar:
        out += ["", f"Merge parity (5 validation prompts): validation loss 4-bit + adapter {mpar['val_loss_4bit_plus_adapter']} "
                f"vs merged fp16 {mpar['val_loss_merged_fp16']} (abs diff {mpar['val_loss_abs_diff']}); verdict agreement "
                f"{mpar['verdict_agreement']}/{mpar['n_prompts']}, identical text {mpar['identical_text']}/{mpar['n_prompts']}. "
                "A small difference is expected: LoRA was trained against NF4-dequantised weights and merged into fp16 weights."]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Evaluation, manual review, RAG
# ---------------------------------------------------------------------------
def eval_section() -> str:
    from src.metrics import comparison_table

    e = _load("eval_results.json")
    if not e:
        return _pending("inference and metrics (notebook Section 2C)")
    out = [f"Test set: **{e['test_size']}** items, identical for every system. Majority baseline predicts "
           f"`{e['majority_verdict']}` for everything. CIs: paired bootstrap, {e['settings']['bootstrap_resamples']:,} "
           f"resamples, seed {e['settings']['bootstrap_seed']}. Decoding: {e['settings']['decoding']}.", "",
           comparison_table(e), ""]
    j = e.get("judge")
    if j:
        out += [f"LLM-as-judge (`{j['model']}`, blind, pointwise, temperature 0, Pydantic-validated JSON):", "",
                "| System | Judged | Excluded | Verdict | Principles | Faithfulness | Actionability | Format | Overall | Hallucination flagged |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for s, v in j["systems"].items():
            if v["n_judged"]:
                out.append(f"| {s} | {v['n_judged']} | {v['n_excluded']} | {v['verdict_correctness']} | {v['principle_grounding']} | "
                           f"{v['faithfulness']} | {v['actionability']} | {v['format_compliance']} | {v['overall']} | "
                           f"{_pct(v['hallucination_detected_rate'])} |")
        out.append("")
    pv, pdf = e.get("per_verdict_accuracy", {}), e.get("per_difficulty_accuracy", {})
    if "ft" in pv and "base_zeroshot" in pv:
        out += ["Verdict accuracy by group:", "", "| Group | n | Base zero-shot | Base 3-shot | Fine-tuned |", "|---|---|---|---|---|"]
        for src, name in ((pv, "verdict"), (pdf, "difficulty")):
            for g, v in src["ft"].items():
                b0 = src.get("base_zeroshot", {}).get(g, {}).get("accuracy")
                b3 = src.get("base_fewshot", {}).get(g, {}).get("accuracy")
                out.append(f"| {name}={g} | {v['n']} | {b0 if b0 is not None else 'n/a'} | {b3 if b3 is not None else 'n/a'} | {v['accuracy']} |")
        out.append("")
    for rel, alt in (("metrics_bar.png", "Metrics by system"), ("confusion_matrices.png", "Verdict confusion matrices")):
        img = _img(rel, alt)
        if img:
            out += [img, ""]
    return "\n".join(out) + "\n"


def review_section() -> str:
    from src.manual_review import summary_table

    s = _load("manual_review_summary.json")
    if not s:
        return _pending("JP labels reports/manual_review.csv (gate G5), then `python -m src.manual_review --score`")
    out = [f"{s['rows_labelled']} of {s['rows_in_sheet']} rows labelled blind (system hidden until scoring).", "",
           summary_table(s), "",
           f"Teacher label noise estimate: {s['teacher_label_noise_estimate']['items_marked_reference_wrong']} of "
           f"{s['teacher_label_noise_estimate']['items_reviewed']} reviewed items had a wrong reference."]
    k = s.get("judge_agreement_all", {})
    if k.get("n_pairs"):
        out.append(f"Agreement between JP's 'hallucinated' label and the judge's hallucination flag: Cohen's kappa "
                   f"{k['cohen_kappa_hallucination']} over {k['n_pairs']} pairs.")
    return "\n".join(out) + "\n"


def rag_section() -> str:
    r = _load("rag_results.json")
    if not r:
        return _pending("RAG fallback (notebook Section 2D)")
    b, a = r["routed_subset_before"], r["routed_subset_after"]
    ob, oa = r["overall_before"], r["overall_after"]
    lines = [f"Gate: perplexity > tau = {r['tau']} (chosen on validation; floor = {r['gate']['floor_percentile']:.0f}th "
             f"percentile = {r['gate']['floor_value']}) or unparseable output. Validation verdict accuracy with the policy "
             f"{r['gate']['val_accuracy_with_policy']} vs without {r['gate']['val_accuracy_without_rag']}.", "",
             f"Test: {r['routed']} of {r['test_size']} items routed to the fallback ({_pct(r['routed_share'])}).", "",
             "| Subset | n | Verdict accuracy | JSON valid | Principles micro-F1 | Invented ID or statute |", "|---|---|---|---|---|---|"]
    for name, m in (("Routed, first answer", b), ("Routed, after RAG", a), ("All test, FT only", ob), ("All test, FT + RAG gate", oa)):
        if m.get("n"):
            lines.append(f"| {name} | {m['n']} | {m['verdict_accuracy']} | {_pct(m['json_valid'])} | {m['principles_micro_f1']} | "
                         f"{_pct(m['invented_id_or_statute_rate'])} |")
    lines += ["", f"Limitation: {r['limitation']}"]
    return "\n".join(lines) + "\n"


def hub_section() -> str:
    h = _load("hub_links.json")
    if not h:
        return _pending("merge and push (notebook cells 2B.9 and 2B.10)")
    return "\n".join(f"- {k}: {v}" for k, v in h.items()) + "\n"


def results_block() -> str:
    parts = [
        "### Dataset", "", "#### Generation", "", generation_section(),
        "#### Filters", "", filter_section(), "#### Split", "", split_section(),
        "#### Human audit", "", audit_section(), "#### Diversity", "", diversity_section(),
        "### Training", "", training_section(),
        "### Hugging Face Hub", "", hub_section(),
        "### Evaluation", "", eval_section(),
        "### Manual review and hallucination rate", "", review_section(),
        "### RAG fallback (bonus)", "", rag_section(),
    ]
    return "\n".join(parts)


def datacard_block() -> str:
    return "\n".join(["#### Generation", "", generation_section(), "#### Filters", "", filter_section(),
                      "#### Split", "", split_section(), "#### Human audit", "", audit_section()])


def replace_between(path: Path, tag: str, content: str) -> bool:
    text = path.read_text(encoding="utf-8")
    pat = re.compile(rf"(<!-- {tag}:START -->)(.*?)(<!-- {tag}:END -->)", re.DOTALL)
    if not pat.search(text):
        print(f"[render] markers for {tag} not found in {path}")
        return False
    new = pat.sub(lambda m: f"{m.group(1)}\n{content}\n{m.group(3)}", text)
    path.write_text(new, encoding="utf-8", newline="\n")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Render README tables from reports/")
    ap.add_argument("--print", action="store_true", help="print the results block instead of writing READMEs")
    args = ap.parse_args()
    block = results_block()
    if args.print:
        print(block)
        return
    ok1 = replace_between(TASK_ROOT / "README.md", "RESULTS", block)
    ok2 = replace_between(TASK_ROOT / "data" / "README.md", "DATACARD", datacard_block())
    print(f"[render] README.md updated: {ok1}; data/README.md updated: {ok2}")


if __name__ == "__main__":
    main()
