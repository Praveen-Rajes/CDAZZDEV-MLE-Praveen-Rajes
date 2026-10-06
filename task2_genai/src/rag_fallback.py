# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'ChromaDB RAG fallback with perplexity gate tuned on validation per plan Phase 10', Date: 2026-10-06
"""RAG fallback (bonus).

- Knowledge base: the rulebook only, one chunk per principle subsection (Rule, Bank standard,
  Typical breach, Compliant pattern) plus the verdict guide and risk guide. Never test data.
- Store: chromadb.PersistentClient(path="task2_genai/chroma_db"), collection "pdpa_policy",
  SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2").
- Confidence: perplexity of the FT model's own greedy output, exp(-mean token log-prob).
- Threshold tau: chosen on VALIDATION only, maximising the fallback policy's verdict accuracy,
  never below the 70th percentile of validation perplexity.
- Fallback: if perplexity > tau or the output failed to parse, retrieve top-3 chunks with the
  scenario as the query, rebuild the user message (Appendix E) and use the second answer.
- Known limitation: the model never saw retrieved context in training (distribution shift).
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import Config, load_text
from .utils import read_jsonl, write_json, write_jsonl

SUBSECTIONS = ("Rule", "Bank standard", "Typical breach", "Compliant pattern")


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
def rulebook_chunks(text: str) -> List[Dict[str, Any]]:
    """Split the rulebook into principle-subsection chunks plus the two guides."""
    chunks: List[Dict[str, Any]] = []
    sections = re.split(r"^## ", text, flags=re.MULTILINE)
    for sec in sections[1:]:
        title, _, body = sec.partition("\n")
        title = title.strip()
        m = re.match(r"(P\d{2})\s+(.*)", title)
        if m:
            pid, name = m.group(1), m.group(2)
            for line in body.splitlines():
                for sub in SUBSECTIONS:
                    prefix = f"{sub}:"
                    if line.startswith(prefix):
                        chunks.append({"id": f"{pid}-{sub.lower().replace(' ', '_')}",
                                       "text": f"{pid} {name}. {sub}: {line[len(prefix):].strip()}",
                                       "metadata": {"principle_id": pid, "subsection": sub}})
        elif title in ("Verdict guide", "Risk rating guide"):
            chunks.append({"id": title.lower().replace(" ", "_"), "text": f"{title}:\n{body.strip()}",
                           "metadata": {"principle_id": "GUIDE", "subsection": title}})
    return chunks


def build_store(cfg: Config, reset: bool = True):
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    path = cfg.path("chroma_dir")
    client = chromadb.PersistentClient(path=str(path))
    ef = SentenceTransformerEmbeddingFunction(model_name=cfg.rag.embed_model)
    if reset:
        try:
            client.delete_collection(cfg.rag.collection)
        except Exception:  # collection did not exist yet
            pass
    coll = client.get_or_create_collection(cfg.rag.collection, embedding_function=ef)
    chunks = rulebook_chunks(load_text(cfg.path("rulebook")))
    if coll.count() == 0:
        coll.add(ids=[c["id"] for c in chunks], documents=[c["text"] for c in chunks],
                 metadatas=[c["metadata"] for c in chunks])
    print(f"[rag] collection '{cfg.rag.collection}' at {path}: {coll.count()} chunks")
    return coll


def retrieve(coll: Any, query: str, k: int) -> List[Dict[str, Any]]:
    res = coll.query(query_texts=[query], n_results=k)
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append({"text": doc, "principle_id": meta["principle_id"], "subsection": meta["subsection"],
                    "distance": round(float(dist), 4)})
    return out


def format_chunks(chunks: List[Dict[str, Any]]) -> str:
    """Numbered list: [1] (P12, Bank standard) ..."""
    lines = []
    for n, c in enumerate(chunks, 1):
        body = c["text"].split(": ", 1)[1] if ": " in c["text"] else c["text"]
        lines.append(f"[{n}] ({c['principle_id']}, {c['subsection']}) {body}")
    return "\n".join(lines)


def rag_user_message(cfg: Config, scenario: str, chunks: List[Dict[str, Any]]) -> str:
    tpl = load_text(cfg.prompt_path("rag_user_template.md")).strip()
    return tpl.replace("{SCENARIO}", scenario).replace("{CHUNKS}", format_chunks(chunks))


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------
def _verdict(p: Dict[str, Any]) -> Optional[str]:
    parsed = p.get("parsed")
    return parsed.get("verdict") if p.get("parse_ok") and isinstance(parsed, dict) else None


def routed(p: Dict[str, Any], tau: float) -> bool:
    ppl = p.get("perplexity")
    return (not p.get("parse_ok")) or ppl is None or not math.isfinite(ppl) or ppl > tau


def policy_accuracy(first: Dict[str, Dict], second: Dict[str, Dict], refs: Dict[str, str], tau: float) -> Tuple[float, int]:
    correct, n_routed = 0, 0
    for i, ref in refs.items():
        use_second = routed(first[i], tau)
        n_routed += use_second
        correct += _verdict(second[i] if use_second else first[i]) == ref
    return correct / len(refs), n_routed


def choose_tau(first: Dict[str, Dict], second: Dict[str, Dict], refs: Dict[str, str], floor_pct: float) -> Dict[str, Any]:
    """Candidate taus = validation perplexities at or above the floor percentile (and the floor itself).
    Pick the highest accuracy; ties go to the higher tau (fewer fallbacks)."""
    ppls = np.array([first[i]["perplexity"] for i in refs if first[i].get("perplexity") is not None], dtype=float)
    floor = float(np.percentile(ppls, floor_pct))
    candidates = sorted({floor, *[float(x) for x in ppls if x >= floor]})
    table = []
    for tau in candidates:
        acc, n_r = policy_accuracy(first, second, refs, tau)
        table.append({"tau": round(tau, 5), "val_accuracy": round(acc, 4), "routed": n_r})
    best = max(table, key=lambda r: (r["val_accuracy"], r["tau"]))
    no_rag_acc = sum(_verdict(first[i]) == refs[i] for i in refs) / len(refs)
    return {"tau": best["tau"], "floor_percentile": floor_pct, "floor_value": round(floor, 5),
            "val_accuracy_with_policy": best["val_accuracy"], "val_accuracy_without_rag": round(no_rag_acc, 4),
            "val_routed": best["routed"], "n_val": len(refs), "candidates": table}


# ---------------------------------------------------------------------------
# Full run (notebook Section 2D)
# ---------------------------------------------------------------------------
def _subset_task_metrics(preds: List[Dict[str, Any]], refs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    from .metrics import component_matrix, example_components, point_metrics

    if not preds:
        return {"n": 0}
    comps = component_matrix([example_components(p, refs[p["id"]]) for p in preds])
    m = point_metrics(comps)
    keep = ("json_valid", "schema_valid", "verdict_accuracy", "verdict_macro_f1", "risk_accuracy",
            "principles_micro_f1", "principles_jaccard", "invented_id_or_statute_rate")
    return {"n": len(preds), **{k: m.get(k) for k in keep}}


def run_rag(cfg: Config, model: Any, tokenizer: Any) -> Dict[str, Any]:
    import json

    from .infer import predict_records, predictions_path, run_system
    from .split_format import scenario_of

    coll = build_store(cfg)
    k = cfg.rag.top_k

    def rag_fn(rec: Dict[str, Any]) -> str:
        return rag_user_message(cfg, scenario_of(rec), retrieve(coll, scenario_of(rec), k))

    # 1. Validation: first pass and fallback answer for every item, so any tau can be simulated.
    val = read_jsonl(cfg.split_path("val"))
    val_refs = {r["id"]: r["meta"]["verdict"] for r in val}
    first_val = {p["id"]: p for p in run_system(cfg, "ft", model, tokenizer, split="val")}
    p2 = cfg.report_path("predictions_ft_val_rag.jsonl")
    second_val_rows = read_jsonl(p2)
    if {r["id"] for r in second_val_rows} != set(val_refs):
        second_val_rows = predict_records(cfg, model, tokenizer, val, "ft_rag", user_fn=rag_fn)
        write_jsonl(p2, second_val_rows)
    second_val = {p["id"]: p for p in second_val_rows}
    gate = choose_tau(first_val, second_val, val_refs, cfg.rag.tau_floor_percentile)
    tau = gate["tau"]
    print(f"[rag] tau={tau} (val acc with policy {gate['val_accuracy_with_policy']} vs without {gate['val_accuracy_without_rag']})")

    # 2. Test: route by the gate, regenerate only the routed items.
    test = read_jsonl(cfg.split_path("test"))
    test_by_id = {r["id"]: r for r in test}
    refs = {r["id"]: json.loads(r["messages"][2]["content"]) for r in test}
    first = {p["id"]: p for p in read_jsonl(predictions_path(cfg, "ft"))}
    if set(first) != set(refs):
        first = {p["id"]: p for p in run_system(cfg, "ft", model, tokenizer, split="test")}
    route_ids = sorted(i for i in refs if routed(first[i], tau))
    p3 = cfg.report_path("predictions_ft_test_rag.jsonl")
    second_rows = [r for r in read_jsonl(p3) if r["id"] in route_ids]
    if {r["id"] for r in second_rows} != set(route_ids):
        second_rows = predict_records(cfg, model, tokenizer, [test_by_id[i] for i in route_ids], "ft_rag", user_fn=rag_fn) if route_ids else []
        write_jsonl(p3, second_rows)
    second = {p["id"]: p for p in second_rows}
    final = [second[i] if i in second else first[i] for i in sorted(refs)]

    result = {
        "tau": tau, "gate": {k2: v for k2, v in gate.items() if k2 != "candidates"}, "gate_candidates": gate["candidates"],
        "test_size": len(refs), "routed": len(route_ids), "routed_share": round(len(route_ids) / len(refs), 4),
        "routed_ids": route_ids,
        "routed_subset_before": _subset_task_metrics([first[i] for i in route_ids], refs),
        "routed_subset_after": _subset_task_metrics([second[i] for i in route_ids], refs),
        "overall_before": _subset_task_metrics([first[i] for i in sorted(refs)], refs),
        "overall_after": _subset_task_metrics(final, refs),
        "limitation": "The model never saw retrieved context during training, so the fallback prompt is a distribution shift.",
    }
    if route_ids:
        ex_id = next((i for i in route_ids if _verdict(second[i]) == refs[i]["verdict"] != _verdict(first[i])), route_ids[0])
        sc = scenario_of(test_by_id[ex_id])
        result["example"] = {"id": ex_id, "scenario": sc, "reference_verdict": refs[ex_id]["verdict"],
                             "first_answer": first[ex_id]["raw_text"], "first_perplexity": first[ex_id].get("perplexity"),
                             "retrieved_chunks": retrieve(coll, sc, k), "second_answer": second[ex_id]["raw_text"],
                             "second_perplexity": second[ex_id].get("perplexity")}
    write_json(cfg.report_path("rag_results.json"), result)
    return result
