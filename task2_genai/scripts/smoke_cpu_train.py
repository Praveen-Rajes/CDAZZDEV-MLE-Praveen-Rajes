# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'CPU smoke test: tiny-model training, inference and metrics on the real code path per plan Phase 4', Date: 2026-10-06
"""Phase 4 smoke test: the same train/infer/metrics code path on CPU with a tiny model.

SmolLM2-135M-Instruct, no quantisation (bitsandbytes needs CUDA), LoRA r=8, 16 examples, 1 epoch,
max_length 512; then generation on 4 examples and metrics on those outputs. Quality is irrelevant.
Uses data/splits if present, otherwise the hand-written test fixtures.

Outputs: reports/smoke/training_log.csv, reports/smoke/smoke_metrics.json
Usage: python scripts/smoke_cpu_train.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT))

from src.config import load_config  # noqa: E402
from src.utils import write_json  # noqa: E402


def main() -> int:
    import numpy as np
    from peft import PeftModel

    from src.infer import load_generation_tokenizer, load_model_fp16, predict_records
    from src.metrics import component_matrix, example_components, point_metrics, rouge_l
    from src.train import load_records, smoke_train, write_csv

    cfg = load_config()
    out_dir = TASK_ROOT / "outputs" / "smoke"
    rep_dir = cfg.report_path("smoke")
    rep_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("== 1. train (CPU, tiny model)")
    tr = smoke_train(cfg, out_dir)
    write_csv(rep_dir / "training_log.csv", tr["table"])
    print(json.dumps(tr["table"], indent=1))

    print("== 2. infer on 4 examples with the trained adapter")
    sm = cfg.smoke
    tok = load_generation_tokenizer(cfg, sm.model_id)
    base = load_model_fp16(sm.model_id, cfg, tok)
    model = PeftModel.from_pretrained(base, tr["adapter_dir"])
    model.eval()
    recs = load_records(cfg, "test", n=sm.n_infer, allow_fixture=True)
    preds = predict_records(cfg, model, tok, recs, "smoke", batch_size=2, max_new_tokens=sm.max_new_tokens)

    print("== 3. metrics on those outputs")
    refs = [json.loads(r["messages"][2]["content"]) for r in recs]
    comps = [example_components(p, ref) for p, ref in zip(preds, refs)]
    rl = rouge_l([p["raw_text"] for p in preds], [r["messages"][2]["content"] for r in recs])
    for c, v in zip(comps, rl):
        c["rougeL_full"] = v
    metrics = point_metrics(component_matrix(comps))
    result = {"training": {k: v for k, v in tr.items() if k != "adapter_dir"}, "n_infer": len(preds),
              "mean_perplexity": float(np.nanmean([p["perplexity"] or np.nan for p in preds])),
              "metrics": metrics, "sample_output": preds[0]["raw_text"][:300], "total_seconds": round(time.time() - t0, 1),
              "note": "Plumbing test only: tiny model, 1 epoch, quality is irrelevant."}
    write_json(rep_dir / "smoke_metrics.json", result)
    print(json.dumps(result, indent=1))
    print(f"SMOKE TEST PASSED in {result['total_seconds']} s -> {rep_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
