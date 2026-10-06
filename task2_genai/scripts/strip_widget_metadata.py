# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'Remove metadata.widgets from the notebook so GitHub renders it, keeping outputs, per plan Phase 11', Date: 2026-10-06
"""Remove only `metadata.widgets` (notebook level and cell level). Outputs stay.

tqdm progress bars in Colab leave widget state without the "state" key, and GitHub's renderer
then shows "Invalid Notebook". This fixes that without touching anything else.

Usage: python task2_genai/scripts/strip_widget_metadata.py [path/to/notebook.ipynb]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / "task2_genai.ipynb"


def strip(path: Path) -> int:
    nb = json.loads(path.read_text(encoding="utf-8"))
    removed = 0
    if "widgets" in nb.get("metadata", {}):
        del nb["metadata"]["widgets"]
        removed += 1
    for cell in nb.get("cells", []):
        if "widgets" in cell.get("metadata", {}):
            del cell["metadata"]["widgets"]
            removed += 1
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return removed


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    n = strip(target)
    print(f"Removed {n} widget metadata block(s) from {target}; outputs kept.")
