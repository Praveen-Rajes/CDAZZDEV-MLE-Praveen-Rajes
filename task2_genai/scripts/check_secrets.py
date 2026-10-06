# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'Secret scanner over tracked files and notebook outputs with pre-commit hook per plan Section 5.2', Date: 2026-10-06
"""Scan the repository for credentials. Exit 1 if anything looks like a secret.

Scans every git-tracked file (and staged files) when run inside a git repo, otherwise every
file under the repo root except ignored folders. Notebook outputs are scanned too (an .ipynb
is plain JSON text).

Usage:
    python task2_genai/scripts/check_secrets.py                 # scan
    python task2_genai/scripts/check_secrets.py --install-hook  # add as a git pre-commit hook
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

PATTERNS = [
    ("groq_key", re.compile(r"gsk_[A-Za-z0-9]{20,}")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("openai_style_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}")),
    ("wandb_key", re.compile(r"wandb_[A-Za-z0-9]{20,}")),
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._\-]{16,}")),
    ("long_hex", re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{40,}(?![0-9a-fA-F])")),
    ("key_assignment", re.compile(r"(?i)\b(GROQ_API_KEY|HF_TOKEN|WANDB_API_KEY|JUDGE_API_KEY)[ \t]*=[ \t]*['\"]?[A-Za-z0-9_\-]{12,}")),
]
# Long hex strings that are not secrets: git/HF commit hashes and cache snapshot folders in URLs or paths.
HEX_ALLOWED_CONTEXT = re.compile(r"(commit/|blob/|tree/|snapshots/|resolve/|revision|sha256|commit_sha|oid)", re.IGNORECASE)
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", "chroma_db", "checkpoints",
             "outputs", "wandb", "hf_cache", "node_modules", ".cache"}
SKIP_FILES = {".env"}
BINARY_EXT = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".safetensors", ".bin", ".pt", ".zip", ".gz", ".parquet", ".sqlite3"}


def git_files() -> List[Path]:
    try:
        tracked = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
        staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=REPO_ROOT, capture_output=True,
                                text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    names = {n for n in (tracked + "\n" + staged).splitlines() if n.strip()}
    return [REPO_ROOT / n for n in sorted(names) if (REPO_ROOT / n).is_file()]


def walk_files() -> List[Path]:
    out = []
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if f in SKIP_FILES:
                continue
            out.append(Path(root) / f)
    return out


def scan_text(text: str) -> List[Tuple[str, int, str]]:
    hits = []
    for name, pat in PATTERNS:
        for m in pat.finditer(text):
            if name == "long_hex":
                ctx = text[max(0, m.start() - 40) : m.start()]
                if HEX_ALLOWED_CONTEXT.search(ctx):
                    continue
            line = text.count("\n", 0, m.start()) + 1
            shown = m.group(0)
            hits.append((name, line, shown[:6] + "..." + f"({len(shown)} chars)"))   # never print the full value
    return hits


def scan(files: Iterable[Path]) -> List[Tuple[Path, str, int, str]]:
    findings = []
    for path in files:
        if path.suffix.lower() in BINARY_EXT or path.name in SKIP_FILES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name, line, shown in scan_text(text):
            findings.append((path, name, line, shown))
    return findings


def install_hook() -> None:
    git_dir = REPO_ROOT / ".git"
    if not git_dir.exists():
        sys.exit("Not a git repository (run `git init` first).")
    hook = git_dir / "hooks" / "pre-commit"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text("#!/bin/sh\npython task2_genai/scripts/check_secrets.py || {\n"
                    "  echo 'check_secrets.py found a possible secret; commit blocked.'; exit 1; }\n", encoding="utf-8")
    try:
        hook.chmod(0o755)
    except OSError:
        pass
    print(f"Installed pre-commit hook at {hook}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan the repo for secrets")
    ap.add_argument("--install-hook", action="store_true")
    args = ap.parse_args()
    if args.install_hook:
        install_hook()
        return
    files = git_files() or walk_files()
    findings = scan(files)
    if findings:
        print(f"POSSIBLE SECRETS FOUND ({len(findings)}):")
        for path, name, line, shown in findings:
            print(f"  {path.relative_to(REPO_ROOT)}:{line}  {name}  {shown}")
        sys.exit(1)
    print(f"check_secrets: OK ({len(files)} files scanned, no secrets found)")


if __name__ == "__main__":
    main()
