<!-- AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'Root README overview with links to each task per plan Section 4', Date: 2026-10-06 -->
# CDAZZDEV Senior MLE Assessment

| Task | Folder | Summary |
|---|---|---|
| Task 1 | [`task1_financial/`](task1_financial/) | (separate task) |
| **Task 2: Generative AI** | [`task2_genai/`](task2_genai/) | QLoRA fine-tune of Llama-3.2-3B-Instruct for PDPA compliance triage at a fictional Sri Lankan bank, with synthetic data from gpt-oss-120b, honest evaluation (ROUGE-L, BERTScore, LLM-as-judge, blind human review) and a RAG fallback |
| Task 3 | [`task3_agentic/`](task3_agentic/) | (separate task) |

## Task 2 at a glance

- **Start here:** [`task2_genai/README.md`](task2_genai/README.md): problem statement, model choices, hyperparameters, rendered
  results, and a step-by-step [how to run](task2_genai/README.md#how-to-run-step-by-step) with a table of
  [where every score and output lives](task2_genai/README.md#where-to-find-every-score-and-output).
- **Notebook (runs on Colab):** [`task2_genai/task2_genai.ipynb`](task2_genai/task2_genai.ipynb)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<gh_user>/CDAZZDEV-MLE-<YourName>/blob/main/task2_genai/task2_genai.ipynb)
- **Build plan:** [`plan.md`](plan.md) (the single source of truth the code follows).

## Repository-wide files

- [`CITATIONS.md`](CITATIONS.md): every AI-assisted file, the teacher and judge models with their prompts, documentation consulted.
- [`REFLECTION.md`](REFLECTION.md): reflection across all tasks (max 600 words).
- [`.env.example`](.env.example): the environment variable names. Real keys live in `.env` (gitignored) or Colab Secrets, never in the repo.
- Secret scan before every commit: `python task2_genai/scripts/check_secrets.py` (install as a pre-commit hook with `--install-hook`).
