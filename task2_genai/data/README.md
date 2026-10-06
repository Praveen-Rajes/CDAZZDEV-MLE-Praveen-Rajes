<!-- AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'Data card for the synthetic PDPA triage dataset per plan Phase 3', Date: 2026-10-06 -->
# Data card: synthetic PDPA compliance triage dataset

## Source

Everything here is **synthetic**. No real bank, customer, employee or internal policy text was used. The organisation is always
"the Bank"; people, vendors and products are fictional; NIC, account and card numbers appear only in partial form.

- **Teacher model:** `openai/gpt-oss-120b` on Groq (free tier), `temperature=1.0`, `top_p=1.0`, `reasoning_effort=low`, strict
  JSON schema (`TeacherBatch`). Optional top-up teacher `openai/gpt-oss-20b`, capped at 25% of seeds and tagged per item in
  `teacher_model`.
- **Teacher system prompt (in full):** [`../prompts/teacher_system_prompt.md`](../prompts/teacher_system_prompt.md); the version
  with the rulebook inserted, exactly as sent, is [`../prompts/teacher_system_prompt.rendered.md`](../prompts/teacher_system_prompt.rendered.md).
- **Rules:** [`../knowledge/pdpa_policy_rulebook.md`](../knowledge/pdpa_policy_rulebook.md), a fictional, simplified policy
  inspired by the structure of Sri Lanka's Personal Data Protection Act No. 9 of 2022 (as amended). Not a statement of the law.
- **Seeds:** [`seeds.jsonl`](seeds.jsonl), 400 balanced stratified specifications over business unit (14), data category (12),
  processing activity (14), artifact type (7), target verdict (30/45/25%), difficulty (50/30/20%) and length (30/45/25%).

## Files

| File | Content |
|---|---|
| `seeds.jsonl` | One seed per line (`seed_id` plus the seven dimensions) |
| `raw/generations.jsonl` | Every accepted teacher item with its seed, teacher model and call id |
| `raw/rejected.jsonl` | Every rejected teacher item with the reason |
| `logs/generation_calls.jsonl` | One line per API call: tokens, cached tokens, latency, status (no prompts, no keys) |
| `clean/all.jsonl` | Items that passed every filter, with the canonical answer JSON |
| `splits/{train,val,test}.jsonl` | Chat format: `{"id", "messages": [system, user, assistant], "meta"}` |
| `audit_log.md` | JP's manual audit of 30 random clean items (gate G3) |

## Processing

1. **Per-item validation at generation time:** strict schema, consistency rules (for example COMPLIANT means no issues and LOW
   risk), verdict equals the seed's target, scenario length within the bucket plus or minus 15%.
2. **Filters, in order:** schema and consistency re-check; leakage (no principle IDs or verdict words in the scenario);
   forbidden content (statute, fine or penalty citations in answers, real Sri Lankan bank names, full-length NIC numbers);
   exact duplicates; near duplicates (`all-MiniLM-L6-v2` cosine >= 0.92).
3. **Split:** stratified by verdict, 80/10/10, seed 42; any val/test item with cosine >= 0.90 to a train item moves to train.
4. **Format:** the assistant turn is the canonical answer JSON (keys in schema order), the same string used as the training
   target, the evaluation reference and the metric input.

## Counts (rendered from `reports/` by `scripts/render_results.py`)

<!-- DATACARD:START -->
#### Generation

_Pending: teacher generation (notebook Section 2A.4)._

#### Filters

_Pending: `python -m src.validate_data` (notebook Section 2A.5)._

#### Split

_Pending: `python -m src.split_format` (notebook Section 2A.7)._

#### Human audit

_Pending: JP's 30-item audit (gate G3), then `python -m src.validate_data --audit-summary`._

<!-- DATACARD:END -->

## Known limitations

- **Single teacher.** All labels come from one model, so the test references share its biases. The 30-item audit and the
  `reference_wrong` error type in the manual review estimate this label noise; they do not remove it.
- **Synthetic distribution.** Real requests are messier, longer and less self-contained than generated ones.
- **Fictional policy.** Verdicts follow the rulebook here, not the law; the dataset is not legal advice.
- **English only**, with Sri Lankan names, towns and LKR amounts for realism.
- **Small test split**, so evaluation intervals are wide.
