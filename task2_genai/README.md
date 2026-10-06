<!-- AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'Task 2 README: problem, choices, hyperparameters, rendered results, how to run and where to find every score', Date: 2026-10-06 -->
# Task 2: Domain-specific fine-tuning pipeline (PDPA compliance triage)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<gh_user>/CDAZZDEV-MLE-<YourName>/blob/main/task2_genai/task2_genai.ipynb)

QLoRA fine-tune of **Llama-3.2-3B-Instruct** (student) on synthetic data written by **gpt-oss-120b** (teacher) and scored by
**qwen3.8-27b** (judge), three different model families on purpose. The model triages internal requests at a fictional Sri Lankan
bank against a data protection rulebook and returns strict JSON.

- Notebook (outputs kept): [`task2_genai.ipynb`](task2_genai.ipynb)
- Merged model and adapter on the Hugging Face Hub: see [Results > Hugging Face Hub](#hugging-face-hub)
- Data card: [`data/README.md`](data/README.md)
- Every number in this README is rendered by [`scripts/render_results.py`](scripts/render_results.py) from JSON/CSV files in
  [`reports/`](reports/). Nothing is typed by hand.

**Contents:** [Problem](#problem-statement) ·
[Model choices](#model-choices) · [Method](#method) · [Teacher prompt](#teacher-system-prompt) ·
[Hyperparameters](#hyperparameters) · [Results](#results) · [Qualitative analysis](#qualitative-analysis) ·
[Limitations](#limitations) · [How to run](#how-to-run-step-by-step) · [Where to find every score](#where-to-find-every-score-and-output) ·
[Command line](#command-line-reference) · [Troubleshooting](#troubleshooting)

## Problem statement

Task: PDPA compliance triage.

Input: one internal request or incident note from an employee of a fictional Sri Lankan bank ("the Bank"), 50 to 240 words, describing a proposed or ongoing activity that involves personal data.

Output: one JSON object with keys in this order:

```json
{
  "rationale": "2 to 4 sentences reasoning from the facts to the verdict",
  "principles": ["P15", "P02"],
  "issues": ["concrete problem tied to a stated fact"],
  "missing_information": [],
  "verdict": "COMPLIANT | NON_COMPLIANT | NEEDS_MORE_INFO",
  "risk_level": "LOW | MEDIUM | HIGH",
  "required_actions": ["verb-first concrete action"]
}
```

Rationale comes first on purpose: for a 3B model, writing the reasoning before the decision gives it tokens to think with.

A response is correct when:

- it is valid JSON with exactly these keys and allowed values,
- the verdict matches what a careful reviewer concludes from the scenario and the rulebook,
- the risk level follows the rulebook's risk guide,
- the principles include the primary relevant principle and nothing clearly irrelevant,
- every issue and action is grounded in facts stated in the scenario,
- it states no invented facts, principle IDs, Act sections, fines or deadlines.

A response is incorrect when the verdict is wrong, or it is unparseable. It is hallucinated when it asserts anything not supported by the scenario or rulebook (full labelling guide in Appendix F).

## Model choices

| Role | Model | Where | Why |
|---|---|---|---|
| Teacher (data generator) | `openai/gpt-oss-120b` | Groq free tier | Strongest free model on Groq today. Supports strict JSON schema (constrained decoding) and prompt caching. |
| Student (fine-tuned) | `meta-llama/Llama-3.2-3B-Instruct` (ungated mirror with identical weights: `unsloth/Llama-3.2-3B-Instruct`) | Colab T4 | Different family from teacher. 3B fits a T4 with room for batch 4 at about 1.2K tokens, merges in fp16 on the GPU, trains fast enough to run a small sweep inside free quota. |
| Judge (LLM-as-judge) | `qwen/qwen3.8-27b` | Groq free tier (preview) | Third family. A judge from the teacher's family would favour the fine-tuned model, because the fine-tuned model imitates the teacher's style. |

What changed from the brief's suggestions (say this in the README, it shows you checked):

- Groq's live model page on 2026-10-06 lists Llama 3.3 70B as Enterprise only. The free plan chat models are gpt-oss-120b, gpt-oss-20b and qwen3.8-27b. So "Groq Llama-3-70B" as teacher is no longer a free option.
- Free plan limits on 2026-10-06: gpt-oss-120b and qwen3.8-27b are each 30 RPM, 1K requests/day, 8K tokens/min, 200K tokens/day. Cached prompt tokens do not count toward limits, and caching works on gpt-oss models only.

Why 3B and not 7B: a 7B model in fp16 is about 14 GB. The T4 has about 15 GB and Colab free has about 12.7 GB of system RAM, so merging a 7B in fp16 is fragile on free Colab. A 3B model gives the same scientific story with time left for a learning-rate sweep, an overfit test and proper evaluation. Smaller model, better data, honest eval.

Fallbacks (config-driven, no code changes):

- Student: `microsoft/Phi-4-mini-instruct` if Llama access or fp16 behaviour is a problem.
- Teacher top-up: `openai/gpt-oss-20b` (separate 200K/day quota), capped at 25 percent of the dataset, tagged per example.
- Judge: if qwen3.8-27b (a preview model) is withdrawn, use a free model from another family through any OpenAI-compatible endpoint (for example Mistral's free tier). Last resort gpt-oss-20b, with the teacher-family bias stated in the README.

Licence note: the Llama 3.2 Community License requires "Llama" at the start of the name of a distributed derivative model and a "Built with Llama" notice. So the Hub repo is `Llama-3.2-3B-PDPA-Triage` and the model card says "Built with Llama".

## Method

Andrej Karpathy's training recipe, applied:

1. **Become one with the data:** hand-audit 30 examples (gate G3), diversity report, length statistics.
2. **End-to-end skeleton and dumb baselines:** majority-class verdict baseline, base model zero-shot, base model 3-shot.
3. **Overfit a tiny batch:** 8 examples, 60 steps; the loss must fall below 0.05 or the pipeline is broken.
4. **Regularise and tune on validation only:** a two-value learning-rate sweep, chosen by validation loss.
5. **Evaluate once on test.** Never tune on test.

Implementation choices worth knowing (each one is in the code with a comment):

- **Completion-only loss.** The dataset is conversational prompt-completion (`prompt = [system, user]`, `completion = [assistant]`),
  so TRL computes the loss on the answer only. Notebook cell 2B.4 decodes a real training batch and prints the supervised tokens to prove it.
- **Pad token is `<|finetune_right_pad_id|>`, never eos.** If padding were masked by the eos id, the model would never learn to stop.
- **Pinned chat-template date.** The Llama 3.2 template prints today's date in every system header. It is pinned to the template's
  own default (`26 Jul 2024`) so training and inference prompts are byte-identical whatever day each runs.
- **Explicit `prepare_model_for_kbit_training` + `get_peft_model`** before `SFTTrainer`, so the trainable-parameter count is printed
  before training and TRL does not repeat the k-bit preparation.
- **`warmup_steps` computed** from 5% of total steps instead of relying on `warmup_ratio`; TRL/transformers renames
  (`max_length`, `eval_strategy`) and the transformers v5 `dtype=` argument are handled by version checks, and anything a library
  version does not accept is printed, never silently dropped.
- **Overfit test uses LoRA dropout 0** and a constant LR: it checks that the pipeline can memorise, not regularisation.
- **Finished runs are reused.** Each run writes `run_summary.json` next to its adapter on Google Drive; re-running the notebook
  does not retrain, and a disconnect resumes from the last epoch checkpoint.
- **Metrics on unparseable outputs** use the raw text (ROUGE-L, BERTScore), and they count as wrong for every task metric.
- **3-shot baseline examples** are fixed: per verdict, the shortest `straightforward` train item (ids saved in `reports/fewshot_ids.json`).

## Teacher system prompt

Full template: [`prompts/teacher_system_prompt.md`](prompts/teacher_system_prompt.md). At runtime `{RULEBOOK}` is replaced with
[`knowledge/pdpa_policy_rulebook.md`](knowledge/pdpa_policy_rulebook.md); the exact text sent is saved as
[`prompts/teacher_system_prompt.rendered.md`](prompts/teacher_system_prompt.rendered.md) and printed in notebook cell 2A.1.
The user message is [`prompts/teacher_user_template.md`](prompts/teacher_user_template.md) with four seeds as JSON.

<details>
<summary>Teacher system prompt (template, in full)</summary>

```text
You are a senior Data Protection Officer at a Sri Lankan licensed commercial bank and an expert writer of realistic training data. You create examples for a small model that triages personal data questions against the Bank's internal policy below.

=== POLICY (the only source of rules) ===
{RULEBOOK}
=== END POLICY ===

YOUR TASK
You will receive a JSON list of seed specifications. For each seed, write one example with two parts:
1. "scenario": a realistic internal text that an employee of the Bank sends to the compliance team.
2. "answer": the triage the model must learn to produce.

SCENARIO RULES
- Follow the seed exactly: business_unit, data_category, processing_activity, artifact_type, length_bucket, difficulty and target_verdict.
- If the combination looks unusual, find a realistic angle (for example, human resources plus data of minors can be a scholarship scheme for staff children).
- Length: short = 50 to 90 words, medium = 90 to 160 words, long = 160 to 240 words.
- Write natural Sri Lankan workplace English. Use Sri Lankan towns, LKR amounts, and a mix of Sinhala, Tamil, Muslim and Burgher names. All people, vendors and products are fictional. Call the organisation "the Bank". Never name a real bank, company or real person.
- Never write full NIC, account or card numbers. If needed, write partial forms such as "NIC ending 4521V".
- Put every fact the answer depends on inside the scenario.
- Never mention principle IDs, the policy's title, or the words "compliant", "non-compliant" or "verdict" in the scenario.
- target_verdict NEEDS_MORE_INFO: leave out the one or two facts that decide the outcome, and make the gap natural (the writer simply did not mention it).
- difficulty "tricky": add one detail that looks like a problem but is not, or one that looks harmless but is the real problem.
- difficulty "multi_principle": the situation must involve two or three principles.
- Vary openings, structure, tone and sentence length across the batch. No two scenarios may start the same way.

ANSWER RULES
- Keys in this order: rationale, principles, issues, missing_information, verdict, risk_level, required_actions.
- rationale: 2 to 4 plain sentences, at most 80 words, reasoning from scenario facts to the verdict.
- principles: 1 to 3 IDs from the policy (P01 to P16), most important first.
- issues: for NON_COMPLIANT, 1 to 3 items, each naming a concrete fact from the scenario and the problem with it. Otherwise an empty list.
- missing_information: for NEEDS_MORE_INFO, 1 to 3 short questions naming the missing decisive facts. Otherwise an empty list.
- verdict: follow the verdict guide; it must equal the seed's target_verdict.
- risk_level: follow the risk rating guide exactly.
- required_actions: 1 to 3 concrete actions, each starting with a verb, at most 25 words each. For COMPLIANT, give safeguards to keep in place.
- Never cite sections of any Act, fines, penalties, deadlines or authorities that are not written in the policy. Never add facts that are not in the scenario.

FORMAT EXAMPLE (format only; do not reuse its topic, wording or structure)
Seed:
{"seed_id": "EXAMPLE", "business_unit": "marketing and CRM", "data_category": "contact details", "processing_activity": "direct marketing campaign", "artifact_type": "internal email", "target_verdict": "NON_COMPLIANT", "difficulty": "multi_principle", "length_bucket": "medium"}
Item:
{"seed_id": "EXAMPLE",
 "scenario": "Subject: SMS blast for the Avurudu home loan offer\nHi all, for the April campaign we want to send a promotional SMS to all 48,000 customers in the core banking mobile number field, including people who gave their number only for OTP and transaction alerts. Nirmala from Digital says the alert sign-up form does not mention promotions. The SMS gateway is our usual local vendor under the existing contract. Can we go ahead on Monday? Thanks, Shehan (Marketing)",
 "answer": {"rationale": "The numbers were collected for OTP and transaction alerts, and the alert form does not cover promotions. Sending a marketing SMS is a new purpose with no recorded opt-in. The local SMS vendor is already under contract, so the vendor arrangement is not the problem.",
  "principles": ["P15", "P02"],
  "issues": ["Promotional SMS would go to customers who gave their number only for OTP and transaction alerts.", "The alert sign-up form does not mention promotions, so there is no recorded marketing opt-in."],
  "missing_information": [],
  "verdict": "NON_COMPLIANT",
  "risk_level": "MEDIUM",
  "required_actions": ["Restrict the campaign to customers with a recorded SMS marketing opt-in.", "Include a free opt-out instruction in every promotional SMS.", "Get DPO confirmation of the filtered list before Monday."]}}

OUTPUT
Return JSON only, matching the schema: {"items": [...]}, one item per seed, in the same order, copying each seed_id exactly.
```

</details>

<details>
<summary>Rulebook inserted at {RULEBOOK} (in full)</summary>

```markdown
# The Bank: Personal Data Protection Policy (training rulebook)

Status: fictional and simplified. Written for a machine learning exercise. Inspired by the structure of Sri Lanka's Personal Data Protection Act No. 9 of 2022, as amended by Act No. 22 of 2025. It is not a statement of the law and is not legal advice. Section numbers of the Act are intentionally not used.

## Definitions
- Personal data: any information that identifies, or can identify, a living person (customer, guarantor, employee, visitor).
- Special category data: health, biometric, genetic, religious or philosophical beliefs, political opinions, racial or ethnic origin, sex life, criminal records, and any personal data of a person under 18.
- Processor: an outside party that handles personal data on the Bank's instructions.
- DPO: the Bank's Data Protection Officer.

## P01 Lawful basis for processing
Rule: Every processing activity needs a lawful basis recorded before it starts: consent, performance of a contract with the person, a legal obligation on the Bank, protecting someone's vital interests, a public interest task, or a legitimate interest that does not override the person's rights.
Bank standard: Consent must be specific, informed, freely given and recorded. Silence or a pre-ticked box is not consent. Consent for one purpose does not cover another.
Typical breach: Starting a new activity with personal data when no basis is recorded.
Compliant pattern: KYC data collected because the law requires it; loan data used to run the loan contract.

## P02 Purpose limitation
Rule: Use personal data only for the purposes stated at collection, or purposes compatible with them.
Bank standard: A new purpose needs DPO review and, where needed, fresh consent or another lawful basis under P01.
Typical breach: Mobile numbers collected for OTP and alerts reused for promotions; CCTV footage reused to rate staff productivity.
Compliant pattern: Transaction data used for fraud monitoring that customers were told about.

## P03 Data minimisation
Rule: Collect and share only what is necessary for the purpose.
Bank standard: Forms, extracts and vendor files must drop fields not needed. Full NIC copies only where a legal requirement exists. Use masked data in test environments.
Typical breach: Sending full customer records to a vendor that needs only names and phone numbers.
Compliant pattern: A courier receives only name, delivery address and phone number.

## P04 Accuracy
Rule: Keep personal data accurate and up to date and correct errors without delay.
Bank standard: Corrections confirmed by the customer must be applied in every system that holds the field, including credit bureau reporting, within 5 working days.
Typical breach: A settled loan still reported to the credit bureau as in default weeks after settlement.
Compliant pattern: An address change updated in core banking, cards and collections the same week.

## P05 Storage limitation and deletion
Rule: Keep personal data only as long as needed for the purpose or required by law, then delete or anonymise it.
Bank standard: Follow the Bank retention schedule. Records with a legal retention period are kept for that period, then destroyed. Data of rejected applicants is kept no longer than 12 months unless a legal claim is pending. Backups and archives follow the same schedule.
Typical breach: Keeping rejected loan applications indefinitely for future marketing.
Compliant pattern: Scheduled purge of expired records with a deletion log.

## P06 Security of processing
Rule: Protect personal data with technical and organisational measures that match the risk.
Bank standard: Role-based access and least privilege; encryption at rest and in transit for customer data; multi-factor authentication for remote access; no customer data on personal devices, personal email or personal messaging apps; access logs reviewed; masked data outside production.
Typical breach: Shared admin passwords; a customer list sent through a personal WhatsApp or Gmail account; an unencrypted USB drive.
Compliant pattern: Vendor access through the Bank VPN with MFA and named accounts.

## P07 Transparency and notice
Rule: At or before collection, tell people who the controller is, why the data is collected, who receives it, how long it is kept, whether it leaves Sri Lanka, and how to use their rights.
Bank standard: Privacy notice on forms, the app and the website; CCTV signs; a call recording announcement; the notice is updated before any new purpose starts.
Typical breach: Recording calls without the announcement; an app collecting location without telling users.
Compliant pattern: The app shows a clear notice and a permission prompt before using location.

## P08 Data subject rights requests
Rule: People can ask to access, correct or erase their data, withdraw consent, and object to some processing.
Bank standard: Log every request in the rights register; verify identity; acknowledge within 5 working days; complete within one month. One extension of up to two more months is allowed only with DPO approval and written reasons to the person. Erasure may be refused where the law requires the Bank to keep the data, but the person must be told why.
Typical breach: Ignoring an erasure request; asking for documents not needed to verify identity.
Compliant pattern: Access request verified at the branch and answered within three weeks.

## P09 Automated decisions and profiling
Rule: Decisions made solely by automated means that significantly affect a person, such as declining credit, need safeguards.
Bank standard: Tell the customer the decision was automated, give the main reasons, and offer human review on request. Models must be documented and monitored for bias. A DPIA (P14) is needed before launch.
Typical breach: Automatic loan declines with no explanation and no way to ask for review.
Compliant pattern: Credit model output reviewed by an officer before any decline.

## P10 Special category and children's data
Rule: Special category data needs explicit consent or another specific justification, plus stronger protection.
Bank standard: Biometric templates are stored encrypted, separate from core banking, and never shared with vendors for their own purposes. Health data only for insurance-linked products with explicit consent. For anyone under 18, consent from a parent or legal guardian. No profiling or marketing aimed at minors.
Typical breach: Marketing campaigns aimed at minor account holders; staff fingerprints from the attendance system reused for another purpose.
Compliant pattern: Guardian-signed consent on file for a minor's savings account.

## P11 Processors and third-party sharing
Rule: Share personal data with outside parties only with a lawful basis. Processors must be bound by a written contract.
Bank standard: The contract covers processing only on Bank instructions, confidentiality, security, breach notice to the Bank within 24 hours, approval of sub-processors, and return or deletion at the end. Vendor due diligence happens before onboarding. Disclosure to police or government bodies only on a written request with a legal basis, routed through Legal and the DPO.
Typical breach: A vendor starts work before a contract is signed; data given to police on a phone call.
Compliant pattern: A signed processing agreement and security review before go-live.

## P12 Cross-border data transfers
Rule: Personal data may leave Sri Lanka only where the Bank ensures the receiving party will protect it to the standard of this policy, using safeguards accepted by the Data Protection Authority, or where another lawful ground applies, such as the person's explicit informed consent or necessity for a contract with the person.
Bank standard: Every transfer, including cloud hosting and remote support access from abroad, needs a transfer assessment approved by the DPO, contractual safeguards, and an entry in the transfer register.
Typical breach: Moving customer data to an overseas software service without an assessment.
Compliant pattern: Assessed and contracted transfer recorded in the register before go-live.

## P13 Personal data breach management
Rule: Personal data breaches must be contained, assessed, recorded and, where required, reported.
Bank standard: Any staff member who suspects a breach reports it to the DPO and Information Security within 24 hours of becoming aware. The DPO assesses the risk and decides on notifying the Data Protection Authority and affected people in line with the Authority's rules. Every breach and near miss goes in the breach register.
Typical breach: Hiding a misdirected email; waiting for an investigation to finish before reporting.
Compliant pattern: Lost laptop with full-disk encryption reported the same day and logged.

## P14 Data protection impact assessment (DPIA)
Rule: High-risk processing needs a DPIA before it starts.
Bank standard: A DPIA is required for new products or systems that process personal data at scale, special category data, systematic monitoring (including CCTV analytics and employee monitoring), automated decisions under P09, new technologies such as AI models trained on customer data, and large cross-border transfers. The DPO signs off.
Typical breach: Launching face recognition at branch entrances with no DPIA.
Compliant pattern: DPIA completed and signed off before a pilot starts.

## P15 Direct marketing
Rule: Use personal data for direct marketing only with the person's consent, and stop when they object.
Bank standard: Opt-in consent recorded per channel (SMS, email, phone). Every message carries a free opt-out. Opt-outs are honoured within 2 working days and kept on a suppression list. Bought or shared lists are not used unless the source consent covers the Bank.
Typical breach: A campaign to alert-only numbers; continuing to call a customer who opted out.
Compliant pattern: Campaign list filtered to recorded opt-ins and the suppression list removed.

## P16 Accountability, records and DPO consultation
Rule: The Bank must be able to show that it complies.
Bank standard: Maintain the data protection management programme, the record of processing activities, consent records and staff training; consult the DPO before new processing; document decisions and approvals.
Typical breach: A new data sharing arrangement agreed by email with no DPO review and no record.
Compliant pattern: Processing activity added to the record with DPO sign-off.

## Verdict guide
- COMPLIANT: the request states enough facts to show the activity meets every relevant principle.
- NON_COMPLIANT: the request states facts that clearly break at least one principle.
- NEEDS_MORE_INFO: the outcome depends on a decisive fact the request does not give (for example whether consent was recorded, whether a contract exists, or where a vendor hosts data).
- Never assume facts in either direction.

## Risk rating guide
- COMPLIANT: always LOW.
- NON_COMPLIANT: HIGH if any high trigger applies; LOW only for a minor procedural gap with no exposure of data; otherwise MEDIUM.
- NEEDS_MORE_INFO: rate as if the missing fact turns out unfavourable, using the same triggers.
- High triggers: special category or children's data is involved in the gap; data has already been disclosed to, or is accessible by, an unauthorised party; a cross-border transfer without safeguards; solely automated decisions with significant effect and no human review; a breach affecting more than 1,000 people.
```

</details>

Other prompts: student [`prompts/student_system_prompt.md`](prompts/student_system_prompt.md), judge
[`prompts/judge_system_prompt.md`](prompts/judge_system_prompt.md) + [`prompts/judge_user_template.md`](prompts/judge_user_template.md),
RAG [`prompts/rag_user_template.md`](prompts/rag_user_template.md).

**Generation budget maths (plan estimates; measured numbers are in [Results](#generation)):** per call (4 examples) about 400 uncached
input tokens, about 1,700 output tokens and about 300 reasoning tokens, so about 2.4K counted tokens, because the roughly 3K-token
system prefix is cached and cached tokens do not count. 200K tokens/day gives about 80 calls, about 320 examples per day from
gpt-oss-120b; 8K TPM allows about 3 calls per minute, so a day's quota takes about 30 minutes. Target: 400 seeds, about 10% rejection,
about 360 accepted. If quota runs out, top up the next day or with gpt-oss-20b (at most 25%, tagged).

## Hyperparameters

`configs/config.yaml` carries the same values with a one-line comment each. Data-driven values (max length, warmup steps, the chosen learning rate and dropout) are rendered in [Results > Training](#training) from `reports/`.

| Parameter | Value | Reason |
|---|---|---|
| Base model | Llama-3.2-3B-Instruct | Different family from teacher and judge; fits T4 for QLoRA and for fp16 merge; instruct tuned, so the zero-shot baseline is meaningful. |
| Quantisation | 4-bit NF4 | Required by the brief. NF4 is designed for normally distributed weights (QLoRA paper) and cuts base weights to about a quarter of fp16. |
| Double quantisation | True | Also quantises the quantisation constants; saves roughly 0.4 bits per parameter for free. |
| Compute dtype | float16 | T4 (Turing) has no native bf16, so fp16 is the fast correct choice. |
| LoRA rank r | 16 | Task is format plus policy mapping over about 300 examples. r=8 risks underfitting the 16-way principle selection; r=64 adds capacity we cannot fill and overfits faster. 16 is the middle ground; trainable params printed. |
| LoRA alpha | 32 | Scaling alpha/r = 2, a common stable setting; keeps update size sensible at r=16. |
| LoRA dropout | 0.1 | QLoRA paper used 0.1 for models up to 13B; small dataset needs regularisation. |
| Target modules | q, k, v, o, gate, up, down projections | QLoRA paper found adapting all linear layers is needed to match full fine-tuning quality; attention-only LoRA underperforms. |
| LoRA bias | none | Standard; training biases adds little and complicates merging. |
| Learning rate | 2e-4 (picked from {1e-4, 2e-4} by val loss) | QLoRA paper default for small models; confirmed by sweep, not assumed. Final value comes from `sweep.csv`. |
| LR scheduler | cosine | Smoothly lowers the LR in the last epoch, which helps validation loss keep falling instead of bouncing. |
| Warmup | 5% of total steps (computed, about 3 steps) | Avoids a large first update on fresh LoRA weights; short because total steps are few. |
| Epochs | 3 | About 16 optimiser steps per epoch at this data size; 3 epochs (about 48 steps) is enough to learn the format and policy mapping. More epochs start memorising (val loss turns up). |
| Per-device train batch size | 4 | Largest size that fits the T4 with checkpointing and headroom, from the memory probe. |
| Gradient accumulation | 4 | Effective batch 16 for stabler gradients on a small, varied dataset. |
| Max sequence length | data-driven (expected 1152 or 1280) | Set to the longest example rounded up to 128, so nothing is truncated (truncation would cut the end-of-turn token). |
| Loss | completion only | Training on the system prompt and scenario wastes capacity and inflates loss with tokens we never generate. |
| Packing | False | Examples are short and few; packing complicates per-example completion masking and makes per-epoch loss harder to read. |
| Optimiser | paged_adamw_8bit | Paged optimiser from the QLoRA paper; 8-bit states save memory and avoid OOM spikes. |
| Weight decay | 0.0 | QLoRA paper setting; regularisation comes from dropout and few epochs. |
| Max grad norm | 0.3 | QLoRA paper setting; guards against spikes in fp16. |
| Gradient checkpointing | True (use_reentrant False) | Trades compute for memory so batch 4 fits; non-reentrant mode is the recommended path with PEFT. |
| fp16 / bf16 | True / False | Matches T4 hardware. |
| Eval and save strategy | epoch (plus step-0 eval) | Rubric asks for per-epoch losses; step 0 gives the starting point. |
| load_best_model_at_end | True, metric eval_loss | Safety net if the last epoch is not the best; with monotonic decrease it is the last epoch anyway. |
| save_total_limit | 3 | Keeps Drive usage small. |
| Logging steps | 2 | About 8 points per epoch, enough to average a per-epoch train loss. |
| Seed / data_seed | 42 / 42 | Reproducibility. |
| Eval batch size | 4 | Same memory budget as training. |
| group_by_length | False | Dataset is small; random order is simpler to reason about. |
| torch_compile | False | Not reliable with 4-bit layers on T4; not worth the risk. |
| NEFTune | not used | Keeps the effect of fine-tuning attributable to data and LoRA alone. |
| Generation (eval) | greedy, max_new_tokens 512 | Deterministic comparisons across systems; 512 covers the longest reference with margin. |
| Teacher sampling | temperature 1.0, top_p 1.0, reasoning low | Diversity from sampling plus seeds; quality enforced by strict schema and validators; low reasoning saves daily tokens. |
| Judge sampling | temperature 0, reasoning none | Repeatable scores; instruct mode is cheaper and enough for rubric scoring. |

## Results

<!-- RESULTS:START -->
### Dataset

#### Generation

_Pending: teacher generation (notebook Section 2A.4)._

#### Filters

_Pending: `python -m src.validate_data` (notebook Section 2A.5)._

#### Split

_Pending: `python -m src.split_format` (notebook Section 2A.7)._

#### Human audit

_Pending: JP's 30-item audit (gate G3), then `python -m src.validate_data --audit-summary`._

#### Diversity

_Pending: `python -m src.diversity` (notebook Section 2A.6)._

### Training

_Pending: LR sweep and final run (notebook Section 2B)._


### Hugging Face Hub

_Pending: merge and push (notebook Section 2B.13)._

### Evaluation

_Pending: inference and metrics (notebook Section 2C)._

### Manual review and hallucination rate

_Pending: JP labels reports/manual_review.csv (gate G5), then `python -m src.manual_review --score`._

### RAG fallback (bonus)

_Pending: RAG fallback (notebook Section 2D)._

<!-- RESULTS:END -->

## Qualitative analysis

<!-- JP writes these two paragraphs from reports/evidence_pack.md (notebook cell 2C.8), citing example IDs. -->
_To be written by JP._

1. _Where fine-tuning helped._
2. _Remaining failure modes and the fix for each._

## Limitations

- **Single teacher.** Labels come from one model; test references share its biases. The 30-item audit and the `reference_wrong`
  count in the manual review estimate the noise. Next step: a human-labelled gold set of about 50 items.
- **Synthetic distribution.** Real requests are messier than generated ones.
- **Small test set** (about 10% of the data), so the bootstrap intervals are wide; they are reported, not hidden.
- **Validation loss is not answer quality.** It measures the likelihood of the teacher's tokens, which is why generation metrics,
  a judge from another family and blind human review are reported too.
- **Fictional policy, English only, not legal advice.**
- **RAG is out of distribution.** The model never saw retrieved context during training.

## How to run (step by step)

Everything runs from the notebook on Google Colab. Each cell calls a module in `src/` and prints the result, so you can also run any
step from a terminal (see [Command line reference](#command-line-reference)).

| Notebook section | Runtime | What it spends | Typical time (plan estimate) |
|---|---|---|---|
| 0 Setup | any | nothing | 3 to 5 min (installs) |
| 2A Data | CPU is enough | Groq teacher quota (resumable, stops at the daily budget) | about 30 min per day of quota |
| 2B Training, merge, push | **T4 GPU** | Colab GPU time | about 30 to 40 min per sweep run, plus probe, overfit test and merge |
| 2C Evaluation | **T4 GPU** | GPU time, Groq judge quota | about 30 to 60 min |
| 2D RAG bonus | **T4 GPU** | GPU time | about 15 min |
| 2E Package | any | nothing | 1 min |

### 1. One-time setup

1. Push this repository to GitHub as `CDAZZDEV-MLE-<YourName>`, then replace `<gh_user>` and `<YourName>` in: the Colab badge at the
   top of this file, the first markdown cell and cell 0.1 of the notebook (`GITHUB_REPO_URL`), and the root `README.md`.
2. Get the keys: a **Groq API key** (console.groq.com, free plan) and a **Hugging Face token with write access**
   (huggingface.co/settings/tokens). Weights & Biases is optional.
3. Open the notebook in Colab (the badge, or File > Upload notebook). Open the **Secrets** panel (key icon on the left) and add
   `GROQ_API_KEY` and `HF_TOKEN` (optional `WANDB_API_KEY`, `JUDGE_API_KEY`), each with notebook access switched on.
   Secrets are never printed, never written to files and never committed.
4. Run **Section 0** (cells 0.1 to 0.5). It mounts Google Drive, clones the repository into `MyDrive/cdazz_task2/repo`, installs the
   latest libraries, writes `requirements-colab-lock.txt` on the first run, loads the secrets and runs the unit tests.
5. Recommended: from then on, open the notebook **from Drive** (`MyDrive/cdazz_task2/repo/task2_genai/task2_genai.ipynb`), so the
   notebook outputs, `data/`, `reports/` and checkpoints are all saved in the same place. (If the repository is private, upload the
   repository folder to `MyDrive/cdazz_task2/repo` instead of cloning.)

Whenever you switch runtime type or reconnect, re-run Section 0 first.

### 2. Build the dataset (Section 2A, CPU runtime is enough)

1. **2A.1** prints the student prompt and the full rendered teacher prompt with its token count.
2. **2A.2 Gate G1:** provider check. It must print a valid JSON reply from the teacher and the judge. If Groq rejects a parameter
   (for example strict mode on the judge), change `configs/config.yaml` (`judge.strict_schema: false`, `extra_body`, or the model id)
   and re-run. Do not change the code.
3. **2A.3** builds the 400 stratified seeds (`data/seeds.jsonl` is already committed; it is only rebuilt if missing).
4. **2A.4a Gate G2:** a 2-call smoke run that shows 8 generated items in a table. Read them. Check that the cache hit rate in the
   printed summary is above zero after the second call.
5. **2A.4b** runs the full generation. It stops cleanly at the daily token budget; run the cell again the next day to continue
   (nothing is lost, seeds already done are skipped). Flags in the cell: `USE_FALLBACK_TEACHER` (gpt-oss-20b top-up, at most 25%),
   `RETRY_REJECTED` (re-attempt rejected seeds once all seeds were tried). Once `TARGET_ACCEPTED` (300) is reached, the cell only
   prints the saved summary and spends nothing.
6. **2A.5** runs the five filters and prints how many items each one dropped.
7. **2A.6 Gate G3:** writes `data/audit_log.md` (30 cards). Fill the four lines under every card (label agrees? scenario realistic?
   errors? notes), then re-run the cell to count the answers. If more than 3 of 30 labels are wrong, fix the teacher prompt and
   regenerate the affected slice before training.
8. **2A.7** diversity report and figures, **2A.8** stratified split and chat JSONL, **2A.9** renders the data card.

### 3. Train, merge and push (Section 2B, T4 GPU)

Runtime > Change runtime type > **T4 GPU**, re-run Section 0, then run 2B top to bottom:

1. **2B.0** (optional) smoke test with a 135M model; catches plumbing problems in a couple of minutes.
2. **2B.1 to 2B.3** GPU check, tokenizer check, token statistics (sets `max_length` from the data and asserts nothing is truncated).
3. **2B.4** label-mask check, **2B.5** memory probe and OOM log, **2B.6** overfit test (must pass).
4. **2B.7** learning-rate sweep (1e-4 and 2e-4, 3 epochs each, step-0 evaluation first). If neither run has a validation loss that
   falls at every epoch, it automatically tries 2 epochs, then dropout 0.15, and records every run.
5. **2B.8** shows the chosen run's per-epoch table and loss curves. The chosen sweep run is the final model; nothing is retrained for show.
6. **2B.9 to 2B.11** push the adapter, merge into fp16 weights, run the parity check, push the merged model (public) with its model
   card, then reload it fresh from the Hub and run one example. Check the Hub page in a logged-out browser window.

### 4. Evaluate (Section 2C, T4 GPU)

1. **2C.1 and 2C.2** generate predictions on the test split for base zero-shot, base 3-shot and the fine-tuned model from the Hub.
2. **2C.3** computes every metric with bootstrap confidence intervals and prints the comparison table.
3. **2C.4** runs the LLM judge (resumable; it stops at the daily budget and continues the next day) and reprints the table with judge scores.
4. **2C.5** shows the metric bar chart, confusion matrices and accuracy by verdict and difficulty.
5. **2C.6 Gate G5:** writes `reports/manual_review.csv` (40 rows: 20 items x 2 systems, shuffled, system hidden). Open it in Google
   Sheets or Excel from Drive, fill `label` (`correct`, `partially_correct` or `hallucinated`) and `error_type` for every row using the
   guide in the notebook, save it back as CSV with the same name. Do not open `reports/manual_review_key.csv` until you have finished.
6. **2C.7** scores the labels (hallucination rate with a Wilson 95% interval, wrong verdicts, reference errors, agreement with the judge).
7. **2C.8** writes the evidence pack (`reports/evidence_pack.md`). Write the two analysis paragraphs in notebook cell 2C.9 and in the
   [Qualitative analysis](#qualitative-analysis) section above, citing example IDs.

### 5. RAG fallback bonus (Section 2D, T4 GPU)

**2D.1** builds the ChromaDB store from the rulebook, picks the perplexity threshold on validation, re-asks the routed test items with
retrieved policy extracts and compares before and after. **2D.2** prints one full before-and-after example.

### 6. Package and submit (Section 2E)

1. **2E.1** renders all README tables from `reports/`, refreshes the model card on the Hub with the final numbers and runs the secret scan.
2. **2E.2** zips `data/` and `reports/` for download (if you worked from Drive they are already in the repository folder).
3. Re-run the whole notebook once (Runtime > Run all). Finished steps reload their saved results and say so, so this is quick and
   spends no quota.
4. File > Download .ipynb into `task2_genai/`, then from the repository root:

```bash
python task2_genai/scripts/strip_widget_metadata.py      # removes metadata.widgets so GitHub renders the notebook
python task2_genai/scripts/render_results.py             # README tables from reports/
python task2_genai/scripts/check_secrets.py              # must print OK
python task2_genai/scripts/check_secrets.py --install-hook   # once: blocks any commit that contains a secret
git add -A && git commit -m "Task 2: results and notebook outputs"
```

5. Open the GitHub repository, the notebook page, the Colab badge and the Hub model in a logged-out browser window.

## Where to find every score and output

| What | File (all under `task2_genai/`) | Produced by |
|---|---|---|
| ROUGE-L (full output, rationale, canonical JSON), BERTScore F1, verdict accuracy and macro-F1, risk accuracy, principles micro-F1 and Jaccard, JSON/schema-valid rate, invented ID or statute rate, bootstrap 95% CIs, confusion matrices | `reports/eval_results.json` | cell 2C.3 / `python -m src.metrics` |
| Comparison table (all systems) | this README, [Results > Evaluation](#evaluation) | `scripts/render_results.py` |
| LLM-as-judge scores per item | `reports/judge_scores.jsonl` (summary in `eval_results.json` under `judge`) | cell 2C.4 / `python -m src.judge` |
| Hallucination rate (Wilson CI), correct and partially correct rates, wrong verdicts, judge kappa | `reports/manual_review_summary.json` | cell 2C.7 / `python -m src.manual_review --score` |
| Labelled review sheet and system key | `reports/manual_review.csv`, `reports/manual_review_key.csv` | cell 2C.6, then JP |
| Per-epoch train and validation loss | `reports/training_log.csv`, `reports/figures/loss_curves.png` | cell 2B.7 |
| Sweep (every run) and chosen config | `reports/sweep.csv`, `reports/training_summary.json` | cell 2B.7 |
| Memory probe / OOM log | `reports/oom_log.md`, `reports/memory_probe.json` | cell 2B.5 |
| Overfit test | `reports/overfit_test.json`, `reports/figures/overfit_test.png` | cell 2B.6 |
| Label-mask check | `reports/label_mask_check.json` | cell 2B.4 |
| Token statistics and `max_length` | `reports/token_stats.json` | cell 2B.3 |
| Merge parity | `reports/merge_parity.json` | cell 2B.10 |
| Hub links | `reports/hub_links.json` | cells 2B.9, 2B.10 |
| Raw predictions with log-probs and perplexity | `reports/predictions_{base_zeroshot,base_fewshot,ft}.jsonl` | cells 2C.1, 2C.2 |
| RAG gate, routed share, before/after metrics and example | `reports/rag_results.json` | cell 2D.1 |
| Dataset: generation, filters, split, audit, diversity | `reports/generation_summary.json`, `filter_report.json`, `split_sizes.json`, `audit_summary.json`, `diversity_report.json` | Section 2A |
| Evidence pack for the analysis | `reports/evidence_pack.md` | cell 2C.8 |
| Figures | `reports/figures/*.png` | Sections 2A, 2B, 2C |
| Checkpoints, adapters, merged weights | Google Drive `MyDrive/cdazz_task2/` (not in git) | Section 2B |

## Command line reference

Local setup (CPU; tests, data pipeline, provider check, metrics):

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r task2_genai/requirements.txt
cp .env.example .env                                    # then fill GROQ_API_KEY, HF_TOKEN
cd task2_genai
```

| Step | Command (run inside `task2_genai/`) |
|---|---|
| Unit tests | `python -m pytest -q` |
| Seeds | `python -m src.seeds` |
| Render teacher prompt and count tokens | `python -m src.generate_data --render-only` |
| Provider check (gate G1) | `python scripts/provider_check.py` |
| Generation smoke run (gate G2) | `python -m src.generate_data --max-calls 2` |
| Full generation (resumable) | `python -m src.generate_data --model openai/gpt-oss-120b --batch-size 4 --max-calls 120` |
| Generation summary (no API calls) | `python -m src.generate_data --summary` |
| Fallback top-up | `python -m src.generate_data --model openai/gpt-oss-20b` |
| Clean | `python -m src.validate_data` |
| Audit cards / audit count | `python -m src.validate_data --audit 30` / `python -m src.validate_data --audit-summary` |
| Diversity report | `python -m src.diversity` |
| Split and format | `python -m src.split_format` |
| CPU smoke test | `python scripts/smoke_cpu_train.py` |
| Training (GPU) | `python -m src.train --stage all` (or `token-stats`, `label-mask`, `probe`, `overfit`, `sweep`) |
| Merge, parity, push (GPU) | `python -m src.merge_push` (`--skip-push` to stay local, `--card-only` to refresh the model card) |
| Inference (GPU) | `python -m src.infer --system base_zeroshot`, `--system base_fewshot`, `--system ft --model <hf_user>/Llama-3.2-3B-PDPA-Triage` |
| Metrics | `python -m src.metrics` |
| Judge | `python -m src.judge` |
| Manual review | `python -m src.manual_review --build`, then `--score`, then `--evidence` |
| README tables | `python scripts/render_results.py` (`--print` to preview) |
| Secret scan | `python scripts/check_secrets.py` |
| Notebook fix for GitHub | `python scripts/strip_widget_metadata.py` |

Every path, model id and hyperparameter is in [`configs/config.yaml`](configs/config.yaml), with a one-line reason per value.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Provider check: 400 on the judge's strict schema | Set `judge.strict_schema: false` in `config.yaml`; Pydantic still validates every reply. |
| A model id is no longer served (for example the preview judge) | Change `judge.model` (another family through any OpenAI-compatible endpoint: set `judge.base_url` and `judge.api_key_env`). Last resort `openai/gpt-oss-20b`, and state the teacher-family bias. |
| `BudgetExhausted` / generation stops | Expected at the daily budget. Re-run the cell the next day, or top up with `USE_FALLBACK_TEACHER = True`. |
| Cache hit rate stays at zero | The system prefix is not byte-stable. Do not spend more quota until it is fixed. |
| Overfit test fails | The pipeline is broken (masking, template, pad token). Check the label-mask output in 2B.4 first. |
| Validation loss does not fall every epoch | The sweep's fallbacks (2 epochs, then dropout 0.15) run automatically; every run is in `sweep.csv`. |
| CUDA out of memory | Keep batch 4 x accumulation 4 with gradient checkpointing (see `oom_log.md`); lower `inference.batch_size` for generation. |
| Colab disconnects mid-training | Re-run Section 0 and the same cell; training resumes from the last checkpoint on Drive. |
| Llama access problems | The default student is the ungated mirror `unsloth/Llama-3.2-3B-Instruct` (identical weights); `student.fallback_model_id` is Phi-4-mini. |
| GitHub shows "Invalid Notebook" | `python task2_genai/scripts/strip_widget_metadata.py` and commit again. |
| Out of Colab GPU quota | Kaggle Notebooks (free T4) run the same notebook; record where the final run happened. |
