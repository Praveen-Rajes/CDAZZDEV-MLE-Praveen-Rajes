# AI-ASSISTED: Claude Code (Claude Opus 5.5), Prompt: 'Load config.yaml into a typed object per plan Phase 0', Date: 2026-10-06
"""Typed access to configs/config.yaml.

Every section is a Pydantic model with extra="forbid", so a typo in the YAML fails
loudly instead of silently falling back to a default.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict

TASK_ROOT = Path(__file__).resolve().parents[1]          # task2_genai/
REPO_ROOT = TASK_ROOT.parent                             # repository root
DEFAULT_CONFIG = TASK_ROOT / "configs" / "config.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ProjectCfg(_Strict):
    seed: int


class PathsCfg(_Strict):
    seeds: str
    generations: str
    rejected: str
    call_log: str
    clean: str
    splits_dir: str
    audit_log: str
    reports_dir: str
    figures_dir: str
    prompts_dir: str
    rulebook: str
    chroma_dir: str
    work_dir: str


class ProviderCfg(_Strict):
    base_url: str
    api_key_env: str
    timeout_s: float


class RateLimitsCfg(_Strict):
    rpm: int
    tpm: int
    daily_token_budget: int


class LLMRoleCfg(_Strict):
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    strict_schema: bool
    extra_body: Dict[str, object]
    rate_limits: RateLimitsCfg
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None


class TeacherFallbackCfg(LLMRoleCfg):
    max_share: float


class JudgeCfg(LLMRoleCfg):
    invalid_retries: int
    systems: List[str]
    shuffle_seed: int


class StudentCfg(_Strict):
    model_id: str
    official_model_id: str
    fallback_model_id: str
    pad_token: Optional[str]
    pin_chat_template_date: Optional[str]
    attn_implementation: str


class SeedsCfg(_Strict):
    n: int
    verdict_quota: Dict[str, float]
    difficulty_quota: Dict[str, float]
    length_quota: Dict[str, float]
    length_words: Dict[str, Tuple[int, int]]
    length_tolerance: float
    business_unit: List[str]
    data_category: List[str]
    processing_activity: List[str]
    artifact_type: List[str]


class GenerationCfg(_Strict):
    batch_size: int
    max_calls: int
    summary_every: int


class FiltersCfg(_Strict):
    embed_model: str
    near_dup_threshold: float
    audit_n: int


class SplitCfg(_Strict):
    train: float
    val: float
    test: float
    seed: int
    cross_split_threshold: float


class QuantCfg(_Strict):
    load_in_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_use_double_quant: bool
    bnb_4bit_compute_dtype: str


class LoraCfg(_Strict):
    r: int
    alpha: int
    dropout: float
    target_modules: List[str]
    bias: str
    task_type: str


class TrainingCfg(_Strict):
    learning_rate: float
    lr_scheduler_type: str
    warmup_fraction: float
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    per_device_eval_batch_size: int
    max_length: Optional[int]
    max_length_multiple: int
    completion_only_loss: bool
    packing: bool
    optim: str
    weight_decay: float
    max_grad_norm: float
    gradient_checkpointing: bool
    fp16: bool
    bf16: bool
    eval_strategy: str
    save_strategy: str
    load_best_model_at_end: bool
    metric_for_best_model: str
    save_total_limit: int
    logging_steps: int
    seed: int
    data_seed: int
    group_by_length: bool
    torch_compile: bool
    report_to: str
    step0_eval: bool
    step0_train_eval: bool


class SweepCfg(_Strict):
    learning_rates: List[float]
    fallback_epochs: int
    fallback_dropout: float


class OverfitCfg(_Strict):
    n_examples: int
    max_steps: int
    learning_rate: float
    lora_dropout: float
    target_loss: float


class MemoryProbeCfg(_Strict):
    per_device_train_batch_size: int
    gradient_checkpointing: bool
    steps: int


class InferenceCfg(_Strict):
    max_new_tokens: int
    batch_size: int
    fewshot_per_verdict: int


class MetricsCfg(_Strict):
    bertscore_lang: str
    bertscore_rescale: bool
    bootstrap_resamples: int
    bootstrap_seed: int
    rationale_max_words_scoring: int


class ManualReviewCfg(_Strict):
    n_items: int
    seed: int
    systems: List[str]


class RagCfg(_Strict):
    collection: str
    embed_model: str
    top_k: int
    tau_floor_percentile: float


class HubCfg(_Strict):
    username: Optional[str]
    merged_repo: str
    adapter_repo: str
    private: bool
    max_shard_size: str


class SmokeCfg(_Strict):
    model_id: str
    n_train: int
    n_eval: int
    n_infer: int
    lora_r: int
    max_length: int
    epochs: int
    max_new_tokens: int


class Config(_Strict):
    project: ProjectCfg
    paths: PathsCfg
    provider: ProviderCfg
    teacher: LLMRoleCfg
    teacher_fallback: TeacherFallbackCfg
    judge: JudgeCfg
    student: StudentCfg
    seeds: SeedsCfg
    generation: GenerationCfg
    filters: FiltersCfg
    split: SplitCfg
    quantization: QuantCfg
    lora: LoraCfg
    training: TrainingCfg
    sweep: SweepCfg
    overfit: OverfitCfg
    memory_probe: MemoryProbeCfg
    inference: InferenceCfg
    metrics: MetricsCfg
    manual_review: ManualReviewCfg
    rag: RagCfg
    hub: HubCfg
    smoke: SmokeCfg

    # ---- path helpers -------------------------------------------------
    def path(self, name: str) -> Path:
        """Absolute path for a key in `paths` (relative entries resolve against task2_genai/)."""
        p = Path(getattr(self.paths, name))
        return p if p.is_absolute() else TASK_ROOT / p

    def prompt_path(self, filename: str) -> Path:
        return self.path("prompts_dir") / filename

    def report_path(self, filename: str) -> Path:
        return self.path("reports_dir") / filename

    def figure_path(self, filename: str) -> Path:
        return self.path("figures_dir") / filename

    def split_path(self, split: str) -> Path:
        return self.path("splits_dir") / f"{split}.jsonl"

    def work_dir(self) -> Path:
        """Checkpoint root: env CDAZZ_WORK_DIR, else config (Drive in Colab), else task2_genai/outputs."""
        env = os.environ.get("CDAZZ_WORK_DIR")
        if env:
            return Path(env)
        p = Path(self.paths.work_dir)
        if p.exists() or p.parent.exists():
            return p
        return TASK_ROOT / "outputs"

    def role_base_url(self, role: LLMRoleCfg) -> str:
        return role.base_url or self.provider.base_url

    def role_api_key_env(self, role: LLMRoleCfg) -> str:
        return role.api_key_env or self.provider.api_key_env


def load_config(path: Optional[Path | str] = None) -> Config:
    """Load and validate the YAML config."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)


def load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
