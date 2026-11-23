# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic configuration for recipes (serialization of the composition API).

Every key changes behavior or is rejected (ADR 0014). Unknown keys are errors,
and keys that were removed raise a message naming their replacement, so a
config can never look like it chose something the runtime ignores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Strict(BaseModel):
    """Forbid unknown keys, and explain removed ones."""

    model_config = ConfigDict(extra="forbid")
    removed_keys: ClassVar[dict[str, str]] = {}

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            hits = [key for key in cls.removed_keys if key in data]
            if hits:
                raise ValueError(
                    "; ".join(f"`{key}` was removed. {cls.removed_keys[key]}" for key in hits)
                )
        return data


class TemporalSplitConfig(_Strict):
    train_end: str
    validation_end: str
    label_delay: str = "7d"


class IdentityTheftConfig(_Strict):
    """The spliced identity-theft task on a pragmatiq book (ADR 0015)."""

    victim_fraction: float = 0.2
    """Share of each probe side's eligible users who become victims."""
    negative_window: str = "7d"
    """How far before the theft day the victim's own card transactions are labelled legitimate."""
    theft_lookback: str = "28d"
    """Theft days fall in this span before the end of their side's label window."""
    donors: Literal["matched", "random"] = "matched"
    """`matched`: same home country and a median amount within `amount_ratio`.
    `random` is easier: the profile gives a foreign donor away."""
    amount_ratio: float = 1.5
    min_prior_card_events: int = 20
    """Card transactions a victim needs before the theft day."""


class PretrainWindowsConfig(_Strict):
    """Tile each entity's history into pretraining windows instead of one sequence.

    Evaluation always pools one embedding per entity; this only widens the
    pretraining corpus, which otherwise sees each entity's most recent
    `sequence_length` events and discards the rest of its history. Windows do
    not overlap.
    """

    events: int = Field(ge=1)
    """Events per pretraining window."""
    max_tokens: int | None = None
    """Row width for the windowed corpus. Defaults to `data.max_tokens`.

    Probe sequences and pretraining windows have different lengths, so they need
    different row widths; padding windows out to `max_tokens` would waste the
    difference on every row.
    """


_PRETRAIN_WINDOWS = "Use data.pretrain_windows: {events, max_tokens}."


class DataConfig(_Strict):
    removed_keys: ClassVar[dict[str, str]] = {
        "pretrain_window_events": _PRETRAIN_WINDOWS,
        "pretrain_max_tokens": _PRETRAIN_WINDOWS,
        "pretrain_window_stride": "Pretraining windows do not overlap; delete the key.",
        "entity_key": "Adapters map their own columns onto TransactionEvent; delete the key.",
        "event_time_key": "Adapters map their own columns onto TransactionEvent; delete the key.",
        "event_order_key": "Adapters map their own columns onto TransactionEvent; delete the key.",
        "pretrain_window_split_by": (
            "Pretraining windows tile each entity's whole history, as its probe histories "
            "do (ADR 0017); delete the key."
        ),
        "pretrain_observation_window": (
            "Pretraining windows use data.observation_window; widen it instead (ADR 0017)."
        ),
    }

    adapter: str = "synthetic"
    """A registered adapter: `synthetic`, `tabformer`, or `pragmatiq`."""
    path: str | None = None
    """Local book root. `pragmatiq` with no path and no `TFM_PRAGMATIQ_ROOT`
    generates a small synthetic fixture in the run directory."""
    observation_window: str = "90d"
    sequence_length: int = 16
    """Most recent events kept per sequence."""
    max_tokens: int = 256
    """Row width: tokens per sequence (flat layouts) or per event (key_value_time)."""
    n_entities: int = 32
    """Entities the `synthetic` adapter and the pragmatiq fixture generate."""
    events_per_entity: int = 16
    """Events per entity for the `synthetic` adapter and the pragmatiq fixture."""
    max_entities: int | None = None
    """Entities sampled from a local book. Defaults to `n_entities`."""
    pretrain_windows: PretrainWindowsConfig | None = None
    """Unset keeps one pretraining sequence per entity."""
    sample_seed: int | None = None
    split_protocol: Literal["timestamp", "precomputed"] = "timestamp"
    temporal_split: TemporalSplitConfig
    fraud_labels: Literal["native", "spliced_identity_theft"] = "native"
    """Where the fraud probe's labels come from.

    `native` reads the book's own event labels. `spliced_identity_theft`
    (pragmatiq only) builds the ADR 0015 task from the unmodified book and
    ignores the book's fraud labels; its sample is drawn uniformly rather than
    stratified by them.
    """
    identity_theft: IdentityTheftConfig = Field(default_factory=IdentityTheftConfig)
    token_cache: str | None = None
    """Directory of a token cache (`tfm cache-tokens`, ADR 0016).

    Key-value-time (pragmatiq): the vocabulary is the cache's, fit on the train
    side of every user it holds, and pretraining reads all of those users from
    disk. Probes keep their `max_entities` sample, which must be inside the cache.

    Flat fields (TabFormer, `transaction_field_tokenizer`): the cache holds the
    train split of the `max_entities` sample, and pretraining windows are read
    from it. `train` then builds no event objects at all; probes still do.
    """

    @model_validator(mode="after")
    def _splice_needs_pragmatiq(self) -> DataConfig:
        if self.fraud_labels == "spliced_identity_theft" and self.adapter != "pragmatiq":
            raise ValueError(
                "data.fraud_labels `spliced_identity_theft` is built from a pragmatiq book; "
                f"data.adapter is {self.adapter!r}."
            )
        return self


_TOKENIZER_CHOOSES_FIELDS = (
    "The tokenizer decides which fields it encodes (features.tokenizer); delete the key."
)


class FeaturesConfig(_Strict):
    removed_keys: ClassVar[dict[str, str]] = {
        "categorical_fields": _TOKENIZER_CHOOSES_FIELDS,
        "numeric_fields": _TOKENIZER_CHOOSES_FIELDS,
        "timestamp_features": _TOKENIZER_CHOOSES_FIELDS,
    }

    tokenizer: str = "transaction_field_tokenizer"
    """A registered tokenizer name. Its layout must be one the granularity accepts."""
    fields: list[str] | None = None
    """`transaction_field_tokenizer` only: the fields each event emits, in order.
    Unset keeps all 12. Drop identity fields such as `CUST`, which repeats the
    user id in every event."""
    categorical_keys: list[str] = Field(default_factory=list)
    """Event keys the key-value-time tokenizers keep categorical although their
    values parse as numbers: codes such as `mcc`, whose numeric order means nothing."""
    hashed_keys: list[str] = Field(default_factory=list)
    """`key_value_time` only. High-cardinality identity keys (merchant, device,
    counterparty) encoded as one hash-bucket token instead of their bytes, so a
    repeated value is one repeated token. Experimental."""
    hash_buckets: int = Field(default=8192, ge=2)
    """Buckets per hashed key; each key has its own range."""
    exclude_keys: list[str] = Field(default_factory=list)
    """`key_value_time` only. Event keys never tokenized: identity keys such as
    TabFormer's `user`, which repeats one value in every event of a history."""


class DecoderConfig(_Strict):
    removed_keys: ClassVar[dict[str, str]] = {
        "vocab_size": (
            "The vocabulary size comes from the fitted tokenizer and is recorded in "
            "tokenizer_manifest.json; delete the key."
        ),
    }

    depth: int = 2
    mask: Literal["causal"] = "causal"
    hidden_size: int = 128
    num_heads: int = 4
    dropout: float = 0.1
    max_seq_len: int = 256
    ffn_size: int | None = None
    """Feed-forward width. Defaults to `4 * hidden_size` when unset."""


class GranularityConfig(_Strict):
    type: Literal["flat", "flat_encoder", "profile_history", "profile_event_decoder"] = "flat"
    """`flat_encoder` is the experimental bidirectional encoder on flat rows (ADR 0019)."""
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)
    """Width and heads for every stack; `depth` is the flat decoder's, the flat
    encoder's, or the v2 history decoder's. `mask` does not apply to `flat_encoder`."""
    profile_depth: int = 1
    event_depth: int = 2
    history_depth: int = 1
    """`profile_history` only: bidirectional history-encoder depth."""
    varlen: bool = False
    """Hierarchical stacks only: run the event encoder padding-free."""

    @model_validator(mode="after")
    def _keys_apply_to_type(self) -> GranularityConfig:
        applies = {
            "flat": set(),
            "flat_encoder": set(),
            "profile_history": {"profile_depth", "event_depth", "history_depth", "varlen"},
            "profile_event_decoder": {"profile_depth", "event_depth", "varlen"},
        }[self.type]
        hierarchical = {"profile_depth", "event_depth", "history_depth", "varlen"}
        # A key left at its default is not a choice, so a dumped config reloads.
        changed = {
            key
            for key in hierarchical & self.model_fields_set
            if getattr(self, key) != type(self).model_fields[key].default
        }
        stray = sorted(changed - applies)
        if stray:
            raise ValueError(f"granularity.type {self.type!r} does not use {stray}.")
        return self


class CompositionConfig(_Strict):
    """One choice per building-block slot (ADR 0014). The runtime builds exactly this."""

    granularity: GranularityConfig = Field(default_factory=GranularityConfig)
    position: Literal["nope", "rope", "time_rope"] = "nope"
    """`rope` declares the rotary positions the HuggingFace Llama backbone applies."""
    attention: str = "eager"
    objective: Literal[
        "next_event", "next_event_fields", "token_event_key_mlm", "masked_fields"
    ] = "next_event"
    """The primary pretraining objective. `objective.auxiliary` adds more."""
    readout: Literal["last_token", "usr_token", "mean", "last_event_mean"] = "last_token"
    heads: dict[str, Literal["binary"]] = Field(default_factory=lambda: {"fraud": "binary"})


class HuggingFaceConfig(_Strict):
    """Llama-family backbone settings not covered by the neutral `DecoderConfig`.

    Each field defaults to `None` and is then derived from `DecoderConfig`, so a
    config that says nothing here keeps the previous behavior. They exist so a
    published reference architecture's exact Llama config can be reproduced.
    """

    pretrained_path: str | None = None
    num_key_value_heads: int | None = None
    """Grouped-query attention KV heads. Defaults to `num_heads // 4`."""
    rope_theta: float | None = None
    """RoPE base. Defaults to the HuggingFace Llama default."""
    tie_word_embeddings: bool = False


class ModelConfig(_Strict):
    removed_keys: ClassVar[dict[str, str]] = {
        "architecture": "Use model.composition.granularity.type.",
        "causal": (
            "Causality follows from model.composition.granularity.type: `flat` and "
            "`profile_event_decoder` are causal, `flat_encoder` and `profile_history` "
            "are bidirectional."
        ),
        "pooling": "Use model.composition.readout (last_token, usr_token, or mean).",
    }

    composition: CompositionConfig = Field(default_factory=CompositionConfig)
    backbone: str = "eager"
    """`flat` only: a name in `tfm.api.backbones` (`eager`, `huggingface` Llama, or
    the `gru` baseline). Unknown names fail when the model is built."""
    huggingface: HuggingFaceConfig = Field(default_factory=HuggingFaceConfig)


class MlmConfig(_Strict):
    """Masking rates and smoothing for `token_event_key_mlm`.

    Defaults are the PRAGMA paper's: 15% token, 10% whole-event, 10% key-level
    masking, of which 10% become `[UNK]` and are excluded from the loss.
    """

    p_token: float = 0.15
    p_event: float = 0.10
    p_key: float = 0.10
    p_unk: float = 0.10
    label_smoothing: float = 0.1


class MaskedFieldsConfig(_Strict):
    """Masking rates for `masked_fields` (ADR 0019): BERT's 15% per token, plus
    10% whole events. Chosen tokens become `<mask>`, a random same-field value,
    or stay, in BERT's 80 / 10 / 10 split."""

    p_field: float = Field(default=0.15, ge=0, le=1)
    p_event: float = Field(default=0.10, ge=0, le=1)
    p_mask_token: float = Field(default=0.8, ge=0, le=1)
    p_random_token: float = Field(default=0.1, ge=0, le=1)


class PastReconstructionConfig(_Strict):
    """`past_reconstruction` auxiliary: rebuild earlier events from the current state."""

    horizon: int = 3
    """Reconstruct events `j-1 ... j-horizon` from slot `j + 1`."""
    weight: float = 1.0


class ContrastiveConfig(_Strict):
    """`subsequence_contrastive` auxiliary: two crops of one history embed close together."""

    crop_fraction: float = 0.7
    """Share of each history's events kept in a crop."""
    temperature: float = 0.1
    weight: float = 1.0


AuxiliaryObjective = Literal[
    "token_event_key_mlm", "past_reconstruction", "subsequence_contrastive"
]


class ObjectiveConfig(_Strict):
    removed_keys: ClassVar[dict[str, str]] = {
        "name": "Use model.composition.objective for the primary objective.",
    }

    mlm: MlmConfig = Field(default_factory=MlmConfig)
    """Applies when the primary objective or an `auxiliary` entry is `token_event_key_mlm`."""
    masked_fields: MaskedFieldsConfig = Field(default_factory=MaskedFieldsConfig)
    """Applies when the primary objective is `masked_fields`."""
    auxiliary: list[AuxiliaryObjective] = Field(default_factory=list)
    """`next_event_fields` only. Each entry adds a loss, summed with its weight."""
    past_reconstruction: PastReconstructionConfig = Field(default_factory=PastReconstructionConfig)
    contrastive: ContrastiveConfig = Field(default_factory=ContrastiveConfig)


class MuonConfig(_Strict):
    """`optimizer.name: muon` only."""

    learning_rate: float = Field(default=0.02, gt=0)
    """Muon's peak rate; `training.learning_rate` stays the AdamW groups' rate."""
    momentum: float = Field(default=0.95, ge=0, lt=1)
    adjust_lr: Literal["original", "match_rms_adamw"] = "original"
    """Muon's per-shape rate adjustment. `match_rms_adamw` scales updates to
    AdamW's RMS, so `learning_rate` can be set near an AdamW rate."""


_MUON = "Use training.optimizer.muon: {learning_rate, momentum, adjust_lr}."


class OptimizerConfig(_Strict):
    """Optimizer and learning-rate schedule. Defaults are constant-rate AdamW."""

    removed_keys: ClassVar[dict[str, str]] = {
        "muon_learning_rate": _MUON,
        "muon_momentum": _MUON,
        "muon_adjust_lr": _MUON,
    }

    name: Literal["adamw", "muon"] = "adamw"
    """`muon` puts hidden matrices on Muon and the rest on AdamW (`tfm.training.optim`).
    Experimental; eager and `ddp` backends only."""
    muon: MuonConfig = Field(default_factory=MuonConfig)
    max_grad_norm: float | None = Field(default=None, gt=0)
    """Clip the global gradient norm to this value; `None` does not clip."""
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    lr_schedule: str = "constant"
    """`constant` or `cosine`."""
    warmup_steps: int = 0
    min_learning_rate: float = 0.0
    """Cosine floor; ignored when `lr_schedule` is `constant`."""
    schedule_steps: int | None = None
    """Cosine horizon in steps. Defaults to `training.max_steps`.

    Setting it above `max_steps` stops training partway through the schedule,
    at a learning rate well above the floor. That is how the Blueprint's
    published checkpoint was taken: step 3000 of a 10000-step cosine.
    """


class TrainingConfig(_Strict):
    removed_keys: ClassVar[dict[str, str]] = {
        "packing_seq_length": "Packed rows are model.composition.granularity.decoder.max_seq_len "
        "tokens; delete the key.",
    }

    max_steps: int = 20
    log_every: int | None = Field(default=None, ge=1)
    """Print step, epoch, loss, learning rate, gradient norm, throughput, and
    time left every this many steps (first rank only). `None` prints nothing;
    every run still writes the per-step curve to `train_log.json`."""
    batch_size: int = 8
    learning_rate: float = 3e-4
    precision: Literal["fp32", "bf16"] = "fp32"
    """Compute precision. `bf16` autocasts the forward pass; master weights,
    gradients, and optimizer state stay fp32. The backend must declare it."""
    backend: str = "eager"
    packing: bool = False
    """Pack sequences into full-context rows for dense next-event supervision.

    Training only. Inference and evaluation always use the padded collate, whose
    one-row-per-entity layout the pooled embeddings require.
    """
    shuffle: bool = False
    """Shuffle training rows once and batch order every epoch, both seeded by `seed`.

    Off by default so existing recipes keep their recorded behavior. Without it
    rows are trained in corpus order, which is entity by entity, and every epoch
    repeats that order.
    """
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)

    @field_validator("backend")
    @classmethod
    def _removed_backends(cls, value: str) -> str:
        if value.strip().lower() == "nemo":
            raise ValueError(
                "training.backend: nemo was removed: it was the fsdp2 loop with the model "
                "built through NeMo AutoModel. Use training.backend: fsdp2."
            )
        return value


class FineTuneConfig(_Strict):
    """Optional fine-tuning arm, reported alongside the frozen probe.

    Off by default: the frozen-embedding probe is the reference baseline, and a
    fine-tuned encoder answers a different question. `method` decides which
    encoder weights move; `head` decides what reads the embedding.
    """

    removed_keys: ClassVar[dict[str, str]] = {
        "cross_layers": "joint_fusion uses FineTune's defaults; delete the key.",
        "plr_frequencies": "joint_fusion uses FineTune's defaults; delete the key.",
    }

    enabled: bool = False
    method: Literal["lora", "full"] = "lora"
    """`lora` trains low-rank adapters on `targets`; `full` unfreezes every weight."""
    head: Literal["linear", "joint_fusion"] = "linear"
    """`linear` is a `Binary` head on the embedding. `joint_fusion` is nuFormer's
    DCNv2 over PLR-embedded raw features and the embedding, trained end to end."""
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    targets: list[str] = Field(default_factory=lambda: ["qkv", "out"])
    max_steps: int = 50
    learning_rate: float = 1e-3
    batch_size: int = 8
    validation_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)
    """Share of training entities held out for early stopping on log-loss.

    0 trains `max_steps` with no validation. Above 0, the best checkpoint is
    kept, and training stops after `patience` checks, `eval_every` steps
    apart, without a gain.
    """
    eval_every: int = Field(default=50, ge=1)
    patience: int = Field(default=3, ge=1)


class SupervisedHeadConfig(_Strict):
    """`flat` only. Add a SupervisedHead objective so the Binary head trains end to end.

    Off by default: the frozen-embedding probe is the reference baseline.
    """

    enabled: bool = False
    loss_weight: float = 1.0
    probe_only: bool = False
    """Stop the head loss at the pooled embedding, leaving pretraining to shape it."""


_SUPERVISED_HEAD = "Use fraud_task.supervised_head: {enabled, loss_weight, probe_only}."


class FraudTaskConfig(_Strict):
    removed_keys: ClassVar[dict[str, str]] = {
        "lora": "Use fraud_task.finetune with `method: lora` (and `enabled: true`).",
        "train_head": _SUPERVISED_HEAD,
        "head_loss_weight": _SUPERVISED_HEAD,
        "head_probe_only": _SUPERVISED_HEAD,
    }

    head: Literal["logistic_regression", "xgboost"] = "logistic_regression"
    use_raw_features: bool = True
    """Report the `raw` probe arm, and `combined` when embeddings are also used."""
    use_tfm_embeddings: bool = True
    """Report the `embedding` probe arm, and `combined` when raw features are also used."""
    supervised_head: SupervisedHeadConfig = Field(default_factory=SupervisedHeadConfig)
    finetune: FineTuneConfig = Field(default_factory=FineTuneConfig)

    @model_validator(mode="after")
    def _some_probe_arm(self) -> FraudTaskConfig:
        if not (self.use_raw_features or self.use_tfm_embeddings):
            raise ValueError(
                "fraud_task.use_raw_features and use_tfm_embeddings are both false, "
                "so no probe arm would run."
            )
        return self


class TabularBaselinesConfig(_Strict):
    """Hand-engineered tabular baselines beside every probe (no encoder involved).

    Each row's features are aggregates of the same truncated history the
    encoder reads, plus the profile fields its tokenizer sees. They stand in
    for the task-specific production model a lift is measured against.
    """

    enabled: bool = False
    top_k: int = Field(default=10, ge=1)
    """Most frequent train-side values kept per categorical key."""
    xgboost_trials: int = Field(default=30, ge=0)
    """Optuna trials for each XGBoost arm (needs the `baselines` extra); 0 with no
    `xgboost_params` skips the XGBoost arms."""
    xgboost_params: dict[str, int | float] | None = None
    """One fixed XGBoost setting for every XGBoost arm instead of a search: early
    stopping on the same inner fold picks only the tree count, so arms differ only
    in their features. Set this or `xgboost_trials`, not both."""
    n_jobs: int = Field(default=8, ge=1)
    device: Literal["cpu", "cuda"] = "cpu"
    """Where every XGBoost arm trains; `cuda` uses XGBoost's GPU `hist` method."""
    exclude_keys: list[str] = Field(default_factory=list)
    """Event or profile keys no tabular feature is built from: identifiers such as a
    user id, which a tree would otherwise memorize per entity."""

    XGBOOST_PARAM_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "max_depth",
            "learning_rate",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "reg_lambda",
            "reg_alpha",
            "scale_pos_weight",
        }
    )

    @model_validator(mode="after")
    def _one_xgboost_protocol(self) -> TabularBaselinesConfig:
        if self.xgboost_params is None:
            return self
        if self.xgboost_trials > 0:
            raise ValueError(
                "Set tabular_baselines.xgboost_params or xgboost_trials, not both; "
                "set xgboost_trials: 0 to use the fixed setting."
            )
        unknown = sorted(set(self.xgboost_params) - self.XGBOOST_PARAM_KEYS)
        if unknown:
            raise ValueError(
                f"Unknown xgboost_params {unknown}; allowed: {sorted(self.XGBOOST_PARAM_KEYS)}."
            )
        return self

    @property
    def xgboost_enabled(self) -> bool:
        return self.xgboost_trials > 0 or self.xgboost_params is not None


class HistoryWindowConfig(_Strict):
    """Also score one row per labelled fraud event, with that event's own history."""

    enabled: bool = False
    max_rows: int | None = Field(default=2000, ge=1)
    """Cap on rows per side, sampled with the probe seed. `None` scores every row."""
    train_negatives_per_positive: float | None = Field(default=None, gt=0)
    """Train side only: keep every positive and sample this many negatives per
    positive, within `max_rows`. Unset samples uniformly. The evaluation side is
    always sampled uniformly, so it keeps its natural rate."""


_HISTORY_WINDOW = (
    "Use evaluation.history_window: {enabled, max_rows, train_negatives_per_positive}."
)


class EvaluationConfig(_Strict):
    removed_keys: ClassVar[dict[str, str]] = {
        "metrics": "Every probe reports the full metric set; delete the key.",
        "report_format": "Runs always write metrics.json and metrics.md; delete the key.",
        "history_window_probe": _HISTORY_WINDOW,
        "history_window_max_rows": _HISTORY_WINDOW,
        "history_window_train_negatives_per_positive": _HISTORY_WINDOW,
    }

    precision_at_k: int = 5
    history_window: HistoryWindowConfig = Field(default_factory=HistoryWindowConfig)
    tabular_baselines: TabularBaselinesConfig = Field(default_factory=TabularBaselinesConfig)
    surprise: bool = False
    """Also score `xgboost_raw_embedding:surprise` on every probe with a history: the
    raw row plus the embedding and the model's next-event surprise on the row's
    latest events, each scored from its prefix (`tfm.evaluation.surprise`). Needs
    the tabular XGBoost arms and a model with a next-event field head."""
    inference_batch_size: int = Field(default=8, ge=1)
    """Rows per forward pass when extracting embeddings."""
    inference_precision: Literal["fp32", "bf16"] = "fp32"
    """`bf16` autocasts embedding extraction; its vectors differ from fp32's in
    low-order bits."""

    @model_validator(mode="after")
    def _surprise_needs_xgboost(self) -> EvaluationConfig:
        baselines = self.tabular_baselines
        if self.surprise and not (baselines.enabled and baselines.xgboost_enabled):
            raise ValueError(
                "evaluation.surprise is scored as an XGBoost arm; enable "
                "tabular_baselines with xgboost_params or xgboost_trials > 0."
            )
        return self


class RuntimeConfig(_Strict):
    device: str = Field(
        default="cpu",
        description="cpu, cuda, cuda:N, or auto (CUDA when available).",
    )


class RecipeConfig(_Strict):
    seed: int = 42
    data: DataConfig
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    fraud_task: FraudTaskConfig = Field(default_factory=FraudTaskConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @model_validator(mode="after")
    def _cross_section_rules(self) -> RecipeConfig:
        kind = self.model.composition.granularity.type
        problems: list[str] = []
        if kind != "flat" and self.model.backbone != "eager":
            problems.append(
                f"model.backbone {self.model.backbone!r} replaces the flat decoder only; "
                f"granularity.type is {kind!r}"
            )
        position = self.model.composition.position
        if kind == "flat" and (self.model.backbone == "huggingface") != (position == "rope"):
            problems.append(
                "model.composition.position must say what the backbone applies: `rope` "
                "with model.backbone huggingface (Llama's rotary positions), `nope` "
                f"otherwise; got backbone {self.model.backbone!r} with {position!r}"
            )
        supervised = self.fraud_task.supervised_head.enabled
        if kind != "flat" and supervised:
            problems.append(
                "fraud_task.supervised_head trains a head jointly on the flat stack only"
            )
        if supervised and self.data.fraud_labels == "spliced_identity_theft":
            problems.append(
                "fraud_task.supervised_head trains on the pretraining corpus, which carries no "
                "spliced labels (ADR 0015)"
            )
        if kind != "flat" and self.training.packing:
            problems.append("training.packing packs flat token streams only")
        if kind == "flat_encoder" and self.features.tokenizer != "transaction_field_tokenizer":
            problems.append(
                "granularity.type flat_encoder masks field tokens by their field's id range, "
                "which transaction_field_tokenizer declares; features.tokenizer is "
                f"{self.features.tokenizer!r}"
            )
        if kind == "flat_encoder" and self.evaluation.surprise:
            problems.append(
                "evaluation.surprise scores next-event predictions, which a bidirectional "
                "flat_encoder does not make"
            )
        if self.objective.auxiliary and self.model.composition.objective != "next_event_fields":
            problems.append(
                "objective.auxiliary adds losses to next_event_fields; "
                f"model.composition.objective is {self.model.composition.objective!r}"
            )
        if len(set(self.objective.auxiliary)) != len(self.objective.auxiliary):
            problems.append("objective.auxiliary lists an objective twice")
        if self.data.token_cache is not None:
            problems.extend(_token_cache_problems(self))
        if kind == "flat" and not self.training.packing:
            problems.extend(_flat_pretrain_row_problems(self))
        if problems:
            raise ValueError("; ".join(problems) + ".")
        return self

    def model_dump_resolved(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def _token_cache_problems(config: RecipeConfig) -> list[str]:
    """What a token cache cannot reproduce (ADR 0016).

    A key-value-time cache reads a pragmatiq book; a flat-field cache reads
    TabFormer's precomputed train split.
    """
    if config.features.tokenizer == "transaction_field_tokenizer":
        return _flat_token_cache_problems(config)
    data = config.data
    problems: list[str] = []
    if data.adapter != "pragmatiq":
        problems.append(
            f"data.token_cache reads a pragmatiq book; data.adapter is {data.adapter!r}"
        )
    if config.features.tokenizer != "key_value_time":
        problems.append(
            "data.token_cache holds key_value_time or transaction_field_tokenizer tokens; "
            f"features.tokenizer is {config.features.tokenizer!r}"
        )
    if data.split_protocol != "timestamp":
        problems.append("data.token_cache fits on a timestamp split; split_protocol is precomputed")
    if data.fraud_labels != "spliced_identity_theft":
        problems.append(
            "data.token_cache holds no labels, so its pretraining rows cannot carry the book's "
            "own fraud labels; use data.fraud_labels spliced_identity_theft"
        )
    return problems


def _flat_token_cache_problems(config: RecipeConfig) -> list[str]:
    data = config.data
    problems: list[str] = []
    if data.adapter != "tabformer":
        problems.append(
            "a transaction_field_tokenizer token cache reads TabFormer; "
            f"data.adapter is {data.adapter!r}"
        )
    if data.split_protocol != "precomputed":
        problems.append(
            "a transaction_field_tokenizer token cache holds TabFormer's precomputed train "
            f"split; split_protocol is {data.split_protocol!r}"
        )
    if data.pretrain_windows is None:
        problems.append(
            "a transaction_field_tokenizer token cache serves pretraining windows; "
            "set data.pretrain_windows"
        )
    if config.training.packing:
        problems.append("a token cache serves padded windows; training.packing must be false")
    return problems


_FLAT_FIELD_TOKENIZERS = ("transaction_field_tokenizer", "rapids_field_tokenizer")


def _flat_pretrain_row_problems(config: RecipeConfig) -> list[str]:
    """Pretraining windows that row or model context would cut silently"""
    data = config.data
    windows = data.pretrain_windows
    if windows is None:
        return []
    problems: list[str] = []
    row_tokens = windows.max_tokens or data.max_tokens
    if config.features.tokenizer in _FLAT_FIELD_TOKENIZERS:
        from tfm.features.tokenizers.financial import FIELD_ORDER

        per_event = len(config.features.fields or FIELD_ORDER) + 1
        window_tokens = per_event * windows.events + 1
        if window_tokens > row_tokens:
            problems.append(
                f"data.pretrain_windows.events {windows.events} is {window_tokens} tokens "
                f"({per_event} per event + 1), wider than its row of {row_tokens} "
                "(data.pretrain_windows.max_tokens, else data.max_tokens)"
            )
    context = max(config.model.composition.granularity.decoder.max_seq_len, data.max_tokens)
    if row_tokens > context:
        problems.append(
            f"data.pretrain_windows.max_tokens {row_tokens} is wider than the model's context "
            f"of {context} (decoder.max_seq_len, or data.max_tokens if larger)"
        )
    return problems


def load_recipe_config(path: str | Path) -> RecipeConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return RecipeConfig.model_validate(raw)
