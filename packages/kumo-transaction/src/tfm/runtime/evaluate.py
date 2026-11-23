# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation stage shared by every stack (slot 4: readout and adaptation).

Every probe here is entity-disjoint, and every label comes from its own side's
period (ADRs 0008 and 0013):

- **fraud**: entity-level. Train rows are sequences at the train cutoff with
  labels from the train window; evaluation rows are sequences at the test
  cutoff with labels from `(validation_end, test_cutoff]`. Entities are split
  with `split_entities`, stratified by the evaluation label.
- **forecast tasks** (`default_12m`, ...): one history per label row, cut at
  the row's own date. Skipped when a row predates the pretraining cutoff,
  because the encoder was then pretrained on events inside its label horizon.
- **history window** (optional): one row per labelled fraud event, beside a
  single-event control arm. Entities are split by whether any of their
  evaluation-window events is labelled fraud, not by their truncated sequence.
- **surprise** (optional): the model's next-event surprise on each row's latest
  events, scored from each event's prefix, joined to the embedding as one more
  XGBoost view.
- **fine-tuning** (optional): the same rows and labels as the frozen probes,
  `history_window` included; the single-event control is not fine-tuned.

With the XGBoost arms enabled, every task is scored by `xgboost_raw` and
`xgboost_raw_embedding` on the same raw row, and `tabular_xgboost` is the
hand-built reference. The logistic `raw` / `embedding` / `combined` stack
runs only without them: it is the CPU smoke path.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from tfm.adaptation.batching import concat_rows, select_rows
from tfm.adaptation.finetune import FineTune, finetune_binary
from tfm.adaptation.probe import probe_arms
from tfm.config import RecipeConfig
from tfm.contracts.batch import TransactionBatch
from tfm.contracts.errors import InsufficientLabelSupportError
from tfm.contracts.schema import TransactionSequence, ensure_utc
from tfm.data.collate import collate_batch, raw_feature_vector
from tfm.data.labels.horizon import available_label_value
from tfm.data.sequence_builders.entity_sequences import build_sequences_at_label_cutoffs
from tfm.data.sequence_builders.event_rows import build_event_history_rows
from tfm.data.splits.entity import EntitySplit, split_entities
from tfm.evaluation.fraud.baselines import evaluate_binary_probes
from tfm.evaluation.surprise import extract_surprise
from tfm.evaluation.tabular import evaluate_tabular_baselines
from tfm.inference.embedding import EmbeddingTable, extract_embeddings, trim_padding
from tfm.runtime.common import parse_duration, stage
from tfm.runtime.data import PreparedData
from tfm.training.lora import LoRA


@dataclass
class ProbeTask:
    """One binary task: batches and labels on each side of an entity-disjoint split.

    The sequences behind each batch, row for row, feed the tabular baselines;
    the frozen embeddings and surprise features, once a probe extracts them,
    feed its embedding arms.
    """

    train_batch: Any
    eval_batch: Any
    y_train: np.ndarray
    y_eval: np.ndarray
    train_sequences: tuple[TransactionSequence, ...] = ()
    eval_sequences: tuple[TransactionSequence, ...] = ()
    train_vectors: np.ndarray | None = None
    eval_vectors: np.ndarray | None = None
    train_surprise: np.ndarray | None = None
    eval_surprise: np.ndarray | None = None

    def summary(self) -> dict[str, int]:
        return {
            "n_train": int(len(self.y_train)),
            "n_eval": int(len(self.y_eval)),
            "n_train_positive": int(self.y_train.sum()),
            "n_eval_positive": int(self.y_eval.sum()),
        }


def _labels(batch: Any) -> np.ndarray:
    return batch.fraud_labels.cpu().numpy().astype(np.float32)


def fraud_entity_split(data: PreparedData, seed: int) -> EntitySplit:
    """Split test-side entities by their evaluation-window label (ADR 0013).

    A spliced task fixed its split before any label existed (ADR 0015); that
    split is returned as is.
    """
    if data.fraud is not None:
        return data.fraud.split
    test = data.batches[2]
    labels = _labels(test)
    return split_entities(
        [(entity, int(label)) for entity, label in zip(test.entity_ids, labels, strict=True)],
        seed=seed,
    )


def event_entity_split(data: PreparedData, seed: int) -> EntitySplit:
    """Split entities by whether any evaluation-window event is labelled fraud (ADR 0013).

    The event-level counterpart of `fraud_entity_split`: its rows come from the
    whole window, so an entity whose fraud falls before the last
    `sequence_length` events still counts as positive. A spliced task's split
    is returned as is.
    """
    if data.fraud is not None:
        return data.fraud.split
    window = data.cutoffs.windows["test"]
    classes: dict[str, int] = {}
    for event in data.events:
        if not window.contains(event.event_timestamp):
            continue
        value = available_label_value(event, "fraud", data.cutoffs.test)
        if value is not None:
            classes[event.entity_id] = max(classes.get(event.entity_id, 0), int(value >= 0.5))
    return split_entities(sorted(classes.items()), seed=seed)


def fraud_task(data: PreparedData, split: EntitySplit) -> ProbeTask:
    """Train-side rows of every non-evaluation entity; test-side rows of evaluation entities."""
    if data.fraud is not None:
        fraud = data.fraud
        return ProbeTask(
            fraud.train_batch,
            fraud.eval_batch,
            _labels(fraud.train_batch),
            _labels(fraud.eval_batch),
            fraud.train_sequences,
            fraud.eval_sequences,
        )
    train, test = data.batches[0], data.batches[2]
    held_out = set(split.evaluation)
    train_rows = [i for i, e in enumerate(train.entity_ids) if e not in held_out]
    eval_rows = [i for i, e in enumerate(test.entity_ids) if e in held_out]
    train_batch = select_rows(train, train_rows)
    eval_batch = select_rows(test, eval_rows)
    return ProbeTask(
        train_batch,
        eval_batch,
        _labels(train_batch),
        _labels(eval_batch),
        tuple(data.train_sequences[i] for i in train_rows),
        tuple(data.test_sequences[i] for i in eval_rows),
    )


def forecast_task(
    data: PreparedData,
    config: RecipeConfig,
    task: str,
    rows: Sequence[Any],
    *,
    seed: int,
) -> ProbeTask | None:
    """An entity-disjoint probe over one history per label row, or `None` to skip."""
    early = [row for row in rows if ensure_utc(row.timestamp) < ensure_utc(data.cutoffs.train)]
    if early:
        warnings.warn(
            f"Skipping the {task!r} probe: {len(early)} of {len(rows)} rows are dated "
            f"before the pretraining cutoff {data.cutoffs.train.isoformat()} (earliest "
            f"{min(r.timestamp for r in early).isoformat()}), so the encoder saw events "
            "inside their label horizon (ADR 0013).",
            stacklevel=2,
        )
        return None
    pairs = build_sequences_at_label_cutoffs(
        data.events,
        rows,
        max_events=config.data.sequence_length,
        observation_window=parse_duration(config.data.observation_window),
    )
    try:
        train_pairs, eval_pairs = split_label_pairs(pairs, seed=seed)
    except InsufficientLabelSupportError as exc:
        warnings.warn(f"Skipping the {task!r} probe: {exc}", stacklevel=2)
        return None

    def side(chosen: list[tuple[Any, Any]]) -> tuple[Any, np.ndarray]:
        batch = collate_batch(
            [seq for seq, _row in chosen], data.tokenizer, max_tokens=config.data.max_tokens
        )
        return batch, np.array([float(row.label) for _seq, row in chosen], dtype=np.float32)

    train_batch, y_train = side(train_pairs)
    eval_batch, y_eval = side(eval_pairs)
    return ProbeTask(
        train_batch,
        eval_batch,
        y_train,
        y_eval,
        tuple(seq for seq, _row in train_pairs),
        tuple(seq for seq, _row in eval_pairs),
    )


def split_label_pairs(
    pairs: Sequence[tuple[Any, Any]], *, seed: int
) -> tuple[list[tuple[Any, Any]], list[tuple[Any, Any]]]:
    """Seeded 70/30 entity-disjoint split of `(sequence, row)` pairs, stratified by label.

    Raises `InsufficientLabelSupportError` when a class is too small to split.
    An entity never lands on both sides: that would score the probe on its own
    training rows (ADR 0008).
    """
    by_entity: dict[str, list[tuple[Any, Any]]] = {}
    for pair in pairs:
        by_entity.setdefault(pair[1].entity_id, []).append(pair)
    classes = [
        (entity, int(any(int(row.label) == 1 for _seq, row in items)))
        for entity, items in by_entity.items()
    ]
    split = split_entities(classes, seed=seed)
    train = [pair for entity in split.train for pair in by_entity[entity]]
    evaluation = [pair for entity in split.evaluation for pair in by_entity[entity]]
    return train, evaluation


COLLATE_ROWS = 16_384
"""History-window rows tokenized at once. A side can hold every labelled event of
its users, too many to tokenize in one call beside a full book in memory."""


def _collate_rows(
    sequences: Sequence[TransactionSequence], tokenizer: Any, *, max_tokens: int
) -> Any:
    """`collate_batch` in chunks of `COLLATE_ROWS`, for probe rows only.

    Flat rows drop their next-token `labels`, which no probe reads, and are cut
    to the longest row, so event-only rows take a few columns, not a history's.
    """
    parts = []
    for start in range(0, len(sequences), COLLATE_ROWS):
        part = collate_batch(
            sequences[start : start + COLLATE_ROWS], tokenizer, max_tokens=max_tokens
        )
        if isinstance(part, TransactionBatch):
            part = trim_padding(dataclasses.replace(part, labels=None))
            part = dataclasses.replace(
                part,
                input_ids=part.input_ids.clone(),
                attention_mask=part.attention_mask.clone(),
            )
        parts.append(part)
    if isinstance(parts[0], TransactionBatch):
        width = max(part.input_ids.shape[1] for part in parts)
        pad = int(getattr(tokenizer, "pad_id", 0))
        parts = [
            dataclasses.replace(
                part,
                input_ids=torch.nn.functional.pad(
                    part.input_ids, (0, width - part.input_ids.shape[1]), value=pad
                ),
                attention_mask=torch.nn.functional.pad(
                    part.attention_mask, (0, width - part.attention_mask.shape[1])
                ),
            )
            for part in parts
        ]
    return concat_rows(parts)


def _history_side(
    data: PreparedData,
    config: RecipeConfig,
    entities: Sequence[str],
    side: str,
    label_cutoff: Any,
    seed: int,
) -> tuple[Any, Any, np.ndarray, tuple[TransactionSequence, ...]] | None:
    ratio = config.evaluation.history_window.train_negatives_per_positive
    rows = build_event_history_rows(
        data.events if data.fraud is None else data.fraud.events,
        entities=entities,
        window=data.cutoffs.windows[side],
        label_cutoff=label_cutoff,
        max_events=config.data.sequence_length,
        observation_window=parse_duration(config.data.observation_window),
        max_rows=config.evaluation.history_window.max_rows,
        seed=seed,
        negatives_per_positive=ratio if side == "train" else None,
    )
    if not rows:
        return None
    max_tokens = config.data.max_tokens
    history = _collate_rows([r.history for r in rows], data.tokenizer, max_tokens=max_tokens)
    single = _collate_rows([r.event_only for r in rows], data.tokenizer, max_tokens=max_tokens)
    # The arms differ only in what the encoder sees; both get the history's aggregates.
    single.numeric_features = torch.tensor(
        [raw_feature_vector(r.history) for r in rows], dtype=torch.float32
    ).reshape(len(rows), -1)
    labels = np.array([r.label for r in rows], dtype=np.float32)
    return history, single, labels, tuple(r.history for r in rows)


def history_window_tasks(
    data: PreparedData, config: RecipeConfig, split: EntitySplit
) -> dict[str, ProbeTask]:
    """`history_window` and its `single_event` control, on `event_entity_split`'s split."""
    held_out = set(split.evaluation)
    if data.fraud is not None:
        train_entities = sorted(split.train)
    else:
        train_entities = sorted({e for e in data.batches[0].entity_ids if e not in held_out})
    seed = config.seed
    train_side = _history_side(data, config, train_entities, "train", data.cutoffs.train, seed)
    eval_side = _history_side(data, config, sorted(held_out), "test", data.cutoffs.test, seed)
    if train_side is None or eval_side is None:
        return {}
    tr_hist, tr_single, y_tr, tr_seq = train_side
    ev_hist, ev_single, y_ev, ev_seq = eval_side
    return {
        "history_window": ProbeTask(tr_hist, ev_hist, y_tr, y_ev, tr_seq, ev_seq),
        "single_event": ProbeTask(tr_single, ev_single, y_tr, y_ev),
    }


def _probe(
    model: nn.Module,
    task: ProbeTask,
    data: PreparedData,
    config: RecipeConfig,
    device: torch.device,
    *,
    prefix: str,
) -> dict[str, dict[str, float]]:
    extract = {
        "device": device,
        "minibatch_size": config.evaluation.inference_batch_size,
        "precision": config.evaluation.inference_precision,
    }
    stage(
        f"evaluate {prefix or 'fraud'}: embedding {len(task.y_train):,} train and "
        f"{len(task.y_eval):,} evaluation rows ({int(task.y_eval.sum())} positive)"
    )
    train = extract_embeddings(model, task.train_batch, **extract)
    evaluation = extract_embeddings(model, task.eval_batch, **extract)
    task.train_vectors, task.eval_vectors = train.vectors, evaluation.vectors
    if config.evaluation.surprise and task.train_sequences:
        task.train_surprise, task.eval_surprise = (
            extract_surprise(
                model,
                batch,
                data.tokenizer,
                device=device,
                minibatch_size=config.evaluation.inference_batch_size,
            ).vectors
            for batch in (task.train_batch, task.eval_batch)
        )
    baselines = config.evaluation.tabular_baselines
    if baselines.enabled and baselines.xgboost_enabled:
        return {}
    arms = probe_arms(
        use_raw_features=config.fraud_task.use_raw_features,
        use_tfm_embeddings=config.fraud_task.use_tfm_embeddings,
    )
    if prefix == "single_event":
        # The control's raw row is the history row's, so its raw arm would repeat
        # `history_window_raw`; only the encoder's input differs.
        arms = tuple(arm for arm in arms if arm != "raw")
    return evaluate_binary_probes(
        train,
        evaluation,
        seed=config.seed,
        k=config.evaluation.precision_at_k,
        y_train=task.y_train,
        y_eval=task.y_eval,
        prefix=prefix,
        head=config.fraud_task.head,
        arms=arms,
    )


def _embedding_views(
    name: str, tasks: Mapping[str, ProbeTask]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Extra readouts of one task's rows, each scored as `xgboost_raw_embedding:<view>`.

    - `surprise`: the embedding joined to the model's next-event surprise.
    - `single_event` (history-window task only): the control's embeddings. The
      rows are the same, so it differs from `xgboost_raw_embedding` only in
      whether the encoder saw the history.
    """
    task = tasks[name]
    views: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if (
        task.train_surprise is not None
        and task.eval_surprise is not None
        and task.train_vectors is not None
        and task.eval_vectors is not None
    ):
        views["surprise"] = (
            np.hstack([task.train_vectors, task.train_surprise]),
            np.hstack([task.eval_vectors, task.eval_surprise]),
        )
    control = tasks.get("single_event")
    if name == "history_window" and control is not None:
        if control.train_vectors is not None and control.eval_vectors is not None:
            views["single_event"] = (control.train_vectors, control.eval_vectors)
    return views


def tabular_baselines(
    tasks: Mapping[str, ProbeTask], data: PreparedData, config: RecipeConfig
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """The tabular arms on every task that carries its sequences; no encoder involved.

    The fraud task's arms are unprefixed, like its other arms. A task whose
    XGBoost inner split holds one class keeps its logistic arm and records why.
    A task whose embeddings were extracted also gets `tabular_xgboost_embedding`.
    """
    cfg = config.evaluation.tabular_baselines
    profiles = {profile.entity_id: profile for profile in data.profiles}
    metrics: dict[str, dict[str, float]] = {}
    manifest: dict[str, Any] = {}
    for name, task in tasks.items():
        if not task.train_sequences or not task.eval_sequences:
            continue
        common: dict[str, Any] = {
            "profiles": profiles,
            "seed": config.seed,
            "top_k": cfg.top_k,
            "coded_keys": tuple(config.features.categorical_keys),
            "exclude_keys": tuple(cfg.exclude_keys),
            "n_jobs": cfg.n_jobs,
            "device": cfg.device,
            "k": config.evaluation.precision_at_k,
            "prefix": "" if name == "fraud" else name,
            "train_embeddings": task.train_vectors,
            "eval_embeddings": task.eval_vectors,
            "embedding_views": _embedding_views(name, tasks),
        }
        args = (task.train_sequences, task.y_train, task.eval_sequences, task.y_eval)
        stage(
            f"evaluate {name}: tabular baselines and XGBoost arms "
            f"({cfg.xgboost_trials} Optuna trials each)"
        )
        try:
            task_metrics, task_manifest = evaluate_tabular_baselines(
                *args,
                xgboost_trials=cfg.xgboost_trials,
                xgboost_params=cfg.xgboost_params,
                **common,
            )
        except InsufficientLabelSupportError as exc:
            task_metrics, task_manifest = evaluate_tabular_baselines(
                *args, xgboost_trials=0, **common
            )
            task_manifest["xgboost_skipped"] = str(exc)
        metrics.update(task_metrics)
        manifest[name] = task_manifest
    return metrics, manifest


def finetune_spec(config: RecipeConfig) -> FineTune:
    cfg = config.fraud_task.finetune
    return FineTune(
        method=cfg.method,
        head=cfg.head,
        lora=LoRA(rank=cfg.rank, alpha=cfg.alpha, dropout=cfg.dropout, targets=tuple(cfg.targets)),
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        validation_fraction=cfg.validation_fraction,
        eval_every=cfg.eval_every,
        patience=cfg.patience,
    )


def evaluate_run(
    model: nn.Module,
    data: PreparedData,
    config: RecipeConfig,
    device: torch.device,
) -> tuple[dict[str, dict[str, float]], dict[str, Any], dict[str, Any]]:
    """Every configured probe and fine-tune. Returns `(metrics, probe_manifest, finetune_manifest)`."""
    metrics: dict[str, dict[str, float]] = {}
    manifest: dict[str, Any] = {"cutoffs": data.cutoffs.as_dict(), "tasks": {}, "skipped": {}}
    tasks: dict[str, ProbeTask] = {}
    split: EntitySplit | None = None
    try:
        split = fraud_entity_split(data, config.seed)
    except InsufficientLabelSupportError as exc:
        warnings.warn(f"Skipping the fraud probe: {exc}", stacklevel=2)
        manifest["skipped"]["fraud"] = str(exc)
    if split is not None:
        tasks["fraud"] = fraud_task(data, split)
        manifest["fraud_split"] = split.as_dict()
        manifest["fraud_labels"] = config.data.fraud_labels
        metrics.update(_probe(model, tasks["fraud"], data, config, device, prefix=""))
    for task, rows in data.label_rows.items():
        probe = forecast_task(data, config, task, rows, seed=config.seed)
        if probe is None:
            manifest["skipped"][task] = "insufficient support or rows predate train cutoff"
            continue
        tasks[task] = probe
        metrics.update(_probe(model, probe, data, config, device, prefix=task))
    baseline_tasks = dict(tasks)
    if config.evaluation.history_window.enabled:
        history_tasks: dict[str, ProbeTask] = {}
        try:
            history_split = event_entity_split(data, config.seed)
        except InsufficientLabelSupportError as exc:
            manifest["skipped"]["history_window"] = str(exc)
        else:
            manifest["history_window_split"] = history_split.as_dict()
            history_tasks = history_window_tasks(data, config, history_split)
            if not history_tasks:
                manifest["skipped"]["history_window"] = "no labelled events on one side"
        for name, probe in history_tasks.items():
            if len(set(probe.y_train.tolist())) < 2 or len(set(probe.y_eval.tolist())) < 2:
                manifest["skipped"][name] = "one class on a side"
                continue
            manifest["tasks"][name] = probe.summary()
            metrics.update(_probe(model, probe, data, config, device, prefix=name))
            baseline_tasks[name] = probe
    for name, probe in tasks.items():
        manifest["tasks"][name] = probe.summary()
    if config.evaluation.tabular_baselines.enabled:
        baseline_metrics, manifest["tabular_baselines"] = tabular_baselines(
            baseline_tasks, data, config
        )
        metrics.update(baseline_metrics)

    finetune_manifest: dict[str, Any] = {"enabled": config.fraud_task.finetune.enabled}
    if config.fraud_task.finetune.enabled:
        spec = finetune_spec(config)
        finetune_manifest.update({"arm": spec.arm, "tasks": {}})
        for name, probe in baseline_tasks.items():
            if name == "single_event" or not len(probe.y_train) or not len(probe.y_eval):
                continue
            task_metrics, task_manifest = finetune_binary(
                model,
                probe.train_batch,
                probe.eval_batch,
                probe.y_train,
                probe.y_eval,
                spec,
                device=device,
                seed=config.seed,
                k=config.evaluation.precision_at_k,
                precision=config.training.precision,
            )
            metrics[f"{spec.arm}_{name}"] = task_metrics
            finetune_manifest["tasks"][name] = task_manifest
    return metrics, manifest, finetune_manifest


__all__ = [
    "EmbeddingTable",
    "ProbeTask",
    "evaluate_run",
    "finetune_spec",
    "forecast_task",
    "fraud_entity_split",
    "fraud_task",
    "event_entity_split",
    "history_window_tasks",
    "split_label_pairs",
    "tabular_baselines",
]
