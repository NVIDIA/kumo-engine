# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation checkpoint selection through the existing trainer progress hook."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch import nn

from tfm.config import TrainingConfig
from tfm.contracts.graph import (
    GraphHistoryBatch,
    GraphSupervisedBatch,
    GraphTrainingBatch,
    SequenceSupervisedBatch,
)
from tfm.contracts.kvt import KeyValueTimeBatch
from tfm.contracts.resources import Closable
from tfm.objectives.next_event_fields import next_event_targets
from tfm.training.progress import TrainingProgress
from tfm.training.trainer import (
    BatchSource,
    TrainableBatch,
    build_trainer,
    move_batch_for_execution,
)


@torch.no_grad()
def validation_loss(
    model: nn.Module, batches: BatchSource[TrainableBatch], device: torch.device
) -> float:
    """Weight future entries, binary samples, or next-event target tokens.

    Refuse a split with no valid supervision; it cannot select a checkpoint.
    Restore the prior training mode, including a frozen hybrid GNN's eval mode.
    """
    training = model.training
    model.eval()
    total, count = 0.0, 0
    source = iter(batches)
    try:
        for original in source:
            if (
                isinstance(original, (GraphTrainingBatch, GraphHistoryBatch, GraphSupervisedBatch))
                and original.graph.sharing != "disjoint"
            ):
                raise ValueError("Validation requires disjoint per-sample graphs")
            batch = move_batch_for_execution(original, device)
            if isinstance(batch, GraphTrainingBatch):
                weight = batch.targets.valid_count
            elif isinstance(batch, (GraphHistoryBatch, GraphSupervisedBatch)):
                weight = batch.graph.batch_size
            elif isinstance(batch, SequenceSupervisedBatch):
                weight = batch.history.batch_size
            elif isinstance(batch, KeyValueTimeBatch):
                weight = next_event_targets(batch).values.numel()
            else:
                raise ValueError("Unsupported batch type for validation selection")
            if not weight:
                continue
            output = model(batch, compute_loss=True)
            if output.loss is None or not bool(torch.isfinite(output.loss)):
                raise ValueError("Validation requires a finite loss")
            total += float(output.loss) * weight
            count += weight
    finally:
        if isinstance(source, Closable):
            source.close()
        model.train(training)
    if not count:
        raise ValueError("Validation split has no valid target entries")
    return total / count


class _SelectionProgress(TrainingProgress):
    def __init__(
        self,
        model: nn.Module,
        batches: BatchSource[TrainableBatch],
        device: torch.device,
        *,
        log_every: int | None = None,
    ) -> None:
        super().__init__(log_every=log_every)
        self.model, self.batches, self.device = model, batches, device
        self.best_loss = math.inf
        self.best_step = 0
        self.best_state: dict[str, Any] | None = None
        self.validation: list[dict[str, float]] = []

    def step(self, step: int, *, loss: float, lr: float, grad_norm: float, tokens: float) -> None:
        super().step(step, loss=loss, lr=lr, grad_norm=grad_norm, tokens=tokens)
        if step % self._batches_per_epoch and step != self._max_steps:
            return
        value = validation_loss(self.model, self.batches, self.device)
        self.validation.append({"step": float(step), "loss": value})
        if value < self.best_loss:
            self.best_loss, self.best_step = value, step
            self.best_state = copy.deepcopy(self.model.state_dict())


class _ObservedBatches:
    """Observe supervision during streaming, without consuming a fitting epoch."""

    def __init__(self, batches: BatchSource[TrainableBatch]) -> None:
        self.batches = batches
        self.supervised = False

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self) -> Iterator[TrainableBatch]:
        source = iter(self.batches)
        try:
            for batch in source:
                self.supervised |= batch.has_supervision
                yield batch
        finally:
            if isinstance(source, Closable):
                source.close()


def fit_selected(
    model: nn.Module,
    train: BatchSource[TrainableBatch],
    validation: BatchSource[TrainableBatch],
    settings: TrainingConfig,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """One fresh optimizer per stage; retain moments across its epochs.

    Select at epoch boundaries and the last step, restore the whole best model
    including buffers, then discard that stage's optimizer. Initially eager only.
    """
    if settings.backend != "eager" or settings.packing or settings.max_steps < 1:
        raise ValueError("Graph stages currently require eager, unpacked, positive-step training")
    if isinstance(train, Sequence) and not any(batch.has_supervision for batch in train):
        raise ValueError("Training split has no valid target entries")
    observed = None if isinstance(train, Sequence) else _ObservedBatches(train)
    progress = _SelectionProgress(model, validation, device, log_every=settings.log_every)
    stats = build_trainer("eager").fit(
        model,
        train if observed is None else observed,
        max_steps=settings.max_steps,
        learning_rate=settings.learning_rate,
        seed=seed,
        device=device,
        optimizer=settings.optimizer,
        shuffle=settings.shuffle if isinstance(train, Sequence) else False,
        precision=settings.precision,
        progress=progress,
    )
    if observed is not None and not observed.supervised:
        raise ValueError("Training split has no valid target entries")
    if progress.best_state is None:
        raise ValueError("No validation checkpoint selected")
    model.load_state_dict(progress.best_state)
    return {
        **stats,
        "best_validation_loss": progress.best_loss,
        "best_step": progress.best_step,
        "validation": progress.validation,
        "curve": progress.curve,
    }
