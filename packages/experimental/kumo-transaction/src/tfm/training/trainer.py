# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight eager training loop (CPU or CUDA)."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Protocol

import torch
from torch import nn

from tfm.config import OptimizerConfig
from tfm.contracts.capabilities import BackendCapabilities
from tfm.contracts.errors import UnsupportedCombinationError
from tfm.training.optim import build_optimizers, set_schedule_factor
from tfm.training.progress import TrainingProgress
from tfm.training.reproducibility import seed_everything


class TrainableBatch(Protocol):
    """A collated batch that can be moved to a device.

    Both `TransactionBatch` (flat decoder) and `KeyValueTimeBatch` (profile /
    history) satisfy this, so one loop serves both stacks.
    """

    def to(self, device: torch.device) -> TrainableBatch: ...


def epoch_order(n: int, *, seed: int, epoch: int, shuffle: bool) -> list[int]:
    """Batch indices for one epoch: in order, or a permutation seeded by `seed` and `epoch`.

    Depends on nothing rank-local, so every data-parallel rank that holds the
    same number of batches walks them in the same pattern.
    """
    if not shuffle:
        return list(range(n))
    generator = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
    return torch.randperm(n, generator=generator).tolist()


def iterate_batches[BatchT: TrainableBatch](
    batches: Sequence[BatchT], *, seed: int = 0, shuffle: bool = False
) -> Iterator[BatchT]:
    epoch = 0
    while True:
        for index in epoch_order(len(batches), seed=seed, epoch=epoch, shuffle=shuffle):
            yield batches[index]
        epoch += 1


def lr_at_step(
    step: int,
    *,
    max_steps: int,
    learning_rate: float,
    optimizer: OptimizerConfig,
) -> float:
    """Learning rate for a zero-indexed step under linear warmup + schedule.

    The cosine decays over `optimizer.schedule_steps` when set, otherwise over
    `max_steps`, so a run can stop before its schedule finishes.
    """
    horizon = max_steps if optimizer.schedule_steps is None else optimizer.schedule_steps
    if horizon < max_steps:
        raise UnsupportedCombinationError(
            f"optimizer.schedule_steps ({horizon}) is shorter than max_steps ({max_steps}); "
            "the schedule would sit at its floor for the rest of the run."
        )
    warmup = max(0, optimizer.warmup_steps)
    if warmup and step < warmup:
        return learning_rate * (step + 1) / warmup
    if optimizer.lr_schedule.strip().lower() == "constant":
        return learning_rate
    if optimizer.lr_schedule.strip().lower() != "cosine":
        raise UnsupportedCombinationError(
            f"Unknown optimizer.lr_schedule {optimizer.lr_schedule!r}; use 'constant' or 'cosine'."
        )
    span = max(1, horizon - warmup)
    progress = min(1.0, (step - warmup) / span)
    floor = optimizer.min_learning_rate
    return floor + 0.5 * (learning_rate - floor) * (1.0 + math.cos(math.pi * progress))


def real_tokens(batch: TrainableBatch) -> float:
    """Tokens a batch trains on: unpadded positions of a flat or key-value-time batch."""
    mask = getattr(batch, "attention_mask", None)
    if mask is None:
        mask = getattr(batch, "token_mask", None)
    return float(mask.sum()) if isinstance(mask, torch.Tensor) else 0.0


AUTOCAST_DTYPES: dict[str, torch.dtype | None] = {"fp32": None, "bf16": torch.bfloat16}
"""`training.precision` to autocast dtype. Master weights stay fp32 either way."""


def train_model[BatchT: TrainableBatch](
    model: nn.Module,
    batches: Sequence[BatchT],
    *,
    max_steps: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    optimizer: OptimizerConfig | None = None,
    shuffle: bool = False,
    precision: str = "fp32",
    loss_log: list[float] | None = None,
    progress: TrainingProgress | None = None,
    world_size: int = 1,
) -> dict[str, float | str]:
    """Train `model` for `max_steps`; return the run's loss and gradient summary.

    `loss_log`, when given, receives every step's loss, for plotting a curve.
    `progress` receives every step's loss, learning rate, gradient norm, and
    token count, and prints a line every `progress.log_every` steps.
    """
    settings = optimizer if optimizer is not None else OptimizerConfig()
    if precision not in AUTOCAST_DTYPES:
        raise UnsupportedCombinationError(
            f"training.precision: {precision!r} is not one of {sorted(AUTOCAST_DTYPES)}."
        )
    compute_dtype = AUTOCAST_DTYPES[precision]
    seed_everything(seed)
    model.to(device)
    model.train()
    optimizers = build_optimizers(model, learning_rate=learning_rate, settings=settings)
    clip = settings.max_grad_norm if settings.max_grad_norm is not None else math.inf
    losses: list[float] = []
    grad_norms: list[float] = []
    stepper = iterate_batches(batches, seed=seed, shuffle=shuffle)
    if progress is not None:
        progress.start(max_steps=max_steps, batches_per_epoch=len(batches), world_size=world_size)
    for step in range(max_steps):
        rate = lr_at_step(
            step, max_steps=max_steps, learning_rate=learning_rate, optimizer=settings
        )
        set_schedule_factor(optimizers, rate / learning_rate if learning_rate else 0.0)
        batch = next(stepper).to(device)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=compute_dtype, enabled=compute_dtype is not None
        ):
            output = model(batch, compute_loss=True)
        assert output.loss is not None
        output.loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        for opt in optimizers:
            opt.step()
        losses.append(float(output.loss.detach().float().cpu()))
        grad_norms.append(float(norm.detach().float().cpu()))
        if progress is not None:
            progress.step(
                step + 1,
                loss=losses[-1],
                lr=rate,
                grad_norm=grad_norms[-1],
                tokens=real_tokens(batch),
            )
    if loss_log is not None:
        loss_log.extend(losses)
    return {
        **training_summary(losses, grad_norms),
        "steps": float(max_steps),
        "precision": precision,
    }


def training_summary(losses: Sequence[float], grad_norms: Sequence[float]) -> dict[str, float]:
    """Scalars a run records about its loss and gradient norms (before clipping)."""
    if not losses:
        nan = float("nan")
        return {"final_loss": nan, "first_loss": nan, "loss_tail_mean": nan}
    tail = losses[-min(100, len(losses)) :]
    finite = [g for g in grad_norms if math.isfinite(g)]
    out = {
        "final_loss": losses[-1],
        "first_loss": losses[0],
        "loss_tail_mean": sum(tail) / len(tail),
        "nonfinite_grad_steps": float(len(grad_norms) - len(finite)),
    }
    if finite:
        ordered = sorted(finite)
        out["grad_norm_median"] = ordered[len(ordered) // 2]
        out["grad_norm_max"] = ordered[-1]
    return out


class Trainer(Protocol):
    """A training backend (ADR 0011). Construct through `build_trainer`."""

    capabilities: BackendCapabilities

    def fit(
        self,
        model: nn.Module,
        batches: Sequence[TrainableBatch],
        *,
        max_steps: int,
        learning_rate: float,
        seed: int,
        device: torch.device,
        optimizer: OptimizerConfig | None = None,
        shuffle: bool = False,
        precision: str = "fp32",
        progress: TrainingProgress | None = None,
    ) -> dict[str, float | str]: ...


class EagerTrainer:
    """The single-process reference loop. Every other backend must match it."""

    capabilities = BackendCapabilities(
        name="eager",
        support_tier="reference",
        backbones=frozenset({"eager", "huggingface", "gru"}),
        batch_layouts=frozenset({"flat_fields", "flat_key_value", "key_value_time"}),
        requires_cuda=False,
        multi_process=False,
        segment_masks=True,
        consolidated_hf_export=True,
        precisions=frozenset({"fp32", "bf16"}),
    )

    def __init__(self, *, checkpoint_dir: Path | None = None) -> None:
        self.checkpoint_dir = checkpoint_dir

    def fit(
        self,
        model: nn.Module,
        batches: Sequence[TrainableBatch],
        *,
        max_steps: int,
        learning_rate: float,
        seed: int,
        device: torch.device,
        optimizer: OptimizerConfig | None = None,
        shuffle: bool = False,
        precision: str = "fp32",
        progress: TrainingProgress | None = None,
    ) -> dict[str, float | str]:
        stats = train_model(
            model,
            batches,
            max_steps=max_steps,
            learning_rate=learning_rate,
            seed=seed,
            device=device,
            optimizer=optimizer,
            shuffle=shuffle,
            precision=precision,
            progress=progress,
        )
        stats["backend"] = "eager"
        stats["hf_checkpoint"] = export_huggingface(model, self.checkpoint_dir)
        return stats


def export_huggingface(model: nn.Module, checkpoint_dir: Path | None) -> str:
    """Write the consolidated HF checkpoint an HF backbone owes (ADR 0011).

    Returns the directory, or "" when there is no directory or no HF backbone.
    """
    inner = getattr(getattr(model, "backbone", None), "model", None)
    if checkpoint_dir is None or not hasattr(inner, "save_pretrained"):
        return ""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    inner.save_pretrained(str(checkpoint_dir))  # type: ignore[union-attr]
    return str(checkpoint_dir)


def build_trainer(name: str, *, checkpoint_dir: Path | None = None) -> Trainer:
    """Construct a registered backend by `training.backend` name."""
    from tfm.training.backends import build_trainer as build_registered

    return build_registered(name, checkpoint_dir=checkpoint_dir)
