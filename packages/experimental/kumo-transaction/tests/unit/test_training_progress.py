# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`TrainingProgress`: what it prints, when, and the curve every run records."""

from __future__ import annotations

import pytest
import torch

from tfm import Decoder, Flat, TrxFM
from tfm.config import TrainingConfig
from tfm.contracts.batch import TransactionBatch
from tfm.training.progress import TrainingProgress
from tfm.training.trainer import real_tokens, train_model


def _run(progress: TrainingProgress, steps: int, *, grad_norm: float = 1.0) -> None:
    progress.start(max_steps=steps, batches_per_epoch=2)
    for step in range(1, steps + 1):
        progress.step(step, loss=1.0 / step, lr=1e-3, grad_norm=grad_norm, tokens=8.0)


def test_prints_every_log_every_steps_and_the_last_step() -> None:
    lines: list[str] = []
    progress = TrainingProgress(log_every=2, write=lines.append, rank=0)

    _run(progress, steps=5)

    assert lines[0] == "training 5 steps, 2 minibatches per epoch"
    assert [line.split(" | ")[0] for line in lines[1:]] == ["step 2/5", "step 4/5", "step 5/5"]
    assert "epoch 2.50" in lines[-1]
    assert "grad norm 1.000" in lines[-1]


def test_records_every_step_without_printing_when_log_every_is_none() -> None:
    lines: list[str] = []
    progress = TrainingProgress(log_every=None, write=lines.append, rank=0)

    _run(progress, steps=3)

    assert lines == []
    assert [row["step"] for row in progress.curve] == [1.0, 2.0, 3.0]
    assert progress.curve[1]["loss"] == pytest.approx(0.5)
    assert progress.curve[1]["epoch"] == pytest.approx(1.0)
    assert set(progress.curve[0]) == {
        "step",
        "epoch",
        "loss",
        "lr",
        "grad_norm",
        "tokens",
        "seconds",
    }


def test_only_the_first_rank_prints() -> None:
    lines: list[str] = []
    progress = TrainingProgress(log_every=1, write=lines.append, rank=1)

    _run(progress, steps=2)

    assert lines == []
    assert len(progress.curve) == 2


def test_an_unclipped_backend_omits_the_gradient_norm() -> None:
    lines: list[str] = []
    progress = TrainingProgress(log_every=1, write=lines.append, rank=0)

    _run(progress, steps=1, grad_norm=float("nan"))

    assert "grad norm" not in lines[-1]


def test_several_processes_are_named_in_the_header() -> None:
    lines: list[str] = []
    progress = TrainingProgress(log_every=1, write=lines.append, rank=0)

    progress.start(max_steps=4, batches_per_epoch=3, world_size=2)

    assert lines == ["training 4 steps on 2 processes, 3 minibatches per epoch per process"]


def test_log_every_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="log_every must be >= 1"):
        TrainingProgress(log_every=0)


def test_config_log_every_defaults_to_silent_and_rejects_zero() -> None:
    assert TrainingConfig().log_every is None
    assert TrainingConfig.model_validate({"log_every": 50}).log_every == 50
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        TrainingConfig.model_validate({"log_every": 0})


def test_real_tokens_counts_unpadded_positions() -> None:
    batch = TransactionBatch(
        input_ids=torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long),
    )
    assert real_tokens(batch) == 5.0


def test_train_model_feeds_every_step_to_progress() -> None:
    torch.manual_seed(0)
    model = TrxFM(
        granularity=Flat(Decoder(depth=1, hidden_size=16, num_heads=2, vocab_size=16, dropout=0.0))
    )
    batch = TransactionBatch(
        input_ids=torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long),
        labels=torch.tensor([[2, 3, -100, -100], [5, -100, -100, -100]], dtype=torch.long),
    )
    lines: list[str] = []
    progress = TrainingProgress(log_every=2, write=lines.append, rank=0)

    stats = train_model(
        model,
        [batch],
        max_steps=3,
        learning_rate=1e-3,
        seed=0,
        device=torch.device("cpu"),
        progress=progress,
    )

    assert stats["steps"] == 3.0
    assert [row["step"] for row in progress.curve] == [1.0, 2.0, 3.0]
    assert all(row["tokens"] == 5.0 for row in progress.curve)
    assert progress.curve[-1]["loss"] == pytest.approx(stats["final_loss"])
    assert len(lines) == 3
