# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Learning-rate schedule and optimizer plumbing."""

from __future__ import annotations

import math

import pytest
import torch

from tfm import Decoder, Flat, TrxFM
from tfm.config import OptimizerConfig
from tfm.contracts.batch import TransactionBatch
from tfm.contracts.errors import UnsupportedCombinationError
from tfm.training.trainer import lr_at_step, train_model


def _rates(optimizer: OptimizerConfig, *, max_steps: int, lr: float = 1.0) -> list[float]:
    return [
        lr_at_step(step, max_steps=max_steps, learning_rate=lr, optimizer=optimizer)
        for step in range(max_steps)
    ]


def test_default_schedule_is_constant() -> None:
    assert _rates(OptimizerConfig(), max_steps=4) == [1.0, 1.0, 1.0, 1.0]


def test_warmup_ramps_linearly_to_the_target_rate() -> None:
    rates = _rates(OptimizerConfig(warmup_steps=4), max_steps=8)

    assert rates[:4] == [0.25, 0.5, 0.75, 1.0]
    assert rates[4:] == [1.0] * 4


def test_cosine_decays_from_the_target_rate_to_the_floor() -> None:
    rates = _rates(
        OptimizerConfig(lr_schedule="cosine", warmup_steps=2, min_learning_rate=0.1),
        max_steps=10,
    )

    assert rates[1] == pytest.approx(1.0)
    assert rates[2] == pytest.approx(1.0)
    assert rates == sorted(rates[:2]) + sorted(rates[2:], reverse=True)
    # Last step is 9, so cosine progress is (9 - 2) / (10 - 2).
    last = 0.1 + 0.5 * 0.9 * (1 + math.cos(math.pi * 7 / 8))
    assert rates[-1] == pytest.approx(last)
    assert all(rate >= 0.1 for rate in rates)
    # The floor is only reached at the step after the horizon.
    assert lr_at_step(
        10,
        max_steps=10,
        learning_rate=1.0,
        optimizer=OptimizerConfig(lr_schedule="cosine", warmup_steps=2, min_learning_rate=0.1),
    ) == pytest.approx(0.1)


def test_schedule_horizon_can_outlast_the_run() -> None:
    """The Blueprint saved step 3000 of a 10000-step cosine, far above the floor."""
    optimizer = OptimizerConfig(
        lr_schedule="cosine", warmup_steps=500, min_learning_rate=6.5e-6, schedule_steps=10_000
    )

    rate = lr_at_step(2999, max_steps=3000, learning_rate=2e-4, optimizer=optimizer)

    progress = (2999 - 500) / (10_000 - 500)
    assert rate == pytest.approx(
        6.5e-6 + 0.5 * (2e-4 - 6.5e-6) * (1 + math.cos(math.pi * progress))
    )
    assert rate > 0.8 * 2e-4
    # Without the horizon, the same run anneals to the floor.
    annealed = OptimizerConfig(lr_schedule="cosine", warmup_steps=500, min_learning_rate=6.5e-6)
    assert lr_at_step(3000, max_steps=3000, learning_rate=2e-4, optimizer=annealed) == (
        pytest.approx(6.5e-6)
    )


def test_schedule_horizon_shorter_than_the_run_is_rejected() -> None:
    with pytest.raises(UnsupportedCombinationError, match="schedule_steps"):
        lr_at_step(
            0,
            max_steps=10,
            learning_rate=1.0,
            optimizer=OptimizerConfig(lr_schedule="cosine", schedule_steps=5),
        )


def test_unknown_schedule_is_rejected() -> None:
    with pytest.raises(UnsupportedCombinationError, match="lr_schedule"):
        lr_at_step(
            0,
            max_steps=4,
            learning_rate=1.0,
            optimizer=OptimizerConfig(lr_schedule="one_cycle"),
        )


def test_train_model_reports_first_and_final_loss() -> None:
    model = TrxFM(granularity=Flat(Decoder(depth=1, hidden_size=16, num_heads=4, vocab_size=20)))
    batch = TransactionBatch(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        labels=torch.tensor([[-100, 2, 3]]),
    )

    curve: list[float] = []
    stats = train_model(
        model,
        [batch],
        max_steps=3,
        learning_rate=1e-2,
        seed=0,
        device=torch.device("cpu"),
        optimizer=OptimizerConfig(
            betas=(0.9, 0.95), weight_decay=0.1, lr_schedule="cosine", warmup_steps=1
        ),
        loss_log=curve,
    )

    assert stats["steps"] == 3.0
    assert isinstance(stats["first_loss"], float)
    assert stats["final_loss"] < stats["first_loss"]
    assert curve[0] == stats["first_loss"] and curve[-1] == stats["final_loss"] and len(curve) == 3
