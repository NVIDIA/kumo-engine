# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`optimizer.name`, the Muon/AdamW parameter split, and gradient clipping."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tfm import Decoder, Flat, TrxFM
from tfm.config import MuonConfig, OptimizerConfig
from tfm.contracts.batch import TransactionBatch
from tfm.contracts.errors import UnsupportedCombinationError
from tfm.training.optim import build_optimizers, muon_parameter_split
from tfm.training.trainer import train_model


def _model() -> TrxFM:
    torch.manual_seed(0)
    return TrxFM(granularity=Flat(Decoder(depth=2, hidden_size=16, num_heads=4, vocab_size=20)))


def _batch() -> TransactionBatch:
    return TransactionBatch(
        input_ids=torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]),
        attention_mask=torch.ones(2, 4, dtype=torch.long),
        labels=torch.tensor([[-100, 2, 3, 4], [-100, 3, 2, 1]]),
    )


def test_muon_takes_hidden_matrices_and_adamw_everything_else() -> None:
    model = nn.Sequential(nn.Embedding(10, 8), nn.Linear(8, 8), nn.LayerNorm(8), nn.Linear(8, 1))
    split = muon_parameter_split(model)
    assert [n for n, _ in split.muon] == ["1.weight"]
    assert {n for n, _ in split.adamw} == {
        "0.weight",
        "1.bias",
        "2.weight",
        "2.bias",
        "3.weight",
        "3.bias",
    }
    every = {id(p) for p in model.parameters()}
    assert {id(p) for _, p in split.muon + split.adamw} == every
    assert not {id(p) for _, p in split.muon} & {id(p) for _, p in split.adamw}


def test_frozen_parameters_join_neither_group() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    model[0].weight.requires_grad_(False)
    split = muon_parameter_split(model)
    assert [n for n, _ in split.muon] == ["1.weight"]


def test_the_default_optimizer_is_one_adamw() -> None:
    opts = build_optimizers(_model(), learning_rate=1e-3, settings=OptimizerConfig())
    assert [type(o).__name__ for o in opts] == ["AdamW"]


def test_muon_groups_carry_their_own_base_rate() -> None:
    opts = build_optimizers(
        _model(),
        learning_rate=1e-3,
        settings=OptimizerConfig(name="muon", muon=MuonConfig(learning_rate=0.02)),
    )
    assert [type(o).__name__ for o in opts] == ["Muon", "AdamW"]
    assert opts[0].param_groups[0]["base_lr"] == 0.02
    assert opts[1].param_groups[0]["base_lr"] == 1e-3


def test_muon_trains_a_decoder_and_follows_the_schedule() -> None:
    settings = OptimizerConfig(
        name="muon", lr_schedule="cosine", warmup_steps=1, muon=MuonConfig(learning_rate=0.02)
    )
    stats = train_model(
        _model(),
        [_batch()],
        max_steps=6,
        learning_rate=1e-2,
        seed=0,
        device=torch.device("cpu"),
        optimizer=settings,
    )
    assert stats["final_loss"] < stats["first_loss"]
    assert stats["grad_norm_max"] >= stats["grad_norm_median"] > 0


def test_clipping_changes_the_update_and_is_off_by_default() -> None:
    def run(clip: float | None) -> float:
        stats = train_model(
            _model(),
            [_batch()],
            max_steps=3,
            learning_rate=5e-2,
            seed=0,
            device=torch.device("cpu"),
            optimizer=OptimizerConfig(max_grad_norm=clip),
        )
        return float(stats["final_loss"])

    unclipped, huge, tight = run(None), run(1e9), run(1e-3)
    assert unclipped == huge
    assert tight != unclipped


def test_fsdp2_rejects_muon_before_training() -> None:
    from tfm.extras.fsdp2 import Fsdp2NextEventTrainer

    with pytest.raises(UnsupportedCombinationError, match="muon"):
        Fsdp2NextEventTrainer().fit(
            _model(),
            [_batch()],
            max_steps=1,
            learning_rate=1e-3,
            seed=0,
            device=torch.device("cpu"),
            optimizer=OptimizerConfig(name="muon"),
            precision="bf16",
        )
