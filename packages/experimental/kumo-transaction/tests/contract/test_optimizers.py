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
from tfm.training.optim import GroupedMuon, build_optimizers, muon_parameter_split
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
    assert [type(o).__name__ for o in opts] == ["GroupedMuon", "AdamW"]
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


SHAPES = [(16, 16), (16, 16), (48, 16), (48, 16), (16, 48), (16, 6), (32, 16)]
"""Repeated, tall, wide and single shapes."""


def _muon_runs(device: str, **kwargs: object) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Five steps of `torch.optim.Muon` and `GroupedMuon` from the same weights and grads."""
    torch.manual_seed(0)
    start = [torch.randn(shape, device=device) for shape in SHAPES]
    oracle = [nn.Parameter(w.clone()) for w in start]
    grouped = [nn.Parameter(w.clone()) for w in start]
    optimizers = (
        torch.optim.Muon(oracle, lr=0.02, weight_decay=0.01, **kwargs),  # type: ignore[arg-type]
        GroupedMuon(grouped, lr=0.02, weight_decay=0.01, **kwargs),  # type: ignore[arg-type]
    )
    generator = torch.Generator(device=device).manual_seed(1)
    for _ in range(5):
        for a, b in zip(oracle, grouped, strict=True):
            grad = torch.randn(a.shape, generator=generator, device=device)
            a.grad, b.grad = grad.clone(), grad.clone()
        for optimizer in optimizers:
            optimizer.step()
    return oracle, grouped


@pytest.mark.parametrize("nesterov", [True, False])
@pytest.mark.parametrize("adjust_lr_fn", ["original", "match_rms_adamw"])
def test_grouped_muon_updates_as_torch_muon_does(nesterov: bool, adjust_lr_fn: str) -> None:
    oracle, grouped = _muon_runs("cpu", nesterov=nesterov, adjust_lr_fn=adjust_lr_fn)
    assert all(torch.equal(a, b) for a, b in zip(oracle, grouped, strict=True))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_grouped_muon_updates_as_torch_muon_does_on_cuda() -> None:
    oracle, grouped = _muon_runs("cuda")
    for a, b in zip(oracle, grouped, strict=True):
        torch.testing.assert_close(b, a, rtol=0, atol=1e-6)


def test_grouped_muon_state_loads_into_torch_muon_and_back() -> None:
    oracle, grouped = _muon_runs("cpu")
    del oracle
    saved = GroupedMuon(grouped, lr=0.02)
    for p in grouped:
        p.grad = torch.ones_like(p)
    saved.step()
    state = saved.state_dict()
    plain = torch.optim.Muon(grouped, lr=0.02)
    plain.load_state_dict(state)
    again = GroupedMuon(grouped, lr=0.02)
    again.load_state_dict(plain.state_dict())
    assert again.state_dict()["state"].keys() == state["state"].keys()
    assert all(
        torch.equal(again.state[p]["momentum_buffer"], saved.state[p]["momentum_buffer"])
        for p in grouped
    )
