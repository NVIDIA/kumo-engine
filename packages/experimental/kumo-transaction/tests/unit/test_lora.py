# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LoRA adaptation of frozen linear layers."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tfm.compositions.profile_history import ProfileHistory
from tfm.contracts.errors import IncompatibleCompositionError
from tfm.layers.profile_history import ProfileHistoryModel
from tfm.training.lora import LoRA, LoraLinear, apply_lora, lora_parameters


def _tiny_stack() -> ProfileHistoryModel:
    torch.manual_seed(0)
    return ProfileHistoryModel(
        ProfileHistory(hidden_size=16, num_heads=4, vocab_size=32, dropout=0.0)
    )


def test_an_adapted_layer_starts_as_an_exact_copy() -> None:
    base = nn.Linear(8, 4)
    hidden = torch.randn(3, 8)

    adapted = LoraLinear(base, LoRA(rank=2))

    torch.testing.assert_close(adapted(hidden), base(hidden))


def test_the_update_is_low_rank_and_scaled() -> None:
    base = nn.Linear(8, 6)
    adapted = LoraLinear(base, LoRA(rank=2, alpha=4.0))
    with torch.no_grad():
        adapted.lora_b.copy_(torch.randn_like(adapted.lora_b))

    delta = adapted.merged_weight() - base.weight

    assert torch.linalg.matrix_rank(delta) <= 2
    assert adapted.scaling == pytest.approx(2.0)
    hidden = torch.randn(3, 8)
    torch.testing.assert_close(
        adapted(hidden), base(hidden) + hidden @ delta.t(), atol=1e-5, rtol=1e-4
    )


def test_apply_lora_freezes_the_base_and_trains_only_adapters() -> None:
    model = _tiny_stack()
    n_before = sum(p.numel() for p in model.parameters())

    stats = apply_lora(model, LoRA(rank=4, targets=("qkv", "out")))

    assert stats.n_adapted > 0
    assert all("qkv" in name or "out" in name for name in stats.adapted_modules)
    trainable = lora_parameters(model)
    assert trainable
    assert all(
        name.rsplit(".", 1)[-1].startswith("lora_")
        for name, p in model.named_parameters()
        if p.requires_grad
    )
    # Only adapters are added; the pretrained weights are all still there, frozen.
    assert stats.n_frozen == n_before
    assert stats.as_dict()["trainable_fraction"] < 0.2


def test_only_adapter_and_head_parameters_receive_gradient() -> None:
    model = _tiny_stack()
    apply_lora(model, LoRA(rank=4))
    hidden = torch.randn(2, 5, 16)

    out = model.event_encoder.blocks[0](hidden, torch.ones(2, 5, dtype=torch.long))
    out.sum().backward()

    grads = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    assert grads
    assert all("lora_" in name for name in grads)


def test_lora_targets_must_match_something() -> None:
    with pytest.raises(IncompatibleCompositionError, match="matched no nn.Linear"):
        apply_lora(_tiny_stack(), LoRA(targets=("not_a_module",)))


def test_invalid_lora_specs_are_rejected() -> None:
    with pytest.raises(IncompatibleCompositionError, match="rank"):
        LoRA(rank=0)
    with pytest.raises(IncompatibleCompositionError, match="target"):
        LoRA(targets=())
