# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A composition's position slot says what its backbone actually applies (Blueprint v1 guide)."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from tfm import Decoder, Flat, NoPE, TrxFM
from tfm.config import RecipeConfig
from tfm.contracts.errors import IncompatibleCompositionError
from tfm.extras.capability import extra_available
from tfm.position.rope import RoPE

VOCAB = 32


def _config(backbone: str, position: str, attention: str = "eager") -> dict[str, Any]:
    return {
        "seed": 0,
        "data": {
            "adapter": "synthetic",
            "max_tokens": 12,
            "temporal_split": {
                "train_end": "2024-01-04T00:00:00Z",
                "validation_end": "2024-01-07T00:00:00Z",
            },
        },
        "model": {
            "backbone": backbone,
            "composition": {
                "position": position,
                "attention": attention,
                "granularity": {
                    "decoder": {"depth": 1, "hidden_size": 32, "num_heads": 4, "max_seq_len": 12}
                },
            },
        },
    }


class _RotaryBackbone(torch.nn.Module):
    position = "rope"
    hidden_size = 8
    vocab_size = VOCAB

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, *, causal: bool):
        return torch.zeros(*input_ids.shape, self.hidden_size)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.zeros(*hidden.shape[:-1], VOCAB)


def _decoder() -> Flat:
    return Flat(Decoder(depth=1, hidden_size=8, num_heads=2, vocab_size=VOCAB))


@pytest.mark.parametrize(
    ("backbone", "position"), [("huggingface", "nope"), ("eager", "rope"), ("gru", "rope")]
)
def test_config_refuses_a_position_the_backbone_does_not_apply(
    backbone: str, position: str
) -> None:
    with pytest.raises(ValueError, match="must say what the backbone applies"):
        RecipeConfig.model_validate(_config(backbone, position))


def test_eager_decoder_refuses_rope() -> None:
    with pytest.raises(IncompatibleCompositionError, match="applies 'nope' positions"):
        TrxFM(granularity=_decoder(), position=RoPE())


def test_rotary_backbone_refuses_nope() -> None:
    with pytest.raises(IncompatibleCompositionError, match="applies 'rope' positions"):
        TrxFM(granularity=_decoder(), position=NoPE(), backbone=_RotaryBackbone())


def test_rotary_backbone_accepts_rope() -> None:
    model = TrxFM(granularity=_decoder(), position=RoPE(), backbone=_RotaryBackbone())
    assert model.position == RoPE()


@pytest.mark.skipif(not extra_available("transformers"), reason="transformers extra not installed")
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_llama_runs_the_declared_attention_kernel(attention: str) -> None:
    from tfm.runtime.pipeline import build_model

    config = RecipeConfig.model_validate(_config("huggingface", "rope", attention))
    model = build_model(config, VOCAB)
    assert model.backbone.model.config._attn_implementation == attention
