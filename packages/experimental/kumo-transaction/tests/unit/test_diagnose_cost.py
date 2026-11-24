# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`cost_from_composition` counts the model that `TrxFM.to_config()` describes."""

from __future__ import annotations

import pytest
from torch import Tensor, nn

from tfm import Decoder, Flat, TrxFM
from tfm.contracts.errors import SchemaError
from tfm.diagnose.cost import cost_from_composition


@pytest.mark.parametrize("ffn_size", [None, 24])
def test_cost_counts_the_eager_backbone_exactly(ffn_size: int | None) -> None:
    model = TrxFM(
        granularity=Flat(
            Decoder(depth=2, hidden_size=16, num_heads=4, vocab_size=20, ffn_size=ffn_size)
        )
    )
    report = cost_from_composition(model.to_config())
    assert report.params.total == sum(p.numel() for p in model.backbone.parameters())


def test_cost_refuses_a_backbone_it_does_not_model() -> None:
    composition = {
        "granularity": {"type": "flat", "decoder": None},
        "backbone": "HuggingFaceCausalBackbone",
    }
    with pytest.raises(SchemaError, match="does not model"):
        cost_from_composition(composition, seq_len=16)


def test_cost_refuses_a_custom_backbone_from_to_config() -> None:
    class Custom(nn.Module):
        hidden_size, vocab_size = 8, 20

        def logits(self, hidden: Tensor) -> Tensor:  # pragma: no cover - never called
            return hidden

    model = TrxFM(granularity=Flat(Decoder(vocab_size=20)), backbone=Custom())
    with pytest.raises(SchemaError, match="Custom"):
        cost_from_composition(model.to_config(), seq_len=16)
