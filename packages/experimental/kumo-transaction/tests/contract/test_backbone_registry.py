# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`model.backbone` resolves through a name registry, like `training.backend`."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from torch import Tensor, nn

from tfm.api import build_backbone, register_backbone, registered_backbones
from tfm.api.model import TrxFM
from tfm.config import RecipeConfig, load_recipe_config
from tfm.contracts.errors import IncompatibleCompositionError
from tfm.runtime import build_model

FLAT = Path("tests/fixtures/configs/flat_decoder.yaml")
V2 = Path("recipes/blueprint_v2/config.smoke.yaml")


class _Lookup(nn.Module):
    """An embedding lookup: the smallest module the backbone contract accepts."""

    position = "nope"

    def __init__(self, config: RecipeConfig, *, vocab_size: int) -> None:
        super().__init__()
        self.hidden_size = config.model.composition.granularity.decoder.hidden_size
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, self.hidden_size)
        self.head = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, input_ids: Tensor, attention_mask: Tensor, *, causal: bool = True) -> Tensor:
        return self.embed(input_ids)

    def logits(self, hidden: Tensor) -> Tensor:
        return self.head(hidden)


def _with_backbone(path: Path, name: str) -> RecipeConfig:
    raw = load_recipe_config(path).model_dump(mode="json")
    raw["model"]["backbone"] = name
    return RecipeConfig.model_validate(raw)


def test_the_builtin_backbones_are_registered() -> None:
    assert {"eager", "gru", "huggingface"} <= set(registered_backbones())
    assert build_backbone(load_recipe_config(FLAT), vocab_size=32) is None


def test_a_registered_backbone_drives_the_flat_stack() -> None:
    register_backbone("test_lookup", _Lookup)
    model = build_model(_with_backbone(FLAT, "test_lookup"), vocab_size=32)
    assert isinstance(model, TrxFM)
    assert isinstance(model.backbone, _Lookup)
    hidden = model.backbone(torch.zeros(2, 5, dtype=torch.long), torch.ones(2, 5))
    assert hidden.shape == (2, 5, model.hidden_size)


def test_an_unknown_backbone_names_the_registered_ones() -> None:
    with pytest.raises(IncompatibleCompositionError, match="Registered backbones: .*eager"):
        build_backbone(_with_backbone(FLAT, "mamba"), vocab_size=32)


def test_a_backbone_on_a_hierarchical_stack_is_refused_at_load() -> None:
    with pytest.raises(ValidationError, match="replaces the flat decoder only"):
        _with_backbone(V2, "gru")


def test_a_lazy_target_must_name_module_and_function() -> None:
    with pytest.raises(ValueError, match="module:function"):
        register_backbone("broken", "tfm.extras.huggingface")
