# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Name registry for the flat stack's history backbone (`model.backbone`).

A backbone replaces the eager decoder inside `TrxFM`. It is a module taking
`(input_ids, attention_mask, *, causal)`, exposing `hidden_size`, `vocab_size`,
`logits(hidden)`, and a `position` attribute naming the positions it applies
itself. Register a builder `(config, *, vocab_size) -> nn.Module` with
`register_backbone("name", builder)`, or, for an optional extra, with an
import path `"module:function"` so selecting it is the only import.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from torch import nn

from tfm.config import RecipeConfig
from tfm.contracts.errors import IncompatibleCompositionError
from tfm.layers.decoder import Decoder

BackboneBuilder = Callable[..., nn.Module]

_BACKBONES: dict[str, BackboneBuilder | str | None] = {}


def register_backbone(name: str, builder: BackboneBuilder | str | None) -> None:
    """`None` keeps the eager decoder; a string is a lazy `module:function` target."""
    if isinstance(builder, str) and ":" not in builder:
        raise ValueError(f"Backbone target must be 'module:function', got {builder!r}.")
    _BACKBONES[name] = builder


def registered_backbones() -> list[str]:
    return sorted(_BACKBONES)


def build_backbone(config: RecipeConfig, vocab_size: int) -> nn.Module | None:
    """The module `model.backbone` names, or `None` for the eager decoder."""
    name = config.model.backbone
    if name not in _BACKBONES:
        raise IncompatibleCompositionError(
            f"Unknown model.backbone {name!r}. Registered backbones: "
            f"{', '.join(registered_backbones())}."
        )
    builder = _BACKBONES[name]
    if builder is None:
        return None
    if isinstance(builder, str):
        module, attribute = builder.split(":", 1)
        builder = getattr(importlib.import_module(module), attribute)
    return builder(config, vocab_size=vocab_size)


def _gru(config: RecipeConfig, *, vocab_size: int) -> Any:
    from tfm.layers.gru import GRUBackbone

    dec = config.model.composition.granularity.decoder
    return GRUBackbone(
        Decoder(
            depth=dec.depth,
            hidden_size=dec.hidden_size,
            num_heads=dec.num_heads,
            dropout=dec.dropout,
            vocab_size=vocab_size,
            max_seq_len=max(dec.max_seq_len, config.data.max_tokens),
        )
    )


register_backbone("eager", None)
register_backbone("gru", _gru)
register_backbone("huggingface", "tfm.extras.huggingface:build_huggingface_backbone")

__all__ = ["BackboneBuilder", "build_backbone", "register_backbone", "registered_backbones"]
