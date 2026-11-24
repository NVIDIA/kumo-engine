# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Readout names, specs, and modules (slot 4, ADR 0014).

Every readout module takes `(states [B, S, D], attention_mask [B, S])` and
returns `[B, D]`, so a model swaps readouts without changing its forward pass.
Valid positions must be left-aligned. Whether a readout is valid for a history
model is decided by `tfm.api.validation`, not here.
"""

from __future__ import annotations

from torch import nn

from tfm.contracts.errors import UnsupportedCombinationError
from tfm.pooling.last_event import LastEventMeanPool
from tfm.pooling.last_token import LastTokenPool, LastTokenPoolModule
from tfm.pooling.mean import MeanPool, MeanPoolModule
from tfm.pooling.usr_token import UsrTokenPool, UsrTokenPoolModule

Readout = LastTokenPool | UsrTokenPool | MeanPool | LastEventMeanPool

_SPECS: dict[str, type] = {
    "last_token": LastTokenPool,
    "usr_token": UsrTokenPool,
    "mean": MeanPool,
    "last_event_mean": LastEventMeanPool,
}


def build_readout(name: str) -> Readout:
    """The readout spec a config names."""
    if name not in _SPECS:
        raise UnsupportedCombinationError(f"Unknown readout {name!r}; available: {sorted(_SPECS)}.")
    spec: Readout = _SPECS[name]()
    return spec


def readout_module(spec: Readout) -> nn.Module:
    """The module that applies `spec`. `LastEventMeanPool` needs token ids, so
    `FlatEncoderModel` applies it itself."""
    if isinstance(spec, LastTokenPool):
        return LastTokenPoolModule()
    if isinstance(spec, MeanPool):
        return MeanPoolModule()
    if isinstance(spec, UsrTokenPool):
        return UsrTokenPoolModule()
    raise UnsupportedCombinationError(f"No readout module for {type(spec).__name__}.")
