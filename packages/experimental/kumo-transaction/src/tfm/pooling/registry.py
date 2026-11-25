# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Readout names, specs, and modules (slot 4, building-block contracts).

Every readout module takes `(states [B, S, D], attention_mask [B, S])` and
returns `[B, D]`, so a model swaps readouts without changing its forward pass.
Valid positions must be left-aligned. Whether a readout is valid for a history
model is decided by `tfm.api.validation`, not here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast, runtime_checkable

from torch import nn

from tfm.contracts.errors import UnsupportedCombinationError
from tfm.contracts.registries import Registry
from tfm.pooling.last_event import LastEventMeanPool
from tfm.pooling.last_token import LastTokenPool, LastTokenPoolModule
from tfm.pooling.mean import MeanPool, MeanPoolModule
from tfm.pooling.usr_token import UsrTokenPool, UsrTokenPoolModule


@runtime_checkable
class ReadoutSpec(Protocol):
    """A user readout declares its name and constructs a [states, mask] -> embedding module."""

    @property
    def name(self) -> str: ...

    def build(self) -> nn.Module: ...


Readout = LastTokenPool | UsrTokenPool | MeanPool | LastEventMeanPool | ReadoutSpec

_SPECS: Registry[Callable[[], Readout]] = Registry("readout")
for _name, _type in (
    ("last_token", LastTokenPool),
    ("usr_token", UsrTokenPool),
    ("mean", MeanPool),
    ("last_event_mean", LastEventMeanPool),
):
    _SPECS.register(_name, _type)


def register_readout(name: str, factory: Callable[[], Readout]) -> None:
    """Register a spec factory with the same lifecycle as built-in readouts."""
    _SPECS.register(name, factory)


def build_readout(name: str) -> Readout:
    """Resolve a registered/importable factory for a readout spec."""
    factory = _SPECS.get(name)
    if not callable(factory):
        raise UnsupportedCombinationError(f"Readout {name!r} must name a spec factory")
    spec = factory()
    if not isinstance(
        spec, (LastTokenPool, UsrTokenPool, MeanPool, LastEventMeanPool, ReadoutSpec)
    ):
        raise UnsupportedCombinationError(f"Readout {name!r} must return a declared readout spec")
    return cast(Readout, spec)


def readout_module(spec: Readout) -> nn.Module:
    """The module that applies `spec`. `LastEventMeanPool` needs token ids, so
    `FlatEncoderModel` applies it itself."""
    if isinstance(spec, LastTokenPool):
        return LastTokenPoolModule()
    if isinstance(spec, MeanPool):
        return MeanPoolModule()
    if isinstance(spec, UsrTokenPool):
        return UsrTokenPoolModule()
    if isinstance(spec, ReadoutSpec):
        module = spec.build()
        if not isinstance(module, nn.Module):
            raise UnsupportedCombinationError("Readout build() must return nn.Module")
        return module
    raise UnsupportedCombinationError(f"No readout module for {type(spec).__name__}.")
