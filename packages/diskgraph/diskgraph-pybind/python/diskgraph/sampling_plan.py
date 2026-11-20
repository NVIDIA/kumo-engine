# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed Python sampling plans matching the native DiskGraph traversal schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ExponentialDecay:
    """Weight edges by ``exp(-rate * delta_seconds)``.

    Args:
        rate: Decay rate per second, corresponding to Rust's ``lambda`` field.
    """

    rate: float


@dataclass
class LinearDecay:
    """Weight edges by ``max(0, 1 - delta_seconds / half_life_seconds)``.

    Args:
        half_life_seconds: Native linear-decay time scale in seconds.
    """

    half_life_seconds: int


@dataclass
class CustomExpression:
    """Identify a native custom weighting expression.

    Args:
        id: Expression identifier interpreted by the native engine.
    """

    id: str


Weighting = ExponentialDecay | LinearDecay | CustomExpression


@dataclass
class WeightedSampling:
    """Select weighted or temporal-weighted neighbors using a native weighting.

    Args:
        weighting: Native weighting function and its parameters.
        temporal: Select Rust's ``temporal_weighted`` mode instead of ``weighted``.
    """

    weighting: Weighting
    temporal: bool = False


SamplingMode = (
    Literal["uniform", "deterministic_uniform", "temporal_most_recent"]
    | WeightedSampling
)
AbsentKeyPolicy = Literal["reject_plan", "uniform_fallback"]


@dataclass
class FanoutSpec:
    """Neighbor count, selection strategy, and missing-key policy for one relation.

    Args:
        fanout: Maximum number of neighbors; zero disables the relation.
        mode: Native selection strategy.
        absent_key_policy: Behavior when a required time/weight key is absent.
    """

    fanout: int
    mode: SamplingMode
    absent_key_policy: AbsentKeyPolicy


@dataclass
class LayerSpec:
    """Per-relation fanouts and an optional fallback for one traversal hop.

    Args:
        fanouts: Overrides keyed by native relation name.
        default_fanout: Selection for relations absent from ``fanouts``;
            None skips them.
    """

    fanouts: dict[str, FanoutSpec] = field(default_factory=dict)
    default_fanout: FanoutSpec | None = None


@dataclass
class SamplingPlan:
    """Ordered traversal hops with ordinary Python list/dictionary containers.

    The dataclasses mirror DiskGraph's Rust sampling types. Rust remains
    responsible for validating graph-specific sampling behavior.

    Args:
        layers: Traversal hops in order from the seed nodes.
    """

    layers: list[LayerSpec] = field(default_factory=list)
