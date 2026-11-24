# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed interfaces for immutable native sampling values."""

from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias

class ExponentialDecay:
    """Exponential edge weighting with a per-second decay rate."""
    def __init__(self, rate: float) -> None: ...
    @property
    def rate(self) -> float: ...

class LinearDecay:
    """Linear edge weighting with a time scale in seconds."""
    def __init__(self, half_life_seconds: int) -> None: ...
    @property
    def half_life_seconds(self) -> int: ...

class CustomExpression:
    """A native custom weighting expression identifier."""
    def __init__(self, id: str) -> None: ...
    @property
    def id(self) -> str: ...

Weighting: TypeAlias = ExponentialDecay | LinearDecay | CustomExpression

class WeightedSampling:
    """A native weighted or temporal-weighted sampling strategy."""
    def __init__(self, weighting: Weighting, temporal: bool = False) -> None: ...
    @property
    def weighting(self) -> Weighting: ...
    @property
    def temporal(self) -> bool: ...

SamplingMode: TypeAlias = (
    Literal["uniform", "deterministic_uniform", "temporal_most_recent"]
    | WeightedSampling
)
AbsentKeyPolicy: TypeAlias = Literal["reject_plan", "uniform_fallback"]

class FanoutSpec:
    """A native neighbor count, selection strategy, and missing-key policy."""
    def __init__(
        self, fanout: int, mode: SamplingMode, absent_key_policy: AbsentKeyPolicy
    ) -> None: ...
    @property
    def fanout(self) -> int: ...
    @property
    def mode(self) -> SamplingMode: ...
    @property
    def absent_key_policy(self) -> AbsentKeyPolicy: ...

class LayerSpec:
    """Immutable native relation overrides and an optional fallback."""
    def __init__(
        self,
        fanouts: Mapping[str, FanoutSpec] | None = None,
        default_fanout: FanoutSpec | None = None,
    ) -> None: ...
    @property
    def fanouts(self) -> Mapping[str, FanoutSpec]: ...
    @property
    def default_fanout(self) -> FanoutSpec | None: ...

class SamplingPlan:
    """An immutable Rust-owned sequence of traversal layers."""
    def __init__(self, layers: Sequence[LayerSpec] | None = None) -> None: ...
    @property
    def layers(self) -> tuple[LayerSpec, ...]: ...
