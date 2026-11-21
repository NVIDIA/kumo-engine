# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable Rust-backed sampling values exposed by PyO3."""

from typing import Literal

from ._c import (
    CustomExpression,
    ExponentialDecay,
    FanoutSpec,
    LayerSpec,
    LinearDecay,
    SamplingPlan,
    WeightedSampling,
)

__all__ = [
    "AbsentKeyPolicy",
    "CustomExpression",
    "ExponentialDecay",
    "FanoutSpec",
    "LayerSpec",
    "LinearDecay",
    "SamplingMode",
    "SamplingPlan",
    "WeightedSampling",
    "Weighting",
]

Weighting = ExponentialDecay | LinearDecay | CustomExpression
SamplingMode = (
    Literal["uniform", "deterministic_uniform", "temporal_most_recent"]
    | WeightedSampling
)
AbsentKeyPolicy = Literal["reject_plan", "uniform_fallback"]
