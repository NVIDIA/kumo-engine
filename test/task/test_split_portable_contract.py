# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pins `kumo.task.split` to the canonical constructors in `kumo_core`.

The split families a model plan can express are defined once, in
`kumo_core.training.splits`: argument order, defaults, derived values,
validation and the repr a plan stores. `kumo.task.split` builds its
registered, mask-generating splits on top of them. These tests hold that
relationship in place -- and hold the two things the platform side must keep
working: `ConfigMixin.config` must not be shadowed, and the repr must not move.
"""

from __future__ import annotations

import pytest
from kumo_core.training.splits import PLAN_SPLITS

from kumo.config import ConfigMixin
from kumo.task.split import (
    ApproxDateOffsetSplit,
    DateOffsetSplit,
    ManualSplit,
    RandomLinkSplit,
    RandomSplit,
    TemporalSplit,
    TimeRangeSplit,
)

PLATFORM_SPLITS = {
    "ApproxDateOffsetSplit": ApproxDateOffsetSplit,
    "DateOffsetSplit": DateOffsetSplit,
    "ManualSplit": ManualSplit,
    "RandomLinkSplit": RandomLinkSplit,
    "RandomSplit": RandomSplit,
    "TemporalSplit": TemporalSplit,
    "TimeRangeSplit": TimeRangeSplit,
}

ARGUMENTS = {
    "ManualSplit": ("SPLIT", {}),
    "RandomSplit": ([0.8, 0.1, 0.1], {}),
    "RandomLinkSplit": ([0.7, 0.3], {}),
    "TemporalSplit": ([0.8, 0.1, 0.1], {}),
    "ApproxDateOffsetSplit": ([0.8, 0.1, 0.1], {}),
    "DateOffsetSplit": ([-2, 0], {}),
    "TimeRangeSplit": ([["2024-01-01", "2024-02-01"],
                        ["2024-02-01", "2024-03-01"]], {}),
}

EXPECTED_NUM_SPLITS = {
    "ApproxDateOffsetSplit": 3,
    "DateOffsetSplit": 3,
    "ManualSplit": 2,
    "RandomLinkSplit": 2,
    "RandomSplit": 3,
    "TemporalSplit": 3,
    "TimeRangeSplit": 2,
}


def test_every_plan_split_family_is_covered() -> None:
    assert (set(PLAN_SPLITS) == set(PLATFORM_SPLITS) == set(ARGUMENTS) ==
            set(EXPECTED_NUM_SPLITS))


@pytest.mark.parametrize("family", sorted(PLATFORM_SPLITS))
def test_platform_split_uses_the_canonical_constructor(family: str) -> None:
    platform = PLATFORM_SPLITS[family]

    assert issubclass(platform, PLAN_SPLITS[family])
    # The constructor is inherited, never redeclared: one definition of the
    # argument order, defaults and validation.
    assert "__init__" not in vars(platform)


@pytest.mark.parametrize("family", sorted(PLATFORM_SPLITS))
def test_platform_split_repr_comes_from_the_canonical_class(
        family: str) -> None:
    args, kwargs = ARGUMENTS[family]
    platform = PLATFORM_SPLITS[family](args, **kwargs)
    canonical = PLAN_SPLITS[family](args, **kwargs)

    # A model plan stores this string, so the two must not drift apart.
    assert repr(platform) == repr(canonical)


@pytest.mark.parametrize("family", sorted(PLATFORM_SPLITS))
def test_platform_split_inherits_the_canonical_stage_count(family: str) -> None:
    args, kwargs = ARGUMENTS[family]
    platform_class = PLATFORM_SPLITS[family]
    canonical = PLAN_SPLITS[family](args, **kwargs)
    platform = platform_class(args, **kwargs)

    assert "num_splits" not in vars(platform_class)
    assert canonical.num_splits == platform.num_splits \
        == EXPECTED_NUM_SPLITS[family]


@pytest.mark.parametrize("family", sorted(PLATFORM_SPLITS))
def test_canonical_class_does_not_shadow_config_serialization(
        family: str) -> None:
    platform = PLATFORM_SPLITS[family]

    # `to_split_config` is deliberately not named `config`: the canonical
    # classes sit ahead of `ConfigMixin` in the MRO, so a `config` there would
    # take over platform serialization.
    assert platform.config is ConfigMixin.config
    assert hasattr(platform, "to_split_config")
