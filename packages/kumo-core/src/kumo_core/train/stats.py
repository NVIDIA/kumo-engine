# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable regression evaluation statistics value types."""

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from pydantic import TypeAdapter


@dataclass
class TargetHistogram:
    bins: list[float] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    bin_width: float | None = 0
    num_bins: int = 0


@dataclass
class TargetHistogramAndStats:
    target_mean: float | None = None
    target_min: float | None = None
    target_max: float | None = None
    target_percentile_25: float | None = None
    target_percentile_75: float | None = None
    target_median: float | None = None
    target_mode: float | None = None
    target_mode_percentage: float | None = None
    target_mode_count: int | None = None
    histogram: TargetHistogram = field(default_factory=TargetHistogram)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TargetHistogramAndStatsPair:
    label: TargetHistogramAndStats
    prediction: TargetHistogramAndStats


@dataclass
class HistogramAndStatsGroup:
    train_table_stats: TargetHistogramAndStats | None
    val_table_stats: TargetHistogramAndStats | None
    test_table_stats: TargetHistogramAndStats | None


@dataclass
class AllTargetRegressionStats:
    label_distribution: HistogramAndStatsGroup
    prediction_distribution: HistogramAndStatsGroup

    @staticmethod
    def empty() -> "AllTargetRegressionStats":
        return AllTargetRegressionStats(
            label_distribution=HistogramAndStatsGroup(None, None, None),
            prediction_distribution=HistogramAndStatsGroup(None, None, None),
        )


class TargetDistributionStats(Protocol):
    target_distribution: Any


def parse_histogram_and_stats(
    stats: TargetDistributionStats | None, ) -> TargetHistogramAndStats | None:
    if stats is None or stats.target_distribution is None:
        return None
    return TypeAdapter(TargetHistogramAndStats).validate_python(
        stats.target_distribution)
