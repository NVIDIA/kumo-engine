# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from kumo_core.train.stats import (
    AllTargetRegressionStats,
    TargetHistogramAndStats,
    parse_histogram_and_stats,
)


def test_regression_stats_have_one_portable_owner():
    parsed = parse_histogram_and_stats(
        SimpleNamespace(
            target_distribution={
                "target_mean": 2.0,
                "histogram": {
                    "bins": [1.0, 2.0],
                    "counts": [3],
                    "bin_width": 1.0,
                    "num_bins": 1,
                },
            }))

    assert isinstance(parsed, TargetHistogramAndStats)
    assert parsed.histogram.counts == [3]
    assert AllTargetRegressionStats.empty(
    ).label_distribution.test_table_stats is None
