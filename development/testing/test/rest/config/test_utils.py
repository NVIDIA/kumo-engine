# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from rest.config.training_job import Metric
from rest.config.utils import filter_metrics, remove_test_prefix


def test_utils():
    metrics = [
        Metric('train_error', 0.5),
        Metric('val_error', 0.5),
        Metric('test_auc', 0.5),
        Metric('test_error', 0.5)
    ]

    metrics_filtered = filter_metrics(metrics, ["train_.*", "val_.*"])
    assert len(metrics_filtered) == 2
    assert all('test_' in metric.name for metric in metrics_filtered)

    metrics_renamed = remove_test_prefix(metrics)
    assert len(metrics_renamed) == 4
    assert all('test_' not in metric.name for metric in metrics_renamed)
