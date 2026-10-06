# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo.metrics import Metrics, MetricsClient
from kumo.metrics.metrics_client import DEFAULT_METRIC_LABEL


def test_metrics_client_get_and_set():
    metrics_client = MetricsClient()
    metrics_client.get_metric(Metrics.TABLE_READ_TIME).clear()
    metrics_client.set_metric_value(Metrics.TABLE_READ_TIME, value=10,
                                    labels={"table_name": "table1"})
    metrics_client.set_metric_value(Metrics.TABLE_READ_TIME, value=10,
                                    labels={"table_name": "table1"})
    metrics_client.set_metric_value(Metrics.TABLE_READ_TIME, value=5,
                                    labels={"table_name": "table2"})

    # Check the values as (sum, count, avg)
    values = metrics_client.get_metric_values(
        metrics_client.get_metric(Metrics.TABLE_READ_TIME))
    assert values["table1"] == (20, 2, 10)
    assert values["table2"] == (5, 1, 5)

    # Check individual metric value.
    values = metrics_client.get_metric_values(
        metrics_client.get_metric(Metrics.TABLE_READ_TIME,
                                  labels={"table_name": "table1"}))
    assert values[DEFAULT_METRIC_LABEL] == (20, 2, 10)
