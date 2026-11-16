# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from typing import List

from rest.config.training_job import Metric


def filter_metrics(metric_list: List[Metric],
                   filter_patterns: List[str]) -> List[Metric]:
    """Filters out metrics from metric_list if `Metric.name`
    matches some pattern in `filter_paterns`."""
    new_metric_list: List[Metric] = []
    for metric in metric_list:
        metric_matches_a_pattern = False
        for pat in filter_patterns:
            if re.match(pat, metric.name):
                metric_matches_a_pattern = True
                break
        if not metric_matches_a_pattern:
            new_metric_list.append(metric)
    return new_metric_list


def remove_test_prefix(metric_list: List[Metric]) -> List[Metric]:
    """Removes the "test_" prefix from `Metric.name`
    if present"""
    for metric in metric_list:
        if re.match("test_.*", metric.name):
            metric.name = re.sub("test_", "", metric.name)
    return metric_list
