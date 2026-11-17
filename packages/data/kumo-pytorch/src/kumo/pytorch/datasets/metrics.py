# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metrics for scoring models against the generated datasets' planted signals."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

__all__ = ["auroc", "auroc_standard_error"]


def auroc(labels: npt.NDArray[np.bool_], scores: npt.NDArray[np.float64]) -> float:
    """Area under the ROC curve; tied scores count one half (Mann-Whitney U)."""
    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError(f"auroc needs equal 1-D arrays, got {labels.shape} and {scores.shape}")
    if np.isnan(scores).any():
        raise ValueError("auroc scores contain NaN")
    positives = int(labels.sum())
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        raise ValueError("auroc needs at least one positive and one negative label")
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    average_rank = np.cumsum(counts) - (counts - 1) / 2.0
    rank_sum = float(average_rank[inverse][labels].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def auroc_standard_error(area: float, positives: int, negatives: int) -> float:
    """Hanley and McNeil (1982) standard error of an AUROC from its class counts."""
    q1 = area / (2.0 - area)
    q2 = 2.0 * area * area / (1.0 + area)
    variance = (
        area * (1.0 - area)
        + (positives - 1) * (q1 - area * area)
        + (negatives - 1) * (q2 - area * area)
    ) / (positives * negatives)
    return math.sqrt(variance)
