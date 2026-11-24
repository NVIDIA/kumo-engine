# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fraud metrics including calibration ECE."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, roc_auc_score


def expected_calibration_error(
    y_true: NDArray[np.floating],
    y_prob: NDArray[np.floating],
    n_bins: int = 10,
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if not np.any(mask):
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += (float(mask.sum()) / n) * abs(acc - conf)
    return float(ece)


def precision_at_k(y_true: NDArray[np.floating], y_score: NDArray[np.floating], k: int) -> float:
    if len(y_true) == 0:
        return 0.0
    k = max(1, min(k, len(y_true)))
    order = np.argsort(-y_score)[:k]
    return float(y_true[order].mean())


def recall_at_k(y_true: NDArray[np.floating], y_score: NDArray[np.floating], k: int) -> float:
    total = float(y_true.sum())
    if total == 0:
        return 0.0
    k = max(1, min(k, len(y_true)))
    order = np.argsort(-y_score)[:k]
    return float(y_true[order].sum() / total)


MIN_TRUSTWORTHY_POSITIVES = 5
"""Below this many evaluation positives, a ranking metric is not a finding.

A probe evaluated on one positive row can only score 0.0 or 1.0 on PR-AUC, so
reporting the number without its support invites reading noise as a result
(ADR 0008). This threshold gates presentation, never computation.
"""


def classification_metrics(
    y_true: NDArray[np.floating],
    y_score: NDArray[np.floating],
    *,
    k: int = 5,
) -> dict[str, float]:
    """Ranking, threshold, and calibration metrics plus their support.

    `n_eval` and `n_eval_positive` travel with the scores so a reader cannot
    see `roc_auc: 1.0` without seeing how many positives produced it.
    """
    y_true = y_true.astype(np.float64)
    y_score = y_score.astype(np.float64)
    unique = np.unique(y_true)
    roc = 0.5 if unique.size < 2 else float(roc_auc_score(y_true, y_score))
    pr = float(average_precision_score(y_true, y_score)) if unique.size >= 1 else 0.0
    pred = (y_score >= 0.5).astype(int)
    precision = float(np.mean(pred[pred == 1] == y_true[pred == 1])) if pred.sum() else 0.0
    recall = (
        float(np.sum((pred == 1) & (y_true == 1)) / np.sum(y_true == 1)) if y_true.sum() else 0.0
    )
    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "precision_at_k": precision_at_k(y_true, y_score, k),
        "recall_at_k": recall_at_k(y_true, y_score, k),
        "calibration": expected_calibration_error(y_true, y_score),
        "precision": precision,
        "recall": recall,
        "n_eval": float(len(y_true)),
        "n_eval_positive": float(y_true.sum()),
    }


def is_supported(metrics: Mapping[str, float]) -> bool:
    """True when a report has enough evaluation positives to be read as one."""
    return metrics.get("n_eval_positive", 0.0) >= MIN_TRUSTWORTHY_POSITIVES
