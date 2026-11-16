# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared metric computation for paper benchmarks."""

from __future__ import annotations

import numpy as np
from kumoapi.task import TaskType


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray | None,
    y_pred: np.ndarray | None,
    task_type: TaskType,
) -> dict[str, float]:
    """Compute core metrics for binary, multiclass, and regression tasks.

    Returns a dict with keys:
    - Binary:  ``roc_auc``, ``f1``, ``ap``
    - Multiclass: ``mrr``, ``acc``, ``f1_macro``
    - Regression: ``mae``, ``mse``, ``rmse``, ``r2``
    """
    # Scikit-learn is only needed when metrics are computed. Keeping the
    # optional import here lets users import the dataset examples without
    # installing benchmark-only dependencies.
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )

    if task_type == TaskType.BINARY_CLASSIFICATION:
        if y_pred_proba is None:
            raise ValueError("Binary classification requires y_pred_proba")
        if y_pred_proba.ndim != 2 or y_pred_proba.shape[1] != 2:
            raise ValueError(
                "Binary y_pred_proba must be 2D with 2 columns "
                "(P(class=0), P(class=1))", )
        y_prob = y_pred_proba[:, 1]
        auc = float(roc_auc_score(y_true, y_prob))
        ap = float(average_precision_score(y_true, y_prob))
        best_f1 = 0.0
        for thres in np.linspace(0.05, 0.95, 19):
            y_hat = np.zeros_like(y_prob, dtype=int)
            y_hat[y_prob > thres] = 1
            f1 = float(f1_score(y_true, y_hat, average="macro"))
            if f1 > best_f1:
                best_f1 = f1
        return {"roc_auc": auc, "f1": best_f1, "ap": ap}

    if task_type == TaskType.MULTICLASS_CLASSIFICATION:
        if y_pred_proba is None and y_pred is None:
            raise ValueError(
                "Multiclass classification requires y_pred_proba or y_pred", )

        y_true_int = y_true.astype(int)
        n = y_true_int.shape[0]
        if n == 0:
            return {"mrr": 0.0, "acc": 0.0, "f1_macro": 0.0}

        # If probability matrix is available, use ranking to compute MRR.
        if y_pred_proba is not None:
            if y_pred_proba.ndim != 2:
                raise ValueError("y_pred_proba must be 2D for multiclass")
            if y_true.shape[0] != y_pred_proba.shape[0]:
                raise ValueError(
                    "y_true and y_pred_proba must have same length", )
            ranks = np.argsort(-y_pred_proba, axis=1)
            y_pred_int = np.argmax(y_pred_proba, axis=1).astype(int)
            mrr_total = 0.0
            for i in range(n):
                row = ranks[i]
                true_class = y_true_int[i]
                matches = np.where(row == true_class)[0]
                if matches.size == 0:
                    continue
                rank = int(matches[0]) + 1
                mrr_total += 1.0 / rank
        else:
            # Fall back to top-1 only: MRR reduces to accuracy.
            if y_pred is None:
                raise ValueError(
                    "Multiclass classification requires y_pred when "
                    "y_pred_proba is None", )
            y_pred_int = np.asarray(y_pred, dtype=int)
            correct = (y_pred_int == y_true_int)
            mrr_total = float(correct.sum())

        mrr = float(mrr_total / n)
        acc = float(accuracy_score(y_true_int, y_pred_int))
        f1_macro = float(f1_score(y_true_int, y_pred_int, average="macro"), )
        return {"mrr": mrr, "acc": acc, "f1_macro": f1_macro}

    if task_type == TaskType.REGRESSION:
        if y_pred is None:
            raise ValueError("Regression requires y_pred")
        mse = float(mean_squared_error(y_true, y_pred))
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    raise ValueError(f"Unsupported task type for metrics: {task_type}")
