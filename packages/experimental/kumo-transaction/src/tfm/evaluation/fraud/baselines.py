# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw / embedding / combined binary probes (logistic or optional XGBoost)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from tfm.evaluation.fraud.metrics import classification_metrics
from tfm.inference.embedding import EmbeddingTable


def _fit_predict_logistic(
    x_train: NDArray[np.floating],
    y_train: NDArray[np.floating],
    x_eval: NDArray[np.floating],
    seed: int,
) -> NDArray[np.float64]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_train)
    x_ev = scaler.transform(x_eval)
    if np.unique(y_train).size < 2:
        return np.full(len(x_eval), float(y_train.mean() if len(y_train) else 0.0))
    clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
    clf.fit(x_tr, y_train.astype(int))
    return clf.predict_proba(x_ev)[:, 1]


def _fit_predict_xgboost(
    x_train: NDArray[np.floating],
    y_train: NDArray[np.floating],
    x_eval: NDArray[np.floating],
    seed: int,
) -> NDArray[np.float64]:
    from tfm.evaluation.tabular import xgboost_classifier

    if np.unique(y_train).size < 2:
        return np.full(len(x_eval), float(y_train.mean() if len(y_train) else 0.0))
    n_pos = float((y_train > 0.5).sum())
    n_neg = float((y_train <= 0.5).sum())
    params = {"max_depth": 4, "learning_rate": 0.1, "scale_pos_weight": n_neg / max(n_pos, 1.0)}
    clf = xgboost_classifier(params, n_estimators=50, seed=seed, n_jobs=1)
    clf.fit(np.asarray(x_train), y_train.astype(int))
    return np.asarray(clf.predict_proba(np.asarray(x_eval))[:, 1], dtype=np.float64)


def fit_predict_probe(
    x_train: NDArray[np.floating],
    y_train: NDArray[np.floating],
    x_eval: NDArray[np.floating],
    *,
    seed: int,
    head: str = "logistic_regression",
) -> NDArray[np.float64]:
    """Fit one binary probe on one feature block and score `x_eval`.

    `head` is `logistic_regression` or `xgboost`. A single-class training label
    vector yields the constant base rate rather than an error, so degenerate
    smoke slices stay runnable. Use `evaluate_binary_probes` for the full
    raw / embedding / combined comparison.
    """
    name = head.strip().lower()
    if name in {"xgboost", "xgb"}:
        return _fit_predict_xgboost(x_train, y_train, x_eval, seed)
    if name not in {"logistic_regression", "logistic"}:
        from tfm.contracts.errors import UnsupportedCombinationError

        raise UnsupportedCombinationError(
            f"Unknown probe head {head!r}; use 'logistic_regression' or 'xgboost'."
        )
    return _fit_predict_logistic(x_train, y_train, x_eval, seed)


def evaluate_binary_probes(
    train: EmbeddingTable,
    eval_table: EmbeddingTable,
    *,
    seed: int,
    k: int = 5,
    y_train: NDArray[np.floating] | None = None,
    y_eval: NDArray[np.floating] | None = None,
    prefix: str = "",
    head: str = "logistic_regression",
    arms: Sequence[str] = ("raw", "embedding", "combined"),
) -> dict[str, dict[str, float]]:
    """Fit the requested raw / embedding / combined probes. Keys are optionally prefixed."""
    unknown = sorted(set(arms) - {"raw", "embedding", "combined"})
    if unknown:
        raise ValueError(f"Unknown probe arms {unknown}.")
    labels_tr = train.fraud_labels if y_train is None else np.asarray(y_train, dtype=np.float32)
    labels_ev = eval_table.fraud_labels if y_eval is None else np.asarray(y_eval, dtype=np.float32)
    stem = f"{prefix}_" if prefix else ""
    inputs = {
        "raw": (train.numeric_features, eval_table.numeric_features),
        "embedding": (train.vectors, eval_table.vectors),
        "combined": (
            np.concatenate([train.numeric_features, train.vectors], axis=1),
            np.concatenate([eval_table.numeric_features, eval_table.vectors], axis=1),
        ),
    }
    reports: dict[str, dict[str, float]] = {}
    for arm in arms:
        x_train, x_eval = inputs[arm]
        scores = fit_predict_probe(x_train, labels_tr, x_eval, seed=seed, head=head)
        reports[f"{stem}{arm}"] = classification_metrics(labels_ev, scores, k=k)
    return reports
