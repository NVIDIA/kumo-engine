# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from kumoapi.task import TaskType

from examples.metrics_utils import compute_metrics as _compute_metrics_shared

if TYPE_CHECKING:
    from examples.paper.datasets.benchmarking_dataset import ForecastingTables

CRPS_QUANTILE_LEVELS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                                           0.8, 0.9)
QUANTILE_COLUMNS: tuple[str, ...] = tuple(f"q{level:g}"
                                          for level in CRPS_QUANTILE_LEVELS)

# Metric-name constants shared across benchmark arms.
METRIC_AUROC = "auroc"
METRIC_AP = "ap"
METRIC_F1 = "f1"  # macro F1 for classification tasks
METRIC_MAE = "mae"


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray | None,
    y_pred: np.ndarray | None,
    task_type: TaskType,
) -> dict[str, float]:
    """Compute core binary, multiclass, and regression metrics."""
    return _compute_metrics_shared(
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        y_pred=y_pred,
        task_type=task_type,
    )


def has_forecast_context(tables: ForecastingTables) -> bool:
    return tables.context is not None and len(tables.context) > 0


def predict_context_frame(tables: ForecastingTables) -> pd.DataFrame:
    """Fit rows plus optional known context (e.g. y9c val year)."""
    train = tables.train.sort_values([tables.entity_col, tables.time_col])
    if not has_forecast_context(tables):
        return train
    context = tables.context.sort_values([tables.entity_col, tables.time_col])
    return pd.concat([train, context], ignore_index=True)


def seasonal_naive_scales_per_entity(
    tables: ForecastingTables,
    *,
    seasonal_period: int,
) -> dict[str, float]:
    """Per-entity in-sample mean |y_t - y_{t-m}| (Hyndman MASE denominator)."""
    m = seasonal_period
    target_col = tables.target_col
    entity_col = tables.entity_col
    scales: dict[str, float] = {}
    history = predict_context_frame(tables)[[
        entity_col,
        target_col,
    ]].sort_values(entity_col)
    for entity, series in history.groupby(entity_col):
        y = series[target_col].to_numpy(dtype=float)
        if len(y) > m:
            scales[str(entity)] = float(np.mean(np.abs(y[m:] - y[:-m])))
    if not scales:
        raise ValueError(
            f"No in-sample seasonal-naive errors (need >{m} history points).")
    return scales


def macro_mase(
    predictions: pd.DataFrame,
    tables: ForecastingTables,
    *,
    seasonal_period: int,
    scales: dict[str, float] | None = None,
) -> float:
    """Macro MASE (mean per-entity MAE / seasonal-naive scale)."""
    if scales is None:
        scales = seasonal_naive_scales_per_entity(
            tables, seasonal_period=seasonal_period)
    entity_col, time_col = tables.entity_col, tables.time_col
    merged = tables.test.merge(predictions, on=[entity_col, time_col],
                               how="inner")
    mase_values: list[float] = []
    for entity, group in merged.groupby(merged[entity_col].astype(str)):
        scale = scales.get(entity, 0.0)
        if scale <= 0:
            continue
        y_true = group[tables.target_col].to_numpy(dtype=float)
        y_pred = group["prediction"].to_numpy(dtype=float)
        mase_values.append(float(np.mean(np.abs(y_true - y_pred))) / scale)
    if not mase_values:
        raise ValueError("No entities with valid seasonal-naive scale.")
    return float(np.mean(mase_values))


def macro_crps(
    predictions: pd.DataFrame,
    tables: ForecastingTables,
    *,
    seasonal_period: int,
    scales: dict[str, float] | None = None,
) -> tuple[float, float]:
    """Macro CRPS (raw, and scaled by the MASE seasonal-naive baseline),
    approximated as twice the mean pinball loss over the deciles. Returns
    ``(nan, nan)`` for point-only models that carry no quantile columns.
    """
    if not all(col in predictions.columns for col in QUANTILE_COLUMNS):
        return float("nan"), float("nan")
    if scales is None:
        scales = seasonal_naive_scales_per_entity(
            tables, seasonal_period=seasonal_period)
    entity_col, time_col = tables.entity_col, tables.time_col
    merged = tables.test.merge(predictions, on=[entity_col, time_col],
                               how="inner")
    levels = np.asarray(CRPS_QUANTILE_LEVELS)
    raw_values: list[float] = []
    scaled_values: list[float] = []
    for entity, group in merged.groupby(merged[entity_col].astype(str)):
        scale = scales.get(entity, 0.0)
        if scale <= 0:
            continue
        y_true = group[tables.target_col].to_numpy(dtype=float)[:, None]
        quantiles = group[list(QUANTILE_COLUMNS)].to_numpy(dtype=float)
        error = y_true - quantiles
        # Pinball loss per (obs, level); CRPS = 2 * mean over both axes.
        pinball = (levels[None, :] - (y_true < quantiles)) * error
        entity_crps = 2.0 * float(pinball.mean())
        raw_values.append(entity_crps)
        scaled_values.append(entity_crps / scale)
    return float(np.mean(raw_values)), float(np.mean(scaled_values))
