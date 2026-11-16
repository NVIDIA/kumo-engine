# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Abstract base class for benchmarking datasets.

Subclasses implement get_enterprise_graph, get_pquery,
get_entity_tables, and get_joined_tables so that generic benchmarks
(Enterprise, single-table, forecasting) run over multiple datasets.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import numpy as np
import pandas as pd
from kumoapi.task import TaskType

from examples.paper.metrics import (
    CRPS_QUANTILE_LEVELS,
    QUANTILE_COLUMNS,
    compute_metrics,
    macro_crps,
    macro_mase,
    seasonal_naive_scales_per_entity,
)
from kumoai_studio.graph import Graph as EnterpriseGraph

# Supervised table target column name used across paper benchmarks
TARGET = "TARGET"


@dataclass(frozen=True)
class SupervisedTables:
    """Train/test splits and metadata for single-table benchmarks."""
    train: pd.DataFrame
    test: pd.DataFrame
    target_col: str
    task_type: TaskType
    entity_col: str | None = None


@dataclass(frozen=True)
class ForecastingTables:
    """Train/context/test splits for entity-level forecasting.

    train: rows used to fit (e.g. through 2023 for y9c).
    context: optional known history between train and test (e.g. 2024 val
        year).
    test: holdout rows to score (e.g. 2025 for y9c).

    test must contain point-in-time observations at freq spacing; aggregate-
    window targets (e.g. M5-style) need a separate evaluation path.

    seasonal_period: season length *m* in units of ``freq`` (MASE denominator).
    feature_preset: calendar features for TabPFN / Chronos / TimesFM.
    lag_timesteps: optional forecasting autoregressive lag count in units of
        ``freq``.
    """
    train: pd.DataFrame
    test: pd.DataFrame
    entity_col: str
    time_col: str
    target_col: str
    freq: str
    seasonal_period: int  # m for MASE, in units of freq
    feature_preset: Literal["default", "calendar"] = "default"
    lag_timesteps: int | None = None
    context: pd.DataFrame | None = None


def forecast_benchmark_entity_ids(
    tables: ForecastingTables,
    max_entities: int | None = None,
) -> list[str]:
    """First N unique entity ids from test rows (insertion order)."""
    entity_col = tables.entity_col
    ids = tables.test[entity_col].astype(str).drop_duplicates()
    if max_entities is None:
        return ids.tolist()
    if max_entities <= 0:
        raise ValueError("max_entities must be positive.")
    return ids.iloc[:max_entities].tolist()


def limit_entities(
    tables: ForecastingTables,
    max_entities: int | None = None,
) -> ForecastingTables:
    """Subset train/context/test to capped entity ids (test row order)."""
    if max_entities is None:
        return tables
    if max_entities <= 0:
        raise ValueError("max_entities must be positive.")
    entity_col = tables.entity_col
    keep = set(forecast_benchmark_entity_ids(tables, max_entities))
    train = tables.train[tables.train[entity_col].isin(keep)].reset_index(
        drop=True)
    test = tables.test[tables.test[entity_col].isin(keep)].reset_index(
        drop=True)
    context = tables.context
    if context is not None and len(context) > 0:
        context = context[context[entity_col].isin(keep)].reset_index(
            drop=True)
    return ForecastingTables(
        train=train,
        test=test,
        entity_col=entity_col,
        time_col=tables.time_col,
        target_col=tables.target_col,
        freq=tables.freq,
        seasonal_period=tables.seasonal_period,
        feature_preset=tables.feature_preset,
        lag_timesteps=tables.lag_timesteps,
        context=context,
    )


def prediction_frame_from_quantiles(
    df: pd.DataFrame,
    *,
    output_selection: str | None = None,
    point_col: str | None = "TARGET_PRED",
) -> pd.DataFrame:
    """Extract point forecast plus decile columns from q_<level> columns."""
    assert output_selection in (None, "mean", "median")
    source = [f"q_{level:g}" for level in CRPS_QUANTILE_LEVELS]
    if not all(col in df.columns for col in source):
        if output_selection == "mean":
            raise ValueError(
                "output_selection='mean' requires quantile predictions "
                "(q_<level> columns).")
        if point_col is None:
            missing = [col for col in source if col not in df]
            raise ValueError(f"Missing quantile prediction columns: {missing}")
        return pd.DataFrame({"prediction": df[point_col].astype(float)},
                            index=df.index)

    deciles = df[source].to_numpy(dtype=float)
    if output_selection == "mean":
        point = deciles.mean(axis=1)
    elif point_col is not None:
        point = df[point_col].astype(float).to_numpy()
    else:
        point = df["q_0.5"].astype(float).to_numpy()
    return pd.DataFrame(
        {
            "prediction": point,
            **{
                col: deciles[:, i]
                for i, col in enumerate(QUANTILE_COLUMNS)
            },
        }, index=df.index)


class BenchmarkingDataset(ABC):
    """Abstract dataset interface used by all paper benchmarks.

    Subclass this to add a new dataset. Implement every abstract
    method, then register the subclass in
    ``examples/paper/datasets/__init__.py``.
    """
    def _set_data_root(
        self,
        data_root: Union[str, Path, None],
        default: str,
    ) -> None:
        """Set _data_root_str and _data_root from data_root or default.

        S3 URIs are kept as strings (Path() can drop slashes and break the URI)
        """
        self._data_root_str = (str(data_root).rstrip("/")
                               if data_root else default.rstrip("/"))
        self._data_root = (None if self._data_root_str.startswith("s3://") else
                           Path(self._data_root_str))

    def _read_parquet(self, name: str) -> pd.DataFrame:
        """Read a parquet file from the dataset data root (local or S3).

        Requires ``_set_data_root`` to have been called in
        ``__init__``.
        """
        data_root_str = getattr(self, "_data_root_str", "")
        data_root = getattr(self, "_data_root", None)
        if data_root_str.startswith("s3://"):
            return pd.read_parquet(f"{data_root_str}/{name}")
        if data_root is None:
            data_root = Path(data_root_str)
        return pd.read_parquet(Path(data_root) / name)

    # -- Single-table benchmarks ----------------------------------

    @abstractmethod
    def get_entity_tables(self) -> SupervisedTables:
        """Return train/test splits from the primary table only."""
        ...

    @abstractmethod
    def get_joined_tables(self) -> SupervisedTables:
        """Return train/test splits from the primary table
        enriched with features from related tables.

        Used by ``benchmark_single_table.py --table-type joined``.

        Same as ``get_entity_tables`` but merge additional
        columns from related tables (e.g. account attributes)
        before splitting.
        """
        ...

    # -- Forecasting benchmarks ----------------------------------

    def get_forecasting_tables(self) -> ForecastingTables:
        """Return train/test splits for entity-level forecasting.

        Datasets that are not temporal regression tasks can leave this
        unimplemented. Used by ``benchmark_forecasting.py``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support forecasting benchmarks.")

    def get_forecasting_labels(
        self,
        predictions: pd.DataFrame,
        tables: "ForecastingTables",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (y_true, y_pred) aligned for metric computation.

        Default: point-in-time — inner join predictions to test on
        (entity_col, time_col).  Override for aggregate-window targets
        (e.g. M5-style monthly sums) where labels must be derived by
        reducing fine-grained predictions rather than joined row-by-row.
        """
        merged = tables.test.merge(
            predictions,
            on=[tables.entity_col, tables.time_col],
        )
        return (
            merged[tables.target_col].to_numpy(dtype=float),
            merged["prediction"].to_numpy(dtype=float),
        )

    def evaluate_forecasting_predictions(
        self,
        predictions: pd.DataFrame,
        tables: "ForecastingTables",
    ) -> dict[str, float]:
        """Regression metrics, macro MASE, and (for probabilistic models)
        macro CRPS scaled by the same seasonal-naive baseline as MASE.
        """
        y_true, y_pred = self.get_forecasting_labels(predictions, tables)
        metrics = compute_metrics(y_true, None, y_pred, TaskType.REGRESSION)
        scales = seasonal_naive_scales_per_entity(
            tables, seasonal_period=tables.seasonal_period)
        metrics["mase"] = macro_mase(predictions, tables,
                                     seasonal_period=tables.seasonal_period,
                                     scales=scales)
        metrics["crps"], metrics["crps_scaled"] = macro_crps(
            predictions, tables, seasonal_period=tables.seasonal_period,
            scales=scales)
        return metrics

    def enterprise_holdout_predictions(
        self,
        holdout: pd.DataFrame,
        tables: "ForecastingTables",
        *,
        output_selection: str | None = None,
    ) -> pd.DataFrame:
        """Convert Enterprise holdout rows into forecasting predictions.

        Multi-quantile models emit ``q_<level>`` columns; the deciles are
        exposed for CRPS and, when ``output_selection='mean'``, integrated
        into the point forecast (otherwise the median ``TARGET_PRED``).
        """
        quantiles = prediction_frame_from_quantiles(
            holdout, output_selection=output_selection)
        out = pd.DataFrame({
            tables.entity_col:
            holdout["ENTITY"].astype(str),
            tables.time_col:
            pd.to_datetime(holdout["TIMESTAMP"]),
        })
        return pd.concat([out, quantiles], axis=1)

    def supports_forecast_holdout(self) -> bool:
        """True when this dataset implements get_forecasting_tables."""
        return (type(self).get_forecasting_tables
                is not BenchmarkingDataset.get_forecasting_tables)

    # -- Enterprise benchmark -------------------------------------

    @abstractmethod
    def get_enterprise_graph(self) -> "EnterpriseGraph":
        """Build and return a ``kumoai_studio.Graph`` for Enterprise.

        Used by ``benchmark_enterprise.py``.
        """
        ...

    @abstractmethod
    def get_pquery(self) -> str:
        """Predictive query string for training."""
        ...

    def get_enterprise_split(self) -> str | None:
        """Optional TimeRangeSplit (or other split) string for enterprise
        training table.

        Return None to use the default split.
        Override to enforce a dataset-specific train/val/test split
        E.g. temporal boundary matching the task table.
        """
        return None

    def configure_train_table_plan(self, plan) -> None:
        """Optionally mutate the TrainingTableGenerationPlan in-place.

        Override to apply dataset-specific settings (e.g. timeframe_step)
        that cannot be expressed via get_enterprise_split alone.
        Default: no-op.
        """

    def configure_model_plan(self, model_plan) -> None:
        """Optionally mutate the ModelPlan in-place before training.

        Override to apply dataset-specific hyperparameter search space
        adjustments (e.g. dropout, max_epochs, num_neighbors, hops).
        Default: no-op.
        """
