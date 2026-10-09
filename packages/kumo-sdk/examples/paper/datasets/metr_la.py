# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""METR-LA: traffic speed forecasting on the Los Angeles highway network.

Dataset:
  s3://kumo-public-datasets/metr_la/
    - sensors.parquet     – 207 loop-detector sensors (sensor_id, sensor_index)
    - observations.parquet – speed_mph every 5 minutes per sensor
                             (timestamp, sensor_id, speed_mph)
    - edges.parquet        – road-network adjacency (from_sensor_id,
                             to_sensor_id, weight)

Task:
  - Regression: predict speed_mph for each sensor.
  - Temporal 70/10/20 split (train / val / test) based on observation
    timestamps.

Enterprise and single-table benchmarks share the same 49-day timestamp
window and test boundary. Single-table folds the Enterprise train and
validation periods into one training split.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import pandas as pd
from kumoapi.model_plan import HopConfig
from kumoapi.task import TaskType

import kumoai_studio as kumo

from .benchmarking_dataset import (
    BenchmarkingDataset,
    ForecastingTables,
    SupervisedTables,
)

DEFAULT_DATA_ROOT = "s3://kumo-public-datasets/metr_la"

SENSORS_TABLE = "sensors"
OBSERVATIONS_TABLE = "observations"
EDGES_TABLE = "edges"

SENSOR_ID = "sensor_id"
FROM_SENSOR_ID = "from_sensor_id"
TO_SENSOR_ID = "to_sensor_id"
TIMESTAMP = "timestamp"
SPEED_MPH = "speed_mph"


class MetrLABenchmarkingDataset(BenchmarkingDataset):
    """METR-LA traffic speed forecasting benchmark."""
    def __init__(
        self,
        data_root: Union[str, Path, None] = None,
    ) -> None:
        """Args:
        data_root: directory or S3 URI containing sensors.parquet and
            observations.parquet.  Defaults to the public S3 bucket.
        """
        self._set_data_root(data_root, DEFAULT_DATA_ROOT)

    # -- data loading -------------------------------------------------

    def _load_sensors(self) -> pd.DataFrame:
        return self._read_parquet("sensors.parquet")

    def _load_observations(self) -> pd.DataFrame:
        return self._read_parquet("observations.parquet")

    def _benchmark_time_bounds(
        self,
        df: pd.DataFrame | None = None,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        if df is None:
            df = self._load_observations()
        test_end = df[TIMESTAMP].max() + pd.Timedelta(minutes=5)
        dataset_start = test_end - pd.Timedelta(days=49)
        return dataset_start, test_end

    def _split_boundaries(
        self,
    ) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
        dataset_start, test_end = self._benchmark_time_bounds()
        total = test_end - dataset_start
        train_end = (dataset_start + total * 0.70).ceil("5min")
        val_end = (dataset_start + total * 0.80).ceil("5min")
        return dataset_start, train_end, val_end, test_end

    # -- Single-table benchmarks --------------------------------------

    def _single_table_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split aligned with the enterprise 70/10/20 boundaries.

        train = enterprise train + val periods  (days  0–39.2, 80%)
        test  = enterprise test period          (days 39.2–49, 20%)

        Baselines do not have a separate validation mechanism, so the
        enterprise val period is folded into training.

        Args:
            df: Dataframe of examples.
            train_ratio: Size of train+val set, if not 80%.
        """
        dataset_start, _, default_val_end, test_end = self._split_boundaries()
        if train_ratio == 0.8:
            val_end = default_val_end
        else:
            val_end = (dataset_start +
                       (test_end - dataset_start) * train_ratio).ceil("5min")

        train = df[(df[TIMESTAMP] >= dataset_start)
                   & (df[TIMESTAMP] < val_end)].reset_index(drop=True)
        test = df[(df[TIMESTAMP] >= val_end)
                  & (df[TIMESTAMP] < test_end)].reset_index(drop=True)
        return train, test

    def get_entity_tables(self, train_ratio: float = 0.8) -> SupervisedTables:
        """Observations-only regression: predict speed_mph from timestamp
        and sensor_id.  Train/test split matches the enterprise window.

        Args:
            train_ratio: Size of train+val set. (default: 0.8)
        """
        obs = self._load_observations()
        obs[SENSOR_ID] = obs[SENSOR_ID].astype(str)
        obs = obs.rename(columns={SPEED_MPH: "TARGET"})
        train, test = self._single_table_split(obs, train_ratio)
        return SupervisedTables(
            train=train,
            test=test,
            target_col="TARGET",
            task_type=TaskType.REGRESSION,
            entity_col=SENSOR_ID,
        )

    def get_joined_tables(self, train_ratio: float = 0.8) -> SupervisedTables:
        """Observations joined with sensors (adds sensor_index feature).
        Train/test split matches the enterprise window.

        Args:
            train_ratio: Size of train+val set. (default: 0.8)
        """
        obs = self._load_observations()
        sensors = self._load_sensors()
        joined = obs.merge(sensors, on=SENSOR_ID, how="left")
        joined[SENSOR_ID] = joined[SENSOR_ID].astype(str)
        joined["sensor_index"] = joined["sensor_index"].astype(str)
        joined = joined.rename(columns={SPEED_MPH: "TARGET"})
        train, test = self._single_table_split(joined, train_ratio)
        return SupervisedTables(
            train=train,
            test=test,
            target_col="TARGET",
            task_type=TaskType.REGRESSION,
            entity_col=SENSOR_ID,
        )

    # -- Forecasting benchmarks ---------------------------------------

    def get_forecasting_tables(self) -> ForecastingTables:
        sup_tables = self.get_entity_tables()
        return ForecastingTables(
            train=sup_tables.train,
            test=sup_tables.test,
            entity_col=SENSOR_ID,
            time_col=TIMESTAMP,
            target_col=sup_tables.target_col,
            freq="5min",
            seasonal_period=288,  # 288 × 5min = 24h
            feature_preset="calendar",
            lag_timesteps=12,  # One hour of 5-minute history.
        )

    # -- Enterprise benchmark ----------------------------------------

    def get_enterprise_graph(self, *,
                             history_only: bool = False) -> kumo.Graph:
        """Build a kumoai_studio.Graph with sensors, observations, and edges tables.

        Uses S3Connector pointing directly at the public dataset bucket,
        avoiding the need to re-upload files.
        """
        connector = kumo.S3Connector(
            name="metr_la_benchmark_source",
            identity=os.environ["KUMO_S3_IDENTITY_ID"],
            root_dir=self._data_root_str,
        )

        sensors_src = connector.table(SENSORS_TABLE)
        observations_src = connector.table(OBSERVATIONS_TABLE)

        sensor_columns = [
            kumo.Column(SENSOR_ID, stype="ID", dtype="string"),
        ]
        if not history_only:
            sensor_columns.append(
                kumo.Column("sensor_index", stype="numerical", dtype="int"))
        sensors_tbl = kumo.Table(
            sensors_src,
            columns=sensor_columns,
            primary_key=SENSOR_ID,
        )

        observations_tbl = kumo.Table(
            observations_src,
            columns=[
                kumo.Column(TIMESTAMP, stype="timestamp", dtype="date"),
                kumo.Column(SENSOR_ID, stype="ID", dtype="string"),
                kumo.Column(SPEED_MPH, stype="numerical", dtype="float"),
            ],
            time_column=TIMESTAMP,
        )

        tables = {
            SENSORS_TABLE: sensors_tbl,
            OBSERVATIONS_TABLE: observations_tbl,
        }
        edges = [
            dict(
                src_table=OBSERVATIONS_TABLE,
                fkey=SENSOR_ID,
                dst_table=SENSORS_TABLE,
            ),
        ]
        if not history_only:
            edges_src = connector.table(EDGES_TABLE)
            edges_tbl = kumo.Table(
                edges_src,
                columns=[
                    kumo.Column(FROM_SENSOR_ID, stype="ID", dtype="string"),
                    kumo.Column(TO_SENSOR_ID, stype="ID", dtype="string"),
                    kumo.Column("weight", stype="numerical", dtype="float"),
                ],
            )
            tables[EDGES_TABLE] = edges_tbl
            edges.extend([
                dict(
                    src_table=EDGES_TABLE,
                    fkey=FROM_SENSOR_ID,
                    dst_table=SENSORS_TABLE,
                ),
                dict(
                    src_table=EDGES_TABLE,
                    fkey=TO_SENSOR_ID,
                    dst_table=SENSORS_TABLE,
                ),
            ])

        return kumo.Graph(
            tables=tables,
            edges=edges,
        )

    def get_pquery(self) -> str:
        """Forecasting pquery for speed_mph per sensor.

        Enterprise forecasting supports SUM, COUNT, COUNT_DISTINCT only.
        Since there is exactly one speed_mph reading per sensor per
        5-minute window, SUM over [0, 5) minutes equals the reading value.
        """
        return self.get_forecast_query(num_timeframes=12)

    def get_forecast_query(self, *, num_timeframes: int) -> str:
        return (f"PREDICT SUM({OBSERVATIONS_TABLE}.{SPEED_MPH}, 0, 5, minutes)"
                f" FORECAST {num_timeframes} TIMEFRAMES\n"
                f"FOR EACH {SENSORS_TABLE}.{SENSOR_ID}")

    def configure_model_plan(self, model_plan) -> None:
        """Tune the model plan for METR-LA traffic forecasting.

        Changes vs. defaults:
        - max_epochs: 8 → 25   (models were peaking at epoch 0; need more room)
        - early_stopping patience: 3 → 5
        - dropout: [0.0] → [0.0, 0.1, 0.2]  (reduce overfitting)
        - num_neighbors: 2 hops, hop1 default 32 (obs stays inferred 512)
        """
        # Epochs / early stopping
        model_plan.optimization.max_epochs = 25
        for es in model_plan.optimization.early_stopping:
            es.patience = 5

        # 2 hops; hop1 keeps the inferred 512 for obs→sensors
        nn = model_plan.neighbor_sampling.num_neighbors[0]
        nn.hop1.default = 32  # road-graph edges in hop1; obs stays inferred
        nn.hop2 = HopConfig(default=32)

        # Dropout search space
        for m in model_plan.model_architecture.model:
            m.dropout = [0.0, 0.1, 0.2]

    def get_enterprise_split(self) -> str:
        """Exact TimeRangeSplit 70/10/20 over the last 49 days.

        The full dataset spans ~118 days at 5-minute resolution (~34 272
        anchors). Computing histogram stats (up to 100 bins/anchor) produces
        ~3.4 M rows which exceeds the 1 M-row Spark-to-Pandas limit.

        MAX_PANDAS_SIZE(1M) / MAX_NUM_BINS(100) = 10 000 max training anchors.
        With a 70% training fraction that means ≤ 49.6 total days.  We use a
        49-day window: training ≈ 34 days = 9 792 anchors -> ~979 200 rows.
        Requires ERROR_ON_NUM_TIME_INTERVALS >= 14 113 on the server.

        Boundaries include time-of-day and intentionally match the tabular
        benchmark split exactly.
        """
        dataset_start, train_end, val_end, test_end = self._split_boundaries()

        def _fmt(ts: pd.Timestamp) -> str:
            return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

        return (f'TimeRangeSplit(['
                f'("{_fmt(dataset_start)}", "{_fmt(train_end)}"), '
                f'("{_fmt(train_end)}", "{_fmt(val_end)}"), '
                f'("{_fmt(val_end)}", "{_fmt(test_end)}")])')
