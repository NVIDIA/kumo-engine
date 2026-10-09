# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Y9C deposit-yield forecasting benchmark.

S3 root: s3://kumo-public-datasets/y9c/remapped_data/v1/kumo_graph_tables/
graph_tables_y9c_deposits_2000/

Unified split: fit through 2023-12-31; 2024 is context (known at predict time);
2025 is the test holdout. Enterprise additionally uses 2024 as a validation
fold during training.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Union, cast

import pandas as pd
from kumoapi.task import TaskType

import kumoai_studio as kumo

from .benchmarking_dataset import (
    TARGET,
    BenchmarkingDataset,
    ForecastingTables,
    SupervisedTables,
    prediction_frame_from_quantiles,
)

DEFAULT_DATA_ROOT = (
    "s3://kumo-public-datasets/y9c/remapped_data/v1/kumo_graph_tables/"
    "graph_tables_y9c_deposits_2000")

INSTITUTIONS_TABLE = "institutions"
DEPOSITS_TABLE = "deposits"

TEMPORAL_TABLES = (
    "balance_sheet",
    "income_statement",
    "asset_quality",
    "capital_ratios",
    DEPOSITS_TABLE,
)

SOURCE_TABLES = (INSTITUTIONS_TABLE, *TEMPORAL_TABLES)

RSSD_ID = "rssd_id"
DATE_COL = "reporting_date_cc_yr_mo_da"
ENTERPRISE_TIME_COL = "enterprise_forecast_time"
TARGET_COL = "yield_total_deposits"

TRAIN_END = pd.Timestamp("2024-01-01")
HOLDOUT_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2026-01-01")

# Enterprise forecasting requires uniformly spaced timestamps for lag features.
# Y9C reports at calendar quarter-ends, so Enterprise uses the same fixed
# quarter abstraction as the benchmark FORECAST query.
ENTERPRISE_TIME_ANCHOR = pd.Timestamp("2000-03-31")
Y9C_TIMEFRAME_DAYS = 91
ENTERPRISE_VAL_START = pd.Timestamp("2022-01-01")

INSTITUTION_FEATURE_COLS = (
    RSSD_ID,
    "abbreviated_state_name",
    "physical_state_code",
    "home_state",
    "bank_count",
    "organization_type",
    "charter_type",
    "financial_holding_company_indicator",
)


class Y9CBenchmarkingDataset(BenchmarkingDataset):
    task_type = TaskType.REGRESSION
    target_col = TARGET_COL

    def __init__(
        self,
        data_root: Union[str, Path, None] = None,
    ) -> None:
        self._set_data_root(data_root, DEFAULT_DATA_ROOT)

    def _load_table(self, name: str) -> pd.DataFrame:
        return self._read_parquet(f"{name}.parquet")

    def _load_deposits(self, *, parse_dates: bool = True) -> pd.DataFrame:
        deposits = self._load_table(DEPOSITS_TABLE)
        if parse_dates:
            deposits = self._parse_temporal_table(deposits)
        return deposits

    def _load_institutions(self, *, compact: bool = False) -> pd.DataFrame:
        institutions = self._load_table(INSTITUTIONS_TABLE)
        if compact:
            cols = [c for c in INSTITUTION_FEATURE_COLS if c in institutions]
            institutions = institutions[cols].copy()
        else:
            institutions = institutions.drop(
                columns=[
                    "legal_name", "entity_short_name", "city_town_text_name"
                ],
                errors="ignore",
            )
        institutions[RSSD_ID] = institutions[RSSD_ID].astype(str)
        return institutions

    @staticmethod
    def _parse_temporal_table(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[RSSD_ID] = df[RSSD_ID].astype(str)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL].astype(str),
                                      format="%Y%m%d")
        return df

    @staticmethod
    def _enterprise_time_for_index(index: int) -> pd.Timestamp:
        return cast(
            pd.Timestamp,
            ENTERPRISE_TIME_ANCHOR +
            pd.Timedelta(days=Y9C_TIMEFRAME_DAYS * index),
        )

    @classmethod
    def _enterprise_time_map(cls, dates) -> dict[pd.Timestamp, pd.Timestamp]:
        dates = sorted(pd.to_datetime(dates).dropna().unique())
        return {
            date: cls._enterprise_time_for_index(i)
            for i, date in enumerate(dates)
        }

    @classmethod
    def _add_enterprise_time(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        mapping = cls._enterprise_time_map(df[DATE_COL])
        df[ENTERPRISE_TIME_COL] = pd.to_datetime(df[DATE_COL]).map(mapping)
        return df

    def _eligible_forecast_entities(self, deposits: pd.DataFrame) -> list[str]:
        deposits = deposits.dropna(subset=[self.target_col])
        test_dates = sorted(deposits.loc[
            (deposits[DATE_COL] >= HOLDOUT_START)
            & (deposits[DATE_COL] < TEST_END),
            DATE_COL,
        ].unique())
        if not test_dates:
            raise ValueError("Y9C holdout has no target rows.")

        test_counts = (deposits[deposits[DATE_COL].isin(test_dates)].groupby(
            RSSD_ID)[DATE_COL].nunique())
        train_counts = (deposits[deposits[DATE_COL] < TRAIN_END].groupby(
            RSSD_ID)[DATE_COL].nunique())
        required_test = len(test_dates)
        entities = sorted(
            entity for entity, n_test in test_counts.items()
            if n_test == required_test and train_counts.get(entity, 0) >= 8)
        if not entities:
            raise ValueError("Y9C has no entities with a complete holdout.")
        return entities

    def _supervised_frame(
        self,
        *,
        joined: bool = False,
        complete_panel: bool = True,
    ) -> pd.DataFrame:
        deposits = self._load_deposits()
        df = deposits[[RSSD_ID, DATE_COL, self.target_col
                       ]].dropna(subset=[self.target_col]).copy()
        if complete_panel:
            # Eligibility uses absolute yield (complete train/test panel).
            abs_panel = deposits[[RSSD_ID, DATE_COL, self.target_col
                                  ]].dropna(subset=[self.target_col])
            keep = set(self._eligible_forecast_entities(abs_panel))
            df = df[df[RSSD_ID].isin(keep)]

        if joined:
            institutions = self._load_institutions(compact=True)
            df = df.merge(institutions, on=RSSD_ID, how="left")

        df = df.rename(columns={self.target_col: TARGET})
        return df.sort_values([RSSD_ID, DATE_COL]).reset_index(drop=True)

    @staticmethod
    def _fit_context_test_split(
            df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        fit = df[df[DATE_COL] < TRAIN_END].reset_index(drop=True)
        context = df[(df[DATE_COL] >= TRAIN_END)
                     & (df[DATE_COL] < HOLDOUT_START)].reset_index(drop=True)
        test = df[(df[DATE_COL] >= HOLDOUT_START)
                  & (df[DATE_COL] < TEST_END)].reset_index(drop=True)
        if len(fit) == 0 or len(test) == 0:
            raise ValueError(
                "Y9C split produced empty fit or test set. Check data root.")
        return fit, context, test

    def _entity_temporal_splits(
        self,
        *,
        joined: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self._fit_context_test_split(
            self._supervised_frame(joined=joined))

    def _supervised_tables(
        self,
        *,
        joined: bool = False,
    ) -> SupervisedTables:
        fit, _context, test = self._entity_temporal_splits(joined=joined)
        return SupervisedTables(
            train=fit,
            test=test,
            target_col=TARGET,
            task_type=TaskType.REGRESSION,
            entity_col=RSSD_ID,
        )

    def get_entity_tables(self) -> SupervisedTables:
        return self._supervised_tables(joined=False)

    def get_joined_tables(self) -> SupervisedTables:
        return self._supervised_tables(joined=True)

    def get_forecasting_tables(self) -> ForecastingTables:
        fit, context, test = self._entity_temporal_splits()
        return ForecastingTables(
            train=fit,
            test=test,
            entity_col=RSSD_ID,
            time_col=DATE_COL,
            target_col=TARGET,
            freq="QE",
            seasonal_period=4,  # 4 × QE = 1 year
            lag_timesteps=4,  # One year of quarterly history.
            context=context,
        )

    def _enterprise_source_tables(self):
        deposits = self._load_deposits()
        deposits = self._add_enterprise_time(deposits)
        deposits[RSSD_ID] = pd.to_numeric(deposits[RSSD_ID],
                                          errors="raise").astype("int64")

        if self._data_root_str.startswith("s3://"):
            connector = kumo.S3Connector(
                name="y9c_benchmark_source",
                identity=os.environ["KUMO_S3_IDENTITY_ID"],
                root_dir=self._data_root_str,
            )
            source_tables = {
                name: connector.table(name)
                for name in SOURCE_TABLES
            }
            upload = kumo.FileUploadConnector(file_type="parquet")
            tmpdir = tempfile.TemporaryDirectory()
            path = str(Path(tmpdir.name) / f"{DEPOSITS_TABLE}.parquet")
            deposits.to_parquet(path, index=False)
            upload.upload(name=DEPOSITS_TABLE, path=path)
            source_tables[DEPOSITS_TABLE] = upload.table(DEPOSITS_TABLE)
            self._enterprise_upload_resources = (upload, tmpdir)
            return connector, source_tables

        connector = kumo.FileUploadConnector(file_type="parquet")
        source_tables = {}
        tmpdir = tempfile.TemporaryDirectory()
        self._enterprise_upload_resources = (connector, tmpdir)
        for name in SOURCE_TABLES:
            if name == DEPOSITS_TABLE:
                path = str(Path(tmpdir.name) / f"{name}.parquet")
                deposits.to_parquet(path, index=False)
            else:
                path = str(Path(self._data_root_str) / f"{name}.parquet")
            connector.upload(name=name, path=path)
            source_tables[name] = connector.table(name)
        return connector, source_tables

    def get_enterprise_graph(
        self,
        *,
        history_only: bool = False,
    ) -> kumo.Graph:
        _connector, source_tables = self._enterprise_source_tables()

        institutions_tbl = kumo.Table.from_source_table(
            source_tables[INSTITUTIONS_TABLE],
            column_names=list(INSTITUTION_FEATURE_COLS),
            primary_key=RSSD_ID,
        ).infer_metadata()

        deposits_columns = None
        if history_only:
            deposits_columns = [
                f"{DEPOSITS_TABLE}_id", RSSD_ID, ENTERPRISE_TIME_COL,
                self.target_col
            ]

        deposits_tbl = kumo.Table.from_source_table(
            source_tables[DEPOSITS_TABLE],
            column_names=deposits_columns,
            primary_key=f"{DEPOSITS_TABLE}_id",
            time_column=ENTERPRISE_TIME_COL,
        ).infer_metadata()

        tables = {
            INSTITUTIONS_TABLE: institutions_tbl,
            DEPOSITS_TABLE: deposits_tbl,
        }
        edges = [
            dict(
                src_table=DEPOSITS_TABLE,
                fkey=RSSD_ID,
                dst_table=INSTITUTIONS_TABLE,
            ),
        ]

        if not history_only:
            for table_name in TEMPORAL_TABLES:
                if table_name == DEPOSITS_TABLE:
                    continue
                tables[table_name] = kumo.Table.from_source_table(
                    source_tables[table_name],
                    primary_key=f"{table_name}_id",
                    time_column=DATE_COL,
                ).infer_metadata()
                edges.append(
                    dict(
                        src_table=table_name,
                        fkey=RSSD_ID,
                        dst_table=INSTITUTIONS_TABLE,
                    ))

        return kumo.Graph(tables=tables, edges=edges)

    def configure_train_table_plan(self, plan) -> None:
        plan.timeframe_step = Y9C_TIMEFRAME_DAYS

    def enterprise_holdout_predictions(
        self,
        holdout: pd.DataFrame,
        tables: ForecastingTables,
        *,
        output_selection: str | None = None,
    ) -> pd.DataFrame:
        deposits = self._add_enterprise_time(self._load_deposits())
        date_map = deposits[[DATE_COL, ENTERPRISE_TIME_COL]].drop_duplicates()
        date_map = dict(zip(date_map[ENTERPRISE_TIME_COL], date_map[DATE_COL]))
        quantiles = prediction_frame_from_quantiles(
            holdout, output_selection=output_selection)
        predictions = pd.DataFrame({
            tables.entity_col:
            holdout["ENTITY"].astype(str),
            tables.time_col:
            pd.to_datetime(holdout["TIMESTAMP"]).map(date_map),
        })
        predictions = pd.concat([predictions, quantiles],
                                axis=1).dropna(subset=[tables.time_col])
        test_dates = set(tables.test[tables.time_col])
        predictions = predictions[predictions[tables.time_col].isin(
            test_dates)]
        if predictions.empty:
            raise ValueError(
                "No Enterprise Y9C holdout rows map to test dates.")
        return predictions

    def get_pquery(self) -> str:
        return (f"PREDICT SUM({DEPOSITS_TABLE}.{TARGET_COL}, 0, "
                f"{Y9C_TIMEFRAME_DAYS}, days) FORECAST 4 TIMEFRAMES\n"
                f"FOR EACH {INSTITUTIONS_TABLE}.{RSSD_ID}")

    def get_enterprise_split(self) -> str:
        dates = pd.Series(
            pd.to_datetime(self._load_deposits()[DATE_COL]).dropna().unique())

        def _synthetic_boundary(real_boundary: pd.Timestamp) -> pd.Timestamp:
            return self._enterprise_time_for_index(
                int((dates < real_boundary).sum()))

        train_end = _synthetic_boundary(ENTERPRISE_VAL_START)
        val_end = _synthetic_boundary(TRAIN_END)
        test_end = _synthetic_boundary(TEST_END)
        fmt = "%Y-%m-%d %H:%M:%S"
        return (f'TimeRangeSplit([("{ENTERPRISE_TIME_ANCHOR.strftime(fmt)}", '
                f'"{train_end.strftime(fmt)}"), '
                f'("{train_end.strftime(fmt)}", "{val_end.strftime(fmt)}"), '
                f'("{val_end.strftime(fmt)}", "{test_end.strftime(fmt)}")])')
