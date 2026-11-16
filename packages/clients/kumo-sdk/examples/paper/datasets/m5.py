# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""M5: Walmart daily retail sales forecasting.

Dataset:
  s3://kumo-public-datasets/m5_forecast_retail_sales/
    - store_dim.parquet       – 10 Walmart stores across CA, TX, WI
    - item_dim.parquet        – 3,049 products (item_id, dept_id, cat_id)
    - item_store_dim.parquet  – 30,490 item-store combos (entity table)
    - m5_sales_long.parquet   – daily unit sales d_1..d_1941 (train + val)
    - m5_sales_eval.parquet   – daily unit sales d_1942..d_1969 (test / eval)
    - snap_facts.parquet      – daily SNAP eligibility per state
    - event_facts.parquet     – calendar events (holidays, sporting, etc.)

Task:
  - Regression: predict daily unit sales per item-store.
  - Split follows the M5 competition private leaderboard:
    train d_1..d_1913, val d_1914..d_1941, test d_1942..d_1969.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import pandas as pd
from kumoapi.task import TaskType

import kumoai_studio as kumo

from .benchmarking_dataset import (
    TARGET,
    BenchmarkingDataset,
    ForecastingTables,
    SupervisedTables,
)

DEFAULT_DATA_ROOT = "s3://kumo-public-datasets/m5_forecast_retail_sales"

TRAIN_END = pd.Timestamp("2016-04-25")
VAL_END = pd.Timestamp("2016-05-23")
TEST_END = pd.Timestamp("2016-06-20")


class M5BenchmarkingDataset(BenchmarkingDataset):
    """M5 Walmart retail sales forecasting benchmark."""
    def __init__(
        self,
        data_root: Union[str, Path, None] = None,
    ) -> None:
        self._set_data_root(data_root, DEFAULT_DATA_ROOT)

    # -- data loading -------------------------------------------------

    def _load_sales(self, *, include_eval: bool = True) -> pd.DataFrame:
        sales = self._read_parquet("m5_sales_long.parquet")
        sales["date"] = pd.to_datetime(sales["date"])
        if include_eval:
            eval_sales = self._read_parquet("m5_sales_eval.parquet")
            eval_sales["date"] = pd.to_datetime(eval_sales["date"])
            sales = pd.concat([sales, eval_sales], ignore_index=True)
        return sales

    # -- splits -------------------------------------------------------

    def _split(
            self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = df[df["date"] < TRAIN_END].reset_index(drop=True)
        context = df[(df["date"] >= TRAIN_END)
                     & (df["date"] < VAL_END)].reset_index(drop=True)
        test = df[(df["date"] >= VAL_END)
                  & (df["date"] < TEST_END)].reset_index(drop=True)
        return train, context, test

    # -- Single-table benchmarks --------------------------------------

    def get_entity_tables(self) -> SupervisedTables:
        sales = self._load_sales()
        sales = sales[["item_store_id", "date", "sales"]].copy()
        sales = sales.rename(columns={"sales": TARGET})
        train, ctx, test = self._split(sales)
        train = pd.concat([train, ctx], ignore_index=True)
        return SupervisedTables(
            train=train,
            test=test,
            target_col=TARGET,
            task_type=TaskType.REGRESSION,
            entity_col="item_store_id",
        )

    def get_joined_tables(self) -> SupervisedTables:
        sales = self._load_sales()
        items = self._read_parquet("item_dim.parquet")
        joined = sales.merge(items, on="item_id", how="left",
                             suffixes=("", "_item"))
        joined = joined[[
            "item_store_id", "date", "sales", "dept_id", "cat_id"
        ]].copy()
        joined = joined.rename(columns={"sales": TARGET})
        train, ctx, test = self._split(joined)
        train = pd.concat([train, ctx], ignore_index=True)
        return SupervisedTables(
            train=train,
            test=test,
            target_col=TARGET,
            task_type=TaskType.REGRESSION,
            entity_col="item_store_id",
        )

    # -- Forecasting benchmarks ---------------------------------------

    def get_forecasting_tables(self) -> ForecastingTables:
        sales = self._load_sales()
        sales = sales[["item_store_id", "date", "sales"]].copy()
        sales = sales.rename(columns={"sales": TARGET})
        train, context, test = self._split(sales)
        return ForecastingTables(
            train=train,
            test=test,
            entity_col="item_store_id",
            time_col="date",
            target_col=TARGET,
            freq="D",
            seasonal_period=7,  # 7 × D = 1 week
            feature_preset="calendar",
            lag_timesteps=7,  # One week of daily history.
            context=context,
        )

    # -- Enterprise benchmark -----------------------------------------

    def get_enterprise_graph(
        self,
        *,
        history_only: bool = False,
    ) -> kumo.Graph:
        connector = kumo.S3Connector(
            name="m5_benchmark_source",
            identity=os.environ["KUMO_S3_IDENTITY_ID"],
            root_dir=self._data_root_str,
        )

        item_store_dim_tbl = kumo.Table(
            connector.table("item_store_dim"),
            columns=[
                kumo.Column("item_store_id", stype="ID", dtype="string"),
                kumo.Column("item_id", stype="ID", dtype="string"),
                kumo.Column("store_id", stype="ID", dtype="string"),
                kumo.Column("dept_id", stype="categorical", dtype="string"),
                kumo.Column("cat_id", stype="categorical", dtype="string"),
            ],
            primary_key="item_store_id",
        )

        sales_cols = [
            kumo.Column("item_store_id", stype="ID", dtype="string"),
            kumo.Column("date", stype="timestamp", dtype="date"),
            kumo.Column("sales", stype="numerical", dtype="int"),
            kumo.Column("sell_price", stype="numerical", dtype="float"),
            kumo.Column("d", stype="ID", dtype="string"),
        ]
        if history_only:
            sales_cols = sales_cols[:3]

        sales_tbl = kumo.Table(
            connector.table("m5_sales_long"),
            columns=sales_cols,
            time_column="date",
        )

        tables = {
            "item_store_dim": item_store_dim_tbl,
            "m5_sales_long": sales_tbl,
        }
        edges = [
            dict(src_table="m5_sales_long", fkey="item_store_id",
                 dst_table="item_store_dim"),
        ]

        if not history_only:
            tables["item_dim"] = kumo.Table(
                connector.table("item_dim"),
                columns=[
                    kumo.Column("item_id", stype="ID", dtype="string"),
                    kumo.Column("dept_id", stype="categorical",
                                dtype="string"),
                    kumo.Column("cat_id", stype="categorical", dtype="string"),
                ],
                primary_key="item_id",
            )
            tables["store_dim"] = kumo.Table(
                connector.table("store_dim"),
                columns=[
                    kumo.Column("store_id", stype="ID", dtype="string"),
                ],
                primary_key="store_id",
            )
            tables["snap_facts"] = kumo.Table(
                connector.table("snap_facts"),
                columns=[
                    kumo.Column("d", stype="ID", dtype="string"),
                    kumo.Column("date", stype="timestamp", dtype="date"),
                    kumo.Column("CA", stype="numerical", dtype="int"),
                    kumo.Column("TX", stype="numerical", dtype="int"),
                    kumo.Column("WI", stype="numerical", dtype="int"),
                ],
                primary_key="d",
            )
            tables["event_facts"] = kumo.Table(
                connector.table("event_facts"),
                columns=[
                    kumo.Column("date", stype="timestamp", dtype="date"),
                    kumo.Column("month", stype="numerical", dtype="int"),
                    kumo.Column("event_name", stype="categorical",
                                dtype="string"),
                    kumo.Column("event_type", stype="categorical",
                                dtype="string"),
                    kumo.Column("d", stype="ID", dtype="string"),
                    kumo.Column("snap_CA", stype="numerical", dtype="int"),
                    kumo.Column("snap_TX", stype="numerical", dtype="int"),
                    kumo.Column("snap_WI", stype="numerical", dtype="int"),
                ],
            )
            edges.extend([
                dict(src_table="item_store_dim", fkey="item_id",
                     dst_table="item_dim"),
                dict(src_table="item_store_dim", fkey="store_id",
                     dst_table="store_dim"),
                dict(src_table="m5_sales_long", fkey="d",
                     dst_table="snap_facts"),
                dict(src_table="event_facts", fkey="d",
                     dst_table="snap_facts"),
            ])

        return kumo.Graph(tables=tables, edges=edges)

    def get_pquery(self) -> str:
        return self.get_forecast_query(num_timeframes=28)

    def get_forecast_query(self, *, num_timeframes: int) -> str:
        return (f"PREDICT SUM(m5_sales_long.sales, 0, 1, days)"
                f" FORECAST {num_timeframes} TIMEFRAMES\n"
                f"FOR EACH item_store_dim.item_store_id")

    def get_enterprise_split(self) -> str:
        fmt = "%Y-%m-%d %H:%M:%S"
        return (f'TimeRangeSplit(['
                f'("2011-01-29 00:00:00", "{TRAIN_END.strftime(fmt)}"), '
                f'("{TRAIN_END.strftime(fmt)}", "{VAL_END.strftime(fmt)}"), '
                f'("{VAL_END.strftime(fmt)}", "{TEST_END.strftime(fmt)}")])')
