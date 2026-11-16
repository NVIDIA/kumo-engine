# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DataCo Supply Chain: multi-class delivery status prediction.

Dataset:
- Original single-table CSV:
  s3://kumo-public-datasets/dataco_supply_chain/csv/
    - DataCoSupplyChainDataset.csv
- Relational split used here:
  s3://kumo-public-datasets/dataco_supply_chain/relational/
    - customer_dim.csv
    - product_dim.csv
    - trans_fact.csv

Task:
- Multi-class classification on Delivery_Status per order line
  (e.g. 'Late delivery', 'Advance shipping', 'Shipping on time',
  'Shipping canceled', etc.).
- Temporal 80/20 split based on the order timestamp.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal

import pandas as pd
from kumoapi.task import TaskType

import kumoai_studio as kumo

from .benchmarking_dataset import TARGET, BenchmarkingDataset, SupervisedTables

ORDERS_TABLE = "orders"
CUSTOMERS_TABLE = "customers"
PRODUCTS_TABLE = "products"
CATEGORIES_TABLE = "categories"
DEPARTMENTS_TABLE = "departments"

CUSTOMER_ID = "Customer_Id"
ORDER_CUSTOMER_ID = "Order_Customer_Id"

PRODUCT_ID = "Product_Card_Id"
ORDER_PRODUCT_ID = "Order_Item_Cardprod_Id"

ORDER_ITEM_ID = "Order_Item_Id"
ORDER_ID = "Order_Id"

CATEGORY_ID = "Category_Id"
CATEGORY_NAME = "Category_Name"
DEPARTMENT_ID = "Department_Id"
DEPARTMENT_NAME = "Department_Name"

RAW_ORDER_DATE_REL = "order_date_DateOrders_"
RAW_ORDER_DATE_SINGLE = "order_date_DateOrders"
ORDER_DATE = "order_date"

DELIVERY_STATUS = "Delivery_Status"
LATE_DELIVERY_RISK = "Late_delivery_risk"

DEFAULT_RELATIONAL_ROOT = (
    "s3://kumo-public-datasets/dataco_supply_chain/relational")
DEFAULT_CSV_ROOT = "s3://kumo-public-datasets/dataco_supply_chain/csv"


class DataCoSupplyChainBenchmarkingDataset(BenchmarkingDataset):

    TARGET_COLUMN = DELIVERY_STATUS

    def __init__(
        self,
        data_root: str | Path | None = None,
        *,
        task: Literal["multiclass", "binary"] = "multiclass",
        positive_label: Literal[
            "Advance shipping",
            "Late delivery",
            "Shipping on time",
            "Shipping canceled",
        ] | None = None,
    ) -> None:
        """Args:
        data_root: directory or S3 URI containing the relational CSVs:
            - customer_dim.csv
            - product_dim.csv
            - trans_fact.csv
            Default:
            s3://kumo-public-datasets/dataco_supply_chain/relational
        task: "binary" (default) or "multiclass". In binary mode the
            target is 1 for rows whose Delivery_Status equals
            ``positive_label`` and 0 otherwise.
        positive_label: Delivery_Status value to treat as the positive
            class when task="binary" (e.g. "Late delivery" or
            "Shipping canceled").
        """
        self._set_data_root(data_root, DEFAULT_RELATIONAL_ROOT)
        # Keep pointer to original single-table CSV root for potential
        # future use.
        self._csv_root_str = DEFAULT_CSV_ROOT

        if task == "multiclass":
            self._task_type = TaskType.MULTICLASS_CLASSIFICATION
            self._positive_label: str | None = None
        else:
            self._task_type = TaskType.BINARY_CLASSIFICATION
            assert positive_label is not None
            self._positive_label = positive_label

    # -- data loading -------------------------------------------------

    def _load_orders(self) -> pd.DataFrame:
        """Load transactional fact table."""
        path = f"{self._data_root_str}/trans_fact.csv"
        df = pd.read_csv(path)
        # Normalise timestamp column name to ORDER_DATE.
        if RAW_ORDER_DATE_REL in df.columns:
            df[ORDER_DATE] = pd.to_datetime(df[RAW_ORDER_DATE_REL])
        elif RAW_ORDER_DATE_SINGLE in df.columns:
            df[ORDER_DATE] = pd.to_datetime(df[RAW_ORDER_DATE_SINGLE])
        else:
            raise ValueError(
                "order date column not found in trans_fact.csv; "
                "expected one of "
                f"{RAW_ORDER_DATE_REL!r} or {RAW_ORDER_DATE_SINGLE!r}", )
        # Drop known leakage-prone proxy for the label.
        df = df.drop(columns=[LATE_DELIVERY_RISK], errors="ignore")
        return df

    def _load_customers(self) -> pd.DataFrame:
        path = f"{self._data_root_str}/customer_dim.csv"
        return pd.read_csv(path)

    def _load_products(self) -> pd.DataFrame:
        path = f"{self._data_root_str}/product_dim.csv"
        return pd.read_csv(path)

    @staticmethod
    def _train_test_split(
        df: pd.DataFrame,
        *,
        time_col: str,
        train_frac: float = 0.8,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Chronological split on *time_col*."""
        df = df.sort_values(time_col).reset_index(drop=True)
        idx = int(len(df) * train_frac)
        return df.iloc[:idx], df.iloc[idx:]

    def _prepare_supervised_orders(self) -> pd.DataFrame:
        """Return orders with encoded target column ``TARGET``.

        Drops rows without a delivery status.
        """
        orders = self._load_orders()
        if DELIVERY_STATUS not in orders.columns:
            raise ValueError(
                f"{DELIVERY_STATUS!r} not found in trans_fact.csv columns", )
        orders = orders.dropna(subset=[DELIVERY_STATUS]).copy()

        if self._task_type == TaskType.MULTICLASS_CLASSIFICATION:
            # Map string labels to contiguous integer class IDs.
            orders[DELIVERY_STATUS] = (orders[DELIVERY_STATUS].astype(
                "category").cat.codes.astype(int))
        else:
            orders[DELIVERY_STATUS] = (
                orders[DELIVERY_STATUS] == self._positive_label).astype(int)

        orders = orders.rename(columns={DELIVERY_STATUS: TARGET})
        # Category and department keys/names live in separate dims for
        # supervised/entity tables; drop them from the entity table.
        orders = orders.drop(
            columns=[
                CATEGORY_ID,
                CATEGORY_NAME,
                DEPARTMENT_ID,
                DEPARTMENT_NAME,
            ],
            errors="ignore",
        )
        return orders

    # -- Single-table benchmarks -------------------------------------

    def get_entity_tables(self) -> SupervisedTables:
        """Order-line level single table (no joined dims)."""
        orders = self._prepare_supervised_orders()
        train, test = self._train_test_split(
            orders,
            time_col=ORDER_DATE,
        )
        return SupervisedTables(
            train=train.reset_index(drop=True),
            test=test.reset_index(drop=True),
            target_col=TARGET,
            task_type=self._task_type,
        )

    def get_joined_tables(self) -> SupervisedTables:
        """Orders enriched with customer and product attributes."""
        orders = self._prepare_supervised_orders()
        customers = self._load_customers()
        products = self._load_products()

        joined = orders.merge(
            customers,
            left_on=ORDER_CUSTOMER_ID,
            right_on=CUSTOMER_ID,
            how="left",
        )
        joined = joined.merge(
            products,
            left_on=ORDER_PRODUCT_ID,
            right_on=PRODUCT_ID,
            how="left",
        )

        train, test = self._train_test_split(
            joined,
            time_col=ORDER_DATE,
        )
        return SupervisedTables(
            train=train.reset_index(drop=True),
            test=test.reset_index(drop=True),
            target_col=TARGET,
            task_type=self._task_type,
        )

    # -- Enterprise benchmark ----------------------------------------

    def get_enterprise_graph(self):
        """Build and return a kumoai_studio.Graph for Enterprise."""
        orders = self._load_orders()
        customers = self._load_customers()
        products = self._load_products()

        categories = None
        if {CATEGORY_ID, CATEGORY_NAME}.issubset(orders.columns):
            categories = (orders[[CATEGORY_ID, CATEGORY_NAME
                                  ]].drop_duplicates().reset_index(drop=True))

        departments = None
        if {DEPARTMENT_ID, DEPARTMENT_NAME}.issubset(orders.columns):
            departments = (orders[[DEPARTMENT_ID, DEPARTMENT_NAME
                                   ]].drop_duplicates().reset_index(drop=True))

        # For enterprise we keep raw string columns; _load_orders already
        # drops the leakage-prone Late_delivery_risk column.
        orders_ent = orders.drop(
            columns=[
                CATEGORY_NAME,
                DEPARTMENT_NAME,
            ],
            errors="ignore",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            orders_path = str(tmp_path / "orders.parquet")
            customers_path = str(tmp_path / "customers.parquet")
            products_path = str(tmp_path / "products.parquet")
            orders_ent.to_parquet(orders_path, index=False)
            customers.to_parquet(customers_path, index=False)
            products.to_parquet(products_path, index=False)

            categories_path = None
            if categories is not None:
                categories_path = str(tmp_path / "categories.parquet")
                categories.to_parquet(categories_path, index=False)

            departments_path = None
            if departments is not None:
                departments_path = str(tmp_path / "departments.parquet")
                departments.to_parquet(departments_path, index=False)

            connector = kumo.FileUploadConnector(file_type="parquet")
            connector.upload(name=ORDERS_TABLE, path=orders_path)
            connector.upload(name=CUSTOMERS_TABLE, path=customers_path)
            connector.upload(name=PRODUCTS_TABLE, path=products_path)
            if categories_path is not None:
                connector.upload(name=CATEGORIES_TABLE, path=categories_path)
            if departments_path is not None:
                connector.upload(name=DEPARTMENTS_TABLE, path=departments_path)

            orders_src = connector.table(ORDERS_TABLE)
            customers_src = connector.table(CUSTOMERS_TABLE)
            products_src = connector.table(PRODUCTS_TABLE)

            orders_pk = (ORDER_ITEM_ID
                         if ORDER_ITEM_ID in orders_ent.columns else ORDER_ID)

            orders_tbl = kumo.Table.from_source_table(
                orders_src,
                primary_key=orders_pk,
                time_column=ORDER_DATE,
            ).infer_metadata()
            customers_tbl = kumo.Table.from_source_table(
                customers_src,
                primary_key=CUSTOMER_ID,
            ).infer_metadata()
            products_tbl = kumo.Table.from_source_table(
                products_src,
                primary_key=PRODUCT_ID,
            ).infer_metadata()

            tables = {
                ORDERS_TABLE: orders_tbl,
                CUSTOMERS_TABLE: customers_tbl,
                PRODUCTS_TABLE: products_tbl,
            }

            if categories_path is not None:
                categories_src = connector.table(CATEGORIES_TABLE)
                categories_tbl = kumo.Table.from_source_table(
                    categories_src,
                    primary_key=CATEGORY_ID,
                ).infer_metadata()
                tables[CATEGORIES_TABLE] = categories_tbl

            if departments_path is not None:
                departments_src = connector.table(DEPARTMENTS_TABLE)
                departments_tbl = kumo.Table.from_source_table(
                    departments_src,
                    primary_key=DEPARTMENT_ID,
                ).infer_metadata()
                tables[DEPARTMENTS_TABLE] = departments_tbl

            edges = [
                dict(
                    src_table=ORDERS_TABLE,
                    fkey=ORDER_CUSTOMER_ID,
                    dst_table=CUSTOMERS_TABLE,
                ),
                dict(
                    src_table=ORDERS_TABLE,
                    fkey=ORDER_PRODUCT_ID,
                    dst_table=PRODUCTS_TABLE,
                ),
            ]
            if categories_path is not None:
                edges.append(
                    dict(
                        src_table=ORDERS_TABLE,
                        fkey=CATEGORY_ID,
                        dst_table=CATEGORIES_TABLE,
                    ), )
            if departments_path is not None:
                edges.append(
                    dict(
                        src_table=ORDERS_TABLE,
                        fkey=DEPARTMENT_ID,
                        dst_table=DEPARTMENTS_TABLE,
                    ), )

            return kumo.Graph(
                tables=tables,
                edges=edges,
            )

    def get_pquery(self) -> str:
        """Predictive query for Delivery_Status per order line.

        In multiclass mode this predicts the raw Delivery_Status labels.
        In binary mode it predicts whether Delivery_Status equals the
        configured positive_label.
        """
        orders = self._load_orders()
        entity_col = (ORDER_ITEM_ID
                      if ORDER_ITEM_ID in orders.columns else ORDER_ID)

        if self._task_type == TaskType.MULTICLASS_CLASSIFICATION:
            lines = [
                f"PREDICT {ORDERS_TABLE}.{DELIVERY_STATUS}",
                f"FOR EACH {ORDERS_TABLE}.{entity_col}",
            ]
            return "\n".join(lines)

        assert self._positive_label is not None
        label = str(self._positive_label).replace('"', '\\"')
        lines = [
            (f'PREDICT {ORDERS_TABLE}.{DELIVERY_STATUS} = '
             f'"{label}"'),
            f"FOR EACH {ORDERS_TABLE}.{entity_col}",
        ]
        return "\n".join(lines)

    def get_enterprise_split(self) -> str:
        """TimeRangeSplit 70/10/20 (train/val/test) on ORDER_DATE."""
        orders = self._load_orders().sort_values(ORDER_DATE).reset_index(
            drop=True)

        n_rows = len(orders)

        idx_train = int(n_rows * 0.70)
        idx_val = max(idx_train + 1, min(n_rows - 1, int(n_rows * 0.80)))

        start_date = orders[ORDER_DATE].iloc[0].date()
        train_end_date = orders[ORDER_DATE].iloc[idx_train - 1].date()
        val_end_date = orders[ORDER_DATE].iloc[idx_val - 1].date()
        last_ts = orders[ORDER_DATE].iloc[-1]
        test_end_date = (last_ts + pd.Timedelta(days=1)).date()

        return (f'TimeRangeSplit(['
                f'("{start_date}", "{train_end_date}"), '
                f'("{train_end_date}", "{val_end_date}"), '
                f'("{val_end_date}", "{test_end_date}")])')
