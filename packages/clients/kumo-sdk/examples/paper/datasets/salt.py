# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SALT RelBench dataset: multi-task auto-complete benchmark.

This module wraps the SALT dataset from RelBench (rel-salt) into the
BenchmarkingDataset interface used by the paper benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from kumoapi.task import TaskType
from relbench.tasks import AutoCompleteTask, get_task

from .benchmarking_dataset import TARGET, BenchmarkingDataset, SupervisedTables

DATASET_NAME = "rel-salt"


@dataclass
class _SaltSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class SALTBenchmarkingDataset(BenchmarkingDataset):
    """Base class for SALT RelBench auto-complete tasks.

    Concrete subclasses set ``TASK_NAME`` to one of the SALT task names,
    such as "item-plant" or "sales-office".
    """

    TASK_NAME: str = ""

    def __init__(
        self,
        data_root: str | Path | None = None,
    ) -> None:
        # ``data_root`` is accepted for API compatibility with other
        # BenchmarkingDatasets
        del data_root

    # -- RelBench helpers ---------------------------------------------

    def _get_relbench_task(self) -> AutoCompleteTask:
        """Return the RelBench AutoCompleteTask for this SALT task."""
        try:
            task = get_task(DATASET_NAME, self.TASK_NAME, download=True)
        except ValueError:
            task = get_task(DATASET_NAME, self.TASK_NAME)
        if not isinstance(task, AutoCompleteTask):
            msg = ("Expected AutoCompleteTask for SALT task "
                   f"{self.TASK_NAME!r}, got {type(task)!r}")
            raise TypeError(msg)
        return task

    def _get_splits(self) -> _SaltSplits:
        """Load train / val / test splits for this task."""
        task = self._get_relbench_task()
        train = task.get_table("train", mask_input_cols=False).df.copy()
        val = task.get_table("val", mask_input_cols=False).df.copy()
        test = task.get_table("test", mask_input_cols=False).df.copy()
        return _SaltSplits(train=train, val=val, test=test)

    # -- Single-table benchmarks --------------------------------------

    def get_entity_tables(self) -> SupervisedTables:
        """Return entity table splits with full features.

        We use the RelBench task splits only to define which entity ids
        belong to train vs. test, and then materialize the underlying
        entity table (with all its columns) from the RelBench database.
        """
        splits = self._get_splits()
        task = self._get_relbench_task()
        target_col = task.target_col
        entity_col = task.entity_col

        # Load the full entity table from the RelBench DB.
        db = task.dataset.get_db(upto_test_timestamp=False)
        entity_df = db.table_dict[task.entity_table].df.copy()

        # Build a per-entity label from the task table.
        task_all = pd.concat(
            [splits.train, splits.val, splits.test],
            ignore_index=True,
        )
        label_by_id = task_all.set_index(entity_col)[target_col]

        # Map labels to contiguous 0..K-1 so GBDT libraries see a dense
        # class index.
        classes = sorted(label_by_id.dropna().unique())
        class_map = {label: idx for idx, label in enumerate(classes)}
        label_by_id = label_by_id.map(class_map).astype("int64")

        # Determine which entities go to test vs. train. RelBench splits
        # are mutually exclusive, so there is no need to adjust ids.
        test_ids = set(splits.test[entity_col].unique())
        train_ids = set(
            pd.concat(
                [
                    splits.train[entity_col],
                    splits.val[entity_col],
                ],
                ignore_index=True,
            ).unique(), )

        # Attach labels to the full entity table and drop columns that
        # might leak the label (original target column plus any columns
        # listed in task.remove_columns for any table).
        entity_df = entity_df.copy()
        entity_df[TARGET] = entity_df[entity_col].map(label_by_id)
        if target_col in entity_df.columns:
            entity_df = entity_df.drop(columns=[target_col])
        leak_cols = {col_name for _tab, col_name in task.remove_columns}
        drop_cols = [c for c in leak_cols if c in entity_df.columns]
        if drop_cols:
            entity_df = entity_df.drop(columns=drop_cols)

        train_df = entity_df[entity_df[entity_col].isin(train_ids)].copy()
        test_df = entity_df[entity_df[entity_col].isin(test_ids)].copy()

        # Drop any rows without labels (entities not present in the task).
        train_df = train_df.dropna(subset=[TARGET])
        test_df = test_df.dropna(subset=[TARGET])

        return SupervisedTables(
            train=train_df.reset_index(drop=True),
            test=test_df.reset_index(drop=True),
            target_col=TARGET,
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
        )

    def get_joined_tables(self) -> SupervisedTables:
        """Return SALT single-table splits enriched with joins.

        Starting from the entity-level splits, left-join a small number
        of related tables from the RelBench database to provide a
        wider feature set.
        """
        base = self.get_entity_tables()
        task = self._get_relbench_task()
        db = task.dataset.get_db(upto_test_timestamp=False)

        train = base.train.copy()
        test = base.test.copy()
        entity_table = task.entity_table

        if entity_table == "salesdocumentitem":
            # Join salesdocument attributes on SALESDOCUMENT.
            sales = db.table_dict["salesdocument"].df.copy()
            key = "SALESDOCUMENT"
            if key in train.columns and key in sales.columns:
                train = train.merge(
                    sales,
                    on=key,
                    how="left",
                    suffixes=("", "_sales"),
                )
                test = test.merge(
                    sales,
                    on=key,
                    how="left",
                    suffixes=("", "_sales"),
                )

            # Join customer attributes on SOLDTOPARTY when available.
            customer = db.table_dict["customer"].df.copy()
            if ("SOLDTOPARTY" in train.columns
                    and "CUSTOMER" in customer.columns):
                train = train.merge(
                    customer,
                    left_on="SOLDTOPARTY",
                    right_on="CUSTOMER",
                    how="left",
                    suffixes=("", "_cust"),
                )
                test = test.merge(
                    customer,
                    left_on="SOLDTOPARTY",
                    right_on="CUSTOMER",
                    how="left",
                    suffixes=("", "_cust"),
                )

        elif entity_table == "salesdocument":
            # Join customer attributes directly if possible.
            customer = db.table_dict["customer"].df.copy()
            if ("SOLDTOPARTY" in train.columns
                    and "CUSTOMER" in customer.columns):
                train = train.merge(
                    customer,
                    left_on="SOLDTOPARTY",
                    right_on="CUSTOMER",
                    how="left",
                    suffixes=("", "_cust"),
                )
                test = test.merge(
                    customer,
                    left_on="SOLDTOPARTY",
                    right_on="CUSTOMER",
                    how="left",
                    suffixes=("", "_cust"),
                )

        # Drop any columns that are marked as removable for leakage
        # control in the RelBench task definition.
        leak_cols = {col_name for _tab, col_name in task.remove_columns}
        for df in (train, test):
            drop_cols = [c for c in leak_cols if c in df.columns]
            if drop_cols:
                df.drop(columns=drop_cols, inplace=True)

        return SupervisedTables(
            train=train.reset_index(drop=True),
            test=test.reset_index(drop=True),
            target_col=base.target_col,
            task_type=base.task_type,
        )

    # -- Enterprise benchmark -----------------------------------------

    def get_enterprise_graph(self):
        msg = "Enterprise graph not implemented for SALT"
        raise NotImplementedError(msg)

    def get_pquery(self) -> str:
        msg = "Enterprise pquery not implemented for SALT"
        raise NotImplementedError(msg)


# -- Concrete SALT tasks ----------------------------------------------


class SALTItemPlantBenchmarkingDataset(SALTBenchmarkingDataset):
    TASK_NAME = "item-plant"


class SALTItemShipPointBenchmarkingDataset(SALTBenchmarkingDataset):
    TASK_NAME = "item-shippoint"


class SALTItemIncotermsBenchmarkingDataset(SALTBenchmarkingDataset):
    TASK_NAME = "item-incoterms"


class SALTSalesOfficeBenchmarkingDataset(SALTBenchmarkingDataset):
    TASK_NAME = "sales-office"


class SALTSalesGroupBenchmarkingDataset(SALTBenchmarkingDataset):
    TASK_NAME = "sales-group"


class SALTSalesPaytermsBenchmarkingDataset(SALTBenchmarkingDataset):
    TASK_NAME = "sales-payterms"


class SALTSalesShipcondBenchmarkingDataset(SALTBenchmarkingDataset):
    TASK_NAME = "sales-shipcond"


class SALTSalesIncotermsBenchmarkingDataset(SALTBenchmarkingDataset):
    TASK_NAME = "sales-incoterms"
