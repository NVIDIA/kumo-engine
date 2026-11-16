# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""IBM AML: accounts + transactions, predict is_laundering.

IBM synthetic anti-money laundering (AML) data: a labeled financial
ecosystem of accounts and transactions. The generator models the full
laundering cycle (placement, layering, integration) and labels
individual transactions as laundering or legitimate.
Task: binary classification predicting is_laundering per transaction.
Data: HI/LI (higher/lower illicit ratio) in small/medium/large sizes;
we use the provided 60/20/20 chronological train/val/test split.

See:
https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml  # noqa: E501
"""

import tempfile
from pathlib import Path

import pandas as pd
from kumoapi.task import TaskType

import kumoai_studio as kumo

from .benchmarking_dataset import TARGET, BenchmarkingDataset, SupervisedTables

# Parquet table names
ACCOUNTS_TABLE = "accounts"
TRANSACTIONS_TABLE = "transactions"

# Column used for time-based train/test split
LAST_TS_COL = "last_ts"

# Dataset variant:
# Higher illicit ratio: HI-Small | HI-Medium | HI-Large
# Lower illicit ratio: LI-Small | LI-Medium | LI-Large
IBM_DATASET_VARIANT = "HI-Small"

# Default data root and CSV filenames (derived from variant)
DEFAULT_DATA_ROOT = f"s3://kumo-public-datasets/ibm-aml/{IBM_DATASET_VARIANT}"
ACCOUNTS_CSV = f"{IBM_DATASET_VARIANT}_accounts.csv"
TRANSACTIONS_CSV = f"{IBM_DATASET_VARIANT}_Trans.csv"

# Enterprise predictive query: account-level (any laundering in window).
# NOTE: enterprise pquery operates at account granularity; the ground-truth
# task is per-transaction (see get_entity_tables).
ENTERPRISE_QUERY = """
PREDICT SUM(transactions.is_laundering, 0, 90, days) > 0
FOR EACH accounts.account_number
WHERE SUM(transactions.amount_paid, -60, 0, days) > 0
"""


class IBMAMLBenchmarkingDataset(BenchmarkingDataset):

    # Target column: exclude from graph so the model cannot see the label.
    TARGET_COLUMN = "is_laundering"

    def __init__(self, data_root: str | Path | None = None) -> None:
        """Args:
        data_root: directory with HI-Small_accounts.csv and HI-Small_Trans.csv
            (local path or S3 URI).
            Default: s3://kumo-public-datasets/ibm-aml/HI-Small
        """
        self._set_data_root(data_root, DEFAULT_DATA_ROOT)

    def _load_accounts(self) -> pd.DataFrame:
        """Load and transform accounts CSV to expected schema."""
        path = f"{self._data_root_str}/{ACCOUNTS_CSV}"
        df = pd.read_csv(path)
        df.columns = [
            "bank_name",
            "bank_id",
            "account_number",
            "entity_id",
            "entity_name",
        ]
        return df

    def _load_transactions(self) -> pd.DataFrame:
        """Load and transform transactions CSV to expected schema."""
        path = f"{self._data_root_str}/{TRANSACTIONS_CSV}"
        df = pd.read_csv(path)
        df.columns = [
            "timestamp",
            "from_bank",
            "from_account",
            "to_bank",
            "to_account",
            "amount_received",
            "receiving_currency",
            "amount_paid",
            "payment_currency",
            "payment_format",
            "is_laundering",
        ]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def get_enterprise_graph(self):
        accounts = self._load_accounts()
        transactions = self._load_transactions()
        with tempfile.TemporaryDirectory() as tmpdir:
            accounts_path = str(Path(tmpdir) / "accounts.parquet")
            transactions_path = str(Path(tmpdir) / "transactions.parquet")
            accounts.to_parquet(accounts_path, index=False)
            transactions.to_parquet(transactions_path, index=False)
            connector = kumo.FileUploadConnector(file_type="parquet")
            if not connector.has_table(ACCOUNTS_TABLE):
                connector.upload(name=ACCOUNTS_TABLE, path=accounts_path)
            if not connector.has_table(TRANSACTIONS_TABLE):
                connector.upload(name=TRANSACTIONS_TABLE,
                                 path=transactions_path)

            accounts_src = connector.table(ACCOUNTS_TABLE)
            transactions_src = connector.table(TRANSACTIONS_TABLE)
            accounts_tbl = kumo.Table.from_source_table(
                accounts_src,
                primary_key="account_number",
            ).infer_metadata()
            transactions_tbl = kumo.Table.from_source_table(
                transactions_src,
                time_column="timestamp",
            ).infer_metadata()

            return kumo.Graph(
                tables={
                    ACCOUNTS_TABLE: accounts_tbl,
                    TRANSACTIONS_TABLE: transactions_tbl,
                },
                edges=[
                    dict(
                        src_table=TRANSACTIONS_TABLE,
                        fkey="from_account",
                        dst_table=ACCOUNTS_TABLE,
                    ),
                ],
            )

    def get_pquery(self) -> str:
        return ENTERPRISE_QUERY.strip()

    def _train_test_split(
        self,
        df: pd.DataFrame,
        time_col: str = LAST_TS_COL,
        train_frac: float = 0.8,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Chronological split on *time_col*."""
        df = df.sort_values(time_col).reset_index(drop=True)
        idx = int(len(df) * train_frac)
        return df.iloc[:idx], df.iloc[idx:]

    def get_entity_tables(self) -> SupervisedTables:
        """Transaction-level train/test splits using only the
        transactions table (no account features).
        """
        txn = self._load_transactions()
        txn = txn.rename(columns={"is_laundering": TARGET})
        train, test = self._train_test_split(
            txn,
            time_col="timestamp",
        )
        return SupervisedTables(
            train=train,
            test=test,
            target_col=TARGET,
            task_type=TaskType.BINARY_CLASSIFICATION,
        )

    def get_joined_tables(self) -> SupervisedTables:
        """Transaction-level train/test splits enriched with sender
        account features (bank_name, bank_id, entity_id,
        entity_name).
        """
        txn = self._load_transactions()
        accounts = self._load_accounts()
        joined = txn.merge(
            accounts,
            left_on="from_account",
            right_on="account_number",
            how="left",
        ).drop(columns=["account_number"])
        joined = joined.rename(columns={"is_laundering": TARGET})
        train, test = self._train_test_split(
            joined,
            time_col="timestamp",
        )
        return SupervisedTables(
            train=train,
            test=test,
            target_col=TARGET,
            task_type=TaskType.BINARY_CLASSIFICATION,
        )
