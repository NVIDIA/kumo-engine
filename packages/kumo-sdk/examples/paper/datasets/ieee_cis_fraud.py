# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""IEEE-CIS Fraud Detection: predict fraudulent e-commerce transactions.

Vesta Corporation's IEEE-CIS Fraud Detection Kaggle competition dataset.
~590K e-commerce transactions (394 features) plus an identity table
(~144K rows) with device/browser information.
Task: binary classification predicting isFraud per transaction.

Raw CSVs (train_transaction.csv, train_identity.csv) are read directly
from S3 or a local directory.

See:
https://www.kaggle.com/competitions/ieee-fraud-detection
"""

import tempfile
from pathlib import Path

import pandas as pd
from kumoapi.task import TaskType

import kumoai_studio as kumo

from .benchmarking_dataset import TARGET, BenchmarkingDataset, SupervisedTables

TRANSACTIONS_TABLE = "transactions"
IDENTITY_TABLE = "identity"

TRANSACTION_ID = "TransactionID"
IDENTITY_ID = "identity_id"
IS_FRAUD = "isFraud"

DEFAULT_DATA_ROOT = "s3://kumo-public-datasets/ieee_cis_fraud"

TRANSACTIONS_CSV = "train_transaction.csv"
IDENTITY_CSV = "train_identity.csv"

# TransactionDT is seconds from an undisclosed reference point.
# We anchor to an arbitrary epoch so the graph has a proper
# datetime-typed time_column (required by the framework).
_EPOCH = pd.Timestamp("2017-12-01")

ENTERPRISE_QUERY = """
PREDICT transactions.isFraud = 1
FOR EACH transactions.TransactionID
"""


class IEEECISFraudBenchmarkingDataset(BenchmarkingDataset):

    TARGET_COLUMN = IS_FRAUD

    def __init__(self, data_root: str | Path | None = None) -> None:
        self._set_data_root(data_root, DEFAULT_DATA_ROOT)

    # -- data loading ---------------------------------------------

    def _load_transactions(self) -> pd.DataFrame:
        path = f"{self._data_root_str}/{TRANSACTIONS_CSV}"
        df = pd.read_csv(path)
        df["timestamp"] = _EPOCH + pd.to_timedelta(df["TransactionDT"],
                                                   unit="s")
        return df

    def _load_identity(self) -> pd.DataFrame:
        path = f"{self._data_root_str}/{IDENTITY_CSV}"
        return pd.read_csv(path)

    def _train_test_split(
        self,
        df: pd.DataFrame,
        train_frac: float = 0.8,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Chronological split on TransactionDT."""
        df = df.sort_values("TransactionDT").reset_index(drop=True)
        idx = int(len(df) * train_frac)
        return df.iloc[:idx], df.iloc[idx:]

    # -- Single-table benchmarks ----------------------------------

    def get_entity_tables(self) -> SupervisedTables:
        txn = self._load_transactions()
        txn = txn.drop(columns=["timestamp"], errors="ignore")
        txn = txn.rename(columns={IS_FRAUD: TARGET})
        train, test = self._train_test_split(txn)
        return SupervisedTables(
            train=train,
            test=test,
            target_col=TARGET,
            task_type=TaskType.BINARY_CLASSIFICATION,
        )

    def get_joined_tables(self) -> SupervisedTables:
        txn = self._load_transactions()
        txn = txn.drop(columns=["timestamp"], errors="ignore")
        identity = self._load_identity()
        joined = txn.merge(
            identity,
            on=TRANSACTION_ID,
            how="left",
        )
        joined = joined.rename(columns={IS_FRAUD: TARGET})
        train, test = self._train_test_split(joined)
        return SupervisedTables(
            train=train,
            test=test,
            target_col=TARGET,
            task_type=TaskType.BINARY_CLASSIFICATION,
        )

    # -- Enterprise benchmark -------------------------------------

    def get_enterprise_graph(self):
        txn = self._load_transactions()
        identity = self._load_identity().copy()
        identity[IDENTITY_ID] = range(len(identity))
        with tempfile.TemporaryDirectory() as tmpdir:
            txn_path = str(Path(tmpdir) / "transactions.parquet")
            id_path = str(Path(tmpdir) / "identity.parquet")
            txn.to_parquet(txn_path, index=False)
            identity.to_parquet(id_path, index=False)

            connector = kumo.FileUploadConnector(file_type="parquet")
            connector.upload(name=TRANSACTIONS_TABLE, path=txn_path)
            connector.upload(name=IDENTITY_TABLE, path=id_path)

            txn_src = connector.table(TRANSACTIONS_TABLE)
            id_src = connector.table(IDENTITY_TABLE)
            txn_tbl = kumo.Table.from_source_table(
                txn_src,
                primary_key=TRANSACTION_ID,
                time_column="timestamp",
            ).infer_metadata()
            id_tbl = kumo.Table.from_source_table(
                id_src,
                primary_key=IDENTITY_ID,
            ).infer_metadata()

            return kumo.Graph(
                tables={
                    TRANSACTIONS_TABLE: txn_tbl,
                    IDENTITY_TABLE: id_tbl,
                },
                edges=[
                    dict(
                        src_table=IDENTITY_TABLE,
                        fkey=TRANSACTION_ID,
                        dst_table=TRANSACTIONS_TABLE,
                    ),
                ],
            )

    def get_pquery(self) -> str:
        return ENTERPRISE_QUERY.strip()

    def get_enterprise_split(self) -> str | None:
        """TimeRangeSplit 70/10/20 (train/val/test) on chronological order.

        Matches temporal ordering used in the single-table benchmarks.
        """
        txn = self._load_transactions().sort_values("timestamp")
        n_rows = len(txn)

        idx_train = max(1, int(n_rows * 0.70))
        idx_val = max(idx_train + 1, min(n_rows - 1, int(n_rows * 0.80)))

        start_date = txn["timestamp"].iloc[0].date()
        train_end_date = txn["timestamp"].iloc[idx_train - 1].date()
        val_end_date = txn["timestamp"].iloc[idx_val - 1].date()
        last_ts = txn["timestamp"].iloc[-1]
        test_end_date = (last_ts + pd.Timedelta(days=1)).date()

        return (f'TimeRangeSplit(['
                f'("{start_date}", "{train_end_date}"), '
                f'("{train_end_date}", "{val_end_date}"), '
                f'("{val_end_date}", "{test_end_date}")])')
