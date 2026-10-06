# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Elliptic Bitcoin: transaction graph, predict illicit vs licit.

Elliptic Bitcoin dataset maps Bitcoin transactions to licit vs illicit
categories.  The graph contains 203,769 transaction nodes connected by
234,355 directed payment-flow edges.  Each transaction has 165 numerical
features and belongs to one of 49 timesteps.
Labels: ~2% illicit (4,545), ~21% licit (42,019), rest unknown.
Task: binary classification predicting illicit (1) vs licit (0).

Canonical train/test split: timestep < 35 = train, >= 35 = test
(following the original paper and PyG convention).

Raw CSVs are downloaded from data.pyg.org (same source PyG uses).

See:
https://arxiv.org/abs/1908.02591
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.EllipticBitcoinDataset.html # noqa: E501
"""

import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from kumoapi.task import TaskType

import kumoai_studio as kumo

from .benchmarking_dataset import TARGET, BenchmarkingDataset, SupervisedTables

TRANSACTIONS_TABLE = "transactions"
PAYMENT_FLOWS_TABLE = "payment_flows"

TRANSACTION_ID = "transaction_id"
FLOW_ID = "flow_id"
TIME_STEP = "time_step"
LABEL = "label"

# Canonical temporal split boundary from the original paper.
TRAIN_TEST_SPLIT_STEP = 35

_BASE_URL = "https://data.pyg.org/datasets/elliptic"
_FEATURES_FILE = "elliptic_txs_features.csv"
_CLASSES_FILE = "elliptic_txs_classes.csv"
_EDGELIST_FILE = "elliptic_txs_edgelist.csv"

DEFAULT_DATA_ROOT = str(Path.home() / ".cache" / "elliptic_bitcoin")

# Epoch for synthetic timestamp: timestamp = EPOCH + time_step days.
_EPOCH = pd.Timestamp("2018-01-01")

# PQuery: predict label per transaction, only labeled rows (label >= 0).
# Train/test split at TRAIN_TEST_SPLIT_STEP is applied via the
# get_enterprise_split method, not in the query.
ENTERPRISE_QUERY = """
PREDICT transactions.label = 1
FOR EACH transactions.transaction_id
WHERE transactions.label >= 0
"""


class EllipticBenchmarkingDataset(BenchmarkingDataset):

    TARGET_COLUMN = LABEL

    def __init__(self, data_root: str | Path | None = None) -> None:
        """Args:
        data_root: directory containing the three Elliptic CSVs
            Default: ~/.cache/elliptic_bitcoin (auto-downloaded).
        """
        self._set_data_root(data_root, DEFAULT_DATA_ROOT)
        self._ensure_downloaded()

    def _ensure_downloaded(self) -> None:
        """Download raw CSVs if not already cached."""
        if self._data_root_str.startswith("s3://"):
            return
        cache = Path(self._data_root_str)
        cache.mkdir(parents=True, exist_ok=True)
        for name in [_FEATURES_FILE, _CLASSES_FILE, _EDGELIST_FILE]:
            if (cache / name).exists():
                continue
            url = f"{_BASE_URL}/{name}.zip"
            zip_path = cache / f"{name}.zip"
            print(f"Downloading {name} from {url} ...")
            urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(cache)
            zip_path.unlink()

    # -- data loading ---------------------------------------------

    def _load_transactions(self) -> pd.DataFrame:
        """Load features + class labels, merge, add synthetic timestamp."""
        root = self._data_root_str

        feat = pd.read_csv(f"{root}/{_FEATURES_FILE}", header=None)
        cols = (["txId", TIME_STEP] +
                [f"feature_{i}" for i in range(feat.shape[1] - 2)])
        feat.columns = cols

        cls_df = pd.read_csv(f"{root}/{_CLASSES_FILE}")
        cls_df["class"] = cls_df["class"].map({"1": 1, "2": 0})
        cls_df = cls_df.rename(columns={"class": LABEL})

        df = feat.merge(cls_df, on="txId", how="left")
        df[TRANSACTION_ID] = range(len(df))

        epoch = pd.Timestamp("2018-01-01")
        df["timestamp"] = epoch + pd.to_timedelta(df[TIME_STEP], unit="D")
        return df

    def _load_payment_flows(self, txn: pd.DataFrame) -> pd.DataFrame:
        """Load edge list and remap original txIds to contiguous IDs."""
        txid_map = dict(zip(txn["txId"], txn[TRANSACTION_ID]))
        edges = pd.read_csv(f"{self._data_root_str}/{_EDGELIST_FILE}")
        df = pd.DataFrame({
            "src_txn_id": edges["txId1"].map(txid_map),
            "dst_txn_id": edges["txId2"].map(txid_map),
        }).dropna()
        df["src_txn_id"] = df["src_txn_id"].astype(int)
        df["dst_txn_id"] = df["dst_txn_id"].astype(int)
        df[FLOW_ID] = range(len(df))
        return df

    def _feature_cols(self, df: pd.DataFrame) -> list[str]:
        return [TIME_STEP
                ] + [c for c in df.columns if c.startswith("feature_")]

    # -- Single-table benchmarks ----------------------------------

    def get_entity_tables(self) -> SupervisedTables:
        txn = self._load_transactions()
        # drops rows without a label
        labeled = txn.dropna(subset=[LABEL]).copy()
        labeled[LABEL] = labeled[LABEL].astype(int)

        keep = self._feature_cols(labeled) + [LABEL]
        labeled = labeled[keep].rename(columns={LABEL: TARGET})

        train = labeled[labeled[TIME_STEP] < TRAIN_TEST_SPLIT_STEP]
        test = labeled[labeled[TIME_STEP] >= TRAIN_TEST_SPLIT_STEP]

        return SupervisedTables(
            train=train.reset_index(drop=True),
            test=test.reset_index(drop=True),
            target_col=TARGET,
            task_type=TaskType.BINARY_CLASSIFICATION,
        )

    def get_joined_tables(self) -> SupervisedTables:
        """Heterogeneous graph, no second table to join.
        Returns the same splits as ``get_entity_tables``.
        """
        return self.get_entity_tables()

    # -- Enterprise benchmark -------------------------------------

    def get_enterprise_graph(self):
        txn = self._load_transactions()
        flows = self._load_payment_flows(txn)

        graph_cols = ([TRANSACTION_ID, "timestamp", TIME_STEP, LABEL] +
                      [c for c in txn.columns if c.startswith("feature_")])
        txn_ent = txn[graph_cols].copy()
        txn_ent[LABEL] = txn_ent[LABEL].fillna(-1).astype(int)

        with tempfile.TemporaryDirectory() as tmpdir:
            txn_path = str(Path(tmpdir) / "transactions.parquet")
            flows_path = str(Path(tmpdir) / "payment_flows.parquet")
            txn_ent.to_parquet(txn_path, index=False)
            flows.to_parquet(flows_path, index=False)

            connector = kumo.FileUploadConnector(file_type="parquet")
            # Always upload so server schema matches this run
            connector.upload(name=TRANSACTIONS_TABLE, path=txn_path)
            connector.upload(name=PAYMENT_FLOWS_TABLE, path=flows_path)

            txn_src = connector.table(TRANSACTIONS_TABLE)
            flows_src = connector.table(PAYMENT_FLOWS_TABLE)
            txn_tbl = kumo.Table.from_source_table(
                txn_src,
                primary_key=TRANSACTION_ID,
                time_column="timestamp",
            ).infer_metadata()
            flows_tbl = kumo.Table.from_source_table(
                flows_src, ).infer_metadata()

            return kumo.Graph(
                tables={
                    TRANSACTIONS_TABLE: txn_tbl,
                    PAYMENT_FLOWS_TABLE: flows_tbl,
                },
                edges=[
                    dict(src_table=PAYMENT_FLOWS_TABLE, fkey="src_txn_id",
                         dst_table=TRANSACTIONS_TABLE),
                    dict(src_table=PAYMENT_FLOWS_TABLE, fkey="dst_txn_id",
                         dst_table=TRANSACTIONS_TABLE),
                ],
            )

    def get_pquery(self) -> str:
        return ENTERPRISE_QUERY.strip()

    def get_enterprise_split(self) -> str | None:
        """TimeRangeSplit for the canonical temporal split:
        train = steps 0..33,
        val = step 34 (one from train),
        test = steps 35..48 (same as time_step >= TRAIN_TEST_SPLIT_STEP).
        """
        train_end = _EPOCH + pd.Timedelta(days=TRAIN_TEST_SPLIT_STEP - 1)
        val_end = _EPOCH + pd.Timedelta(days=TRAIN_TEST_SPLIT_STEP)
        test_end = _EPOCH + pd.Timedelta(days=49)
        return (f'TimeRangeSplit(['
                f'("{_EPOCH.date()}", "{train_end.date()}"), '
                f'("{train_end.date()}", "{val_end.date()}"), '
                f'("{val_end.date()}", "{test_end.date()}")])')
