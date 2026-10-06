# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DBInfer Benchmark (DBB) datasets.

Wraps datasets from the 4DBInfer multi-table benchmark
(https://github.com/awslabs/multi-table-benchmark) into
the BenchmarkingDataset interface.

Data is auto-downloaded on first use from
https://data.dgl.ai/mtbench/.  At runtime only PyYAML,
pandas and numpy are required (no ``dbinfer_bench``
import needed -- avoids the DGL C++ graphbolt
dependency).

See Also:
--------
4DBInfer paper: https://arxiv.org/abs/2404.18209
"""

import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml
from kumoapi.task import TaskType
from tqdm import tqdm

from .benchmarking_dataset import TARGET, BenchmarkingDataset, SupervisedTables

_DBB_SOURCE = "https://data.dgl.ai/mtbench"
_DBB_VERSION = "20240304"

# -- download helpers -----------------------------------------


def _default_cache_dir() -> Path:
    default = Path.home() / ".cache" / "dbinfer"
    return Path(os.environ.get("DBB_DATASET_HOME", default))


def _ensure_downloaded(
    dataset_name: str,
    cache_dir: Path,
) -> Path:
    """Download and extract a DBB dataset if not cached."""
    ds_dir = cache_dir / dataset_name
    version_file = ds_dir / "VERSION"
    if version_file.exists():
        if version_file.read_text().strip() == _DBB_VERSION:
            return ds_dir

    ds_dir.mkdir(parents=True, exist_ok=True)
    tarname = f"{_DBB_VERSION}-{dataset_name}.tar"
    url = f"{_DBB_SOURCE}/{tarname}"
    dl_path = cache_dir / tarname
    print(f"Downloading {dataset_name} from {url} ...")
    resp = requests.get(url, stream=True, verify=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with tqdm(total=total, unit="B", unit_scale=True) as bar:
        with open(dl_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    print(f"Extracting {dl_path} ...")
    with tarfile.open(dl_path, "r:*") as tar:
        tar.extractall(path=cache_dir)
    dl_path.unlink(missing_ok=True)
    version_file.write_text(_DBB_VERSION)
    return ds_dir


# -- base class -----------------------------------------------


class DBBBenchmarkingDataset(BenchmarkingDataset):
    """Base class for all 4DBInfer dataset wrappers.

    Subclasses only need to set ``DATASET_NAME`` and ``TASK_NAME``.
    Everything else is derived from the 4DBInfer ``metadata.yaml`` on disk.
    """

    DATASET_NAME: str
    TASK_NAME: str

    def __init__(
        self,
        data_root: str | Path | None = None,
        **kwargs,
    ) -> None:
        if data_root is not None:
            self._ds_dir = Path(data_root)
        else:
            self._ds_dir = _ensure_downloaded(
                self.DATASET_NAME,
                _default_cache_dir(),
            )
        with open(self._ds_dir / "metadata.yaml") as f:
            self._meta: dict = yaml.safe_load(f)
        self._task_meta = self._find_task()
        self._task_type_cache: TaskType | None = None

    # -- metadata helpers -------------------------------------

    def _find_task(self) -> dict:
        for t in self._meta["tasks"]:
            if t["name"] == self.TASK_NAME:
                return t
        avail = [t["name"] for t in self._meta["tasks"]]
        raise ValueError(f"Task {self.TASK_NAME!r} not in {avail}")

    def _table_meta(self, name: str) -> dict:
        for t in self._meta["tables"]:
            if t["name"] == name:
                return t
        raise ValueError(f"Table {name!r} not found")

    @property
    def _target_column_table_name(self) -> str:
        """Table that contains the target column."""
        return self._task_meta["target_table"]

    @property
    def _target_col(self) -> str:
        return self._task_meta["target_column"]

    # -- data loading -----------------------------------------

    def _excluded_columns(self) -> dict[str, set[str]]:
        """Per-table columns to drop from task splits to avoid leakage.

        Keys are table names, values are sets of column names. Base
        implementation returns an empty mapping; subclasses may override.
        """
        return {}

    def _load_table(self, name: str) -> pd.DataFrame:
        tbl = self._table_meta(name)
        src = self._ds_dir / tbl["source"]
        fmt = tbl.get("format", "parquet")
        if fmt == "parquet":
            df = pd.read_parquet(src)
        else:
            df = pd.DataFrame({k: v for k, v in np.load(src).items()})

        # Restrict to only the columns declared in the metadata for this table.
        # Parquet files often contain extra columns (e.g. Score, LastAccessDate
        # in StackExchange) that are lifetime aggregates updated well after the
        # prediction anchor — keeping them leaks future information.
        meta_cols = {c["name"] for c in tbl["columns"]}
        extra_cols = [c for c in df.columns if c not in meta_cols]
        if extra_cols:
            print(
                f"[{self.DATASET_NAME}/{name}] dropping columns: {extra_cols}")
            df = df.drop(columns=extra_cols)

        # Drop any per-table columns marked as excluded by subclasses.
        excluded = self._excluded_columns().get(name, set())
        if excluded:
            drop_cols = [c for c in excluded if c in df.columns]
            if drop_cols:
                df = df.drop(columns=drop_cols)
        return df

    def _load_split(self, split: str) -> pd.DataFrame:
        src = self._task_meta["source"]
        path = self._ds_dir / src.format(split=split)
        fmt = self._task_meta.get("format", "parquet")
        if fmt == "parquet":
            df = pd.read_parquet(path)
        else:
            npz = np.load(path)
            df = pd.DataFrame({k: v for k, v in npz.items()})

        # Drop any columns whose names are marked as excluded by subclasses,
        # regardless of their originating table. This is safer for pre-joined
        # task splits where table identity may not be explicit.
        excl_map = self._excluded_columns()
        if excl_map:
            all_excluded: set[str] = set()
            for cols in excl_map.values():
                all_excluded.update(cols)
            # Never drop the target column from splits; it is the label.
            target = self._target_col
            if target in all_excluded:
                all_excluded.remove(target)
            drop_cols = [c for c in all_excluded if c in df.columns]
            if drop_cols:
                df = df.drop(columns=drop_cols)
        return df

    # -- type inference ---------------------------------------

    def _infer_task_type(self) -> TaskType:
        if self._task_type_cache is not None:
            return self._task_type_cache

        task_kind = self._task_meta.get("task_type")
        if task_kind == "regression":
            tt = TaskType.REGRESSION
        elif task_kind == "retrieval":
            tt = TaskType.LINK_PREDICTION
        else:
            train = self._load_split("train")
            n_cls = train[self._target_col].nunique()
            if n_cls <= 2:
                tt = TaskType.BINARY_CLASSIFICATION
            else:
                tt = TaskType.MULTICLASS_CLASSIFICATION

        self._task_type_cache = tt
        return tt

    def _build_class_map(
        self,
        *dfs: pd.DataFrame,
    ) -> dict | None:
        """Build str->int class map if target is
        non-numeric.  Returns None if already numeric.
        """
        if self._task_meta.get("task_type") == "regression":
            return None
        sample = dfs[0][self._target_col]
        if np.issubdtype(sample.dtype, np.number):
            return None
        all_vals: set = set()
        for df in dfs:
            all_vals.update(df[self._target_col].unique())
        return {v: i for i, v in enumerate(sorted(all_vals))}

    # -- column helpers for tabular ---------------------------

    def _col_dtypes(self) -> dict[str, str]:
        return {c["name"]: c["dtype"] for c in self._task_meta["columns"]}

    def _prepare_tabular(
        self,
        df: pd.DataFrame,
        class_map: dict | None = None,
    ) -> pd.DataFrame:
        """Prepare a task-split DF for tabular models.

        - Drop primary_key columns
        - Convert datetime columns to int64 (ns epoch)
        - Encode string target to int (if class_map)
        - Rename target -> TARGET
        """
        df = df.copy()
        dtypes = self._col_dtypes()
        for col, dt in dtypes.items():
            if col not in df.columns:
                continue
            if dt == "datetime":
                df[col] = (pd.to_datetime(df[col]).astype("int64"))
            elif dt == "primary_key" or dt == "foreign_key":
                df = df.drop(columns=[col])
        if class_map is not None:
            df[self._target_col] = (df[self._target_col].map(class_map))
        df = df.rename(columns={self._target_col: TARGET})
        return df

    # -- single-table benchmarks ------------------------------

    def _build_supervised_tables(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
    ) -> SupervisedTables:
        """Common helper to build SupervisedTables from splits."""
        task_type = self._infer_task_type()
        class_map = self._build_class_map(train, val, test)

        train_full = pd.concat([train, val], ignore_index=True)
        train_df = self._prepare_tabular(train_full, class_map)
        test_df = self._prepare_tabular(test, class_map)

        return SupervisedTables(
            train=train_df.reset_index(drop=True),
            test=test_df.reset_index(drop=True),
            target_col=TARGET,
            task_type=task_type,
        )

    def get_entity_tables(self) -> SupervisedTables:
        train = self._load_split("train")
        val = self._load_split("validation")
        test = self._load_split("test")

        return self._build_supervised_tables(train, val, test)

    def _one_hop_links(self) -> list[tuple[str, str, str]]:
        """Tables one hop from the task: (table_name, fk_col, pk_col).

        Uses task columns with dtype foreign_key whose link_to table
        exists in the dataset. Returns list of (tbl_name, fk_col, pk_col)
        for left-join: task[fk_col] = tbl[pk_col].
        """
        table_names = {t["name"] for t in self._meta["tables"]}
        out: list[tuple[str, str, str]] = []
        for c in self._task_meta["columns"]:
            if c["dtype"] != "foreign_key":
                continue
            parts = c["link_to"].split(".")
            if len(parts) != 2:
                continue
            pk_table, pk_col = parts[0], parts[1]
            if pk_table not in table_names:
                continue
            out.append((pk_table, c["name"], pk_col))
        return out

    def _join_one_hop(
        self,
        df: pd.DataFrame,
        one_hop: list[tuple[str, str, str]],
    ) -> pd.DataFrame:
        """Left-join df with each one-hop table on (fk_col, pk_col)."""
        for tbl_name, fk_col, pk_col in one_hop:
            if fk_col not in df.columns:
                continue
            other = self._load_table(tbl_name)
            df = df.merge(
                other,
                left_on=fk_col,
                right_on=pk_col,
                how="left",
                suffixes=("", "_join"),
            )
            if pk_col != fk_col and pk_col in df.columns:
                df = df.drop(columns=[pk_col])
            drop_join = [c for c in df.columns if c.endswith("_join")]
            if drop_join:
                df = df.drop(columns=drop_join)
        return df

    def get_joined_tables(self) -> SupervisedTables:
        train = self._load_split("train")
        val = self._load_split("validation")
        test = self._load_split("test")

        one_hop = self._one_hop_links()
        if one_hop:
            train = self._join_one_hop(train, one_hop)
            val = self._join_one_hop(val, one_hop)
            test = self._join_one_hop(test, one_hop)

        return self._build_supervised_tables(train, val, test)

    # -- enterprise (not implemented) -------------------------

    def get_pquery(self) -> str:
        raise NotImplementedError("Enterprise pquery not implemented for "
                                  f"DBB/{self.DATASET_NAME}")

    def get_enterprise_graph(self):
        raise NotImplementedError("Enterprise graph not implemented for "
                                  f"DBB/{self.DATASET_NAME}")


# -- relationship-attribute base class ------------------------


class DBBRelationshipBenchmarkingDataset(DBBBenchmarkingDataset):
    """Base class for 4DBInfer relationship-attribute tasks.

    In a relationship-attribute task the prediction target lives on an
    interaction/edge row that has two (or more) foreign keys and no
    natural primary key.

    Subclasses only need to set ``DATASET_NAME`` and ``TASK_NAME``.
    """


# -- concrete datasets ----------------------------------------


class DBBAVSRepeater(DBBRelationshipBenchmarkingDataset):
    """AVS / repeater: binary classification (customer retention)."""

    DATASET_NAME = "avs"
    TASK_NAME = "repeater"


class DBBOutbrainCTR(DBBRelationshipBenchmarkingDataset):
    """Outbrain-small / ctr: binary classification (ad click-through)."""

    DATASET_NAME = "outbrain-small"
    TASK_NAME = "ctr"


class DBBDigineticaCTR(DBBRelationshipBenchmarkingDataset):
    """Diginetica / ctr: binary classification (item click)."""

    DATASET_NAME = "diginetica"
    TASK_NAME = "ctr"


class DBBDigineticaPurchase(DBBBenchmarkingDataset):
    """Diginetica / purchase: retrieval (item purchase ranking)."""

    DATASET_NAME = "diginetica"
    TASK_NAME = "purchase"


class DBBRetailRocketCVR(DBBRelationshipBenchmarkingDataset):
    """RetailRocket / cvr: binary classification (add-to-cart)."""

    DATASET_NAME = "retailrocket"
    TASK_NAME = "cvr"


class DBBAmazonChurn(DBBBenchmarkingDataset):
    """Amazon / churn: binary classification (user churn)."""

    DATASET_NAME = "amazon"
    TASK_NAME = "churn"


class DBBAmazonRating(DBBRelationshipBenchmarkingDataset):
    """Amazon / rating: regression (review rating 1–5)."""

    DATASET_NAME = "amazon"
    TASK_NAME = "rating"

    def _excluded_columns(self) -> dict[str, set[str]]:
        # Avoid information leakage from contemporaneous review text fields.
        return {
            "Review": {
                "review_text",
                "summary",
            },
        }


class DBBAmazonPurchase(DBBBenchmarkingDataset):
    """Amazon / purchase: retrieval (product purchase ranking)."""

    DATASET_NAME = "amazon"
    TASK_NAME = "purchase"


class DBBStackExchangeChurn(DBBBenchmarkingDataset):
    """StackExchange / churn: binary classification (user activity)."""

    DATASET_NAME = "stackexchange"
    TASK_NAME = "churn"


class DBBStackExchangeUpvote(DBBBenchmarkingDataset):
    """StackExchange / upvote: binary classification (post popularity)."""

    DATASET_NAME = "stackexchange"
    TASK_NAME = "upvote"


class DBBMAGVenue(DBBBenchmarkingDataset):
    """MAG / venue: multiclass classification (349 venues)."""

    DATASET_NAME = "mag"
    TASK_NAME = "venue"

    def _excluded_columns(self) -> dict[str, set[str]]:
        # Avoid using the venue label column of the same paper as a feature.
        base = super()._excluded_columns()
        out: dict[str, set[str]] = {k: set(v) for k, v in base.items()}
        tbl = self._target_column_table_name
        col = self._target_col
        out.setdefault(tbl, set()).add(col)
        return out


class DBBMAGCite(DBBBenchmarkingDataset):
    """MAG / cite: retrieval (citation target ranking)."""

    DATASET_NAME = "mag"
    TASK_NAME = "cite"


class DBBSeznamCharge(DBBBenchmarkingDataset):
    """Seznam / charge: multiclass classification (8 charge types).

    For each (client, timestamp) we predict Probehnuto.sluzba (joined).
    Target column lives in Probehnuto.
    """

    DATASET_NAME = "seznam"
    TASK_NAME = "charge"


class DBBSeznamPrepay(DBBBenchmarkingDataset):
    """Seznam / prepay: multiclass classification (8 prepay types).

    Prediction is per (client, timestamp), analogous to DBBSeznamCharge.
    """

    DATASET_NAME = "seznam"
    TASK_NAME = "prepay"
