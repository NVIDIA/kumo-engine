# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FiGraph: dynamic heterogeneous graph for financial anomaly detection.

FiGraph (https://github.com/XiaoguangWang23/FiGraph) provides snapshots
by year with five node types (Listed Company, Unlisted Company, Audit
Institution, Human, Other Entity) and four edge types (Investment,
Related-party Transaction, Supply Chain, Audit). Only listed companies
have attributes and labels; the rest are background nodes. Binary
classification: predict financial fraud (anomaly) vs normal.

Data is read from S3 by default:
s3://kumo-public-datasets/figraph/{year}/
  - listed_companies.parquet
  - edges.parquet

This module loads all snapshots 2014–2022 and exposes:
- a single-table view over listed companies with a temporal split
  (train: 2014–2020, test: 2021–2022)
- a heterogeneous temporal graph with five entity tables and one
  temporal feature table for listed companies.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import SupportsInt, cast

import pandas as pd
from kumoapi.task import TaskType

import kumoai_studio as kumo

from .benchmarking_dataset import TARGET, BenchmarkingDataset, SupervisedTables
from .figraph_common import (
    EDGES_PARQUET,
    ENTITY_TABLE_BY_PREFIX,
    FIGRAPH_ENTERPRISE_QUERY,
    FIGRAPH_S3_BASE,
    LISTED_FEATURES_TABLE,
    LISTED_PARQUET,
    LISTED_TABLE,
    MAX_YEAR,
    MIN_YEAR,
    PREFIX_A,
    PREFIX_H,
    PREFIX_O,
    PREFIX_ORDER,
    PREFIX_U,
    TEST_START_YEAR,
    TRAIN_END_YEAR,
    node_prefix,
)


class FiGraphBenchmarkingDataset(BenchmarkingDataset):
    """FiGraph dataset for single-table and Enterprise benchmarks."""

    TARGET_COLUMN = "Label"

    def __init__(self, data_root: str | Path | None = None) -> None:
        """Args:
        data_root: Directory or S3 prefix containing per-year subdirs:
            {root}/{year}/listed_companies.parquet
            {root}/{year}/edges.parquet
            Default: s3://kumo-public-datasets/figraph
        """
        default = FIGRAPH_S3_BASE
        self._set_data_root(data_root, default)
        if not self._data_root_str.startswith("s3://"):
            base = Path(self._data_root_str)
            probe = base / str(MIN_YEAR) / LISTED_PARQUET
            if not probe.exists():
                raise FileNotFoundError(
                    f"FiGraph parquet not found at {probe}. "
                    "Run download_figraph for this year range or use an "
                    "S3 data_root.")

    # -- Helpers ----------------------------------------------------

    def _read_year_parquet(self, year: int, name: str) -> pd.DataFrame:
        """Read a parquet file from a specific snapshot year."""
        root_str = getattr(self, "_data_root_str", "")
        root_path = getattr(self, "_data_root", None)
        if root_str.startswith("s3://"):
            return pd.read_parquet(f"{root_str}/{year}/{name}")
        if root_path is None:
            root_path = Path(root_str)
        return pd.read_parquet(root_path / str(year) / name)

    def _load_all_listed(self) -> pd.DataFrame:
        """Load listed companies for all years, keep Year metadata."""
        frames: list[pd.DataFrame] = []
        for year in range(MIN_YEAR, MAX_YEAR + 1):
            try:
                df_year = self._read_year_parquet(year, LISTED_PARQUET)
            except FileNotFoundError:
                raise FileNotFoundError(
                    "Missing FiGraph listed_companies.parquet for year "
                    f"{year} under {self._data_root_str}.") from None
            if "Year" not in df_year.columns:
                df_year = df_year.copy()
                df_year["Year"] = year
            if ("nodeID" in df_year.columns
                    and "node_id" not in df_year.columns):
                df_year = df_year.rename(columns={"nodeID": "node_id"})
            frames.append(df_year)
        if not frames:
            raise FileNotFoundError(
                f"No listed companies data under {self._data_root_str}.")
        df = pd.concat(frames, ignore_index=True)
        return df

    def _add_timestamp_per_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a synthetic timestamp column based on Year and row order."""
        if "Year" not in df.columns:
            raise ValueError("FiGraph listed companies missing Year column.")
        frames: list[pd.DataFrame] = []
        for year, grp in df.groupby("Year", sort=True):
            grp = grp.reset_index(drop=True)
            year_int = cast(SupportsInt, year)
            epoch = pd.Timestamp(
                year=int(year_int),
                month=1,
                day=1,
            )
            grp = grp.copy()
            grp["timestamp"] = epoch + pd.to_timedelta(grp.index, unit="D")
            frames.append(grp)
        return pd.concat(frames, ignore_index=True)

    def _load_all_listed_with_timestamp(self) -> pd.DataFrame:
        """All listed-company rows across snapshots with timestamp."""
        df = self._load_all_listed()
        df = self._add_timestamp_per_year(df)
        return df

    def _read_edges_parquet(self) -> pd.DataFrame:
        """Load and concatenate edges for all years, tagging with Year."""
        frames: list[pd.DataFrame] = []
        for year in range(MIN_YEAR, MAX_YEAR + 1):
            try:
                df_year = self._read_year_parquet(year, EDGES_PARQUET)
            except FileNotFoundError:
                raise FileNotFoundError(
                    "Missing FiGraph edges.parquet for year "
                    f"{year} under {self._data_root_str}.") from None
            if df_year.shape[1] < 3:
                raise ValueError(
                    "FiGraph edges parquet must have at least 3 columns, "
                    f"got {df_year.shape[1]} for year {year}.")
            df_year = df_year.iloc[:, :3].copy()
            df_year.columns = ["node1", "node2", "edge_type"]
            df_year["Year"] = year
            frames.append(df_year)
        if not frames:
            raise FileNotFoundError(
                f"No FiGraph edges data under {self._data_root_str}.")
        return pd.concat(frames, ignore_index=True)

    # -- Single-table benchmarks ----------------------------------

    def get_entity_tables(self) -> SupervisedTables:
        """Listed companies only over all years.

        Tabular features + binary Label (fraud vs normal), with a
        temporal split by year:
        - train: Years 2014–2020
        - test: Years 2021–2022
        """
        df = self._load_all_listed_with_timestamp()
        if self.TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column {self.TARGET_COLUMN!r} not in "
                             f"FiGraph columns: {list(df.columns)}")
        df = df.copy()
        df[self.TARGET_COLUMN] = pd.to_numeric(
            df[self.TARGET_COLUMN],
            errors="coerce",
        ).astype("Int64")
        df = df.dropna(subset=[self.TARGET_COLUMN])

        train_mask = (df["Year"] >= MIN_YEAR) & (df["Year"] <= TRAIN_END_YEAR)
        test_mask = df["Year"] >= TEST_START_YEAR
        train = df[train_mask]
        test = df[test_mask]
        if train.empty or test.empty:
            raise ValueError(
                "FiGraph train/test split is empty; check data and years.")
        train = train.rename(columns={self.TARGET_COLUMN: TARGET})
        test = test.rename(columns={self.TARGET_COLUMN: TARGET})
        train = train.drop(columns=["timestamp"], errors="ignore")
        test = test.drop(columns=["timestamp"], errors="ignore")
        return SupervisedTables(
            train=train,
            test=test,
            target_col=TARGET,
            task_type=TaskType.BINARY_CLASSIFICATION,
        )

    def get_joined_tables(self) -> SupervisedTables:
        return self.get_entity_tables()

    # -- Enterprise graph construction ----------------------------

    def _build_graph_tables(
        self,
        include_label_in_features: bool = False,
    ) -> tuple[dict[str, pd.DataFrame], list[tuple[str, str, str]],
               pd.DataFrame, dict[str, pd.DataFrame]]:
        """Build df_dict and graph edges for the relational graph.

        Returns:
        - df_dict: dictionary of dataframes for each table
        - graph_edges: list of (table_name, fkey, dst_table) tuples
        - listed_features: dataframe of listed company features
        - edge_dfs: dictionary of dataframes for each edge table
        """
        listed_all = self._load_all_listed_with_timestamp()
        drop = [
            c for c in listed_all.columns
            if "mda" in c.lower() or c == "ManaDiscAnal"
        ]
        listed_all = listed_all.drop(columns=drop, errors="ignore")
        if "node_id" not in listed_all.columns:
            raise ValueError("FiGraph listed companies missing node_id.")
        listed_entities = (listed_all[[
            "node_id"
        ]].drop_duplicates().reset_index(drop=True))
        if include_label_in_features:
            listed_features = listed_all.reset_index(drop=True)
        else:
            listed_features = listed_all.drop(
                columns=[self.TARGET_COLUMN],
                errors="ignore").reset_index(drop=True)
        listed_features["feature_row_id"] = range(len(listed_features))

        edges_df = self._read_edges_parquet()
        edges_df["node1"] = edges_df["node1"].astype(str).str.strip()
        edges_df["node2"] = edges_df["node2"].astype(str).str.strip()
        edges_df["_p1"] = edges_df["node1"].map(node_prefix)
        edges_df["_p2"] = edges_df["node2"].map(node_prefix)
        edges_df = edges_df[edges_df["_p1"].astype(bool)
                            & edges_df["_p2"].astype(bool)].copy()

        all_node_ids = pd.concat([edges_df["node1"],
                                  edges_df["node2"]]).drop_duplicates()
        all_node_ids = all_node_ids.astype(str).str.strip().str.upper()

        def partition_by_prefix(series: pd.Series) -> dict[str, pd.DataFrame]:
            s = series.astype(str).str.strip().str.upper()
            out: dict[str, pd.DataFrame] = {}
            for p in PREFIX_ORDER:
                mask = s.str[0] == p
                if mask.any():
                    out[p] = pd.DataFrame({"node_id": series[mask].unique()})
            return out

        partitions = partition_by_prefix(all_node_ids)

        df_dict: dict[str, pd.DataFrame] = {}
        df_dict[LISTED_TABLE] = listed_entities
        df_dict[LISTED_FEATURES_TABLE] = listed_features
        for p in (PREFIX_U, PREFIX_A, PREFIX_H, PREFIX_O):
            if p in partitions and not partitions[p].empty:
                df_dict[ENTITY_TABLE_BY_PREFIX[p]] = partitions[p]

        edge_dfs: dict[str, pd.DataFrame] = {}
        for (_p1, _p2), grp in edges_df.groupby(["_p1", "_p2"]):
            et_name = f"edges_{_p1}_{_p2}"
            tbl = grp[["node1", "node2", "edge_type", "Year"]].copy()
            tbl = tbl.rename(columns={"node1": "src_id", "node2": "dst_id"})
            tbl["timestamp"] = pd.Timestamp(year=int(grp["Year"].iloc[0]),
                                            month=1, day=1)
            tbl["edge_id"] = range(len(tbl))
            edge_dfs[et_name] = tbl
        for et_name, tbl in edge_dfs.items():
            df_dict[et_name] = tbl

        graph_edges: list[tuple[str, str, str]] = []
        for et_name in edge_dfs:
            _, src_prefix, dst_prefix = et_name.split("_", 2)
            src_table = ENTITY_TABLE_BY_PREFIX[src_prefix]
            dst_table = ENTITY_TABLE_BY_PREFIX[dst_prefix]
            if src_table in df_dict and dst_table in df_dict:
                graph_edges.append((et_name, "src_id", src_table))
                graph_edges.append((et_name, "dst_id", dst_table))
        graph_edges.append((LISTED_FEATURES_TABLE, "node_id", LISTED_TABLE))

        return (df_dict, graph_edges, listed_features, edge_dfs)

    # -- Enterprise benchmark -------------------------------------

    def get_enterprise_graph(self):
        """Build the Kumo Graph over the shared table set: same tables
        and edges as the local relational graph.

        Includes listed_companies, listed_company_features (with Label for
        training), U/A/H/O entity tables, and all directed edge tables.
        """
        df_dict, graph_edges, _listed_features, edge_dfs = (
            self._build_graph_tables(include_label_in_features=True))
        kumo_edges = [
            dict(src_table=src_tab, fkey=fk, dst_table=dst_tab)
            for src_tab, fk, dst_tab in graph_edges
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            connector = kumo.FileUploadConnector(file_type="parquet")
            tables = {}
            for name, df in df_dict.items():
                path = str(tmp / f"{name}.parquet")
                df.to_parquet(path, index=False)
                connector.upload(name=name, path=path)
                src = connector.table(name)
                pk = ("feature_row_id" if name == LISTED_FEATURES_TABLE else
                      "edge_id" if name in edge_dfs else "node_id")
                time_col = ("timestamp" if name in (LISTED_FEATURES_TABLE,
                                                    *edge_dfs) else None)
                tbl = kumo.Table.from_source_table(
                    src, primary_key=pk,
                    time_column=time_col).infer_metadata()
                tables[name] = tbl
            return kumo.Graph(tables=tables, edges=kumo_edges)

    def get_pquery(self) -> str:
        """PQL for fraud prediction per company-year, matching task table."""
        return FIGRAPH_ENTERPRISE_QUERY.strip()

    def get_enterprise_split(self) -> str | None:
        """TimeRangeSplit matching task table:
        - train 2014–2019,
        - val 2020,
        - test 2021–2022.
        """
        return ('TimeRangeSplit(['
                f'("{MIN_YEAR}-01-01", "{TRAIN_END_YEAR - 1}-12-31"), '
                f'("{TRAIN_END_YEAR}-01-01", "{TRAIN_END_YEAR}-12-31"), '
                f'("{TEST_START_YEAR}-01-01", "{MAX_YEAR}-12-31")])')
