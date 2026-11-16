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
- a projected graph with only listed entities: one entity table, one
  temporal feature table, and one projected L-L edge table (direct
  L-L plus 2-hop L-X-L via U/A/H/O).
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pandas as pd

import kumoai_studio as kumo

from .benchmarking_dataset import BenchmarkingDataset, SupervisedTables
from .figraph import FiGraphBenchmarkingDataset
from .figraph_common import (
    EDGES_PARQUET,
    FIGRAPH_ENTERPRISE_QUERY,
    FIGRAPH_S3_BASE,
    LISTED_FEATURES_TABLE,
    LISTED_PARQUET,
    LISTED_TABLE,
    MAX_YEAR,
    MIN_YEAR,
    PREFIX_A,
    PREFIX_H,
    PREFIX_L,
    PREFIX_O,
    PREFIX_U,
    TEST_START_YEAR,
    node_prefix,
)

# Single edge table for L-L projected graph.
LISTED_EDGES_TABLE = "listed_edges"

# Projection path labels for 2-hop edges (middle node type).
PROJECTION_PATH_BY_PREFIX = {
    PREFIX_U: "L-U-L",
    PREFIX_A: "L-A-L",
    PREFIX_H: "L-H-L",
    PREFIX_O: "L-O-L",
}

# Cache subdir for graph tables (under user cache).
GRAPH_CACHE_SUBDIR = "graph_tables"


def _graph_cache_dir(data_root: str) -> Path:
    """Return cache directory for graph tables for this data_root."""
    key = hashlib.sha256(str(data_root).encode()).hexdigest()[:16]
    return Path.home() / ".cache" / "figraph" / GRAPH_CACHE_SUBDIR / key


def _graph_cache_exists(cache_dir: Path) -> bool:
    """True if all three parquet files exist in cache."""
    return ((cache_dir / "listed_entities.parquet").exists()
            and (cache_dir / "listed_company_features.parquet").exists()
            and (cache_dir / "listed_edges.parquet").exists())


def _build_projected_ll_edges(
    edges_df: pd.DataFrame,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Build a single L-L edge table from raw edges, year by year.

    - Direct L-L: each raw edge with both endpoints L becomes one row
      (both directions) with projection_path="L-L".
    - 2-hop: only pairs where both edges share the same year (year1 ==
      year2); group by (middle node, year), then pair L neighbors.

    Processes one year at a time to limit memory. If cache_dir is set,
    writes each year's projected edges to
    cache_dir / "listed_edges_{year}.parquet", then concatenates from
    disk. Otherwise accumulates per-year DataFrames and concats in memory.

    Expects edges_df to have: node1, node2, edge_type, Year, _p1, _p2.
    Returns DataFrame: src_id, dst_id, year, timestamp, edge_id,
    projection_path, edge_type.
    """
    years = sorted(edges_df["Year"].dropna().unique().astype(int).tolist())
    if not years:
        return pd.DataFrame(columns=pd.Index([
            "src_id",
            "dst_id",
            "year",
            "timestamp",
            "edge_id",
            "projection_path",
            "edge_type",
        ]))

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    year_dfs: list[pd.DataFrame] = []
    non_l_prefixes = (PREFIX_U, PREFIX_A, PREFIX_H, PREFIX_O)

    for year in years:
        print(f"Processing edges for year {year}")
        edges_y = edges_df[edges_df["Year"] == year]
        rows: list[dict] = []

        # Direct L-L for this year.
        ll = edges_y[(edges_y["_p1"] == PREFIX_L)
                     & (edges_y["_p2"] == PREFIX_L)]
        for _, r in ll.iterrows():
            src, dst = r["node1"], r["node2"]
            etype = str(r["edge_type"]).strip()
            for s, d in [(src, dst), (dst, src)]:
                rows.append({
                    "src_id": s,
                    "dst_id": d,
                    "year": year,
                    "projection_path": "L-L",
                    "edge_type": etype,
                })

        # 2-hop for this year only (same year for both edges).
        for mid_prefix in non_l_prefixes:
            path_label = PROJECTION_PATH_BY_PREFIX[mid_prefix]
            mask_l_first = ((edges_y["_p1"] == PREFIX_L)
                            & (edges_y["_p2"] == mid_prefix))
            mask_l_second = ((edges_y["_p1"] == mid_prefix)
                             & (edges_y["_p2"] == PREFIX_L))
            subset = edges_y[mask_l_first | mask_l_second].copy()
            if subset.empty:
                continue
            subset["_l"] = subset.apply(
                lambda r: (r["node1"] if r["_p1"] == PREFIX_L else r["node2"]),
                axis=1,
            )
            subset["_mid"] = subset.apply(
                lambda r: (r["node2"] if r["_p1"] == PREFIX_L else r["node1"]),
                axis=1,
            )
            subset["_t"] = subset["edge_type"].astype(str).str.strip()
            for _mid, grp in subset.groupby("_mid"):
                neighbors = list(zip(grp["_l"], grp["_t"]))
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        l1, t1 = neighbors[i]
                        l2, t2 = neighbors[j]
                        if l1 == l2:
                            continue
                        etype = f"{t1}"
                        rows.append({
                            "src_id": l1,
                            "dst_id": l2,
                            "year": year,
                            "projection_path": path_label,
                            "edge_type": etype,
                        })
                        rows.append({
                            "src_id": l2,
                            "dst_id": l1,
                            "year": year,
                            "projection_path": path_label,
                            "edge_type": etype,
                        })

        if not rows:
            continue
        df_y = pd.DataFrame(rows)
        if cache_dir is not None:
            path = cache_dir / f"listed_edges_{year}.parquet"
            print(f"Saving {path} with {len(df_y)} edges for year {year}.")
            df_y.to_parquet(path, index=False)
        else:
            year_dfs.append(df_y)

    if cache_dir is not None:
        print(f"Reading cached edges from {cache_dir} for {len(years)} years")
        year_dfs = [
            pd.read_parquet(cache_dir / f"listed_edges_{y}.parquet")
            for y in years
            if (cache_dir / f"listed_edges_{y}.parquet").exists()
        ]
    if not year_dfs:
        return pd.DataFrame(columns=pd.Index([
            "src_id",
            "dst_id",
            "year",
            "timestamp",
            "edge_id",
            "projection_path",
            "edge_type",
        ]))

    out = pd.concat(year_dfs, ignore_index=True)
    print(f"Concatenated {len(out)} edges from {len(year_dfs)} years")
    out["timestamp"] = out["year"].apply(
        lambda y: pd.Timestamp(year=int(y), month=1, day=1))
    out["edge_id"] = range(len(out))
    return out


class FiGraphProjectedBenchmarkingDataset(BenchmarkingDataset):
    """FiGraph dataset for Enterprise benchmarks on the projected graph."""

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
        """Add a synthetic timestamp column: Jan 1 of each row's Year.

        Keeps the temporal range 2014-01-01 to 2022-01-01 (no row-index
        offset that would push the range beyond the data years).
        """
        if "Year" not in df.columns:
            raise ValueError("FiGraph listed companies missing Year column.")
        df = df.copy()
        df["timestamp"] = df["Year"].apply(
            lambda y: pd.Timestamp(year=int(y), month=1, day=1))
        return df

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
        """Single-table view reused from FiGraphBenchmarkingDataset.

        This dataset focuses on the projected L-L graph, but for
        compatibility with utilities that expect a single-table view
        (e.g. metric selection in the enterprise runner), delegate to
        the base FiGraph dataset implementation.
        """
        base = FiGraphBenchmarkingDataset(self._data_root_str)
        return base.get_entity_tables()

    def get_joined_tables(self) -> SupervisedTables:
        return self.get_entity_tables()

    # -- Enterprise graph construction ------------------------------

    def _build_graph_tables(
        self,
        include_label_in_features: bool = False,
    ) -> tuple[dict[str, pd.DataFrame], list[tuple[str, str, str]],
               pd.DataFrame]:
        """Build df_dict and graph_edges for the projected L-L graph.

        Returns (df_dict, graph_edges, listed_features). Cache stores
        listed_company_features with Label; if include_label_in_features
        is False, Label is dropped from the returned listed_features.
        """
        cache_dir = _graph_cache_dir(self._data_root_str)
        listed_entities = listed_features = listed_edges = None
        if _graph_cache_exists(cache_dir):
            print(f"Loading cached graph tables from {cache_dir}")
            listed_entities = pd.read_parquet(cache_dir /
                                              "listed_entities.parquet")
            listed_features = pd.read_parquet(
                cache_dir / "listed_company_features.parquet")
            listed_edges = pd.read_parquet(cache_dir / "listed_edges.parquet")
        need_rebuild = listed_entities is None
        if not need_rebuild and include_label_in_features:
            assert listed_features is not None
            if self.TARGET_COLUMN not in listed_features.columns:
                need_rebuild = True
        if need_rebuild:
            if listed_entities is None:
                print("Building graph tables from scratch")
            else:
                print("Rebuilding (cache missing Label for Enterprise)")
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
            listed_features = listed_all.reset_index(drop=True)
            listed_features["feature_row_id"] = range(len(listed_features))
            edges_df = self._read_edges_parquet()
            edges_df["node1"] = edges_df["node1"].astype(str).str.strip()
            edges_df["node2"] = edges_df["node2"].astype(str).str.strip()
            edges_df["_p1"] = edges_df["node1"].map(node_prefix)
            edges_df["_p2"] = edges_df["node2"].map(node_prefix)
            edges_df = edges_df[edges_df["_p1"].astype(bool)
                                & edges_df["_p2"].astype(bool)].copy()
            cache_dir.mkdir(parents=True, exist_ok=True)
            listed_edges = _build_projected_ll_edges(edges_df,
                                                     cache_dir=cache_dir)
            listed_entities.to_parquet(cache_dir / "listed_entities.parquet",
                                       index=False)
            listed_features.to_parquet(
                cache_dir / "listed_company_features.parquet", index=False)
            listed_edges.to_parquet(cache_dir / "listed_edges.parquet",
                                    index=False)
        assert listed_entities is not None
        assert listed_features is not None
        assert listed_edges is not None
        if not include_label_in_features:
            listed_features = listed_features.drop(
                columns=[self.TARGET_COLUMN], errors="ignore")
        df_dict = {
            LISTED_TABLE: listed_entities,
            LISTED_FEATURES_TABLE: listed_features,
            LISTED_EDGES_TABLE: listed_edges,
        }
        graph_edges = [
            (LISTED_EDGES_TABLE, "src_id", LISTED_TABLE),
            (LISTED_EDGES_TABLE, "dst_id", LISTED_TABLE),
            (LISTED_FEATURES_TABLE, "node_id", LISTED_TABLE),
        ]
        return (df_dict, graph_edges, listed_features)

    # -- Enterprise benchmark -------------------------------------

    def get_enterprise_graph(self):
        """Build Kumo Graph over the projected L-L table set.

        Same schema: listed_companies, listed_company_features (with
        Label), listed_edges. Uses _build_graph_tables(True).
        """
        df_dict, graph_edges, _ = (self._build_graph_tables(
            include_label_in_features=True))
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
                if name == LISTED_FEATURES_TABLE:
                    pk, time_col = "feature_row_id", "timestamp"
                elif name == LISTED_EDGES_TABLE:
                    pk, time_col = "edge_id", "timestamp"
                else:
                    pk, time_col = "node_id", None
                tbl = kumo.Table.from_source_table(
                    src, primary_key=pk,
                    time_column=time_col).infer_metadata()
                tables[name] = tbl
            return kumo.Graph(tables=tables, edges=kumo_edges)

    def get_pquery(self) -> str:
        """PQL for fraud prediction per company-year, matching task table."""
        return FIGRAPH_ENTERPRISE_QUERY.strip()

    def get_enterprise_split(self) -> str | None:
        """TimeRangeSplit from the same table the PQL uses (Label >= 0).

        Uses the listed_company_features table (with Label and timestamp)
        to choose train/val/test years such that both train and val have
        at least one row under the same filter as the PQL. Returns None
        only if there are fewer than two non-empty pre-test years.
        """
        _, _, listed_features = self._build_graph_tables(
            include_label_in_features=True, )
        if (self.TARGET_COLUMN not in listed_features.columns
                or "timestamp" not in listed_features.columns):
            return None

        mask = pd.to_numeric(
            listed_features[self.TARGET_COLUMN],
            errors="coerce",
        ).notna()
        ts = pd.to_datetime(
            listed_features.loc[mask]["timestamp"],
            utc=False,
        )

        test_start = pd.Timestamp(year=TEST_START_YEAR, month=1, day=1)
        context = ts[ts < test_start]
        if len(context) == 0:
            return None

        # Count rows per pre-test year and keep only non-empty years.
        year_counts = (context.dt.year.value_counts().sort_index())
        non_empty_years = year_counts.index.tolist()
        if len(non_empty_years) < 2:
            return None

        val_year = non_empty_years[-1]
        train_end_year = non_empty_years[-2]
        train_start_year = non_empty_years[0]

        return ("TimeRangeSplit(["
                f'("{train_start_year}-01-01", "{train_end_year}-12-31"), '
                f'("{val_year}-01-01", "{val_year}-12-31"), '
                f'("{TEST_START_YEAR}-01-01", "{MAX_YEAR}-12-31")])')
