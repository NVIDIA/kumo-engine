"""End-to-end tests for execute_tt_plan_materialize_with_split.

These tests exercise the full TT-gen pipeline:
  produce → accumulate → downsample → split → col_stats → write parquet

They call the real pybind function against a real index — no mocking.
Assertions are on inputs/outputs (parquet files, summary dicts, row
counts) so they remain valid regardless of internal implementation
(in-memory vs scratch-file streaming).
"""
import json
import os
import subprocess

import pyarrow as pa
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq
import pytest

from diskgraph._c import execute_tt_plan_materialize_with_split


# ---------------------------------------------------------------------------
# Shared graph fixture
# ---------------------------------------------------------------------------

def _build_test_graph(root):
    """Build a small graph: 3 users, 5 products, 30 orders.

    Orders span timestamps 1000-5000 (seconds), giving 5 temporal
    anchors at stride 1000. Each user has orders to multiple products,
    producing ~30-50 TT rows for LIST_DISTINCT(orders.product_id).
    """
    data_dir = root / "data"
    data_dir.mkdir()
    index_dir = root / "index"

    # Users (entity table)
    users = pa.table({
        "user_id": pa.array(["alice", "bob", "carol"]),
        "age": pa.array([25, 30, 35]),
    })
    pq.write_table(users, str(data_dir / "users.parquet"))

    # Products (target entity table)
    products = pa.table({
        "product_id": pa.array(["p1", "p2", "p3", "p4", "p5"]),
    })
    pq.write_table(products, str(data_dir / "products.parquet"))

    # Orders (fact table) — 30 rows spanning timestamps 1000-5000
    user_ids, product_ids, timestamps = [], [], []
    for uid in ["alice", "bob", "carol"]:
        for ts in [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500]:
            user_ids.append(uid)
            product_ids.append(f"p{(hash(uid + str(ts)) % 5) + 1}")
            timestamps.append(ts)

    orders = pa.table({
        "order_id": pa.array([f"o{i}" for i in range(len(user_ids))]),
        "user_id": pa.array(user_ids),
        "product_id": pa.array(product_ids),
        "ts": pa.array(timestamps, type=pa.int64()),
    })
    pq.write_table(orders, str(data_dir / "orders.parquet"))

    config = {
        "dataset_name": "test_materialize",
        "num_shards": 1,
        "tables": [
            {
                "name": "users",
                "columns": [
                    {"name": "user_id", "ingested_dtype": "string"},
                    {"name": "age", "ingested_dtype": "int64"},
                ],
                "primary_key": "user_id",
                "source": {"type": "parquet", "paths": [str(data_dir / "users.parquet")]},
            },
            {
                "name": "products",
                "columns": [{"name": "product_id", "ingested_dtype": "string"}],
                "primary_key": "product_id",
                "source": {"type": "parquet", "paths": [str(data_dir / "products.parquet")]},
            },
            {
                "name": "orders",
                "columns": [
                    {"name": "order_id", "ingested_dtype": "string"},
                    {"name": "user_id", "ingested_dtype": "string"},
                    {"name": "product_id", "ingested_dtype": "string"},
                    {"name": "ts", "ingested_dtype": "int64"},
                ],
                "primary_key": "order_id",
                "fkeys": [
                    {
                        "src_column": "user_id",
                        "dst_table": "users",
                        "reverse": {"time": {"table": "orders", "column": "ts", "unit": "s"}},
                    },
                    {
                        "src_column": "product_id",
                        "dst_table": "products",
                        "reverse": {"time": {"table": "orders", "column": "ts", "unit": "s"}},
                    },
                ],
                "source": {"type": "parquet", "paths": [str(data_dir / "orders.parquet")]},
            },
        ],
    }

    config_path = root / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    subprocess.run(
        ["cargo", "run", "--release", "--bin", "ingest", "--",
         "--source", str(config_path),
         "--dest", "local",
         "--output", str(index_dir)],
        cwd=project_root,
        check=True,
        capture_output=True,
    )
    return str(index_dir)


@pytest.fixture(scope="session")
def test_graph(tmp_path_factory):
    root = tmp_path_factory.mktemp("materialize_e2e")
    return _build_test_graph(root)


def _build_filter_graph(root):
    """Graph designed to exercise the LP orphan + temporal filter.

    - products table: 3 products with staggered `added_date`
        p_valid_early  (added 0)     — valid from start
        p_valid_late   (added 4000)  — NOT valid at anchor=2000
        p_deleted      (added 0, deleted 2000)  — NOT valid at anchor=2000
      (No `p_orphan` in products — but orders reference it, so it's a
      dangling FK that must be filtered regardless of temporal logic.)

    - orders reference {p_valid_early, p_valid_late, p_deleted, p_orphan}
      at various timestamps. `p_orphan` is not a real product.

    With a single anchor at ts=2000 and target window [0, 3000]:
      - Without filter: targets = {p_valid_early, p_valid_late, p_deleted, p_orphan}
      - With filter (orphan only, no temporal):
        targets = {p_valid_early, p_valid_late, p_deleted}  (p_orphan dropped)
      - With filter (orphan + temporal):
        targets = {p_valid_early}  (future + deleted + orphan dropped)
    """
    data_dir = root / "data"
    data_dir.mkdir()
    index_dir = root / "index"

    users = pa.table({"user_id": pa.array(["alice"])})
    pq.write_table(users, str(data_dir / "users.parquet"))

    # Products with temporal metadata (create/end time as second-precision
    # timestamps, matching the ingest's expectation for create_time columns).
    ts_s = pa.timestamp("s")
    products = pa.table({
        "product_id": pa.array(["p_valid_early", "p_valid_late", "p_deleted"]),
        "added_date":   pa.array([0,    4000, 0],    type=ts_s),
        "deleted_date": pa.array([None, None, 2000], type=ts_s),
    })
    pq.write_table(products, str(data_dir / "products.parquet"))

    # Orders reference all 4 products (including orphan p_orphan that is
    # not in the products table). Timestamps at 1000-2500.
    orders = pa.table({
        "order_id":   pa.array(["o1", "o2", "o3", "o4"]),
        "user_id":    pa.array(["alice"] * 4),
        "product_id": pa.array(
            ["p_valid_early", "p_valid_late", "p_deleted", "p_orphan"]
        ),
        "ts":         pa.array([1000, 1500, 2000, 2500], type=pa.int64()),
    })
    pq.write_table(orders, str(data_dir / "orders.parquet"))

    config = {
        "dataset_name": "test_materialize_filter",
        "num_shards": 1,
        "tables": [
            {
                "name": "users",
                "columns": [{"name": "user_id", "ingested_dtype": "string"}],
                "primary_key": "user_id",
                "source": {"type": "parquet", "paths": [str(data_dir / "users.parquet")]},
            },
            {
                "name": "products",
                "columns": [
                    {"name": "product_id", "ingested_dtype": "string"},
                    {"name": "added_date", "ingested_dtype": "timestamp"},
                    {"name": "deleted_date", "ingested_dtype": "timestamp"},
                ],
                "primary_key": "product_id",
                "create_time": "added_date",
                "end_time": "deleted_date",
                "source": {"type": "parquet", "paths": [str(data_dir / "products.parquet")]},
            },
            {
                "name": "orders",
                "columns": [
                    {"name": "order_id", "ingested_dtype": "string"},
                    {"name": "user_id", "ingested_dtype": "string"},
                    {"name": "product_id", "ingested_dtype": "string"},
                    {"name": "ts", "ingested_dtype": "int64"},
                ],
                "primary_key": "order_id",
                "fkeys": [
                    {
                        "src_column": "user_id",
                        "dst_table": "users",
                        "reverse": {"time": {"table": "orders", "column": "ts", "unit": "s"}},
                    },
                    {
                        "src_column": "product_id",
                        "dst_table": "products",
                        "reverse": {"time": {"table": "orders", "column": "ts", "unit": "s"}},
                    },
                ],
                "source": {"type": "parquet", "paths": [str(data_dir / "orders.parquet")]},
            },
        ],
    }
    config_path = root / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    subprocess.run(
        ["cargo", "run", "--release", "--bin", "ingest", "--",
         "--source", str(config_path),
         "--dest", "local",
         "--output", str(index_dir)],
        cwd=project_root,
        check=True,
        capture_output=True,
    )
    return str(index_dir)


@pytest.fixture(scope="session")
def filter_graph(tmp_path_factory):
    root = tmp_path_factory.mktemp("materialize_filter_e2e")
    return _build_filter_graph(root)


# ---------------------------------------------------------------------------
# Plan builder helpers
# ---------------------------------------------------------------------------

def _make_plan(timestamps, split=None, explode=True, include_frequency=True):
    """Build a LIST_DISTINCT(orders.product_id) plan for the test graph."""
    plan = {
        "entity": {
            "table": "users",
            "column": "user_id",
            "output_time_col_name": "TIMESTAMP",
            "default_time_value": 0,
            "time_col": None,
            "valid_to_col": None,
            "filter": None,
        },
        "temporal": {
            "timestamps": timestamps,
            "forecast_length": 1,
            "lag_timesteps": None,
        },
        "prediction_type": {
            "link": {"explode": explode, "include_frequency": include_frequency}
        },
        "split": split,
        "stages": [
            {
                "ops": [
                    {
                        "aggregate": {
                            "alias": "agg_1",
                            "path": [
                                {
                                    "src_table": "users",
                                    "src_col": "user_id",
                                    "dst_table": "orders",
                                    "dst_col": "user_id",
                                }
                            ],
                            "func": "list_distinct",
                            "target_col": "product_id",
                            "time_col": "ts",
                            "window": [0, 2000],
                            "window_unit": "seconds",
                            "row_filter": None,
                            "limit": None,
                            "target_entity_table": "products",
                        }
                    }
                ],
                "filter": None,
            }
        ],
    }
    return json.dumps(plan)


def _fixed_split_config():
    return json.dumps({"kind": "FixedSplit", "ratio": [0.6, 0.2, 0.2]})


def _ds_config(enabled=False, max_rows=999999):
    return json.dumps({
        "enabled": enabled,
        "max_rows": max_rows,
        "preserve_pre_downsample_tt": True,
        "is_link_prediction": True,
        "list_distinct_target_is_array": False,
        "uses_arl_labels": False,
    })


def _make_filter_plan(
    timestamps,
    target_entity_table=None,
    window=(-5000, 5000),
):
    """Plan for filter_graph: LIST_DISTINCT(orders.product_id) for users.

    target_entity_table=None disables orphan+temporal filtering.
    target_entity_table="products" enables both (automatic temporal if
    the products table has create_time_col / end_time_col).
    """
    plan = {
        "entity": {
            "table": "users", "column": "user_id",
            "output_time_col_name": "TIMESTAMP",
            "default_time_value": 0,
            "time_col": None, "valid_to_col": None, "filter": None,
        },
        "temporal": {
            "timestamps": timestamps,
            "forecast_length": 1, "lag_timesteps": None,
        },
        "prediction_type": {
            "link": {"explode": True, "include_frequency": True}
        },
        "split": None,
        "stages": [{
            "ops": [{
                "aggregate": {
                    "alias": "agg_1",
                    "path": [{
                        "src_table": "users", "src_col": "user_id",
                        "dst_table": "orders", "dst_col": "user_id",
                    }],
                    "func": "list_distinct",
                    "target_col": "product_id",
                    "time_col": "ts",
                    "window": list(window),
                    "window_unit": "seconds",
                    "row_filter": None,
                    "limit": None,
                    "target_entity_table": target_entity_table,
                }
            }],
            "filter": None,
        }],
    }
    return json.dumps(plan)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMaterializeBasic:
    """Verify the output parquet is correct."""

    def test_output_parquet_exists_with_correct_schema(self, test_graph, tmp_path):
        output = str(tmp_path / "tt.parquet")
        plan = _make_plan(timestamps=[1000, 2000, 3000])
        result = execute_tt_plan_materialize_with_split(
            plan_json=plan,
            index_path=test_graph,
            output_path=output,
            split_config_json=_fixed_split_config(),
            split_family_hint="FixedSplit",
            downsampling_config_json=_ds_config(),
        )
        assert os.path.exists(output)
        pf = pq.ParquetFile(output)
        cols = {f.name for f in pf.schema_arrow}
        assert "ENTITY" in cols
        assert "TIMESTAMP" in cols
        assert "TARGET" in cols
        assert "arange" in cols
        assert pf.metadata.num_rows == result["rows_written"]
        assert result["rows_written"] > 0

    def test_rows_written_matches_parquet(self, test_graph, tmp_path):
        output = str(tmp_path / "tt.parquet")
        plan = _make_plan(timestamps=[1000, 2000, 3000, 4000])
        result = execute_tt_plan_materialize_with_split(
            plan_json=plan,
            index_path=test_graph,
            output_path=output,
            split_config_json=_fixed_split_config(),
            split_family_hint="FixedSplit",
            downsampling_config_json=_ds_config(),
        )
        pf = pq.ParquetFile(output)
        assert pf.metadata.num_rows == result["rows_written"]

    def test_more_timestamps_produce_more_rows(self, test_graph, tmp_path):
        out1 = str(tmp_path / "tt1.parquet")
        out2 = str(tmp_path / "tt2.parquet")
        r1 = execute_tt_plan_materialize_with_split(
            plan_json=_make_plan(timestamps=[2000]),
            index_path=test_graph, output_path=out1,
            split_config_json=_fixed_split_config(),
            split_family_hint="FixedSplit",
            downsampling_config_json=_ds_config(),
        )
        r2 = execute_tt_plan_materialize_with_split(
            plan_json=_make_plan(timestamps=[2000, 3000, 4000]),
            index_path=test_graph, output_path=out2,
            split_config_json=_fixed_split_config(),
            split_family_hint="FixedSplit",
            downsampling_config_json=_ds_config(),
        )
        assert r2["rows_written"] > r1["rows_written"]


class TestMaterializeSplit:
    """Verify split assignment produces non-empty train/val/test."""

    def test_fixed_split_has_all_stages(self, test_graph, tmp_path):
        output = str(tmp_path / "tt.parquet")
        plan = _make_plan(timestamps=[1000, 2000, 3000, 4000])
        result = execute_tt_plan_materialize_with_split(
            plan_json=plan,
            index_path=test_graph, output_path=output,
            split_config_json=_fixed_split_config(),
            split_family_hint="FixedSplit",
            downsampling_config_json=_ds_config(),
        )
        # Count from split_assignments bytes: 0=train, 1=val, 2=test
        sa = result["split_assignments"]
        assert sa is not None and len(sa) > 0
        train = sum(1 for b in sa if b == 0)
        val = sum(1 for b in sa if b == 1)
        test = sum(1 for b in sa if b == 2)
        assert train > 0, f"train should be > 0, got train={train}"
        assert val > 0, f"val should be > 0, got val={val}"
        assert test > 0, f"test should be > 0, got test={test}"
        assert train + val + test == result["rows_written"]

    def test_split_assignments_bytes_present(self, test_graph, tmp_path):
        output = str(tmp_path / "tt.parquet")
        plan = _make_plan(timestamps=[1000, 2000, 3000])
        result = execute_tt_plan_materialize_with_split(
            plan_json=plan,
            index_path=test_graph, output_path=output,
            split_config_json=_fixed_split_config(),
            split_family_hint="FixedSplit",
            downsampling_config_json=_ds_config(),
        )
        assert result.get("split_assignments") is not None
        assert len(result["split_assignments"]) > 0


class TestMaterializeColStats:
    """Verify col_stats are produced."""

    def test_col_stats_json_non_empty(self, test_graph, tmp_path):
        output = str(tmp_path / "tt.parquet")
        plan = _make_plan(timestamps=[1000, 2000, 3000])
        result = execute_tt_plan_materialize_with_split(
            plan_json=plan,
            index_path=test_graph, output_path=output,
            split_config_json=_fixed_split_config(),
            split_family_hint="FixedSplit",
            downsampling_config_json=_ds_config(),
        )
        col_stats_json = result.get("train_table_col_stats_json")
        assert col_stats_json is not None, "col_stats should be present"
        col_stats = json.loads(col_stats_json)
        assert col_stats.get("kind") == "tt_train_table_col_stats"
        stats = col_stats.get("stats", {})
        assert len(stats) > 0, f"col_stats.stats should have entries, got {list(stats.keys())[:5]}"

    def test_col_stats_has_expected_columns(self, test_graph, tmp_path):
        output = str(tmp_path / "tt.parquet")
        plan = _make_plan(timestamps=[1000, 2000, 3000])
        result = execute_tt_plan_materialize_with_split(
            plan_json=plan,
            index_path=test_graph, output_path=output,
            split_config_json=_fixed_split_config(),
            split_family_hint="FixedSplit",
            downsampling_config_json=_ds_config(),
        )
        col_stats_json = result.get("train_table_col_stats_json")
        assert col_stats_json is not None, "train_table_col_stats_json missing from result"
        col_stats = json.loads(col_stats_json)
        keys = list(col_stats.get("stats", {}).keys())
        # Inline stats produce encoder-ready distribution stats for generated columns.
        # This plan is LIST_DISTINCT (link prediction) with include_frequency=True,
        # so we expect FREQUENCY and ENTITY_INCLUDE_BIT stats.
        # ENTITY/TIMESTAMP stats are in exact_stats (separate field), not here.
        has_frequency = any("__kumo_internal_frequency_column__" in k for k in keys)
        has_include_bit = any("__kumo_internal_entity_inclusive_bit_column__" in k for k in keys)
        assert has_frequency, f"missing FREQUENCY stats in {keys[:10]}"
        assert has_include_bit, f"missing ENTITY_INCLUDE_BIT stats in {keys[:10]}"
        # Verify key format matches kumo_col_stats_v1
        assert col_stats.get("kind") == "tt_train_table_col_stats", \
            f"unexpected kind: {col_stats.get('kind')}"


class TestMaterializeDownsampling:
    """Verify downsampling limits output rows."""

    def test_downsampling_limits_rows(self, test_graph, tmp_path):
        output_full = str(tmp_path / "full.parquet")
        output_ds = str(tmp_path / "ds.parquet")
        plan = _make_plan(timestamps=[1000, 2000, 3000, 4000])

        r_full = execute_tt_plan_materialize_with_split(
            plan_json=plan,
            index_path=test_graph, output_path=output_full,
            split_config_json=_fixed_split_config(),
            split_family_hint="FixedSplit",
            downsampling_config_json=_ds_config(enabled=False),
        )
        full_rows = r_full["rows_written"]
        assert full_rows > 5, "need enough rows to test downsampling"

        max_rows = max(full_rows // 2, 1)
        r_ds = execute_tt_plan_materialize_with_split(
            plan_json=plan,
            index_path=test_graph, output_path=output_ds,
            split_config_json=_fixed_split_config(),
            split_family_hint="FixedSplit",
            downsampling_config_json=_ds_config(enabled=True, max_rows=max_rows),
        )
        assert r_ds["rows_written"] <= max_rows
        assert r_ds["rows_written"] < full_rows

    def test_downsampling_preserves_pre_downsample_count(self, test_graph, tmp_path):
        output = str(tmp_path / "ds.parquet")
        plan = _make_plan(timestamps=[1000, 2000, 3000, 4000])
        result = execute_tt_plan_materialize_with_split(
            plan_json=plan,
            index_path=test_graph, output_path=output,
            split_config_json=_fixed_split_config(),
            split_family_hint="FixedSplit",
            downsampling_config_json=_ds_config(enabled=True, max_rows=3),
        )
        assert result["pre_downsample_row_count"] >= result["rows_written"]

    def test_no_downsampling_when_under_limit(self, test_graph, tmp_path):
        output = str(tmp_path / "tt.parquet")
        plan = _make_plan(timestamps=[2000])
        result = execute_tt_plan_materialize_with_split(
            plan_json=plan,
            index_path=test_graph, output_path=output,
            split_config_json=_fixed_split_config(),
            split_family_hint="FixedSplit",
            downsampling_config_json=_ds_config(enabled=True, max_rows=999999),
        )
        # Under limit → no downsampling applied
        assert result["rows_written"] == result["pre_downsample_row_count"]


class TestMaterializeArtifacts:
    """Verify output artifacts."""

    def test_summary_json_has_expected_fields(self, test_graph, tmp_path):
        output = str(tmp_path / "tt.parquet")
        plan = _make_plan(timestamps=[1000, 2000, 3000])
        result = execute_tt_plan_materialize_with_split(
            plan_json=plan,
            index_path=test_graph, output_path=output,
            split_config_json=_fixed_split_config(),
            split_family_hint="FixedSplit",
            downsampling_config_json=_ds_config(),
        )
        assert "rows_written" in result
        assert "summary_json" in result
        assert "split_family" in result
        assert "split_assignments" in result
        assert "train_table_col_stats_json" in result
        summary = json.loads(result["summary_json"])
        assert "stats" in summary
        assert "rows_written" in summary

    def test_split_family_matches_config(self, test_graph, tmp_path):
        output = str(tmp_path / "tt.parquet")
        plan = _make_plan(timestamps=[1000, 2000])
        result = execute_tt_plan_materialize_with_split(
            plan_json=plan,
            index_path=test_graph, output_path=output,
            split_config_json=_fixed_split_config(),
            split_family_hint="FixedSplit",
            downsampling_config_json=_ds_config(),
        )
        assert result["split_family"] == "FixedSplit"


class TestMaterializeParquetContract:
    """Validate output parquet against the contract downstream consumers rely on.

    The final TT parquet is consumed by kumo's pipeline via stricter readers
    (e.g. pyarrow.dataset) and row-count helpers (io_utils.parquet_util.is_empty).
    These impose constraints beyond what pyarrow.parquet.read_table enforces:
    - unique column names (duplicates raise ArrowInvalid in dataset API)
    - readable via dataset API (not just direct parquet read)
    - arange column is a valid monotonic row index 0..N-1

    These tests exist because regressions in the post-DS write path (e.g.
    double-appending arange) only surface in downstream readers, not in our
    own validation.
    """

    def _assert_unique_columns(self, parquet_path: str) -> None:
        """Fail if the parquet schema has duplicate column names."""
        pf = pq.ParquetFile(parquet_path)
        names = [f.name for f in pf.schema_arrow]
        duplicates = [n for n in set(names) if names.count(n) > 1]
        assert not duplicates, (
            f"parquet has duplicate column names {duplicates}; "
            f"breaks pyarrow.dataset and downstream readers. schema: {names}"
        )

    def _assert_dataset_readable(self, parquet_path: str) -> int:
        """Read via pyarrow.dataset (the strict API kumo uses).

        Returns row count. Fails if schema has duplicates or file is malformed —
        mirrors what io_utils.parquet_util.is_empty() does under the hood.
        """
        ds = pa_ds.dataset(parquet_path, format="parquet")
        return ds.count_rows()

    def test_final_parquet_has_unique_columns_no_ds(self, test_graph, tmp_path):
        """Output parquet without downsampling must have unique column names."""
        output = str(tmp_path / "tt.parquet")
        plan = _make_plan(timestamps=[1000, 2000, 3000])
        execute_tt_plan_materialize_with_split(
            plan_json=plan, index_path=test_graph, output_path=output,
            split_config_json=_fixed_split_config(),
            split_family_hint="FixedSplit",
            downsampling_config_json=_ds_config(enabled=False),
        )
        self._assert_unique_columns(output)

    def test_final_parquet_has_unique_columns_with_ds(self, test_graph, tmp_path):
        """Output parquet WITH downsampling must have unique column names.

        Regression test: a prior streaming impl wrote arange in scratch AND
        appended it again post-downsample, producing two 'arange' columns in
        the final parquet. Broke pyarrow.dataset reads → false 'empty TT'.
        """
        output = str(tmp_path / "tt.parquet")
        plan = _make_plan(timestamps=[1000, 2000, 3000, 4000])
        execute_tt_plan_materialize_with_split(
            plan_json=plan, index_path=test_graph, output_path=output,
            split_config_json=_fixed_split_config(),
            split_family_hint="FixedSplit",
            downsampling_config_json=_ds_config(enabled=True, max_rows=3),
        )
        self._assert_unique_columns(output)

    def test_pre_downsample_parquet_has_unique_columns(self, test_graph, tmp_path):
        """The preserved pre_downsample artifact must also have unique columns."""
        output = str(tmp_path / "tt.parquet")
        plan = _make_plan(timestamps=[1000, 2000, 3000, 4000])
        execute_tt_plan_materialize_with_split(
            plan_json=plan, index_path=test_graph, output_path=output,
            split_config_json=_fixed_split_config(),
            split_family_hint="FixedSplit",
            downsampling_config_json=_ds_config(enabled=True, max_rows=3),
        )
        pre_path = output + ".pre_downsample.parquet"
        if os.path.exists(pre_path):
            self._assert_unique_columns(pre_path)

    def test_final_parquet_readable_via_dataset_api(self, test_graph, tmp_path):
        """Output must be readable via pyarrow.dataset.

        kumo's io_utils.parquet_util.is_empty() uses pyarrow.dataset; if that
        fails due to schema issues, the file is silently treated as empty.
        """
        output = str(tmp_path / "tt.parquet")
        plan = _make_plan(timestamps=[1000, 2000, 3000])
        result = execute_tt_plan_materialize_with_split(
            plan_json=plan, index_path=test_graph, output_path=output,
            split_config_json=_fixed_split_config(),
            split_family_hint="FixedSplit",
            downsampling_config_json=_ds_config(enabled=True, max_rows=3),
        )
        row_count = self._assert_dataset_readable(output)
        assert row_count == result["rows_written"]
        assert row_count > 0

    def test_arange_is_monotonic_row_index(self, test_graph, tmp_path):
        """arange must be 0..N-1 contiguous on the final parquet.

        After downsampling, arange should be re-indexed to match the final
        row count — not carry gaps from pre-downsample rows that were filtered.
        """
        output = str(tmp_path / "tt.parquet")
        plan = _make_plan(timestamps=[1000, 2000, 3000, 4000])
        execute_tt_plan_materialize_with_split(
            plan_json=plan, index_path=test_graph, output_path=output,
            split_config_json=_fixed_split_config(),
            split_family_hint="FixedSplit",
            downsampling_config_json=_ds_config(enabled=True, max_rows=3),
        )
        table = pq.read_table(output)
        arange = table.column("arange").to_pylist()
        assert arange == list(range(len(arange))), (
            f"arange must be 0..N-1 on final parquet; got {arange}"
        )


class TestLpRhsEntityTableFiltering:
    """E2E: verify target_entity_table in the plan JSON drives the orphan
    + temporal filter correctly on the executor.

    The fixture `filter_graph` has:
    - products: {p_valid_early, p_valid_late, p_deleted} with temporal cols
    - orders referencing {p_valid_early, p_valid_late, p_deleted, p_orphan}
    - anchor ts=2000
    """

    def _final_targets(self, output_path):
        """Collect the set of target values from the final parquet."""
        table = pq.read_table(output_path)
        return set(table.column("TARGET").to_pylist())

    def test_no_filter_keeps_all_referenced_products(self, filter_graph, tmp_path):
        """Without target_entity_table: orphan + temporally invalid targets
        both pass through (baseline — shows what the filter is catching)."""
        output = str(tmp_path / "tt.parquet")
        plan = _make_filter_plan(timestamps=[2000], target_entity_table=None)
        execute_tt_plan_materialize_with_split(
            plan_json=plan, index_path=filter_graph, output_path=output,
            split_config_json=_fixed_split_config(),
            split_family_hint="FixedSplit",
            downsampling_config_json=_ds_config(enabled=False),
        )
        targets = self._final_targets(output)
        # All 4 targets present including the orphan and the temporally-invalid
        assert targets == {
            "p_valid_early", "p_valid_late", "p_deleted", "p_orphan"
        }, f"got {targets}"

    def test_orphan_and_temporal_filter_drops_invalid_targets(
        self, filter_graph, tmp_path
    ):
        """With target_entity_table='products' and create_time/end_time on
        products: orphan + future + deleted targets all get dropped.

        At anchor=2000:
          - p_valid_early (added=0, no delete)    → KEEP
          - p_valid_late  (added=4000)            → DROP (future)
          - p_deleted     (added=0, deleted=2000) → DROP (deleted)
          - p_orphan      (not in products)       → DROP (orphan)
        """
        output = str(tmp_path / "tt.parquet")
        plan = _make_filter_plan(
            timestamps=[2000], target_entity_table="products"
        )
        execute_tt_plan_materialize_with_split(
            plan_json=plan, index_path=filter_graph, output_path=output,
            split_config_json=_fixed_split_config(),
            split_family_hint="FixedSplit",
            downsampling_config_json=_ds_config(enabled=False),
        )
        targets = self._final_targets(output)
        assert targets == {"p_valid_early"}, (
            f"expected only temporally-valid non-orphan target; got {targets}"
        )

    def test_filter_config_error_on_missing_target_table(
        self, filter_graph, tmp_path
    ):
        """target_entity_table pointing at a non-existent table must error
        loudly (not silently disable the filter)."""
        output = str(tmp_path / "tt.parquet")
        plan = _make_filter_plan(
            timestamps=[2000], target_entity_table="nonexistent_table"
        )
        with pytest.raises(Exception) as excinfo:
            execute_tt_plan_materialize_with_split(
                plan_json=plan, index_path=filter_graph, output_path=output,
                split_config_json=_fixed_split_config(),
                split_family_hint="FixedSplit",
                downsampling_config_json=_ds_config(enabled=False),
            )
        assert "nonexistent_table" in str(excinfo.value)

    def test_filter_active_at_later_anchor(self, filter_graph, tmp_path):
        """At anchor=5000 (after p_valid_late is added, p_deleted is gone):
          - p_valid_early → KEEP
          - p_valid_late  (added=4000 <= 5000) → KEEP
          - p_deleted     (deleted=2000 <= 5000) → still DROP
          - p_orphan      → DROP
        """
        output = str(tmp_path / "tt.parquet")
        # Broaden window to include all orders at this later anchor
        plan = _make_filter_plan(
            timestamps=[5000], target_entity_table="products",
        )
        execute_tt_plan_materialize_with_split(
            plan_json=plan, index_path=filter_graph, output_path=output,
            split_config_json=_fixed_split_config(),
            split_family_hint="FixedSplit",
            downsampling_config_json=_ds_config(enabled=False),
        )
        targets = self._final_targets(output)
        assert targets == {"p_valid_early", "p_valid_late"}, (
            f"expected valid + now-created targets; got {targets}"
        )
