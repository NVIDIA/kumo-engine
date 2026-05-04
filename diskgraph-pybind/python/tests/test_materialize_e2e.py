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

def _make_plan(timestamps, split=None, explode=True, include_frequency=True, target_stype="numerical"):
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
    plan["target_stype"] = target_stype
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
        "target_stype": "categorical",
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

    def test_manual_split_uses_projected_split_column(self, test_graph, tmp_path):
        output = str(tmp_path / "tt.parquet")
        split = {
            "kind": "ManualSplit",
            "split_column": "users.user_id",
            "manual_train_values": ["alice"],
            "manual_val_values": ["bob"],
            "manual_test_values": ["carol"],
        }
        result = execute_tt_plan_materialize_with_split(
            plan_json=_make_plan(timestamps=[1000, 2000, 3000], split=split),
            index_path=test_graph,
            output_path=output,
            split_config_json=json.dumps(split),
            split_family_hint="ManualSplit",
            downsampling_config_json=_ds_config(),
        )

        assert result["rows_written"] > 0
        assignments = result["split_assignments"]
        assert assignments is not None
        assert len(assignments) == result["rows_written"]
        entities = pq.read_table(output, columns=["ENTITY"]).column("ENTITY").to_pylist()
        expected = {"alice": 0, "bob": 1, "carol": 2}
        assert len(entities) == len(assignments)
        seen = set()
        for entity, assignment in zip(entities, assignments):
            assert assignment == expected[entity]
            seen.add(entity)
        assert seen == set(expected)


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
        # This plan uses LIST_DISTINCT (link prediction) with target_stype=categorical,
        # so TARGET stats should use categorical format (CATEGORY_COUNTS).
        plan = _make_plan(timestamps=[1000, 2000, 3000], target_stype="categorical")
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
        # so we expect TARGET (categorical), FREQUENCY (numerical), and ENTITY_INCLUDE_BIT stats.
        has_target = any("TARGET" in k and "kumo.categorical" in k for k in keys)
        has_category_counts = any("CATEGORY_COUNTS" in k for k in keys)
        has_frequency = any("__kumo_internal_frequency_column__" in k for k in keys)
        has_include_bit = any("__kumo_internal_entity_inclusive_bit_column__" in k for k in keys)
        has_entity = any("ENTITY" in k and "kumo.categorical" in k for k in keys)
        has_timestamp = any("TIMESTAMP" in k and "kumo.numerical" in k for k in keys)
        assert has_target, f"missing TARGET categorical stats in {keys[:10]}"
        assert has_category_counts, f"missing CATEGORY_COUNTS in {keys[:10]}"
        assert has_frequency, f"missing FREQUENCY stats in {keys[:10]}"
        assert has_include_bit, f"missing ENTITY_INCLUDE_BIT stats in {keys[:10]}"
        assert has_entity, f"missing ENTITY categorical stats in {keys[:10]}"
        assert has_timestamp, f"missing TIMESTAMP numerical stats in {keys[:10]}"
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


# ---------------------------------------------------------------------------
# Multicategorical TT-gen e2e (kumo#28400 follow-up — Item A)
#
# Mirrors the Rust integration cube for LIST_DISTINCT × multicat × source
# shape, but exercised through the Python pybind materialize entrypoint so the
# full pipeline (ingest persists `diskgraph_config.json` with `semantic_type`
# → executor reads `column_semantic_type` at TT-gen → writer emits multicat
# stats + array-output target column) is covered end-to-end from one process.
#
# Canonical fixture (shape-agnostic logical content):
#   user 0:  ts=100 tags=[a,b], ts=200 tags=[b,c]
#   user 1:  ts=300 tags=[a],   ts=400 tags=[d,e]
#   user 2:  no orders
#
# Anchor ts=5000, window [-5000, 0] seconds → all 4 orders included.
# Per-user distinct token sets (LIST_DISTINCT semantics):
#   user 0 → {a,b,c} (b appears in both rows)
#   user 1 → {a,d,e}
#   user 2 → {}
#
# Per-token frequency (number of source rows that contributed each token):
#   user 0:  a=1, b=2, c=1
#   user 1:  a=1, d=1, e=1
#
# Cross-shape parity: utf8_delim (post-#28478 normalized to StringArray at
# ingest) and string_array shapes must produce byte-identical output. The
# int64_array shape uses tokens 1..=5 under the bijection a=1,b=2,c=3,d=4,e=5
# and is decoded back via MULTICAT_TOKEN_INT_MAP for comparison.
# ---------------------------------------------------------------------------

MULTICAT_TOKEN_INT_MAP = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
MULTICAT_INT_TO_TOKEN = {v: k for k, v in MULTICAT_TOKEN_INT_MAP.items()}


def _build_multicat_graph(root, shape):
    """Build a 3-user / 4-order multicat-source fixture.

    `shape` ∈ {"utf8_delim", "string_array", "int64_array"} — drives the
    on-disk parquet layout of the `tags` column and the diskgraph config's
    `ingested_dtype` + delimiter declaration. All three shapes declare
    `semantic_type=multicategorical` so ingest persists the stype to
    `diskgraph_config.json` for TT-gen to read at runtime.
    """
    data_dir = root / "data"
    data_dir.mkdir()
    index_dir = root / "index"

    users = pa.table({"user_id": pa.array([0, 1, 2], type=pa.int64())})
    pq.write_table(users, str(data_dir / "users.parquet"))

    order_user_id = [0, 0, 1, 1]
    order_ts = [100, 200, 300, 400]
    tags_per_row = [["a", "b"], ["b", "c"], ["a"], ["d", "e"]]

    if shape == "utf8_delim":
        tags_arr = pa.array([",".join(toks) for toks in tags_per_row], type=pa.string())
    elif shape == "string_array":
        tags_arr = pa.array(tags_per_row, type=pa.list_(pa.string()))
    elif shape == "int64_array":
        tags_arr = pa.array(
            [[MULTICAT_TOKEN_INT_MAP[t] for t in toks] for toks in tags_per_row],
            type=pa.list_(pa.int64()),
        )
    else:
        raise ValueError(f"unknown shape: {shape}")

    orders = pa.table({
        "order_id": pa.array(list(range(len(order_user_id))), type=pa.int64()),
        "user_id": pa.array(order_user_id, type=pa.int64()),
        "ts": pa.array(order_ts, type=pa.int64()),
        "tags": tags_arr,
    })
    pq.write_table(orders, str(data_dir / "orders.parquet"))

    if shape == "utf8_delim":
        tags_col = {
            "name": "tags",
            "ingested_dtype": "string",
            "semantic_type": "multicategorical",
            "multicategorical_delimiter": ",",
        }
    elif shape == "string_array":
        tags_col = {
            "name": "tags",
            "ingested_dtype": "string_array",
            "semantic_type": "multicategorical",
        }
    else:  # int64_array
        tags_col = {
            "name": "tags",
            "ingested_dtype": "int64_array",
            "semantic_type": "multicategorical",
        }

    config = {
        "dataset_name": f"test_multicat_{shape}",
        "num_shards": 1,
        "tables": [
            {
                "name": "users",
                "columns": [{"name": "user_id", "ingested_dtype": "int64"}],
                "primary_key": "user_id",
                "source": {"type": "parquet", "paths": [str(data_dir / "users.parquet")]},
            },
            {
                "name": "orders",
                "columns": [
                    {"name": "order_id", "ingested_dtype": "int64"},
                    {"name": "user_id", "ingested_dtype": "int64"},
                    {"name": "ts", "ingested_dtype": "int64"},
                    tags_col,
                ],
                "primary_key": "order_id",
                "fkeys": [
                    {
                        "src_column": "user_id",
                        "dst_table": "users",
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
def multicat_graph_utf8_delim(tmp_path_factory):
    root = tmp_path_factory.mktemp("multicat_e2e_utf8_delim")
    return _build_multicat_graph(root, "utf8_delim")


@pytest.fixture(scope="session")
def multicat_graph_string_array(tmp_path_factory):
    root = tmp_path_factory.mktemp("multicat_e2e_string_array")
    return _build_multicat_graph(root, "string_array")


@pytest.fixture(scope="session")
def multicat_graph_int64_array(tmp_path_factory):
    root = tmp_path_factory.mktemp("multicat_e2e_int64_array")
    return _build_multicat_graph(root, "int64_array")


def _make_multicat_plan(timestamps=(5000,), target_is_array=True, include_frequency=True):
    """LIST_DISTINCT(orders.tags) plan with target_stype=multicategorical.

    target_is_array=True (explode=False) is the multilabel link-prediction
    shape: TARGET column is List<Utf8>, FREQUENCY column is List<Int64>,
    one row per entity (including empty-list rows for entities with no
    in-window orders).
    """
    plan = {
        "entity": {
            "table": "users", "column": "user_id",
            "output_time_col_name": "TIMESTAMP",
            "default_time_value": 0,
            "time_col": None, "valid_to_col": None, "filter": None,
        },
        "temporal": {
            "timestamps": list(timestamps),
            "forecast_length": 1,
            "lag_timesteps": None,
        },
        "prediction_type": {
            "link": {
                "explode": not target_is_array,
                "include_frequency": include_frequency,
            }
        },
        "target_stype": "multicategorical",
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
                    "target_col": "tags",
                    "time_col": "ts",
                    "window": [-5000, 0],
                    "window_unit": "seconds",
                    "row_filter": None,
                    "limit": None,
                    "target_entity_table": None,
                }
            }],
            "filter": None,
        }],
    }
    return json.dumps(plan)


def _ds_config_multicat(target_is_array=True):
    return json.dumps({
        "enabled": False,
        "max_rows": 999999,
        "preserve_pre_downsample_tt": True,
        "is_link_prediction": True,
        "list_distinct_target_is_array": target_is_array,
        "uses_arl_labels": False,
    })


class TestMulticatListDistinctMaterialize:
    """End-to-end materialize for multilabel-shaped LIST_DISTINCT plans.

    Closes Item A from PR #230's deferred list. Validates that the on-disk
    stype channel introduced by #28400 (`diskgraph_config.json` written at
    ingest, read by TT-gen via `column_semantic_type`) drives per-token
    observation through the full Python pybind pipeline across all three
    multicat source-column on-disk shapes.
    """

    def _arrayout_rows(self, parquet_path):
        """Return list of (entity, target_list, freq_list) rows.

        TARGET is List<Utf8>; FREQUENCY is List<Int64>.
        """
        table = pq.read_table(parquet_path)
        entities = table.column("ENTITY").to_pylist()
        targets = table.column("TARGET").to_pylist()
        freqs = table.column("__kumo_internal_frequency_column__").to_pylist()
        return list(zip(entities, targets, freqs))

    def _per_entity_token_freq(self, rows, decode_int=False):
        """Map entity → sorted [(token, freq)] across the array-output rows.

        Skips entities whose target list is empty (matches the exploded-mode
        behavior, where empty entities emit no rows at all).
        """
        out = {}
        for entity, target_list, freq_list in rows:
            if not target_list:
                continue
            assert freq_list is not None, "include_frequency=True must populate FREQUENCY"
            assert len(target_list) == len(freq_list), (
                f"TARGET and FREQUENCY lengths diverge: {len(target_list)} vs {len(freq_list)}"
            )
            pairs = []
            for tok, freq in zip(target_list, freq_list):
                if decode_int:
                    tok = MULTICAT_INT_TO_TOKEN[int(tok)]
                pairs.append((tok, int(freq)))
            pairs.sort()
            out[int(entity)] = pairs
        return out

    def _expected_canonical_token_freq(self):
        """Per-entity (token, frequency) under the canonical multicat fixture.

        user 0: rows [a,b]+[b,c] → a=1, b=2, c=1
        user 1: rows [a]+[d,e]   → a=1, d=1, e=1
        user 2: no orders → no entry (empty-list row dropped by filter)
        """
        return {
            0: [("a", 1), ("b", 2), ("c", 1)],
            1: [("a", 1), ("d", 1), ("e", 1)],
        }

    def _run(self, index_path, output_path):
        return execute_tt_plan_materialize_with_split(
            plan_json=_make_multicat_plan(),
            index_path=index_path,
            output_path=output_path,
            split_config_json=_fixed_split_config(),
            split_family_hint="FixedSplit",
            downsampling_config_json=_ds_config_multicat(target_is_array=True),
        )

    def test_arrayout_schema_string_array(self, multicat_graph_string_array, tmp_path):
        """TARGET=List<Utf8>, FREQUENCY=List<Int64>, one row per entity (3)."""
        output = str(tmp_path / "tt.parquet")
        result = self._run(multicat_graph_string_array, output)

        pf = pq.ParquetFile(output)
        schema = pf.schema_arrow
        cols = {f.name: f.type for f in schema}
        assert "ENTITY" in cols
        assert "TIMESTAMP" in cols
        assert "TARGET" in cols
        assert "__kumo_internal_frequency_column__" in cols

        target_type = cols["TARGET"]
        assert pa.types.is_list(target_type), f"TARGET must be list type; got {target_type}"
        assert pa.types.is_string(target_type.value_type), (
            f"TARGET value_type must be string; got {target_type.value_type}"
        )

        freq_type = cols["__kumo_internal_frequency_column__"]
        assert pa.types.is_list(freq_type), f"FREQUENCY must be list type; got {freq_type}"
        assert pa.types.is_int64(freq_type.value_type), (
            f"FREQUENCY value_type must be int64; got {freq_type.value_type}"
        )

        # Three users → three rows in array mode (including empty-list user).
        assert pf.metadata.num_rows == 3
        assert result["rows_written"] == 3

    def test_arrayout_token_freq_string_array(self, multicat_graph_string_array, tmp_path):
        """Per-entity (token, freq) matches canonical expectations."""
        output = str(tmp_path / "tt.parquet")
        self._run(multicat_graph_string_array, output)
        rows = self._arrayout_rows(output)
        observed = self._per_entity_token_freq(rows)
        assert observed == self._expected_canonical_token_freq(), (
            f"per-entity token/freq mismatch on string_array; got {observed}"
        )

    def test_arrayout_consistency_string_array_vs_utf8_delim(
        self, multicat_graph_string_array, multicat_graph_utf8_delim, tmp_path
    ):
        """Cross-shape parity: utf8_delim (normalized at ingest by #28478) and
        native string_array must produce byte-equivalent per-entity output."""
        out_str = str(tmp_path / "tt_str.parquet")
        out_delim = str(tmp_path / "tt_delim.parquet")
        self._run(multicat_graph_string_array, out_str)
        self._run(multicat_graph_utf8_delim, out_delim)

        rows_str = self._arrayout_rows(out_str)
        rows_delim = self._arrayout_rows(out_delim)
        a = self._per_entity_token_freq(rows_str)
        b = self._per_entity_token_freq(rows_delim)
        assert a == b, f"string_array vs utf8_delim diverge\nstring={a}\ndelim={b}"

    def test_arrayout_intlist_decoded_matches_canonical(
        self, multicat_graph_int64_array, tmp_path
    ):
        """Int64 source tokens stringify to "1".."5" in TARGET; decoded via
        the bijection they reproduce the same per-entity token sets."""
        output = str(tmp_path / "tt.parquet")
        self._run(multicat_graph_int64_array, output)
        rows = self._arrayout_rows(output)
        observed = self._per_entity_token_freq(rows, decode_int=True)
        assert observed == self._expected_canonical_token_freq(), (
            f"int64_array decoded token/freq mismatch; got {observed}"
        )

    def test_col_stats_multi_category_counts_per_token(
        self, multicat_graph_string_array, tmp_path
    ):
        """Stats payload must observe per-token, not whole-cell.

        Pre-#28400 the array-output writer arm only observed `targets[0]` per
        row, undercounting `targets[1..]`. Pre-fix `update_exact_stats` also
        stringified `Value::List` to "<list>", collapsing every entity's
        labels into one synthetic category. Both regressions are caught here.
        """
        output = str(tmp_path / "tt.parquet")
        result = self._run(multicat_graph_string_array, output)
        col_stats = json.loads(result["train_table_col_stats_json"])
        keys = list(col_stats.get("stats", {}).keys())

        target_counts_keys = [
            k for k in keys
            if "TARGET" in k and "MULTI_CATEGORY_COUNTS" in k
        ]
        assert target_counts_keys, (
            f"missing TARGET MULTI_CATEGORY_COUNTS in {keys[:20]}"
        )
        counts_entry = col_stats["stats"][target_counts_keys[0]]
        pairs = counts_entry["value"]
        observed = {tok: count for tok, count in pairs}

        # Whole-cell regression net: "<list>" placeholder must not appear.
        assert "<list>" not in observed, (
            f"TARGET stats collapsed Value::List to <list> placeholder; got {observed}"
        )

        # Per-token frequency across array-output rows under the canonical
        # fixture:  a×2 (user 0 + user 1), b×1, c×1, d×1, e×1.
        expected = {"a": 2, "b": 1, "c": 1, "d": 1, "e": 1}
        for tok, want in expected.items():
            assert observed.get(tok) == want, (
                f"token {tok!r} count {observed.get(tok)} != expected {want}; "
                f"full counts = {observed}"
            )
