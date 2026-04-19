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
        col_stats = json.loads(result["train_table_col_stats_json"])
        keys = list(col_stats.get("stats", {}).keys())
        # Should have ENTITY, TARGET, TIMESTAMP stats at minimum
        has_entity = any("ENTITY" in k for k in keys)
        has_target = any("TARGET" in k for k in keys)
        has_timestamp = any("TIMESTAMP" in k for k in keys)
        assert has_entity, f"missing ENTITY stats in {keys[:10]}"
        assert has_target, f"missing TARGET stats in {keys[:10]}"
        assert has_timestamp, f"missing TIMESTAMP stats in {keys[:10]}"


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
