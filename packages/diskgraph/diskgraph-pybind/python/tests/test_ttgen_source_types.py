# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training tables compute with source column types the way they would with their casts."""

import decimal
import json
import pathlib

import diskgraph
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from diskgraph import DiskGraphTTBuilder, EdgeRef, RowFilter, SamplerEngine, execute_tt_gen

EVENTS = EdgeRef("users", "id", "events", "user_id")


@pytest.fixture(scope="module")
def index(tmp_path_factory) -> str:
    root = tmp_path_factory.mktemp("source_types")
    amount = pa.decimal128(10, 2)
    pq.write_table(pa.table({
        "id": pa.array([1, 2], pa.int64()),
        "balance": pa.array([decimal.Decimal("9.00"), decimal.Decimal("10.00")], amount),
    }), root / "users.parquet")
    pq.write_table(pa.table({
        "id": pa.array([10, 11, 12], pa.int64()),
        "user_id": pa.array([1, 1, 2], pa.int64()),
        "time": pa.array([100, 200, 100], pa.timestamp("s")),
        "amount": pa.array([decimal.Decimal(v) for v in ("9.00", "10.00", "1.00")], amount),
    }), root / "events.parquet")
    decimal_column = {"ingested_dtype": "decimal128", "precision": 10, "scale": 2}
    diskgraph.ingest(json.dumps({"dataset_name": "source_types", "num_shards": 1, "tables": [
        {"name": "users", "primary_key": "id",
         "columns": [{"name": "id", "ingested_dtype": "int64"},
                     {"name": "balance", **decimal_column}],
         "source": {"type": "parquet", "paths": [str(root / "users.parquet")]}},
        {"name": "events", "primary_key": "id", "create_time": "time",
         "columns": [{"name": "id", "ingested_dtype": "int64"},
                     {"name": "user_id", "ingested_dtype": "int64"},
                     {"name": "time", "ingested_dtype": "timestamp"},
                     {"name": "amount", **decimal_column}],
         "fkeys": [{"src_column": "user_id", "dst_table": "users",
                    "reverse": {"time": {"table": "events", "column": "time", "unit": "s"}}}],
         "source": {"type": "parquet", "paths": [str(root / "events.parquet")]}},
    ]}), str(root / "index"))
    return str(root / "index")


def _targets(index: str, build, stype: str = "numerical") -> list:
    builder = DiskGraphTTBuilder()
    builder.node_prediction()
    builder.for_each("users", "id")
    builder.set_target_stype(stype)
    build(builder)
    staging = pathlib.Path(execute_tt_gen(builder.build(), index, "tt")["staging_dir"])
    table = pa.concat_tables(pq.read_table(p) for p in sorted(staging.rglob("part-*.parquet")))
    return sorted(zip(table["ENTITY1"].to_pylist(), table["TARGET1"].to_pylist()))


@pytest.mark.parametrize("func, expected", [
    ("sum", [19.0, 1.0]), ("avg", [9.5, 1.0]), ("min", [9.0, 1.0]), ("max", [10.0, 1.0]),
])
def test_decimal_reductions_use_the_values(index, func, expected):
    targets = _targets(index, lambda b: b.aggregate("t", EVENTS, func, target_col="amount"))
    assert targets == list(zip([1, 2], expected))


@pytest.mark.parametrize("row_filter, expected", [
    (RowFilter.compare("amount", ">", 9.5), [1, 0]),
    (RowFilter.compare("amount", "==", decimal.Decimal("10")), [1, 0]),
    (RowFilter.compare_to_entity("amount", "<", "balance"), [0, 1]),
])
def test_decimal_filters_compare_values(index, row_filter, expected):
    targets = _targets(index, lambda b: b.aggregate("t", EVENTS, "count", row_filter=row_filter))
    assert targets == list(zip([1, 2], expected))


def test_decimal_targets_are_numbers(index):
    assert _targets(index, lambda b: b.column("t", "users", "balance")) == [(1, 9.0), (2, 10.0)]

    def larger(builder):
        builder.column("balance", "users", "balance")
        builder.compare("t", "balance", ">", 9.5)

    assert _targets(index, larger, "categorical") == [(1, False), (2, True)]


@pytest.mark.parametrize("operation", ["min", "max"])
def test_temporal_decimal_extrema_are_refused(index, operation):
    engine = SamplerEngine(index)
    relation = next(r["name"] for r in json.loads(engine.get_type_mapping())["edge_relations"]
                    if r["src_type"] == "users")
    plan = {"frames": [{"name": "f", "root_type": "users", "path": [
        {"relation": relation, "destination_type": "events"}], "time_column": "time",
        "start_seconds": -1000, "stop_seconds": 0, "distinct": True, "columns": ["amount"]}],
        "sequences": [],
        "aggregates": [{"name": "x", "frame": "f", "operation": operation, "column": "amount"}]}
    roots = {"users": engine.lookup_node_ids("users", pa.array([1], pa.int64()))}
    with pytest.raises(diskgraph.DiskgraphConfigError, match="decimal column"):
        engine.query_temporal(roots, {"users": [1000]}, json.dumps(plan))
