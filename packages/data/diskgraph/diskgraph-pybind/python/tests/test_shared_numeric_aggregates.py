# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared execution preserves each frontend's row order and numeric policies."""

import json
import math

import diskgraph
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture(scope="module")
def numeric_graph(tmp_path_factory):
    root = tmp_path_factory.mktemp("shared_numeric")
    pq.write_table(pa.table({"id": list(range(1, 9))}), root / "users.parquet")
    owners = [1, 1, 1, 2, 3, 3, 4, 4] + [6] * 600
    values = [1e16, -1e16, 1.0, None, float("inf"), 2.0, -0.0, 0.0]
    values += [None if i % 7 == 0 else float(i) for i in range(1, 601)]
    times = [1, 2, 3, 1, 1, 2, 1, 2] + list(range(1, 601))
    owners += [7, 7, 8, 8, 8]
    values += [float("nan"), 2.0, 1.0, 1.0, 1.0]
    times += [1, 2, 1, 2, 3]
    integer = [0] * (len(owners) - 5) + [2, 1, 2**53, 1, 1]
    pq.write_table(
        pa.table({"id": range(len(owners)), "owner": owners,
                  "time": pa.array(times, type=pa.timestamp("s")), "value": values, "integer": integer}),
        root / "events.parquet",
    )
    config = {"dataset_name": "shared_numeric", "num_shards": 1, "tables": [
        {"name": "users", "primary_key": "id", "columns": [{"name": "id", "ingested_dtype": "int64"}],
         "source": {"type": "parquet", "paths": [str(root / "users.parquet")]}},
        {"name": "events", "primary_key": "id", "create_time": "time",
         "columns": [{"name": name, "ingested_dtype": dtype} for name, dtype in
                     [("id", "int64"), ("owner", "int64"), ("time", "timestamp"), ("value", "float64"), ("integer", "int64")]],
         "fkeys": [{"src_column": "owner", "dst_table": "users", "reverse": {
             "time": {"table": "events", "column": "time", "unit": "s"}}}],
         "source": {"type": "parquet", "paths": [str(root / "events.parquet")]}},
    ]}
    diskgraph.ingest(
        json.dumps(config), str(root / "index"), col_stats=False,
        context_threads=2, encode_threads=2, pk_map_threads=2, edges_map_threads=2, index_threads=2,
    )
    engine = diskgraph.SamplerEngine(str(root / "index"), num_threads=1)
    relation = next(
        r["name"] for r in json.loads(engine.get_type_mapping())["edge_relations"]
        if r["src_type"] == "users"
    )
    return engine, relation


def query(numeric_graph, keys, *, general=False, future=False, operations=("count", "sum", "mean")):
    engine, relation = numeric_graph
    aggregates = [
        {"name": "target:" + operation, "frame": "f", "operation": operation,
         "column": None if operation == "count" else "value"}
        for operation in operations
    ]
    plan = {"frames": [{"name": "f", "root_type": "users", "path": [
        {"relation": relation, "destination_type": "events"}], "time_column": "time",
        "start_seconds": 0 if future else -2000, "stop_seconds": 2000 if future else 0,
        "distinct": True, "columns": ["value"]}],
        "sequences": [], "aggregates": aggregates}
    if general:
        # A variance consumer keeps the existing general frame evaluator. The
        # requested count/sum/mean still have exactly the same input rows.
        plan["aggregates"].append({"name": "variance", "frame": "f",
                                   "operation": "var_population", "column": "value"})
    result = engine.query_temporal(
        {"users": engine.lookup_node_ids("users", pa.array(keys, type=pa.int64()))},
        {"users": [0 if future else 1000] * len(keys)}, json.dumps(plan), output_format="model",
    )
    return {a["name"]: a["values"] for a in result["aggregates"] if a["name"] != "variance"}


def test_grouped_targets_preserve_cancellation_nulls_and_chunk_boundaries(numeric_graph):
    keys = [1, 2, 5, 6]
    actual = query(numeric_graph, keys)
    reference = query(numeric_graph, keys, general=True)
    for name in actual:
        assert actual[name].equals(reference[name]), name
    assert actual["target:count"].to_pylist() == [3, 1, 0, 600]
    assert actual["target:sum"].to_pylist()[:3] == [1.0, None, 0.0]
    assert actual["target:mean"].to_pylist()[:3] == [1.0 / 3, None, None]


@pytest.mark.parametrize("key", [3, 4])
def test_nonfinite_and_signed_zero_keep_general_numeric_policy(numeric_graph, key):
    actual = query(numeric_graph, [key], operations=("sum", "mean", "min", "max"))
    reference = query(numeric_graph, [key], general=True, operations=("sum", "mean", "min", "max"))
    for name in actual:
        for left, right in zip(actual[name].to_pylist(), reference[name].to_pylist()):
            assert left == right or math.isnan(left) and math.isnan(right), name
            if left == 0.0:
                assert math.copysign(1, left) == math.copysign(1, right), name


def test_numeric_group_keeps_empty_anchor_output_types(numeric_graph):
    result = query(numeric_graph, [])
    assert result["target:count"].type == pa.int64()
    assert result["target:sum"].type == pa.float64()
    assert all(len(values) == 0 for values in result.values())


def test_forward_window_targets_use_the_shared_evaluator(numeric_graph):
    actual = query(numeric_graph, [1, 2, 5, 6], future=True)
    reference = query(numeric_graph, [1, 2, 5, 6], general=True, future=True)
    assert actual["target:count"].to_pylist() == [3, 1, 0, 600]
    for name in actual:
        assert actual[name].equals(reference[name]), name


@pytest.mark.parametrize("key, operation, expected", [
    (3, "mean", 2.0), (7, "mean", 2.0), (7, "min", 2.0), (7, "max", 2.0),
])
def test_pql_policy_survives_explicit_anchor_lowering(numeric_graph, key, operation, expected):
    """PQL numeric policy is preserved even when bulk execution declines NaN/inf."""
    engine, relation = numeric_graph
    builder = diskgraph.DiskGraphTTBuilder()
    builder.for_each("users", "id")
    builder.node_prediction()
    builder.aggregate("value", diskgraph.EdgeRef("users", "id", "events", "owner"),
                      "avg" if operation == "mean" else operation,
                      target_col="value", time_col="time", window=(-2000, 0),
                      window_unit="seconds")
    roots = {"users": engine.lookup_node_ids("users", pa.array([key], type=pa.int64()))}
    result = engine.query_aggregate_plan(roots, {"users": [1000]}, builder.build(),
                                         output_format="model")
    assert result["aggregates"][0]["values"].to_pylist() == [expected]


@pytest.mark.parametrize("operation", ["min", "max"])
def test_integer_extrema_preserve_frontend_output_contract(numeric_graph, operation):
    engine, relation = numeric_graph
    builder = diskgraph.DiskGraphTTBuilder()
    builder.for_each("users", "id")
    builder.node_prediction()
    builder.aggregate("value", diskgraph.EdgeRef("users", "id", "events", "owner"), operation,
                      target_col="integer", time_col="time", window=(-2000, 0),
                      window_unit="seconds")
    roots = {"users": engine.lookup_node_ids("users", pa.array([7], type=pa.int64()))}
    pql = engine.query_aggregate_plan(roots, {"users": [1000]}, builder.build(),
                                     output_format="model")["aggregates"][0]["values"]
    frame = {"name": "f", "root_type": "users", "path": [
        {"relation": relation, "destination_type": "events"}], "time_column": "time",
        "start_seconds": -2000, "stop_seconds": 0, "distinct": True, "columns": ["integer"]}
    reduction = {"name": "value", "frame": "f", "operation": operation, "column": "integer"}
    row = engine.query_temporal(roots, {"users": [1000]}, json.dumps({
        "frames": [frame], "aggregates": [reduction]}), output_format="model")["aggregates"][0]["values"]
    assert pql.type == pa.float64()
    assert row.type == pa.int64()
    assert pql.to_pylist() == row.to_pylist() == [1 if operation == "min" else 2]


def test_large_integer_sum_keeps_order_and_explicit_exact_contract(numeric_graph):
    engine, relation = numeric_graph
    roots = {"users": engine.lookup_node_ids("users", pa.array([8], type=pa.int64()))}
    builder = diskgraph.DiskGraphTTBuilder()
    builder.for_each("users", "id")
    builder.node_prediction()
    builder.aggregate("value", diskgraph.EdgeRef("users", "id", "events", "owner"), "sum",
                      target_col="integer", time_col="time", window=(-2000, 0),
                      window_unit="seconds")
    pql = engine.query_aggregate_plan(roots, {"users": [1000]}, builder.build(),
                                     output_format="model")["aggregates"][0]["values"]
    frame = {"name": "f", "root_type": "users", "path": [
        {"relation": relation, "destination_type": "events"}], "time_column": "time",
        "start_seconds": -2000, "stop_seconds": 0, "distinct": True, "columns": ["integer"]}
    reduction = {"name": "value", "frame": "f", "operation": "sum", "column": "integer"}
    def run():
        return engine.query_temporal(roots, {"users": [1000]}, json.dumps({
            "frames": [frame], "aggregates": [reduction]}), output_format="model")["aggregates"][0]["values"]
    # Chronological and PQL traversal order intentionally differ. A floating
    # reduction preserves that order; exact integer SUM is order-independent.
    assert pql.to_pylist() == [float(2**53 + 2)]
    assert run().to_pylist() == [float(2**53)]
    frame["preserve_traversal_order"] = True
    assert run().equals(pql)
    frame["preserve_traversal_order"] = False
    reduction["sum_type"] = "int64"
    exact = run()
    assert exact.type == pa.int64()
    assert exact.to_pylist() == [2**53 + 2]
