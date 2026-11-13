# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common TT plan execution: explicit anchors, overlap reuse and bounded chunks."""

import json

import pyarrow as pa
import pytest
from test_shared_numeric_aggregates import numeric_graph as _numeric_graph


@pytest.fixture(scope="module")
def numeric_graph(tmp_path_factory):
    return _numeric_graph.__wrapped__(tmp_path_factory)


def program(relation):
    frames = [
        {
            "name": name,
            "root_type": "users",
            "path": [{"relation": relation, "destination_type": "events"}],
            "time_column": "time",
            "start_seconds": start,
            "stop_seconds": 0,
            "distinct": True,
            "columns": ["value"],
        }
        for name, start in [("full", -1000), ("recent", -500)]
    ]
    aggregates = [
        {
            "name": name + ":" + operation,
            "frame": name,
            "operation": operation,
            "column": None if operation == "count" else "value",
        }
        for name in ["full", "recent"]
        for operation in ["count", "sum", "var_population"]
    ]
    return {"frames": frames, "aggregates": aggregates}


def staged(fragment):
    return {
        "entity": {"table": "users", "column": "id"},
        "temporal": None, "prediction_type": "node",
        "stages": [{"filter": None, "ops": [
            {"row_aggregate": {"rows": next(f for f in fragment["frames"] if f["name"] == a["frame"]), "reduction": a}}
            for a in fragment["aggregates"]
        ]}],
    }


def values(result):
    return {entry["name"]: entry["values"] for entry in result["aggregates"]}


def test_overlapping_frames_decode_once_and_keep_independent_answers(numeric_graph):
    engine, relation = numeric_graph
    plan = program(relation)
    roots = {
        "users": engine.lookup_node_ids("users", pa.array([6, 1, 6], type=pa.int64()))
    }
    cutoffs = {"users": [1000, 1000, 1000]}
    combined = engine.query_temporal(
        roots, cutoffs, json.dumps(plan), output_format="model"
    )
    assert combined["stats"]["frames_materialized"] == 1
    assert combined["stats"]["rows_decoded"] == 1203
    for frame in plan["frames"]:
        separate = {
            "frames": [frame],
            "aggregates": [
                a for a in plan["aggregates"] if a["frame"] == frame["name"]
            ],
        }
        reference = engine.query_temporal(
            roots, cutoffs, json.dumps(separate), output_format="model"
        )
        for name, expected in values(reference).items():
            assert values(combined)[name].equals(expected), name


@pytest.mark.parametrize("chunk", [1, 2, 7])
def test_dgtt_plan_anchor_chunks_preserve_order_duplicates_and_types(
    numeric_graph, chunk
):
    engine, relation = numeric_graph
    fragment = program(relation)
    plan = staged(fragment)
    roots = {
        "users": engine.lookup_node_ids("users", pa.array([6, 1, 6], type=pa.int64()))
    }
    cutoffs = {"users": [1000, 1000, 1000]}
    actual = engine.query_aggregate_plan(
        roots, cutoffs, json.dumps(plan), anchor_batch_size=chunk
    )
    expected = engine.query_temporal(
        roots, cutoffs, json.dumps(fragment), output_format="model"
    )
    for name, reference in values(expected).items():
        assert values(actual)[name].equals(reference), name
    assert values(actual)["full:count"].to_pylist() == [600, 3, 600]


def test_plan_does_not_silently_drop_staged_side_effects(numeric_graph):
    engine, relation = numeric_graph
    plan = staged(program(relation))
    plan["stages"][0]["filter"] = {"non_empty_list": {"alias": "full:count"}}
    with pytest.raises(Exception, match="stage side effects"):
        engine.query_aggregate_plan({"users": []}, {"users": []}, json.dumps(plan))


@pytest.mark.parametrize("clock", ["milliseconds", "microseconds", "nanoseconds"])
def test_explicit_plan_preserves_bounded_nonseconds_windows(tmp_path, clock):
    import diskgraph
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"id": [1]}), tmp_path / "users.parquet")
    pq.write_table(
        pa.table(
            {
                "id": [1, 2, 3, 4],
                "owner": [1, 1, 1, 1],
                "time": pa.array([1000, 1001, 2000, 2500], type=pa.timestamp("ms")),
                "value": [999.0, 3.0, 5.0, 100.0],
            }
        ),
        tmp_path / "events.parquet",
    )
    config = {
        "dataset_name": "clock_compat",
        "timestamp_unit": clock,
        "num_shards": 1,
        "tables": [
            {
                "name": "users",
                "primary_key": "id",
                "columns": [{"name": "id", "ingested_dtype": "int64"}],
                "source": {
                    "type": "parquet",
                    "paths": [str(tmp_path / "users.parquet")],
                },
            },
            {
                "name": "events",
                "primary_key": "id",
                "create_time": "time",
                "columns": [
                    {"name": n, "ingested_dtype": t}
                    for n, t in [
                        ("id", "int64"),
                        ("owner", "int64"),
                        ("time", "timestamp"),
                        ("value", "float64"),
                    ]
                ],
                "fkeys": [
                    {
                        "src_column": "owner",
                        "dst_table": "users",
                        "reverse": {
                            "time": {"table": "events", "column": "time", "unit": "ms"}
                        },
                    }
                ],
                "source": {
                    "type": "parquet",
                    "paths": [str(tmp_path / "events.parquet")],
                },
            },
        ],
    }
    diskgraph.ingest(
        json.dumps(config),
        str(tmp_path / "index"),
        col_stats=False,
        context_threads=2,
        encode_threads=2,
        pk_map_threads=2,
        edges_map_threads=2,
        index_threads=2,
    )
    engine = diskgraph.SamplerEngine(str(tmp_path / "index"), num_threads=1)
    relation = next(
        r["name"]
        for r in json.loads(engine.get_type_mapping())["edge_relations"]
        if r["src_type"] == "users"
    )
    fragment = program(relation)
    roots = {"users": engine.lookup_node_ids("users", pa.array([1], type=pa.int64()))}
    for frame in fragment["frames"]:
        frame["start_seconds"] = -1
    cutoffs = {"users": [2]}
    expected = engine.query_temporal(
        roots, cutoffs, json.dumps(fragment), output_format="model"
    )
    plan = staged(fragment)
    actual = engine.query_aggregate_plan(
        roots, cutoffs, json.dumps(plan), anchor_batch_size=1
    )
    for name, reference in values(expected).items():
        assert values(actual)[name].equals(reference), name
    assert values(actual)["full:count"].to_pylist() == [2]
    assert values(actual)["full:sum"].to_pylist() == [8.0]
    # Both endpoints are checked in the index clock: include1.001s/2s,
    # exclude the lower endpoint1s and future2.5s. Also verify unbounded input.
    for frame in fragment["frames"]:
        frame.update(time_column=None, start_seconds=None, stop_seconds=None)
    plan = staged(fragment)
    actual = engine.query_aggregate_plan(
        roots, cutoffs, json.dumps(plan), anchor_batch_size=1
    )
    assert values(actual)["full:count"].to_pylist() == [4]
    assert values(actual)["full:sum"].to_pylist() == [1107.0]


def test_pql_and_fluent_share_executable_row_ops(numeric_graph):
    import diskgraph
    engine, relation = numeric_graph
    builder = diskgraph.DiskGraphTTBuilder()
    builder.for_each("users", "id")
    builder.node_prediction()
    builder.aggregate("total", diskgraph.EdgeRef("users", "id", "events", "owner"), "sum", target_col="value", time_col="time", window=(-500,0), window_unit="seconds")
    plan = json.loads(builder.build())
    # Keep compiler/forecasting consumers compatible; execution lowers this to
    # the same RowAggregate stages emitted directly by the fluent frontend.
    assert "aggregate" in plan["stages"][0]["ops"][0]
    assert not plan.get("aggregate_program")
    roots={"users":engine.lookup_node_ids("users",pa.array([1,6],type=pa.int64()))}
    cutoffs={"users":[1000,1000]}
    actual=engine.query_aggregate_plan(roots,cutoffs,json.dumps(plan),output_format="model")
    fragment=program(relation)
    reference=engine.query_temporal(roots,cutoffs,json.dumps(fragment),output_format="model")
    assert values(actual)["total"].to_pylist()==values(reference)["recent:sum"].to_pylist()


def test_prepared_plan_reuses_typed_operations_across_chunks(numeric_graph):
    engine, relation = numeric_graph
    fragment = program(relation)
    plan = staged(fragment)
    prepared = engine.prepare_aggregate_plan("users", json.dumps(plan))
    roots = engine.lookup_node_ids("users", pa.array([6, 1, 6], type=pa.int64()))
    expected = values(engine.query_temporal({"users": roots}, {"users": [1000]*3}, json.dumps(fragment), output_format="model"))
    actual = {name: [] for name in expected}
    for root in roots:
        result = values(prepared.execute([root], [1000], anchor_batch_size=1))
        for name in actual: actual[name].extend(result[name].to_pylist())
    assert actual == {name: array.to_pylist() for name, array in expected.items()}
    with pytest.raises(Exception, match="equal length"):
        prepared.execute(roots, [1000])


@pytest.mark.parametrize("limit", [None, 1])
def test_pql_filtered_scalar_preserves_filter_and_limit(numeric_graph, limit):
    import diskgraph
    engine, _ = numeric_graph
    builder = diskgraph.DiskGraphTTBuilder()
    builder.for_each("users", "id")
    builder.node_prediction()
    builder.aggregate("filtered", diskgraph.EdgeRef("users", "id", "events", "owner"), "sum", target_col="value", time_col="time", window=(-1000, 0), window_unit="seconds")
    plan = json.loads(builder.build())
    op = plan["stages"][0]["ops"][0]["aggregate"]
    op["row_filter"] = {"compare": {"lhs": {"column": "value"}, "op": "gt", "rhs": {"constant": {"float": 1.0}}}}
    op["limit"] = limit
    roots = engine.lookup_node_ids("users", pa.array([1, 6], type=pa.int64()))
    prepared = engine.prepare_aggregate_plan("users", json.dumps(plan))
    result = values(prepared.execute(roots, [1000, 1000]))["filtered"].to_pylist()
    # User1 retains1e16; user6 retains non-null values greater than1. Numeric SUM's historical
    # limit does not cap traversed rows; preserving it must not invent a cap.
    assert result == [1e16, sum(i for i in range(2, 601) if i % 7 != 0)]
