# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporal query output shapes: empty anchor batches and refused projections."""

import json

import diskgraph
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

AGGREGATES = [
    {"name": "count", "frame": "f", "operation": "count"},
    {"name": "max", "frame": "f", "operation": "max", "column": "value"},
    {"name": "mode_id", "frame": "f", "operation": "mode", "column": "id"},
    {"name": "mean", "frame": "f", "operation": "mean", "column": "value"},
    {"name": "sum", "frame": "f", "operation": "sum", "column": "id", "sum_type": "int64"},
]


@pytest.fixture(scope="module")
def events_graph(tmp_path_factory):
    root = tmp_path_factory.mktemp("query_shapes")
    pq.write_table(pa.table({"id": [1]}), root / "users.parquet")
    pq.write_table(
        pa.table(
            {
                "id": [10, 11, 12],
                "owner": [1, 1, 1],
                "time": pa.array([10, 11, 12], type=pa.timestamp("s")),
                "value": pa.array([-0.0, 0.0, 1.0], type=pa.float64()),
            }
        ),
        root / "events.parquet",
    )
    config = {
        "dataset_name": "query_shapes",
        "num_shards": 1,
        "tables": [
            {
                "name": "users",
                "primary_key": "id",
                "columns": [{"name": "id", "ingested_dtype": "int64"}],
                "source": {"type": "parquet", "paths": [str(root / "users.parquet")]},
            },
            {
                "name": "events",
                "primary_key": "id",
                "create_time": "time",
                "columns": [
                    {"name": name, "ingested_dtype": dtype}
                    for name, dtype in [
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
                            "time": {"table": "events", "column": "time", "unit": "s"}
                        },
                    }
                ],
                "source": {"type": "parquet", "paths": [str(root / "events.parquet")]},
            },
        ],
    }
    diskgraph.ingest(json.dumps(config), str(root / "index"), col_stats=False)
    engine = diskgraph.SamplerEngine(str(root / "index"))
    relation = next(
        r["name"]
        for r in json.loads(engine.get_type_mapping())["edge_relations"]
        if r["src_type"] == "users"
    )
    plan = {
        "frames": [
            {
                "name": "f",
                "root_type": "users",
                "path": [{"relation": relation, "destination_type": "events"}],
                "time_column": "time",
                "start_seconds": None,
                "stop_seconds": 0,
                "distinct": True,
                "columns": ["id", "value"],
            }
        ],
        "sequences": [
            {
                "name": "s",
                "frame": "f",
                "columns": ["id", "value"],
                "order": "ascending",
                "limit": None,
            }
        ],
        "aggregates": AGGREGATES,
    }
    return engine, plan


def run(engine, plan, ids, cutoffs, output_format):
    return engine.query_temporal(
        {"users": ids}, {"users": cutoffs}, json.dumps(plan), output_format=output_format
    )


@pytest.mark.parametrize("output_format", ["inspectable", "model"])
def test_empty_anchor_batch_returns_typed_empty_results(events_graph, output_format):
    engine, plan = events_graph
    ids = engine.lookup_node_ids("users", pa.array([1]))
    full = run(engine, plan, ids, [20], output_format)
    empty = run(engine, plan, ids[:0], [], output_format)

    (sequence,) = empty["sequences"]
    assert sequence["batch"].num_rows == 0
    assert sequence["batch"].schema == full["sequences"][0]["batch"].schema
    for key in ("offsets", "node_ids", "timestamps"):
        values = sequence[key]
        if output_format == "model":
            assert values.type == full["sequences"][0][key].type
            values = values.to_pylist()
        assert values == ([0] if key == "offsets" else [])

    assert [a["name"] for a in empty["aggregates"]] == [a["name"] for a in AGGREGATES]
    for got, reference in zip(empty["aggregates"], full["aggregates"]):
        if output_format == "model":
            assert len(got["values"]) == 0
            assert got["values"].type == reference["values"].type
        else:
            assert got["values"] == []


def test_empty_anchor_batch_lengths(events_graph):
    engine, plan = events_graph
    ids = engine.lookup_node_ids("users", pa.array([1]))
    full = engine.query_temporal_lengths({"users": ids}, {"users": [20]}, json.dumps(plan))
    empty = engine.query_temporal_lengths(
        {"users": ids[:0]}, {"users": []}, json.dumps(plan)
    )
    assert full["sequences"][0]["lengths"].to_pylist() == [3]
    lengths = empty["sequences"][0]["lengths"]
    assert lengths.type == full["sequences"][0]["lengths"].type
    assert lengths.to_pylist() == []


@pytest.mark.parametrize("method", ["query_temporal", "query_temporal_lengths"])
def test_zero_column_sequence_is_refused(events_graph, method):
    engine, plan = events_graph
    plan = {**plan, "sequences": [{**plan["sequences"][0], "columns": []}]}
    ids = engine.lookup_node_ids("users", pa.array([1]))
    with pytest.raises(
        diskgraph.DiskgraphConfigError,
        match='sequence "s" must project at least one column',
    ):
        getattr(engine, method)({"users": ids}, {"users": [20]}, json.dumps(plan))
