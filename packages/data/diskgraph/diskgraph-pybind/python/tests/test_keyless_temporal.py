# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keyless temporal destinations preserve source occurrences and native IDs."""

import copy
import json

import diskgraph
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture(scope="module", params=[1, 3])
def keyless_graph(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(f"keyless_{request.param}")
    pq.write_table(pa.table({"id": [1, 2]}), root / "users.parquet")
    pq.write_table(
        pa.table(
            {
                "owner": [1, 1, 1, 1],
                "time": pa.array([9, 9, 10, 11], type=pa.timestamp("s")),
                "value": [2, 2, 3, 100],
                "detail": ["same", "same", "cutoff", "future"],
            }
        ),
        root / "events.parquet",
    )
    config = {
        "dataset_name": "keyless",
        "num_shards": request.param,
        "tables": [
            {
                "name": "users",
                "primary_key": "id",
                "columns": [{"name": "id", "ingested_dtype": "int64"}],
                "source": {"type": "parquet", "paths": [str(root / "users.parquet")]},
            },
            {
                "name": "events",
                "create_time": "time",
                "columns": [
                    {"name": name, "ingested_dtype": dtype}
                    for name, dtype in [
                        ("owner", "int64"),
                        ("time", "timestamp"),
                        ("value", "int64"),
                        ("detail", "string"),
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
    engine = diskgraph.SamplerEngine(str(root / "index"), num_threads=2)
    relation = next(
        r["name"]
        for r in json.loads(engine.get_type_mapping())["edge_relations"]
        if r["src_type"] == "users" and r["dst_type"] == "events"
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
                "columns": ["value"],
            }
        ],
        "sequences": [
            {"name": "s", "frame": "f", "columns": ["value"], "order": "ascending"}
        ],
        "aggregates": [
            {"name": "count", "frame": "f", "operation": "count"},
            {"name": "sum", "frame": "f", "operation": "sum", "column": "value"},
        ],
    }
    return engine, plan


def values(value):
    return value.to_pylist() if hasattr(value, "to_pylist") else value


def query(engine, plan, output_format, ids=(1, 2), cutoff=10):
    roots = engine.lookup_node_ids("users", pa.array(ids, type=pa.int64()))
    return engine.query_temporal(
        {"users": roots},
        {"users": [cutoff] * len(roots)},
        json.dumps(plan),
        output_format=output_format,
    )


@pytest.mark.parametrize("output_format", ["inspectable", "model"])
def test_keyless_sequences_aggregates_and_empty_history(keyless_graph, output_format):
    engine, plan = keyless_graph
    result = query(engine, plan, output_format)
    assert [values(a["values"]) for a in result["aggregates"]] == [[3, 0], [7, 0]]
    sequence = result["sequences"][0]
    assert sequence["batch"].column("value").to_pylist() == [2, 2, 3]
    assert values(sequence["offsets"]) == [0, 3, 3]
    assert values(sequence["timestamps"]) == [9, 9, 10]
    ids = values(sequence["node_ids"])
    assert len(set(ids)) == 3
    again = query(engine, plan, output_format)["sequences"][0]
    assert values(again["node_ids"]) == ids
    assert sequence["batch"].schema.names == ["value"]


@pytest.mark.parametrize("output_format", ["inspectable", "model"])
def test_keyless_window_and_bounded_selection(keyless_graph, output_format):
    engine, original = keyless_graph
    plan = copy.deepcopy(original)
    plan["frames"][0]["start_seconds"] = -1
    result = query(engine, plan, output_format)
    assert [values(a["values"]) for a in result["aggregates"]] == [[1, 0], [3, 0]]
    assert result["sequences"][0]["batch"].column("value").to_pylist() == [3]
    plan["aggregates"] = []
    plan["frames"][0]["start_seconds"] = None
    plan["sequences"][0].update(limit=1, order="descending")
    limited = query(engine, plan, output_format)["sequences"][0]
    full_ids = values(
        query(engine, original, output_format)["sequences"][0]["node_ids"]
    )
    assert values(limited["node_ids"]) == [full_ids[-1]]
    assert limited["batch"].column("value").to_pylist() == [3]
    roots = engine.lookup_node_ids("users", pa.array([1, 2]))
    lengths = engine.query_temporal_lengths(
        {"users": roots}, {"users": [10, 10]}, json.dumps(plan)
    )
    assert values(lengths["sequences"][0]["lengths"]) == [1, 0]


@pytest.mark.parametrize("output_format", ["inspectable", "model"])
def test_keyless_count_without_projected_columns(keyless_graph, output_format):
    engine, original = keyless_graph
    plan = copy.deepcopy(original)
    plan["frames"][0].update(columns=[], time_column=None, stop_seconds=None)
    plan["sequences"] = []
    plan["aggregates"] = plan["aggregates"][:1]
    result = query(engine, plan, output_format)
    assert result["stats"]["rows_decoded"] == 0
    assert values(result["aggregates"][0]["values"]) == [
        4,
        0,
    ]
    assert (
        values(query(engine, plan, output_format, ids=())["aggregates"][0]["values"])
        == []
    )


@pytest.mark.parametrize("output_format", ["inspectable", "model"])
def test_keyless_shared_frame_defers_sequence_columns(keyless_graph, output_format):
    engine, original = keyless_graph
    plan = copy.deepcopy(original)
    plan["frames"][0]["columns"] = ["value", "detail"]
    plan["sequences"][0].update(columns=["detail"], limit=1, order="descending")
    result = query(engine, plan, output_format, cutoff=20)
    assert [values(a["values"]) for a in result["aggregates"]] == [[4, 0], [107, 0]]
    assert result["sequences"][0]["batch"].column("detail").to_pylist() == ["future"]
