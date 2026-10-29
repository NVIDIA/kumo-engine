# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded sequences must match the full-frame path, including timestamp ties."""

import copy
import json
from datetime import UTC

import diskgraph
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture(scope="module")
def history_graph(tmp_path_factory):
    root = tmp_path_factory.mktemp("sequence_limit")
    pq.write_table(pa.table({"id": [1, 2, 3]}), root / "users.parquet")
    rng = np.random.default_rng(71)
    times = 1_700_000_000 + rng.integers(0, 12, 100)
    orders = pa.table(
        {
            "id": rng.permutation(np.arange(-50, 50)),
            "owner": np.where(np.arange(100) < 80, 1, 3),
            "time": pa.array(times, type=pa.timestamp("s")),
            "other_time": pa.array(times[::-1], type=pa.timestamp("s")),
            "value": [None if i % 7 == 0 else float(i) for i in range(100)],
        }
    )
    pq.write_table(orders, root / "orders.parquet")
    config = {
        "dataset_name": "sequence_limit",
        "num_shards": 2,
        "tables": [
            {
                "name": "users",
                "primary_key": "id",
                "columns": [{"name": "id", "ingested_dtype": "int64"}],
                "source": {"type": "parquet", "paths": [str(root / "users.parquet")]},
            },
            {
                "name": "orders",
                "primary_key": "id",
                "create_time": "time",
                "columns": [
                    {"name": name, "ingested_dtype": dtype}
                    for name, dtype in [
                        ("id", "int64"),
                        ("owner", "int64"),
                        ("time", "timestamp"),
                        ("other_time", "timestamp"),
                        ("value", "float64"),
                    ]
                ],
                "fkeys": [
                    {
                        "src_column": "owner",
                        "dst_table": "users",
                        "reverse": {
                            "time": {"table": "orders", "column": "time", "unit": "s"}
                        },
                    }
                ],
                "source": {"type": "parquet", "paths": [str(root / "orders.parquet")]},
            },
        ],
    }
    diskgraph.ingest(
        json.dumps(config),
        str(root / "index"),
        col_stats=False,
        context_threads=2,
        encode_threads=2,
        pk_map_threads=2,
        edges_map_threads=2,
        index_threads=2,
    )
    engine = diskgraph.SamplerEngine(str(root / "index"))
    mapping = json.loads(engine.get_type_mapping())
    relations = mapping["edge_relations"]
    reverse = next(r["name"] for r in relations if r["src_type"] == "users")
    forward = next(r["name"] for r in relations if r["src_type"] == "orders")
    return engine, orders.to_pylist(), reverse, forward, config


def plan_for(
    history_graph, *, limit=3, order="descending", clock="time", distinct=True
):
    return {
        "frames": [
            {
                "name": "f",
                "root_type": "users",
                "path": [{"relation": history_graph[2], "destination_type": "orders"}],
                "time_column": clock,
                "start_seconds": -8,
                "stop_seconds": 0,
                "distinct": distinct,
                "columns": ["id", "value"],
            }
        ],
        "sequences": [
            {
                "name": "s",
                "frame": "f",
                "columns": ["id", "value"],
                "order": order,
                "limit": limit,
            }
        ],
        "aggregates": [],
    }


def query(history_graph, plan, output_format="inspectable"):
    engine = history_graph[0]
    ids = engine.lookup_node_ids("users", pa.array([1, 2, 3, 1]))
    return engine.query_temporal(
        {"users": ids},
        {"users": [1_700_000_010, 1_700_000_010, 1_700_000_008, 1_700_000_005]},
        json.dumps(plan),
        output_format=output_format,
    )


@pytest.mark.parametrize("order", ["ascending", "descending"])
@pytest.mark.parametrize("limit", [1, 3, 20, 100, None])
@pytest.mark.parametrize("distinct", [True, False])
def test_limited_sequence_matches_source(history_graph, order, limit, distinct):
    plan = plan_for(history_graph, order=order, limit=limit, distinct=distinct)
    result = query(history_graph, plan)["sequences"][0]
    offsets = [0]
    expected = []
    for owner, cutoff in [(1, 10), (2, 10), (3, 8), (1, 5)]:
        rows = [
            r
            for r in history_graph[1]
            if r["owner"] == owner
            and 1_700_000_000 + cutoff - 8
            <= r["time"].replace(tzinfo=UTC).timestamp()
            < 1_700_000_000 + cutoff
        ]
        rows.sort(
            key=lambda r: (r["time"], str(r["id"])), reverse=order == "descending"
        )
        expected.extend({k: r[k] for k in ("id", "value")} for r in rows[:limit])
        offsets.append(len(expected))
    assert result["offsets"] == offsets
    assert result["batch"].to_pylist() == expected


def test_limit_reduces_feature_decoding_but_keeps_boundary_ties(history_graph):
    limited = query(history_graph, plan_for(history_graph))
    full = query(history_graph, plan_for(history_graph, limit=None))
    assert limited["stats"]["rows_decoded"] < full["stats"]["rows_decoded"]
    # A timestamp tie may straddle the limit: all tied keys must reach the old
    # primary-key sorter, not just whichever edges happen to be encountered first.
    assert limited["stats"]["rows_decoded"] > limited["sequences"][0]["batch"].num_rows


@pytest.mark.parametrize(
    "guard", ["aggregate", "multiple_sequences", "other_clock", "multihop"]
)
def test_unsupported_fast_paths_keep_full_frame(history_graph, guard):
    plan = plan_for(history_graph)
    if guard == "aggregate":
        plan["aggregates"] = [
            {"name": "total", "frame": "f", "operation": "sum", "column": "value"}
        ]
    elif guard == "multiple_sequences":
        plan["sequences"].append(
            {**plan["sequences"][0], "name": "other", "order": "ascending"}
        )
    elif guard == "other_clock":
        plan["frames"][0]["time_column"] = "other_time"
    else:
        plan["frames"][0]["path"] += [
            {"relation": history_graph[3], "destination_type": "users"},
            {"relation": history_graph[2], "destination_type": "orders"},
        ]
        plan["frames"][0]["distinct"] = False
    full_plan = copy.deepcopy(plan)
    for seq in full_plan["sequences"]:
        seq["limit"] = None
    limited, full = query(history_graph, plan), query(history_graph, full_plan)
    assert limited["stats"] == full["stats"]
    assert limited["aggregates"] == full["aggregates"]
    for a, b in zip(limited["sequences"], full["sequences"]):
        rows = b["batch"].to_pylist()
        expected = []
        for start, stop in zip(b["offsets"], b["offsets"][1:]):
            expected.extend(rows[start:stop][:3])
        assert a["batch"].to_pylist() == expected


def test_model_and_length_paths_agree(history_graph):
    plan = plan_for(history_graph)
    inspected = query(history_graph, plan)["sequences"][0]
    model = query(history_graph, plan, "model")["sequences"][0]
    for name in ("offsets", "timestamps", "node_ids"):
        assert model[name].to_pylist() == inspected[name]
    assert model["batch"].equals(inspected["batch"])
    engine = history_graph[0]
    ids = engine.lookup_node_ids("users", pa.array([1, 2, 3, 1]))
    lengths = engine.query_temporal_lengths(
        {"users": ids},
        {"users": [1_700_000_010, 1_700_000_010, 1_700_000_008, 1_700_000_005]},
        json.dumps(plan),
    )["sequences"][0]["lengths"].to_pylist()
    assert lengths == np.diff(inspected["offsets"]).tolist()


@pytest.mark.parametrize("order", ["ascending", "descending"])
@pytest.mark.parametrize("output_order", ["ascending", "descending"])
@pytest.mark.parametrize("limit", [1, 3, 100, None])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("output_format", ["inspectable", "model"])
def test_output_order_preserves_selected_rows_and_aggregates(
    history_graph, order, output_order, limit, shared, output_format
):
    plan = plan_for(history_graph, order=order, limit=limit)
    if shared:
        plan["aggregates"] = [
            {"name": "total", "frame": "f", "operation": "sum", "column": "value"}
        ]
    original = query(history_graph, plan, output_format)
    plan["sequences"][0]["output_order"] = output_order
    actual = query(history_graph, plan, output_format)
    before, after = original["sequences"][0], actual["sequences"][0]
    offsets = before["offsets"]
    if output_format == "model":
        offsets = offsets.to_pylist()
    indices = [
        row
        for start, stop in zip(offsets, offsets[1:])
        for row in (
            range(start, stop)
            if order == output_order
            else range(stop - 1, start - 1, -1)
        )
    ]
    assert after["batch"].equals(before["batch"].take(pa.array(indices, type=pa.int64())))
    for name in ("node_ids", "timestamps"):
        expected, result = before[name], after[name]
        if output_format == "model":
            expected, result = expected.to_pylist(), result.to_pylist()
        assert result == [expected[i] for i in indices]
    assert (
        after["offsets"].to_pylist() if output_format == "model" else after["offsets"]
    ) == offsets
    assert actual["stats"] == original["stats"]
    for a, b in zip(actual["aggregates"], original["aggregates"]):
        a, b = a["values"], b["values"]
        assert a.equals(b) if output_format == "model" else a == b


def test_randomized_limits_match_aggregate_shared_full_frame(history_graph):
    rng = np.random.default_rng(93)
    for _ in range(160):
        plan = plan_for(
            history_graph,
            limit=int(rng.integers(1, 110)),
            order=rng.choice(["ascending", "descending"]),
            distinct=bool(rng.integers(2)),
        )
        frame = plan["frames"][0]
        frame["start_seconds"] = int(rng.integers(-20, 5))
        frame["stop_seconds"] = frame["start_seconds"] + int(rng.integers(1, 20))
        if rng.integers(3) == 0:
            frame["time_column"] = "other_time"
        if rng.integers(4) == 0:
            frame["start_seconds"] = None
        if rng.integers(4) == 0:
            frame["stop_seconds"] = None
        reference = copy.deepcopy(plan)
        # A full-window consumer prevents early row pruning.
        reference["aggregates"] = [
            {"name": "all_rows", "frame": "f", "operation": "count"}
        ]
        actual = query(history_graph, plan)["sequences"][0]
        expected = query(history_graph, reference)["sequences"][0]
        for key in ("name", "offsets", "timestamps", "node_ids"):
            assert actual[key] == expected[key]
        assert actual["batch"].equals(expected["batch"])


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_limits_still_rejected(history_graph, limit):
    with pytest.raises(diskgraph.DiskgraphConfigError):
        query(history_graph, plan_for(history_graph, limit=limit))


def test_duplicate_source_keys_preserve_distinct_and_repeated_rows(
    history_graph, tmp_path
):
    config = copy.deepcopy(history_graph[4])
    original = pq.read_table(config["tables"][1]["source"]["paths"][0])
    path = tmp_path / "duplicates.parquet"
    pq.write_table(pa.concat_tables([original] * 3), path)
    config["tables"][1]["source"]["paths"] = [str(path)]
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
    duplicate_graph = (
        diskgraph.SamplerEngine(str(tmp_path / "index")),
        *history_graph[1:],
    )
    for distinct in (True, False):
        for order in ("ascending", "descending"):
            for limit in (1, 3, 20):
                plan = plan_for(
                    duplicate_graph, distinct=distinct, order=order, limit=limit
                )
                full = copy.deepcopy(plan)
                full["aggregates"] = [
                    {"name": "all", "frame": "f", "operation": "count"}
                ]
                actual, expected = (
                    query(duplicate_graph, plan),
                    query(duplicate_graph, full),
                )
                a, b = actual["sequences"][0], expected["sequences"][0]
                for key in ("offsets", "timestamps", "node_ids"):
                    assert a[key] == b[key]
                assert a["batch"].equals(b["batch"])


@pytest.fixture(
    params=[
        (bits, anchor, "s", False)
        for bits in (8, 16, 32)
        for anchor in (0, 1_700_000_000)
    ]
    + [(32, 0, "s", True), (16, 0, "day", False)]
)
def fallback_clock_graph(history_graph, tmp_path, request):
    bits, anchor, unit, weighted = request.param
    ceiling = anchor + (1 << (bits - 1)) - 1
    config = copy.deepcopy(history_graph[4])
    orders = pa.table(
        {
            "id": [1, 2, 3],
            "owner": [1, 1, 1],
            "time": pa.array(
                [ceiling - 20, ceiling - 10, ceiling + 200], type=pa.timestamp("s")
            ),
            "value": [10.0, 20.0, 30.0],
        }
    )
    path = tmp_path / "orders.parquet"
    pq.write_table(orders, path)
    config["tables"][1]["columns"] = [
        column
        for column in config["tables"][1]["columns"]
        if column["name"] != "other_time"
    ]
    config["tables"][1]["source"]["paths"] = [str(path)]
    reverse = config["tables"][1]["fkeys"][0]["reverse"]
    reverse["time"].update(bitwidth=bits, anchor_time=anchor, unit=unit)
    if weighted:
        reverse["weight"] = {"table": "orders", "expr": "value", "dtype": "float32"}
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
    return diskgraph.SamplerEngine(str(tmp_path / "index")), ceiling, history_graph


@pytest.mark.parametrize("limit", [1, 2])
@pytest.mark.parametrize("order", ["ascending", "descending"])
def test_clipped_or_coarse_timestamps_cannot_displace_eligible_rows(
    fallback_clock_graph, limit, order
):
    engine, ceiling, original = fallback_clock_graph
    plan = plan_for(original, limit=limit, order=order)
    plan["frames"][0]["start_seconds"] = None
    ids = engine.lookup_node_ids("users", pa.array([1]))
    result = engine.query_temporal(
        {"users": ids}, {"users": [ceiling + 100]}, json.dumps(plan)
    )
    expected_ids = [1, 2] if order == "ascending" else [2, 1]
    assert (
        result["sequences"][0]["batch"].column("id").to_pylist() == expected_ids[:limit]
    )
    # A clipped or coarse clock needs the existing full-frame rowpack check.
    assert result["stats"]["rows_decoded"] == 3


@pytest.mark.parametrize(
    "operation",
    [
        "count",
        "count_distinct",
        "sum",
        "mean",
        "min",
        "max",
        "std_population",
        "std_sample",
        "var_population",
        "var_sample",
        "recency",
        "mode",
        "grouped",
        "nested_grouped",
    ],
)
@pytest.mark.parametrize("order", ["ascending", "descending"])
@pytest.mark.parametrize("distinct", [True, False])
def test_shared_history_defers_only_sequence_columns(
    history_graph, operation, order, distinct
):
    plan = plan_for(history_graph, order=order, distinct=distinct)
    plan["frames"][0]["columns"] += ["owner", "other_time"]
    plan["sequences"][0]["columns"] += ["owner", "other_time"]
    aggregate = {"name": "a", "frame": "f", "operation": operation}
    required = {"id", "time"}
    if operation in {"grouped", "nested_grouped"}:
        required.update(("owner", "value"))
        aggregate.update(
            operation="mean",
            column="subtotal",
            groups=[
                {
                    "keys": ["owner"],
                    "values": [
                        {"name": "subtotal", "operation": "sum", "column": "value"}
                    ],
                }
            ],
        )
        if operation == "nested_grouped":
            aggregate["groups"].append(
                {
                    "keys": ["owner"],
                    "values": [
                        {"name": "second", "operation": "mean", "column": "subtotal"}
                    ],
                }
            )
            aggregate["column"] = "second"
    elif operation != "count":
        aggregate["column"] = "time" if operation == "recency" else "value"
        required.add(aggregate["column"])
    plan["aggregates"] = [aggregate]
    unlimited = copy.deepcopy(plan)
    unlimited["sequences"][0]["limit"] = None
    full = query(history_graph, unlimited)
    for output_format in ("inspectable", "model"):
        actual = query(history_graph, plan, output_format)
        a, b = actual["sequences"][0], full["sequences"][0]
        selected = [
            row
            for start, stop in zip(b["offsets"], b["offsets"][1:])
            for row in range(start, min(start + 3, stop))
        ]
        assert a["batch"].equals(b["batch"].take(pa.array(selected, type=pa.int64())))
        for name in ("node_ids", "timestamps"):
            values = a[name].to_pylist() if output_format == "model" else a[name]
            assert values == [b[name][row] for row in selected]
        values = actual["aggregates"][0]["values"]
        assert (values.to_pylist() if output_format == "model" else values) == full[
            "aggregates"
        ][0]["values"]
        assert actual["stats"]["edges_visited"] == full["stats"]["edges_visited"]
        # Shared columns (e.g. amount) are decoded once, not again for the history.
        n, k = full["stats"]["rows_decoded"], len(selected)
        assert actual["stats"]["values_decoded"] == n * len(required) + k * (
            5 - len(required)
        )
        assert actual["stats"]["values_decoded"] < n * 5


@pytest.mark.parametrize(
    "shape", ["other_clock", "multihop", "unwindowed", "empty", "future"]
)
def test_shared_history_keeps_existing_row_selection(history_graph, shape):
    plan = plan_for(history_graph)
    plan["frames"][0]["columns"].append("owner")
    plan["sequences"][0]["columns"].append("owner")
    plan["aggregates"] = [
        {"name": "total", "frame": "f", "operation": "sum", "column": "value"},
        {"name": "count", "frame": "f", "operation": "count"},
    ]
    frame = plan["frames"][0]
    if shape == "other_clock":
        frame["time_column"] = "other_time"
    elif shape == "multihop":
        frame["path"] += [
            {"relation": history_graph[3], "destination_type": "users"},
            {"relation": history_graph[2], "destination_type": "orders"},
        ]
        frame["distinct"] = False
    elif shape == "unwindowed":
        frame["start_seconds"] = frame["stop_seconds"] = None
    elif shape == "empty":
        frame.update(start_seconds=100, stop_seconds=200)
    else:
        frame.update(start_seconds=0, stop_seconds=10)
    unlimited = copy.deepcopy(plan)
    unlimited["sequences"][0]["limit"] = None
    actual, full = query(history_graph, plan), query(history_graph, unlimited)
    assert actual["aggregates"] == full["aggregates"]
    a, b = actual["sequences"][0], full["sequences"][0]
    selected = [
        row
        for start, stop in zip(b["offsets"], b["offsets"][1:])
        for row in range(start, min(start + 3, stop))
    ]
    assert a["batch"].equals(b["batch"].take(pa.array(selected, type=pa.int64())))
    assert actual["stats"]["values_decoded"] <= full["stats"]["values_decoded"]


def test_shared_history_keeps_column_refusals(history_graph):
    plan = plan_for(history_graph)
    plan["aggregates"] = [{"name": "count", "frame": "f", "operation": "count"}]
    plan["sequences"][0]["columns"].append("undeclared")
    with pytest.raises(diskgraph.DiskgraphConfigError, match="did not project column"):
        query(history_graph, plan)
    plan["frames"][0]["columns"].append("undeclared")
    with pytest.raises(diskgraph.DiskgraphConfigError, match="has no column"):
        query(history_graph, plan)


@pytest.mark.parametrize("limit", [20, 100])
def test_shared_history_avoids_extra_pass_for_low_selectivity(history_graph, limit):
    plan = plan_for(history_graph, limit=limit)
    plan["frames"][0]["columns"].append("owner")
    plan["sequences"][0]["columns"].append("owner")
    plan["aggregates"] = [{"name": "count", "frame": "f", "operation": "count"}]
    unlimited = copy.deepcopy(plan)
    unlimited["sequences"][0]["limit"] = None
    actual, full = query(history_graph, plan), query(history_graph, unlimited)
    assert actual["stats"] == full["stats"]
    a, b = actual["sequences"][0], full["sequences"][0]
    selected = [
        row
        for start, stop in zip(b["offsets"], b["offsets"][1:])
        for row in range(start, min(start + limit, stop))
    ]
    assert a["batch"].equals(b["batch"].take(pa.array(selected, type=pa.int64())))


@pytest.mark.parametrize("limit,deferred", [(8, True), (9, False)])
def test_shared_history_selectivity_boundary(history_graph, limit, deferred):
    plan = plan_for(history_graph, limit=limit)
    plan["frames"][0]["columns"].append("owner")
    plan["sequences"][0]["columns"].append("owner")
    plan["aggregates"] = [{"name": "count", "frame": "f", "operation": "count"}]
    result = query(history_graph, plan)
    # There are 99 eligible candidates. The caps select 24 vs 27 rows: either
    # side of the conservative one-quarter boundary, without changing answers.
    returned = result["sequences"][0]["batch"].num_rows
    assert returned == limit * 3
    assert result["stats"]["rows_decoded"] == 99 + (returned if deferred else 0)


def test_shared_history_preserves_types_and_floating_point_order(
    history_graph, tmp_path
):
    config = copy.deepcopy(history_graph[4])
    source = pq.read_table(config["tables"][1]["source"]["paths"][0])
    source = source.set_column(
        source.schema.get_field_index("value"),
        "value",
        pa.array([1e16, 1.0, -1e16, None] * 25),
    )
    extra = {
        "description": (pa.array([None, "alpha", "beta", "alpha"] * 25), "string"),
        "tags": (pa.array([None, [], ["a", "b"], ["c"]] * 25), "string_array"),
        "enabled": (pa.array([True, False, None, True] * 25), "bool"),
    }
    for name, (values, dtype) in extra.items():
        source = source.append_column(name, values)
        config["tables"][1]["columns"].append({"name": name, "ingested_dtype": dtype})
    path = tmp_path / "typed.parquet"
    pq.write_table(source, path)
    config["tables"][1]["source"]["paths"] = [str(path)]
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
    graph = (diskgraph.SamplerEngine(str(tmp_path / "index")), *history_graph[1:])
    for operation in ("sum", "mean", "std_sample", "mode"):
        plan = plan_for(graph)
        plan["frames"][0]["columns"] += list(extra)
        plan["sequences"][0]["columns"] = [
            "tags",
            "value",
            "description",
            "enabled",
            "id",
        ]
        plan["aggregates"] = [
            {
                "name": "a",
                "frame": "f",
                "operation": operation,
                "column": "description" if operation == "mode" else "value",
            }
        ]
        unlimited = copy.deepcopy(plan)
        unlimited["sequences"][0]["limit"] = None
        actual, full = query(graph, plan), query(graph, unlimited)
        assert actual["aggregates"] == full["aggregates"]
        a, b = actual["sequences"][0], full["sequences"][0]
        selected = [
            row
            for start, stop in zip(b["offsets"], b["offsets"][1:])
            for row in range(start, min(start + 3, stop))
        ]
        assert a["batch"].equals(b["batch"].take(pa.array(selected, type=pa.int64())))
        assert actual["stats"]["values_decoded"] < full["stats"]["values_decoded"]


def _window_plan(original, *, start, limit):
    plan = plan_for(original, limit=limit)
    plan["frames"][0]["start_seconds"] = start
    plan["frames"][0]["columns"] = ["id"]
    plan["sequences"][0]["columns"] = ["id"]
    return plan


def _ids_and_lengths(engine, plan, cutoff):
    ids = engine.lookup_node_ids("users", pa.array([1]))
    result = engine.query_temporal({"users": ids}, {"users": [cutoff]}, json.dumps(plan))
    lengths = engine.query_temporal_lengths(
        {"users": ids}, {"users": [cutoff]}, json.dumps(plan)
    )["sequences"][0]["lengths"].to_pylist()
    return sorted(result["sequences"][0]["batch"].column("id").to_pylist()), lengths


@pytest.mark.parametrize("limit", [1, None])
def test_window_above_clock_ceiling_keeps_clipped_row(fallback_clock_graph, limit):
    # The row at ceiling + 200 is stored at the ceiling; its real time is in the window.
    engine, ceiling, original = fallback_clock_graph
    plan = _window_plan(original, start=-200, limit=limit)
    ids, lengths = _ids_and_lengths(engine, plan, ceiling + 300)
    assert ids == [3]
    assert lengths == [1]


@pytest.mark.parametrize("limit", [1, 3, None])
def test_lengths_match_rows_when_a_clipped_row_is_outside_the_window(
    fallback_clock_graph, limit
):
    engine, ceiling, original = fallback_clock_graph
    plan = _window_plan(original, start=None, limit=limit)
    ids, lengths = _ids_and_lengths(engine, plan, ceiling + 100)
    assert ids == ([2] if limit == 1 else [1, 2])
    assert lengths == [len(ids)]


@pytest.fixture(params=[(bits, anchor) for bits in (8, 16, 32) for anchor in (0, 1_700_000_000)])
def floor_clock_graph(history_graph, tmp_path, request):
    bits, anchor = request.param
    # The lowest code is reserved for NULL, so the lowest exact time is one above it.
    floor = anchor - (1 << (bits - 1)) + 1
    config = copy.deepcopy(history_graph[4])
    orders = pa.table(
        {
            "id": [1, 2, 3],
            "owner": [1, 1, 1],
            "time": pa.array([floor - 50, floor + 10, floor + 20], type=pa.timestamp("s")),
            "value": [10.0, 20.0, 30.0],
        }
    )
    path = tmp_path / "orders.parquet"
    pq.write_table(orders, path)
    config["tables"][1]["columns"] = [
        column for column in config["tables"][1]["columns"] if column["name"] != "other_time"
    ]
    config["tables"][1]["source"]["paths"] = [str(path)]
    config["tables"][1]["fkeys"][0]["reverse"]["time"].update(
        bitwidth=bits, anchor_time=anchor, unit="s"
    )
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
    return diskgraph.SamplerEngine(str(tmp_path / "index")), floor, history_graph


@pytest.mark.parametrize("limit", [1, None])
@pytest.mark.parametrize(
    ("start", "cutoff_offset", "expected"),
    [(-200, 15, [1, 2]), (-10, 15, [2]), (-200, -40, [1])],
)
def test_rows_below_clock_floor_use_their_real_time(
    floor_clock_graph, limit, start, cutoff_offset, expected
):
    # The row at floor - 50 is stored in the NULL code; it is a real time, not NULL.
    engine, floor, original = floor_clock_graph
    plan = _window_plan(original, start=start, limit=limit)
    ids, lengths = _ids_and_lengths(engine, plan, floor + cutoff_offset)
    if limit is None:
        assert ids == expected
    else:
        assert len(ids) == 1 and ids[0] in expected
    assert lengths == [len(ids)]
