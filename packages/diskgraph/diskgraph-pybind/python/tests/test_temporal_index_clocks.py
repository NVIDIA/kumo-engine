# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporal queries must compare stored times and windows in the same clock."""

import json

import diskgraph
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from test_index_clocks import CLOCKS, DAY, EPOCH_DAY, TICK_NS


def make_graph(root, clock, ids, owners, times):
    root.mkdir()
    pq.write_table(pa.table({"id": [1, 2, 3]}), root / "users.parquet")
    pq.write_table(pa.table({
        "id": ids,
        "owner": owners,
        "time": times,
        "other_time": times,
        "amount": [float(i + 1) for i in range(len(ids))],
        "text": ["payload " * 20] * len(ids),
    }), root / "events.parquet")
    config = {
        "dataset_name": "temporal_clocks", "num_shards": 2,
        "timestamp_unit": clock,
        "tables": [
            {
                "name": "users", "primary_key": "id",
                "columns": [{"name": "id", "ingested_dtype": "int64"}],
                "source": {"type": "parquet", "paths": [str(root / "users.parquet")]},
            },
            {
                "name": "events", "primary_key": "id", "create_time": "time",
                "columns": [
                    {"name": name, "ingested_dtype": dtype}
                    for name, dtype in [("id", "int64"), ("owner", "int64"),
                                        ("time", "timestamp"), ("other_time", "timestamp"),
                                        ("amount", "float64"), ("text", "string")]
                ],
                "fkeys": [{"src_column": "owner", "dst_table": "users", "reverse": {
                    "time": {"table": "events", "column": "time", "unit": "s"}
                }}],
                "source": {"type": "parquet", "paths": [str(root / "events.parquet")]},
            },
        ],
    }
    diskgraph.ingest(json.dumps(config), str(root / "index"), col_stats=False,
                     context_threads=2, encode_threads=2, pk_map_threads=2,
                     edges_map_threads=2, index_threads=2)
    engine = diskgraph.SamplerEngine(str(root / "index"))
    relations = json.loads(engine.get_type_mapping())["edge_relations"]
    relation = next(r["name"] for r in relations if r["src_type"] == "users")
    return engine, relation


def plan_for(relation, *, time_column="time", start=-6 * DAY, limit=1, shared=False):
    return {
        "frames": [{"name": "f", "root_type": "users", "path": [
            {"relation": relation, "destination_type": "events"}],
            "time_column": time_column, "start_seconds": start, "stop_seconds": 0,
            "distinct": True, "columns": ["id", "amount", "text"]}],
        "sequences": [{"name": "s", "frame": "f", "columns": ["id", "text"],
                       "order": "descending", "limit": limit}],
        "aggregates": ([
            {"name": "n", "frame": "f", "operation": "count"},
            {"name": "total", "frame": "f", "operation": "sum", "column": "amount"},
            {"name": "age", "frame": "f", "operation": "recency", "column": time_column},
        ] if shared else []),
    }


def values(value):
    return value.to_pylist() if isinstance(value, pa.Array) else value


def run(engine, plan, owners, anchors, output_format="model"):
    roots = {"users": engine.lookup_node_ids("users", pa.array(owners, type=pa.int64()))}
    cutoffs = {"users": anchors}
    result = engine.query_temporal(roots, cutoffs, json.dumps(plan), output_format=output_format)
    lengths = engine.query_temporal_lengths(roots, cutoffs, json.dumps(plan))
    sequence = result["sequences"][0]
    assert lengths["sequences"][0]["lengths"].to_pylist() == [
        b - a for a, b in zip(values(sequence["offsets"]), values(sequence["offsets"])[1:])
    ]
    return result, lengths


@pytest.fixture(scope="module", params=CLOCKS)
def day_graph(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("temporal_clocks") / request.param
    # Start-boundary rows, an exact timestamp tie, and one future row.
    times = pa.array([(EPOCH_DAY + d) * DAY for d in (0, 1, 3, 3, 6, 9)], pa.timestamp("s"))
    engine, relation = make_graph(root, request.param, [90, 80, 2, 10, 50, 60], [1] * 6, times)
    return engine, relation, request.param


@pytest.mark.parametrize("output_format", ["inspectable", "model"])
@pytest.mark.parametrize("time_column", ["time", "other_time"])
@pytest.mark.parametrize("shared", [False, True])
def test_all_clocks_select_the_same_rows(day_graph, output_format, time_column, shared):
    engine, relation, _ = day_graph
    plan = plan_for(relation, time_column=time_column, limit=2, shared=shared)
    result, lengths = run(engine, plan, [1, 2, 1],
                          [(EPOCH_DAY + d) * DAY for d in (6, 6, 3)], output_format)
    sequence = result["sequences"][0]
    assert sequence["batch"].column("id").to_pylist() == [50, 2, 2, 10]
    assert values(sequence["offsets"]) == [0, 2, 2, 4]
    assert values(sequence["timestamps"]) == [(EPOCH_DAY + d) * DAY for d in (6, 3, 3, 3)]
    if shared:
        aggregates = {a["name"]: values(a["values"]) for a in result["aggregates"]}
        assert aggregates == {"n": [4, 0, 4], "total": [14.0, 0.0, 10.0],
                              "age": [0.0, None, 0.0]}
    if time_column == "time":
        assert lengths["stats"]["rowpack_timestamps"] == 0
    else:
        assert lengths["stats"]["rowpack_timestamps"] > 0


def test_unbounded_and_delivery_order(day_graph):
    engine, relation, _ = day_graph
    plan = plan_for(relation, start=None, limit=None)
    plan["sequences"][0]["output_order"] = "ascending"
    result, _ = run(engine, plan, [1], [(EPOCH_DAY + 6) * DAY])
    assert result["sequences"][0]["batch"].column("id").to_pylist() == [90, 80, 10, 2, 50]
    plan["frames"][0]["stop_seconds"] = None
    result, _ = run(engine, plan, [1], [(EPOCH_DAY + 6) * DAY])
    assert result["sequences"][0]["batch"].column("id").to_pylist() == [90, 80, 10, 2, 50, 60]


def test_future_window_aggregate_labels(day_graph):
    engine, relation, _ = day_graph
    plan = plan_for(relation, start=0, limit=2, shared=True)
    plan["frames"][0]["stop_seconds"] = 6 * DAY
    result, _ = run(engine, plan, [1, 1], [(EPOCH_DAY + d) * DAY for d in (6, 0)])
    assert result["sequences"][0]["batch"].column("id").to_pylist() == [60, 50, 2]
    aggregates = {a["name"]: values(a["values"]) for a in result["aggregates"]}
    assert aggregates == {"n": [1, 4], "total": [6.0, 14.0], "age": [-3 * DAY, -6 * DAY]}


def test_recency_uses_the_same_rounded_anchor_as_filtering(day_graph):
    engine, relation, clock = day_graph
    plan = plan_for(relation, shared=True)
    anchor = (EPOCH_DAY + 6) * DAY + 4000
    result, _ = run(engine, plan, [1], [anchor])
    rounded = anchor * 10**9 // TICK_NS[clock] * TICK_NS[clock] // 10**9
    assert values(result["aggregates"][2]["values"]) == [rounded - (EPOCH_DAY + 6) * DAY]


@pytest.mark.parametrize("shared", [False, True])
def test_clock_support_preserves_selective_feature_decoding(day_graph, shared):
    engine, relation, _ = day_graph
    plan = plan_for(relation, start=None, limit=1, shared=shared)
    result, _ = run(engine, plan, [1], [(EPOCH_DAY + 6) * DAY])
    assert result["sequences"][0]["batch"].column("id").to_pylist() == [50]
    # Five rows qualify. A sole history decodes only its chosen row; a shared
    # aggregate must read all five amounts, but only the chosen row's text.
    assert result["stats"]["rows_decoded"] == (6 if shared else 1)
    assert result["stats"]["values_decoded"] == (16 if shared else 4)


@pytest.mark.parametrize("output_format", ["inspectable", "model"])
def test_multiple_windows_share_normalized_anchors(day_graph, output_format):
    engine, relation, clock = day_graph
    plan = plan_for(relation, limit=2, shared=True)
    plan["frames"].append({**plan["frames"][0], "name": "future",
                           "start_seconds": 0, "stop_seconds": 6 * DAY})
    plan["sequences"].append({**plan["sequences"][0], "name": "future", "frame": "future"})
    plan["aggregates"] += [
        {**aggregate, "name": "future_" + aggregate["name"], "frame": "future"}
        for aggregate in plan["aggregates"]
    ]
    anchors = [(EPOCH_DAY + 6) * DAY + 4000, (EPOCH_DAY + 6) * DAY, (EPOCH_DAY + 3) * DAY]
    roots = {"users": engine.lookup_node_ids("users", pa.array([1, 2, 1]))}
    cutoffs = {"users": anchors}
    result = engine.query_temporal(roots, cutoffs, json.dumps(plan), output_format=output_format)
    lengths = engine.query_temporal_lengths(roots, cutoffs, json.dumps(plan))
    for sequence, counts, ids, expected in zip(
        result["sequences"], lengths["sequences"],
        [[50, 2, 2, 10], [60, 60, 50]], [[2, 0, 2], [1, 0, 2]],
    ):
        assert sequence["batch"].column("id").to_pylist() == ids
        assert counts["lengths"].to_pylist() == expected
    rounded = anchors[0] * 10**9 // TICK_NS[clock] * TICK_NS[clock] // 10**9
    age = rounded - (EPOCH_DAY + 6) * DAY
    assert {a["name"]: values(a["values"]) for a in result["aggregates"]} == {
        "n": [4, 0, 4], "total": [14.0, 0.0, 10.0], "age": [age, None, 0],
        "future_n": [1, 0, 2], "future_total": [6.0, 0.0, 11.0],
        "future_age": [age - 3 * DAY, None, -6 * DAY],
    }


@pytest.fixture(scope="module", params=CLOCKS[:3])
def fine_graph(tmp_path_factory, request):
    base = 1_700_000_000 * 10**9
    times = [base + n for n in (100_000_000, 900_000_000, 900_000_000, 1_100_000_000)]
    times += [-1_900_000_000, -1_100_000_000, -900_000_000, 100_000_000]
    root = tmp_path_factory.mktemp("fine_clocks") / request.param
    return make_graph(root, request.param, [9, 2, 10, 20, 29, 22, 21, 30],
                      [1] * 4 + [2] * 4, pa.array(times, pa.timestamp("ns")))


@pytest.mark.parametrize("time_column", ["time", "other_time"])
@pytest.mark.parametrize("shared", [False, True])
def test_subsecond_selection_and_negative_epochs(fine_graph, time_column, shared):
    engine, relation = fine_graph
    plan = plan_for(relation, time_column=time_column, start=-1, limit=1, shared=shared)
    plan["sequences"][0]["columns"].append(time_column)
    result, _ = run(engine, plan, [1, 2], [1_700_000_001, 0])
    sequence = result["sequences"][0]
    assert sequence["batch"].column("id").to_pylist() == [2, 21]
    assert values(sequence["timestamps"]) == [1_700_000_000, -1]
    column = sequence["batch"].column(time_column)
    scale = {"ns": 10**9, "us": 10**6, "ms": 10**3}[column.type.unit]
    assert column.cast(pa.int64()).to_pylist() == [
        1_700_000_000 * scale + 9 * scale // 10, -9 * scale // 10]
    if shared:
        aggregates = {a["name"]: values(a["values"]) for a in result["aggregates"]}
        assert aggregates["n"] == [3, 1]
        assert aggregates["total"] == [6.0, 7.0]
        assert aggregates["age"] == pytest.approx([0.1, 0.9])


@pytest.mark.parametrize("output_format", ["inspectable", "model"])
def test_timestamp_reductions_do_not_collapse_subseconds(fine_graph, output_format):
    engine, relation = fine_graph
    plan = plan_for(relation, start=-1, limit=None)
    plan["aggregates"] = [
        {"name": "distinct", "frame": "f", "operation": "count_distinct", "column": "time"},
        {"name": "mode_count", "frame": "f", "groups": [{"keys": ["time"], "values": [
            {"name": "n", "operation": "count"}]}], "operation": "max", "column": "n"},
        {"name": "grouped_age", "frame": "f", "groups": [{"keys": ["owner"], "values": [
            {"name": "last", "operation": "max", "column": "time"}]}],
            "operation": "recency", "column": "last"},
    ]
    plan["aggregates"] += [
        {"name": operation, "frame": "f", "operation": operation, "column": "time"}
        for operation in ("min", "max", "mode")
    ]
    plan["frames"][0]["columns"].append("owner")
    result, _ = run(engine, plan, [1, 3], [1_700_000_001] * 2, output_format)
    assert {a["name"]: values(a["values"]) for a in result["aggregates"]} == {
        "distinct": [2, 0], "mode_count": [2, None], "grouped_age": [pytest.approx(0.1), None],
        "min": [1_700_000_000, None], "max": [1_700_000_000, None],
        "mode": [1_700_000_000, None]}
    if output_format == "model":
        for aggregate in result["aggregates"][-3:]:
            assert aggregate["values"].type == pa.int64()


@pytest.mark.parametrize("method", ["query_temporal", "query_temporal_lengths"])
def test_subtick_windows_are_rejected(day_graph, method):
    engine, relation, clock = day_graph
    if clock not in ("hours", "days"):
        pytest.skip("a one-second bound is representable in this clock")
    plan = plan_for(relation, start=-1)
    roots = {"users": engine.lookup_node_ids("users", pa.array([1]))}
    with pytest.raises(diskgraph.DiskgraphConfigError, match="shorter than one tick"):
        getattr(engine, method)(roots, {"users": [(EPOCH_DAY + 6) * DAY]}, json.dumps(plan))
