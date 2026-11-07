# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coarse edge clocks are conservative filters; row clocks decide membership."""

import copy
import json

import diskgraph
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def build_graph(root, *, unit="minute", bits=16, anchor=0, null=False, weighted=False,
                adversarial=False, index_clock="seconds", times_override=None):
    step = {"minute": 60, "hour": 3600, "day": 86400}[unit]
    # Interior bucket boundaries and timestamps beyond both encoded endpoints.
    floor, ceiling = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    offsets = [bucket * step + delta for bucket in range(-8, 9) for delta in (-1, 0, 1)]
    offsets += [(floor - 2) * step, (ceiling + 2) * step]
    times = [anchor + offset for offset in offsets]
    if times_override is not None:
        times = list(times_override)
    if adversarial:
        times += [anchor + delta for delta in [1, 2, step // 2, step // 2, step - 1, step]]
    if null:
        times.append(None)
    n = len(times)
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"id": [1, 2]}), root / "users.parquet")
    ids = np.random.default_rng(158).permutation(n).tolist() if adversarial else list(range(n))
    events = pa.table({
        "id": ids, "owner": [1] * n,
        "time": pa.array(times, type=pa.timestamp("s")),
        "reference_time": pa.array(times, type=pa.timestamp("s")),
        "amount": [float(i) for i in range(n)],
    })
    if adversarial:
        events = events.take(pa.array(np.random.default_rng(82).permutation(n)))
    pq.write_table(events, root / "events.parquet")
    config = {
        "dataset_name": "coarse_filter", "num_shards": 1, "timestamp_unit": index_clock,
        "tables": [
            {"name": "users", "primary_key": "id",
             "columns": [{"name": "id", "ingested_dtype": "int64"}],
             "source": {"type": "parquet", "paths": [str(root / "users.parquet")]}},
            {"name": "events", "primary_key": "id",
             "columns": [{"name": name, "ingested_dtype": dtype} for name, dtype in
                         [("id", "int64"), ("owner", "int64"), ("time", "timestamp"),
                          ("reference_time", "timestamp"), ("amount", "float64")]],
             "source": {"type": "parquet", "paths": [str(root / "events.parquet")]},
             "fkeys": [{"src_column": "owner", "dst_table": "users", "reverse": {
                 "time": {"table": "events", "column": "time", "unit": unit,
                          "bitwidth": bits, "anchor_time": anchor}}}]},
        ],
    }
    if weighted:
        config["tables"][1]["fkeys"][0]["reverse"]["weight"] = {
            "table": "events", "expr": "amount", "dtype": "float32"}
    diskgraph.ingest(json.dumps(config), str(root / "index"), col_stats=False,
                     context_threads=2, encode_threads=2, pk_map_threads=2,
                     edges_map_threads=2, index_threads=2)
    engine = diskgraph.SamplerEngine(str(root / "index"))
    relations = json.loads(engine.get_type_mapping())["edge_relations"]
    reverse = next(r["name"] for r in relations if r["src_type"] == "users")
    forward = next(r["name"] for r in relations if r["src_type"] == "events")
    return engine, reverse, forward, times, step


@pytest.fixture(scope="module", params=[("minute", 32, 1_700_000_003, False),
                                      ("hour", 16, 0, False), ("day", 8, -17, False),
                                      ("minute", 16, 0, True)])
def coarse_graph(tmp_path_factory, request):
    unit, bits, anchor, weighted = request.param
    return build_graph(tmp_path_factory.mktemp("coarse"), unit=unit, bits=bits,
                       anchor=anchor, weighted=weighted), anchor


def make_plan(reverse, *, lower, upper, limit=None, clock="time"):
    return {
        "frames": [{"name": "f", "root_type": "users",
                    "path": [{"relation": reverse, "destination_type": "events"}],
                    "time_column": clock, "start_seconds": lower, "stop_seconds": upper,
                    "distinct": True, "columns": ["id", "amount"]}],
        "sequences": [{"name": "s", "frame": "f", "columns": ["id", "amount"],
                       "order": "descending", "limit": limit}],
        "aggregates": [],
    }


def run(engine, roots, cutoffs, plan):
    return engine.query_temporal({"users": roots}, {"users": cutoffs}, json.dumps(plan))


def assert_same(actual, expected):
    assert actual["aggregates"] == expected["aggregates"]
    for a, b in zip(actual["sequences"], expected["sequences"], strict=True):
        assert a.keys() == b.keys()
        for key in a:
            if key == "batch":
                assert a[key].equals(b[key])
            else:
                assert a[key] == b[key]


@pytest.mark.parametrize("shape", ["sequence", "shared", "multiple", "multihop", "duplicates"])
def test_coarse_matches_full_row_filter(coarse_graph, shape):
    (engine, reverse, forward, times, step), anchor = coarse_graph
    roots = engine.lookup_node_ids("users", pa.array([1, 2, 1]))
    cutoffs = [anchor, anchor, anchor + step // 2]
    windows = [(None, 0), (0, None), (None, None), (-1, 0), (0, 1),
               (-step, 0), (-step + 1, 1), (-2 * step, 2 * step)]
    rng = np.random.default_rng(85)
    windows += [(int(lo), int(lo + width)) for lo, width in
                zip(rng.integers(-8 * step, 8 * step, 10), rng.integers(1, 3 * step, 10))]
    # Exercise actual clipped endpoint rows, not just interior buckets.
    windows += [(time - anchor - 1, time - anchor) for time in times[-2:]]
    for lower, upper in windows:
        for limit in (None, 1, 3):
            plan = make_plan(reverse, lower=lower, upper=upper, limit=limit)
            if shape == "shared":
                plan["aggregates"] = [{"name": "sum", "frame": "f", "operation": "sum", "column": "amount"}]
            if shape == "multiple":
                plan["sequences"].append(dict(plan["sequences"][0], name="ascending", order="ascending"))
            if shape in ("multihop", "duplicates"):
                plan["frames"][0]["path"] += [
                    {"relation": forward, "destination_type": "users"},
                    {"relation": reverse, "destination_type": "events"}]
                plan["frames"][0]["distinct"] = shape != "duplicates"
            reference = copy.deepcopy(plan)
            reference["frames"][0]["time_column"] = "reference_time"
            actual, expected = run(engine, roots, cutoffs, plan), run(engine, roots, cutoffs, reference)
            assert_same(actual, expected)
            if shape == "sequence":
                selected = [i for i, time in enumerate(times)
                            if (lower is None or time > anchor + lower)
                            and (upper is None or time <= anchor + upper)]
                selected.sort(key=lambda i: times[i], reverse=True)
                selected = selected if limit is None else selected[:limit]
                sequence = actual["sequences"][0]
                assert sequence["batch"].column("id").to_pylist()[:sequence["offsets"][1]] == selected
            lengths = engine.query_temporal_lengths({"users": roots}, {"users": cutoffs}, json.dumps(plan))
            for length, sequence in zip(lengths["sequences"], actual["sequences"], strict=True):
                assert length["lengths"].to_pylist() == np.diff(sequence["offsets"]).tolist()


def test_coarse_filter_reduces_decode_work(coarse_graph):
    (engine, reverse, _, _, step), anchor = coarse_graph
    roots = engine.lookup_node_ids("users", pa.array([1]))
    plan = make_plan(reverse, lower=-step, upper=0)
    reference = copy.deepcopy(plan)
    reference["frames"][0]["time_column"] = "reference_time"
    actual, expected = run(engine, roots, [anchor], plan), run(engine, roots, [anchor], reference)
    assert_same(actual, expected)
    assert actual["stats"]["edges_visited"] < expected["stats"]["edges_visited"]
    assert actual["stats"]["rows_decoded"] < expected["stats"]["rows_decoded"] / 2
    lengths = [engine.query_temporal_lengths({"users": roots}, {"users": [anchor]}, json.dumps(p))
               for p in (plan, reference)]
    assert lengths[0]["stats"]["rowpack_timestamps"] < lengths[1]["stats"]["rowpack_timestamps"] / 2
    assert lengths[0]["stats"]["edges_visited"] < lengths[1]["stats"]["edges_visited"]


@pytest.mark.parametrize("degree", [256, 131072])
def test_indexed_seek_skips_newer_adjacency_without_history_cap(tmp_path, degree):
    anchor = 1_700_000_003
    engine, reverse, _, _, _ = build_graph(
        tmp_path, unit="minute", bits=32, anchor=anchor,
        times_override=list(range(anchor + 1, anchor + degree + 1)))
    roots = engine.lookup_node_ids("users", pa.array([1, 2]))
    cutoff = anchor + degree // 100
    plan = make_plan(reverse, lower=None, upper=0, limit=None)
    plan["aggregates"] = [{"name": "all_amount", "frame": "f",
                           "operation": "sum", "column": "amount"}]
    reference = copy.deepcopy(plan)
    reference["frames"][0]["time_column"] = "reference_time"
    actual = run(engine, roots, [cutoff, cutoff], plan)
    expected = run(engine, roots, [cutoff, cutoff], reference)
    assert_same(actual, expected)
    assert actual["sequences"][0]["batch"].column("id").to_pylist() == list(
        reversed(range(degree // 100)))
    # Include the partial minute bucket; no capped-history early stop is used.
    assert actual["stats"]["edges_visited"] <= degree // 100 + 60
    assert expected["stats"]["edges_visited"] == degree
    lengths = [engine.query_temporal_lengths({"users": roots}, {"users": [cutoff, cutoff]},
                                            json.dumps(p)) for p in (plan, reference)]
    for result in lengths:
        assert result["sequences"][0]["lengths"].to_pylist() == [degree // 100, 0]
    assert lengths[0]["stats"]["edges_visited"] <= degree // 100 + 60
    assert lengths[1]["stats"]["edges_visited"] == degree


def test_coarse_null_clock_still_raises(tmp_path):
    engine, reverse, _, _, _ = build_graph(tmp_path, null=True)
    roots = engine.lookup_node_ids("users", pa.array([1]))
    for clock in ("time", "reference_time"):
        plan = make_plan(reverse, lower=100_000, upper=100_001, clock=clock)
        with pytest.raises(diskgraph.DiskgraphConfigError, match="null timestamp"):
            run(engine, roots, [0], plan)
        with pytest.raises(diskgraph.DiskgraphConfigError, match="null timestamp"):
            engine.query_temporal_lengths({"users": roots}, {"users": [0]}, json.dumps(plan))


@pytest.mark.parametrize("unit,bits,weighted", [("minute", 8, False), ("hour", 16, True),
                                              ("day", 32, False)])
def test_coarse_refinement_ties_and_complete_aggregates(tmp_path, unit, bits, weighted):
    anchor = -1_700_000_003
    engine, reverse, forward, _, step = build_graph(
        tmp_path, unit=unit, bits=bits, anchor=anchor, weighted=weighted, adversarial=True)
    table = pq.read_table(tmp_path / "events.parquet", columns=["id", "owner", "time", "amount"])
    table = table.set_column(table.schema.get_field_index("time"), "time",
                             table["time"].cast(pa.timestamp("s")).cast(pa.int64()))
    source = table.to_pylist()
    root_keys = [1, 2, 1, 1]
    cutoffs = [anchor, anchor, anchor + step // 2, anchor + step - 1]
    roots = engine.lookup_node_ids("users", pa.array(root_keys))
    windows = [(None, 0), (-step, 0), (-1, 0), (0, 1), (-step, step), (None, None)]
    windows += [(row["time"] - anchor - 1, row["time"] - anchor)
                for row in source if abs(row["time"] - anchor) > 100 * step]
    for duplicates in (False, True):
        for lower, upper in windows:
            for order in ("ascending", "descending"):
                plan = make_plan(reverse, lower=lower, upper=upper, limit=2)
                plan["sequences"][0]["order"] = order
                plan["aggregates"] = [{"name": "all_amount", "frame": "f",
                                       "operation": "sum", "column": "amount"}]
                if duplicates:
                    plan["frames"][0]["path"] += [
                        {"relation": forward, "destination_type": "users"},
                        {"relation": reverse, "destination_type": "events"}]
                    plan["frames"][0]["distinct"] = False
                actual = run(engine, roots, cutoffs, plan)
                reference = copy.deepcopy(plan)
                reference["frames"][0]["time_column"] = "reference_time"
                assert_same(actual, run(engine, roots, cutoffs, reference))
                seq = actual["sequences"][0]
                returned = seq["batch"].column("id").to_pylist()
                expected_lengths = []
                for i, (owner, cutoff) in enumerate(zip(root_keys, cutoffs, strict=True)):
                    eligible = [row for row in source if row["owner"] == owner
                                and (lower is None or row["time"] > cutoff + lower)
                                and (upper is None or row["time"] <= cutoff + upper)]
                    if duplicates:
                        eligible *= sum(row["owner"] == owner for row in source)
                    # Public RDL tie order uses (type name, repr(primary key)).
                    eligible.sort(key=lambda row: (row["time"], repr(row["id"])),
                                  reverse=order == "descending")
                    selected = [row["id"] for row in eligible[:2]]
                    assert returned[seq["offsets"][i]:seq["offsets"][i + 1]] == selected
                    expected_lengths.append(len(selected))
                    expected_sum = sum(row["amount"] for row in eligible)
                    assert actual["aggregates"][0]["values"][i] == expected_sum
                lengths = engine.query_temporal_lengths({"users": roots}, {"users": cutoffs},
                                                        json.dumps(plan))
                assert lengths["sequences"][0]["lengths"].to_pylist() == expected_lengths


@pytest.mark.parametrize("index_clock", ["seconds", "milliseconds", "microseconds",
                                         "nanoseconds", "hours", "days"])
def test_index_clock_units_preserve_seconds_query_bounds(tmp_path, index_clock):
    # Whole-day source times are exactly representable in every index clock.
    day = 86400
    times = [day * value for value in [-3, -2, -1, 0, 1, 1, 2, 3]]
    engine, reverse, _, _, _ = build_graph(
        tmp_path, unit="hour", bits=8, index_clock=index_clock, times_override=times)
    roots = engine.lookup_node_ids("users", pa.array([1, 2, 1]))
    cutoffs = [0, 0, day + 1379]
    for lower, upper in [(None, 0), (-day, 0), (0, day), (-2 * day, 2 * day)]:
        for order in ("ascending", "descending"):
            plan = make_plan(reverse, lower=lower, upper=upper, limit=2)
            plan["sequences"][0]["order"] = order
            actual = run(engine, roots, cutoffs, plan)
            reference = copy.deepcopy(plan)
            reference["frames"][0]["time_column"] = "reference_time"
            assert_same(actual, run(engine, roots, cutoffs, reference))
            expected_lengths = []
            seq = actual["sequences"][0]
            for anchor, cutoff in enumerate(cutoffs):
                eligible = [] if anchor == 1 else [
                    i for i, time in enumerate(times)
                    if (lower is None or time > cutoff + lower)
                    and (upper is None or time <= cutoff + upper)]
                eligible.sort(key=lambda i: (times[i], repr(i)), reverse=order == "descending")
                selected = eligible[:2]
                assert seq["batch"].column("id").to_pylist()[
                    seq["offsets"][anchor]:seq["offsets"][anchor + 1]] == selected
                expected_lengths.append(len(selected))
            lengths = engine.query_temporal_lengths({"users": roots}, {"users": cutoffs},
                                                    json.dumps(plan))
            assert lengths["sequences"][0]["lengths"].to_pylist() == expected_lengths
