# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Columnar request boundaries preserve the existing graph/temporal contracts."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json

import diskgraph
import numpy as np
import pyarrow as pa
import pytest


class ArrowOnly:
    """Expose buffers but fail if the binding falls back to Python iteration."""

    def __init__(self, array):
        self.array = array

    def __arrow_c_array__(self, requested_schema=None):
        return self.array.__arrow_c_array__(requested_schema)

    def __iter__(self):
        raise AssertionError("input was expanded into Python scalars")


def collect(events):
    batches = defaultdict(list)
    for event in events:
        if event["type"] != "stats":
            assert isinstance(event["batch"], pa.RecordBatch)
            key = (event["type"], event.get("rel_id", event.get("type_id")))
            batches[key].append(event["batch"])
    return {key: pa.Table.from_batches(parts) for key, parts in batches.items()}


def assert_batches_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key].equals(expected[key]), key


def sample_plan():
    relations = [
        "orders_parquet_user_id_to_users_user_id_reverse",
        "orders_parquet_merchant_id_to_merchants_merchant_id",
    ]
    return json.dumps({"layers": [
        {"fanouts": {relation: {
            "fanout": 10, "mode": "deterministic_uniform",
            "absent_key_policy": "uniform_fallback",
        }}, "default_fanout": None}
        for relation in relations
    ]})


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize("dtype", [pa.uint64(), pa.int64()])
def test_array_sampling_matches_lists_with_repeated_anchors(orders_graph, fused, swapped, dtype):
    engine = diskgraph.SamplerEngine(orders_graph, random_seed=41, num_threads=1)
    # Same source entity, different anchor time; one root has no neighbors.
    keys = pa.array(["u1", "u1", "u3", "u2"])
    ids = engine.lookup_node_ids("users", keys)
    times = [1609459200, 1609718400, 1609718400, 1609718400]
    options = dict(fused=fused, swap_edge_direction=swapped,
                   seed_timestamp_unit=diskgraph.TimeUnit.Seconds)
    expected = collect(engine.sample(
        {"users": ids}, sample_plan(), seed_timestamps={"users": times}, **options))
    # Sliced input buffers must retain the logical offset, not read the prefix.
    column = pa.array([None] + ids + [None], type=dtype).slice(1, len(ids))
    cutoffs = pa.array([None] + times + [None], type=pa.int64()).slice(1, len(times))
    actual = collect(engine.sample(
        {"users": ArrowOnly(column)}, sample_plan(),
        seed_timestamps={"users": ArrowOnly(cutoffs)}, **options))
    assert_batches_equal(actual, expected)
    from_keys = collect(engine.sample_from_keys(
        {"users": ArrowOnly(keys)}, sample_plan(),
        seed_timestamps={"users": ArrowOnly(cutoffs)}, **options))
    assert_batches_equal(from_keys, expected)
    # The native Arrow topology can feed NumPy/tensors without Python row objects.
    for (kind, _), table in actual.items():
        if kind == "node":
            owners = table["id_a"].combine_chunks().to_numpy(zero_copy_only=True)
            assert owners.dtype == np.uint64
            assert np.all(owners < len(keys))


@pytest.mark.parametrize("table,keys", [
    ("signed_ids", pa.array([-5, 7, 7], type=pa.int64())),
    ("unsigned_ids", pa.array([2**63 + 1, 7], type=pa.uint64())),
    ("string_ids", pa.array(["0xAB", "  a ", "0xAB"])),
])
def test_lookup_model_output_and_key_sampling(typed_primary_key_graph, table, keys):
    engine = diskgraph.SamplerEngine(typed_primary_key_graph)
    expected = engine.lookup_node_ids(table, keys)
    ids = engine.lookup_node_ids(table, ArrowOnly(keys), output_format="model")
    assert ids.type == pa.uint64()
    assert ids.to_pylist() == expected
    assert engine.canonical_node_mask(table, ids, keys) == [True] * len(keys)
    plan = json.dumps({"layers": []})
    assert_batches_equal(
        collect(engine.sample_from_keys({table: keys}, plan, fused=True)),
        collect(engine.sample({table: expected}, plan, fused=True)),
    )


def test_lookup_missing_and_empty(orders_graph):
    engine = diskgraph.SamplerEngine(orders_graph)
    keys = pa.array(["u1", "missing", "u1"])
    ids = engine.lookup_node_ids("users", keys, output_format="model")
    assert ids.type == pa.uint64()
    assert ids.to_pylist() == engine.lookup_node_ids("users", keys)
    assert ids.null_count == 1
    empty = engine.lookup_node_ids("users", pa.array([], type=pa.string()), output_format="model")
    assert empty.type == pa.uint64() and len(empty) == 0
    with pytest.raises(diskgraph.DiskgraphDataError, match="batch offset 1"):
        engine.sample_from_keys({"users": keys}, sample_plan(), fused=True)
    with pytest.raises(diskgraph.DiskgraphDataError, match="null"):
        engine.sample({"users": ids}, sample_plan(), fused=True)
    with pytest.raises(diskgraph.DiskgraphConfigError, match="output_format"):
        engine.lookup_node_ids("users", keys, output_format="invalid")


@pytest.mark.parametrize("ids,error,match", [
    (pa.array([0, None], type=pa.uint64()), diskgraph.DiskgraphDataError, "null"),
    (pa.array([-1], type=pa.int64()), diskgraph.DiskgraphDataError, "negative"),
    (pa.array([0.0]), TypeError, "uint64 or int64"),
    (pa.array(["0"]), TypeError, "uint64 or int64"),
    (pa.chunked_array([[0], [1]], type=pa.uint64()), TypeError, "combine chunks"),
])
def test_invalid_id_columns_fail_before_sampling(orders_graph, ids, error, match):
    engine = diskgraph.SamplerEngine(orders_graph)
    with pytest.raises(error, match=match):
        engine.sample({"users": ids}, sample_plan(), fused=True)
    with pytest.raises(error, match=match):
        engine.fetch_features([{"node_type": "users", "node_ids": ids, "feature_names": []}])


@pytest.mark.parametrize("times,error,match", [
    (pa.array([None], type=pa.int64()), diskgraph.DiskgraphDataError, "null"),
    (pa.array([1.5]), TypeError, "int64"),
    (pa.array([1], type=pa.timestamp("s")), TypeError, "declared time unit"),
    (pa.array([], type=pa.int64()), diskgraph.DiskgraphConfigError, "equal length"),
    ([], diskgraph.DiskgraphConfigError, "equal length"),
])
def test_invalid_cutoffs_fail_before_sampling(orders_graph, times, error, match):
    engine = diskgraph.SamplerEngine(orders_graph)
    with pytest.raises(error, match=match):
        engine.sample({"users": [0]}, sample_plan(), fused=True,
                      seed_timestamps={"users": times})
    with pytest.raises(error, match=match):
        engine.sample_from_keys({"users": pa.array(["u1"])}, sample_plan(), fused=True,
                                seed_timestamps={"users": times})


def test_unknown_timestamp_type_and_invalid_keys(orders_graph):
    engine = diskgraph.SamplerEngine(orders_graph)
    with pytest.raises(diskgraph.DiskgraphConfigError, match="matching seed type"):
        engine.sample({"users": [0]}, sample_plan(), fused=True,
                      seed_timestamps={"not_users": [0]})
    with pytest.raises(diskgraph.DiskgraphConfigError, match="unknown node type"):
        engine.sample_from_keys({"not_users": pa.array(["u1"])}, sample_plan(), fused=True)
    with pytest.raises(diskgraph.DiskgraphDataError, match="null primary key"):
        engine.sample_from_keys({"users": pa.array([None], type=pa.string())}, sample_plan(), fused=True)


def test_array_feature_fetch_preserves_projection_duplicates_and_lifetime(orders_graph):
    engine = diskgraph.SamplerEngine(orders_graph)
    ids = engine.lookup_node_ids("users", pa.array(["u2", "u1", "u2"]))
    expected = collect(engine.fetch_features([
        {"node_type": "users", "node_ids": ids, "feature_names": ["age"]}]))
    iterator = engine.fetch_features([
        {"node_type": "users", "node_ids": ArrowOnly(pa.array(ids, type=pa.uint64())),
         "feature_names": ["age"]}])
    # The async request owns its imported data even after the Python engine is gone.
    del engine
    actual = collect(iterator)
    assert_batches_equal(actual, expected)
    (table,) = actual.values()
    assert table["age"].to_pylist() == [30, 25, 30]


def temporal_plan():
    return json.dumps({
        "frames": [{"name": "orders", "root_type": "users", "path": [{
            "relation": "orders_parquet_user_id_to_users_user_id_reverse",
            "destination_type": "orders_parquet",
        }], "time_column": "timestamp", "stop_seconds": 0, "distinct": True,
            "columns": ["amount", "order_id"]}],
        "sequences": [{"name": "recent", "frame": "orders", "columns": ["amount"],
                       "order": "descending", "limit": 2}],
        "aggregates": [{"name": "total", "frame": "orders", "operation": "sum", "column": "amount"}],
    })


@pytest.mark.parametrize("empty", [False, True])
def test_temporal_arrays_match_lists_and_lengths(sorted_orders_graph, empty):
    engine = diskgraph.SamplerEngine(sorted_orders_graph)
    keys = pa.array([] if empty else ["u1", "u1", "u2", "u3"], type=pa.string())
    ids = engine.lookup_node_ids("users", keys, output_format="model")
    times = pa.array([] if empty else [1609459200, 1609718400, 1609718400, 1609718400], type=pa.int64())
    expected = engine.query_temporal({"users": ids.to_pylist()}, {"users": times.to_pylist()},
                                     temporal_plan(), output_format="model")
    actual = engine.query_temporal({"users": ArrowOnly(ids)}, {"users": ArrowOnly(times)},
                                   temporal_plan(), output_format="model")
    for got, want in zip(actual["sequences"], expected["sequences"]):
        for key in ("batch", "offsets", "node_ids", "timestamps"):
            assert got[key].equals(want[key])
    for got, want in zip(actual["aggregates"], expected["aggregates"]):
        assert got["values"].equals(want["values"])
    lengths = engine.query_temporal_lengths({"users": ids}, {"users": times}, temporal_plan())
    expected_lengths = np.diff(actual["sequences"][0]["offsets"].to_numpy())
    np.testing.assert_array_equal(lengths["sequences"][0]["lengths"].to_numpy(), expected_lengths)


def test_empty_sampling_and_concurrent_calls(orders_graph):
    engine = diskgraph.SamplerEngine(orders_graph, random_seed=41, num_threads=1)
    empty = collect(engine.sample_from_keys({"users": pa.array([], type=pa.string())},
                                           sample_plan(), fused=True))
    assert not empty
    keys = pa.array(["u2", "u1", "u1"])
    expected = collect(engine.sample_from_keys({"users": keys}, sample_plan(), fused=True))
    def run(_):
        return collect(engine.sample_from_keys({"users": ArrowOnly(keys)}, sample_plan(), fused=True))
    with ThreadPoolExecutor(max_workers=3) as pool:
        for actual in pool.map(run, range(6)):
            assert_batches_equal(actual, expected)
