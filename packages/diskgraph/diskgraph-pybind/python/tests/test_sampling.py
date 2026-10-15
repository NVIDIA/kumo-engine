# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from diskgraph._c import DiskgraphConfigError, DiskgraphDataError, DiskgraphSystemError

import json
import subprocess
from pathlib import Path
import diskgraph as diskgraph_py
from diskgraph import DiskgraphConfigError
import pyarrow as pa
import threading
import time
from decimal import Decimal

def test_load(orders_graph):
    dg = diskgraph_py.SamplerEngine(orders_graph)
    assert dg is not None


def test_benchmark_uses_selected_graph(orders_graph, tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"layers": [{"fanouts": {}, "default_fanout": {
        "fanout": 3, "mode": "uniform", "absent_key_policy": "uniform_fallback",
    }}]}))
    result = subprocess.run(
        ["cargo", "run", "--release", "--bin", "bench_sampler", "--",
         "--index", orders_graph, "--seed-type", "orders_parquet", "--plan", str(plan)],
        cwd=Path(__file__).resolve().parents[3], check=True, capture_output=True, text=True,
    )
    edges = next(line for line in result.stdout.splitlines() if line.startswith("Total Edges:"))
    assert int(edges.split(":")[1]) > 0


def test_canonical_node_mask_uses_typed_primary_keys(orders_graph,
                                                     list_features_graph,
                                                     typed_primary_key_graph):
    string_engine = diskgraph_py.SamplerEngine(orders_graph)
    assert string_engine.canonical_node_mask(
        "users", [0, 1, 0], pa.array(["u1", "u2", "u2"])
    ) == [True, True, False]

    integer_engine = diskgraph_py.SamplerEngine(list_features_graph)
    assert integer_engine.canonical_node_mask(
        "items", [0, 1, 1], pa.array([0, 1, 2], type=pa.int64())
    ) == [True, True, False]

    typed_engine = diskgraph_py.SamplerEngine(typed_primary_key_graph)
    assert typed_engine.canonical_node_mask(
        "signed_ids", [0, 1, 2], pa.array([-5, 7, 7], type=pa.int64())
    ) == [True, True, False]
    assert typed_engine.canonical_node_mask(
        "unsigned_ids", [0, 1, 0],
        pa.array([7, 2**63 + 1, 2**63 + 1], type=pa.uint64()),
    ) == [True, True, False]
    assert typed_engine.canonical_node_mask(
        "string_ids", [0, 1, 0], pa.array(["  a ", "0xAB", "0xAB"])
    ) == [True, True, False]
    assert typed_engine.canonical_node_mask(
        "bytes_ids", [0, 1, 0],
        pa.array([b"\x00\xff", b"raw", b"raw"], type=pa.binary()),
    ) == [True, True, False]

    uuid_one = bytes.fromhex("00" * 15 + "01")
    assert typed_engine.canonical_node_mask(
        "uuid_ids", [0], pa.array([uuid_one], type=pa.binary()),
    ) == [True]
    assert typed_engine.canonical_node_mask(
        "uuid_ids", [1],
        pa.array(["00000000-0000-0000-0000-000000000002"]),
    ) == [True]
    assert typed_engine.canonical_node_mask(
        "uuid256_ids", [0], pa.array([bytes([1]) * 32], type=pa.binary())
    ) == [True]
    assert typed_engine.canonical_node_mask(
        "uuid256_ids", [1], pa.array(["02" * 32])
    ) == [True]
    ipv4_mapped = b"\x00" * 10 + b"\xff\xff\xc0\x00\x02\x01"
    assert typed_engine.canonical_node_mask(
        "ip_ids", [0], pa.array([ipv4_mapped], type=pa.binary())
    ) == [True]
    assert typed_engine.canonical_node_mask(
        "ip_ids", [1], pa.array(["2001:db8::1"])
    ) == [True]
    assert typed_engine.canonical_node_mask(
        "decimal128_ids", [0, 1, 2, 0],
        pa.array([
            Decimal("123.0000"), Decimal("7.0000"),
            Decimal("7.0000"), Decimal("999.0000"),
        ], type=pa.decimal128(38, 4)),
    ) == [True, True, False, False]
    assert typed_engine.canonical_node_mask(
        "decimal256_ids", [0, 1, 2, 0],
        pa.array([
            Decimal("123.0000"), Decimal("7.0000"),
            Decimal("7.0000"), Decimal("999.0000"),
        ], type=pa.decimal256(76, 4)),
    ) == [True, True, False, False]

    with pytest.raises(TypeError, match="incompatible with Arrow type"):
        integer_engine.canonical_node_mask("items", [0], pa.array(["0"]))
    with pytest.raises(DiskgraphDataError, match="exactly 16 bytes"):
        typed_engine.canonical_node_mask(
            "uuid_ids", [0], pa.array([b"short"], type=pa.binary())
        )


def test_canonical_node_mask_two_50k_typed_batches(typed_primary_key_graph):
    engine = diskgraph_py.SamplerEngine(typed_primary_key_graph)
    cases = [
        ("signed_ids", [-5, 7, 7, 999], [0, 1, 2, 0], pa.int64(), 8_334),
        (
            "unsigned_ids",
            [7, 2**63 + 1, 99, 7],
            [0, 1, 0, 1],
            pa.uint64(),
            8_334,
        ),
        (
            "string_ids",
            ["  a ", "0xAB", "missing", "  a "],
            [0, 1, 0, 1],
            pa.string(),
            8_333,
        ),
        (
            "bytes_ids",
            [b"\x00\xff", b"raw", b"missing", b"\x00\xff"],
            [0, 1, 0, 1],
            pa.binary(),
            8_333,
        ),
        (
            "decimal128_ids",
            [
                Decimal("123.0000"),
                Decimal("7.0000"),
                Decimal("7.0000"),
                Decimal("999.0000"),
            ],
            [0, 1, 2, 0],
            pa.decimal128(38, 4),
            8_333,
        ),
        (
            "decimal256_ids",
            [
                Decimal("123.0000"),
                Decimal("7.0000"),
                Decimal("7.0000"),
                Decimal("999.0000"),
            ],
            [0, 1, 2, 0],
            pa.decimal256(76, 4),
            8_333,
        ),
    ]
    expected_pattern = [True, True, False, False]
    assert sum(case[-1] for case in cases) == 50_000

    for batch_index in range(2):
        for node_type, key_pattern, node_pattern, data_type, count in cases:
            order = [(index + batch_index) % 4 for index in range(count)]
            expected = [expected_pattern[index] for index in order]
            keys = pa.array([key_pattern[index] for index in order], type=data_type)
            node_ids = [node_pattern[index] for index in order]
            actual = engine.canonical_node_mask(node_type, node_ids, keys)
            assert len(actual) == count
            assert all(got == want for got, want in zip(actual, expected)), (
                f"{node_type} batch {batch_index} returned an incorrect canonical mask"
            )


def test_decimal256_pk_table_does_not_block_sampler_initialization(
        typed_primary_key_graph):
    engine = diskgraph_py.SamplerEngine(typed_primary_key_graph)
    assert engine.canonical_node_mask(
        "signed_ids", [0], pa.array([-5], type=pa.int64())
    ) == [True]

def test_schema_mapping(orders_graph):
    dg = diskgraph_py.SamplerEngine(orders_graph)
    mapping_json = dg.get_type_mapping()
    mapping = json.loads(mapping_json)
    assert "node_types" in mapping
    assert "users" in mapping["node_types"]
    assert "orders_parquet" in mapping["node_types"]

def test_fused_sampling_1hop(orders_graph):
    dg = diskgraph_py.SamplerEngine(orders_graph)

    # User 0 -> Orders
    plan = {
        "layers": [
            {
                "fanouts": {
                    "orders_parquet_user_id_to_users_user_id_reverse": {
                        "fanout": 5, "mode": "uniform", "absent_key_policy": "uniform_fallback"
                    }
                },
                "default_fanout": None
            }
        ]
    }

    iterator = dg.sample({"users": [0]}, json.dumps(plan), fused=True)

    events = []
    for event in iterator:
        events.append(event)

    # Validation
    types = [e["type"] for e in events]
    assert "stats" in types

    # Check Stats
    stats_event = next(e for e in events if e["type"] == "stats")
    stats = json.loads(stats_event["stats"])
    assert stats["total_nodes"] > 0
    assert stats["total_edges"] > 0

    # Check Features
    feature_events = [e for e in events if e["type"] == "feature"]
    assert len(feature_events) > 0

    # Check Topology
    edge_events = [e for e in events if e["type"] == "edge"]
    assert len(edge_events) > 0

    # Check Schema (PyArrow Batch)
    first_feat = feature_events[0]["batch"]
    assert isinstance(first_feat, pa.RecordBatch)
    assert first_feat.num_rows > 0

def test_standalone_sampling(orders_graph):
    dg = diskgraph_py.SamplerEngine(orders_graph)

    # User 0 -> Orders (Topology Only)
    plan = {
        "layers": [
            {
                "fanouts": {
                    "orders_parquet_user_id_to_users_user_id_reverse": {
                        "fanout": 5, "mode": "uniform", "absent_key_policy": "uniform_fallback"
                    }
                },
                "default_fanout": None
            }
        ]
    }

    iterator = dg.sample({"users": [0]}, json.dumps(plan), fused=False)

    events = []
    for event in iterator:
        events.append(event)

    # Verify NO features returned
    feature_events = [e for e in events if e["type"] == "feature"]
    assert len(feature_events) == 0, "Standalone mode should not return features"

    # Verify Topology
    edge_events = [e for e in events if e["type"] == "edge"]
    assert len(edge_events) > 0

    # Verify Node Mapping
    node_events = [e for e in events if e["type"] == "node"]
    assert len(node_events) > 0

def test_fetch_features(orders_graph):
    dg = diskgraph_py.SamplerEngine(orders_graph)

    # Fetch features for User 0 and Order 0 (if exists)
    # We construct a batch request
    req = [
        {
            "node_type": "users",
            "node_ids": [0, 1],
            "feature_names": [] # All features
        }
    ]

    iterator = dg.fetch_features(req)

    events = []
    for event in iterator:
        events.append(event)

    # Validation
    feature_events = [e for e in events if e["type"] == "feature"]
    assert len(feature_events) > 0

    # Check Batch Content
    batch = feature_events[0]["batch"]
    assert isinstance(batch, pa.RecordBatch)
    assert batch.num_rows > 0

    # Check Error Handling (Bad Type)
    bad_req = [{"node_type": "invalid_type", "node_ids": [0], "feature_names": []}]
    iter_bad = dg.fetch_features(bad_req)
    with pytest.raises(DiskgraphConfigError, match="Unknown node type"):
        for _ in iter_bad: pass

def test_fused_sampling_2hop_consistency(orders_graph):
    dg = diskgraph_py.SamplerEngine(orders_graph)

    # User -> Orders -> Merchants
    plan = {
        "layers": [
            {
                "fanouts": {
                    "orders_parquet_user_id_to_users_user_id_reverse": {
                        "fanout": 5, "mode": "uniform", "absent_key_policy": "uniform_fallback"
                    }
                },
                "default_fanout": None
            },
            {
                "fanouts": {
                    "orders_parquet_merchant_id_to_merchants_merchant_id": {
                        "fanout": 5, "mode": "uniform", "absent_key_policy": "uniform_fallback"
                    }
                },
                "default_fanout": None
            }
        ]
    }

    iterator = dg.sample({"users": [0]}, json.dumps(plan), fused=True)

    # Track counts per type
    node_counts = {} # type_id -> total_rows
    feature_counts = {} # type_id -> total_rows

    for event in iterator:
        if event["type"] == "node":
            tid = event["type_id"]
            node_counts[tid] = node_counts.get(tid, 0) + event["batch"].num_rows
        elif event["type"] == "feature":
            tid = event["type_id"]
            feature_counts[tid] = feature_counts.get(tid, 0) + event["batch"].num_rows

    # Verify strict 1:1 mapping
    # Every node in the topology MUST have a corresponding feature row
    for tid, count in node_counts.items():
        assert tid in feature_counts, f"Missing features for type {tid}"
        assert feature_counts[tid] == count, f"Mismatch for type {tid}: {count} nodes vs {feature_counts[tid]} features"

def test_concurrency_gil(orders_graph):
    dg = diskgraph_py.SamplerEngine(orders_graph)

    def worker():
        # Simple 1-hop
        plan = { "layers": [{ "fanouts": { "orders_parquet_user_id_to_users_user_id_reverse": { "fanout": 5, "mode": "uniform", "absent_key_policy": "uniform_fallback" } }, "default_fanout": None }] }

        # Run multiple queries per thread
        for _ in range(5):
            iterator = dg.sample({"users": [0]}, json.dumps(plan), fused=True)
            count = 0
            for _ in iterator:
                count += 1
            assert count > 0

    threads = []
    start = time.time()
    for _ in range(4):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    end = time.time()

    # If GIL was held, this would be serial and slow.
    # 4 threads * 5 reqs = 20 reqs.
    # Should be fast.
    print(f"Concurrency test took {end - start:.2f}s")

def test_invalid_plan(orders_graph):
    dg = diskgraph_py.SamplerEngine(orders_graph)
    with pytest.raises(DiskgraphConfigError):
        dg.sample({"users": [0]}, "{invalid_json", fused=True)

def test_temporal_sampling(orders_graph):
    dg = diskgraph_py.SamplerEngine(orders_graph)

    # Simple 1-hop plan
    plan = {
        "layers": [
            {
                "fanouts": {
                    "orders_parquet_user_id_to_users_user_id_reverse": {
                        "fanout": 5, "mode": "uniform", "absent_key_policy": "uniform_fallback"
                    }
                },
                "default_fanout": None
            }
        ]
    }

    # Pass seed_timestamps (epoch microseconds)
    timestamps = {"users": [1609459200000000]} # Jan 1 2021

    iterator = dg.sample(
        {"users": [0]},
        json.dumps(plan),
        fused=True,
        seed_timestamps=timestamps
    )

    events = []
    for event in iterator:
        events.append(event)

    # Basic verification that it runs without error
    types = [e["type"] for e in events]
    assert "stats" in types
    stats_event = next(e for e in events if e["type"] == "stats")
    stats = json.loads(stats_event["stats"])
    assert "total_nodes" in stats
