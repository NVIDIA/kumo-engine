import pytest
import json
import diskgraph_pybind as diskgraph_py
import pyarrow as pa
import threading
import time

def test_load(ifood_graph):
    dg = diskgraph_py.SamplerEngine(ifood_graph)
    assert dg is not None

def test_schema_mapping(ifood_graph):
    dg = diskgraph_py.SamplerEngine(ifood_graph)
    mapping_json = dg.get_type_mapping()
    mapping = json.loads(mapping_json)
    assert "node_types" in mapping
    assert "users" in mapping["node_types"]
    assert "orders_parquet" in mapping["node_types"]

def test_fused_sampling_1hop(ifood_graph):
    dg = diskgraph_py.SamplerEngine(ifood_graph)

    # User 0 -> Orders
    plan = {
        "layers": [
            {
                "fanouts": {
                    "orders_parquet_user_id_and_centroid_unique_numerical_to_users_user_id_and_centroid_unique_numerical_reverse": {
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

def test_standalone_sampling(ifood_graph):
    dg = diskgraph_py.SamplerEngine(ifood_graph)

    # User 0 -> Orders (Topology Only)
    plan = {
        "layers": [
            {
                "fanouts": {
                    "orders_parquet_user_id_and_centroid_unique_numerical_to_users_user_id_and_centroid_unique_numerical_reverse": {
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

def test_fetch_features(ifood_graph):
    dg = diskgraph_py.SamplerEngine(ifood_graph)

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
    with pytest.raises(RuntimeError, match="Unknown node type"):
        for _ in iter_bad: pass

def test_fused_sampling_2hop_consistency(ifood_graph):
    dg = diskgraph_py.SamplerEngine(ifood_graph)

    # User -> Orders -> Merchants
    plan = {
        "layers": [
            {
                "fanouts": {
                    "orders_parquet_user_id_and_centroid_unique_numerical_to_users_user_id_and_centroid_unique_numerical_reverse": {
                        "fanout": 5, "mode": "uniform", "absent_key_policy": "uniform_fallback"
                    }
                },
                "default_fanout": None
            },
            {
                "fanouts": {
                    "orders_parquet_merchant_id_numerical_to_merchants_merchant_id_numerical": {
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

def test_concurrency_gil(ifood_graph):
    dg = diskgraph_py.SamplerEngine(ifood_graph)

    def worker():
        # Simple 1-hop
        plan = { "layers": [{ "fanouts": { "orders_parquet_user_id_and_centroid_unique_numerical_to_users_user_id_and_centroid_unique_numerical_reverse": { "fanout": 5, "mode": "uniform", "absent_key_policy": "uniform_fallback" } }, "default_fanout": None }] }

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

def test_invalid_plan(ifood_graph):
    dg = diskgraph_py.SamplerEngine(ifood_graph)
    with pytest.raises(ValueError):
        dg.sample({"users": [0]}, "{invalid_json", fused=True)
