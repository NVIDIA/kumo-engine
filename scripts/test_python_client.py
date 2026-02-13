#!/usr/bin/env python3
"""Smoke test for the DiskgraphNeighborSampler Python client.

Usage:
    # Start the gateway first (in another terminal):
    #   cargo run --release -p diskgraph-gateway -- \
    #       --data ~/diskgraph-data/diskgraph-index/social_sampling_50m --port 8815

    python scripts/test_python_client.py \
        --server grpc://localhost:8815 \
        --seed-type users \
        --seed-ids 0,1,2 \
        --hops 2 \
        --fanout 10
"""
import argparse
import json
import sys

import pyarrow.flight as flight


def test_schema_mapping(client: flight.FlightClient):
    """Test the get_type_name_mapping action."""
    print("=" * 60)
    print("TEST: get_type_name_mapping action")
    print("=" * 60)

    action = flight.Action("get_type_name_mapping", b"")
    results = list(client.do_action(action))
    mapping = json.loads(results[0].body.to_pybytes())

    print(f"  node_types: {mapping['node_types']}")
    for i, rel in enumerate(mapping["edge_relations"]):
        print(f"  edge_relation[{i}]: {rel['src_type']} -> {rel['name']} -> {rel['dst_type']}")

    assert len(mapping["node_types"]) > 0, "No node types"
    assert len(mapping["edge_relations"]) > 0, "No edge relations"
    for rel in mapping["edge_relations"]:
        if rel["name"]:
            assert "src_type" in rel, f"Missing src_type in {rel}"
            assert "dst_type" in rel, f"Missing dst_type in {rel}"
    print("  PASSED\n")
    return mapping


def test_sampling(
    client: flight.FlightClient,
    seed_type: str,
    seed_ids: list,
    num_hops: int,
    fanout: int,
):
    """Test a sampling request end-to-end."""
    print("=" * 60)
    print(f"TEST: sampling {len(seed_ids)} seeds of type '{seed_type}', "
          f"{num_hops} hops, fanout={fanout}")
    print("=" * 60)

    layers = []
    for _ in range(num_hops):
        layers.append({
            "fanouts": {},
            "default_fanout": {
                "fanout": fanout,
                "mode": "uniform",
                "absent_key_policy": "uniform_fallback",
            },
        })

    request = {
        "dataset_id": "00000000-0000-0000-0000-000000000000",
        "seed_nodes": {seed_type: seed_ids},
        "seed_timestamps": None,
        "plan": {"layers": layers},
        "flags": {
            "replace": False,
            "directed": True,
            "disjoint": True,
            "return_edge_id": False,
            "adaptive": False,
        },
    }

    ticket = flight.Ticket(json.dumps(request).encode("utf-8"))
    reader = client.do_get(ticket)

    edges_by_rel = {}
    nodes_by_type = {}
    stats = None
    batch_count = 0

    for chunk in reader:
        batch = chunk.data
        if batch is None or batch.num_rows == 0:
            md = chunk.app_metadata
            if md is not None and len(md) > 0:
                stats = json.loads(md.to_pybytes())
            continue

        batch_count += 1
        row_type = batch.column("row_type").to_pylist()
        id_a = batch.column("id_a").to_pylist()
        id_b = batch.column("id_b").to_pylist()
        meta_type = batch.column("meta_type").to_pylist()

        for i in range(batch.num_rows):
            if row_type[i] == 0:
                rel_id = meta_type[i]
                edges_by_rel.setdefault(rel_id, []).append(
                    (id_a[i], id_b[i])
                )
            elif row_type[i] == 1:
                type_id = meta_type[i]
                nodes_by_type.setdefault(type_id, []).append(id_b[i])

    print(f"\n  Received {batch_count} Arrow batches")
    print(f"  Edges by relation ID:")
    for rel_id, edges in sorted(edges_by_rel.items()):
        print(f"    rel_id={rel_id}: {len(edges)} edges "
              f"(first 3: {edges[:3]})")

    print(f"  Nodes by type ID:")
    for type_id, nids in sorted(nodes_by_type.items()):
        print(f"    type_id={type_id}: {len(nids)} nodes "
              f"(first 5: {nids[:5]})")

    if stats:
        print(f"\n  Stats: total_nodes={stats['total_nodes']}, "
              f"total_edges={stats['total_edges']}")
        print(f"  nodes_per_type: {stats['nodes_per_type']}")
        print(f"  edges_per_type: "
              f"{[c for c in stats['edges_per_type'] if any(x > 0 for x in c)]}")

        # Validate node counts match stats.
        for type_id, nids in nodes_by_type.items():
            expected = sum(stats["nodes_per_type"][type_id])
            assert len(nids) == expected, (
                f"type_id={type_id}: got {len(nids)} nodes, "
                f"stats says {expected}"
            )
        print("  Node counts match stats: OK")

    assert len(nodes_by_type) > 0, "No nodes returned"
    print("  PASSED\n")


def test_pyg_sampler(
    server_address: str,
    seed_type: str,
    seed_ids: list,
    num_hops: int,
    fanout: int,
):
    """Test via the DiskgraphNeighborSampler class."""
    print("=" * 60)
    print("TEST: DiskgraphNeighborSampler (PyG integration)")
    print("=" * 60)

    # Add kumo repo to path so we can import the module.
    import os
    kumo_repo = os.path.join(os.path.dirname(__file__), "..", "..", "kumo")
    if os.path.isdir(kumo_repo):
        sys.path.insert(0, kumo_repo)

    try:
        from kumo.engine.diskgraph_sampler import (
            DiskgraphNeighborSampler,
        )
    except ImportError as e:
        print(f"  SKIPPED: could not import DiskgraphNeighborSampler: {e}")
        return

    # First fetch schema mapping to know edge/node types.
    client = flight.connect(server_address)
    action = flight.Action("get_type_name_mapping", b"")
    results = list(client.do_action(action))
    mapping = json.loads(results[0].body.to_pybytes())
    del client

    node_types = mapping["node_types"]
    edge_types = []
    for rel in mapping["edge_relations"]:
        if rel["name"]:
            edge_types.append((rel["src_type"], rel["name"], rel["dst_type"]))

    sampler = DiskgraphNeighborSampler(
        server_address=server_address,
        num_neighbors=[fanout] * num_hops,
        edge_types=edge_types,
        node_types=node_types,
        replace=False,
        subgraph_type="directional",
        disjoint=True,
        temporal_strategy="uniform",
        time_attr=None,
    )

    import torch
    seed = {seed_type: torch.tensor(seed_ids, dtype=torch.long)}
    result = sampler._sample(seed)

    print(f"\n  HeteroSamplerOutput:")
    print(f"    node types: {list(result.node.keys())}")
    for k, v in result.node.items():
        print(f"      {k}: shape={v.shape}, dtype={v.dtype}, first={v[:5].tolist()}")

    print(f"    edge types (row): {list(result.row.keys())}")
    for k, v in result.row.items():
        print(f"      {k}: {v.shape[0]} edges")

    if result.num_sampled_nodes:
        print(f"    num_sampled_nodes: {result.num_sampled_nodes}")
    if result.num_sampled_edges:
        print(f"    num_sampled_edges:")
        for k, v in result.num_sampled_edges.items():
            if any(x > 0 for x in v):
                print(f"      {k}: {v}")

    # Validation.
    assert len(result.node) > 0, "No nodes in output"
    print("  PASSED\n")


def main():
    parser = argparse.ArgumentParser(description="Diskgraph Python client smoke test")
    parser.add_argument("--server", default="grpc://localhost:8815",
                        help="Arrow Flight server address")
    parser.add_argument("--seed-type", default="users",
                        help="Seed node type name")
    parser.add_argument("--seed-ids", default="0,1,2",
                        help="Comma-separated seed node IDs")
    parser.add_argument("--hops", type=int, default=2,
                        help="Number of hops")
    parser.add_argument("--fanout", type=int, default=10,
                        help="Fanout per hop")
    args = parser.parse_args()

    seed_ids = [int(x) for x in args.seed_ids.split(",")]

    print(f"Connecting to {args.server}...")
    client = flight.connect(args.server)

    # 1. Schema mapping.
    test_schema_mapping(client)

    # 2. Raw sampling.
    test_sampling(client, args.seed_type, seed_ids, args.hops, args.fanout)

    # 3. PyG sampler integration.
    test_pyg_sampler(args.server, args.seed_type, seed_ids, args.hops, args.fanout)

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
