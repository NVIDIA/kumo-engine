commit a91fd1cf6461d6b539037d81fae6a97d74a61592
Author: mpelko-kumo <miha.pelko@kumo.ai>
Date:   Thu Feb 12 17:52:13 2026 +0000

    fix: correct edge direction convention and stats shape for kumoml compat
    
    Two contract mismatches prevented E2E training from working:
    
    1. edges_per_type was allocated with num_hops+1 slots, creating a trailing
       zero. kumoml reads num_sampled_edges[key][-i] for edge trimming, so the
       trailing zero caused no-trim at layer 1 → assertion failure.
       Fix: allocate num_hops slots (nodes_per_type correctly stays num_hops+1).
    
    2. diskgraph BFS traverses edges forward (col = further-from-seed), but
       kumoml expects PyG's CSC convention (col = closer-to-seed). The GNN
       intentionally skips edge trimming at layer 0, relying on col being
       bounded by closer-to-seed nodes.
       Fix: gateway reverses edge direction in handle_edge() using a
       forward↔reverse relation map built from *_src_to_dst ↔ *_dst_to_src
       naming convention. Stats edges_per_type is swapped accordingly.
    
    Also adds diagnostic scripts for regression testing.
    
    Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

diff --git a/scripts/diagnose_hop_ordering.py b/scripts/diagnose_hop_ordering.py
new file mode 100644
index 0000000..850172a
--- /dev/null
+++ b/scripts/diagnose_hop_ordering.py
@@ -0,0 +1,396 @@
+#!/usr/bin/env python3
+"""Diagnostic script to check whether edges in diskgraph sampling output
+are ordered by hop within each relation type.
+
+This tests the hypothesis from HOP_ORDERING_FIX_PLAN.md:
+- PyG expects edges ordered by hop per relation, so that
+  num_sampled_edges[rel][k] can be used to slice the edge tensor.
+- The gateway's EdgeBuffer flushes at 64k boundaries, potentially
+  mixing hops.
+
+Usage:
+    python scripts/diagnose_hop_ordering.py --server grpc://localhost:8815
+"""
+import argparse
+import json
+import sys
+import os
+
+import pyarrow.flight as flight
+
+
+def get_schema(client):
+    action = flight.Action("get_schema_mapping", b"")
+    results = list(client.do_action(action))
+    return json.loads(results[0].body.to_pybytes())
+
+
+def sample_raw(client, seed_type, seed_ids, num_hops, fanout):
+    """Issue a sampling request and return raw per-relation edges + stats."""
+    layers = []
+    for _ in range(num_hops):
+        layers.append({
+            "fanouts": {},
+            "default_fanout": {
+                "fanout": fanout,
+                "mode": "uniform",
+                "absent_key_policy": "uniform_fallback",
+            },
+        })
+
+    request = {
+        "dataset_id": "00000000-0000-0000-0000-000000000000",
+        "seed_nodes": {seed_type: seed_ids},
+        "seed_timestamps": None,
+        "plan": {"layers": layers},
+        "flags": {
+            "replace": False,
+            "directed": True,
+            "disjoint": True,
+            "return_edge_id": False,
+            "adaptive": False,
+        },
+    }
+
+    ticket = flight.Ticket(json.dumps(request).encode("utf-8"))
+    reader = client.do_get(ticket)
+
+    # Collect edges in arrival order per relation
+    edges_by_rel = {}   # rel_id -> [(src, dst)] in arrival order
+    nodes_by_type = {}  # type_id -> [global_id] in arrival order
+    stats = None
+
+    for chunk in reader:
+        batch = chunk.data
+        if batch is None or batch.num_rows == 0:
+            md = chunk.app_metadata
+            if md is not None and len(md) > 0:
+                stats = json.loads(md.to_pybytes())
+            continue
+
+        row_type = batch.column("row_type").to_pylist()
+        id_a = batch.column("id_a").to_pylist()
+        id_b = batch.column("id_b").to_pylist()
+        meta_type = batch.column("meta_type").to_pylist()
+
+        for i in range(batch.num_rows):
+            if row_type[i] == 0:  # edge
+                rel_id = meta_type[i]
+                edges_by_rel.setdefault(rel_id, []).append((id_a[i], id_b[i]))
+            elif row_type[i] == 1:  # node
+                type_id = meta_type[i]
+                nodes_by_type.setdefault(type_id, []).append(id_b[i])
+
+    return edges_by_rel, nodes_by_type, stats
+
+
+def check_hop_ordering(edges_by_rel, nodes_by_type, stats, mapping):
+    """Check whether edges are correctly ordered by hop per relation.
+
+    For each relation r and hop k, the edges in segment
+    [offset_k, offset_k + num_sampled_edges[r][k]) should have
+    dst_local < sum(num_sampled_nodes[dst_type][0..k+1]).
+
+    If edges are hop-ordered, hop-0 edges should have dst values within
+    the hop-0+hop-1 node range (i.e., nodes discovered at seed + hop 0),
+    and hop-1 edges can reference any node in the full sampled set.
+    """
+    print("\n" + "=" * 70)
+    print("DIAGNOSTIC: Hop-ordering check")
+    print("=" * 70)
+
+    rel_info = {}
+    for i, rel in enumerate(mapping["edge_relations"]):
+        rel_info[i] = rel
+
+    type_name_to_id = {name: i for i, name in enumerate(mapping["node_types"])}
+
+    nodes_per_type = stats["nodes_per_type"]  # type_id -> [count_hop0, count_hop1, ...]
+    edges_per_type = stats["edges_per_type"]  # rel_id -> [count_hop0, count_hop1, ...]
+
+    any_failure = False
+
+    for rel_id, edges in sorted(edges_by_rel.items()):
+        rel = rel_info.get(rel_id, {})
+        rel_name = rel.get("name", f"rel_{rel_id}")
+        src_type_name = rel.get("src_type", "?")
+        dst_type_name = rel.get("dst_type", "?")
+        src_type_id = type_name_to_id.get(src_type_name, -1)
+        dst_type_id = type_name_to_id.get(dst_type_name, -1)
+
+        hop_counts = edges_per_type[rel_id] if rel_id < len(edges_per_type) else []
+        # Filter out trailing zeros
+        while hop_counts and hop_counts[-1] == 0:
+            hop_counts = hop_counts[:-1]
+
+        if not hop_counts or all(c == 0 for c in hop_counts):
+            continue
+
+        total_edges_from_stats = sum(hop_counts)
+        total_edges_actual = len(edges)
+
+        print(f"\n  Relation {rel_id}: {src_type_name} --[{rel_name}]--> {dst_type_name}")
+        print(f"    edges_per_hop (from stats): {hop_counts}")
+        print(f"    total edges: stats={total_edges_from_stats}, actual={total_edges_actual}")
+
+        if total_edges_from_stats != total_edges_actual:
+            print(f"    ** MISMATCH: stats total != actual edge count")
+            any_failure = True
+
+        # Compute cumulative node counts for dst type
+        dst_node_hops = nodes_per_type[dst_type_id] if dst_type_id >= 0 and dst_type_id < len(nodes_per_type) else []
+        src_node_hops = nodes_per_type[src_type_id] if src_type_id >= 0 and src_type_id < len(nodes_per_type) else []
+
+        dst_cumsum = []
+        s = 0
+        for c in dst_node_hops:
+            s += c
+            dst_cumsum.append(s)
+
+        src_cumsum = []
+        s = 0
+        for c in src_node_hops:
+            s += c
+            src_cumsum.append(s)
+
+        print(f"    dst type '{dst_type_name}' (id={dst_type_id}): nodes_per_hop={dst_node_hops}, cumsum={dst_cumsum}")
+        print(f"    src type '{src_type_name}' (id={src_type_id}): nodes_per_hop={src_node_hops}, cumsum={src_cumsum}")
+
+        # Now check each hop segment
+        offset = 0
+        for hop_k, count_k in enumerate(hop_counts):
+            if count_k == 0:
+                offset += count_k
+                continue
+
+            segment = edges[offset : offset + count_k]
+            if not segment:
+                offset += count_k
+                continue
+
+            src_vals = [e[0] for e in segment]
+            dst_vals = [e[1] for e in segment]
+            src_max = max(src_vals)
+            dst_max = max(dst_vals)
+
+            # At hop k, valid dst indices are [0, dst_cumsum[k+1])
+            # because hop k edges connect nodes discovered at hop k (source)
+            # to nodes discovered at hop k+1 (destination) — but destination
+            # local IDs are from the global per-type space, so they can be
+            # anywhere in [0, dst_cumsum[k+1]).
+            #
+            # More precisely, for layer k of the GNN:
+            # x_dst is trimmed to dst_cumsum[k] nodes (all nodes up to hop k).
+            # So edge_index[1] (dst) must be < dst_cumsum[k].
+            #
+            # Wait — let's be precise about the PyG convention:
+            # num_sampled_nodes includes seeds at index 0.
+            # Layer 0 edges go from hop-0 discovered nodes TO seeds.
+            # So for layer 0: x_dst = nodes[0:num_nodes[0]] (seeds only).
+            # For layer 1: x_dst = nodes[0:num_nodes[0]+num_nodes[1]] (seeds + hop0).
+            #
+            # Actually PyG processes layers in REVERSE:
+            # layer[-1] (deepest) processes hop[-1] edges first.
+            # But the edge segments are ordered hop0, hop1, ...
+            # and are sliced accordingly.
+            #
+            # The key invariant: for hop k edges, dst must be < dst_cumsum[k].
+            # This is because these edges land on nodes from hops [0..k-1]
+            # (already discovered), not on newly discovered hop-k nodes.
+
+            # Bound for dst: must be < cumulative nodes up to and including hop k
+            # (hop k edges connect TO nodes in hops 0..k)
+            dst_bound = dst_cumsum[hop_k] if hop_k < len(dst_cumsum) else dst_cumsum[-1]
+
+            # Bound for src: must be < cumulative nodes up to and including hop k+1
+            src_bound = src_cumsum[hop_k + 1] if (hop_k + 1) < len(src_cumsum) else src_cumsum[-1]
+
+            hop_ok = dst_max < dst_bound
+            print(f"    hop {hop_k}: {count_k} edges, "
+                  f"src_max={src_max}, dst_max={dst_max}, "
+                  f"dst_bound={dst_bound} (cumsum[{hop_k}]), "
+                  f"{'OK' if hop_ok else '** FAIL **'}")
+
+            if not hop_ok:
+                any_failure = True
+                # Show some offending edges
+                offending = [(i, s, d) for i, (s, d) in enumerate(segment) if d >= dst_bound]
+                print(f"      {len(offending)} edges have dst >= {dst_bound}")
+                for idx, s, d in offending[:5]:
+                    print(f"        edge[{offset+idx}]: src={s}, dst={d}")
+                if len(offending) > 5:
+                    print(f"        ... and {len(offending)-5} more")
+
+            offset += count_k
+
+    print()
+    if any_failure:
+        print("RESULT: ** HOP ORDERING VIOLATION DETECTED **")
+        print("Edges are NOT correctly ordered by hop per relation.")
+        print("This confirms the hypothesis in HOP_ORDERING_FIX_PLAN.md.")
+    else:
+        print("RESULT: All hop segments pass bounds check.")
+        print("Edges appear to be correctly ordered by hop.")
+    print()
+    return not any_failure
+
+
+def check_pyg_sampler(server_address, mapping, num_hops, fanout, seed_type, seed_ids):
+    """Run through the DiskgraphNeighborSampler and check the same invariant."""
+    print("\n" + "=" * 70)
+    print("DIAGNOSTIC: DiskgraphNeighborSampler hop-ordering check")
+    print("=" * 70)
+
+    kumo_repo = os.path.join(os.path.dirname(__file__), "..", "..", "kumo")
+    if os.path.isdir(kumo_repo):
+        sys.path.insert(0, os.path.abspath(kumo_repo))
+
+    try:
+        from kumo.engine.diskgraph_sampler import DiskgraphNeighborSampler
+    except ImportError as e:
+        print(f"  SKIPPED: {e}")
+        return True
+
+    import torch
+
+    node_types = mapping["node_types"]
+    edge_types = [
+        (rel["src_type"], rel["name"], rel["dst_type"])
+        for rel in mapping["edge_relations"] if rel["name"]
+    ]
+
+    sampler = DiskgraphNeighborSampler(
+        server_address=server_address,
+        num_neighbors=[fanout] * num_hops,
+        edge_types=edge_types,
+        node_types=node_types,
+        replace=False,
+        subgraph_type="directional",
+        disjoint=True,
+        temporal_strategy="uniform",
+        time_attr=None,
+    )
+
+    seed = {seed_type: torch.tensor(seed_ids, dtype=torch.long)}
+    result = sampler._sample(seed)
+
+    any_failure = False
+
+    print(f"\n  num_sampled_nodes: {result.num_sampled_nodes}")
+    print(f"  num_sampled_edges:")
+    for k, v in result.num_sampled_edges.items():
+        if any(x > 0 for x in v):
+            print(f"    {k}: {v}")
+
+    for edge_type, hop_counts in result.num_sampled_edges.items():
+        row = result.row.get(edge_type)
+        col = result.col.get(edge_type)
+        if row is None or col is None:
+            continue
+
+        hop_counts_list = hop_counts if isinstance(hop_counts, list) else hop_counts.tolist()
+        # strip trailing zeros
+        while hop_counts_list and hop_counts_list[-1] == 0:
+            hop_counts_list = hop_counts_list[:-1]
+
+        if not hop_counts_list:
+            continue
+
+        src_type, rel_name, dst_type = edge_type
+
+        # Get node hop counts for dst type
+        dst_node_hops = result.num_sampled_nodes.get(dst_type, [])
+        if hasattr(dst_node_hops, 'tolist'):
+            dst_node_hops = dst_node_hops.tolist()
+
+        dst_cumsum = []
+        s = 0
+        for c in dst_node_hops:
+            s += c
+            dst_cumsum.append(s)
+
+        total_dst_nodes = result.node[dst_type].shape[0] if dst_type in result.node else 0
+
+        print(f"\n  Edge type: {edge_type}")
+        print(f"    total edges: row={row.shape[0]}, col={col.shape[0]}")
+        print(f"    hop_counts: {hop_counts_list}")
+        print(f"    dst '{dst_type}': {total_dst_nodes} nodes, per-hop={dst_node_hops}, cumsum={dst_cumsum}")
+
+        offset = 0
+        for hop_k, count_k in enumerate(hop_counts_list):
+            if count_k == 0:
+                offset += count_k
+                continue
+
+            col_segment = col[offset : offset + count_k]
+            row_segment = row[offset : offset + count_k]
+
+            col_max = col_segment.max().item() if count_k > 0 else -1
+            row_max = row_segment.max().item() if count_k > 0 else -1
+
+            # dst bound: cumsum up to hop k
+            dst_bound = dst_cumsum[hop_k] if hop_k < len(dst_cumsum) else dst_cumsum[-1]
+
+            hop_ok = col_max < dst_bound
+            print(f"    hop {hop_k}: {count_k} edges, "
+                  f"row_max={row_max}, col_max={col_max}, "
+                  f"dst_bound={dst_bound}, "
+                  f"{'OK' if hop_ok else '** FAIL **'}")
+
+            if not hop_ok:
+                any_failure = True
+                offending_mask = col_segment >= dst_bound
+                n_offending = offending_mask.sum().item()
+                print(f"      {n_offending} edges have col >= {dst_bound}")
+
+            offset += count_k
+
+    print()
+    if any_failure:
+        print("RESULT: ** HOP ORDERING VIOLATION in PyG sampler output **")
+    else:
+        print("RESULT: PyG sampler output passes hop-ordering check.")
+    print()
+    return not any_failure
+
+
+def main():
+    parser = argparse.ArgumentParser(description="Diagnose hop-ordering in diskgraph sampling")
+    parser.add_argument("--server", default="grpc://localhost:8815")
+    parser.add_argument("--seed-type", default="users")
+    parser.add_argument("--num-seeds", type=int, default=64,
+                        help="Number of seed nodes")
+    parser.add_argument("--hops", type=int, default=2)
+    parser.add_argument("--fanout", type=int, default=10)
+    args = parser.parse_args()
+
+    seed_ids = list(range(args.num_seeds))
+
+    print(f"Connecting to {args.server}...")
+    client = flight.connect(args.server)
+
+    mapping = get_schema(client)
+    print(f"Schema: {len(mapping['node_types'])} node types, "
+          f"{len(mapping['edge_relations'])} relations")
+
+    print(f"\nSampling: {args.num_seeds} {args.seed_type} seeds, "
+          f"{args.hops} hops, fanout={args.fanout}")
+
+    # Raw Arrow Flight check
+    edges_by_rel, nodes_by_type, stats = sample_raw(
+        client, args.seed_type, seed_ids, args.hops, args.fanout
+    )
+    raw_ok = check_hop_ordering(edges_by_rel, nodes_by_type, stats, mapping)
+
+    # PyG sampler check
+    pyg_ok = check_pyg_sampler(
+        args.server, mapping, args.hops, args.fanout,
+        args.seed_type, seed_ids
+    )
+
+    if not raw_ok or not pyg_ok:
+        sys.exit(1)
+
+
+if __name__ == "__main__":
+    main()
