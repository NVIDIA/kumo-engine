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

diff --git a/scripts/diagnose_edge_direction.py b/scripts/diagnose_edge_direction.py
new file mode 100644
index 0000000..56ef1f4
--- /dev/null
+++ b/scripts/diagnose_edge_direction.py
@@ -0,0 +1,417 @@
+#!/usr/bin/env python3
+"""Diagnostic: test whether diskgraph sampler output satisfies
+the GNN layer-trim invariant used by kumo's message-passing layers.
+
+The kumo GNN conv does:
+    out = self.aggr(x_src[edge_index[0]], edge_index[1], dim_size=x_dst.size(0))
+
+Where x_dst is trimmed per GNN layer using num_sampled_nodes.
+For a 2-layer GNN processing L hops:
+  - Layer L-1 (deepest): x_dst trimmed to total - outermost_ring nodes
+    Only uses hop-(L-1) edges (if trim_to_layer edge slicing is active)
+    OR all edges (if only node tensors are trimmed)
+  - Layer 0 (shallowest): x_dst trimmed to seed nodes
+
+The critical invariant:
+  For each GNN layer, edge_index[1] (col) must be < x_dst.size(0).
+
+This script checks whether diskgraph's output satisfies this invariant
+under two processing models:
+  A) "trim edges per layer" — each layer only sees its hop's edges
+  B) "all edges, trim nodes only" — each layer sees ALL edges but with trimmed nodes
+
+Usage:
+    python scripts/diagnose_edge_direction.py --server grpc://localhost:8815
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
+    action = flight.Action("get_type_name_mapping", b"")
+    results = list(client.do_action(action))
+    return json.loads(results[0].body.to_pybytes())
+
+
+def sample_via_pyg(server_address, mapping, num_hops, fanout, seed_type, seed_ids):
+    """Run through DiskgraphNeighborSampler and return the output."""
+    kumo_repo = os.path.join(os.path.dirname(__file__), "..", "..", "kumo")
+    if os.path.isdir(kumo_repo):
+        sys.path.insert(0, os.path.abspath(kumo_repo))
+
+    from kumo.engine.diskgraph_sampler import DiskgraphNeighborSampler
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
+    return sampler._sample(seed)
+
+
+def check_hop_ordering(result):
+    """Check if edges are actually hop-ordered by verifying BFS source ranges."""
+    print("\n" + "=" * 70)
+    print("CHECK 1: Are edges hop-ordered?")
+    print("=" * 70)
+
+    all_ok = True
+    for et, hop_counts in result.num_sampled_edges.items():
+        row = result.row.get(et)
+        if row is None:
+            continue
+
+        src_type, _, dst_type = et
+        hop_list = hop_counts if isinstance(hop_counts, list) else hop_counts.tolist()
+        while hop_list and hop_list[-1] == 0:
+            hop_list = hop_list[:-1]
+        if not hop_list:
+            continue
+
+        src_node_hops = result.num_sampled_nodes.get(src_type, [])
+        if hasattr(src_node_hops, 'tolist'):
+            src_node_hops = src_node_hops.tolist()
+
+        src_cumsum = []
+        s = 0
+        for c in src_node_hops:
+            s += c
+            src_cumsum.append(s)
+
+        offset = 0
+        print(f"\n  {et}")
+        print(f"    hop_counts: {hop_list}")
+        print(f"    src '{src_type}' cumsum: {src_cumsum}")
+        for hop_k, count_k in enumerate(hop_list):
+            if count_k == 0:
+                offset += count_k
+                continue
+            segment = row[offset:offset + count_k]
+            seg_min = segment.min().item()
+            seg_max = segment.max().item()
+
+            # Hop-k BFS sources should be from hops [0..k] of the src type
+            expected_max = src_cumsum[hop_k] - 1 if hop_k < len(src_cumsum) else src_cumsum[-1] - 1
+
+            ok = seg_max <= expected_max
+            status = "OK" if ok else "** MIXED HOPS **"
+            print(f"    hop {hop_k}: {count_k} edges, row=[{seg_min}..{seg_max}], "
+                  f"expected_row_max={expected_max} {status}")
+            if not ok:
+                all_ok = False
+            offset += count_k
+
+    print(f"\n  {'PASS: edges are hop-ordered' if all_ok else 'FAIL: hop mixing detected'}")
+    return all_ok
+
+
+def check_gnn_invariant_with_edge_trim(result):
+    """Model A: GNN trims both edges AND nodes per layer (PyG trim_to_layer).
+    Layer k removes outermost hop's edges and nodes.
+    """
+    print("\n" + "=" * 70)
+    print("CHECK 2: GNN invariant with per-layer edge trimming")
+    print("  (each layer only processes its hop's edges)")
+    print("=" * 70)
+
+    num_hops = max(
+        len(v) if isinstance(v, list) else v.numel()
+        for v in result.num_sampled_nodes.values()
+    ) - 1  # -1 because first entry is seeds
+
+    all_ok = True
+    for layer in range(num_hops):
+        # Layer 0 = deepest (processes last hop)
+        # Layer num_hops-1 = shallowest (processes first hop)
+        # BUT in trim_to_layer: layer 0 = shallowest (no trim)
+        # Let's use GNN layer numbering: 0=shallowest, L-1=deepest
+        # Deepest layer processes hop (L-1), shallowest processes hop 0
+        hop = num_hops - 1 - layer  # hop for this GNN layer
+
+        for et, hop_counts in result.num_sampled_edges.items():
+            col = result.col.get(et)
+            if col is None:
+                continue
+
+            src_type, _, dst_type = et
+            hop_list = hop_counts if isinstance(hop_counts, list) else hop_counts.tolist()
+
+            if hop >= len(hop_list) or hop_list[hop] == 0:
+                continue
+
+            # Get the hop segment
+            offset = sum(hop_list[:hop])
+            count = hop_list[hop]
+            col_segment = col[offset:offset + count]
+
+            if col_segment.numel() == 0:
+                continue
+
+            col_max = col_segment.max().item()
+
+            # x_dst trimmed: remove hops beyond the current layer
+            # At deepest GNN layer (processing last hop):
+            #   x_dst = total - last_hop_nodes = cumsum[num_hops-1]
+            # At shallowest GNN layer (processing first hop):
+            #   x_dst = seeds only = cumsum[0]
+            dst_node_hops = result.num_sampled_nodes.get(dst_type, [])
+            if hasattr(dst_node_hops, 'tolist'):
+                dst_node_hops = dst_node_hops.tolist()
+
+            # For GNN layer processing hop k:
+            # x_dst = total_dst - sum(dst_node_hops[k+1:])
+            # = sum(dst_node_hops[0:k+1]) = cumsum[k]
+            # Wait: num_sampled_nodes has seeds at index 0, hop-0 at index 1, etc.
+            # So cumsum[k+1] = seeds + hop0 + ... + hopk nodes
+            dst_cumsum = []
+            s = 0
+            for c in dst_node_hops:
+                s += c
+                dst_cumsum.append(s)
+
+            # GNN layer for hop k trims x_dst to cumsum[k+1]
+            # (includes seeds + nodes discovered at hops 0..k)
+            x_dst_size = dst_cumsum[hop + 1] if (hop + 1) < len(dst_cumsum) else dst_cumsum[-1]
+
+            ok = col_max < x_dst_size
+            if not ok:
+                all_ok = False
+            status = "OK" if ok else "** FAIL **"
+            print(f"  GNN layer {layer} (hop {hop}): {et}")
+            print(f"    col_max={col_max}, x_dst_size={x_dst_size} "
+                  f"(cumsum[{hop+1}]) {status}")
+
+    print(f"\n  {'PASS' if all_ok else 'FAIL: col exceeds trimmed x_dst'}")
+    return all_ok
+
+
+def check_gnn_invariant_no_edge_trim(result):
+    """Model B: GNN uses ALL edges at each layer, only trims node tensors.
+    This is what likely happens in kumo's GNN.
+    """
+    print("\n" + "=" * 70)
+    print("CHECK 3: GNN invariant WITHOUT edge trimming")
+    print("  (each layer sees ALL edges, only node tensors are trimmed)")
+    print("=" * 70)
+
+    num_hops = max(
+        len(v) if isinstance(v, list) else v.numel()
+        for v in result.num_sampled_nodes.values()
+    ) - 1
+
+    all_ok = True
+    for layer in range(num_hops):
+        hop = num_hops - 1 - layer
+
+        for et in result.row.keys():
+            col = result.col.get(et)
+            if col is None or col.numel() == 0:
+                continue
+
+            src_type, _, dst_type = et
+            col_max = col.max().item()
+
+            dst_node_hops = result.num_sampled_nodes.get(dst_type, [])
+            if hasattr(dst_node_hops, 'tolist'):
+                dst_node_hops = dst_node_hops.tolist()
+
+            dst_cumsum = []
+            s = 0
+            for c in dst_node_hops:
+                s += c
+                dst_cumsum.append(s)
+
+            x_dst_size = dst_cumsum[hop + 1] if (hop + 1) < len(dst_cumsum) else dst_cumsum[-1]
+
+            ok = col_max < x_dst_size
+            if not ok:
+                all_ok = False
+            status = "OK" if ok else "** FAIL **"
+            print(f"  GNN layer {layer} (hop {hop}): {et}")
+            print(f"    col_max={col_max} (over ALL edges), x_dst_size={x_dst_size} {status}")
+
+    print(f"\n  {'PASS' if all_ok else 'FAIL: col exceeds trimmed x_dst (no edge trim)'}")
+    return all_ok
+
+
+def check_reversed_direction(result, mapping):
+    """Check what happens if we swap row/col and reverse edge types.
+    For each relation, use the reverse relation's edge_type with swapped row/col.
+    """
+    print("\n" + "=" * 70)
+    print("CHECK 4: Hypothetical fix — reverse edge types")
+    print("  (swap row↔col and use reverse relation name)")
+    print("=" * 70)
+
+    # Build forward↔reverse mapping
+    rels = mapping["edge_relations"]
+    # Group by (src_type, dst_type, table) pairs — forward and reverse share the same table
+    # Actually, let's just pair by index: rel 0↔1, 2↔3, 4↔5
+    # (comments fwd/rev, likes fwd/rev, follows fwd/rev)
+    pairs = {}
+    for i, rel in enumerate(rels):
+        name = rel["name"]
+        if not name:
+            continue
+        # Parse: edge_{table}_{direction}
+        # src_to_dst ↔ dst_to_src
+        if "src_to_dst" in name:
+            partner = name.replace("src_to_dst", "dst_to_src")
+        elif "dst_to_src" in name:
+            partner = name.replace("dst_to_src", "src_to_dst")
+        else:
+            continue
+        for j, rel2 in enumerate(rels):
+            if rel2["name"] == partner:
+                pairs[i] = j
+                break
+
+    print(f"  Forward↔Reverse pairs: {pairs}")
+
+    # After reversal, for edge type (A, rel, B) → becomes (B, rev_rel, A)
+    # row = old col (B indices), col = old row (A indices)
+    # Now col = A = frontier (closer to seed) → should satisfy trim bounds
+
+    num_hops = max(
+        len(v) if isinstance(v, list) else v.numel()
+        for v in result.num_sampled_nodes.values()
+    ) - 1
+
+    all_ok = True
+    for et in result.row.keys():
+        # Find the reverse edge type
+        src_type, rel_name, dst_type = et
+
+        # After reversal: col contains the OLD row values (src_type = frontier)
+        # The GNN would process edge type (dst_type, rev_name, src_type)
+        # with x_dst = src_type features (the frontier type)
+        reversed_col = result.row[et]  # old row = src_type indices
+
+        if reversed_col.numel() == 0:
+            continue
+
+        col_max = reversed_col.max().item()
+
+        # The trimmed node count for the OLD src_type (now the col/dst type)
+        dst_node_hops = result.num_sampled_nodes.get(src_type, [])
+        if hasattr(dst_node_hops, 'tolist'):
+            dst_node_hops = dst_node_hops.tolist()
+
+        dst_cumsum = []
+        s = 0
+        for c in dst_node_hops:
+            s += c
+            dst_cumsum.append(s)
+
+        hop_counts = result.num_sampled_edges.get(et, [])
+        if hasattr(hop_counts, 'tolist'):
+            hop_counts = hop_counts.tolist()
+        while hop_counts and hop_counts[-1] == 0:
+            hop_counts = hop_counts[:-1]
+
+        # Check per-hop segments
+        offset = 0
+        for hop_k, count_k in enumerate(hop_counts):
+            if count_k == 0:
+                offset += count_k
+                continue
+
+            seg_col = reversed_col[offset:offset + count_k]
+            seg_col_max = seg_col.max().item()
+
+            # After reversal, the col type is src_type (frontier)
+            # At the GNN layer for hop k: x_dst = cumsum[k+1]
+            x_dst_size = dst_cumsum[hop_k + 1] if (hop_k + 1) < len(dst_cumsum) else dst_cumsum[-1]
+
+            ok = seg_col_max < x_dst_size
+            if not ok:
+                all_ok = False
+            status = "OK" if ok else "** FAIL **"
+            print(f"  {et} (reversed) hop {hop_k}: col_max={seg_col_max}, "
+                  f"x_dst_size={x_dst_size} {status}")
+
+            offset += count_k
+
+    print(f"\n  {'PASS: reversing edge types fixes the invariant' if all_ok else 'FAIL'}")
+    return all_ok
+
+
+def main():
+    parser = argparse.ArgumentParser()
+    parser.add_argument("--server", default="grpc://localhost:8815")
+    parser.add_argument("--seed-type", default="users")
+    parser.add_argument("--num-seeds", type=int, default=64)
+    parser.add_argument("--hops", type=int, default=2)
+    parser.add_argument("--fanout", type=int, default=10)
+    args = parser.parse_args()
+
+    seed_ids = list(range(args.num_seeds))
+    client = flight.connect(args.server)
+    mapping = get_schema(client)
+    del client
+
+    print(f"Sampling: {args.num_seeds} {args.seed_type} seeds, "
+          f"{args.hops} hops, fanout={args.fanout}")
+
+    result = sample_via_pyg(
+        args.server, mapping, args.hops, args.fanout,
+        args.seed_type, seed_ids
+    )
+
+    print(f"\nnum_sampled_nodes:")
+    for k, v in result.num_sampled_nodes.items():
+        print(f"  {k}: {v}")
+    print(f"num_sampled_edges:")
+    for k, v in result.num_sampled_edges.items():
+        vl = v if isinstance(v, list) else v.tolist()
+        if any(x > 0 for x in vl):
+            print(f"  {k}: {vl}")
+
+    hop_ordered = check_hop_ordering(result)
+    gnn_with_trim = check_gnn_invariant_with_edge_trim(result)
+    gnn_no_trim = check_gnn_invariant_no_edge_trim(result)
+    reversal_fix = check_reversed_direction(result, mapping)
+
+    print("\n" + "=" * 70)
+    print("SUMMARY")
+    print("=" * 70)
+    print(f"  Edges hop-ordered:              {'YES' if hop_ordered else 'NO'}")
+    print(f"  GNN OK with edge trim:          {'YES' if gnn_with_trim else 'NO'}")
+    print(f"  GNN OK without edge trim:       {'YES' if gnn_no_trim else 'NO'}")
+    print(f"  Edge type reversal would fix:   {'YES' if reversal_fix else 'NO'}")
+
+    if hop_ordered and not gnn_with_trim:
+        print("\n  DIAGNOSIS: Edges are hop-ordered but GNN trim still fails.")
+        print("  This means the issue is NOT about edge ordering.")
+        print("  The issue is about BFS DIRECTION: diskgraph puts discovered")
+        print("  (further-from-seed) nodes in col, but the GNN expects")
+        print("  closer-to-seed nodes in col for its layer trimming to work.")
+    elif hop_ordered and gnn_with_trim and not gnn_no_trim:
+        print("\n  DIAGNOSIS: Edges are hop-ordered and GNN works WITH edge trimming")
+        print("  but fails WITHOUT it. Check if kumo's GNN uses trim_to_layer.")
+
+    if not hop_ordered or not gnn_with_trim or not gnn_no_trim:
+        sys.exit(1)
+
+
+if __name__ == "__main__":
+    main()
