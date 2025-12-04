# diskgraph-index

Rust crate for building and reading per-shard on-disk indices for the Diskgraph service.

## Overview

`diskgraph-index` ingests a single node shard and its aligned edge shards, producing two files:

- `offsets.bin`: delta-encoded byte offsets for each node’s edge blob.
- `edges.bin`: packed edge blobs containing relation-prefixed counts and contiguous relation segments.

At query time a shard is mmapped and edge lists are served zero-copy by slicing into the edge blob.

## On-disk layout (per shard)

- **Offsets header:** `OffsetFileHeader` with version, node count, total offsets, chunk size, and `edge_blob_word_size` (bytes).
- **Offsets body:** Chunks of `DELTA_CHUNK` (64) offsets:
  - `u16 count`
  - `u64 base` (absolute byte offset for the first node in the chunk)
  - `(count-1) * u16` deltas to subsequent offsets (must fit in `u16`).
- **Edge blob (edges.bin):** For each node:
  - Prefix: varint counts for every relation, in a fixed order.
  - Data: For each relation (in the same order), `count * entry_bytes` packed edges.
    - Entry packing: `dst_id` masked to the minimum bits needed for the destination type, followed by optional `dest_sort_key` bits (if present for that relation). Relation IDs are implicit by position, not stored per edge.

All offsets are byte-based (not words).

### Visuals

Single relation (R0 only):

```
offsets.bin
[header][chunk0: count|base|d1|d2|...]
                   |
                   v
edges.bin
+--------+------------------------------------+
| node0  | cnt(R0) | R0 edge0 | R0 edge1 ...  |
+--------+------------------------------------+
| node1  | cnt(R0) | R0 edges ...             |
+--------+------------------------------------+
| node2  | cnt(R0) | ...                      |
+--------+------------------------------------+
```

Multiple relations (R0, R1):

```
offsets.bin
[header][chunk0: count|base|d1|d2|...]
                   |
                   v
edges.bin
+--------+-------------------------------------------------------------+
| node0  | cnt(R0) | cnt(R1) | R0 edges ... | R1 edges ...             |
+--------+-------------------------------------------------------------+
| node1  | cnt(R0) | cnt(R1) | R0 edges ... | R1 edges ...             |
+--------+-------------------------------------------------------------+
```

Counts are varints in prefix order (R0 then R1), followed by contiguous segments for each relation in the same order.

## Build algorithm (ShardBuilder)

Input: one node shard (sorted by `id`) and all aligned edge shards (sorted by `src_id`, then `dest_sort_key` desc when present).

Steps:
1. Precompute relation layouts:
   - `dst_bits` = min bits to hold `max_id` of destination node type (fallback 64).
   - Masks for `dst_id` and optional `dest_sort_key`.
   - `entry_bytes` = ceil((dst_bits + sort_bits)/8).
2. Stream nodes and edges:
   - Node stream and each edge stream are async Parquet readers (`ParquetRecordBatchStreamBuilder`), caching column indices and downcast arrays per batch.
   - Merge edges across relations with a heap, preserving global ordering and validating sort order.
3. For each node (single pass):
   - Collect edges into per-relation buckets (reused vectors).
   - Compute per-relation counts; estimate node blob size.
   - If the resulting offset delta would exceed `u16::MAX`, run truncation:
     - Greedy leveling of the longest relation lists downward until the node blob fits the delta budget.
     - Truncate buckets accordingly; record counters.
   - Write varint counts (per relation), then packed edges for that relation.
   - Append the cumulative byte offset to the offsets writer.
4. Offsets writer delta-encodes offsets in fixed chunks; writes header once finalized.

Outputs: `ShardBuildOutput` with file paths, `NodeEdgeStats`, `BuildStats`, and `RelationLayout` list for the shard.

### Truncation behavior
- Trigger: projected offset delta > `u16::MAX`.
- Strategy: iteratively level the longest relation lists down (binary search on length) until the node blob fits the delta budget; preserves relative order within each relation; truncation is per node, per relation.
- Counters: `truncated_nodes`, `truncated_edges_total`, `truncated_edges_per_relation`.

### Validations and graceful failures
- Edge ordering: edges must be sorted by `src_id`, then `dest_sort_key` descending (when present). Violation → error.
- Shard alignment: edges must not continue after nodes are exhausted (DanglingEdges).
- Offset overflow: prevented by truncation; residual overflow → error.
- Count/byte overflows: guarded with checked arithmetic; errors on overflow.

## Read algorithm (ShardReader)

Input: shard directory plus `RelationLayout` list (must match build order).

Steps:
1. Mmap `offsets.bin` and `edges.bin`.
2. Parse offset chunks once; chunk index is deterministic: `chunk = node_idx / DELTA_CHUNK`.
3. For a lookup:
   - Decode start/end offsets for the node (two chunk-local delta decodes).
   - Slice `edges.bin` for `[start, end)`.
   - Wrap in `EdgeListView` which:
     - Decodes all relation counts from the prefix (varints).
     - Returns `RelationSegment` slices (data + layout) for zero-copy iteration.

### Efficiency considerations
- No binary search for offsets; chunk position is arithmetic.
- Edge list access is zero-copy; consumer can decode in place using `RelationLayout` masks.
- Relation IDs are implicit by segment order; no per-entry relation bits stored.

## Failure modes (read path)
- Invalid offsets header/chunk mismatch → `InvalidMetadata`.
- Offset pointing outside edge blob or inconsistent delta encoding → `InvalidMetadata`.
- Relation count/segment length mismatch (prefix exceeds blob, or not enough data) → I/O error.

## Counters and stats
- `NodeEdgeStats`: total edges and bytes written for the shard.
- `BuildStats`: nodes/edges ingested, nodes truncated, total truncated edges, and per-relation truncation counts. Returned from build for monitoring/alerting.

## Edge cases handled
- Missing `dest_sort_key`: relation stores only `dst_id` with minimal bits.
- Large fanout nodes: truncated to fit offset delta budget; truncation recorded.
- Sparse shards: empty edge lists are encoded as zero counts; offsets remain monotonic.
- Destination ID width: minimized per relation using destination type `max_id`; masks precomputed.

## Expectations and assumptions
- Input Parquet shards are already sorted:
  - Nodes by `id` ascending.
  - Edges by `src_id` ascending and `dest_sort_key` descending when present.
- Sharding alignment: edges for shard N reference only source IDs in shard N.
- Offsets are always byte-based; `edge_blob_word_size` kept as 1 for future compatibility.
- Relation order in metadata is stable and shared between build and read.
