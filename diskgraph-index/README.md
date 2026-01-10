# diskgraph-index

Rust crate for building and reading per-shard on-disk indices for the Diskgraph service.

## Overview

`diskgraph-index` ingests node shard ranges (from metadata) and aligned edge shard parts, producing two files (plus optional node data). Edge inputs can be a single `*.bin` file or a directory of `*.bin` parts for the shard. Node data is enabled per node type in metadata (`has_node_data`), with one or more `node_data_paths` per shard and a rowpack context file referenced by `node_data_compression.context_path`.

- `offsets.bin`: delta-encoded byte offsets for each node’s edge blob.
- `edges.bin`: packed edge blobs containing relation-prefixed counts and contiguous relation segments (and optional node data payloads).

At query time a shard is mmapped and edge lists are served zero-copy by slicing into the edge blob.

## On-disk layout (per shard)

- **Offsets header:** `OffsetFileHeader` with version, node count, total offsets, chunk size, and `edge_blob_word_size` (bytes).
- **Offsets body:** Chunks of `DELTA_CHUNK` (64) offsets:
  - `u16 count`
  - `u64 base` (absolute byte offset for the first node in the chunk)
  - `(count-1) * u16` deltas to subsequent offsets (must fit in `u16`).
- **Edge blob (edges.bin):** For each node:
  - If the node has zero edges across all relations, nothing is written; its start/end offsets are equal.
  - Otherwise: Prefix compact varint counts for every relation (fixed order), then optional node data length, then per relation `count * entry_bytes` packed edges, then optional node data payload.
    - Count encoding: 1 byte for 0–63 (`00xxxxxx`), 2 bytes for 64–16,383 (`01xxxxxx xxxxxxxx`). Higher values are truncated by the builder (hard cap at the encoding limit, 0x3FFF edges per relation per node).
    - Entry packing: `dst_id` masked to the minimum bits needed for the destination type, followed by optional `dest_sort_key` bits (if present for that relation). Relation IDs are implicit by position, not stored per edge.
    - Node data length: `u16` byte length (little-endian) stored immediately after the counts when node data is enabled. Node data bytes follow all relation segments (rowpack-compressed).

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
+--------+-------------------------------------------------------------------------+
| node0  | cnt(R0) | cnt(R1) | [node_len] | R0 edges ... | R1 edges ... | node |
+--------+-------------------------------------------------------------------------+
| node1  | cnt(R0) | cnt(R1) | [node_len] | R0 edges ... | R1 edges ... | data |
+--------+-------------------------------------------------------------------------+
```

Counts use the compact 1–2 byte scheme in prefix order (R0 then R1), followed by optional `node_len` (u16), then contiguous segments for each relation in the same order. Node data bytes (when enabled) come last.

## Build algorithm (ShardBuilder)

Input: node shard ranges from metadata and all aligned edge shard parts (input order does not need to be sorted).

Steps:
1. Precompute relation layouts:
   - `dst_bits` = min bits to hold `max_id` of destination node type (fallback 64).
   - Masks for `dst_id` and optional `dest_sort_key`.
   - `entry_bytes` = ceil((dst_bits + sort_bits)/8); `dest_sort_key_bits` supports multiples of 8 up to 64.
2. Load and sort edge parts:
   - For each relation + shard, load all `*.bin` parts into memory.
   - Sort by `src_id` (radix), then sort by `dest_sort_key` descending within each `src_id` when present.
3. Stream edges and generate node ids:
   - Node ids are generated from the shard range in metadata.
   - Each edge stream yields rows ordered by `src_id`.
   - Merge edges across relations by draining the per-relation streams for the current node id.
3. For each node (single pass):
   - Collect edges into per-relation buckets (reused vectors).
   - Compute per-relation counts; cap each relation at 10k edges to keep counts within the compact varint range.
   - Estimate node blob size (counts + optional node data length/payload + edges). If the resulting offset delta would exceed `u16::MAX`, run truncation:
     - Greedy leveling of the longest relation lists downward (bulk drops of the current longest plateau) until the node blob fits the delta budget.
     - Truncate buckets accordingly; record counters.
   - Write compact varint counts (per relation), optional `u16` node data length, then packed edges for that relation, then node data bytes.
   - Append the cumulative byte offset to the offsets writer.
4. Offsets writer delta-encodes offsets in fixed chunks; writes header once finalized.

Outputs: `ShardBuildOutput` with file paths, `NodeEdgeStats`, `BuildStats`, and `RelationLayout` list for the shard.

### Truncation behavior
- Trigger: projected offset delta > `u16::MAX`.
- Strategy: iteratively level the longest relation lists down in bulk (plateau drops) until the node blob fits the delta budget; preserves relative order within each relation; truncation is per node, per relation.
- Counters: `truncated_nodes`, `truncated_edges_total`, `truncated_edges_per_relation`.

### Validations and graceful failures
- Shard alignment: edges must not continue after nodes are exhausted (DanglingEdges).
- Zero-edge nodes: allowed. Offsets may repeat (delta 0) when start==end; reader treats zero-length slices as all-zero counts.
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
     - If node data is enabled, reads a `u16` node data length and slices node data bytes after all relation segments.
     - Returns `RelationSegment` slices (data + layout) for zero-copy iteration.

### Efficiency considerations
- No binary search for offsets; chunk position is arithmetic.
- Edge list access is zero-copy; consumer can decode in place using `RelationLayout` masks.
- Relation IDs are implicit by segment order; no per-entry relation bits stored.
- When node data is enabled, `EdgeListView::node_data()` returns the raw bytes for the node (no padding).

## Failure modes (read path)
- Invalid offsets header/chunk mismatch → `InvalidMetadata`.
- Offset pointing outside edge blob or inconsistent delta encoding → `InvalidMetadata`.
- Relation count/segment length mismatch (prefix exceeds blob, or not enough data) → I/O error.
- Node data length overflow or trailing bytes when node data is enabled → I/O error.

## Counters and stats
- `NodeEdgeStats`: total edges and bytes written for the shard.
- `BuildStats`: nodes/edges ingested, nodes truncated, total truncated edges, per-relation truncation counts, and how many relation lists were capped at the varint limit. Returned from build for monitoring/alerting.

## Edge cases handled
- Missing `dest_sort_key`: relation stores only `dst_id` with minimal bits.
- Large fanout nodes: truncated to fit offset delta budget; truncation recorded.
- Sparse shards: empty edge lists are encoded as zero counts; offsets remain monotonic.
- Destination ID width: minimized per relation using destination type `max_id`; masks precomputed.

## Shard explorer (TUI)

`cargo run -p diskgraph-index --bin explore -- <path>` opens a REPL to inspect a pre-built index. Expected layout includes `index_metadata.json` plus `nodes/` and `edges/` shards.

Commands inside the REPL:
- `tables` — list node types with shard and node counts.
- `use <table>` — switch the active table for `node`/`benchmark`.
- `info` — print relation layouts and node count for the active table.
- `node <id>` — show counts and a few edges per relation for the node in the active table.
- `benchmark <threads> [duration]` — run a 20s random lookup benchmark across N threads, including edge + node data reads (duration like `30s` or `5m`).
- `benchmarkall <threads> [duration]` — run a 20s random lookup benchmark across all tables (round robin).
- `quit` / `exit` / `Ctrl+D` — leave the explorer.

Helper script:
- `scripts/explore_local.sh /path/to/index`

## Expectations and assumptions
- Input edge shard parts may be unsorted; the builder sorts by `src_id` and `dest_sort_key` as needed.
- Sharding alignment: edges for shard N reference only source IDs in shard N.
- Offsets are always byte-based; `edge_blob_word_size` kept as 1 for future compatibility.
- Relation order in metadata is stable and shared between build and read.
