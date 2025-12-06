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
  - If the node has zero edges across all relations, nothing is written; its start/end offsets are equal.
  - Otherwise: Prefix compact varint counts for every relation (fixed order), then per relation `count * entry_bytes` packed edges.
    - Count encoding: 1 byte for 0–63 (`00xxxxxx`), 2 bytes for 64–16,383 (`01xxxxxx xxxxxxxx`). Higher values are truncated by the builder (hard cap at the encoding limit, 0x3FFF edges per relation per node).
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

Counts use the compact 1–2 byte scheme in prefix order (R0 then R1), followed by contiguous segments for each relation in the same order.

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
   - Compute per-relation counts; cap each relation at 10k edges to keep counts within the compact varint range.
   - Estimate node blob size. If the resulting offset delta would exceed `u16::MAX`, run truncation:
     - Greedy leveling of the longest relation lists downward (bulk drops of the current longest plateau) until the node blob fits the delta budget.
     - Truncate buckets accordingly; record counters.
   - Write compact varint counts (per relation), then packed edges for that relation.
   - Append the cumulative byte offset to the offsets writer.
4. Offsets writer delta-encodes offsets in fixed chunks; writes header once finalized.

Outputs: `ShardBuildOutput` with file paths, `NodeEdgeStats`, `BuildStats`, and `RelationLayout` list for the shard.

### Truncation behavior
- Trigger: projected offset delta > `u16::MAX`.
- Strategy: iteratively level the longest relation lists down in bulk (plateau drops) until the node blob fits the delta budget; preserves relative order within each relation; truncation is per node, per relation.
- Counters: `truncated_nodes`, `truncated_edges_total`, `truncated_edges_per_relation`.

### Validations and graceful failures
- Edge ordering: edges must be sorted by `src_id`, then `dest_sort_key` descending (when present). Violation → error.
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
- `BuildStats`: nodes/edges ingested, nodes truncated, total truncated edges, per-relation truncation counts, and how many relation lists were capped at the varint limit. Returned from build for monitoring/alerting.

## Edge cases handled
- Missing `dest_sort_key`: relation stores only `dst_id` with minimal bits.
- Large fanout nodes: truncated to fit offset delta budget; truncation recorded.
- Sparse shards: empty edge lists are encoded as zero counts; offsets remain monotonic.
- Destination ID width: minimized per relation using destination type `max_id`; masks precomputed.

## Cloud ingest tests

We support end-to-end ingest/read tests against S3, GCS, and ADLS using local emulators (MinIO, fake-gcs-server, Azurite) or real cloud endpoints. Tests fail fast if misconfigured and actually read/write data for two relations per shard with multiple row groups and multiple offset chunks.

- Quick run with emulators (requires Docker):
  ```bash
  ./diskgraph-index/run_cloud_tests.sh
  ```
  This spins up emulators, creates buckets/containers, sets env vars, runs `cargo test -p diskgraph-index --test cloud_ingest`, then tears down containers.

- Manual run against your own endpoints (IAM/ADC/MSI preferred; omit keys to use ambient creds):
  ```bash
  # S3 / MinIO style
  RUN_CLOUD_TESTS=1 \
  S3_BUCKET=your-bucket \
  S3_ENDPOINT=https://your-s3-endpoint \
  S3_REGION=us-east-1 \
  cargo test -p diskgraph-index --test cloud_ingest -- --ignored

  # GCS (must be https; provide service account or ADC)
  RUN_CLOUD_TESTS=1 \
  GCS_BUCKET=your-bucket \
  GCS_ENDPOINT=https://storage.googleapis.com \
  cargo test -p diskgraph-index --test cloud_ingest -- --ignored

  # ADLS/Azurite
  RUN_CLOUD_TESTS=1 \
  AZURE_ACCOUNT=devstoreaccount1 \
  AZURE_CONTAINER=test-container \
  AZURE_ENDPOINT=http://127.0.0.1:10000/devstoreaccount1 \
  AZURE_USE_EMULATOR=1 \
  cargo test -p diskgraph-index --test cloud_ingest -- --ignored
  ```

Notes:
- Tests require `RUN_CLOUD_TESTS=1` and the relevant `S3_*`, `GCS_*`, or `AZURE_*` vars. With `RUN_CLOUD_TESTS=1`, missing vars panic instead of silently skipping.
- GCS client requires `https`; for emulators we inject a minimal service account and use the JSON API for HTTP endpoints.
- For real clouds, prefer role-based auth: AWS IAM (omit `S3_ACCESS_KEY`/`S3_SECRET_KEY`), GCP ADC/workload identity (omit `GCS_SERVICE_ACCOUNT_*`), Azure MSI/client creds (omit `AZURE_ACCESS_KEY`/`AZURE_SAS_KEY`). Emulators still need the fake keys provided by `run_cloud_tests.sh`.
- Data sets include two relations (with and without `dest_sort_key`), enough nodes to span multiple offset chunks, and row groups to exercise multi-group Parquet reads.

## Expectations and assumptions
- Input Parquet shards are already sorted:
  - Nodes by `id` ascending.
  - Edges by `src_id` ascending and `dest_sort_key` descending when present.
- Sharding alignment: edges for shard N reference only source IDs in shard N.
- Offsets are always byte-based; `edge_blob_word_size` kept as 1 for future compatibility.
- Relation order in metadata is stable and shared between build and read.
