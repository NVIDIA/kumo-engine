# DiskGraph Ingest

> **Related docs**: Config format details in [`diskgraph-ingest/README.md`](../diskgraph-ingest/README.md). Rowpack encoding in [`diskgraph-rowpack/README.md`](../diskgraph-rowpack/README.md). Index on-disk layout in [`diskgraph-index/README.md`](../diskgraph-index/README.md). PtrHash algorithm in [`diskgraph-ptrhash/README.md`](../diskgraph-ptrhash/README.md). Dataset specification in [`specs/README.md`](../specs/README.md).

## Overview

Ingest converts raw data (parquet, CSV, Snowflake, Databricks) into the DiskGraph index — a memory-mapped, sharded graph representation optimized for O(1) node lookup and sequential edge traversal.

The pipeline is streaming and single-pass over the data. No intermediate materialization to disk except final output and temporary shard files. Everything is parallel by default.

```
Raw Data (parquet/CSV/cloud)
  → Phase 0: Context (schema inference, rowpack config)
  → Phase 1: Encode (node data → rowpack, FK extraction, sharding)
  → Phase 2: PK Map (MPHF construction for PK → node ID)
  → Phase 3: Edges Map (FK resolution → sorted adjacency lists)
  → Phase 4: Index Build (compress, layout, metadata)
  → Phase 5: Finalize (cleanup temp, write stats)
```

## Config Format

**File**: `diskgraph-ingest/src/graph_config.rs`. Full JSON schema with examples in [`diskgraph-ingest/README.md`](../diskgraph-ingest/README.md).

The ingest config is a JSON file defining the graph schema:

```rust
GraphConfig {
    dataset_name: String,           // Graph identifier
    tables: Vec<TableSpec>,         // Node and edge table definitions
    num_shards: Option<usize>,      // Partition count (default: 1)
    rowpack: Option<RowPackConfig>, // Global encoding parameters
    edge_format: Option<EdgeStorageFormat>, // Binary (default) or Parquet
    coercion_mode: CoercionMode,    // FK→PK type compatibility (Lossless/Strict)
}
```

### TableSpec

Each table defines its source, schema, keys, and relationships:

```rust
TableSpec {
    name: String,                   // Logical table name
    source: DataSource,             // Parquet paths, Snowflake, Databricks, etc.
    columns: Vec<ColumnSpec>,       // Column names, types, encoding hints
    primary_key: Option<String>,    // PK column (enables MPHF construction)
    create_time: Option<String>,    // Entity creation timestamp (for temporal filtering)
    end_time: Option<String>,       // Entity expiry timestamp
    fkeys: Vec<FKeySpec>,           // Foreign key relationships
    edge_table: Option<EdgeTableSpec>, // If this is an edge table (src_fk, dst_fk, sort_key)
}
```

### Key Design Choices

**`num_shards`**: Controls parallelism and memory pressure. Each shard gets its own node data file, PK map, and edge files. More shards = more parallelism during index build and query, but more files and higher overhead for small tables. A table with 500 nodes across 3 shards produces empty shards with sentinel values — this caused binary search failures (PR #134).

**`coercion_mode`**: Determines how FK→PK type mismatches are handled. `Lossless` rejects incompatible types. `Strict` allows safe coercions (int32→int64). The coercion matrix (Same/Safe/Conditional/Lossy/Incompatible) is documented in Issue #30.

**`edge_table` vs node table with `fkeys`**: Edge tables have explicit src/dst FKs and optional sort keys (timestamp, weight). Node tables with `fkeys` define implicit edges (the FK column points to another table's PK). The config converter detects which is which — auto-detection based on column types was disabled (PR #108) because it misclassified tables like HnM transactions.

### DataSource Types

```rust
DataSource::Parquet { paths }     // Local or cloud (s3://, gs://, az://)
DataSource::Csv { paths }
DataSource::Snowflake { connection_id, table_or_view }
DataSource::Databricks { connection_id, catalog, schema, table }
DataSource::BigQuery { connection_id, ... }
```

### ColumnSpec

```rust
ColumnSpec {
    name: String,
    ingested_dtype: IngestedType,      // How data is stored in index
    source_dtype: Option<String>,      // Original Arrow type
    timestamp_format: Option<String>,  // For string timestamp parsing
    timestamp_unit: Option<TimestampUnit>,
    semantic_type: Option<SemanticType>, // ML encoding hint (ID, categorical, numerical, text, etc.)
}
```

**IngestedType** covers: Int8-64, UInt8-64, Float16/32/64, Bool, String, Bytes, Uuid, Uuid256, Ip, Date, Timestamp, Decimal128/256, Float16/32/64Array.

## Phase 0: Context (Schema Inference)

**Function**: `IngestContext::build_node_contexts()`

Samples up to 100K rows per table to build encoding configuration:

1. **RowPackContext construction**: Detects string dictionaries (coverage threshold, BPE encoding), numeric ranges for varint optimization, float array formats
2. **Text sampling**: For col stats text features (avg_tokens, glove_overlap), samples from up to 3 parquet files to handle null-heavy first files
3. **Timestamp format detection**: Tries ISO8601, Unix epoch (seconds/millis/micros/nanos), integer-encoded dates (YYYYMMDD, YYYYMMDDHHMMSS), custom patterns. Downloads up to 15 files per table (capped after PR #104 fixed unbounded downloads)

**Parallelism**: Per-table tasks via MPSC channel. Main thread collects results.

**What to watch when changing**: The context phase determines encoding parameters that affect every subsequent phase. A wrong type inference here propagates silently until training produces wrong features. Test on datasets with diverse column types — especially string timestamps, integer-encoded dates, and JSON-encoded float arrays.

## Phase 1: Encode (Node Data)

**Function**: `IngestContext::build_temp_shards()`

The encode phase reads raw data and produces:
- Rowpack-encoded node data blobs (one per shard per table)
- Primary key values (for MPHF construction in Phase 2)
- Foreign key values (for edge resolution in Phase 3)

### Rowpack Format

Node data is encoded in rowpack — a row-oriented format optimized for O(1) random access to individual nodes. See [`diskgraph-rowpack/README.md`](../diskgraph-rowpack/README.md) for the full format specification.

**Why rowpack over Parquet/Arrow?** Graph traversal accesses individual nodes by ID. Parquet is columnar (optimized for full-column scans). Arrow RecordBatches carry per-batch schema overhead. Rowpack encodes each node as a standalone binary blob using shared per-table dictionaries — fast to decode a single row, dense storage (1-bit bools, zigzag varint integers, BPE-compressed strings).

The RowPackContext (dictionaries, alphabet maps, encoding strategies) is built once during Phase 0 from a sample and shared read-only during encode.

### Sharding

Nodes are distributed across shards by hashing the primary key: `gxhash(pk_bytes) % num_shards`. Consistent hashing ensures the same PK always maps to the same shard.

### Batching Architecture

The encode phase uses a producer-consumer pipeline:

```
Loader pool → resident source (local file or complete cloud object in RAM)
  → ordered node-file tasks or edge row-group partitions → shared encode pool
    → NodeBatcher: buffers rows by shard, flushes at buffer limit
    → SyncSender<NodeWriteMessage>
      → NodeWritePool: N consumer threads write to shard files
```

One reader abstraction covers local files and complete S3, GCS, or ADLS
downloads. A bounded loader pool feeds work to the full encode pool. Node files
remain single ordered tasks so primary-key deduplication and PK-less row IDs are
stable within each source file; edge files can fan row groups out in parallel.
Cloud buffers default to at most 25% of the tighter of host RAM and the active
cgroup limit.

`NodeBatcher` maintains per-shard buffers for node data, primary keys, and foreign keys. It flushes when any shard buffer exceeds `EDGE_BATCH_TARGET_BYTES / num_shards`. This keeps memory bounded regardless of input size.

Sort key data (timestamps, weights for edge ordering) is bundled into `NodeWriteBatch` — not sent through a separate pipeline. This was a deliberate design choice after PR #51 fixed a sort key misalignment bug where separate batcher pipelines flushed at different rates, causing row N in the fkey temp file to refer to a different source row than row N in the sort key temp file.

### Col Stats Observer

If `--col-stats` is enabled, the encode phase attaches a `ColumnValueObserver` to the encoder. See [col_stats.md](../diskgraph-ingest/col_stats.md) for details. The observer adds 3-22% wall time overhead depending on dataset characteristics.

### Duplicate PK Handling

When multiple rows share the same PK value, the first encoded row wins and later
duplicates are dropped. Duplicate counts are tracked per table and reported in
ingest stats, with up to 10 sample values for debugging.

**What to watch when changing**: The encode phase's memory usage is bounded by batch buffer sizes, not by table size. If you add per-row state that grows with the table, you'll break the streaming property. Keep per-row state in the observer pattern (see col stats) or accumulate in fixed-size structures.

### Timestamp Validity and Row Dropping

Rows with invalid `create_time` (time_col) values are dropped during encode to match Spark's behavior. Two checks run in sequence:

**Phase 2 null check (Arrow bitmap)**: Before encode, the `process_node_batch!` macro checks the Arrow validity bitmap for the time_col column. Arrow-null timestamps cause the row to be skipped via `continue` — no encode, no PK/FK/sort key entries. This is the original null-timestamp drop path.

**Post-encode tag check**: After `encode_row_observed`, `write_node_row_from_access` inspects the time_col's 2-bit tag in the encoded rowpack data. If the tag is 0 (null), the encoded buffers are truncated and the function returns early — no PK, FK, or sort key entries are created. This catches:
- Out-of-range Date32 values: `get_timestamp()` in `access.rs` returns `Err` for days outside `[-106752, 24855]` (~1677 to ~2038). The lower bound is pandas `Timestamp.min`; the upper bound prevents i32 sort key overflow (epoch seconds exceed `i32::MAX` after 2038-01-19).
- String timestamp parse failures: unparseable strings also return `Err`, producing a null tag.

**Edge tables**: The `process_edge_batch!` macro additionally calls `access.get_timestamp()` on the time_col after the Arrow null check. If it returns `Err` or `Ok(None)`, the edge row is dropped via `continue` before reaching `write_edge_row_from_access`.

**Why two checks?** The Arrow bitmap only reflects parquet-level nulls. Values that are non-null in parquet but semantically invalid (out-of-range dates, malformed strings) pass the bitmap check but fail during encode. The post-encode tag check catches these. For non-time-col timestamp columns, the invalid value is encoded as a null feature (tag=0) but the row is kept — only the time_col triggers a row drop.

**Reporting**: Dropped rows are counted in `node_data_dropped` (node tables) and `edge_dropped` (edge tables), aggregated into `dropped_rows_counts` in `ingest_stats.json`. Parse failures are recorded in `col_stats.json` via `observe_parse_failure`.

## Phase 2: PK Map (MPHF Construction)

**Function**: `IngestContext::build_sharded_pk_maps()`

Builds a minimal perfect hash function (MPHF) for each table's primary key. This enables O(1) PK → node ID lookup during edge resolution (Phase 3) and at query time (TT gen, sampling).

### PtrHash Algorithm

DiskGraph uses PtrHash (partitioned bucketing MPHF). See [`diskgraph-ptrhash/README.md`](../diskgraph-ptrhash/README.md) for algorithm details and local fork changes.

Key properties:
- **2.4 bits per key** storage overhead (compared to ~8 bytes per key for a standard HashMap)
- O(1) lookup with zero false positives (perfect hash — every key maps to a unique slot)
- Memory-mapped after construction — immutable, lock-free concurrent reads

### Key Types

All primary keys are normalized to one of 5 key types:

| KeyType | Source DTypes | Storage |
|---------|-------------|---------|
| Int64 | int/uint/float/decimal → u64 | 8 bytes |
| VarBytes | string/bytes/decimal256 | variable |
| Uuid | 16-byte UUID | 16 bytes |
| Uuid256 | 32-byte UUID | 32 bytes |
| Ip | IPv4/IPv6 | 16 bytes |

### File Format

```
pk_hash/{shard}/mphf.bin    — MPHF binary (epserde serialized)
pk_hash/{shard}/data.bin    — PK data with header:
  Offset 0:  magic "DGPKDAT\0" (8 bytes)
  Offset 8:  version 3 (4 bytes)
  Offset 12: key_type tag (1 byte)
  Offset 13: key_length (4 bytes)
  Offset 17: entry_count (8 bytes)
  Offset 25: [u32 node_id_local] × entry_count
```

**What to watch when changing**: The MPHF is the foundation of all node lookups. `contains_pk_value()` (used for orphaned FK filtering in TT gen) and `node_id_from_pk()` (used for edge resolution) both depend on it. If you change the key type mapping or hash function, all downstream lookups break silently — nodes resolve to wrong IDs. Always verify with a round-trip test: insert PK, look up PK, verify node ID.

## Phase 3: Edges Map

**Function**: `IngestContext::build_edges_mapped()`

Reads FK values from Phase 1 temp files, resolves each FK to a destination node ID via the PK map from Phase 2, and builds sorted edge lists per source node.

### Edge Resolution

For each FK value:
1. Look up destination table's PK map: `dst_table.node_id_from_pk(fk_value)`
2. If found: create edge entry `(src_node_id, dst_node_id, sort_key)`
3. If not found: FK points to non-existent PK (orphaned FK). The edge is dropped. This is expected for dirty data.

### Edge Binary Format

See [`diskgraph-index/README.md`](../diskgraph-index/README.md) for the full on-disk edge layout. Summary:

Edges are bit-packed: `[dst_node_id (dst_width bits) | sort_key (sort_width bits)]` per entry. `dst_width` is derived from `max(dst_node_id)`. Total bits per edge = dst_width + sort_width. Edges for a single node are stored contiguously, grouped by relation type.

### Sort Key Encoding

Sort keys enable temporal filtering during graph traversal. A sort key combines:
- **Timestamp**: Delta-encoded from an anchor value, quantized to Int8/Int16/Int32
- **Weight**: Raw value as Byte/UInt16/UInt32/Float16/Float32

The sort key is stored inline with the destination node ID in the edge binary. This is cache-friendly — when traversing edges for a node and filtering by time window, the timestamp is immediately available without a separate lookup.

### Edge Layout Per Node

Edges for a single source node are stored contiguously, grouped by relation (edge type). The `RelationLayout` struct precomputes bit widths and masks for decoding:

```rust
RelationLayout {
    dst_mask: u128,       // Bitmask for extracting dst_node_id
    sort_mask: u128,      // Bitmask for extracting sort_key
    entry_bytes: usize,   // Bytes per edge entry
}
```

**What to watch when changing**: Edge format changes affect sampling, TT gen, and feature serving — every component that traverses edges. The bit-packing is tightly coupled to the metadata (dst_width, sort_width). If you change the encoding, update `RelationLayout`, `iter_edge()` in `ttgen_traversal.rs`, and the sampler's edge decoder. Test on datasets with both small tables (few bits for dst) and large tables (many bits).

## Phase 4: Index Build

**Function**: `IngestContext::build_index()`

Reads temp files from Phases 1-3 and produces the final index:
- Compresses node data with LZ4 or Zstd (configurable via `NodeDataCompression`)
- Lays out edges per-node in the final binary format
- Writes `index_metadata.json` with complete schema

Uses `GraphIndexBuilder` from `diskgraph-index`. Memory budget defaults to 70% of system RAM (`DEFAULT_INDEX_MEM_BUDGET_FRAC`). If the budget is exceeded, the builder processes shards sequentially.

## Phase 5: Finalize

Cleanup and stats output:

1. **cleanup_temp()**: Parallel deletion of `_tmp/` directory. Uses `std::thread::scope` for scoped parallelism — not background fire-and-forget, which caused race conditions with Temporal retries (PR #123).

2. **CleanupGuard**: RAII guard that removes the output directory on failure (drop). On success, `disarm()` prevents cleanup. Ensures no partial index survives a failed ingest.

3. **Stats output**: `ingest_stats.json` (rows, dropped PKs, timing), `edge_stats.json` (Kumo-format edge statistics), optionally `col_stats.json` (see [col_stats.md](../diskgraph-ingest/col_stats.md)).

## Config Converter

**File**: `diskgraph-pybind/python/diskgraph/config_converter.py` (Python wrapper) + `diskgraph-ingest/src/config_converter/` (Rust implementation)

Converts Kumo's `GraphConfig` (SDK-level graph definition) to DiskGraph's ingest config:

1. **Table type detection**: Node vs edge tables. Auto-detection disabled (PR #108) — now relies on explicit schema.
2. **FK resolution**: Maps FK columns to PK columns in destination tables. Detects type coercions needed.
3. **Timestamp format detection**: Samples values, tries multiple parsers, selects format.
4. **num_shards estimation**: Based on total data size and target shard size.
5. **Source path expansion**: Resolves parquet directories to individual file paths.

## Output Directory Structure

See [`specs/README.md`](../specs/README.md) for the formal dataset specification including `index_metadata.json` schema.

```
{output}/
  index_metadata.json              # Schema, shard metadata, edge relations
  ingest_stats.json                # Ingest statistics
  col_stats.json                   # Column statistics (if --col-stats)
  edge_stats.json                  # Kumo-format edge stats
  kumo_col_stats/                  # Per-table col stats
    col_stats_full/{table}.json
    col_stats_min/{table}.json
  nodes/{table}/
    node_data.ctx                  # RowPackContext (encoding schema)
    shard-{id}/
      node_data.bin                # Compressed rowpack data
      edges.bin                    # Edge adjacency lists
      pk_data.bin                  # PK map data
      pk_hash.bin                  # MPHF binary
```

## Developing: Changing Ingest

### Adding a New Column Type

1. Add variant to `IngestedType` in `diskgraph-core/src/metadata.rs`
2. Add encoding in rowpack: `encode_value()` and `decode_value()` in `diskgraph-rowpack`
3. Add Arrow→IngestedType mapping in the config converter
4. Add `ColumnValueObserver` callback if col stats needs to observe this type
5. Test round-trip: encode → decode → compare against original Arrow data

### Adding a New Data Source

1. Add variant to `DataSource` enum in `graph_config.rs`
2. Implement the reader (return `RecordBatch` iterator) in `diskgraph-ingest/src/readers/`
3. Add CLI flag and parsing
4. The rest of the pipeline (encode, PK map, edges) is source-agnostic

### Changing Sharding

Sharding affects every phase and every downstream consumer. The shard hash function (`gxhash(pk_bytes) % num_shards`) must be deterministic — changing it invalidates all existing indexes. If you need a different distribution, add a new sharding strategy rather than modifying the existing one.

Empty shards are legal but must be handled everywhere: binary search in `find_shard`, PK map construction, edge iteration. The sentinel values (`min_id=1, max_id=0`) for empty shards have caused bugs before (PR #134).

### Changing the Edge Format

Edge format is the most sensitive part of the index. It's read by:
- `diskgraph-index/src/layout.rs` (RelationLayout, edge decoding)
- `diskgraph-train-table/src/ttgen_traversal.rs` (TT gen edge iteration)
- `diskgraph-pybind/src/sampler.rs` (sampling during training)

All three must agree on bit widths, masks, and sort key encoding. Change all three together. Add integration tests that verify edge traversal returns correct neighbors and sort keys.

### Memory Considerations

The encode phase is streaming — memory usage is bounded by batch buffer sizes, not table size. If you add per-table state that grows with data, you'll break this property. The col stats OOM (PR #61) was caused by exactly this: `HashMap::clear()` retained bucket arrays in an unbounded channel, accumulating 100+ GB.

Keep per-row state small (fits L1/L2). Keep per-table state bounded. Keep per-ingest state proportional to schema size, not data size.

## Running Ingest

### CLI

```bash
cargo run --release --bin ingest -- \
  --source config.json \
  --output /path/to/index \
  --col-stats \
  --cat-count-strategy hybrid
```

### Key Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--source` | required | Config JSON or directory |
| `--output` | required | Index output path |
| `--dest` | local | Storage backend (local, s3, gcs, adls) |
| `--parallel` | true | Enable parallel processing |
| `--col-stats` | false | Compute column statistics |
| `--cat-count-strategy` | none | Category counting: none, hybrid, graduation |
| `--index-mem-budget-bytes` | 70% RAM | Memory limit for index build |
| `--mem-profile` | false | Track peak memory per phase |

### Library API

```rust
pub fn ingest_with_stats(
    graph_config_path: &Path,
    output: &str,
    dest: Dest,
    options: IngestOptions,
) -> Result<(DatasetMetadata, IngestStats)>
```

## Performance Constants

| Constant | Value | Rationale |
|----------|-------|-----------|
| NODE_BATCH_SIZE | 100,000 | Parquet read batch size |
| SAMPLE_ROW_LIMIT | 100,000 | Context phase sample cap |
| Worker threads | ~105% CPU count | Slightly over-subscribe for I/O overlap |
| DEFAULT_INDEX_MEM_BUDGET_FRAC | 0.7 | 70% system RAM for index build |
| GXHASH_SHARD_SEED | 0x6a09e667f3bcc909 | Deterministic shard assignment |
| MAX_VAR_KEY_LEN | 65,535 | Max PK/FK key size in bytes |
