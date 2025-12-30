# DiskGraph

DiskGraph is a Rust-based system for ingesting large typed, bidirectional graphs into an efficient on-disk layout and serving low-latency neighborhood sampling queries. It includes ingestion tooling, an index builder/reader, and TUI explorers for local and cloud datasets.

## Workspace Layout

- `diskgraph-core/` — shared types and traits.
  - `dataset.rs` — dataset lifecycle state, registry trait, in-memory registry.
  - `metadata.rs` — `index_metadata.json` schema and on-disk size estimation.
  - `sampling.rs` — sampling plan definitions, request/response shapes, sampler trait.
  - `storage.rs` — storage abstraction, local storage stub, compression interface (`storage/compression.rs`).
  - `errors.rs` — common error type.
- `diskgraph-datagenerator/` — synthetic dataset generator for testing.
  - Produces dataset directories with `index_metadata.json` and sharded edge tables; node shard ranges are recorded in metadata.
  - Includes a massive-scale streaming generator and helper scripts to push directly to S3/GCS/ADLS or the local filesystem for large benchmarks.
  - Guarantees: node shard ranges recorded in metadata; edge shards sorted by `(src_id, dest_sort_key desc when present)`; shard alignment (edge shard N only references src IDs from node shard N of the source type); per-shard id ranges are recorded in metadata for routing; optional fields exercised across scenarios.
- `diskgraph-index/` — shard-level index builder/reader.
  - Ingests node shard ranges (from metadata) and aligned edge shards into compact on-disk files (`offsets.bin`, `edges.bin`) with delta-encoded offsets and mmap-friendly edge blobs.
  - Includes a mmap-based shard reader and a TUI explorer that streams Parquet shards from local/cloud storage and caches the built shard locally for inspection.
- `diskgraph-ingest/` — ingestion pipeline from source CSV/Parquet + `source_metadata.json` into index-ready Parquet + `index_metadata.json`.
  - Builds primary key maps, converts node/edge tables, writes node payloads compressed with rowpack, and emits per-table rowpack context files.
  - Includes scripts for local and cloud ingestion plus performance profiling.
- `diskgraph-rowpack/` — row-oriented node payload compression codec.
  - Builds per-table contexts from samples and encodes each node row independently for fast decode in the index/explorer.

## What’s Implemented

- End-to-end ingestion from source CSV/Parquet into `index_metadata.json` + node/edge Parquet outputs.
- Rowpack compression for node payloads (context + per-row encoding).
- Shard-level index builder/reader and explorer with local/cloud support.

## Building and Running

```bash
cargo build --workspace

# Generate synthetic datasets (coverage of all scenarios)
cargo run -p diskgraph-datagenerator -- --output /tmp/dg-data --scenario coverage --nodes 10000 --edges 50000 --node-shards 4
```

## Quick Start (Workflow)

1) Generate source tables + `source_metadata.json` (see `diskgraph-datagenerator` scripts).  
2) Run `diskgraph-ingest` to produce `index_metadata.json` + node/edge Parquet outputs.  
3) Run `diskgraph-index` explorer to build and query the on-disk index locally or from cloud.  
