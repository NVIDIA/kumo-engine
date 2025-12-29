# DiskGraph

DiskGraph is a Rust-based system for ingesting large typed, bidirectional graphs into an efficient on-disk layout and serving low-latency neighborhood sampling queries. The project ships as both a library and an HTTP service; Python bindings will wrap the same core semantics.

## Workspace Layout

- `diskgraph-core/` — shared types and traits.
  - `dataset.rs` — dataset lifecycle state, registry trait, in-memory registry.
  - `metadata.rs` — index_metadata.json schema and on-disk size estimation.
  - `sampling.rs` — sampling plan definitions, request/response shapes, sampler trait.
  - `storage.rs` — storage abstraction, local storage stub, compression interface (`storage/compression.rs`).
  - `errors.rs` — common error type.
- `diskgraph-datagenerator/` — synthetic dataset generator for testing.
  - Produces dataset directories with `index_metadata.json` and sharded edge tables; node shard ranges are recorded in metadata.
  - Includes a massive-scale streaming generator and helper scripts to push directly to S3/GCS/ADLS or the local filesystem for large benchmarks.
  - Guarantees: node shard ranges recorded in metadata; edge shards sorted by `(src_id, dest_sort_key desc when present)`; shard alignment (edge shard N only references src IDs from node shard N of the source type); per-shard id ranges are recorded in metadata for routing; optional fields exercised across scenarios.
- `diskgraph-index/` — shard-level index builder/reader.
  - Will ingest node shard ranges (from metadata) and aligned edge shards into compact on-disk files (`offsets.bin`, `edges.bin`) with delta-encoded offsets and mmap-friendly edge blobs.
  - Includes a mmap-based shard reader that returns zero-copy edge list views (encoding is stubbed until compression/packing is finalized) and a TUI explorer that streams Parquet shards from local/cloud storage and caches the built shard locally for inspection.
- `diskgraph-ingest/` — ingestion coordinator.
  - Loads index_metadata.json, runs disk admission control, registers datasets, and invokes storage ingestion (materialization is stubbed until layout/compression are chosen).
  - Emits progress callbacks via `IngestObserver`.
  - Endpoints: health, list datasets, load dataset, unload dataset, sample.
  - Wires together `IngestCoordinator`, registry, and storage.

## What’s Implemented vs Stubbed

- Implemented: metadata parsing, size estimation heuristic, dataset registry/control-plane, HTTP wiring, disk reservation checks (logical), sampling API surface.
- Stubbed: compression/decompression, Parquet ingestion/materialization, on-disk layout, actual sampling logic; placeholder storage returns empty neighborhoods and `NotImplemented` for compression.

## Building and Running

```bash
cargo build --workspace

# Run the HTTP service (default port 8080)

# Generate synthetic datasets (coverage of all scenarios)
cargo run -p diskgraph-datagenerator -- --output /tmp/dg-data --scenario coverage --nodes 10000 --edges 50000 --node-shards 4
```

## Quick Start (Workflow)

1) Prepare a dataset directory with `index_metadata.json` and Parquet shards (see `metadata.rs` for expected fields).  
2) Start the service and POST `/datasets/load` with `{ "dataset_root": "/abs/path" }`.  
3) Send sampling requests to `/datasets/{id}/sample` with a `SamplingRequest` payload (see `sampling.rs`).  

Until the storage layer is implemented, sampling responses will be empty and ingestion does not write on-disk structures; use this scaffold to iterate on layout and algorithms.
