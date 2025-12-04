# DiskGraph

DiskGraph is a Rust-based system for ingesting large typed, bidirectional graphs into an efficient on-disk layout and serving low-latency neighborhood sampling queries. The project ships as both a library and an HTTP service; Python bindings will wrap the same core semantics.

## Workspace Layout

- `diskgraph-core/` — shared types and traits.
  - `dataset.rs` — dataset lifecycle state, registry trait, in-memory registry.
  - `metadata.rs` — metadata.json schema and on-disk size estimation.
  - `sampling.rs` — sampling plan definitions, request/response shapes, sampler trait.
  - `storage.rs` — storage abstraction, local storage stub, compression interface (`storage/compression.rs`).
  - `errors.rs` — common error type.
- `diskgraph-ingest/` — ingestion coordinator.
  - Loads metadata.json, runs disk admission control, registers datasets, and invokes storage ingestion (materialization is stubbed until layout/compression are chosen).
  - Emits progress callbacks via `IngestObserver`.
- `diskgraph-service/` — axum HTTP service.
  - Endpoints: health, list datasets, load dataset, unload dataset, sample.
  - Wires together `IngestCoordinator`, registry, and storage.

## What’s Implemented vs Stubbed

- Implemented: metadata parsing, size estimation heuristic, dataset registry/control-plane, HTTP wiring, disk reservation checks (logical), sampling API surface.
- Stubbed: compression/decompression, Parquet ingestion/materialization, on-disk layout, actual sampling logic; placeholder storage returns empty neighborhoods and `NotImplemented` for compression.

## Building and Running

```bash
cargo build --workspace

# Run the HTTP service (default port 8080)
DISKGRAPH_DATA_PATH=/path/to/data cargo run -p diskgraph-service
```

## Quick Start (Workflow)

1) Prepare a dataset directory with `metadata.json` and Parquet shards (see `metadata.rs` for expected fields).  
2) Start the service and POST `/datasets/load` with `{ "dataset_root": "/abs/path" }`.  
3) Send sampling requests to `/datasets/{id}/sample` with a `SamplingRequest` payload (see `sampling.rs`).  

Until the storage layer is implemented, sampling responses will be empty and ingestion does not write on-disk structures; use this scaffold to iterate on layout and algorithms.
