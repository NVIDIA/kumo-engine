# diskgraph-core

Shared data types and service contracts for the DiskGraph pipeline. This crate defines the logical schemas and traits that connect the ingestion pipeline, the Arrow Flight gateway, and the native sampling engine.

## What lives here

- **Batch Sampling API** (`sampling.rs`)
  - `BatchSamplingRequest`: The root entry point for graph traversal.
  - `SamplingPlan` & `LayerSpec`: Multi-hop heterogeneous fanout definitions.
  - `SamplingFlags`: Configuration for replacement, disjointness, and adaptive sampling.
  - `SamplingMode`: Logic for Uniform, Temporal, and Weighted neighbor selection.
  - `AbsentKeyPolicy`: Policy for handling relations missing weight or time keys.
  - `Weighting`: Definitions for temporal decay and custom scoring expressions.
- **Feature Fetching API** (`features.rs`)
  - `BatchFeatureRequest`: Request structure for retrieving node attributes.
- **Index Metadata** (`metadata.rs`)
  - `DatasetMetadata`, `NodeTypeMetadata`, `EdgeRelationMetadata`.
  - SortKey packing definitions (`timestamp` MSB, `weight` LSB).
  - `estimate_ondisk_bytes()` heuristic used for admission control.
- **Storage Contracts** (`storage.rs`)
  - `GraphStore` trait: Interface for topology management and neighbor sampling.
  - `FeatureStore` trait: Interface for retrieving node features.
  - `LocalGraphEngine`: Skeleton implementation combining both store capabilities.
  - `storage::compression` defines a `CompressionCodec` interface and a placeholder implementation.
- **Dataset Lifecycle** (`dataset.rs`)
  - `DatasetId`, `DatasetState`, `DatasetHandle`, `DatasetReservation`.
  - `DatasetRegistry` trait for managing loaded graphs.
- **Errors** (`errors.rs`)
  - `DiskgraphError` shared across the service and CLI layers.

## Design Principles

1.  **Columnar First**: All data-heavy APIs return `arrow_array::RecordBatch`. This enables zero-copy integration with PyTorch/GPU via DLPack.
2.  **Batch Oriented**: Replaces single-node logical requests with vectorized batch requests to maximize server-side throughput.
3.  **Kumo Alignment**: Strictly separates graph topology (`GraphStore`) from node features (`FeatureStore`), matching Kumo's architectural concepts.
4.  **Protocol Agnostic**: Definitions are pure Rust structs with `serde` traits, suitable for binary serialization (e.g., `bincode`) in Arrow Flight `cmd` fields.

## Legacy Notes (Ingestion & Metadata)

- `DatasetMetadata` mirrors the `index_metadata.json` produced by ingest. The field names and shapes are kept stable so the service can read them directly.
- `estimate_ondisk_bytes()` intentionally overestimates on-disk usage to keep admission control safe.
- `LocalGraphEngine` stubs (like ingest/reserve) are placeholders for now; the primary implementation path is via the new sampling/feature traits.
- `PlaceholderCompressionCodec` returns `NotImplemented` so callers can differentiate missing algorithms from runtime failures.
