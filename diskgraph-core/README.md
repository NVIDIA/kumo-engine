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
  - `GraphStorage` trait: The unified interface for sampling and feature retrieval, returning Arrow `RecordBatch` streams.
  - `StorageConfig`: Basic path and capacity configuration for storage backends.
  - `LocalGraphStorage` skeleton that tracks datasets but leaves layout/compression to downstream crates.
  - `storage::compression` defines a `CompressionCodec` interface and a placeholder implementation.
- **Dataset Lifecycle** (`dataset.rs`)
  - `DatasetId`, `DatasetState`, `DatasetHandle`, `DatasetReservation`.
  - `DatasetRegistry` trait for managing loaded graphs.
- **Errors** (`errors.rs`)
  - `DiskgraphError` shared across the service and CLI layers.

## Design Principles

1.  **Columnar First**: All data-heavy APIs return `arrow_array::RecordBatch`. This enables zero-copy integration with PyTorch/GPU via DLPack.
2.  **Batch Oriented**: Replaces single-node logical requests with vectorized batch requests to maximize server-side throughput.
3.  **Protocol Agnostic**: Definitions are pure Rust structs with `serde` traits, suitable for binary serialization (e.g., `bincode`) in Arrow Flight `cmd` fields.
4.  **Type-Local Re-indexing**: The sampler is expected to perform Global-to-Local ID mapping per node type, ensuring the client receives tensors ready for GNN adjacency matrices.

## Example (constructing a Batch Request)

```rust
use diskgraph_core::sampling::{BatchSamplingRequest, SamplingPlan, LayerSpec, FanoutSpec, SamplingMode, SamplingFlags};
use uuid::Uuid;

let request = BatchSamplingRequest {
    dataset_id: Uuid::new_v4(),
    seed_nodes: vec![1, 2, 3],
    seed_timestamps: None,
    plan: SamplingPlan {
        layers: vec![
            LayerSpec {
                fanouts: [
                    ("User__Click__Item".to_string(), FanoutSpec {
                        fanout: 10,
                        mode: SamplingMode::TemporalMostRecent,
                    }),
                ].into(),
            },
        ],
    },
    flags: SamplingFlags {
        replace: false,
        directed: true,
        disjoint: true,
        return_edge_id: false,
        adaptive: false,
    },
};
```

## Legacy Notes (Ingestion & Metadata)

- `DatasetMetadata` mirrors the `index_metadata.json` produced by ingest. The field names and shapes are kept stable so the service can read them directly.
- `estimate_ondisk_bytes()` intentionally overestimates on-disk usage to keep admission control safe.
- `LocalGraphStorage` stubs (like ingest/reserve) are placeholders for now; the primary implementation path is via the new sampling/feature traits.
- `PlaceholderCompressionCodec` returns `NotImplemented` so callers can differentiate missing algorithms from runtime failures.