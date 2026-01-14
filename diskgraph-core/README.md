# diskgraph-core

Shared data types and service contracts for the DiskGraph pipeline. This crate keeps the ingestion/runtime/sampling layers consistent by defining metadata structs, dataset lifecycle state, sampling plans, and storage interfaces. It deliberately avoids heavy implementations of compression or on-disk layout.

## What lives here

- **Dataset lifecycle** (`dataset.rs`)
  - `DatasetId`, `DatasetState`, `DatasetHandle`, `DatasetReservation`
  - `DatasetRegistry` trait plus an in-memory implementation for tests and bootstrapping.
- **Index metadata** (`metadata.rs`)
  - `DatasetMetadata`, `NodeTypeMetadata`, `EdgeRelationMetadata`, shard metadata.
  - Optional dest-sort-key metadata and encoding details.
  - `estimate_ondisk_bytes()` heuristic used for admission control.
- **Sampling contracts** (`sampling.rs`)
  - `SamplingPlan`, `HopPlan`, `DestSortKeyMode`, `AbsentKeyPolicy`, `Weighting`.
  - `SamplingRequest`/`SampledNeighborhood` payloads.
  - `Sampler` trait and a `RoutingSampler` that forwards to storage.
- **Storage contracts** (`storage.rs`)
  - `StorageConfig`, `GraphStorage` trait.
  - `LocalGraphStorage` skeleton that tracks datasets but leaves layout/compression to downstream crates.
  - `storage::compression` defines a `CompressionCodec` interface and a placeholder implementation.
- **Errors** (`errors.rs`)
  - `DiskgraphError` shared across the service and CLI layers.

## Usage notes

- `DatasetMetadata` mirrors the `index_metadata.json` produced by ingest. The field names and shapes are kept stable so the service can read them directly.
- `estimate_ondisk_bytes()` intentionally overestimates on-disk usage to keep admission control safe.
- `LocalGraphStorage` does not yet implement disk accounting or on-disk removal; those are stubs that return placeholders.
- `PlaceholderCompressionCodec` returns `NotImplemented` so callers can differentiate missing algorithms from runtime failures.

## Example (wiring a sampler)

```rust
use diskgraph_core::sampling::{RoutingSampler, SamplingRequest, SamplingPlan, HopPlan, AbsentKeyPolicy};
use diskgraph_core::storage::{LocalGraphStorage, StorageConfig};

let storage = LocalGraphStorage::new(StorageConfig::new("/mnt/localssd/diskgraph"));
let sampler = RoutingSampler::new(storage);

let request = SamplingRequest {
    dataset_id: uuid::Uuid::new_v4(),
    root_id: 42,
    anchor_time: None,
    plan: SamplingPlan {
        root_type: "User".to_string(),
        hops: vec![HopPlan {
            relation: "user_order".to_string(),
            fanout: 25,
            dest_sort_key_mode: None,
            absent_key_policy: AbsentKeyPolicy::UniformWithoutWeights,
        }],
        include_edge_ids: false,
    },
};

// sampler.sample(request).await?;
```
