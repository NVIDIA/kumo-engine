# DiskGraph Sampler

A high-performance, out-of-core graph sampling engine designed for Kumo.AI's heterogeneous GNN workloads.

## Overview

`diskgraph-sampler` implements a vectorized, memory-efficient sampling engine that traverses large-scale graphs stored on disk (via `diskgraph-index` layouts) and streams results directly to clients via Apache Arrow Flight (via `diskgraph-gateway`).

## Key Features

*   **Streaming Architecture:** Emits edges incrementally ("Edges First") to bound memory usage, while buffering node mappings for efficient "Nodes Last" delivery.
*   **Zero-Copy Design:** Compatible with Arrow Flight for direct-to-GPU transfer. Internal traversal uses `mmap` views without intermediate copies.
*   **Hub Deduping:** Optimizes traversal for "Hub Nodes" (nodes with high indegree from the frontier) by sorting and deduplicating access requests, ensuring adjacency lists are parsed exactly once per layer.
*   **O(k) Uniform Sampling:** Uses a "Sorted Index Jump" algorithm (Algorithm L variant) to sample $k$ neighbors from a degree $N$ node in $O(k \log k)$ time, bypassing the $O(N)$ scan of standard reservoir sampling.
*   **Temporal Sampling:** Supports deterministic "Most Recent" sampling with optimized SortKey filtering.
*   **Heterogeneous Support:** Natively handles multiple Node and Edge types with per-type fanout configurations.

## Architecture

### `Sampler`
The core driver.
1.  **Plan Compilation:** Converts string-based `SamplingPlan` to integer-based `CompiledSamplingPlan` for fast lookups.
2.  **BFS Traversal:** Iterates layers. For each layer, groups the frontier by Shard, sorts by Local ID, and iterates.
3.  **Kernel:** Opens `EdgeListView` lazily/optimally and dispatches to `sample_from_view`.

### `SamplingState` & Pooling
To minimize allocator thrashing:
*   **Pooled State:** `StatePool` recycles `SamplingState` objects.
*   **Zero-Allocation Reset:** `adaptive_reset` clears vectors without deallocating capacity, unless they exceed a "Hero Query" threshold (262k items), in which case it shrinks them to prevent permanent bloat.
*   **Dense Stats:** Uses `Vec<Vec<u64>>` for fast statistics tracking.

### `OutputHandler`
A trait for decoupling traversal from serialization.
*   `handle_edge`: Called for every sampled edge. Implementation (Gateway) buffers and streams Arrow batches.
*   `finalize`: Called at end. Implementation serializes buffered nodes and stats.

## Usage

```rust
let sampler = Sampler::new(readers, info, decoders);
let mut state = pool.acquire();
let mut output = MyOutputHandler::new();

sampler.sample(&request, &mut state, &mut output)?;
```
