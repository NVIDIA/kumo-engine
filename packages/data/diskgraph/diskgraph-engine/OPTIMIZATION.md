# Sampler Optimizations

This document summarizes the key optimizations implemented in `diskgraph-sampler` to achieve high-throughput, memory-efficient sampling.

## Architectural Optimizations

### 1. IO Loop Inversion & Hub Deduping
*   **Key Idea:** Adjacency lists for high-indegree nodes should be accessed exactly once per layer.
*   **Optimization:** The traversal loop is inverted to group by `Shard -> Node (Sorted) -> Relation`. By processing frontier nodes in sorted order, we detect duplicate visits to the same physical node ("hubs") and reuse the open file views and parsed segments.

### 2. Lazy-but-Eager Structure Parsing
*   **Key Idea:** Minimize redundant scanning of the on-disk edge list headers.
*   **Optimization:** The kernel parses all relation segments (counts and offsets) once per node visit. This avoids re-decoding the varint headers for every individual edge relation requested by the sampling plan.

### 3. O(k) Uniform Sampling (Sorted Index Jumps)
*   **Key Idea:** Uniform sampling should not require scanning the entire adjacency list.
*   **Optimization:** Implemented a jump-based sampling algorithm. For fanout $k$ and degree $N$, the engine generates and sorts $k$ random indices upfront, then uses `EdgeUnpacker::advance` to skip directly to target edges. Complexity is reduced from $O(N)$ to $O(k \log k)$.

### 4. Integer-Only Hot Path
*   **Key Idea:** Remove string hashing and map lookups from the hot sampling loop.
*   **Optimization:** A plan compilation step resolves all string names (Node Types, Relations) to integer IDs once per request. The traversal engine uses `usize` indexing into pre-allocated vectors for all state and statistics tracking.

### 5. Memory Pooling & Adaptive Shrinking
*   **Key Idea:** Minimize allocator overhead while preventing unbounded memory growth from outliers.
*   **Optimization:**
    *   **Pooling:** The engine uses a `StatePool` to recycle `SamplingState` objects across requests. This amortizes the cost of allocating large vectors for nodes and frontiers, maintaining a "warm" capacity for steady-state workloads.
    *   **Adaptive Reset:** To handle "Hero Queries" (outliers touching millions of nodes) without permanently bloating the pool, `adaptive_reset` checks vector capacity on return. If a vector's capacity exceeds a safety threshold (262k items), it uses `shrink_to` to release memory to the OS while maintaining a warm baseline, preventing both OOM and re-allocation churn.

### 6. Unified Fanout Spec
*   **Key Idea:** Simplify configuration and lookups for complex heterogeneous graphs.
*   **Optimization:** Added support for a `default_fanout` per layer. This allows the sampling loop to apply a fallback strategy for all relations not explicitly configured, reducing plan size and compilation time.

## Low-Level Kernel Optimizations

### 7. Vectorized Edge Unpacking
*   **Key Idea:** Use bit-level parallelism to decode packed edges at line speed.
*   **Optimization:** The `EdgeUnpacker` processes edges in batches (e.g., 128). it reads 64-bit or 128-bit words and uses pre-computed bitmasks and shifts to extract Destination IDs and SortKeys in a single CPU cycle, minimizing branch mispredictions.

### 8. Lazy SortKey Decoding
*   **Key Idea:** Defer expensive timestamp and weight parsing until strictly necessary.
*   **Optimization:** Since SortKeys pack timestamps in the Most Significant Bits, temporal filters perform raw integer comparisons (`sort_key < anchor_packed`). Full decoding to `i64` or `f32` is only performed on-demand for final output serialization.

### 9. Zero-Copy Segment View
*   **Key Idea:** Eliminate intermediate buffers and data copies between disk and network.
*   **Optimization:** All views and segments hold direct references (`&[u8]`) into memory-mapped storage. Combined with Arrow Flight, data flows from disk to client with zero intermediate heap copies in the hot path.
