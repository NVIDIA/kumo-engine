# Design Doc: Kumo Diskgraph Native Sampling & Feature Fetching

## 1. Overview
This document outlines the design for implementing a native, out-of-core sampling and feature fetching engine for `kumo-diskgraph`. The goal is to replace the legacy `pyg_lib`-based in-memory architecture (used in `kumo-graphengine`) with a high-throughput, disk-aware solution that supports:
1.  **Feature Parity**: Matching all legacy sampling behaviors (temporal, replacement, disjointness) and defaults.
2.  **Advanced Capabilities**: Native weighted sampling and edge-level attributes.
3.  **Scalability**: Handling datasets larger than RAM via memory-mapped disk storage.
4.  **Performance**: Leveraging Arrow Flight for zero-copy streaming to clients.

## 2. API Specification (`diskgraph-core`)

The core definitions will be refactored to move from single-node logical requests to batched, execution-ready specifications.

### 2.1. Sampling Request
The primary entry point for graph traversal.

```rust
// diskgraph-core/src/sampling.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSamplingRequest {
    /// Target dataset identifier.
    pub dataset_id: DatasetId,
    
    /// Batch of seed node IDs to start the random walk.
    pub seed_nodes: Vec<u64>,
    
    /// Optional batch of timestamps (1:1 with seed_nodes). 
    /// If present, acts as the anchor time for temporal filtering.
    pub seed_timestamps: Option<Vec<i64>>,
    
    /// The recursive sampling strategy.
    pub plan: SamplingPlan,
    
    /// Global behavioral modifiers.
    pub flags: SamplingFlags,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingFlags {
    /// If true, allows selecting the same neighbor multiple times to satisfy fanout.
    /// Default: false
    pub replace: bool,
    
    /// If true, follows edge direction. If false, treats graph as undirected.
    /// Default: true (Legacy: 'subgraph_type' != 'induced')
    pub directed: bool,
    
    /// If true, ensures nodes sampled at hop K are treated as distinct seeds for hop K+1.
    /// Required for correct temporal sampling.
    /// Default: true (Matches Kumo 'SamplerKwargs' default)
    pub disjoint: bool,
    
    /// If true, output includes edge identifiers (required for edge feature fetching).
    /// Default: false (Legacy hardcoded false, but recommended true for Diskgraph)
    pub return_edge_id: bool,
    
    /// If true, enables dynamic fanout compensation (Metapath-aware/Adaptive).
    /// Default: false
    pub adaptive: bool,
}
```

### 2.2. Sampling Strategy
We replace the linear `HopPlan` with a layer-based heterogeneous spec.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingPlan {
    /// Ordered list of layers (Hop 0, Hop 1, ...).
    pub layers: Vec<LayerSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSpec {
    /// Fanout configuration per edge type (e.g., "User:Click:Item").
    pub fanouts: HashMap<String, FanoutSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FanoutSpec {
    /// Maximum number of neighbors to sample.
    pub fanout: u32,
    /// Strategy for selecting neighbors from the candidate list.
    pub mode: SamplingMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingMode {
    /// Random uniform selection. (Legacy: "uniform")
    Uniform,
    
    /// Deterministic selection of the most recent edges. (Legacy: "last")
    TemporalMostRecent,
    
    /// Probabilistic selection based on edge weight.
    Weighted { weighting: Weighting },
    
    /// Probabilistic selection based on time-decayed weight.
    TemporalWeighted { weighting: Weighting },
}
```

### 2.3. Feature Fetching Request
New request type for the follow-up step to retrieve node features.

```rust
// diskgraph-core/src/features.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchFeatureRequest {
    pub dataset_id: DatasetId,
    
    /// The node type (table name) to fetch features for.
    pub node_type: String,
    
    /// The batch of global node IDs to fetch.
    pub node_ids: Vec<u64>,
    
    /// Optional: List of column names to fetch. If None, fetch all available features.
    pub feature_projection: Option<Vec<String>>,
}
```

### 2.4. Storage Trait Update
Update `GraphStorage` to support both operations returning Arrow-compatible types.

```rust
// diskgraph-core/src/storage.rs

#[async_trait]
pub trait GraphStorage: Send + Sync {
    // ... existing methods ...

    /// Perform a sampling pass.
    async fn sample(&self, request: BatchSamplingRequest) -> Result<arrow_array::RecordBatch>;

    /// Retrieve node features for a batch of IDs.
    async fn get_features(&self, request: BatchFeatureRequest) -> Result<arrow_array::RecordBatch>;
}
```

## 3. RPC Framework: Arrow Flight

We will use **Apache Arrow Flight** as the unified transport protocol for both sampling and feature retrieval.

### 3.1. Rationale
*   **Zero-Copy**: Allows mapping the underlying `RecordBatch` directly to GPU memory (via PyTorch/DLPack) without the serialization overhead of Protobuf or gRPC.
*   **Streaming**: Bypasses gRPC's 2GB message limit, enabling massive sampling batches to be streamed as a sequence of chunks.
*   **Unified Data Plane**: Both graph topology (COO lists) and node features (dense tables) map naturally to Arrow's columnar format.

### 3.2. Architecture: `diskgraph-gateway`
This new crate hosts the RPC server and implements the Flight service.
*   **Routing**: The `do_get` method inspects the `FlightDescriptor` command:
    *   `BatchSamplingRequest` $\to$ Dispatched to `diskgraph-sampler`.
    *   `BatchFeatureRequest` $\to$ Dispatched to `diskgraph-rowpack` reader.
*   **Streaming**: The server generates `FlightData` chunks iteratively, ensuring the memory footprint remains proportional to the batch size, not the total graph size.

### 3.3. Sampling Response Schema (Multiplexed Stream)
To minimize RPC overhead and provide pre-indexed tensors, the sampler returns a **Multiplexed Dictionary-Encoded Stream**. This single schema transports both Edges (structure) and Nodes (mapping) in one unified byte stream.

| Column | Type | Use Case |
| :--- | :--- | :--- |
| `row_type` | `Int8` | **0** for Edge, **1** for Node. |
| `id_a` | `Int64` | Edge: **Src Local ID** \| Node: **Local ID** (Batch-specific index). |
| `id_b` | `Int64` | Edge: **Dst Local ID** \| Node: **Global ID** (Storage ID). |
| `meta_type` | `Dict<String>` | Edge: **Relation Type** \| Node: **Node Table Name**. |
| `meta_val` | `Int64` | Edge: **Global Edge ID** \| Node: **Batch Index** (for disjoint). |

#### **3.3.1. Metadata (Stats)**
The aggregate counts (`num_sampled_nodes`, `num_sampled_edges`) required for `HeteroSamplerOutput` are computed by the server and sent as a serialized buffer in the **`app_metadata`** field of the final `FlightData` message. This keeps the data stream clean while providing the structural metadata needed for zero-copy slicing in Python.

#### **3.3.2. Data Ordering & Optimization**
The server implements an **"Edges First, Nodes Last"** strategy to maximize throughput and minimize framing overhead:
1.  **Streaming Edges**: As the sampler traverses the graph (BFS), sampled edges are streamed immediately in large chunks (`row_type=0`). This ensures the client starts receiving the bulk data without waiting for the full traversal.
2.  **Buffering Nodes**: To avoid flushing tiny `RecordBatch`es for the small initial hops (e.g., seeds), the server buffers newly discovered node mappings (`GlobalID -> LocalID`) in memory.
3.  **Flushing Nodes**: Once the traversal is complete, all buffered node mappings are flushed as one or more large, homogeneous batches (`row_type=1`).

**Benefits**:
*   **Batch Purity**: The client receives long sequences of pure Edge batches followed by pure Node batches, allowing vectorized appending to target tensors without per-row logic.
*   **Client Efficiency**: By sending **Local IDs** directly, the client avoids expensive hash-map lookups for re-indexing.

### 3.4. Feature Fetching Response
The feature fetching response streams `RecordBatch`es containing the requested feature columns.

#### **Schema Definition**
The schema is dynamic, based on the `feature_projection` requested by the client.
*   `node_id`: `Int64` (The Global Node ID, used for alignment verification).
*   `feat_{name}`: Native Arrow types mapping directly to the `diskgraph` physical storage types.

This format is natively compatible with `KumoFeatureStore`'s existing `TensorMapper` logic, allowing the client to feed the Flight stream directly into the encoding pipeline with zero overhead.

## 4. Sampling Algorithm (`diskgraph-sampler`)

The sampler implements a "Middle Path" vectorized approach, integrating sampling logic with on-the-fly re-indexing to produce tensors ready for direct model consumption.

### 4.1. Data Layout & Access
*   **Edges**: Accessed via `edges.bin` (packed payload: `relation_id | dst_id | sort_key`).
*   **Offsets**: Accessed via `offsets.bin` (delta-encoded indices).
*   **Sort Key**: `u64` packing `timestamp` (MSB) and `weight`, enabling optimized temporal scans where the traversal can terminate early once the time barrier is reached.

### 4.2. Algorithm Execution Flow

**Step 1: Initialization**
*   **ActiveFrontier**: Initialized with seed nodes. If `disjoint=true`, each node entry carries its `SeedIndex` to maintain lineage.
*   **LocalIndexMap**: Tracks nodes seen in the current batch to assign contiguous Local IDs.
    *   **If `disjoint=false`**: Keyed by `GlobalID`. Multiple paths to the same node are merged.
    *   **If `disjoint=true`**: Keyed by `(SeedIndex, GlobalID)`. This unrolls the graph into a forest of trees, where a physical node may have multiple Local IDs (one per seed).
*   **NodeBuffer**: An in-memory buffer to hold node mapping metadata (`GlobalID, LocalID, Type, SeedIndex`) to be flushed at the end.
*   **StatsAccumulator**: Tracks aggregate `num_nodes` and `num_edges` per hop and type.

**Step 2: Layer Iteration (0..L)**
For each `LayerSpec` in the plan:

1.  **Group & Fetch**: Group `ActiveFrontier` nodes by source table type and use vectorized `extract_batch_keys` to fetch adjacency list offsets in bulk.
2.  **Neighbor Access (Kernel)**: Access `edges.bin` via memory map. For each active node, retrieve its neighbor list slice from disk.
3.  **Filtering & Selection**: Unpack the `SortKey` for each neighbor. Filter by `anchor_time` and apply the `SamplingMode` (Temporal/Weighted/Uniform) to select up to `fanout` neighbors.
4.  **Re-Indexing & Edge Emission**:
    *   For each selected edge `(GlobalSrc, GlobalDst)` originating from `SeedIndex`:
        *   **Lookup Src**: Retrieve `LocalSrc` from `LocalIndexMap`.
        *   **Lookup/Assign Dst**:
            *   **Key**: Use `GlobalDst` (if merged) or `(SeedIndex, GlobalDst)` (if disjoint).
            *   **Check**: If the key is new, assign the next `LocalID`, add to `LocalIndexMap`, append to `NodeBuffer`, and add to `NextFrontier` for the next hop.
            *   **Reuse**: If existing, retrieve the previously assigned `LocalID` (deduplication within the seed's tree or globally).
        *   **Emit Edge**: Write the **Local IDs** `(LocalSrc, LocalDst)` along with `EdgeType` and `EdgeID` to the Arrow Builder (`row_type=0`).
    *   **Streaming**: Flush the builder if the chunk threshold is reached.
5.  **Stats Update**: Increment `StatsAccumulator` counts for the current hop's edges and newly discovered nodes.

**Step 3: Finalize**
1.  **Flush Edges**: Send any remaining edge records in the builder.
2.  **Flush Nodes**: Serialize the entire `NodeBuffer` into large Arrow `RecordBatch`es (`row_type=1`) and stream them. 
3.  **Send Stats**: Serialize the `StatsAccumulator` and attach it to the `app_metadata` field of the final Flight message.

### 4.3. Reasoning: Server-Side Indexing
Performing the re-indexing on the server during traversal is significantly more efficient than post-processing on the client. Since the sampler must already track visited nodes to handle multi-hop logic and avoid redundant work (especially for `disjoint=false`), assigning local IDs adds negligible overhead. By providing local IDs directly in the stream, we eliminate the need for expensive hashing and dictionary lookups in the Python client, enabling $O(1)$ construction of PyTorch tensors.

## 5. Feature Fetching Algorithm (`diskgraph-rowpack`)

The feature fetching logic optimizes for bulk retrieval.

1.  **Locate Shards**: Map `node_ids` to `shard_idx` and `local_offset` using loaded `NodeShardMetadata`.
2.  **Vectorized Read**:
    *   Group IDs by shard.
    *   Access `node_data.bin`. If data is `rowpack` encoded (compressed row groups), determine which groups cover the requested offsets.
    *   Read/Decompress required row groups.
3.  **Arrow Conversion**:
    *   Decode the `rowpack` binary format directly into Arrow Builders to minimize allocation.
4.  **Alignment**: Ensure the output `RecordBatch` contains a `node_id` column to allow the client to map features back to the graph structure.

## 6. Client Integration (`diskgraph-python` & `kumo`)

The client side leverages the pre-computed Local IDs and structural metadata to reconstruct `HeteroSamplerOutput` with minimal CPU overhead.

### 6.1. Sampler Client (`KumoDiskgraphSampler`)
The client binding (implemented in Rust/PyO3) consumes the Flight stream and exposes a Python-friendly API.

#### **Reconstruction Algorithm**
1.  **Stream Consumption & Concatenation**:
    *   Iteratively read `RecordBatch` chunks from the Flight stream.
    *   **Edges**: Append `Edge` chunks (`row_type=0`) to contiguous internal vectors.
    *   **Nodes**: Append `Node` chunks (`row_type=1`) to node mapping vectors.
    *   *Optimization*: Since the server buffers nodes to send them as a single final batch, the client typically performs only one large concatenation for nodes, minimizing memory reallocations.
2.  **Efficient Demultiplexing (Run-Length Boundary Detection)**:
    *   The server guarantees that edges and nodes are **grouped by type** within the stream.
    *   The client performs a vectorized $O(N)$ scan on the `meta_type` integer indices to identify the offsets where types change (boundaries).
    *   **Overhead Note**: While this scan is a new step compared to the legacy format, it is extremely fast (SIMD-ready integer comparisons). It replaces the legacy overhead of parsing dozens of distinct Protobuf fields and many small tensor headers.
3.  **Zero-Copy Slicing & Tensor Creation**:
    *   Using the boundary offsets found in step 2, the client creates $O(1)$ views (slices) of the concatenated data.
    *   These slices are mapped directly to `torch.Tensor` (via DLPack). **No data is copied during this step.**
    *   The resolution of integer indices to actual string names (e.g., `2` $\to$ `"User:Click:Item"`) happens only once per group.
4.  **Structural Metadata**:
    *   The client uses the `Stats Batch` to populate `num_sampled_nodes` and `num_sampled_edges`.
    *   Because the stream is sent in BFS order, the hop counts align perfectly with the concatenated data slices.

#### **Performance Trade-off Summary**
*   **Legacy**: High overhead from Protobuf deserialization and many small allocations for a dictionary of tensors.
*   **New Design**: Introduces a single large concatenation (Memory Copy) and a linear type-scan (CPU), but eliminates the complex Protobuf-to-Object decoding path. The result is a more cache-friendly and predictable reconstruction process.

### 6.2. Feature Client (`KumoFeatureStore`)
*   **Input**: Calls `diskgraph-gateway` via Flight with a `BatchFeatureRequest`.
*   **Processing**:
    *   Receives a stream of `RecordBatch`es where columns correspond to features.
    *   Converts these batches into a `dict[str, pa.Array]` (or `pa.Table`).
    *   Passes this dictionary directly to `TensorMappers.convert_raw_to_tensor_frame`.
*   **Compatibility**: This format is natively compatible with Kumo's existing feature processing pipeline, which already operates on PyArrow arrays. This eliminates the legacy overhead of decoding Protobuf byte blobs.

### 6.3. Defaults & Configuration
The Python wrapper (`KumoDiskgraphSampler`) enforces Kumo-specific defaults to ensure backward compatibility:
*   `temporal_strategy` $\to$ Defaults to `'last'` (TemporalMostRecent).
*   `disjoint` $\to$ Defaults to `True` (required for correct temporal semantics).
*   `return_edge_id` $\to$ Defaults to `True` (to support edge features/weights).

## 7. Performance Analysis

### 7.1. Optimizations
*   **SortKey Packing**: Enables $O(1)$ (best-case) temporal sampling.
*   **Server-Side Re-indexing**: Eliminates expensive Python-side `unique()`/hashing.
*   **Zero-Copy**: Data path is Arrow end-to-end.

### 7.2. Complexity
*   **Time**: $O(N \cdot d_{avg})$ where $N$ is batch size. Weighted sampling forces linear scan.
*   **Memory**: Proportional to output batch size. Strictly out-of-core.

## 8. Implementation Plan

1.  **Refactor Core**: Update `diskgraph-core` with new Request/Flags structs.
2.  **Gateway Setup**: Create `diskgraph-gateway` crate with Flight server.
3.  **Sampler Engine**: Implement `diskgraph-sampler` (Uniform/Temporal) with re-indexing.
4.  **Weighted Support**: Add `SortKey` unpacking and A-Res algorithm.
5.  **Feature Fetching**: Implement `get_features` using `rowpack` decoding.
6.  **Client Binding**: Update Python client to use Flight and set defaults (e.g., `temporal_strategy='last'`, `disjoint=True`).

## 9. Design Highlights

### **1. Transport: Arrow Flight with Zero-Copy**
Replaces legacy gRPC/Protobuf with **Apache Arrow Flight** to bypass the 2GB message limit and enable massive, scalable sampling. Data flows from disk $\to$ server RAM $\to$ network $\to$ client RAM $\to$ GPU memory (via DLPack) with **minimal serialization**, relying on Arrow's columnar memory layout.

### **2. Server-Side Re-Indexing (Zero Client CPU Overhead)**
The `diskgraph-sampler` (Rust) performs the **Global-to-Local ID mapping** during the traversal. It maintains a `HashMap` of visited nodes per batch and emits edges using `0..N` local indices. This eliminates the need for expensive Python-side `unique()` or dictionary lookups, allowing the client to construct PyTorch adjacency matrices in $O(1)$ time.

### **3. Multiplexed Dictionary-Encoded Stream**
Instead of sending separate streams, **Edges** (topology) and **Nodes** (mapping) are interleaved in a single Arrow Flight stream using a `row_type` discriminator. **Dictionary Encoding** compresses repetitive string metadata (e.g., `"User:Click:Item"`) to 1 byte per row, significantly reducing bandwidth. The client uses efficient **vectorized run-length scanning** on these integer codes to "demultiplex" the stream without expensive string parsing.

### **4. "Edges First, Nodes Last" Buffering Strategy**
The server streams **Edge batches** immediately as they are generated to keep memory footprint low. It **buffers Node mappings** in memory and flushes them as a single large batch at the end of the stream. This ensures the client receives long, homogeneous chunks of data, maximizing SIMD efficiency during reconstruction.

### **5. Integrated Metadata (Stats Batch)**
A lightweight **Stats Batch** is sent in the final message's `app_metadata`. It provides the exact counts (`num_sampled_nodes`, `num_sampled_edges`) per hop. This allows the client to **instantaneously slice** the flat tensors into per-hop layers without iterating over the data to count items.

### **6. Temporal & Weighted Optimization**
The `SortKey` packs `Timestamp` (MSB) and `Weight` (LSB) into a single `u64`. This physical sorting on disk enables **$O(1)$ temporal sampling** (best-case), as the sampler can stop reading an adjacency list as soon as it hits the time horizon or fanout limit.

### **7. Native Feature Fetching Integration**
Feature fetching uses the same Arrow Flight transport. The response format (`RecordBatch` with native types) is designed to plug directly into Kumo's existing `TensorMappers`, removing the need for legacy Protobuf blob decoding.

### **8. Full Parity & Correctness**
Supports all legacy flags: `disjoint` (for temporal correctness), `replace` (for fixed tensor sizes), and `directed`. Handles **Heterogeneous Graphs** natively by maintaining type-local index spaces, matching PyTorch Geometric's `HeteroData` requirements.