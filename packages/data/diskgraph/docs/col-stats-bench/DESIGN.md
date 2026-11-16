# Column Statistics for DiskGraph: Design Document

**Author**: DiskGraph Team
**Status**: Design Phase Complete — Ready for Implementation
**Date**: 2026-02-11
**Target Audience**: Distinguished engineers unfamiliar with DiskGraph internals

---

## Executive Summary

### What is DiskGraph?

DiskGraph is a high-performance Rust-based graph ingestion and indexing engine, designed to replace Kumo's existing Spark-based `graphengine` (built on RocksDB). Kumo is an enterprise machine learning platform that trains Graph Neural Networks (GNNs) on relational data. The core workflow involves:

1. **Data ingestion**: Raw CSV/Parquet tables → Graph structure (nodes + edges)
2. **Training**: GNN models consume the graph to make predictions

The existing Spark-based pipeline has three main stages:
- **Spark Ingest**: Read raw tables, materialize edges (FK→PK joins), write to RocksDB
- **GraphEngine (RocksDB)**: Store and sample graph neighborhoods during training
- **Training**: PyTorch Geometric models on sampled subgraphs

DiskGraph replaces all of this with a single Rust pipeline that achieves:
- **10-30× faster ingestion** (parallel Parquet processing vs. Spark shuffle)
- **Lower storage overhead** (custom row-pack encoding vs. RocksDB)
- **Single-pass processing** (no Spark shuffle or external sort)

**Scale**: DiskGraph targets graphs with 100B+ nodes. Our production test dataset (`alt_detection_15pct`) has:
- 10-15B nodes
- 1605 Parquet files
- ~1.3 TB total size
- Typical ingest time: 16-18 minutes on 64-core machines with NVMe SSDs

### What Are Column Statistics and Why DiskGraph Needs Them

Machine learning models require **feature encoding**: converting raw column values (strings, numbers) into numerical representations (embeddings, normalized floats) that neural networks can process.

**Kumo-ml's encoder selection** depends on per-column statistics:
- **Categorical columns**: Needs exact frequency counts to determine vocabulary size and whether to use index encoding (exact lookup) vs. hash encoding (feature hashing)
- **Numerical columns**: Needs mean/std/min/max/quantiles for normalization and encoder selection (StandardScaler, MinMaxScaler, RobustScaler, etc.)
- **Text columns**: Needs token statistics for text encoder selection (not in Phase 1 scope)

**Current workflow** (inefficient):
1. DiskGraph ingests raw data → graph index
2. **Separate PySpark job** reads the same Parquet files again → computes column stats
3. kumo-ml receives stats → selects encoders → trains models

**Goal**: Eliminate step 2 by computing column statistics **inline during DiskGraph ingestion**.

**Challenge**: The encode stage already processes ~1.3TB at 1-2 GB/s. Adding stats collection must have **minimal overhead** (target: <5%, acceptable: <10-12%) to avoid doubling total ingest time.

### Key Design Decision: Observer Pattern with Streaming Algorithms

We will compute statistics **inline during the per-row encoding loop** using an observer pattern:
- **Phase 1 (this design)**: Streaming algorithms with O(1) memory and O(1) per-value cost (Welford online moments, HyperLogLog cardinality, Reservoir sampling for quantiles)
- **Phase 2 (future)**: Expensive stats requiring O(N) memory (category frequency histograms with HashMap) will be done in a **separate post-encode pass** (following Spark's ANALYZE TABLE pattern)

**Why this is safe**: The observer pattern allows us to measure overhead incrementally. Each statistic is added one at a time with benchmarking. If cumulative overhead exceeds 10-12%, we stop and move remaining stats to Phase 2.

---

## 1. Background: DiskGraph System Architecture

### 1.1 Crate Structure

DiskGraph is organized into focused Rust crates:

| Crate | Purpose |
|-------|---------|
| **diskgraph-ingest** | Orchestration: reads GraphConfig, dispatches parallel tasks, coordinates write pools, produces final index |
| **diskgraph-rowpack** | Encoding format: converts Arrow arrays → compact binary row format. Includes RowAccess (universal type converter) and RowPackContextBuilder (sampling phase) |
| **diskgraph-infer** | Schema inference: scans CSV/Parquet → infers column types → produces GraphConfig JSON |
| **diskgraph-index** | Index storage: on-disk format for node data, edge adjacency, and sampling |
| **diskgraph-core** | Shared types: Key, IngestedType, data structures |
| **diskgraph-datagenerator** | Testing: generates synthetic Parquet data |

### 1.2 Data Model

**Tables come in two flavors:**

1. **Node tables**: Represent entities (users, products, transactions)
   - Have a **primary key** (PK) column
   - Have **feature columns** (what we need stats for)
   - May have **foreign key** (FK) columns pointing to other tables

2. **Edge tables**: Represent explicit relationships (user_follows_user, product_in_category)
   - Store only `src`, `dst`, and optional `sort_key` columns
   - No feature columns → **no stats needed for edge tables**

**Example schema:**

```
users (node table)
  ├─ user_id (PK, String)
  ├─ age (Int32)               ← needs stats: mean, std, etc.
  ├─ country (String)          ← needs stats: cardinality, category counts
  ├─ balance (Float64)         ← needs stats
  └─ created_at (Timestamp)    ← timestamp, no ML encoding

transactions (node table)
  ├─ txn_id (PK, String)
  ├─ user_id (FK → users)      ← identifier, stats = cardinality only
  ├─ amount (Float64)          ← needs stats
  └─ timestamp (Timestamp)

edge_follows (edge table)
  ├─ src (FK → users)
  └─ dst (FK → users)
  (no feature columns, skip stats entirely)
```

### 1.3 Type System

#### IngestedType Enum (Physical Types)

DiskGraph normalizes Arrow/Parquet types into 19 **physical storage types**:

```rust
// diskgraph-ingest/src/graph_config.rs
pub enum IngestedType {
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float32, Float64,
    Bool,
    String,
    Bytes,
    Uuid,         // 16-byte UUID
    Uuid256,      // 32-byte UUID
    Ip,           // 16-byte IPv4/IPv6
    Date,         // Days since epoch
    Timestamp,    // (seconds, nanoseconds)
    Decimal128, Decimal256,
    Float32Array, Float64Array, Float16Array,  // embeddings
}
```

These types determine **how values are stored** in rowpack format.

#### SemanticType Enum (ML Encoding Intent)

Column statistics depend on **semantic type** (what the column represents for ML):

```rust
// diskgraph-ingest/src/graph_config.rs
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SemanticType {
    Numerical,        // Continuous values: age, price, temperature
    Categorical,      // Discrete labels: country, product_category, boolean
    Id,               // Identifiers: primary keys, foreign keys
    Text,             // Free-form text: product descriptions, user bios
    Timestamp,        // Date/time values (for edge ordering, not ML features)
    Sequence,         // Ordered lists: embeddings, time series
    Multicategorical, // Concatenated categories: "tag1,tag2,tag3"
    Unsupported,      // Binary, unknown, or not meaningful for ML
}
```

**Mapping examples:**
- `Int32` named `"age"` → `Numerical`
- `Int32` named `"user_id"` → `Id`
- `String` with 200 unique values → `Categorical`
- `String` with 10M unique values → `Id` (too many for categorical encoding)
- `Float64Array` → `Sequence` (embedding vector)

#### ColumnSpec (The Unified Descriptor)

Each column is described by a `ColumnSpec` in the `GraphConfig`:

```rust
pub struct ColumnSpec {
    pub name: String,
    pub ingested_dtype: IngestedType,     // Physical storage type
    pub semantic_type: Option<SemanticType>,  // ML encoding type (inferred)
    pub source_dtype: Option<String>,     // Original Parquet type
    pub fixed_len: Option<usize>,         // For fixed-width strings/arrays
    pub timestamp_format: Option<String>, // For string→timestamp parsing
    pub timestamp_unit: Option<TimestampUnit>,
    pub precision: Option<u32>,           // Decimal precision
    pub scale: Option<i32>,               // Decimal scale
}
```

**Key relationship**: `ingested_dtype` determines storage and initial heuristics. `semantic_type` determines which stats to collect.

### 1.4 RowAccess: The Universal Type Converter

`RowAccess` (in `diskgraph-rowpack/src/access.rs`) is a critical abstraction that provides **typed access to Arrow array values** with **coercions**:

```rust
pub struct RowAccess<'p> {
    row: usize,
    cols_ptr: *const ArrayRef,   // Points to Arrow batch columns
    hints: Vec<RowAccessHints>,   // Per-column parsing hints
    // ...
}

impl RowAccess {
    // Type-specific accessors (17+ Arrow types per accessor):
    pub fn get_int64(&self, idx: usize) -> Result<Option<i64>>;
    pub fn get_float64(&self, idx: usize) -> Result<Option<f64>>;
    pub fn with_string<F, R>(&self, idx: usize, f: F) -> Result<Option<R>>
        where F: FnOnce(&str) -> Result<R>;
    pub fn get_timestamp(&self, idx: usize) -> Result<Option<(i64, u32)>>;
    // ... many more
}
```

**Key capability — Coercions**:
- `get_int64()` on an `Int32` array → automatically widens to `i64`
- `with_string()` on an `Int64` array → formats as decimal string
- `with_string()` on a `Binary` array → formats as UUID/IP string (if hint provided)
- `get_timestamp()` on a `String` array → parses ISO 8601 using provided format
- `get_timestamp()` on an `Int64` array → interprets as epoch microseconds

**Why this matters for stats**: We cannot observe stats **before** RowAccess, because:
- Timestamps might be stored as `Int64` (epoch) or `String` (ISO 8601) — we need the **final** coerced value
- Binary UUIDs are formatted to 36-character strings — stats must count the string, not 16 bytes
- String integers like `"12345"` are parsed to `i64` — stats must count the integer

**Implication**: Statistics **must** be collected **inside the encoding loop**, after RowAccess has performed all coercions.

---

## 2. Background: DiskGraph Ingest Pipeline

### 2.1 Pipeline Phases (High Level)

The `ingest()` function (in `diskgraph-ingest/src/lib.rs`) orchestrates six phases:

```
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 0: Infer (separate binary)                                    │
│  └─ Scan parquet files → infer schema → GraphConfig JSON             │
└──────────────────────────────────────────────────────────────────────┘
         ↓ GraphConfig JSON
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 1: Context (build_node_contexts)                              │
│  └─ Sample 10K rows per table → RowPackContextBuilder               │
│     → infer encoding params (dictionary, varint widths, etc.)        │
│     → write .ctx files                                               │
│     → infer SemanticType (NEW in this design)                        │
└──────────────────────────────────────────────────────────────────────┘
         ↓ .ctx files + SemanticType per column
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 2: Encode (build_temp_shards)                  ← WE ARE HERE  │
│  └─ For each table, for each file (1605 files):                      │
│     ├─ Open parquet → read batches (100K rows each)                  │
│     ├─ For each row:                                                 │
│     │   ├─ Extract PK/FK keys                                        │
│     │   └─ encode_row_from_access_into_buffers()                     │
│     │       → RowAccess reads every column                           │
│     │       → encode to rowpack format                               │
│     │       → write to shards                                        │
│     └─ Write temp node/edge files                                    │
└──────────────────────────────────────────────────────────────────────┘
         ↓ Temp shards (.shard files)
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 3: PK Map (build_primary_key_maps)                            │
│  └─ Read all node files → build HashMap<PK, NodeID>                  │
└──────────────────────────────────────────────────────────────────────┘
         ↓ PK→NodeID maps
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 4: Edge Map (build_edge_lists)                                │
│  └─ Read edge keys → resolve FK→NodeID → build adjacency lists       │
└──────────────────────────────────────────────────────────────────────┘
         ↓ Edge adjacency lists
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 5: Index (build_index)                                        │
│  └─ Merge shards → build final on-disk index                         │
└──────────────────────────────────────────────────────────────────────┘
         ↓ Final index (ready for training)
```

**Phase timing on alt_detection_15pct (1.3TB, 10-15B nodes):**
- Context: ~2-3 minutes
- **Encode: ~16-18 minutes** (THE BOTTLENECK)
- PK Map: ~3-4 minutes
- Edge Map: ~4-5 minutes
- Index: ~8-10 minutes
- **Total: ~35-40 minutes**

**Why encode is the bottleneck**: It reads every single parquet file, every single row, every single column value. It's CPU-bound (parallel processing across 64 cores) and I/O-bound (reading 1.3TB from NVMe SSDs at 1-2 GB/s).

**Our goal**: Add stats collection to the encode phase with <10-12% overhead (acceptable: 16-18 min → 18-20 min).

### 2.2 Context Phase Deep Dive (Where SemanticType is Inferred)

```rust
// diskgraph-ingest/src/lib.rs:1468
fn build_node_contexts(&mut self) -> Result<()> {
    // Create one WorkItem per node table (parallel execution)
    for table in &tables {
        if table.edge_table.is_some() { continue; }  // Skip edge tables

        tasks.push(WorkItem {
            run: Box::new(move |_scratch| {
                // 1. Create RowPackContextBuilder for this table
                let node_columns = rowpack_columns(&table)?;
                let mut node_builder = RowPackContextBuilder::new(
                    node_columns,
                    rowpack_config
                );

                // 2. Sample up to 10K rows
                let mut sampled = 0usize;
                let sample_limit = rowpack_config.sample_rows().min(10_000);

                for path in table.get_paths() {
                    // Open parquet file
                    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
                        .with_batch_size(100_000)
                        .build()?;

                    for batch in reader {
                        access.set_cols(&batch.columns());
                        for row in 0..batch.num_rows() {
                            if sampled >= sample_limit { break; }
                            access.set_row(row);
                            node_builder.observe_row_from_access(&mut access)?;
                            sampled += 1;
                        }
                    }
                }

                // 3. Compute unique counts from samples
                let unique_counts = node_builder.sample_unique_counts();
                // Returns Vec<(num_unique, count)> per column
                // - For integers: from HashSet<i64> (capped at 1M)
                // - For strings: from HashMap<String, u32> (capped at 1M)

                // 4. Infer SemanticType per column
                let semantic_types: Vec<SemanticType> = table.columns
                    .iter()
                    .zip(unique_counts.iter())
                    .map(|(col, &(num_unique, count))| {
                        let role = ColumnRole {
                            is_pk: table.primary_key == Some(col.name),
                            is_fk: table.fkeys.contains(col.name),
                            name: &col.name,
                        };
                        let sample = ColumnSampleStats { num_unique, count };
                        infer_semantic_type(col, &role, Some(&sample))
                    })
                    .collect();

                // 5. Finalize context and write .ctx file
                let node_context = node_builder.finish()?;
                target.write_bytes(&node_context_path, node_context.to_bytes())?;

                // 6. Send SemanticType back to main thread
                tx.send(Ok((table.name.clone(), semantic_types)))?;
                Ok(())
            })
        });
    }

    // Execute all tasks in parallel
    run_tasks(tasks, &pool)?;

    // Collect results and populate semantic_type on each column
    for result in results {
        let (table_name, semantic_types) = result?;
        let table = self.source.tables.find(|t| t.name == table_name)?;
        for (col, stype) in table.columns.iter_mut().zip(semantic_types) {
            col.semantic_type = Some(stype);  // ← populated before encode phase
        }
    }
    Ok(())
}
```

**Key output**: By the end of the context phase, every `ColumnSpec` has a populated `semantic_type` field. The encode phase can read this to decide which stats to collect per column.

### 2.3 Encode Phase Deep Dive (Where Stats Will Be Collected)

```rust
// diskgraph-ingest/src/lib.rs:1761
fn build_temp_shards(&mut self) -> Result<()> {
    // Create write pools (background threads writing to disk)
    let node_pool = NodeWritePool::new(target, 16);
    let edge_pool = EdgeSrcWritePool::new(target, 16);

    // Create one WorkItem per file per table
    // Example: 10 tables × 160 files/table = 1600 WorkItems
    let mut tasks = Vec::new();
    for table in &tables {
        for file_idx in 0..table.get_paths().len() {
            tasks.push(WorkItem {
                bucket: None,  // No bucketing, just parallel dispatch
                run: Box::new(move |_scratch| {
                    if table.edge_table.is_some() {
                        Self::build_temp_edge_file(...)
                    } else {
                        Self::build_temp_node_file(...)  // ← THE KEY FUNCTION
                    }
                })
            });
        }
    }

    // Dispatch to TaskPool (64 worker threads)
    let pool = self.make_pool::<()>();
    run_tasks(tasks, &pool)?;
    Ok(())
}
```

#### Inside `build_temp_node_file` (The Per-File Encode Loop)

```rust
// diskgraph-ingest/src/lib.rs:2112
fn build_temp_node_file(
    table: &TableSpec,
    file_idx: usize,
    num_shards: usize,
    // ... many parameters
) -> Result<TempNodeFileResult> {
    // 1. Load RowPackContext (encoding decisions from context phase)
    let ctx_bytes = fs::read(&temp_node_context_path(&table.name))?;
    let node_context = RowPackContext::from_bytes(&ctx_bytes)?;
    let node_encoder = RowPackEncoder::new(node_context)?;

    // 2. Setup RowAccess (Arrow→value accessor)
    let row_access_hints = build_row_access_hints(table);
    let mut access = RowAccess::new(row_access_hints, stats);

    // 3. Setup batching infrastructure
    let mut batcher = NodeBatcher::new(...);  // Batches writes to node pool
    let mut key_bufs: Vec<Vec<u8>> = vec![Vec::new(); table.columns.len()];
    let mut f16_buf: Vec<f16> = Vec::new();

    // 4. Open parquet file and process batches
    let path = table.get_paths()[file_idx];
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_batch_size(100_000)  // NODE_BATCH_SIZE
        .build()?;

    for batch in reader {
        // Set up Arrow column pointers
        let mut cols = Vec::with_capacity(table.columns.len());
        for col in &table.columns {
            let idx = batch.schema().index_of(&col.name)?;
            cols.push(batch.column(idx).clone());
        }
        access.set_cols(&cols);

        // 5. THE CRITICAL MACRO: process_node_batch!
        process_node_batch!(batch);
    }

    Ok(TempNodeFileResult { total_rows, node_data_dropped })
}
```

#### The `process_node_batch!` Macro (The Hot Loop)

```rust
// diskgraph-ingest/src/lib.rs:2212 (inside build_temp_node_file)
macro_rules! process_node_batch {
    ($batch:expr) => {
        // Precompute shard assignments (if PK is present)
        let pk_array = pk_idx.and_then(|idx| access.array(idx).ok());
        let precomputed = precompute_shards(pk_array, pk_col, num_shards);

        // Precompute FK keys (fast paths for common conversions)
        let mut fkeys_pre = Vec::with_capacity(rel_specs.len());
        for (_, idx, _, _, target_type) in &rel_specs {
            let array = access.array(*idx).unwrap();
            let kind = edge_fast_path_for_array(array, target_type);
            if kind.is_some() {
                fkeys_pre.push(extract_batch_keys(array, kind.unwrap()).ok());
            } else {
                fkeys_pre.push(None);
            }
        }

        // THE PER-ROW LOOP ← WHERE STATS OBSERVATION HAPPENS
        for row in 0..$batch.num_rows() {
            access.set_row(row);
            total_rows += 1;

            Self::write_node_row_from_access(
                table,
                pk_idx,
                &rel_specs,
                &fkeys_pre,
                row,
                &mut access,        // ← RowAccess positioned at this row
                &node_encoder,      // ← Will call encode_row_from_access_into_buffers
                &mut batcher,
                &mut dest_fkey_batcher,
                &mut sort_key_batchers,
                &mut row_id,
                pk_col,
                pk_pre,
                &mut key_bufs,      // ← Buffers for extracted keys
                &mut f16_buf,
                &mut node_data_dropped,
                num_shards,
                key_failures,
                timestamp_valid,
                shard_override,
            )?;
        }
    };
}
```

#### Inside `write_node_row_from_access` (The Per-Row Function)

```rust
// diskgraph-ingest/src/lib.rs:2435
fn write_node_row_from_access(
    table: &TableSpec,
    pk_idx: Option<usize>,
    rel_specs: &[(Arc<str>, usize, usize, String, IngestedType)],
    fkeys_pre: &[Option<Vec<Option<Key>>>],
    row: usize,
    access: &mut RowAccess<'_>,
    node_encoder: &Option<RowPackEncoder>,
    // ... many batchers
) -> Result<()> {
    // 1. Extract primary key (if present)
    for buf in key_bufs.iter_mut() { buf.clear(); }
    let shard = if let (Some(idx), Some(pk_col)) = (pk_idx, pk_col) {
        let buf = &mut key_bufs[idx];
        key_from_access_into_buf(access, idx, pk_col, &pk_col.ingested_dtype, buf)?;
        shard_for_key(buf, num_shards)
    } else {
        row_id % num_shards
    };

    // 2. Extract foreign keys
    for (i, (_rel_name, rel_idx, _rel_pos, _fk_col_name, target_type)) in rel_specs.iter().enumerate() {
        let buf = &mut key_bufs[*rel_idx];
        if let Some(precomputed_keys) = &fkeys_pre[i] {
            // Fast path: use precomputed key
            if let Some(Some(key)) = precomputed_keys.get(row) {
                write_key_to_buf(buf, key)?;
            }
        } else {
            // Slow path: extract via RowAccess
            let col = &table.columns[*rel_idx];
            key_from_access_into_buf(access, *rel_idx, col, target_type, buf)?;
        }
    }

    // 3. Encode rowpack data (ALL FEATURE COLUMNS) ← STATS GO HERE
    if let Some(encoder) = node_encoder {
        let mut tags = Vec::new();
        let mut bools = Vec::new();
        let mut data = Vec::new();

        // THE CRITICAL CALL ← THIS READS EVERY COLUMN VALUE
        let lens = encoder.encode_row_from_access_into_buffers(
            access,
            &mut tags,
            &mut bools,
            &mut data,
            &mut f16_buf,
            &key_bufs,  // Already-extracted keys (avoid re-reading)
        )?;

        // 4. Send encoded row to write pool
        batcher.add(...)?;
    }

    Ok(())
}
```

#### Inside `encode_row_from_access_into_buffers` (The Encoding Heart)

```rust
// diskgraph-rowpack/src/lib.rs:2297
pub fn encode_row_from_access_into_buffers(
    &self,
    access: &mut RowAccess<'_>,
    tags: &mut Vec<u8>,
    bools: &mut Vec<u8>,
    data: &mut Vec<u8>,
    f16_buf: &mut Vec<f16>,
    key_bufs: &[Vec<u8>],  // Pre-extracted keys (optimization)
) -> Result<RowPackRowLens> {
    let tag_start = tags.len();
    let bool_start = bools.len();
    let data_start = data.len();

    // THE COLUMN LOOP ← ITERATES OVER EVERY COLUMN
    for (idx, col) in self.ctx.columns.iter().enumerate() {
        let tag = match col.logical_type {
            IngestedType::Bool => match access.get_bool(idx).ok().flatten() {
                None => 0u8,  // Null tag
                Some(v) => {
                    bool_writer.write_bits(if v { 1 } else { 0 }, 1);
                    1u8  // Present tag
                }
            },
            IngestedType::Int64 => match access.get_int64(idx).ok().flatten() {
                None => 0u8,
                Some(v) => {
                    // Encode using varint/fixed-width based on context
                    encode_int(data, signed, varint, bits, v as i128)?;
                    1u8
                }
            },
            IngestedType::Float64 => match access.get_float64(idx).ok().flatten() {
                None => 0u8,
                Some(v) => {
                    encode_float_with_tag(data, col.logical_type, storage, v)?
                }
            },
            IngestedType::String => {
                // THE TRICKY CASE ← Strings are zero-copy borrowed
                access.with_string(idx, |v| {
                    // v: &str borrowed from Arrow buffer (no allocation)
                    encode_string(
                        data,
                        v,
                        col,
                        self.dict_maps[idx].as_ref(),
                        self.alphabet_maps[idx].as_ref(),
                        self.bpe_tries[idx].as_ref(),
                        None,
                        None,
                    )
                })?.unwrap_or(0u8)
            }
            IngestedType::Uuid => {
                // Use pre-extracted key if available (optimization)
                let buf = &key_bufs[idx];
                if buf.len() == 16 {
                    data.extend_from_slice(buf);
                    1u8
                } else {
                    match access.get_uuid(idx).ok().flatten() {
                        None => 0u8,
                        Some(v) => {
                            data.extend_from_slice(&v);
                            1u8
                        }
                    }
                }
            }
            IngestedType::Timestamp => match access.get_timestamp(idx).ok().flatten() {
                None => 0u8,
                Some((secs, nanos)) => {
                    write_varint(zigzag_encode(secs), data);
                    write_varint(nanos as u64, data);
                    1u8
                }
            },
            IngestedType::Float32Array => match access.float_list_view(idx)? {
                None => 0u8,
                Some(view) => {
                    encode_float_list_with_tag(data, logical_type, storage, view, f16_buf)?
                }
            },
            // ... 11 more type arms
        };
        tags.write_bits(tag, 2);  // 2-bit tag: 0=null, 1=present, 2/3=variants
    }

    // Pack tag/bool/data buffers
    out.reserve(tags.buf.len() + bools.buf.len() + data.len() + 8);
    tags.append_to(out);
    bools.append_to(out);
    out.extend_from_slice(&data);

    Ok(RowPackRowLens {
        tag_bits: (tags.len() - tag_start) * 8,
        bool_bits: (bools.len() - bool_start) * 8,
        data_bytes: data.len() - data_start,
    })
}
```

**Key observations:**

1. **Every column is read via RowAccess**: Even if a column isn't part of a key, it's still encoded into rowpack format.
2. **Strings are zero-copy**: `with_string()` borrows `&str` from Arrow buffer — no heap allocation.
3. **Coercions happen here**: `get_int64()` on an `Int32` array, `get_timestamp()` on a `String` array, etc.
4. **This is the only place we can observe correct values**: Before RowAccess = raw Arrow types. After RowAccess = coerced types ready for rowpack.

**Implication for stats**: We need to hook **inside this loop**, **after each RowAccess read**.

---

## 3. Background: Column Statistics in kumo-ml

### 3.1 The Stats kumo-ml Needs

Kumo-ml's encoder selection logic (in `kumoml/common/data/encoders.py`) requires different statistics depending on semantic type:

#### Numerical Columns

**Stats needed:**
- `MEAN`: Arithmetic mean (for standardization: `(x - mean) / std`)
- `STD`: Standard deviation (for standardization)
- `MIN` / `MAX`: Range (for MinMaxScaler: `(x - min) / (max - min)`)
- `KURTOSIS`: Excess kurtosis (for detecting heavy-tailed distributions → RobustScaler)
- `NUM_UNIQUE`: Approximate cardinality (to detect "fake" numerical columns like ZIP codes)
- `QUANTILES`: Q25, Q50, Q75 (for RobustScaler: `(x - median) / IQR`, IQR = Q75 - Q25)

**Encoder selection logic:**
```python
# kumoml/enterprise/nn/fused_encoder/fused_numerical.py
if kurtosis > 10:
    # Heavy-tailed distribution → use RobustScaler (median + IQR)
    encoder = RobustScaler()
elif (max_val - min_val) < 1e-6:
    # Constant column → skip normalization
    encoder = Identity()
elif min_val >= 0:
    # Non-negative → consider log transform
    if max_val / (min_val + 1e-9) > 1000:
        encoder = MinLogScaler()  # log(1 + x)
    else:
        encoder = StandardScaler()  # (x - mean) / std
else:
    encoder = StandardScaler()
```

#### Categorical Columns

**Stats needed:**
- `NUM_UNIQUE`: Exact or approximate cardinality (drives Index vs. Hash vs. Embedding decision)
- `CATEGORY_COUNTS`: Exact frequency histogram `Vec<(String, u64)>` sorted by count descending

**Encoder selection logic:**
```python
# kumoml/common/data/encoders.py
min_occ = max(int(math.sqrt(stats.num_unique)), 2)
num_unique_pass_min_occ = len([
    name for name, occ in stats.category_count if occ > min_occ
])

if num_unique_pass_min_occ <= 4000:
    if num_unique_pass_min_occ <= unique_threshold:
        # Low cardinality → exact one-hot encoding
        encoder = IndexEncoder(min_occ=min_occ)
    elif num_unique_pass_min_occ <= 20 * unique_threshold:
        # Medium cardinality → feature hashing
        encoder = HashEncoder(num_components=unique_threshold)
    else:
        # High cardinality → learned embedding
        encoder = EmbeddingEncoder(embedding_dim=...)
else:
    # Very high cardinality → learned embedding
    encoder = EmbeddingEncoder(embedding_dim=...)
```

**Critical detail — min_occ filtering**:
- `min_occ = max(√num_unique, 2)`
- Categories with count ≤ min_occ are filtered out (mapped to "unknown" token)
- **At 10K unique → min_occ = 100** (drop categories with ≤100 occurrences)
- **At 1M unique → min_occ = 1000** (drop categories with ≤1000 occurrences)

**Why exact counts matter**:
- Categories near the threshold can flip between included/excluded based on small count errors
- The **frequency ordering** determines embedding table layout (most frequent = index 0) → must be reproducible for model checkpointing

#### Text Columns

**Stats needed:**
- `AVG_TOKENS`: Average number of space-separated tokens (threshold = 3 to distinguish text from short categorical) — **Phase 1 inline** (~15ns per string, byte scan)
- `NON_CHAR_FRACTION`: Fraction of non-alphabetic bytes (to detect non-text noise) — **Phase 1 inline** (~15ns per string, byte scan)
- `STRING_GLOVE_OVERLAP`: Fraction of tokens present in GloVe vocabulary — **Phase 2** (requires external vocabulary file)

#### Id Columns

**Stats needed:**
- `NUM_UNIQUE`: Approximate cardinality only (sanity check that FK cardinality ≈ PK cardinality of referenced table)

#### Timestamp / Sequence Columns

**Stats needed:**
- Timestamp: `MIN` / `MAX` (for date range validation)
- Sequence: `MIN_LENGTH` / `MAX_LENGTH` (for RNN/Transformer encoder config)

### 3.2 Thresholds and Constants

```python
# kumoml/common/data/encoders.py
MAX_CATEGORIES = 1_000_000           # Max vocabulary size for IndexEncoder
UNIQUE_INT_THRESHOLD = 0.05          # Integer columns with >5% unique → Numerical, else Categorical
MAX_AUTO_CATEGORICAL_THRESHOLD = 3   # Max unique for auto-detect categorical (bools, enums)
AVG_TOKENS_THRESHOLD = 3             # Strings with >3 avg tokens → Text, else Categorical
```

### 3.3 Why Category Counts Must Be Exact

**Count-Min Sketch is NOT suitable** because:

1. **Only over-counts, never under-counts**: CMS guarantees `estimate >= true_count`. Categories near min_occ threshold get incorrectly promoted, never demoted. This breaks the filtering logic.

2. **Error magnitude at scale**: With 5B values and 1M CMS buckets, average error per bucket ≈ 5000 (5B / 1M). For a category with true count 1000 near min_occ=1000, CMS might report 5000-6000 → completely wrong.

3. **Frequency ordering must be reproducible**: Model training uses the same vocabulary across multiple training runs (checkpointing, hyperparameter tuning). Approximate counts cause non-deterministic ordering.

**Solution for Phase 1**: Skip exact category counts (move to Phase 2). Use HyperLogLog for NUM_UNIQUE only (0.8% relative error, sufficient for encoder selection).

**Solution for Phase 2**: Use exact `HashMap<String, u64>` with a cap (e.g., 2M entries). Beyond cap, stop inserting new keys but keep counting existing ones. Accept that ultra-high-cardinality columns (>2M unique) will have incomplete counts → fallback to EmbeddingEncoder anyway.

---

## 4. Design: SemanticType Inference (Step 0 — DONE)

**Objective**: Before the encode phase begins, every column must have a populated `semantic_type` field so we know which stats to collect.

### 4.1 Implementation (Already Complete)

**Added to `graph_config.rs`:**

```rust
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SemanticType {
    Numerical, Categorical, Id, Text, Timestamp, Sequence, Multicategorical, Unsupported,
}

pub struct ColumnSpec {
    // ... existing fields
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_type: Option<SemanticType>,  // ← NEW FIELD
}
```

**Inference logic (`infer_semantic_type()`):**

```rust
pub fn infer_semantic_type(
    col: &ColumnSpec,
    role: &ColumnRole,
    sample: Option<&ColumnSampleStats>,
) -> SemanticType {
    // 1. Role-based rules (PK/FK are always Id, regardless of dtype)
    if role.is_pk || role.is_fk {
        return SemanticType::Id;
    }

    // 2. Dtype-based rules (unambiguous types)
    match col.ingested_dtype {
        Bool => return SemanticType::Categorical,
        Date | Timestamp => return SemanticType::Timestamp,
        Uuid | Uuid256 | Ip => return SemanticType::Id,
        Float32Array | Float64Array | Float16Array => return SemanticType::Sequence,
        Float32 | Float64 | Decimal128 | Decimal256 => return SemanticType::Numerical,
        Bytes => return SemanticType::Unsupported,
        _ => {}  // Int* and String are ambiguous — need heuristics + stats
    }

    // 3. Name heuristics (for remaining Int*/String types)
    if contains_identifier(role.name) {
        return SemanticType::Id;
    }

    // 4. Stats-based rules (for remaining Int*/String types)
    let Some(sample) = sample else {
        // No sample stats → fallback to dtype
        return match col.ingested_dtype {
            String => SemanticType::Categorical,
            _ => SemanticType::Numerical,  // Int* without stats → assume numerical
        };
    };

    if sample.count == 0 { return SemanticType::Unsupported; }

    let num_unique = sample.num_unique;
    let unique_fraction = num_unique as f64 / sample.count as f64;

    // Very low cardinality (≤3 unique) → always categorical (boolean-like)
    if num_unique <= MAX_AUTO_CATEGORICAL_THRESHOLD {
        return SemanticType::Categorical;
    }

    // Within categorical range (≤1M unique)
    if num_unique <= MAX_CATEGORIES {
        // For integers: high uniqueness (>5%) → numerical, otherwise categorical
        if is_int_type(&col.ingested_dtype) && unique_fraction > UNIQUE_INT_THRESHOLD {
            return SemanticType::Numerical;
        }
        return SemanticType::Categorical;
    }

    // Very high cardinality (>1M unique) → Text for strings, Numerical for ints
    match col.ingested_dtype {
        String => SemanticType::Text,
        _ => SemanticType::Numerical,
    }
}
```

**Stats tracking in `RowPackContextBuilder`:**

```rust
// diskgraph-rowpack/src/lib.rs
#[derive(Debug, Clone)]
struct ColumnStats {
    count: usize,
    null_count: usize,
    int_uniques: HashSet<i64>,  // ← NEW (capped at 1M)
    string_counts: HashMap<String, u32>,  // Already existed (capped at 1M)
    // ... other fields
}

impl ColumnStats {
    fn observe_int64(&mut self, v: i64) {
        self.count += 1;
        if self.int_uniques.len() < 1_000_000 {
            self.int_uniques.insert(v);
        }
    }

    fn observe_string(&mut self, v: &str, config: &RowPackConfig) {
        self.count += 1;
        if self.string_counts.len() < 1_000_000 {
            *self.string_counts.entry(v.to_owned()).or_insert(0) += 1;
        }
        // ... also track max_len, avg_len for dictionary encoding decision
    }
}

pub fn sample_unique_counts(&self) -> Vec<(usize, usize)> {
    self.stats.iter().zip(self.columns.iter()).map(|(s, logical)| {
        let num_unique = match logical {
            IngestedType::Int8 | IngestedType::Int16 | IngestedType::Int32
                | IngestedType::Int64 | IngestedType::UInt8 | IngestedType::UInt16
                | IngestedType::UInt32 | IngestedType::UInt64 => {
                s.int_uniques.len()
            }
            IngestedType::String => {
                s.string_counts.len()
            }
            _ => 0,  // Other types: stype unambiguous from dtype
        };
        (num_unique, s.count)
    }).collect()
}
```

**Result**: By the end of `build_node_contexts()`, every `ColumnSpec.semantic_type` is populated. The encode phase can read this field to decide which stats to collect.

---

## 5. Design: Observer Pattern (Step 1 — Next Implementation)

### 5.1 Problem Statement

**Goal**: Collect statistics on values **after RowAccess coercions** but **inside the encoding loop**.

**Constraint**: Cannot modify `encode_row_from_access_into_buffers()` signature — it's called from thousands of places and is performance-critical.

**Constraint**: Cannot extract string values outside the `with_string()` closure — strings are zero-copy borrowed from Arrow buffers and extracting them requires heap allocation (300K mallocs per 100K-row batch = ~16% overhead measured).

**Requirements:**
1. Zero overhead when stats collection is disabled
2. Correct values (post-coercion)
3. Zero allocation for strings
4. Minimal code duplication

### 5.2 Why Other Approaches Don't Work

#### Approach 1: Column-at-a-Time (Before Per-Row Loop)

**Idea**: Iterate over Arrow arrays before entering the per-row loop.

```rust
// REJECTED APPROACH
for (idx, col) in table.columns.iter().enumerate() {
    let array = access.array(idx)?;
    match col.ingested_dtype {
        IngestedType::Int64 => {
            let int_array = array.as_any().downcast_ref::<Int64Array>()?;
            for value in int_array.iter() {
                if let Some(v) = value {
                    observe_int64(idx, v);
                }
            }
        }
        // ... 18 more type arms
    }
}
```

**Problem**: This observes **raw Arrow values**, not **coerced values**. Examples of what we'd miss:
- Timestamps stored as `Int64` (epoch microseconds) → RowAccess coerces to `(i64, u32)` (seconds, nanos)
- Timestamps stored as `String` ("2023-01-15T12:34:56Z") → RowAccess parses to `(i64, u32)`
- Binary UUIDs (16 bytes) → RowAccess formats to 36-character string
- `Int32` columns read via `get_int64()` → RowAccess widens to `i64`

**Verdict**: Cannot use. Stats would be on wrong values.

#### Approach 2: Read-Then-Encode Refactor

**Idea**: Split encoding into two phases:
1. Read all values via RowAccess → store in temporary buffers
2. Encode from temporary buffers

```rust
// REJECTED APPROACH
struct RowValues {
    ints: Vec<Option<i64>>,
    floats: Vec<Option<f64>>,
    strings: Vec<Option<String>>,  // ← HEAP ALLOCATION
    // ...
}

fn read_row_from_access(access: &mut RowAccess) -> RowValues {
    let mut values = RowValues::default();
    for (idx, col) in columns.iter().enumerate() {
        match col.logical_type {
            IngestedType::Int64 => {
                values.ints[idx] = access.get_int64(idx).ok().flatten();
            }
            IngestedType::String => {
                values.strings[idx] = access.with_string(idx, |s| Ok(s.to_owned()))?;
                // ← ALLOCATES String on every row
            }
            // ...
        }
    }
    values
}
```

**Problem**: String extraction requires `s.to_owned()` → heap allocation. Measured overhead:
- 100K-row batch with 10 string columns = 1M strings
- 1M heap allocations per batch
- **Measured overhead: ~16%** on string-heavy workloads

**Verdict**: Cannot use. Overhead too high.

### 5.3 Chosen Approach: Observer Pattern

**Key insight**: `encode_row_from_access_into_buffers()` already reads every value via RowAccess. We can make it **generic over an observer** and call the observer **inline, after each value is read**.

#### Observer Trait

```rust
// diskgraph-rowpack/src/lib.rs
pub trait ColumnValueObserver {
    fn observe_i64(&mut self, col_idx: usize, value: i64);
    fn observe_f64(&mut self, col_idx: usize, value: f64);
    fn observe_string(&mut self, col_idx: usize, value: &str);  // ← Zero-copy!
    fn observe_timestamp(&mut self, col_idx: usize, secs: i64, nanos: u32);
    fn observe_float_list(&mut self, col_idx: usize, len: usize);
    fn observe_null(&mut self, col_idx: usize);
    // ... one method per IngestedType category
}

pub struct NoOpObserver;
impl ColumnValueObserver for NoOpObserver {
    #[inline(always)]
    fn observe_i64(&mut self, _col_idx: usize, _value: i64) {
        // Empty — will be eliminated by LLVM
    }
    #[inline(always)]
    fn observe_f64(&mut self, _col_idx: usize, _value: f64) {}
    #[inline(always)]
    fn observe_string(&mut self, _col_idx: usize, _value: &str) {}
    #[inline(always)]
    fn observe_timestamp(&mut self, _col_idx: usize, _secs: i64, _nanos: u32) {}
    #[inline(always)]
    fn observe_float_list(&mut self, _col_idx: usize, _len: usize) {}
    #[inline(always)]
    fn observe_null(&mut self, _col_idx: usize) {}
}
```

#### Generic Encode Function

```rust
// diskgraph-rowpack/src/lib.rs
impl RowPackEncoder {
    pub fn encode_row_from_access_into_buffers(
        &self,
        access: &mut RowAccess<'_>,
        tags: &mut Vec<u8>,
        bools: &mut Vec<u8>,
        data: &mut Vec<u8>,
        f16_buf: &mut Vec<f16>,
        key_bufs: &[Vec<u8>],
    ) -> Result<RowPackRowLens> {
        // Delegate to generic version with NoOpObserver
        self.encode_row_observed::<NoOpObserver>(
            access, tags, bools, data, f16_buf, key_bufs, &mut NoOpObserver
        )
    }

    pub fn encode_row_observed<O: ColumnValueObserver>(
        &self,
        access: &mut RowAccess<'_>,
        tags: &mut Vec<u8>,
        bools: &mut Vec<u8>,
        data: &mut Vec<u8>,
        f16_buf: &mut Vec<f16>,
        key_bufs: &[Vec<u8>],
        observer: &mut O,  // ← NEW PARAMETER
    ) -> Result<RowPackRowLens> {
        for (idx, col) in self.ctx.columns.iter().enumerate() {
            let tag = match col.logical_type {
                IngestedType::Int64 => match access.get_int64(idx).ok().flatten() {
                    None => {
                        observer.observe_null(idx);  // ← OBSERVE NULL
                        0u8
                    }
                    Some(v) => {
                        observer.observe_i64(idx, v);  // ← OBSERVE VALUE
                        encode_int(data, signed, varint, bits, v as i128)?;
                        1u8
                    }
                },
                IngestedType::String => {
                    access.with_string(idx, |v| {
                        observer.observe_string(idx, v);  // ← INSIDE CLOSURE (zero-copy)
                        encode_string(data, v, col, ...)
                    })?.unwrap_or_else(|| {
                        observer.observe_null(idx);
                        0u8
                    })
                }
                IngestedType::Timestamp => match access.get_timestamp(idx).ok().flatten() {
                    None => {
                        observer.observe_null(idx);
                        0u8
                    }
                    Some((secs, nanos)) => {
                        observer.observe_timestamp(idx, secs, nanos);  // ← OBSERVE
                        write_varint(zigzag_encode(secs), data);
                        write_varint(nanos as u64, data);
                        1u8
                    }
                },
                // ... 16 more type arms with observer calls
            };
            tags.write_bits(tag, 2);
        }
        Ok(...)
    }
}
```

**Key properties:**

1. **Zero-copy strings**: `observe_string(idx, v)` is called **inside** the `with_string()` closure, borrowing `&str` directly from Arrow buffer. No heap allocation.

2. **Correct coercions**: Observer sees the **same values** that get encoded. If RowAccess coerces `Int32` → `i64`, observer sees `i64`. If RowAccess parses string timestamp, observer sees `(secs, nanos)`.

3. **Zero overhead when disabled**: `NoOpObserver` methods are `#[inline(always)]` and empty. LLVM monomorphization eliminates them entirely. The generic `encode_row_observed<NoOpObserver>` compiles to the same code as today's non-generic version.

4. **Minimal code duplication**: The encoding logic is written once. Only the observer trait needs to be implemented per stats collector.

### 5.4 Call Sites

**When stats are disabled (current behavior):**

```rust
// diskgraph-ingest/src/lib.rs
let lens = encoder.encode_row_from_access_into_buffers(
    access, &mut tags, &mut bools, &mut data, &mut f16_buf, &key_bufs
)?;
```

**When stats are enabled (new code path):**

```rust
// diskgraph-ingest/src/lib.rs
let lens = if let Some(observer) = &mut file_stats_observer {
    encoder.encode_row_observed(
        access, &mut tags, &mut bools, &mut data, &mut f16_buf, &key_bufs, observer
    )?
} else {
    encoder.encode_row_from_access_into_buffers(
        access, &mut tags, &mut bools, &mut data, &mut f16_buf, &key_bufs
    )?
};
```

### 5.5 Benchmarking Step 1

**Verification steps:**

1. **Assembly inspection**: Compile with `encode_row_observed::<NoOpObserver>` and verify that the generated assembly is **identical** to the current `encode_row_from_access_into_buffers` (modulo register allocation).

2. **Wall-clock benchmark**: Run full ingest on synthetic data (100M rows, 50 columns). Compare:
   - Baseline (current code)
   - With observer infrastructure (calling `encode_row_observed::<NoOpObserver>`)
   - Expected: 0% overhead (within noise, <1%)

3. **If overhead detected**: Likely causes:
   - LLVM failed to eliminate observer calls (check `-C opt-level=3`, `#[inline(always)]`)
   - Generic function duplication → instruction cache pressure (profile with `perf`)
   - Solution: Investigate with `cargo asm`, `perf record`, or accept minor overhead if <2%

---

## 6. Design: Stats Accumulation and Merge

### 6.1 Per-File vs. Per-Thread Accumulators

**Trade-off**:

| Strategy | Pros | Cons |
|----------|------|------|
| **Per-file accumulator** (chosen) | Simple. Clear ownership. Merge only at file completion. | One accumulator per file × 1605 files × 7.2MB = 11.5 GB if all files in flight (not realistic — only 64 threads). |
| **Shared accumulator (Arc<Mutex>)** | Constant memory (one per table × 7.2MB × 10 tables = 72 MB). | Lock contention on merge. |
| **Per-thread accumulator** | Zero contention. | 64 threads × 10 tables × 7.2MB = 4.6 GB. Requires TaskPool refactor to return thread-local state. |

**Chosen: Per-file accumulator + shared merge target**

**Why**:
- Only 64 files in flight at a time (64 worker threads) → 64 × 7.2MB = 460 MB, not 11.5 GB
- Merge happens **after file completes** (5-7 seconds of work) → lock hold time is 8ms per file → contention negligible (<0.2% of thread time)
- Simpler than per-thread accumulators (no TaskPool refactor needed)

### 6.2 Memory Budget

**Per-column stats memory:**

| Stat | Size | Columns | Total |
|------|------|---------|-------|
| Welford (M1, M2, M3, M4, min, max, count) | 56 bytes | All feature columns | 50 × 56B = 2.8 KB |
| HyperLogLog (p=14, 16K registers) | 16 KB | All feature columns | 50 × 16KB = 800 KB |
| Reservoir (k=10K samples × 16 bytes, value+priority) | 160 KB | Numerical only (≈20) | 20 × 160KB = 3.2 MB |
| Timestamp min/max (2 × (i64, u32)) | 24 bytes | Timestamp columns (≈5) | 5 × 24B = 120 B |
| Sequence length (min, max) | 16 bytes | Sequence columns (≈5) | 5 × 16B = 80 B |
| Text stats (token/nonchar counts) | 24 bytes | Text columns (≈5) | 5 × 24B = 120 B |
| **Per-file total** | | | **~4.2 MB** |

**System memory budget:**
- 64 threads × 1 file in-flight × 4.2 MB = **270 MB** (per-file accumulators)
- 10 tables × 1 shared accumulator × 4.2 MB = **42 MB** (shared merge targets)
- **Total: ~310 MB** (negligible on 128-256 GB machines)

### 6.3 FileColStats Structure

```rust
// diskgraph-ingest/src/col_stats.rs (new file)
use diskgraph_ingest::graph_config::SemanticType;

pub struct FileColStats {
    pub table_name: String,
    pub columns: Vec<ColumnStatsCollector>,
}

pub enum ColumnStatsCollector {
    Numerical {
        welford: WelfordAccumulator,
        hll: HyperLogLog,
        reservoir: ReservoirSampler,
        null_count: u64,
    },
    Categorical {
        hll: HyperLogLog,
        null_count: u64,
    },
    Id {
        hll: HyperLogLog,
        null_count: u64,
    },
    Text {
        hll: HyperLogLog,
        token_count: u64,      // Sum of tokens across all strings
        non_char_count: u64,   // Sum of non-alpha bytes across all strings
        string_count: u64,     // Number of non-null strings observed
        null_count: u64,
    },
    Timestamp {
        min: Option<(i64, u32)>,
        max: Option<(i64, u32)>,
        null_count: u64,
    },
    Sequence {
        min_len: Option<usize>,
        max_len: Option<usize>,
        null_count: u64,
    },
    Skip,  // PK columns, Unsupported types, edge table columns
}

impl FileColStats {
    pub fn new(table: &TableSpec) -> Self {
        let columns = table.columns.iter().map(|col| {
            // Determine which collector based on semantic_type
            match col.semantic_type.unwrap_or(SemanticType::Unsupported) {
                SemanticType::Numerical => ColumnStatsCollector::Numerical {
                    welford: WelfordAccumulator::new(),
                    hll: HyperLogLog::new(14),  // p=14, 16KB
                    reservoir: ReservoirSampler::new(10_000),
                },
                SemanticType::Categorical => ColumnStatsCollector::Categorical {
                    hll: HyperLogLog::new(14),
                },
                SemanticType::Id => ColumnStatsCollector::Id {
                    hll: HyperLogLog::new(14),
                },
                SemanticType::Text => ColumnStatsCollector::Text {
                    hll: HyperLogLog::new(14),
                    token_count: 0,
                    non_char_count: 0,
                    string_count: 0,
                },
                SemanticType::Timestamp => ColumnStatsCollector::Timestamp {
                    min: None,
                    max: None,
                },
                SemanticType::Sequence => ColumnStatsCollector::Sequence {
                    min_len: None,
                    max_len: None,
                },
                _ => ColumnStatsCollector::Skip,
            }
        }).collect();

        Self { table_name: table.name.clone(), columns }
    }
}
```

### 6.4 Observer Implementation

```rust
// diskgraph-ingest/src/col_stats.rs
impl ColumnValueObserver for FileColStats {
    fn observe_i64(&mut self, col_idx: usize, value: i64) {
        if let Some(collector) = self.columns.get_mut(col_idx) {
            match collector {
                ColumnStatsCollector::Numerical { welford, hll, reservoir, .. } => {
                    welford.add(value as f64);
                    hll.add_i64(value);
                    reservoir.add(value as f64);
                }
                ColumnStatsCollector::Categorical { hll, .. }
                | ColumnStatsCollector::Id { hll, .. } => {
                    hll.add_i64(value);
                }
                _ => {}
            }
        }
    }

    fn observe_f64(&mut self, col_idx: usize, value: f64) {
        if !value.is_finite() { return; }  // NaN/Inf would poison Welford
        if let Some(collector) = self.columns.get_mut(col_idx) {
            match collector {
                ColumnStatsCollector::Numerical { welford, hll, reservoir, .. } => {
                    welford.add(value);
                    hll.add_f64(value);  // Hash the bits
                    reservoir.add(value);
                }
                _ => {}
            }
        }
    }

    fn observe_string(&mut self, col_idx: usize, value: &str) {
        if let Some(collector) = self.columns.get_mut(col_idx) {
            match collector {
                ColumnStatsCollector::Categorical { hll, .. }
                | ColumnStatsCollector::Id { hll, .. } => {
                    hll.add_str(value);
                }
                ColumnStatsCollector::Text { hll, token_count, non_char_count, string_count, .. } => {
                    hll.add_str(value);
                    // Short-circuit: scan at most 512 bytes for token/non_char stats.
                    // Sufficient for text-vs-categorical classification (threshold: avg_tokens > 3).
                    let scan = &value.as_bytes()[..value.len().min(512)];
                    *token_count += scan.iter().filter(|&&b| b == b' ').count() as u64 + 1;
                    *non_char_count += scan.iter().filter(|&&b| !b.is_ascii_alphabetic()).count() as u64;
                    *string_count += 1;
                }
                _ => {}
            }
        }
    }

    fn observe_timestamp(&mut self, col_idx: usize, secs: i64, nanos: u32) {
        debug_assert!(nanos < 1_000_000_000, "un-normalized nanos: {nanos}");
        if let Some(collector) = self.columns.get_mut(col_idx) {
            if let ColumnStatsCollector::Timestamp { min, max, .. } = collector {
                let ts = (secs, nanos);
                *min = Some(min.map_or(ts, |m| if ts < m { ts } else { m }));
                *max = Some(max.map_or(ts, |m| if ts > m { ts } else { m }));
            }
        }
    }

    fn observe_float_list(&mut self, col_idx: usize, len: usize) {
        if let Some(collector) = self.columns.get_mut(col_idx) {
            if let ColumnStatsCollector::Sequence { min_len, max_len } = collector {
                *min_len = Some(min_len.map_or(len, |m| m.min(len)));
                *max_len = Some(max_len.map_or(len, |m| m.max(len)));
            }
        }
    }

    fn observe_null(&mut self, col_idx: usize) {
        if let Some(collector) = self.columns.get_mut(col_idx) {
            match collector {
                ColumnStatsCollector::Numerical { null_count, .. }
                | ColumnStatsCollector::Categorical { null_count, .. }
                | ColumnStatsCollector::Id { null_count, .. }
                | ColumnStatsCollector::Text { null_count, .. }
                | ColumnStatsCollector::Timestamp { null_count, .. }
                | ColumnStatsCollector::Sequence { null_count, .. } => {
                    *null_count += 1;
                }
                ColumnStatsCollector::Skip => {}
            }
        }
    }
}
```

### 6.5 Plumbing: How FileColStats Gets Back to the Main Thread

`WorkItem::run` returns `Result<()>`, not `Result<FileColStats>`. Three options:

**Option A (chosen): Lock inside the worker closure.** The worker captures
`Arc<Mutex<TableColStats>>` and locks/merges at the end of `build_temp_node_file`,
before returning `Ok(())`. The shared accumulator is just another `Arc` captured by
the closure, alongside the existing write pool `Arc`s.

```rust
// Inside build_temp_shards, when creating WorkItems:
let shared_stats: Arc<Mutex<HashMap<String, TableColStats>>> = ...;

tasks.push(WorkItem {
    run: Box::new(move |_scratch| {
        let mut file_stats = FileColStats::new(&table);
        // ... build_temp_node_file with observer ...
        // After file is fully processed:
        shared_stats.lock().unwrap()
            .entry(table.name.clone())
            .or_insert_with(|| TableColStats::new(&table))
            .merge(file_stats)?;
        Ok(())
    })
});
```

**Why not Option B (side channel)**: Would require a `crossbeam::Sender` and a
dedicated receiver thread or post-loop drain. More plumbing for no benefit — the
lock hold time (~3ms) is negligible vs file processing time (5-7s).

**Why not Option C (change WorkItem return type)**: TaskPool is shared infrastructure
across the codebase. Changing its return type is too invasive for this feature.

### 6.6 Merge Strategy

**Shared merge target**:

```rust
// diskgraph-ingest/src/col_stats.rs
pub struct TableColStats {
    pub table_name: String,
    pub columns: Vec<ColumnStatsCollector>,
}

impl TableColStats {
    pub fn merge(&mut self, file_stats: FileColStats) -> Result<()> {
        ensure!(
            self.columns.len() == file_stats.columns.len(),
            "column count mismatch during merge"
        );

        for (dst, src) in self.columns.iter_mut().zip(file_stats.columns.into_iter()) {
            match (dst, src) {
                (
                    ColumnStatsCollector::Numerical { welford: w_dst, hll: h_dst, reservoir: r_dst },
                    ColumnStatsCollector::Numerical { welford: w_src, hll: h_src, reservoir: r_src },
                ) => {
                    w_dst.merge(w_src);
                    h_dst.merge(&h_src);
                    r_dst.merge(&r_src);
                }
                (
                    ColumnStatsCollector::Categorical { hll: h_dst },
                    ColumnStatsCollector::Categorical { hll: h_src },
                ) => {
                    h_dst.merge(&h_src);
                }
                // ... similar for other types
                _ => {}  // Type mismatch, skip (shouldn't happen)
            }
        }
        Ok(())
    }
}
```

**Contention analysis**:

```
Lock hold time per merge:
  - Welford merge: 4 FP ops × 50 columns = ~200 cycles = ~0.1 µs
  - HLL merge: 16K register max ops × 50 columns × 2ns/op = ~1.6 ms
  - Reservoir merge: sort 20K entries × 20 columns = ~1 ms
  - Total: ~3 ms per file

File processing time: 5-7 seconds

Lock contention: 3ms / 5000ms = 0.06% of thread time

Verdict: Negligible. Proceed with Arc<Mutex>.
```

### 6.6 Merge Algorithms

#### Welford Online Algorithm (Parallel Merge)

**Single-pass update**:

```rust
struct WelfordAccumulator {
    count: u64,
    m1: f64,  // mean
    m2: f64,  // variance × (n-1)
    m3: f64,  // skewness accumulator
    m4: f64,  // kurtosis accumulator
    min: f64,  // init to f64::INFINITY
    max: f64,  // init to f64::NEG_INFINITY
}

impl WelfordAccumulator {
    fn new() -> Self {
        Self {
            count: 0, m1: 0.0, m2: 0.0, m3: 0.0, m4: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    fn add(&mut self, x: f64) {
        self.count += 1;
        let n = self.count as f64;
        let delta = x - self.m1;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * (n - 1.0);

        self.m1 += delta_n;
        self.m4 += term1 * delta_n2 * (n * n - 3.0 * n + 3.0)
            + 6.0 * delta_n2 * self.m2
            - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;

        self.min = self.min.min(x);
        self.max = self.max.max(x);
    }
}
```

**Parallel merge** (Chan et al., 1983):

```rust
impl WelfordAccumulator {
    fn merge(&mut self, other: Self) {
        if other.count == 0 { return; }

        let n_a = self.count as f64;
        let n_b = other.count as f64;
        let n = n_a + n_b;
        let delta = other.m1 - self.m1;
        let delta2 = delta * delta;
        let delta3 = delta2 * delta;
        let delta4 = delta3 * delta;

        // Merge moments
        self.m1 = (n_a * self.m1 + n_b * other.m1) / n;

        self.m2 += other.m2 + delta2 * n_a * n_b / n;

        self.m3 += other.m3
            + delta3 * n_a * n_b * (n_a - n_b) / (n * n)
            + 3.0 * delta * (n_a * other.m2 - n_b * self.m2) / n;

        self.m4 += other.m4
            + delta4 * n_a * n_b * (n_a * n_a - n_a * n_b + n_b * n_b) / (n * n * n)
            + 6.0 * delta2 * (n_a * n_a * other.m2 + n_b * n_b * self.m2) / (n * n)
            + 4.0 * delta * (n_a * other.m3 - n_b * self.m3) / n;

        self.count = n as u64;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    fn mean(&self) -> f64 { self.m1 }
    fn variance(&self) -> f64 {
        if self.count < 2 { return 0.0; }
        self.m2 / (self.count as f64 - 1.0)
    }
    fn std_dev(&self) -> f64 { self.variance().sqrt() }
    fn skewness(&self) -> f64 {
        if self.m2 < 1e-10 { return 0.0; }  // near-constant → undefined, report 0
        let n = self.count as f64;
        (n.sqrt() * self.m3) / self.m2.powf(1.5)
    }
    fn kurtosis(&self) -> f64 {
        if self.m2 < 1e-10 { return 0.0; }  // near-constant → avoid FP garbage
        let n = self.count as f64;
        (n * self.m4) / (self.m2 * self.m2) - 3.0  // Excess kurtosis
    }
}
```

**Correctness**: Merge formula is exact. No approximation.

#### HyperLogLog (Approximate Cardinality)

**Structure**:

```rust
struct HyperLogLog {
    p: u8,              // Precision (14 → 16384 registers)
    registers: Vec<u8>, // 2^p registers (16KB for p=14)
}

impl HyperLogLog {
    fn new(p: u8) -> Self {
        let m = 1usize << p;  // p=14 → 16384 registers
        Self {
            p,
            registers: vec![0; m],
        }
    }

    fn add_i64(&mut self, value: i64) {
        let hash = gxhash::hash(&value.to_le_bytes());
        self.add_hash(hash);
    }

    fn add_f64(&mut self, value: f64) {
        let hash = gxhash::hash(&value.to_le_bytes());
        self.add_hash(hash);
    }

    fn add_str(&mut self, value: &str) {
        let hash = gxhash::hash(value.as_bytes());
        self.add_hash(hash);
    }

    fn add_hash(&mut self, hash: u64) {
        // Use first p bits for register index
        let idx = (hash & ((1 << self.p) - 1)) as usize;
        // Count leading zeros in remaining bits + 1
        let remaining = hash >> self.p;
        let leading_zeros = remaining.leading_zeros() as u8 + 1;
        // Update register with max
        self.registers[idx] = self.registers[idx].max(leading_zeros);
    }

    fn merge(&mut self, other: &Self) {
        assert_eq!(self.p, other.p);
        for (r_dst, r_src) in self.registers.iter_mut().zip(&other.registers) {
            *r_dst = (*r_dst).max(*r_src);  // Element-wise max
        }
    }

    fn cardinality(&self) -> u64 {
        let m = self.registers.len() as f64;
        let raw_estimate = self.alpha_m(m) * m * m
            / self.registers.iter().map(|&r| 2.0f64.powi(-(r as i32))).sum::<f64>();

        // Bias correction for small/large cardinalities
        if raw_estimate <= 2.5 * m {
            // Small range correction
            let zeros = self.registers.iter().filter(|&&r| r == 0).count() as f64;
            if zeros > 0.0 {
                (m * (m / zeros).ln()).round() as u64
            } else {
                raw_estimate.round() as u64
            }
        } else if raw_estimate <= (1u64 << 32) as f64 / 30.0 {
            raw_estimate.round() as u64
        } else {
            // Large range correction
            (-((1u64 << 32) as f64) * (1.0 - raw_estimate / (1u64 << 32) as f64).ln())
                .round() as u64
        }
    }

    fn alpha_m(&self, m: f64) -> f64 {
        match self.p {
            4 => 0.673,
            5 => 0.697,
            6 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / m),
        }
    }
}
```

**Error**: ~0.8% relative standard error for p=14. At 1M unique → estimate within 8K.

**Memory**: 16 KB per column (16384 registers × 1 byte).

**Merge**: O(2^p) = O(16384) element-wise max. ~0.4 ms per column.

#### Reservoir Sampling with Priority Keys (Efraimidis-Spirakis)

The naive proportional-slice approach to reservoir merge is biased: taking the
first N samples from a partition introduces selection bias (early-inserted values
are overrepresented). We use the Efraimidis-Spirakis (2006) approach instead:
each sample carries a random priority key, and merge simply keeps the top-k by
priority. This is provably correct.

For uniform (unweighted) sampling, the priority key simplifies to a random
`u ~ Uniform(0,1)`. The reservoir keeps the k values with the **largest** keys.

**Structure**:

```rust
struct ReservoirSampler {
    k: usize,                      // Reservoir size (10,000)
    samples: Vec<(f64, f64)>,      // (value, priority_key)
    count: u64,                    // Total values seen
    min_key: f64,                  // Current minimum key in reservoir (threshold)
    rng: SmallRng,                 // Fast RNG
}

impl ReservoirSampler {
    fn new(k: usize) -> Self {
        Self {
            k,
            samples: Vec::with_capacity(k),
            count: 0,
            min_key: 0.0,
            rng: SmallRng::seed_from_u64(42),
        }
    }

    fn add(&mut self, value: f64) {
        self.count += 1;
        let key: f64 = self.rng.gen();  // Uniform(0, 1)

        if self.samples.len() < self.k {
            // Filling phase: always insert
            self.samples.push((value, key));
            if self.samples.len() == self.k {
                // Reservoir full — compute threshold
                self.min_key = self.samples.iter()
                    .map(|&(_, k)| k)
                    .fold(f64::INFINITY, f64::min);
            }
        } else if key > self.min_key {
            // Replacement phase: new key beats current minimum
            // Find and replace the sample with min_key
            let min_idx = self.samples.iter()
                .enumerate()
                .min_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap())
                .unwrap().0;
            self.samples[min_idx] = (value, key);
            self.min_key = self.samples.iter()
                .map(|&(_, k)| k)
                .fold(f64::INFINITY, f64::min);
        }
        // else: key <= min_key, skip (most common case for large count)
    }

    fn merge(&mut self, other: &Self) {
        // Combine both reservoirs, keep top-k by priority key.
        // Provably correct: each row from the combined stream has
        // equal probability k/N_total of appearing in final reservoir.
        let mut combined: Vec<(f64, f64)> = Vec::with_capacity(
            self.samples.len() + other.samples.len()
        );
        combined.extend_from_slice(&self.samples);
        combined.extend_from_slice(&other.samples);

        // Sort by priority key descending, keep top k
        combined.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        combined.truncate(self.k);

        self.samples = combined;
        self.count += other.count;
        self.min_key = self.samples.last()
            .map(|&(_, k)| k)
            .unwrap_or(0.0);
    }

    fn quantile(&self, q: f64) -> Option<f64> {
        if self.samples.is_empty() { return None; }
        let mut values: Vec<f64> = self.samples.iter().map(|&(v, _)| v).collect();
        values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((values.len() - 1) as f64 * q).round() as usize;
        Some(values[idx])
    }
}
```

**Correctness**: Each row in the combined stream has equal probability `k / N_total`
of appearing in the final reservoir, regardless of partition sizes or merge order.
The priority key is the proof: only the top-k keys survive globally, so partition
boundaries don't matter.

**Note on determinism**: Quantile estimates are non-deterministic across runs because
file processing order (64 threads, work-stealing) determines which priority keys
each partition generates. This is acceptable — small variations in Q25/Q50/Q75 have
negligible impact on encoder selection. kumo-ml's existing PySpark `approxQuantile`
is also non-deterministic.

**Memory**: 10K samples × 16 bytes (value + key) = 160 KB per numerical column.

**Cost**: O(1) amortized per value. During filling phase: ~3 ns. After filling:
one RNG call + comparison against `min_key` → most values rejected immediately
(key < min_key, probability = 1 - k/count ≈ 1 for large count). ~3 ns per value.
Merge: sort 20K entries = ~50 µs per column.

---

## 7. Design: Phase 1 vs. Phase 2 Stats Split

### 7.1 Phase 1: Inline During Encode (Streaming Algorithms)

**Criteria for Phase 1 inclusion:**
1. O(1) memory per column (or small constant, e.g., 64 KB for HLL)
2. O(1) or O(log k) time per value
3. No external dependencies

**Phase 1 stats:**

| Stat | Algorithm | Memory per Column | Time per Value | Applies To |
|------|-----------|-------------------|----------------|------------|
| **MEAN / STD / KURTOSIS** | Welford online (m2 < ε guard) | 56 B | ~3 ns (4 FP ops) | Numerical |
| **MIN / MAX** | Running comparison (init ±∞) | 16 B | ~1 ns | Numerical, Timestamp |
| **NUM_UNIQUE** | HyperLogLog (p=14) | 16 KB | ~10 ns (hash + register update) | All (except Skip) |
| **QUANTILES** | Reservoir + E-S priority keys (k=10K) | 160 KB | ~3 ns (RNG + threshold check) | Numerical |
| **AVG_TOKENS** | Count spaces (first 512 bytes) | 8 B | ~5-15 ns (byte scan, short-circuit) | Text |
| **NON_CHAR_FRACTION** | Count non-alpha (first 512 bytes) | 8 B | ~5-15 ns (byte scan, short-circuit) | Text |
| **NULL_COUNT** | Increment counter | 8 B | ~1 ns | All |
| **SEQ_MIN_LENGTH / SEQ_MAX_LENGTH** | Running min/max | 16 B | ~1 ns | Sequence |

**Total overhead estimate (pessimistic)**:

```
Per-value cost for 50-column table:
  20 Numerical: (3 + 10 + 3 + 1) × 20 = 340 ns
  15 Categorical: 10 × 15 = 150 ns
  5 Text: (10 + 15 + 15) × 5 = 200 ns
  5 Id: 10 × 5 = 50 ns
  5 Timestamp: 1 × 5 = 5 ns
  Total: 745 ns per row

Baseline per-row cost (measured): 5000-8000 ns (encode + key extraction)
Overhead: 745 / 5000 = 15% (pessimistic) to 745 / 8000 = 9% (optimistic)

Actual expected: 7-12% (some columns skip stats, fast paths avoid RowAccess)
```

**Why Phase 1 stats are safe**: Even if overhead reaches 10-12%, we avoid a **second pass** over 1.3 TB of data (which would cost 16-18 minutes = 100% overhead).

### 7.2 Phase 2: Separate Post-Encode Pass (Expensive Stats)

**Criteria for Phase 2 deferral:**
1. O(N) memory per column (where N = cardinality or row count)
2. Expensive per-value operations (HashMap insert, string clone)
3. External dependencies (GloVe vocabulary, pre-trained models)

**Phase 2 stats:**

| Stat | Algorithm | Memory per Column | Why Deferred |
|------|-----------|-------------------|--------------|
| **CATEGORY_COUNTS** | HashMap<String, u64> | Up to 68 MB (1M entries × 68 bytes) | Expensive HashMap insert + string clone per value. ~100-200ns overhead per string value. For 10 string columns × 10B rows = 100B string inserts = ~10-20 minutes overhead. |
| **GLOVE_COVERAGE** | Lookup tokens in GloVe vocab | 1 GB (GloVe model) | Requires loading external vocabulary file. Text preprocessing pipeline. |
| **MULTI_CATEGORY_SEPARATOR** | Infer separator from samples | Negligible | Complex heuristic logic, edge case handling. |

**Implementation approach**: Follow Spark's pattern:
- Spark inline stats: min/max/null_count per parquet file (fast)
- Spark ANALYZE TABLE: HyperLogLog for NDV, histograms (separate job, slower)

**Phase 2 will be a separate command**:

```bash
# Phase 1 (inline)
diskgraph-ingest --parallel /path/to/data --output /path/to/index

# Phase 2 (post-encode)
diskgraph-stats --index /path/to/index --output /path/to/encoder_stats.json
```

**Why this is acceptable**: Kumo-ml already does a separate stats pass today (PySpark). Phase 2 would still be faster than PySpark (Rust, no shuffle, can read rowpack directly from index instead of re-reading parquet).

---

## 8. How Industry Does It

| System | Inline Stats | Separate Stats | Notes |
|--------|-------------|----------------|-------|
| **Parquet** | min/max/null_count per row group | None | Basic stats only, no advanced cardinality estimation. |
| **Apache Spark** | min/max/null_count | ANALYZE TABLE (HLL for NDV, histograms) | Separate job, can take hours on large tables. |
| **Delta Lake** | Same as Parquet | OPTIMIZE ZORDER + ANALYZE TABLE | Per-file stats inline, global stats separate. |
| **Apache Iceberg** | min/max/null_count per file | Table-level stats via metadata files | Manifest files track per-file stats, global stats computed on-demand. |
| **DuckDB** | min/max/null_count + HLL for NDV | None | Inline HLL during INSERT, very close to our approach. |
| **Apache Druid** | min/max + HLL per segment | Bitmap indexes (separate) | Inline HLL during ingestion, closest match to our design. |
| **ClickHouse** | min/max per granule (8192 rows) | Sampling for histograms (separate) | Deterministic sampling, fast approximate stats. |

**Our approach (Phase 1 + Phase 2) is most similar to Druid**: Inline streaming algorithms (HLL, Welford) during ingestion, with optional post-processing for expensive stats (histograms, indexes).

**Differentiator**: We use **Reservoir sampling for quantiles** (uncommon in columnar stores, which typically use t-digest or q-digest). Reservoir is simpler and sufficient for our use case (10K samples for 10B rows = 0.0001% sample rate, but unbiased).

---

## 9. Estimated Overhead Budget

**Target**: <5% encode stage overhead (optimistic)
**Acceptable**: <10-12% encode stage overhead (conservative)
**Baseline**: 16-18 minutes encode stage on alt_detection_15pct

| Step | What | Expected Overhead | Cumulative |
|------|------|-------------------|------------|
| **0** | SemanticType inference (context phase only) | 0% (separate phase) | 0% |
| **1** | Observer infrastructure (NoOpObserver) | 0% (compiled away) | **0%** |
| **2** | Benchmark data generation | N/A | N/A |
| **3** | FileColStats + shared accumulator + wiring | ~0.5% (branch per batch, mutex lock per file) | **0.5%** |
| **4** | Welford (mean/std/kurtosis) + min/max | ~1% (4 FP ops per numerical value) | **1.5%** |
| **5** | HLL (num_unique, p=14) | ~1-2% (hash + register update per value) | **3-4%** |
| **6** | Reservoir (quantiles) | ~0.5% (RNG + conditional swap per numerical value) | **4-5%** |
| **7** | Timestamp + Sequence stats | ~0% (trivial min/max) | **4-5%** |
| **8** | Text stats (avg_tokens, non_char_fraction) | ~1% (byte scans per text value) | **5-6%** |

**Optimistic case**: 5-6% overhead → 16 min → 16.8-17 min (acceptable)
**Pessimistic case**: 10-12% overhead → 16 min → 17.6-18 min (acceptable)
**Unacceptable case**: >20% overhead → 16 min → >19 min (fall back: move more stats to Phase 2)

**Mitigation if overhead exceeds 12%**:
1. Move Reservoir sampling to Phase 2 (saves ~0.5%)
2. Move text stats to Phase 2 (saves ~1%)
3. Reduce HLL precision to p=14 (16KB instead of 64KB per column, saves memory and merge time, negligible speed impact)

---

## 10. Implementation Order (Cross-Reference to IMPLEMENTATION_PLAN.md)

| Step | Description | Expected Duration |
|------|-------------|-------------------|
| **0** | SemanticType + inference | DONE (1 week) |
| **1** | Observer trait + NoOpObserver + encode_row_observed | 1-2 days |
| **2** | Benchmark data generation (generate.py) | 1 day |
| **3** | FileColStats + shared accumulator + wiring | 2-3 days |
| **4** | Welford (mean/std/kurtosis) + min/max | 2 days |
| **5** | HLL (num_unique) | 2-3 days |
| **6** | Reservoir (quantiles) | 2 days |
| **7** | Timestamp + Sequence stats | 1 day |
| **8** | Text stats (avg_tokens, non_char_fraction) | 1-2 days |
| **9** | JSON output + verification script (verify.py) | 1-2 days |

**Total**: 2-3 weeks for Phase 1 implementation, assuming incremental benchmarking after each step.

---

## 11. Risks and Mitigations

### Risk 1: LLVM Codegen Bloat from Generic Observer

**Risk**: Making `encode_row_observed<O>()` generic over `O: ColumnValueObserver` could cause LLVM to generate multiple copies of the function (one per observer type), increasing binary size and causing instruction cache thrashing.

**Detection**:
- **Binary size**: Check that binary size increases by <5% after adding observer infrastructure
- **Assembly inspection**: Use `cargo asm` to verify that `encode_row_observed::<NoOpObserver>` has similar instruction count to the baseline non-generic version
- **Performance**: Wall-clock benchmark on synthetic data (100M rows). If overhead >1% with NoOpObserver, investigate.

**Mitigation**:
- Add `#[inline(always)]` to observer trait methods
- Consider using dynamic dispatch (`dyn ColumnValueObserver`) if monomorphization causes issues (slight overhead, but removes codegen duplication)
- Profile with `perf record` to check for icache misses

### Risk 2: HLL Hash Function Quality (gxhash)

**Risk**: gxhash is designed for hash table performance (minimize low-bit collisions),
not for statistical uniformity across all 64 bits. HLL accuracy depends on
well-distributed bits in the leading-zeros region. A weak hash could cause register
clustering and inflated cardinality estimates.

**Detection**:
- Unit tests with known cardinalities (100, 10K, 1M, 10M unique values)
- Compare actual relative error against theoretical 0.8% for p=14
- If error exceeds 2× theoretical (>1.6%), the hash function is suspect

**Mitigation**:
- gxhash uses AES-NI, which has excellent avalanche properties (AES S-box is
  specifically designed for diffusion). Likely sufficient.
- If tests show poor accuracy: switch to xxhash64 or wyhash. Both have proven
  avalanche properties and are ~2-3ns per hash (vs gxhash's ~2ns). Negligible
  speed difference.
- Benchmark during Step 5 (HLL implementation) before committing to a hash function.

### Risk 3: HLL Memory Overhead with Many Columns

**Risk**: 16 KB per column × 100 columns × 64 files in flight = 100 MB. On tables
with many columns (e.g., 200+ feature columns), memory could be significant.

**Detection**:
- Monitor RSS during encode stage with `htop` or `systemctl status`

**Mitigation**:
- Only allocate HLL for columns that need it (skip PK, Timestamp, Unsupported)
- p=14 already uses only 16 KB per column (vs 64 KB at p=16)

### Risk 4: Reservoir Sampling Correctness with Parallel Merge

**Risk**: Weighted reservoir merge is tricky. If implemented incorrectly, quantile estimates could be biased or unstable.

**Detection**:
- Unit tests: Generate known distributions (Uniform, Normal, Exponential), split into partitions, merge, compare quantiles to numpy/scipy ground truth
- Test unequal partition sizes (e.g., 1M + 10K samples) to ensure weighting is correct
- Test edge cases: empty partitions, single-value partitions

**Mitigation**:
- Use proven weighted merge formula (Efraimidis & Spirakis, 2006)
- Add property-based tests (proptest crate) to fuzz merge logic
- Accept that quantiles are approximate (not exact) — sufficient for encoder selection

### Risk 5: Lock Contention on Shared Accumulator

**Risk**: 64 threads all trying to merge into `Arc<Mutex<TableColStats>>` at the same time could cause lock contention, serializing the encode stage.

**Detection**:
- Measure time spent holding mutex lock (add instrumentation)
- Profile with `perf` to check for `futex` syscalls (indicates blocking)
- Monitor CPU utilization — if encode threads are idle, contention is likely

**Mitigation**:
- Current design: merge at **file completion** (every 5-7 seconds), not per-row or per-batch → lock hold time ~8 ms every 5 seconds → negligible
- If contention detected: switch to per-thread accumulators (requires TaskPool refactor to return thread-local state)
- Alternative: use lock-free concurrent HashMap (crossbeam or similar) — more complex, likely unnecessary

### Risk 6: NaN/Infinity Poisoning Welford Accumulator

**Risk**: A single NaN or infinity value from a corrupted Parquet file poisons the
entire Welford accumulator — all subsequent means and variances become NaN. The merge
then propagates NaN to the global accumulator, destroying stats for the whole table.

**Mitigation** (already in design):
- `observe_f64()` guards with `if !value.is_finite() { return; }`
- Optionally track `nan_count` and `inf_count` for data quality reporting
- Cost: one branch per float value (~0 ns, branch predictor handles it)

### Risk 7: Kurtosis Instability for Near-Constant Columns

**Risk**: When m2 ≈ 0 (near-constant column), kurtosis = n * m4 / m2² produces
floating-point garbage that could incorrectly trigger RobustScaler (kurtosis > 10
threshold), which then divides by IQR ≈ 0, producing NaN features.

**Mitigation** (already in design):
- `kurtosis()` and `skewness()` return 0.0 when `m2 < 1e-10`
- Near-constant columns correctly fall through to `max - min < 1e-6` check

### Risk 8: Benchmark Data Not Representative

**Risk**: Synthetic data might not match production characteristics (skewed distributions, outliers, missing values, encoding edge cases), causing benchmarks to underestimate overhead.

**Detection**:
- Run final benchmarks on **real production data** (alt_detection_15pct) in addition to synthetic
- Compare stats output against kumo-ml's existing PySpark stats to ensure correctness
- Profile with `perf` on both synthetic and production data to check for differences in hotspots

**Mitigation**:
- Design synthetic data to match production (Zipf distributions for categorical, lognormal for numerical, realistic null rates)
- Include edge cases in synthetic data (very long strings, extreme outliers, all-null columns)
- Run incremental benchmarks on small slices of production data (e.g., 1% sample) before full benchmark

---

## 12. Open Questions and Future Work

### JSON Output Schema (Step 9)

The output must match kumo-ml's `ColumnStatistics` expected format. Preliminary schema:

```json
{
  "tables": {
    "<table_name>": {
      "columns": {
        "<col_name>": {
          "semantic_type": "numerical|categorical|id|text|timestamp|sequence",
          "count": 10000000,
          "null_count": 5000,
          "mean": 42.3,
          "std": 15.7,
          "min": 18.0,
          "max": 80.0,
          "kurtosis": -1.2,
          "skewness": 0.3,
          "num_unique": 63,
          "quantiles": { "q25": 30.0, "q50": 42.0, "q75": 55.0 },
          "avg_tokens": null,
          "non_char_fraction": null,
          "seq_min_length": null,
          "seq_max_length": null,
          "timestamp_min": null,
          "timestamp_max": null
        }
      }
    }
  }
}
```

Fields are null when not applicable to the semantic type. The exact field names and
nesting must be validated against `kumoml/common/data/column_statistics.py` before
Step 9 implementation.

### Phase 2 Design Questions

1. **Category counts storage format**: Should Phase 2 use HashMap<String, u64> or a more compact format (e.g., arrow array)? HashMap is simple but memory-heavy for high cardinality.

2. **Category counts cap**: What should the cap be? Options:
   - 1M entries (kumo-ml MAX_CATEGORIES)
   - 2M entries (allows some headroom)
   - 10M entries (covers ultra-high cardinality, but 680 MB per column)

3. **Phase 2 implementation approach**: Should Phase 2 re-read parquet files or read rowpack from the index? Reading rowpack is faster (already decoded, no Arrow overhead) but requires implementing a rowpack reader.

### Performance Optimizations

1. **SIMD for Welford**: Can Welford updates be vectorized? Arrow arrays are contiguous, so we could process 8 f64 values at a time with AVX-512. Likely marginal gains (~2x faster), but Welford is already cheap (~3 ns per value).

2. **SIMD for HLL**: Hashing is the bottleneck. Investigate SIMD hash functions (e.g., xxhash with AVX2). Again, likely marginal.

3. **Fast paths for stats**: Similar to FK key extraction fast paths, could we have fast paths for stats? E.g., if an entire Arrow Int64Array maps to Numerical, skip RowAccess and iterate the array directly. Trade-off: code complexity vs. speed.

### Correctness Verification

1. **Integration tests**: Generate synthetic data with known stats (e.g., 1M rows with mean=100, std=15, 10K unique categories), run ingest with stats, verify output JSON matches ground truth.

2. **Differential testing**: Run kumo-ml's existing PySpark stats pipeline and DiskGraph stats on the same data, compare outputs (within tolerances for HLL/reservoir approximations).

3. **Stress testing**: Run on pathological inputs (all-null columns, single-value columns, extreme outliers, 1B-row columns) to ensure no panics or incorrect results.

---

## 13. Appendix: Key File References

| File | Lines | Purpose |
|------|-------|---------|
| `diskgraph-ingest/src/graph_config.rs` | 93-170 | SemanticType enum, ColumnSpec, IngestedType enum |
| `diskgraph-rowpack/src/lib.rs` | 675-870 | RowPackContextBuilder, sample_unique_counts() |
| `diskgraph-rowpack/src/lib.rs` | 2250-2650 | encode_row_from_access_into_buffers() — the critical per-row loop |
| `diskgraph-rowpack/src/access.rs` | 1-100 | RowAccess, RowAccessHints, BinaryStringHint |
| `diskgraph-ingest/src/lib.rs` | 160-195 | ingest() pipeline phases |
| `diskgraph-ingest/src/lib.rs` | 1468-1730 | build_node_contexts() — context phase |
| `diskgraph-ingest/src/lib.rs` | 1730-1870 | build_temp_shards() — encode phase start |
| `diskgraph-ingest/src/lib.rs` | 2112-2432 | build_temp_node_file() — per-file processing |
| `diskgraph-ingest/src/lib.rs` | 2212-2278 | process_node_batch! macro — the hot loop |
| `diskgraph-ingest/src/lib.rs` | 2435-2500 | write_node_row_from_access() — per-row function |
| `docs/col-stats-bench/IMPLEMENTATION_PLAN.md` | Full | Step-by-step implementation plan (this design's companion) |
| `docs/col_stats_knowledge.md` | Full | Background on kumo-ml stats usage (**Note**: describes an earlier column-at-a-time approach that was rejected; this design doc supersedes it for architectural decisions) |

---

## 14. Appendix: ASCII Art Diagrams

### DiskGraph Ingest Pipeline (Full)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DISKGRAPH INGEST PIPELINE                       │
└─────────────────────────────────────────────────────────────────────────┘

                           ┌─────────────────┐
                           │  Raw CSV/Parquet │
                           └────────┬─────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  Phase 0: Infer     │
                         │  (diskgraph-infer)  │
                         │  ─ Scan 10K rows    │
                         │  ─ Infer schema     │
                         └──────────┬──────────┘
                                    │ GraphConfig.json
                         ┌──────────▼──────────┐
                         │  Phase 1: Context   │
                         │  ─ Sample 10K rows  │
                         │  ─ Build RowPackCtx │
                         │  ─ Infer SemanticType│
                         └──────────┬──────────┘
                                    │ .ctx files + semantic_type
                         ┌──────────▼──────────┐
                         │  Phase 2: Encode    │  ← STATS COLLECTED HERE
                         │  ─ Read ALL rows    │
                         │  ─ Extract PK/FK    │
                         │  ─ Encode rowpack   │
                         │  ─ Write shards     │
                         │  ─ Observe stats    │
                         └──────────┬──────────┘
                                    │ Temp shards + FileColStats
                         ┌──────────▼──────────┐
                         │  Phase 3: PK Map    │
                         │  ─ PK → NodeID hash │
                         └──────────┬──────────┘
                                    │ PK maps
                         ┌──────────▼──────────┐
                         │  Phase 4: Edge Map  │
                         │  ─ FK → NodeID      │
                         │  ─ Build adjacency  │
                         └──────────┬──────────┘
                                    │ Edge lists
                         ┌──────────▼──────────┐
                         │  Phase 5: Index     │
                         │  ─ Merge shards     │
                         │  ─ Build final index│
                         └──────────┬──────────┘
                                    │
                           ┌────────▼─────────┐
                           │  DiskGraph Index  │
                           │  (ready for ML)   │
                           └──────────────────┘
```

### Encode Phase Data Flow (Detailed)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      ENCODE PHASE (build_temp_shards)                   │
└─────────────────────────────────────────────────────────────────────────┘

   Main Thread                     Worker Thread (64×)
   ───────────                     ────────────────────
        │
        │  Create WorkItems (1605 files)
        ├──────────────────────┐
        │                      │
        │   Dispatch to        │
        │   TaskPool           ▼
        │                 ┌─────────────────────┐
        │                 │ build_temp_node_file│
        │                 └──────────┬──────────┘
        │                            │
        │                            │ 1. Load RowPackContext (.ctx file)
        │                            │ 2. Create RowAccess
        │                            │ 3. Create FileColStats (if --col-stats)
        │                            │
        │                 ┌──────────▼──────────┐
        │                 │ Open parquet file   │
        │                 └──────────┬──────────┘
        │                            │
        │                 ┌──────────▼──────────┐
        │                 │ For each RecordBatch│
        │                 │ (100K rows each)    │
        │                 └──────────┬──────────┘
        │                            │
        │                 ┌──────────▼──────────┐
        │                 │ process_node_batch! │
        │                 │ ─ Precompute shards │
        │                 │ ─ Extract FK keys   │
        │                 │ ─ For each row:     │
        │                 │   write_node_row... │
        │                 └──────────┬──────────┘
        │                            │
        │                 ┌──────────▼──────────┐
        │                 │ write_node_row_...  │
        │                 │ ─ Extract PK key    │
        │                 │ ─ encode_row_...    │
        │                 │   ↓ calls observer  │
        │                 │ ─ Send to NodeWriter│
        │                 └──────────┬──────────┘
        │                            │
        │                 ┌──────────▼──────────┐
        │                 │ encode_row_observed │
        │                 │ <O: ColumnValue...> │
        │                 │                     │
        │                 │ For each column:    │
        │                 │   ─ Read via RowAcc │
        │                 │   ─ observer.obs... │
        │                 │   ─ Encode to buf   │
        │                 └──────────┬──────────┘
        │                            │
        │                            │ FileColStats accumulates
        │                            │
        │                 ┌──────────▼──────────┐
        │                 │ File complete       │
        │                 │ Return FileColStats │
        │                 └──────────┬──────────┘
        │                            │
        │  Receive via channel       │
        ◄────────────────────────────┘
        │
        │  Lock shared TableColStats
        │  Merge FileColStats
        │  (8ms per file)
        │
        ▼
   ┌─────────────────────────┐
   │ All files complete      │
   │ Write encoder_stats.json│
   └─────────────────────────┘
```

### Observer Pattern Flow (Per-Row)

```
┌─────────────────────────────────────────────────────────────────────────┐
│            OBSERVER PATTERN (encode_row_observed<O>)                    │
└─────────────────────────────────────────────────────────────────────────┘

   encode_row_observed<FileColStats>(access, ..., observer)
        │
        │  For col_idx in 0..num_columns:
        │
        ├──▶ match col.ingested_dtype {
        │        IngestedType::Int64 => {
        │            let val = access.get_int64(col_idx)?  ── Arrow → i64 coercion
        │                                                     (Int32 → i64, etc.)
        │            if let Some(v) = val {
        │                observer.observe_i64(col_idx, v)  ──┐
        │                encode_int(data, v)                 │
        │            } else {                                │
        │                observer.observe_null(col_idx)  ────┤
        │            }                                        │
        │        }                                            │
        │        IngestedType::String => {                   │
        │            access.with_string(col_idx, |s| {       │
        │                observer.observe_string(col_idx, s) ├─ ZERO-COPY BORROW
        │                encode_string(data, s)              │
        │            })?                                      │
        │        }                                            │
        │        // ... 17 more type arms                    │
        │    }                                                │
        │                                                     │
        ▼                                                     ▼
                                          FileColStats.observe_i64(idx, v)
                                               │
                                               ├─ Numerical: welford.add(v)
                                               ├─ Numerical: hll.add_i64(v)
                                               ├─ Numerical: reservoir.add(v)
                                               ├─ Categorical: hll.add_i64(v)
                                               └─ Id: hll.add_i64(v)

   Result: Every value observed AFTER coercion, INSIDE encoding loop,
           with zero-copy string access. NoOpObserver compiles to zero
           overhead (LLVM eliminates empty inline functions).
```

---

## 15. Conclusion

This design provides a **complete, production-ready plan** for adding column statistics to DiskGraph's ingest pipeline with **minimal overhead** (<10-12%) and **zero risk** to the existing pipeline.

**Key innovations:**
1. **Observer pattern** for zero-copy stats collection inside the encoding loop
2. **SemanticType-driven** collection (only collect stats that matter for each column)
3. **Streaming algorithms** (Welford, HLL, Reservoir) for O(1) memory and bounded overhead
4. **Two-phase approach** (inline streaming stats, deferred expensive stats) to balance performance and completeness
5. **Incremental implementation** with benchmarking after each step to detect regressions early

**Next steps:** Begin implementation with Step 1 (Observer trait) and proceed incrementally through Steps 2-9, benchmarking after each addition.

---

**Questions for reviewers:**
1. Is the 10-12% overhead target acceptable given that it eliminates a full second pass (100% overhead)?
2. Should we implement Phase 2 (category counts) now or defer until Phase 1 is validated in production?
3. Any concerns about the observer pattern's interaction with LLVM optimization?
4. Should gxhash be benchmarked against xxhash64/wyhash for HLL quality before committing?
5. Is the 512-byte short-circuit for text stats aggressive enough, or should we scan the full string?
