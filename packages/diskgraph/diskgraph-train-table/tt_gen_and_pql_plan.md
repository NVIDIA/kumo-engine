# Training Table Generation & PQL Plan Compilation

## Pipeline Overview

```
PQL Query (Kumo SDK)
  → ValidatedPredictiveQuery (AST)
  → DiskGraphPlanCompiler.compile() (Python)
  → plan_json (DiskGraphTTPlan, serialized JSON)
  → resolve_plan() (Rust, main thread)
  → ResolvedPlan (runtime IR with index handles)
  → execute_with_summary() (Rust, parallel entity loop)
  → Parquet TT + split assignments + stats artifacts + manifest
```

## Plan Compilation (Python)

**File**: `diskgraph-pybind/python/diskgraph/compiler.py`

`DiskGraphPlanCompiler.compile()` translates PQL AST to DiskGraph execution plan:

1. **Temporal spec**: `builder.temporal(timestamps, forecast_length, lag_timesteps)`
2. **Entity spec**: `builder.for_each(entity_table, entity_col, ...)`
3. **Prediction type**: `builder.node_prediction()` or `builder.link_prediction(explode, include_frequency)`
4. **Target**: `_traverse_target(query.target_ast, query)` — recursively walks AST, emitting `builder.aggregate()`, `builder.column()`, `builder.compare()` calls
5. **Split**: `builder.split_ratio(split_spec)` — one `SplitSpec`; unit scaling and family gating happen Rust-side in `make_split_spec`
6. **Output**: `builder.build()` → JSON string

### Key Parameters

| Parameter | Source | Effect |
|-----------|--------|--------|
| `explode` | `not ALLOW_ARRAY_TARGETS` (env var, default False) | True = one row per target item; False = list column |
| `include_frequency` | Set for RANK queries | Adds `__kumo_internal_frequency_column__` |
| `list_distinct_target_is_array` | `allow_array_targets()` | Controls LP output format |

## Plan Data Structures (Rust)

**File**: `diskgraph-core/src/tt_builder.rs`

```
DiskGraphTTPlan
  ├── entity_spec: EntitySpec {table, column, time_col, ...}
  ├── temporal: Option<TemporalSpec> {timestamps, forecast_length, lag}
  ├── prediction_type: PredictionType {Node | Link{explode, include_frequency} | PassThrough | Forecasting}
  ├── stages: Vec<Stage>
  │     └── Stage {ops: Vec<Op>, filter: Option<StageFilter>}
  │           └── Op::Aggregate {alias, path: Vec<EdgeRef>, func, target_col, window, row_filter, limit}
  │           └── Op::Column {alias, table, column, path: Vec<EdgeRef>}
  │           └── Op::Compare {alias, lhs, op, rhs}
  └── split: Option<SplitSpec>

EdgeRef {src_table, src_col, dst_table, dst_col}
```

## Plan Resolution (Rust)

**File**: `diskgraph-train-table/src/ttgen_resolve.rs`

### Why ResolvedPlan Exists

The `DiskGraphTTPlan` (JSON) uses string identifiers: table names, column names, edge references like `{src_table: "transactions", src_col: "customer_id", dst_table: "customers", dst_col: "customer_id"}`. Every entity in the parallel loop would need to look up "transactions" in the index, find the column index for "customer_id", locate the edge relation, etc. — string hashing and map lookups on every iteration, millions of times.

`resolve_plan()` runs once on the main thread and converts all string references to direct handles:

- Table names → `Arc<TableReader>` (pointer to memory-mapped index data)
- Column names → rowpack column indices (integer offsets for direct decode)
- Edge references → relation indices + `RelationLayout` (precomputed bit widths and masks for edge decoding)
- Sort key encoders → `SortKeyEncoder` (precomputed for temporal window comparisons)
- Filters → `ResolvedRowFilter` with column indices resolved against table contexts
- For LP ListDistinct: `target_pk_table` → `Arc<TableReader>` for the FK destination table (PR #173)

The result is `ResolvedPlan` — a fully resolved intermediate representation where every operation is a direct index/pointer access with zero string lookups. Wrapped in `Arc` and shared immutably across all rayon threads.

### Why ResolvedPlan Is Internal

`ResolvedPlan` holds `Arc<TableReader>` references into the `GraphIndexReader`'s memory-mapped data. These are lifetime-bound to the index — they can't outlive it, can't be serialized, can't cross process boundaries. The plan JSON (`DiskGraphTTPlan`) is the serializable, portable, cross-language contract. `ResolvedPlan` is the runtime-only optimization that eliminates repeated lookups.

This is the same pattern as a compiled SQL query plan: the SQL string is the interface, the physical plan with table handles is internal to the engine.

### Key Resolution Steps

```
DiskGraphTTPlan (JSON, string references)
  → resolve_plan() [main thread, once]
    → resolve entity table, time_col index, valid_to_col index, entity filter
    → for each stage:
        → for each op:
            → resolve_path(): EdgeRef strings → ResolvedEdge with relation_idx, layout, encoder
            → resolve target column: string → rowpack column index
            → resolve row filter: column names → indices, remote filters → resolved paths
            → resolve FK destination: for LP ListDistinct, find target PK table (PR #173)
    → infer target dtype from last op
  → Arc::new(ResolvedPlan) [shared across all threads]
```

`resolve_fk_destination()` deserves note: for LP `ListDistinct`, it scans the destination table's edge relations to find which table the target column is an FK to. If found, validates the table has a PK map (hard error if not). This enables the executor to filter orphaned FK values without any per-row schema validation.

## Execution (Rust)

**File**: `diskgraph-train-table/src/ttgen_executor.rs`

### Threading Model

```
Main thread:
  open index → resolve_plan() → Arc<ResolvedPlan>
  spawn rayon parallel iterator

Rayon thread pool (N = num_cpus):
  num_chunks = min(N * 8, num_entities) chunks
  each chunk: execute_chunk(&plan, entity_range, sender)
    → owns: ChannelBatchWriter (sends RecordBatches via crossbeam channel)
    → owns: ChunkStats (per-chunk accumulator, merged after all chunks)
    → owns: aliases Vec, link_frequencies Vec (reused across entities)

Consumer thread (Python via pyo3):
  TTPlanIterator.__next__() → recv() from channel
  GIL released during recv (py.detach)
```

All threads share `&ResolvedPlan` (immutable reference through Arc). No mutable shared state during the entity loop. Per-chunk state (`aliases`, `link_frequencies`, `ChunkStats`) is thread-local.

The chunk count (N × 8) is deliberately higher than thread count. With equal-sized chunks, the last-finishing chunk bottlenecks the entire job. With 8x oversubscription, rayon's work-stealing redistributes load — fast-finishing threads steal chunks from slow ones. This matters when entity processing time varies (some entities have many timestamps and neighbors, others have one).

### Entity Loop

```
execute_entity(plan, entity_node):
  1. Read entity PK value from rowpack
  2. Skip if duplicate PK (MPHF canonical check — ingest may create dupes)
  3. Read entity time_col and valid_to_col (if present)
  4. Evaluate entity filter (if present) — skip if fails
     Exception: if lag_timesteps > 0, keep filtered entities with include_bit=false
  5a. Static query: execute_stages once, write_target once
  5b. Temporal query:
      - Binary search: first_valid = partition_point(ts < time_col)
      - Binary search: last_valid = partition_point(ts <= valid_to)
      - For each ts in [first_valid, last_valid):
          clear aliases, clear link_frequencies
          execute_stages() → if passes filter → write_target()
```

### Stage Execution

Each stage contains ordered ops. Ops execute sequentially within a stage (ops may depend on earlier ops in the same stage via aliases). Results stored in `aliases: Vec<Option<Value>>` — indexed by alias number, pre-allocated to `num_aliases` size.

Aggregates are the expensive ops. Each aggregate:
1. Calls `traverse_and_accumulate()` which walks multi-hop edge paths
2. At each destination node: applies temporal window filter (sort key range), applies row filter
3. Accumulates via function-specific logic: COUNT, SUM, AVG, MIN, MAX, FIRST, LAST, CountDistinct, ListDistinct

Edge traversal (`ttgen_traversal.rs`) decodes edges directly from memory-mapped binary blobs — no deserialization, just bit manipulation on mmap'd data. Sort keys are encoded inline with destination node IDs for cache-friendly temporal filtering.

### Target Writing

`write_target()` inspects `PredictionType`:
- **Node**: One row per (entity, timestamp) with scalar target
- **Link (explode=true)**: One row per (entity, timestamp, target_item) — empty target lists produce 0 rows
- **Link (explode=false)**: One row per (entity, timestamp) with list-valued target column
- **PassThrough**: Only (entity, timestamp) columns — for LP RHS candidate generation

### LP Target FK Filtering

For `ListDistinct`, each target value is checked against the FK destination table's PK map before inclusion. Orphaned FK values (e.g., article_ids that don't exist in the articles table) are skipped.

The check is a single MPHF lookup (~40ns): hash value → shard → index into mmap'd data → validate. Lock-free — immutable memory-mapped data, no thread synchronization. The `target_pk_table` reference is resolved once at plan resolution time; the executor never does schema validation per row.

### Back-Pressure and Output

`ChannelBatchWriter` buffers rows into Arrow `RecordBatch`es (~1024 rows each) and sends via `crossbeam::bounded(16)`. When the channel is full (16 batches buffered), rayon workers block on send — this is the back-pressure mechanism. If the Python consumer (Parquet writer) falls behind, workers pause rather than accumulating unbounded memory.

## Artifact Production

**File**: `diskgraph-pybind/python/diskgraph/tt_artifacts.py`

`execute_tt_plan_with_artifacts()` produces:
1. **Parquet TT**: Training table with `arange` column appended
2. **Split assignment**: Binary blob (`DGSPLT` magic + row→stage mapping)
3. **Stats artifact**: `TTExecutionSummary` + plan JSON
4. **Split stats**: Per-stage row counts
5. **Train table col stats**: Column-level stats from native executor
6. **Train table edges**: FK edge parquets for downstream materialization
7. **Manifest**: JSON with artifact paths and SHA256 checksums

## Edge Artifact Export

**File**: `diskgraph-pybind/python/diskgraph/tt_artifacts.py` (in `export_tt_artifacts`)

For LP training tables, the TT has 2 foreign keys (entity FK + target FK). The export produces:
- `fkey_cols.parquet`: Maps each TT row to node indices in the destination tables
- Reverse edge parquets: COO format `(row, col)` for graph construction

When a target FK value doesn't resolve to a valid node (orphaned FK), the fkey_cols entry gets `-1`. The edge parquet drops these entries, causing a size mismatch between entity and target edges. This was the root cause of the HnM LP mask corruption bug (fixed in PR #173 by filtering orphaned FKs during TT gen).

## Integration with Kumo

### Invocation

**File**: `kumo/data/generic/diskgraph_train_table.py`

`generate_train_table_via_diskgraph()`:
1. Creates `DiskGraphPlanCompiler`, compiles PQL to plan JSON
2. Calls `execute_tt_plan_with_artifacts(plan_json, index_path, output_path, ...)`
3. Returns `(train_table_path, manifest_path)`

### Strict Path (native_full mode)

When `deploy_env().is_diskgraph == True`:
- DG does everything: TT gen + stats + splits + edge artifacts
- Spark is completely skipped
- `materialize_training_task` uses `_materialize_training_table_without_exact_stats()` with `skip_compute_stats=True`
- Artifacts validated via `get_validated_artifact_bundle()` (SHA256 checksums)

### Materialization Flow

```
materialize_training_table activity (TT gen workflow):
  → DG TT gen → parquet + manifest
  → execute_split → correct masks (train/val/test)
  → save_data → masks to gen-traintable-job/.../materialized/

materialize_training_task activity (training workflow):
  → load_data from materialized path (loads correct masks from .pt files)
  → task.materialize() → early return (has_data=True)
  → task._materialize() → add_table() for edge setup
  → task.save() → writes masks to trainingjob/.../tasks/NA/
```

Masks survive this flow because `load_data()` loads the correct `.pt` files from the first save, and `add_table()` doesn't modify them (as long as both fkey sizes match — guaranteed by the orphaned FK fix).

## Performance Summary

See [agent_design_and_performance.md](agent_design_and_performance.md) for general principles.

The key performance properties of TT gen are covered in the sections above. Summary of the design budget:

| Operation | Per-what | Cost | Notes |
|-----------|----------|------|-------|
| Plan resolution (string → handle) | Once | Milliseconds | Main thread only |
| Entity PK read | Per entity | ~10ns | Rowpack decode from mmap |
| MPHF canonical check | Per entity | ~40ns | Lock-free mmap read |
| Timestamp binary search | Per entity | O(log N) | `partition_point` |
| Stage op execution | Per entity × timestamp | Variable | Edge traversal dominates |
| Edge decode | Per edge | ~5ns | Bit manipulation on mmap |
| Sort key comparison | Per edge | ~3ns | Inline with edge data |
| FK existence check | Per target value (LP) | ~40ns | Lock-free MPHF lookup |
| Alias read/write | Per op | ~1ns | Vec index, stack-local |
| Channel send | Per ~1024 rows | ~100ns | Amortized, blocks on full |

Nothing on the per-entity or per-row hot path does allocation, string comparison, lock acquisition, or atomic operations. All shared data access is through immutable references (`&ResolvedPlan`, mmap reads).

## Developing: Correctness Considerations

### The Ground Truth Is Spark

DiskGraph TT gen must produce the same output as GraphEngine's Spark pipeline for the same query and data. "Same" means: same (ENTITY, TIMESTAMP, TARGET) triples, same column values, same row count. When they differ, DG is wrong until proven otherwise — verify against raw source data before concluding Spark is wrong.

Divergences found so far fall into categories (see [agent_debugging.md](agent_debugging.md) for the full list):
- **Spark filters implicitly, DG doesn't**: Orphaned FKs dropped by Spark JOIN (#173), NULLs dropped by Spark in aggregations (#139), empty accumulators dropped by Spark INNER JOIN (#52)
- **Timestamp unit mismatches**: Seconds vs nanoseconds at boundaries (#172, #170)
- **Different aggregation semantics**: `>=` vs `>` at temporal boundaries (#170)

### Testing Changes to the Executor

When modifying the entity loop, stage execution, or target writing:

1. **Run the integration tests**: `cargo test --workspace` covers 200+ tests including TT gen integration tests that compare against known-good outputs.

2. **Compare against GE on a real dataset.** Use the citi benchmark (small, 249 TT rows) for fast iteration. Diff the TT parquets row-by-row:
   ```python
   ge = pd.read_parquet("/path/to/ge/tt")
   dg = pd.read_parquet("/path/to/dg/tt")
   ge_set = set(zip(ge['ENTITY'].astype(str), ge['TIMESTAMP'].astype(str), ge['TARGET'].astype(str)))
   dg_set = set(zip(dg['ENTITY'].astype(str), dg['TIMESTAMP'].astype(str), dg['TARGET'].astype(str)))
   assert ge_set == dg_set, f"GE-only: {len(ge_set - dg_set)}, DG-only: {len(dg_set - ge_set)}"
   ```

3. **Test multiple query types.** Regression, binary classification, link prediction (exploded), multiclass — each takes a different path through `write_target()` and has different edge cases.

4. **Test with production-like data.** Synthetic benchmarks often have clean data (no orphaned FKs, no NULL columns, no empty target lists). Production datasets have all of these. HnM (31M transactions, orphaned article FKs), Faire (564M engagements, high-cardinality retailers), and Intuit (fraud detection, temporal with create_time) are good coverage.

### Adding a New Aggregation Function

1. Add the variant to `AggFunc` enum in `diskgraph-core/src/tt_builder.rs`.
2. Add the Python builder method in `diskgraph-pybind/src/tt_builder.rs`.
3. Add the accumulation logic in `execute_aggregate()` in `ttgen_executor.rs`. Follow the existing pattern: traverse, apply row filter, read column value, accumulate. Handle NULL values explicitly — `read_column_value_nullable()` for proper NULL propagation.
4. Add resolution logic in `ttgen_resolve.rs` if the new function needs additional resolved state (like `target_pk_table` for ListDistinct).
5. Add compiler support in `compiler.py` — map the PQL AST node to the new `AggFunc`.
6. Write integration tests: one per query type that uses the new function. Compare against Spark output.

### Adding a New Op Type

1. Add variant to `Op` enum (`tt_builder.rs`), `ResolvedOp` enum (`ttgen_resolve.rs`).
2. Add resolution in `resolve_op()`.
3. Add execution in `execute_stages()` — the op must write its result to `aliases[alias_idx]`.
4. If the op can be a target (the final output), update `write_target()` and `infer_target_dtype()`.

### Changes That Affect Edge Artifacts

The edge artifact export (`export_tt_artifacts`) reads the TT parquet and builds fkey_cols + reverse edge parquets. Changes to TT gen that affect:
- **Row count** (adding/removing rows): Edge artifacts must match. The fkey_cols parquet has one row per TT row.
- **Column schema** (adding/removing/renaming columns): The `arange` column and FK columns are read by name.
- **FK resolution** (which node IDs target values map to): Affects fkey_cols values. Entries with unresolvable FKs get `-1` and are dropped from edge parquets — both fkey edges must have the same count after dropping, otherwise the LP BP codepath in `graph_data.py:add_table()` resets masks.

### The Two-Save Mask Problem

If your change affects how `materialize_training_table` or `materialize_training_task` handle the training table, be aware: masks are written twice. The first write (in `materialize_training_table`) produces correct masks. The second write (in `materialize_training_task` via `task.save()`) must not overwrite them with incorrect values. Between the two saves, `task._materialize() → add_table()` runs — and for LP, it processes fkey edges which can modify `table_data` (masks, fkey tensor, num_nodes). Test by checking the mask .pt files on disk after training materialization completes.
