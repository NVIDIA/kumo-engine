# TT-gen internals (`diskgraph-train-table`)

This document explains the training-table generator used by DiskGraph predictive queries.
It is intended for both developers and coding agents working in this repo.

`diskgraph-train-table` executes a resolved predictive-query plan against an ingested DiskGraph index and streams Arrow `RecordBatch` output for materialization into parquet and downstream artifacts.

## Scope

This crate is responsible for:

- opening a local DiskGraph index
- resolving a `DiskGraphTTPlan` into table/relation/column handles
- iterating entity nodes, optionally across temporal anchor timestamps
- traversing graph edges and evaluating aggregations / filters
- emitting output batches for node prediction, link prediction, pass-through, and forecasting
- producing execution summary stats used by the artifact pipeline

This crate does **not**:

- parse PQL itself
- compile the user query into a plan
- own final parquet materialization details like `arange` append and manifest assembly
- own Python-side split assignment reconstruction for all split families

## Main entry points

### Rust

Primary executor APIs live in `src/ttgen_executor.rs`:

- `execute(plan, index_path)`
  - returns `(Receiver<RecordBatch>, JoinHandle<TTGenResult<u64>>)`
  - convenience wrapper when only the row count is needed
- `execute_with_summary(plan, index_path)`
  - returns `(Receiver<RecordBatch>, JoinHandle<TTGenResult<TTExecutionSummary>>)`
  - preferred entry point for consumers that need stats payloads

Error type:

- `src/errors.rs`
  - `TTGenError::SchemaError`
  - `TTGenError::InvalidPlan`
  - `TTGenError::Io`
  - `TTGenError::SystemError`

### Plan contract

The executor consumes `diskgraph_core::tt_builder::DiskGraphTTPlan`.
That plan is the boundary between Python compilation/builder logic and Rust execution.
See:

- `diskgraph-core/src/tt_builder.rs`
- `diskgraph-pybind/src/tt_builder.rs`
- `diskgraph-pybind/python/diskgraph/compiler.py`

### Python

Relevant Python-facing entry points:

- `diskgraph-pybind/src/tt_executor.rs`
  - `execute_tt_plan(...)`
- `diskgraph-pybind/python/diskgraph/executor.py`
  - `generate_training_table(...)`

Pybind error mapping is defined in `diskgraph-pybind/src/lib.rs`:

- `SchemaError -> DiskgraphDataError`
- `InvalidPlan -> DiskgraphConfigError`
- `Io/SystemError -> DiskgraphSystemError`

Keep this contract stable when adding new error paths.

## Code map

- `src/ttgen_executor.rs`
  - top-level orchestration, entity loop, stage execution, output dispatch, summary handling
- `src/ttgen_resolve.rs`
  - converts string-based plan references into resolved handles and indices
- `src/ttgen_traversal.rs`
  - edge iteration, time-window binary search, rowpack column reads
- `src/ttgen_filter.rs`
  - row-filter and stage-filter evaluation
- `src/ttgen_output.rs`
  - buffered Arrow batch writer and output schemas
- `src/ttgen_stats.rs`
  - execution summary and split/stat reconstruction payloads
- `src/ttgen_accumulator.rs`
  - accumulator helpers and value hashing
- `tests/ttgen_integration.rs`
  - end-to-end semantic coverage

## Execution pipeline

### 1) Open index and resolve the plan

`execute_with_summary()` opens `index_metadata.json` via `GraphIndexReader::open(...)`, then calls `resolve_plan(...)`.

Resolution does all string-to-handle work once on the main thread:

- entity table lookup
- entity PK/time/valid_to column resolution
- alias index assignment
- edge relation resolution
- target dtype inference
- stage and filter resolution
- temporal timestamp sorting
- entity-column dependency collection for filter RHS evaluation

Important details:

- relation lookup disambiguates same-src/same-dst relations using relation names and the edge-ref columns
- plan timestamps are defensively sorted ascending before execution
- if the entity spec omits `time_col` / `valid_to_col`, resolution falls back to index metadata `create_time_col` / `end_time_col`

### 2) Chunk the entity space

The executor parallelizes over entity node ID ranges:

- `num_chunks = min(num_threads * 8, num_entities).max(1)`
- each chunk is processed by rayon
- each worker gets its own `ChannelBatchWriter`, alias buffer, and stats accumulator

This design keeps shared state small:

- immutable resolved plan is wrapped in `Arc`
- workers only share the bounded `crossbeam_channel::Sender<RecordBatch>`

### 3) Iterate entities

For each entity node, the executor:

1. reads the entity PK for output
2. skips duplicate-PK nodes using `is_canonical_node(...)`
3. reads entity `time_col` / `valid_to_col` when present
4. pre-resolves entity columns referenced by filters
5. evaluates the entity-level filter
6. emits either:
   - one row for static plans, or
   - one row per valid anchor timestamp for temporal plans

#### Duplicate PK handling

Ingest can assign multiple node IDs to duplicate PK values, but the MPHF points at the canonical row only.
TT-gen therefore skips non-canonical nodes to avoid duplicate output rows.

### 4) Determine valid temporal anchors

For temporal plans, TT-gen uses the sorted anchor list and binary-search boundaries:

- first valid anchor: `timestamp >= entity.time_col`
- last valid anchor: `timestamp <= entity.valid_to_col`

Behavior:

- missing `time_col` means the entity is valid from the start
- missing `valid_to_col` means the entity never expires
- if an entity has a configured `time_col` but its value is null in a non-temporal plan, the row is dropped

### 5) Execute stages

Stages are flat and ordered. Each stage has:

- `ops: Vec<Op>`
- optional `StageFilter`

Each op writes into an alias slot. After all ops in a stage run, the stage filter is evaluated. If it fails, remaining stages are skipped and no output row is emitted for that `(entity, timestamp)` pair.

Supported op families:

- `Aggregate`
- `HasEdges`
- `Column`
- `Compare`
- `LogicalOp`
- `Autoregression`

## Traversal and temporal windows

### Edge iteration

`iter_edge(...)` in `ttgen_traversal.rs` decodes relation entries from the index.
Edges are assumed to be sorted by sort key in descending order (newest first).

When a temporal window exists, TT-gen binary-searches the matching edge slice.
The window semantics are:

- `(lo, hi]`
- lower bound exclusive
- upper bound inclusive

This matches Kumo temporal join semantics.

### First hop only

Window filtering is applied only on the **first** hop of a path.
That matches the expected temporal join shape where the first hop is the reverse-FK edge carrying the time sort key.
Subsequent hops traverse all edges.

### Fixed vs calendar windows

Resolved windows are represented as:

- `ResolvedWindow::Fixed`
  - seconds/minutes/hours/days/weeks
- `ResolvedWindow::Calendar`
  - months/years

Calendar offsets use `chrono::Months` so they match `pd.DateOffset(months=N)` semantics instead of treating months as a fixed number of days.

## Filter semantics

There are two filter systems.

### Row filters

`RowFilter` runs during traversal on destination rows.
It can compare:

- local destination columns
- remote values reached by a sub-traversal
- entity columns pre-resolved for the current entity
- literal constants

Null behavior follows SQL-ish tri-state conventions:

- `IS NULL` / `IS NOT NULL` are handled explicitly
- any other comparison with a null side fails the filter

### Stage filters

`StageFilter` runs after a stage finishes.
Its LHS is always a computed alias.
RHS can be:

- constant
- entity column
- remote value
- another alias

If a stage filter fails, later stages do not execute.

`NonEmptyList` is the explicit list-valued admission filter. It passes only for a present `Value::List` with at least one item and is used when empty-list validity depends on the compiled task semantics.

### `HasEdges`

`HasEdges` is the fast existence-check variant.
It traverses until the first matching row and returns a boolean.
Use this when a query only needs existence semantics instead of a full aggregate.

## Aggregate semantics

Implemented aggregate functions:

- `Count`
- `Sum`
- `Avg`
- `Min`
- `Max`
- `CountDistinct`
- `ListDistinct`
- `First`
- `Last`

Important details:

- `Count` counts matching traversed rows after row-filter application
- `Sum` returns `0.0` for an empty traversal, while `Avg`/`Min`/`Max` return `None`
- `CountDistinct` hashes `Value` instances
- `ListDistinct` preserves first-seen order and also captures per-value occurrence counts for link-prediction frequency output
- `ListDistinct` returns a present empty `Value::List` when no values match; consumers must not globally reinterpret that result as null or missing
- `limit` provides early termination, mainly for `ListDistinct`
- because edges iterate newest-first:
  - `Last` = first matching item encountered
  - `First` = keep overwriting until traversal finishes

## Prediction/output modes

The output schema is selected from `PredictionType` plus inferred dtypes.
Writer logic lives in `src/ttgen_output.rs`.

### Node prediction

Columns:

- `ENTITY`
- `TIMESTAMP` (Arrow timestamp ns)
- `TARGET`
- optional `ENTITY_INCLUDE_BIT`

### Link prediction, exploded

One row per distinct target.

Columns:

- `ENTITY`
- `TIMESTAMP`
- `TARGET`
- optional `__kumo_internal_frequency_column__`
- optional `ENTITY_INCLUDE_BIT`

### Link prediction, array

One row per `(entity, timestamp)` with list-valued target output.

Columns:

- `ENTITY`
- `TIMESTAMP`
- `TARGET: List<...>`
- optional `__kumo_internal_frequency_column__: List<Int64>`
- optional `ENTITY_INCLUDE_BIT`

### Pass-through

Columns:

- `ENTITY`
- `TIMESTAMP`
- optional `ENTITY_INCLUDE_BIT`

Used for candidate-generation style flows.

### Forecasting

Columns:

- `ENTITY`
- `TIMESTAMP`
- `AUTOREGRESSION_COL`
- optional `YOY_COL`
- optional `ENTITY_INCLUDE_BIT`

### Timestamp unit in output

Internally the executor works with integer anchor timestamps and emits them as epoch nanoseconds by multiplying by `1_000_000_000` during batch flush.
This matches Kumo's `datetime64[ns]` convention on the Python side.

## `ENTITY_INCLUDE_BIT`

When `lag_timesteps > 0`, entities that fail the entity filter are not fully dropped.
Instead TT-gen can still emit rows with `ENTITY_INCLUDE_BIT=false` so downstream models can retain autoregressive context while excluding those rows from train-stage counting.

When `lag_timesteps <= 0`, entity filter failure drops the entity entirely.

## Stats and split payloads

Execution summary logic lives in `src/ttgen_stats.rs`.

`TTExecutionSummary` includes:

- total rows written
- exact stats payloads for entity/timestamp/target/frequency/include-bit
- split payload metadata used by downstream artifact assembly

Current split payload behavior:

- some split families are marked `supports_native_execution=true`
- some are counts-only reconstruction
- `RandomSplit` / `RandomLinkSplit` are currently non-authoritative for native mask production
- `ManualSplit` reconstruction only works when splitting on `TIMESTAMP` or `PSEUDO_TIMESTAMP` and values are parseable epoch-second integers

This split payload is a summary/reconstruction aid, not the full replacement for all downstream split-assignment logic.

## Materialization boundary

This crate streams `RecordBatch` output only.
The pybind layer handles follow-on steps such as:

- consuming the batch stream
- appending contiguous `arange`
- writing parquet
- building split stats / train-table stats artifacts
- optional edge-bundle artifact assembly

If you change TT output schemas or stats payloads, audit both:

- `diskgraph-pybind/src/tt_executor.rs`
- `diskgraph-pybind/python/diskgraph/tt_artifacts.py`

## Performance model

Key design choices:

- one-time plan resolution
- per-worker reusable alias buffers
- bounded channel backpressure (`16` batches)
- batch flush at `65536` rows
- no intermediate heap allocation for path traversal beyond the decoded edge vector per hop
- per-chunk stats merged after execution

Known hotspots typically include:

- rowpack column decoding
- edge decoding in `iter_edge(...)`
- row-filter evaluation inside traversal loops
- repeated aggregation over many timestamps for temporal plans

## Common gotchas

### 1) Relation ambiguity

When multiple relations connect the same src/dst tables, `resolve_path()` relies on relation-name heuristics using `src_col` and `dst_col`.
If you change ingest-side relation naming, verify TT resolution still disambiguates correctly.

### 2) Null handling is intentional

Do not casually replace `Option`-based filter behavior with default values.
The current executor depends on null-aware semantics to match Kumo behavior.

### 3) `TemporalSpec.timestamps` and output timestamp conversion

TT-gen executes over the integer anchor timestamps carried in the plan and converts them to Arrow nanosecond timestamps only at output flush time.
If you change timestamp units or plan encoding, audit both the executor and the Python consumers together.

### 4) `ListDistinct` order matters

`ListDistinct` preserves encounter order, and frequency output is aligned with that order.
If you refactor it, keep values and counts in lockstep.

### 5) `First` / `Last` are easy to invert

Because traversal is newest-first, `Last` is the first encountered matching edge, while `First` must scan through the full traversal.

## Testing guidance

Primary semantic coverage lives in:

- `diskgraph-train-table/tests/ttgen_integration.rs`

Coverage includes:

- basic aggregations
- row filters and stage filters
- compare ops and logical ops
- entity features
- temporal windows and calendar months
- multi-hop traversal
- link prediction modes
- `ListDistinct` limits and duplicates
- `lag_timesteps` / `ENTITY_INCLUDE_BIT`
- batch prediction execution modes

When changing semantics, add or update integration tests first.
For localized behavior, also add unit tests in the touched module.

## Contributing checklist

When touching TT-gen:

- preserve typed `TTGenError` behavior
- preserve pybind exception mapping
- keep `RecordBatch` schema changes synchronized with Python consumers
- verify temporal boundary semantics `(lo, hi]`
- verify `First`/`Last` behavior under descending edge order
- verify stats payload and split payload compatibility
- run the relevant Rust tests, especially `ttgen_integration`

## Recommended read order for future changes

1. `diskgraph-core/src/tt_builder.rs`
2. `diskgraph-train-table/src/ttgen_resolve.rs`
3. `diskgraph-train-table/src/ttgen_executor.rs`
4. `diskgraph-train-table/src/ttgen_traversal.rs`
5. `diskgraph-train-table/src/ttgen_filter.rs`
6. `diskgraph-train-table/src/ttgen_output.rs`
7. `diskgraph-train-table/src/ttgen_stats.rs`
8. `diskgraph-train-table/tests/ttgen_integration.rs`

That path gives the fastest accurate mental model of the subsystem.
