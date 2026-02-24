# TT-Gen Executor Specification

## Overview

The TT-gen executor takes a **declarative query plan** (`DiskGraphTTPlan`) and a **graph index** (on-disk directory), and produces **output Parquet shards** containing training table rows.

Each output row has exactly 3 columns:
- `entity_id` (u64): the node ID of the entity
- `timestamp` (i64): the observation timestamp (0 for non-temporal plans)
- `target` (f64): the final computed value

## Graph Model

The graph consists of:
- **Node types** (tables): each has a name, a set of nodes (0..max_id), and optional typed columnar data per node (called "node data")
- **Edge relations**: directed edges from a source node type to a destination node type. Each edge connects a source node to a destination node. Edges may carry a **sort key** derived from a timestamp column, enabling time-ordered binary search.

Node data columns have types: Int64, Float64, String, Bool, Timestamp (stored as epoch seconds).

Edges are stored per source node, sorted by sort key (ascending). A node may have 0 or more outgoing edges per relation.

## Plan Structure

```
DiskGraphTTPlan {
    entity: EntitySpec,          // which table is the entity
    temporal: Option<TemporalSpec>,  // timestamps for temporal plans
    prediction_type: PredictionType, // Node or Link
    stages: Vec<Stage>,          // ordered stages to execute
}
```

### EntitySpec

```
EntitySpec {
    table: String,          // entity node type name
    column: String,         // primary key column (metadata only, not used in execution)
    time_col: Option<String>,      // column on entity table: entity "birth" time
    valid_to_col: Option<String>,  // column on entity table: entity "expiry" time
    filter: Option<RowFilter>,     // entity-level pre-filter
}
```

### TemporalSpec

```
TemporalSpec {
    timestamps: Vec<TimestampEntry>,  // sorted ascending by start
    forecast_length: i64,
    lag_timesteps: Option<i64>,
}

TimestampEntry {
    start: i64,   // observation timestamp (epoch seconds)
    end: i64,     // end of forecast window
}
```

### Stage

```
Stage {
    ops: Vec<Op>,                    // operations to execute (in order)
    filter: Option<StageFilter>,     // filter applied after all ops in this stage
}
```

### Op (Operation)

Each op produces a named value (alias) that is stored and can be referenced later.

```
Op::Aggregate {
    alias: String,
    path: Vec<EdgeRef>,        // edge traversal path (1+ hops)
    func: AggFunc,             // aggregation function
    target_col: Option<String>,// column to aggregate on destination table
    time_col: Option<String>,  // (metadata, not used in execution)
    window: Option<(i64, i64)>,// relative time window (lo, hi) offsets
    window_unit: Option<TimeUnit>, // unit for window offsets
    row_filter: Option<RowFilter>, // per-row filter during traversal
    limit: Option<u32>,        // early-exit limit (for ListDistinct)
}

Op::Column {
    alias: String,
    table: String,     // which table to read from
    column: String,    // which column to read
}

Op::Compare {
    alias: String,
    lhs: String,       // alias name of a previously computed op
    op: CompareOp,     // comparison operator
    rhs: Value,        // constant to compare against
}
```

### EdgeRef

```
EdgeRef {
    src_table: String,
    src_col: String,   // (metadata only)
    dst_table: String,
    dst_col: String,   // (metadata only)
}
```

### AggFunc

- `Count` — count of edges (after filters). No target_col needed. Result: integer.
- `Sum` — sum of target_col values across edges. Result: float.
- `Avg` — mean of target_col values. Result: float. Returns NaN if 0 edges.
- `Min` — minimum of target_col values. Result: float. Returns NaN if 0 edges.
- `Max` — maximum of target_col values. Result: float. Returns NaN if 0 edges.
- `CountDistinct` — count of distinct target_col values. Result: integer.
- `ListDistinct` — collect distinct target_col values into a list. Result: list of values. When converted to f64 for output, the **list length** is used.
- `First` — value of target_col from the first edge (by sort key order). If no target_col, returns the destination node ID. Returns 0 if no edges.
- `Last` — value of target_col from the last edge (by sort key order). Same defaults as First.

### CompareOp

`Eq`, `Neq`, `Gt`, `Gte`, `Lt`, `Lte`

### Value

Runtime values used in filters and as computation results:
- `Int(i64)`
- `Float(f64)`
- `Str(String)`
- `Bool(bool)`
- `List(Vec<Value>)`

**Conversion to f64** (for output target column):
- Int → cast to f64
- Float → as-is
- Bool → true=1.0, false=0.0
- Str → 0.0
- List → length as f64

**Comparison rules**:
- Same numeric types: standard comparison
- Int vs Float: Int promoted to f64
- Str vs Str: lexicographic
- Bool vs Bool: false < true
- List vs Int/Float: list length compared against the number
- List vs List: compared by length
- Mismatched types (e.g., Int vs Str): always false

### TimeUnit

`Seconds`, `Minutes`, `Hours`, `Days`, `Weeks`, `Months`, `Years`

Window offsets are multiplied by the unit (e.g., Days → offset * 86400 seconds).

## Filters

### RowFilter (applied per-edge during traversal)

```
RowFilter::Compare { lhs: RowFilterLhs, op: CompareOp, rhs: RowFilterRhs }
RowFilter::And(Vec<RowFilter>)
RowFilter::Or(Vec<RowFilter>)
RowFilter::Not(Box<RowFilter>)
```

**RowFilterLhs** — where the left-hand value comes from:
- `Column(String)` — column on the current destination node being visited
- `Remote { path: Vec<EdgeRef>, column: String }` — follow a sub-traversal from the current destination node, read a column on the final node. Takes the first edge at each hop (FK semantics: many-to-one).

**RowFilterRhs** — where the right-hand value comes from:
- `Constant(Value)` — literal value
- `EntityColumn(String)` — column on the entity node (resolved once per entity, reused across all edges)
- `Remote { path: Vec<EdgeRef>, column: String }` — sub-traversal from the current destination node (same as LHS Remote)

### StageFilter (applied after all ops in a stage)

```
StageFilter::Compare { alias: String, op: CompareOp, rhs: StageFilterRhs }
StageFilter::And(Vec<StageFilter>)
StageFilter::Or(Vec<StageFilter>)
StageFilter::Not(Box<StageFilter>)
```

The `alias` field references a previously computed op result.

**StageFilterRhs**:
- `Constant(Value)` — literal value
- `EntityColumn(String)` — column on the entity node
- `Remote { path: Vec<EdgeRef>, column: String }` — sub-traversal from the entity node
- `Alias(String)` — reference another previously computed op

## Execution Semantics

### 1. Entity Loop

For each entity node (0..num_entities):

### 2. Entity Pre-filtering

If `entity.filter` is set, evaluate it on the entity node. If false, skip this entity entirely (no output rows).

The entity filter is a `RowFilter` where the "current table" is the entity table itself. `RowFilterLhs::Column` reads from the entity table. `RowFilterRhs::EntityColumn` also reads from the entity table (they're the same table in this context).

### 3. Entity Time Filtering (temporal only)

If `entity.time_col` is set, read the entity's time column value `T`. The entity only exists for timestamps where `timestamp.start >= T`. Formally: skip timestamps where `timestamp.start < T`.

If `entity.valid_to_col` is set, read the entity's valid_to column value `V`. The entity expires after `V`. Formally: skip timestamps where `timestamp.start > V`.

If the time column or valid_to column is NULL (node has no data or column is null), the constraint is not applied (entity exists for all timestamps / never expires).

The valid range of timestamps is the intersection of both constraints.

### 4. Temporal vs Non-Temporal

**Non-temporal** (`temporal` is None):
- Execute stages once per entity
- Output row has timestamp=0
- One row per entity (if not filtered out)

**Temporal** (`temporal` is Some):
- For each valid timestamp in the temporal spec:
  - Execute stages with the current observation timestamp
  - Output row has timestamp=ts.start
- One row per (entity, timestamp) pair (if not filtered out)

### 5. Stage Execution

Stages execute in order. Each stage:
1. Execute all ops in the stage (in order)
2. Evaluate the stage filter (if present)
3. If the stage filter returns false, **stop all stages** for this (entity, timestamp) — no output row

### 6. Op Execution

**Op::Aggregate**: Traverse the edge path starting from the entity node. At each final destination, optionally apply row_filter and accumulate according to func.

**Op::Column**: Read a column value from the specified table for the current entity node.

**Op::Compare**: Read the alias value (previously computed), compare against rhs constant, produce Bool result.

### 7. Edge Traversal (Aggregate)

The `path` field specifies one or more edge hops. For a single-hop path `[A→B]`:
- Iterate all edges from the entity in relation A→B
- For each edge, the destination node is visited

For a multi-hop path `[A→B, B→C]`:
- From the entity, iterate all edges in relation A→B
- For each intermediate node in B, iterate all edges in relation B→C
- The callback fires for each final destination in C

**Time Window** (only applied on the LAST hop):
- If `window` is set to `(lo, hi)` with unit:
  - Compute absolute bounds: `[observation_ts + lo*unit_secs, observation_ts + hi*unit_secs)`
  - Only edges whose sort key falls within this range are visited
  - The range is inclusive on the low end, exclusive on the high end
  - Binary search is used (edges are sorted by sort key)
- Intermediate hops (not the last) always iterate ALL edges (no window)

**Row Filter** (only applied on final destination):
- After reaching the final destination node, evaluate the row filter
- If false, skip this destination (don't accumulate)

### 8. Accumulation

During traversal, each visited (and not filtered) destination contributes to the accumulator:

- **Count**: increment counter
- **Sum**: add target column value (as f64)
- **Avg**: online mean (Welford's algorithm)
- **Min/Max**: track running min/max (as f64)
- **CountDistinct**: hash target column value, insert into set, return set size
- **ListDistinct**: hash target column value, if new insert into values list. If `limit` is set, stop traversal after collecting `limit` distinct values.
- **First**: take the first destination's target column value and stop
- **Last**: keep overwriting with each destination's target column value

For numeric reads: if a column is Int, it's cast to f64. If the column value is NULL or the node has no data, 0.0 is used.

### 9. Output

The **target** value is the result of the LAST op's alias in the LAST stage. It is converted to f64 for output.

A row is emitted only if ALL stages passed (no stage filter returned false) AND the target alias was computed (is not None).

### 10. Parallelism

Entities are partitioned across threads. Each thread writes to a separate Parquet shard. The output is the union of all shards. Entity processing is independent — no cross-entity state.

## Edge Cases and Boundary Conditions

1. **Entity with no edges**: Aggregate ops return their "empty" value (Count=0, Sum=0.0, Avg=NaN, Min=NaN, Max=NaN, CountDistinct=0, ListDistinct=empty list, First=0, Last=0).

2. **Entity with no node data**: Column reads return 0 (Int) or 0.0. EntityColumn filter RHS returns 0.

3. **NULL column values**: Treated as 0 for numeric operations. For CountDistinct/ListDistinct, NULL values are skipped (not counted).

4. **Empty timestamp list**: Temporal plan with empty timestamps produces no output.

5. **Time window with no matching edges**: Returns the "empty" value for the aggregation function.

6. **Stage filter failure**: All subsequent stages are skipped, no output row for this (entity, timestamp).

7. **Multiple ops in one stage**: All execute before the stage filter is evaluated. Later ops can reference aliases from earlier ops in the SAME stage.

8. **Cross-stage alias references**: StageFilter can reference aliases from the current or any prior stage. Op::Compare can reference aliases from any prior op.

9. **Multi-hop with window**: Window applies only to the last hop. Intermediate hops return all edges regardless of time.

10. **Remote sub-traversal in filter**: Follows FK edges (takes first edge at each hop). If no edge exists at any hop, returns the default value (0 for int, 0.0 for float).

11. **ListDistinct with limit**: Stops traversal entirely once `limit` distinct values are collected. The remaining edges are NOT visited.

12. **First with no target_col**: Returns the destination node ID as Int.

13. **Compare type mismatch in filter**: Returns false (e.g., comparing Int to Str).

14. **valid_to < time_col**: Entity has no valid timestamps (empty range). No output rows.

15. **Non-temporal plan with window**: Window offsets are relative to observation_ts. For non-temporal, observation_ts is conceptually 0, so the window is [0+lo, 0+hi). This is usually nonsensical but technically works.

## Test Infrastructure

The test builds an in-memory graph index on disk in a temp directory. The graph index consists of:
- `index_metadata.json` — dataset metadata (node types, edge relations, sort key config)
- `nodes/{table}/shard-00000/edges.bin` — binary edge data (or node data for leaf tables)
- `nodes/{table}/shard-00000/offsets.bin` — offset index
- `nodes/{table}/node_data.ctx` — rowpack schema for node data columns

The executor is invoked via `ttgen_executor::execute(plan, index_path, output_dir)`.

Output is read back from the Parquet shards and sorted by (entity_id, timestamp) for deterministic comparison.

### Available Graph Builders

**Basic graph** (`build_test_index`): 2 users (no node data) → 3 items (price: f64, category: i64). User 0 has 4 edges at timestamps 100, 200, 300, 400 to items 0, 1, 2, 0. User 1 has no edges.

**Extended graph** (`build_extended_test_index`): 3 users (user_level: i64, create_time: i64, user_score: i64) → 3 items (price: f64, category: i64). User 0 (level=1, time=50, score=100) has 4 edges. User 1 (level=2, time=100, score=200) has no edges. User 2 (level=1, time=500, score=300) has 1 edge to item 1 at ts=600.

**3-table graph** (`build_three_table_test_index`): Same as extended + brands table (brand_rating: i64). Items have FK edges to brands: item 0→brand 0, item 1→brand 1, item 2→brand 0. Brand 0 rating=25, brand 1 rating=15.

### Helper Functions
- `user_to_item_edge()` — single-hop edge path from users to items
- `item_to_brand_edge()` — single-hop edge path from items to brands
- `user_to_item_to_brand_path()` — two-hop path from users through items to brands
- `read_output_rows(shard_paths)` — reads Parquet shards into sorted Vec<OutputRow>
- `index_root(base)` — returns the index subdirectory path
