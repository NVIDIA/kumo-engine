# DiskGraphTTBuilder: Design & Rationale

This document explains the design of the `DiskGraphTTPlan` intermediate
representation and the `DiskGraphTTBuilder` Python API. Every structural
decision is traced back to either a PQL grammar constraint, an execution
requirement, or a codebase constraint.

## Context

Kumo's current training-table pipeline:

```
PQL string
  -> Python parser (ANTLR4) -> AST
  -> Python semantic analysis (name resolution, type inference, FK path finding)
  -> Python plan compiler -> Spark execution plan
  -> Spark materialization (SQL joins, aggregations)
  -> GraphEngine (RocksDB) storage
  -> PyG tensors -> ML training
```

DiskGraph replaces the Spark + GraphEngine portion. The graph topology
is already in the DiskGraph index — edges can be traversed directly
without SQL joins. The question is where to draw the execution boundary.

## Decision 1: Execution boundary

Three options:

**A. DiskGraph as pure storage.** Keep Spark for execution, DiskGraph
replaces RocksDB. This doesn't solve the core problem: Spark
materialization (shuffles, intermediate tables) is the bottleneck, not
storage. The graph topology is in the index — traversal shouldn't
require a join.

**B. DiskGraph as execution engine.** Python sends a plan, Rust
executes end-to-end. One boundary crossing (plan in, results out).
Rust can short-circuit, pipeline, parallelize internally.

**C. Hybrid.** Rust exposes primitives, Python orchestrates step by
step. N boundary crossings for N stages. Rust can't optimize across
calls.

**Choice: B.** The performance win requires Rust to see the whole plan
at once. Short-circuiting (skip remaining computation when an entity
fails a filter) is impossible if Python orchestrates step by step.

## Decision 2: What crosses the boundary

Python builds a plan, Rust executes it. What format?

**A. Pass the PQL AST.** Rust must understand PQL semantics (a moving
target). Tight coupling — every PQL grammar change requires Rust
changes.

**B. Pass a normalized execution plan.** Python does semantic analysis
(name resolution, FK path finding, type checking). Rust receives a
fully-resolved plan with no ambiguity. Python = compiler, Rust =
runtime.

**C. Pass SQL.** Re-invents the materialization model we're escaping.

**Choice: B.** The plan format is a stable contract. Either side can
evolve independently. The plan is self-contained — the executor needs
only the plan and the index, not the original schema.

Note: this doesn't preclude moving parsing/semantics to Rust later. The
plan format doesn't care who produces it. Today Python produces it
because Kumo's semantic layer (data catalog, schema resolution, FK path
finding) is deeply integrated with the Python platform. Moving it to
Rust would be a large rewrite with no immediate performance benefit
(parsing runs once, in milliseconds).

## PQL: the source language

PQL is defined by an ANTLR4 grammar (`kumo/kumo/pquery/grammar/PQLGrammar.g4`).
Every query has one shape:

```
PREDICT <target> [RANK|CLASSIFY] [TOP k]
FOR [EACH] <entity>
[WHERE <entity_filter>]
[ASSUMING <whatif_condition>]
```

### Target

```
target: condition | aggregation | column;
```

Three cases:
- **column**: `PREDICT LOAN.STATUS FOR EACH LOAN.LOAN_ID` — direct lookup, with an optional validated many-to-one relationship path when the column belongs to a related table.
- **aggregation**: `PREDICT SUM(ORDERS.AMOUNT, 0, 30, days) FOR EACH USERS.USER_ID`
  — traverse edges, aggregate.
- **condition**: `PREDICT LOAN.STATUS = "paid" FOR EACH LOAN.LOAN_ID` — boolean.

### Aggregation

```
aggregation:
    AGGR '(' (column | filtered_column) ',' (INT | NEG_INFTY) ',' INT ')'
    | AGGR '(' (column | filtered_column) ',' (INT | NEG_INFTY) ',' INT ',' TIME_UNIT ')'
    | AGGR '(' (column | filtered_column) ')';
```

The inner argument is `column | filtered_column` — never another
aggregation. **Nested aggregations (`SUM(AVG(X))`) are rejected by the
grammar**, not by semantic analysis. Extending the grammar to allow
nesting is syntactically trivial but semantically deep (the inner
aggregation needs a grouping key that PQL has no concept of).

Functions: `SUM`, `AVG`, `MIN`, `MAX`, `COUNT`, `COUNT_DISTINCT`,
`FIRST`, `LAST`, `LIST_DISTINCT`.

### Filtering in PQL

Filtering appears in four places with two distinct semantics:

**Row-level (check a column on a single row, no computation needed):**

1. Inside aggregation: `SUM(ORDERS.AMOUNT WHERE ORDERS.STATUS = 'paid', 0, 30)`.
   Filters which rows to include during traversal.
2. Entity WHERE, simple column: `FOR EACH USERS.ID WHERE USERS.AGE > 25`.
   Filters which entities to consider.

**Result-level (check a computed value, ops must execute first):**

3. Entity WHERE with aggregation:
   `FOR EACH USERS.ID WHERE SUM(TRANS.PRICE, -60, 0) > 0`.
   Requires traversal + aggregation before the filter can evaluate.
4. ASSUMING clause: `ASSUMING COUNT(ORDERS.*, 0, 30) > 0`.
   Post-computation filter on the target or related aggregation.

Both ASSUMING and the entity WHERE can reference any table reachable
from the entity via foreign keys, not just the target table.

### Joins

No explicit JOIN syntax. Multi-table references (`PREDICT ORDERS.AMOUNT
FOR EACH USERS.USER_ID`) are resolved by the Python semantic layer,
which finds FK paths in the graph schema. The plan receives explicit
edge references — the executor never does name resolution.

## Deriving the IR

The IR is designed bottom-up from the grammar, optimizing for
completeness (covers all PQL constructs), minimality (no redundant
types), and executability (the backend needs no further analysis).

### Leaf types

Direct encoding of grammar terminals:

```rust
enum Value { Int(i64), Float(f64), Str(String), Bool(bool), List(Vec<Value>) }
enum AggFunc { Count, Sum, Avg, Min, Max, CountDistinct, ListDistinct }
enum CompareOp { Eq, Neq, Gt, Gte, Lt, Lte, IsNull, IsNotNull, StartsWith, EndsWith, Contains, NotContains, In }
enum TimeUnit { Seconds, Minutes, Hours, Days, Weeks, Months, Years }
```

### Edge references

The grammar infers joins from FK relationships. The IR makes them
explicit — each hop is a directed edge between two (table, column)
pairs:

```rust
struct EdgeRef {
    src_table: String, src_col: String,
    dst_table: String, dst_col: String,
}
```

Multi-hop traversal: `Vec<EdgeRef>`. The Python compiler resolves the
FK path and emits explicit hops. The executor follows them without
schema access.

### Operations

Each grammar production that *computes a value* maps to an op:

**`column` (bare column target):**

`PREDICT LOAN.STATUS FOR EACH LOAN.LOAN_ID` reads a column value for the entity row. A bare related-table column may traverse a validated many-to-one relationship first. One entity still produces at most one value, without aggregation.

```rust
Op::Column { alias: String, table: String, column: String, path: Vec<EdgeRef> }
```

No `row_filter` field because there's nothing to filter. An empty path reads the entity row directly. A non-empty path follows a validated relationship from the entity row before reading the column; this is valid for unique many-to-one lookup targets. One-to-many related-table targets still require an aggregation to define which related rows contribute.

**`aggregation`:**

Traverse from entity to target table, optionally filter rows, aggregate.

```rust
Op::Aggregate {
    alias: String,
    path: Vec<EdgeRef>,
    func: AggFunc,
    target_col: Option<String>,  // None = COUNT(*)
    time_col: Option<String>,
    window: Option<(i64, i64)>,
    window_unit: Option<TimeUnit>,
    row_filter: Option<RowFilter>,
}
```

`target_col: Option` because `COUNT(ORDERS.*)` counts rows without a
specific column. `path: Vec<EdgeRef>` handles both single-hop
(`[A->B]`) and multi-hop (`[A->B, B->C]`) without a separate op type.

**`condition` (boolean comparison):**

`PREDICT LOAN.STATUS = "paid"` or as part of ASSUMING. Always built on
a prior result — takes an alias and compares to a constant.

```rust
Op::Compare {
    alias: String,
    lhs: String,        // alias of a prior op
    op: CompareOp,
    rhs: Value,
}
```

References prior ops by alias (SSA-style). The executor maintains an
`alias -> value` map. Flat references are simpler to execute than nested
expression trees.

### Why three ops and not more

Every PQL target is one of: column lookup, aggregation, or condition.
Aggregation is parameterized by function — `SUM`, `COUNT`, `AVG` share
the same structure (traverse, filter, accumulate), differing only in the
accumulator. Separate `CountEdges`, `SumColumn`, `AvgColumn` ops would
proliferate types that share identical structure.

### Two filter types

Filters have the shape `LHS op RHS`. The LHS and RHS types differ
between the two filter types because they operate at different execution
phases:

**RowFilter** — evaluated per-row during edge traversal (inner loop):

```rust
enum RowFilterLhs {
    Column(String),                                    // column on current table
    Remote { path: Vec<EdgeRef>, column: String },     // column via sub-traversal
}
enum RowFilterRhs {
    Constant(Value),                                   // literal
    EntityColumn(String),                              // entity table column
    Remote { path: Vec<EdgeRef>, column: String },     // column via sub-traversal
}
enum RowFilter {
    Compare { lhs: RowFilterLhs, op: CompareOp, rhs: RowFilterRhs },
    And(Vec<RowFilter>), Or(Vec<RowFilter>), Not(Box<RowFilter>),
}
```

**StageFilter** — evaluated per-entity between stages (outer loop):

```rust
enum StageFilterRhs {
    Constant(Value),                                   // literal
    EntityColumn(String),                              // entity table column
    Remote { path: Vec<EdgeRef>, column: String },     // column via sub-traversal
    Alias(String),                                     // computed value from prior op
}
enum StageFilter {
    Compare { alias: String, op: CompareOp, rhs: StageFilterRhs },
    NonEmptyList { alias: String },
    And(Vec<StageFilter>), Or(Vec<StageFilter>), Not(Box<StageFilter>),
}
```

`NonEmptyList` expresses an explicit list-valued admission rule without relying on the generic convention that a numeric comparison against a list uses the list length. The compiler uses it for multilabel ranking targets, where an empty relevant-item set supplies no ranking supervision; multilabel classification does not use it because an empty set is a valid all-negative label.

**Why separate LHS types:** RowFilter LHS reads from the row being
traversed (a column or a remote lookup). StageFilter LHS is always a
computed alias. These can't be interchanged — aliases don't exist during
traversal, and row columns aren't accessible between stages.

**Why separate RHS types:** StageFilterRhs has `Alias` for comparing
two computed values (e.g. `WHERE SUM(...) > AVG(...)`). RowFilterRhs
does not — if the comparison involves a computed alias, the filter is
inherently stage-level, not row-level.

**Why LHS/RHS parameterisation instead of separate variants:** Each
filter comparison has the shape `<source₁> op <source₂>`. Without
parameterisation, each combination of LHS source × RHS source would be
a separate enum variant (2 × 3 = 6 for RowFilter). With
parameterisation, one `Compare` variant handles all combinations. Adding
a new source type scales additively (2+3 → 3+3) instead of
multiplicatively (6 → 9).

**Mapping from PQL (current and planned):**

| # | PQL pattern | Filter | LHS | RHS |
|---|---|---|---|---|
| 1 | `WHERE ORDERS.STATUS = 'paid'` (in agg) | Row | Column | Constant |
| 2 | `FOR EACH USERS.ID WHERE USERS.AGE > 25` | Row (entity) | Column | Constant |
| 3 | `WHERE SUM(...) > 0` | Stage | alias | Constant |
| 4 | `ASSUMING COUNT(...) > 0` | Stage | alias | Constant |
| 5 | `WHERE ORDERS.CURRENCY = USERS.PREF` | Row | Column | EntityColumn |
| 6 | `WHERE SUM(...) > USERS.CREDIT_LIMIT` | Stage | alias | EntityColumn |
| 7 | `WHERE MERCHANTS.COUNTRY = 'US'` (multi-hop) | Row | Remote | Constant |
| 8 | `WHERE ORDERS.CURR = MERCHANTS.DEFAULT` | Row | Column | Remote |
| 9 | `WHERE MERCHANTS.COUNTRY = USERS.COUNTRY` | Row | Remote | EntityColumn |
| 10 | `WHERE SUM(...) > COMPANY.LIMIT` | Stage | alias | Remote |
| 11 | `WHERE SUM(...) > AVG(...)` | Stage | alias | Alias |

Rows 1–4 are expressible in PQL today. Rows 5–6 require entity=target
comparison (planned — small AST change). Rows 7–9 require multi-hop
filtering syntax (planned — requires recursive sampling in executor).
Rows 10–11 are natural combinations of the above.

### Stages

```rust
struct Stage {
    ops: Vec<Op>,
    filter: Option<StageFilter>,
}
```

A plan is `Vec<Stage>`. The executor processes stages in order. If a
stage's filter evaluates to false, all remaining stages are skipped.

**Why stages are required (not just an optimization):**

PQL's entity WHERE can contain aggregations:

```sql
PREDICT SUM(transactions.price, 0, 90, days)
FOR EACH customers.customer_id
WHERE SUM(transactions.price, -60, 0, days) > 0
ASSUMING COUNT(transactions.*, 0, 90, days) > 0
```

This produces three stages:

```
Stage 1: ops=[Aggregate("_entity_check", Sum, window=(-60,0))]
         filter=_entity_check > 0

Stage 2: ops=[Aggregate("_assumption", Count, window=(0,90))]
         filter=_assumption > 0

Stage 3: ops=[Aggregate("target", Sum, window=(0,90))]
         filter=None
```

Stage 1 cannot be a `RowFilter` on `EntitySpec` because it requires
traversal and aggregation — it's not a simple column check. It must be
an op + StageFilter in its own stage.

**Stages also enable short-circuit optimization.** A compound filter
(`AND(A, B)`) evaluates both conditions after both ops execute. Separate
stages let the executor bail out between them — if the cheap check in
stage 1 fails, the expensive ops in stages 2-3 never run.

**Why not a flat `Vec<Op>` with filter annotations?** A failing filter
always means "skip everything remaining" (no partial skips). Stages make
this scope explicit. Within a stage, all ops always execute. The filter
is checked between stages. No "half-executed stage" state — simplifies
error handling, progress tracking, and debugging.

**Why not a tree?** PQL targets are simple enough that linear ordering
suffices. Stages are a linearized topological sort of the dependency
graph, with filters as cut points. A tree adds complexity for generality
PQL doesn't need.

### Top-level plan

```rust
struct DiskGraphTTPlan {
    entity: EntitySpec,
    temporal: Option<TemporalSpec>,
    prediction_type: PredictionType,
    stages: Vec<Stage>,
    top_k: Option<u32>,
}
```

- `entity`: from PQL `FOR EACH` + simple `WHERE` column filter.
  Includes both `time_col` (create-time) and `valid_to_col` for
  temporal filtering of seed entities (which are not visited via edges,
  so timestamps must be read from rowpack).
- `temporal`: from the training framework (timestamp windows, forecast
  length). Split labels (train/val/test) are a postprocessing concern
  and not included — the executor just produces rows for each
  `(entity, timestamp)` pair.
- `prediction_type`: `Node` or `Link { explode }`. For link prediction,
  the target entity info is already in the `ListDistinct` aggregate's
  `EdgeRef` path — no redundant fields needed.
- `stages`: linearized ops + filters, derived from PQL target +
  aggregation-based entity WHERE + ASSUMING.
- `top_k`: from PQL `TOP k`. Optimization hint for `ListDistinct` —
  executor can stop after k items.

## Codebase placement

**Types in `diskgraph-core/src/tt_builder.rs`.** `diskgraph-core` holds
shared contracts (metadata, sampling requests, features). Both the
pybind layer (Python -> Rust) and the future executor (Rust) import from
here.

**PyO3 builder in `diskgraph-pybind/src/tt_builder.rs`.** Separate
crate because PyO3 requires `crate-type = ["cdylib"]`, which cannot be
depended on by other Rust crates. `diskgraph-pybind` is a leaf — nothing
depends on it. It wraps `diskgraph-core` types with Python-friendly
method signatures and string-to-enum parsing.

## The builder pattern

`DiskGraphTTBuilder` is a mutable Python class. PQL is processed
incrementally — the Python compiler walks the AST and calls builder
methods (`for_each`, `aggregate`, `filter`, etc.). `build()` freezes
everything into an immutable `DiskGraphTTPlan`, serialized as JSON.

JSON because the plan crosses a process boundary. JSON is debuggable
(print, log, diff), language-agnostic, and the plan is small metadata.
Serialization cost is negligible vs execution.

## Naming

`DiskGraphTTBuilder` / `DiskGraphTTPlan`:

- `DiskGraph` prefix: disambiguates from kumoapi's `TrainingTableGenPlan`
  (Python, Spark-based). Engineers working on Kumo see both.
- `TT`: training table. `DiskGraphTrainingTableBuilder` is 35
  characters. Well-understood abbreviation in the Kumo codebase.
- `Builder` vs `Plan`: builder is mutable (Python accumulates ops),
  plan is immutable (Rust consumes it). Separate names for separate
  lifecycles.

## Extensibility

Adding a new op: add a variant to `Op`. Serde handles backward
compatibility (unknown variants can be handled gracefully).

Adding nested aggregations: would require a new op variant (e.g.,
`Op::NestedAggregate` with inner/outer functions and a grouping key for
the intermediate entity). The grammar change is one production; the
semantic implications are deep (what does the inner aggregation group
by?).

Moving parsing to Rust: the plan format is the contract. Replace the
Python builder with a Rust parser that emits the same `DiskGraphTTPlan`.
The executor doesn't change.
