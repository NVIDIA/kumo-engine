# DiskGraph

DiskGraph is a graph ingestion, storage, sampling, and train-table-generation system. This glossary captures the domain language used to describe graph data, train-time artifacts, and how they connect.

## Language

**Parent graph snapshot**:
One finalized DiskGraph index state for a graph at a specific ingest generation.
_Avoid_: Current graph, live graph

**Seed table**:
A table of per-example task rows that anchors sampling by linking each row to a parent entity row.
_Avoid_: Target table

**Seed row**:
One staging row in a seed table that carries task-local supervision and links to one parent entity row.
_Avoid_: Graph node, real node

**Unresolvable seed row**:
A seed row whose `ENTITY1` value does not match any parent entity row in the parent graph snapshot.
_Avoid_: Bad node, invalid graph node

**Observation timestamp**:
The seed row timestamp that defines the history boundary visible from the parent graph for that example.
_Avoid_: Generic event time, target event time

**Target value**:
The task-local supervision carried by a seed row, such as the label or RHS value(s) consumed by the learning objective. In v1 (PR270), target values are stored as rowpack columns and not materialized as structural FK edges; future versions may materialize RHS target values as structural FK edges (a parallel `seed.edges.bin` for the target side).
_Avoid_: Assuming target values are permanently excluded from structural FK materialization; the v1 implementation stores them as supervision columns, not because the architecture forbids edges, but because RHS edge materialization is not yet implemented.

**DG-native seed-table path**:
The new DiskGraph runtime path that ingests a seed table into the parent graph snapshot and samples from that combined view.
_Avoid_: Legacy path, implicit fallback

**Training seed table**:
A seed table produced for training workloads and carrying training-time supervision semantics.
_Avoid_: Generic seed table object

**Prediction seed table**:
A seed table produced for prediction workloads and carrying prediction-time request semantics.
_Avoid_: Generic seed table object

**Seed-row feature**:
A task-local example feature carried on a seed row and consumed with the seed bag rather than as part of graph topology.
_Avoid_: Graph node feature, structural feature

**Parent-untouched invariant**:
The rule that seed-table ingest may create artifacts only under the seed-table subdirectory and must not modify the parent graph snapshot.
_Avoid_: In-place parent patching, parent augmentation

**Single-seed runtime view**:
The runtime view that opens exactly one seed table against exactly one parent graph snapshot.
_Avoid_: Multi-seed merged runtime

**Persisted seed-row id**:
The local row id assigned by the persisted seed table after ingest-time drops and filtering.
_Avoid_: Raw TT input row id, pre-ingest row position

**Runtime-ready seed table**:
A persisted seed table in which every stored row is immediately sampleable by the runtime.
_Avoid_: Best-effort staging dump, partially invalid seed table

**Transitional sidecar**:
An artifact kept only for compatibility with older consumers and not part of the core DG-native seed-table contract.
_Avoid_: Core runtime semantic artifact

**Seed-table extension artifact**:
A seed table understood only in relation to the parent graph snapshot it extends, not as a standalone dataset.
_Avoid_: Standalone graph dataset

## Relationships

- A **seed table** belongs to exactly one **parent graph snapshot**
- Each **seed row** links to exactly one parent entity row through `ENTITY1`
- A **seed row** is not part of the graph topology; sampling enters the real graph through its linked parent entity row
- An **unresolvable seed row** is dropped during seed-table ingest and counted, but does not make the ingest fail
- An **observation timestamp** on a **seed row** limits which parent-graph history is visible for that example
- A **target value** belongs to the **seed row** as supervision; in v1 it is not materialized as a structural FK edge, but future versions may generate structural FK edges for RHS target values
- The **DG-native seed-table path** supports exactly one structural entity slot in v1 and rejects unsupported shapes explicitly
- A **training seed table** and a **prediction seed table** are distinct domain objects even when they share the same storage pattern
- A **seed-row feature** belongs to the **seed row** as task-local example data and is supported in v1
- The **parent-untouched invariant** requires the parent graph snapshot to remain byte-for-byte unchanged during seed-table ingest
- A **single-seed runtime view** combines exactly one **seed table** with exactly one **parent graph snapshot**
- A **prediction seed table** normally omits `TARGET1`; absence of supervision is part of its shape
- Each persisted **seed row** has exactly one structural entry link into the parent graph
- A **persisted seed-row id** is the runtime identity used after ingest-time dropping and filtering
- A **runtime-ready seed table** stores only rows that are immediately sampleable after ingest
- A **transitional sidecar** may accompany a seed table without defining the seed table's DG-native meaning
- A **seed-table extension artifact** is meaningful only together with its parent graph snapshot

## Example dialogue

> **Dev:** "Can I keep using this **seed table** after re-ingesting the graph?"
> **Domain expert:** "No. The **seed table** belongs to one **parent graph snapshot** only, so a new parent snapshot requires a rebuilt seed table."

## Flagged ambiguities

- "target table" was used to mean the on-disk task-row artifact; resolved: the canonical term is **seed table**. Use "target" only for labels / target expressions / `TARGET1`.
- "seed row" was at risk of being treated like a graph node; resolved: a **seed row** is staging data, not part of the real graph topology.
- Missing `ENTITY1` matches might have implied a fatal ingest error; resolved: they create **unresolvable seed rows**, which are dropped and counted non-fatally.
- "TIMESTAMP" could have meant many time concepts; resolved: it is the **observation timestamp** that gates visible parent-graph history for the seed row.
- The hop scope of the **observation timestamp** is still open: it is not yet resolved whether the timestamp constrains only the first parent-graph expansion or every hop in the sampled subgraph.
- `TARGET1` could have been mistaken for a settled structural edge endpoint; partially resolved: in v1 it is a **target value** stored as task-local supervision in the rowpack, not materialized as structural FK edges — but RHS structural FK materialization (a parallel `seed.edges.bin`) is architecturally possible and not yet implemented.
- Unsupported shapes on the **DG-native seed-table path** are not silently routed elsewhere; v1 rejects them explicitly.
- Shared storage shape does not imply one shared workflow concept; **training seed table** and **prediction seed table** remain distinct domain objects.
- `FEAT*` / `LAG_FEAT*` might have been treated as out of scope; resolved: they are **seed-row features** and are supported in v1 as task-local example features.
- The seed-table subdirectory could have been treated as permission to patch the parent; resolved: the **parent-untouched invariant** forbids any parent mutation.
- Runtime composition could have grown into “one parent + many seed tables”; resolved: v1 uses a **single-seed runtime view** only.
- Prediction shape could have carried a dummy `TARGET1`; resolved: a **prediction seed table** normally has no `TARGET1` at all.
- High-level discussion could have drifted back to the workflow object name; resolved: when discussing the DiskGraph-side idea, use **prediction seed table** even at a high level.
- The hop-0 entry shape could have grown into zero-or-many structural links; resolved: each persisted **seed row** has exactly one structural entry link.
- Raw TT row order could have been treated as the stable runtime identity; resolved from `tt-ng`: use the **persisted seed-row id** assigned after ingest-time drops.
- The persisted seed table could have been treated as a maybe-valid staging dump; resolved: it is a **runtime-ready seed table**.
- `arange.bin` could have become part of the new path's meaning by accident; resolved from `tt-ng`: it is a **transitional sidecar** only.
- A seed table could have been treated as a reusable standalone dataset; resolved for now: it is a **seed-table extension artifact**.
