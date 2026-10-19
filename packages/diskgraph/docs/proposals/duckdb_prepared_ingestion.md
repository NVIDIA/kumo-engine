# DuckDB preparation with GxHash for DiskGraph ingestion

Status: proposed; performance and runtime compatibility are unverified.

Prototype DuckDB preparation of node IDs and sorted, resolved edges, followed
by ordered Arrow streams into DiskGraph's native feature encoders and shard
writers. Use DiskGraph-compatible GxHash partitioning throughout.

```text
Source Parquet
  -> DuckDB: normalize keys, validate rows, handle duplicates
  -> GxHash shard assignment and node-ID assignment
  -> endpoint joins and directed relation generation
  -> edge sorting
  -> ordered Arrow batches
  -> DiskGraph: feature encoding, edge packing, shard files, metadata
```

The goal is to avoid repeated mapping, redistribution, sorting, and intermediate
key files. DuckDB uses the same machine's resources; its joins and sorts can
still spill. Lower total runtime, memory, or scratch usage must be measured.

## 1. Establish the baseline

Measure representative integer-key, string-key, skewed, and larger-than-memory
datasets using current ingestion. Record:

- Total time and time in Encode, PkMap, EdgesMap, and Index.
- Peak process memory, peak scratch usage, and disk bytes read and written.
- Node and edge counts, key widths, degree skew, and resource limits.

Use the current [ingest implementation](../../diskgraph-ingest/src/lib.rs),
[key mapping](../../diskgraph-ingest/src/mapping.rs), and
[index builder](../../diskgraph-index/src/builder.rs) as the behavioral reference.
Identify whether mapping, endpoint resolution, sorting, or encoding dominates.

## 2. Define the prepared-input contract

| Dataset | Contents | Required order |
| --- | --- | --- |
| Nodes | Node type, shard ID, node ID, original key, features | Node-ID order within each shard |
| Directed edges | Relation, source shard, endpoint IDs, sort key | Relation and source ID, then configured neighbor order |
| Layout | Shard counts, ID ranges, schemas, sort-key metadata, partitioning version | Configuration input |

Preserve current key normalization, duplicate selection, invalid-row handling,
orphan handling, and relationship direction semantics. Finalize row eligibility
before assigning IDs so node features and endpoint mappings remain aligned.

Fix the shard count before preparation. Persist mappings and layout metadata
for the build; retries must reuse them. Specify exact Arrow types and safe
numeric conversions. Ordering and writer state must survive batch boundaries.

## 3. Expose compatible GxHash partitioning to DuckDB

Reuse the native key serialization and shard calculation rather than
independently implementing the algorithm:

```text
canonical key bytes
  -> GxHash with DiskGraph's pinned version and seed
  -> unsigned hash % shard_count
```

The current shard seed is `0x6a09e667f3bcc909`; the ingest crate currently pins
GxHash `3.5.0`. Treat the implementation, canonical encoding, seed, and shard
count as a versioned compatibility contract, not interchangeable hash settings.
See [constants](../../diskgraph-ingest/src/consts.rs) and
[dependencies](../../diskgraph-ingest/Cargo.toml).

Evaluate a batched native DuckDB function or extension that shares the relevant
DiskGraph implementation. Avoid a Python callback per key. Follow the
[DiskGraph ownership workflow](../../../../tools/migrations/diskgraph/README.md)
for implementation changes.

Add cross-engine test vectors for integer boundaries, negative integers, empty
and multibyte strings, and supported UUID/IP representations. Check canonical
encoding, hash outputs, and shard assignments on supported architectures.

Persist the partitioning version and shard count with build metadata. Matching
GxHash preserves shard assignment; warehouse or DuckDB numbering does not
necessarily reproduce the current local node IDs.

## 4. Implement DuckDB preparation

Start with integer keys and one timestamp sort key:

1. Read source Parquet and apply normalization, validity, and duplicate rules.
2. Assign shards through the compatible GxHash function.
3. Assign local node IDs and cumulative shard offsets.
4. Materialize reusable primary-key-to-node-ID mappings.
5. Join edge endpoints and any referenced sort-key columns.
6. Generate forward and reverse records according to the graph configuration.
7. Sort by relation, source ID, and configured neighbor order within each shard.

Expand key-type support after correctness is established. Match encoded
sort-key semantics, including composite keys and timestamp boundaries.
Materialize intermediates only where reuse or recovery justifies their cost.

## 5. Add DiskGraph's prepared-input path

Reuse native encoders and storage writers. The new path must:

- Accept supplied node IDs, layout metadata, and ordered Arrow batches.
- Bypass repeated partitioning, endpoint resolution, and edge sorting.
- Preserve feature-to-ID alignment and validate ordering across batches.
- Handle isolated nodes, empty shards, and zero-degree relations.
- Produce required primary-key lookup artifacts, column statistics, and metadata.
- Route subsequent key lookups using the same GxHash assignment and supplied IDs.

Identify schema inference, statistics, or layout decisions that require an
earlier pass. Do not assume every artifact can be constructed in one pass.
Current ingestion cannot obtain these savings merely by receiving sorted data;
the mapping and sorting stages must explicitly be bypassed.

## 6. Bound resources and preserve recovery

Share CPU and memory budgets between DuckDB and DiskGraph, allowing headroom
for allocations outside DuckDB's managed memory. Bound Arrow queues and active
shard writers. Account for DuckDB spill files in peak scratch usage.

Avoid rebuilding endpoint joins for every shard. Measure repeated scans,
duplicate join state, and sort work under parallel consumption. Preserve each
stream's order while allowing independent shards to progress concurrently.

Commit completed shard files only after validation. Restart incomplete shards
using persisted mappings. Reject resumes with incompatible schemas, shard
counts, hash contracts, or node-ID assignments.

## 7. Validate correctness and performance

Compare graph semantics through original keys where node-ID assignments differ.
Verify retained features and relationships, duplicate/null/orphan handling,
neighbor order, timestamps, key lookups, statistics, and downstream query and
sampling behavior. Open generated artifacts with the existing DiskGraph reader.

Run benchmarks sequentially under identical resource limits:

| Variant | Purpose |
| --- | --- |
| Current ingestion | Baseline |
| DuckDB GxHash, mappings, and joins; DiskGraph sorting | Isolate preparation gains |
| Full DuckDB preparation and sorting; DiskGraph streaming writer | Evaluate the complete proposal |

Include hash integration, mapping construction, materialization, spilling,
Arrow transfer, required lookup construction, and final index writing in total
cost. Measure overall process memory and scratch usage, not only individual
operators. Adopt the new path only for demonstrated gains and keep the current
path available until correctness and performance are established.

## First milestone

Verify identical GxHash assignments, then build one node type and one
bidirectional relationship through the prepared-input path. Include an isolated
node, an empty shard, and an adjacency list spanning batches. Validate the index
with the existing reader and compare end-to-end resource usage with the baseline.
