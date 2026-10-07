# Debugging DiskGraph

This guide is distilled from 40+ PRs of bringing DiskGraph from prototype to production parity with GraphEngine. The patterns here are not hypothetical — each one caused real failures.

## The Three Debugging Rules

**1. Verify with data, not code.** Read the actual parquet. Print the actual tensor. Compare the actual TT rows. Code analysis generates hypotheses; data confirms or kills them. The HnM LP bug was initially attributed to `explode=false` based on code reading — the actual path was `explode=true` with orphaned FKs, only found by diffing TT parquets row-by-row.

**2. Compare against GraphEngine.** GE has been in production for years. When DG produces different results, run the same query on both, diff the outputs. The diff tells you exactly what's different — extra rows, missing rows, wrong values. Then verify against raw source data to determine which backend is correct.

**3. Use file-based logging.** The devserver runs multiple processes in tmux panes. Python logger output gets mixed, truncated, and lost across panes and subprocesses. Write to `/tmp/debug.log` directly:
```python
with open('/tmp/debug.log', 'a') as f:
    f.write(f'[function] key={value}, shape={tensor.shape}\n')
```

## Failure Patterns by Frequency

### 1. Timestamp Unit Mismatches (3+ PRs: #172, #170, kumo #27774)

DG Rust code, DG parquet output, and Kumo Python code may each use different timestamp units (seconds, milliseconds, nanoseconds). This is the single most common integration bug.

**How it manifests:** Empty splits, wrong temporal windows, entities appearing at wrong timestamps, all rows assigned to one split stage.

**Examples:**
- `TimeRangeSplit` passed second boundaries but executor compared nanosecond parquet timestamps → all rows misassigned → empty train set (#172)
- `kumo_timestamp_` prefix on internal timestamps broke `pd.Timestamp` parsing → split boundary computation failed (kumo #27774)
- Temporal sampling used `>=` instead of `>` at boundary → 60% fewer neighbors for entities where `time_col == seed_time` (#170)

**How to debug:** Print the raw timestamp values at both sides of every comparison. Check the unit. Check inclusive vs exclusive. The bug is always at a boundary.

### 2. Spark Implicit Behavior That DG Must Handle Explicitly (5+ PRs: #139, #173, #52, #169, kumo #27475)

GE runs through Spark which silently handles many edge cases. DG's graph traversal includes everything the index has. Every Spark implicit behavior is a potential divergence:

| Spark Does Implicitly | DG Must Do Explicitly | PR |
|----------------------|----------------------|-----|
| JOIN filters orphaned FKs | Check PK existence in ListDistinct | #173 |
| NULL values skipped in aggregations | Use `read_column_value_nullable()` | #139 |
| INNER JOIN drops entities with no matching rows | `Accumulator::finalize()` returns None for empty | #52 |
| `drop_na` on time columns | Skip NULL time_col rows in col stats | #169 |
| Deduplication within partitions | Explicit PK canonical check | Built into MPHF |

**How to debug:** Compare TT row counts. If DG has more rows, it's including something Spark filters. If DG has fewer, it's filtering too aggressively. Check raw source data to determine ground truth.

### 3. Subprocess Cache Isolation (4+ PRs: kumo #27665, #27728, #27754, diskgraph-e2e)

Kumo runs activities in `process_pool_activity` — each in a fresh subprocess. In-memory caches, seeded stats, Spark sessions are all empty. Any state that was set up in a previous activity is gone.

**How it manifests:** Training works but prediction fails (different subprocess). Encoder inference fails despite stats being computed earlier. Spark can't read S3 (session not initialized in this subprocess).

**Examples:**
- Col stats seeded in one activity, empty in the next subprocess → encoder inference fails (kumo #27665)
- Spark session created without S3 JARs in fresh subprocess → `ClassNotFoundException` (kumo #27754)
- Prediction table gen has no col stats workaround that training has → validation fails (kumo diskgraph-e2e)

**How to debug:** Check if the failing code runs in a different subprocess than the code that set up state. The fix is always: re-seed from disk, not from memory.

### 4. pyo3 GIL Boundary Issues (#158, #59)

Rust code called from Python via PyO3 must carefully manage the GIL. Two patterns cause deadlocks:

- **Joining a thread that logs:** `handle.join()` with GIL held + background thread calls `tracing::info!()` which routes through pyo3-log and needs GIL → deadlock. Fix: `py.detach()` before `handle.join()`.
- **Blocking recv with GIL held:** Channel recv blocks the Python thread. Fix: `py.detach()` around `recv()` in the streaming iterator.

**Rule:** Never hold the GIL while blocking on a Rust thread that might call any pyo3 function. This includes logging.

### 5. Edge Cases in Graph Structure (#134, #147, #51, #108)

Small or degenerate graph structures expose assumptions in the code:

- **Empty shards:** 500-node table gets 3 shards → 2 empty with sentinel values → binary search breaks (#134)
- **Sort key misalignment:** Separate batcher pipelines flush at different rates → row N in fkey file ≠ row N in sort key file (#51)
- **Edge table misdetection:** `should_be_edge_table()` classified HnM transactions as edge table based on column types → missing col stats (#108)
- **Binary search regression:** Multi-threaded merge accidentally reverted O(log n) to O(n) → caught by unit test (#147)

**How to debug:** These are always data-dependent. Test on datasets with extreme characteristics: very small tables (< shard count), tables with all-NULL columns, tables with 0 edges.

### 6. Path and Format Mismatches (kumo #27789, #27757, #27728, #107)

The DG↔Kumo integration has many path conventions, column naming conventions, and format expectations that must align:

- Custom TT edge materialization: DG uses `train_table`, custom TT uses `custom_train_table` → fuzzy match fails (kumo #27789)
- Manifest path: trailing `/` → wrong `.manifest.json` path (kumo #27757)
- Parquet directory vs file: `_sha256_file()` called on a directory (kumo #27728)
- FREQUENCY column: DG outputs `FREQUENCY`, Kumo expects `__kumo_internal_frequency_column__` (#107)

**How to debug:** Print the actual path. `ls` the actual directory. These bugs are always obvious once you look at the real filesystem state.

## Debugging Procedures

### Comparing TT Gen Output

The most productive debugging technique. Save both TTs and diff:

```python
import pandas as pd

ge = pd.read_parquet("/path/to/ge/train_table")
dg = pd.read_parquet("/path/to/dg/train_table")

print(f"GE: {ge.shape}, DG: {dg.shape}")

# Row-level diff
ge_set = set(zip(ge['ENTITY'].astype(str), ge['TIMESTAMP'].astype(str), ge['TARGET'].astype(str)))
dg_set = set(zip(dg['ENTITY'].astype(str), dg['TIMESTAMP'].astype(str), dg['TARGET'].astype(str)))
print(f"GE-only: {len(ge_set - dg_set)}, DG-only: {len(dg_set - ge_set)}, Both: {len(ge_set & dg_set)}")

# For DG-only rows, verify against raw data
for ent, ts, tgt in sorted(dg_set - ge_set)[:5]:
    raw = source_table[(source_table['entity'] == ent)]
    # Check if this row should exist given the query window
```

### Checking Masks

When training fails with missing metrics (`val_map@10 not available`), check the mask .pt files:

```python
import torch
for name in ['train_mask', 'val_mask', 'test_mask']:
    m = torch.load(f"/path/to/trainingjob/tasks/NA/table_name/{name}.pt", weights_only=True)
    print(f"{name}: sum={int(m.sum())}, shape={m.shape}")
```

If val_mask and test_mask are all zeros, trace where masks get written. There are two save points: `materialize_training_table` (correct masks) and `task.save()` in `materialize_training_task` (can overwrite). Check both.

### Checking Edge Parquets

When LP training fails, check if both fkey edges have the same size:

```python
import os, pandas as pd
base = "/path/to/trainingjob/persist/edges"
for d in sorted(os.listdir(base)):
    f = os.path.join(base, d, "edge.parquet")
    if os.path.exists(f):
        df = pd.read_parquet(f)
        print(f"{d}: {len(df)} rows")
```

If sizes differ, check `fkey_cols.parquet` for `-1` values (unresolvable FKs).

### Checking FK Resolution

```python
import pandas as pd, glob
parts = sorted(glob.glob("/path/to/artifacts/train_table_edges/train_table_fkey_cols.parquet/part-*.parquet"))
fkey_df = pd.concat([pd.read_parquet(p) for p in parts])
print(f"TARGET == -1: {(fkey_df['TARGET'] == -1).sum()} rows (orphaned FKs)")
```

### Rebuilding After Changes

```bash
# Rust changes in kumo-diskgraph:
bash kumo-tests/scripts/update_diskgraph.sh --build-only
# Verify .so timestamp is newer:
ls -la kumo/venv/lib/python3.11/site-packages/diskgraph/*.so
# Delete pycache:
find kumo/venv -name "__pycache__" -path "*diskgraph*" -exec rm -rf {} +
# Smoke test:
python -c "from diskgraph import TTPlanIterator; print('OK')"
```

### Devserver State

Clean state between runs. Stale caches cause mysterious failures:
```bash
rm -rf /mnt/localssd/kumo-home/jobs /mnt/localssd/kumo-home/sync
rm -rf ~/.cache/kumo/jobs ~/.cache/kumo/sync ~/.cache/kumo/data-persist-dir
rm -rf /mnt/localssd/NA-*  # DG index dirs
rm -rf /mnt/localssd/diskgraph-data/*  # DG index data
```

Or use `start_devserver.sh` which handles all of this.

**Always clean dataplane root when reproducing bugs.** The DG index, kumo job cache, and data-persist-dir all carry state from previous runs. A "fix" that works without cleaning may just be reusing cached artifacts from a previous (correct) run. A bug that only reproduces on the second run may be caused by stale cached state. When in doubt, nuke everything and start fresh.

## Isolating Bugs to DiskGraph

The devserver is a complex system — Temporal, REST API, kumo-watch, Spark, plus DG. When a bug occurs, the first step is isolating whether it's in DG (Rust) or in the kumo integration (Python).

### Reproduce with pybind directly

If the bug is in TT gen, reproduce it without the devserver:

```python
from diskgraph import DiskGraphPlanCompiler, execute_tt_plan_with_artifacts

# Use the same index and plan that the devserver used.
# The plan JSON is logged by the devserver — grep for it in the data worker pane.
# The index path is at /mnt/localssd/diskgraph-data/{graph_key}/{version}/

result = execute_tt_plan_with_artifacts(
    plan_json=plan_json,
    index_path="/mnt/localssd/diskgraph-data/NA-xxx/yyy/",
    output_path="/tmp/tt_debug/",
    # ... same args as the devserver used
)
```

If this reproduces the bug, it's a DG Rust issue. If it doesn't, the bug is in the kumo integration layer (materialize_training_task, edge materialization, mask handling, etc.).

### Reproduce with the ingest binary

If the bug is in ingest, run the binary standalone:

```bash
cargo run --release --bin ingest -- \
  --source /path/to/diskgraph_config.json \
  --output /tmp/test_index \
  --col-stats --cat-count-strategy hybrid
```

Capture the config from a previous devserver run — it's written to the DG data directory during ingest.

### Reproduce with the sampler

If the bug is in sampling/features, create a minimal Python script that loads the index and calls `SamplerEngine` directly, without the kumo training loop.

### Key principle: shrink the reproduction

The devserver runs ~15 processes across tmux panes. Errors in one pane cause cascading failures in others. Logs get mixed. A direct pybind or binary reproduction eliminates all of this noise and lets you iterate in seconds instead of minutes.

Once you have a standalone reproduction, you can add Rust logging, rebuild (`cargo build --release`), reinstall the wheel, and re-run — without restarting the devserver.

## Debugging Ingest

Ingest bugs are data-dependent — they show up on specific datasets with specific schema characteristics. The ingest binary can be run standalone for fast iteration.

### Running Ingest Standalone

```bash
cargo run --release --bin ingest -- \
  --source config.json \
  --output /tmp/test_index \
  --col-stats \
  --cat-count-strategy hybrid
```

This bypasses the devserver and Temporal entirely. Useful for iterating on ingest bugs without the 30-second devserver startup.

### Verifying Index Correctness

After ingest, check basic index properties:

```python
from diskgraph import DiskGraphEngine
engine = DiskGraphEngine("/tmp/test_index")
# Check node counts per table
for table in engine.table_names():
    print(f"{table}: {engine.node_count(table)} nodes")
```

Compare node counts against raw parquet row counts. If they differ, check for:
- **Duplicate PK deduplication**: DG deduplicates by PK during ingest. If the source has duplicate PKs (e.g., Tubi synthetic data with PKs repeated 56x across parquet partitions), DG will have fewer nodes than raw row count.
- **NULL rows dropped**: DG may drop rows with NULL PKs.

### Common Ingest Bugs

**Empty shards** (#134): Small tables (e.g., 500 nodes) get the same shard count as large tables. Most shards end up empty with sentinel values (`min_id=1, max_id=0`). Binary search breaks on inverted ranges. Test with tables that have fewer rows than shard count.

**Sort key misalignment** (#51): When a table has multiple parquet files, separate batcher pipelines (sort key vs node data) can flush at different rates. Row N in the fkey temp file may refer to a different source row than row N in the sort key temp file. Only manifests through wrong time-windowed aggregation results — very hard to detect without comparing against GE. Fixed by bundling sort key data into node write batches.

**Edge table misdetection** (#108): The config converter's `should_be_edge_table()` heuristic can misclassify node tables as edge tables based on column types (e.g., HnM transactions where `sales_channel_id` as int64/ID passes the numeric weight check). Symptoms: missing col stats for the misclassified table. Fixed by disabling auto-detection.

**String-encoded float arrays** (#113): Production datasets (e.g., Faire EMBEDDING column) store embeddings as JSON strings `"[-0.19, 0.27, ...]"` instead of native float arrays. DG needs to parse these during ingest. Uses `fast-float2` crate for ~5x faster parsing.

**Timestamp format detection** (#75, #104): The converter downloads parquet files to detect timestamp formats. Without limits, it downloads every file for unhandled Arrow temporal types. Cap at 15 files per table. Handle native Arrow temporal types. Detect integer-encoded dates (YYYYMMDD, YYYYMMDDHHMMSS).

## Debugging Sampling

Sampling bugs affect training metrics but don't cause crashes — the model trains but produces worse results than GE. These are the hardest bugs to detect because you need to compare metric quality, not just check for errors.

### Verifying Neighbor Counts

For a specific entity, compare the sampled neighbors against what GE samples:

```python
# After a training run, check the sampled subgraph
# Kumo's trainer logs subgraph sizes — check kumo.log for:
#   "num_sampled_nodes", "num_sampled_edges"
# Compare these between GE and DG runs on the same data
```

When DG produces fewer neighbors, check:
- **Temporal boundary semantics** (#170): DG used `>=` (excludes edges at seed timestamp), GE uses `>` (includes them). For entities where `time_col == seed_time`, the `>=` filter drops all edges from that timestamp. This caused 60% fewer neighbors on datasets where destination table time_col equals the seed time. Fixed by changing to `>`.
- **CSC direction mapping** (kumo #27475): Forward/reverse edge counterpart mapping was incorrect for link prediction. The sampling direction determines which neighbors are reachable.

### Verifying Feature Values

When training metrics diverge despite correct neighbor counts:

```python
# In diskgraph_sampler.py, the feature store serves features via:
#   engine.fetch_features(table, node_ids, col_indices)
# Compare raw features against reading the parquet directly:
import pandas as pd
raw = pd.read_parquet(f"s3://bucket/{table}")
# Compare values for specific node IDs
```

**Fusion mode stale cache** (kumo #27475): `_get_fused_tf()` ignores `attr.index` and returns a stale cache from `get_subgraph_size()` calibration. Causes shape mismatches and memory leaks on large datasets. Symptoms: training works on small data but crashes or produces wrong features on large data. Fixed by clearing cache after calibration + 0-hop sample fallback.

**Missing time attribute** (kumo #27756): `DiskgraphFeatureStore` set `time_col_name=None` for graph tables instead of reading from `time_col_dict`. The explain step and temporal features depend on this. Symptoms: val AP drops significantly (0.57→0.65 after fix on Intuit fraud dataset).

### Ghost Node Stores (#27304)

The Rust sampler reports sampling stats for ALL schema types, including types with 0 sampled nodes. Kumo's trainer creates "ghost" node stores for these unsampled types, which fail downstream validation. Symptoms: trainer check failures on datasets with many node types where some types have no sampled entities. Fixed by filtering `num_sampled_nodes`/`num_sampled_edges` to only include types present in the sampled dicts.

## What To Check First

| Symptom | Check |
|---------|-------|
| `val_map@10 not available` | Mask .pt files (val_mask sum should be > 0) |
| Training crashes in AutoML | Devserver tmux pane 8 (kumo-watch) for error |
| TT gen different row count | Diff TT parquets row-by-row |
| Wrong metric values | Compare GE vs DG using custom TT path (same input data) |
| `ClassNotFoundException` for S3 | Spark session missing JARs — check `dependency_path` |
| Encoder inference fails | Col stats not seeded — check `_ensure_diskgraph_col_stats_in_cache` |
| Prediction fails but training works | Subprocess isolation — check if prediction re-seeds state |
| Deadlock (process hangs) | GIL held while joining Rust thread — check pyo3-log usage |
| Training metrics worse than GE | Compare neighbor counts and feature values — check sampling boundaries |
| Missing col stats for a table | Edge table misdetection or ingest cache skip — check config converter |
| Ingest crashes on specific dataset | Empty shards, string-encoded arrays, timestamp format — run ingest standalone |
| Memory leak during training | Fusion mode stale cache — check `_get_fused_tf()` cache clearing |

## Hypotheses That Are Usually Wrong

These come up repeatedly during debugging and waste time. Check before going down these paths:

**"GE has a bug."** GE has been in production for years. If DG produces different results, DG is almost always the one that's different. Verify against raw source data before concluding GE is wrong. (Exception: the HnM orphaned FK case where DG was technically including valid data that GE's Spark join happened to filter — but even there, DG needed to match GE's behavior for correctness.)

**"The boundary is off by one."** The actual boundary semantics `(start, end]` are well-defined and consistent. When results look like an off-by-one, it's almost always a unit mismatch (seconds vs nanoseconds) or a completely different code path being taken. Check the units first.

**"The Rust code is wrong."** Often the Rust code is correct but the Python integration layer has a mismatch — wrong path, wrong column name, wrong argument order, missing prefix strip. Check the Python↔Rust boundary first. The Rust code has extensive unit and integration tests; the glue code often doesn't.

**"It's a race condition."** DG's parallel executor has no shared mutable state on the hot path. The entity loop, stage execution, and target writing are all deterministic given the same input. When results vary between runs, it's more likely: Rust HashMap iteration order (non-deterministic but not a race), subprocess cache state (empty vs populated), or Spark non-determinism upstream.

**"The split is computed wrong."** Splits are almost always computed correctly by `execute_split`. The bug is usually that correct masks get overwritten later — by `task.save()` in `materialize_training_task`, or by `add_table()` in `graph_data.py` during LP edge processing. Trace all the save points, not just the computation.
