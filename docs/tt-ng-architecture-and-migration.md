# Streaming Train-Table Generation: Architecture and Migration

> Design doc for the PR that introduces streaming train-table generation
> to `diskgraph-train-table`. Intended audience: engineers reviewing the
> PR or working on adjacent code. Self-contained — concrete details
> below; in-file module docstrings under `diskgraph-train-table/src/`
> carry the implementation-level context.

## 1. What this PR changes

The legacy train-table generator was a ~6000-line `tt_executor.rs`
monolith in `diskgraph-pybind/`, plus ~2800 LOC of Python helpers
(`split_assignments.py`, `tt_artifacts.py`, `task_dataset_*.py`,
their tests). It ran TT-gen, split-assignment, manifest-writing, and
prediction-table production in one place; pybind orchestrated
column-typed accumulators inline; splits used a permutation-shuffle
that required all rows in memory.

This PR replaces that with a streaming two-pass architecture in a
dedicated `diskgraph-train-table/` crate. Pass A streams rows from the
entity table in parallel, building tt.parquet and per-worker
accumulators. Pass B re-reads tt.parquet column-projected to make
per-row split decisions and writes split.bin via lock-free `pwrite`.
The pybind layer becomes a thin shim: deserialize plan JSON, call into
the train-table crate, return paths.

PR diff: **+17891 / -7713 / 36 files** against `tt-ng`. Of that:

- ~8.5k LOC are new modules in `diskgraph-train-table/src/` (8 files)
- ~3.1k LOC update existing modules in `diskgraph-train-table/src/`
- ~5k LOC of tests (3 files)
- ~1.1k LOC of design docs
- -7.7k LOC removes the legacy pybind monolith and its Python helpers

## 2. Why we needed this rearchitecture

Three concrete blockers in the legacy pipeline:

1. **Memory at scale.** Permutation-shuffle splits required `Vec<u64>`
   of size N; at 50B rows that's 400GB. `HashSet<u64>` cardinality
   tracking added another ~400GB. Production-scale jobs OOM'd.
2. **Two-pass stats.** Col stats were computed in a second pass over
   the produced parquet because the first pass didn't carry the
   accumulators forward. Doubled I/O on the largest table.
3. **Architectural lock-in.** TT-gen, prediction-serving, and
   split-assignment shared the same pybind monolith and Python
   helpers. Evolving any of them broke the others. The legacy path
   had no abstraction boundaries to migrate against.

The new architecture targets each: streaming accumulators (HLL +
t-digest replace HashSet/sort), pass-A-then-pass-B with shared
accumulator state, and explicit module/trait boundaries
(`SplitDecider`, `RowProcessor`, `StatsMerger`, `ChannelBatchWriter`).

## 3. Architecture overview

### Before

```
┌────────── kumo Python ──────────┐
│ train_table_generator.py        │
└──────────────┬──────────────────┘
               │  plan + flags
               ▼
┌────── pybind monolith (~6k LOC) ────────────────────────┐
│ diskgraph-pybind/src/tt_executor.rs                     │
│   - reads entity table                                  │
│   - builds train table in memory                        │
│   - second pass: col stats                              │
│   - permutation-shuffle splits via                      │
│     diskgraph-pybind/src/split_assignments.rs           │
│   - manifest builder (Python helpers in                 │
│     diskgraph-pybind/python/diskgraph/tt_artifacts.py)  │
│   - emits tt.parquet, split assignments,                │
│     edge bundles, all stats artifacts                   │
└─────────────────────────────────────────────────────────┘
               ▼
        kumo consumes (training)
```

Single monolith. No abstraction boundaries. Memory footprint scales
with row count.

### After

```
┌────────── kumo Python ──────────────────────┐
│ train_table_generator.py builds plan_json   │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌──── diskgraph-pybind shim (~955 LOC) ───────┐
│ execute_tt_plan(plan_json, idx, output_dir) │
│   serde_json::from_str → DiskGraphTTPlan    │
│   call execute_tt_pass(...)                 │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌──── diskgraph-train-table crate ────────────────────────────┐
│                                                             │
│ Pass A — parallel chunked entity-table walk                 │
│  ├─ N rayon workers, each owns a RowProcessor<U, D>         │
│  ├─ ChannelBatchWriter → bounded(16) channel → writer       │
│  │   thread → tt.parquet                                    │
│  └─ Per-worker ChunkStats (welford+tdigest+HLL+counts)      │
│                                                             │
│ Merge — fold ChunkStats → StatsMerger                       │
│         decider.derive_cutoffs() (stateful families only)   │
│                                                             │
│ Optional regroup — only for LP+array+RandomLinkSplit        │
│                                                             │
│ Pass B — parallel column-projected re-read of tt.parquet    │
│  ├─ Per row group, decoder projects only required columns   │
│  ├─ Decider decides each row's stage byte                   │
│  ├─ Workers pwrite to disjoint offsets in split.bin         │
│  │   (lock-free, no atomics, no channels)                   │
│  └─ Per-stage stats accumulator (split_stats module)        │
│                                                             │
│ Bridge passes — transitional sidecars                       │
│  ├─ bridge_pass: arange.bin                                 │
│  └─ kumo_contract_bridge: split_stats.json,                 │
│     train_table_col_stats.json                              │
│                                                             │
│ Manifest — single Rust source of truth, sha256-checksummed  │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌── kumo legacy GraphStore bridge ──────────────────────────┐
│ kumo/data/generic/_diskgraph_legacy_graphstore_bridge.py  │
│   1. Inject arange column into tt.parquet                 │
│   2. Rename ENTITY1→ENTITY, TARGET1→TARGET                │
│   3. Refresh manifest train_table sha256                  │
│   4. Call diskgraph.export_tt_artifacts to produce        │
│      reverse-edges + fkey_cols + mask                     │
│   5. Write train_table_edges manifest entry               │
└──────────────┬────────────────────────────────────────────┘
               │
               ▼
        kumo consumes (training)
```

Crate-with-modules architecture. Streaming throughout. Explicit
abstraction at every boundary.

## 4. Key APIs

### Pybind entry

```rust
// diskgraph-pybind/src/tt_executor.rs:268
#[pyfunction]
pub fn execute_tt_plan(
    py: Python<'_>,
    plan_json: &str,
    index_path: &str,
    output_dir: &str,
) -> PyResult<Py<PyAny>>
```

The only TT-gen entry into the executor. Returns a Python dict with
`tt_parquet_path`, `manifest_path`, `rows_written`, `summary`. No
streaming exposure across the FFI boundary.

### Custom-TT pybind entry

```rust
// diskgraph-pybind/src/custom_tt_entry.rs
#[pyfunction]
pub fn build_train_table_artifacts_for_existing_table(
    py: Python<'_>,
    output_path: &str,
    index_path: &str,
    entity_table_name: &str,
    entity_pk_column: &str,
    target_stype: &str,
    input_timestamp_unit: &str,        // REQUIRED — no default
    split_spec_json: Option<&str>,
    weight_col: Option<&str>,
) -> PyResult<Py<PyAny>>
```

The rehydration entry kumo's
`ensure_diskgraph_native_artifacts_for_train_table` calls when an
already-materialized training-table parquet needs DG sidecars
(manifest, split.bin, col_stats, weight.bin). Builds a
`PredictionType::Node` plan with **empty stages** (every output column
read from the customer parquet) and dispatches into
`execute_tt_pass_with_source(..., Some(custom_source))`.

`input_timestamp_unit` is parsed via `TimeUnit::from_name`
(`nanoseconds`/`microseconds`/`milliseconds`/`seconds`) and stored on
`CustomTtSource`. See §9 for why it's required at the bridge boundary
and not on `DiskGraphTTPlan`.

### Rust orchestrator

```rust
// diskgraph-train-table/src/ttgen_executor.rs:169
pub fn execute_tt_pass<U: UnitSystem>(
    plan: &DiskGraphTTPlan<U>,
    index_path: &str,
    output_dir: &str,
) -> TTGenResult<TTPassOutput>
```

`TTPassOutput` carries the produced paths plus the finalized stats
payload. `U` is `Seconds` in production; the pybind builder converts
nanosecond/millisecond/etc. inputs to seconds at construction
(`abs_from_unit::<Seconds>(value, unit)` in `tt_builder.rs:396`), so
the deserialization side trusts JSON values are already in seconds.

### SplitDecider trait

```rust
// diskgraph-train-table/src/split_decider.rs
pub(crate) trait SplitDecider<U: UnitSystem>: Send + Sync {
    type BatchView<'a>: Send where Self: 'a;       // GAT — borrows from &'a RecordBatch

    fn batch_view<'a>(&self, batch: &'a RecordBatch, row_offset: u64)
        -> TTGenResult<Self::BatchView<'a>>;
    fn decide_for_view(&self, view: &Self::BatchView<'_>, row_idx: usize) -> u8;

    fn accumulate(&mut self, row: &TTRowRef<'_, U>);
    fn derive_cutoffs(&mut self);

    /// Pass-A-time per-row decide, used by codepaths that don't go through
    /// the BatchView (e.g. tests, regroup driver). Most impls share logic
    /// with `decide_for_view`.
    fn decide(&self, row: &TTRowRef<'_, U>) -> u8;

    /// Stateless families: no-op. Stateful: combine accumulators (used to
    /// fold per-rayon-worker deciders after pass A).
    fn merge(&mut self, other: &Self) where Self: Sized;

    fn required_columns(&self) -> Vec<String>;
}
```

13 impls cover all kumo split families. Generic-dispatched:
`RowProcessor<U, D>` is monomorphized over `D`, so calls inline at the
per-row hot path. **No `&dyn`, no `Box<dyn>`, no vtable** — verified
during benchmarking.

The `BatchView` GAT lifts column lookup and Arrow downcast OUT of the
per-row loop into per-batch construction (e.g. `TimestampNanosecondArray`
gets downcast once per `RecordBatch`, not once per row).

### Pass B lock-free pwrite

```rust
// diskgraph-train-table/src/pass_b.rs:211
pub(crate) fn run_pass_b<U: UnitSystem, D: SplitDecider<U>>(
    tt_parquet: &Path,
    decider: &D,
    split_bin_path: &Path,
    total_rows: u64,
    out_per_stage_stats: &mut Option<PerStageStatsAccumulator>,
) -> TTGenResult<StageCounts>
```

Workers parallel-iterate row groups. Each worker:

1. Opens its own `File` for parquet read (kernel page cache shared).
2. Decodes its row group with `ProjectionMask` from
   `decider.required_columns()`.
3. Builds per-worker `Vec<u8>` of stage bytes.
4. `Arc<File>::write_at(buf, offset)` — `pwrite` to disjoint regions
   of `split.bin`. `set_len(N)` is called BEFORE the parallel scope
   so no size-update kernel contention happens during pass B.

No mutex, no atomic, no channel.

### RowProcessor / ChunkStats / StatsMerger

```rust
// diskgraph-train-table/src/ttgen_stats.rs
pub struct ChunkStats<U: UnitSystem> {
    // welford / t-digest / HLL / LocalColumnCounts per column
    // num_links_per_ts: HashMap<Abs<U>, u64>
    // ...
}

pub struct StatsMerger<U: UnitSystem> { /* folded ChunkStats */ }

pub struct RowProcessor<U: UnitSystem, D: SplitDecider<U>> {
    pub stats: ChunkStats<U>,
    pub decider: D,
}

impl<U, D> RowProcessor<U, D> {
    pub fn observe(&mut self, row: &TTRowRef<'_, U>) {
        // 1. fan out to ChunkStats (welford, tdigest, HLL, counts)
        // 2. self.decider.accumulate(row) (stateful families)
    }
}
```

`StatsMerger::finalize()` produces `TrainTableStatsPayload` which embeds
`KumoContractStats` (the kumo-shaped subset, see §8).

### Manifest

Single Rust source of truth (`diskgraph-train-table/src/manifest.rs`).
Schema documented in §8.

## 5. Module-level diff: what's NEW, UPDATED, or DELETED

### `diskgraph-train-table/` (the crate the PR adds to)

| Module | Status in this PR | Role |
|---|---|---|
| `lib.rs` | UPDATED | Crate root, registers all modules. |
| `ttgen_executor.rs` | UPDATED (+1448 LOC) | Orchestrator. `execute_tt_pass` entry. Per-chunk worker dispatch. Pass A → merge → optional regroup → pass B → bridges → manifest. |
| `ttgen_stats.rs` | UPDATED (+1236 LOC) | `ChunkStats`, `StatsMerger`, `RowProcessor`, `RowObserver`, `TTRowRef`, `TrainTableStatsPayload`, `KumoContractStats`, `build_inline_col_stats`. |
| `ttgen_output.rs` | UPDATED (+338 LOC) | `ChannelBatchWriter`, `OutputMode` enum (Node / LinkExploded / LinkArray / PassThrough / Forecasting). Multi-entity slot support (`ENTITY1..N`). |
| `ttgen_resolve.rs` | UPDATED (+99 LOC) | Resolves plan ops, projects split-column values, ColumnSplit branch, ManualSplit-on-TIMESTAMP pseudo-column skip. |
| `ttgen_accumulator.rs` | unchanged | Hash-based accumulator helpers. |
| `ttgen_filter.rs` | unchanged | Per-row filter eval. |
| `ttgen_traversal.rs` | unchanged | Multi-hop traversal helpers. |
| `errors.rs` | UPDATED (+9 LOC) | Error types (added a few variants). |
| `pass_b.rs` | **NEW** (+1541 LOC) | Lock-free parallel column-projected pass B. Two entries: `run_pass_b` (decider-driven) and `run_pass_b_from_inline_stage_column` (post-regroup). |
| `split_decider.rs` | **NEW** (+2959 LOC) | `SplitDecider` trait + 11 impls + `BatchView` cache types + helpers. |
| `regroup.rs` | **NEW** (+1267 LOC) | Per-link regroup pass for `LP + array TARGET + RandomLinkSplit/CustomLinkSplit`. Explodes ListArrays at pass-B time, assigns per-link stage, regroups into per-(row, stage) sub-arrays. |
| `manifest.rs` | **NEW** (+674 LOC) | `TTManifest` struct + serde, `ArtifactRef`, `ManifestChecksums`, sha256 helpers, `write_manifest`. |
| `split_stats.rs` | **NEW** (+428 LOC) | `PerStageStatsAccumulator` per rayon worker; folded at end. Bolts onto pass B (no extra pass). |
| `bridge_pass.rs` | **NEW** (+124 LOC, transitional) | Writes `arange.bin` sidecar. Header docstring: "DELETE WHEN ML MIGRATES". |
| `kumo_contract_bridge.rs` | **NEW** (+802 LOC, transitional) | Emits `split_stats.json` + `train_table_col_stats.json` sidecars. Header: "DELETE WHEN KUMO MIGRATES OFF SIDECAR JSON". |
| `custom_tt_chunk_processor.rs` | **NEW** (+664 LOC) | Mirror of pass A's worker for the custom-TT path (customer-supplied parquet input instead of PQuery-driven entity walk). |

### `diskgraph-pybind/` (the layer the PR thins out)

| File | Status | Role |
|---|---|---|
| `src/tt_executor.rs` | UPDATED (~6000 → ~955 LOC, net -4558) | Thin shim: `execute_tt_plan` dispatches to train-table crate. ~600 LOC of surviving prediction-artifact helpers + new pybind shims. |
| `src/lib.rs` | UPDATED (+52 LOC) | Module registration. |
| `src/tt_builder.rs` | UPDATED (+34 LOC) | Plan builder. `parse_time_unit` + `abs_from_unit::<Seconds>` does the unit conversion at construction time. |
| `python/diskgraph/__init__.py` | UPDATED | Exports new symbols. |
| `python/diskgraph/executor.py` | UPDATED (+31 LOC) | Thin wrapper. |
| `src/split_assignments.rs` | **DELETED** (-516 LOC) | Permutation-shuffle splits replaced by streaming-friendly `SplitDecider` impls. Required all rows in memory; incompatible with 50B-row scale. |
| `python/diskgraph/{split_assignments,tt_artifacts,task_dataset_*,_exceptions}.py` | **DELETED** (-2832 LOC total) | Python helpers obsolete; manifest builder moved to Rust (`manifest.rs`); task_dataset format predated tt.parquet. |
| `python/tests/test_{materialize_e2e,split_assignments,task_dataset_mock,tt_artifacts}.py` | **DELETED** (-1856 LOC total) | Tests for deleted code. New e2e coverage is in `diskgraph-train-table/tests/`. |

## 6. The 13 SplitDecider impls

| Family | What it observes | When it decides | Streaming-friendly? |
|---|---|---|---|
| `NoSplitDecider` | nothing | always returns `0` (train) | trivial |
| `TimeRangeDecider` | timestamp, two cutoffs from plan | `ts < val_start ? train : ts < test_start ? val : test` | yes |
| `BacktestingDecider` | timestamp, single cutoff from plan | `ts ≥ backtest_ts ? test : train` | yes |
| `ManualDecider` | split column value, per-stage value lists from plan | lookup in lists | yes |
| `RandomDecider` | entity_pk + ts, hashed with seed | `hash_value(entity, ts, seed)` mod cumulative cutoffs | yes |
| `RandomLinkDecider` | per-row case wraps `RandomDecider`. The per-link case for LP+array is handled by `route.rs`'s `hash_entity_ts_link_to_unit_interval` driver, which reuses the same `splitmix64` + `hash_value` machinery and adds the link's target value, so a link's stage does not depend on its list position. | inherits / `(source_entity, ts, link target, seed)` hash | yes |
| `FixedDecider` | row offset | indexes pre-computed `Vec<u8>` | yes (pre-baked) |
| `TemporalDecider` | timestamp | pass A t-digests; `derive_cutoffs()` computes q=0.8 / q=0.9; pass B is TimeRange-like | yes (no buffer; t-digest is O(1) memory) |
| `DateOffsetDecider` | timestamp + offset (calendar months when `month_offsets` set, else fixed `Dur<U>`) | anchor-relative bucket | yes |
| `ApproxDateOffsetDecider` | tolerant DateOffset | similar | yes |
| `ColumnDeciderManual` | user split column, per-stage value lists | lookup in lists | yes |
| `ColumnDeciderNumericExplicit` | user split column (numeric) | sorted-cutoffs bucket | yes |
| `ColumnDeciderRatioNumeric` | user split column (numeric) | quantile-based ratio | yes |

Sharp principle: time-based / column-based / pre-baked / manual deciders
assign deterministically by their input key (same key → same stage IS
the customer's intent, by contract). Only the random family is
"shuffle randomly per-row".

## 7. Random hash correctness: the entity-aware fix

### What was wrong

An earlier commit on this branch had `RandomDecider` hashing only
`(ts_raw, seed)`. Documented as "ts is sufficient for uniformity when
timestamps span a wide range" — a fragile assumption about customer
data shape, not an algorithmic invariant.

Concrete failure modes:

1. Customer with batch ingest (10k+ rows at one midnight UTC ts): all
   land in train OR all in val OR all in test.
2. Customer with skewed ts distribution (30% of rows on one popular
   day): aggregate ratios skew from 80/10/10 to ~70/40/-10.
3. Customer adds new data at a fresh ts range (all-2026 rows): all
   hash to one stage; model never sees the new data in training.

`amazon_10k` happens to span ~10 years of timestamps so the bug never
manifested in CI. Production data with batch-aligned timestamps would
have triggered it.

### The fix (commit `cbf420d7`)

Hash `(entity_pk, ts, seed)` for `RandomDecider`, and
`(source_row_entity_pk, ts, link_idx, seed)` for the regroup per-link
hash:

```rust
// New helper in split_decider.rs
fn hash_value(v: &Value, h: u64) -> u64 {
    match v {
        Value::Int(i)   => splitmix64(h ^ (*i as u64) ^ 0x01),
        Value::Float(f) => {
            let bits = if f.is_nan() { f64::NAN.to_bits() }
                       else if *f == 0.0 { 0u64 }
                       else { f.to_bits() };
            splitmix64(h ^ bits ^ 0x02)
        }
        Value::Str(s)   => /* chunk-by-8 splitmix mix, with 0x03 discriminator */,
        Value::Bool(b)  => splitmix64(h ^ (*b as u64) ^ 0x04),
        Value::List(_)  => /* recursive, with 0x05 */,
    }
}

// RandomDecider now hashes (entity_pk, ts, seed)
fn decide_entity_ts(&self, entity_pk: &Value, ts_raw: i64) -> u8 {
    let h0 = splitmix64(self.seed);
    let h1 = hash_value(entity_pk, h0);
    let h2 = splitmix64(h1 ^ (ts_raw as u64));
    bucket_by_cutoffs(unit_interval_from(h2), &self.cutoffs)
}
```

Per-variant discriminator bytes (`0x01..0x05`) prevent cross-variant
collisions (`Int(0x6e)` and `Str("\x6e")` get distinct hashes, tested).
NaN/-0 canonicalization for Float so equivalent values hash equal.

`required_columns()` extended to `["TIMESTAMP", "ENTITY1"]` so pass B's
`ProjectionMask` reads ENTITY1 in addition to TIMESTAMP. New
`EntityTsBatchView` caches both column refs once per batch.

13 new tests covering same-ts entity distinction, ts-clustered
aggregate ratios, determinism, cross-seed divergence, all `Value`
variants, cross-variant collision avoidance, the regroup parallel.

### Test relaxation (commit `7ef3b4ac`)

The fix exposed a latent flake: an integration test asserted
byte-for-byte equality of `split.bin` across two runs of the same
plan. The pre-fix bug masked it (every same-ts row got the same
stage, so reordering tt.parquet didn't change byte sequence). After
fix, rayon's row-order non-determinism produces different byte
sequences across runs even though stage *counts* match exactly.

The byte-for-byte assertion was over-strict on an implementation
detail. Pass A's row order is non-deterministic by design — the
bounded(16) channel + writer thread appends in arrival order. Tag-and-
reorder at the writer would impose head-of-line blocking + serialize
the parallel pipeline at 50B-row scale; sort-by-(entity,ts) post-write
is O(N log N) and unacceptable.

The customer-facing invariant is "same data + same seed → same
trained model", which depends on the (entity, ts) → stage multiset,
not on which arange index a row got. Relaxed the assertion to compare
per-(entity, ts) → stage maps. 20/20 deterministic passes after.

## 8. Communication with kumo

### Plan flowing in (kumo → DG)

`plan_json` is JSON-serialized `DiskGraphTTPlan<Seconds>`. Kumo's
`train_table_generator.py` runs its `compiler.compile(...)` sites
(line 1403 onward — entity, target, whatif, etc.) to assemble the
plan, then dispatches via
`get_service().generate_train_table(...)` at line 1164. The service
serializes the plan to JSON and calls our pybind `execute_tt_plan`.

**Timestamp unit handling**: kumo passes raw i64 + a `unit` string
("nanoseconds", "seconds", etc.) at builder construction. The pybind
builder calls `abs_from_unit::<Seconds>(value, unit)` which converts
to seconds-domain `Abs<Seconds>` BEFORE serialization. The
deserializer trusts the JSON values are already in seconds — soft
contract on the JSON boundary, enforced at construction.

### Bundle flowing out (DG → kumo)

After `execute_tt_plan` + `apply_legacy_bridge`, kumo gets `output_path`
(tt.parquet) and `manifest_path`. It calls `validate_manifest(manifest_path)`
which sha256-checks all referenced artifacts and parses sub-artifact
JSONs into the dict.

### Manifest schema

```json
{
  "version": 1,
  "row_count": 1322,
  "source_mode": "pquery",
  "split_mode": "compute",
  "split_kind": "ApproxDateOffsetSplit",
  "stage_counts": {"train": 1322, "val": 214, "test": 87, "unused": 0},
  "train_table":           {"path": "...", "format": "tt_parquet_v1"},
  "split":                 {"path": "...", "format": "split_assignment_row_stage_bin"},
  "stats":                 {"path": "...", "format": "tt_stats_json"},
  "split_stats":           {"path": "...", "format": "tt_split_stats_json"} | null,
  "train_table_col_stats": {"path": "...", "format": "tt_train_table_col_stats_json"} | null,
  "train_table_edges":     {...edge bundle...} | null,
  "arange_sidecar":        {"path": "..."} | null,
  "checksums": {
    "train_table_sha256": "...", "split_sha256": "...", "stats_sha256": "...",
    "split_stats_sha256": "...", "train_table_col_stats_sha256": "...",
    "train_table_edges_sha256": "..."
  }
}
```

Optional sections elide via `serde(skip_serializing_if = "Option::is_none")`.
Format strings are matched verbatim by kumo's reader at
`kumo/engine/diskgraph_native_tt_artifacts.py`.

### `version: 1` — what it actually means

It's an **internal artifact-schema number owned by DG**, NOT tied to
kumo's REST v1/v2 object model. (`kumoapi` is a flat package with no
v1/v2 split; kumo's manifest reader does a literal `if version != 1:`
integer compare with no kumoapi imports; sibling artifacts inside the
same bundle version independently — `stats.json` already accepts both
v1 and v2 for `tt_stats` while the manifest stays at v1.)

What the version field actually does today:

1. **Fail-fast sanity check** — catches "wrong file at this path" or
   "wheel/kumo out of sync" failures with a clean error message
   instead of cryptic `KeyError` deep in the parser.
2. **Hedge for future incompatible changes** — if/when a non-additive
   change is needed (rename a required field, restructure nested
   objects, change a format string), bumping to 2 is possible. Today
   every change has been additive (new `Option<T>` fields with
   `skip_serializing_if`), so the version has stayed at 1.

When v2 plausibly happens: when kumo's Rust-native dataloader replaces
the legacy GraphStore, the transitional sidecars all delete and the
manifest restructures. That breaking change would land as `version: 2`
with a coordinated kumo-side dual-version reader during transition.

### What kumo does with each piece

| Field | Consumer | Permanent? |
|---|---|---|
| `tt.parquet` | kumo connector (pkey='arange'), Python sampler, materializer, future Rust dataloader | **permanent** |
| `split.bin` | kumo split_module builds train/val/test masks | **permanent** |
| `stats.json` (KumoContractStats) | `train_table_generator.py:1259` validator + class-weighting in `baseline/train/global_module.py` | **permanent (target_distribution is load-bearing)** |
| `manifest.json` | `validate_artifact_bundle` | **permanent** |
| `split_stats.json` sidecar | `parse_split_stats_artifact` (display-only per grep) | transitional |
| `train_table_col_stats.json` sidecar | `seed_train_table_col_stats_from_manifest` for encoder inference | transitional |
| `train_table_edges` (in manifest) + reverse-edge dirs + fkey_cols + mask | legacy GraphStore via `_materialize_train_table_edges_from_manifest_bundle` | transitional |
| `arange.bin` sidecar | (none today — kumo bridge re-injects into tt.parquet) | transitional |

### `target_distribution` — the load-bearing field

For Categorical / Boolean target (e.g. binary classification):
```json
{
  "count": {"True": 1234, "False": 567},
  "percentage": {"True": 0.685, "False": 0.315}
}
```

For Numerical target:
```json
{
  "target_mean": 3.42, "target_min": 1.0, "target_max": 5.0,
  "target_percentile_25": 2.0, "target_percentile_75": 4.0,
  "target_median": 3.0,
  "target_mode": null, "target_mode_percentage": null, "target_mode_count": null,
  "histogram": {"bins": [...], "counts": [...], "bin_width": 0.4, "num_bins": 10}
}
```

Used by kumo's class-weighted loss in
`kumo/baseline/train/global_module.py`. Without it, an imbalanced
binary classifier trivially learns the majority class. For numerical
regression, `target_mean` / `target_min` / `target_max` drive z-score
normalization.

This is the **single most load-bearing field** of `KumoContractStats`.
It survives in end-state when everything else gets stripped.

## 9. Bridge code (transitional, isolated, deletable)

Kumo's training pipeline today expects several artifacts that don't
correspond to the end-state DG output shape: an `arange` index column
inside tt.parquet (a row-offset column), per-link reverse-edges
parquet dirs (consumed by the legacy GraphStore), col-stats sidecar
JSON keyed by logical column names. These are bridged today by
clearly-isolated transitional code that will delete when kumo
migrates.

Three structural properties make end-state migration easy:

### (a) Bridges live in dedicated files with deletion docstrings

| File | LOC | When it deletes |
|---|---|---|
| `diskgraph-train-table/src/bridge_pass.rs` | 124 | When ML migrates off arange-column-in-parquet |
| `diskgraph-train-table/src/kumo_contract_bridge.rs` | 802 | When kumo migrates off sidecar JSON to summary['stats'] |
| `diskgraph-train-table/src/custom_tt_chunk_processor.rs` | ~750 | When kumo's Rust dataloader replaces the legacy custom-TT path (and a spec-driven custom-TT API supersedes it) |
| `diskgraph-pybind/src/custom_tt_entry.rs` | ~180 | Deletes with the chunk processor |
| `diskgraph-pybind/python/diskgraph/tt_artifacts.py` | ~180 | Deletes with the entry |
| `kumo/data/generic/_diskgraph_legacy_graphstore_bridge.py` | ~400 | When kumo's Rust dataloader replaces legacy GraphStore (carries `normalize_legacy_parquet_for_rehydration` pre-bridge + `_inject_arange_weight_and_patch_checksum` combined post-bridge) |
| `kumo/engine/diskgraph_native_tt_artifact_rehydration.py` | ~250 | When kumo emits DG sidecars natively at TT-gen time |
| `kumo/task/task.py:591` (`_PHYSICAL_TO_LOGICAL` alias map) | ~12 | Deletes with the bridge above |
| `kumo/data/generic/diskgraph_batch_pred.py` (DG→kumo column rename) | ~15 | Deletes with the bridge above |
| `kumo-tests/kumotests/comparison/comparators.py` `_TRAIN_TABLE_COL_ALIASES` | ~25 | When graphengine retires (no more comparison) |

Total transitional code: ~2700 LOC across 10 files. Each file has a
header docstring naming its consumers and explicit deletion plan.

### (b) Each bridge has a single call site

Bridges aren't woven through the codebase:

- `bridge_pass::write_arange_sidecar` — called once in
  `ttgen_executor.rs` after pass B drains.
- `kumo_contract_bridge::write_all` — called once at the end of
  `execute_tt_pass_with_source`, between manifest construction and
  `write_manifest`.
- `apply_legacy_bridge` — called once from `diskgraph_train_table.py`
  after `execute_tt_plan` returns.

Removing each is a one-line edit + the file delete.

### (c) Permanent surface is clearly separated

`KumoContractStats` (in `ttgen_stats.rs`) is the permanent
kumo-contract surface. It will not be deleted — kumo's reader migrates
from sidecar JSON to `summary['stats']`, which uses this struct
verbatim. Migration is "delete the sidecars, keep the payload". No
re-architecture.

The manifest's optional fields (`split_stats`, `train_table_col_stats`,
`train_table_edges`, `arange_sidecar`) are all `Option<ArtifactRef>`
with `skip_serializing_if`. Drop them from the struct and they elide
cleanly.

### What the bridges do, concretely

The kumo-side `apply_legacy_bridge` (PQ + custom-TT post-bridge):

1. **`_inject_arange_weight_and_patch_checksum`** — single rewrite pass
   that appends both the int64 arange row-index column AND, when the
   manifest declares `weight_sidecar` non-null, an f32 weight column
   (loaded from the DGWGHT-formatted weight.bin), then renames
   `ENTITY1→ENTITY`, `TARGET1→TARGET` (slot-0 only) and refreshes
   `manifest['checksums']['train_table_sha256']`. One read, one write,
   one checksum patch — chosen over two-pass to avoid double I/O at
   scale. Idempotent: skips already-present columns on retried runs.
2. **Call `diskgraph.export_tt_artifacts`** to produce reverse-edges
   parquet dirs + fkey_cols + mask for the legacy GraphStore consumer.
3. **Write `train_table_edges` manifest entry** with inner+outer
   sha256 checksums.

The kumo-side **custom-TT pre-bridge** `normalize_legacy_parquet_for_rehydration`:

1. Read customer training-table parquet (via `pq.read_table`,
   preserving Arrow types).
2. Rename `ENTITY → ENTITY1`, `TARGET → TARGET1` so DG's canonical
   multi-entity column names line up with what
   `output_mode_for_custom_tt` reads. Only renames when the legacy name
   exists AND the canonical name doesn't.
3. Write to `<output_dir>/normalized_input.parquet` and return the
   path. The pybind entry consumes this path; kumo's post-bridge
   reverses the rename.

The DG-side `kumo_contract_bridge::write_all`:

1. **Emit `split_stats.json`** with per-stage `{num_train_rows, num_entities}`.
2. **Emit `train_table_col_stats.json`** with `{col_name__stype__stat__approx: value}` keys (kumo's encoder inference reads this).

The DG-side `bridge_pass::write_arange_sidecar`:

1. Write `arange.bin` (N × 8 bytes, little-endian Int64). Currently
   re-injected into tt.parquet by the kumo bridge; in end-state, the
   Rust dataloader reads it directly.

The DG-side custom-TT chunk processor `weight_bin.rs`:

1. Optional emit. When `plan.weight_col` is `None`, **no weight.bin
   written** and the manifest's `weight_sidecar` field is `None`. When
   `Some(col_name)`, the chunk processor reads the f32 column from the
   customer parquet and writes `weight.bin` with format `DGWGHT\0\0` +
   `version=1` (LE u32) + `dtype_tag=1 (f32)` (LE u32) + `row_count`
   (LE u64) + payload (N × 4 bytes f32). Manifest's `weight_sidecar`
   gains the path + sha256.

### Custom-TT design decisions (locked)

These shape the bridge boundary between kumo's existing pipeline and
DG's typed-Arrow custom-TT entry. Each is settled — re-litigation
should reference this section first.

1. **`input_timestamp_unit` declared at the bridge, not inferred.**
   Kumo's canonical TIMESTAMP representation is int64 epoch
   nanoseconds (Spark `LongType` written by the upstream
   materialization activity, well before our pre-bridge runs). DG's
   typed-Arrow path self-describes; raw Int64 columns do not. Rather
   than adding a magnitude heuristic (rejected — every guess can be
   weaponized by an unusual timestamp range) or defaulting to ns
   (rejected — defaults are inference), the pybind entry takes
   `input_timestamp_unit: &str` as a **required** kwarg parsed via
   `TimeUnit::from_name`. Kumo's rehydration glue passes
   `"nanoseconds"` literally, declaring its convention at the bridge.
   `read_timestamp_secs` uses the declared unit only for `Int64Array`;
   typed `Timestamp(Nanosecond/Microsecond/Millisecond/Second)` arrays
   self-describe and ignore the declaration.

2. **The unit lives on `CustomTtSource`, not `DiskGraphTTPlan`.** PQL
   never reads raw int64 timestamps — its executor builds typed
   `TimestampNanosecondArray`s end-to-end. Carrying the field on the
   shared plan struct would force a meaningless `Option<>` on every
   PQL call site. `CustomTtSource` is constructed only at the bridge,
   so the field is required and unconditional there.

3. **Reuse `diskgraph_core::time::TimeUnit`.** No bespoke enum. The
   existing `TimeUnit` is serde-friendly (`snake_case`) and ships with
   `from_name(&str) -> Option<TimeUnit>` for string-to-enum parsing.
   Non-sub-second variants (`Minutes`, `Hours`, ...) return `None`
   from `read_timestamp_secs` (not meaningful for epoch timestamps).

4. **`PredictionType::Node` + empty stages, never `PassThrough`.**
   `PassThrough` in the executor drops `TARGET1` (it was designed for
   inference-only paths that emit `(ENTITY, TIMESTAMP)` only). Empty
   stages on a `Node` plan preserves `TARGET1` while still skipping
   plan compilation entirely (every output column comes from the
   customer parquet). Verified by probe before the design was locked.

5. **`target_is_bool_dtype` reads from the resolved `OutputMode`, not
   `plan.primary_target_dtype()`.** For empty-stages plans,
   `infer_target_dtype` falls through to `DataType::Float64` — losing
   the Boolean signal needed for JSON-bool emission in
   `CATEGORY_COUNTS`. The chunk processor's
   `output_mode_for_custom_tt` already reads the customer parquet's
   actual TARGET1 dtype; `ChunkStats::target_is_bool_dtype` is
   initialized from that resolved value. (Note: kumoapi's `Stype` enum
   has no `boolean` variant — bool targets use stype `"categorical"`.
   The dtype flag is what triggers correct JSON-bool emission, not
   the stype string.)

6. **Option X (accepted parquet duplication) for rehydration.** The
   chunk processor rewrites `tt.parquet` in `output_dir` even though
   the customer parquet is already on disk a directory up. Inefficient
   at scale but transitional code (deletes when end-state arrives), so
   we don't engineer around it. The whole rehydration path is
   bounded-cost: kumo's call site's parquet is at most
   `train_table_size`, not `index_size`.

7. **Combined arange + weight injection in one rewrite pass.** Kumo's
   post-bridge rewrites `tt.parquet` exactly once, appending both
   columns and patching the checksum together. Two passes would double
   the I/O cost on the row count axis (the single hot dimension in
   custom-TT's I/O budget).

## 10. End-state vision

When kumo's Rust-native dataloader replaces the legacy GraphStore (the
major end-state migration):

- Python sampler dies. Rust sampler in DG resolves fkeys and emits
  sub-graph tensors directly to `next(dataloader)`.
- Legacy GraphStore + materialize_training_task activity dies.
- `arange.bin` sidecar dies — Rust dataloader iterates parquet
  natively; row-index column not needed.
- `fkey_cols.parquet`, `mask.parquet`, reverse-edge parquet dirs die —
  Rust sampler resolves fkeys on-the-fly via the index's pkmap
  (microseconds per row).
- `train_table_col_stats.json` sidecar dies — col stats flow through
  `summary['stats']` directly to kumo's encoder inference; or kumo's
  Rust dataloader does inference natively.
- `split_stats.json` sidecar dies — split logic moves entirely into
  DG; kumo no longer reads per-stage stats.
- `train_table_edges` manifest entry dies along with the GraphStore.
- Most kumo-side `task.py`, `materialize_training_task`,
  split-and-mask code dies.

What stays:

- `tt.parquet` (canonical TT data).
- `split.bin` (per-row stage byte).
- `KumoContractStats.target_distribution` (class weighting in
  `global_module.py`).
- `KumoContractStats.{min,max,anchor}_date`, counts.
- `manifest.json` (artifact manifest — possibly bumped to v2 at that
  point).

## 11. Migration path — deletion order

Each step is a small, reviewable PR. None require coordinated rollouts
because kumo and DG ship separately and the sidecars are
emitted-and-consumed in lock-step.

1. **Kumo's Rust-native dataloader replaces legacy GraphStore + Python
   sampler + PyG.** The major migration; the rest follow.
2. `git rm kumo/data/generic/_diskgraph_legacy_graphstore_bridge.py` +
   simplify `diskgraph_train_table.py` (one-line revert).
3. `git rm test/data/generic/test_diskgraph_legacy_graphstore_bridge.py`.
4. Revert `kumo/task/task.py` alias map (~12 lines).
5. Revert `kumo/data/generic/diskgraph_batch_pred.py` rename (~15 lines).
6. `git rm diskgraph-train-table/src/bridge_pass.rs` + drop
   `arange_path` from `TTPassOutput` + drop `arange_sidecar` from
   `manifest.rs`.
7. `git rm diskgraph-train-table/src/kumo_contract_bridge.rs` + drop
   `kumo_contract_bridge::write_all` call + drop `split_stats` +
   `train_table_col_stats` `Option<ArtifactRef>` fields from
   `manifest.rs`.
8. **Delete Pass A col_stats accumulation in ttgen.** Once seed-tables
   PR (#270) merged, every TT subdir produced by `execute_tt_plan` runs
   through DG ingest at `train_tables/<tt_id>/<stage>/` and gets a
   `col_stats.json` from the standard DG ingest observer — the same
   stats (Welford + TDigest + HLL) that Pass A's `ChunkStats` produces.
   The Pass A accumulation is now redundant work on the hot path. Delete:
   - `diskgraph-train-table/src/ttgen_stats.rs` — `ChunkStats`
     struct + `build_inline_col_stats` builder.
   - `WelfordAccumulator` / `TDigestEstimator` / `HyperLogLog` fields
     and update logic in `ttgen_executor.rs` (around lines 1456-1481)
     and `custom_tt_chunk_processor.rs` (around lines 389-406).
   - `train_table_col_stats.json` emission (already gated on the
     `Option<ArtifactRef>` removed in step 7).
   - Kumo's `seed_train_table_col_stats_from_manifest` (moved to step 9).

   Consumers must read `<tt_id>/<stage>/col_stats.json` from the DG
   sub-index instead of `train_table_col_stats.json`. The `discover_tts`
   + merged-reader path exposes these; the consumer redirect is the
   prerequisite for this deletion.

   **Estimated savings on the hot path:** Pass A's per-row accumulation
   is ~50 cycles × N_columns. On a 100M-row TT, that's tens of seconds
   of CPU per worker eliminated. Memory: ~MBs per worker (bounded by
   accumulator state, not row count).
9. Revert `kumo-tests` comparator alias when graphengine retires.
10. Drop kumo's `seed_train_table_col_stats_from_manifest`,
    `_materialize_train_table_edges_from_manifest_bundle`, and
    `validate_artifact_bundle`'s now-unreachable strict checks.

## 12. Performance characteristics

### Memory budgets (verified at micro-scale, projected to 50B rows)

| Component | Per-worker memory | Bound |
|---|---|---|
| Welford accumulator per column | ~32 bytes | constant |
| t-digest per quantile-tracked column | ~1KB (default centroids) | constant |
| HLL per cardinality-tracked column | ~16KB (precision 14) | constant |
| `LocalColumnCounts` per categorical column | bounded by top-N (typically 1024) | constant |
| `num_links_per_ts` HashMap | bounded by unique ts cardinality (~1M typical, low-MB) | data-dependent but small |
| `ChunkStats` per worker | sum of above ≈ ~256KB | constant |
| `StageCounts` per worker | 4 × u64 = 32 bytes | constant |

At 64 workers × 256KB ≈ 16MB total for stats. Plus the bounded(16)
channel buffer ≈ ~16MB of pending RecordBatches. Plus `split.bin` of
size N (~50GB for 50B rows). Plus the index pages in kernel page
cache.

The legacy permutation `Vec<u64>` of size N (400GB at 50B rows) and
`HashSet<u64>` cardinality fields (another 400GB) are gone.

### Concurrency model

- **Pass A**: rayon parallel chunked iteration over the entity table.
  Workers send to `bounded(16)` crossbeam channel; one writer thread
  drains to `tt.parquet` via `ArrowWriter`. Channel back-pressure
  prevents unbounded buffering.
- **Pass B**: rayon parallel iteration over parquet row groups. Each
  worker `pwrite`s to disjoint offsets in `Arc<File>`. No mutex, no
  atomic, no channel. `set_len(N)` called BEFORE the parallel scope
  so no kernel size-update contention.
- **Stats merging**: per-worker `ChunkStats` folded sequentially in
  the orchestrator after pass A. O(num_workers), not O(N).

### Per-row hot path

- `RowProcessor<U, D>` is monomorphized over `D: SplitDecider<U>`.
  Decider calls inline. No vtable.
- `SplitDecider::BatchView<'a>` (GAT) caches column lookup + Arrow
  downcast at batch construction. Per-row decide is a memory load + a
  hash mix + a bucket compare.
- `ChunkStats::observe_row` fans out to per-column accumulators; each
  is branchless inline arithmetic.

### Performance gates

- `cargo test --release -p diskgraph-train-table --test ttgen_integration`:
  131 tests, p50 wall = 3.44s. Gate: ≤3.7s.
- Wheel build + install via `update_diskgraph.sh --build-only`: ~22s.
- amazon_10k full e2e (graphengine ref + diskgraph candidate):
  ~10 minutes total.

## 13. Testing strategy

| Layer | Count | Coverage |
|---|---|---|
| `diskgraph-train-table/src/*` unit tests | 232 | Per-module accumulator, decider, batch-view, regroup, manifest serde, stats payload shapes. |
| `tests/ttgen_integration.rs` | 131 | Integration: each split family × each pass-A/B path. p50 3.44s. |
| `tests/ttgen_e2e.rs` | 16 | Manifest schema round-trip, kumo-contract sidecar shapes, multi-entity output, bridge behavior. |
| `tests/custom_tt_integration.rs` | 15 (3 ignored) | Custom-TT path with synthetic customer parquets. |
| Total | **394 passed, 0 failed** | |

Plus on the consumer side:

| Layer | Count |
|---|---|
| kumo `pytest test/data -k diskgraph` | 35 (30 baseline + 5 bridge tests) |
| kumo-tests comparison (graphengine vs diskgraph on amazon_10k) | 3/3 stages PASSED, 1/1 fixture PASSED |

## 14. Open follow-ups

| Item | Why deferred | Effort |
|---|---|---|
| LP+array+CustomLinkSplit regroup parallelization | Rare case, v0 sequential is fine; not on 50B-row hot path | small |
| Per-stage entity HLL (split_stats `num_entities` exact count) | Pass-A architectural change vs pass-B projection extension; kumo only displays this field per grep, so not blocking | medium |
| Delete Pass A col_stats accumulation (redundant with DG ingest observer post-PR #270) | Needs consumer redirect to `<tt_id>/<stage>/col_stats.json` first. See §11 step 8 for the full deletion. Hot-path win: ~50 cycles × N_columns per row eliminated | small (deletion) + medium (consumer audit) |
| Manifest version 2 design | Premature until a non-additive change is actually needed | reactive |
| Multi-file parquet output (parquet dataset mode) | Would change kumo's reader contract; not worth it for the row-order-flake fix | medium |
| Replace legacy `RandomSplit` permutation byte-for-byte parity | Structural — streaming model is incompatible. Aggregate ratios match; per-row assignments differ. Documented divergence. | won't fix |

(Forecasting `OutputMode::Forecasting` ships in this PR — see
`ttgen_output.rs:44,196` and `ttgen_executor.rs:1303,1644`.
DateOffset calendar-month support also ships via the
`month_offsets` plan field consumed by `DateOffsetDecider`.)

## 15. References

- Typed timestamp unit system: `docs/epoch_seconds_design.md`
- Module-level docstrings in each `diskgraph-train-table/src/*.rs`
  carry in-file context including concurrency model, schema, and
  deletion plan for transitional bridges.
- Kumo-side bridge: `kumo/data/generic/_diskgraph_legacy_graphstore_bridge.py`
  (header docstring documents what each step does and when each
  deletes).

---

**Status**: 394/0 tests pass; amazon_10k e2e validated end-to-end
against graphengine reference; `--custom-tt` comparison on
`amazon_10k_max_rating_count_distinct` (binary fixture) passes 3/3
stages with the full kumo→DG rehydration path exercised; entity-aware
random hash shipping; transitional bridges isolated and deletable.

**Next milestone**: kumo Rust-native dataloader migration triggers
the deletion wave (steps 1-9 in §11).
