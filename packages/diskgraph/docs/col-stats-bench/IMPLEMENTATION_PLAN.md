# Column Statistics Implementation Plan

## Overview

Collect per-column statistics during DiskGraph's encode stage so downstream ML
encoders (kumo-ml) don't need a separate PySpark pass. Stats must be collected
with near-zero overhead when disabled and <5% overhead when enabled.

---

## Step 0: SemanticType + Stype Inference — DONE

### What was implemented

**`diskgraph-ingest/src/graph_config.rs`**:
- `SemanticType` enum: Numerical, Categorical, Id, Text, Timestamp, Sequence,
  Multicategorical, Unsupported.
- `semantic_type: Option<SemanticType>` field on `ColumnSpec` with
  `serde(default)` — existing configs without this field deserialize to `None`.
- `infer_semantic_type()` decision tree: dtype-based rules first, then name
  heuristics, then sample stats thresholds matching kumo-ml's `EncoderGenerator`.
- `ColumnRole` and `ColumnSampleStats` types for inference inputs.

**`diskgraph-rowpack/src/lib.rs`**:
- `int_uniques: HashSet<i64>` in `ColumnStats` — tracks unique integer values
  during context-phase sampling (10K rows), capped at `MAX_SAMPLE_UNIQUES = 1M`.
- `sample_unique_counts()` method on `RowPackContextBuilder` — returns
  `(num_unique, count)` per column using `int_uniques.len()` for ints and
  `string_counts.len()` for strings.

**`diskgraph-ingest/src/lib.rs`**:
- Context phase (`build_node_contexts`) wired to call `sample_unique_counts()`
  after 10K-row sampling, then `infer_semantic_type()` per column.
- Inferred `semantic_type` populated on `ColumnSpec` before encode phase begins.
- Channel extended to carry `Vec<SemanticType>` back to main thread.

---

## Step 1: Observer Pattern Infrastructure

### Architecture

**`ColumnValueObserver` trait** defined in `diskgraph-rowpack/src/lib.rs`:

```rust
pub trait ColumnValueObserver {
    fn observe_i64(&mut self, col_idx: usize, value: i64);
    fn observe_f64(&mut self, col_idx: usize, value: f64);
    fn observe_string(&mut self, col_idx: usize, value: &str);
    fn observe_timestamp(&mut self, col_idx: usize, secs: i64, nanos: u32);
    fn observe_float_list(&mut self, col_idx: usize, len: usize);
    fn observe_null(&mut self, col_idx: usize);
    // ... other type arms
}

pub struct NoOpObserver;
impl ColumnValueObserver for NoOpObserver {
    // All methods empty — LLVM eliminates via monomorphization
}
```

When col stats are **disabled**, `encode_row_from_access_into_buffers` calls
the new generic `encode_row_observed` with `NoOpObserver`. LLVM monomorphizes
and eliminates all observer calls. The compiled code is identical to today's.

When col stats are **enabled**, a real `ColStatsObserver` is passed in. Each
`observe_*` method feeds the value into the appropriate streaming algorithm
(Welford, HLL, reservoir, etc.).

### Why Observer Pattern

Three approaches were considered:

**Column-at-a-time (iterate raw Arrow arrays before per-row loop):**
Misses RowAccess coercions (timestamps as Int64/String, binary→UUID, etc.).

**Refactor to read-then-encode (read in ingest, pass values to rowpack):**
Strings can't be extracted without heap allocation — `with_string` borrows
`&str` inside a closure. 300K mallocs/batch = ~16% overhead.

**Observer pattern (piggyback on rowpack encode):** ← chosen
Observer called inside `with_string` closure → borrows `&str` from Arrow
buffer, zero allocation. Correct coercions by construction. Zero overhead
when disabled via monomorphization.

### Benchmark: Step 1

After implementing with `NoOpObserver`:
- Verify zero overhead via asm inspection and wall-clock benchmarks
- If overhead detected, investigate icache effects from function duplication

---

## Step 2: Stats Accumulation + Merge Strategy

### Approach: Shared Accumulator (Option A)

One `Arc<Mutex<HashMap<String, TableColStats>>>` per ingest run. Each worker
thread, after finishing a file, locks the mutex and merges its per-file stats.

**Why this works:**
- Lock hold time per merge: ~8ms (dominated by HLL register max-merge)
- File processing time: ~5-7 seconds
- Lock contention: <0.2% of thread time
- Memory: one accumulator per table × ~7.2MB = ~72MB for 10 tables. Constant.

**Why not map-reduce (dedicated reducer threads):**
Stats accumulation happens inline during encoding (~10ns per value via streaming
algorithms). You can't send 40M individual values per file through a channel —
the channel overhead alone exceeds the accumulation cost. Reducers can only
receive already-accumulated per-file stats, making them equivalent to Option A
but with fewer encode threads.

**Why not per-thread scratch (Option B):**
Each thread needs accumulators for every table it touches. 64 threads × 10
tables × 7.2MB = ~4.6GB. Worse memory than Option A. Also requires TaskPool
modification to return scratch state.

### FileColStats Structure

```rust
pub struct FileColStats {
    cols: Vec<ColumnCollector>,
}

pub enum ColumnCollector {
    Numerical {
        welford: WelfordAccumulator,  // mean/std/kurtosis/min/max
        hll: HyperLogLog,            // num_unique
        reservoir: ReservoirSampler,  // quantiles
    },
    Categorical {
        hll: HyperLogLog,
    },
    Id {
        hll: HyperLogLog,
    },
    Text {
        hll: HyperLogLog,
        token_count: u64,     // sum of tokens per value (for avg_tokens)
        non_char_count: u64,  // sum of non-alpha bytes (for non_char_fraction)
        string_count: u64,    // number of non-null strings observed
    },
    Timestamp {
        min_secs: i64, min_nanos: u32,
        max_secs: i64, max_nanos: u32,
    },
    Sequence {
        min_len: usize,
        max_len: usize,
    },
    Skip,  // PK columns, unsupported types
}
```

### Merge Formulas

| Stat | Merge | Complexity |
|------|-------|-----------|
| Welford (mean/var/skew/kurt) | Parallel combine formula (Chan et al.) | O(1) |
| HLL | Element-wise max of register arrays | O(2^p) = O(65536) |
| Reservoir (k=10K) | Weighted merge proportional to count | O(k) |
| Min/Max | min of mins, max of maxes | O(1) |
| Count/NullCount | Additive | O(1) |
| token_count, non_char_count | Additive | O(1) |

---

## Phase 1 Stats: What Goes Where

### Inline During Encode (observer)

| Stat | Algorithm | Per-value cost | Applies to |
|------|-----------|---------------|-----------|
| MEAN/STD/KURTOSIS | Welford online (M1-M4) | ~3ns | Numerical |
| MIN/MAX | Running comparison | ~1ns | Numerical, Timestamp |
| NUM_UNIQUE | HyperLogLog (p=14, 16KB/col) | ~10ns (hash) | All except Skip |
| QUANTILES Q25/Q50/Q75 | Reservoir sampling (k=10K) | ~3ns | Numerical |
| AVG_TOKENS | Count spaces per string | ~15ns | Text |
| NON_CHAR_FRACTION | Count non-alpha bytes per string | ~15ns | Text |
| SEQUENCE LENGTH | List length | ~1ns | Sequence |

**Total per-value overhead estimate:**
- Numerical columns: ~17ns (Welford + HLL + Reservoir + min/max)
- String/Text columns: ~40ns (HLL + avg_tokens + non_char)
- Id columns: ~10ns (HLL only)
- Timestamp columns: ~3ns (min/max only)
- Sequence columns: ~1ns (length only)

**Per-file overhead (800K rows, 50 cols):** ~600-800ms on a 5-7s file = ~10-12%.
With semantic_type-based skipping (Id columns only HLL, Timestamp only min/max),
effective overhead closer to **7-8%**. Measured per-step as each stat is added.

### Separate Post-Encode Pass (Phase 2)

| Stat | Reason |
|------|--------|
| CATEGORY_COUNTS (HashMap<Value, u64>) | Up to 1M entries per column, expensive HashMap insert + string clone. ~100-200ns per value. |
| GloVe coverage | Requires loading external vocabulary. Better in kumo-ml. |
| Multicategorical separator detection | Separator inference on samples, complex logic. |

### Context Phase Only (10K sample, already done)

| Stat | Reason |
|------|--------|
| num_unique for stype inference | Used to populate semantic_type before encode |
| String max_len, avg_len | Already tracked by RowPackContextBuilder |

---

## Implementation Order + Benchmarking

Each step: implement → benchmark → verify no regression → next step.

| Step | What | Files Changed | Expected Overhead |
|------|------|---------------|-------------------|
| 0 | SemanticType + stype inference | graph_config.rs, lib.rs, rowpack | **DONE** |
| 1 | Observer trait + NoOpObserver + encode_row_observed | rowpack/lib.rs | 0% (compiled away) |
| 2 | Benchmark data generation (diverse schema) | generate.py | N/A |
| 3 | FileColStats + shared accumulator + wiring | col_stats.rs, lib.rs | ~0.5% (infrastructure) |
| 4 | Welford (mean/std/kurtosis) + min/max | col_stats.rs | ~1% |
| 5 | HLL (num_unique) | col_stats.rs | ~1-2% (hash per value) |
| 6 | Reservoir (quantiles) | col_stats.rs | ~0.5% |
| 7 | Timestamp + Sequence stats | col_stats.rs | ~0% |
| 8 | Text stats (avg_tokens, non_char_fraction) | col_stats.rs | ~1% |
| 9 | JSON output + correctness verification | col_stats.rs | ~0% |

---

## Benchmark Data Generation

Update `docs/col-stats-bench/generate.py` to produce diverse data:

### Schema

**users** (node table, 10 columns):

| Column | Arrow Type | SemanticType | Distribution |
|--------|-----------|--------------|-------------|
| user_id | Utf8 | Id (PK) | Sequential hex IDs |
| username | Utf8 | Categorical | High cardinality (1M unique), Zipf |
| country | Utf8 | Categorical | Low cardinality (200 unique), uniform |
| status | Utf8 | Categorical | Very low cardinality (5 unique), uniform |
| age | Int32 | Numerical | Uniform(18, 80) |
| balance | Float64 | Numerical | Normal(5000, 2000) |
| score | Float32 | Numerical | Lognormal(3.0, 1.5) |
| signup_year | Int16 | Numerical | Uniform(2000, 2025) |
| is_active | Boolean | Categorical | 50/50 |
| created_at | Timestamp(us) | Timestamp | Uniform(2020, 2025) as Arrow Timestamp |

**products** (node table, 8 columns):

| Column | Arrow Type | SemanticType | Distribution |
|--------|-----------|--------------|-------------|
| product_id | Utf8 | Id (PK) | Sequential hex IDs |
| category | Utf8 | Categorical | Medium cardinality (500 unique) |
| brand | Utf8 | Categorical | High cardinality (50K unique), Zipf |
| price | Float64 | Numerical | Lognormal(3.0, 1.5) |
| weight_kg | Float32 | Numerical | Exponential(2.0) |
| rating | Float32 | Numerical | Normal(3.5, 1.0) clamped to [1,5] |
| in_stock | Boolean | Categorical | 85/15 |
| last_updated | Int64 | Timestamp | Epoch microseconds |

**devices** (node table, 7 columns):

| Column | Arrow Type | SemanticType | Distribution |
|--------|-----------|--------------|-------------|
| device_id | Utf8 | Id (PK) | Sequential hex IDs |
| device_type | Utf8 | Categorical | Low cardinality (50 unique) |
| os_version | Utf8 | Categorical | Medium cardinality (200 unique) |
| screen_size | Float32 | Numerical | Uniform(4.0, 32.0) |
| battery_mah | Int32 | Numerical | Normal(5000, 2000) |
| embeddings | List\<Float32\> | Sequence | Length uniform(64, 256) |
| firmware_ts | Utf8 | Timestamp | ISO 8601 strings |

### Sizes

Default: 100M rows/table (100 files × 1M rows/file).
Quick test: 10M rows/table (10 files).

---

## Correctness Verification

### Unit Tests

- **Welford**: Known distributions (Uniform, Normal, Exponential)
- **HLL**: Known cardinalities (100, 10K, 1M), verify within 1% error
- **Reservoir**: Compare Q25/Q50/Q75 against exact numpy quantiles
- **Welford merge**: Split data, merge partitions, compare against single-pass

### Integration Test

- Generate small dataset (1M rows)
- Run ingest with `--col-stats`
- Compare each stat against pandas/numpy on same parquet files
- Script: `docs/col-stats-bench/verify.py`

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `diskgraph-ingest/src/graph_config.rs` | ColumnSpec, SemanticType, infer_semantic_type() |
| `diskgraph-rowpack/src/lib.rs` | ColumnValueObserver trait, encode_row_observed, int_uniques |
| `diskgraph-rowpack/src/access.rs` | RowAccess (value reader with coercions) |
| `diskgraph-ingest/src/col_stats.rs` | FileColStats, TableColStats, merge, JSON output (TBD) |
| `diskgraph-ingest/src/lib.rs` | Wiring: shared accumulator, observer in encode path |
| `diskgraph-ingest/src/bin/ingest.rs` | --col-stats CLI flag |
| `docs/col-stats-bench/generate.py` | Benchmark data generation |
| `docs/col-stats-bench/run_bench.sh` | Benchmark runner |
| `docs/col-stats-bench/verify.py` | Correctness verification |
