# Column Statistics

## Context: Where Col Stats Fits in Ingest

DiskGraph ingest has four phases:

1. **Schema inference** — detect column types, timestamp formats, FK relationships from parquet metadata + sampling
2. **Encode** — row-by-row conversion from Arrow to rowpack format, building node data blobs. Col stats piggybacks here.
3. **PK map** — build MPHF (minimal perfect hash function) for primary key → node ID lookup
4. **Edges map** — build sorted adjacency lists from FK columns

Col stats runs during phase 2. Output: `col_stats.json` alongside the index.

Encode is 26-40% of total ingest wall time. The rest (pk_map, edges_map, I/O) is unaffected by col stats. On edge-dominated datasets (Faire: 553M edges), encode is a small fraction and col stats overhead is invisible (<2%). On small datasets where encode dominates (ifood, databricks), overhead is measurable (19-22%).

## The Observer Architecture

Col stats uses the `ColumnValueObserver` trait in `diskgraph-rowpack`. The encode pipeline calls typed observer methods (`observe_i64`, `observe_f64`, `observe_string`, `observe_timestamp`, `observe_float_list`, `observe_null`) for each value as it passes through the encoder. The observer implementation accumulates statistics without owning or copying the data.

This is a general-purpose extensibility point. Any computation that needs per-row access to column values during ingest — without a separate data read — can implement `ColumnValueObserver`. Col stats is the first consumer, but the pattern supports future uses:

- **Data quality checks** — NULL rates, outlier detection, value range validation
- **Histogram construction** — for query optimization or data profiling
- **Sampling** — reservoir sampling for representative row subsets
- **Schema drift detection** — comparing current ingest against previous stats

The key design properties:
- **Zero-copy**: Observer receives references (`&str`, `i64`, `f64`), never allocates per value
- **Per-file accumulation**: `FileColStats` accumulates within a single file's encode. Thread-safe because each file is processed by one thread.
- **Merge-at-end**: `TableColStats` merges per-file results using algorithm-specific merge (Welford, HLL register-max, T-Digest sort-merge). Merge happens once per file, not per row.
- **No backpressure**: Observer returns void — it can't slow down the encode pipeline. If accumulation is expensive, it must be amortized (see category counting).

## What Gets Computed

| Column Type | Stats | Algorithm | Per-Column Memory |
|-------------|-------|-----------|-------------------|
| Numerical | count, null_count, mean, std, min, max, skewness, kurtosis | Welford online accumulator (exact, f64) | 64B |
| Numerical | q25, q50, q75 | T-Digest (δ=200, <0.02% range-relative error) | 3.2KB |
| Numerical | num_unique | HyperLogLog (p=14, <1% avg error, <4% worst) | 16KB |
| Categorical/ID | qualifying set + exact counts | Hybrid strategy (see below) | Variable |
| Text | avg_tokens, max_tokens, non_char_fraction, glove_overlap | 10K-row sample during context phase + GloVe bloom filter (5MB) | Negligible |
| Timestamp | count, null_count, min, max | Exact counters | 32B |
| Sequence (float list) | count, null_count, min_len, max_len, mean, std | Exact (f64) | 64B |

Non-category stats total ~18KB per column — fits L1 cache. All support exact parallel merge.

## Category Counting

The ML framework uses category information to choose encoders:
- **Index encoder** (≤1M qualifying categories): value→index mapping for learned embeddings. Needs the qualifying set and its exact counts.
- **Hash encoder** (>1M qualifying categories): hashes values directly. Needs only cardinality from HLL.

A category "qualifies" if it appears ≥ `min_occ = sqrt(num_unique)` times. Columns exceeding 2M uniques abort counting immediately.

### Production Strategy: Hybrid (Exact Counts)

The Kumo integration uses **Hybrid** exclusively. Hybrid produces exact category counts for every qualifying category — full parity with what Spark computes. This was a deliberate choice: the ML team requires exact count parity to validate that DG-trained models match GE-trained models. Approximate counts could mask real divergences during the migration.

Local `HashMap<u64, u64>` per thread capped at 5K entries (~120KB, L2). On overflow, flush to shared `DashMap<u64, AtomicU64>` (256 shards). All keys are `u64` gxhash — uniform type, no string allocation on hot path, identity hashing in DashMap. String recovery via write-once map at output only.

### Future Optimization: Graduation (Approximate Counts)

Graduation builds on Hybrid. Key insight: once a category crosses `min_occ`, stop counting it. On Zipfian distributions, popular categories graduate within first few files — after warmup, 80%+ of rows skip counting entirely.

The graduated set is stored as `RwLock<Arc<HashSet<u64>>>`. Threads clone the `Arc` (~1ns) per flush and read from an immutable `HashSet` — zero per-read synchronization. Pure memory reads at 3-5ns.

**Per-row hot path:**
```
h = hash(value)
if local_grad.contains(&h) { return }    // ~3-5ns, immutable, zero sync
local_map.insert_or_increment(h)          // ~5ns, L2
```

Graduation finds 99.6-100% of qualifying categories. All misses are within 1.00-1.07× of `min_occ` — categories at the boundary. This is **not yet used in production** — requires ML team sign-off that approximate counts don't affect model quality. Available via `--cat-count-strategy graduation` CLI flag.

### Subsampling (Not Recommended)

A `--row-subsample` flag enables 30% per-row subsampling — skip 70% of rows randomly during category counting. Cheapest option (~5% overhead) but lowest quality: recall drops to 72-100% with misses up to 1.44× min_occ. Retained for experimentation only. Not suitable for production because the accuracy gap is too large for edge-case columns.

### Why Category Counting Is Hard

Unlike streaming stats (~18KB/column, L1), category counting tracks individual values across 200 threads on billions of rows. Per-row access to any shared structure costs ~40ns. With C categorical columns, the floor is `C × 40ns` against 5-8µs encode. At 18 columns = 11%; at 179 columns approaches 100%.

Approaches tried:

| Approach | Outcome | Why |
|----------|---------|-----|
| Per-thread HashMap, merge at end | +384% | 1M uniques × 200 threads = 4.8GB cold data, O(N) merge |
| Single DashMap, probe per row | +107% | Shard lock contention at 200 threads |
| DashSet for graduated set | +107% | Atomic reader count per `contains()` bounces cache lines across cores |
| **Frozen `Arc<HashSet>` + local buffer** | **+10-20%** | Zero per-read sync, local buffer in L2 |

The DashSet failure is instructive: even "read-only" concurrent data structures have per-access atomic overhead. `DashSet::contains()` does `fetch_add` + `fetch_sub` on a shard reader count — two atomic ops per probe at 10-15ns each. With 200 threads, the shard counter cache line bounces constantly. The frozen `Arc<HashSet>` pattern eliminates this entirely — clone once per flush, then pure immutable reads.

### Design Constants

| Constant | Value | Rationale |
|----------|-------|-----------|
| MAX_CATEGORIES | 1,000,000 | Above this → Hash encoder, no qualifying set needed |
| Local map cap | 5,000 entries | ~120KB, fits L2. Low-cardinality columns stay far smaller. |
| min_occ floor | 10,000 (→ min_occ ≥ 100) | Prevents premature graduation before HLL stabilizes (~1 file, ~6M rows) |
| Graduation batch | 64 | Rebuild frozen set every 64 new qualifiers |
| HLL precision | 14 | 16,384 registers, 0.81% expected error, 16KB/col |
| T-Digest δ | 200 | <0.02% range-relative quantile error, 3.2KB/col |
| GloVe bloom | 5MB, k=7 | 0.55% FPR for ~4.1M-word vocabulary |

## Output and Kumo Consumption

### Output Files

```
{index_path}/kumo_col_stats/
  col_stats_full/{table_name}.json   # Full stats with encoder-compatible fields
  col_stats_min/{table_name}.json    # Minimal stats
```

### How Kumo Reads Stats

DG col stats must reach Kumo's `StatsManager` before encoder inference runs. The flow has multiple handoff points because Kumo runs activities in isolated subprocesses:

```
DG ingest (Rust, encode phase)
  → col_stats JSON written to index directory
  → diskgraph_ingest_activity copies to kumo data backend cache
  → [subprocess boundary — cache is empty]
  → _ensure_diskgraph_col_stats_in_cache() re-seeds from DG index
  → StatsManager reads from cache for encoder inference
  → seed_train_stats_cache_from_native_artifact() for training-specific stats
```

`_ensure_diskgraph_col_stats_in_cache()` always overwrites from the DG index — the authoritative source. Don't rely on activity-to-activity cache propagation which breaks across subprocess boundaries and on cached-index early returns.

## Fixed Issues

| Issue | Fix | PR |
|-------|-----|-----|
| Local HashMap retained capacity → 100+ GB OOM | `*local = None` instead of `local.clear()` | #61 |
| Categorical ordering non-deterministic | Secondary alphabetical sort to break tied counts | #112 |
| Float64 categoricals ignored | Added float64 observation | #169 |
| Bool/float category keys wrong JSON type | Emit correct JSON types | #169 |
| Train table stats used only train split rows | Use all rows | #169 |
| NULL time_col rows counted | Skip NULLs (match Spark `drop_na`) | #169 |
| stats_per_timeframe empty timestamps | Use produced timestamps from split payload | #171 |
| stats_per_timeframe fake row counts | Real per-timestamp counts | #171 |
| Cache empty in subprocesses | Always re-seed from disk | kumo diskgraph-e2e |
| CATEGORY_COUNTS missing for binary TARGET | Seed from manifest before `task.materialize` | kumo #27665 |
| Ingest workflow returns early on cached index | Clean `NA-*` dirs to force re-ingest | kumo-tests |
| Encoder inference before stats seeded | Seed before inference; `ignore_encoders=True` fallback | kumo #27728 |

## Developing: Adding or Changing Statistics

### Adding a New Statistic

To add a new per-column statistic (e.g., a new distributional measure, a domain-specific check):

1. **Add observer method** to `ColumnValueObserver` trait in `diskgraph-rowpack/src/lib.rs` if the existing typed callbacks (`observe_i64`, `observe_string`, etc.) don't cover your needs. Usually they do — most new stats just need a different accumulator behind the same callbacks.

2. **Add accumulator state** to `FileColStats` in `col_stats.rs`. Keep it small — this lives in the per-row hot path. If your stat needs >1KB per column, it will affect L1 cache pressure across all columns. Benchmark on discord-safety (179 columns) to check.

3. **Add merge logic** to `TableColStats`. Your accumulator must support merging partial results from different files/threads. If it doesn't (e.g., requires seeing all values in order), it can't run in the streaming architecture — you'll need a separate pass, which means re-reading data.

4. **Add JSON output fields** and update both `col_stats_full` and `col_stats_min` schemas.

5. **Update Kumo consumption.** New stats that affect encoder inference need to flow through the `_ensure_diskgraph_col_stats_in_cache()` → `StatsManager` path. Test that the stat arrives correctly across the subprocess boundary.

6. **Benchmark.** Measure wall time overhead on at least two datasets: one small where encode dominates (databricks, 31GB) and one large where encode is a fraction (alt_detection, 2.2TB or faire, 127GB). Report overhead as percentage of total ingest wall time, not encode time alone.

### Changing Category Counting

Category counting is the most performance-sensitive part. Changes here must be benchmarked on various datasets (179 cat cols, 10B rows). Any change that adds per-row work to the hot path (even a single branch) can add measurable overhead at 179 columns.

The frozen `Arc<HashSet>` pattern for the graduated set is load-bearing — don't replace it with `DashSet` or `RwLock` reads per row without benchmarking. The difference between 3ns (immutable read) and 25ns (atomic read) is 22ns × 179 columns × 10B rows = hours of added runtime.

### Changing Stats Format

The JSON output is consumed by Kumo's `StatsManager`. Changing field names, removing fields, or changing semantics breaks the Python consumption path. Test with a full E2E run: ingest → TT gen → materialize → training. Col stats bugs often don't surface until encoder inference, which runs much later in the pipeline.

## File Locations

| Component | Path |
|-----------|------|
| Core implementation | `diskgraph-ingest/src/col_stats.rs` |
| Pipeline integration | `diskgraph-ingest/src/lib.rs` |
| Observer trait | `diskgraph-rowpack/src/lib.rs` |
| GloVe bloom data/builder | `diskgraph-ingest/data/glove_bloom.bin`, `src/bin/build_glove_bloom.rs` |
| Category counting design doc | `docs/category_counts.md` |
| Benchmarks | `docs/col-stats-bench/` |
