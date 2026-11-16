# Design Decisions

Architectural choices made during DiskGraph development and the constraints that shaped them.

## Streaming Single-Pass Architecture

DiskGraph computes everything — column statistics, TT generation, split assignments, edge artifacts — in streaming single-pass pipelines. No separate preprocessing steps, no multi-pass algorithms, no intermediate materialization to disk (except final output).

**Why:** Production datasets are 100GB-2TB. Multi-pass means re-reading data. Disk I/O is the bottleneck. A 5% overhead added to one pass beats a 0% overhead second pass that doubles I/O.

**Implications:** Every algorithm must be streaming-compatible. Welford for stats, HLL for cardinality, T-Digest for quantiles. Category counting required the most design effort because it inherently needs to track state across the full input (see [col_stats.md](../diskgraph-ingest/col_stats.md)).

## MPHF-Based PK Maps

Primary key → node ID resolution uses minimal perfect hash functions (epserde + ptrhash). Memory-mapped, immutable after construction, O(1) lookup with zero thread synchronization.

**Why:** PK lookups happen at multiple points — FK resolution during ingest, entity resolution during TT gen, orphaned FK filtering during ListDistinct. Each point is on a hot path in a parallel context. Any locking or atomic overhead per lookup compounds across billions of rows × 200 threads.

**Implications:** All PK-dependent operations are lock-free. The `contains_pk_value()` function added in PR #173 follows this pattern — it's a pure read on mmap data, safe to call from any thread at any time.

## Manifest-Based Artifact Validation

All TT gen artifacts (parquet, split assignments, stats, edge bundles) are referenced by a JSON manifest with SHA256 checksums. The manifest is the single source of truth for what was produced.

**Why:** The Kumo pipeline runs artifacts through multiple Temporal activities, each in its own subprocess. Artifacts move between local disk, S3, and cache directories. Without checksums, a corrupted copy or stale cache silently produces wrong results. With checksums, corruption is detected at load time.

**Where:** `diskgraph_native_tt_artifacts.py:validate_artifact_bundle()` verifies every artifact against its manifest checksum before use.

## Strict Path (native_full)

When `deploy_env().is_diskgraph == True`, DG handles the entire pipeline: ingest, col stats, TT gen, splits, edge artifacts. No Spark fallback. If an artifact is missing, the pipeline fails with a clear error instead of falling back to Spark.

**Why:** Mixed Spark/DG execution creates impossible-to-debug inconsistencies. Stats computed by Spark and DG can differ subtly (NULL handling, timestamp precision, categorical ordering). Using one system end-to-end eliminates cross-system divergence.

**The alternative (shadow mode)** runs DG in parallel with Spark for validation. Useful during development but not for production — doubles compute cost.

## Rayon Parallel Entity Loop

TT gen parallelizes across entities using rayon with fine-grained chunks (`num_threads × 8`). Each chunk processes a range of entity IDs independently.

**Why work-stealing over fixed partitioning:** Entity processing time varies — some entities have many timestamps and neighbors, others have few. Rayon's work-stealing distributes load dynamically. More chunks than threads (8x) ensures the last finishing chunk doesn't bottleneck.

**Per-chunk state:** `aliases: Vec<Option<Value>>` and `link_frequencies: Vec<i64>` are pre-allocated per chunk and reused across entities. No per-entity allocation.

**Back-pressure:** A `bounded(16)` channel between rayon workers and the Parquet writer prevents unbounded memory growth. Workers block on send if the writer falls behind.

## pyo3 Streaming Iterator

TT gen results stream from Rust to Python via `TTPlanIterator` — a PyO3 class implementing `__iter__`/`__next__`. Each call to `__next__` receives one `RecordBatch` from a crossbeam channel.

**Why streaming over disk shards:** The original pipeline wrote Parquet shards to disk, then read them back in Python. Streaming eliminates the I/O round-trip and reduces peak disk usage. Back-pressure naturally throttles the Rust producer.

**GIL management:** `__next__` releases the GIL via `py.detach()` before blocking on `recv()`. The background Rust thread may use `tracing::info!()` which routes through pyo3-log and needs the GIL — if `__next__` held the GIL while blocking, deadlock (PR #158).

## Edge Bundle from TT Gen

DG produces edge artifacts (fkey_cols, reverse edges, masks) during TT gen alongside the training table. These are bundled in the manifest and reused during materialization.

**Why not materialize edges separately:** Edge materialization in Spark reads the training table, joins with destination tables, and writes edge parquets — expensive I/O. DG already has the graph index and the TT data during TT gen. Producing edges at the same time avoids re-reading.

**The fkey_cols parquet** maps each TT row to node indices in destination tables. When a target FK value doesn't resolve to a valid node (orphaned FK), the entry gets `-1`. Edge materialization drops these, which can cause fkey size mismatches (PR #173 fixes this upstream).

## Custom TT Path for Comparison Testing

Cross-backend comparison tests use a custom TT workflow: generate TT on GE, export to S3, both backends train from the same custom TT via `TrainingTable.update()`.

**Why not compare TT gen directly:** TT gen differences are expected during development (different temporal join semantics, NULL handling, FK resolution). Training comparison requires identical input data. The custom TT path isolates training behavior from TT gen behavior.

**Limitation:** Custom TT can't test static queries (no TIMESTAMP column) or queries where the TT structure differs between backends.

## Col Stats Re-Seeding Pattern

`_ensure_diskgraph_col_stats_in_cache()` always overwrites Kumo's cache from the DG index, even if the cache appears populated.

**Why always overwrite:** Kumo runs activities in isolated subprocesses. Each subprocess starts with an empty in-memory cache. The on-disk cache may be stale (previous graph version), partially populated (ingest workflow returned early on cached index), or missing encoder-specific fields (CATEGORY_COUNTS). The DG index is the authoritative source — always copying from it guarantees correctness regardless of cache state.

## Non-Blocking SDK + Filesystem Polling

Test scripts use `non_blocking=True` for SDK calls and poll the filesystem for completion signals (`kumo.log`, `best_trainer.json`).

**Why not SDK blocking calls:** The Kumo SDK's `training_job.wait()` relies on Temporal workflow tracking. On the local devserver, Temporal tracking often doesn't complete properly, causing indefinite hangs. Filesystem polling is reliable and debuggable — you can `cat` the files to see progress.

## Split Convention: (start, end]

Exclusive start, inclusive end for all temporal boundaries. The `timeframe_offset` is added to timestamps before boundary comparison.

**Why this convention:** Matches Kumo's `boundaries_to_masks` semantics. Prevents double-counting at boundaries (a timestamp equal to a boundary belongs to exactly one split stage).

## Key Lessons

**Spark's implicit behaviors are DG's explicit bugs.** Every Spark join, NULL skip, deduplication, and type coercion is a behavior DG must consciously handle. The list grows with each new dataset: orphaned FKs (#173), NULL-as-zero (#139), empty accumulator rows (#52), time column NULLs (#169).

**Timestamp units cause the most integration bugs.** Three separate PRs (#172, #170, kumo #27774) fixed timestamp unit mismatches. Always carry the unit alongside the value. Always check at every boundary.

**Subprocess isolation breaks everything that relies on in-memory state.** Stats caches, Spark sessions, seeded encoders — all empty in a new subprocess. Always re-seed from disk at the start of every activity.

**The two-save problem:** Both `materialize_training_table` (TT gen workflow) and `materialize_training_task` (training workflow) write mask .pt files. The second must not overwrite the first's correct values. Any code between `load_data` and `task.save()` that modifies `table_data` (like `add_table()` in LP) can corrupt masks.

**Data beats code analysis.** Compare actual TT parquets. Check actual .pt mask tensors. Read actual edge parquet row counts. Hypotheses from code reading are wrong more often than they're right — the codebase has 40+ PRs of edge cases that look obvious in hindsight but weren't predictable from reading code alone.


# Performance Principles for DiskGraph Development

## Core Philosophy

DiskGraph processes billions of rows across 200+ threads. Every nanosecond on the hot path matters. The entity loop in TT gen and the row-by-row encode in ingest are the two critical hot paths — design decisions here compound across billions of iterations.

## Hot Path vs Slow Path

**Classify every operation as hot or slow before implementing.**

- **Hot path**: Runs per-row or per-entity-timestamp. Must be single-digit nanoseconds. No allocations, no syscalls, no thread synchronization.
- **Slow path**: Runs per-flush, per-file, or once per job. Allocations, locks, and I/O are acceptable here.

Examples:
| Operation | Path | Budget |
|-----------|------|--------|
| Check if category graduated | Hot | 3-5ns (HashSet lookup on immutable data) |
| Flush local map to global DashMap | Slow | Microseconds (every ~5K rows) |
| Resolve plan (string→index handles) | Once | Milliseconds (main thread, before parallel work) |
| PK existence check for orphaned FK filtering | Hot | ~40ns (MPHF hash + mmap read) |
| Rebuild graduated set on promotion | Rare | Microseconds (once per qualifying category) |

## Thread Synchronization: Avoid on Hot Path

**Rule: Zero synchronization on the per-row hot path.**

Even "read-only" synchronization has cost:
- `RwLock::read()`: atomic `fetch_add` + `fetch_sub` on reader count = ~10-15ns + cache line bouncing across 200 cores
- `DashSet::contains()`: atomic operations on shard counter = ~10-15ns + contention
- `Arc::clone()`: atomic `fetch_add` = ~1ns (acceptable if done per-flush, not per-row)

**Pattern: Clone-and-read.** Instead of taking a read lock per row:
```rust
// BAD: RwLock read on every row — atomic ops per access
let graduated = self.graduated.read();
if graduated.contains(&h) { return; }

// GOOD: Clone Arc once per flush, read immutable HashSet freely
let local_grad: Arc<HashSet<u64>> = self.graduated.read().clone(); // once per flush
// ... per row:
if local_grad.contains(&h) { return; } // pure memory read, 3-5ns, no sync
```

## Memory Hierarchy: Design for Cache

### Per-Thread Data Must Fit L1/L2

- L1 cache: ~32KB per core. Per-column accumulators (Welford = 64B, HLL = 16KB) fit here.
- L2 cache: ~256KB per core. Local HashMap capped at 5K entries (~120KB) fits here.
- L3 cache: Shared, ~30-50MB. Global structures (DashMap shards) live here.
- RAM: Avoid random access patterns on large structures — each cache miss = ~100ns.

**Low-cardinality columns naturally fit cache.** A column with 30 uniques uses ~700 bytes in the local map. Don't size data structures for the worst case — let them grow naturally.

### Avoid Large Per-Thread HashMaps/HashSets

Per-thread `HashMap<u64, u64>` with 1M entries = ~24MB per thread × 200 threads = 4.8GB. Merge at end = proportional to total size. Benchmarked at +384% overhead.

**Solution**: Cap local maps (5K entries), flush to shared structure periodically. Tradeoff: periodic contention at flush vs unbounded memory.

### Hot Path Memory Access Pattern

Best to worst:
1. **Register/stack** (~0.5ns): Counters, loop variables
2. **L1 cache** (~1ns): Small per-column accumulators (Welford, counters)
3. **L2 cache** (~5ns): Local HashMap (capped at ~120KB)
4. **Immutable shared read** (~3-5ns): Frozen HashSet via Arc (no cache line invalidation)
5. **L3 cache** (~15ns): DashMap shard lookup (occasional, at flush boundaries)
6. **DRAM** (~100ns): Random access into large structures — avoid on hot path

## Memory-Mapped I/O

DiskGraph index data (node data, edges, PK maps) is memory-mapped via `memmap2::Mmap`. This means:
- **Immutable after construction** — safe for concurrent reads without synchronization
- **OS-managed caching** — frequently accessed pages stay in RAM, cold pages evicted
- **Zero-copy reads** — no deserialization, just pointer arithmetic into mapped memory

The PK existence check (`contains_pk_value`) is a pure mmap read: hash → shard → MPHF index → u32 lookup → node validation. All on immutable memory-mapped data. Safe to call from 200 threads concurrently with zero synchronization.

## Algorithm Design Patterns

### Streaming Over Batching

Process data row-by-row during existing passes (encode, TT gen) rather than buffering entire tables. This:
- Avoids O(N) memory allocation
- Keeps working set in cache
- Piggybacks on existing I/O

### Early Exit / Short-Circuit

- Graduation: once a category qualifies, stop counting it. 80%+ of rows skip counting after warmup.
- ListDistinct `limit`: `values.len() < limit` exits traversal early when TOP K is reached.
- Entity filter: skip entire entity before timestamp iteration.
- Stage filter: skip target write if filter fails.

### Compute Once, Share Immutably

- Plan resolution runs once on main thread, produces `Arc<ResolvedPlan>` shared by all worker threads.
- Table readers wrapped in `Arc<TableReader>` — one instance, many readers.
- Graduated set: rebuilt rarely, shared via `Arc<HashSet>` — many readers, no per-read sync.

### Hash-Based Keys

Category counting uses `u64` gxhash of values instead of original strings. Benefits:
- Uniform 8-byte key type for all column types
- No string allocation on hot path
- Identity hashing in DashMap (key is already well-distributed)
- String recovery via separate write-once map (slow path only, at output)

## Profiling Insights

Profile host-to-device copies, embedding lookup, and data loading separately. Record the dataset generation recipe, hardware, and command with each result.

### Key Optimization Opportunities

1. **`pin_memory=True`** for DataLoader — avoids pageable→pinned copy, directly DMA to GPU
2. **Reduce RHS embedding size** — pruning, dimensionality reduction, or quantization
3. **Overlap CPU data prep with GPU compute** — prefetch next batch while current trains

## Anti-Patterns

| Anti-Pattern | Why It's Bad | Better Approach |
|-------------|-------------|-----------------|
| Per-thread HashMap with 1M entries | 24MB × 200 threads = 4.8GB, cache-hostile merge | Cap at 5K, flush to shared structure |
| DashSet for read-mostly data | Atomic reader count per access, cache line bouncing | Frozen `Arc<HashSet>`, zero-sync reads |
| String keys in hot-path maps | Allocation + hashing per row | Pre-hash to u64, recover strings at output |
| RwLock read on every row | Two atomics per access (acquire + release) | Clone Arc per-flush, read immutable snapshot |
| Global structure per-row insert | Lock contention at 200 threads | Local buffer + periodic flush |
| Large allocations in entity loop | GC pressure, fragmentation | Pre-allocate and reuse (`aliases` vec, `link_frequencies` vec) |
| Checking invariants per-row | Branching + cache pollution | Validate at resolve time (once), use `expect` for invariants |

## Measuring Performance

Always measure **wall time of the full pipeline**, not just the stage you changed. An encoding regression can be hidden when primary-key or edge-map construction dominates.

Use generated workloads that cover large row counts, wide tables, high-cardinality columns, and many edges. Keep the generation settings with the results.

Profile with `cargo flamegraph` or the built-in `pprof` support. Focus on the innermost loops — the entity processing function and the per-row encode callback.
