# Category Counts: Design & Optimization

## Purpose

Category counts map each distinct value in a categorical/ID column to its frequency. The ML pipeline uses these to decide encoder parameters (e.g., embedding table sizes, rare-category thresholds). Only **Categorical** and **Id** semantic types are eligible — edge tables, text, numerical, timestamp, and sequence columns are excluded.

## The Problem

Counting categories during encode is expensive because:
1. **High cardinality** — ID columns can have millions of unique values (e.g., `msclkid` with 650K uniques, `source_activity_id` with 62M)
2. **Concurrent access** — encode runs on 28+ threads writing to the same counters
3. **Memory pressure** — naively storing all string→count pairs can consume gigabytes
4. **String allocation** — every `observe_string` call could allocate if we store raw strings

## Strategies

Four strategies, selected via `--cat-count-strategy`:

### GlobalDashMap

The baseline concurrent approach. Two `DashMap`s per column:

```
counts:  DashMap<u64, u64>    // hash → count
strings: DashMap<u64, String> // hash → string (for JSON recovery)
```

Every row does a DashMap insert/update. Simple but high contention — every thread hits the same sharded map on every row.

**Identity hasher** (`IdentityBuildHasher`): keys are already hashed (gxhash), so the DashMap's internal hash just passes them through. Eliminates double-hashing overhead.

### Direct

`DashMap<String, u64>` — stores the actual string as the key. No hash indirection, no string recovery map needed. But every lookup does string comparison and every new entry clones the string. Mainly useful as a correctness baseline.

### Hybrid

Reduces DashMap contention by absorbing high-frequency values in a per-thread local buffer:

```
Per-thread:  HashMap<u64, u64>  (cap 5,000 — fits in L2 cache at ~80KB)
             HashMap<u64, String> (string recovery)
Per-column:  DashMap<u64, u64>  (shared, same as GlobalDashMap)
```

**Hot path**: check local HashMap first (~3ns). If hit, increment locally — zero contention. If local is full, spill directly to the global DashMap.

**Flush**: at file boundaries, merge all local counts into the global DashMap in one batch (fewer lock acquisitions than per-row).

The 5K cap is chosen so the local HashMap (keys + values + string recovery) stays under ~80KB — comfortably in L2 cache on ARM (Graviton).

### Graduation

The most sophisticated strategy. Designed for the insight that **we don't need exact counts** — we only need to know which categories occur "often enough."

A category **graduates** when its count exceeds `min_occ = sqrt(max(10000, unique_count))`. Once graduated, we stop counting it entirely. This eliminates work for high-frequency values (which dominate row counts).

**Data structures** (per column):

| Structure | Type | Purpose | Access pattern |
|-----------|------|---------|----------------|
| `graduated` | `ArcSwap<HashSet<u64>>` | Frozen set of graduated hashes | Read: lock-free `load()` from all threads |
| `pending` | `DashMap<u64, u64>` | Counts for not-yet-graduated values | Write: batch merge from local buffers |
| `strings` | `DashMap<u64, String>` | Hash → string recovery (write-once) | Write: on first occurrence only |
| `inbox` | `Mutex<Vec<u64>>` | Buffer of newly graduated hashes | Write: push on graduation, drain on rebuild |
| `cached_min_occ` | `AtomicU64` | Current threshold (adapts as unique_count grows) | Read: relaxed atomic |

Plus the same per-thread local `HashMap<u64, u64>` buffer as Hybrid (5K cap).

**Hot path for each row**:
1. Check per-thread `grad_snapshot` (`Arc<HashSet<u64>>`) — if graduated, **return immediately** (~3-5ns)
2. Check local HashMap — if hit, increment locally
3. If local full → batch-merge into `pending` DashMap, check for new graduations, refresh snapshot

**ArcSwap for the graduated set**: the graduated set is read by every thread on every row but only written when categories graduate (rare). `ArcSwap` gives lock-free reads via `load()` (returns a `Guard`, ~1ns) and atomic pointer swap on rebuild. Threads cache a local `Arc<HashSet>` snapshot and only refresh it on flush.

**Inbox batching**: newly graduated hashes go into a `Mutex<Vec<u64>>` inbox. When the inbox reaches 64 entries, the graduated `HashSet` is rebuilt (clone old + add new + swap). This amortizes the rebuild cost.

**Output**: graduated categories are emitted with count = `min_occ` (the threshold at graduation time). This is intentional — downstream only needs "frequent enough", not exact counts.

## Per-Row Subsampling

All strategies support **30% row subsampling** via `xorshift64*` PRNG:

```rust
fn should_sample_row(&mut self) -> bool {
    self.rng ^= self.rng >> 12;
    self.rng ^= self.rng << 25;
    self.rng ^= self.rng >> 27;
    (self.rng.wrapping_mul(0x2545F4914F6CDD1D) >> 11) < ROW_SUBSAMPLE_THRESHOLD
}
```

- PRNG seeded uniquely per file (`file_idx * 0x517CC1B727220A95 | 1`)
- Counts are scaled by `÷ 0.3` at output time (except Graduation which outputs `min_occ`)
- This replaced an earlier file-level skip approach which was too coarse (skipping entire files gave poor recall on tables with few files)

## Abort Mechanism

`MAX_CATEGORIES = 1,000,000` per column. If `unique_count` exceeds this, the column is flagged `aborted` and all further observation calls return immediately. The column's category counts are omitted from the output JSON.

This prevents unbounded memory growth on high-cardinality columns (e.g., free-text IDs misclassified as categorical).

## Performance (Databricks, 31GB, 41 categorical columns)

| Config | Encode time | Overhead vs baseline |
|--------|-------------|---------------------|
| No col-stats | 27.9s | — |
| Col-stats, no categories | 31.7s | +14% |
| Col-stats + Hybrid | 49.4s | +77% |
| Col-stats + Graduation | 49.0s | +76% |

Category counting dominates the col-stats overhead (+63% on top of basic stats). Hybrid ≈ Graduation on this dataset. The choice matters more on larger datasets where Graduation's early stopping saves significant work on high-frequency categories.

## Key Design Decisions

1. **Hash-based counting** (not string-based): hashing once via gxhash + identity hasher on DashMap avoids repeated string comparisons. String recovery is write-once (first occurrence only).

2. **Local buffers fit in L2**: 5K entries × 16 bytes (u64 key + u64 value) = 80KB. L2 on Graviton is 1MB/core, so the local buffer stays hot.

3. **ArcSwap over RwLock**: graduated set is read ~1B times, written ~100 times. RwLock would still require atomic increments on the reader count. ArcSwap's `load()` is a single atomic load (~1ns).

4. **30% subsample, not 10%**: 30% gives good recall while still cutting 70% of DashMap contention. At 10B rows, 30% = 3B observations which is manageable.

5. **min_occ = sqrt(max(10K, unique))**: adapts to column cardinality. For 1M uniques → threshold 1000, for 10K uniques → threshold 100. Categories below this threshold are noise for encoder selection.
