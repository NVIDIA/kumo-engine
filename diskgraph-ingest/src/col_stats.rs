//! Per-column statistics collection during the encode stage.
//!
//! `FileColStats` implements `ColumnValueObserver` and accumulates per-file
//! statistics using streaming algorithms. After a file is processed, the
//! per-file stats are merged into a shared `TableColStats` accumulator
//! via a mutex.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwap;
use dashmap::DashMap;
use parking_lot::Mutex;

use crate::graph_config::SemanticType;
use diskgraph_rowpack::{ColumnValueObserver, FloatListView};
use serde_json::{json, Value as JsonValue};

// ---------------------------------------------------------------------------
// Category counts constants
// ---------------------------------------------------------------------------

/// Maximum number of unique categories per column before aborting collection.
pub const MAX_CATEGORIES: u64 = 1_000_000;

/// Local HashMap capacity for Hybrid approach (fits in L2 cache at ~80KB).
const HYBRID_LOCAL_CAP: usize = 5_000;

/// Per-row subsampling probability (30%).
const ROW_SUBSAMPLE_PROB: f64 = 0.3;

/// Threshold for xorshift64* PRNG: sample if (rng >> 11) < this value.
const ROW_SUBSAMPLE_THRESHOLD: u64 = (0.3 * (1u64 << 53) as f64) as u64;

// ---------------------------------------------------------------------------
// GloVe bloom filter (embedded, ~5MB)
// ---------------------------------------------------------------------------

/// Pre-built bloom filter for the GloVe combined vocabulary (~4.1M words).
/// Used for the STRING_GLOVE_OVERLAP stat (approximate).
/// FPR ≈ 0.55%, fits in L3 cache. Read-only, zero contention.
static GLOVE_BLOOM_BYTES: &[u8] = include_bytes!("../data/glove_bloom.bin");

const BLOOM_NUM_BITS: u64 = 40_000_000;
const BLOOM_NUM_HASHES: u32 = 7;
const BLOOM_FNV_SEED1: u64 = 0x517cc1b727220a95;
const BLOOM_FNV_SEED2: u64 = 0x6c62272e07bb0142;

/// Bloom filter backed by a static byte slice (zero-copy from `include_bytes!`).
pub struct GloveBloom;

impl GloveBloom {
    /// Check if a token (already lowercased + punctuation-stripped) might be
    /// in the GloVe combined vocabulary.
    #[inline]
    pub fn might_contain(token: &[u8]) -> bool {
        let h1 = Self::fnv1a_64(token, BLOOM_FNV_SEED1);
        let h2 = Self::fnv1a_64(token, BLOOM_FNV_SEED2);
        for i in 0..BLOOM_NUM_HASHES {
            let bit_idx = h1.wrapping_add((i as u64).wrapping_mul(h2)) % BLOOM_NUM_BITS;
            let byte_idx = (bit_idx / 8) as usize;
            let bit_offset = (bit_idx % 8) as u8;
            if GLOVE_BLOOM_BYTES[byte_idx] & (1 << bit_offset) == 0 {
                return false;
            }
        }
        true
    }

    #[inline(always)]
    fn fnv1a_64(bytes: &[u8], seed: u64) -> u64 {
        let mut h = seed;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }
}

// ---------------------------------------------------------------------------
// Text sample stats (computed during context phase on ~10K rows)
// ---------------------------------------------------------------------------

/// Text column statistics computed on a small sample during the context phase.
/// Matches Python's approx-only text stats (1K-10K row sample).
#[derive(Clone, Debug)]
pub struct TextSampleStats {
    pub avg_tokens: f64,
    pub max_tokens: u64,
    pub non_char_fraction: f64,
    pub string_truncation_scale: f64,
    pub glove_overlap: f64,
}

/// Compute text sample stats from a slice of string values using
/// `basic_english_normalize` tokenization (lowercase + punctuation → space + split).
/// Operates on full strings (no 512-byte truncation for tokenization).
pub fn compute_text_sample_stats(values: &[&str]) -> TextSampleStats {
    if values.is_empty() {
        return TextSampleStats {
            avg_tokens: 0.0,
            max_tokens: 0,
            non_char_fraction: 0.0,
            string_truncation_scale: 1.0,
            glove_overlap: 0.0,
        };
    }

    let mut token_sum: u64 = 0;
    let mut max_tokens: u64 = 0;
    let mut total_tokens_for_non_char: u64 = 0;
    let mut non_char_tokens: u64 = 0;
    let mut glove_hits: u64 = 0;
    let mut glove_total: u64 = 0;
    let mut truncation_scale_sum: f64 = 0.0;

    for &value in values {
        let byte_len = value.len();
        // Truncation scale: byte_len / min(byte_len, 512).
        truncation_scale_sum += if byte_len <= 512 {
            1.0
        } else {
            byte_len as f64 / 512.0
        };

        // basic_english_normalize: lowercase + replace punctuation with space + split
        // We process the full string (not truncated to 512).
        let bytes = value.as_bytes();
        let mut buf = vec![0u8; bytes.len()];
        buf.copy_from_slice(bytes);
        for b in &mut buf {
            b.make_ascii_lowercase();
            if matches!(
                *b,
                b'\'' | b'"' | b'.' | b',' | b'(' | b')' | b'!' | b'?' | b';' | b':'
            ) {
                *b = b' ';
            }
        }

        // Split on whitespace, count tokens.
        let mut i = 0;
        let mut row_tokens: u64 = 0;
        while i < buf.len() {
            while i < buf.len() && buf[i].is_ascii_whitespace() {
                i += 1;
            }
            let start = i;
            while i < buf.len() && !buf[i].is_ascii_whitespace() {
                i += 1;
            }
            if start < i {
                row_tokens += 1;
                glove_total += 1;
                if GloveBloom::might_contain(&buf[start..i]) {
                    glove_hits += 1;
                }
                // non_char: token doesn't match [a-zA-Z]+ (already lowercased, so check a-z)
                let is_alpha_only = buf[start..i].iter().all(|b| b.is_ascii_lowercase());
                total_tokens_for_non_char += 1;
                if !is_alpha_only {
                    non_char_tokens += 1;
                }
            }
        }
        token_sum += row_tokens;
        if row_tokens > max_tokens {
            max_tokens = row_tokens;
        }
    }

    let count = values.len() as f64;
    TextSampleStats {
        avg_tokens: token_sum as f64 / count,
        max_tokens,
        non_char_fraction: if total_tokens_for_non_char > 0 {
            non_char_tokens as f64 / total_tokens_for_non_char as f64
        } else {
            0.0
        },
        string_truncation_scale: truncation_scale_sum / count,
        glove_overlap: if glove_total > 0 {
            glove_hits as f64 / glove_total as f64
        } else {
            0.0
        },
    }
}

// ---------------------------------------------------------------------------
// Identity hasher for DashMap<u64, _> (keys are already hashed)
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _bytes: &[u8]) {
        unreachable!("IdentityHasher only supports u64 keys");
    }
    #[inline(always)]
    fn write_u64(&mut self, i: u64) {
        self.0 = i;
    }
}

#[derive(Clone, Default)]
struct IdentityBuildHasher;

impl BuildHasher for IdentityBuildHasher {
    type Hasher = IdentityHasher;
    fn build_hasher(&self) -> Self::Hasher {
        IdentityHasher(0)
    }
}

// ---------------------------------------------------------------------------
// Category count strategy
// ---------------------------------------------------------------------------

/// Strategy for collecting per-column category frequency counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CategoryCountStrategy {
    /// No category counting (default).
    None,
    /// Per-thread small HashMap (5K cap) + spill to global DashMap.
    Hybrid,
    /// Categories "graduate" once count exceeds min_occ; graduated set is read-only.
    Graduation,
}

// ---------------------------------------------------------------------------
// Shared per-column category count state (used by all approaches)
// ---------------------------------------------------------------------------

/// Internal storage for category counts.
enum CountStorage {
    /// Hash-based: hash→count + hash→string recovery (Hybrid).
    Hashed {
        counts: DashMap<u64, u64, IdentityBuildHasher>,
        strings: DashMap<u64, String, IdentityBuildHasher>,
    },
    /// Graduation v2: ArcSwap frozen graduated set + DashMap pending + inbox coordinator.
    Graduation {
        /// Frozen graduated set — swapped atomically via ArcSwap.
        graduated: ArcSwap<HashSet<u64>>,
        /// Concurrent accumulation for not-yet-graduated categories.
        pending: DashMap<u64, u64, IdentityBuildHasher>,
        /// Write-once string recovery (hash → string) for all observed categories.
        strings: DashMap<u64, String, IdentityBuildHasher>,
        /// Graduation candidates waiting to be merged into the graduated set.
        inbox: Mutex<Vec<u64>>,
        /// Cached min_occ threshold: sqrt(max(10_000, unique_count)).
        cached_min_occ: AtomicU64,
    },
}

// ---------------------------------------------------------------------------
// Graduation helpers
// ---------------------------------------------------------------------------

/// Integer square root via Newton's method with f64 seed + one refinement.
pub fn isqrt(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let mut x = (n as f64).sqrt() as u64;
    // One Newton refinement to handle f64 rounding
    x = (x + n / x.max(1)) / 2;
    // Correct off-by-one
    if x > 0 && x * x > n {
        x -= 1;
    }
    // Ensure we didn't undershoot
    while (x + 1) * (x + 1) <= n {
        x += 1;
    }
    x
}

/// Compute min_occ threshold: sqrt(max(10_000, unique_count)), minimum 2.
pub fn compute_min_occ(unique_count: u64) -> u64 {
    isqrt(unique_count.max(10_000)).max(2)
}

/// Shared per-column category count state. One instance per eligible column,
/// shared across all encode threads via `Arc`.
pub struct SharedColumnCounts {
    storage: CountStorage,
    unique_count: AtomicU64,
    aborted: AtomicBool,
}

impl SharedColumnCounts {
    /// Create hash-based storage (for GlobalDashMap & Hybrid).
    pub fn new_hashed() -> Self {
        Self {
            storage: CountStorage::Hashed {
                counts: DashMap::with_capacity_and_hasher_and_shard_amount(
                    1024,
                    IdentityBuildHasher,
                    256,
                ),
                strings: DashMap::with_capacity_and_hasher_and_shard_amount(
                    1024,
                    IdentityBuildHasher,
                    256,
                ),
            },
            unique_count: AtomicU64::new(0),
            aborted: AtomicBool::new(false),
        }
    }

    /// Create graduation v2 storage (ArcSwap frozen set + DashMap pending + inbox).
    pub fn new_graduation() -> Self {
        Self {
            storage: CountStorage::Graduation {
                graduated: ArcSwap::from_pointee(HashSet::new()),
                pending: DashMap::with_capacity_and_hasher_and_shard_amount(
                    1024,
                    IdentityBuildHasher,
                    256,
                ),
                strings: DashMap::with_capacity_and_hasher_and_shard_amount(
                    1024,
                    IdentityBuildHasher,
                    256,
                ),
                inbox: Mutex::new(Vec::new()),
                cached_min_occ: AtomicU64::new(100), // sqrt(10_000)
            },
            unique_count: AtomicU64::new(0),
            aborted: AtomicBool::new(false),
        }
    }

    /// Observe a hashed value into shared DashMap. Called from Hybrid spill path.
    #[inline]
    fn observe_hashed(&self, hash: u64, value_str: &str) {
        if self.aborted.load(Ordering::Relaxed) {
            return;
        }
        if let CountStorage::Hashed { counts, strings } = &self.storage {
            let mut is_new = false;
            counts
                .entry(hash)
                .and_modify(|c| *c += 1)
                .or_insert_with(|| {
                    is_new = true;
                    1
                });
            if is_new {
                strings.entry(hash).or_insert_with(|| value_str.to_owned());
                if self.unique_count.fetch_add(1, Ordering::Relaxed) + 1 > MAX_CATEGORIES {
                    self.aborted.store(true, Ordering::Relaxed);
                }
            }
        }
    }

    /// Merge a batch of local counts into the shared DashMap (Hybrid flush).
    fn merge_local_batch(&self, local: &mut LocalColumnCounts) {
        if self.aborted.load(Ordering::Relaxed) {
            return;
        }
        if let CountStorage::Hashed { counts, strings } = &self.storage {
            for (&hash, &count) in &local.counts {
                let mut is_new = false;
                counts
                    .entry(hash)
                    .and_modify(|c| *c += count)
                    .or_insert_with(|| {
                        is_new = true;
                        count
                    });
                if is_new {
                    if let Some(s) = local.strings.remove(&hash) {
                        strings.entry(hash).or_insert(s);
                    }
                    if self.unique_count.fetch_add(1, Ordering::Relaxed) + 1 > MAX_CATEGORIES {
                        self.aborted.store(true, Ordering::Relaxed);
                        return;
                    }
                }
            }
        }
    }

    /// Merge a batch of local counts into graduation storage.
    /// Returns the current graduated snapshot for caching by the caller.
    fn merge_local_batch_graduation(&self, local: &mut LocalColumnCounts) -> Arc<HashSet<u64>> {
        if let CountStorage::Graduation {
            graduated,
            pending,
            strings,
            inbox,
            cached_min_occ,
        } = &self.storage
        {
            let grad_snapshot = graduated.load_full();
            if self.aborted.load(Ordering::Relaxed) {
                return grad_snapshot;
            }
            let min_occ = cached_min_occ.load(Ordering::Relaxed);
            let mut newly_graduated = Vec::new();
            for (&hash, &count) in &local.counts {
                // Skip already graduated
                if grad_snapshot.contains(&hash) {
                    continue;
                }
                // Check if this hash was ever seen before (strings is write-once).
                // Must check before inserting the string below.
                let is_truly_new = !strings.contains_key(&hash);
                // Write-once string recovery
                if let Some(s) = local.strings.remove(&hash) {
                    strings.entry(hash).or_insert(s);
                }
                // Merge count into pending
                let total = {
                    let mut entry = pending.entry(hash).or_insert(0);
                    *entry += count;
                    *entry
                };
                if total >= min_occ {
                    newly_graduated.push(hash);
                    pending.remove(&hash);
                } else if total == count && is_truly_new {
                    // Truly new unique — hash never seen before in any thread.
                    // Avoids double-counting when a graduated category is
                    // re-inserted into pending from a stale snapshot.
                    let uc = self.unique_count.fetch_add(1, Ordering::Relaxed) + 1;
                    if uc > MAX_CATEGORIES {
                        self.aborted.store(true, Ordering::Relaxed);
                        return grad_snapshot;
                    }
                }
            }
            // Update cached_min_occ
            let uc = self.unique_count.load(Ordering::Relaxed);
            let new_min_occ = compute_min_occ(uc);
            cached_min_occ.store(new_min_occ, Ordering::Relaxed);
            // Push newly graduated to inbox; rebuild if inbox >= 64
            if !newly_graduated.is_empty() {
                let mut inbox_guard = inbox.lock();
                inbox_guard.extend(newly_graduated);
                if inbox_guard.len() >= 64 {
                    let to_add: Vec<u64> = inbox_guard.drain(..).collect();
                    self.rebuild_graduated(&to_add);
                    drop(inbox_guard);
                    return graduated.load_full();
                }
            }
            grad_snapshot
        } else {
            Arc::new(HashSet::new())
        }
    }

    /// Rebuild the graduated ArcSwap set by adding new hashes.
    fn rebuild_graduated(&self, new_hashes: &[u64]) {
        if let CountStorage::Graduation { graduated, .. } = &self.storage {
            let old = graduated.load();
            let mut new_set = HashSet::with_capacity(old.len() + new_hashes.len());
            new_set.extend(old.iter());
            new_set.extend(new_hashes.iter());
            graduated.store(Arc::new(new_set));
        }
    }

    /// Flush remaining inbox entries into the graduated set (call before drain_to_map).
    fn finish_graduation(&self) {
        if let CountStorage::Graduation { inbox, .. } = &self.storage {
            let mut inbox_guard = inbox.lock();
            if !inbox_guard.is_empty() {
                let to_add: Vec<u64> = inbox_guard.drain(..).collect();
                drop(inbox_guard);
                self.rebuild_graduated(&to_add);
            }
        }
    }

    /// Drain into a HashMap<String, u64> for JSON output. Returns None if aborted.
    pub fn drain_to_map(&self) -> Option<HashMap<String, u64>> {
        if self.aborted.load(Ordering::Relaxed) {
            return None;
        }
        match &self.storage {
            CountStorage::Hashed { counts, strings } => {
                let mut result = HashMap::with_capacity(counts.len());
                for entry in counts.iter() {
                    let hash = *entry.key();
                    let count = *entry.value();
                    if let Some(string_entry) = strings.get(&hash) {
                        result.insert(string_entry.value().clone(), count);
                    }
                }
                Some(result)
            }
            CountStorage::Graduation {
                graduated,
                pending,
                strings,
                cached_min_occ,
                ..
            } => {
                // Flush remaining inbox into graduated set before draining.
                self.finish_graduation();
                let min_occ = cached_min_occ.load(Ordering::Relaxed);
                let grad_set = graduated.load();
                let mut result = HashMap::with_capacity(grad_set.len() + pending.len() / 2);
                // 1. Graduated categories — known to exceed min_occ during encode.
                for &hash in grad_set.iter() {
                    if let Some(string_entry) = strings.get(&hash) {
                        result.insert(string_entry.value().clone(), min_occ);
                    }
                }
                // 2. Final sweep of pending — include categories whose accumulated
                //    count meets min_occ. This catches categories that were never
                //    checked at the right moment during encode.
                for entry in pending.iter() {
                    let count = *entry.value();
                    if count >= min_occ {
                        let hash = *entry.key();
                        if let Some(string_entry) = strings.get(&hash) {
                            result.insert(string_entry.value().clone(), count);
                        }
                    }
                }
                Some(result)
            }
        }
    }

    pub fn is_aborted(&self) -> bool {
        self.aborted.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Per-file local category count state (Approach 1 & 3)
// ---------------------------------------------------------------------------

/// Per-file local category counts for a single column.
pub struct LocalColumnCounts {
    counts: HashMap<u64, u64>,
    strings: HashMap<u64, String>,
}

impl LocalColumnCounts {
    fn with_capacity(cap: usize) -> Self {
        Self {
            counts: HashMap::with_capacity(cap),
            strings: HashMap::with_capacity(cap),
        }
    }

    /// Observe a value locally. Returns true if the key is new.
    #[inline]
    fn observe(&mut self, hash: u64, value_str: &str) -> bool {
        use std::collections::hash_map::Entry;
        match self.counts.entry(hash) {
            Entry::Occupied(mut e) => {
                *e.get_mut() += 1;
                false
            }
            Entry::Vacant(e) => {
                e.insert(1);
                self.strings.insert(hash, value_str.to_owned());
                true
            }
        }
    }

    fn len(&self) -> usize {
        self.counts.len()
    }

    fn clear(&mut self) {
        self.counts.clear();
        self.strings.clear();
    }
}

// ---------------------------------------------------------------------------
// Table-level shared category count state
// ---------------------------------------------------------------------------

/// Per-table shared category count state. Created before encode, shared across
/// all file tasks for a table via `Arc`.
pub struct TableCategoryCounts {
    pub strategy: CategoryCountStrategy,
    /// Per column index → shared state. None for non-eligible columns.
    pub columns: Vec<Option<Arc<SharedColumnCounts>>>,
    /// Whether per-row subsampling is enabled (30% sampling, scale ÷0.3).
    pub row_subsample: bool,
    /// Monotonic file counter for PRNG seeding (unique seed per file).
    file_counter: AtomicU64,
}

impl TableCategoryCounts {
    /// Create shared state for a table given its column semantic types.
    /// Only Categorical and Id columns get category count collection.
    pub fn new(
        strategy: CategoryCountStrategy,
        semantic_types: &[Option<SemanticType>],
        row_subsample: bool,
    ) -> Self {
        let columns = semantic_types
            .iter()
            .map(|stype| match stype {
                Some(SemanticType::Categorical) | Some(SemanticType::Id) => {
                    let shared = match strategy {
                        CategoryCountStrategy::Graduation => SharedColumnCounts::new_graduation(),
                        _ => SharedColumnCounts::new_hashed(),
                    };
                    Some(Arc::new(shared))
                }
                _ => None,
            })
            .collect();
        Self {
            strategy,
            columns,
            row_subsample,
            file_counter: AtomicU64::new(0),
        }
    }

    /// Get the next file index (for PRNG seeding).
    pub fn next_file_idx(&self) -> u64 {
        self.file_counter.fetch_add(1, Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// HyperLogLog — re-exported from diskgraph-core
// ---------------------------------------------------------------------------

pub use diskgraph_core::accumulators::HyperLogLog;

// ---------------------------------------------------------------------------
// Reservoir sampler (Efraimidis-Spirakis with priority keys)
// ---------------------------------------------------------------------------

/// Weighted reservoir sampler using Efraimidis-Spirakis priority keys.
///
/// Each item gets a random priority key u ~ Uniform(0,1). The reservoir holds
/// the top-k items by priority. Merge: combine two reservoirs, sort by priority
/// descending, take top-k. This is provably correct and order-independent.
///
/// The internal array is **unsorted** during observation for performance.
/// Only sorted on demand (quantile computation, merge output).
pub struct ReservoirSampler {
    /// (value, priority_key) pairs — unsorted during observation.
    samples: Vec<(f64, f64)>,
    k: usize,
    /// Minimum priority in the reservoir (for fast rejection).
    min_priority: f64,
    /// Index of the element with min priority.
    min_idx: usize,
    /// Thread-local RNG for generating priority keys.
    rng: u64,
}

impl ReservoirSampler {
    pub const DEFAULT_K: usize = 1_000;

    pub fn new(k: usize) -> Self {
        Self {
            samples: Vec::with_capacity(k),
            k,
            min_priority: f64::INFINITY,
            min_idx: 0,
            rng: 0x517cc1b727220a95, // fixed seed per file — non-deterministic across files
        }
    }

    /// Observe a value. Generates a random priority key and inserts if it beats
    /// the current minimum.
    #[inline]
    pub fn observe(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        let priority = self.next_uniform();
        if self.samples.len() < self.k {
            if priority < self.min_priority {
                self.min_priority = priority;
                self.min_idx = self.samples.len();
            }
            self.samples.push((value, priority));
        } else if priority > self.min_priority {
            // Replace the min element
            self.samples[self.min_idx] = (value, priority);
            // Rescan for new min (cache-friendly sequential read)
            self.min_priority = f64::INFINITY;
            for (i, &(_, p)) in self.samples.iter().enumerate() {
                if p < self.min_priority {
                    self.min_priority = p;
                    self.min_idx = i;
                }
            }
        }
    }

    /// Estimate quantiles from the reservoir samples.
    /// Returns (q25, q50, q75). Returns None if no samples.
    pub fn quantiles(&self) -> Option<(f64, f64, f64)> {
        if self.samples.is_empty() {
            return None;
        }
        let mut values: Vec<f64> = self.samples.iter().map(|(v, _)| *v).collect();
        values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let n = values.len();
        let q25 = values[(n as f64 * 0.25) as usize];
        let q50 = values[(n as f64 * 0.50) as usize];
        let q75 = values[(n as f64 * 0.75) as usize];
        Some((q25, q50, q75))
    }

    /// Merge another reservoir into this one.
    /// Combine all samples, sort by priority descending, take top-k.
    pub fn merge(&mut self, other: &ReservoirSampler) {
        self.samples.extend_from_slice(&other.samples);
        self.samples
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        self.samples.truncate(self.k);
        // Recompute min
        self.min_priority = f64::INFINITY;
        self.min_idx = 0;
        for (i, &(_, p)) in self.samples.iter().enumerate() {
            if p < self.min_priority {
                self.min_priority = p;
                self.min_idx = i;
            }
        }
    }

    /// xorshift64* PRNG → uniform f64 in (0, 1).
    #[inline(always)]
    fn next_uniform(&mut self) -> f64 {
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let val = self.rng.wrapping_mul(0x2545F4914F6CDD1D);
        // Convert to (0, 1) range
        (val >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

// ---------------------------------------------------------------------------
// Reservoir sampler — Algorithm L variant (skip-ahead, fewer RNG calls)
// ---------------------------------------------------------------------------

/// Reservoir sampler using E-S priority keys with Algorithm L skip-ahead.
///
/// After the reservoir fills, instead of generating a priority for every item,
/// computes the geometric skip to the next item that would enter. This reduces
/// RNG calls from O(n) to O(k·log(n/k)) but keeps the same O(k) rescan per
/// replacement.
pub struct ReservoirSamplerL {
    samples: Vec<(f64, f64)>,
    k: usize,
    min_priority: f64,
    min_idx: usize,
    rng: u64,
    items_seen: u64,
    next_accept: u64,
    filled: bool,
}

impl ReservoirSamplerL {
    pub const DEFAULT_K: usize = 10_000;

    pub fn new(k: usize) -> Self {
        Self {
            samples: Vec::with_capacity(k),
            k,
            min_priority: f64::INFINITY,
            min_idx: 0,
            rng: 0x517cc1b727220a95,
            items_seen: 0,
            next_accept: 0,
            filled: false,
        }
    }

    #[inline]
    pub fn observe(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.items_seen += 1;

        if !self.filled {
            let priority = self.next_uniform();
            if priority < self.min_priority {
                self.min_priority = priority;
                self.min_idx = self.samples.len();
            }
            self.samples.push((value, priority));
            if self.samples.len() == self.k {
                self.filled = true;
                self.compute_next_accept();
            }
            return;
        }

        if self.items_seen != self.next_accept {
            return;
        }

        // Accept: generate priority conditioned on > min_priority → U(min_priority, 1)
        let u = self.next_uniform();
        let priority = self.min_priority + u * (1.0 - self.min_priority);
        self.samples[self.min_idx] = (value, priority);
        self.rescan_min();
        self.compute_next_accept();
    }

    fn compute_next_accept(&mut self) {
        let u = self.next_uniform();
        let skip = if self.min_priority > 1e-15 && self.min_priority < 1.0 - 1e-15 {
            let s = (u.ln() / self.min_priority.ln()).floor();
            if s.is_finite() && s >= 0.0 {
                s as u64
            } else {
                0
            }
        } else {
            0
        };
        self.next_accept = self.items_seen + skip + 1;
    }

    fn rescan_min(&mut self) {
        self.min_priority = f64::INFINITY;
        for (i, &(_, p)) in self.samples.iter().enumerate() {
            if p < self.min_priority {
                self.min_priority = p;
                self.min_idx = i;
            }
        }
    }

    pub fn quantiles(&self) -> Option<(f64, f64, f64)> {
        if self.samples.is_empty() {
            return None;
        }
        let mut values: Vec<f64> = self.samples.iter().map(|(v, _)| *v).collect();
        values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let n = values.len();
        Some((
            values[(n as f64 * 0.25) as usize],
            values[(n as f64 * 0.50) as usize],
            values[(n as f64 * 0.75) as usize],
        ))
    }

    pub fn merge(&mut self, other: &ReservoirSamplerL) {
        self.samples.extend_from_slice(&other.samples);
        self.samples
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        self.samples.truncate(self.k);
        self.rescan_min();
    }

    #[inline(always)]
    fn next_uniform(&mut self) -> f64 {
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let val = self.rng.wrapping_mul(0x2545F4914F6CDD1D);
        (val >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

// ---------------------------------------------------------------------------
// T-Digest (buffer-merge, δ=200)
// ---------------------------------------------------------------------------

/// Streaming quantile estimator using the t-digest algorithm (Dunning 2019).
///
/// Maintains a sorted set of centroids (mean, weight). Incoming values are
/// buffered; when the buffer fills, it is sorted and merged into the centroid
/// list with compression controlled by the k1 scale function. Gives excellent
/// accuracy at both tails and median, with O(δ) memory.
pub struct TDigestEstimator {
    centroids: Vec<(f64, f64)>, // (mean, weight), sorted by mean
    max_size: usize,            // δ — compression parameter
    count: f64,
    buffer: Vec<f64>,
    buf_cap: usize,
}

impl TDigestEstimator {
    pub const DEFAULT_DELTA: usize = 200;

    pub fn new(delta: usize) -> Self {
        Self {
            centroids: Vec::new(),
            max_size: delta,
            count: 0.0,
            buffer: Vec::with_capacity(delta),
            buf_cap: delta,
        }
    }

    #[inline]
    pub fn observe(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.buffer.push(value);
        if self.buffer.len() >= self.buf_cap {
            self.flush();
        }
    }

    pub fn flush(&mut self) {
        if self.buffer.is_empty() {
            return;
        }
        self.buffer
            .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let buf_centroids: Vec<(f64, f64)> = self.buffer.drain(..).map(|v| (v, 1.0)).collect();
        self.merge_centroids(buf_centroids);
    }

    fn merge_centroids(&mut self, other: Vec<(f64, f64)>) {
        // Merge two sorted centroid lists, then compress
        let mut combined = Vec::with_capacity(self.centroids.len() + other.len());
        let (mut i, mut j) = (0, 0);
        while i < self.centroids.len() && j < other.len() {
            if self.centroids[i].0 <= other[j].0 {
                combined.push(self.centroids[i]);
                i += 1;
            } else {
                combined.push(other[j]);
                j += 1;
            }
        }
        combined.extend_from_slice(&self.centroids[i..]);
        combined.extend_from_slice(&other[j..]);

        self.count = combined.iter().map(|(_, w)| w).sum();
        self.centroids = self.compress(combined);
    }

    fn compress(&self, sorted: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
        if sorted.is_empty() || self.count == 0.0 {
            return sorted;
        }
        let total = self.count;
        let mut result: Vec<(f64, f64)> = Vec::with_capacity(self.max_size);
        let mut cur_mean = sorted[0].0;
        let mut cur_weight = sorted[0].1;
        let mut weight_so_far = 0.0;

        for &(mean, weight) in &sorted[1..] {
            let q = (weight_so_far + (cur_weight + weight) / 2.0) / total;
            let limit = 4.0 * total / self.max_size as f64 * q * (1.0 - q);
            if cur_weight + weight <= limit {
                let new_weight = cur_weight + weight;
                cur_mean = (cur_mean * cur_weight + mean * weight) / new_weight;
                cur_weight = new_weight;
            } else {
                weight_so_far += cur_weight;
                result.push((cur_mean, cur_weight));
                cur_mean = mean;
                cur_weight = weight;
            }
        }
        result.push((cur_mean, cur_weight));
        result
    }

    fn quantile_inner(&self, q: f64) -> f64 {
        if self.centroids.is_empty() {
            return 0.0;
        }
        if self.centroids.len() == 1 {
            return self.centroids[0].0;
        }
        let target = q * self.count;
        let mut cum = 0.0;
        for i in 0..self.centroids.len() {
            let (mean, weight) = self.centroids[i];
            if cum + weight > target {
                if i == 0 {
                    return mean;
                }
                let prev_mean = self.centroids[i - 1].0;
                let prev_weight = self.centroids[i - 1].1;
                let prev_mid = cum - prev_weight / 2.0;
                let cur_mid = cum + weight / 2.0;
                let frac = (target - prev_mid) / (cur_mid - prev_mid);
                return prev_mean + (mean - prev_mean) * frac.clamp(0.0, 1.0);
            }
            cum += weight;
        }
        self.centroids.last().unwrap().0
    }

    pub fn quantiles(&mut self) -> Option<(f64, f64, f64)> {
        self.flush();
        if self.centroids.is_empty() {
            return None;
        }
        Some((
            self.quantile_inner(0.25),
            self.quantile_inner(0.50),
            self.quantile_inner(0.75),
        ))
    }

    pub fn merge(&mut self, other: &mut TDigestEstimator) {
        other.flush();
        self.flush();
        let other_centroids = other.centroids.clone();
        self.merge_centroids(other_centroids);
    }
}

// ---------------------------------------------------------------------------
// WelfordAccumulator — re-exported from diskgraph-core
// ---------------------------------------------------------------------------

pub use diskgraph_core::accumulators::WelfordAccumulator;

// ---------------------------------------------------------------------------
// FileColStats — per-file observer
// ---------------------------------------------------------------------------

/// Per-file column statistics observer.
///
/// Created once per file, passed as the observer to `encode_row_observed`.
/// After the file is processed, merged into the shared accumulator.
pub struct FileColStats {
    cols: Vec<ColumnStatsCollector>,
    /// Category count strategy for this file.
    cat_strategy: CategoryCountStrategy,
    /// Shared per-column category count state (for abort checks + DashMaps).
    cat_shared: Option<Arc<TableCategoryCounts>>,
    /// Per-file local category count state per column (Hybrid & Graduation).
    /// Indexed by col_idx. None for non-eligible columns.
    cat_local: Vec<Option<LocalColumnCounts>>,
    /// Whether per-row subsampling is enabled.
    row_subsample: bool,
    /// xorshift64* PRNG state for per-row subsampling.
    rng: u64,
    /// Per-column cached graduated set snapshots (Graduation strategy only).
    grad_snapshots: Vec<Option<Arc<HashSet<u64>>>>,
}

/// Per-column stats collector, chosen by SemanticType.
pub enum ColumnStatsCollector {
    Numerical {
        null_count: u64,
        welford: WelfordAccumulator,
        hll: HyperLogLog,
        tdigest: TDigestEstimator,
    },
    Categorical {
        count: u64,
        null_count: u64,
        hll: HyperLogLog,
    },
    Id {
        count: u64,
        null_count: u64,
        hll: HyperLogLog,
    },
    Text {
        count: u64,
        null_count: u64,
        hll: HyperLogLog,
    },
    Timestamp {
        count: u64,
        null_count: u64,
        parse_fail_count: u64,
        min_secs: i64,
        max_secs: i64,
    },
    Sequence {
        count: u64,
        null_count: u64,
        min_len: usize,
        max_len: usize,
        welford: WelfordAccumulator,
    },
    /// Skip this column (PK, Unsupported, Multicategorical).
    Skip,
}

impl FileColStats {
    /// Create a new observer for a file, given the semantic types of each column.
    /// No category count collection.
    pub fn new(semantic_types: &[Option<SemanticType>]) -> Self {
        let cols = semantic_types
            .iter()
            .map(ColumnStatsCollector::from_semantic_type)
            .collect();
        let n = semantic_types.len();
        Self {
            cols,
            cat_strategy: CategoryCountStrategy::None,
            cat_shared: None,
            cat_local: (0..n).map(|_| None).collect(),
            row_subsample: false,
            rng: 1,
            grad_snapshots: (0..n).map(|_| None).collect(),
        }
    }

    /// Create a new observer for a file with category count collection.
    pub fn new_with_categories(
        semantic_types: &[Option<SemanticType>],
        table_cat: Arc<TableCategoryCounts>,
    ) -> Self {
        let cols = semantic_types
            .iter()
            .map(ColumnStatsCollector::from_semantic_type)
            .collect();
        let strategy = table_cat.strategy;
        let row_subsample = table_cat.row_subsample;
        let file_idx = table_cat.next_file_idx();
        // Seed PRNG: unique per file, must be odd (non-zero)
        let rng = file_idx.wrapping_mul(0x517CC1B727220A95) | 1;
        // Create per-file local state (Hybrid and Graduation need local buffers)
        let cat_local = match strategy {
            CategoryCountStrategy::Hybrid | CategoryCountStrategy::Graduation => table_cat
                .columns
                .iter()
                .map(|shared| {
                    shared
                        .as_ref()
                        .map(|_| LocalColumnCounts::with_capacity(HYBRID_LOCAL_CAP))
                })
                .collect(),
            _ => (0..semantic_types.len()).map(|_| None).collect(),
        };
        // Graduation: init grad_snapshots as empty sets for eligible columns
        let grad_snapshots = match strategy {
            CategoryCountStrategy::Graduation => table_cat
                .columns
                .iter()
                .map(|shared| shared.as_ref().map(|_| Arc::new(HashSet::new())))
                .collect(),
            _ => (0..semantic_types.len()).map(|_| None).collect(),
        };
        Self {
            cols,
            cat_strategy: strategy,
            cat_shared: Some(table_cat),
            cat_local,
            row_subsample,
            rng,
            grad_snapshots,
        }
    }

    /// Merge another `FileColStats` into this one (per-column dispatch).
    /// Used on the channel receive side to accumulate per-table stats.
    pub fn merge(&mut self, other: &mut FileColStats) {
        assert_eq!(self.cols.len(), other.cols.len());
        for (dst, src) in self.cols.iter_mut().zip(other.cols.iter_mut()) {
            dst.merge(src);
        }
    }

    /// Flush local category counts to shared state (called at file boundary).
    /// Handles Hybrid, Graduation, and early abort checks.
    pub fn flush_categories(&mut self) {
        match self.cat_strategy {
            CategoryCountStrategy::Hybrid => {
                if let Some(ref shared) = self.cat_shared {
                    for (col_idx, local) in self.cat_local.iter_mut().enumerate() {
                        if let Some(local) = local.as_mut() {
                            if let Some(ref col_shared) = shared.columns[col_idx] {
                                col_shared.merge_local_batch(local);
                            }
                            local.clear();
                        }
                    }
                }
            }
            CategoryCountStrategy::Graduation => {
                if let Some(ref shared) = self.cat_shared {
                    for (col_idx, local) in self.cat_local.iter_mut().enumerate() {
                        if let Some(local) = local.as_mut() {
                            if let Some(ref col_shared) = shared.columns[col_idx] {
                                let snap = col_shared.merge_local_batch_graduation(local);
                                self.grad_snapshots[col_idx] = Some(snap);
                            }
                            local.clear();
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Check for columns where parse failures or nulls exceed 80% of total rows.
    /// Returns an error if any timestamp column has a high parse failure rate,
    /// indicating a likely format mismatch. High null rates are allowed — a
    /// column that is mostly or entirely null is valid (e.g. DELETED_DATE).
    pub fn check_parse_failures(&self, column_names: &[String], table_name: &str) -> anyhow::Result<()> {
        for (col, name) in self.cols.iter().zip(column_names.iter()) {
            if let ColumnStatsCollector::Timestamp {
                count,
                null_count,
                parse_fail_count,
                ..
            } = col
            {
                let total = count + null_count + parse_fail_count;
                if total == 0 {
                    continue;
                }
                if *parse_fail_count > 0 {
                    let fail_frac = *parse_fail_count as f64 / total as f64;
                    eprintln!(
                        "WARNING: table '{}' column '{}': {} parse failures out of {} rows ({:.1}%)",
                        table_name,
                        name,
                        parse_fail_count,
                        total,
                        fail_frac * 100.0,
                    );
                    // Only error on high parse failure rate — nulls are valid data.
                    let non_null = total - *null_count;
                    if non_null > 0 && *parse_fail_count as f64 / non_null as f64 > 0.5 {
                        return Err(anyhow::anyhow!(
                            "table '{}' column '{}': {:.1}% of non-null rows failed to parse \
                             ({} parse failures out of {} non-null, {} null out of {} total). \
                             This likely indicates a timestamp format mismatch.",
                            table_name,
                            name,
                            *parse_fail_count as f64 / non_null as f64 * 100.0,
                            parse_fail_count,
                            non_null,
                            null_count,
                            total,
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Serialize accumulated stats to a JSON object keyed by column name.
    /// Category counts are included from the shared state (Approach 2&3) or
    /// local accumulator (Approach 1).
    pub fn to_json(
        &mut self,
        column_names: &[String],
        table_cat: Option<&TableCategoryCounts>,
        text_sample_stats: Option<&HashMap<String, TextSampleStats>>,
    ) -> JsonValue {
        // Collect category counts first (borrows self.cat_local, self.cat_shared)
        // to avoid overlapping borrows with self.cols.iter_mut().
        let cat_counts: Vec<Option<HashMap<String, u64>>> = (0..self.cols.len())
            .map(|col_idx| {
                get_category_counts_for_col(self.cat_strategy, &self.cat_local, table_cat, col_idx)
            })
            .collect();

        let mut map = serde_json::Map::new();
        for (col_idx, (col, name)) in self.cols.iter_mut().zip(column_names.iter()).enumerate() {
            if let Some(mut val) = col.to_json() {
                // Add category counts if available
                if let Some(ref cc) = cat_counts[col_idx] {
                    if let JsonValue::Object(ref mut obj) = val {
                        let json_counts: serde_json::Map<String, JsonValue> = cc
                            .iter()
                            .map(|(k, v)| (k.clone(), JsonValue::Number((*v).into())))
                            .collect();
                        obj.insert(
                            "category_counts".to_string(),
                            JsonValue::Object(json_counts),
                        );
                    }
                }
                // Merge text sample stats (from context phase) for text columns.
                if let Some(tss) = text_sample_stats.and_then(|m| m.get(name)) {
                    if let JsonValue::Object(ref mut obj) = val {
                        obj.insert("avg_tokens".to_string(), json!(tss.avg_tokens));
                        obj.insert("max_tokens".to_string(), json!(tss.max_tokens));
                        obj.insert(
                            "non_char_fraction".to_string(),
                            json!(tss.non_char_fraction),
                        );
                        obj.insert(
                            "string_truncation_scale".to_string(),
                            json!(tss.string_truncation_scale),
                        );
                        obj.insert("glove_overlap".to_string(), json!(tss.glove_overlap));
                    }
                }
                map.insert(name.clone(), val);
            }
        }
        JsonValue::Object(map)
    }
}

/// Free function to get category counts for a column (avoids self borrow issues).
fn get_category_counts_for_col(
    strategy: CategoryCountStrategy,
    _cat_local: &[Option<LocalColumnCounts>],
    table_cat: Option<&TableCategoryCounts>,
    col_idx: usize,
) -> Option<HashMap<String, u64>> {
    match strategy {
        CategoryCountStrategy::None => None,
        CategoryCountStrategy::Hybrid | CategoryCountStrategy::Graduation => {
            let cat = table_cat?;
            let col_shared = cat.columns.get(col_idx)?.as_ref()?;
            let mut counts = col_shared.drain_to_map()?;
            // Scale counts for per-row subsampling (except Graduation which outputs min_occ).
            if cat.row_subsample && strategy != CategoryCountStrategy::Graduation {
                for v in counts.values_mut() {
                    *v = (*v as f64 / ROW_SUBSAMPLE_PROB).round() as u64;
                }
            }
            Some(counts)
        }
    }
}

impl ColumnStatsCollector {
    fn from_semantic_type(stype: &Option<SemanticType>) -> Self {
        match stype {
            Some(SemanticType::Numerical) => ColumnStatsCollector::Numerical {
                null_count: 0,
                welford: WelfordAccumulator::new(),
                hll: HyperLogLog::new(),
                tdigest: TDigestEstimator::new(TDigestEstimator::DEFAULT_DELTA),
            },
            Some(SemanticType::Categorical) => ColumnStatsCollector::Categorical {
                count: 0,
                null_count: 0,
                hll: HyperLogLog::new(),
            },
            Some(SemanticType::Id) => ColumnStatsCollector::Id {
                count: 0,
                null_count: 0,
                hll: HyperLogLog::new(),
            },
            Some(SemanticType::Text) => ColumnStatsCollector::Text {
                count: 0,
                null_count: 0,
                hll: HyperLogLog::new(),
            },
            Some(SemanticType::Timestamp) => ColumnStatsCollector::Timestamp {
                count: 0,
                null_count: 0,
                parse_fail_count: 0,
                min_secs: i64::MAX,
                max_secs: i64::MIN,
            },
            Some(SemanticType::Sequence) => ColumnStatsCollector::Sequence {
                count: 0,
                null_count: 0,
                min_len: usize::MAX,
                max_len: 0,
                welford: WelfordAccumulator::new(),
            },
            Some(SemanticType::Multicategorical) | Some(SemanticType::Unsupported) | None => {
                ColumnStatsCollector::Skip
            }
        }
    }

    /// Merge another collector into this one. Both must be the same variant.
    fn merge(&mut self, other: &mut ColumnStatsCollector) {
        match (self, other) {
            (
                ColumnStatsCollector::Numerical {
                    null_count,
                    welford,
                    hll,
                    tdigest,
                },
                ColumnStatsCollector::Numerical {
                    null_count: nc2,
                    welford: w2,
                    hll: h2,
                    tdigest: td2,
                },
            ) => {
                *null_count += *nc2;
                welford.merge(w2);
                hll.merge(h2);
                tdigest.merge(td2);
            }
            (
                ColumnStatsCollector::Categorical {
                    count,
                    null_count,
                    hll,
                },
                ColumnStatsCollector::Categorical {
                    count: c2,
                    null_count: nc2,
                    hll: h2,
                },
            ) => {
                *count += *c2;
                *null_count += *nc2;
                hll.merge(h2);
            }
            (
                ColumnStatsCollector::Id {
                    count,
                    null_count,
                    hll,
                },
                ColumnStatsCollector::Id {
                    count: c2,
                    null_count: nc2,
                    hll: h2,
                },
            ) => {
                *count += *c2;
                *null_count += *nc2;
                hll.merge(h2);
            }
            (
                ColumnStatsCollector::Text {
                    count,
                    null_count,
                    hll,
                },
                ColumnStatsCollector::Text {
                    count: c2,
                    null_count: nc2,
                    hll: h2,
                },
            ) => {
                *count += *c2;
                *null_count += *nc2;
                hll.merge(h2);
            }
            (
                ColumnStatsCollector::Timestamp {
                    count,
                    null_count,
                    parse_fail_count,
                    min_secs,
                    max_secs,
                },
                ColumnStatsCollector::Timestamp {
                    count: c2,
                    null_count: nc2,
                    parse_fail_count: pf2,
                    min_secs: ms2,
                    max_secs: xs2,
                },
            ) => {
                *count += *c2;
                *null_count += *nc2;
                *parse_fail_count += *pf2;
                if *ms2 < *min_secs {
                    *min_secs = *ms2;
                }
                if *xs2 > *max_secs {
                    *max_secs = *xs2;
                }
            }
            (
                ColumnStatsCollector::Sequence {
                    count,
                    null_count,
                    min_len,
                    max_len,
                    welford,
                },
                ColumnStatsCollector::Sequence {
                    count: c2,
                    null_count: nc2,
                    min_len: ml2,
                    max_len: xl2,
                    welford: w2,
                },
            ) => {
                *count += *c2;
                *null_count += *nc2;
                if *ml2 < *min_len {
                    *min_len = *ml2;
                }
                if *xl2 > *max_len {
                    *max_len = *xl2;
                }
                welford.merge(w2);
            }
            (ColumnStatsCollector::Skip, ColumnStatsCollector::Skip) => {}
            _ => {
                // Mismatched variants — should not happen if semantic types are consistent
                debug_assert!(false, "ColumnStatsCollector::merge variant mismatch");
            }
        }
    }

    /// Serialize this collector's stats to JSON. Returns None for Skip.
    #[allow(clippy::wrong_self_convention)] // needs &mut self for tdigest flush
    fn to_json(&mut self) -> Option<JsonValue> {
        match self {
            ColumnStatsCollector::Numerical {
                null_count,
                welford,
                hll,
                tdigest,
            } => {
                let total = welford.count + *null_count;
                let (q25, q50, q75) = tdigest.quantiles().unwrap_or((0.0, 0.0, 0.0));
                Some(json!({
                    "semantic_type": "numerical",
                    "count": total,
                    "null_count": *null_count,
                    "mean": welford.mean(),
                    "std": welford.std_dev(),
                    "min": welford.min,
                    "max": welford.max,
                    "skewness": welford.skewness(),
                    "kurtosis": welford.kurtosis(),
                    "q25": q25,
                    "q50": q50,
                    "q75": q75,
                    "num_unique": hll.estimate() as u64,
                }))
            }
            ColumnStatsCollector::Categorical {
                count,
                null_count,
                hll,
            } => Some(json!({
                "semantic_type": "categorical",
                "count": *count + *null_count,
                "null_count": *null_count,
                "num_unique": hll.estimate() as u64,
            })),
            ColumnStatsCollector::Id {
                count,
                null_count,
                hll,
            } => Some(json!({
                "semantic_type": "id",
                "count": *count + *null_count,
                "null_count": *null_count,
                "num_unique": hll.estimate() as u64,
            })),
            ColumnStatsCollector::Text {
                count,
                null_count,
                hll,
            } => Some(json!({
                "semantic_type": "text",
                "count": *count + *null_count,
                "null_count": *null_count,
                "num_unique": hll.estimate() as u64,
            })),
            ColumnStatsCollector::Timestamp {
                count,
                null_count,
                parse_fail_count,
                min_secs,
                max_secs,
            } => {
                let total = *count + *null_count + *parse_fail_count;
                Some(json!({
                    "semantic_type": "timestamp",
                    "count": total,
                    "null_count": *null_count,
                    "parse_fail_count": *parse_fail_count,
                    "min_secs": *min_secs,
                    "max_secs": *max_secs,
                }))
            }
            ColumnStatsCollector::Sequence {
                count,
                null_count,
                min_len,
                max_len,
                welford,
            } => Some(json!({
                "semantic_type": "sequence",
                "count": *count + *null_count,
                "null_count": *null_count,
                "min_len": *min_len,
                "max_len": *max_len,
                "sequence_mean": welford.mean(),
                "sequence_std": welford.std_dev(),
            })),
            ColumnStatsCollector::Skip => None,
        }
    }
}

impl FileColStats {
    /// xorshift64* PRNG step → returns true if this row should be sampled.
    #[inline(always)]
    fn should_sample_row(&mut self) -> bool {
        if !self.row_subsample {
            return true;
        }
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        (self.rng.wrapping_mul(0x2545F4914F6CDD1D) >> 11) < ROW_SUBSAMPLE_THRESHOLD
    }

    /// Observe a value for category counts (string-typed categorical/id column).
    /// Dispatches to the appropriate strategy.
    #[inline]
    fn observe_category_string(&mut self, col_idx: usize, hash: u64, value: &str) {
        if !self.should_sample_row() {
            return;
        }
        match self.cat_strategy {
            CategoryCountStrategy::None => {}
            CategoryCountStrategy::Hybrid => {
                if let Some(ref shared) = self.cat_shared {
                    if let Some(ref col_shared) = shared.columns[col_idx] {
                        if col_shared.is_aborted() {
                            return;
                        }
                        if let Some(ref mut local) = self.cat_local[col_idx] {
                            if local.counts.contains_key(&hash) {
                                *local.counts.get_mut(&hash).unwrap() += 1;
                            } else if local.len() < HYBRID_LOCAL_CAP {
                                local.observe(hash, value);
                            } else {
                                col_shared.observe_hashed(hash, value);
                            }
                        }
                    }
                }
            }
            CategoryCountStrategy::Graduation => {
                if let Some(ref shared) = self.cat_shared {
                    if let Some(ref col_shared) = shared.columns[col_idx] {
                        if col_shared.is_aborted() {
                            return;
                        }
                        // Check local graduated snapshot first (~3-5ns)
                        if let Some(ref gs) = self.grad_snapshots[col_idx] {
                            if gs.contains(&hash) {
                                return;
                            }
                        }
                        // Then local HashMap (same as Hybrid)
                        if let Some(ref mut local) = self.cat_local[col_idx] {
                            if local.counts.contains_key(&hash) {
                                *local.counts.get_mut(&hash).unwrap() += 1;
                            } else if local.len() < HYBRID_LOCAL_CAP {
                                local.observe(hash, value);
                            } else {
                                let snap = col_shared.merge_local_batch_graduation(local);
                                local.clear();
                                // Refresh graduated snapshot
                                self.grad_snapshots[col_idx] = Some(snap);
                                local.observe(hash, value);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Observe a value for category counts (integer-typed categorical/id column).
    #[inline]
    fn observe_category_i64(&mut self, col_idx: usize, hash: u64, value: i64) {
        if !self.should_sample_row() {
            return;
        }
        match self.cat_strategy {
            CategoryCountStrategy::None => {}
            CategoryCountStrategy::Hybrid => {
                if let Some(ref shared) = self.cat_shared {
                    if let Some(ref col_shared) = shared.columns[col_idx] {
                        if col_shared.is_aborted() {
                            return;
                        }
                        if let Some(ref mut local) = self.cat_local[col_idx] {
                            if local.counts.contains_key(&hash) {
                                *local.counts.get_mut(&hash).unwrap() += 1;
                            } else if local.len() < HYBRID_LOCAL_CAP {
                                use std::collections::hash_map::Entry;
                                match local.counts.entry(hash) {
                                    Entry::Occupied(mut e) => *e.get_mut() += 1,
                                    Entry::Vacant(e) => {
                                        e.insert(1);
                                        local.strings.insert(hash, value.to_string());
                                    }
                                }
                            } else {
                                // Spill to global — inline for i64
                                if let CountStorage::Hashed { counts, strings } =
                                    &col_shared.storage
                                {
                                    let mut is_new = false;
                                    counts.entry(hash).and_modify(|c| *c += 1).or_insert_with(
                                        || {
                                            is_new = true;
                                            1
                                        },
                                    );
                                    if is_new {
                                        strings.entry(hash).or_insert_with(|| value.to_string());
                                        if col_shared.unique_count.fetch_add(1, Ordering::Relaxed)
                                            + 1
                                            > MAX_CATEGORIES
                                        {
                                            col_shared.aborted.store(true, Ordering::Relaxed);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            CategoryCountStrategy::Graduation => {
                if let Some(ref shared) = self.cat_shared {
                    if let Some(ref col_shared) = shared.columns[col_idx] {
                        if col_shared.is_aborted() {
                            return;
                        }
                        // Check local graduated snapshot first
                        if let Some(ref gs) = self.grad_snapshots[col_idx] {
                            if gs.contains(&hash) {
                                return;
                            }
                        }
                        // Then local HashMap (same as Hybrid)
                        if let Some(ref mut local) = self.cat_local[col_idx] {
                            if local.counts.contains_key(&hash) {
                                *local.counts.get_mut(&hash).unwrap() += 1;
                            } else if local.len() < HYBRID_LOCAL_CAP {
                                use std::collections::hash_map::Entry;
                                match local.counts.entry(hash) {
                                    Entry::Occupied(mut e) => *e.get_mut() += 1,
                                    Entry::Vacant(e) => {
                                        e.insert(1);
                                        local.strings.insert(hash, value.to_string());
                                    }
                                }
                            } else {
                                let snap = col_shared.merge_local_batch_graduation(local);
                                local.clear();
                                self.grad_snapshots[col_idx] = Some(snap);
                                // Observe the current value in the now-empty local buffer
                                use std::collections::hash_map::Entry;
                                match local.counts.entry(hash) {
                                    Entry::Occupied(mut e) => *e.get_mut() += 1,
                                    Entry::Vacant(e) => {
                                        e.insert(1);
                                        local.strings.insert(hash, value.to_string());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

impl ColumnValueObserver for FileColStats {
    #[inline]
    fn observe_i64(&mut self, col_idx: usize, value: i64) {
        // Pre-compute hash for categorical/id (shared between HLL and category counts).
        // This immutable borrow is released before the mutable match below.
        let cat_hash = match &self.cols[col_idx] {
            ColumnStatsCollector::Categorical { .. } | ColumnStatsCollector::Id { .. } => {
                Some(HyperLogLog::hash_i64(value))
            }
            _ => None,
        };

        match &mut self.cols[col_idx] {
            ColumnStatsCollector::Numerical {
                welford,
                hll,
                tdigest,
                ..
            } => {
                let fval = value as f64;
                welford.observe(fval);
                hll.observe_i64(value);
                tdigest.observe(fval);
            }
            ColumnStatsCollector::Categorical { count, hll, .. } => {
                *count += 1;
                hll.observe_hash(cat_hash.unwrap());
            }
            ColumnStatsCollector::Id { count, hll, .. } => {
                *count += 1;
                hll.observe_hash(cat_hash.unwrap());
            }
            ColumnStatsCollector::Timestamp { count, .. } => *count += 1,
            _ => {}
        }

        // Category counts — after releasing the borrow on self.cols
        if let Some(hash) = cat_hash {
            self.observe_category_i64(col_idx, hash, value);
        }
    }

    #[inline]
    fn observe_f64(&mut self, col_idx: usize, value: f64) {
        if let ColumnStatsCollector::Numerical {
            welford,
            hll,
            tdigest,
            ..
        } = &mut self.cols[col_idx]
        {
            welford.observe(value);
            hll.observe_f64(value);
            tdigest.observe(value);
        }
    }

    #[inline]
    fn observe_string(&mut self, col_idx: usize, value: &str) {
        // Pre-compute hash for categorical/id (shared between HLL and category counts).
        let cat_hash = match &self.cols[col_idx] {
            ColumnStatsCollector::Categorical { .. } | ColumnStatsCollector::Id { .. } => {
                Some(HyperLogLog::hash_bytes(value.as_bytes()))
            }
            _ => None,
        };

        match &mut self.cols[col_idx] {
            ColumnStatsCollector::Categorical { count, hll, .. } => {
                *count += 1;
                hll.observe_hash(cat_hash.unwrap());
            }
            ColumnStatsCollector::Id { count, hll, .. } => {
                *count += 1;
                hll.observe_hash(cat_hash.unwrap());
            }
            ColumnStatsCollector::Text { count, hll, .. } => {
                *count += 1;
                hll.observe_bytes(value.as_bytes());
            }
            _ => {}
        }

        // Category counts — after releasing the borrow on self.cols
        if let Some(hash) = cat_hash {
            self.observe_category_string(col_idx, hash, value);
        }
    }

    #[inline]
    fn observe_timestamp(&mut self, col_idx: usize, secs: i64) {
        if let ColumnStatsCollector::Timestamp {
            count,
            min_secs,
            max_secs,
            ..
        } = &mut self.cols[col_idx]
        {
            *count += 1;
            if secs < *min_secs {
                *min_secs = secs;
            }
            if secs > *max_secs {
                *max_secs = secs;
            }
        }
    }

    #[inline]
    fn observe_float_list(&mut self, col_idx: usize, view: &FloatListView) {
        if let ColumnStatsCollector::Sequence {
            count,
            min_len,
            max_len,
            welford,
            ..
        } = &mut self.cols[col_idx]
        {
            let len = view.len();
            *count += 1;
            if len < *min_len {
                *min_len = len;
            }
            if len > *max_len {
                *max_len = len;
            }
            view.for_each_f64(|v| welford.observe(v));
        }
    }

    #[inline]
    fn observe_null(&mut self, col_idx: usize) {
        match &mut self.cols[col_idx] {
            ColumnStatsCollector::Numerical { null_count, .. } => *null_count += 1,
            ColumnStatsCollector::Categorical { null_count, .. } => *null_count += 1,
            ColumnStatsCollector::Id { null_count, .. } => *null_count += 1,
            ColumnStatsCollector::Text { null_count, .. } => *null_count += 1,
            ColumnStatsCollector::Timestamp { null_count, .. } => *null_count += 1,
            ColumnStatsCollector::Sequence { null_count, .. } => *null_count += 1,
            ColumnStatsCollector::Skip => {}
        }
    }

    #[inline]
    fn observe_parse_failure(&mut self, col_idx: usize) {
        if let ColumnStatsCollector::Timestamp {
            parse_fail_count, ..
        } = &mut self.cols[col_idx]
        {
            *parse_fail_count += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_welford_basic() {
        let mut w = WelfordAccumulator::new();
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            w.observe(v);
        }
        assert_eq!(w.count, 8);
        assert!((w.mean() - 5.0).abs() < 1e-10);
        // Sample variance = M2/(n-1) = 32/7 ≈ 4.571
        let expected_var = 32.0 / 7.0;
        assert!((w.variance() - expected_var).abs() < 1e-10);
        assert!((w.std_dev() - expected_var.sqrt()).abs() < 1e-10);
        assert_eq!(w.min, 2.0);
        assert_eq!(w.max, 9.0);
    }

    #[test]
    fn test_welford_nan_inf_skipped() {
        let mut w = WelfordAccumulator::new();
        w.observe(1.0);
        w.observe(f64::NAN);
        w.observe(f64::INFINITY);
        w.observe(f64::NEG_INFINITY);
        w.observe(3.0);
        assert_eq!(w.count, 2);
        assert!((w.mean() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_welford_empty() {
        let w = WelfordAccumulator::new();
        assert_eq!(w.count, 0);
        assert_eq!(w.mean(), 0.0);
        assert_eq!(w.variance(), 0.0);
        assert_eq!(w.skewness(), 0.0);
        assert_eq!(w.kurtosis(), 0.0);
        assert_eq!(w.min, f64::INFINITY);
        assert_eq!(w.max, f64::NEG_INFINITY);
    }

    #[test]
    fn test_welford_merge() {
        // Split data into two halves, merge, compare against single-pass
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut single = WelfordAccumulator::new();
        for &v in &data {
            single.observe(v);
        }

        let mut a = WelfordAccumulator::new();
        for &v in &data[..5] {
            a.observe(v);
        }
        let mut b = WelfordAccumulator::new();
        for &v in &data[5..] {
            b.observe(v);
        }
        a.merge(&b);

        assert_eq!(a.count, single.count);
        assert!((a.mean() - single.mean()).abs() < 1e-10);
        assert!((a.variance() - single.variance()).abs() < 1e-10);
        assert!((a.skewness() - single.skewness()).abs() < 1e-6);
        assert!((a.kurtosis() - single.kurtosis()).abs() < 1e-6);
        assert_eq!(a.min, single.min);
        assert_eq!(a.max, single.max);
    }

    #[test]
    fn test_hll_cardinality() {
        let mut hll = HyperLogLog::new();
        let n = 100_000;
        for i in 0..n {
            hll.observe_i64(i);
        }
        let est = hll.estimate();
        let error = (est - n as f64).abs() / n as f64;
        assert!(
            error < 0.02,
            "HLL error too high: {:.1}% (est={:.0})",
            error * 100.0,
            est
        );
    }

    #[test]
    fn test_hll_strings() {
        let mut hll = HyperLogLog::new();
        let n = 50_000;
        for i in 0..n {
            let s = format!("value_{i}");
            hll.observe_bytes(s.as_bytes());
        }
        let est = hll.estimate();
        let error = (est - n as f64).abs() / n as f64;
        assert!(
            error < 0.03,
            "HLL string error: {:.1}% (est={:.0})",
            error * 100.0,
            est
        );
    }

    #[test]
    fn test_hll_merge() {
        let mut a = HyperLogLog::new();
        let mut b = HyperLogLog::new();
        for i in 0..50_000 {
            a.observe_i64(i);
        }
        for i in 50_000..100_000 {
            b.observe_i64(i);
        }
        a.merge(&b);
        let est = a.estimate();
        let error = (est - 100_000.0).abs() / 100_000.0;
        assert!(
            error < 0.02,
            "HLL merge error: {:.1}% (est={:.0})",
            error * 100.0,
            est
        );
    }

    #[test]
    fn test_reservoir_quantiles() {
        let mut r = ReservoirSampler::new(10_000);
        // Uniform [0, 1000)
        for i in 0..100_000 {
            r.observe(i as f64);
        }
        let (q25, q50, q75) = r.quantiles().unwrap();
        // With 100K values uniform in [0, 100000), quantiles should be ~25K, ~50K, ~75K
        assert!((q25 - 25_000.0).abs() < 2_000.0, "q25={q25}");
        assert!((q50 - 50_000.0).abs() < 2_000.0, "q50={q50}");
        assert!((q75 - 75_000.0).abs() < 2_000.0, "q75={q75}");
    }

    #[test]
    fn test_reservoir_merge() {
        let mut a = ReservoirSampler::new(10_000);
        let mut b = ReservoirSampler::new(10_000);
        for i in 0..50_000 {
            a.observe(i as f64);
        }
        for i in 50_000..100_000 {
            b.observe(i as f64);
        }
        a.merge(&b);
        let (q25, q50, q75) = a.quantiles().unwrap();
        assert!((q25 - 25_000.0).abs() < 2_000.0, "q25={q25}");
        assert!((q50 - 50_000.0).abs() < 2_000.0, "q50={q50}");
        assert!((q75 - 75_000.0).abs() < 2_000.0, "q75={q75}");
    }

    #[test]
    fn test_welford_uniform_kurtosis() {
        // Uniform distribution has kurtosis = -1.2
        let mut w = WelfordAccumulator::new();
        let n = 100_000;
        for i in 0..n {
            w.observe(i as f64 / n as f64);
        }
        // Excess kurtosis of uniform is -1.2
        assert!(
            (w.kurtosis() - (-1.2)).abs() < 0.05,
            "kurtosis={}",
            w.kurtosis()
        );
    }

    #[test]
    fn test_tdigest_quantiles() {
        let mut td = TDigestEstimator::new(200);
        for i in 0..100_000 {
            td.observe(i as f64);
        }
        let (q25, q50, q75) = td.quantiles().unwrap();
        assert!((q25 - 25_000.0).abs() < 500.0, "q25={q25}");
        assert!((q50 - 50_000.0).abs() < 500.0, "q50={q50}");
        assert!((q75 - 75_000.0).abs() < 500.0, "q75={q75}");
    }

    #[test]
    fn test_tdigest_merge() {
        let mut a = TDigestEstimator::new(200);
        let mut b = TDigestEstimator::new(200);
        for i in 0..50_000 {
            a.observe(i as f64);
        }
        for i in 50_000..100_000 {
            b.observe(i as f64);
        }
        a.merge(&mut b);
        let (q25, q50, q75) = a.quantiles().unwrap();
        assert!((q25 - 25_000.0).abs() < 500.0, "q25={q25}");
        assert!((q50 - 50_000.0).abs() < 500.0, "q50={q50}");
        assert!((q75 - 75_000.0).abs() < 500.0, "q75={q75}");
    }

    #[test]
    fn test_tdigest_lognormal() {
        // Lognormal has skewed quantiles — t-digest should handle this well
        let mut td = TDigestEstimator::new(200);
        // Generate lognormal-like values: exp(uniform * 5)
        for i in 0..100_000 {
            let x = ((i as f64) / 100_000.0 * 5.0).exp();
            td.observe(x);
        }
        let (q25, q50, q75) = td.quantiles().unwrap();
        // exp(1.25)≈3.49, exp(2.5)≈12.18, exp(3.75)≈42.52
        assert!((q25 - 3.49).abs() < 0.5, "q25={q25}");
        assert!((q50 - 12.18).abs() < 1.0, "q50={q50}");
        assert!((q75 - 42.52).abs() < 3.0, "q75={q75}");
    }

    #[test]
    fn test_reservoir_l_quantiles() {
        let mut r = ReservoirSamplerL::new(10_000);
        for i in 0..100_000 {
            r.observe(i as f64);
        }
        let (q25, q50, q75) = r.quantiles().unwrap();
        assert!((q25 - 25_000.0).abs() < 2_000.0, "q25={q25}");
        assert!((q50 - 50_000.0).abs() < 2_000.0, "q50={q50}");
        assert!((q75 - 75_000.0).abs() < 2_000.0, "q75={q75}");
    }

    // -----------------------------------------------------------------------
    // Graduation tests
    // -----------------------------------------------------------------------

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_isqrt() {
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(1), 1);
        assert_eq!(isqrt(4), 2);
        assert_eq!(isqrt(9), 3);
        assert_eq!(isqrt(10), 3);
        assert_eq!(isqrt(10_000), 100);
        assert_eq!(isqrt(100_000), 316);
        assert_eq!(isqrt(1_000_000), 1_000);
        // Verify: 316^2 = 99856 <= 100000, 317^2 = 100489 > 100000
        assert!(316 * 316 <= 100_000);
        assert!(317 * 317 > 100_000);
    }

    #[test]
    fn test_compute_min_occ() {
        // Small values clamp to 10_000 → sqrt(10_000) = 100
        assert_eq!(compute_min_occ(0), 100);
        assert_eq!(compute_min_occ(100), 100);
        assert_eq!(compute_min_occ(9_999), 100);
        assert_eq!(compute_min_occ(10_000), 100);
        // Larger values: sqrt(100_000) = 316
        assert_eq!(compute_min_occ(100_000), 316);
        // min_occ always >= 2
        assert!(compute_min_occ(0) >= 2);
    }

    /// Helper: build a LocalColumnCounts with a single hash observed `count` times.
    fn local_batch(hash: u64, value: &str, count: u64) -> LocalColumnCounts {
        let mut local = LocalColumnCounts::with_capacity(16);
        local.counts.insert(hash, count);
        local.strings.insert(hash, value.to_owned());
        local
    }

    #[test]
    fn test_graduation_basic() {
        let shared = SharedColumnCounts::new_graduation();
        let hash = HyperLogLog::hash_bytes(b"hello");
        // Merge a batch with count=200 (well above default min_occ=100)
        let mut local = local_batch(hash, "hello", 200);
        shared.merge_local_batch_graduation(&mut local);
        let result = shared.drain_to_map().unwrap();
        assert!(
            result.contains_key("hello"),
            "expected 'hello' in graduated set"
        );
    }

    #[test]
    fn test_graduation_below_threshold() {
        let shared = SharedColumnCounts::new_graduation();
        let hash = HyperLogLog::hash_bytes(b"rare");
        // Merge a batch with count=1 — should NOT graduate
        let mut local = local_batch(hash, "rare", 1);
        shared.merge_local_batch_graduation(&mut local);
        let result = shared.drain_to_map().unwrap();
        assert!(
            !result.contains_key("rare"),
            "'rare' should not be in graduated set"
        );
    }

    #[test]
    fn test_graduation_i64() {
        let shared = SharedColumnCounts::new_graduation();
        let hash = HyperLogLog::hash_i64(42);
        let mut local = local_batch(hash, "42", 200);
        shared.merge_local_batch_graduation(&mut local);
        let result = shared.drain_to_map().unwrap();
        assert!(result.contains_key("42"), "expected '42' in graduated set");
    }

    #[test]
    fn test_graduation_abort() {
        let shared = SharedColumnCounts::new_graduation();
        // Insert > MAX_CATEGORIES unique values via batches → should abort
        let batch_size = 1000;
        for batch_start in (0..(MAX_CATEGORIES + 100)).step_by(batch_size) {
            let mut local = LocalColumnCounts::with_capacity(batch_size);
            for i in
                batch_start..std::cmp::min(batch_start + batch_size as u64, MAX_CATEGORIES + 100)
            {
                let hash = HyperLogLog::hash_i64(i as i64);
                local.counts.insert(hash, 1);
                local.strings.insert(hash, i.to_string());
            }
            shared.merge_local_batch_graduation(&mut local);
            if shared.is_aborted() {
                break;
            }
        }
        // Should have been aborted by MAX_CATEGORIES
        assert!(shared.is_aborted());
        assert!(shared.drain_to_map().is_none());
    }

    #[test]
    fn test_graduation_threshold_adapts() {
        // With many unique categories, min_occ should increase
        let shared = SharedColumnCounts::new_graduation();
        // Insert 50_000 unique values (1 each) via batches
        let batch_size = 5000;
        for batch_start in (0..50_000u64).step_by(batch_size) {
            let mut local = LocalColumnCounts::with_capacity(batch_size);
            for i in batch_start..batch_start + batch_size as u64 {
                let hash = HyperLogLog::hash_i64(i as i64);
                local.counts.insert(hash, 1);
                local.strings.insert(hash, i.to_string());
            }
            shared.merge_local_batch_graduation(&mut local);
        }
        // Now min_occ should be sqrt(50000) ≈ 223
        // Re-observe value 0 with count=300 to exceed threshold
        let hash = HyperLogLog::hash_i64(0);
        let mut local = local_batch(hash, "0", 300);
        shared.merge_local_batch_graduation(&mut local);
        let result = shared.drain_to_map().unwrap();
        assert!(
            result.contains_key("0"),
            "expected '0' to graduate after 300+ observations"
        );
    }

    #[test]
    fn test_text_sample_stats_basic() {
        let values = vec![
            "hello world",        // 2 tokens, both alpha, both in GloVe
            "foo bar baz",        // 3 tokens, all alpha
            "http://example.com", // punctuation → spaces → multiple tokens
        ];
        let stats = compute_text_sample_stats(&values);
        // basic_english_normalize: lowercase + replace '\".,()!?;: with space + split
        // "hello world" → ["hello", "world"] = 2 tokens
        // "foo bar baz" → ["foo", "bar", "baz"] = 3 tokens
        // "http://example.com" → "http //example com" → ["http", "//example", "com"] = 3 tokens
        //   (/ is not in punctuation list, so stays)
        assert!(stats.avg_tokens > 0.0, "avg_tokens should be > 0");
        assert!(stats.max_tokens >= 3, "max_tokens should be >= 3");
        assert!(stats.glove_overlap >= 0.0 && stats.glove_overlap <= 1.0);
        assert!(stats.non_char_fraction >= 0.0 && stats.non_char_fraction <= 1.0);
        assert!(
            (stats.string_truncation_scale - 1.0).abs() < 1e-9,
            "short strings should have scale 1.0"
        );
    }

    #[test]
    fn test_text_sample_stats_empty() {
        let stats = compute_text_sample_stats(&[]);
        assert_eq!(stats.avg_tokens, 0.0);
        assert_eq!(stats.max_tokens, 0);
        assert_eq!(stats.glove_overlap, 0.0);
        assert!((stats.string_truncation_scale - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_text_sample_stats_truncation_scale() {
        // Create a string > 512 bytes
        let long = "a".repeat(1024);
        let stats = compute_text_sample_stats(&[&long]);
        assert!(
            (stats.string_truncation_scale - 2.0).abs() < 1e-9,
            "1024/512 = 2.0"
        );
    }
}
