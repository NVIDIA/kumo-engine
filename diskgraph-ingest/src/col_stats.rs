//! Per-column statistics collection during the encode stage.
//!
//! `FileColStats` implements `ColumnValueObserver` and accumulates per-file
//! statistics using streaming algorithms. After a file is processed, the
//! per-file stats are merged into a shared `TableColStats` accumulator
//! via a mutex.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::graph_config::{ColumnSpec, IngestedType, SemanticType};
use diskgraph_rowpack::{ColumnValueObserver, FloatListView, Int64ListView, StringListView};
use serde_json::{json, Value as JsonValue};

// ---------------------------------------------------------------------------
// Category counts constants
// ---------------------------------------------------------------------------

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
// Category counting infrastructure — re-exported from diskgraph-core
// ---------------------------------------------------------------------------

pub use diskgraph_core::accumulators::{
    CategoryCountStrategy, CountStorage, IdentityBuildHasher, IdentityHasher, LocalColumnCounts,
    MulticatInputKind, SharedColumnCounts, build_multicat_emit_stats, compute_min_occ,
    isqrt, kumo_entry, kumo_key, kumo_safe_div, MAX_CATEGORIES,
};

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
    /// Categorical, Id, and Multicategorical columns get category count collection.
    /// Multicategorical reuses the same machinery: one DashMap entry per distinct
    /// *token* (not per row), fed by the per-input-kind tokenization paths in
    /// `ColumnStatsCollector::Multicategorical`.
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
                Some(SemanticType::Multicategorical) => {
                    let shared = match strategy {
                        CategoryCountStrategy::Graduation => {
                            SharedColumnCounts::new_graduation_uncapped()
                        }
                        _ => SharedColumnCounts::new_hashed_uncapped(),
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

// ---------------------------------------------------------------------------
// TDigestEstimator & WelfordAccumulator — re-exported from diskgraph-core
// ---------------------------------------------------------------------------

pub use diskgraph_core::accumulators::TDigestEstimator;
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
        invalid_count: u64,
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
        /// Up to 3 sample values that failed to parse (for error messages).
        parse_fail_samples: Vec<String>,
        min_secs: i64,
        max_secs: i64,
        hll: HyperLogLog,
    },
    Sequence {
        count: u64,
        null_count: u64,
        min_len: usize,
        max_len: usize,
        welford: WelfordAccumulator,
    },
    /// Image URL column — stores only universal stats (count, null_count).
    /// Produces `kumo.image` keys in col_stats output.
    ImageUrl { count: u64, null_count: u64 },
    /// Multicategorical column — one row produces N tokens; per-token
    /// counting reuses the existing `SharedColumnCounts` / `LocalColumnCounts`
    /// machinery via `observe_category_{string,i64}`. This variant carries
    /// only universal per-column state; token counts live in the shared
    /// per-column category counts, same as Categorical/Id.
    Multicategorical {
        /// Non-null rows observed.
        count: u64,
        /// Null rows observed.
        null_count: u64,
        /// HLL over per-token hashes — source of `NUM_UNIQUE_MULTI`.
        hll: HyperLogLog,
        /// Tokenization dispatch; frozen at construction.
        input_kind: MulticatInputKind,
        /// Emit-only: separator string for `MULTI_CATEGORIES_SEPARATOR`.
        /// `Some(sep)` for `StringScalar` with a configured delimiter;
        /// `None` otherwise (native lists, or single-token-per-row string).
        separator: Option<String>,
        /// Hot-path split hint for the config-converter's ASCII delimiters.
        /// Keeps scalar-string observation from cloning the separator per row.
        separator_byte: Option<u8>,
    },
    /// Skip this column (PK, Unsupported).
    Skip,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TimestampParseFailureSummary {
    total_rows: u64,
    source_non_null_rows: u64,
    source_null_rows: u64,
    parse_failures: u64,
}

impl TimestampParseFailureSummary {
    fn new(count: u64, null_count: u64, parse_fail_count: u64) -> Option<Self> {
        let total_rows = count.saturating_add(null_count);
        if total_rows == 0 {
            return None;
        }
        Some(Self {
            total_rows,
            source_non_null_rows: count.saturating_add(parse_fail_count),
            source_null_rows: null_count.saturating_sub(parse_fail_count),
            parse_failures: parse_fail_count,
        })
    }

    fn parse_failure_pct_of_non_null(self) -> f64 {
        if self.source_non_null_rows == 0 {
            0.0
        } else {
            self.parse_failures as f64 / self.source_non_null_rows as f64 * 100.0
        }
    }

    fn parse_failure_pct_of_total(self) -> f64 {
        self.parse_failures as f64 / self.total_rows as f64 * 100.0
    }
}

impl FileColStats {
    /// Create a new observer for a file. No category count collection.
    ///
    /// `columns` supplies the per-column `semantic_type`, `ingested_dtype`,
    /// and (for multicategorical string columns) `multicategorical_delimiter`
    /// used to construct each collector.
    pub fn new(columns: &[ColumnSpec]) -> Self {
        let cols = columns
            .iter()
            .map(|c| {
                ColumnStatsCollector::from_column_type(
                    &c.semantic_type,
                    &c.ingested_dtype,
                    c.multicategorical_delimiter.as_deref(),
                )
            })
            .collect();
        let n = columns.len();
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
        columns: &[ColumnSpec],
        table_cat: Arc<TableCategoryCounts>,
    ) -> Self {
        let cols: Vec<ColumnStatsCollector> = columns
            .iter()
            .map(|c| {
                ColumnStatsCollector::from_column_type(
                    &c.semantic_type,
                    &c.ingested_dtype,
                    c.multicategorical_delimiter.as_deref(),
                )
            })
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
            _ => (0..columns.len()).map(|_| None).collect(),
        };
        // Graduation: init grad_snapshots as empty sets for eligible columns
        let grad_snapshots = match strategy {
            CategoryCountStrategy::Graduation => table_cat
                .columns
                .iter()
                .map(|shared| shared.as_ref().map(|_| Arc::new(HashSet::new())))
                .collect(),
            _ => (0..columns.len()).map(|_| None).collect(),
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
                        if let Some(inner) = local.as_mut() {
                            if let Some(ref col_shared) = shared.columns[col_idx] {
                                col_shared.merge_local_batch(inner);
                            }
                        }
                        // Drop the local HashMap entirely to free bucket arrays.
                        // After merging to shared, local counts are never reused
                        // and the FileColStats will sit in an unbounded channel
                        // until all files complete — retaining capacity across
                        // 19K+ files would consume 100+ GB.
                        *local = None;
                    }
                }
            }
            CategoryCountStrategy::Graduation => {
                if let Some(ref shared) = self.cat_shared {
                    for (col_idx, local) in self.cat_local.iter_mut().enumerate() {
                        if let Some(inner) = local.as_mut() {
                            if let Some(ref col_shared) = shared.columns[col_idx] {
                                let snap = col_shared.merge_local_batch_graduation(inner);
                                self.grad_snapshots[col_idx] = Some(snap);
                            }
                        }
                        *local = None;
                    }
                }
            }
            _ => {}
        }
    }

    /// Warn when timestamp parse failures were observed.
    ///
    /// Timestamp parse failures are diagnostics, not fatal errors: create-time
    /// rows are dropped by ingest, and non-time timestamp features encode as
    /// null. `parse_fail_count` is a subset of `null_count`, so totals are
    /// derived from valid + encoded-null rows and do not add parse failures a
    /// second time.
    pub fn warn_parse_failures(&self, column_names: &[String], table_name: &str) {
        for (col, name) in self.cols.iter().zip(column_names.iter()) {
            if let ColumnStatsCollector::Timestamp {
                count,
                null_count,
                parse_fail_count,
                parse_fail_samples,
                ..
            } = col
            {
                let Some(summary) =
                    TimestampParseFailureSummary::new(*count, *null_count, *parse_fail_count)
                else {
                    continue;
                };
                if summary.parse_failures > 0 {
                    let samples = if parse_fail_samples.is_empty() {
                        String::new()
                    } else {
                        format!(
                            " [{}]",
                            parse_fail_samples
                                .iter()
                                .map(|s| format!("'{}'", s))
                                .collect::<Vec<_>>()
                                .join(", "),
                        )
                    };
                    tracing::warn!(
                        table = table_name,
                        column = name.as_str(),
                        parse_failures = summary.parse_failures,
                        source_non_null_rows = summary.source_non_null_rows,
                        source_null_rows = summary.source_null_rows,
                        total_rows = summary.total_rows,
                        pct_of_non_null = format_args!(
                            "{:.1}%",
                            summary.parse_failure_pct_of_non_null()
                        ),
                        pct_of_total = format_args!(
                            "{:.1}%",
                            summary.parse_failure_pct_of_total()
                        ),
                        samples = samples.as_str(),
                        "timestamp parse failures detected; invalid timestamp values were treated as null or dropped according to time-column semantics",
                    );
                }
            }
        }
    }

    /// Serialize accumulated stats to a JSON object keyed by column name.
    /// Category counts are included from the shared state (Approach 2&3) or
    /// local accumulator (Approach 1).
    pub fn to_json(
        &mut self,
        column_names: &[String],
        ingested_types: &[IngestedType],
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
                // Add category counts if available, as array of [key, count] pairs
                // matching the format GraphEngine/Spark produces.
                if let Some(ref cc) = cat_counts[col_idx] {
                    if let JsonValue::Object(ref mut obj) = val {
                        let is_int = ingested_types
                            .get(col_idx)
                            .is_some_and(|t| t.is_integer_like());
                        let json_counts: Vec<JsonValue> = cc
                            .iter()
                            .map(|(k, v)| {
                                let key_val = if is_int {
                                    k.parse::<i64>()
                                        .map(|n| JsonValue::Number(n.into()))
                                        .unwrap_or_else(|_| JsonValue::String(k.clone()))
                                } else {
                                    JsonValue::String(k.clone())
                                };
                                json!([key_val, v])
                            })
                            .collect();
                        obj.insert("category_counts".to_string(), JsonValue::Array(json_counts));
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

#[inline]
fn single_byte_ascii(s: &str) -> Option<u8> {
    if s.len() == 1 && s.is_ascii() {
        Some(s.as_bytes()[0])
    } else {
        None
    }
}

impl ColumnStatsCollector {
    /// Construct the per-column collector for an ingest scan.
    ///
    /// Multicategorical columns need both the declared `ingested_type` (to pick
    /// the tokenization path) and, for `IngestedType::String`, the
    /// config-time-detected `multicategorical_delimiter`. An unsupported
    /// combination of `Multicategorical + IngestedType` falls back to `Skip` —
    /// the convert layer is responsible for rejecting such columns upstream
    /// with a clear error, so this fallback should be unreachable in practice.
    fn from_column_type(
        stype: &Option<SemanticType>,
        ingested_type: &IngestedType,
        multicategorical_delimiter: Option<&str>,
    ) -> Self {
        match stype {
            Some(SemanticType::Numerical) => ColumnStatsCollector::Numerical {
                null_count: 0,
                invalid_count: 0,
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
                parse_fail_samples: Vec::new(),
                min_secs: i64::MAX,
                max_secs: i64::MIN,
                hll: HyperLogLog::new(),
            },
            Some(SemanticType::Sequence) => ColumnStatsCollector::Sequence {
                count: 0,
                null_count: 0,
                min_len: usize::MAX,
                max_len: 0,
                welford: WelfordAccumulator::new(),
            },
            Some(SemanticType::ImageUrl) => ColumnStatsCollector::ImageUrl {
                count: 0,
                null_count: 0,
            },
            Some(SemanticType::Multicategorical) => {
                let input_kind = match ingested_type {
                    IngestedType::String => Some(MulticatInputKind::SingleStringToken),
                    IngestedType::StringArray => Some(MulticatInputKind::StringArray),
                    IngestedType::Int64Array => Some(MulticatInputKind::Int64Array),
                    IngestedType::Float16Array
                    | IngestedType::Float32Array
                    | IngestedType::Float64Array => Some(MulticatInputKind::FloatArray),
                    _ => None,
                };
                match input_kind {
                    Some(kind) => ColumnStatsCollector::Multicategorical {
                        count: 0,
                        null_count: 0,
                        hll: HyperLogLog::new(),
                        input_kind: kind,
                        // Emit `MULTI_CATEGORIES_SEPARATOR` whenever a delimiter
                        // is on the column config — including for columns that
                        // were pre-split into `StringArray` at config-time
                        // (kumo-ai/kumo#28478). The stat documents the original
                        // tokenization choice; downstream readers don't need it
                        // to reconstruct tokens (the on-disk shape is already a
                        // list) but it preserves stat-schema parity with the
                        // pre-normalization world.
                        separator: multicategorical_delimiter.map(String::from),
                        separator_byte: multicategorical_delimiter.and_then(single_byte_ascii),
                    },
                    // Defensive: the convert layer should already have rejected
                    // this combination. Falling back to Skip keeps ingest going
                    // instead of panicking mid-scan.
                    None => ColumnStatsCollector::Skip,
                }
            }
            Some(SemanticType::Unsupported) | None => ColumnStatsCollector::Skip,
        }
    }

    /// Merge another collector into this one. Both must be the same variant.
    fn merge(&mut self, other: &mut ColumnStatsCollector) {
        match (self, other) {
            (
                ColumnStatsCollector::Numerical {
                    null_count,
                    invalid_count,
                    welford,
                    hll,
                    tdigest,
                },
                ColumnStatsCollector::Numerical {
                    null_count: nc2,
                    invalid_count: ic2,
                    welford: w2,
                    hll: h2,
                    tdigest: td2,
                },
            ) => {
                *null_count += *nc2;
                *invalid_count += *ic2;
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
                    parse_fail_samples,
                    min_secs,
                    max_secs,
                    hll,
                },
                ColumnStatsCollector::Timestamp {
                    count: c2,
                    null_count: nc2,
                    parse_fail_count: pf2,
                    parse_fail_samples: pfs2,
                    min_secs: ms2,
                    max_secs: xs2,
                    hll: hll2,
                },
            ) => {
                *count += *c2;
                *null_count += *nc2;
                *parse_fail_count += *pf2;
                hll.merge(hll2);
                // Keep up to 3 sample values across merges.
                for s in pfs2 {
                    if parse_fail_samples.len() < 3 {
                        parse_fail_samples.push(s.clone());
                    }
                }
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
            (
                ColumnStatsCollector::ImageUrl { count, null_count },
                ColumnStatsCollector::ImageUrl {
                    count: c2,
                    null_count: nc2,
                },
            ) => {
                *count += *c2;
                *null_count += *nc2;
            }
            (
                ColumnStatsCollector::Multicategorical {
                    count,
                    null_count,
                    hll,
                    input_kind,
                    separator: _,
                    separator_byte: _,
                },
                ColumnStatsCollector::Multicategorical {
                    count: c2,
                    null_count: nc2,
                    hll: h2,
                    input_kind: ik2,
                    separator: _,
                    separator_byte: _,
                },
            ) => {
                // Defensive: same stype on both sides must resolve to the same
                // tokenization path. If this ever trips, the two FileColStats
                // were built from inconsistent ColumnSpecs.
                debug_assert_eq!(
                    *input_kind, *ik2,
                    "multicategorical merge: input_kind mismatch"
                );
                *count += *c2;
                *null_count += *nc2;
                hll.merge(h2);
                // Token counts merge through SharedColumnCounts — no per-file
                // state to combine here. The separator is read from the column
                // config on both sides; it doesn't need merging.
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
                invalid_count,
                welford,
                hll,
                tdigest,
            } => {
                let total = welford.count + *null_count + *invalid_count;
                let (q25, q50, q75) = tdigest.quantiles().unwrap_or((0.0, 0.0, 0.0));
                Some(json!({
                    "semantic_type": "numerical",
                    "count": total,
                    "null_count": *null_count,
                    "invalid_count": *invalid_count,
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
                hll,
                ..
            } => {
                let total = (*count).saturating_add(*null_count);
                Some(json!({
                    "semantic_type": "timestamp",
                    "count": total,
                    "null_count": *null_count,
                    "parse_fail_count": *parse_fail_count,
                    "num_unique": hll.estimate() as u64,
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
            ColumnStatsCollector::ImageUrl { count, null_count } => Some(json!({
                "semantic_type": "image_url",
                "count": *count + *null_count,
                "null_count": *null_count,
            })),
            ColumnStatsCollector::Multicategorical {
                count,
                null_count,
                hll,
                ..
            } => Some(json!({
                "semantic_type": "multicategorical",
                "count": *count + *null_count,
                "null_count": *null_count,
                "num_unique_multi": hll.estimate() as u64,
            })),
            ColumnStatsCollector::Skip => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Kumo-format stats output (replaces Python bridge)
// ---------------------------------------------------------------------------

/// Format an epoch-second timestamp to kumo timestamp string.
/// Sentinels (i64::MAX/MIN) and unparseable values → "kumo_timestamp_NaT".
fn format_kumo_timestamp(epoch_secs: i64) -> String {
    if epoch_secs == i64::MAX || epoch_secs == i64::MIN {
        return "kumo_timestamp_NaT".to_string();
    }
    match chrono::DateTime::from_timestamp(epoch_secs, 0) {
        Some(dt) => format!("kumo_timestamp_{}", dt.format("%Y-%m-%d %H:%M:%S")),
        None => "kumo_timestamp_NaT".to_string(),
    }
}

fn kumo_stype_key(col: &ColumnStatsCollector) -> &'static str {
    match col {
        ColumnStatsCollector::Numerical { .. } => "kumo.numerical",
        ColumnStatsCollector::Categorical { .. } => "kumo.categorical",
        ColumnStatsCollector::Id { .. } => "kumo.ID",
        ColumnStatsCollector::Text { .. } => "kumo.text",
        ColumnStatsCollector::Timestamp { .. } => "kumo.timestamp",
        ColumnStatsCollector::Sequence { .. } => "kumo.sequence",
        ColumnStatsCollector::ImageUrl { .. } => "kumo.image",
        ColumnStatsCollector::Multicategorical { .. } => "kumo.multicategorical",
        ColumnStatsCollector::Skip => "",
    }
}

/// Build category count JSON array: [[key, count], ...].
/// For integer-like columns, keys are emitted as JSON numbers.
fn build_cat_count_json(
    col_idx: usize,
    cat_counts: &[Option<HashMap<String, u64>>],
    ingested_types: &[IngestedType],
) -> JsonValue {
    let Some(ref cc) = cat_counts[col_idx] else {
        return json!([]);
    };
    let ingested_type = ingested_types.get(col_idx);
    let is_bool = matches!(ingested_type, Some(IngestedType::Bool));
    // Integer-emitting classification includes `Int64Array` so that a
    // `list<int64> + multicategorical` column emits numeric keys byte-matching
    // what `tt_executor` produces for train-table multicat stats.
    let is_int = (ingested_type.is_some_and(|t| t.is_integer_like())
        || matches!(ingested_type, Some(IngestedType::Int64Array)))
        && !is_bool;
    // Float-emitting classification includes the `Float*Array` shapes for the
    // same reason. The token strings stored in `SharedColumnCounts::strings`
    // come from `format!("{}", f)` / `to_string()` and parse cleanly back
    // into JSON numbers for the common cases; non-finite paths (Inf,
    // subnormals) fall back to a string key via the `.parse::<f64>().ok()`
    // guard below.
    let is_float = matches!(
        ingested_type,
        Some(IngestedType::Float32)
            | Some(IngestedType::Float64)
            | Some(IngestedType::Float16Array)
            | Some(IngestedType::Float32Array)
            | Some(IngestedType::Float64Array)
    );
    let pairs: Vec<JsonValue> = cc
        .iter()
        .map(|(k, v)| {
            let key_val = if is_bool {
                // Emit JSON bools to match Spark's True/False representation.
                match k.as_str() {
                    "1" | "true" => JsonValue::Bool(true),
                    "0" | "false" => JsonValue::Bool(false),
                    _ => JsonValue::String(k.clone()),
                }
            } else if is_int {
                k.parse::<i64>()
                    .map(|n| JsonValue::Number(n.into()))
                    .unwrap_or_else(|_| JsonValue::String(k.clone()))
            } else if is_float {
                k.parse::<f64>()
                    .ok()
                    .and_then(serde_json::Number::from_f64)
                    .map(JsonValue::Number)
                    .unwrap_or_else(|| JsonValue::String(k.clone()))
            } else {
                JsonValue::String(k.clone())
            };
            json!([key_val, v])
        })
        .collect();
    JsonValue::Array(pairs)
}

/// Truncate CATEGORY_COUNTS values to `limit` entries for minified output.
pub fn minify_kumo_stats(
    stats: &serde_json::Map<String, JsonValue>,
    limit: usize,
) -> serde_json::Map<String, JsonValue> {
    stats
        .iter()
        .map(|(key, value)| {
            if key.ends_with("CATEGORY_COUNTS__False") || key.ends_with("CATEGORY_COUNTS__True") {
                let mut val = value.clone();
                if let Some(obj) = val.as_object_mut() {
                    if let Some(arr) = obj.get_mut("value").and_then(|v| v.as_array_mut()) {
                        arr.truncate(limit);
                    }
                }
                (key.clone(), val)
            } else {
                (key.clone(), value.clone())
            }
        })
        .collect()
}

impl FileColStats {
    /// Produce kumo-format stats for one table.
    ///
    /// Each column produces stat entries keyed as
    /// `"{col}__{kumo.stype}__{STAT}__{True|False}"` with values
    /// `{"value": .., "approx": bool, "data_version": str, "num_entries": int}`.
    ///
    /// The output is written directly to per-table JSON files and copied
    /// into the kumo table cache by the Python `diskgraph_col_stats_activity`.
    pub fn to_kumo_json(
        &mut self,
        column_names: &[String],
        ingested_types: &[IngestedType],
        table_cat: Option<&TableCategoryCounts>,
        text_sample_stats: Option<&HashMap<String, TextSampleStats>>,
        data_version: &str,
    ) -> serde_json::Map<String, JsonValue> {
        // Pre-collect category counts (same pattern as to_json to avoid borrow issues).
        let cat_counts: Vec<Option<HashMap<String, u64>>> = (0..self.cols.len())
            .map(|col_idx| {
                get_category_counts_for_col(self.cat_strategy, &self.cat_local, table_cat, col_idx)
            })
            .collect();

        let mut map = serde_json::Map::new();
        let dv = data_version;

        for (col_idx, col) in self.cols.iter_mut().enumerate() {
            let name = &column_names[col_idx];
            let sk = kumo_stype_key(col);

            match col {
                ColumnStatsCollector::Numerical {
                    null_count,
                    invalid_count,
                    welford,
                    hll,
                    tdigest,
                } => {
                    let count = welford.count + *null_count + *invalid_count;
                    // Base stats
                    map.insert(
                        kumo_key(name, sk, "COUNT", false),
                        kumo_entry(json!(count), false, dv, count),
                    );
                    map.insert(
                        kumo_key(name, sk, "NUM_NA", false),
                        kumo_entry(json!(*null_count), false, dv, count),
                    );
                    map.insert(
                        kumo_key(name, sk, "NA_FRACTION", false),
                        kumo_entry(json!(kumo_safe_div(*null_count, count)), false, dv, count),
                    );
                    map.insert(
                        kumo_key(name, sk, "INVALID_FRACTION", false),
                        kumo_entry(json!(kumo_safe_div(*invalid_count, count)), false, dv, count),
                    );
                    // Numerical-specific
                    map.insert(
                        kumo_key(name, sk, "MEAN", false),
                        kumo_entry(json!(welford.mean()), false, dv, count),
                    );
                    map.insert(
                        kumo_key(name, sk, "STD", false),
                        kumo_entry(json!(welford.std_dev()), false, dv, count),
                    );
                    map.insert(
                        kumo_key(name, sk, "MIN", false),
                        kumo_entry(json!(welford.min), false, dv, count),
                    );
                    map.insert(
                        kumo_key(name, sk, "MAX", false),
                        kumo_entry(json!(welford.max), false, dv, count),
                    );
                    let (q25, q50, q75) = tdigest.quantiles().unwrap_or((0.0, 0.0, 0.0));
                    map.insert(
                        kumo_key(name, sk, "QUANTILES", false),
                        kumo_entry(json!([q25, q50, q75]), false, dv, count),
                    );
                    map.insert(
                        kumo_key(name, sk, "QUANTILE25", false),
                        kumo_entry(json!(q25), false, dv, count),
                    );
                    map.insert(
                        kumo_key(name, sk, "MEDIAN", false),
                        kumo_entry(json!(q50), false, dv, count),
                    );
                    map.insert(
                        kumo_key(name, sk, "QUANTILE75", false),
                        kumo_entry(json!(q75), false, dv, count),
                    );
                    map.insert(
                        kumo_key(name, sk, "MEDIAN", true),
                        kumo_entry(json!(q50), true, dv, count),
                    );
                    let kurt = welford.kurtosis();
                    map.insert(
                        kumo_key(name, sk, "KURTOSIS", false),
                        kumo_entry(json!(kurt), false, dv, count),
                    );
                    map.insert(
                        kumo_key(name, sk, "KURTOSIS", true),
                        kumo_entry(json!(kurt), true, dv, count),
                    );
                    let num_unique = hll.estimate() as u64;
                    let num_unique = if num_unique > count {
                        count
                    } else {
                        num_unique
                    };
                    map.insert(
                        kumo_key(name, sk, "NUM_UNIQUE", false),
                        kumo_entry(json!(num_unique), false, dv, count),
                    );
                    map.insert(
                        kumo_key(name, sk, "UNIQUE_FRACTION", false),
                        kumo_entry(json!(kumo_safe_div(num_unique, count)), false, dv, count),
                    );
                }
                ColumnStatsCollector::Categorical {
                    count,
                    null_count,
                    hll,
                } => {
                    let total = *count + *null_count;
                    map.insert(
                        kumo_key(name, sk, "COUNT", false),
                        kumo_entry(json!(total), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "NUM_NA", false),
                        kumo_entry(json!(*null_count), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "NA_FRACTION", false),
                        kumo_entry(json!(kumo_safe_div(*null_count, total)), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "INVALID_FRACTION", false),
                        kumo_entry(json!(0.0), false, dv, total),
                    );
                    // Category counts
                    let values = build_cat_count_json(col_idx, &cat_counts, ingested_types);
                    map.insert(
                        kumo_key(name, sk, "CATEGORY_COUNTS", false),
                        kumo_entry(values, false, dv, total),
                    );
                    let num_unique = hll.estimate() as u64;
                    let num_unique = if num_unique > total {
                        total
                    } else {
                        num_unique
                    };
                    map.insert(
                        kumo_key(name, sk, "NUM_UNIQUE", false),
                        kumo_entry(json!(num_unique), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "UNIQUE_FRACTION", false),
                        kumo_entry(json!(kumo_safe_div(num_unique, total)), false, dv, total),
                    );
                }
                ColumnStatsCollector::Id {
                    count,
                    null_count,
                    hll,
                } => {
                    let total = *count + *null_count;
                    map.insert(
                        kumo_key(name, sk, "COUNT", false),
                        kumo_entry(json!(total), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "NUM_NA", false),
                        kumo_entry(json!(*null_count), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "NA_FRACTION", false),
                        kumo_entry(json!(kumo_safe_div(*null_count, total)), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "INVALID_FRACTION", false),
                        kumo_entry(json!(0.0), false, dv, total),
                    );
                    let values = build_cat_count_json(col_idx, &cat_counts, ingested_types);
                    map.insert(
                        kumo_key(name, sk, "CATEGORY_COUNTS", false),
                        kumo_entry(values, false, dv, total),
                    );
                    let num_unique = hll.estimate() as u64;
                    let num_unique = if num_unique > total {
                        total
                    } else {
                        num_unique
                    };
                    map.insert(
                        kumo_key(name, sk, "NUM_UNIQUE", false),
                        kumo_entry(json!(num_unique), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "UNIQUE_FRACTION", false),
                        kumo_entry(json!(kumo_safe_div(num_unique, total)), false, dv, total),
                    );
                }
                ColumnStatsCollector::Text {
                    count,
                    null_count,
                    hll,
                } => {
                    let total = *count + *null_count;
                    map.insert(
                        kumo_key(name, sk, "COUNT", false),
                        kumo_entry(json!(total), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "NUM_NA", false),
                        kumo_entry(json!(*null_count), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "NA_FRACTION", false),
                        kumo_entry(json!(kumo_safe_div(*null_count, total)), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "INVALID_FRACTION", false),
                        kumo_entry(json!(0.0), false, dv, total),
                    );
                    // Text sample stats (approx=true)
                    if let Some(tss) = text_sample_stats.and_then(|m| m.get(name.as_str())) {
                        map.insert(
                            kumo_key(name, sk, "STRING_AVG_TOKENS", true),
                            kumo_entry(json!(tss.avg_tokens), true, dv, total),
                        );
                        map.insert(
                            kumo_key(name, sk, "STRING_MAX_TOKENS", true),
                            kumo_entry(json!(tss.max_tokens), true, dv, total),
                        );
                        map.insert(
                            kumo_key(name, sk, "STRING_AVG_NON_CHAR", true),
                            kumo_entry(json!(tss.non_char_fraction), true, dv, total),
                        );
                        map.insert(
                            kumo_key(name, sk, "STRING_GLOVE_OVERLAP", true),
                            kumo_entry(json!(tss.glove_overlap), true, dv, total),
                        );
                    }
                    let num_unique = hll.estimate() as u64;
                    let num_unique = if num_unique > total {
                        total
                    } else {
                        num_unique
                    };
                    map.insert(
                        kumo_key(name, sk, "NUM_UNIQUE", false),
                        kumo_entry(json!(num_unique), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "UNIQUE_FRACTION", false),
                        kumo_entry(json!(kumo_safe_div(num_unique, total)), false, dv, total),
                    );
                }
                ColumnStatsCollector::Timestamp {
                    count: c,
                    null_count,
                    parse_fail_count,
                    min_secs,
                    max_secs,
                    hll,
                    ..
                } => {
                    let total = (*c).saturating_add(*null_count);
                    map.insert(
                        kumo_key(name, sk, "COUNT", false),
                        kumo_entry(json!(total), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "NUM_NA", false),
                        kumo_entry(json!(*null_count), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "NA_FRACTION", false),
                        kumo_entry(json!(kumo_safe_div(*null_count, total)), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "INVALID_FRACTION", false),
                        kumo_entry(json!(0.0), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "DROPPED_ROWS", false),
                        kumo_entry(json!(*parse_fail_count), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "MIN", false),
                        kumo_entry(json!(format_kumo_timestamp(*min_secs)), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "MAX", false),
                        kumo_entry(json!(format_kumo_timestamp(*max_secs)), false, dv, total),
                    );
                    let num_unique = (hll.estimate() as u64).min(total);
                    map.insert(
                        kumo_key(name, sk, "NUM_UNIQUE", false),
                        kumo_entry(json!(num_unique), false, dv, total),
                    );
                }
                ColumnStatsCollector::Sequence {
                    count: c,
                    null_count,
                    min_len,
                    max_len,
                    ..
                } => {
                    let total = *c + *null_count;
                    map.insert(
                        kumo_key(name, sk, "COUNT", false),
                        kumo_entry(json!(total), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "NUM_NA", false),
                        kumo_entry(json!(*null_count), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "NA_FRACTION", false),
                        kumo_entry(json!(kumo_safe_div(*null_count, total)), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "INVALID_FRACTION", false),
                        kumo_entry(json!(0.0), false, dv, total),
                    );
                    // Always emit min/max length (Python bridge emits even sentinels).
                    map.insert(
                        kumo_key(name, sk, "SEQUENCE_MIN_LENGTH", false),
                        kumo_entry(json!(*min_len), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "SEQUENCE_MAX_LENGTH", false),
                        kumo_entry(json!(*max_len), false, dv, total),
                    );
                }
                ColumnStatsCollector::ImageUrl { count, null_count } => {
                    let total = *count + *null_count;
                    map.insert(
                        kumo_key(name, sk, "COUNT", false),
                        kumo_entry(json!(total), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "NUM_NA", false),
                        kumo_entry(json!(*null_count), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "NA_FRACTION", false),
                        kumo_entry(json!(kumo_safe_div(*null_count, total)), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "INVALID_FRACTION", false),
                        kumo_entry(json!(0.0), false, dv, total),
                    );
                }
                ColumnStatsCollector::Multicategorical {
                    count,
                    null_count,
                    hll,
                    separator,
                    ..
                } => {
                    // Sort category counts count-desc then token-asc to
                    // byte-match tt_executor's train-table output.
                    let mut sorted_cats: Vec<(String, u64)> = cat_counts[col_idx]
                        .as_ref()
                        .map(|cc| cc.iter().map(|(k, v)| (k.clone(), *v)).collect())
                        .unwrap_or_default();
                    sorted_cats.sort_by(|a, b| {
                        b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0))
                    });
                    let pairs = build_multicat_emit_stats(
                        name,
                        sk,
                        dv,
                        *count,
                        *null_count,
                        hll,
                        separator.as_deref(),
                        &sorted_cats,
                    );
                    for (key, value) in pairs {
                        map.insert(key, value);
                    }
                }
                ColumnStatsCollector::Skip => {}
            }
        }
        map
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
    /// Gated by per-row subsampling; see `observe_category_string_inner` for
    /// the sample-check-bypass variant used from the multicategorical path
    /// (row-level sampling is enforced once at the top of the row observer,
    /// not per token — see §3.1 of the multicat design).
    #[inline]
    fn observe_category_string(&mut self, col_idx: usize, hash: u64, value: &str) {
        if !self.should_sample_row() {
            return;
        }
        self.observe_category_string_inner(col_idx, hash, value);
    }

    /// Sample-check-bypass variant of `observe_category_string`. The caller is
    /// responsible for doing the row-level sample check exactly once per row
    /// before the per-token loop.
    #[inline]
    fn observe_category_string_inner(&mut self, col_idx: usize, hash: u64, value: &str) {
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
    /// Gated by per-row subsampling; see `observe_category_i64_inner`.
    #[inline]
    fn observe_category_i64(&mut self, col_idx: usize, hash: u64, value: i64) {
        if !self.should_sample_row() {
            return;
        }
        self.observe_category_i64_inner(col_idx, hash, value);
    }

    /// Sample-check-bypass variant of `observe_category_i64`. See
    /// `observe_category_string_inner` for rationale.
    #[inline]
    fn observe_category_i64_inner(&mut self, col_idx: usize, hash: u64, value: i64) {
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
                                        col_shared.record_new_unique();
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

impl FileColStats {
    /// Multicategorical `SingleStringToken` row observer. Called from
    /// `observe_string` when the collector's `input_kind` is
    /// `SingleStringToken` — i.e. a `(stype=multicategorical,
    /// IngestedType::String)` column with no `multicategorical_delimiter`
    /// declared. Each cell contributes exactly one token (the trimmed
    /// cell value); HLL observes it for `NUM_UNIQUE_MULTI` and category
    /// counts honor row-level subsampling. Nulls take the `observe_null`
    /// path upstream; this method is only entered for non-null rows.
    ///
    /// Pre-#28478 this method also handled the `(String, delim)` shape by
    /// splitting on the configured delimiter inline. After the
    /// normalize-at-ingest flip such columns become `IngestedType::StringArray`
    /// at convert time and dispatch through `observe_string_list`, leaving
    /// only the single-token-per-row case here.
    #[inline]
    fn observe_multicat_single_string_token(&mut self, col_idx: usize, raw: &str) {
        if let ColumnStatsCollector::Multicategorical {
            count,
            input_kind: MulticatInputKind::SingleStringToken,
            ..
        } = &mut self.cols[col_idx]
        {
            *count += 1;
        } else {
            return;
        }
        let count_category = self.should_sample_row();
        self.observe_multicat_token_str(col_idx, raw.trim(), count_category);
    }

    /// Per-token string primitive for Multicategorical columns. Updates the
    /// per-column HLL (for `NUM_UNIQUE_MULTI`) for every token. If
    /// `count_category` is true, it also feeds the hash into the shared
    /// category counts through `observe_category_string_inner`.
    ///
    /// Hot path — shares hashing + HLL + category-count infra with the
    /// Categorical arm of `observe_string`. `bin/multicat_microbench`
    /// pins the per-token cost; keep it ≈ Categorical observe.
    #[inline]
    fn observe_multicat_token_str(&mut self, col_idx: usize, tok: &str, count_category: bool) {
        let hash = HyperLogLog::hash_bytes(tok.as_bytes());
        if let ColumnStatsCollector::Multicategorical { hll, .. } = &mut self.cols[col_idx] {
            hll.observe_hash(hash);
        }
        if count_category {
            self.observe_category_string_inner(col_idx, hash, tok);
        }
    }

    /// Per-token i64 primitive for Multicategorical columns (Int64Array
    /// input kind). Identity-hashed via `HyperLogLog::hash_i64`, matching
    /// the integer categorical path.
    #[inline]
    fn observe_multicat_token_i64(&mut self, col_idx: usize, n: i64, count_category: bool) {
        let hash = HyperLogLog::hash_i64(n);
        if let ColumnStatsCollector::Multicategorical { hll, .. } = &mut self.cols[col_idx] {
            hll.observe_hash(hash);
        }
        if count_category {
            self.observe_category_i64_inner(col_idx, hash, n);
        }
    }
}

// Hot path. Per-token cost dominates encode time for Multicategorical and
// large Categorical/Id columns. The Multicat per-token primitives
// (`observe_multicat_token_str`, `observe_multicat_token_i64`) share their
// hashing + HLL + category-count infrastructure with the Categorical/Id arms
// of `observe_string` / `observe_i64`. If you change either path, rerun
// `cargo run --release --bin multicat_microbench` and confirm
//   C2 - C3 ≈ C1   (multicat token observe ≈ Categorical observe)
// stays within ~10%. Drift here amplifies linearly across hundreds of
// millions of tokens on list-heavy fixtures.
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
            ColumnStatsCollector::Timestamp {
                count,
                min_secs,
                max_secs,
                hll,
                ..
            } => {
                *count += 1;
                if value < *min_secs {
                    *min_secs = value;
                }
                if value > *max_secs {
                    *max_secs = value;
                }
                hll.observe_i64(value);
            }
            _ => {}
        }

        // Category counts — after releasing the borrow on self.cols
        if let Some(hash) = cat_hash {
            self.observe_category_i64(col_idx, hash, value);
        }
    }

    #[inline]
    fn observe_f64(&mut self, col_idx: usize, value: f64) {
        // Pre-compute hash for categorical/id (shared between HLL and category counts).
        let cat_hash = match &self.cols[col_idx] {
            ColumnStatsCollector::Categorical { .. } | ColumnStatsCollector::Id { .. } => {
                Some(HyperLogLog::hash_f64(value))
            }
            _ => None,
        };

        match &mut self.cols[col_idx] {
            ColumnStatsCollector::Numerical {
                invalid_count,
                welford,
                hll,
                tdigest,
                ..
            } => {
                if value.is_finite() {
                    welford.observe(value);
                    hll.observe_f64(value);
                    tdigest.observe(value);
                } else {
                    *invalid_count += 1;
                }
            }
            ColumnStatsCollector::Categorical { count, hll, .. } => {
                *count += 1;
                hll.observe_hash(cat_hash.unwrap());
            }
            ColumnStatsCollector::Id { count, hll, .. } => {
                *count += 1;
                hll.observe_hash(cat_hash.unwrap());
            }
            _ => {}
        }

        // Category counts — after releasing the borrow on self.cols
        if let Some(hash) = cat_hash {
            let str_value = value.to_string();
            self.observe_category_string(col_idx, hash, &str_value);
        }
    }

    #[inline]
    fn observe_string(&mut self, col_idx: usize, value: &str) {
        // Multicategorical scalar-string path: one row → N tokens. Dispatch
        // early so the rest of this method keeps its categorical/id/text
        // one-token-per-row shape.
        if matches!(
            &self.cols[col_idx],
            ColumnStatsCollector::Multicategorical {
                input_kind: MulticatInputKind::SingleStringToken,
                ..
            }
        ) {
            self.observe_multicat_single_string_token(col_idx, value);
            return;
        }

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
            ColumnStatsCollector::ImageUrl { count, .. } => {
                *count += 1;
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
            hll,
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
            hll.observe_i64(secs);
        }
    }

    #[inline]
    fn observe_float_list(&mut self, col_idx: usize, view: &FloatListView) {
        match &mut self.cols[col_idx] {
            ColumnStatsCollector::Sequence {
                count,
                min_len,
                max_len,
                welford,
                ..
            } => {
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
            ColumnStatsCollector::Multicategorical {
                input_kind: MulticatInputKind::FloatArray,
                count,
                ..
            } => {
                *count += 1;
                // Fall through to the per-token loop below. Releasing the &mut
                // borrow on self.cols first.
                let count_category = self.should_sample_row();
                // Per-token loop: widen to f64, drop NaN, stringify via Display.
                // HLL sees every token; category counts are gated once per row.
                // `format!("{}", f)` is the unavoidable per-element allocation;
                // graduated tokens only see this cost the first time they
                // appear (and only for unique new token strings).
                view.for_each_non_null_f64(|f| {
                    if f.is_nan() {
                        return;
                    }
                    let s = format!("{}", f);
                    self.observe_multicat_token_str(col_idx, &s, count_category);
                });
            }
            _ => {}
        }
    }

    #[inline]
    fn observe_int64_list(&mut self, col_idx: usize, view: &Int64ListView) {
        if matches!(
            &self.cols[col_idx],
            ColumnStatsCollector::Multicategorical {
                input_kind: MulticatInputKind::Int64Array,
                ..
            }
        ) {
            if let ColumnStatsCollector::Multicategorical { count, .. } = &mut self.cols[col_idx] {
                *count += 1;
            }
            let count_category = self.should_sample_row();
            view.for_each_i64(|n| {
                // Integer tokens are identity-hashed via the same primitive
                // categorical/id integer columns use; strings stored in
                // `LocalColumnCounts::strings` are `n.to_string()`, matching
                // tt_executor's stringified-int emit.
                self.observe_multicat_token_i64(col_idx, n, count_category);
            });
        }
    }

    #[inline]
    fn observe_string_list(&mut self, col_idx: usize, view: &StringListView) {
        if matches!(
            &self.cols[col_idx],
            ColumnStatsCollector::Multicategorical {
                input_kind: MulticatInputKind::StringArray,
                ..
            }
        ) {
            if let ColumnStatsCollector::Multicategorical { count, .. } = &mut self.cols[col_idx] {
                *count += 1;
            }
            let count_category = self.should_sample_row();
            // `view.for_each_str` drops null elements; legitimate empty strings
            // still reach the callback and remain empty tokens after trim.
            // `trim()` is a borrowed subslice — zero heap activity.
            view.for_each_str(|s| {
                self.observe_multicat_token_str(col_idx, s.trim(), count_category);
            });
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
            ColumnStatsCollector::ImageUrl { null_count, .. } => *null_count += 1,
            ColumnStatsCollector::Multicategorical { null_count, .. } => *null_count += 1,
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

    /// Build a minimal `Vec<ColumnSpec>` for tests that only care about
    /// `semantic_type`. All columns get `IngestedType::String` and no
    /// multicat delimiter; callers that need a specific ingested type
    /// should use `make_test_cols_with_types` instead.
    fn make_test_cols(stypes: &[Option<SemanticType>]) -> Vec<ColumnSpec> {
        stypes
            .iter()
            .enumerate()
            .map(|(i, s)| ColumnSpec {
                name: format!("c{i}"),
                source_dtype: None,
                ingested_dtype: IngestedType::String,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                precision: None,
                scale: None,
                semantic_type: *s,
                multicategorical_delimiter: None,
            })
            .collect()
    }

    /// Like `make_test_cols`, but lets the caller pick per-column
    /// `ingested_dtype` and `multicategorical_delimiter`.
    fn make_test_cols_with_types(
        entries: &[(Option<SemanticType>, IngestedType, Option<&str>)],
    ) -> Vec<ColumnSpec> {
        entries
            .iter()
            .enumerate()
            .map(|(i, (stype, dtype, delim))| ColumnSpec {
                name: format!("c{i}"),
                source_dtype: None,
                ingested_dtype: dtype.clone(),
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                precision: None,
                scale: None,
                semantic_type: *stype,
                multicategorical_delimiter: delim.map(str::to_string),
            })
            .collect()
    }

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
    fn test_shared_counts_cap_policy() {
        let capped = SharedColumnCounts::new_hashed();
        capped.unique_count.store(MAX_CATEGORIES, Ordering::Relaxed);
        capped.record_new_unique();
        assert!(capped.is_aborted());
        assert!(capped.drain_to_map().is_none());

        let uncapped = SharedColumnCounts::new_hashed_uncapped();
        uncapped
            .unique_count
            .store(MAX_CATEGORIES, Ordering::Relaxed);
        uncapped.record_new_unique();
        assert!(!uncapped.is_aborted());
        uncapped.observe_hashed(HyperLogLog::hash_bytes(b"a"), "a");
        uncapped.observe_hashed(HyperLogLog::hash_bytes(b"b"), "b");
        uncapped.observe_hashed(HyperLogLog::hash_bytes(b"c"), "c");
        let counts = uncapped.drain_to_map().unwrap();
        assert_eq!(counts.len(), 3);
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

    // -----------------------------------------------------------------------
    // Kumo-format stats tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_kumo_timestamp_normal() {
        assert_eq!(
            format_kumo_timestamp(1537401600),
            "kumo_timestamp_2018-09-20 00:00:00"
        );
    }

    #[test]
    fn test_format_kumo_timestamp_sentinels() {
        assert_eq!(format_kumo_timestamp(i64::MAX), "kumo_timestamp_NaT");
        assert_eq!(format_kumo_timestamp(i64::MIN), "kumo_timestamp_NaT");
    }

    #[test]
    fn test_to_kumo_json_numerical() {
        let stypes = vec![Some(SemanticType::Numerical)];
        let mut stats = FileColStats::new(&make_test_cols(&stypes));
        stats.observe_f64(0, 10.0);
        stats.observe_f64(0, 20.0);
        stats.observe_null(0);

        let col_names = vec!["age".to_string()];
        let ingested_types = vec![IngestedType::Float64];
        let map = stats.to_kumo_json(&col_names, &ingested_types, None, None, "v1");

        // Check base stats
        let count_entry = map.get("age__kumo.numerical__COUNT__False").unwrap();
        assert_eq!(count_entry["value"], 3);
        assert_eq!(count_entry["data_version"], "v1");
        assert_eq!(count_entry["num_entries"], 3);

        let na_entry = map.get("age__kumo.numerical__NUM_NA__False").unwrap();
        assert_eq!(na_entry["value"], 1);

        // Check numerical-specific stats
        assert!(map.contains_key("age__kumo.numerical__MEAN__False"));
        assert!(map.contains_key("age__kumo.numerical__STD__False"));
        assert!(map.contains_key("age__kumo.numerical__MIN__False"));
        assert!(map.contains_key("age__kumo.numerical__MAX__False"));
        assert!(map.contains_key("age__kumo.numerical__QUANTILES__False"));
        assert!(map.contains_key("age__kumo.numerical__KURTOSIS__True"));
        assert!(map.contains_key("age__kumo.numerical__NUM_UNIQUE__False"));
        assert!(map.contains_key("age__kumo.numerical__UNIQUE_FRACTION__False"));

        let mean = map.get("age__kumo.numerical__MEAN__False").unwrap();
        assert_eq!(mean["value"], 15.0);
    }

    #[test]
    fn test_to_kumo_json_timestamp() {
        let stypes = vec![Some(SemanticType::Timestamp)];
        let mut stats = FileColStats::new(&make_test_cols(&stypes));
        stats.observe_timestamp(0, 1537401600); // 2018-09-20
        stats.observe_timestamp(0, 1700000000); // 2023-11-14
        stats.observe_null(0);

        let col_names = vec!["ts".to_string()];
        let ingested_types = vec![IngestedType::Timestamp];
        let map = stats.to_kumo_json(&col_names, &ingested_types, None, None, "v2");

        let min_entry = map.get("ts__kumo.timestamp__MIN__False").unwrap();
        assert_eq!(min_entry["value"], "kumo_timestamp_2018-09-20 00:00:00");

        let max_entry = map.get("ts__kumo.timestamp__MAX__False").unwrap();
        assert_eq!(max_entry["value"], "kumo_timestamp_2023-11-14 22:13:20");

        assert!(map.contains_key("ts__kumo.timestamp__DROPPED_ROWS__False"));
        assert!(map.contains_key("ts__kumo.timestamp__NUM_UNIQUE__False"));
    }

    #[test]
    fn test_to_kumo_json_timestamp_sentinel() {
        let stypes = vec![Some(SemanticType::Timestamp)];
        let mut stats = FileColStats::new(&make_test_cols(&stypes));
        // No valid timestamps observed → min=MAX, max=MIN (sentinels)
        stats.observe_null(0);

        let col_names = vec!["ts".to_string()];
        let ingested_types = vec![IngestedType::Timestamp];
        let map = stats.to_kumo_json(&col_names, &ingested_types, None, None, "v1");

        let min_entry = map.get("ts__kumo.timestamp__MIN__False").unwrap();
        assert_eq!(min_entry["value"], "kumo_timestamp_NaT");
        let max_entry = map.get("ts__kumo.timestamp__MAX__False").unwrap();
        assert_eq!(max_entry["value"], "kumo_timestamp_NaT");
    }

    #[test]
    fn timestamp_parse_failure_summary_does_not_double_count_failed_rows() {
        let summary = TimestampParseFailureSummary::new(0, 3, 3).unwrap();
        assert_eq!(summary.total_rows, 3);
        assert_eq!(summary.source_non_null_rows, 3);
        assert_eq!(summary.source_null_rows, 0);
        assert_eq!(summary.parse_failure_pct_of_non_null(), 100.0);
        assert_eq!(summary.parse_failure_pct_of_total(), 100.0);

        let mixed = TimestampParseFailureSummary::new(2, 3, 1).unwrap();
        assert_eq!(mixed.total_rows, 5);
        assert_eq!(mixed.source_non_null_rows, 3);
        assert_eq!(mixed.source_null_rows, 2);
        assert!((mixed.parse_failure_pct_of_non_null() - 33.333333).abs() < 0.000001);
        assert_eq!(mixed.parse_failure_pct_of_total(), 20.0);
    }

    #[test]
    fn test_to_kumo_json_skip_column() {
        let stypes = vec![Some(SemanticType::Unsupported)];
        let mut stats = FileColStats::new(&make_test_cols(&stypes));
        let col_names = vec!["x".to_string()];
        let ingested_types = vec![IngestedType::String];
        let map = stats.to_kumo_json(&col_names, &ingested_types, None, None, "v1");
        assert!(map.is_empty());
    }

    /// Normalize a kumo stats map to a deterministic JSON string for snapshot
    /// comparison.  Category count arrays are sorted by stringified key so that
    /// HashMap iteration order doesn't matter.
    fn canonicalize_kumo_map(map: &serde_json::Map<String, JsonValue>) -> String {
        let mut sorted: Vec<(&String, &JsonValue)> = map.iter().collect();
        sorted.sort_by_key(|(k, _)| (*k).clone());

        let entries: Vec<String> = sorted
            .into_iter()
            .map(|(k, v)| {
                let mut val = v.clone();
                // Sort CATEGORY_COUNTS array by stringified first element
                if k.contains("CATEGORY_COUNTS") {
                    if let Some(obj) = val.as_object_mut() {
                        if let Some(arr) = obj.get_mut("value").and_then(|v| v.as_array_mut()) {
                            arr.sort_by(|a, b| {
                                let ak = a.get(0).map(|v| v.to_string()).unwrap_or_default();
                                let bk = b.get(0).map(|v| v.to_string()).unwrap_or_default();
                                ak.cmp(&bk)
                            });
                        }
                    }
                }
                format!(
                    "{}: {}",
                    serde_json::to_string(k).unwrap(),
                    serde_json::to_string(&val).unwrap()
                )
            })
            .collect();
        format!("{{\n{}\n}}", entries.join(",\n"))
    }

    /// Exact-output snapshot test for to_kumo_json().
    ///
    /// Exercises all 6 supported column types plus unsupported/skip.
    /// Every key and value is asserted against a hardcoded expected JSON.
    /// Category count arrays are sorted by key for determinism.
    ///
    /// If the Rust stats output changes, this test will fail — update the
    /// snapshot after verifying the new output is correct.
    #[test]
    fn test_to_kumo_json_exact_snapshot() {
        let stypes = vec![
            Some(SemanticType::Numerical),   // 0: "age"
            Some(SemanticType::Categorical), // 1: "status"
            Some(SemanticType::Id),          // 2: "user_id"
            Some(SemanticType::Text),        // 3: "bio"
            Some(SemanticType::Timestamp),   // 4: "created_at"
            Some(SemanticType::Sequence),    // 5: "embedding"
            Some(SemanticType::Unsupported), // 6: "skip_me"
        ];
        let mut stats = FileColStats::new(&make_test_cols(&stypes));

        // Numerical: 10, 20, 30 + 1 null → count=4
        stats.observe_f64(0, 10.0);
        stats.observe_f64(0, 20.0);
        stats.observe_f64(0, 30.0);
        stats.observe_null(0);
        // Categorical: "active"×2, "inactive"×1, 1 null → count=4
        stats.observe_string(1, "active");
        stats.observe_string(1, "inactive");
        stats.observe_string(1, "active");
        stats.observe_null(1);
        // Id: "u1", "u2" → count=2
        stats.observe_string(2, "u1");
        stats.observe_string(2, "u2");
        // Text: 2 values, 1 null → count=3
        stats.observe_string(3, "hello world");
        stats.observe_string(3, "foo bar baz");
        stats.observe_null(3);
        // Timestamp: 2 values, 1 source null, 1 parse fail encoded as null → count=4
        stats.observe_timestamp(4, 1537401600);
        stats.observe_timestamp(4, 1700000000);
        stats.observe_null(4);
        stats.observe_parse_failure(4);
        stats.observe_null(4);
        // Sequence: 2 nulls only → count=2
        stats.observe_null(5);
        stats.observe_null(5);

        let col_names: Vec<String> = [
            "age",
            "status",
            "user_id",
            "bio",
            "created_at",
            "embedding",
            "skip_me",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let ingested_types = vec![
            IngestedType::Float64,
            IngestedType::String,
            IngestedType::String,
            IngestedType::String,
            IngestedType::Timestamp,
            IngestedType::Float32Array,
            IngestedType::String,
        ];
        let mut text_stats = HashMap::new();
        text_stats.insert(
            "bio".to_string(),
            TextSampleStats {
                avg_tokens: 2.5,
                max_tokens: 3,
                non_char_fraction: 0.0,
                string_truncation_scale: 1.0,
                glove_overlap: 0.8,
            },
        );

        let map = stats.to_kumo_json(&col_names, &ingested_types, None, Some(&text_stats), "v42");
        let canonical = canonicalize_kumo_map(&map);

        // ---- Assert exact key set ----
        let keys: Vec<&String> = {
            let mut v: Vec<_> = map.keys().collect();
            v.sort();
            v
        };
        assert!(
            !keys.iter().any(|k| k.starts_with("skip_me__")),
            "Unsupported column must not appear"
        );

        // ---- Assert every entry exactly ----
        // Helper: assert a single stat entry's 4 fields
        fn e(
            map: &serde_json::Map<String, JsonValue>,
            key: &str,
            val: JsonValue,
            approx: bool,
            dv: &str,
            n: u64,
        ) {
            let entry = map.get(key).unwrap_or_else(|| panic!("missing key: {key}"));
            let obj = entry.as_object().unwrap();
            assert_eq!(obj["approx"], json!(approx), "approx @ {key}");
            assert_eq!(obj["data_version"], json!(dv), "data_version @ {key}");
            assert_eq!(obj["num_entries"], json!(n), "num_entries @ {key}");
            // For floats allow small tolerance
            let actual = &obj["value"];
            match (&val, actual) {
                (JsonValue::Number(expected), JsonValue::Number(got)) => {
                    let ef = expected.as_f64().unwrap();
                    let gf = got.as_f64().unwrap();
                    assert!(
                        (ef - gf).abs() < 1e-9,
                        "value @ {key}: expected {ef}, got {gf}"
                    );
                }
                _ => assert_eq!(*actual, val, "value @ {key}"),
            }
        }

        // ---- Numerical: age ----
        // 3 observations + 1 null = count 4
        // mean=20, std=10, min=10, max=30, kurtosis=-1.5
        // With only 3 data points, tdigest quantiles = [10, 20, 30]
        // HLL with 3 distinct f64s → num_unique 3 (exact for small n)
        e(
            &map,
            "age__kumo.numerical__COUNT__False",
            json!(4),
            false,
            "v42",
            4,
        );
        e(
            &map,
            "age__kumo.numerical__NUM_NA__False",
            json!(1),
            false,
            "v42",
            4,
        );
        e(
            &map,
            "age__kumo.numerical__NA_FRACTION__False",
            json!(0.25),
            false,
            "v42",
            4,
        );
        e(
            &map,
            "age__kumo.numerical__INVALID_FRACTION__False",
            json!(0.0),
            false,
            "v42",
            4,
        );
        e(
            &map,
            "age__kumo.numerical__MEAN__False",
            json!(20.0),
            false,
            "v42",
            4,
        );
        e(
            &map,
            "age__kumo.numerical__STD__False",
            json!(10.0),
            false,
            "v42",
            4,
        );
        e(
            &map,
            "age__kumo.numerical__MIN__False",
            json!(10.0),
            false,
            "v42",
            4,
        );
        e(
            &map,
            "age__kumo.numerical__MAX__False",
            json!(30.0),
            false,
            "v42",
            4,
        );
        e(
            &map,
            "age__kumo.numerical__KURTOSIS__False",
            json!(-1.5),
            false,
            "v42",
            4,
        );
        e(
            &map,
            "age__kumo.numerical__KURTOSIS__True",
            json!(-1.5),
            true,
            "v42",
            4,
        );
        // Quantiles are TDigest approximations — assert structure and reasonable range
        for qkey in [
            "age__kumo.numerical__QUANTILE25__False",
            "age__kumo.numerical__MEDIAN__False",
            "age__kumo.numerical__QUANTILE75__False",
            "age__kumo.numerical__MEDIAN__True",
        ] {
            let entry = map.get(qkey).unwrap_or_else(|| panic!("missing {qkey}"));
            let v = entry["value"].as_f64().unwrap();
            assert!(
                (10.0..=30.0).contains(&v),
                "{qkey} value {v} out of [10,30]"
            );
            assert_eq!(entry["data_version"], json!("v42"));
        }
        {
            let q = map.get("age__kumo.numerical__QUANTILES__False").unwrap();
            let arr = q["value"].as_array().unwrap();
            assert_eq!(arr.len(), 3);
            for v in arr {
                let f = v.as_f64().unwrap();
                assert!((10.0..=30.0).contains(&f), "quantile {f} out of range");
            }
            // Monotonicity: q25 <= median <= q75
            assert!(arr[0].as_f64().unwrap() <= arr[1].as_f64().unwrap());
            assert!(arr[1].as_f64().unwrap() <= arr[2].as_f64().unwrap());
        }
        // NUM_UNIQUE: HLL exact for 3 values
        {
            let nu = map.get("age__kumo.numerical__NUM_UNIQUE__False").unwrap();
            let v = nu["value"].as_u64().unwrap();
            assert!((2..=4).contains(&v), "NUM_UNIQUE should be ~3, got {v}");
        }
        {
            let uf = map
                .get("age__kumo.numerical__UNIQUE_FRACTION__False")
                .unwrap();
            let v = uf["value"].as_f64().unwrap();
            assert!(
                v > 0.0 && v <= 1.0,
                "UNIQUE_FRACTION should be in (0,1], got {v}"
            );
        }

        // ---- Categorical: status ----
        // 3 observations + 1 null = count 4, no cat counts (strategy=None)
        e(
            &map,
            "status__kumo.categorical__COUNT__False",
            json!(4),
            false,
            "v42",
            4,
        );
        e(
            &map,
            "status__kumo.categorical__NUM_NA__False",
            json!(1),
            false,
            "v42",
            4,
        );
        e(
            &map,
            "status__kumo.categorical__NA_FRACTION__False",
            json!(0.25),
            false,
            "v42",
            4,
        );
        e(
            &map,
            "status__kumo.categorical__INVALID_FRACTION__False",
            json!(0.0),
            false,
            "v42",
            4,
        );
        e(
            &map,
            "status__kumo.categorical__CATEGORY_COUNTS__False",
            json!([]),
            false,
            "v42",
            4,
        );
        {
            let nu = map
                .get("status__kumo.categorical__NUM_UNIQUE__False")
                .unwrap();
            let v = nu["value"].as_u64().unwrap();
            assert!(
                (1..=4).contains(&v),
                "status NUM_UNIQUE should be ~2, got {v}"
            );
        }

        // ---- Id: user_id ----
        // 2 observations, 0 null = count 2
        e(
            &map,
            "user_id__kumo.ID__COUNT__False",
            json!(2),
            false,
            "v42",
            2,
        );
        e(
            &map,
            "user_id__kumo.ID__NUM_NA__False",
            json!(0),
            false,
            "v42",
            2,
        );
        e(
            &map,
            "user_id__kumo.ID__NA_FRACTION__False",
            json!(0.0),
            false,
            "v42",
            2,
        );
        e(
            &map,
            "user_id__kumo.ID__INVALID_FRACTION__False",
            json!(0.0),
            false,
            "v42",
            2,
        );
        e(
            &map,
            "user_id__kumo.ID__CATEGORY_COUNTS__False",
            json!([]),
            false,
            "v42",
            2,
        );
        {
            let nu = map.get("user_id__kumo.ID__NUM_UNIQUE__False").unwrap();
            let v = nu["value"].as_u64().unwrap();
            assert!(
                (1..=2).contains(&v),
                "user_id NUM_UNIQUE should be ~2, got {v}"
            );
        }

        // ---- Text: bio ----
        // 2 observations + 1 null = count 3
        e(
            &map,
            "bio__kumo.text__COUNT__False",
            json!(3),
            false,
            "v42",
            3,
        );
        e(
            &map,
            "bio__kumo.text__NUM_NA__False",
            json!(1),
            false,
            "v42",
            3,
        );
        e(
            &map,
            "bio__kumo.text__NA_FRACTION__False",
            json!(1.0 / 3.0),
            false,
            "v42",
            3,
        );
        e(
            &map,
            "bio__kumo.text__INVALID_FRACTION__False",
            json!(0.0),
            false,
            "v42",
            3,
        );
        e(
            &map,
            "bio__kumo.text__STRING_AVG_TOKENS__True",
            json!(2.5),
            true,
            "v42",
            3,
        );
        e(
            &map,
            "bio__kumo.text__STRING_MAX_TOKENS__True",
            json!(3),
            true,
            "v42",
            3,
        );
        e(
            &map,
            "bio__kumo.text__STRING_AVG_NON_CHAR__True",
            json!(0.0),
            true,
            "v42",
            3,
        );
        e(
            &map,
            "bio__kumo.text__STRING_GLOVE_OVERLAP__True",
            json!(0.8),
            true,
            "v42",
            3,
        );
        {
            let nu = map.get("bio__kumo.text__NUM_UNIQUE__False").unwrap();
            let v = nu["value"].as_u64().unwrap();
            assert!((1..=3).contains(&v), "bio NUM_UNIQUE should be ~2, got {v}");
        }

        // ---- Timestamp: created_at ----
        // 2 observations + 1 source null + 1 parse failure encoded as null = count 4
        e(
            &map,
            "created_at__kumo.timestamp__COUNT__False",
            json!(4),
            false,
            "v42",
            4,
        );
        e(
            &map,
            "created_at__kumo.timestamp__NUM_NA__False",
            json!(2),
            false,
            "v42",
            4,
        );
        e(
            &map,
            "created_at__kumo.timestamp__NA_FRACTION__False",
            json!(0.5),
            false,
            "v42",
            4,
        );
        e(
            &map,
            "created_at__kumo.timestamp__INVALID_FRACTION__False",
            json!(0.0),
            false,
            "v42",
            4,
        );
        e(
            &map,
            "created_at__kumo.timestamp__DROPPED_ROWS__False",
            json!(1),
            false,
            "v42",
            4,
        );
        e(
            &map,
            "created_at__kumo.timestamp__MIN__False",
            json!("kumo_timestamp_2018-09-20 00:00:00"),
            false,
            "v42",
            4,
        );
        e(
            &map,
            "created_at__kumo.timestamp__MAX__False",
            json!("kumo_timestamp_2023-11-14 22:13:20"),
            false,
            "v42",
            4,
        );
        e(
            &map,
            "created_at__kumo.timestamp__NUM_UNIQUE__False",
            json!(2),
            false,
            "v42",
            4,
        );

        // ---- Sequence: embedding ----
        // 0 observations + 2 null = count 2
        e(
            &map,
            "embedding__kumo.sequence__COUNT__False",
            json!(2),
            false,
            "v42",
            2,
        );
        e(
            &map,
            "embedding__kumo.sequence__NUM_NA__False",
            json!(2),
            false,
            "v42",
            2,
        );
        e(
            &map,
            "embedding__kumo.sequence__NA_FRACTION__False",
            json!(1.0),
            false,
            "v42",
            2,
        );
        e(
            &map,
            "embedding__kumo.sequence__INVALID_FRACTION__False",
            json!(0.0),
            false,
            "v42",
            2,
        );
        // min_len=usize::MAX, max_len=0 (sentinel values from zero observations)
        {
            let min_l = map
                .get("embedding__kumo.sequence__SEQUENCE_MIN_LENGTH__False")
                .unwrap();
            assert_eq!(min_l["value"], json!(usize::MAX));
        }
        {
            let max_l = map
                .get("embedding__kumo.sequence__SEQUENCE_MAX_LENGTH__False")
                .unwrap();
            assert_eq!(max_l["value"], json!(0));
        }

        // ---- Total key count ----
        // numerical: 4 base + MEAN STD MIN MAX (4) + QUANTILES Q25 MEDIAN Q75 MEDIAN(approx) (5)
        //          + KURTOSIS(exact) KURTOSIS(approx) (2) + NUM_UNIQUE UNIQUE_FRACTION (2) = 17
        // categorical: 4 base + CATEGORY_COUNTS + NUM_UNIQUE + UNIQUE_FRACTION = 7
        // id: same = 7
        // text: 4 base + 4 text stats + NUM_UNIQUE + UNIQUE_FRACTION = 10
        // timestamp: 4 base + DROPPED_ROWS + MIN + MAX + NUM_UNIQUE = 8
        // sequence: 4 base + SEQ_MIN + SEQ_MAX = 6
        // Total = 17 + 7 + 7 + 10 + 8 + 6 = 55
        assert_eq!(
            map.len(),
            55,
            "Expected exactly 55 stat entries, got {}",
            map.len()
        );

        // ---- Verify canonical JSON is stable ----
        // Re-run and assert identical canonical output
        let canonical2 = canonicalize_kumo_map(&map);
        assert_eq!(
            canonical, canonical2,
            "Canonical output should be deterministic"
        );
    }

    #[test]
    fn test_minify_kumo_stats_truncates_category_counts() {
        let mut stats = serde_json::Map::new();
        let big_list: Vec<JsonValue> = (0..50).map(|i| json!([format!("cat_{i}"), i])).collect();
        stats.insert(
            "col__kumo.categorical__CATEGORY_COUNTS__False".to_string(),
            json!({"value": big_list, "approx": false, "data_version": "v1", "num_entries": 100}),
        );
        stats.insert(
            "col__kumo.categorical__COUNT__False".to_string(),
            json!({"value": 100, "approx": false, "data_version": "v1", "num_entries": 100}),
        );

        let minified = minify_kumo_stats(&stats, 20);

        // COUNT untouched
        let count = minified.get("col__kumo.categorical__COUNT__False").unwrap();
        assert_eq!(count["value"], 100);

        // CATEGORY_COUNTS truncated to 20
        let cc = minified
            .get("col__kumo.categorical__CATEGORY_COUNTS__False")
            .unwrap();
        assert_eq!(cc["value"].as_array().unwrap().len(), 20);
    }

    // =========================================================================
    // Multicategorical collector tests
    //
    // These exercise the §6.1 spec from the design doc at
    // `knowledge/execution-notes/28187/design_doc.md`. They bypass the
    // `config_converter` layer by constructing `FileColStats` directly via
    // `make_test_cols_with_types` with explicit `IngestedType` +
    // `multicategorical_delimiter`, so they test the collector's observation
    // and emit logic in isolation.
    //
    // Category-count assertions use `CategoryCountStrategy::Hybrid` so every
    // observed token ends up in `pending` (no graduation threshold to
    // approximate). Helper `drain_counts` pulls the exact per-token tallies.
    // =========================================================================

    /// Build a `FileColStats` with `CategoryCountStrategy::Hybrid` so per-token
    /// counts can be asserted exactly. Returns both the stats observer and
    /// the shared `TableCategoryCounts` (needed for the emit call and for
    /// draining counts post-observe).
    fn make_hybrid_stats(
        entries: &[(Option<SemanticType>, IngestedType, Option<&str>)],
    ) -> (FileColStats, std::sync::Arc<TableCategoryCounts>) {
        let cols = make_test_cols_with_types(entries);
        let stypes: Vec<Option<SemanticType>> = cols.iter().map(|c| c.semantic_type).collect();
        let table_cat = std::sync::Arc::new(TableCategoryCounts::new(
            CategoryCountStrategy::Hybrid,
            &stypes,
            false,
        ));
        let stats = FileColStats::new_with_categories(&cols, std::sync::Arc::clone(&table_cat));
        (stats, table_cat)
    }

    /// Drain per-column category counts for a single column index.
    fn drain_counts(
        table_cat: &TableCategoryCounts,
        col_idx: usize,
    ) -> std::collections::HashMap<String, u64> {
        table_cat.columns[col_idx]
            .as_ref()
            .and_then(|shared| shared.drain_to_map())
            .unwrap_or_default()
    }

    // --- SingleStringToken input kind --------------------------------------
    //
    // This input kind handles `(stype=multicategorical, IngestedType::String)`
    // columns with no `multicategorical_delimiter` declared — i.e. each cell
    // contributes exactly one token. Multicat columns that arrive with a
    // delimiter are normalized to `IngestedType::StringArray` at convert
    // time (kumo-ai/kumo#28478) and exercised through the `StringArray`
    // section below + the `ingest_multicategorical_e2e` integration tests.

    #[test]
    fn col_stats_multicat_single_string_token_each_cell_is_one_token() {
        // Declared `None` → each row's whole value is one token (after trim).
        let (mut stats, tc) = make_hybrid_stats(&[(
            Some(SemanticType::Multicategorical),
            IngestedType::String,
            None,
        )]);
        stats.observe_string(0, "hello world");
        stats.observe_string(0, "hello world");
        stats.observe_string(0, "foo");
        stats.flush_categories();
        let counts = drain_counts(&tc, 0);
        assert_eq!(counts.get("hello world").copied(), Some(2));
        assert_eq!(counts.get("foo").copied(), Some(1));
        assert_eq!(counts.len(), 2);
    }

    #[test]
    fn col_stats_multicat_single_string_token_null_row() {
        let (mut stats, tc) = make_hybrid_stats(&[(
            Some(SemanticType::Multicategorical),
            IngestedType::String,
            None,
        )]);
        stats.observe_null(0);
        stats.observe_string(0, "tag");
        stats.flush_categories();
        let counts = drain_counts(&tc, 0);
        assert_eq!(counts.get("tag").copied(), Some(1));
        assert_eq!(counts.len(), 1);
        match &stats.cols[0] {
            ColumnStatsCollector::Multicategorical {
                count, null_count, ..
            } => {
                assert_eq!(*count, 1);
                assert_eq!(*null_count, 1);
            }
            _ => panic!("expected Multicategorical collector"),
        }
    }

    // --- StringArray input kind --------------------------------------------

    #[test]
    fn col_stats_multicat_stringarray_tokens_as_is() {
        let view = build_string_list_view(&[Some("a"), Some("b")]);
        let view2 = build_string_list_view(&[Some("b"), Some("c"), Some("a")]);
        let (mut stats, tc) = make_hybrid_stats(&[(
            Some(SemanticType::Multicategorical),
            IngestedType::StringArray,
            None,
        )]);
        stats.observe_string_list(0, &view);
        stats.observe_string_list(0, &view2);
        stats.flush_categories();
        let counts = drain_counts(&tc, 0);
        assert_eq!(counts.get("a").copied(), Some(2));
        assert_eq!(counts.get("b").copied(), Some(2));
        assert_eq!(counts.get("c").copied(), Some(1));
        assert_eq!(counts.len(), 3);
    }

    #[test]
    fn col_stats_multicat_stringarray_empty_list_vs_null_row() {
        // Empty list → non-null row, zero tokens. Null row → null_count += 1.
        let (mut stats, tc) = make_hybrid_stats(&[(
            Some(SemanticType::Multicategorical),
            IngestedType::StringArray,
            None,
        )]);
        let empty = build_string_list_view(&[]);
        let x = build_string_list_view(&[Some("x")]);
        stats.observe_string_list(0, &empty);
        stats.observe_null(0);
        stats.observe_string_list(0, &x);
        stats.flush_categories();
        let counts = drain_counts(&tc, 0);
        assert_eq!(counts.get("x").copied(), Some(1));
        assert_eq!(counts.len(), 1);
        match &stats.cols[0] {
            ColumnStatsCollector::Multicategorical {
                count, null_count, ..
            } => {
                assert_eq!(*count, 2);
                assert_eq!(*null_count, 1);
            }
            _ => panic!("expected Multicategorical collector"),
        }
    }

    #[test]
    fn col_stats_multicat_stringarray_null_elements_dropped() {
        // Null list elements are dropped; legitimate empty-string elements
        // still count as empty tokens.
        let view = build_native_string_list_view(&[Some("a"), None, Some(""), Some("b")]);
        let (mut stats, tc) = make_hybrid_stats(&[(
            Some(SemanticType::Multicategorical),
            IngestedType::StringArray,
            None,
        )]);
        stats.observe_string_list(0, &view);
        stats.flush_categories();
        let counts = drain_counts(&tc, 0);
        assert_eq!(counts.get("a").copied(), Some(1));
        assert_eq!(counts.get("").copied(), Some(1));
        assert_eq!(counts.get("b").copied(), Some(1));
        assert_eq!(counts.len(), 3);
    }

    #[test]
    fn col_stats_multicat_stringarray_trim_applied() {
        // §4.1.1 policy: list elements also trim leading/trailing whitespace.
        let view = build_string_list_view(&[Some(""), Some(" b"), Some("c ")]);
        let (mut stats, tc) = make_hybrid_stats(&[(
            Some(SemanticType::Multicategorical),
            IngestedType::StringArray,
            None,
        )]);
        stats.observe_string_list(0, &view);
        stats.flush_categories();
        let counts = drain_counts(&tc, 0);
        assert_eq!(counts.get("").copied(), Some(1));
        assert_eq!(counts.get("b").copied(), Some(1));
        assert_eq!(counts.get("c").copied(), Some(1));
        assert_eq!(counts.len(), 3);
    }

    // --- Int64Array input kind ---------------------------------------------

    #[test]
    fn col_stats_multicat_intarray_tokens_stringified() {
        let v1 = build_int64_list_view(&[Some(10), Some(20)]);
        let v2 = build_int64_list_view(&[Some(10), Some(-3)]);
        let (mut stats, tc) = make_hybrid_stats(&[(
            Some(SemanticType::Multicategorical),
            IngestedType::Int64Array,
            None,
        )]);
        stats.observe_int64_list(0, &v1);
        stats.observe_int64_list(0, &v2);
        stats.flush_categories();
        let counts = drain_counts(&tc, 0);
        assert_eq!(counts.get("10").copied(), Some(2));
        assert_eq!(counts.get("20").copied(), Some(1));
        assert_eq!(counts.get("-3").copied(), Some(1));
        assert_eq!(counts.len(), 3);
    }

    #[test]
    fn col_stats_multicat_intarray_null_elements_dropped() {
        let view = build_native_int64_list_view(&[Some(5), None, Some(7)]);
        let (mut stats, tc) = make_hybrid_stats(&[(
            Some(SemanticType::Multicategorical),
            IngestedType::Int64Array,
            None,
        )]);
        stats.observe_int64_list(0, &view);
        stats.flush_categories();
        let counts = drain_counts(&tc, 0);
        assert_eq!(counts.get("5").copied(), Some(1));
        assert_eq!(counts.get("7").copied(), Some(1));
        assert_eq!(counts.len(), 2);
    }

    #[test]
    fn col_stats_multicat_intarray_negative_and_extremes() {
        let view = build_int64_list_view(&[Some(0), Some(i64::MIN), Some(i64::MAX)]);
        let (mut stats, tc) = make_hybrid_stats(&[(
            Some(SemanticType::Multicategorical),
            IngestedType::Int64Array,
            None,
        )]);
        stats.observe_int64_list(0, &view);
        stats.flush_categories();
        let counts = drain_counts(&tc, 0);
        assert_eq!(counts.get("0").copied(), Some(1));
        assert_eq!(counts.get("-9223372036854775808").copied(), Some(1));
        assert_eq!(counts.get("9223372036854775807").copied(), Some(1));
    }

    // --- FloatArray input kind ---------------------------------------------

    #[test]
    fn col_stats_multicat_floatarray_tokens_stringified() {
        let view = build_float64_list_view(&[Some(1.0), Some(2.5), Some(1.0)]);
        let (mut stats, tc) = make_hybrid_stats(&[(
            Some(SemanticType::Multicategorical),
            IngestedType::Float64Array,
            None,
        )]);
        stats.observe_float_list(0, &view);
        stats.flush_categories();
        let counts = drain_counts(&tc, 0);
        // Rust `Display`: 1.0 → "1", 2.5 → "2.5".
        assert_eq!(counts.get("1").copied(), Some(2));
        assert_eq!(counts.get("2.5").copied(), Some(1));
        assert_eq!(counts.len(), 2);
    }

    #[test]
    fn col_stats_multicat_floatarray_nan_dropped() {
        let view = build_float64_list_view(&[Some(f64::NAN), Some(1.0), Some(f64::NAN)]);
        let (mut stats, tc) = make_hybrid_stats(&[(
            Some(SemanticType::Multicategorical),
            IngestedType::Float64Array,
            None,
        )]);
        stats.observe_float_list(0, &view);
        stats.flush_categories();
        let counts = drain_counts(&tc, 0);
        assert_eq!(counts.get("1").copied(), Some(1));
        assert_eq!(counts.len(), 1);
        // The row itself is non-null; NaN elements don't bump null_count.
        match &stats.cols[0] {
            ColumnStatsCollector::Multicategorical {
                count, null_count, ..
            } => {
                assert_eq!(*count, 1);
                assert_eq!(*null_count, 0);
            }
            _ => panic!("expected Multicategorical collector"),
        }
    }

    #[test]
    fn col_stats_multicat_floatarray_null_elements_dropped() {
        let view = build_native_float64_list_view(&[Some(1.0), None, Some(2.5)]);
        let (mut stats, tc) = make_hybrid_stats(&[(
            Some(SemanticType::Multicategorical),
            IngestedType::Float64Array,
            None,
        )]);
        stats.observe_float_list(0, &view);
        stats.flush_categories();
        let counts = drain_counts(&tc, 0);
        assert_eq!(counts.get("1").copied(), Some(1));
        assert_eq!(counts.get("2.5").copied(), Some(1));
        assert!(!counts.contains_key("0"));
        assert_eq!(counts.len(), 2);
    }

    #[test]
    fn col_stats_multicat_floatarray_signed_zero_distinct() {
        let view = build_float64_list_view(&[Some(0.0), Some(-0.0)]);
        let (mut stats, tc) = make_hybrid_stats(&[(
            Some(SemanticType::Multicategorical),
            IngestedType::Float64Array,
            None,
        )]);
        stats.observe_float_list(0, &view);
        stats.flush_categories();
        let counts = drain_counts(&tc, 0);
        // Rust Display: -0.0 → "-0", 0.0 → "0".
        assert_eq!(counts.get("0").copied(), Some(1));
        assert_eq!(counts.get("-0").copied(), Some(1));
    }

    #[test]
    fn col_stats_multicat_floatarray_infinity() {
        let view = build_float64_list_view(&[Some(f64::INFINITY), Some(f64::NEG_INFINITY)]);
        let (mut stats, tc) = make_hybrid_stats(&[(
            Some(SemanticType::Multicategorical),
            IngestedType::Float64Array,
            None,
        )]);
        stats.observe_float_list(0, &view);
        stats.flush_categories();
        let counts = drain_counts(&tc, 0);
        assert_eq!(counts.get("inf").copied(), Some(1));
        assert_eq!(counts.get("-inf").copied(), Some(1));
    }

    // --- Merge -------------------------------------------------------------

    #[test]
    fn col_stats_multicat_merge_accumulates_row_counts() {
        // Merge only exercises the per-column `count`, `null_count`, and HLL;
        // token counts live in the shared `TableCategoryCounts` so they merge
        // transparently across per-file `FileColStats` instances.
        let cols = make_test_cols_with_types(&[(
            Some(SemanticType::Multicategorical),
            IngestedType::String,
            Some(","),
        )]);
        let mut a = FileColStats::new(&cols);
        let mut b = FileColStats::new(&cols);
        a.observe_string(0, "x,y");
        a.observe_null(0);
        b.observe_string(0, "z");
        a.merge(&mut b);
        match &a.cols[0] {
            ColumnStatsCollector::Multicategorical {
                count, null_count, ..
            } => {
                assert_eq!(*count, 2);
                assert_eq!(*null_count, 1);
            }
            _ => panic!("expected Multicategorical collector"),
        }
    }

    // --- Emit path --------------------------------------------------------

    #[test]
    fn col_stats_multicat_emit_keys_match_tt_executor_format() {
        // Post-#28478 the `(stype=multicategorical, dtype=String, delim)`
        // shape is normalized to `IngestedType::StringArray` at convert
        // time. The collector still records the original delimiter so the
        // emitted `MULTI_CATEGORIES_SEPARATOR` stat preserves the
        // tokenization documentation; observation flows through
        // `observe_string_list` from the access layer's batch-tokenized
        // list. Mirror that here: declare StringArray + delim, feed the
        // pre-tokenized list views directly.
        let (mut stats, tc) = make_hybrid_stats(&[(
            Some(SemanticType::Multicategorical),
            IngestedType::StringArray,
            Some(","),
        )]);
        let row0 = build_string_list_view(&[Some("kids"), Some("sale")]);
        let row1 = build_string_list_view(&[Some("kids")]);
        stats.observe_string_list(0, &row0);
        stats.observe_string_list(0, &row1);
        stats.observe_null(0);
        stats.flush_categories();

        let col_names = vec!["tags".to_string()];
        let ingested_types = vec![IngestedType::StringArray];
        let map = stats.to_kumo_json(&col_names, &ingested_types, Some(&tc), None, "v1");

        // The three keys tt_executor also emits for multicat stats.
        let sep = map
            .get("tags__kumo.multicategorical__MULTI_CATEGORIES_SEPARATOR__False")
            .expect("separator key present");
        assert_eq!(sep["value"], ",");

        let counts = map
            .get("tags__kumo.multicategorical__MULTI_CATEGORY_COUNTS__False")
            .expect("category-counts key present");
        let arr = counts["value"].as_array().expect("array");
        // Sorted count-desc, then token-asc: [["kids",2],["sale",1]].
        assert_eq!(arr[0][0], "kids");
        assert_eq!(arr[0][1], 2);
        assert_eq!(arr[1][0], "sale");
        assert_eq!(arr[1][1], 1);

        let uniq = map
            .get("tags__kumo.multicategorical__NUM_UNIQUE_MULTI__True")
            .expect("num-unique key present");
        // HLL on 2 distinct tokens — estimate is tight on tiny sets.
        assert_eq!(uniq["approx"], true);
        assert!(uniq["value"].as_u64().unwrap() >= 1);

        // Universal stats present with the kumo.multicategorical stype key.
        assert!(map.contains_key("tags__kumo.multicategorical__COUNT__False"));
        assert!(map.contains_key("tags__kumo.multicategorical__NUM_NA__False"));
    }

    #[test]
    fn col_stats_multicat_emit_stringarray_separator_is_null() {
        let view = build_string_list_view(&[Some("a"), Some("b")]);
        let (mut stats, tc) = make_hybrid_stats(&[(
            Some(SemanticType::Multicategorical),
            IngestedType::StringArray,
            None,
        )]);
        stats.observe_string_list(0, &view);
        stats.flush_categories();
        let col_names = vec!["tags".to_string()];
        let ingested_types = vec![IngestedType::StringArray];
        let map = stats.to_kumo_json(&col_names, &ingested_types, Some(&tc), None, "v1");
        let sep = map
            .get("tags__kumo.multicategorical__MULTI_CATEGORIES_SEPARATOR__False")
            .expect("separator key present");
        assert_eq!(sep["value"], JsonValue::Null);
    }

    // --- Helpers for building list views (using rowpack's Parsed variant to
    //     avoid having to construct Arrow arrays for unit tests) ----------

    fn build_string_list_view(elems: &[Option<&str>]) -> StringListView {
        // rowpack's `Parsed` variant holds owned `String`s and does not expose
        // element-level nulls. Use `build_native_string_list_view` for tests
        // that need to exercise Arrow null bits.
        let owned: Vec<String> = elems
            .iter()
            .map(|e| {
                e.expect("Parsed StringListView cannot represent null elements")
                    .to_string()
            })
            .collect();
        let end = owned.len();
        StringListView {
            start: 0,
            end,
            values: diskgraph_rowpack::StringListValues::Parsed(owned),
        }
    }

    fn build_int64_list_view(elems: &[Option<i64>]) -> Int64ListView {
        // Parsed values cannot represent null elements. Use
        // `build_native_int64_list_view` for Arrow null-bit tests.
        let owned: Vec<i64> = elems
            .iter()
            .map(|e| e.expect("Parsed Int64ListView cannot represent null elements"))
            .collect();
        let end = owned.len();
        Int64ListView {
            start: 0,
            end,
            values: diskgraph_rowpack::Int64ListValues::Parsed(owned),
        }
    }

    fn build_float64_list_view(elems: &[Option<f64>]) -> FloatListView {
        // `FloatListValues::Parsed` only carries f32; for the NaN-drop /
        // signed-zero tests we use f32 with `as f32` which preserves these
        // behaviors for the values in scope. Use `build_native_float64_list_view`
        // for Arrow null-bit tests.
        let owned: Vec<f32> = elems
            .iter()
            .map(|e| e.expect("Parsed FloatListView cannot represent null elements") as f32)
            .collect();
        let end = owned.len();
        FloatListView {
            start: 0,
            end,
            values: diskgraph_rowpack::FloatListValues::Parsed(owned),
        }
    }

    fn build_native_string_list_view(elems: &[Option<&str>]) -> StringListView {
        let end = elems.len();
        let array = arrow_array::StringArray::from(elems.to_vec());
        StringListView {
            start: 0,
            end,
            values: diskgraph_rowpack::StringListValues::Utf8(
                std::sync::Arc::new(array) as arrow_array::ArrayRef
            ),
        }
    }

    fn build_native_int64_list_view(elems: &[Option<i64>]) -> Int64ListView {
        let end = elems.len();
        let array = arrow_array::Int64Array::from(elems.to_vec());
        Int64ListView {
            start: 0,
            end,
            values: diskgraph_rowpack::Int64ListValues::Native(
                std::sync::Arc::new(array) as arrow_array::ArrayRef
            ),
        }
    }

    fn build_native_float64_list_view(elems: &[Option<f64>]) -> FloatListView {
        let end = elems.len();
        let array = arrow_array::Float64Array::from(elems.to_vec());
        FloatListView {
            start: 0,
            end,
            values: diskgraph_rowpack::FloatListValues::F64(
                std::sync::Arc::new(array) as arrow_array::ArrayRef
            ),
        }
    }
}
