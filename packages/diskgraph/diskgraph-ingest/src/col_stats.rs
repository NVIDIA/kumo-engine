// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-column statistics collection during the encode stage.
//!
//! `FileColStats` implements `ColumnValueObserver` and accumulates per-file
//! statistics using streaming algorithms. After a file is processed, the
//! per-file stats are merged into a shared `TableColStats` accumulator
//! via a mutex.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::graph_config::{ColumnSpec, IngestedType, SemanticType};
use diskgraph_core::time::{split_parts, TimeUnit};
use diskgraph_rowpack::{ColumnValueObserver, FloatListView, IntListView, StringListView};
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
    build_multicat_emit_stats, compute_min_occ, isqrt, kumo_entry, kumo_key, kumo_safe_div,
    CategoryCountStrategy, CountStorage, IdentityBuildHasher, IdentityHasher,
    LocalColumnCounts, MulticatInputKind, SharedColumnCounts, MAX_CATEGORIES,
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
    /// Categorical, Id, and Multicategorical columns get category count
    /// collection, each capped at `MAX_CATEGORIES` uniques so Encode memory
    /// stays bounded. Multicategorical reuses the same machinery: one DashMap
    /// entry per distinct *token* (not per row), fed by the per-input-kind
    /// tokenization paths in `ColumnStatsCollector::Multicategorical`.
    pub fn new(
        strategy: CategoryCountStrategy,
        semantic_types: &[Option<SemanticType>],
        row_subsample: bool,
    ) -> Self {
        Self::new_with_max_categories(strategy, semantic_types, row_subsample, MAX_CATEGORIES)
    }

    /// Construct table counts with an injectable Hybrid feature cap.
    /// Production uses [`MAX_CATEGORIES`]; focused tests use a small cap while
    /// exercising the same observation and publication paths.
    pub fn new_with_max_categories(
        strategy: CategoryCountStrategy,
        semantic_types: &[Option<SemanticType>],
        row_subsample: bool,
        max_categories: u64,
    ) -> Self {
        assert!(
            strategy != CategoryCountStrategy::Hybrid || !row_subsample,
            "Hybrid category counts require row_subsample=false so retained occurrence counts are exact"
        );
        let columns = semantic_types
            .iter()
            .map(|stype| match stype {
                // Id columns are counted alongside Categorical: some models
                // consume their category counts, and the shared map is capped
                // so there is no memory reason to skip them.
                Some(SemanticType::Categorical) | Some(SemanticType::Id) => {
                    let shared = match strategy {
                        CategoryCountStrategy::Graduation => SharedColumnCounts::new_graduation(),
                        _ => SharedColumnCounts::new_hashed_capped_with_max_categories(
                            max_categories,
                        ),
                    };
                    Some(Arc::new(shared))
                }
                Some(SemanticType::Multicategorical) => {
                    // Token cardinality can be proportional to the complete
                    // input. Use the same capped/abort policy as scalar
                    // categorical columns so Encode memory remains bounded.
                    let shared = match strategy {
                        CategoryCountStrategy::Graduation => SharedColumnCounts::new_graduation(),
                        _ => SharedColumnCounts::new_hashed_capped_with_max_categories(
                            max_categories,
                        ),
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
    /// The index clock that timestamp statistics are in.
    clock: TimeUnit,
}

/// Per-column stats collector, chosen by SemanticType.
pub enum ColumnStatsCollector {
    Numerical {
        null_count: u64,
        parse_fail_count: u64,
        invalid_count: u64,
        welford: WelfordAccumulator,
        hll: HyperLogLog,
        tdigest: TDigestEstimator,
    },
    Categorical {
        count: u64,
        null_count: u64,
        parse_fail_count: u64,
        hll: HyperLogLog,
    },
    Id {
        count: u64,
        null_count: u64,
        parse_fail_count: u64,
        hll: HyperLogLog,
    },
    Text {
        count: u64,
        null_count: u64,
        parse_fail_count: u64,
        hll: HyperLogLog,
    },
    Timestamp {
        count: u64,
        null_count: u64,
        parse_fail_count: u64,
        min: i64,
        max: i64,
        hll: HyperLogLog,
    },
    Sequence {
        count: u64,
        null_count: u64,
        parse_fail_count: u64,
        min_len: usize,
        max_len: usize,
        welford: WelfordAccumulator,
    },
    Latlong {
        count: u64,
        null_count: u64,
        parse_fail_count: u64,
        invalid_count: u64,
        min_len: usize,
        max_len: usize,
        welford: WelfordAccumulator,
    },
    /// Image URL column — stores universal column statistics.
    /// Produces `kumo.image` keys in col_stats output.
    ImageUrl {
        count: u64,
        null_count: u64,
        parse_fail_count: u64,
    },
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
        parse_fail_count: u64,
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
            clock: TimeUnit::Seconds,
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
            clock: TimeUnit::Seconds,
        }
    }

    pub fn with_clock(mut self, clock: TimeUnit) -> Self {
        self.clock = clock;
        self
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

    pub fn warn_parse_failures(&self, column_names: &[String], table_name: &str) {
        for (col, name) in self.cols.iter().zip(column_names) {
            let count = col.parse_fail_count();
            if count > 0 {
                tracing::warn!(
                    "{count} invalid values found in {table_name}.{name}. Invalid values are parsed as null. Please correct and retrain if needed."
                );
            }
        }
    }

    /// Serialize accumulated stats to a JSON object keyed by column name.
    /// Category counts are read (not consumed) from the shared state.
    /// Reference for the `col_stats.json` that [`write_col_stats`] streams.
    #[cfg(any(test, feature = "test-util"))]
    pub fn to_json(
        &mut self,
        column_names: &[String],
        ingested_types: &[IngestedType],
        table_cat: Option<&TableCategoryCounts>,
        text_sample_stats: Option<&HashMap<String, TextSampleStats>>,
    ) -> JsonValue {
        let cat_counts = self.copy_category_counts(table_cat, column_names);
        let cat_capped = self.category_capped(table_cat);
        let has_counts: Vec<bool> = cat_counts.iter().map(Option::is_some).collect();
        let mut map = self.summary_json(column_names, &has_counts, &cat_capped, text_sample_stats);
        for (col_idx, cc) in cat_counts.iter().enumerate() {
            let (Some(cc), Some(name)) = (cc, column_names.get(col_idx)) else {
                continue;
            };
            let Some(JsonValue::Object(obj)) = map.get_mut(name) else {
                continue;
            };
            // Array of [key, count] pairs matching the format GraphEngine/Spark produces.
            let is_int = ingested_types
                .get(col_idx)
                .is_some_and(|t| t.is_integer_like());
            let mut sorted: Vec<_> = cc.iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
            let json_counts: Vec<JsonValue> = sorted
                .into_iter()
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
        JsonValue::Object(map)
    }

    /// `to_json` with each `category_counts` left as `[]` for the caller to fill.
    fn summary_json(
        &mut self,
        column_names: &[String],
        has_counts: &[bool],
        cat_capped: &[Option<bool>],
        text_sample_stats: Option<&HashMap<String, TextSampleStats>>,
    ) -> serde_json::Map<String, JsonValue> {
        let mut map = serde_json::Map::new();
        for (col_idx, (col, name)) in self.cols.iter_mut().zip(column_names.iter()).enumerate() {
            if let Some(mut val) = col.to_json(self.clock) {
                if has_counts[col_idx] {
                    if let JsonValue::Object(ref mut obj) = val {
                        obj.insert("category_counts".to_string(), json!([]));
                        if let Some(capped) = cat_capped[col_idx] {
                            obj.insert("category_counts_capped".to_string(), json!(capped));
                        }
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
        map
    }

    #[cfg(any(test, feature = "test-util"))]
    fn copy_category_counts(
        &self,
        table_cat: Option<&TableCategoryCounts>,
        column_names: &[String],
    ) -> Vec<Option<HashMap<String, u64>>> {
        (0..self.cols.len())
            .map(|col_idx| {
                get_category_counts_for_col(
                    self.cat_strategy,
                    table_cat,
                    col_idx,
                    column_names.get(col_idx).map_or("", String::as_str),
                )
            })
            .collect()
    }

    /// Call after the counts were read: publication can set the capped flag.
    #[cfg(any(test, feature = "test-util"))]
    fn category_capped(&self, table_cat: Option<&TableCategoryCounts>) -> Vec<Option<bool>> {
        (0..self.cols.len())
            .map(|col_idx| {
                (self.cat_strategy == CategoryCountStrategy::Hybrid)
                    .then(|| {
                        table_cat?
                            .columns
                            .get(col_idx)?
                            .as_ref()
                            .map(|shared| shared.is_capped())
                    })
                    .flatten()
            })
            .collect()
    }

    /// Consumes the table's category counts for publication, sorting
    /// columns in parallel. Returns per column the sorted counts and, for
    /// Hybrid, whether they were capped. Errors if anything else still holds
    /// the counts, since they are moved out.
    fn take_published_counts(
        &mut self,
        table_name: &str,
        table_cat: Option<Arc<TableCategoryCounts>>,
        column_names: &[String],
    ) -> anyhow::Result<PublishedColumns> {
        let n = self.cols.len();
        let mut counts: Vec<Option<Vec<(String, u64)>>> = (0..n).map(|_| None).collect();
        let mut capped: Vec<Option<bool>> = vec![None; n];
        // Drop this observer's handle so the table's counts can be owned.
        self.cat_shared = None;
        let Some(table_cat) =
            table_cat.filter(|_| self.cat_strategy != CategoryCountStrategy::None)
        else {
            return Ok((counts, capped));
        };
        let still_shared = || {
            anyhow::anyhow!(
                "category counts for table {table_name} are still shared at publication"
            )
        };
        let table_cat = Arc::try_unwrap(table_cat).map_err(|_| still_shared())?;
        let scale =
            table_cat.row_subsample && self.cat_strategy != CategoryCountStrategy::Graduation;
        let mut jobs = Vec::new();
        for (col_idx, shared) in table_cat.columns.into_iter().enumerate().take(n) {
            if let Some(shared) = shared {
                let shared = Arc::try_unwrap(shared).map_err(|_| still_shared())?;
                shared.warn_collisions_once(column_names.get(col_idx).map_or("", String::as_str));
                jobs.push((col_idx, shared));
            }
        }
        let workers = diskgraph_limits::effective_parallelism().min(jobs.len());
        let jobs = std::sync::Mutex::new(jobs);
        let done: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|_| {
                    scope.spawn(|| {
                        let mut done = Vec::new();
                        loop {
                            let job = jobs.lock().unwrap_or_else(|e| e.into_inner()).pop();
                            let Some((col_idx, shared)) = job else { break };
                            let mut published = shared.into_sorted_counts();
                            if scale {
                                for (_, v) in published.counts.iter_mut().flatten() {
                                    *v = (*v as f64 / ROW_SUBSAMPLE_PROB).round() as u64;
                                }
                            }
                            done.push((col_idx, published));
                        }
                        done
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
                .collect()
        });
        for (col_idx, published) in done {
            counts[col_idx] = published.counts;
            capped[col_idx] =
                (self.cat_strategy == CategoryCountStrategy::Hybrid).then_some(published.capped);
        }
        Ok((counts, capped))
    }
}

/// Per column: sorted category counts and, for Hybrid, whether they were capped.
type PublishedColumns = (Vec<Option<Vec<(String, u64)>>>, Vec<Option<bool>>);

/// Free function to get category counts for a column (avoids self borrow issues).
#[cfg(any(test, feature = "test-util"))]
fn get_category_counts_for_col(
    strategy: CategoryCountStrategy,
    table_cat: Option<&TableCategoryCounts>,
    col_idx: usize,
    column_name: &str,
) -> Option<HashMap<String, u64>> {
    match strategy {
        CategoryCountStrategy::None => None,
        CategoryCountStrategy::Hybrid | CategoryCountStrategy::Graduation => {
            let cat = table_cat?;
            let col_shared = cat.columns.get(col_idx)?.as_ref()?;
            col_shared.warn_collisions_once(column_name);
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
    fn parse_fail_count(&self) -> u64 {
        match self {
            Self::Numerical {
                parse_fail_count, ..
            }
            | Self::Categorical {
                parse_fail_count, ..
            }
            | Self::Id {
                parse_fail_count, ..
            }
            | Self::Text {
                parse_fail_count, ..
            }
            | Self::Timestamp {
                parse_fail_count, ..
            }
            | Self::Sequence {
                parse_fail_count, ..
            }
            | Self::Latlong {
                parse_fail_count, ..
            }
            | Self::ImageUrl {
                parse_fail_count, ..
            }
            | Self::Multicategorical {
                parse_fail_count, ..
            } => *parse_fail_count,
            Self::Skip => 0,
        }
    }

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
                parse_fail_count: 0,
                invalid_count: 0,
                welford: WelfordAccumulator::new(),
                hll: HyperLogLog::new(),
                tdigest: TDigestEstimator::new(TDigestEstimator::DEFAULT_DELTA),
            },
            Some(SemanticType::Categorical) => ColumnStatsCollector::Categorical {
                count: 0,
                null_count: 0,
                parse_fail_count: 0,
                hll: HyperLogLog::new(),
            },
            Some(SemanticType::Id) => ColumnStatsCollector::Id {
                count: 0,
                null_count: 0,
                parse_fail_count: 0,
                hll: HyperLogLog::new(),
            },
            Some(SemanticType::Text) => ColumnStatsCollector::Text {
                count: 0,
                null_count: 0,
                parse_fail_count: 0,
                hll: HyperLogLog::new(),
            },
            Some(SemanticType::Timestamp) => ColumnStatsCollector::Timestamp {
                count: 0,
                null_count: 0,
                parse_fail_count: 0,
                min: i64::MAX,
                max: i64::MIN,
                hll: HyperLogLog::new(),
            },
            Some(SemanticType::Sequence) => ColumnStatsCollector::Sequence {
                count: 0,
                null_count: 0,
                parse_fail_count: 0,
                min_len: usize::MAX,
                max_len: 0,
                welford: WelfordAccumulator::new(),
            },
            Some(SemanticType::Latlong) => ColumnStatsCollector::Latlong {
                count: 0,
                null_count: 0,
                parse_fail_count: 0,
                invalid_count: 0,
                min_len: usize::MAX,
                max_len: 0,
                welford: WelfordAccumulator::new(),
            },
            Some(SemanticType::ImageUrl) => ColumnStatsCollector::ImageUrl {
                count: 0,
                null_count: 0,
                parse_fail_count: 0,
            },
            Some(SemanticType::Multicategorical) => {
                let input_kind = match ingested_type {
                    IngestedType::String => Some(MulticatInputKind::SingleStringToken),
                    IngestedType::StringArray => Some(MulticatInputKind::StringArray),
                    IngestedType::Int8Array
                    | IngestedType::Int16Array
                    | IngestedType::Int32Array
                    | IngestedType::Int64Array => Some(MulticatInputKind::Int64Array),
                    IngestedType::Float16Array
                    | IngestedType::Float32Array
                    | IngestedType::Float64Array => Some(MulticatInputKind::FloatArray),
                    _ => None,
                };
                match input_kind {
                    Some(kind) => ColumnStatsCollector::Multicategorical {
                        count: 0,
                        null_count: 0,
                        parse_fail_count: 0,
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
                    parse_fail_count,
                    invalid_count,
                    welford,
                    hll,
                    tdigest,
                },
                ColumnStatsCollector::Numerical {
                    null_count: nc2,
                    parse_fail_count: pf2,
                    invalid_count: ic2,
                    welford: w2,
                    hll: h2,
                    tdigest: td2,
                },
            ) => {
                *null_count += *nc2;
                *parse_fail_count += *pf2;
                *invalid_count += *ic2;
                welford.merge(w2);
                hll.merge(h2);
                tdigest.merge(td2);
            }
            (
                ColumnStatsCollector::Categorical {
                    count,
                    null_count,
                    parse_fail_count,
                    hll,
                },
                ColumnStatsCollector::Categorical {
                    count: c2,
                    null_count: nc2,
                    parse_fail_count: pf2,
                    hll: h2,
                },
            ) => {
                *count += *c2;
                *null_count += *nc2;
                *parse_fail_count += *pf2;
                hll.merge(h2);
            }
            (
                ColumnStatsCollector::Id {
                    count,
                    null_count,
                    parse_fail_count,
                    hll,
                },
                ColumnStatsCollector::Id {
                    count: c2,
                    null_count: nc2,
                    parse_fail_count: pf2,
                    hll: h2,
                },
            ) => {
                *count += *c2;
                *null_count += *nc2;
                *parse_fail_count += *pf2;
                hll.merge(h2);
            }
            (
                ColumnStatsCollector::Text {
                    count,
                    null_count,
                    parse_fail_count,
                    hll,
                },
                ColumnStatsCollector::Text {
                    count: c2,
                    null_count: nc2,
                    parse_fail_count: pf2,
                    hll: h2,
                },
            ) => {
                *count += *c2;
                *null_count += *nc2;
                *parse_fail_count += *pf2;
                hll.merge(h2);
            }
            (
                ColumnStatsCollector::Timestamp {
                    count,
                    null_count,
                    parse_fail_count,
                    min,
                    max,
                    hll,
                },
                ColumnStatsCollector::Timestamp {
                    count: c2,
                    null_count: nc2,
                    parse_fail_count: pf2,
                    min: ms2,
                    max: xs2,
                    hll: hll2,
                },
            ) => {
                *count += *c2;
                *null_count += *nc2;
                *parse_fail_count += *pf2;
                hll.merge(hll2);
                if *ms2 < *min {
                    *min = *ms2;
                }
                if *xs2 > *max {
                    *max = *xs2;
                }
            }
            (
                ColumnStatsCollector::Sequence {
                    count,
                    null_count,
                    parse_fail_count,
                    min_len,
                    max_len,
                    welford,
                },
                ColumnStatsCollector::Sequence {
                    count: c2,
                    null_count: nc2,
                    parse_fail_count: pf2,
                    min_len: ml2,
                    max_len: xl2,
                    welford: w2,
                },
            ) => {
                *count += *c2;
                *null_count += *nc2;
                *parse_fail_count += *pf2;
                if *ml2 < *min_len {
                    *min_len = *ml2;
                }
                if *xl2 > *max_len {
                    *max_len = *xl2;
                }
                welford.merge(w2);
            }
            (
                ColumnStatsCollector::Latlong {
                    count,
                    null_count,
                    parse_fail_count,
                    invalid_count,
                    min_len,
                    max_len,
                    welford,
                },
                ColumnStatsCollector::Latlong {
                    count: c2,
                    null_count: nc2,
                    parse_fail_count: pf2,
                    invalid_count: ic2,
                    min_len: ml2,
                    max_len: xl2,
                    welford: w2,
                },
            ) => {
                *count += *c2;
                *null_count += *nc2;
                *parse_fail_count += *pf2;
                *invalid_count += *ic2;
                if *ml2 < *min_len {
                    *min_len = *ml2;
                }
                if *xl2 > *max_len {
                    *max_len = *xl2;
                }
                welford.merge(w2);
            }
            (
                ColumnStatsCollector::ImageUrl {
                    count,
                    null_count,
                    parse_fail_count,
                },
                ColumnStatsCollector::ImageUrl {
                    count: c2,
                    null_count: nc2,
                    parse_fail_count: pf2,
                },
            ) => {
                *count += *c2;
                *null_count += *nc2;
                *parse_fail_count += *pf2;
            }
            (
                ColumnStatsCollector::Multicategorical {
                    count,
                    null_count,
                    parse_fail_count,
                    hll,
                    input_kind,
                    separator: _,
                    separator_byte: _,
                },
                ColumnStatsCollector::Multicategorical {
                    count: c2,
                    null_count: nc2,
                    parse_fail_count: pf2,
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
                *parse_fail_count += *pf2;
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
    fn to_json(&mut self, clock: TimeUnit) -> Option<JsonValue> {
        match self {
            ColumnStatsCollector::Numerical {
                null_count,
                parse_fail_count,
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
                    "parse_fail_count": *parse_fail_count,
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
                parse_fail_count,
                hll,
            } => Some(json!({
                "semantic_type": "categorical",
                "count": *count + *null_count,
                "null_count": *null_count,
                "parse_fail_count": *parse_fail_count,
                "num_unique": hll.estimate() as u64,
            })),
            ColumnStatsCollector::Id {
                count,
                null_count,
                parse_fail_count,
                hll,
            } => Some(json!({
                "semantic_type": "id",
                "count": *count + *null_count,
                "null_count": *null_count,
                "parse_fail_count": *parse_fail_count,
                "num_unique": hll.estimate() as u64,
            })),
            ColumnStatsCollector::Text {
                count,
                null_count,
                parse_fail_count,
                hll,
            } => Some(json!({
                "semantic_type": "text",
                "count": *count + *null_count,
                "null_count": *null_count,
                "parse_fail_count": *parse_fail_count,
                "num_unique": hll.estimate() as u64,
            })),
            ColumnStatsCollector::Timestamp {
                count,
                null_count,
                parse_fail_count,
                min,
                max,
                hll,
                ..
            } => {
                let total = (*count).saturating_add(*null_count);
                let mut stats = json!({
                    "semantic_type": "timestamp",
                    "count": total,
                    "null_count": *null_count,
                    "parse_fail_count": *parse_fail_count,
                    "num_unique": hll.estimate() as u64,
                });
                let suffix = timestamp_stat_suffix(clock);
                stats[format!("min_{suffix}")] = json!(*min);
                stats[format!("max_{suffix}")] = json!(*max);
                if clock != TimeUnit::Seconds {
                    stats["timestamp_unit"] = json!(clock);
                }
                Some(stats)
            }
            ColumnStatsCollector::Sequence {
                count,
                null_count,
                parse_fail_count,
                min_len,
                max_len,
                welford,
            } => {
                // Keep the internal sentinel intact for merge, but do not
                // expose usize::MAX for a column with no populated rows.
                let (emitted_min_len, emitted_max_len) = if *count == 0 {
                    (0, 0)
                } else {
                    (*min_len, *max_len)
                };
                Some(json!({
                    "semantic_type": "sequence",
                    "count": *count + *null_count,
                    "null_count": *null_count,
                    "parse_fail_count": *parse_fail_count,
                    "min_len": emitted_min_len,
                    "max_len": emitted_max_len,
                    "sequence_mean": welford.mean(),
                    "sequence_std": welford.std_dev(),
                }))
            }
            ColumnStatsCollector::Latlong {
                count,
                null_count,
                parse_fail_count,
                invalid_count,
                min_len,
                max_len,
                welford,
            } => Some(json!({
                "semantic_type": "latlong",
                "count": *count + *null_count,
                "null_count": *null_count,
                "parse_fail_count": *parse_fail_count,
                "invalid_count": *invalid_count,
                "min_len": *min_len,
                "max_len": *max_len,
                "sequence_mean": welford.mean(),
                "sequence_std": welford.std_dev(),
            })),
            ColumnStatsCollector::ImageUrl {
                count,
                null_count,
                parse_fail_count,
            } => Some(json!({
                "semantic_type": "image_url",
                "count": *count + *null_count,
                "null_count": *null_count,
                "parse_fail_count": *parse_fail_count,
            })),
            ColumnStatsCollector::Multicategorical {
                count,
                null_count,
                parse_fail_count,
                hll,
                ..
            } => Some(json!({
                "semantic_type": "multicategorical",
                "count": *count + *null_count,
                "null_count": *null_count,
                "parse_fail_count": *parse_fail_count,
                "num_unique_multi": hll.estimate() as u64,
            })),
            ColumnStatsCollector::Skip => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Kumo-format stats output (replaces Python bridge)
// ---------------------------------------------------------------------------

/// The `col_stats.json` key suffix of timestamp bounds (`min_secs` for seconds).
fn timestamp_stat_suffix(clock: TimeUnit) -> &'static str {
    match clock {
        TimeUnit::Nanoseconds => "ns",
        TimeUnit::Microseconds => "us",
        TimeUnit::Milliseconds => "ms",
        TimeUnit::Hours => "hours",
        TimeUnit::Days => "days",
        TimeUnit::Seconds => "secs",
        other => unreachable!("{other:?} is not an index clock"),
    }
}

/// Format a timestamp in `clock` ticks to kumo timestamp string.
/// Sentinels (i64::MAX/MIN) and unparseable values → "kumo_timestamp_NaT".
fn format_kumo_timestamp(ticks: i64, clock: TimeUnit) -> String {
    if ticks == i64::MAX || ticks == i64::MIN {
        return "kumo_timestamp_NaT".to_string();
    }
    let (epoch_secs, nanos) = split_parts(ticks, clock);
    match chrono::DateTime::from_timestamp(epoch_secs, nanos) {
        Some(dt) => format!("kumo_timestamp_{}", dt.format("%Y-%m-%d %H:%M:%S%.f")),
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
        ColumnStatsCollector::Latlong { .. } => "kumo.latlong",
        ColumnStatsCollector::ImageUrl { .. } => "kumo.image",
        ColumnStatsCollector::Multicategorical { .. } => "kumo.multicategorical",
        ColumnStatsCollector::Skip => "",
    }
}

/// Build category count JSON array: [[key, count], ...].
/// For integer-like columns, keys are emitted as JSON numbers.
#[cfg(any(test, feature = "test-util"))]
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
        || matches!(
            ingested_type,
            Some(IngestedType::Int8Array)
                | Some(IngestedType::Int16Array)
                | Some(IngestedType::Int32Array)
                | Some(IngestedType::Int64Array)
        ))
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
    let mut sorted: Vec<_> = cc.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    let pairs: Vec<JsonValue> = sorted
        .into_iter()
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
#[cfg(any(test, feature = "test-util"))]
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
    /// Reads the counts without consuming them; a test reference for what
    /// [`write_col_stats`] and [`FileColStats::into_kumo_stats`] produce.
    #[cfg(any(test, feature = "test-util"))]
    pub fn to_kumo_json(
        &mut self,
        column_names: &[String],
        ingested_types: &[IngestedType],
        table_cat: Option<&TableCategoryCounts>,
        text_sample_stats: Option<&HashMap<String, TextSampleStats>>,
        data_version: &str,
    ) -> serde_json::Map<String, JsonValue> {
        let cat_counts = self.copy_category_counts(table_cat, column_names);
        let cat_capped = self.category_capped(table_cat);
        let cat_lens: Vec<Option<usize>> = cat_counts
            .iter()
            .map(|c| c.as_ref().map(HashMap::len))
            .collect();
        let mut map = self.kumo_stats(
            column_names,
            &cat_lens,
            &cat_capped,
            text_sample_stats,
            data_version,
        );
        for (col_idx, cc) in cat_counts.iter().enumerate() {
            let Some(cc) = cc else { continue };
            let col = &self.cols[col_idx];
            let (stat, values) = match col {
                ColumnStatsCollector::Categorical { .. } | ColumnStatsCollector::Id { .. } => (
                    "CATEGORY_COUNTS",
                    build_cat_count_json(col_idx, &cat_counts, ingested_types),
                ),
                ColumnStatsCollector::Multicategorical { .. } => {
                    // Count-desc then token-asc, byte-matching tt_executor.
                    let mut sorted: Vec<(&String, &u64)> = cc.iter().collect();
                    sorted.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
                    let values: Vec<JsonValue> = sorted
                        .into_iter()
                        .map(|(tok, cnt)| json!([tok, cnt]))
                        .collect();
                    ("MULTI_CATEGORY_COUNTS", json!(values))
                }
                _ => continue,
            };
            let key = kumo_key(&column_names[col_idx], kumo_stype_key(col), stat, false);
            if let Some(JsonValue::Object(entry)) = map.get_mut(&key) {
                entry.insert("value".to_string(), values);
            }
        }
        map
    }

    /// `to_kumo_json` with every category-count `value` left as `[]`;
    /// `cat_lens` are the published category counts per column.
    fn kumo_stats(
        &mut self,
        column_names: &[String],
        cat_lens: &[Option<usize>],
        cat_capped: &[Option<bool>],
        text_sample_stats: Option<&HashMap<String, TextSampleStats>>,
        data_version: &str,
    ) -> serde_json::Map<String, JsonValue> {
        // The category map enumerates every distinct value only under
        // non-subsampled Hybrid. Graduation drops sub-threshold categories and
        // subsampling only sees a fraction, so those paths cannot use the map's
        // length as the exact unique count.
        let cat_map_is_exhaustive =
            self.cat_strategy == CategoryCountStrategy::Hybrid && !self.row_subsample;

        let mut map = serde_json::Map::new();
        let dv = data_version;

        for (col_idx, col) in self.cols.iter_mut().enumerate() {
            let name = &column_names[col_idx];
            let sk = kumo_stype_key(col);

            match col {
                ColumnStatsCollector::Numerical {
                    null_count,
                    parse_fail_count,
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
                        kumo_key(name, sk, "NUM_PARSE_FAILURES", false),
                        kumo_entry(json!(*parse_fail_count), false, dv, count),
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
                        kumo_entry(
                            json!(kumo_safe_div(*invalid_count, count)),
                            false,
                            dv,
                            count,
                        ),
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
                    parse_fail_count,
                    hll,
                } => {
                    let total = *count + *null_count;
                    map.insert(
                        kumo_key(name, sk, "COUNT", false),
                        kumo_entry(json!(total), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "NUM_PARSE_FAILURES", false),
                        kumo_entry(json!(*parse_fail_count), false, dv, total),
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
                    let values = json!([]);
                    map.insert(
                        kumo_key(name, sk, "CATEGORY_COUNTS", false),
                        kumo_entry(values, false, dv, total),
                    );
                    let capped = cat_capped[col_idx].unwrap_or(false);
                    // Use the exact distinct count from the count-capped
                    // category map only when it is exhaustive (non-subsampled
                    // Hybrid). Graduation drops sub-threshold categories,
                    // subsampling sees only a fraction, and an aborted or
                    // absent map is None — all fall back to the HLL estimate.
                    let use_hll = !cat_map_is_exhaustive || capped;
                    let num_unique = match cat_lens[col_idx] {
                        Some(len) if !use_hll => len as u64,
                        _ => hll.estimate() as u64,
                    }
                    .min(total);
                    // Compatibility debt: scalar cardinality has historically
                    // used False keys and flags even when sourced from HLL.
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
                    parse_fail_count,
                    hll,
                } => {
                    let total = *count + *null_count;
                    map.insert(
                        kumo_key(name, sk, "COUNT", false),
                        kumo_entry(json!(total), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "NUM_PARSE_FAILURES", false),
                        kumo_entry(json!(*parse_fail_count), false, dv, total),
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
                    let values = json!([]);
                    map.insert(
                        kumo_key(name, sk, "CATEGORY_COUNTS", false),
                        kumo_entry(values, false, dv, total),
                    );
                    let capped = cat_capped[col_idx].unwrap_or(false);
                    // Use the exact distinct count from the count-capped
                    // category map only when it is exhaustive (non-subsampled
                    // Hybrid). Graduation drops sub-threshold categories,
                    // subsampling sees only a fraction, and an aborted or
                    // absent map is None — all fall back to the HLL estimate.
                    let use_hll = !cat_map_is_exhaustive || capped;
                    let num_unique = match cat_lens[col_idx] {
                        Some(len) if !use_hll => len as u64,
                        _ => hll.estimate() as u64,
                    }
                    .min(total);
                    // Compatibility debt: scalar cardinality has historically
                    // used False keys and flags even when sourced from HLL.
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
                    parse_fail_count,
                    hll,
                } => {
                    let total = *count + *null_count;
                    map.insert(
                        kumo_key(name, sk, "COUNT", false),
                        kumo_entry(json!(total), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "NUM_PARSE_FAILURES", false),
                        kumo_entry(json!(*parse_fail_count), false, dv, total),
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
                    // Use the exact distinct count from the count-capped
                    // category map only when it is exhaustive (non-subsampled
                    // Hybrid). Graduation drops sub-threshold categories,
                    // subsampling sees only a fraction, and an aborted or
                    // absent map is None — all fall back to the HLL estimate.
                    let num_unique = match cat_lens[col_idx] {
                        Some(len) if cat_map_is_exhaustive => len as u64,
                        _ => hll.estimate() as u64,
                    }
                    .min(total);
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
                    min,
                    max,
                    hll,
                    ..
                } => {
                    let total = (*c).saturating_add(*null_count);
                    map.insert(
                        kumo_key(name, sk, "COUNT", false),
                        kumo_entry(json!(total), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "NUM_PARSE_FAILURES", false),
                        kumo_entry(json!(*parse_fail_count), false, dv, total),
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
                        kumo_entry(
                            json!(format_kumo_timestamp(*min, self.clock)),
                            false,
                            dv,
                            total,
                        ),
                    );
                    map.insert(
                        kumo_key(name, sk, "MAX", false),
                        kumo_entry(
                            json!(format_kumo_timestamp(*max, self.clock)),
                            false,
                            dv,
                            total,
                        ),
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
                    parse_fail_count,
                    min_len,
                    max_len,
                    ..
                } => {
                    let total = *c + *null_count;
                    let (emitted_min_len, emitted_max_len) = if *c == 0 {
                        (0, 0)
                    } else {
                        (*min_len, *max_len)
                    };
                    map.insert(
                        kumo_key(name, sk, "COUNT", false),
                        kumo_entry(json!(total), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "NUM_PARSE_FAILURES", false),
                        kumo_entry(json!(*parse_fail_count), false, dv, total),
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
                    // Emit 0/0 for an all-missing sequence; otherwise expose
                    // the non-empty observations' length bounds.
                    map.insert(
                        kumo_key(name, sk, "SEQUENCE_MIN_LENGTH", false),
                        kumo_entry(json!(emitted_min_len), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "SEQUENCE_MAX_LENGTH", false),
                        kumo_entry(json!(emitted_max_len), false, dv, total),
                    );
                }
                ColumnStatsCollector::Latlong {
                    count: c,
                    null_count,
                    parse_fail_count,
                    invalid_count,
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
                        kumo_key(name, sk, "NUM_PARSE_FAILURES", false),
                        kumo_entry(json!(*parse_fail_count), false, dv, total),
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
                        kumo_entry(
                            json!(kumo_safe_div(*invalid_count, total)),
                            false,
                            dv,
                            total,
                        ),
                    );
                    map.insert(
                        kumo_key(name, sk, "SEQUENCE_MIN_LENGTH", false),
                        kumo_entry(json!(*min_len), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "SEQUENCE_MAX_LENGTH", false),
                        kumo_entry(json!(*max_len), false, dv, total),
                    );
                }
                ColumnStatsCollector::ImageUrl {
                    count,
                    null_count,
                    parse_fail_count,
                } => {
                    let total = *count + *null_count;
                    map.insert(
                        kumo_key(name, sk, "COUNT", false),
                        kumo_entry(json!(total), false, dv, total),
                    );
                    map.insert(
                        kumo_key(name, sk, "NUM_PARSE_FAILURES", false),
                        kumo_entry(json!(*parse_fail_count), false, dv, total),
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
                    parse_fail_count,
                    hll,
                    separator,
                    ..
                } => {
                    map.insert(
                        kumo_key(name, sk, "NUM_PARSE_FAILURES", false),
                        kumo_entry(json!(*parse_fail_count), false, dv, *count + *null_count),
                    );
                    let pairs = build_multicat_emit_stats(
                        name,
                        sk,
                        dv,
                        *count,
                        *null_count,
                        hll,
                        separator.as_deref(),
                        &[],
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
                        if let Some(ref mut local) = self.cat_local[col_idx] {
                            if local.increment_if_present(hash, value) {
                                return;
                            }
                            // Shared state owns membership: count/admit the
                            // first occurrence before caching only future
                            // unmerged deltas locally. This prevents a hot
                            // early value from being stranded in a worker map
                            // while unrelated workers fill the global cap.
                            if col_shared.observe_hashed(hash, value)
                                && local.len() < HYBRID_LOCAL_CAP
                            {
                                local.cache_admitted(hash, value);
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
                        if let Some(ref mut local) = self.cat_local[col_idx] {
                            if local.increment_i64_if_present(hash, value) {
                                return;
                            }
                            if col_shared.observe_hashed_i64(hash, value)
                                && local.len() < HYBRID_LOCAL_CAP
                            {
                                local.cache_admitted_i64(hash, value);
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
        let hash = HyperLogLog::hash_str(tok);
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
                min,
                max,
                hll,
                ..
            } => {
                *count += 1;
                if value < *min {
                    *min = value;
                }
                if value > *max {
                    *max = value;
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
        // Canonicalize categorical floats exactly as they are serialized. This
        // keeps HLL membership, retained counts, and JSON identity aligned for
        // values such as signed zero and non-finite float spellings.
        let cat_value = match &self.cols[col_idx] {
            ColumnStatsCollector::Categorical { .. } | ColumnStatsCollector::Id { .. } => {
                // The mapper treats both IEEE signed-zero encodings as the
                // same scalar key, so category statistics must do the same.
                // Multicategorical float-token spelling is intentionally
                // unchanged and continues through `observe_float_list`.
                let canonical = if value == 0.0 { 0.0 } else { value };
                let string = canonical.to_string();
                Some((HyperLogLog::hash_str(&string), string))
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
                hll.observe_hash(cat_value.as_ref().unwrap().0);
            }
            ColumnStatsCollector::Id { count, hll, .. } => {
                *count += 1;
                hll.observe_hash(cat_value.as_ref().unwrap().0);
            }
            _ => {}
        }

        // Category counts — after releasing the borrow on self.cols
        if let Some((hash, string)) = cat_value {
            self.observe_category_string(col_idx, hash, &string);
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
                Some(HyperLogLog::hash_str(value))
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
                hll.observe_str(value);
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
            min,
            max,
            hll,
            ..
        } = &mut self.cols[col_idx]
        {
            *count += 1;
            if secs < *min {
                *min = secs;
            }
            if secs > *max {
                *max = secs;
            }
            hll.observe_i64(secs);
        }
    }

    #[inline]
    fn observe_float_list(&mut self, col_idx: usize, view: &FloatListView) {
        match &mut self.cols[col_idx] {
            ColumnStatsCollector::Sequence {
                count,
                null_count,
                min_len,
                max_len,
                welford,
                ..
            } => {
                let len = view.len();
                // An empty fixed-width sequence carries no dimensionality.
                // Treat it as a missing feature for stats purposes so nullable
                // embeddings derive their width from populated rows. The
                // rowpack value remains an empty list and downstream tensor
                // mapping pads it with NaNs at the derived width.
                if len == 0 {
                    *null_count += 1;
                    return;
                }
                *count += 1;
                if len < *min_len {
                    *min_len = len;
                }
                if len > *max_len {
                    *max_len = len;
                }
                view.for_each_f64(|v| welford.observe(v));
            }
            ColumnStatsCollector::Latlong {
                count,
                invalid_count,
                min_len,
                max_len,
                welford,
                ..
            } => {
                let len = view.len();
                if len != 2 || view.has_nulls() {
                    *invalid_count += 1;
                    return;
                }
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
    fn observe_int64_list(&mut self, col_idx: usize, view: &IntListView) {
        match &mut self.cols[col_idx] {
            // Sequence stype + intlist dtype: mirror observe_float_list's
            // Sequence arm so int-list embedding columns produce non-sentinel
            // SEQUENCE_MIN_LENGTH / SEQUENCE_MAX_LENGTH stats. Without this,
            // Kumo's snapshot validator rejects the table with "Missing
            // embedding statistics" for any int-list sequence column.
            ColumnStatsCollector::Sequence {
                count,
                null_count,
                min_len,
                max_len,
                welford,
                ..
            } => {
                let len = view.len();
                if len == 0 {
                    *null_count += 1;
                    return;
                }
                *count += 1;
                if len < *min_len {
                    *min_len = len;
                }
                if len > *max_len {
                    *max_len = len;
                }
                view.for_each_i64(|v| welford.observe(v as f64));
            }
            ColumnStatsCollector::Latlong {
                count,
                invalid_count,
                min_len,
                max_len,
                welford,
                ..
            } => {
                let len = view.len();
                if len != 2 || view.has_nulls() {
                    *invalid_count += 1;
                    return;
                }
                *count += 1;
                if len < *min_len {
                    *min_len = len;
                }
                if len > *max_len {
                    *max_len = len;
                }
                view.for_each_i64(|v| welford.observe(v as f64));
            }
            ColumnStatsCollector::Multicategorical {
                input_kind: MulticatInputKind::Int64Array,
                count,
                ..
            } => {
                *count += 1;
                // Fall through to the per-token loop below. Releasing the &mut
                // borrow on self.cols first.
                let count_category = self.should_sample_row();
                view.for_each_i64(|n| {
                    // Integer tokens are identity-hashed via the same primitive
                    // categorical/id integer columns use; strings stored in
                    // `LocalColumnCounts::strings` are `n.to_string()`, matching
                    // tt_executor's stringified-int emit.
                    self.observe_multicat_token_i64(col_idx, n, count_category);
                });
            }
            _ => {}
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
            ColumnStatsCollector::Latlong { null_count, .. } => *null_count += 1,
            ColumnStatsCollector::ImageUrl { null_count, .. } => *null_count += 1,
            ColumnStatsCollector::Multicategorical { null_count, .. } => *null_count += 1,
            ColumnStatsCollector::Skip => {}
        }
    }

    #[inline]
    fn observe_parse_failure(&mut self, col_idx: usize) {
        match &mut self.cols[col_idx] {
            ColumnStatsCollector::Numerical {
                parse_fail_count, ..
            }
            | ColumnStatsCollector::Categorical {
                parse_fail_count, ..
            }
            | ColumnStatsCollector::Id {
                parse_fail_count, ..
            }
            | ColumnStatsCollector::Text {
                parse_fail_count, ..
            }
            | ColumnStatsCollector::Timestamp {
                parse_fail_count, ..
            }
            | ColumnStatsCollector::Sequence {
                parse_fail_count, ..
            }
            | ColumnStatsCollector::ImageUrl {
                parse_fail_count, ..
            }
            | ColumnStatsCollector::Multicategorical {
                parse_fail_count, ..
            } => *parse_fail_count += 1,
            ColumnStatsCollector::Latlong {
                parse_fail_count,
                invalid_count,
                ..
            } => {
                *parse_fail_count += 1;
                *invalid_count += 1;
            }
            ColumnStatsCollector::Skip => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming publication
// ---------------------------------------------------------------------------

/// One table's inputs to [`write_col_stats`].
pub struct ColStatsTable<'a> {
    pub name: String,
    pub stats: FileColStats,
    pub column_names: Vec<String>,
    pub ingested_types: Vec<IngestedType>,
    /// Consumed: nothing else may hold it.
    pub table_cat: Option<Arc<TableCategoryCounts>>,
    pub text_sample_stats: Option<&'a HashMap<String, TextSampleStats>>,
    /// Kumo-format files are written only when set.
    pub data_version: Option<String>,
}

/// Writes `col_stats.json` and, per table with a data version,
/// `kumo_col_stats/col_stats_{full,min}/<table>.json`.
///
/// Output bytes equal pretty-serializing `FileColStats::to_json` and
/// serializing [`FileColStats::to_kumo_json`] / [`minify_kumo_stats`],
/// but each column's category map is consumed, and its sorted counts are
/// rendered in parallel and streamed to the files instead of being copied
/// into JSON values.
pub fn write_col_stats(
    target: &crate::io::OutputTarget,
    tables: Vec<ColStatsTable<'_>>,
) -> anyhow::Result<()> {
    let mut by_name: HashMap<String, ColStatsTable<'_>> =
        tables.into_iter().map(|t| (t.name.clone(), t)).collect();
    // Same table order as a root `serde_json::Map`.
    let order: serde_json::Map<String, JsonValue> = by_name
        .keys()
        .map(|name| (name.clone(), JsonValue::Null))
        .collect();
    write_stream(target, "col_stats.json", |out| {
        write_object(out, &order, Some(0), |out, name, _, depth| {
            let table = by_name.remove(name).expect("table listed in order");
            publish_table(target, table, out, depth)
        })
    })
}

fn publish_table(
    target: &crate::io::OutputTarget,
    table: ColStatsTable<'_>,
    summary_out: &mut dyn Write,
    summary_depth: Option<usize>,
) -> anyhow::Result<()> {
    let ColStatsTable {
        name,
        mut stats,
        column_names,
        ingested_types,
        table_cat,
        text_sample_stats,
        data_version,
    } = table;
    let (counts, capped) = stats.take_published_counts(&name, table_cat, &column_names)?;
    let has_counts: Vec<bool> = counts.iter().map(Option::is_some).collect();
    let summary = stats.summary_json(&column_names, &has_counts, &capped, text_sample_stats);
    let kumo = match &data_version {
        Some(dv) => stats.kumo_col_stats(
            counts,
            &capped,
            &column_names,
            &ingested_types,
            text_sample_stats,
            dv,
        ),
        None => KumoColStats {
            stats: serde_json::Map::new(),
            counts,
            keys: Vec::new(),
        },
    };
    let summary_arrays: HashMap<String, CatArray<'_>> = kumo
        .counts
        .iter()
        .zip(&column_names)
        .zip(&ingested_types)
        .filter_map(|((counts, col_name), ingested_type)| {
            let kind = if ingested_type.is_integer_like() {
                CatKeyKind::Int
            } else {
                CatKeyKind::Str
            };
            Some((col_name.clone(), (counts.as_deref()?, kind)))
        })
        .collect();

    std::thread::scope(|scope| {
        let kumo_writer = data_version.is_some().then(|| {
            let (name, kumo) = (&name, &kumo);
            scope.spawn(move || -> anyhow::Result<()> {
                use crate::col_stats_export::{COL_STATS_DIR, COL_STATS_FULL, COL_STATS_MIN};
                for (subdir, limit) in [
                    (COL_STATS_FULL, usize::MAX),
                    (COL_STATS_MIN, MIN_CATEGORY_COUNTS),
                ] {
                    let rel = format!("{COL_STATS_DIR}/{subdir}/{name}.json");
                    write_stream(target, &rel, |out| kumo.write(&kumo.stats, limit, out))?;
                }
                Ok(())
            })
        });
        write_spliced(
            summary_out,
            &summary,
            "category_counts",
            &summary_arrays,
            usize::MAX,
            summary_depth,
        )?;
        if let Some(handle) = kumo_writer {
            handle
                .join()
                .unwrap_or_else(|e| std::panic::resume_unwind(e))?;
        }
        Ok(())
    })
}

/// Kumo-format stats for one table, holding its consumed category counts.
pub struct KumoColStats {
    /// Every category-count `value` is `[]` here; [`Self::write`] fills them in.
    pub stats: serde_json::Map<String, JsonValue>,
    counts: Vec<Option<Vec<(String, u64)>>>,
    /// Stat key, column index and key type of each category-count array.
    keys: Vec<(String, usize, CatKeyKind)>,
}

impl KumoColStats {
    /// Serializes `stats` (this table's stats or a map derived from them)
    /// with each category-count array filled in, at most `limit` entries each.
    pub fn write(
        &self,
        stats: &serde_json::Map<String, JsonValue>,
        limit: usize,
        out: &mut dyn Write,
    ) -> anyhow::Result<()> {
        let arrays: HashMap<String, CatArray<'_>> = self
            .keys
            .iter()
            .filter_map(|(key, col_idx, kind)| {
                Some((key.clone(), (self.counts[*col_idx].as_deref()?, *kind)))
            })
            .collect();
        write_spliced(out, stats, "value", &arrays, limit, None)
    }

    pub fn to_json_string(
        &self,
        stats: &serde_json::Map<String, JsonValue>,
        limit: usize,
    ) -> anyhow::Result<String> {
        let mut out = Vec::new();
        self.write(stats, limit, &mut out)?;
        Ok(String::from_utf8(out)?)
    }
}

impl Drop for KumoColStats {
    fn drop(&mut self) {
        // Freeing millions of strings is slow on one thread.
        std::thread::scope(|scope| {
            for column in std::mem::take(&mut self.counts).into_iter().flatten() {
                if column.len() > 64 * 1024 {
                    scope.spawn(move || drop(column));
                }
            }
        });
    }
}

impl FileColStats {
    /// Kumo-format stats for one table, consuming its category counts.
    /// Errors if anything else still holds `table_cat`.
    pub fn into_kumo_stats(
        mut self,
        table_name: &str,
        column_names: &[String],
        ingested_types: &[IngestedType],
        table_cat: Option<Arc<TableCategoryCounts>>,
        text_sample_stats: Option<&HashMap<String, TextSampleStats>>,
        data_version: &str,
    ) -> anyhow::Result<KumoColStats> {
        let (counts, capped) = self.take_published_counts(table_name, table_cat, column_names)?;
        Ok(self.kumo_col_stats(
            counts,
            &capped,
            column_names,
            ingested_types,
            text_sample_stats,
            data_version,
        ))
    }

    fn kumo_col_stats(
        &mut self,
        counts: Vec<Option<Vec<(String, u64)>>>,
        capped: &[Option<bool>],
        column_names: &[String],
        ingested_types: &[IngestedType],
        text_sample_stats: Option<&HashMap<String, TextSampleStats>>,
        data_version: &str,
    ) -> KumoColStats {
        let lens: Vec<Option<usize>> = counts.iter().map(|c| c.as_ref().map(Vec::len)).collect();
        let stats = self.kumo_stats(column_names, &lens, capped, text_sample_stats, data_version);
        let keys = self
            .cols
            .iter()
            .enumerate()
            .filter(|(col_idx, _)| counts[*col_idx].is_some())
            .filter_map(|(col_idx, col)| {
                let (stat, kind) = match col {
                    ColumnStatsCollector::Categorical { .. } | ColumnStatsCollector::Id { .. } => (
                        "CATEGORY_COUNTS",
                        kumo_cat_key_kind(ingested_types.get(col_idx)),
                    ),
                    ColumnStatsCollector::Multicategorical { .. } => {
                        ("MULTI_CATEGORY_COUNTS", CatKeyKind::Str)
                    }
                    _ => return None,
                };
                let key = kumo_key(column_names.get(col_idx)?, kumo_stype_key(col), stat, false);
                Some((key, col_idx, kind))
            })
            .collect();
        KumoColStats {
            stats,
            counts,
            keys,
        }
    }
}

/// `CATEGORY_COUNTS` entries kept in `col_stats_min`.
pub const MIN_CATEGORY_COUNTS: usize = 20;

fn write_stream(
    target: &crate::io::OutputTarget,
    rel: &str,
    write: impl FnOnce(&mut dyn Write) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let path = target.join(rel);
    if matches!(target, crate::io::OutputTarget::Local { .. }) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut out = std::io::BufWriter::with_capacity(1 << 20, target.sink(&path)?);
    write(&mut out)?;
    out.into_inner().map_err(|e| e.into_error())?.finish()?;
    Ok(())
}

// serde_json's layout: compact (`None`) or pretty with a two-space indent at
// nesting depth `Some(d)`.

fn write_indent(out: &mut dyn Write, depth: usize) -> std::io::Result<()> {
    for _ in 0..depth {
        out.write_all(b"  ")?;
    }
    Ok(())
}

fn write_object(
    out: &mut dyn Write,
    map: &serde_json::Map<String, JsonValue>,
    depth: Option<usize>,
    mut write_value: impl FnMut(&mut dyn Write, &str, &JsonValue, Option<usize>) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    out.write_all(b"{")?;
    let inner = depth.map(|d| d + 1);
    for (i, (key, value)) in map.iter().enumerate() {
        if i > 0 {
            out.write_all(b",")?;
        }
        if let Some(d) = inner {
            out.write_all(b"\n")?;
            write_indent(out, d)?;
        }
        serde_json::to_writer(&mut *out, key)?;
        out.write_all(if depth.is_some() { b": " } else { b":" })?;
        write_value(out, key, value, inner)?;
    }
    if let (false, Some(d)) = (map.is_empty(), depth) {
        out.write_all(b"\n")?;
        write_indent(out, d)?;
    }
    out.write_all(b"}")?;
    Ok(())
}

fn write_value(out: &mut dyn Write, value: &JsonValue, depth: Option<usize>) -> anyhow::Result<()> {
    match depth {
        None => serde_json::to_writer(out, value)?,
        Some(depth) => serde_json::to_writer_pretty(Indent { out, depth }, value)?,
    }
    Ok(())
}

/// Writes `map`, emitting `arrays[key]` in place of `map[key][field]`.
fn write_spliced(
    out: &mut dyn Write,
    map: &serde_json::Map<String, JsonValue>,
    field: &str,
    arrays: &HashMap<String, CatArray<'_>>,
    limit: usize,
    depth: Option<usize>,
) -> anyhow::Result<()> {
    write_object(out, map, depth, |out, key, value, depth| {
        match (arrays.get(key), value) {
            (Some(&(counts, kind)), JsonValue::Object(obj)) => {
                write_object(out, obj, depth, |out, key, value, depth| {
                    if key == field {
                        write_cat_array(out, &counts[..counts.len().min(limit)], kind, depth)
                    } else {
                        write_value(out, value, depth)
                    }
                })
            }
            _ => write_value(out, value, depth),
        }
    })
}

/// `[[key, count], ...]`, rendered in parallel slices.
fn write_cat_array(
    out: &mut dyn Write,
    counts: &[(String, u64)],
    kind: CatKeyKind,
    depth: Option<usize>,
) -> anyhow::Result<()> {
    if counts.is_empty() {
        out.write_all(b"[]")?;
        return Ok(());
    }
    out.write_all(b"[")?;
    let workers = diskgraph_limits::effective_parallelism();
    let slice_len = counts.len().div_ceil(workers).max(16 * 1024);
    let rendered: Vec<Vec<u8>> = if counts.len() <= slice_len {
        vec![render_pairs(counts, kind, depth, false)]
    } else {
        std::thread::scope(|scope| {
            let handles: Vec<_> = counts
                .chunks(slice_len)
                .enumerate()
                .map(|(i, slice)| scope.spawn(move || render_pairs(slice, kind, depth, i > 0)))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
                .collect()
        })
    };
    for bytes in rendered {
        out.write_all(&bytes)?;
    }
    if let Some(d) = depth {
        out.write_all(b"\n")?;
        write_indent(out, d)?;
    }
    out.write_all(b"]")?;
    Ok(())
}

fn render_pairs(
    pairs: &[(String, u64)],
    kind: CatKeyKind,
    depth: Option<usize>,
    leading_comma: bool,
) -> Vec<u8> {
    let pad = |buf: &mut Vec<u8>, levels: usize| {
        if depth.is_some() {
            buf.push(b'\n');
            buf.extend(std::iter::repeat_n(b' ', 2 * levels));
        }
    };
    let d = depth.unwrap_or(0);
    let mut buf: Vec<u8> = Vec::with_capacity(pairs.len() * 48);
    for (i, (key, count)) in pairs.iter().enumerate() {
        if i > 0 || leading_comma {
            buf.push(b',');
        }
        pad(&mut buf, d + 1);
        buf.push(b'[');
        pad(&mut buf, d + 2);
        // Writes into a Vec cannot fail.
        let _ = serde_json::to_writer(&mut buf, &CatKey(key, kind));
        buf.push(b',');
        pad(&mut buf, d + 2);
        let _ = write!(buf, "{count}");
        pad(&mut buf, d + 1);
        buf.push(b']');
    }
    buf
}

/// Adds `depth` indent levels after every newline.
struct Indent<'a> {
    out: &'a mut dyn Write,
    depth: usize,
}

impl Write for Indent<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        for line in buf.split_inclusive(|&b| b == b'\n') {
            self.out.write_all(line)?;
            if line.ends_with(b"\n") {
                write_indent(self.out, self.depth)?;
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.out.flush()
    }
}

/// Sorted `(value, count)` pairs and how to type their keys.
type CatArray<'a> = (&'a [(String, u64)], CatKeyKind);

/// JSON type of a published category key.
#[derive(Clone, Copy)]
enum CatKeyKind {
    Str,
    Int,
    Bool,
    Float,
}

fn kumo_cat_key_kind(ingested_type: Option<&IngestedType>) -> CatKeyKind {
    match ingested_type {
        Some(IngestedType::Bool) => CatKeyKind::Bool,
        Some(t)
            if t.is_integer_like()
                || matches!(
                    t,
                    IngestedType::Int8Array
                        | IngestedType::Int16Array
                        | IngestedType::Int32Array
                        | IngestedType::Int64Array
                ) =>
        {
            CatKeyKind::Int
        }
        Some(
            IngestedType::Float32
            | IngestedType::Float64
            | IngestedType::Float16Array
            | IngestedType::Float32Array
            | IngestedType::Float64Array,
        ) => CatKeyKind::Float,
        _ => CatKeyKind::Str,
    }
}

struct CatKey<'a>(&'a str, CatKeyKind);

impl serde::Serialize for CatKey<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let CatKey(key, kind) = *self;
        match kind {
            CatKeyKind::Int => {
                if let Ok(n) = key.parse::<i64>() {
                    return s.serialize_i64(n);
                }
            }
            CatKeyKind::Bool => match key {
                "1" | "true" => return s.serialize_bool(true),
                "0" | "false" => return s.serialize_bool(false),
                _ => {}
            },
            CatKeyKind::Float => {
                if let Some(f) = key.parse::<f64>().ok().filter(|f| f.is_finite()) {
                    return s.serialize_f64(f);
                }
            }
            CatKeyKind::Str => {}
        }
        s.serialize_str(key)
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
            hll.observe_str(&s);
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
        let hash = HyperLogLog::hash_str("hello");
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
        let hash = HyperLogLog::hash_str("rare");
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
        let capped = SharedColumnCounts::new_hashed_capped_with_max_categories(2);
        assert!(capped.observe_hashed(HyperLogLog::hash_str("a"), "a"));
        assert!(capped.observe_hashed(HyperLogLog::hash_str("b"), "b"));
        assert!(!capped.observe_hashed(HyperLogLog::hash_str("c"), "c"));
        assert!(capped.observe_hashed(HyperLogLog::hash_str("a"), "a"));
        assert!(capped.is_capped());
        assert!(!capped.is_aborted());
        assert_eq!(
            capped.drain_to_map().unwrap(),
            std::collections::HashMap::from([("a".to_owned(), 2), ("b".to_owned(), 1)])
        );

        let exhaustive = SharedColumnCounts::new_hashed_with_max_categories(1);
        assert!(exhaustive.observe_hashed(HyperLogLog::hash_str("a"), "a"));
        assert!(exhaustive.observe_hashed(HyperLogLog::hash_str("b"), "b"));
        assert!(exhaustive.is_aborted());
        assert!(!exhaustive.is_capped());
        assert!(exhaustive.drain_to_map().is_none());
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
            format_kumo_timestamp(1537401600, TimeUnit::Seconds),
            "kumo_timestamp_2018-09-20 00:00:00"
        );
    }

    #[test]
    fn test_format_kumo_timestamp_sentinels() {
        assert_eq!(
            format_kumo_timestamp(i64::MAX, TimeUnit::Seconds),
            "kumo_timestamp_NaT"
        );
        assert_eq!(
            format_kumo_timestamp(i64::MIN, TimeUnit::Seconds),
            "kumo_timestamp_NaT"
        );
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
    fn kumo_timestamps_carry_a_fraction_only_when_one_is_stored() {
        let at = |ticks, clock| format_kumo_timestamp(ticks, clock);
        assert_eq!(
            at(1_700_000_000_123_456_789, TimeUnit::Nanoseconds),
            "kumo_timestamp_2023-11-14 22:13:20.123456789"
        );
        assert_eq!(
            at(1_700_000_000_250, TimeUnit::Milliseconds),
            "kumo_timestamp_2023-11-14 22:13:20.250"
        );
        assert_eq!(
            at(1_700_000_000_000, TimeUnit::Milliseconds),
            "kumo_timestamp_2023-11-14 22:13:20"
        );
        assert_eq!(
            at(1_700_000_000, TimeUnit::Seconds),
            "kumo_timestamp_2023-11-14 22:13:20"
        );
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
    fn parse_failures_merge_without_counting_source_nulls() {
        for stype in [
            SemanticType::Numerical,
            SemanticType::Categorical,
            SemanticType::Id,
            SemanticType::Text,
            SemanticType::Timestamp,
            SemanticType::Sequence,
            SemanticType::Latlong,
            SemanticType::ImageUrl,
            SemanticType::Multicategorical,
        ] {
            let cols = make_test_cols(&[Some(stype)]);
            let mut first = FileColStats::new(&cols);
            let mut second = FileColStats::new(&cols);
            let names = vec!["value".into()];
            let types = vec![IngestedType::String];
            let key = kumo_key(
                "value",
                kumo_stype_key(&first.cols[0]),
                "NUM_PARSE_FAILURES",
                false,
            );
            let raw = first.to_json(&names, &types, None, None);
            assert_eq!(raw["value"]["parse_fail_count"], 0);
            let hosted = first.to_kumo_json(&names, &types, None, None, "v1");
            assert_eq!(hosted[&key]["value"], 0);

            first.observe_null(0);
            first.observe_parse_failure(0);
            first.observe_null(0);
            second.observe_parse_failure(0);
            second.observe_null(0);
            first.merge(&mut second);

            assert_eq!(first.cols[0].parse_fail_count(), 2);
            let raw = first.to_json(&names, &types, None, None);
            assert_eq!(raw["value"]["parse_fail_count"], 2);
            assert_eq!(raw["value"]["null_count"], 3);
            let hosted = first.to_kumo_json(&names, &types, None, None, "v1");
            assert_eq!(hosted[&key]["value"], 2);
            assert_eq!(hosted[&key]["num_entries"], 3);
        }
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
            assert_eq!(nu["approx"], json!(false));
            assert_eq!(nu["data_version"], json!("v42"));
            assert_eq!(nu["num_entries"], json!(4));
            assert!(!map.contains_key("status__kumo.categorical__NUM_UNIQUE__True"));
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
            assert_eq!(nu["approx"], json!(false));
            assert_eq!(nu["data_version"], json!("v42"));
            assert_eq!(nu["num_entries"], json!(2));
            assert!(!map.contains_key("user_id__kumo.ID__NUM_UNIQUE__True"));
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
        // Internal sentinels are normalized to 0/0 when every row is null.
        {
            let min_l = map
                .get("embedding__kumo.sequence__SEQUENCE_MIN_LENGTH__False")
                .unwrap();
            assert_eq!(min_l["value"], json!(0));
        }
        {
            let max_l = map
                .get("embedding__kumo.sequence__SEQUENCE_MAX_LENGTH__False")
                .unwrap();
            assert_eq!(max_l["value"], json!(0));
        }

        // The six supported columns each emit NUM_PARSE_FAILURES, including zero.
        for (name, sk, failures) in [
            ("age", "kumo.numerical", 0),
            ("status", "kumo.categorical", 0),
            ("user_id", "kumo.ID", 0),
            ("bio", "kumo.text", 0),
            ("created_at", "kumo.timestamp", 1),
            ("embedding", "kumo.sequence", 0),
        ] {
            assert_eq!(
                map[&kumo_key(name, sk, "NUM_PARSE_FAILURES", false)]["value"],
                failures
            );
        }
        assert_eq!(map.len(), 61);

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
    // These bypass the `config_converter` layer by constructing `FileColStats`
    // directly via `make_test_cols_with_types` with explicit `IngestedType` +
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

    #[test]
    fn id_columns_get_category_accumulator() {
        // Id columns now get a (capped) accumulator alongside Categorical:
        // some models consume their category counts and the shared map is
        // bounded by MAX_CATEGORIES, so there is no reason to skip them.
        let tc = TableCategoryCounts::new(
            CategoryCountStrategy::Hybrid,
            &[Some(SemanticType::Categorical), Some(SemanticType::Id)],
            false,
        );
        assert!(tc.columns[0].is_some());
        assert!(tc.columns[1].is_some());
    }

    #[test]
    fn id_column_category_counts_populate() {
        // An Id column's category counts populate through the normal observe
        // path (capped), rather than being skipped.
        let (mut stats, tc) = make_hybrid_stats(&[(
            Some(SemanticType::Id),
            IngestedType::String,
            None,
        )]);
        stats.observe_string(0, "u1");
        stats.observe_string(0, "u1");
        stats.observe_string(0, "u2");
        stats.flush_categories();
        assert_eq!(
            drain_counts(&tc, 0),
            std::collections::HashMap::from([("u1".into(), 2), ("u2".into(), 1)]),
        );
    }

    #[test]
    fn hybrid_small_cap_publishes_exact_counts_with_legacy_cardinality_flags() {
        let cols = make_test_cols_with_types(&[(
            Some(SemanticType::Categorical),
            IngestedType::String,
            None,
        )]);
        let table_cat = std::sync::Arc::new(TableCategoryCounts::new_with_max_categories(
            CategoryCountStrategy::Hybrid,
            &[Some(SemanticType::Categorical)],
            false,
            2,
        ));
        let mut first = FileColStats::new_with_categories(&cols, std::sync::Arc::clone(&table_cat));
        let mut second =
            FileColStats::new_with_categories(&cols, std::sync::Arc::clone(&table_cat));

        // "hot" is admitted immediately rather than living only in the first
        // worker's local cache. A third distinct value closes admissions, but
        // both workers keep contributing exact deltas for retained values.
        first.observe_string(0, "hot");
        second.observe_string(0, "cold");
        second.observe_string(0, "excluded");
        first.observe_string(0, "hot");
        second.observe_string(0, "cold");
        first.observe_string(0, "hot");
        first.flush_categories();
        second.flush_categories();
        first.merge(&mut second);

        let output = first.to_kumo_json(
            &["feature".to_owned()],
            &[IngestedType::String],
            Some(&table_cat),
            None,
            "v1",
        );
        assert_eq!(
            output["feature__kumo.categorical__CATEGORY_COUNTS__False"]["value"],
            json!([["hot", 3], ["cold", 2]])
        );
        assert!(!output.contains_key("feature__kumo.categorical__CATEGORY_COUNTS_CAPPED__False"));
        assert!(!output.contains_key("feature__kumo.categorical__NUM_UNIQUE__True"));
        let num_unique = &output["feature__kumo.categorical__NUM_UNIQUE__False"];
        assert_eq!(num_unique["value"], json!(3));
        assert_eq!(num_unique["approx"], json!(false));
        assert!(!output.contains_key("feature__kumo.categorical__UNIQUE_FRACTION__True"));
        let unique_fraction = &output["feature__kumo.categorical__UNIQUE_FRACTION__False"];
        assert_eq!(unique_fraction["value"], json!(0.5));
        assert_eq!(unique_fraction["approx"], json!(false));
    }

    #[test]
    fn hybrid_multicat_small_cap_publishes_exact_token_counts_and_cap_status() {
        let cols = make_test_cols_with_types(&[(
            Some(SemanticType::Multicategorical),
            IngestedType::String,
            None,
        )]);
        let table_cat = std::sync::Arc::new(TableCategoryCounts::new_with_max_categories(
            CategoryCountStrategy::Hybrid,
            &[Some(SemanticType::Multicategorical)],
            false,
            2,
        ));
        let mut stats =
            FileColStats::new_with_categories(&cols, std::sync::Arc::clone(&table_cat));

        stats.observe_string(0, "red");
        stats.observe_string(0, "blue");
        stats.observe_string(0, "excluded");
        stats.observe_string(0, "red");
        stats.observe_string(0, "blue");
        stats.observe_string(0, "red");
        stats.flush_categories();

        let output = stats.to_kumo_json(
            &["tags".to_owned()],
            &[IngestedType::String],
            Some(&table_cat),
            None,
            "v1",
        );
        assert_eq!(
            output["tags__kumo.multicategorical__MULTI_CATEGORY_COUNTS__False"]["value"],
            json!([["red", 3], ["blue", 2]])
        );
        assert!(!output
            .contains_key("tags__kumo.multicategorical__MULTI_CATEGORY_COUNTS_CAPPED__False"));
        let num_unique = &output["tags__kumo.multicategorical__NUM_UNIQUE_MULTI__True"];
        assert_eq!(num_unique["value"], json!(3));
        assert_eq!(num_unique["approx"], json!(true));
    }

    #[test]
    fn exactly_at_hybrid_cap_is_exhaustive_until_a_value_is_excluded() {
        let cols =
            make_test_cols_with_types(&[(Some(SemanticType::Id), IngestedType::Int64, None)]);
        let table_cat = std::sync::Arc::new(TableCategoryCounts::new_with_max_categories(
            CategoryCountStrategy::Hybrid,
            &[Some(SemanticType::Id)],
            false,
            2,
        ));
        let mut stats =
            FileColStats::new_with_categories(&cols, std::sync::Arc::clone(&table_cat));
        stats.observe_i64(0, 10);
        stats.observe_i64(0, 20);
        stats.observe_i64(0, 10);
        stats.flush_categories();

        let output = stats.to_kumo_json(
            &["id_feature".to_owned()],
            &[IngestedType::Int64],
            Some(&table_cat),
            None,
            "v1",
        );
        assert_eq!(
            output["id_feature__kumo.ID__CATEGORY_COUNTS__False"]["value"],
            json!([[10, 2], [20, 1]])
        );
        assert!(!output.contains_key("id_feature__kumo.ID__CATEGORY_COUNTS_CAPPED__False"));
        assert_eq!(
            output["id_feature__kumo.ID__NUM_UNIQUE__False"]["value"],
            json!(2)
        );
    }

    #[test]
    fn scalar_float_signed_zero_uses_one_mapper_identity() {
        let cols = make_test_cols_with_types(&[(
            Some(SemanticType::Categorical),
            IngestedType::Float64,
            None,
        )]);
        let table_cat = std::sync::Arc::new(TableCategoryCounts::new_with_max_categories(
            CategoryCountStrategy::Hybrid,
            &[Some(SemanticType::Categorical)],
            false,
            2,
        ));
        let mut stats =
            FileColStats::new_with_categories(&cols, std::sync::Arc::clone(&table_cat));

        stats.observe_f64(0, 0.0);
        stats.observe_f64(0, -0.0);
        stats.flush_categories();

        let output = stats.to_kumo_json(
            &["float_feature".to_owned()],
            &[IngestedType::Float64],
            Some(&table_cat),
            None,
            "v1",
        );
        assert_eq!(
            output["float_feature__kumo.categorical__CATEGORY_COUNTS__False"]["value"],
            json!([[0.0, 2]])
        );
        assert_eq!(
            output["float_feature__kumo.categorical__NUM_UNIQUE__False"]["value"],
            json!(1)
        );
    }

    #[test]
    #[should_panic(expected = "Hybrid category counts require row_subsample=false")]
    fn hybrid_subsampling_is_rejected_for_exact_feature_counts() {
        TableCategoryCounts::new(
            CategoryCountStrategy::Hybrid,
            &[Some(SemanticType::Categorical)],
            true,
        );
    }

    #[test]
    fn graduation_id_num_unique_falls_back_to_hll_not_empty_map() {
        // Under Graduation a single value seen once stays below min_occ and never
        // graduates, so the drained category map is empty. NUM_UNIQUE must fall
        // back to the HLL estimate (~1, clamped by total=1), not the empty map's
        // length (0). Non-subsampled Hybrid stays on the exact map length.
        let cols =
            make_test_cols_with_types(&[(Some(SemanticType::Id), IngestedType::String, None)]);
        let stypes: Vec<Option<SemanticType>> = cols.iter().map(|c| c.semantic_type).collect();
        let table_cat = std::sync::Arc::new(TableCategoryCounts::new(
            CategoryCountStrategy::Graduation,
            &stypes,
            false,
        ));
        let mut stats =
            FileColStats::new_with_categories(&cols, std::sync::Arc::clone(&table_cat));
        stats.observe_string(0, "solo");
        stats.flush_categories();

        let col_names = vec!["uid".to_string()];
        let ingested_types = vec![IngestedType::String];
        let map = stats.to_kumo_json(&col_names, &ingested_types, Some(&table_cat), None, "v1");
        let nu = map.get("uid__kumo.ID__NUM_UNIQUE__False").unwrap();
        assert_eq!(
            nu["value"].as_u64().unwrap(),
            1,
            "Graduation NUM_UNIQUE must be the HLL estimate (~1), not the empty map length (0)",
        );
        assert_eq!(nu["approx"], json!(false));
        assert_eq!(nu["data_version"], json!("v1"));
        assert_eq!(nu["num_entries"], json!(1));
        assert!(!map.contains_key("uid__kumo.ID__NUM_UNIQUE__True"));
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
        let v1 = build_int_list_view(&[Some(10), Some(20)]);
        let v2 = build_int_list_view(&[Some(10), Some(-3)]);
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
        let view = build_native_int_list_view(&[Some(5), None, Some(7)]);
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
        let view = build_int_list_view(&[Some(0), Some(i64::MIN), Some(i64::MAX)]);
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

    // Sequence stype + intlist dtype: observe_int64_list must update the
    // Sequence collector's count/min_len/max_len. Regression for the gap
    // that surfaced as "Missing embedding statistics. Embedding column
    // 'array_int_col' is missing sequence length statistics" during
    // Databricks data-variety ingest.
    #[test]
    fn col_stats_sequence_intarray_updates_min_max_length() {
        let v1 = build_int_list_view(&[Some(1), Some(2), Some(3)]);
        let v2 = build_int_list_view(&[Some(10), Some(20)]);
        let v3 = build_int_list_view(&[Some(7), Some(8), Some(9), Some(10)]);
        let (mut stats, _tc) =
            make_hybrid_stats(&[(Some(SemanticType::Sequence), IngestedType::Int64Array, None)]);
        stats.observe_int64_list(0, &v1);
        stats.observe_int64_list(0, &v2);
        stats.observe_int64_list(0, &v3);
        match &stats.cols[0] {
            ColumnStatsCollector::Sequence {
                count,
                null_count,
                min_len,
                max_len,
                ..
            } => {
                assert_eq!(*count, 3);
                assert_eq!(*null_count, 0);
                assert_eq!(*min_len, 2);
                assert_eq!(*max_len, 4);
            }
            _ => panic!("expected Sequence collector"),
        }
    }

    #[test]
    fn col_stats_sequence_floatarray_derives_width_from_non_empty_rows() {
        let empty = build_float64_list_view(&[]);
        let populated = build_float64_list_view(&[Some(1.0), Some(2.0), Some(3.0)]);
        let (mut stats, _tc) = make_hybrid_stats(&[(
            Some(SemanticType::Sequence),
            IngestedType::Float64Array,
            None,
        )]);
        let (mut populated_stats, _tc) = make_hybrid_stats(&[(
            Some(SemanticType::Sequence),
            IngestedType::Float64Array,
            None,
        )]);

        stats.observe_float_list(0, &empty);
        populated_stats.observe_float_list(0, &populated);
        stats.merge(&mut populated_stats);

        match &stats.cols[0] {
            ColumnStatsCollector::Sequence {
                count,
                null_count,
                min_len,
                max_len,
                ..
            } => {
                assert_eq!(*count, 1);
                assert_eq!(*null_count, 1);
                assert_eq!(*min_len, 3);
                assert_eq!(*max_len, 3);
            }
            _ => panic!("expected Sequence collector"),
        }

        let map = stats.to_kumo_json(
            &["embedding".to_string()],
            &[IngestedType::Float64Array],
            None,
            None,
            "v1",
        );
        assert_eq!(
            map["embedding__kumo.sequence__COUNT__False"]["value"],
            json!(2)
        );
        assert_eq!(
            map["embedding__kumo.sequence__NUM_NA__False"]["value"],
            json!(1)
        );
        assert_eq!(
            map["embedding__kumo.sequence__SEQUENCE_MIN_LENGTH__False"]["value"],
            json!(3)
        );
        assert_eq!(
            map["embedding__kumo.sequence__SEQUENCE_MAX_LENGTH__False"]["value"],
            json!(3)
        );
    }

    #[test]
    fn col_stats_sequence_intarray_treats_empty_rows_as_missing() {
        let empty = build_int_list_view(&[]);
        let populated = build_int_list_view(&[Some(1), Some(2)]);
        let (mut stats, _tc) =
            make_hybrid_stats(&[(Some(SemanticType::Sequence), IngestedType::Int64Array, None)]);

        stats.observe_int64_list(0, &empty);
        stats.observe_int64_list(0, &populated);

        match &stats.cols[0] {
            ColumnStatsCollector::Sequence {
                count,
                null_count,
                min_len,
                max_len,
                ..
            } => {
                assert_eq!(*count, 1);
                assert_eq!(*null_count, 1);
                assert_eq!(*min_len, 2);
                assert_eq!(*max_len, 2);
            }
            _ => panic!("expected Sequence collector"),
        }
    }

    #[test]
    fn col_stats_sequence_all_empty_normalizes_emitted_lengths_across_partitions() {
        let empty = build_float64_list_view(&[]);
        let (mut stats, _tc) = make_hybrid_stats(&[(
            Some(SemanticType::Sequence),
            IngestedType::Float64Array,
            None,
        )]);
        let (mut other, _tc) = make_hybrid_stats(&[(
            Some(SemanticType::Sequence),
            IngestedType::Float64Array,
            None,
        )]);

        stats.observe_float_list(0, &empty);
        other.observe_float_list(0, &empty);
        other.observe_null(0);
        stats.merge(&mut other);

        match &stats.cols[0] {
            ColumnStatsCollector::Sequence {
                count,
                null_count,
                min_len,
                max_len,
                ..
            } => {
                assert_eq!(*count, 0);
                assert_eq!(*null_count, 3);
                assert_eq!(*min_len, usize::MAX);
                assert_eq!(*max_len, 0);
            }
            _ => panic!("expected Sequence collector"),
        }

        let summary = stats.to_json(
            &["embedding".to_string()],
            &[IngestedType::Float64Array],
            None,
            None,
        );
        assert_eq!(summary["embedding"]["count"], json!(3));
        assert_eq!(summary["embedding"]["null_count"], json!(3));
        assert_eq!(summary["embedding"]["min_len"], json!(0));
        assert_eq!(summary["embedding"]["max_len"], json!(0));

        let map = stats.to_kumo_json(
            &["embedding".to_string()],
            &[IngestedType::Float64Array],
            None,
            None,
            "v1",
        );
        assert_eq!(
            map["embedding__kumo.sequence__COUNT__False"]["value"],
            json!(3)
        );
        assert_eq!(
            map["embedding__kumo.sequence__NUM_NA__False"]["value"],
            json!(3)
        );
        assert_eq!(
            map["embedding__kumo.sequence__NA_FRACTION__False"]["value"],
            json!(1.0)
        );
        assert_eq!(
            map["embedding__kumo.sequence__SEQUENCE_MIN_LENGTH__False"]["value"],
            json!(0)
        );
        assert_eq!(
            map["embedding__kumo.sequence__SEQUENCE_MAX_LENGTH__False"]["value"],
            json!(0)
        );
    }

    #[test]
    fn col_stats_sequence_empty_does_not_hide_mixed_nonzero_widths() {
        let empty = build_float64_list_view(&[]);
        let width_32 = build_float64_list_view(&vec![Some(1.0); 32]);
        let width_31 = build_float64_list_view(&vec![Some(1.0); 31]);
        let (mut stats, _tc) = make_hybrid_stats(&[(
            Some(SemanticType::Sequence),
            IngestedType::Float64Array,
            None,
        )]);

        stats.observe_float_list(0, &empty);
        stats.observe_float_list(0, &width_32);
        stats.observe_float_list(0, &width_31);

        let map = stats.to_kumo_json(
            &["embedding".to_string()],
            &[IngestedType::Float64Array],
            None,
            None,
            "v1",
        );
        assert_eq!(
            map["embedding__kumo.sequence__COUNT__False"]["value"],
            json!(3)
        );
        assert_eq!(
            map["embedding__kumo.sequence__NUM_NA__False"]["value"],
            json!(1)
        );
        assert_eq!(
            map["embedding__kumo.sequence__SEQUENCE_MIN_LENGTH__False"]["value"],
            json!(31)
        );
        assert_eq!(
            map["embedding__kumo.sequence__SEQUENCE_MAX_LENGTH__False"]["value"],
            json!(32)
        );
    }

    #[test]
    fn col_stats_latlong_rejects_non_length_two_and_inner_nulls() {
        let valid = build_float64_list_view(&[Some(37.7749), Some(-122.4194)]);
        let wrong_len = build_float64_list_view(&[Some(37.7749), Some(-122.4194), Some(1.0)]);
        let inner_null = build_native_float64_list_view(&[Some(37.7749), None]);
        let (mut stats, _tc) = make_hybrid_stats(&[(
            Some(SemanticType::Latlong),
            IngestedType::Float64Array,
            None,
        )]);

        stats.observe_float_list(0, &valid);
        stats.observe_float_list(0, &wrong_len);
        stats.observe_float_list(0, &inner_null);

        match &stats.cols[0] {
            ColumnStatsCollector::Latlong {
                count,
                null_count,
                invalid_count,
                min_len,
                max_len,
                ..
            } => {
                assert_eq!(*count, 1);
                assert_eq!(*null_count, 0);
                assert_eq!(*invalid_count, 2);
                assert_eq!(*min_len, 2);
                assert_eq!(*max_len, 2);
            }
            _ => panic!("expected Latlong collector"),
        }
    }

    #[test]
    fn col_stats_latlong_emits_length_two_for_kumo_json() {
        let valid = build_int_list_view(&[Some(37), Some(-122)]);
        let (mut stats, _tc) =
            make_hybrid_stats(&[(Some(SemanticType::Latlong), IngestedType::Int64Array, None)]);
        stats.observe_int64_list(0, &valid);
        stats.observe_parse_failure(0);
        stats.observe_null(0);
        stats.observe_parse_failure(0);
        stats.observe_null(0);

        let col_names = vec!["location".to_string()];
        let ingested_types = vec![IngestedType::Int64Array];
        let map = stats.to_kumo_json(&col_names, &ingested_types, None, None, "v1");

        let count = map
            .get("location__kumo.latlong__COUNT__False")
            .expect("count key present");
        assert_eq!(count["value"], json!(3));

        let num_na = map
            .get("location__kumo.latlong__NUM_NA__False")
            .expect("num-na key present");
        assert_eq!(num_na["value"], json!(2));

        let min_len = map
            .get("location__kumo.latlong__SEQUENCE_MIN_LENGTH__False")
            .expect("min length key present");
        assert_eq!(min_len["value"], json!(2));

        let max_len = map
            .get("location__kumo.latlong__SEQUENCE_MAX_LENGTH__False")
            .expect("max length key present");
        assert_eq!(max_len["value"], json!(2));

        let invalid = map
            .get("location__kumo.latlong__INVALID_FRACTION__False")
            .expect("invalid-fraction key present");
        let invalid_fraction = invalid["value"].as_f64().unwrap();
        assert!((invalid_fraction - (2.0 / 3.0)).abs() < 1e-12);
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

    fn build_int_list_view(elems: &[Option<i64>]) -> IntListView {
        // Parsed values cannot represent null elements. Use
        // `build_native_int_list_view` for Arrow null-bit tests.
        let owned: Vec<i64> = elems
            .iter()
            .map(|e| e.expect("Parsed IntListView cannot represent null elements"))
            .collect();
        let end = owned.len();
        IntListView {
            start: 0,
            end,
            values: diskgraph_rowpack::IntListValues::Parsed(owned),
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

    fn build_native_int_list_view(elems: &[Option<i64>]) -> IntListView {
        let end = elems.len();
        let array = arrow_array::Int64Array::from(elems.to_vec());
        IntListView {
            start: 0,
            end,
            values: diskgraph_rowpack::IntListValues::I64(
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

    const ODD_NAME: &str = "a\"quoted\nname";

    type BuiltTable = (
        String,
        Vec<ColumnSpec>,
        FileColStats,
        Arc<TableCategoryCounts>,
    );

    /// Deterministic multi-table input covering every category-count shape.
    fn build_publish_input(seed: u64) -> Vec<BuiltTable> {
        let mut rng = seed;
        let mut next = move |n: u64| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng % n
        };
        let cat = Some(SemanticType::Categorical);
        let entries = [
            (cat, IngestedType::String, None),
            (Some(SemanticType::Id), IngestedType::Int64, None),
            (
                Some(SemanticType::Multicategorical),
                IngestedType::String,
                Some(","),
            ),
            (cat, IngestedType::Bool, None),
            (cat, IngestedType::Float64, None),
            (cat, IngestedType::String, None), // collisions
            (cat, IngestedType::String, None), // never observed
            (Some(SemanticType::Text), IngestedType::String, None),
            (Some(SemanticType::Numerical), IngestedType::Float64, None),
            (cat, IngestedType::String, None), // every value ties on count
        ];
        let tables = [
            ("users", 64, Some("v1")),
            ("orders", 1_000, Some("v2")),
            ("aux", 64, None),
        ];
        tables
            .into_iter()
            .map(|(name, cap, _)| {
                let mut cols = make_test_cols_with_types(&entries);
                cols[0].name = ODD_NAME.to_string();
                let stypes: Vec<_> = cols.iter().map(|c| c.semantic_type).collect();
                let tc = Arc::new(TableCategoryCounts::new_with_max_categories(
                    CategoryCountStrategy::Hybrid,
                    &stypes,
                    false,
                    cap,
                ));
                let mut merged = FileColStats::new_with_categories(&cols, Arc::clone(&tc));
                for _ in 0..3 {
                    let mut file = FileColStats::new_with_categories(&cols, Arc::clone(&tc));
                    for row in 0..400 {
                        let k = next(150);
                        file.observe_string(0, &format!("v{k}\u{e9}\t"));
                        file.observe_i64(1, k as i64 - 75);
                        file.observe_string(2, &format!("t{},t{}", next(90), next(90)));
                        file.observe_string(3, ["true", "false", "1", "x"][next(4) as usize]);
                        file.observe_string(4, &format!("{}.5", next(40)));
                        file.observe_string(5, &format!("c{}", next(30)));
                        file.observe_string(7, "some text");
                        file.observe_f64(8, next(1000) as f64);
                        file.observe_string(9, &format!("tie{}", row % 40));
                        if next(10) == 0 {
                            file.observe_null(0);
                        }
                    }
                    file.flush_categories();
                    merged.merge(&mut file);
                }
                let shared = tc.columns[5].as_ref().unwrap();
                for (value, n) in [("collide-a", 3), ("collide-b", 5), ("collide-c", 5)] {
                    for _ in 0..n {
                        shared.observe_hashed(0x0C01_11DE, value);
                    }
                }
                (name.to_string(), cols, merged, tc)
            })
            .collect()
    }

    fn local_target(dir: &tempfile::TempDir) -> crate::io::OutputTarget {
        crate::io::OutputTarget::parse(dir.path().to_str().unwrap(), crate::io::Dest::Local)
            .unwrap()
    }

    fn into_publish_tables<'a>(
        input: Vec<BuiltTable>,
        versions: [Option<&str>; 3],
        text_stats: &'a HashMap<String, TextSampleStats>,
    ) -> Vec<ColStatsTable<'a>> {
        input
            .into_iter()
            .zip(versions)
            .map(|((name, cols, stats, tc), dv)| ColStatsTable {
                name,
                stats,
                column_names: cols.iter().map(|c| c.name.clone()).collect(),
                ingested_types: cols.iter().map(|c| c.ingested_dtype.clone()).collect(),
                table_cat: Some(tc),
                text_sample_stats: Some(text_stats),
                data_version: dv.map(str::to_string),
            })
            .collect()
    }

    #[test]
    fn write_col_stats_matches_serde_value_output() {
        let versions = [Some("v1"), Some("v2"), None];
        let text_stats: HashMap<String, TextSampleStats> = [(
            "c7".to_string(),
            TextSampleStats {
                avg_tokens: 2.0,
                max_tokens: 2,
                non_char_fraction: 0.1,
                string_truncation_scale: 1.0,
                glove_overlap: 0.5,
            },
        )]
        .into();

        for seed in [1, 7, 42, 1234, 99_991] {
            let mut expected_root = serde_json::Map::new();
            let mut expected_files = Vec::new();
            for ((name, cols, mut stats, tc), dv) in
                build_publish_input(seed).into_iter().zip(versions)
            {
                let names: Vec<String> = cols.iter().map(|c| c.name.clone()).collect();
                let types: Vec<_> = cols.iter().map(|c| c.ingested_dtype.clone()).collect();
                expected_root.insert(
                    name.clone(),
                    stats.to_json(&names, &types, Some(&tc), Some(&text_stats)),
                );
                if let Some(dv) = dv {
                    let full = stats.to_kumo_json(&names, &types, Some(&tc), Some(&text_stats), dv);
                    let min = minify_kumo_stats(&full, MIN_CATEGORY_COUNTS);
                    expected_files.push((format!("col_stats_full/{name}.json"), full));
                    expected_files.push((format!("col_stats_min/{name}.json"), min));
                }
            }

            let dir = tempfile::tempdir().unwrap();
            let tables = into_publish_tables(build_publish_input(seed), versions, &text_stats);
            write_col_stats(&local_target(&dir), tables).unwrap();

            // Every published shape is present.
            let users = &expected_root["users"];
            assert_eq!(users[ODD_NAME]["category_counts_capped"], true);
            assert_eq!(
                users[ODD_NAME]["category_counts"].as_array().unwrap().len(),
                64
            );
            assert_eq!(users["c6"]["category_counts"], json!([]));
            let collided = users["c5"]["category_counts"].as_array().unwrap();
            assert!(collided.iter().any(|p| p[0] == "collide-a"));
            let ties = users["c9"]["category_counts"].as_array().unwrap();
            assert_eq!(ties.len(), 40);
            assert!(ties.iter().all(|p| p[1] == ties[0][1]));
            let orders = &expected_root["orders"];
            assert_eq!(orders[ODD_NAME]["category_counts_capped"], false);
            assert!(orders["c1"]["category_counts"][0][0].is_i64());
            assert_eq!(expected_files.len(), 4);

            let read = |rel: &str| std::fs::read(dir.path().join(rel)).unwrap();
            assert_eq!(
                read("col_stats.json"),
                serde_json::to_vec_pretty(&JsonValue::Object(expected_root)).unwrap(),
                "seed {seed}"
            );
            for (rel, map) in expected_files {
                assert_eq!(
                    read(&format!("kumo_col_stats/{rel}")),
                    serde_json::to_vec(&map).unwrap(),
                    "seed {seed} {rel}"
                );
            }
            assert!(!dir
                .path()
                .join("kumo_col_stats/col_stats_full/aux.json")
                .exists());
        }
    }

    #[test]
    fn write_col_stats_fails_while_counts_are_shared() {
        let text_stats = HashMap::new();
        let input = build_publish_input(3);
        let extra_holder = Arc::clone(&input[0].3);
        let dir = tempfile::tempdir().unwrap();
        let tables = into_publish_tables(input, [Some("v1"), Some("v2"), None], &text_stats);
        let err = write_col_stats(&local_target(&dir), tables).unwrap_err();
        assert!(err.to_string().contains("still shared"), "{err}");
        drop(extra_holder);
    }

    #[test]
    fn write_col_stats_with_no_tables_writes_empty_object() {
        let dir = tempfile::tempdir().unwrap();
        write_col_stats(&local_target(&dir), Vec::new()).unwrap();
        assert_eq!(
            std::fs::read(dir.path().join("col_stats.json")).unwrap(),
            b"{}"
        );
    }
}
