//! Shared streaming accumulators used by both col-stats (ingestion) and
//! TT-gen (training table generation).
//!
//! These are pure data structures with no crate-specific dependencies beyond
//! `gxhash` (for HyperLogLog hashing).

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwap;
use dashmap::DashMap;
use parking_lot::Mutex;
use serde_json::{json, Value as JsonValue};

// ---------------------------------------------------------------------------
// HyperLogLog cardinality estimator
// ---------------------------------------------------------------------------

/// HyperLogLog cardinality estimator with 2^14 = 16384 registers.
///
/// Hash function: gxhash (hardware-accelerated).
pub struct HyperLogLog {
    registers: Box<[u8; Self::M]>,
}

impl Default for HyperLogLog {
    fn default() -> Self {
        Self::new()
    }
}

impl HyperLogLog {
    const P: u32 = 14;
    const M: usize = 1 << Self::P; // 16384
    /// Fixed seed for deterministic hashing. All HLL instances must use the
    /// same hash function for register-max merge to be valid.
    const HASH_SEED: i64 = 0x517c_c1b7_2722_0a95_u64 as i64;

    pub fn new() -> Self {
        Self {
            registers: Box::new([0u8; Self::M]),
        }
    }

    /// Compute a u64 hash for a byte slice using the fixed HLL seed.
    /// Used by both HLL and category counts to share a single hash computation.
    #[inline]
    pub fn hash_bytes(bytes: &[u8]) -> u64 {
        let mut hasher = gxhash::GxHasher::with_seed(Self::HASH_SEED);
        hasher.write(bytes);
        hasher.finish()
    }

    /// Compute a u64 hash for an i64 value using the fixed HLL seed.
    #[inline]
    pub fn hash_i64(value: i64) -> u64 {
        let mut hasher = gxhash::GxHasher::with_seed(Self::HASH_SEED);
        hasher.write_i64(value);
        hasher.finish()
    }

    /// Compute a u64 hash for a f64 value using the fixed HLL seed.
    #[inline]
    pub fn hash_f64(value: f64) -> u64 {
        let mut hasher = gxhash::GxHasher::with_seed(Self::HASH_SEED);
        hasher.write_u64(value.to_bits());
        hasher.finish()
    }

    /// Hash and observe a byte slice.
    #[inline]
    pub fn observe_bytes(&mut self, bytes: &[u8]) {
        self.observe_hash(Self::hash_bytes(bytes));
    }

    /// Observe an i64 value.
    #[inline]
    pub fn observe_i64(&mut self, value: i64) {
        self.observe_hash(Self::hash_i64(value));
    }

    /// Observe a f64 value (uses bit representation for hashing).
    #[inline]
    pub fn observe_f64(&mut self, value: f64) {
        self.observe_hash(Self::hash_f64(value));
    }

    #[inline(always)]
    pub fn observe_hash(&mut self, hash: u64) {
        let idx = (hash >> (64 - Self::P)) as usize;
        let remaining = (hash << Self::P) | (1 << (Self::P - 1));
        let rho = remaining.leading_zeros() as u8 + 1;
        if rho > self.registers[idx] {
            self.registers[idx] = rho;
        }
    }

    /// Estimate cardinality using the HLL algorithm with bias correction.
    pub fn estimate(&self) -> f64 {
        let m = Self::M as f64;
        let alpha = match Self::M {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / m),
        };
        let sum: f64 = self
            .registers
            .iter()
            .map(|&r| 2.0_f64.powi(-(r as i32)))
            .sum();
        let raw = alpha * m * m / sum;

        // Small range correction
        if raw <= 2.5 * m {
            let zeros = self.registers.iter().filter(|&&r| r == 0).count();
            if zeros > 0 {
                return m * (m / zeros as f64).ln();
            }
        }
        raw
    }

    /// Merge another HLL into this one (element-wise max of registers).
    pub fn merge(&mut self, other: &HyperLogLog) {
        for (a, &b) in self.registers.iter_mut().zip(other.registers.iter()) {
            if b > *a {
                *a = b;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Welford online accumulator (mean, variance, skewness, kurtosis, min, max)
// ---------------------------------------------------------------------------

/// Streaming accumulator for mean, variance, std_dev, skewness, kurtosis,
/// min, and max using Welford's online algorithm.
///
/// Supports parallel merge via Chan et al.'s formula.
pub struct WelfordAccumulator {
    pub count: u64,
    pub m1: f64,
    pub m2: f64,
    pub m3: f64,
    pub m4: f64,
    pub min: f64,
    pub max: f64,
}

impl Default for WelfordAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl WelfordAccumulator {
    pub fn new() -> Self {
        Self {
            count: 0,
            m1: 0.0,
            m2: 0.0,
            m3: 0.0,
            m4: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Observe a single value. NaN/Inf values are silently skipped.
    #[inline]
    pub fn observe(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        let n1 = self.count;
        self.count += 1;
        let n = self.count as f64;
        let delta = value - self.m1;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * n1 as f64;

        self.m1 += delta_n;
        self.m4 += term1 * delta_n2 * (n * n - 3.0 * n + 3.0) + 6.0 * delta_n2 * self.m2
            - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;

        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }

    pub fn mean(&self) -> f64 {
        self.m1
    }

    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / (self.count - 1) as f64
    }

    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    pub fn skewness(&self) -> f64 {
        if self.m2 < 1e-10 {
            return 0.0;
        }
        let n = self.count as f64;
        (n.sqrt() * self.m3) / self.m2.powf(1.5)
    }

    pub fn kurtosis(&self) -> f64 {
        if self.m2 < 1e-10 {
            return 0.0;
        }
        let n = self.count as f64;
        (n * self.m4) / (self.m2 * self.m2) - 3.0
    }

    /// Merge another accumulator into this one (Chan et al. parallel formula).
    pub fn merge(&mut self, other: &WelfordAccumulator) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = WelfordAccumulator {
                count: other.count,
                m1: other.m1,
                m2: other.m2,
                m3: other.m3,
                m4: other.m4,
                min: other.min,
                max: other.max,
            };
            return;
        }
        let na = self.count as f64;
        let nb = other.count as f64;
        let n = na + nb;
        let delta = other.m1 - self.m1;
        let delta2 = delta * delta;
        let delta3 = delta2 * delta;
        let delta4 = delta2 * delta2;

        let new_m1 = (na * self.m1 + nb * other.m1) / n;
        let new_m2 = self.m2 + other.m2 + delta2 * na * nb / n;
        let new_m3 = self.m3
            + other.m3
            + delta3 * na * nb * (na - nb) / (n * n)
            + 3.0 * delta * (na * other.m2 - nb * self.m2) / n;
        let new_m4 = self.m4
            + other.m4
            + delta4 * na * nb * (na * na - na * nb + nb * nb) / (n * n * n)
            + 6.0 * delta2 * (na * na * other.m2 + nb * nb * self.m2) / (n * n)
            + 4.0 * delta * (na * other.m3 - nb * self.m3) / n;

        self.count += other.count;
        self.m1 = new_m1;
        self.m2 = new_m2;
        self.m3 = new_m3;
        self.m4 = new_m4;
        if other.min < self.min {
            self.min = other.min;
        }
        if other.max > self.max {
            self.max = other.max;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- WelfordAccumulator tests --

    #[test]
    fn test_welford_basic() {
        let mut w = WelfordAccumulator::new();
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            w.observe(v);
        }
        assert_eq!(w.count, 5);
        assert!((w.mean() - 3.0).abs() < 1e-10);
        assert!((w.variance() - 2.5).abs() < 1e-10);
        assert!((w.std_dev() - 2.5f64.sqrt()).abs() < 1e-10);
        assert_eq!(w.min, 1.0);
        assert_eq!(w.max, 5.0);
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
        assert_eq!(w.std_dev(), 0.0);
        assert_eq!(w.skewness(), 0.0);
        assert_eq!(w.kurtosis(), 0.0);
    }

    #[test]
    fn test_welford_merge() {
        let mut a = WelfordAccumulator::new();
        for v in [1.0, 2.0, 3.0] {
            a.observe(v);
        }
        let mut b = WelfordAccumulator::new();
        for v in [4.0, 5.0] {
            b.observe(v);
        }
        let mut combined = WelfordAccumulator::new();
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            combined.observe(v);
        }
        a.merge(&b);
        assert_eq!(a.count, combined.count);
        assert!((a.mean() - combined.mean()).abs() < 1e-10);
        assert!((a.variance() - combined.variance()).abs() < 1e-10);
        assert!((a.std_dev() - combined.std_dev()).abs() < 1e-10);
        assert_eq!(a.min, combined.min);
        assert_eq!(a.max, combined.max);
    }

    #[test]
    fn test_welford_merge_empty() {
        let mut a = WelfordAccumulator::new();
        a.observe(5.0);
        let b = WelfordAccumulator::new();
        a.merge(&b);
        assert_eq!(a.count, 1);
        assert!((a.mean() - 5.0).abs() < 1e-10);

        let mut c = WelfordAccumulator::new();
        let mut d = WelfordAccumulator::new();
        d.observe(7.0);
        c.merge(&d);
        assert_eq!(c.count, 1);
        assert!((c.mean() - 7.0).abs() < 1e-10);
    }

    // -- HyperLogLog tests --

    #[test]
    fn test_hll_cardinality() {
        let mut hll = HyperLogLog::new();
        for i in 0..10_000i64 {
            hll.observe_i64(i);
        }
        let est = hll.estimate();
        // Within 5% of true value
        assert!(
            (est - 10_000.0).abs() / 10_000.0 < 0.05,
            "HLL estimate {} too far from 10000",
            est
        );
    }

    #[test]
    fn test_hll_strings() {
        let mut hll = HyperLogLog::new();
        for i in 0..5_000 {
            hll.observe_bytes(format!("item_{}", i).as_bytes());
        }
        let est = hll.estimate();
        assert!(
            (est - 5_000.0).abs() / 5_000.0 < 0.05,
            "HLL string estimate {} too far from 5000",
            est
        );
    }

    #[test]
    fn test_hll_merge() {
        let mut a = HyperLogLog::new();
        let mut b = HyperLogLog::new();
        for i in 0..5_000i64 {
            a.observe_i64(i);
        }
        for i in 5_000..10_000i64 {
            b.observe_i64(i);
        }
        a.merge(&b);
        let est = a.estimate();
        assert!(
            (est - 10_000.0).abs() / 10_000.0 < 0.05,
            "HLL merge estimate {} too far from 10000",
            est
        );
    }

    #[test]
    fn test_hll_empty() {
        let hll = HyperLogLog::new();
        assert_eq!(hll.estimate(), 0.0);
    }

    #[test]
    fn test_hll_duplicates() {
        let mut hll = HyperLogLog::new();
        // Insert same value many times
        for _ in 0..10_000 {
            hll.observe_i64(42);
        }
        let est = hll.estimate();
        assert!(est < 2.0, "HLL should estimate ~1 for duplicates, got {}", est);
    }

    #[test]
    fn test_shared_counts_cap_policy() {
        let capped = SharedColumnCounts::new_hashed();
        capped.unique_count.store(MAX_CATEGORIES, Ordering::Relaxed);
        capped.record_new_unique();
        assert!(capped.is_aborted());

        let uncapped = SharedColumnCounts::new_hashed_uncapped();
        uncapped
            .unique_count
            .store(MAX_CATEGORIES, Ordering::Relaxed);
        uncapped.record_new_unique();
        assert!(!uncapped.is_aborted());

        let capped_grad = SharedColumnCounts::new_graduation();
        capped_grad
            .unique_count
            .store(MAX_CATEGORIES, Ordering::Relaxed);
        capped_grad.record_new_unique();
        assert!(capped_grad.is_aborted());

        let uncapped_grad = SharedColumnCounts::new_graduation_uncapped();
        uncapped_grad
            .unique_count
            .store(MAX_CATEGORIES, Ordering::Relaxed);
        uncapped_grad.record_new_unique();
        assert!(!uncapped_grad.is_aborted());
    }

    #[test]
    fn test_shared_counts_string_byte_budget() {
        let budget = Arc::new(CategoryStringBudget::new(5_000));
        let counts = SharedColumnCounts::new_hashed_with_string_budget(Arc::clone(&budget));
        counts.observe_hashed(1, "first");
        assert!(!counts.is_aborted());
        counts.observe_hashed(2, "second");
        assert!(counts.is_aborted());
        assert!(budget.used_bytes() <= 5_000);
    }

    // -- TDigestEstimator tests --

    #[test]
    fn test_tdigest_quantile_at_arbitrary_q() {
        // Feed a known distribution (uniform 0..1000) and verify that
        // `quantile_at` is accurate at off-quartile probes.
        let mut d = TDigestEstimator::new(TDigestEstimator::DEFAULT_DELTA);
        for v in 0..1000 {
            d.observe(v as f64);
        }
        // For uniform[0, N], the empirical q-quantile is ≈ q*(N-1).
        // T-digest is most accurate at the tails (k1 scale function),
        // so we accept ~0.5% absolute error of the range (== ~5 here).
        let tol = 5.0;
        for q in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99] {
            let est = d.quantile_at(q);
            let truth = q * 999.0;
            assert!(
                (est - truth).abs() < tol,
                "q={q}: got {est}, expected ≈{truth} (tol={tol})"
            );
        }
    }

    #[test]
    fn test_tdigest_quantile_at_empty_returns_zero() {
        let mut d = TDigestEstimator::new(TDigestEstimator::DEFAULT_DELTA);
        assert_eq!(d.quantile_at(0.5), 0.0);
    }

    #[test]
    fn test_tdigest_quantile_at_flushes_buffer() {
        // After a few observations smaller than buf_cap, internal buffer
        // is non-empty. quantile_at must flush before answering.
        let mut d = TDigestEstimator::new(TDigestEstimator::DEFAULT_DELTA);
        for v in [10.0, 20.0, 30.0, 40.0, 50.0] {
            d.observe(v);
        }
        // Without flush these would be buffered. quantile_at flushes.
        let q50 = d.quantile_at(0.5);
        assert!((10.0..=50.0).contains(&q50), "median {q50} out of range");
    }
}

// ---------------------------------------------------------------------------
// TDigest quantile estimator
// ---------------------------------------------------------------------------

/// Streaming quantile estimator using the t-digest algorithm (Dunning 2019).
///
/// Maintains a sorted set of centroids (mean, weight). Incoming values are
/// buffered; when the buffer fills, it is sorted and merged into the centroid
/// list with compression controlled by the k1 scale function. Gives excellent
/// accuracy at both tails and median, with O(δ) memory.
#[derive(Debug, Clone)]
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

    /// Estimate the value at quantile `q ∈ [0, 1]`.
    ///
    /// Takes `&mut self` because it must `flush()` the buffered observations
    /// into the centroid list before scanning. Callers holding only `&self`
    /// must clone the digest first (cheap — bounded by `delta` centroids).
    ///
    /// Returns `0.0` for an empty digest. For `q` outside `[0, 1]`, behavior
    /// follows `quantile_inner` (clamps to the empirical extremes).
    pub fn quantile_at(&mut self, q: f64) -> f64 {
        self.flush();
        self.quantile_inner(q)
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
// Identity hasher for DashMap<u64, _> (keys are already hashed)
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
pub struct IdentityHasher(u64);

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
pub struct IdentityBuildHasher;

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

/// Maximum number of unique categories per column before aborting collection.
pub const MAX_CATEGORIES: u64 = 1_000_000;
const CATEGORY_ENTRY_OVERHEAD_BYTES: u64 = 256;
const CATEGORY_COLUMN_OVERHEAD_BYTES: u64 = 256 * 1024;
// At output time a recovery string can coexist in the retained count map,
// root stats, full Kumo stats, and minified Kumo stats. Only one serialized
// buffer is live at a time: local mirrors upload from the durable file and
// remote OutputTargets synchronously await each put. JSON escaping can grow
// every input byte to six bytes ("\u00XX"), and Vec growth may retain up to
// twice its final length. Charge this complete worst-case fan-out before
// allocating the retained string or its map entry.
const CATEGORY_RETAINED_COPY_COUNT: u64 = 4;
const CATEGORY_JSON_ESCAPE_FACTOR: u64 = 6;
const CATEGORY_SERIALIZED_VEC_CAPACITY_FACTOR: u64 = 2;
const CATEGORY_MATERIALIZATION_MULTIPLIER: u64 = CATEGORY_RETAINED_COPY_COUNT
    + CATEGORY_JSON_ESCAPE_FACTOR * CATEGORY_SERIALIZED_VEC_CAPACITY_FACTOR;

/// Encode-wide conservative budget for retained category recovery strings and
/// their count/string map entries. Reservations are monotonic: moved entries
/// keep their charge, while dropped duplicates may conservatively retain it.
pub struct CategoryStringBudget {
    max_bytes: u64,
    used_bytes: AtomicU64,
}

impl CategoryStringBudget {
    pub fn new(max_bytes: u64) -> Self {
        Self {
            max_bytes,
            used_bytes: AtomicU64::new(0),
        }
    }

    fn try_reserve_bytes(&self, bytes: u64) -> bool {
        self.used_bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
                used.checked_add(bytes)
                    .filter(|next| *next <= self.max_bytes)
            })
            .is_ok()
    }

    fn try_reserve(&self, string_bytes: usize) -> bool {
        self.try_reserve_bytes(
            (string_bytes as u64)
                .saturating_add(CATEGORY_ENTRY_OVERHEAD_BYTES)
                .saturating_mul(CATEGORY_MATERIALIZATION_MULTIPLIER),
        )
    }

    pub fn try_reserve_column(&self) -> bool {
        self.try_reserve_bytes(CATEGORY_COLUMN_OVERHEAD_BYTES)
    }

    pub fn used_bytes(&self) -> u64 {
        self.used_bytes.load(Ordering::Relaxed)
    }
}

/// Internal storage for category counts.
pub enum CountStorage {
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
    pub storage: CountStorage,
    pub unique_count: AtomicU64,
    pub aborted: AtomicBool,
    max_categories: Option<u64>,
    string_budget: Option<Arc<CategoryStringBudget>>,
}

impl SharedColumnCounts {
    /// Create hash-based storage (for GlobalDashMap & Hybrid).
    pub fn new_hashed() -> Self {
        Self::new_hashed_with_cap(Some(MAX_CATEGORIES), None)
    }

    pub fn new_hashed_with_string_budget(budget: Arc<CategoryStringBudget>) -> Self {
        Self::new_hashed_with_cap(Some(MAX_CATEGORIES), Some(budget))
    }

    /// Create hash-based storage without an ingestion-side unique cap.
    pub fn new_hashed_uncapped() -> Self {
        Self::new_hashed_with_cap(None, None)
    }

    fn new_hashed_with_cap(
        max_categories: Option<u64>,
        string_budget: Option<Arc<CategoryStringBudget>>,
    ) -> Self {
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
            max_categories,
            string_budget,
        }
    }

    /// Create graduation v2 storage (ArcSwap frozen set + DashMap pending + inbox).
    pub fn new_graduation() -> Self {
        Self::new_graduation_with_cap(Some(MAX_CATEGORIES), None)
    }

    pub fn new_graduation_with_string_budget(budget: Arc<CategoryStringBudget>) -> Self {
        Self::new_graduation_with_cap(Some(MAX_CATEGORIES), Some(budget))
    }

    /// Create graduation v2 storage without an ingestion-side unique cap.
    pub fn new_graduation_uncapped() -> Self {
        Self::new_graduation_with_cap(None, None)
    }

    fn new_graduation_with_cap(
        max_categories: Option<u64>,
        string_budget: Option<Arc<CategoryStringBudget>>,
    ) -> Self {
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
            max_categories,
            string_budget,
        }
    }

    /// Reserve one new recovery-string/map entry before allocating it. A
    /// failed reservation aborts collection for this column without affecting
    /// graph ingestion.
    pub fn reserve_string_bytes(&self, bytes: usize) -> bool {
        if self.aborted.load(Ordering::Relaxed) {
            return false;
        }
        if self
            .string_budget
            .as_ref()
            .is_some_and(|budget| !budget.try_reserve(bytes))
        {
            self.aborted.store(true, Ordering::Relaxed);
            return false;
        }
        true
    }

    /// Record a newly observed unique category and abort only when this counter
    /// was constructed with an explicit cap.
    #[inline]
    pub fn record_new_unique(&self) {
        let new_unique = self.unique_count.fetch_add(1, Ordering::Relaxed) + 1;
        if let Some(max_categories) = self.max_categories {
            if new_unique > max_categories {
                self.aborted.store(true, Ordering::Relaxed);
            }
        }
    }

    /// Observe a hashed value into shared DashMap. Called from Hybrid spill path.
    #[inline]
    pub fn observe_hashed(&self, hash: u64, value_str: &str) {
        if self.aborted.load(Ordering::Relaxed) {
            return;
        }
        if let CountStorage::Hashed { counts, strings } = &self.storage {
            match counts.entry(hash) {
                dashmap::mapref::entry::Entry::Occupied(mut entry) => *entry.get_mut() += 1,
                dashmap::mapref::entry::Entry::Vacant(entry) => {
                    if !self.reserve_string_bytes(value_str.len()) {
                        return;
                    }
                    entry.insert(1);
                    strings.entry(hash).or_insert_with(|| value_str.to_owned());
                    self.record_new_unique();
                }
            }
        }
    }

    /// Merge a batch of local counts into the shared DashMap (Hybrid flush).
    pub fn merge_local_batch(&self, local: &mut LocalColumnCounts) {
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
                    self.record_new_unique();
                    if self.aborted.load(Ordering::Relaxed) {
                        return;
                    }
                }
            }
        }
    }

    /// Merge a batch of local counts into graduation storage.
    /// Returns the current graduated snapshot for caching by the caller.
    pub fn merge_local_batch_graduation(&self, local: &mut LocalColumnCounts) -> Arc<HashSet<u64>> {
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
                    self.record_new_unique();
                    if self.aborted.load(Ordering::Relaxed) {
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
// Per-file local category count state (Hybrid & Graduation)
// ---------------------------------------------------------------------------

/// Per-file local category counts for a single column.
pub struct LocalColumnCounts {
    pub counts: HashMap<u64, u64>,
    pub strings: HashMap<u64, String>,
}

impl LocalColumnCounts {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            counts: HashMap::with_capacity(cap),
            strings: HashMap::with_capacity(cap),
        }
    }

    /// Observe a value locally. Returns true if the key is new.
    #[inline]
    pub fn observe(&mut self, hash: u64, value_str: &str) -> bool {
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

    pub fn len(&self) -> usize {
        self.counts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    pub fn clear(&mut self) {
        self.counts.clear();
        self.strings.clear();
    }
}

// ---------------------------------------------------------------------------
// Stat formatting helpers (kumo_key, kumo_entry, kumo_safe_div)
// ---------------------------------------------------------------------------

pub fn kumo_key(col: &str, stype: &str, stat: &str, approx: bool) -> String {
    let approx_str = if approx { "True" } else { "False" };
    format!("{col}__{stype}__{stat}__{approx_str}")
}

pub fn kumo_entry(value: JsonValue, approx: bool, dv: &str, n: u64) -> JsonValue {
    json!({"value": value, "approx": approx, "data_version": dv, "num_entries": n})
}

pub fn kumo_safe_div(num: u64, den: u64) -> f64 {
    if den == 0 { 0.0 } else { num as f64 / den as f64 }
}

// ---------------------------------------------------------------------------
// Multicategorical primitives
// ---------------------------------------------------------------------------

/// Dispatch discriminator for multicategorical tokenization. Frozen at
/// collector construction based on the column's `IngestedType`. The inner
/// loop specializes on this to avoid per-row branching on the list variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MulticatInputKind {
    /// IngestedType::String, no `multicategorical_delimiter` declared —
    /// the whole cell is treated as one token. Multicat columns that
    /// arrive with a delimiter are normalized to `IngestedType::StringArray`
    /// at config conversion (kumo-ai/kumo#28478) and dispatch through the
    /// `StringArray` arm below; this variant is the single-token-per-row
    /// degenerate case only.
    SingleStringToken,
    /// IngestedType::StringArray — iterate list elements; nulls become "".
    /// Reached for both native `List<Utf8>` parquet sources and for
    /// `Utf8`-scalar sources whose declared `multicategorical_delimiter`
    /// caused the convert-time normalize pass to flip them.
    StringArray,
    /// IngestedType::Int64Array — iterate list elements; nulls become 0.
    Int64Array,
    /// IngestedType::Float16Array/Float32Array/Float64Array — iterate, widen
    /// to f64, drop NaN, stringify via Display; nulls become 0.0.
    FloatArray,
}

/// Split `raw` by a single ASCII byte separator and call `f` for each token.
/// Does not allocate.
#[inline]
pub fn for_each_split_ascii_byte(raw: &str, separator: u8, mut f: impl FnMut(&str)) {
    let mut start = 0;
    for (idx, byte) in raw.bytes().enumerate() {
        if byte == separator {
            f(&raw[start..idx]);
            start = idx + 1;
        }
    }
    f(&raw[start..]);
}

/// Build kumo-format MULTI_* key-value pairs for multicategorical stats.
///
/// `category_counts` must be pre-sorted (count-desc, token-asc).
#[allow(clippy::too_many_arguments)]
pub fn build_multicat_emit_stats(
    name: &str,
    stype: &str,
    dv: &str,
    count: u64,
    null_count: u64,
    hll: &HyperLogLog,
    separator: Option<&str>,
    category_counts: &[(String, u64)],
) -> Vec<(String, JsonValue)> {
    let total = count + null_count;
    let mut pairs = Vec::with_capacity(7);

    // Universal stats (match Categorical/Id shape)
    pairs.push((
        kumo_key(name, stype, "COUNT", false),
        kumo_entry(json!(total), false, dv, total),
    ));
    pairs.push((
        kumo_key(name, stype, "NUM_NA", false),
        kumo_entry(json!(null_count), false, dv, total),
    ));
    pairs.push((
        kumo_key(name, stype, "NA_FRACTION", false),
        kumo_entry(json!(kumo_safe_div(null_count, total)), false, dv, total),
    ));
    pairs.push((
        kumo_key(name, stype, "INVALID_FRACTION", false),
        kumo_entry(json!(0.0), false, dv, total),
    ));

    // Multicat-specific: separator
    let sep_value = match separator {
        Some(s) => JsonValue::String(s.to_owned()),
        None => JsonValue::Null,
    };
    pairs.push((
        kumo_key(name, stype, "MULTI_CATEGORIES_SEPARATOR", false),
        kumo_entry(sep_value, false, dv, total),
    ));

    // MULTI_CATEGORY_COUNTS: [[token, count], ...]
    let values: Vec<JsonValue> = category_counts
        .iter()
        .map(|(tok, cnt)| json!([tok, cnt]))
        .collect();
    pairs.push((
        kumo_key(name, stype, "MULTI_CATEGORY_COUNTS", false),
        kumo_entry(json!(values), false, dv, total),
    ));

    // NUM_UNIQUE_MULTI (approx=true, HLL estimate)
    let num_unique = hll.estimate() as u64;
    pairs.push((
        kumo_key(name, stype, "NUM_UNIQUE_MULTI", true),
        kumo_entry(json!(num_unique), true, dv, total),
    ));

    pairs
}
