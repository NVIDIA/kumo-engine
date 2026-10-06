// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared streaming accumulators used by both col-stats (ingestion) and
//! TT-gen (training table generation).
//!
//! These are pure data structures with no crate-specific dependencies beyond
//! `gxhash` and `ptr_hash` (for value hashing).

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use arc_swap::{ArcSwap, ArcSwapOption};
use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};
use serde_json::{json, Value as JsonValue};

// ---------------------------------------------------------------------------
// HyperLogLog cardinality estimator
// ---------------------------------------------------------------------------

/// HyperLogLog cardinality estimator with 2^14 = 16384 registers.
///
/// Hash function: [`HyperLogLog::hash_str`], `hash_i64`, `hash_f64`.
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

    /// Compute a u64 hash for a string using the fixed HLL seed.
    /// Used by both HLL and category counts to share a single hash computation.
    #[inline]
    pub fn hash_str(value: &str) -> u64 {
        ptr_hash::hash::hash_str(value, Self::HASH_SEED as u64)
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

    /// Hash and observe a string.
    #[inline]
    pub fn observe_str(&mut self, value: &str) {
        self.observe_hash(Self::hash_str(value));
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
            hll.observe_str(&format!("item_{}", i));
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
        let capped = SharedColumnCounts::new_hashed_capped_with_max_categories(2);
        assert!(capped.observe_hashed(1, "one"));
        assert!(capped.observe_hashed(2, "two"));
        assert!(!capped.is_capped());
        assert!(!capped.observe_hashed(3, "three"));
        assert!(capped.is_capped());
        assert!(!capped.is_aborted());
        assert!(capped.observe_hashed(1, "one"));
        assert_eq!(capped.drain_to_map().unwrap()["one"], 2);

        let zero_cap = SharedColumnCounts::new_hashed_capped_with_max_categories(0);
        assert!(!zero_cap.observe_hashed(1, "excluded"));
        assert_eq!(zero_cap.unique_count.load(Ordering::Relaxed), 0);
        assert!(zero_cap.is_capped());
        assert!(zero_cap
            .admitted_snapshot
            .load_full()
            .is_some_and(|snapshot| snapshot.is_empty()));

        let caller_capped = SharedColumnCounts::new_hashed_with_max_categories(2);
        caller_capped.unique_count.store(1, Ordering::Relaxed);
        caller_capped.record_new_unique();
        assert!(!caller_capped.is_aborted());
        caller_capped.record_new_unique();
        assert!(caller_capped.is_aborted());

        let capped_grad = SharedColumnCounts::new_graduation();
        capped_grad
            .unique_count
            .store(MAX_CATEGORIES, Ordering::Relaxed);
        capped_grad.record_new_unique();
        assert!(capped_grad.is_aborted());
    }

    #[test]
    fn retained_counts_include_local_deltas_before_and_after_cap() {
        let counts = SharedColumnCounts::new_hashed_capped_with_max_categories(2);
        let mut local = LocalColumnCounts::with_capacity(2);

        assert!(counts.observe_hashed(1, "hot"));
        local.cache_admitted(1, "hot");
        assert!(local.increment_if_present(1, "hot"));
        assert!(counts.observe_hashed(2, "cold"));
        assert!(!counts.observe_hashed(3, "excluded"));
        assert!(local.increment_if_present(1, "hot"));

        counts.merge_local_batch(&mut local);
        let result = counts.drain_to_map().unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result["hot"], 3);
        assert_eq!(result["cold"], 1);
        assert!(!result.contains_key("excluded"));
        assert!(counts.is_capped());
    }

    #[test]
    fn frozen_snapshot_rejects_excluded_hash_without_dashmap_access() {
        let counts = SharedColumnCounts::new_hashed_capped_with_max_categories(1);
        assert!(counts.observe_hashed(1, "retained"));
        assert!(counts.admitted_snapshot.load().is_some());

        assert!(!counts.observe_hashed(2, "excluded"));
        let accesses_after_first_rejection = counts.hashed_map_accesses.load(Ordering::Relaxed);
        for _ in 0..100 {
            assert!(!counts.observe_hashed(2, "excluded"));
        }

        assert_eq!(
            counts.hashed_map_accesses.load(Ordering::Relaxed),
            accesses_after_first_rejection,
        );
        assert!(counts.is_capped());
        assert_eq!(
            counts.drain_to_map().unwrap(),
            HashMap::from([("retained".to_owned(), 1)])
        );
    }

    #[test]
    fn rejected_local_key_does_not_skip_a_later_retained_delta() {
        let counts = SharedColumnCounts::new_hashed_capped_with_max_categories(1);
        assert!(counts.observe_hashed(1, "retained"));

        let mut local = LocalColumnCounts::with_capacity(2);
        local.counts.insert(2, 7);
        local.strings.insert(2, "excluded".to_owned());
        local.counts.insert(1, 4);
        local.strings.insert(1, "retained".to_owned());
        counts.merge_local_batch(&mut local);

        let result = counts.drain_to_map().unwrap();
        assert_eq!(result, HashMap::from([("retained".to_owned(), 5)]));
        assert!(counts.is_capped());
    }

    #[test]
    fn colliding_values_are_counted_apart() {
        for counts in [
            SharedColumnCounts::new_hashed(),
            SharedColumnCounts::new_hashed_with_max_categories(10),
        ] {
            let mut local = LocalColumnCounts::with_capacity(1);
            local.observe(7, "left");
            local.observe(7, "right");
            local.observe(7, "right");
            counts.merge_local_batch(&mut local);
            assert!(counts.observe_hashed(7, "right"));
            assert!(counts.observe_hashed(7, "third"));
            assert!(counts.observe_hashed_i64(8, 11));
            assert!(counts.observe_hashed_i64(8, 12));
            assert_eq!(counts.unique_count.load(Ordering::Relaxed), 5);
            assert_eq!(
                counts.drain_to_map().unwrap(),
                HashMap::from([
                    ("left".to_owned(), 1),
                    ("right".to_owned(), 3),
                    ("third".to_owned(), 1),
                    ("11".to_owned(), 1),
                    ("12".to_owned(), 1),
                ])
            );
        }
    }

    #[test]
    fn colliding_value_respects_the_cap() {
        let counts = SharedColumnCounts::new_hashed_capped_with_max_categories(2);
        assert!(counts.observe_hashed(7, "left"));
        assert!(counts.observe_hashed(7, "right"));
        assert!(!counts.observe_hashed(7, "late"));
        assert!(!counts.observe_hashed(9, "other"));
        assert!(counts.observe_hashed(7, "right"));
        assert!(counts.is_capped());
        assert_eq!(
            counts.drain_to_map().unwrap(),
            HashMap::from([("left".to_owned(), 1), ("right".to_owned(), 2)])
        );

        let exhaustive = SharedColumnCounts::new_hashed_with_max_categories(1);
        assert!(exhaustive.observe_hashed(7, "left"));
        exhaustive.observe_hashed(7, "right");
        assert!(exhaustive.is_aborted());
    }

    #[test]
    fn concurrent_colliding_values_count_exactly() {
        use std::thread;

        let values: Vec<String> = (0..48).map(|i| format!("value-{i}")).collect();
        for cap in [MAX_CATEGORIES, 20] {
            let counts = Arc::new(SharedColumnCounts::new_hashed_capped_with_max_categories(
                cap,
            ));
            let handles: Vec<_> = (0..8)
                .map(|worker| {
                    let (counts, values) = (Arc::clone(&counts), values.clone());
                    thread::spawn(move || {
                        let mut local = LocalColumnCounts::with_capacity(4);
                        let mut admitted = HashMap::<String, u64>::new();
                        for step in 0..20_000usize {
                            let value = &values[(step * 7 + worker) % values.len()];
                            let hash = (value.len() % 4) as u64;
                            if local.increment_if_present(hash, value)
                                || counts.observe_hashed(hash, value)
                            {
                                *admitted.entry(value.clone()).or_default() += 1;
                                local.cache_admitted(hash, value);
                            }
                            if step % 1_000 == 0 {
                                counts.merge_local_batch(&mut local);
                                local.clear();
                            }
                        }
                        counts.merge_local_batch(&mut local);
                        admitted
                    })
                })
                .collect();
            let mut expected = HashMap::<String, u64>::new();
            for handle in handles {
                for (value, count) in handle.join().unwrap() {
                    *expected.entry(value).or_default() += count;
                }
            }
            let drained = counts.drain_to_map().unwrap();
            assert_eq!(drained.len() as u64, cap.min(48));
            for (value, count) in drained {
                assert_eq!(expected[&value], count, "{value}");
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn long_gxhash_twins_hash_apart() {
        use std::hash::Hasher as _;
        let gxhash = |bytes: &[u8]| {
            let mut hasher = gxhash::GxHasher::with_seed(HyperLogLog::HASH_SEED);
            hasher.write(bytes);
            hasher.finish()
        };
        let (left, right) = ptr_hash::hash::gxhash_twins();
        assert_ne!(left, right);
        assert_eq!(gxhash(left.as_bytes()), gxhash(right.as_bytes()));
        assert_ne!(HyperLogLog::hash_str(&left), HyperLogLog::hash_str(&right));
        assert_eq!(
            HyperLogLog::hash_str("sixteen-bytes-ok"),
            gxhash(b"sixteen-bytes-ok")
        );
    }

    #[test]
    fn local_clear_removes_integer_identity() {
        let mut local = LocalColumnCounts::with_capacity(1);
        local.cache_admitted_i64(7, 11);
        assert!(local.increment_i64_if_present(7, 11));

        local.clear();

        assert!(!local.increment_i64_if_present(7, 11));
        assert!(local.integer_values.is_empty());
        assert!(local.strings.is_empty());
        assert!(local.counts.is_empty());
    }

    #[test]
    #[should_panic(expected = "exact category occurrence count overflowed u64")]
    fn shared_count_overflow_fails_explicitly() {
        let counts = SharedColumnCounts::new_hashed();
        assert!(counts.observe_hashed(7, "value"));
        let CountStorage::Hashed { counts: values, .. } = &counts.storage else {
            unreachable!();
        };
        values.get_mut(&7).unwrap().count = u64::MAX;
        counts.observe_hashed(7, "value");
    }

    #[test]
    #[should_panic(expected = "exact local category occurrence count overflowed u64")]
    fn local_fast_path_count_overflow_fails_explicitly() {
        let mut local = LocalColumnCounts::with_capacity(1);
        local.cache_admitted(7, "value");
        *local.counts.get_mut(&7).unwrap() = u64::MAX;
        local.increment_if_present(7, "value");
    }

    #[test]
    fn concurrent_boundary_overshoot_is_counted_before_final_trim() {
        use std::thread;

        let barrier = Arc::new(std::sync::Barrier::new(2));
        let mut inner = SharedColumnCounts::new_hashed_capped_with_max_categories(1);
        inner.admission_barrier = Some(Arc::clone(&barrier));
        let counts = Arc::new(inner);
        let handles: Vec<_> = [(0, "a"), (u64::MAX, "z")]
            .into_iter()
            .map(|(hash, value)| {
                let counts = Arc::clone(&counts);
                thread::spawn(move || assert!(counts.observe_hashed(hash, value)))
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(counts.unique_count.load(Ordering::Relaxed), 2);
        assert!(counts.is_capped());
        let snapshot = counts
            .admitted_snapshot
            .load_full()
            .expect("closed admissions publish a membership snapshot");
        assert_eq!(snapshot.len(), 2);
        assert!(snapshot.contains(&0));
        assert!(snapshot.contains(&u64::MAX));
        for _ in 0..4 {
            assert!(counts.observe_hashed(u64::MAX, "z"));
        }
        assert_eq!(
            counts.drain_to_map().unwrap(),
            HashMap::from([("z".to_owned(), 5)])
        );
    }

    #[test]
    fn concurrent_workers_count_one_retained_membership_exactly() {
        use std::thread;

        let counts = Arc::new(SharedColumnCounts::new_hashed_capped_with_max_categories(2));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let counts = Arc::clone(&counts);
                thread::spawn(move || {
                    for _ in 0..1_000 {
                        assert!(counts.observe_hashed(11, "shared"));
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(counts.unique_count.load(Ordering::Relaxed), 1);
        assert_eq!(counts.drain_to_map().unwrap()["shared"], 8_000);
        assert!(!counts.is_capped());
    }

    #[test]
    fn graduation_counts_first_batch_at_min_occ_as_unique() {
        // A brand-new hash whose first local batch already meets min_occ
        // graduates immediately; it must still register as one unique category.
        let counts = SharedColumnCounts::new_graduation();
        let mut local = LocalColumnCounts::with_capacity(1);
        // Default min_occ is sqrt(10_000) = 100, so this graduates on merge.
        local.observe(1, "hot");
        *local.counts.get_mut(&1).unwrap() = 100;
        counts.merge_local_batch_graduation(&mut local);
        assert_eq!(counts.unique_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn graduation_counts_same_hash_across_locals_once() {
        // Two workers flush the same new hash concurrently. A barrier lines up
        // their `merge_local_batch_graduation` calls so both race the write-once
        // `strings` slot; only the single-arbitration path counts the category
        // once. The old racy `contains_key`+`or_insert` would double-count here.
        use std::sync::Barrier;
        use std::thread;

        for _ in 0..64 {
            let counts = Arc::new(SharedColumnCounts::new_graduation());
            let barrier = Arc::new(Barrier::new(2));
            let handles: Vec<_> = (0..2)
                .map(|_| {
                    let counts = Arc::clone(&counts);
                    let barrier = Arc::clone(&barrier);
                    thread::spawn(move || {
                        let mut local = LocalColumnCounts::with_capacity(1);
                        local.observe(9, "warm");
                        barrier.wait();
                        counts.merge_local_batch_graduation(&mut local);
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
            assert_eq!(counts.unique_count.load(Ordering::Relaxed), 1);
        }
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

/// Maximum number of category values published per feature column.
pub const MAX_CATEGORIES: u64 = 1_000_000;

/// Typed category identity. Integer-backed categories stay unformatted until
/// publication; string-backed values retain their canonical spelling.
pub enum CategoryIdentity {
    String(String),
    I64(i64),
}

/// A canonical value and its exact occurrence count. Keeping typed identity
/// beside the hash makes the DashMap entry the single admission/counting
/// arbitration point and lets us detect otherwise-silent 64-bit collisions.
/// A value whose hash is already held by a different value is counted in
/// `CountStorage::Hashed::collided` instead.
pub struct HashedCategoryCount {
    value: CategoryIdentity,
    count: u64,
}

/// Internal storage for category counts.
pub enum CountStorage {
    /// Hash-based exact retained counts (Hybrid).
    Hashed {
        counts: DashMap<u64, HashedCategoryCount, IdentityBuildHasher>,
        collided: Mutex<Vec<HashedCategoryCount>>,
    },
    /// Graduation v2: ArcSwap frozen graduated set + DashMap pending + inbox coordinator.
    /// Identity is the hash alone; rejected with col_stats, so never published.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CategoryLimitPolicy {
    /// Feature statistics retain admitted keys and reject only new keys once
    /// the bound is reached.
    RetainCapped,
    /// Classification targets require an exhaustive vocabulary and abort when
    /// their independent ceiling is exceeded.
    Abort,
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
    admissions_closed: AtomicBool,
    /// Readers cover concurrent admission; the snapshot publisher takes the
    /// write side before enumerating stable membership.
    admission_gate: RwLock<()>,
    /// Shared frozen membership used to reject excluded hashes before any
    /// DashMap shard access.
    admitted_snapshot: ArcSwapOption<HashSet<u64>>,
    capped: AtomicBool,
    collision_warned: AtomicBool,
    max_categories: Option<u64>,
    limit_policy: CategoryLimitPolicy,
    #[cfg(test)]
    hashed_map_accesses: AtomicU64,
    #[cfg(test)]
    admission_barrier: Option<Arc<std::sync::Barrier>>,
}

impl SharedColumnCounts {
    /// Create hash-based storage (for GlobalDashMap & Hybrid).
    pub fn new_hashed() -> Self {
        Self::new_hashed_capped_with_max_categories(MAX_CATEGORIES)
    }

    /// Create retained hash-based feature storage with a caller-specific cap.
    ///
    /// Primarily useful for focused ingest tests. Production feature stats use
    /// [`MAX_CATEGORIES`]. Concurrent distinct admissions can overshoot by at
    /// most the number of simultaneous in-progress admission operations; the
    /// complete admitted counts are trimmed deterministically at publication.
    pub fn new_hashed_capped_with_max_categories(max_categories: u64) -> Self {
        Self::new_hashed_with_cap(Some(max_categories), CategoryLimitPolicy::RetainCapped)
    }

    /// Create hash-based storage with a caller-specific unique-category cap.
    ///
    /// This constructor preserves the exhaustive-or-fail contract used for
    /// classification targets. It is deliberately separate from the retained
    /// feature-vocabulary policy used by [`Self::new_hashed`].
    pub fn new_hashed_with_max_categories(max_categories: u64) -> Self {
        Self::new_hashed_with_cap(Some(max_categories), CategoryLimitPolicy::Abort)
    }

    fn new_hashed_with_cap(max_categories: Option<u64>, limit_policy: CategoryLimitPolicy) -> Self {
        Self {
            storage: CountStorage::Hashed {
                counts: DashMap::with_capacity_and_hasher_and_shard_amount(
                    1024,
                    IdentityBuildHasher,
                    256,
                ),
                collided: Mutex::new(Vec::new()),
            },
            unique_count: AtomicU64::new(0),
            aborted: AtomicBool::new(false),
            admissions_closed: AtomicBool::new(false),
            admission_gate: RwLock::new(()),
            admitted_snapshot: ArcSwapOption::empty(),
            capped: AtomicBool::new(false),
            collision_warned: AtomicBool::new(false),
            max_categories,
            limit_policy,
            #[cfg(test)]
            hashed_map_accesses: AtomicU64::new(0),
            #[cfg(test)]
            admission_barrier: None,
        }
    }

    /// Create graduation v2 storage (ArcSwap frozen set + DashMap pending + inbox).
    pub fn new_graduation() -> Self {
        Self::new_graduation_with_cap(Some(MAX_CATEGORIES))
    }

    fn new_graduation_with_cap(max_categories: Option<u64>) -> Self {
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
            admissions_closed: AtomicBool::new(false),
            admission_gate: RwLock::new(()),
            admitted_snapshot: ArcSwapOption::empty(),
            capped: AtomicBool::new(false),
            collision_warned: AtomicBool::new(false),
            max_categories,
            limit_policy: CategoryLimitPolicy::Abort,
            #[cfg(test)]
            hashed_map_accesses: AtomicU64::new(0),
            #[cfg(test)]
            admission_barrier: None,
        }
    }

    /// Record a newly observed unique category and abort only when this counter
    /// was constructed with an explicit cap.
    #[inline]
    pub fn record_new_unique(&self) {
        let new_unique = self.unique_count.fetch_add(1, Ordering::Relaxed) + 1;
        if let Some(max_categories) = self.max_categories {
            if new_unique > max_categories {
                match self.limit_policy {
                    CategoryLimitPolicy::RetainCapped => {
                        self.capped.store(true, Ordering::Relaxed);
                        self.admissions_closed.store(true, Ordering::Relaxed);
                    }
                    CategoryLimitPolicy::Abort => {
                        self.aborted.store(true, Ordering::Relaxed);
                    }
                }
            }
        }
    }

    fn checked_add_count(current: &mut u64, delta: u64) {
        *current = current
            .checked_add(delta)
            .expect("exact category occurrence count overflowed u64");
    }

    #[inline]
    fn record_hashed_map_access(&self) {
        #[cfg(test)]
        self.hashed_map_accesses.fetch_add(1, Ordering::Relaxed);
    }

    /// Publish one immutable membership snapshot after admissions close.
    ///
    /// The write guard waits for every admission reader that could already
    /// have passed the cap check, so concurrent overshoot winners are included
    /// before snapshot misses become authoritative.
    fn publish_admitted_snapshot(&self) {
        if self.admitted_snapshot.load().is_some() {
            return;
        }
        let _admission_guard = self.admission_gate.write();
        if self.admitted_snapshot.load().is_some() {
            return;
        }
        let CountStorage::Hashed { counts, .. } = &self.storage else {
            return;
        };
        let snapshot = counts.iter().map(|entry| *entry.key()).collect();
        self.admitted_snapshot.store(Some(Arc::new(snapshot)));
    }

    #[inline]
    fn reject_from_admitted_snapshot(&self, hash: u64) -> bool {
        let snapshot = self.admitted_snapshot.load();
        if snapshot
            .as_ref()
            .is_some_and(|admitted| !admitted.contains(&hash))
        {
            if !self.capped.load(Ordering::Relaxed) {
                self.capped.store(true, Ordering::Relaxed);
            }
            true
        } else {
            false
        }
    }

    /// Retained-policy admission of one new distinct value. Callers hold the
    /// admission gate for reading.
    fn try_admit_retained(&self, publish_snapshot: &mut bool) -> bool {
        let at_limit = self.max_categories.is_some_and(|max_categories| {
            self.unique_count.load(Ordering::Relaxed) >= max_categories
        });
        if self.admissions_closed.load(Ordering::Acquire) || at_limit {
            self.admissions_closed.store(true, Ordering::Release);
            self.capped.store(true, Ordering::Relaxed);
            *publish_snapshot = true;
            return false;
        }
        #[cfg(test)]
        if let Some(barrier) = &self.admission_barrier {
            barrier.wait();
        }
        let new_unique = self.unique_count.fetch_add(1, Ordering::Relaxed) + 1;
        if let Some(max_categories) = self.max_categories {
            if new_unique >= max_categories {
                // Freeze membership at capacity. Admission readers already in
                // flight may still win; the publisher waits for all of them.
                self.admissions_closed.store(true, Ordering::Release);
                *publish_snapshot = true;
                if new_unique > max_categories {
                    self.capped.store(true, Ordering::Relaxed);
                }
            }
        }
        true
    }

    /// Exhaustive-policy admission: always admits, aborts past the cap.
    fn admit_exhaustive(&self) -> bool {
        let new_unique = self.unique_count.fetch_add(1, Ordering::Relaxed) + 1;
        if self
            .max_categories
            .is_some_and(|max_categories| new_unique > max_categories)
        {
            self.aborted.store(true, Ordering::Relaxed);
        }
        true
    }

    /// Counts a value whose hash is held by a different value.
    fn merge_collided<F, V>(
        collided: &Mutex<Vec<HashedCategoryCount>>,
        delta: u64,
        identity_matches: F,
        value_factory: V,
        admit: impl FnOnce() -> bool,
    ) -> bool
    where
        F: Fn(&CategoryIdentity) -> bool,
        V: FnOnce() -> CategoryIdentity,
    {
        let mut collided = collided.lock();
        if let Some(stored) = collided
            .iter_mut()
            .find(|stored| identity_matches(&stored.value))
        {
            Self::checked_add_count(&mut stored.count, delta);
            return true;
        }
        if !admit() {
            return false;
        }
        collided.push(HashedCategoryCount {
            value: value_factory(),
            count: delta,
        });
        true
    }

    fn merge_hashed_with<F, V>(
        &self,
        hash: u64,
        delta: u64,
        identity_matches: F,
        value_factory: V,
    ) -> bool
    where
        F: Fn(&CategoryIdentity) -> bool,
        V: FnOnce() -> CategoryIdentity,
    {
        if self.aborted.load(Ordering::Relaxed) {
            return false;
        }
        let CountStorage::Hashed { counts, collided } = &self.storage else {
            return false;
        };

        if self.limit_policy == CategoryLimitPolicy::RetainCapped {
            // Once published, an absent hash is known never to have won
            // admission. This bypasses all DashMap shard locks for the
            // high-cardinality tail.
            if self.reject_from_admitted_snapshot(hash) {
                return false;
            }

            // Retained hashes remain on the exact shared counting path and do
            // not need the admission gate after membership is frozen.
            self.record_hashed_map_access();
            if let Some(mut stored) = counts.get_mut(&hash) {
                if identity_matches(&stored.value) {
                    Self::checked_add_count(&mut stored.count, delta);
                    return true;
                }
            }

            let mut publish_snapshot = false;
            let admitted = {
                // Consistent lock order is admission gate before DashMap entry.
                // The snapshot publisher takes the write guard before
                // iterating DashMap shards, so it can never wait on a reader
                // that is itself waiting for the write guard.
                let _admission_guard = self.admission_gate.read();
                if self.reject_from_admitted_snapshot(hash) {
                    return false;
                }

                self.record_hashed_map_access();
                match counts.entry(hash) {
                    dashmap::mapref::entry::Entry::Occupied(mut entry) => {
                        let stored = entry.get_mut();
                        if identity_matches(&stored.value) {
                            Self::checked_add_count(&mut stored.count, delta);
                            true
                        } else {
                            Self::merge_collided(
                                collided,
                                delta,
                                identity_matches,
                                value_factory,
                                || self.try_admit_retained(&mut publish_snapshot),
                            )
                        }
                    }
                    dashmap::mapref::entry::Entry::Vacant(entry) => {
                        let admitted = self.try_admit_retained(&mut publish_snapshot);
                        if admitted {
                            entry.insert(HashedCategoryCount {
                                value: value_factory(),
                                count: delta,
                            });
                        }
                        admitted
                    }
                }
            };
            if publish_snapshot {
                self.publish_admitted_snapshot();
            }
            admitted
        } else {
            self.record_hashed_map_access();
            match counts.entry(hash) {
                dashmap::mapref::entry::Entry::Occupied(mut entry) => {
                    let stored = entry.get_mut();
                    if identity_matches(&stored.value) {
                        Self::checked_add_count(&mut stored.count, delta);
                        true
                    } else {
                        Self::merge_collided(
                            collided,
                            delta,
                            identity_matches,
                            value_factory,
                            || self.admit_exhaustive(),
                        )
                    }
                }
                dashmap::mapref::entry::Entry::Vacant(entry) => {
                    entry.insert(HashedCategoryCount {
                        value: value_factory(),
                        count: delta,
                    });
                    self.admit_exhaustive()
                }
            }
        }
    }

    fn merge_hashed(&self, hash: u64, value_str: &str, delta: u64) -> bool {
        self.merge_hashed_with(
            hash,
            delta,
            |stored| matches!(stored, CategoryIdentity::String(value) if value == value_str),
            || CategoryIdentity::String(value_str.to_owned()),
        )
    }

    fn merge_hashed_i64(&self, hash: u64, value: i64, delta: u64) -> bool {
        self.merge_hashed_with(
            hash,
            delta,
            |stored| matches!(stored, CategoryIdentity::I64(stored) if *stored == value),
            || CategoryIdentity::I64(value),
        )
    }

    /// Count one occurrence in shared Hybrid storage. Returns whether the
    /// category was already admitted or won admission.
    #[inline]
    pub fn observe_hashed(&self, hash: u64, value_str: &str) -> bool {
        self.merge_hashed(hash, value_str, 1)
    }

    /// Count an integer-backed category without formatting cached, admitted,
    /// or rejected observations. Typed identity remains an integer until the
    /// retained vocabulary is formatted once during publication.
    #[inline]
    pub fn observe_hashed_i64(&self, hash: u64, value: i64) -> bool {
        self.merge_hashed_i64(hash, value, 1)
    }

    /// Merge a batch of local counts into the shared DashMap (Hybrid flush).
    pub fn merge_local_batch(&self, local: &mut LocalColumnCounts) {
        if self.aborted.load(Ordering::Relaxed) {
            return;
        }
        if matches!(&self.storage, CountStorage::Hashed { .. }) {
            for (&hash, &count) in &local.counts {
                if let Some(integer_value) = local.integer_values.get(&hash) {
                    self.merge_hashed_i64(hash, *integer_value, count);
                } else if let Some(value) = local.strings.get(&hash) {
                    self.merge_hashed(hash, value, count);
                } else {
                    continue;
                }
                if self.aborted.load(Ordering::Relaxed) {
                    return;
                }
            }
            for (hash, value, count) in &local.collided {
                self.merge_hashed(*hash, value, *count);
                if self.aborted.load(Ordering::Relaxed) {
                    return;
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
                // `strings` is write-once; the vacant/occupied entry is the single
                // arbitration point, so concurrent flushes of the same new hash
                // count it as unique exactly once rather than once per worker.
                let is_truly_new = match strings.entry(hash) {
                    dashmap::mapref::entry::Entry::Occupied(_) => false,
                    dashmap::mapref::entry::Entry::Vacant(slot) => match local.strings.remove(&hash)
                    {
                        Some(value) => {
                            slot.insert(value);
                            true
                        }
                        None => false,
                    },
                };
                // Winning the write-once `strings` slot is the unique-count event:
                // count it once here, independent of pending total or graduation, so
                // a concurrent `pending` update can't skip it. A re-inserted graduated
                // category is already in `strings` (not truly new), so no double-count.
                if is_truly_new {
                    self.record_new_unique();
                    if self.aborted.load(Ordering::Relaxed) {
                        return grad_snapshot;
                    }
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
            CountStorage::Hashed { counts, collided } => {
                let mut entries = Vec::with_capacity(counts.len());
                let publish = |stored: &HashedCategoryCount| {
                    let value = match &stored.value {
                        CategoryIdentity::String(value) => value.clone(),
                        CategoryIdentity::I64(value) => value.to_string(),
                    };
                    (value, stored.count)
                };
                entries.extend(counts.iter().map(|entry| publish(entry.value())));
                entries.extend(collided.lock().iter().map(publish));
                if let Some(max_categories) = self.max_categories {
                    if entries.len() > max_categories as usize {
                        self.capped.store(true, Ordering::Relaxed);
                        entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                        entries.truncate(max_categories as usize);
                    }
                }
                Some(entries.into_iter().collect())
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

    /// Values counted on the collision path.
    pub fn collided_count(&self) -> usize {
        match &self.storage {
            CountStorage::Hashed { collided, .. } => collided.lock().len(),
            CountStorage::Graduation { .. } => 0,
        }
    }

    /// Warns once per column if any value took the collision path.
    pub fn warn_collisions_once(&self, column: &str) {
        let colliding_values = self.collided_count();
        if colliding_values > 0 && !self.collision_warned.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                column,
                colliding_values,
                "category values share a 64-bit hash with another value; counted separately"
            );
        }
    }

    pub fn is_aborted(&self) -> bool {
        self.aborted.load(Ordering::Relaxed)
    }

    /// Whether at least one category was excluded or an internal concurrent
    /// admission overshoot requires final trimming.
    pub fn is_capped(&self) -> bool {
        self.capped.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Per-file local category count state (Hybrid & Graduation)
// ---------------------------------------------------------------------------

/// Per-file local category counts for a single column.
pub struct LocalColumnCounts {
    pub counts: HashMap<u64, u64>,
    pub strings: HashMap<u64, String>,
    integer_values: HashMap<u64, i64>,
    /// `(hash, value, count)` for values whose hash `strings` holds for another value.
    collided: Vec<(u64, String, u64)>,
}

impl LocalColumnCounts {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            counts: HashMap::with_capacity(cap),
            strings: HashMap::with_capacity(cap),
            // Keep string-backed categorical caches at their prior allocation
            // footprint. Integer identity storage allocates lazily only when
            // an integer category actually wins admission.
            integer_values: HashMap::new(),
            collided: Vec::new(),
        }
    }

    /// Observe a value locally. Returns true if the key is new.
    #[inline]
    pub fn observe(&mut self, hash: u64, value_str: &str) -> bool {
        use std::collections::hash_map::Entry;
        match self.counts.entry(hash) {
            Entry::Occupied(mut e) => {
                let count = if self.strings.get(&hash).map(String::as_str) == Some(value_str) {
                    e.get_mut()
                } else if let Some((_, _, count)) = self
                    .collided
                    .iter_mut()
                    .find(|(h, value, _)| *h == hash && value == value_str)
                {
                    count
                } else {
                    self.collided.push((hash, value_str.to_owned(), 1));
                    return true;
                };
                *count = count
                    .checked_add(1)
                    .expect("exact local category occurrence count overflowed u64");
                false
            }
            Entry::Vacant(e) => {
                e.insert(1);
                self.strings.insert(hash, value_str.to_owned());
                true
            }
        }
    }

    /// Increment a cached admitted key without allocating on a miss.
    #[inline]
    pub fn increment_if_present(&mut self, hash: u64, value_str: &str) -> bool {
        let Some(count) = self.counts.get_mut(&hash) else {
            return false;
        };
        if self.strings.get(&hash).map(String::as_str) != Some(value_str) {
            return false;
        }
        *count = count
            .checked_add(1)
            .expect("exact local category occurrence count overflowed u64");
        true
    }

    /// Integer equivalent of [`Self::increment_if_present`]. The typed
    /// identity makes the collision check allocation-free.
    #[inline]
    pub fn increment_i64_if_present(&mut self, hash: u64, value: i64) -> bool {
        let Some(count) = self.counts.get_mut(&hash) else {
            return false;
        };
        if self.integer_values.get(&hash) != Some(&value) {
            return false;
        }
        *count = count
            .checked_add(1)
            .expect("exact local category occurrence count overflowed u64");
        true
    }

    /// Cache a key whose current observation was already counted by shared
    /// admission. Subsequent local values are unmerged deltas only. A hash
    /// already cached for another value stays uncached.
    pub fn cache_admitted(&mut self, hash: u64, value_str: &str) {
        if let std::collections::hash_map::Entry::Vacant(e) = self.counts.entry(hash) {
            e.insert(0);
            self.strings.insert(hash, value_str.to_owned());
        }
    }

    /// Cache a newly admitted integer without formatting it. Publication owns
    /// the one eventual integer-to-string conversion.
    pub fn cache_admitted_i64(&mut self, hash: u64, value: i64) {
        if let std::collections::hash_map::Entry::Vacant(e) = self.counts.entry(hash) {
            e.insert(0);
            self.integer_values.insert(hash, value);
        }
    }

    pub fn len(&self) -> usize {
        self.counts.len() + self.collided.len()
    }

    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    pub fn clear(&mut self) {
        self.counts.clear();
        self.strings.clear();
        self.integer_values.clear();
        self.collided.clear();
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
