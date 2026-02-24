//! Shared streaming accumulators used by both col-stats (ingestion) and
//! TT-gen (training table generation).
//!
//! These are pure data structures with no crate-specific dependencies beyond
//! `gxhash` (for HyperLogLog hashing).

use std::hash::Hasher;

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
}
