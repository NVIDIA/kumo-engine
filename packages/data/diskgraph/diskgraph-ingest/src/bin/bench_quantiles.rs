// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Microbenchmark: compare quantile estimator throughput and accuracy.
//!
//! Estimators:
//!   1. Reservoir E-S (k=1000)
//!   2. Reservoir E-S (k=10000)
//!   3. Reservoir E-S + Algorithm L skip-ahead (k=10000)
//!   4. T-Digest (δ=200)
//!   5. KLL Sketch (k=200, ε ≈ 0.5%)
//!
//! Usage:
//!   cargo run --release --bin bench_quantiles [-- --n 10000000]

use std::time::Instant;

// ============================================================================
// 1. Reservoir E-S (classic, per-item RNG)
// ============================================================================

struct ReservoirES {
    samples: Vec<(f64, f64)>, // (value, priority)
    k: usize,
    min_priority: f64,
    min_idx: usize,
    rng: u64,
}

impl ReservoirES {
    fn new(k: usize) -> Self {
        Self {
            samples: Vec::with_capacity(k),
            k,
            min_priority: f64::INFINITY,
            min_idx: 0,
            rng: 0x517cc1b727220a95,
        }
    }

    #[inline]
    fn observe(&mut self, value: f64) {
        let priority = self.next_uniform();
        if self.samples.len() < self.k {
            if priority < self.min_priority {
                self.min_priority = priority;
                self.min_idx = self.samples.len();
            }
            self.samples.push((value, priority));
        } else if priority > self.min_priority {
            self.samples[self.min_idx] = (value, priority);
            self.rescan_min();
        }
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

    fn quantiles(&self) -> (f64, f64, f64) {
        let mut values: Vec<f64> = self.samples.iter().map(|(v, _)| *v).collect();
        values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let n = values.len();
        (
            values[(n as f64 * 0.25) as usize],
            values[(n as f64 * 0.50) as usize],
            values[(n as f64 * 0.75) as usize],
        )
    }

    fn merge(&mut self, other: &ReservoirES) {
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

// ============================================================================
// 2. Reservoir E-S + Algorithm L (skip-ahead, fewer RNG calls)
// ============================================================================

struct ReservoirL {
    samples: Vec<(f64, f64)>, // (value, priority)
    k: usize,
    min_priority: f64,
    min_idx: usize,
    rng: u64,
    items_seen: u64,
    next_accept: u64,
    filled: bool,
}

impl ReservoirL {
    fn new(k: usize) -> Self {
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
    fn observe(&mut self, value: f64) {
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

        // Accept this item — generate priority conditioned on > min_priority
        // U(min_priority, 1) = min_priority + U*(1-min_priority)
        let u = self.next_uniform();
        let priority = self.min_priority + u * (1.0 - self.min_priority);
        self.samples[self.min_idx] = (value, priority);
        self.rescan_min();
        self.compute_next_accept();
    }

    fn compute_next_accept(&mut self) {
        let u = self.next_uniform();
        // Gap ~ Geometric(1 - min_priority)
        // skip = floor(ln(u) / ln(min_priority))
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

    fn quantiles(&self) -> (f64, f64, f64) {
        let mut values: Vec<f64> = self.samples.iter().map(|(v, _)| *v).collect();
        values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let n = values.len();
        (
            values[(n as f64 * 0.25) as usize],
            values[(n as f64 * 0.50) as usize],
            values[(n as f64 * 0.75) as usize],
        )
    }

    fn merge(&mut self, other: &ReservoirL) {
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

// ============================================================================
// 3. T-Digest (δ=200)
// ============================================================================

struct TDigest {
    centroids: Vec<(f64, f64)>, // (mean, weight)
    max_size: usize,            // δ — max centroids
    count: f64,
    buffer: Vec<f64>,
    buf_cap: usize,
}

impl TDigest {
    fn new(delta: usize) -> Self {
        let buf_cap = delta; // flush when buffer reaches δ
        Self {
            centroids: Vec::new(),
            max_size: delta,
            count: 0.0,
            buffer: Vec::with_capacity(buf_cap),
            buf_cap,
        }
    }

    #[inline]
    fn observe(&mut self, value: f64) {
        self.buffer.push(value);
        if self.buffer.len() >= self.buf_cap {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if self.buffer.is_empty() {
            return;
        }
        self.buffer
            .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        // Merge buffer values into centroids (each with weight 1)
        let buf_centroids: Vec<(f64, f64)> = self.buffer.drain(..).map(|v| (v, 1.0)).collect();
        self.merge_centroids(buf_centroids);
    }

    fn merge_centroids(&mut self, other: Vec<(f64, f64)>) {
        // Merge two sorted centroid lists, then compress
        let mut combined = Vec::with_capacity(self.centroids.len() + other.len());

        let mut i = 0;
        let mut j = 0;
        while i < self.centroids.len() && j < other.len() {
            if self.centroids[i].0 <= other[j].0 {
                combined.push(self.centroids[i]);
                i += 1;
            } else {
                combined.push(other[j]);
                j += 1;
            }
        }
        while i < self.centroids.len() {
            combined.push(self.centroids[i]);
            i += 1;
        }
        while j < other.len() {
            combined.push(other[j]);
            j += 1;
        }

        // Update total count
        self.count = combined.iter().map(|(_, w)| w).sum();

        // Compress using k1 scale function
        self.centroids = self.compress(combined);
    }

    fn compress(&self, sorted: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
        if sorted.is_empty() {
            return sorted;
        }
        let total = self.count;
        if total == 0.0 {
            return sorted;
        }

        let mut result: Vec<(f64, f64)> = Vec::with_capacity(self.max_size);
        let mut cur_mean = sorted[0].0;
        let mut cur_weight = sorted[0].1;
        let mut weight_so_far = 0.0;

        for &(mean, weight) in &sorted[1..] {
            // Quantile of the proposed merged centroid's center
            let q = (weight_so_far + (cur_weight + weight) / 2.0) / total;
            // Max weight at quantile q: use k1 scale → limit = 4*n/δ * q*(1-q)
            let limit = 4.0 * total / self.max_size as f64 * q * (1.0 - q);
            if cur_weight + weight <= limit {
                // Merge into current centroid
                let new_weight = cur_weight + weight;
                cur_mean = (cur_mean * cur_weight + mean * weight) / new_weight;
                cur_weight = new_weight;
            } else {
                // Emit current centroid, start new one
                weight_so_far += cur_weight;
                result.push((cur_mean, cur_weight));
                cur_mean = mean;
                cur_weight = weight;
            }
        }
        result.push((cur_mean, cur_weight));
        result
    }

    fn quantile(&self, q: f64) -> f64 {
        // Ensure flushed (caller should call flush first)
        assert!(
            self.buffer.is_empty(),
            "call flush() before querying quantiles"
        );
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
                // Interpolate within this centroid
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

    fn quantiles(&mut self) -> (f64, f64, f64) {
        self.flush();
        (
            self.quantile(0.25),
            self.quantile(0.50),
            self.quantile(0.75),
        )
    }

    fn merge_from(&mut self, other: &mut TDigest) {
        other.flush();
        self.flush();
        let other_centroids = other.centroids.clone();
        self.merge_centroids(other_centroids);
    }
}

// ============================================================================
// 4. KLL Sketch
// ============================================================================

struct KllSketch {
    compactors: Vec<Vec<f64>>,
    k: usize,
    count: u64,
    rng: u64,
}

impl KllSketch {
    fn new(k: usize) -> Self {
        Self {
            compactors: vec![Vec::new()],
            k,
            count: 0,
            rng: 0x517cc1b727220a95,
        }
    }

    fn capacity_at_level(&self, level: usize) -> usize {
        // Capacity at level h: ceil(k * (2/3)^h), minimum 2
        let cap = (self.k as f64 * (2.0_f64 / 3.0).powi(level as i32)).ceil() as usize;
        cap.max(2)
    }

    #[inline]
    fn observe(&mut self, value: f64) {
        self.count += 1;
        self.compactors[0].push(value);
        if self.compactors[0].len() >= self.capacity_at_level(0) {
            self.compact(0);
        }
    }

    fn compact(&mut self, level: usize) {
        // Sort the compactor
        self.compactors[level].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        // Random offset: keep even or odd indexed elements
        let offset = (self.next_u64() & 1) as usize;

        // Collect every other element starting from offset
        let promoted: Vec<f64> = self.compactors[level]
            .iter()
            .skip(offset)
            .step_by(2)
            .copied()
            .collect();

        // Clear this level
        self.compactors[level].clear();

        // Push promoted to next level
        if level + 1 >= self.compactors.len() {
            self.compactors.push(Vec::new());
        }
        self.compactors[level + 1].extend(promoted);

        // Check if next level also needs compaction
        if self.compactors[level + 1].len() >= self.capacity_at_level(level + 1) {
            self.compact(level + 1);
        }
    }

    fn quantiles(&self) -> (f64, f64, f64) {
        // Collect all items with their weights
        let mut weighted: Vec<(f64, u64)> = Vec::new();
        for (level, compactor) in self.compactors.iter().enumerate() {
            let weight = 1u64 << level;
            for &value in compactor {
                weighted.push((value, weight));
            }
        }
        weighted.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let total_weight: u64 = weighted.iter().map(|(_, w)| w).sum();

        let find_quantile = |q: f64| -> f64 {
            let target = (q * total_weight as f64) as u64;
            let mut cum = 0u64;
            for &(value, weight) in &weighted {
                cum += weight;
                if cum > target {
                    return value;
                }
            }
            weighted.last().map(|(v, _)| *v).unwrap_or(0.0)
        };

        (
            find_quantile(0.25),
            find_quantile(0.50),
            find_quantile(0.75),
        )
    }

    fn merge(&mut self, other: &KllSketch) {
        // Ensure we have enough levels
        while self.compactors.len() < other.compactors.len() {
            self.compactors.push(Vec::new());
        }
        // Merge corresponding levels
        for (level, other_compactor) in other.compactors.iter().enumerate() {
            self.compactors[level].extend(other_compactor);
            // Check capacity
            while self.compactors[level].len() >= self.capacity_at_level(level) {
                self.compact(level);
            }
        }
        self.count += other.count;
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        self.rng.wrapping_mul(0x2545F4914F6CDD1D)
    }
}

// ============================================================================
// Data generation
// ============================================================================

fn generate_uniform(n: usize) -> Vec<f64> {
    let mut rng: u64 = 0xdeadbeefcafe1234;
    (0..n)
        .map(|_| {
            rng ^= rng >> 12;
            rng ^= rng << 25;
            rng ^= rng >> 27;
            let val = rng.wrapping_mul(0x2545F4914F6CDD1D);
            (val >> 11) as f64 * (1.0 / (1u64 << 53) as f64) * 1_000_000.0
        })
        .collect()
}

fn generate_normal(n: usize, mean: f64, std: f64) -> Vec<f64> {
    // Box-Muller transform
    let mut rng: u64 = 0xcafebabe12345678;
    let mut result = Vec::with_capacity(n);
    while result.len() < n {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        let u1 = (rng.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        let u2 = (rng.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        let u1 = u1.max(1e-300); // avoid log(0)
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).sin();
        result.push(mean + z0 * std);
        if result.len() < n {
            result.push(mean + z1 * std);
        }
    }
    result
}

fn generate_lognormal(n: usize, mu: f64, sigma: f64) -> Vec<f64> {
    generate_normal(n, mu, sigma)
        .into_iter()
        .map(|x| x.exp())
        .collect()
}

fn exact_quantiles(data: &[f64]) -> (f64, f64, f64) {
    let mut sorted = data.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    (
        sorted[(n as f64 * 0.25) as usize],
        sorted[(n as f64 * 0.50) as usize],
        sorted[(n as f64 * 0.75) as usize],
    )
}

// ============================================================================
// Benchmark runner
// ============================================================================

fn run_benchmark(name: &str, data: &[f64], exact: (f64, f64, f64)) {
    let n = data.len();
    let range = exact.2 - exact.0; // IQR for relative error

    println!("\n{}", "=".repeat(60));
    println!("  Distribution: {name}  (n={n})");
    println!(
        "  Exact: Q25={:.2} Q50={:.2} Q75={:.2}",
        exact.0, exact.1, exact.2
    );
    println!("{}", "=".repeat(60));
    println!(
        "  {:30} {:>10} {:>8} {:>8} {:>8} {:>8}",
        "Estimator", "ns/value", "Q25 err%", "Q50 err%", "Q75 err%", "Memory"
    );
    println!(
        "  {:-<30} {:-<10} {:-<8} {:-<8} {:-<8} {:-<8}",
        "", "", "", "", "", ""
    );

    // Helper for error computation
    let pct_err = |est: f64, exact_val: f64| -> f64 {
        if range.abs() < 1e-10 {
            (est - exact_val).abs()
        } else {
            (est - exact_val).abs() / range * 100.0
        }
    };

    // 1. Reservoir E-S k=1000
    {
        let start = Instant::now();
        let mut r = ReservoirES::new(1_000);
        for &v in data {
            r.observe(v);
        }
        let elapsed = start.elapsed();
        let (q25, q50, q75) = r.quantiles();
        let mem = 1_000 * 16; // 16 bytes per (f64, f64)
        println!(
            "  {:30} {:>10.1} {:>7.2}% {:>7.2}% {:>7.2}% {:>7}",
            "Reservoir ES k=1000",
            elapsed.as_nanos() as f64 / n as f64,
            pct_err(q25, exact.0),
            pct_err(q50, exact.1),
            pct_err(q75, exact.2),
            format!("{}KB", mem / 1024),
        );
    }

    // 2. Reservoir E-S k=10000
    {
        let start = Instant::now();
        let mut r = ReservoirES::new(10_000);
        for &v in data {
            r.observe(v);
        }
        let elapsed = start.elapsed();
        let (q25, q50, q75) = r.quantiles();
        let mem = 10_000 * 16;
        println!(
            "  {:30} {:>10.1} {:>7.2}% {:>7.2}% {:>7.2}% {:>7}",
            "Reservoir ES k=10000",
            elapsed.as_nanos() as f64 / n as f64,
            pct_err(q25, exact.0),
            pct_err(q50, exact.1),
            pct_err(q75, exact.2),
            format!("{}KB", mem / 1024),
        );
    }

    // 3. Reservoir Algorithm L k=10000
    {
        let start = Instant::now();
        let mut r = ReservoirL::new(10_000);
        for &v in data {
            r.observe(v);
        }
        let elapsed = start.elapsed();
        let (q25, q50, q75) = r.quantiles();
        let mem = 10_000 * 16;
        println!(
            "  {:30} {:>10.1} {:>7.2}% {:>7.2}% {:>7.2}% {:>7}",
            "Reservoir L k=10000",
            elapsed.as_nanos() as f64 / n as f64,
            pct_err(q25, exact.0),
            pct_err(q50, exact.1),
            pct_err(q75, exact.2),
            format!("{}KB", mem / 1024),
        );
    }

    // 4. T-Digest (δ=200)
    {
        let start = Instant::now();
        let mut td = TDigest::new(200);
        for &v in data {
            td.observe(v);
        }
        let (q25, q50, q75) = td.quantiles();
        let elapsed = start.elapsed();
        let mem = 200 * 16 + 200 * 8; // centroids + buffer
        println!(
            "  {:30} {:>10.1} {:>7.2}% {:>7.2}% {:>7.2}% {:>7}",
            "T-Digest δ=200",
            elapsed.as_nanos() as f64 / n as f64,
            pct_err(q25, exact.0),
            pct_err(q50, exact.1),
            pct_err(q75, exact.2),
            format!("{}KB", mem / 1024),
        );
    }

    // 5. KLL Sketch (k=200)
    {
        let start = Instant::now();
        let mut kll = KllSketch::new(200);
        for &v in data {
            kll.observe(v);
        }
        let elapsed = start.elapsed();
        let (q25, q50, q75) = kll.quantiles();
        // Memory: sum of all compactor capacities
        let mem_items: usize = kll.compactors.iter().map(|c| c.len()).sum();
        println!(
            "  {:30} {:>10.1} {:>7.2}% {:>7.2}% {:>7.2}% {:>7}",
            "KLL k=200",
            elapsed.as_nanos() as f64 / n as f64,
            pct_err(q25, exact.0),
            pct_err(q50, exact.1),
            pct_err(q75, exact.2),
            format!("{}KB", mem_items * 8 / 1024),
        );
    }

    // 6. T-Digest (δ=100) — smaller
    {
        let start = Instant::now();
        let mut td = TDigest::new(100);
        for &v in data {
            td.observe(v);
        }
        let (q25, q50, q75) = td.quantiles();
        let elapsed = start.elapsed();
        let mem = 100 * 16 + 100 * 8;
        println!(
            "  {:30} {:>10.1} {:>7.2}% {:>7.2}% {:>7.2}% {:>7}",
            "T-Digest δ=100",
            elapsed.as_nanos() as f64 / n as f64,
            pct_err(q25, exact.0),
            pct_err(q50, exact.1),
            pct_err(q75, exact.2),
            format!("{}KB", mem / 1024),
        );
    }
}

fn run_merge_benchmark(name: &str, data: &[f64], exact: (f64, f64, f64)) {
    let n = data.len();
    let range = exact.2 - exact.0;
    let num_partitions = 100; // simulate 100 files
    let chunk_size = n / num_partitions;

    let pct_err = |est: f64, exact_val: f64| -> f64 {
        if range.abs() < 1e-10 {
            (est - exact_val).abs()
        } else {
            (est - exact_val).abs() / range * 100.0
        }
    };

    println!("\n  Merge test ({num_partitions} partitions, {name}):");
    println!(
        "  {:30} {:>10} {:>8} {:>8} {:>8}",
        "Estimator", "merge μs", "Q25 err%", "Q50 err%", "Q75 err%"
    );
    println!("  {:-<30} {:-<10} {:-<8} {:-<8} {:-<8}", "", "", "", "", "");

    // Reservoir ES k=10000 merge
    {
        let mut parts: Vec<ReservoirES> = Vec::new();
        for chunk in data.chunks(chunk_size) {
            let mut r = ReservoirES::new(10_000);
            for &v in chunk {
                r.observe(v);
            }
            parts.push(r);
        }
        let start = Instant::now();
        let mut acc = parts.remove(0);
        for p in &parts {
            acc.merge(p);
        }
        let elapsed = start.elapsed();
        let (q25, q50, q75) = acc.quantiles();
        println!(
            "  {:30} {:>10.0} {:>7.2}% {:>7.2}% {:>7.2}%",
            "Reservoir ES k=10000",
            elapsed.as_micros(),
            pct_err(q25, exact.0),
            pct_err(q50, exact.1),
            pct_err(q75, exact.2),
        );
    }

    // Reservoir L k=10000 merge
    {
        let mut parts: Vec<ReservoirL> = Vec::new();
        for chunk in data.chunks(chunk_size) {
            let mut r = ReservoirL::new(10_000);
            for &v in chunk {
                r.observe(v);
            }
            parts.push(r);
        }
        let start = Instant::now();
        let mut acc = parts.remove(0);
        for p in &parts {
            acc.merge(p);
        }
        let elapsed = start.elapsed();
        let (q25, q50, q75) = acc.quantiles();
        println!(
            "  {:30} {:>10.0} {:>7.2}% {:>7.2}% {:>7.2}%",
            "Reservoir L k=10000",
            elapsed.as_micros(),
            pct_err(q25, exact.0),
            pct_err(q50, exact.1),
            pct_err(q75, exact.2),
        );
    }

    // T-Digest merge
    {
        let mut parts: Vec<TDigest> = Vec::new();
        for chunk in data.chunks(chunk_size) {
            let mut td = TDigest::new(200);
            for &v in chunk {
                td.observe(v);
            }
            td.flush();
            parts.push(td);
        }
        let start = Instant::now();
        let mut acc = parts.remove(0);
        for p in &mut parts {
            acc.merge_from(p);
        }
        let elapsed = start.elapsed();
        let (q25, q50, q75) = acc.quantiles();
        println!(
            "  {:30} {:>10.0} {:>7.2}% {:>7.2}% {:>7.2}%",
            "T-Digest δ=200",
            elapsed.as_micros(),
            pct_err(q25, exact.0),
            pct_err(q50, exact.1),
            pct_err(q75, exact.2),
        );
    }

    // KLL merge
    {
        let mut parts: Vec<KllSketch> = Vec::new();
        for chunk in data.chunks(chunk_size) {
            let mut kll = KllSketch::new(200);
            for &v in chunk {
                kll.observe(v);
            }
            parts.push(kll);
        }
        let start = Instant::now();
        let mut acc = parts.remove(0);
        for p in &parts {
            acc.merge(p);
        }
        let elapsed = start.elapsed();
        let (q25, q50, q75) = acc.quantiles();
        println!(
            "  {:30} {:>10.0} {:>7.2}% {:>7.2}% {:>7.2}%",
            "KLL k=200",
            elapsed.as_micros(),
            pct_err(q25, exact.0),
            pct_err(q50, exact.1),
            pct_err(q75, exact.2),
        );
    }
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.strip_prefix("--n=").map(String::from).or(Some(s)))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000_000);

    println!("Quantile Estimator Benchmark");
    println!("n = {n} values per distribution");

    // Generate data
    println!("\nGenerating data...");
    let uniform = generate_uniform(n);
    let normal = generate_normal(n, 5000.0, 2000.0);
    let lognormal = generate_lognormal(n, 3.0, 1.5);

    // Exact quantiles
    println!("Computing exact quantiles...");
    let exact_uniform = exact_quantiles(&uniform);
    let exact_normal = exact_quantiles(&normal);
    let exact_lognormal = exact_quantiles(&lognormal);

    // Run observation benchmarks
    run_benchmark("Uniform [0, 1M)", &uniform, exact_uniform);
    run_benchmark("Normal(5000, 2000)", &normal, exact_normal);
    run_benchmark("Lognormal(3.0, 1.5)", &lognormal, exact_lognormal);

    // Run merge benchmarks
    run_merge_benchmark("Uniform", &uniform, exact_uniform);
    run_merge_benchmark("Normal", &normal, exact_normal);
    run_merge_benchmark("Lognormal", &lognormal, exact_lognormal);

    println!("\nDone.");
}
