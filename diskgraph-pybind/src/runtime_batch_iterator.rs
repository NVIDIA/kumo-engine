use arrow::pyarrow::ToPyArrow;
use arrow_array::{Array, Int64Array, RecordBatch, UInt32Array, UInt64Array};
use diskgraph_core::sampling::{NegativeSamplingSpec, SamplingFlags, SamplingPlan};
use diskgraph_engine::engine::DiskGraphEngine;
use diskgraph_index::graph::TableReader;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::seq::SliceRandom;
use rand::Rng;
use rand::{rngs::StdRng, SeedableRng};
use serde::Deserialize;
use std::sync::Arc;

use crate::{map_engine_err, DiskgraphConfigError, DiskgraphDataError, DiskgraphSystemError};

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn epoch_shuffle_seed(base_seed: u64, epoch: u64, stage_key: &str) -> u64 {
    let mut h = splitmix64(base_seed);
    h ^= splitmix64(epoch);
    for byte in stage_key.as_bytes() {
        h = splitmix64(h ^ u64::from(*byte));
    }
    h
}

/// Rounds only affect how well-mixed the output is. Every round is a
/// bijection, so the ordering is a valid permutation at any count -- nothing
/// about correctness depends on this number.
///
/// More rounds mix better and each is a few XORs, so there is no reason to
/// skimp; the format-preserving-encryption standards built on this
/// construction use 8 (FF3-1) and 10 (FF1). We take 10 and check the result
/// directly: see `a_shard_is_spread_evenly_across_the_table` and
/// `consecutive_rows_of_a_shard_are_far_apart`.
const FEISTEL_ROUNDS: usize = 10;

/// Below this many rows, shuffle by materialising the order instead.
///
/// This is a COMPATIBILITY boundary, not a correctness or quality one. The
/// generator is a valid permutation at any size, and the mixing tests run
/// well above this.
///
/// Below a million rows the problem being solved does not exist: a million
/// u64 offsets is 8 MB. Keeping Fisher-Yates there means every table under
/// that size produces a BYTE-IDENTICAL ordering, which confines the
/// behaviour change to the tables actually large enough to need it.
const MATERIALISE_BELOW: u64 = 1_000_000;

/// A seeded pseudorandom permutation of `[0, n)`, evaluable POINTWISE in
/// constant time and constant memory.
///
/// # Why this exists
///
/// The ordering it replaces was materialised: `(0..n).collect()` followed by a
/// Fisher-Yates shuffle -- 334 MB and ~1.6 s for a 41.8M-row stage table --
/// then read exactly once, front to back, and dropped. It was a stream stored
/// as an array, and it cost O(all rows) however few rows the caller wanted.
///
/// # What is needed
///
/// To avoid the array we need to answer "which row ends up at position p?" on
/// demand. That function must be:
///
///   1. a BIJECTION on `[0, n)` -- non-negotiable. If two inputs mapped to one
///      output, some rows would be trained on twice and others never, with
///      correct shapes, correct counts and nothing to notice.
///   2. pointwise -- computable for any `p` without computing the rest.
///   3. seeded -- so shards stay reproducible and disjoint.
///
/// A hash fails (1): hashes collide by design. A linear map `(a*p + b) % n`
/// satisfies (1) but is a fixed-stride rotation -- it reorders without mixing,
/// so any structure in the input order survives.
///
/// # How Feistel gets both
///
/// Split a value into halves `(l, r)` and apply `l' = r`, `r' = l ^ F(r)`.
/// This is invertible for ANY `F`, because `r = l'` and `l = r' ^ F(l')` --
/// undoing it only ever EVALUATES `F`, never inverts it. The XOR carries the
/// invertibility; `F` supplies only the mixing, which is why `F` can be chosen
/// purely for speed.
///
/// Each round is a bijection, so any number of rounds composed is one too.
/// Round count affects only how well mixed the output is, never whether it is
/// a permutation -- correctness and quality stay cleanly separated. See
/// `FEISTEL_ROUNDS` for why the count is 10 and not the textbook 3-4.
///
/// # Cycle walking, for domains that are not powers of two
///
/// Feistel works on a fixed bit width, so take the smallest `2^(2*half_bits)`
/// covering `n` and re-apply until the result lands back inside `[0, n)`.
///
/// That is still a bijection: the permutation partitions the padded space into
/// disjoint cycles, so stepping along a cycle from a point inside `[0, n)`
/// must return to `[0, n)` -- it is a closed loop containing that point. Taking
/// each cycle to its next in-range element is thus a bijection of the in-range
/// subset onto itself.
///
/// It is also fast: the in-range fraction is `n / 2^(2*half_bits)`, never below
/// one half, so the expected number of steps is under 2 for any `n`. This is
/// how NIST's format-preserving encryption (FF1, FF3) handles arbitrary
/// domains.
#[derive(Clone)]
struct FeistelPerm {
    keys: [u64; FEISTEL_ROUNDS],
    n: u64,
    /// Widths of the two halves. UNBALANCED when `ceil(log2 n)` is odd, so
    /// the block is always the true minimal power of two >= n. A balanced
    /// split would force an even block width, rounding the domain up by two
    /// bits instead of one whenever the ideal width is odd -- which drops the
    /// in-range fraction from >= 0.5 to ~0.25 and doubles the expected
    /// cycle-walk cost for roughly half of all table sizes. FF1 and FF3 split
    /// unevenly for the same reason.
    left_bits: u32,
    right_bits: u32,
}

impl FeistelPerm {
    fn new(n: u64, seed: u64) -> Self {
        // The true minimal block: ceil(log2 n) bits, floored at 2 so neither
        // half is zero-width (a zero-width half is not invertible).
        let total_bits = if n <= 2 {
            2
        } else {
            (64 - (n - 1).leading_zeros()).max(2)
        };
        let left_bits = total_bits / 2;
        let right_bits = total_bits - left_bits;
        let mut keys = [0u64; FEISTEL_ROUNDS];
        let mut h = splitmix64(seed);
        for k in keys.iter_mut() {
            h = splitmix64(h);
            *k = h;
        }
        Self { keys, n, left_bits, right_bits }
    }

    #[inline]
    fn mask(bits: u32) -> u64 {
        if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 }
    }

    #[inline]
    fn permute_block(&self, x: u64) -> u64 {
        // With unequal halves the two swap WIDTHS as well as roles each
        // round, so the widths are tracked alongside the values. FEISTEL_ROUNDS
        // is even, which returns them to the original layout for reassembly.
        let (mut a, mut b) = (self.left_bits, self.right_bits);
        let mut l = x >> b;
        let mut r = x & Self::mask(b);
        for i in 0..FEISTEL_ROUNDS {
            let nl = r;
            r = (l ^ (splitmix64(r ^ self.keys[i]) & Self::mask(a))) & Self::mask(a);
            l = nl;
            std::mem::swap(&mut a, &mut b);
        }
        (l << self.right_bits) | r
    }

    /// The image of `i` under the permutation.
    ///
    /// `i` MUST be `< n`. Out of range the cycle walk still terminates and
    /// returns a value in `[0, n)`, so an over-read would look entirely
    /// plausible while being wrong -- unlike the materialised path, which
    /// panics on the same mistake. Assert so both fail identically and
    /// loudly rather than one of them silently.
    #[inline]
    fn at(&self, i: u64) -> u64 {
        debug_assert!(
            i < self.n,
            "FeistelPerm::at({i}) out of range for domain {}",
            self.n
        );
        if self.n <= 1 {
            return i;
        }
        let mut x = i;
        loop {
            x = self.permute_block(x);
            if x < self.n {
                return x;
            }
        }
    }
}

/// One worker's slice of the epoch, generated on demand.
///
/// Replaces the materialised `Vec<u64>` of row offsets. The contract is
/// deliberately identical to what it replaces: worker `w` of `k` receives the
/// elements of the (possibly shuffled) order at positions `w, w+k, w+2k, ...`,
/// so coverage and disjointness hold for exactly the same reason they did
/// before -- every position belongs to exactly one worker.
/// How this epoch's ordering is obtained. Both variants describe the SAME
/// kind of object -- a permutation of `[0, n)` -- and differ only in whether
/// it is stored or computed.
#[derive(Clone)]
enum Order {
    /// `shuffle=false`: position p holds row p.
    Identity,
    /// Small domains: the classic Fisher-Yates array, shared between the
    /// stream's clones so it is stored once.
    Materialised(Arc<Vec<u64>>),
    /// Large domains: computed pointwise, no allocation.
    Generated(FeistelPerm),
}

#[derive(Clone)]
struct ShardStream {
    order: Order,
    stride: u64,
    start: u64,
    /// Added to every emitted value. Non-zero only in NodeRange mode, where
    /// the values ARE global node ids rather than stage-table offsets.
    base: u64,
    len: usize,
}

impl ShardStream {
    fn empty() -> Self {
        Self { order: Order::Identity, stride: 1, start: 0, base: 0, len: 0 }
    }

    fn new(
        input_count: u64,
        num_workers: u64,
        worker_id: u64,
        shuffle: bool,
        rnd_seed: Option<u64>,
        epoch: u64,
        stage_key: &str,
        base: u64,
    ) -> PyResult<Self> {
        if num_workers == 0 {
            return Err(DiskgraphConfigError::new_err("num_workers must be >= 1"));
        }
        if worker_id >= num_workers {
            return Err(DiskgraphConfigError::new_err(format!(
                "worker_id {worker_id} out of range for num_workers {num_workers}"
            )));
        }
        if shuffle && rnd_seed.is_none() {
            return Err(DiskgraphConfigError::new_err(
                "rnd_seed is required when shuffle=True",
            ));
        }
        // Count of positions p < input_count with p % num_workers == worker_id.
        // Closed form, so `len()` costs nothing even for billions of rows.
        let len = if worker_id >= input_count {
            0usize
        } else {
            (input_count - worker_id).div_ceil(num_workers) as usize
        };
        let seed = epoch_shuffle_seed(rnd_seed.unwrap_or(0), epoch, stage_key);
        let order = if !shuffle {
            Order::Identity
        } else if input_count < MATERIALISE_BELOW {
            // Small domain: materialise, both because Feistel is weakest here
            // and because 8 MB is not worth avoiding. Byte-identical to the
            // ordering this replaces.
            let mut v: Vec<u64> = (0..input_count).collect();
            v.shuffle(&mut StdRng::seed_from_u64(seed));
            Order::Materialised(Arc::new(v))
        } else {
            Order::Generated(FeistelPerm::new(input_count, seed))
        };
        Ok(Self { order, stride: num_workers, start: worker_id, base, len })
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    /// The `j`-th element of this worker's shard.
    #[inline]
    fn at(&self, j: usize) -> u64 {
        debug_assert!(
            j < self.len,
            "ShardStream::at({j}) past the end of a {}-element shard",
            self.len
        );
        let pos = self.start + (j as u64) * self.stride;
        let v = match &self.order {
            Order::Identity => pos,
            Order::Materialised(v) => v[pos as usize],
            Order::Generated(p) => p.at(pos),
        };
        v + self.base
    }

    fn take(&self, from: usize, count: usize) -> Vec<u64> {
        (from..from + count).map(|j| self.at(j)).collect()
    }
}

/// The materialised ordering `ShardStream` replaces. Kept, test-only, as the
/// reference the unshuffled path is asserted byte-identical against -- the
/// clearest statement that this refactor changes only the shuffled order.
#[cfg(test)]
fn make_worker_offsets(
    input_count: u64,
    num_workers: u64,
    worker_id: u64,
    shuffle: bool,
    rnd_seed: Option<u64>,
    epoch: u64,
    stage_key: &str,
) -> PyResult<Vec<u64>> {
    if num_workers == 0 {
        return Err(DiskgraphConfigError::new_err("num_workers must be >= 1"));
    }
    if worker_id >= num_workers {
        return Err(DiskgraphConfigError::new_err(format!(
            "worker_id {worker_id} out of range for num_workers {num_workers}"
        )));
    }
    if shuffle && rnd_seed.is_none() {
        return Err(DiskgraphConfigError::new_err(
            "rnd_seed is required when shuffle=True",
        ));
    }
    let mut order: Vec<u64> = (0..input_count).collect();
    if shuffle {
        let mut rng =
            StdRng::seed_from_u64(epoch_shuffle_seed(rnd_seed.unwrap_or(0), epoch, stage_key));
        order.shuffle(&mut rng);
    }
    Ok(order
        .into_iter()
        .enumerate()
        .filter_map(|(pos, offset)| {
            if (pos as u64) % num_workers == worker_id {
                Some(offset)
            } else {
                None
            }
        })
        .collect())
}

#[derive(Debug, Deserialize)]
struct WeightedSeedSamplingConfig {
    buckets: Vec<WeightedSeedBucketConfig>,
    num_samples: u64,
}

#[derive(Debug, Deserialize)]
struct WeightedSeedBucketConfig {
    name: String,
    node_ids_path: String,
    count: u64,
    mass: f64,
    cdf_path: Option<String>,
    total_weight: Option<f64>,
}

struct U64Sidecar {
    path: String,
    mmap: Option<memmap2::Mmap>,
    len: u64,
}

impl U64Sidecar {
    fn open(path: &str) -> PyResult<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| DiskgraphSystemError::new_err(format!("failed to open {path}: {e}")))?;
        let byte_len = file
            .metadata()
            .map_err(|e| DiskgraphSystemError::new_err(format!("failed to stat {path}: {e}")))?
            .len();
        if byte_len % 8 != 0 {
            return Err(DiskgraphDataError::new_err(format!(
                "u64 sidecar {path} has {byte_len} bytes, not a multiple of 8"
            )));
        }
        let mmap = if byte_len == 0 {
            None
        } else {
            Some(unsafe { memmap2::Mmap::map(&file) }.map_err(|e| {
                DiskgraphSystemError::new_err(format!("failed to mmap {path}: {e}"))
            })?)
        };
        Ok(Self {
            path: path.to_string(),
            mmap,
            len: byte_len / 8,
        })
    }

    fn len(&self) -> u64 {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn get(&self, index: u64) -> PyResult<u64> {
        if index >= self.len {
            return Err(DiskgraphDataError::new_err(format!(
                "u64 sidecar {} index {index} out of range for len {}",
                self.path, self.len
            )));
        }
        let mmap = self.mmap.as_ref().ok_or_else(|| {
            DiskgraphDataError::new_err(format!("u64 sidecar {} is empty", self.path))
        })?;
        let offset = (index as usize) * 8;
        let mut raw = [0u8; 8];
        raw.copy_from_slice(&mmap[offset..offset + 8]);
        Ok(u64::from_le_bytes(raw))
    }
}

struct F64Sidecar {
    path: String,
    mmap: Option<memmap2::Mmap>,
    len: u64,
}

impl F64Sidecar {
    fn open(path: &str) -> PyResult<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| DiskgraphSystemError::new_err(format!("failed to open {path}: {e}")))?;
        let byte_len = file
            .metadata()
            .map_err(|e| DiskgraphSystemError::new_err(format!("failed to stat {path}: {e}")))?
            .len();
        if byte_len % 8 != 0 {
            return Err(DiskgraphDataError::new_err(format!(
                "f64 sidecar {path} has {byte_len} bytes, not a multiple of 8"
            )));
        }
        let mmap = if byte_len == 0 {
            None
        } else {
            Some(unsafe { memmap2::Mmap::map(&file) }.map_err(|e| {
                DiskgraphSystemError::new_err(format!("failed to mmap {path}: {e}"))
            })?)
        };
        Ok(Self {
            path: path.to_string(),
            mmap,
            len: byte_len / 8,
        })
    }

    fn len(&self) -> u64 {
        self.len
    }

    fn get(&self, index: u64) -> PyResult<f64> {
        if index >= self.len {
            return Err(DiskgraphDataError::new_err(format!(
                "f64 sidecar {} index {index} out of range for len {}",
                self.path, self.len
            )));
        }
        let mmap = self.mmap.as_ref().ok_or_else(|| {
            DiskgraphDataError::new_err(format!("f64 sidecar {} is empty", self.path))
        })?;
        let offset = (index as usize) * 8;
        let mut raw = [0u8; 8];
        raw.copy_from_slice(&mmap[offset..offset + 8]);
        Ok(f64::from_le_bytes(raw))
    }

    fn partition_point_le(&self, draw: f64) -> PyResult<u64> {
        let mut left = 0u64;
        let mut right = self.len;
        while left < right {
            let mid = left + (right - left) / 2;
            if self.get(mid)? <= draw {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        Ok(left)
    }
}

struct WeightedSeedBucket {
    name: String,
    node_ids: U64Sidecar,
    cdf: Option<F64Sidecar>,
    mass: f64,
    total_weight: Option<f64>,
}

impl WeightedSeedBucket {
    fn open(config: WeightedSeedBucketConfig) -> PyResult<Self> {
        if !config.mass.is_finite() || config.mass < 0.0 {
            return Err(DiskgraphConfigError::new_err(format!(
                "weighted_seed_sampling bucket {} has invalid mass {}",
                config.name, config.mass
            )));
        }
        let node_ids = U64Sidecar::open(&config.node_ids_path)?;
        if node_ids.len() != config.count {
            return Err(DiskgraphDataError::new_err(format!(
                "weighted_seed_sampling bucket {} count mismatch: manifest={} file={}",
                config.name,
                config.count,
                node_ids.len()
            )));
        }
        let cdf = match config.cdf_path {
            Some(path) => {
                let cdf = F64Sidecar::open(&path)?;
                if cdf.len() != config.count {
                    return Err(DiskgraphDataError::new_err(format!(
                        "weighted_seed_sampling bucket {} cdf count mismatch: manifest={} file={}",
                        config.name,
                        config.count,
                        cdf.len()
                    )));
                }
                Some(cdf)
            }
            None => None,
        };
        if let Some(cdf_ref) = &cdf {
            let total = config.total_weight.ok_or_else(|| {
                DiskgraphConfigError::new_err(format!(
                    "weighted_seed_sampling bucket {} missing total_weight",
                    config.name
                ))
            })?;
            if !total.is_finite() || total <= 0.0 {
                return Err(DiskgraphConfigError::new_err(format!(
                    "weighted_seed_sampling bucket {} has invalid total_weight {}",
                    config.name, total
                )));
            }
            if config.count > 0 {
                let observed = cdf_ref.get(config.count - 1)?;
                let tolerance = total.abs().max(1.0) * 1e-9;
                if (observed - total).abs() > tolerance {
                    return Err(DiskgraphDataError::new_err(format!(
                        "weighted_seed_sampling bucket {} total_weight mismatch: manifest={} file={}",
                        config.name, total, observed
                    )));
                }
            }
            Ok(Self {
                name: config.name,
                node_ids,
                cdf,
                mass: config.mass,
                total_weight: Some(total),
            })
        } else {
            Ok(Self {
                name: config.name,
                node_ids,
                cdf,
                mass: config.mass,
                total_weight: None,
            })
        }
    }

    fn sample_node_id(&self, rng: &mut StdRng) -> PyResult<u64> {
        if self.node_ids.is_empty() {
            return Err(DiskgraphDataError::new_err(format!(
                "weighted_seed_sampling bucket {} is empty",
                self.name
            )));
        }
        let index = if let Some(cdf) = self.cdf.as_ref() {
            let total = self.total_weight.expect("cdf bucket has total_weight");
            let draw = rng.gen_range(0.0..total);
            let index = cdf.partition_point_le(draw)?;
            if index >= self.node_ids.len() {
                self.node_ids.len() - 1
            } else {
                index
            }
        } else {
            rng.gen_range(0..self.node_ids.len())
        };
        self.node_ids.get(index)
    }
}

#[allow(clippy::too_many_arguments)]
fn make_weighted_stage_node_ids(
    config_json: &str,
    stage_table: &TableReader,
    stage_key: &str,
    num_workers: u64,
    worker_id: u64,
    rnd_seed: Option<u64>,
    epoch: u64,
) -> PyResult<Vec<u64>> {
    let config: WeightedSeedSamplingConfig = serde_json::from_str(config_json).map_err(|e| {
        DiskgraphConfigError::new_err(format!("Invalid weighted_seed_sampling JSON: {e}"))
    })?;
    if config.num_samples == 0 {
        return Ok(Vec::new());
    }
    let buckets = config
        .buckets
        .into_iter()
        .map(WeightedSeedBucket::open)
        .collect::<PyResult<Vec<_>>>()?;
    let total_mass: f64 = buckets.iter().map(|bucket| bucket.mass).sum();
    if total_mass <= 0.0 {
        return Err(DiskgraphConfigError::new_err(
            "weighted_seed_sampling has zero total sampling mass",
        ));
    }
    let mut rng = StdRng::seed_from_u64(epoch_shuffle_seed(
        rnd_seed.unwrap_or(0),
        epoch,
        &format!("{stage_key}:weighted_seed_sampling"),
    ));
    let mut local_ids = Vec::new();
    for pos in 0..config.num_samples {
        let draw = rng.gen_range(0.0..total_mass);
        let mut running = 0.0;
        let mut selected = buckets.last().ok_or_else(|| {
            DiskgraphConfigError::new_err("weighted_seed_sampling requires at least one bucket")
        })?;
        for bucket in &buckets {
            if bucket.mass <= 0.0 {
                continue;
            }
            running += bucket.mass;
            if draw < running {
                selected = bucket;
                break;
            }
        }
        let node_id = selected.sample_node_id(&mut rng)?;
        if pos % num_workers != worker_id {
            continue;
        }
        if stage_table.shard_for_node(node_id).is_none() {
            return Err(DiskgraphDataError::new_err(format!(
                "weighted_seed_sampling node id {node_id} out of range for stage table {stage_key}"
            )));
        }
        local_ids.push(node_id);
    }
    Ok(local_ids)
}

// ---------------------------------------------------------------------------
// TT-NG temporal-LP timestamp grouping — LOAD side.
//
// When `group_by_timestamp` is set, the iterator yields SINGLE-TIMESTAMP
// batches by consuming the per-stage `bucket_index.bin` + bucket list the
// ingest side recorded in `diskgraph_manifest.json` (see
// `diskgraph-ingest/src/ts_grouping.rs` + `tt_executor.rs`). The bin is a flat
// little-endian `u64` array of FINAL node-ids; bucket `b` occupies
// `[offset_b, offset_b + count_b)` (units = node-ids, byte range ×8).
//
// The schedule is LAZY and O(D)+O(one bucket) RAM (D = #buckets = distinct
// timestamps): we hold only the ordered bucket list + a prefix-sum of
// per-bucket batch counts + the bin path, and read/permute one bucket at a time
// as batches are pulled. Determinism: bucket ORDER and WITHIN-bucket order
// depend only on `(base_seed, epoch, stage_key[, ts])`, never on `worker_id`,
// so every worker computes the identical global order and then keeps its own
// residue class of global batch indices (`i % num_workers == worker_id`) — a
// true partition of all node-ids, single-timestamp by construction.
// ---------------------------------------------------------------------------

/// One bucket in the (epoch-shuffled) schedule: its timestamp (shuffle-seed
/// input + metadata, units = SECONDS — never compared to `seed_time` ns), its
/// `[offset, offset + count)` slice of `bucket_index.bin` (units = node-ids),
/// and the global batch index of its FIRST batch (`batch_start`). The bucket
/// owns global batch indices `[batch_start, batch_start + ceil(count/bs))`.
struct ScheduledBucket {
    ts: i64,
    offset: u64,
    count: u64,
    batch_start: u64,
}

/// Lazy single-timestamp batch schedule. Holds O(D) state (the ordered buckets,
/// bin path, and params) plus a cache of at most ONE materialized (permuted)
/// bucket's node-ids. Owned by the iterator when `group_by_timestamp` is set.
struct BucketSchedule {
    /// Buckets in epoch-shuffled order, each tagged with its global batch range
    /// start. `batch_start` is the running prefix-sum of `ceil(count/bs)`.
    buckets: Vec<ScheduledBucket>,
    /// Total batches across all buckets (`Σ ceil(count_b / batch_size)`).
    total_batches: u64,
    /// Absolute path of `bucket_index.bin`.
    bin_path: std::path::PathBuf,
    /// Shuffle base seed and epoch (within-bucket permutations key off these
    /// plus the bucket `ts`).
    base_seed: u64,
    epoch: u64,
    stage_key: String,
    batch_size: usize,
    num_workers: u64,
    worker_id: u64,
    /// Next owned global batch index to emit. Owned indices are
    /// `{ worker_id, worker_id + num_workers, ... } ∩ [0, total_batches)`.
    next_global: u64,
    /// Cache of the currently-materialized bucket: its index into `buckets` and
    /// its permuted node-ids. `None` until the first read; replaced whenever a
    /// batch lands in a different bucket. Because owned global indices are
    /// monotonic, bucket reads are monotonic → each bucket is read ~once.
    cached: Option<(usize, Vec<u64>)>,
}

impl BucketSchedule {
    /// Build the schedule from the manifest bucket list. `raw_buckets` are the
    /// `(ts, offset, count)` triples from `diskgraph_manifest.json` (ingest
    /// order = ascending ts). The schedule shuffles the BUCKET ORDER by
    /// `epoch_shuffle_seed(base, epoch, stage_key)` and prefix-sums the
    /// per-bucket batch counts. O(D) RAM.
    #[allow(clippy::too_many_arguments)]
    fn build(
        raw_buckets: Vec<RawBucket>,
        bin_path: std::path::PathBuf,
        base_seed: u64,
        epoch: u64,
        stage_key: String,
        batch_size: usize,
        num_workers: u64,
        worker_id: u64,
    ) -> Self {
        // Shuffle the bucket order deterministically by (base, epoch, stage_key)
        // — identical across workers.
        let mut order: Vec<usize> = (0..raw_buckets.len()).collect();
        let mut rng = StdRng::seed_from_u64(epoch_shuffle_seed(base_seed, epoch, &stage_key));
        order.shuffle(&mut rng);

        let mut buckets = Vec::with_capacity(raw_buckets.len());
        let mut acc: u64 = 0;
        for idx in order {
            let (ts, offset, count) = raw_buckets[idx];
            // Empty buckets contribute zero batches; skip so binary search and
            // the prefix-sum stay tight (an empty bucket would share a
            // batch_start with its successor and never be selected anyway).
            if count == 0 {
                continue;
            }
            let nbatches = count.div_ceil(batch_size as u64);
            buckets.push(ScheduledBucket {
                ts,
                offset,
                count,
                batch_start: acc,
            });
            acc += nbatches;
        }
        let total_batches = acc;
        BucketSchedule {
            buckets,
            total_batches,
            bin_path,
            base_seed,
            epoch,
            stage_key,
            batch_size,
            num_workers,
            worker_id,
            next_global: worker_id,
            cached: None,
        }
    }

    /// This worker's batch count = number of owned global indices in
    /// `[0, total_batches)`.
    fn worker_batch_count(&self) -> u64 {
        worker_batch_count(self.total_batches, self.num_workers, self.worker_id)
    }

    /// Resolve a global batch index to `(bucket index, chunk index within the
    /// bucket)` via binary search over the per-bucket batch ranges. `g` must be
    /// `< total_batches`.
    fn locate(&self, g: u64) -> (usize, u64) {
        // partition_point: first bucket whose batch_start > g; the owning bucket
        // is the one before it.
        let after = self
            .buckets
            .partition_point(|b| b.batch_start <= g);
        debug_assert!(after >= 1, "g < total_batches implies a non-empty prefix");
        let b = after - 1;
        let chunk = g - self.buckets[b].batch_start;
        (b, chunk)
    }

    /// Read bucket `b`'s node-id slice from `bucket_index.bin` and apply the
    /// within-bucket shuffle keyed by `(base, epoch, "{stage_key}:{ts}")`.
    /// O(count_b) RAM. Caller caches the result.
    fn materialize_bucket(&self, b: usize) -> PyResult<Vec<u64>> {
        use std::os::unix::fs::FileExt;
        let bucket = &self.buckets[b];
        let f = std::fs::File::open(&self.bin_path).map_err(|e| {
            DiskgraphSystemError::new_err(format!(
                "open bucket_index.bin {}: {e}",
                self.bin_path.display()
            ))
        })?;
        let mut bytes = vec![0u8; (bucket.count as usize) * 8];
        f.read_exact_at(&mut bytes, bucket.offset * 8).map_err(|e| {
            DiskgraphDataError::new_err(format!(
                "read bucket [{}, {}) from {}: {e}",
                bucket.offset,
                bucket.offset + bucket.count,
                self.bin_path.display()
            ))
        })?;
        let mut ids: Vec<u64> = bytes
            .as_chunks::<8>().0.iter()
            .map(|c| u64::from_le_bytes(*c))
            .collect();
        let key = format!("{}:{}", self.stage_key, bucket.ts);
        let mut rng =
            StdRng::seed_from_u64(epoch_shuffle_seed(self.base_seed, self.epoch, &key));
        ids.shuffle(&mut rng);
        Ok(ids)
    }

    /// Pull this worker's next single-timestamp batch of FINAL node-ids, or
    /// `None` when the worker's owned global indices are exhausted.
    fn next_node_ids(&mut self) -> PyResult<Option<Vec<u64>>> {
        if self.next_global >= self.total_batches {
            return Ok(None);
        }
        let g = self.next_global;
        self.next_global += self.num_workers;

        let (b, chunk) = self.locate(g);
        // (Re)materialize the bucket if it isn't the cached one. Owned indices
        // are monotonic ⇒ bucket index is non-decreasing ⇒ ~one read per bucket.
        if self.cached.as_ref().map(|(idx, _)| *idx) != Some(b) {
            let ids = self.materialize_bucket(b)?;
            self.cached = Some((b, ids));
        }
        let ids = &self.cached.as_ref().unwrap().1;
        let bs = self.batch_size as u64;
        let start = (chunk * bs) as usize;
        let end = (((chunk + 1) * bs).min(self.buckets[b].count)) as usize;
        Ok(Some(ids[start..end].to_vec()))
    }
}

/// A bucket as recorded in the manifest: `(ts, offset, count)` (matches the
/// `tt_executor` manifest encoding). Units: `ts` seconds, `offset`/`count`
/// node-ids.
type RawBucket = (i64, u64, u64);

/// Locate the `bucket_index` entry for the stage whose `table_name` matches
/// `stage_table` in `<ext_root>/diskgraph_manifest.json`, returning
/// `(bin_abs_path, raw_buckets)`. The bin `path` is recorded relative to the
/// extension root. FAILS LOUD if the manifest, the stage, or its `bucket_index`
/// is missing — `group_by_timestamp` was requested, so a missing artifact is an
/// error, never a silent fallback.
fn read_bucket_index(
    ext_root: &std::path::Path,
    stage_table: &str,
) -> PyResult<(std::path::PathBuf, Vec<RawBucket>)> {
    let manifest_path = ext_root.join("diskgraph_manifest.json");
    let bytes = std::fs::read(&manifest_path).map_err(|e| {
        DiskgraphConfigError::new_err(format!(
            "group_by_timestamp=True but cannot read {}: {e}",
            manifest_path.display()
        ))
    })?;
    let manifest: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
        DiskgraphConfigError::new_err(format!(
            "invalid diskgraph_manifest.json at {}: {e}",
            manifest_path.display()
        ))
    })?;
    let stages = manifest
        .get("task_dataset")
        .and_then(|td| td.get("stages"))
        .and_then(|s| s.as_object())
        .ok_or_else(|| {
            DiskgraphConfigError::new_err(format!(
                "diskgraph_manifest.json has no task_dataset.stages: {}",
                manifest_path.display()
            ))
        })?;
    // Stages are keyed by stage_key (train/val/test); the iterator only knows
    // the table_name, so match on the stage's `table_name`. table_names are
    // guaranteed distinct across stages (each stage emits its own table), but
    // assert it here so a manifest bug surfaces loudly instead of silently
    // picking an arbitrary first match.
    let mut matches = stages
        .values()
        .filter(|v| v.get("table_name").and_then(|t| t.as_str()) == Some(stage_table));
    let stage_entry = matches.next().ok_or_else(|| {
        DiskgraphConfigError::new_err(format!(
            "group_by_timestamp=True but no stage with table_name {stage_table:?} \
             in {}",
            manifest_path.display()
        ))
    })?;
    if matches.next().is_some() {
        return Err(DiskgraphConfigError::new_err(format!(
            "diskgraph_manifest.json has multiple stages with table_name {stage_table:?} \
             in {} (stage table_names must be distinct)",
            manifest_path.display()
        )));
    }
    let bucket_index = stage_entry.get("bucket_index").ok_or_else(|| {
        DiskgraphConfigError::new_err(format!(
            "group_by_timestamp=True but stage {stage_table:?} has no bucket_index \
             in {} (ingest did not emit timestamp grouping for this stage)",
            manifest_path.display()
        ))
    })?;
    let rel_path = bucket_index
        .get("path")
        .and_then(|p| p.as_str())
        .ok_or_else(|| {
            DiskgraphConfigError::new_err(format!(
                "bucket_index for stage {stage_table:?} has no string `path`"
            ))
        })?;
    let buckets_json = bucket_index
        .get("buckets")
        .and_then(|b| b.as_array())
        .ok_or_else(|| {
            DiskgraphConfigError::new_err(format!(
                "bucket_index for stage {stage_table:?} has no `buckets` array"
            ))
        })?;
    let mut raw = Vec::with_capacity(buckets_json.len());
    for b in buckets_json {
        let ts = b.get("ts").and_then(|v| v.as_i64());
        let offset = b.get("offset").and_then(|v| v.as_u64());
        let count = b.get("count").and_then(|v| v.as_u64());
        match (ts, offset, count) {
            (Some(ts), Some(offset), Some(count)) => raw.push((ts, offset, count)),
            _ => {
                return Err(DiskgraphConfigError::new_err(format!(
                    "malformed bucket entry {b} for stage {stage_table:?}"
                )))
            }
        }
    }
    Ok((ext_root.join(rel_path), raw))
}

/// Number of global batch indices in `[0, total_batches)` that fall in worker
/// `worker_id`'s residue class `i % num_workers == worker_id`.
fn worker_batch_count(total_batches: u64, num_workers: u64, worker_id: u64) -> u64 {
    if worker_id >= total_batches {
        0
    } else {
        (total_batches - worker_id).div_ceil(num_workers)
    }
}

fn py_record_batch_list<'py>(
    py: Python<'py>,
    batches: &[RecordBatch],
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for batch in batches {
        out.append(batch.to_pyarrow(py)?)?;
    }
    Ok(out)
}

fn py_array<'py>(py: Python<'py>, array: &dyn Array) -> PyResult<Bound<'py, PyAny>> {
    array.to_data().to_pyarrow(py)
}

fn py_u64_array<'py>(py: Python<'py>, values: Vec<u64>) -> PyResult<Bound<'py, PyAny>> {
    UInt64Array::from(values).to_data().to_pyarrow(py)
}

fn py_u32_array<'py>(py: Python<'py>, values: Vec<u32>) -> PyResult<Bound<'py, PyAny>> {
    UInt32Array::from(values).to_data().to_pyarrow(py)
}

fn py_bool_array<'py>(py: Python<'py>, values: Vec<bool>) -> PyResult<Bound<'py, PyAny>> {
    arrow_array::BooleanArray::from(values)
        .to_data()
        .to_pyarrow(py)
}

// ---------------------------------------------------------------------------
// RuntimeBatchIterator — merged-reader path (replaces the legacy
// dual-engine `TaskBatchIterator`).
//
// Iterator over runtime batches matching PR-28765's
// `runtime_batch_to_hetero` contract. One engine over the merged
// (parent + seed-table extension) catalog; per-seed parent-row resolution is
// done by Phase-1 ingest, so this hot loop is just:
//
//   1. Pick the next `batch_size` row offsets for this worker shard.
//   2. `engine.sample_runtime_batch(...)` — walks `seed.edges.bin` to
//      recover parent row ids, reads seed rowpack, calls `engine.sample`,
//      and returns a `RuntimeBatch`.
//   3. `runtime_batch_to_pydict(...)` — converts the Rust struct to the
//      PyDict shape the trainer adapter expects, zero-copy where possible
//      (via `to_data().to_pyarrow` and `RecordBatch.to_pyarrow`).
// ---------------------------------------------------------------------------

use diskgraph_engine::output::{GraphChannel, RawFeatureBank, RuntimeBatch, TaskBag};

/// Convert our Rust `RuntimeBatch` to PR-28765's PyDict shape. All Arrow
/// payloads cross the FFI boundary via Arrow C Data Interface (zero-copy);
/// the only allocations are the Python dict scaffolding itself.
fn runtime_batch_to_pydict<'py>(
    py: Python<'py>,
    batch: &RuntimeBatch,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);

    // task ----
    let task = PyDict::new(py);
    let TaskBag {
        input_ids,
        tt_stage_node_ids,
        seed_node_ids,
        seed_timestamps,
        target,
        weight,
        amount_m,
        num_positives,
        rhs_seed_type,
        neg_mask,
        lp_pos_counts,
        lp_pos_rhs_global,
        lp_pos_mask,
        lp_pos_weight,
        lp_pos_local,
        lp_exclude_counts,
        lp_exclude_rhs_global,
    } = &batch.task;
    task.set_item("input_ids", py_u32_array(py, input_ids.clone())?)?;
    task.set_item(
        "tt_stage_node_ids",
        py_u64_array(py, tt_stage_node_ids.clone())?,
    )?;
    task.set_item("seed_node_ids", py_u64_array(py, seed_node_ids.clone())?)?;
    if let Some(ts) = seed_timestamps {
        let arr = Int64Array::from(ts.clone());
        task.set_item("seed_timestamps", py_array(py, &arr)?)?;
    }
    if let Some(tb) = target {
        task.set_item("target", tb.to_pyarrow(py)?)?;
    }
    if let Some(w) = weight {
        let arr = arrow_array::Float32Array::from(w.clone());
        task.set_item("weight", py_array(py, &arr)?)?;
    }
    // Negative-sampling shape (M, B, rhs type). Scalars convert directly.
    task.set_item("amount_m", *amount_m)?;
    task.set_item("num_positives", *num_positives)?;
    match rhs_seed_type {
        Some(t) => task.set_item("rhs_seed_type", t)?,
        None => task.set_item("rhs_seed_type", py.None())?,
    }
    // `neg_mask` is a bool array of length `B*amount_m`, anchor-major and
    // parallel to the `[B:B+B*M]` rhs negative slots in
    // graphs.main.node_ids[rhs_seed_type]. `true` = use the slot in the loss;
    // `false` = the slot holds a dense placeholder id (temporal-resample budget
    // exhausted) that the loss must ignore. Because masked slots keep a dense
    // placeholder, the `[B, M]` layout / `dst_neg_index` reshape is unaffected —
    // the loss applies the mask. Empty when no negative sampling was requested.
    task.set_item("neg_mask", py_bool_array(py, neg_mask.clone())?)?;
    // Ragged multi-positive LP label payload (FK-free resolution, design §S5).
    // Anchor-major. `lp_pos_counts` is `[B]` (per-anchor positive count);
    // `lp_pos_rhs_global` / `lp_pos_mask` are `[ΣP]`. Emitted only when non-empty
    // — the LP label path populates them; every non-LP path leaves the keys
    // absent. `lp_pos_weight` / `lp_pos_local` are emitted only when Some (S4).
    if !lp_pos_counts.is_empty() {
        task.set_item("lp_pos_counts", py_u32_array(py, lp_pos_counts.clone())?)?;
        let rhs = Int64Array::from(lp_pos_rhs_global.clone());
        task.set_item("lp_pos_rhs_global", py_array(py, &rhs)?)?;
        task.set_item("lp_pos_mask", py_bool_array(py, lp_pos_mask.clone())?)?;
    }
    if let Some(w) = lp_pos_weight {
        let arr = arrow_array::Float32Array::from(w.clone());
        task.set_item("lp_pos_weight", py_array(py, &arr)?)?;
    }
    if let Some(local) = lp_pos_local {
        task.set_item("lp_pos_local", py_u32_array(py, local.clone())?)?;
    }
    // STATIC LP past-visit exclusion (eval-only). Emitted only when non-empty,
    // so train / predict batches leave the keys absent (the adapter's contract).
    if !lp_exclude_counts.is_empty() {
        task.set_item(
            "lp_exclude_counts",
            py_u32_array(py, lp_exclude_counts.clone())?,
        )?;
        let exclude_rhs = Int64Array::from(lp_exclude_rhs_global.clone());
        task.set_item("lp_exclude_rhs_global", py_array(py, &exclude_rhs)?)?;
    }
    dict.set_item("task", task)?;

    // graphs ----
    let graphs = PyDict::new(py);
    for (channel_name, channel) in &batch.graphs {
        graphs.set_item(channel_name, graph_channel_to_pydict(py, channel)?)?;
    }
    dict.set_item("graphs", graphs)?;

    // raw_features ----
    let raw_features = PyDict::new(py);
    for (ns, bank) in &batch.raw_features {
        raw_features.set_item(ns, raw_feature_bank_to_pydict(py, ns, bank)?)?;
    }
    dict.set_item("raw_features", raw_features)?;

    Ok(dict)
}

fn graph_channel_to_pydict<'py>(
    py: Python<'py>,
    channel: &GraphChannel,
) -> PyResult<Bound<'py, PyDict>> {
    let ch = PyDict::new(py);

    // node_ids (per type)
    let node_ids = PyDict::new(py);
    for (nt, ids) in &channel.node_ids {
        node_ids.set_item(nt, py_u64_array(py, ids.clone())?)?;
    }
    ch.set_item("node_ids", node_ids)?;

    // batch_dict (per type)
    let batch_dict = PyDict::new(py);
    for (nt, idxs) in &channel.batch_dict {
        batch_dict.set_item(nt, py_u32_array(py, idxs.clone())?)?;
    }
    ch.set_item("batch_dict", batch_dict)?;

    // edge_index_dict — grouped by (src, rel, dst). Each entry: {edge_type:{src,rel,dst}, src:[u32], dst:[u32]}.
    let edge_index_dict = PyList::empty(py);
    for (rel_idx, (srcs, dsts)) in channel.edges.iter().enumerate() {
        if srcs.is_empty() {
            continue;
        }
        let rel_id = rel_idx as u16;
        let (src, rel, dst) = channel
            .rel_id_to_names
            .get(&rel_id)
            .cloned()
            .unwrap_or_else(|| ("?".to_string(), format!("rel_{rel_id}"), "?".to_string()));
        let key = PyDict::new(py);
        key.set_item("src", src)?;
        key.set_item("rel", rel)?;
        key.set_item("dst", dst)?;
        let entry = PyDict::new(py);
        entry.set_item("edge_type", key)?;
        entry.set_item("src", py_u32_array(py, srcs.clone())?)?;
        entry.set_item("dst", py_u32_array(py, dsts.clone())?)?;
        edge_index_dict.append(entry)?;
    }
    ch.set_item("edge_index_dict", edge_index_dict)?;

    // num_sampled_nodes_dict — per-type per-hop counts: {name: [hop0, hop1, ...]}.
    let num_sampled_nodes = PyDict::new(py);
    for (nt, counts) in &channel.num_sampled_nodes {
        num_sampled_nodes.set_item(nt, py_u64_array(py, counts.clone())?)?;
    }
    ch.set_item("num_sampled_nodes_dict", num_sampled_nodes)?;

    // num_sampled_edges_dict — per-edge-type per-hop counts. Mirrors the
    // edge_index_dict entry shape ({edge_type:{src,rel,dst}, counts:[...]}) but
    // emits EVERY edge type the plan can reach, INCLUDING all-zero ones (no
    // `is_empty` skip), so the adapter can materialize every schema edge store
    // for layer-wise / explainer indexing — matching PyG NeighborLoader.
    let num_sampled_edges = PyList::empty(py);
    for (names, counts) in &channel.num_sampled_edges {
        let (src, rel, dst) = names;
        let key = PyDict::new(py);
        key.set_item("src", src)?;
        key.set_item("rel", rel)?;
        key.set_item("dst", dst)?;
        let entry = PyDict::new(py);
        entry.set_item("edge_type", key)?;
        entry.set_item("counts", py_u64_array(py, counts.clone())?)?;
        num_sampled_edges.append(entry)?;
    }
    ch.set_item("num_sampled_edges_dict", num_sampled_edges)?;

    Ok(ch)
}

fn raw_feature_bank_to_pydict<'py>(
    py: Python<'py>,
    namespace: &str,
    bank: &RawFeatureBank,
) -> PyResult<Bound<'py, PyDict>> {
    let block = PyDict::new(py);
    block.set_item("key", namespace)?;
    let num_rows: usize = bank.batches.iter().map(|b| b.num_rows()).sum();
    block.set_item("num_rows", num_rows)?;
    block.set_item("record_batches", py_record_batch_list(py, &bank.batches)?)?;
    Ok(block)
}

/// Worker-sharded iterator over `RuntimeBatch`es of a seed-table stage.
/// Consumed via the merged-reader path (single engine over parent +
/// extension).
#[pyclass]
pub struct RuntimeBatchIterator {
    reader: Arc<diskgraph_index::graph::GraphIndexReader>,
    engine: Arc<DiskGraphEngine>,
    stage_table_catalog_key: String,
    stage_table: Arc<TableReader>,
    plan: SamplingPlan,
    flags: SamplingFlags,
    /// This worker's slice of the epoch, generated on demand rather
    /// than materialised. See `ShardStream`.
    stream: ShardStream,
    /// Worker-local explicit stage-table node ids to emit in caller order.
    /// These are final stage node ids (`task.tt_stage_node_ids`), not dense
    /// offsets.
    selected_stage_node_ids: Option<Vec<u64>>,
    /// Worker-local replacement-sampled stage node ids for binary
    /// WeightMode.SAMPLE. Built once per iterator epoch from binary label
    /// node-id sidecars.
    weighted_stage_node_ids: Option<Vec<u64>>,
    cursor: usize,
    batch_size: usize,
    drop_last: bool,
    /// Optional engine-side global negative RHS sampling. `None` => no
    /// negatives (byte-identical to the pre-negatives path).
    negatives: Option<NegativeSamplingSpec>,
    /// FK-free LP resolution (design §S5). When set, the engine reads the seed
    /// set from the `entity_node_id` rowpack column instead of walking
    /// `seed.edges.bin`. `None` => legacy FK-edge path.
    entity_node_type: Option<String>,
    /// FK-free LP resolution: when set, the engine emits the ragged `lp_pos_*`
    /// labels from the `target_node_ids` rowpack column. `None` => raw target
    /// passthrough (non-LP / pred path).
    target_node_type: Option<String>,
    /// NodeRange seed mode (two-tower MIPS RHS pass). When set, the iterator
    /// seeds the FULL `[0, num_rhs)` catalog of this node type directly (the
    /// `offsets` ARE the rhs global node-ids, no stage-table map) at the fixed
    /// `rhs_seed_time`, bypassing stage / entity-target resolution and negatives.
    /// `None` => stage-table seeding (the default path).
    rhs_seed_node_type: Option<String>,
    /// Fixed anchor timestamp applied to every NodeRange seed. Required when
    /// `rhs_seed_node_type` is set.
    rhs_seed_time: Option<i64>,
    /// TT-NG temporal-LP timestamp grouping (LOAD side). When `Some`, the
    /// iterator yields single-timestamp batches from this lazy schedule instead
    /// of the flat `offsets` path; `offsets` is left empty in that mode. `None`
    /// => the existing flat `make_worker_offsets` path (byte-identical).
    bucket_schedule: Option<BucketSchedule>,
    /// Whether a TIMESTAMP column in the seed rowpack should be used as the
    /// temporal sampler anchor. Static task datasets may still carry a pseudo
    /// TIMESTAMP column for split/schema compatibility; those must not anchor
    /// temporal-most-recent sampling.
    use_seed_timestamps: bool,
}

/// An opened graph index, reusable across epochs and shards.
///
/// WHY THIS EXISTS. Constructing an iterator used to fuse two things with very
/// different costs and lifetimes:
///
///   * WHICH INDEX -- mmap every shard, build every row-pack and sort-key
///     decoder, spawn a rayon pool. ~1.6 s on DoorDash. Depends only on
///     `(parent_root, tt_id, rnd_seed, num_threads)`.
///   * WHICH ROWS -- the epoch and the shard identity. Microseconds.
///
/// Because both happened in one constructor, changing the cheap thing forced
/// redoing the expensive one. A training run therefore paid the full index
/// open once per epoch in every DataLoader worker, and again on every
/// intra-epoch replay -- which is why `persistent_workers` alone never helped:
/// the processes survive, but `__iter__` rebuilds inside them.
///
/// Holding a `RuntimeIndex` and calling `iter()` per epoch pays the open once.
/// Measured on DoorDash: 3218.8 ms per construction becomes ~1.55 s for the
/// first and 0.1 ms for every one after.
///
/// `rnd_seed` and `num_threads` belong here rather than on `iter()` because the
/// engine is built from them -- and `num_threads` in particular is not a
/// performance-only knob: the engine's parallel neighbour sampling produces
/// different (individually reproducible) batches for different thread counts,
/// so two iterators that disagree on it are not sampling the same graph.
#[pyclass]
pub struct RuntimeIndex {
    reader: Arc<diskgraph_index::graph::GraphIndexReader>,
    engine: Arc<DiskGraphEngine>,
    namespace: String,
}

#[pymethods]
impl RuntimeIndex {
    #[new]
    #[pyo3(signature = (parent_root, tt_id, rnd_seed=None, num_threads=None))]
    pub fn new(
        parent_root: String,
        tt_id: String,
        rnd_seed: Option<u64>,
        num_threads: Option<usize>,
    ) -> PyResult<Self> {
        let parent_path = std::path::Path::new(&parent_root);
        let reader = diskgraph_index::graph::GraphIndexReader::open_with_tt(parent_path, &tt_id)
            .map_err(|e| DiskgraphSystemError::new_err(e.to_string()))?;
        let namespace = tt_id.rsplit('/').next().unwrap_or(&tt_id).to_string();
        let reader = Arc::new(reader);
        let engine = Arc::new(
            DiskGraphEngine::from_reader(reader.clone(), rnd_seed, num_threads)
                .map_err(map_engine_err)?,
        );
        Ok(Self { reader, engine, namespace })
    }

    /// Build an iterator over one (stage, epoch, shard) of this index.
    ///
    /// Cheap: no mmap, no decoders, no thread pool. Call it once per epoch, or
    /// once per chunk when work is handed out dynamically.
    #[pyo3(signature = (
        stage_table,
        sampling_plan_json,
        num_workers,
        worker_id,
        shuffle,
        rnd_seed,
        epoch,
        batch_size,
        drop_last=false,
        fused=None,
        swap_edge_direction=true,
        neg_rhs_node_type=None,
        neg_amount_m=None,
        neg_rnd_seed=None,
        replace=false,
        disjoint=true,
        adaptive=false,
        entity_node_type=None,
        target_node_type=None,
        rhs_seed_node_type=None,
        rhs_seed_time=None,
        group_by_timestamp=false,
        weighted_seed_sampling_json=None,
        selected_stage_node_ids=None,
        use_seed_timestamps=true,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn iter(
        &self,
        stage_table: String,
        sampling_plan_json: String,
        num_workers: u64,
        worker_id: u64,
        shuffle: bool,
        rnd_seed: Option<u64>,
        epoch: u64,
        batch_size: usize,
        drop_last: bool,
        fused: Option<bool>,
        swap_edge_direction: bool,
        neg_rhs_node_type: Option<String>,
        neg_amount_m: Option<usize>,
        neg_rnd_seed: Option<u64>,
        replace: bool,
        disjoint: bool,
        adaptive: bool,
        entity_node_type: Option<String>,
        target_node_type: Option<String>,
        rhs_seed_node_type: Option<String>,
        rhs_seed_time: Option<i64>,
        group_by_timestamp: bool,
        weighted_seed_sampling_json: Option<String>,
        selected_stage_node_ids: Option<Vec<u64>>,
        use_seed_timestamps: bool,
    ) -> PyResult<RuntimeBatchIterator> {
        RuntimeBatchIterator::build(
            Arc::clone(&self.reader),
            Arc::clone(&self.engine),
            self.namespace.clone(),
            stage_table,
            sampling_plan_json,
            num_workers,
            worker_id,
            shuffle,
            rnd_seed,
            epoch,
            batch_size,
            drop_last,
            fused,
            swap_edge_direction,
            neg_rhs_node_type,
            neg_amount_m,
            neg_rnd_seed,
            replace,
            disjoint,
            adaptive,
            entity_node_type,
            target_node_type,
            rhs_seed_node_type,
            rhs_seed_time,
            group_by_timestamp,
            weighted_seed_sampling_json,
            selected_stage_node_ids,
            use_seed_timestamps,
        )
    }
}

impl RuntimeBatchIterator {
    #[allow(clippy::too_many_arguments)]
    fn build(
        reader: Arc<diskgraph_index::graph::GraphIndexReader>,
        engine: Arc<DiskGraphEngine>,
        namespace: String,
        stage_table: String,
        sampling_plan_json: String,
        num_workers: u64,
        worker_id: u64,
        shuffle: bool,
        rnd_seed: Option<u64>,
        epoch: u64,
        batch_size: usize,
        drop_last: bool,
        // Sampler dispatch: fused interleaves feature fetching with the edge
        // walk (cache-locality win for small per-batch feature working sets);
        // non-fused walks first + fetches features in a parallel map-reduce
        // (amortizes memory latency for large working sets). Both produce
        // the same RuntimeBatch shape.
        //
        // `None` (default) → DG picks via `engine::choose_fused_mode` over
        // (reader, plan, batch_size). Explicit `True`/`False` overrides for
        // callers that have profiled their workload. Long-term the explicit
        // override goes away and DG fully owns the decision (Phase 5 legacy
        // deletion — kumo's sampler abstraction is gone by then anyway).
        fused: Option<bool>,
        // PyG CSC convention: swap edge src/dst (and rel id via reverse_rel_map)
        // when recording edges. Default true matches PR 269 + what PR 28765's
        // adapter expects.
        swap_edge_direction: bool,
        // Engine-side global negative RHS sampling (LP training). Passed as 3
        // optional scalars (cleaner than a dict given the existing
        // signature-based arg list). `neg_rhs_node_type=None` (default)
        // disables negatives; when set, `neg_amount_m` is required.
        // `neg_rnd_seed=None` (default) ⇒ `thread_rng` (random negatives, the
        // training/prod default); `Some(x)` ⇒ deterministic draw (tests).
        neg_rhs_node_type: Option<String>,
        neg_amount_m: Option<usize>,
        neg_rnd_seed: Option<u64>,
        // Sampler flags exposed for callers (#307): with-replacement neighbor
        // draw, per-seed disjoint neighborhoods, and adaptive sampling.
        replace: bool,
        disjoint: bool,
        adaptive: bool,
        // FK-free LP resolution (design §S5). `entity_node_type` => read the seed
        // set from the `entity_node_id` rowpack column (no `seed.edges.bin`
        // walk). `target_node_type` => emit ragged `lp_pos_*` labels from the
        // `target_node_ids` column (positives are LABELS, never rooted). Both
        // `None` (default) preserves the legacy FK-edge / raw-target path.
        entity_node_type: Option<String>,
        target_node_type: Option<String>,
        // NodeRange seed mode (two-tower MIPS RHS pass). When `rhs_seed_node_type`
        // is set, the iterator seeds the FULL `[0, num_rhs)` catalog of that node
        // type directly at the fixed `rhs_seed_time` (required), bypassing
        // stage-table seeding / entity-target resolution / negatives. `None`
        // (default) preserves the stage-table seeding path byte-for-byte.
        rhs_seed_node_type: Option<String>,
        rhs_seed_time: Option<i64>,
        // TT-NG temporal-LP timestamp grouping (LOAD side). `false` (default) =>
        // the flat `make_worker_offsets` path, byte-identical. `true` => yield
        // single-timestamp batches from the per-stage `bucket_index.bin` the
        // ingest side recorded; FAILS LOUD if that artifact is missing.
        // Mutually exclusive with NodeRange mode (the RHS catalog pass has no
        // per-row timestamp to group on).
        group_by_timestamp: bool,
        weighted_seed_sampling_json: Option<String>,
        // XAI / targeted replay mode: explicit stage-table node ids, in caller
        // order. These are the same ids emitted as task.tt_stage_node_ids.
        selected_stage_node_ids: Option<Vec<u64>>,
        use_seed_timestamps: bool,
    ) -> PyResult<Self> {
        if batch_size == 0 {
            return Err(DiskgraphConfigError::new_err("batch_size must be >= 1"));
        }
        if selected_stage_node_ids.is_some() {
            if shuffle {
                return Err(DiskgraphConfigError::new_err(
                    "selected_stage_node_ids is incompatible with shuffle=True",
                ));
            }
            if drop_last {
                return Err(DiskgraphConfigError::new_err(
                    "selected_stage_node_ids is incompatible with drop_last=True",
                ));
            }
            if group_by_timestamp {
                return Err(DiskgraphConfigError::new_err(
                    "selected_stage_node_ids is incompatible with group_by_timestamp=True",
                ));
            }
            if rhs_seed_node_type.is_some() {
                return Err(DiskgraphConfigError::new_err(
                    "selected_stage_node_ids is incompatible with rhs_seed_node_type (NodeRange mode)",
                ));
            }
        }
        if weighted_seed_sampling_json.is_some() {
            if selected_stage_node_ids.is_some() {
                return Err(DiskgraphConfigError::new_err(
                    "weighted_seed_sampling is incompatible with selected_stage_node_ids",
                ));
            }
            if drop_last {
                return Err(DiskgraphConfigError::new_err(
                    "weighted_seed_sampling is incompatible with drop_last=True",
                ));
            }
            if group_by_timestamp {
                return Err(DiskgraphConfigError::new_err(
                    "weighted_seed_sampling is incompatible with group_by_timestamp=True",
                ));
            }
            if rhs_seed_node_type.is_some() {
                return Err(DiskgraphConfigError::new_err(
                    "weighted_seed_sampling is incompatible with rhs_seed_node_type (NodeRange mode)",
                ));
            }
        }
        if group_by_timestamp && rhs_seed_node_type.is_some() {
            return Err(DiskgraphConfigError::new_err(
                "group_by_timestamp is incompatible with rhs_seed_node_type (NodeRange mode)",
            ));
        }
        if group_by_timestamp && shuffle && rnd_seed.is_none() {
            return Err(DiskgraphConfigError::new_err(
                "rnd_seed is required when group_by_timestamp=True (used for the \
                 bucket-order + within-bucket shuffle)",
            ));
        }
        if num_workers == 0 {
            return Err(DiskgraphConfigError::new_err("num_workers must be >= 1"));
        }
        if worker_id >= num_workers {
            return Err(DiskgraphConfigError::new_err(format!(
                "worker_id {worker_id} out of range for num_workers {num_workers}"
            )));
        }
        if rhs_seed_node_type.is_some() && rhs_seed_time.is_none() {
            return Err(DiskgraphConfigError::new_err(
                "rhs_seed_time is required when rhs_seed_node_type is set (NodeRange mode)",
            ));
        }
        let negatives = match neg_rhs_node_type {
            Some(rhs_node_type) => {
                let amount_m = neg_amount_m.ok_or_else(|| {
                    DiskgraphConfigError::new_err(
                        "neg_amount_m is required when neg_rhs_node_type is set",
                    )
                })?;
                Some(NegativeSamplingSpec {
                    rhs_node_type,
                    amount_m,
                    rnd_seed: neg_rnd_seed,
                })
            }
            None => None,
        };
        // Stage table in the merged catalog is namespaced as `<namespace>/<bare>`;
        // accept either form from the caller.
        let stage_catalog_key = if stage_table.contains('/') {
            stage_table.clone()
        } else {
            format!("{namespace}/{stage_table}")
        };
        let stage_table_arc = reader.table(&stage_catalog_key).ok_or_else(|| {
            DiskgraphConfigError::new_err(format!(
                "unknown stage table in merged catalog: {stage_catalog_key}"
            ))
        })?;
        let selected_stage_node_ids = if let Some(ids) = selected_stage_node_ids {
            let mut local_ids = Vec::new();
            for (idx, node_id) in ids.into_iter().enumerate() {
                if (idx as u64) % num_workers == worker_id {
                    if stage_table_arc.shard_for_node(node_id).is_none() {
                        return Err(DiskgraphDataError::new_err(format!(
                            "selected_stage_node_ids[{idx}]={node_id} is not a valid node id for stage table `{stage_catalog_key}`"
                        )));
                    }
                    local_ids.push(node_id);
                }
            }
            Some(local_ids)
        } else {
            None
        };
        let weighted_stage_node_ids = if let Some(config_json) = weighted_seed_sampling_json {
            Some(make_weighted_stage_node_ids(
                &config_json,
                stage_table_arc.as_ref(),
                &stage_catalog_key,
                num_workers,
                worker_id,
                rnd_seed,
                epoch,
            )?)
        } else {
            None
        };
        let sampling_plan: SamplingPlan = serde_json::from_str(&sampling_plan_json)
            .map_err(|e| DiskgraphConfigError::new_err(format!("Invalid plan JSON: {e}")))?;
        // Grouped (single-timestamp) path: build the lazy bucket schedule from
        // the per-stage `bucket_index.bin` the ingest side recorded. `offsets`
        // is unused in this mode (left empty); FAILS LOUD on a missing artifact.
        let bucket_schedule = if group_by_timestamp {
            let ext_root = reader.tt_extension_root().ok_or_else(|| {
                DiskgraphConfigError::new_err(
                    "group_by_timestamp=True but reader has no TT extension root",
                )
            })?;
            let bare_table = stage_catalog_key
                .rsplit('/')
                .next()
                .unwrap_or(&stage_catalog_key);
            let (bin_path, raw_buckets) = read_bucket_index(ext_root, bare_table)?;
            Some(BucketSchedule::build(
                raw_buckets,
                bin_path,
                rnd_seed.unwrap_or(0),
                epoch,
                stage_catalog_key.clone(),
                batch_size,
                num_workers,
                worker_id,
            ))
        } else {
            None
        };
        // NodeRange mode: seeds are the RHS catalog node-ids `[lo, hi]` (dense by
        // ingest guarantee), NOT stage-table row offsets. Build the offsets over
        // the dense range and add `lo` so each offset IS the global node-id. The
        // stage table is ignored for seeding in this mode.
        let stream = if selected_stage_node_ids.is_some() || weighted_stage_node_ids.is_some() {
            ShardStream::empty()
        } else if group_by_timestamp {
            // Grouped path drives iteration from `bucket_schedule`, not the stream.
            ShardStream::empty()
        } else if let Some(rhs_type) = rhs_seed_node_type.as_deref() {
            let (lo, hi) = engine
                .node_type_id_range(rhs_type)
                .map_err(map_engine_err)?;
            let num_rhs = hi - lo + 1;
            // `base = lo` so each emitted value IS the global node id, which is
            // what the materialised path achieved with a post-pass over the Vec.
            ShardStream::new(
                num_rhs, num_workers, worker_id, shuffle, rnd_seed, epoch, rhs_type, lo,
            )?
        } else {
            ShardStream::new(
                stage_table_arc.node_count(),
                num_workers,
                worker_id,
                shuffle,
                rnd_seed,
                epoch,
                &stage_catalog_key,
                0,
            )?
        };
        // `fused=None` → DG picks via heuristic. Explicit override wins.
        // Telemetry so trainers can see what was chosen and why.
        let chosen_fused = match fused {
            Some(v) => v,
            None => {
                let h = diskgraph_engine::engine::choose_fused_mode(
                    reader.as_ref(),
                    &sampling_plan,
                    batch_size,
                );
                tracing::info!(
                    chosen_fused = h,
                    batch_size = batch_size,
                    stage_table = %stage_catalog_key,
                    "DG-side fused-mode heuristic chose mode for this iterator"
                );
                h
            }
        };
        Ok(Self {
            reader,
            engine,
            stage_table_catalog_key: stage_catalog_key,
            stage_table: stage_table_arc,
            plan: sampling_plan,
            flags: SamplingFlags {
                replace,
                directed: true,
                disjoint,
                return_edge_id: false,
                adaptive,
                fused_feature_fetching: chosen_fused,
                swap_edge_direction,
            },
            stream,
            selected_stage_node_ids,
            weighted_stage_node_ids,
            cursor: 0,
            batch_size,
            drop_last,
            negatives,
            entity_node_type,
            target_node_type,
            rhs_seed_node_type,
            rhs_seed_time,
            bucket_schedule,
            use_seed_timestamps,
        })
    }

}

#[pymethods]
impl RuntimeBatchIterator {
    /// Backwards-compatible constructor: open an index and immediately
    /// iterate it. Equivalent to `RuntimeIndex(...).iter(...)`, and kept so
    /// existing callers need no change. Callers that build many iterators
    /// over one index should hold a `RuntimeIndex` instead -- that is the
    /// whole point of the split, and this shim cannot do it for them.
    #[new]
    #[pyo3(signature = (
        parent_root,
        tt_id,
        stage_table,
        sampling_plan_json,
        num_workers,
        worker_id,
        shuffle,
        rnd_seed,
        epoch,
        batch_size,
        drop_last=false,
        num_threads=None,
        fused=None,
        swap_edge_direction=true,
        neg_rhs_node_type=None,
        neg_amount_m=None,
        neg_rnd_seed=None,
        replace=false,
        disjoint=true,
        adaptive=false,
        entity_node_type=None,
        target_node_type=None,
        rhs_seed_node_type=None,
        rhs_seed_time=None,
        group_by_timestamp=false,
        weighted_seed_sampling_json=None,
        selected_stage_node_ids=None,
        use_seed_timestamps=true,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        parent_root: String,
        tt_id: String,
        stage_table: String,
        sampling_plan_json: String,
        num_workers: u64,
        worker_id: u64,
        shuffle: bool,
        rnd_seed: Option<u64>,
        epoch: u64,
        batch_size: usize,
        drop_last: bool,
        num_threads: Option<usize>,
        fused: Option<bool>,
        swap_edge_direction: bool,
        neg_rhs_node_type: Option<String>,
        neg_amount_m: Option<usize>,
        neg_rnd_seed: Option<u64>,
        replace: bool,
        disjoint: bool,
        adaptive: bool,
        entity_node_type: Option<String>,
        target_node_type: Option<String>,
        rhs_seed_node_type: Option<String>,
        rhs_seed_time: Option<i64>,
        group_by_timestamp: bool,
        weighted_seed_sampling_json: Option<String>,
        selected_stage_node_ids: Option<Vec<u64>>,
        use_seed_timestamps: bool,
    ) -> PyResult<Self> {
        let index = RuntimeIndex::new(parent_root, tt_id, rnd_seed, num_threads)?;
        Self::build(
            Arc::clone(&index.reader),
            Arc::clone(&index.engine),
            index.namespace.clone(),
            stage_table,
            sampling_plan_json,
            num_workers,
            worker_id,
            shuffle,
            rnd_seed,
            epoch,
            batch_size,
            drop_last,
            fused,
            swap_edge_direction,
            neg_rhs_node_type,
            neg_amount_m,
            neg_rnd_seed,
            replace,
            disjoint,
            adaptive,
            entity_node_type,
            target_node_type,
            rhs_seed_node_type,
            rhs_seed_time,
            group_by_timestamp,
            weighted_seed_sampling_json,
            selected_stage_node_ids,
            use_seed_timestamps,
        )
    }

    pub fn num_samples(&self) -> usize {
        if let Some(ids) = self.selected_stage_node_ids.as_ref() {
            return ids.len();
        }
        if let Some(ids) = self.weighted_stage_node_ids.as_ref() {
            return ids.len();
        }
        // Grouped path: report the worker's batch count (offsets is empty).
        match &self.bucket_schedule {
            Some(sched) => sched.worker_batch_count() as usize,
            None => self.stream.len(),
        }
    }

    /// This worker's batch count for one epoch. For the grouped path this is the
    /// number of owned global batch indices; Python's `__len__` calls it.
    pub fn worker_batch_count(&self) -> usize {
        if let Some(ids) = self.selected_stage_node_ids.as_ref() {
            return ids.len().div_ceil(self.batch_size);
        }
        if let Some(ids) = self.weighted_stage_node_ids.as_ref() {
            return ids.len().div_ceil(self.batch_size);
        }
        match &self.bucket_schedule {
            Some(sched) => sched.worker_batch_count() as usize,
            None => {
                let remaining = self.stream.len();
                if self.drop_last {
                    remaining / self.batch_size
                } else {
                    remaining.div_ceil(self.batch_size)
                }
            }
        }
    }

    pub fn stage_size(&self) -> u64 {
        self.stage_table.node_count()
    }

    pub fn next_batch(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        // Grouped (single-timestamp) path: pull the next bucket-chunk of FINAL
        // node-ids and feed them DIRECTLY to the engine (they are already final
        // node-ids — no stage `node_id_from_offset` map, no NodeRange seed).
        //
        // INVARIANT: ingest writes `bucket_index.bin` entries as `shard.min_id +
        // local` (see ts_grouping::merge_stage) — the SAME formula
        // `node_id_from_offset` uses to map a dense offset to a final node-id.
        // So a grouped id IS a valid final node-id of the stage table, and the
        // flat path's offset→node-id remap would be the identity here. This holds
        // for any shard layout (the +min_id term carries the shard base). The
        // `debug_assert!` below catches a future ingest/load skew (e.g. a stage
        // sub-index whose shard ranges don't match what ingest wrote) in dev
        // builds; it is compiled out of the release .so.
        //
        // Disjoint-field borrows: `sched` borrows ONLY `self.bucket_schedule`
        // mutably, leaving every other field free to borrow immutably. This lets
        // the `detach` (GIL-release) closure capture only the specific Rust
        // fields the sample needs (no `&self`, no `Py<…>` field) so it stays
        // `Send`. `next_node_ids` (which triggers `materialize_bucket`'s blocking
        // file IO) touches NO Python, so we fold it INTO the closure: both the
        // per-bucket IO and the multi-threaded rayon sample run with the GIL
        // released. Only `runtime_batch_to_pydict` (Python construction) stays
        // under the GIL.
        if let Some(sched) = self.bucket_schedule.as_mut() {
            let engine = &self.engine;
            let reader = &self.reader;
            let stage_table = &self.stage_table;
            let stage_table_catalog_key = &self.stage_table_catalog_key;
            let plan = &self.plan;
            let flags = &self.flags;
            let negatives = self.negatives.as_ref();
            let entity_node_type = self.entity_node_type.as_deref();
            let target_node_type = self.target_node_type.as_deref();
            let runtime_batch = py.detach(|| -> PyResult<Option<RuntimeBatch>> {
                let node_ids = match sched.next_node_ids()? {
                    Some(ids) => ids,
                    None => return Ok(None),
                };
                debug_assert!(
                    node_ids
                        .iter()
                        .all(|id| stage_table.shard_for_node(*id).is_some()),
                    "grouped seed id not a valid stage-table node-id — ingest \
                     bucket_index / load stage sub-index shard layouts disagree",
                );
                let rb = engine
                    .sample_runtime_batch(
                        reader.as_ref(),
                        stage_table_catalog_key,
                        &node_ids,
                        diskgraph_core::dataset::DatasetId::nil(),
                        plan,
                        flags,
                        negatives,
                        entity_node_type,
                        target_node_type,
                        self.use_seed_timestamps,
                        None,
                    )
                    .map_err(map_engine_err)?;
                Ok(Some(rb))
            })?;
            let runtime_batch = match runtime_batch {
                Some(rb) => rb,
                None => return Ok(None),
            };
            let dict = runtime_batch_to_pydict(py, &runtime_batch)?;
            return Ok(Some(dict.into()));
        }

        if let Some(selected_ids) = self.selected_stage_node_ids.as_ref() {
            let remaining = selected_ids.len().saturating_sub(self.cursor);
            if remaining == 0 {
                return Ok(None);
            }
            let take = self.batch_size.min(remaining);
            let stage_node_ids = selected_ids[self.cursor..self.cursor + take].to_vec();
            self.cursor += take;

            let engine = &self.engine;
            let reader = &self.reader;
            let stage_table_catalog_key = &self.stage_table_catalog_key;
            let plan = &self.plan;
            let flags = &self.flags;
            let negatives = self.negatives.as_ref();
            let entity_node_type = self.entity_node_type.as_deref();
            let target_node_type = self.target_node_type.as_deref();
            let runtime_batch = py
                .detach(|| {
                    engine.sample_runtime_batch(
                        reader.as_ref(),
                        stage_table_catalog_key,
                        &stage_node_ids,
                        diskgraph_core::dataset::DatasetId::nil(),
                        plan,
                        flags,
                        negatives,
                        entity_node_type,
                        target_node_type,
                        self.use_seed_timestamps,
                        None,
                    )
                })
                .map_err(map_engine_err)?;

            let dict = runtime_batch_to_pydict(py, &runtime_batch)?;
            return Ok(Some(dict.into()));
        }

        if let Some(weighted_ids) = self.weighted_stage_node_ids.as_ref() {
            let remaining = weighted_ids.len().saturating_sub(self.cursor);
            if remaining == 0 {
                return Ok(None);
            }
            let take = self.batch_size.min(remaining);
            let stage_node_ids = weighted_ids[self.cursor..self.cursor + take].to_vec();
            self.cursor += take;

            let engine = &self.engine;
            let reader = &self.reader;
            let stage_table_catalog_key = &self.stage_table_catalog_key;
            let plan = &self.plan;
            let flags = &self.flags;
            let negatives = self.negatives.as_ref();
            let entity_node_type = self.entity_node_type.as_deref();
            let target_node_type = self.target_node_type.as_deref();
            let runtime_batch = py
                .detach(|| {
                    engine.sample_runtime_batch(
                        reader.as_ref(),
                        stage_table_catalog_key,
                        &stage_node_ids,
                        diskgraph_core::dataset::DatasetId::nil(),
                        plan,
                        flags,
                        negatives,
                        entity_node_type,
                        target_node_type,
                        self.use_seed_timestamps,
                        None,
                    )
                })
                .map_err(map_engine_err)?;

            let dict = runtime_batch_to_pydict(py, &runtime_batch)?;
            return Ok(Some(dict.into()));
        }

        let remaining = self.stream.len().saturating_sub(self.cursor);
        if remaining == 0 {
            return Ok(None);
        }
        let take = self.batch_size.min(remaining);
        if take < self.batch_size && self.drop_last {
            return Ok(None);
        }
        let batch_offsets = self.stream.take(self.cursor, take);
        self.cursor += take;

        // NodeRange mode: the offsets ARE the rhs global node-ids (built over the
        // dense `[lo, hi]` catalog range), so there is no stage-table
        // `node_id_from_offset` map — pass them straight to the NodeRange engine
        // path with the fixed seed_time.
        if let Some(rhs_type) = self.rhs_seed_node_type.as_deref() {
            let seed_time = self
                .rhs_seed_time
                .expect("rhs_seed_time present when rhs_seed_node_type is set (validated in new)");
            // Release the GIL around the multi-threaded rayon sample (the engine
            // is GIL-free pure Rust); re-acquire only to build the PyDict.
            // Capture only the Rust fields needed so the closure stays `Send`.
            let engine = &self.engine;
            let reader = &self.reader;
            let plan = &self.plan;
            let flags = &self.flags;
            let runtime_batch = py
                .detach(|| {
                    engine.sample_runtime_batch(
                        reader.as_ref(),
                        // stage_table is unused in NodeRange mode; pass the rhs type.
                        rhs_type,
                        &batch_offsets,
                        diskgraph_core::dataset::DatasetId::nil(),
                        plan,
                        flags,
                        None,
                        None,
                        None,
                        true,
                        Some(diskgraph_engine::engine::NodeRangeSeed {
                            rhs_node_type: rhs_type,
                            seed_time,
                        }),
                    )
                })
                .map_err(map_engine_err)?;
            let dict = runtime_batch_to_pydict(py, &runtime_batch)?;
            return Ok(Some(dict.into()));
        }

        // Map offsets → stage table node ids. With a single-shard stage
        // sub-index (Phase-1 invariant) these are equal, but `node_id_from_offset`
        // does the right thing regardless.
        let mut stage_node_ids = Vec::with_capacity(batch_offsets.len());
        for offset in &batch_offsets {
            let node_id = self
                .stage_table
                .node_id_from_offset(*offset)
                .ok_or_else(|| {
                    DiskgraphDataError::new_err(format!(
                        "offset {offset} out of range for stage table {}",
                        self.stage_table_catalog_key
                    ))
                })?;
            stage_node_ids.push(node_id);
        }

        // Sidecars (weight.bin, etc.) are auto-detected inside the engine
        // via `reader.tt_extension_root()` — no path threading at this layer.
        // Release the GIL around the multi-threaded rayon sample (the engine is
        // GIL-free pure Rust); re-acquire only to build the PyDict. Capture only
        // the Rust fields needed so the closure stays `Send` (no `&self`).
        let engine = &self.engine;
        let reader = &self.reader;
        let stage_table_catalog_key = &self.stage_table_catalog_key;
        let plan = &self.plan;
        let flags = &self.flags;
        let negatives = self.negatives.as_ref();
        let entity_node_type = self.entity_node_type.as_deref();
        let target_node_type = self.target_node_type.as_deref();
        let runtime_batch = py
            .detach(|| {
                engine.sample_runtime_batch(
                    reader.as_ref(),
                    stage_table_catalog_key,
                    &stage_node_ids,
                    diskgraph_core::dataset::DatasetId::nil(),
                    plan,
                    flags,
                    negatives,
                    entity_node_type,
                    target_node_type,
                    self.use_seed_timestamps,
                    None, // node_range off (stage-table path)
                )
            })
            .map_err(map_engine_err)?;

        let dict = runtime_batch_to_pydict(py, &runtime_batch)?;
        Ok(Some(dict.into()))
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.next_batch(py)
    }

    /// Number of batches this worker yields in one epoch. For the grouped path
    /// this is the count of owned global batch indices; for the flat path it is
    /// `ceil(num_samples / batch_size)` (or floor when `drop_last`).
    fn __len__(&self) -> usize {
        self.worker_batch_count()
    }
}

/// Free function: list finalized tt_ids under `<parent_root>/train_tables/`.
/// Mirrors `diskgraph_index::graph::discover_tts`; surfaces the admin API
/// so Python can ask "what tt_ids are available?" without opening the
/// reader first.
#[pyfunction]
fn discover_tts(parent_root: String) -> PyResult<Vec<String>> {
    let path = std::path::Path::new(&parent_root);
    diskgraph_index::graph::discover_tts(path)
        .map_err(|e| DiskgraphSystemError::new_err(e.to_string()))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RuntimeIndex>()?;
    m.add_class::<RuntimeBatchIterator>()?;
    m.add_function(wrap_pyfunction!(discover_tts, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::make_worker_offsets;
    use super::{read_bucket_index, worker_batch_count, BucketSchedule, F64Sidecar, U64Sidecar};
    use pyo3::Python;
    use std::collections::HashSet;
    use std::io::Write;

    // -----------------------------------------------------------------------
    // TT-NG temporal-LP timestamp grouping (LOAD side) — schedule tests.
    // -----------------------------------------------------------------------

    /// Write a synthetic `bucket_index.bin` from a list of `(ts, [node_ids])`
    /// buckets laid out back-to-back (the ingest contract), returning the path,
    /// the raw `(ts, offset, count)` triples, and the full node-id set.
    fn write_bin(
        dir: &std::path::Path,
        buckets: &[(i64, Vec<u64>)],
    ) -> (std::path::PathBuf, Vec<(i64, u64, u64)>, HashSet<u64>) {
        let path = dir.join("bucket_index.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        let mut raw = Vec::new();
        let mut all = HashSet::new();
        let mut offset: u64 = 0;
        for (ts, ids) in buckets {
            for id in ids {
                f.write_all(&id.to_le_bytes()).unwrap();
                assert!(all.insert(*id), "duplicate node-id {id} in fixture");
            }
            raw.push((*ts, offset, ids.len() as u64));
            offset += ids.len() as u64;
        }
        (path, raw, all)
    }

    /// Drain a worker's schedule into the list of its per-batch node-id vecs.
    fn drain(sched: &mut BucketSchedule) -> Vec<Vec<u64>> {
        let mut out = Vec::new();
        while let Some(ids) = sched.next_node_ids().unwrap() {
            out.push(ids);
        }
        out
    }

    #[test]
    fn u64_sidecar_mmap_reads_by_index_without_heap_decoding() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("ids.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        for value in [7u64, 11, 42] {
            f.write_all(&value.to_le_bytes()).unwrap();
        }
        drop(f);

        let sidecar = U64Sidecar::open(path.to_str().unwrap()).unwrap();
        assert_eq!(sidecar.len(), 3);
        assert!(!sidecar.is_empty());
        assert_eq!(sidecar.get(0).unwrap(), 7);
        assert_eq!(sidecar.get(2).unwrap(), 42);
        assert!(sidecar.get(3).is_err());
    }

    #[test]
    fn f64_sidecar_binary_searches_cumulative_weights() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("cdf.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        for value in [0.5f64, 2.5, 6.5] {
            f.write_all(&value.to_le_bytes()).unwrap();
        }
        drop(f);

        let sidecar = F64Sidecar::open(path.to_str().unwrap()).unwrap();
        assert_eq!(sidecar.len(), 3);
        assert_eq!(sidecar.partition_point_le(0.0).unwrap(), 0);
        assert_eq!(sidecar.partition_point_le(0.5).unwrap(), 1);
        assert_eq!(sidecar.partition_point_le(2.0).unwrap(), 1);
        assert_eq!(sidecar.partition_point_le(6.4).unwrap(), 2);
    }

    /// THE partition-invariant test. For a sweep of worker counts, batch sizes,
    /// and bucket-size distributions (buckets smaller/larger than batch_size, a
    /// partial tail, single-node buckets), collect every batch from every
    /// worker for one epoch and assert: the union equals the full node-id set
    /// with no duplicates (a true partition), and every batch is single-ts.
    #[test]
    fn bucket_schedule_is_a_single_ts_partition() {
        let tmp = tempfile::tempdir().unwrap();
        // ts -> node-ids for that timestamp. Distributions chosen to exercise
        // buckets < / == / > batch_size and partial tails. Node-ids are
        // arbitrary (not 0..n) to confirm we feed FINAL ids, not offsets.
        let bucket_dists: Vec<Vec<(i64, Vec<u64>)>> = vec![
            // Mixed sizes incl. a single-node bucket and a big one.
            vec![
                (100, vec![7]),
                (50, vec![10, 11, 12, 13, 14]),
                (200, vec![20, 21]),
                (75, (1000..1017).collect()),
                (10, vec![900, 901, 902]),
            ],
            // All buckets smaller than any tested batch_size.
            vec![(1, vec![1, 2]), (2, vec![3]), (3, vec![4, 5]), (4, vec![6])],
            // One large bucket → forces multi-chunk within a single ts.
            vec![(9, (0..23).collect())],
            // Two equal buckets exactly a multiple of batch_size 2.
            vec![(5, vec![100, 101, 102, 103]), (6, vec![200, 201])],
            // INTERIOR empty bucket: a `(ts, [])` entry between non-empty ones
            // exercises the `count == 0` skip in `build` (the empty bucket must
            // contribute zero batches and never be selected, yet the partition of
            // the remaining ids must stay exact and single-ts).
            vec![
                (11, vec![300, 301, 302]),
                (12, vec![]),
                (13, vec![400, 401]),
            ],
        ];

        for (di, dist) in bucket_dists.iter().enumerate() {
            let sub = tmp.path().join(format!("dist{di}"));
            std::fs::create_dir_all(&sub).unwrap();
            let (bin_path, raw, all_ids) = write_bin(&sub, dist);
            // Map node-id -> its ts, to assert single-ts batches.
            let ts_of: std::collections::HashMap<u64, i64> = dist
                .iter()
                .flat_map(|(ts, ids)| ids.iter().map(move |id| (*id, *ts)))
                .collect();

            for &num_workers in &[1u64, 2, 3, 4] {
                for &batch_size in &[1usize, 2, 3, 5, 8] {
                    let mut union: Vec<u64> = Vec::new();
                    let mut total_batches_seen = 0u64;
                    for worker_id in 0..num_workers {
                        let mut sched = BucketSchedule::build(
                            raw.clone(),
                            bin_path.clone(),
                            42,
                            0,
                            "train".to_string(),
                            batch_size,
                            num_workers,
                            worker_id,
                        );
                        // __len__ must match the batches actually yielded.
                        let reported = sched.worker_batch_count();
                        let batches = drain(&mut sched);
                        assert_eq!(
                            batches.len() as u64,
                            reported,
                            "dist{di} nw={num_workers} bs={batch_size} w{worker_id}: \
                             reported len != yielded"
                        );
                        total_batches_seen += batches.len() as u64;
                        for batch in &batches {
                            assert!(!batch.is_empty(), "no empty batch should be emitted");
                            assert!(batch.len() <= batch_size, "batch exceeds batch_size");
                            // I1: single timestamp per batch.
                            let first_ts = ts_of[&batch[0]];
                            for id in batch {
                                assert_eq!(
                                    ts_of[id], first_ts,
                                    "dist{di} bs={batch_size}: batch mixes timestamps"
                                );
                            }
                            union.extend_from_slice(batch);
                        }
                    }
                    // I2 no-dup + I3 complete: exact partition of all node-ids.
                    let uniq: HashSet<u64> = union.iter().copied().collect();
                    assert_eq!(
                        uniq.len(),
                        union.len(),
                        "dist{di} nw={num_workers} bs={batch_size}: node-id emitted twice"
                    );
                    assert_eq!(
                        uniq, all_ids,
                        "dist{di} nw={num_workers} bs={batch_size}: union != full set"
                    );
                    // Σ over workers of len == B (total global batches).
                    let one = BucketSchedule::build(
                        raw.clone(),
                        bin_path.clone(),
                        42,
                        0,
                        "train".to_string(),
                        batch_size,
                        1,
                        0,
                    );
                    assert_eq!(
                        total_batches_seen, one.total_batches,
                        "dist{di} nw={num_workers} bs={batch_size}: Σ len != B"
                    );
                }
            }
        }
    }

    /// An all-empty / zero-bucket schedule (`total_batches == 0`) drained
    /// end-to-end: every worker must yield nothing, report a batch count of 0,
    /// and `total_batches` must be 0. Covers both an empty bucket list and a
    /// list of only-empty buckets.
    #[test]
    fn bucket_schedule_zero_batches_drains_empty() {
        let tmp = tempfile::tempdir().unwrap();
        // Two zero-bucket cases: no buckets at all, and only-empty buckets.
        let cases: Vec<Vec<(i64, Vec<u64>)>> =
            vec![vec![], vec![(1, vec![]), (2, vec![]), (3, vec![])]];
        for (ci, dist) in cases.iter().enumerate() {
            let sub = tmp.path().join(format!("zero{ci}"));
            std::fs::create_dir_all(&sub).unwrap();
            let (bin_path, raw, all_ids) = write_bin(&sub, dist);
            assert!(all_ids.is_empty(), "zero-bucket fixture has no ids");
            for &num_workers in &[1u64, 2, 4] {
                for worker_id in 0..num_workers {
                    let mut sched = BucketSchedule::build(
                        raw.clone(),
                        bin_path.clone(),
                        42,
                        0,
                        "train".to_string(),
                        4,
                        num_workers,
                        worker_id,
                    );
                    assert_eq!(sched.total_batches, 0, "case{ci}: total_batches != 0");
                    assert_eq!(
                        sched.worker_batch_count(),
                        0,
                        "case{ci} w{worker_id}: reported a batch"
                    );
                    let batches = drain(&mut sched);
                    assert!(
                        batches.is_empty(),
                        "case{ci} w{worker_id}: yielded a batch from a zero-bucket schedule"
                    );
                }
            }
        }
    }

    /// `num_workers > total_batches`: some workers own ZERO global batch indices.
    /// Drive the full drain (not just the arithmetic) and assert every worker
    /// yields exactly its reported count, the high-id workers yield nothing, and
    /// the union across all workers is still an exact, no-dup, single-ts
    /// partition with Σ len == total_batches.
    #[test]
    fn bucket_schedule_more_workers_than_batches() {
        let tmp = tempfile::tempdir().unwrap();
        // 3 ids in 3 single-id buckets at batch_size 1 => exactly 3 batches.
        let dist: Vec<(i64, Vec<u64>)> = vec![(10, vec![500]), (20, vec![600]), (30, vec![700])];
        let (bin_path, raw, all_ids) = write_bin(tmp.path(), &dist);
        let ts_of: std::collections::HashMap<u64, i64> = dist
            .iter()
            .flat_map(|(ts, ids)| ids.iter().map(move |id| (*id, *ts)))
            .collect();

        let total = BucketSchedule::build(
            raw.clone(),
            bin_path.clone(),
            42,
            0,
            "train".to_string(),
            1,
            1,
            0,
        )
        .total_batches;
        assert_eq!(total, 3, "expected 3 total batches");

        let num_workers = 5u64; // > total_batches (3): workers 3 and 4 own nothing.
        let mut union: Vec<u64> = Vec::new();
        let mut total_seen = 0u64;
        let mut workers_with_zero = 0u64;
        for worker_id in 0..num_workers {
            let mut sched = BucketSchedule::build(
                raw.clone(),
                bin_path.clone(),
                42,
                0,
                "train".to_string(),
                1,
                num_workers,
                worker_id,
            );
            let reported = sched.worker_batch_count();
            let batches = drain(&mut sched);
            assert_eq!(
                batches.len() as u64,
                reported,
                "w{worker_id}: reported len != yielded"
            );
            if reported == 0 {
                workers_with_zero += 1;
            }
            total_seen += batches.len() as u64;
            for batch in &batches {
                assert!(!batch.is_empty(), "no empty batch should be emitted");
                let first_ts = ts_of[&batch[0]];
                for id in batch {
                    assert_eq!(ts_of[id], first_ts, "batch mixes timestamps");
                }
                union.extend_from_slice(batch);
            }
        }
        assert_eq!(
            workers_with_zero, 2,
            "expected exactly 2 workers (3,4) to own zero batches"
        );
        let uniq: HashSet<u64> = union.iter().copied().collect();
        assert_eq!(uniq.len(), union.len(), "node-id emitted twice");
        assert_eq!(uniq, all_ids, "union != full set");
        assert_eq!(total_seen, total, "Σ len != total_batches");
    }

    /// Determinism: same `(base, epoch)` => identical schedule independent of a
    /// second construction; different `epoch` => different global order but
    /// still a valid partition; the order is independent of `worker_id`.
    #[test]
    fn bucket_schedule_is_deterministic_and_epoch_varying() {
        let tmp = tempfile::tempdir().unwrap();
        let dist: Vec<(i64, Vec<u64>)> = (0..12)
            .map(|t| (t as i64, vec![(t * 10) as u64, (t * 10 + 1) as u64]))
            .collect();
        let (bin_path, raw, all_ids) = write_bin(tmp.path(), &dist);

        let build = |epoch: u64, worker_id: u64, num_workers: u64| {
            let mut s = BucketSchedule::build(
                raw.clone(),
                bin_path.clone(),
                7,
                epoch,
                "train".to_string(),
                3,
                num_workers,
                worker_id,
            );
            drain(&mut s)
        };

        // Same (base, epoch) replays byte-identically.
        assert_eq!(build(0, 0, 1), build(0, 0, 1));
        // Different epoch => different order (probabilistic but safe at 12 ts).
        assert_ne!(build(0, 0, 1), build(1, 0, 1));

        // The GLOBAL order (num_workers=1) is independent of worker_id: a
        // worker's batches are exactly the global batches at its residue class.
        let global = build(3, 0, 1);
        let nw = 3u64;
        let mut reassembled: Vec<Vec<u64>> = vec![Vec::new(); global.len()];
        for w in 0..nw {
            let wb = build(3, w, nw);
            // worker w owns global indices w, w+nw, ...
            for (k, batch) in wb.into_iter().enumerate() {
                let g = w as usize + k * nw as usize;
                reassembled[g] = batch;
            }
        }
        assert_eq!(
            reassembled, global,
            "per-worker residue classes != global order"
        );

        // Different epoch still a full partition.
        let mut u = HashSet::new();
        for w in 0..nw {
            for b in build(5, w, nw) {
                for id in b {
                    assert!(u.insert(id), "dup across workers at epoch 5");
                }
            }
        }
        assert_eq!(u, all_ids);
    }

    /// `worker_batch_count` matches the residue-class arithmetic and Σ == B.
    #[test]
    fn worker_batch_count_sums_to_total() {
        for total in [0u64, 1, 5, 7, 16] {
            for nw in [1u64, 2, 3, 4] {
                let sum: u64 = (0..nw).map(|w| worker_batch_count(total, nw, w)).sum();
                assert_eq!(sum, total, "Σ worker_batch_count != total ({total},{nw})");
            }
        }
    }

    /// Fail-loud: `read_bucket_index` errors clearly when the manifest is
    /// missing, the stage is absent, or it has no `bucket_index`. Formatting a
    /// `PyErr` requires the interpreter, so the message assertion runs under the
    /// GIL.
    #[test]
    fn read_bucket_index_fails_loud() {
        Python::initialize();
        let tmp = tempfile::tempdir().unwrap();
        // No manifest at all.
        assert!(read_bucket_index(tmp.path(), "stage").is_err());

        // Manifest present but stage has no bucket_index.
        let manifest = serde_json::json!({
            "task_dataset": {
                "stages": {
                    "train": { "table_name": "stage", "row_count": 3 }
                }
            }
        });
        std::fs::write(
            tmp.path().join("diskgraph_manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let err = read_bucket_index(tmp.path(), "stage").unwrap_err();
        Python::attach(|py| {
            assert!(
                err.value(py).to_string().contains("no bucket_index"),
                "expected a 'no bucket_index' error, got: {}",
                err.value(py)
            );
        });
        // Unknown stage table.
        assert!(read_bucket_index(tmp.path(), "nope").is_err());

        // Well-formed bucket_index round-trips through read_bucket_index.
        let manifest = serde_json::json!({
            "task_dataset": {
                "stages": {
                    "train": {
                        "table_name": "stage",
                        "bucket_index": {
                            "path": "stage/bucket_index.bin",
                            "buckets": [
                                { "ts": 10, "offset": 0, "count": 2 },
                                { "ts": 20, "offset": 2, "count": 1 }
                            ]
                        }
                    }
                }
            }
        });
        std::fs::write(
            tmp.path().join("diskgraph_manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let (bin_path, raw) = read_bucket_index(tmp.path(), "stage").unwrap();
        assert_eq!(bin_path, tmp.path().join("stage/bucket_index.bin"));
        assert_eq!(raw, vec![(10, 0, 2), (20, 2, 1)]);
    }

    // ---- ShardStream / Feistel -------------------------------------------
    //
    // The catastrophic failure for a pointwise permutation is silent: if it is
    // not a bijection, some seeds are trained on twice and others never, with
    // nothing to notice. So bijectivity is tested EXHAUSTIVELY over every small
    // domain rather than sampled.

    use super::{FeistelPerm, ShardStream, MATERIALISE_BELOW};

    #[test]
    fn feistel_is_a_bijection_for_every_small_domain() {
        for n in 1u64..=512 {
            let perm = FeistelPerm::new(n, 0xC0FFEE ^ n);
            let mut seen = HashSet::new();
            for i in 0..n {
                let v = perm.at(i);
                assert!(v < n, "n={n} i={i} produced {v}, outside [0,n)");
                assert!(seen.insert(v), "n={n} produced {v} twice: not a bijection");
            }
            assert_eq!(seen.len() as u64, n);
        }
    }

    #[test]
    fn feistel_is_a_bijection_for_awkward_domains() {
        // Just above and just below powers of two, where cycle walking works
        // hardest (the padded domain is nearly twice the real one).
        for n in [1u64, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 63, 64, 65, 1023, 1024,
                  1025, 4095, 4096, 4097, 65535, 65536, 65537] {
            let perm = FeistelPerm::new(n, 42);
            let mut seen = HashSet::new();
            for i in 0..n {
                let v = perm.at(i);
                assert!(v < n, "n={n}: {v} out of range");
                assert!(seen.insert(v), "n={n}: {v} repeated");
            }
        }
    }

    #[test]
    fn feistel_actually_shuffles() {
        // A bijection that is the identity would pass every test above.
        let n = 10_000u64;
        let perm = FeistelPerm::new(n, 7);
        let fixed = (0..n).filter(|i| perm.at(*i) == *i).count();
        // Fixed points of a random permutation are Poisson(1); 50 would mean
        // the permutation is barely moving anything.
        assert!(fixed < 50, "{fixed} fixed points: this is not shuffling");
    }

    #[test]
    fn shards_cover_every_row_exactly_once() {
        // The property the loader actually depends on.
        for (n, k) in [(100u64, 7u64), (1000, 16), (41, 41), (41, 64), (1, 8),
                       (9973, 13)] {
            let mut seen: Vec<u64> = Vec::new();
            for w in 0..k {
                let st = ShardStream::new(n, k, w, true, Some(5), 3, "train", 0)
                    .unwrap();
                for j in 0..st.len() {
                    seen.push(st.at(j));
                }
            }
            seen.sort_unstable();
            let expect: Vec<u64> = (0..n).collect();
            assert_eq!(seen, expect, "n={n} k={k}: shards are not a partition");
        }
    }

    #[test]
    fn len_matches_what_the_stream_yields() {
        // `len()` is a closed form and feeds Lightning's val_check_batch
        // scheduling, so an off-by-one is a scheduling bug, not an error.
        for n in 0u64..200 {
            for k in 1u64..=17 {
                for w in 0..k {
                    let st =
                        ShardStream::new(n, k, w, false, None, 0, "s", 0).unwrap();
                    let counted =
                        (0..n).filter(|p| p % k == w).count();
                    assert_eq!(st.len(), counted, "n={n} k={k} w={w}");
                }
            }
        }
    }

    #[test]
    fn unshuffled_stream_matches_the_materialised_path_exactly() {
        // With shuffle=false the new path must be byte-identical to the old
        // one; only the shuffled ordering is allowed to differ.
        for (n, k) in [(10u64, 3u64), (100, 7), (64, 8), (5, 9)] {
            for w in 0..k {
                let old = make_worker_offsets(n, k, w, false, None, 0, "train").unwrap();
                let st = ShardStream::new(n, k, w, false, None, 0, "train", 0).unwrap();
                let new: Vec<u64> = (0..st.len()).map(|j| st.at(j)).collect();
                assert_eq!(old, new, "n={n} k={k} w={w}");
            }
        }
    }

    #[test]
    fn small_domains_match_the_materialised_shuffle_exactly() {
        // Below the threshold the ordering must be byte-identical to what it
        // replaces, so tasks with modest train tables see no change at all.
        for (n, k) in [(10u64, 3u64), (1000, 8), (99_999, 16)] {
            for w in 0..k {
                let old = make_worker_offsets(n, k, w, true, Some(7), 2, "train")
                    .unwrap();
                let st =
                    ShardStream::new(n, k, w, true, Some(7), 2, "train", 0).unwrap();
                let new: Vec<u64> = (0..st.len()).map(|j| st.at(j)).collect();
                assert_eq!(old, new, "n={n} k={k} w={w} diverged below threshold");
            }
        }
    }

    #[test]
    fn large_domains_still_partition_exactly() {
        // Above the threshold the generator takes over; coverage must survive
        // the switch.
        let n = MATERIALISE_BELOW + 12_345;
        let k = 8u64;
        let mut seen: Vec<u64> = Vec::new();
        for w in 0..k {
            let st = ShardStream::new(n, k, w, true, Some(3), 1, "train", 0).unwrap();
            for j in 0..st.len() {
                seen.push(st.at(j));
            }
        }
        seen.sort_unstable();
        assert_eq!(seen, (0..n).collect::<Vec<u64>>());
    }

    /// Positions sampled per scale. 1M gives ~100k per decile, so one
    /// standard deviation is ~316 and the 5% tolerance below is ~15 sigma --
    /// enough to catch any bias worth caring about, while still running in
    /// well under a second.
    const MIXING_SAMPLES: u64 = 1_000_000;

    /// Sizes the mixing tests sweep. Chosen so BOTH block layouts are
    /// covered: an even `ceil(log2 n)` gives balanced halves, an odd one
    /// gives the unbalanced split. An all-even list would leave the
    /// unbalanced path with correctness coverage (exhaustive bijectivity)
    /// but no evidence it actually mixes -- and unbalanced Feistel is a
    /// known source of subtle non-mixing.
    ///
    ///   10_000_000      ceil(log2) = 24  even
    ///   67_108_865      ceil(log2) = 27  ODD   (2^26 + 1)
    ///   1_000_000_000   ceil(log2) = 30  even
    ///   5_000_000_000   ceil(log2) = 33  ODD
    ///   10_000_000_000  ceil(log2) = 34  even
    const MIXING_SIZES: [u64; 5] = [
        10_000_000,
        67_108_865,
        1_000_000_000,
        5_000_000_000,
        10_000_000_000,
    ];

    #[test]
    fn a_shard_is_spread_evenly_across_the_table() {
        // Does a worker's shard come from all over the table, or does it
        // clump? Train tables are typically written in timestamp order, so a
        // shard drawn from one region would give temporally correlated
        // batches -- precisely what shuffling exists to prevent.
        //
        // Sampled ACROSS the whole shard rather than from its prefix: a
        // permutation that behaved well early and degraded later would
        // otherwise pass. Run at 10B as well as 10M, which costs the same
        // because the domain is never materialised.
        for n in MIXING_SIZES {
            let st = ShardStream::new(n, 16, 3, true, Some(99), 0, "train", 0)
                .unwrap();
            let len = st.len() as u64;
            let samples = MIXING_SAMPLES.min(len);
            let mut deciles = [0u64; 10];
            for i in 0..samples {
                let j = (i * len / samples) as usize;
                deciles[((st.at(j) / (n / 10)).min(9)) as usize] += 1;
            }
            let expected = samples as f64 / 10.0;
            for (d, count) in deciles.iter().enumerate() {
                let off = (*count as f64 - expected).abs() / expected;
                assert!(
                    off < 0.05,
                    "n={n}: decile {d} holds {count} of {samples}, expected \
                     ~{expected:.0} ({:.2}% off): shard is not spread evenly",
                    off * 100.0
                );
            }
        }
    }

    #[test]
    fn consecutive_rows_of_a_shard_are_far_apart() {
        // Catches what a bijection test cannot: a linear map (a*p + b) % n is
        // a perfect permutation that emits a fixed stride, so it reorders
        // without mixing. For unrelated pairs from [0, n) the mean absolute
        // gap is n/3.
        //
        // Measured over consecutive positions (which is the property in
        // question) but from blocks spread across the shard, so late regions
        // are covered too.
        for n in MIXING_SIZES {
            let st = ShardStream::new(n, 8, 1, true, Some(4), 0, "train", 0)
                .unwrap();
            let len = st.len() as u64;
            let blocks = 100u64;
            let per_block = (MIXING_SAMPLES / blocks).min(len / blocks.max(1));
            let (mut total_gap, mut pairs) = (0u128, 0u128);
            for b in 0..blocks {
                let start = b * len / blocks;
                for j in start + 1..(start + per_block).min(len) {
                    total_gap +=
                        u128::from(st.at(j as usize - 1).abs_diff(st.at(j as usize)));
                    pairs += 1;
                }
            }
            let mean_gap = (total_gap / pairs.max(1)) as f64;
            let expected = n as f64 / 3.0;
            let off = (mean_gap - expected).abs() / expected;
            assert!(
                off < 0.05,
                "n={n}: mean gap over {pairs} pairs is {mean_gap:.0}, expected \
                 ~{expected:.0} ({:.2}% off): the ordering carries structure",
                off * 100.0
            );
        }
    }

    #[test]
    fn the_padded_domain_is_never_more_than_twice_the_real_one() {
        // Cycle walking costs `domain / n` evaluations per row on average, so
        // this ratio IS the cost. A balanced split forces an even block width,
        // which for odd ideal widths rounds the domain up two bits instead of
        // one -- ratio 4 rather than 2, for roughly half of all table sizes.
        // The unbalanced split keeps it under 2 everywhere.
        let mut worst = (0u64, 0.0f64);
        for n in (2u64..2000).chain([
            (1 << 20) + 1,
            (1 << 26) + 1,   // odd ideal width: the band that used to cost 4x
            100_000_000,     // likewise
            41_756_131,      // even ideal width: was already fine
            1_000_000_000,
            10_000_000_000,
        ]) {
            let p = FeistelPerm::new(n, 1);
            let domain = 1u128 << (p.left_bits + p.right_bits);
            let ratio = domain as f64 / n as f64;
            // <= rather than <, for one degenerate case: n = 2 wants a
            // 1-bit block, but a zero-width half is not invertible, so it is
            // floored at 2 bits and lands exactly on 2.0. Academic -- anything
            // under MATERIALISE_BELOW never reaches this path.
            assert!(
                ratio <= 2.0,
                "n={n}: padded domain {domain} is {ratio:.2}x the real one; \
                 cycle walking would cost that many evaluations per row"
            );
            if ratio > worst.1 {
                worst = (n, ratio);
            }
        }
        assert!(worst.1 <= 2.0, "worst ratio {:.2} at n={}", worst.1, worst.0);
    }

    #[test]
    fn epoch_changes_the_generated_order_too() {
        // The existing epoch test runs below MATERIALISE_BELOW, so it only
        // covers Fisher-Yates. If the generated path keyed on the base seed
        // alone, every epoch would replay the identical order -- silent,
        // plausible, and bad for convergence. Check it where the generator
        // is actually in use, and on both block layouts.
        for n in [10_000_000u64, 67_108_865] {
            let a = ShardStream::new(n, 4, 1, true, Some(11), 0, "train", 0).unwrap();
            let b = ShardStream::new(n, 4, 1, true, Some(11), 1, "train", 0).unwrap();
            let take = 10_000usize.min(a.len());
            let va: Vec<u64> = (0..take).map(|j| a.at(j)).collect();
            let vb: Vec<u64> = (0..take).map(|j| b.at(j)).collect();
            assert_ne!(va, vb, "n={n}: epoch did not change the generated order");
            let same = va.iter().zip(&vb).filter(|(x, y)| x == y).count();
            assert!(
                same * 100 < take,
                "n={n}: {same}/{take} positions unchanged across epochs -- \
                 the epoch is barely affecting the key"
            );
        }
    }

    #[test]
    fn stream_is_deterministic_across_instances() {
        let a = ShardStream::new(5000, 4, 1, true, Some(11), 2, "train", 0).unwrap();
        let b = ShardStream::new(5000, 4, 1, true, Some(11), 2, "train", 0).unwrap();
        let va: Vec<u64> = (0..a.len()).map(|j| a.at(j)).collect();
        let vb: Vec<u64> = (0..b.len()).map(|j| b.at(j)).collect();
        assert_eq!(va, vb);
    }

    #[test]
    fn epoch_changes_the_order_but_not_the_coverage() {
        let e0 = ShardStream::new(1000, 1, 0, true, Some(1), 0, "train", 0).unwrap();
        let e1 = ShardStream::new(1000, 1, 0, true, Some(1), 1, "train", 0).unwrap();
        let v0: Vec<u64> = (0..e0.len()).map(|j| e0.at(j)).collect();
        let v1: Vec<u64> = (0..e1.len()).map(|j| e1.at(j)).collect();
        assert_ne!(v0, v1, "epoch did not change the shuffle");
        let (mut s0, mut s1) = (v0.clone(), v1.clone());
        s0.sort_unstable();
        s1.sort_unstable();
        assert_eq!(s0, s1, "epochs must cover the same rows");
    }

    #[test]
    fn base_offsets_every_value_for_node_range_mode() {
        let st = ShardStream::new(50, 4, 2, true, Some(3), 0, "rhs", 1_000).unwrap();
        for j in 0..st.len() {
            let v = st.at(j);
            assert!((1_000..1_050).contains(&v), "{v} outside the shifted range");
        }
    }

    #[test]
    fn make_worker_offsets_shards_without_shuffle() {
        assert_eq!(
            make_worker_offsets(10, 3, 0, false, None, 0, "train").unwrap(),
            vec![0, 3, 6, 9]
        );
        assert_eq!(
            make_worker_offsets(10, 3, 1, false, None, 0, "train").unwrap(),
            vec![1, 4, 7]
        );
        assert_eq!(
            make_worker_offsets(10, 3, 2, false, None, 0, "train").unwrap(),
            vec![2, 5, 8]
        );
    }

    #[test]
    fn make_worker_offsets_handles_empty_and_more_workers_than_rows() {
        assert_eq!(
            make_worker_offsets(0, 4, 0, false, None, 0, "train").unwrap(),
            Vec::<u64>::new()
        );
        assert_eq!(
            make_worker_offsets(2, 4, 0, false, None, 0, "train").unwrap(),
            vec![0]
        );
        assert_eq!(
            make_worker_offsets(2, 4, 1, false, None, 0, "train").unwrap(),
            vec![1]
        );
        assert_eq!(
            make_worker_offsets(2, 4, 2, false, None, 0, "train").unwrap(),
            Vec::<u64>::new()
        );
        assert_eq!(
            make_worker_offsets(2, 4, 3, false, None, 0, "train").unwrap(),
            Vec::<u64>::new()
        );
    }

    #[test]
    fn make_worker_offsets_shuffle_is_deterministic_and_covers_all_offsets() {
        let a0 = make_worker_offsets(20, 3, 0, true, Some(42), 0, "train").unwrap();
        let a1 = make_worker_offsets(20, 3, 1, true, Some(42), 0, "train").unwrap();
        let a2 = make_worker_offsets(20, 3, 2, true, Some(42), 0, "train").unwrap();
        assert_eq!(
            a0,
            make_worker_offsets(20, 3, 0, true, Some(42), 0, "train").unwrap()
        );
        assert_ne!(
            a0,
            make_worker_offsets(20, 3, 0, false, None, 0, "train").unwrap()
        );

        let mut all = [a0, a1, a2].concat();
        all.sort_unstable();
        assert_eq!(all, (0..20).collect::<Vec<u64>>());
    }

    #[test]
    fn make_worker_offsets_shuffle_varies_by_epoch_and_stage() {
        let epoch0 = make_worker_offsets(50, 2, 0, true, Some(42), 0, "train").unwrap();
        let epoch1 = make_worker_offsets(50, 2, 0, true, Some(42), 1, "train").unwrap();
        let val0 = make_worker_offsets(50, 2, 0, true, Some(42), 0, "val").unwrap();

        assert_ne!(epoch0, epoch1);
        assert_ne!(epoch0, val0);
        assert_eq!(
            epoch1,
            make_worker_offsets(50, 2, 0, true, Some(42), 1, "train").unwrap()
        );

        let mut all = [
            make_worker_offsets(50, 2, 0, true, Some(42), 1, "train").unwrap(),
            make_worker_offsets(50, 2, 1, true, Some(42), 1, "train").unwrap(),
        ]
        .concat();
        all.sort_unstable();
        assert_eq!(all, (0..50).collect::<Vec<u64>>());
    }

    #[test]
    fn make_worker_offsets_rejects_invalid_worker_config() {
        assert!(make_worker_offsets(10, 0, 0, false, None, 0, "train").is_err());
        assert!(make_worker_offsets(10, 3, 3, false, None, 0, "train").is_err());
        assert!(make_worker_offsets(10, 3, 0, true, None, 0, "train").is_err());
    }

    // Ported from PR 268 (tt-ng) — these locked invariants on the shuffle
    // sharder added in PR 266. We preserve `make_worker_offsets` verbatim
    // under the new module name, so the tests come along with it.

    #[test]
    fn make_worker_offsets_shuffle_varies_by_rnd_seed() {
        let seed_a = make_worker_offsets(50, 2, 0, true, Some(42), 0, "train").unwrap();
        let seed_b = make_worker_offsets(50, 2, 0, true, Some(43), 0, "train").unwrap();
        let seed_a_replay = make_worker_offsets(50, 2, 0, true, Some(42), 0, "train").unwrap();

        assert_ne!(
            seed_a, seed_b,
            "different rnd_seed at same (epoch, stage, worker) should differ"
        );
        assert_eq!(
            seed_a, seed_a_replay,
            "same (rnd_seed, epoch, stage, worker) must replay byte-identical"
        );
    }

    #[test]
    fn make_worker_offsets_disjoint_across_workers_at_fixed_epoch() {
        let num_workers: u64 = 4;
        let input_count: u64 = 50;
        let mut shards: Vec<Vec<u64>> = Vec::with_capacity(num_workers as usize);
        for w in 0..num_workers {
            shards.push(
                make_worker_offsets(input_count, num_workers, w, true, Some(42), 3, "train")
                    .unwrap(),
            );
        }

        // Pairwise disjoint (not just sorted union — every pair has zero overlap).
        for i in 0..shards.len() {
            for j in (i + 1)..shards.len() {
                let a: std::collections::HashSet<_> = shards[i].iter().copied().collect();
                for offset in &shards[j] {
                    assert!(
                        !a.contains(offset),
                        "worker {} and worker {} overlap on offset {}",
                        i,
                        j,
                        offset
                    );
                }
            }
        }

        // Union covers exactly [0, input_count).
        let mut union: Vec<u64> = shards.into_iter().flatten().collect();
        union.sort_unstable();
        assert_eq!(union, (0..input_count).collect::<Vec<u64>>());
    }

    #[test]
    fn make_worker_offsets_multi_epoch_sweep_all_distinct_and_replayable() {
        let epochs = 5u64;
        let mut orderings: Vec<Vec<u64>> = Vec::with_capacity(epochs as usize);
        for e in 0..epochs {
            orderings.push(make_worker_offsets(64, 2, 0, true, Some(42), e, "train").unwrap());
        }

        // All epochs produce pairwise-distinct orderings on the same worker.
        // (Probabilistic for tiny input sizes; 64 rows is safe.)
        for i in 0..orderings.len() {
            for j in (i + 1)..orderings.len() {
                assert_ne!(
                    orderings[i], orderings[j],
                    "epoch {} and epoch {} produced identical ordering on worker 0",
                    i, j
                );
            }
        }

        // Each epoch is byte-identical on replay.
        for (e, ordering) in orderings.iter().enumerate() {
            let replay = make_worker_offsets(64, 2, 0, true, Some(42), e as u64, "train").unwrap();
            assert_eq!(*ordering, replay, "epoch {} replay diverged", e);
        }
    }
}
