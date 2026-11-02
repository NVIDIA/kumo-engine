// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python-independent index ownership and worker-sharded batch iteration.

use diskgraph_core::sampling::{NegativeSamplingSpec, SamplingFlags, SamplingPlan};
use diskgraph_engine::{engine::DiskGraphEngine, output::RuntimeBatch};
use diskgraph_index::graph::TableReader;
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use serde::Deserialize;
use std::{collections::VecDeque, sync::Arc};

mod errors;
pub use errors::{Result, RuntimeError};

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

/// Keep Fisher–Yates for small domains; avoid offset arrays for large ones.
/// This also preserves the ordinary shuffler's existing small-table order.
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
        Self {
            keys,
            n,
            left_bits,
            right_bits,
        }
    }

    #[inline]
    fn mask(bits: u32) -> u64 {
        if bits >= 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        }
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

impl Order {
    fn shuffled(count: u64, seed: u64) -> Self {
        Self::shuffled_below(count, seed, MATERIALISE_BELOW)
    }

    fn shuffled_below(count: u64, seed: u64, materialise_below: u64) -> Self {
        if count < materialise_below {
            let mut offsets: Vec<u64> = (0..count).collect();
            offsets.shuffle(&mut StdRng::seed_from_u64(seed));
            Self::Materialised(Arc::new(offsets))
        } else {
            Self::Generated(FeistelPerm::new(count, seed))
        }
    }

    #[inline]
    fn at(&self, position: u64) -> u64 {
        match self {
            Self::Identity => position,
            Self::Materialised(offsets) => offsets[position as usize],
            Self::Generated(permutation) => permutation.at(position),
        }
    }

    fn stored_rows(&self) -> u64 {
        match self {
            Self::Materialised(offsets) => offsets.len() as u64,
            _ => 0,
        }
    }
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
        Self {
            order: Order::Identity,
            stride: 1,
            start: 0,
            base: 0,
            len: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        input_count: u64,
        num_workers: u64,
        worker_id: u64,
        shuffle: bool,
        rnd_seed: Option<u64>,
        epoch: u64,
        stage_key: &str,
        base: u64,
    ) -> Result<Self> {
        if num_workers == 0 {
            return Err(RuntimeError::config("num_workers must be >= 1"));
        }
        if worker_id >= num_workers {
            return Err(RuntimeError::config(format!(
                "worker_id {worker_id} out of range for num_workers {num_workers}"
            )));
        }
        if shuffle && rnd_seed.is_none() {
            return Err(RuntimeError::config(
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
        } else {
            Order::shuffled(input_count, seed)
        };
        Ok(Self {
            order,
            stride: num_workers,
            start: worker_id,
            base,
            len,
        })
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
        self.order.at(pos) + self.base
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
) -> Result<Vec<u64>> {
    if num_workers == 0 {
        return Err(RuntimeError::config("num_workers must be >= 1"));
    }
    if worker_id >= num_workers {
        return Err(RuntimeError::config(format!(
            "worker_id {worker_id} out of range for num_workers {num_workers}"
        )));
    }
    if shuffle && rnd_seed.is_none() {
        return Err(RuntimeError::config(
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

/// A TT's mmapped `fact_pseudo_time.bin` plus the seed pseudo time.
pub struct PseudoTimeSeed {
    node_type: String,
    mmap: memmap2::Mmap,
    seed_time: u8,
}

impl PseudoTimeSeed {
    /// Check the extension manifest for pseudo times; reject malformed manifests.
    pub fn is_present(reader: &diskgraph_index::graph::GraphIndexReader) -> Result<bool> {
        let Some(root) = reader.tt_extension_root() else {
            return Ok(false);
        };
        let path = root.join("manifest.json");
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(error) => {
                return Err(RuntimeError::system(format!("{}: {error}", path.display())));
            }
        };
        let manifest: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|error| RuntimeError::data(format!("{}: {error}", path.display())))?;
        let present = manifest
            .get("fact_pseudo_time")
            .is_some_and(|value| !value.is_null());
        if !present
            && manifest["version"].as_u64()
                == Some(diskgraph_train_table::manifest::FACT_PSEUDO_TIME_MANIFEST_VERSION as u64)
        {
            return Err(RuntimeError::data(format!(
                "{}: manifest version 4 requires fact_pseudo_time",
                path.display()
            )));
        }
        Ok(present)
    }

    /// Resolve a stage cutoff, respecting explicit overrides; reject invalid settings.
    pub fn for_stage(
        reader: &diskgraph_index::graph::GraphIndexReader,
        stage_table: &str,
        seed_time: Option<i64>,
        disable_pseudo_time: bool,
    ) -> Result<Option<Self>> {
        if disable_pseudo_time {
            if seed_time.is_some() {
                return Err(RuntimeError::config(
                    "disable_pseudo_time cannot be combined with pseudo_seed_time",
                ));
            }
            return Ok(None);
        }
        if let Some(time) = seed_time {
            return Self::open(reader, time).map(Some);
        }
        if !Self::is_present(reader)? {
            return Ok(None);
        }
        let root = reader.tt_extension_root().unwrap();
        let path = root.join("diskgraph_manifest.json");
        let bytes = std::fs::read(&path)
            .map_err(|e| RuntimeError::data(format!("{}: {e}", path.display())))?;
        let manifest: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| RuntimeError::data(format!("{}: {e}", path.display())))?;
        let stages = manifest["task_dataset"]["stages"]
            .as_object()
            .ok_or_else(|| RuntimeError::data("fact pseudo times require task_dataset.stages"))?;
        let bare_table = stage_table.rsplit('/').next().unwrap_or(stage_table);
        let stage = stages.iter().find_map(|(stage, entry)| {
            (entry["table_name"].as_str() == Some(bare_table)).then_some(stage.as_str())
        });
        match stage {
            Some("train" | "val") => Self::open(reader, 1).map(Some),
            Some("test") => Self::open(reader, 2).map(Some),
            Some("predict") => {
                tracing::warn!(
                    stage_table,
                    "Prediction samples all facts without a pseudo-time cutoff"
                );
                Ok(None)
            }
            _ => Err(RuntimeError::config(format!(
                "Cannot infer pseudo-time cutoff for {stage_table:?}; pass pseudo_seed_time explicitly"
            ))),
        }
    }

    /// Map the pseudo-time sidecar; reject invalid cutoffs, metadata, or payloads.
    pub fn open(reader: &diskgraph_index::graph::GraphIndexReader, seed_time: i64) -> Result<Self> {
        use diskgraph_train_table::manifest::{read_manifest, FACT_PSEUDO_TIME_FORMAT};
        use diskgraph_train_table::pseudo_time::fact_pseudo_time_payload;
        if !(0..=3).contains(&seed_time) {
            return Err(RuntimeError::config(
                "pseudo_seed_time must be between 0 and 3",
            ));
        }
        let seed_time = seed_time as u8;
        let root = reader.tt_extension_root().ok_or_else(|| {
            RuntimeError::config("pseudo_seed_time requires a train-table extension")
        })?;
        let manifest = read_manifest(&root.join("manifest.json")).map_err(RuntimeError::from)?;
        let sidecar = manifest.fact_pseudo_time.ok_or_else(|| {
            RuntimeError::config(format!(
                "pseudo_seed_time set but {} has no fact_pseudo_time sidecar",
                root.display()
            ))
        })?;
        if sidecar.format != FACT_PSEUDO_TIME_FORMAT {
            return Err(RuntimeError::data(format!(
                "unsupported fact_pseudo_time format {:?}",
                sidecar.format
            )));
        }
        let path = root.join(&sidecar.path);
        let file = std::fs::File::open(&path)
            .map_err(|e| RuntimeError::system(format!("failed to open {}: {e}", path.display())))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| RuntimeError::system(format!("failed to mmap {}: {e}", path.display())))?;
        fact_pseudo_time_payload(&mmap)
            .map_err(|e| RuntimeError::data(format!("{}: {e}", path.display())))?;
        Ok(Self {
            node_type: sidecar.node_type,
            mmap,
            seed_time,
        })
    }

    /// Borrow the mapped fact times and cutoff for engine sampling.
    pub fn filter(&self) -> diskgraph_engine::engine::PseudoTimeFilter<'_> {
        diskgraph_engine::engine::PseudoTimeFilter {
            node_type: &self.node_type,
            times: &self.mmap[diskgraph_train_table::pseudo_time::FACT_PSEUDO_TIME_HEADER_LEN..],
            seed_time: self.seed_time,
        }
    }
}

struct U64Sidecar {
    path: String,
    mmap: Option<memmap2::Mmap>,
    len: u64,
}

impl U64Sidecar {
    fn open(path: &str) -> Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| RuntimeError::system(format!("failed to open {path}: {e}")))?;
        let byte_len = file
            .metadata()
            .map_err(|e| RuntimeError::system(format!("failed to stat {path}: {e}")))?
            .len();
        if byte_len % 8 != 0 {
            return Err(RuntimeError::data(format!(
                "u64 sidecar {path} has {byte_len} bytes, not a multiple of 8"
            )));
        }
        let mmap = if byte_len == 0 {
            None
        } else {
            Some(
                unsafe { memmap2::Mmap::map(&file) }
                    .map_err(|e| RuntimeError::system(format!("failed to mmap {path}: {e}")))?,
            )
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

    fn get(&self, index: u64) -> Result<u64> {
        if index >= self.len {
            return Err(RuntimeError::data(format!(
                "u64 sidecar {} index {index} out of range for len {}",
                self.path, self.len
            )));
        }
        let mmap = self
            .mmap
            .as_ref()
            .ok_or_else(|| RuntimeError::data(format!("u64 sidecar {} is empty", self.path)))?;
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
    fn open(path: &str) -> Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| RuntimeError::system(format!("failed to open {path}: {e}")))?;
        let byte_len = file
            .metadata()
            .map_err(|e| RuntimeError::system(format!("failed to stat {path}: {e}")))?
            .len();
        if byte_len % 8 != 0 {
            return Err(RuntimeError::data(format!(
                "f64 sidecar {path} has {byte_len} bytes, not a multiple of 8"
            )));
        }
        let mmap = if byte_len == 0 {
            None
        } else {
            Some(
                unsafe { memmap2::Mmap::map(&file) }
                    .map_err(|e| RuntimeError::system(format!("failed to mmap {path}: {e}")))?,
            )
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

    fn get(&self, index: u64) -> Result<f64> {
        if index >= self.len {
            return Err(RuntimeError::data(format!(
                "f64 sidecar {} index {index} out of range for len {}",
                self.path, self.len
            )));
        }
        let mmap = self
            .mmap
            .as_ref()
            .ok_or_else(|| RuntimeError::data(format!("f64 sidecar {} is empty", self.path)))?;
        let offset = (index as usize) * 8;
        let mut raw = [0u8; 8];
        raw.copy_from_slice(&mmap[offset..offset + 8]);
        Ok(f64::from_le_bytes(raw))
    }

    fn partition_point_le(&self, draw: f64) -> Result<u64> {
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
    fn open(config: WeightedSeedBucketConfig) -> Result<Self> {
        if !config.mass.is_finite() || config.mass < 0.0 {
            return Err(RuntimeError::config(format!(
                "weighted_seed_sampling bucket {} has invalid mass {}",
                config.name, config.mass
            )));
        }
        let node_ids = U64Sidecar::open(&config.node_ids_path)?;
        if node_ids.len() != config.count {
            return Err(RuntimeError::data(format!(
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
                    return Err(RuntimeError::data(format!(
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
                RuntimeError::config(format!(
                    "weighted_seed_sampling bucket {} missing total_weight",
                    config.name
                ))
            })?;
            if !total.is_finite() || total <= 0.0 {
                return Err(RuntimeError::config(format!(
                    "weighted_seed_sampling bucket {} has invalid total_weight {}",
                    config.name, total
                )));
            }
            if config.count > 0 {
                let observed = cdf_ref.get(config.count - 1)?;
                let tolerance = total.abs().max(1.0) * 1e-9;
                if (observed - total).abs() > tolerance {
                    return Err(RuntimeError::data(format!(
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

    fn sample_node_id(&self, rng: &mut StdRng) -> Result<u64> {
        if self.node_ids.is_empty() {
            return Err(RuntimeError::data(format!(
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
) -> Result<Vec<u64>> {
    let config: WeightedSeedSamplingConfig = serde_json::from_str(config_json)
        .map_err(|e| RuntimeError::config(format!("Invalid weighted_seed_sampling JSON: {e}")))?;
    if config.num_samples == 0 {
        return Ok(Vec::new());
    }
    let buckets = config
        .buckets
        .into_iter()
        .map(WeightedSeedBucket::open)
        .collect::<Result<Vec<_>>>()?;
    let total_mass: f64 = buckets.iter().map(|bucket| bucket.mass).sum();
    if total_mass <= 0.0 {
        return Err(RuntimeError::config(
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
            RuntimeError::config("weighted_seed_sampling requires at least one bucket")
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
            return Err(RuntimeError::data(format!(
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
// Permute batch slots globally and row positions within each timestamp.
struct ScheduledBucket {
    offset: u64,
    count: u64,
    batch_start: u64,
    seed: u64,
}

// A bucket order is stored only if it spans under this many batches and fits
// the cache on its own; otherwise it is generated.
const STORED_BUCKET_BATCHES: u64 = 64;
const MAX_STORED_ROWS: u64 = 32 * 1024 * 1024 / size_of::<u64>() as u64;
const MAX_STORED_ENTRIES: usize = 16;

struct BucketOrders {
    entries: VecDeque<(usize, Order)>,
    stored_rows: u64,
    store_below: u64,
}

impl BucketOrders {
    fn new(batch_size: usize) -> Self {
        let store_below = STORED_BUCKET_BATCHES
            .saturating_mul(batch_size as u64)
            .min(MAX_STORED_ROWS + 1);
        Self::with_store_below(store_below)
    }

    fn with_store_below(store_below: u64) -> Self {
        Self {
            entries: VecDeque::new(),
            stored_rows: 0,
            store_below,
        }
    }

    fn get(&mut self, bucket: usize, count: u64, seed: u64) -> &Order {
        if let Some(position) = self.entries.iter().position(|(id, _)| *id == bucket) {
            let entry = self.entries.remove(position).unwrap();
            self.entries.push_back(entry);
        } else {
            let rows = if count < self.store_below { count } else { 0 };
            while self.entries.len() >= MAX_STORED_ENTRIES
                || self.stored_rows + rows > MAX_STORED_ROWS
            {
                let Some((_, evicted)) = self.entries.pop_front() else {
                    break;
                };
                self.stored_rows -= evicted.stored_rows();
            }
            let order = Order::shuffled_below(count, seed, self.store_below);
            self.stored_rows += order.stored_rows();
            self.entries.push_back((bucket, order));
        }
        &self.entries.back().unwrap().1
    }
}

struct BucketSchedule {
    buckets: Vec<ScheduledBucket>,
    total_batches: u64,
    ids: U64Sidecar,
    batches: Order,
    row_orders: BucketOrders,
    batch_size: usize,
    num_workers: u64,
    worker_id: u64,
    next_global: u64,
}

impl BucketSchedule {
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
    ) -> Result<Self> {
        if batch_size == 0 || num_workers == 0 || worker_id >= num_workers {
            return Err(RuntimeError::config(
                "invalid bucket batch size or worker assignment",
            ));
        }
        let ids = U64Sidecar::open(&bin_path.to_string_lossy())?;
        let mut buckets = Vec::with_capacity(raw_buckets.len());
        let mut total_batches = 0u64;
        for (ts, offset, count) in raw_buckets {
            if offset > ids.len() || count > ids.len() - offset {
                return Err(RuntimeError::data(format!(
                    "bucket at timestamp {ts} exceeds {}",
                    bin_path.display()
                )));
            }
            if count == 0 {
                continue;
            }
            buckets.push(ScheduledBucket {
                offset,
                count,
                batch_start: total_batches,
                seed: epoch_shuffle_seed(base_seed, epoch, &format!("{stage_key}:rows:{ts}")),
            });
            total_batches = total_batches
                .checked_add(count.div_ceil(batch_size as u64))
                .ok_or_else(|| RuntimeError::data("bucket batch count overflow"))?;
        }
        Ok(Self {
            buckets,
            total_batches,
            ids,
            row_orders: BucketOrders::new(batch_size),
            batches: Order::shuffled(
                total_batches,
                epoch_shuffle_seed(base_seed, epoch, &format!("{stage_key}:batches")),
            ),
            batch_size,
            num_workers,
            worker_id,
            next_global: worker_id,
        })
    }

    fn worker_batch_count(&self) -> u64 {
        worker_batch_count(self.total_batches, self.num_workers, self.worker_id)
    }

    fn locate(&self, g: u64) -> (usize, u64) {
        let b = self.buckets.partition_point(|b| b.batch_start <= g) - 1;
        (b, g - self.buckets[b].batch_start)
    }

    fn next_node_ids(&mut self) -> Result<Option<Vec<u64>>> {
        if self.next_global >= self.total_batches {
            return Ok(None);
        }
        let g = self.batches.at(self.next_global);
        self.next_global = self.next_global.saturating_add(self.num_workers);
        let (b, chunk) = self.locate(g);
        let bucket = &self.buckets[b];
        let start = chunk * self.batch_size as u64;
        let end = start
            .saturating_add(self.batch_size as u64)
            .min(bucket.count);
        let rows = self.row_orders.get(b, bucket.count, bucket.seed);
        (start..end)
            .map(|position| self.ids.get(bucket.offset + rows.at(position)))
            .collect::<Result<Vec<_>>>()
            .map(Some)
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
) -> Result<(std::path::PathBuf, Vec<RawBucket>)> {
    let manifest_path = ext_root.join("diskgraph_manifest.json");
    let bytes = std::fs::read(&manifest_path).map_err(|e| {
        RuntimeError::config(format!(
            "group_by_timestamp=True but cannot read {}: {e}",
            manifest_path.display()
        ))
    })?;
    let manifest: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
        RuntimeError::config(format!(
            "invalid diskgraph_manifest.json at {}: {e}",
            manifest_path.display()
        ))
    })?;
    let stages = manifest
        .get("task_dataset")
        .and_then(|td| td.get("stages"))
        .and_then(|s| s.as_object())
        .ok_or_else(|| {
            RuntimeError::config(format!(
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
        RuntimeError::config(format!(
            "group_by_timestamp=True but no stage with table_name {stage_table:?} \
             in {}",
            manifest_path.display()
        ))
    })?;
    if matches.next().is_some() {
        return Err(RuntimeError::config(format!(
            "diskgraph_manifest.json has multiple stages with table_name {stage_table:?} \
             in {} (stage table_names must be distinct)",
            manifest_path.display()
        )));
    }
    let bucket_index = stage_entry.get("bucket_index").ok_or_else(|| {
        RuntimeError::config(format!(
            "group_by_timestamp=True but stage {stage_table:?} has no bucket_index \
             in {} (ingest did not emit timestamp grouping for this stage)",
            manifest_path.display()
        ))
    })?;
    let rel_path = bucket_index
        .get("path")
        .and_then(|p| p.as_str())
        .ok_or_else(|| {
            RuntimeError::config(format!(
                "bucket_index for stage {stage_table:?} has no string `path`"
            ))
        })?;
    let buckets_json = bucket_index
        .get("buckets")
        .and_then(|b| b.as_array())
        .ok_or_else(|| {
            RuntimeError::config(format!(
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
                return Err(RuntimeError::config(format!(
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

/// Worker-sharded iterator over `RuntimeBatch`es of a seed-table stage.
/// Consumed via the merged-reader path (single engine over parent +
/// extension).
pub struct RuntimeBatchIterator {
    reader: Arc<diskgraph_index::graph::GraphIndexReader>,
    engine: Arc<DiskGraphEngine>,
    stage_table_catalog_key: String,
    stage_table: Arc<TableReader>,
    plan: SamplingPlan,
    /// The plan's feature projection, resolved against this graph ONCE at build
    /// time and shared by every batch this iterator yields (`Arc` clone =
    /// refcount bump). Resolving per batch would redo the same node-type and
    /// column-name string matching for every batch and every layer, and would
    /// also re-emit the build-time projection diagnostics on every batch.
    feature_projection: std::sync::Arc<diskgraph_engine::state::ResolvedProjection>,
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
    /// Static LP: hide fact rows with pseudo time above the seed's.
    pseudo_time: Option<PseudoTimeSeed>,
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
///     decoder, spawn a rayon pool. Depends only on
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
/// Later epochs reuse the mapped files, decoders, and thread pool.
///
/// `rnd_seed` and `num_threads` belong here rather than on `iter()` because the
/// engine is built from them -- and `num_threads` in particular is not a
/// performance-only knob: the engine's parallel neighbour sampling produces
/// different (individually reproducible) batches for different thread counts,
/// so two iterators that disagree on it are not sampling the same graph.
pub struct RuntimeIndex {
    reader: Arc<diskgraph_index::graph::GraphIndexReader>,
    engine: Arc<DiskGraphEngine>,
    namespace: String,
}
impl RuntimeIndex {
    /// Open the graph and training-table extension; propagate typed index errors.
    pub fn new(
        parent_root: String,
        tt_id: String,
        rnd_seed: Option<u64>,
        num_threads: Option<usize>,
    ) -> Result<Self> {
        let parent_path = std::path::Path::new(&parent_root);
        let reader = diskgraph_index::graph::GraphIndexReader::open_with_tt(parent_path, &tt_id)
            .map_err(|e| RuntimeError::system(e.to_string()))?;
        let namespace = tt_id.rsplit('/').next().unwrap_or(&tt_id).to_string();
        let reader = Arc::new(reader);
        let engine = Arc::new(
            DiskGraphEngine::from_reader(reader.clone(), rnd_seed, num_threads)
                .map_err(RuntimeError::from)?,
        );
        Ok(Self {
            reader,
            engine,
            namespace,
        })
    }

    /// Build an iterator over one (stage, epoch, shard) of this index.
    ///
    /// Cheap: no mmap, no decoders, no thread pool. Call it once per epoch, or
    /// once per chunk when work is handed out dynamically.
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
        pseudo_seed_time: Option<i64>,
        disable_pseudo_time: bool,
        // Feature projection pushdown: `node type -> the columns the caller
        // will actually consume`. Resolved ONCE in `build()` into a
        // `ResolvedProjection` shared by every batch this iterator yields.
        //
        // A plain parameter rather than a key inside `sampling_plan_json`:
        // `SamplingPlan` describes the TRAVERSAL, is embedded in
        // `BatchSamplingRequest`, is deep-cloned per batch and is bincode-
        // serialized -- none of which should be true of a caller-intent hint
        // that is read exactly once. Being a parameter also means a NEW kumo
        // against an OLD pinned wheel fails LOUDLY (`TypeError: ... unexpected
        // keyword argument 'projections'`) at iterator construction, rather
        // than silently degrading to select-all the way an ignored JSON key
        // would. A silently-inert projection is precisely the failure mode this
        // feature guards against everywhere else, so failing loud is correct.
        //
        // Per-type states, and ONLY these three (see
        // `SamplerEngine::resolve_projection`):
        //   * key ABSENT      -> select all for that type
        //   * `[]`            -> select NOTHING; the type is skipped entirely
        //   * `["a", "b"]`    -> materialize exactly these
        // and `None` for the whole argument => select all, everywhere.
        //
        // `[]` meaning "nothing" is why this parameter is `projections`
        // and not the earlier `feature_columns`: under that name `[]` meant
        // select-all. The two spellings are structurally identical
        // (`Option<HashMap<String, Vec<String>>>`), so the NAME is the entire
        // skew tripwire and must never be aliased back.
        projections: Option<std::collections::HashMap<String, Vec<String>>>,
    ) -> Result<RuntimeBatchIterator> {
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
            pseudo_seed_time,
            disable_pseudo_time,
            projections,
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
        pseudo_seed_time: Option<i64>,
        disable_pseudo_time: bool,
        // See `RuntimeIndex::iter`; `None` => select-all, `[]` per key => skip.
        projections: Option<std::collections::HashMap<String, Vec<String>>>,
    ) -> Result<Self> {
        if batch_size == 0 {
            return Err(RuntimeError::config("batch_size must be >= 1"));
        }
        if selected_stage_node_ids.is_some() {
            if shuffle {
                return Err(RuntimeError::config(
                    "selected_stage_node_ids is incompatible with shuffle=True",
                ));
            }
            if drop_last {
                return Err(RuntimeError::config(
                    "selected_stage_node_ids is incompatible with drop_last=True",
                ));
            }
            if group_by_timestamp {
                return Err(RuntimeError::config(
                    "selected_stage_node_ids is incompatible with group_by_timestamp=True",
                ));
            }
            if rhs_seed_node_type.is_some() {
                return Err(RuntimeError::config(
                    "selected_stage_node_ids is incompatible with rhs_seed_node_type (NodeRange mode)",
                ));
            }
        }
        if weighted_seed_sampling_json.is_some() {
            if selected_stage_node_ids.is_some() {
                return Err(RuntimeError::config(
                    "weighted_seed_sampling is incompatible with selected_stage_node_ids",
                ));
            }
            if drop_last {
                return Err(RuntimeError::config(
                    "weighted_seed_sampling is incompatible with drop_last=True",
                ));
            }
            if group_by_timestamp {
                return Err(RuntimeError::config(
                    "weighted_seed_sampling is incompatible with group_by_timestamp=True",
                ));
            }
            if rhs_seed_node_type.is_some() {
                return Err(RuntimeError::config(
                    "weighted_seed_sampling is incompatible with rhs_seed_node_type (NodeRange mode)",
                ));
            }
        }
        if group_by_timestamp && rhs_seed_node_type.is_some() {
            return Err(RuntimeError::config(
                "group_by_timestamp is incompatible with rhs_seed_node_type (NodeRange mode)",
            ));
        }
        if group_by_timestamp && shuffle && rnd_seed.is_none() {
            return Err(RuntimeError::config(
                "rnd_seed is required when group_by_timestamp=True (used for the \
                 bucket-order + within-bucket shuffle)",
            ));
        }
        if num_workers == 0 {
            return Err(RuntimeError::config("num_workers must be >= 1"));
        }
        if worker_id >= num_workers {
            return Err(RuntimeError::config(format!(
                "worker_id {worker_id} out of range for num_workers {num_workers}"
            )));
        }
        if rhs_seed_node_type.is_some() && rhs_seed_time.is_none() {
            return Err(RuntimeError::config(
                "rhs_seed_time is required when rhs_seed_node_type is set (NodeRange mode)",
            ));
        }
        let negatives = match neg_rhs_node_type {
            Some(rhs_node_type) => {
                let amount_m = neg_amount_m.ok_or_else(|| {
                    RuntimeError::config("neg_amount_m is required when neg_rhs_node_type is set")
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
            RuntimeError::config(format!(
                "unknown stage table in merged catalog: {stage_catalog_key}"
            ))
        })?;
        let selected_stage_node_ids = if let Some(ids) = selected_stage_node_ids {
            let mut local_ids = Vec::new();
            for (idx, node_id) in ids.into_iter().enumerate() {
                if (idx as u64) % num_workers == worker_id {
                    if stage_table_arc.shard_for_node(node_id).is_none() {
                        return Err(RuntimeError::data(format!(
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
            .map_err(|e| RuntimeError::config(format!("Invalid plan JSON: {e}")))?;
        // Grouped (single-timestamp) path: build the lazy bucket schedule from
        // the per-stage `bucket_index.bin` the ingest side recorded. `offsets`
        // is unused in this mode (left empty); FAILS LOUD on a missing artifact.
        let bucket_schedule = if group_by_timestamp {
            let ext_root = reader.tt_extension_root().ok_or_else(|| {
                RuntimeError::config("group_by_timestamp=True but reader has no TT extension root")
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
            )?)
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
                .map_err(RuntimeError::from)?;
            let num_rhs = hi - lo + 1;
            // `base = lo` so each emitted value IS the global node id, which is
            // what the materialised path achieved with a post-pass over the Vec.
            ShardStream::new(
                num_rhs,
                num_workers,
                worker_id,
                shuffle,
                rnd_seed,
                epoch,
                rhs_type,
                lo,
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
        let pseudo_time = PseudoTimeSeed::for_stage(
            reader.as_ref(),
            &stage_table,
            pseudo_seed_time,
            disable_pseudo_time,
        )?;
        // Explicit fused mode takes precedence over feature projection.
        let projections =
            if fused == Some(true) && projections.as_ref().is_some_and(|p| !p.is_empty()) {
                tracing::warn!(
                    "fused=True does not support feature projection; ignoring feature \
                 projections and fetching all non-reserved columns"
                );
                None
            } else {
                projections
            };
        let projections_non_empty = projections.as_ref().is_some_and(|p| !p.is_empty());
        let chosen_fused = match (fused, projections_non_empty) {
            (Some(v), _) => v,
            (None, true) => {
                tracing::info!(
                    batch_size = batch_size,
                    stage_table = %stage_catalog_key,
                    "Feature projection supplied; forcing non-fused dispatch \
                     (fused-mode projection pushdown is not supported)"
                );
                false
            }
            (None, false) => {
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
        // Resolve the plan's feature projection ONCE, here, against this graph.
        // Doing it before the struct literal keeps `engine` borrowable (the
        // literal moves it) and, more importantly, means the projection
        // diagnostics in `resolve_projection` are emitted exactly once per
        // iterator instead of once per batch.
        let feature_projection =
            std::sync::Arc::new(engine.resolve_projection(&projections.unwrap_or_default()));
        Ok(Self {
            reader,
            engine,
            stage_table_catalog_key: stage_catalog_key,
            stage_table: stage_table_arc,
            feature_projection,
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
            pseudo_time,
        })
    }
}
impl RuntimeBatchIterator {
    /// Return this worker's sample count, or its batch count in timestamp-grouped mode.
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

    /// Return the number of rows in the selected stage table.
    pub fn stage_size(&self) -> u64 {
        self.stage_table.node_count()
    }

    /// Yield the next native batch, or None at exhaustion; propagate sampling errors.
    pub fn next_batch(&mut self) -> Result<Option<RuntimeBatch>> {
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
            let pseudo_time = self.pseudo_time.as_ref().map(PseudoTimeSeed::filter);
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
            let runtime_batch = engine
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
                    pseudo_time,
                    Some(std::sync::Arc::clone(&self.feature_projection)),
                )
                .map_err(RuntimeError::from)?;
            return Ok(Some(runtime_batch));
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
            let pseudo_time = self.pseudo_time.as_ref().map(PseudoTimeSeed::filter);
            let runtime_batch = engine
                .sample_runtime_batch(
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
                    pseudo_time,
                    Some(std::sync::Arc::clone(&self.feature_projection)),
                )
                .map_err(RuntimeError::from)?;
            return Ok(Some(runtime_batch));
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
            let pseudo_time = self.pseudo_time.as_ref().map(PseudoTimeSeed::filter);
            let runtime_batch = engine
                .sample_runtime_batch(
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
                    pseudo_time,
                    Some(std::sync::Arc::clone(&self.feature_projection)),
                )
                .map_err(RuntimeError::from)?;
            return Ok(Some(runtime_batch));
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
            let engine = &self.engine;
            let reader = &self.reader;
            let plan = &self.plan;
            let flags = &self.flags;
            let pseudo_time = self.pseudo_time.as_ref().map(PseudoTimeSeed::filter);
            let projection = &self.feature_projection;
            let runtime_batch = engine
                .sample_runtime_batch(
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
                    pseudo_time,
                    Some(std::sync::Arc::clone(projection)),
                )
                .map_err(RuntimeError::from)?;
            return Ok(Some(runtime_batch));
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
                    RuntimeError::data(format!(
                        "offset {offset} out of range for stage table {}",
                        self.stage_table_catalog_key
                    ))
                })?;
            stage_node_ids.push(node_id);
        }

        // Sidecars (weight.bin, etc.) are auto-detected inside the engine
        // via `reader.tt_extension_root()` — no path threading at this layer.
        let engine = &self.engine;
        let reader = &self.reader;
        let stage_table_catalog_key = &self.stage_table_catalog_key;
        let plan = &self.plan;
        let flags = &self.flags;
        let negatives = self.negatives.as_ref();
        let entity_node_type = self.entity_node_type.as_deref();
        let target_node_type = self.target_node_type.as_deref();
        let pseudo_time = self.pseudo_time.as_ref().map(PseudoTimeSeed::filter);
        let runtime_batch = engine
            .sample_runtime_batch(
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
                pseudo_time,
                Some(std::sync::Arc::clone(&self.feature_projection)),
            )
            .map_err(RuntimeError::from)?;
        Ok(Some(runtime_batch))
    }
}
#[cfg(test)]
mod tests {
    use super::make_worker_offsets;
    use super::{
        read_bucket_index, worker_batch_count, BucketOrders, BucketSchedule, F64Sidecar, U64Sidecar,
    };
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

        let mut bucket_dists = bucket_dists;
        bucket_dists.push(
            (0..20)
                .map(|ts| (ts, (ts as u64 * 31..(ts as u64 + 1) * 31).collect()))
                .collect(),
        );

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
                        )
                        .unwrap();
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
                    )
                    .unwrap();
                    assert_eq!(
                        total_batches_seen, one.total_batches,
                        "dist{di} nw={num_workers} bs={batch_size}: Σ len != B"
                    );
                }
            }
        }
    }

    #[test]
    fn bucket_orders_store_only_buckets_under_64_batches() {
        let mut cache = BucketOrders::new(10);
        assert_eq!(cache.get(0, 639, 1).stored_rows(), 639);
        assert_eq!(cache.get(1, 640, 1).stored_rows(), 0);
        let order = cache.get(1, 640, 1);
        let mut seen: Vec<u64> = (0..640).map(|p| order.at(p)).collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..640).collect::<Vec<_>>());
    }

    #[test]
    fn bucket_orders_preserve_selection_across_memory_budget_evictions() {
        let mut cache = BucketOrders::new(20_000);
        let positions = [0, 1, 400_000, 799_999];
        let mut expected = Vec::new();
        for bucket in 0..6 {
            let order = cache.get(bucket, 800_000, bucket as u64);
            expected.push(positions.map(|position| order.at(position)));
            let bytes: usize = cache
                .entries
                .iter()
                .map(|(_, order)| order.stored_rows() as usize * size_of::<u64>())
                .sum();
            assert!(bytes <= 32 * 1024 * 1024);
        }
        for (bucket, expected) in expected.iter().enumerate() {
            let order = cache.get(bucket, 800_000, bucket as u64);
            assert_eq!(&positions.map(|position| order.at(position)), expected);
        }
    }

    #[test]
    fn a_bucket_larger_than_the_cache_is_generated_whatever_the_cache_holds() {
        let generated = FeistelPerm::new(4_500_000, 17);
        let positions = [0, 1, 2_250_000, 4_499_999];
        let expected = positions.map(|position| generated.at(position));
        let mut cache = BucketOrders::new(100_000);
        assert_eq!(cache.get(0, 4_500_000, 17).stored_rows(), 0);
        assert_eq!(
            positions.map(|p| cache.get(0, 4_500_000, 17).at(p)),
            expected
        );
        let mut warm = BucketOrders::new(100_000);
        for bucket in 1..20 {
            warm.get(bucket, 200_000, bucket as u64);
        }
        assert_eq!(warm.get(0, 4_500_000, 17).stored_rows(), 0);
        assert_eq!(
            positions.map(|p| warm.get(0, 4_500_000, 17).at(p)),
            expected
        );
    }

    #[test]
    fn capped_timestamp_batches_give_each_row_equal_inclusion() {
        let tmp = tempfile::tempdir().unwrap();
        let dist = vec![
            (10, (0..31).collect()),
            (20, (31..101).collect()),
            (30, vec![101, 102, 103]),
        ];
        let (path, raw, _) = write_bin(tmp.path(), &dist);
        let mut counts = [0usize; 104];
        let mut mixed = 0;
        for epoch in 0..2000 {
            let mut schedule = BucketSchedule::build(
                raw.clone(),
                path.clone(),
                17,
                epoch,
                "train".into(),
                8,
                1,
                0,
            )
            .unwrap();
            let mut selected = HashSet::new();
            let mut timestamps = HashSet::new();
            for _ in 0..2 {
                let batch = schedule.next_node_ids().unwrap().unwrap();
                timestamps.insert(if batch[0] < 31 {
                    10
                } else if batch[0] < 101 {
                    20
                } else {
                    30
                });
                for row in batch {
                    assert!(selected.insert(row));
                    counts[row as usize] += 1;
                }
            }
            mixed += usize::from(timestamps.len() > 1);
        }
        let expected = 2000.0 * 2.0 / 14.0;
        for count in counts {
            assert!(
                (count as f64 - expected).abs() < 65.0,
                "row selected {count} times"
            );
        }
        assert!(mixed > 900);
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
                    )
                    .unwrap();
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
        .unwrap()
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
            )
            .unwrap();
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
            )
            .unwrap();
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
    /// missing, the stage is absent, or it has no `bucket_index`.
    #[test]
    fn read_bucket_index_fails_loud() {
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
        assert!(
            err.to_string().contains("no bucket_index"),
            "expected a 'no bucket_index' error, got: {err}"
        );
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
        for n in [
            1u64, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 63, 64, 65, 1023, 1024, 1025, 4095, 4096, 4097,
            65535, 65536, 65537,
        ] {
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
        for (n, k) in [
            (100u64, 7u64),
            (1000, 16),
            (41, 41),
            (41, 64),
            (1, 8),
            (9973, 13),
        ] {
            let mut seen: Vec<u64> = Vec::new();
            for w in 0..k {
                let st = ShardStream::new(n, k, w, true, Some(5), 3, "train", 0).unwrap();
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
                    let st = ShardStream::new(n, k, w, false, None, 0, "s", 0).unwrap();
                    let counted = (0..n).filter(|p| p % k == w).count();
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
                let old = make_worker_offsets(n, k, w, true, Some(7), 2, "train").unwrap();
                let st = ShardStream::new(n, k, w, true, Some(7), 2, "train", 0).unwrap();
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
            let st = ShardStream::new(n, 16, 3, true, Some(99), 0, "train", 0).unwrap();
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
            let st = ShardStream::new(n, 8, 1, true, Some(4), 0, "train", 0).unwrap();
            let len = st.len() as u64;
            let blocks = 100u64;
            let per_block = (MIXING_SAMPLES / blocks).min(len / blocks.max(1));
            let (mut total_gap, mut pairs) = (0u128, 0u128);
            for b in 0..blocks {
                let start = b * len / blocks;
                for j in start + 1..(start + per_block).min(len) {
                    total_gap += u128::from(st.at(j as usize - 1).abs_diff(st.at(j as usize)));
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
            (1 << 26) + 1, // odd ideal width: the band that used to cost 4x
            100_000_000,   // likewise
            41_756_131,    // even ideal width: was already fine
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
        assert!(
            worst.1 <= 2.0,
            "worst ratio {:.2} at n={}",
            worst.1,
            worst.0
        );
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
