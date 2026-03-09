//! # PtrHash: Minimal Perfect Hashing at RAM Throughput
//!
//! PtrHash builds a _minimal perfect hash function_, that is,
//! a hash function that maps a fixed set of keys to `{0, ..., n-1}`.
//!
//! PtrHash was developed for large key sets of at least 1 million keys, and has been tested up to 10^11 keys.
//! It only uses 2.4 bits per key.
//!
//! It can also be used for arbitrary small sets.
//! In this case, the space efficiency will be less due to a relatively large constant overhead.
//!
//! See the GitHub [readme](https://github.com/ragnargrootkoerkamp/ptrhash)
//! or paper ([arXiv](https://arxiv.org/abs/2502.15539), [blog version](https://curiouscoding.nl/posts/ptrhash/))
//! for details on the algorithm and performance.
//!
//! Construction uses a restartable key source plus a reusable scratch buffer
//! via `try_new_from_source_with_scratch`.
//!
//! ## Partitioning
//!
//! By default, PtrHash partitions the keys into multiple parts.
//! This speeds up construction in two ways:
//! - smaller parts have better cache locality, and
//! - parts can be constructed in parallel.
//!
//! However, at query time there is a small overhead to compute the part of each key.
//! To avoid this, set [`PtrHashParams::single_part`] to `true`.

/// Customizable Hasher trait.
pub mod hash;
/// Extendable backing storage trait and types.
pub mod pack;
/// Internal utilities used by ptrhash.
pub mod util;
pub use scratch::PtrHashScratch;

pub mod bucket_fn;
mod bucket_idx;
mod build;
mod radix;
mod reduce;
mod scratch;
mod sort_buckets;
#[cfg(test)]
mod test;

use bitvec::{bitvec, vec::BitVec};
use bucket_fn::BucketFn;
use bucket_fn::CubicEps;
use cacheline_ef::CachelineEfVec;
use tracing::trace;
use tracing::warn;
use mem_dbg::MemSize;
use pack::MutPacked;
use std::{borrow::Borrow, default::Default, marker::PhantomData};

use crate::{hash::*, pack::Packed, reduce::*};

#[derive(Debug)]
pub(crate) enum SortPartsError {
    DuplicateHashes,
    PartOverflow,
}

/// Parameters for PtrHash construction.
///
/// While all fields are public, prefer [`PtrHashParams::default()`].
#[derive(Clone, Copy, Debug, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[cfg_attr(feature = "epserde", deep_copy)]
pub struct PtrHashParams<BF> {
    /// Set to false to disable remapping to a minimal PHF.
    pub remap: bool,
    /// Use `n/alpha` slots approximately.
    pub alpha: f64,
    /// Use average bucket size lambda.
    pub lambda: f64,
    /// Bucket function
    pub bucket_fn: BF,
    /// Force using a single part.
    pub single_part: bool,
}

impl PtrHashParams<CubicEps> {
    /// Default parameters for inputs >1M keys.
    ///
    /// Takes `2.4` bits/key, and trades off space and speed.
    /// - `alpha=0.99`
    /// - `lambda=3.5`
    /// - `bucket_fn=CubicEps`
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self {
            remap: true,
            alpha: 0.99,
            lambda: 3.5,
            bucket_fn: CubicEps,
            single_part: false,
        }
    }
}

impl Default for PtrHashParams<CubicEps> {
    fn default() -> Self {
        Self::default()
    }
}

/// Type alias to simplify construction.
///
/// [`PtrHash`] has a large number of generics, partly to support epserde.
/// [`DefaultPtrHash`] fills in most values.
///
/// Use this as [`DefaultPtrHash::try_new_from_source_with_scratch()`].
pub type DefaultPtrHash<Hx = hash::GxHash64, Key = u64, BF = bucket_fn::CubicEps> =
    PtrHash<Key, BF, CachelineEfVec, Hx, Vec<u8>>;

/// Trait that keys must satisfy.
pub trait KeyT: Send + Sync + std::hash::Hash {}
impl<T: Send + Sync + std::hash::Hash> KeyT for T {}

/// Restartable source of keys for PtrHash construction.
pub trait KeySource<Key: KeyT> {
    type Iter<'a>: Iterator<Item = &'a Key> + Clone
    where
        Self: 'a,
        Key: 'a;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn iter(&self) -> Self::Iter<'_>;
}

impl<Key: KeyT + Eq, V, S: std::hash::BuildHasher> KeySource<Key>
    for std::collections::HashMap<Key, V, S>
{
    type Iter<'a>
        = std::collections::hash_map::Keys<'a, Key, V>
    where
        Self: 'a,
        Key: 'a;

    fn len(&self) -> usize {
        std::collections::HashMap::len(self)
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.keys()
    }
}

// Some fixed algorithmic decisions.
type Rp = FastReduce;
type Rb = FastReduce;
type RemSlots = FM32;
type Pilot = u64;
type PilotHash = u64;

/// PtrHash datastructure.
/// It is recommended to use PtrHash with default types.
///
/// - `Key`: The type of keys to hash.
/// - `BF`: The bucket function to use. Inferred from `PtrHashParams` when calling `PtrHash::new()`.
/// - `F`: The packing to use for remapping free slots, default `CachelineEf`.
/// - `Hx`: The hasher to use for keys, default `GxHash64`.
/// - `V`: The pilots type. Usually `Vec<u8>`, or `&[u8]` for Epserde.
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[derive(Clone, MemSize)]
pub struct PtrHash<
    Key: KeyT + ?Sized = u64,
    BF: BucketFn = bucket_fn::CubicEps,
    F: Packed = CachelineEfVec,
    Hx: Hasher<Key> = hash::GxHash64,
    V: AsRef<[u8]> = Vec<u8>,
> {
    params: PtrHashParams<BF>,

    /// The number of keys.
    n: usize,
    /// The total number of parts.
    parts: usize,
    /// The maximal number of parts per shard.
    /// The last shard may have fewer parts.
    parts_per_shard: usize,
    /// The total number of slots.
    slots_total: usize,
    /// The total number of buckets.
    buckets_total: usize,
    /// The number of slots per part, always a power of 2.
    slots: usize,
    /// The number of buckets per part.
    buckets: usize,

    // Precomputed fast modulo operations.
    /// Fast %parts.
    rem_parts: Rp,
    /// Fast %b.
    rem_buckets: Rb,
    /// Fast %b_total.
    rem_buckets_total: Rb,

    /// Fast %s when there is only a single part.
    rem_slots: RemSlots,

    // Computed state.
    /// The global seed.
    seed: u64,
    /// The pilots.
    pilots: V,
    /// Remap the out-of-bound slots to free slots.
    remap: F,
    _key: PhantomData<Key>,
    _hx: PhantomData<Hx>,
}

/// An empty PtrHash instance. Mostly useless, but may be convenient.
impl<Key: KeyT, BF: BucketFn, F: MutPacked, Hx: Hasher<Key>> Default
    for PtrHash<Key, BF, F, Hx, Vec<u8>>
where
    PtrHashParams<BF>: Default,
{
    fn default() -> Self {
        PtrHash {
            params: <PtrHashParams<BF> as Default>::default(),

            n: 0,
            parts: 0,
            parts_per_shard: 0,
            slots_total: 0,
            buckets_total: 0,
            slots: 0,
            buckets: 0,
            rem_parts: FastReduce::new(0),
            rem_buckets: FastReduce::new(0),
            rem_buckets_total: FastReduce::new(0),
            rem_slots: RemSlots::new(1),
            seed: 0,
            pilots: vec![],
            remap: F::default(),
            _key: PhantomData,
            _hx: PhantomData,
        }
    }
}

/// Construction methods.
impl<Key: KeyT, BF: BucketFn, F: MutPacked, Hx: Hasher<Key>> PtrHash<Key, BF, F, Hx, Vec<u8>> {
    /// Build from a restartable key source.
    pub fn try_new_from_source_with_scratch<S: KeySource<Key>>(
        source: &S,
        params: PtrHashParams<BF>,
        scratch: &mut PtrHashScratch<Hx::H>,
    ) -> Option<Self> {
        let mut ptr_hash = Self::init(source.len(), params);
        ptr_hash.compute_pilots_with_scratch(source.iter(), scratch)?;
        Some(ptr_hash)
    }

    /// Only initialize the parameters; do not compute the pilots yet.
    fn init(n: usize, mut params: PtrHashParams<BF>) -> Self {
        assert!(n < (1 << 40), "Number of keys must be less than 2^40.");

        const SMALL_N_THRESHOLD: usize = 1_000_000;
        if n <= SMALL_N_THRESHOLD {
            // Small-n path: bias toward more buckets and higher slack to avoid
            // indistinguishable buckets on tiny inputs.
            if params.alpha < 0.995 {
                params.alpha = 0.995;
            }
            if params.lambda > 2.0 {
                params.lambda = 2.0;
            }
        }

        let shards = 1;

        // Formula of Vigna, eps-cost-sharding.
        let eps = 1.0 - params.alpha;
        let x = n as f64 * eps * eps / 2.0;
        let target_parts = if x <= 1.0 { 1.0 } else { x / x.ln() };
        let parts_per_shard = (target_parts.floor() as usize) / shards;
        let mut parts = parts_per_shard.max(1) * shards;

        let mut keys_per_part;
        let mut slots_per_part;
        let mut buckets_per_part;
        let mut slots_total;
        let mut buckets_total;

        loop {
            keys_per_part = n / parts;
            slots_per_part = (keys_per_part as f64 / params.alpha) as usize;
            // Avoid powers of two, since then %S does not depend on all bits.
            if slots_per_part.is_power_of_two() {
                slots_per_part += 1;
            }
            slots_total = parts * slots_per_part;
            // Add a few extra buckets to avoid collisions for small n.
            buckets_per_part = (keys_per_part as f64 / params.lambda).ceil() as usize + 3;
            buckets_total = parts * buckets_per_part;

            if parts == 1 {
                break;
            }

            // Ensure the expected maximum part size fits the available slots.
            let exp_keys_per_part = n as f64 / parts as f64;
            let stddev = exp_keys_per_part.sqrt();
            let stddevs_away = ((parts as f64).ln() * 2.0).sqrt();
            let exp_max = exp_keys_per_part + stddev * stddevs_away;
            let buf_max = exp_max + 2.0 * stddev;
            if buf_max < slots_per_part as f64 {
                break;
            }

            parts = (parts / 2).max(1);
        }

        let parts_per_shard = parts / shards;

        trace!("        keys: {n:>10}");
        trace!("      shards: {shards:>10}");
        trace!("       parts: {parts:>10}");
        trace!("   slots/prt: {slots_per_part:>10}");
        trace!("   slots tot: {slots_total:>10}");
        trace!("  real alpha: {:>10.4}", n as f64 / slots_total as f64);
        trace!(" buckets/prt: {buckets_per_part:>10}");
        trace!(" buckets tot: {buckets_total:>10}");
        trace!("keys/ bucket: {:>13.2}", n as f64 / buckets_total as f64);

        params
            .bucket_fn
            .set_buckets_per_part(buckets_per_part as u64);

        Self {
            params,
            n,
            parts,
            parts_per_shard,
            slots_total,
            slots: slots_per_part,
            buckets_total,
            buckets: buckets_per_part,
            rem_parts: Rp::new(parts),
            rem_buckets: Rb::new(buckets_per_part),
            rem_buckets_total: Rb::new(buckets_total),
            rem_slots: RemSlots::new(slots_per_part),
            seed: 0,
            pilots: Default::default(),
            remap: F::default(),
            _key: PhantomData,
            _hx: PhantomData,
        }
    }

    fn compute_pilots_with_scratch<'a>(
        &mut self,
        keys: impl Iterator<Item = impl Borrow<Key>> + Clone + 'a,
        scratch: &mut PtrHashScratch<Hx::H>,
    ) -> Option<()> {
        // Initialize arrays;
        scratch.taken.clear();
        scratch.pilots.clear();

        let mut tries = 0;
        const MAX_TRIES: usize = 10;

        let mut rng = fastrand::Rng::with_seed(31415);

        // Loop over global seeds `s`.
        's: loop {
            tries += 1;
            if tries > MAX_TRIES {
                warn!("PtrHash failed to find a global seed after {MAX_TRIES} tries.");
                return None;
            }
            if tries > 1 {
                trace!("NEW TRY Try {tries} for global seed.");
            }

            // Choose a global seed s.
            self.seed = rng.u64(..);
            trace!(
                "PtrHash build try {} seed={} n={} parts={} slots/part={} buckets/part={} alpha={} lambda={}",
                tries,
                self.seed,
                self.n,
                self.parts,
                self.slots,
                self.buckets,
                self.params.alpha,
                self.params.lambda
            );

            // Reset output-memory.
            scratch.pilots.clear();
            scratch.pilots.resize(self.buckets_total, 0);

            // TODO: Compress taken on the fly, instead of pre-allocating the entire thing.
            for taken in scratch.taken.iter_mut() {
                taken.clear();
                taken.resize(self.slots, false);
            }
            scratch
                .taken
                .resize_with(self.parts, || bitvec![0; self.slots]);

            scratch.hashes.clear();
            scratch.hashes.reserve(self.n);
            scratch
                .hashes
                .extend(keys.clone().map(|key| self.hash_key(key.borrow())));

            // Determine the buckets.
            let sorted = self.sort_parts_with_scratch(
                0,
                &mut scratch.hashes,
                &mut scratch.part_starts,
                &mut scratch.hashes_tmp,
            );
            if sorted.is_err() {
                continue 's;
            }

            // Compute pilots.
            let mut hashes = Vec::new();
            let mut part_starts = Vec::new();
            std::mem::swap(&mut scratch.hashes, &mut hashes);
            std::mem::swap(&mut scratch.part_starts, &mut part_starts);
            let shard_ok = self.build_shard(0, &hashes, &part_starts, scratch);
            std::mem::swap(&mut scratch.hashes, &mut hashes);
            std::mem::swap(&mut scratch.part_starts, &mut part_starts);
            if !shard_ok {
                trace!("Could not find pilots");
                continue 's;
            }

            let remap = self.remap_free_slots_with_scratch(&scratch.taken, &mut scratch.free_slots);
            if remap.is_err() {
                trace!("Failed to construct CachelineEF");
                continue 's;
            }

            break 's;
        }

        // Pack the data.
        self.pilots.clone_from(&scratch.pilots);

        Some(())
    }

    fn remap_free_slots_with_scratch(
        &mut self,
        taken: &Vec<BitVec>,
        free_slots: &mut Vec<u64>,
    ) -> Result<(), ()> {
        assert_eq!(
            taken.iter().map(|t| t.count_zeros()).sum::<usize>(),
            self.slots_total - self.n,
            "Not the right number of free slots left!\n total slots {} - n {}",
            self.slots_total,
            self.n
        );

        if !self.params.remap || self.slots_total == self.n {
            return Ok(());
        }

        // Compute the free spots.
        free_slots.clear();
        free_slots.reserve(self.slots_total - self.n);
        let get = |t: &Vec<BitVec>, idx: usize| t[idx / self.slots][idx % self.slots];
        for i in taken
            .iter()
            .enumerate()
            .flat_map(|(p, t)| {
                let offset = p * self.slots;
                t.iter_zeros().map(move |i| offset + i)
            })
            .take_while(|&i| i < self.n)
        {
            while !get(taken, self.n + free_slots.len()) {
                free_slots.push(i as u64);
            }
            free_slots.push(i as u64);
        }
        self.remap = MutPacked::try_new(free_slots.clone()).ok_or(())?;
        Ok(())
    }
}

/// Indexing methods.
impl<Key: KeyT, BF: BucketFn, F: Packed, Hx: Hasher<Key>, V: AsRef<[u8]>>
    PtrHash<Key, BF, F, Hx, V>
{
    #[inline]
    fn hash_key(&self, key: &Key) -> Hx::H {
        Hx::hash(key, self.seed)
    }

    #[inline]
    fn hash_pilot(&self, pilot: Pilot) -> PilotHash {
        MulHash::hash(&pilot, self.seed)
    }

    #[inline]
    fn part(&self, hx: Hx::H) -> usize {
        self.rem_parts.reduce(hx.high())
    }

    #[inline]
    fn bucket_in_part(&self, hx: u64) -> usize {
        if BF::B_OUTPUT {
            let b = self.params.bucket_fn.call(hx);
            debug_assert!(b < self.buckets as u64);
            b as usize
        } else {
            self.rem_buckets.reduce(self.params.bucket_fn.call(hx))
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// `self.index()` always returns below this bound.
    /// Should be around `n/alpha ~ 1.01*n`.
    pub fn max_index(&self) -> usize {
        self.slots_total
    }

    /// Get a non-minimal index of the given key, in `[0, n/alpha)`.
    /// Use `index` to get a key in `[0, n)`.
    #[inline]
    pub fn index_no_remap(&self, key: &Key) -> usize {
        let hx = self.hash_key(key);
        let b = self.bucket(hx);
        let pilot = self.pilots.as_ref().index(b);
        self.slot(hx, pilot)
    }

    /// Get the index for `key` in `[0, n)`.
    #[inline]
    pub fn index(&self, key: &Key) -> usize {
        let hx = self.hash_key(key);
        let b = self.bucket(hx);
        let p = self.pilots.as_ref().index(b);
        let slot = self.slot(hx, p);
        if slot < self.n {
            slot
        } else {
            self.remap.index(slot - self.n) as usize
        }
    }

    /// See bucket.rs for additional implementations.
    /// Returns the offset in the slots array for the current part and the bucket index.
    fn bucket(&self, hx: Hx::H) -> usize {
        if BF::LINEAR {
            return self.rem_buckets_total.reduce(hx.high());
        }

        // Extract the high bits for part selection; do normal bucket
        // computation within the part using the remaining bits.
        // NOTE: This is somewhat slow, but doing better is hard.
        let (part, hx) = self.rem_parts.reduce_with_remainder(hx.high());
        let bucket = self.bucket_in_part(hx);
        part * self.buckets + bucket
    }

    /// Slot uses the 64 low bits of the hash.
    fn slot(&self, hx: Hx::H, pilot: u64) -> usize {
        (self.part(hx) * self.slots) + self.slot_in_part(hx, pilot)
    }

    fn slot_in_part(&self, hx: Hx::H, pilot: Pilot) -> usize {
        self.slot_in_part_hp(hx, self.hash_pilot(pilot))
    }

    /// Slot uses the 64 low bits of the hash.
    fn slot_in_part_hp(&self, hx: Hx::H, hp: PilotHash) -> usize {
        self.rem_slots.reduce(hx.low() ^ hp)
    }
}
