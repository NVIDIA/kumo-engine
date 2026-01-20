//! Implementations of the `Hash` trait that abstracts over 64 and 128-bit hashes.
use mem_dbg::MemSize;

use crate::KeyT;
use gxhash::GxHasher;
use std::fmt::Debug;
use std::hash::Hasher as StdHasher;

/// The `Hash` trait returns a 64 or 128-bit `Hash`. From this, two `u64` values are extracted.
///
/// When 64-bit hashes are enough, we simply return the same hash (the `u64`
/// `Self` value) as the low and high part.
///
/// When 128-bit hashes are needed, the two functions return the low/high half of bits.
///
/// Our method never needs the full hash value, and instead uses the two hashes
/// in different places to extract sufficient entropy.
pub trait Hash: Copy + Debug + Default + Send + Sync + Eq {
    /// Returns the low 64bits of the hash.
    fn low(&self) -> u64;
    /// Returns the high 64bits of the hash.
    fn high(&self) -> u64;
}

impl Hash for u64 {
    fn low(&self) -> u64 {
        *self
    }
    fn high(&self) -> u64 {
        *self
    }
}

impl Hash for u128 {
    fn low(&self) -> u64 {
        *self as u64
    }
    fn high(&self) -> u64 {
        (*self >> 64) as u64
    }
}

/// Wrapper trait for various hash functions.
pub trait Hasher<Key: ?Sized>: Clone + Sync {
    type H: Hash;
    fn hash(x: &Key, seed: u64) -> Self::H;
}

fn to_bytes<Key: ?Sized>(x: &Key) -> &[u8] {
    unsafe { std::slice::from_raw_parts(x as *const Key as *const u8, std::mem::size_of_val(x)) }
}

/// Multiply the key by a mixing constant.
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[derive(Clone)]
pub(crate) struct MulHash;

/// Fast 64-bit hash with gxhash.
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[derive(Clone, MemSize)]
pub struct GxHash64;
/// 128-bit gxhash output.
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[derive(Clone, MemSize)]
pub struct GxHash128;

impl MulHash {
    // Reuse the mixing constant from FxHash.
    pub const C: u64 = 0x517cc1b727220a95;
}
impl Hasher<u64> for MulHash {
    type H = u64;
    fn hash(x: &u64, seed: u64) -> u64 {
        Self::C.wrapping_mul(*x ^ seed)
    }
}

impl<Key: KeyT + ?Sized> Hasher<Key> for GxHash64 {
    type H = u64;
    #[inline(always)]
    fn hash(x: &Key, seed: u64) -> u64 {
        let mut hasher = GxHasher::with_seed(seed as i64);
        x.hash(&mut hasher);
        hasher.finish()
    }
}

impl Hasher<u64> for GxHash128 {
    type H = u128;
    #[inline(always)]
    fn hash(x: &u64, seed: u64) -> u128 {
        gxhash::gxhash128(to_bytes(x), seed as i64)
    }
}

impl Hasher<[u8]> for GxHash128 {
    type H = u128;
    #[inline(always)]
    fn hash(x: &[u8], seed: u64) -> u128 {
        gxhash::gxhash128(to_bytes(x), seed as i64)
    }
}

impl Hasher<Vec<u8>> for GxHash128 {
    type H = u128;
    #[inline(always)]
    fn hash(x: &Vec<u8>, seed: u64) -> u128 {
        gxhash::gxhash128(to_bytes(x.as_slice()), seed as i64)
    }
}

impl<const N: usize> Hasher<[u8; N]> for GxHash128 {
    type H = u128;
    #[inline(always)]
    fn hash(x: &[u8; N], seed: u64) -> u128 {
        gxhash::gxhash128(to_bytes(x), seed as i64)
    }
}

impl Hasher<&[u8]> for GxHash128 {
    type H = u128;
    #[inline(always)]
    fn hash(x: &&[u8], seed: u64) -> u128 {
        gxhash::gxhash128(to_bytes(*x), seed as i64)
    }
}

impl<const N: usize> Hasher<&[u8; N]> for GxHash128 {
    type H = u128;
    #[inline(always)]
    fn hash(x: &&[u8; N], seed: u64) -> u128 {
        gxhash::gxhash128(to_bytes(*x), seed as i64)
    }
}
