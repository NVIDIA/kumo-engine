//! Implementations of the `Hash` trait that abstracts over 64 and 128-bit hashes.
use mem_dbg::MemSize;

use crate::KeyT;
use gxhash::GxHasher;
use xxhash_rust::xxh3::xxh3_64_with_seed;
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
    /// Only [`PkHash`] honours `fallback`; the rest ignore it.
    fn hash(x: &Key, seed: u64, fallback: bool) -> Self::H;
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
pub struct PkHash;
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
    fn hash(x: &u64, seed: u64, _fallback: bool) -> u64 {
        Self::C.wrapping_mul(*x ^ seed)
    }
}

impl<Key: KeyT + ?Sized> Hasher<Key> for PkHash {
    type H = u64;
    #[inline(always)]
    fn hash(x: &Key, seed: u64, fallback: bool) -> u64 {
        if fallback {
            let mut bytes = KeyBytes::default();
            x.hash(&mut bytes);
            return xxh3_64_with_seed(bytes.as_slice(), seed);
        }
        let mut hasher = GxHasher::with_seed(seed as i64);
        x.hash(&mut hasher);
        hasher.finish()
    }
}

/// 64-bit hash of a string that keys an exact identity: gxhash up to
/// [`MAX_GXHASH_INJECTIVE_LEN`] bytes, xxh3 above, where gxhash collisions
/// are constructible.
///
/// [`MAX_GXHASH_INJECTIVE_LEN`]: crate::MAX_GXHASH_INJECTIVE_LEN
#[inline]
pub fn hash_str(value: &str, seed: u64) -> u64 {
    let bytes = value.as_bytes();
    if bytes.len() > crate::MAX_GXHASH_INJECTIVE_LEN {
        return xxh3_64_with_seed(bytes, seed);
    }
    let mut hasher = GxHasher::with_seed(seed as i64);
    hasher.write(bytes);
    hasher.finish()
}

/// Two distinct printable 33-byte strings that gxhash cannot separate, under
/// any seed.
///
/// `compress_all` folds `a = s[1..17]` into `b = s[17..33]` as
/// `aesenc(a, 0) ^ b`. Changing `a[15]` moves that fold by a fixed `delta`,
/// which the same change to `b` cancels.
#[cfg(all(any(test, feature = "test-utils"), target_arch = "x86_64"))]
pub fn gxhash_twins() -> (String, String) {
    use std::arch::x86_64::*;
    const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789._-+";
    let fold = |x: u8| {
        let mut a = [0u8; 16];
        a[15] = x;
        let mut out = [0u8; 16];
        unsafe {
            let v = _mm_loadu_si128(a.as_ptr() as *const __m128i);
            let v = _mm_aesenc_si128(v, _mm_setzero_si128());
            _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, v);
        }
        out
    };
    for &x in CHARS {
        for &y in CHARS.iter().filter(|&&y| y != x) {
            let (fx, fy) = (fold(x), fold(y));
            let mut left = b"category-witness-0000.example.com".to_vec();
            left[16] = x;
            let mut right = left.clone();
            right[16] = y;
            let fits = (0..16).all(|i| {
                let delta = fx[i] ^ fy[i];
                if delta == 0 {
                    return true;
                }
                let Some(c) = CHARS.iter().find(|&&c| CHARS.contains(&(c ^ delta))) else {
                    return false;
                };
                left[17 + i] = *c;
                right[17 + i] = c ^ delta;
                true
            });
            if fits {
                return (
                    String::from_utf8(left).unwrap(),
                    String::from_utf8(right).unwrap(),
                );
            }
        }
    }
    unreachable!("no printable gxhash twins")
}

/// Collects the bytes `Hash` feeds a hasher, so xxh3 sees what gxhash would.
struct KeyBytes {
    inline: [u8; 64],
    len: usize,
    spill: Vec<u8>,
}

impl Default for KeyBytes {
    fn default() -> Self {
        Self { inline: [0u8; 64], len: 0, spill: Vec::new() }
    }
}

impl KeyBytes {
    fn as_slice(&self) -> &[u8] {
        if self.spill.is_empty() {
            &self.inline[..self.len]
        } else {
            &self.spill
        }
    }
}

impl StdHasher for KeyBytes {
    fn write(&mut self, bytes: &[u8]) {
        if !self.spill.is_empty() {
            self.spill.extend_from_slice(bytes);
        } else if self.len + bytes.len() <= self.inline.len() {
            self.inline[self.len..self.len + bytes.len()].copy_from_slice(bytes);
            self.len += bytes.len();
        } else {
            self.spill.reserve(self.len + bytes.len());
            self.spill.extend_from_slice(&self.inline[..self.len]);
            self.spill.extend_from_slice(bytes);
        }
    }

    fn finish(&self) -> u64 {
        unreachable!("KeyBytes only collects")
    }
}

impl Hasher<u64> for GxHash128 {
    type H = u128;
    #[inline(always)]
    fn hash(x: &u64, seed: u64, _fallback: bool) -> u128 {
        gxhash::gxhash128(to_bytes(x), seed as i64)
    }
}

impl Hasher<[u8]> for GxHash128 {
    type H = u128;
    #[inline(always)]
    fn hash(x: &[u8], seed: u64, _fallback: bool) -> u128 {
        gxhash::gxhash128(to_bytes(x), seed as i64)
    }
}

impl Hasher<Vec<u8>> for GxHash128 {
    type H = u128;
    #[inline(always)]
    fn hash(x: &Vec<u8>, seed: u64, _fallback: bool) -> u128 {
        gxhash::gxhash128(to_bytes(x.as_slice()), seed as i64)
    }
}

impl<const N: usize> Hasher<[u8; N]> for GxHash128 {
    type H = u128;
    #[inline(always)]
    fn hash(x: &[u8; N], seed: u64, _fallback: bool) -> u128 {
        gxhash::gxhash128(to_bytes(x), seed as i64)
    }
}

impl Hasher<&[u8]> for GxHash128 {
    type H = u128;
    #[inline(always)]
    fn hash(x: &&[u8], seed: u64, _fallback: bool) -> u128 {
        gxhash::gxhash128(to_bytes(*x), seed as i64)
    }
}

impl<const N: usize> Hasher<&[u8; N]> for GxHash128 {
    type H = u128;
    #[inline(always)]
    fn hash(x: &&[u8; N], seed: u64, _fallback: bool) -> u128 {
        gxhash::gxhash128(to_bytes(*x), seed as i64)
    }
}
