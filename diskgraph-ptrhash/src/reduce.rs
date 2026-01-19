use mem_dbg::MemSize;

use crate::util::mul_high;

pub trait Reduce: Copy + Sync + std::fmt::Debug {
    /// Reduce into the range [0, d).
    fn new(d: usize) -> Self;
    /// Reduce a (uniform random 64 bit) number into the range [0, d).
    fn reduce(self, h: u64) -> usize;
    /// Reduce a (uniform random 64 bit) number into the range [0, d),
    /// and also return a remainder that can be used for further reductions.
    fn reduce_with_remainder(self, _h: u64) -> (usize, u64) {
        unimplemented!();
    }
}

/// FastReduce64
/// Taken from https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
/// NOTE: This only uses the lg(n) high-order bits of entropy from the hash.
#[derive(Copy, Clone, Debug, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[cfg_attr(feature = "epserde", repr(C))]
#[cfg_attr(feature = "epserde", zero_copy)]
pub struct FastReduce {
    d: u64,
}
impl Reduce for FastReduce {
    fn new(d: usize) -> Self {
        Self { d: d as u64 }
    }
    fn reduce(self, h: u64) -> usize {
        mul_high(self.d, h) as usize
    }
    fn reduce_with_remainder(self, h: u64) -> (usize, u64) {
        let r = self.d as u128 * h as u128;
        ((r >> 64) as usize, r as u64)
    }
}

/// FastMod32, using the low 32 bits of the hash.
/// Taken from https://github.com/lemire/fastmod/blob/master/include/fastmod.h
#[derive(Copy, Clone, Debug, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[cfg_attr(feature = "epserde", deep_copy)]
pub struct FM32 {
    d: u64,
    m: u64,
}
impl Reduce for FM32 {
    fn new(d: usize) -> Self {
        assert!(d <= u32::MAX as usize);
        let d = d as u64;
        let d = d.max(1);
        Self {
            d,
            m: u64::MAX / d + 1,
        }
    }
    fn reduce(self, h: u64) -> usize {
        let lowbits = self.m.wrapping_mul(h);
        ((lowbits as u128 * self.d as u128) >> 64) as usize
    }
}
