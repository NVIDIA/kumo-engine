//! Bucket functions used to skew bucket size distributions.
use mem_dbg::MemSize;

use crate::util::mul_high;
use std::fmt::Debug;

pub trait BucketFn: Clone + Copy + Sync + Debug {
    const LINEAR: bool = false;
    const B_OUTPUT: bool = false;
    fn set_buckets_per_part(&mut self, _b: u64) {}
    fn call(&self, x: u64) -> u64;
}

/// `x * x * (1 + x)/2 * 255/256 + x/256`
#[derive(Clone, Copy, Debug, MemSize, Default)]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[cfg_attr(feature = "epserde", repr(C))]
#[cfg_attr(feature = "epserde", zero_copy)]
pub struct CubicEps;

impl BucketFn for CubicEps {
    fn call(&self, x: u64) -> u64 {
        mul_high(mul_high(x, x), (x >> 1) | (1 << 63)) / 256 * 255 + x / 256
    }
}
