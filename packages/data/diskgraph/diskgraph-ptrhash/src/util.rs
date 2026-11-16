//! Internal utilities for ptrhash implementation.

pub(crate) fn mul_high(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) >> 64) as u64
}
