// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Indexed lookup over descending keys, shared by sampling and temporal queries.
//!
//! Callers establish that keys are sorted and bounds use the same encoding or
//! a monotone decoded clock. This layer does not infer clock compatibility or
//! replace exact row filtering for quantized/clipped timestamps.

use std::ops::Range;

/// Locate `(lower, upper]` without materializing keys. Open endpoints include
/// the corresponding end of the index. Equal keys are never split. Empty or
/// reversed windows produce an empty range, including when bounds lie outside
/// the index. `key_at` must return keys in descending order.
#[inline]
pub fn descending_range_by<K: Ord + Copy>(
    len: usize,
    lower: Option<K>,
    upper: Option<K>,
    mut key_at: impl FnMut(usize) -> K,
) -> Range<usize> {
    let start = upper.map_or(0, |upper| partition_point_by(len, |i| key_at(i) > upper));
    let end = lower.map_or(len, |lower| {
        start + partition_point_by(len - start, |i| key_at(start + i) > lower)
    });
    start..end
}

/// First false position of a monotone true-prefix predicate. The endpoint
/// checks keep empty-prefix and full-range requests constant work.
#[inline]
pub fn partition_point_by(len: usize, mut pred: impl FnMut(usize) -> bool) -> usize {
    if len == 0 || !pred(0) {
        return 0;
    }
    if len == 1 || pred(len - 1) {
        return len;
    }
    let (mut lo, mut hi) = (1, len - 1);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if pred(mid) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_matches_scan_with_ties_and_open_or_reversed_bounds() {
        for keys in [vec![], vec![2], vec![3, 3, 2, 0, 0, -2]] {
            for lower in [None, Some(-3), Some(0), Some(2), Some(3), Some(4)] {
                for upper in [None, Some(-3), Some(0), Some(2), Some(3), Some(4)] {
                    let range = descending_range_by(keys.len(), lower, upper, |i| keys[i]);
                    let expected: Vec<_> = keys.iter().copied().filter(|&key| {
                        lower.is_none_or(|lo| key > lo) && upper.is_none_or(|hi| key <= hi)
                    }).collect();
                    assert_eq!(&keys[range], expected);
                }
            }
        }
    }

    #[test]
    fn selective_lookup_is_logarithmic_and_all_history_checks_endpoints() {
        let len = 1 << 20;
        let mut probes = 0;
        let range = descending_range_by(len, Some(0), Some(10_485), |i| {
            probes += 1;
            len - i
        });
        assert_eq!(range, len - 10_485..len);
        assert!(probes < 50, "{probes} probes");
        probes = 0;
        assert_eq!(descending_range_by(len, None, Some(len), |i| {
            probes += 1;
            len - i
        }), 0..len);
        assert_eq!(probes, 1);
    }
}
