// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conservative indexed-clock admission, independent of traversal and projection.
//!
//! A rejection guarantees that the stored bucket cannot contain a row in the
//! requested `(lower, upper]` window. Admission only produces candidates: callers
//! must preserve exact row-clock checks, ordering, duplicates, and null behavior.
//! It cannot justify truncating a sequence or an aggregate's input.
//!
//! This is NOT a binary-search predicate. Clipped/sentinel codes at both ends
//! remain candidates, so admission need not form one contiguous range. A future
//! range seeker must handle those regions explicitly or use its scan fallback.

use crate::metadata::{EdgeSortKeyMetadata, EdgeSortKeyTimestampKind};
use crate::sortkey::SortKeyDecoder;
use crate::time::TimeUnit;

/// A non-sentinel key's bucket end; exact only at one-second resolution.
/// Floor/NULL and ceiling codes may represent clipped timestamps and return None.
#[inline]
pub fn unclipped_bucket_end(decoder: &SortKeyDecoder, key: u64) -> Option<i64> {
    let timestamp = decoder.decode(key).0?;
    (Some(timestamp) != decoder.decode(u64::MAX).0).then_some(timestamp)
}

/// Compatible indexed clock for conservative candidate admission.
///
/// Bounds and decoded row clocks use integral epoch seconds. For step S, an
/// upward-rounded key ending at T represents integer times [T-S+1, T].
/// Timestamp bits precede weight bits; weights never affect window eligibility.
pub struct IndexedTimeFilter {
    decoder: SortKeyDecoder,
    step: i64,
}

impl IndexedTimeFilter {
    /// None requires the caller's exact row-clock fallback. Missing, differently
    /// scaled, or mismatched clocks, invalid steps, and unsafe decoding arithmetic cannot
    /// be used to reject candidates. This also validates one-second encodings.
    pub fn matching(
        metadata: Option<&EdgeSortKeyMetadata>,
        destination: &str,
        clock: &str,
        index_clock: TimeUnit,
    ) -> Option<Self> {
        // Query bounds are seconds. Other index clocks require row decoding;
        // a matching column name or step of one does not establish equal units.
        if index_clock != TimeUnit::Seconds {
            return None;
        }
        let metadata = metadata?;
        let timestamp = metadata.timestamp.as_ref()?;
        let encoding = &timestamp.encoding;
        if timestamp.table != destination || timestamp.column != clock || encoding.step < 1 {
            return None;
        }
        let bits = match encoding.kind {
            EdgeSortKeyTimestampKind::Int8 => 8,
            EdgeSortKeyTimestampKind::Int16 => 16,
            EdgeSortKeyTimestampKind::Int32 => 32,
            // Full-width keys are currently emitted for subsecond index clocks.
            // Keep unsupported encodings on the exact row-clock fallback.
            EdgeSortKeyTimestampKind::Int64 => return None,
        };
        // Decoder arithmetic multiplies before adding. Validate both operations
        // across the encoded range, including extremes retained as sentinels.
        for delta in [-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1] {
            delta
                .checked_mul(encoding.step)?
                .checked_add(encoding.anchor)?;
        }
        Some(Self {
            decoder: SortKeyDecoder::new(metadata),
            step: encoding.step,
        })
    }

    #[inline]
    pub fn is_exact(&self) -> bool {
        self.step == 1
    }

    #[inline]
    pub fn decoder(&self) -> &SortKeyDecoder {
        &self.decoder
    }

    /// False proves the bucket is outside `(lower, upper]`; true requires exact
    /// refinement. None denotes an open endpoint. Sentinel codes always survive
    /// so callers retain clipped rows and their existing null-clock errors.
    #[inline]
    pub fn may_overlap(&self, key: u64, lower: Option<i64>, upper: Option<i64>) -> bool {
        unclipped_bucket_end(&self.decoder, key).is_none_or(|end| {
            let first = i128::from(end) - i128::from(self.step) + 1;
            lower.is_none_or(|bound| end > bound)
                && upper.is_none_or(|bound| first <= i128::from(bound))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coarse_metadata(step: i64, anchor: i64) -> EdgeSortKeyMetadata {
        use crate::metadata::{EdgeSortKeyTimestampEncoding, EdgeSortKeyTimestampMetadata};
        EdgeSortKeyMetadata {
            width_bits: 8,
            timestamp: Some(EdgeSortKeyTimestampMetadata {
                table: "events".into(),
                column: "time".into(),
                encoding: EdgeSortKeyTimestampEncoding {
                    kind: EdgeSortKeyTimestampKind::Int8,
                    anchor,
                    step,
                },
            }),
            weight: None,
        }
    }

    #[test]
    fn coarse_buckets_never_drop_an_in_window_time() {
        for step in [1, 60, 3600, 86400] {
            for anchor in [-1_700_000_003, 0, 1_700_000_003] {
                let meta = coarse_metadata(step, anchor);
                let clock =
                    IndexedTimeFilter::matching(Some(&meta), "events", "time", TimeUnit::Seconds)
                        .unwrap();
                // Include both clipped endpoints, negative deltas, and every
                // exact/before/after bucket boundary in the interior.
                for bucket in -130i64..=130 {
                    for remainder in [-1, 0, 1, step - 1] {
                        let time = anchor + bucket * step + remainder;
                        let diff = time.saturating_sub(anchor);
                        let delta = diff.div_euclid(step) + i64::from(diff.rem_euclid(step) > 0);
                        let key = ((delta.clamp(-128, 127) as i8 as u8) ^ 128) as u64;
                        for width in [1, step, step + 1] {
                            assert!(clock.may_overlap(key, Some(time - width), Some(time)));
                            assert!(clock.may_overlap(key, None, Some(time)));
                            assert!(clock.may_overlap(key, Some(time - 1), None));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn coarse_window_endpoints_and_ambiguous_codes() {
        let meta = coarse_metadata(60, 0);
        let clock = IndexedTimeFilter::matching(Some(&meta), "events", "time", TimeUnit::Seconds)
            .unwrap();
        let key = 129; // Bucket (0, 60], or integer times [1, 60].
        assert!(!clock.may_overlap(key, Some(60), None));
        assert!(!clock.may_overlap(key, None, Some(0)));
        assert!(clock.may_overlap(key, Some(59), Some(60)));
        assert!(clock.may_overlap(key, Some(0), Some(1)));
        for key in [0, 255] {
            assert!(clock.may_overlap(key, Some(100_000), Some(100_001)));
            assert!(clock.may_overlap(key, Some(-100_001), Some(-100_000)));
        }
    }

    #[test]
    fn coarse_clock_mismatch_and_overflow_fall_back() {
        for (step, anchor) in [
            (0, 0),
            (-1, 0),
            (i64::MAX, 0),
            (60, i64::MAX),
            (60, i64::MIN),
            (1, i64::MAX),
            (1, i64::MIN),
        ] {
            let meta = coarse_metadata(step, anchor);
            assert!(
                IndexedTimeFilter::matching(Some(&meta), "events", "time", TimeUnit::Seconds).is_none()
            );
        }
        let meta = coarse_metadata(60, 0);
        assert!(
            IndexedTimeFilter::matching(None, "events", "time", TimeUnit::Seconds).is_none()
        );
        let mut missing = meta.clone();
        missing.timestamp = None;
        assert!(
            IndexedTimeFilter::matching(Some(&missing), "events", "time", TimeUnit::Seconds).is_none()
        );
        assert!(
            IndexedTimeFilter::matching(Some(&meta), "other", "time", TimeUnit::Seconds).is_none()
        );
        assert!(
            IndexedTimeFilter::matching(Some(&meta), "events", "other", TimeUnit::Seconds).is_none()
        );
        assert!(!IndexedTimeFilter::matching(Some(&meta), "events", "time", TimeUnit::Seconds)
            .unwrap()
            .is_exact());
    }

    #[test]
    fn differently_scaled_and_full_width_clocks_use_row_fallback() {
        let mut meta = coarse_metadata(1, 0);
        for unit in TimeUnit::INDEX_CLOCKS {
            assert_eq!(
                IndexedTimeFilter::matching(Some(&meta), "events", "time", unit).is_some(),
                unit == TimeUnit::Seconds,
            );
        }
        meta.width_bits = 64;
        meta.timestamp.as_mut().unwrap().encoding.kind = EdgeSortKeyTimestampKind::Int64;
        assert!(
            IndexedTimeFilter::matching(Some(&meta), "events", "time", TimeUnit::Seconds).is_none()
        );
    }

    #[test]
    fn admitted_sentinels_do_not_form_a_binary_search_range() {
        let meta = coarse_metadata(60, 0);
        let clock = IndexedTimeFilter::matching(Some(&meta), "events", "time", TimeUnit::Seconds)
            .unwrap();
        // Descending encoded time: ceiling, newer interior, matching bucket,
        // older interior, floor/NULL. A range seeker must retain the endpoints
        // separately or fall back; partition_point(may_overlap) is invalid.
        let admitted = [255, 130, 129, 128, 0]
            .map(|key| clock.may_overlap(key, Some(0), Some(1)));
        assert_eq!(admitted, [true, false, true, false, true]);
    }
}
