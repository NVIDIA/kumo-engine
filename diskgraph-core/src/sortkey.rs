use crate::metadata::{EdgeSortKeyMetadata, EdgeSortKeyTimestampKind, EdgeSortKeyWeightKind};
use half::f16;

type WeightDecoderFn = fn(u64) -> f32;

/// Helper to decode the packed u64 sort key into its components (timestamp and weight).
/// Optimized for high-throughput loop execution by pre-computing bitmasks and shifts.
#[derive(Clone)] // Clone is cheap (just u64s and a fn ptr)
pub struct SortKeyDecoder {
    // Extraction
    weight_mask: u64,
    ts_shift: u8, // Equals weight_bits

    // Timestamp Decoding (Arithmetic)
    ts_sign_toggle_mask: u64, // 1 << (bits - 1)
    ts_extend_shift: u32,     // 64 - bits
    ts_min_value: i64,        // The specific "NaT" value for this bit width
    ts_step: i64,
    ts_anchor: i64,
    has_timestamp: bool,

    // Weight Decoding
    weight_decoder: WeightDecoderFn,
    has_weight: bool,
}

impl SortKeyDecoder {
    pub fn new(meta: &EdgeSortKeyMetadata) -> Self {
        let timestamp_bits = meta
            .timestamp
            .as_ref()
            .map(|ts| match ts.encoding.kind {
                EdgeSortKeyTimestampKind::Int8 => 8,
                EdgeSortKeyTimestampKind::Int16 => 16,
                EdgeSortKeyTimestampKind::Int32 => 32,
            })
            .unwrap_or(0);

        let weight_bits = meta
            .weight
            .as_ref()
            .map(|w| match w.encoding.dtype {
                EdgeSortKeyWeightKind::Byte => 8,
                EdgeSortKeyWeightKind::UInt16 => 16,
                EdgeSortKeyWeightKind::UInt32 => 32,
                EdgeSortKeyWeightKind::Float16 => 16,
                EdgeSortKeyWeightKind::Float32 => 32,
            })
            .unwrap_or(0);

        let (ts_anchor, ts_step) = meta
            .timestamp
            .as_ref()
            .map(|ts| (ts.encoding.anchor, ts.encoding.step))
            .unwrap_or((0, 1));

        let weight_mask = if weight_bits > 0 {
            (1u64 << weight_bits) - 1
        } else {
            0
        };

        let ts_sign_toggle_mask = if timestamp_bits > 0 {
            1u64 << (timestamp_bits - 1)
        } else {
            0
        };

        let ts_extend_shift = if timestamp_bits > 0 {
            64 - timestamp_bits as u32
        } else {
            0
        };

        let ts_min_value = if timestamp_bits > 0 && timestamp_bits < 64 {
            -(1i64 << (timestamp_bits - 1))
        } else {
            i64::MIN
        };

        let weight_decoder = if let Some(w) = &meta.weight {
            match w.encoding.dtype {
                EdgeSortKeyWeightKind::Byte => decode_weight_byte,
                EdgeSortKeyWeightKind::UInt16 => decode_weight_u16,
                EdgeSortKeyWeightKind::UInt32 => decode_weight_u32,
                EdgeSortKeyWeightKind::Float16 => decode_weight_f16,
                EdgeSortKeyWeightKind::Float32 => decode_weight_f32,
            }
        } else {
            decode_weight_noop
        };

        Self {
            weight_mask,
            ts_shift: weight_bits,
            ts_sign_toggle_mask,
            ts_extend_shift,
            ts_min_value,
            ts_step,
            ts_anchor,
            has_timestamp: timestamp_bits > 0,
            weight_decoder,
            has_weight: weight_bits > 0,
        }
    }

    /// Unpack the timestamp and weight from the raw u64 sort key.
    /// Returns (timestamp_secs, weight_val).
    #[inline(always)]
    pub fn decode(&self, raw: u64) -> (Option<i64>, Option<f32>) {
        let weight_raw = raw & self.weight_mask;
        let ts_raw = raw >> self.ts_shift;

        let timestamp = if self.has_timestamp {
            // Ordered Signed Decoding:
            // 1. Flip the sign bit back (encoding was val ^ sign_bit)
            let toggled = ts_raw ^ self.ts_sign_toggle_mask;

            // 2. Sign extend to 64-bit
            // (x << shift) as i64 >> shift
            let delta = (toggled << self.ts_extend_shift) as i64 >> self.ts_extend_shift;

            // 3. Check NaT
            if delta == self.ts_min_value {
                None
            } else {
                // 4. Apply scale/offset
                Some(
                    delta
                        .saturating_mul(self.ts_step)
                        .saturating_add(self.ts_anchor),
                )
            }
        } else {
            None
        };

        let weight = if self.has_weight {
            Some((self.weight_decoder)(weight_raw))
        } else {
            None
        };

        (timestamp, weight)
    }
}

// --- Weight Decoder Implementations ---

fn decode_weight_noop(_raw: u64) -> f32 {
    0.0
}

fn decode_weight_byte(raw: u64) -> f32 {
    raw as f32
}

fn decode_weight_u16(raw: u64) -> f32 {
    raw as f32
}

fn decode_weight_u32(raw: u64) -> f32 {
    raw as f32
}

fn decode_weight_f32(raw: u64) -> f32 {
    let bits = raw as u32;
    // Inverse of ordered_f32_bits:
    // if top bit is 1 (was positive): bits ^ 0x8000_0000
    // if top bit is 0 (was negative): !bits
    let decoded_bits = if bits & 0x8000_0000 != 0 {
        bits ^ 0x8000_0000
    } else {
        !bits
    };
    f32::from_bits(decoded_bits)
}

fn decode_weight_f16(raw: u64) -> f32 {
    let bits = raw as u16;
    let decoded_bits = if bits & 0x8000 != 0 {
        bits ^ 0x8000
    } else {
        !bits
    };
    f16::from_bits(decoded_bits).to_f32()
}

/// Encode a signed integer into an unsigned integer that preserves sort order.
///
/// This is the encoding half of the sign-bit-flip trick:
/// `value ^ (1 << (bits - 1))` maps signed i(bits) → u(bits) such that
/// smaller signed values map to smaller unsigned values.
///
/// Used by `SortKeyEncoder` and by tests for round-trip validation.
pub fn ordered_signed_to_u64(value: i64, bits: u8) -> u64 {
    let mask = 1u64 << (bits - 1);
    let raw = match bits {
        8 => (value as i8 as u8) as u64,
        16 => (value as i16 as u16) as u64,
        32 => (value as i32 as u32) as u64,
        _ => value as u64,
    };
    raw ^ mask
}

/// Encoder for sort keys: converts a timestamp (i64 seconds) into a packed u64
/// sort key suitable for binary search on the edge index.
///
/// Counterpart to `SortKeyDecoder`. Constructed from the same `EdgeSortKeyMetadata`.
#[derive(Clone)]
pub struct SortKeyEncoder {
    ts_anchor: i64,
    ts_step: i64,
    timestamp_bits: u8,
    weight_bits: u8,
    ts_sign_toggle_mask: u64,
    has_timestamp: bool,
}

impl SortKeyEncoder {
    pub fn new(meta: &EdgeSortKeyMetadata) -> Self {
        let timestamp_bits = meta
            .timestamp
            .as_ref()
            .map(|ts| match ts.encoding.kind {
                EdgeSortKeyTimestampKind::Int8 => 8,
                EdgeSortKeyTimestampKind::Int16 => 16,
                EdgeSortKeyTimestampKind::Int32 => 32,
            })
            .unwrap_or(0);

        let weight_bits = meta
            .weight
            .as_ref()
            .map(|w| match w.encoding.dtype {
                EdgeSortKeyWeightKind::Byte => 8,
                EdgeSortKeyWeightKind::UInt16 => 16,
                EdgeSortKeyWeightKind::UInt32 => 32,
                EdgeSortKeyWeightKind::Float16 => 16,
                EdgeSortKeyWeightKind::Float32 => 32,
            })
            .unwrap_or(0);

        let (ts_anchor, ts_step) = meta
            .timestamp
            .as_ref()
            .map(|ts| (ts.encoding.anchor, ts.encoding.step))
            .unwrap_or((0, 1));

        let ts_sign_toggle_mask = if timestamp_bits > 0 {
            1u64 << (timestamp_bits - 1)
        } else {
            0
        };

        Self {
            ts_anchor,
            ts_step,
            timestamp_bits,
            weight_bits,
            ts_sign_toggle_mask,
            has_timestamp: timestamp_bits > 0,
        }
    }

    /// Returns true if this relation has a timestamp component in the sort key.
    #[inline]
    pub fn has_timestamp(&self) -> bool {
        self.has_timestamp
    }

    /// Encode a timestamp as a lower-bound sort key (weight bits = 0).
    /// Suitable for binary search: all edges at this timestamp have sort_key >= result.
    #[inline]
    pub fn encode_lower(&self, timestamp_secs: i64) -> u64 {
        if !self.has_timestamp {
            return 0;
        }
        let delta = timestamp_secs.saturating_sub(self.ts_anchor).div_euclid(self.ts_step);
        // Truncate to timestamp_bits width, then flip sign bit for unsigned ordering
        let truncated = match self.timestamp_bits {
            8 => (delta as i8 as u8) as u64,
            16 => (delta as i16 as u16) as u64,
            32 => (delta as i32 as u32) as u64,
            _ => delta as u64,
        };
        let raw = truncated ^ self.ts_sign_toggle_mask;
        raw << self.weight_bits
    }

    /// Encode as upper-bound sort key (all weight bits set).
    /// All edges at this timestamp have sort_key <= result.
    #[inline]
    pub fn encode_upper(&self, timestamp_secs: i64) -> u64 {
        let lower = self.encode_lower(timestamp_secs);
        if self.weight_bits > 0 {
            lower | ((1u64 << self.weight_bits) - 1)
        } else {
            lower
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::*;

    fn ordered_f32_bits(value: f32) -> u32 {
        let bits = value.to_bits();
        if bits & 0x8000_0000 != 0 {
            !bits
        } else {
            bits ^ 0x8000_0000
        }
    }

    #[test]
    fn test_sort_key_decoder() {
        // Setup: 32-bit TS + 32-bit Float Weight
        let meta = EdgeSortKeyMetadata {
            width_bits: 64,
            timestamp: Some(EdgeSortKeyTimestampMetadata {
                table: "t".into(),
                column: "c".into(),
                encoding: EdgeSortKeyTimestampEncoding {
                    kind: EdgeSortKeyTimestampKind::Int32,
                    anchor: 1000,
                    step: 10,
                },
            }),
            weight: Some(EdgeSortKeyWeightMetadata {
                table: "t".into(),
                expr: "e".into(),
                encoding: EdgeSortKeyWeightEncoding {
                    dtype: EdgeSortKeyWeightKind::Float32,
                },
            }),
        };

        let decoder = SortKeyDecoder::new(&meta);

        let target_ts = 1500i64; // delta = (1500 - 1000) / 10 = 50
        let target_w = 0.5f32;

        let delta = 50i64;
        let ts_enc = ordered_signed_to_u64(delta, 32);
        let w_enc = ordered_f32_bits(target_w) as u64;

        let packed = (ts_enc << 32) | w_enc;

        let (ts, w) = decoder.decode(packed);

        assert_eq!(ts, Some(target_ts));
        assert_eq!(w, Some(target_w));
    }

    #[test]
    fn test_decoder_negative_values() {
        // Setup: 16-bit TS + 16-bit Float Weight
        let meta = EdgeSortKeyMetadata {
            width_bits: 32,
            timestamp: Some(EdgeSortKeyTimestampMetadata {
                table: "t".into(),
                column: "c".into(),
                encoding: EdgeSortKeyTimestampEncoding {
                    kind: EdgeSortKeyTimestampKind::Int16,
                    anchor: 0,
                    step: 1,
                },
            }),
            weight: None, // No weight
        };
        let decoder = SortKeyDecoder::new(&meta);

        let target_ts = -12345i64;
        let ts_enc = ordered_signed_to_u64(target_ts, 16);
        // packed is just ts_enc because no weight bits

        let (ts, _) = decoder.decode(ts_enc);
        assert_eq!(ts, Some(target_ts));
    }

    #[test]
    fn test_decoder_nat() {
        let meta = EdgeSortKeyMetadata {
            width_bits: 32,
            timestamp: Some(EdgeSortKeyTimestampMetadata {
                table: "t".into(),
                column: "c".into(),
                encoding: EdgeSortKeyTimestampEncoding {
                    kind: EdgeSortKeyTimestampKind::Int16,
                    anchor: 0,
                    step: 1,
                },
            }),
            weight: None,
        };
        let decoder = SortKeyDecoder::new(&meta);

        let nat_val = i16::MIN as i64; // -32768
        let enc = ordered_signed_to_u64(nat_val, 16);
        let (ts, _) = decoder.decode(enc);
        assert_eq!(ts, None); // Should decode to None
    }

    #[test]
    fn test_weight_only() {
        let meta = EdgeSortKeyMetadata {
            width_bits: 32,
            timestamp: None,
            weight: Some(EdgeSortKeyWeightMetadata {
                table: "t".into(),
                expr: "e".into(),
                encoding: EdgeSortKeyWeightEncoding {
                    dtype: EdgeSortKeyWeightKind::Float32,
                },
            }),
        };
        let decoder = SortKeyDecoder::new(&meta);

        let target_w = -10.5f32;
        let enc = ordered_f32_bits(target_w) as u64;
        let (ts, w) = decoder.decode(enc);

        assert_eq!(ts, None);
        assert_eq!(w, Some(target_w));
    }

    #[test]
    fn test_no_components() {
        let meta = EdgeSortKeyMetadata {
            width_bits: 0,
            timestamp: None,
            weight: None,
        };
        let decoder = SortKeyDecoder::new(&meta);
        let (ts, w) = decoder.decode(12345); // Raw value shouldn't matter
        assert_eq!(ts, None);
        assert_eq!(w, None);
    }

    #[test]
    fn test_timestamp_int32_boundary() {
        // 32-bit TS (Max supported), No Weight
        let meta = EdgeSortKeyMetadata {
            width_bits: 32,
            timestamp: Some(EdgeSortKeyTimestampMetadata {
                table: "t".into(),
                column: "c".into(),
                encoding: EdgeSortKeyTimestampEncoding {
                    kind: EdgeSortKeyTimestampKind::Int32,
                    anchor: 0,
                    step: 1,
                },
            }),
            weight: None,
        };

        let decoder = SortKeyDecoder::new(&meta);
        let val = i32::MIN as i64 + 1; // Near min
        let enc = ordered_signed_to_u64(val, 32);
        let (ts, _) = decoder.decode(enc);
        assert_eq!(ts, Some(val));
    }

    #[test]
    fn test_weight_types_integration() {
        // Test UInt16 Weight
        let meta = EdgeSortKeyMetadata {
            width_bits: 16,
            timestamp: None,
            weight: Some(EdgeSortKeyWeightMetadata {
                table: "t".into(),
                expr: "e".into(),
                encoding: EdgeSortKeyWeightEncoding {
                    dtype: EdgeSortKeyWeightKind::UInt16,
                },
            }),
        };
        let decoder = SortKeyDecoder::new(&meta);
        let val = 60000u16;
        let (ts, w) = decoder.decode(val as u64);
        assert_eq!(ts, None);
        assert_eq!(w, Some(60000.0));

        // Test Float16 Weight
        let meta_f16 = EdgeSortKeyMetadata {
            width_bits: 16,
            timestamp: None,
            weight: Some(EdgeSortKeyWeightMetadata {
                table: "t".into(),
                expr: "e".into(),
                encoding: EdgeSortKeyWeightEncoding {
                    dtype: EdgeSortKeyWeightKind::Float16,
                },
            }),
        };
        let decoder_f16 = SortKeyDecoder::new(&meta_f16);
        let val_f16 = 1.5f32;
        let val_f16_half = f16::from_f32(val_f16);

        // Manual Ordered Encoding for F16
        let bits = val_f16_half.to_bits();
        let enc = if bits & 0x8000 != 0 {
            !bits
        } else {
            bits ^ 0x8000
        };

        let (_, w) = decoder_f16.decode(enc as u64);
        assert_eq!(w, Some(val_f16));
    }

    #[test]
    fn test_timestamp_overflow() {
        // Test saturating arithmetic
        let meta = EdgeSortKeyMetadata {
            width_bits: 32,
            timestamp: Some(EdgeSortKeyTimestampMetadata {
                table: "t".into(),
                column: "c".into(),
                encoding: EdgeSortKeyTimestampEncoding {
                    kind: EdgeSortKeyTimestampKind::Int32,
                    anchor: i64::MAX - 10,
                    step: 100,
                },
            }),
            weight: None,
        };
        let decoder = SortKeyDecoder::new(&meta);

        // Delta = 1. Result = Anchor + 100 -> Overflow
        let enc = ordered_signed_to_u64(1, 32);
        let (ts, _) = decoder.decode(enc);
        assert_eq!(ts, Some(i64::MAX)); // Saturated

        // Delta = -1. Result = Anchor - 100.
        let enc_neg = ordered_signed_to_u64(-1, 32);
        let (ts_neg, _) = decoder.decode(enc_neg);
        assert_eq!(ts_neg, Some(i64::MAX - 10 - 100)); // Correct
    }

    #[test]
    fn test_tight_packing() {
        // 8-bit TS + 8-bit Weight
        let meta = EdgeSortKeyMetadata {
            width_bits: 16,
            timestamp: Some(EdgeSortKeyTimestampMetadata {
                table: "t".into(),
                column: "c".into(),
                encoding: EdgeSortKeyTimestampEncoding {
                    kind: EdgeSortKeyTimestampKind::Int8,
                    anchor: 0,
                    step: 1,
                },
            }),
            weight: Some(EdgeSortKeyWeightMetadata {
                table: "t".into(),
                expr: "e".into(),
                encoding: EdgeSortKeyWeightEncoding {
                    dtype: EdgeSortKeyWeightKind::Byte,
                },
            }),
        };
        let decoder = SortKeyDecoder::new(&meta);

        // TS = -5 (encoded 8-bit), Weight = 255 (encoded 8-bit)
        let ts_delta = -5i64;
        let w_val = 255u64;

        let ts_enc = ordered_signed_to_u64(ts_delta, 8); // 8 bits
        let w_enc = w_val; // 8 bits

        // Pack: TS in upper 8, Weight in lower 8
        // Total 16 bits used, but usually packed in u64 at bottom?
        // Logic: val >> weight_bits.
        // So TS is at bits [8..16].

        let packed = (ts_enc << 8) | w_enc;

        let (ts, w) = decoder.decode(packed);

        assert_eq!(ts, Some(-5));
        assert_eq!(w, Some(255.0));
    }

    // --- SortKeyEncoder tests ---

    #[test]
    fn test_encoder_round_trip_32bit() {
        // 32-bit TS, no weight
        let meta = EdgeSortKeyMetadata {
            width_bits: 32,
            timestamp: Some(EdgeSortKeyTimestampMetadata {
                table: "t".into(),
                column: "c".into(),
                encoding: EdgeSortKeyTimestampEncoding {
                    kind: EdgeSortKeyTimestampKind::Int32,
                    anchor: 1000,
                    step: 10,
                },
            }),
            weight: None,
        };
        let encoder = SortKeyEncoder::new(&meta);
        let decoder = SortKeyDecoder::new(&meta);

        // Round trip: encode_lower → decode should give back the same timestamp
        // (within quantization: timestamp is quantized to step)
        for ts in [1000, 1500, 2000, 500, 0, -1000] {
            let encoded = encoder.encode_lower(ts);
            let (decoded_ts, _) = decoder.decode(encoded);
            let expected = ((ts - 1000) / 10) * 10 + 1000; // quantized
            assert_eq!(decoded_ts, Some(expected), "round trip failed for ts={}", ts);
        }
    }

    #[test]
    fn test_encoder_round_trip_with_weight() {
        // 32-bit TS + 32-bit weight
        let meta = EdgeSortKeyMetadata {
            width_bits: 64,
            timestamp: Some(EdgeSortKeyTimestampMetadata {
                table: "t".into(),
                column: "c".into(),
                encoding: EdgeSortKeyTimestampEncoding {
                    kind: EdgeSortKeyTimestampKind::Int32,
                    anchor: 0,
                    step: 1,
                },
            }),
            weight: Some(EdgeSortKeyWeightMetadata {
                table: "t".into(),
                expr: "e".into(),
                encoding: EdgeSortKeyWeightEncoding {
                    dtype: EdgeSortKeyWeightKind::Float32,
                },
            }),
        };
        let encoder = SortKeyEncoder::new(&meta);
        let decoder = SortKeyDecoder::new(&meta);

        // encode_lower should have weight bits = 0
        let lo = encoder.encode_lower(100);
        let (ts_lo, _) = decoder.decode(lo);
        assert_eq!(ts_lo, Some(100));

        // encode_upper should have all weight bits set
        let hi = encoder.encode_upper(100);
        let (ts_hi, _) = decoder.decode(hi);
        assert_eq!(ts_hi, Some(100));
        // upper should be > lower (for binary search range to work)
        assert!(hi > lo, "encode_upper should be > encode_lower");
    }

    #[test]
    fn test_encoder_ordering() {
        // Encoded sort keys should preserve timestamp ordering
        let meta = EdgeSortKeyMetadata {
            width_bits: 32,
            timestamp: Some(EdgeSortKeyTimestampMetadata {
                table: "t".into(),
                column: "c".into(),
                encoding: EdgeSortKeyTimestampEncoding {
                    kind: EdgeSortKeyTimestampKind::Int32,
                    anchor: 0,
                    step: 1,
                },
            }),
            weight: None,
        };
        let encoder = SortKeyEncoder::new(&meta);

        let timestamps: Vec<i64> = vec![-1000, -100, -1, 0, 1, 100, 1000, 10000];
        let encoded: Vec<u64> = timestamps.iter().map(|&ts| encoder.encode_lower(ts)).collect();

        for i in 1..encoded.len() {
            assert!(
                encoded[i] > encoded[i - 1],
                "ordering violated: ts[{}]={} (enc={}) should be > ts[{}]={} (enc={})",
                i, timestamps[i], encoded[i],
                i - 1, timestamps[i - 1], encoded[i - 1]
            );
        }
    }

    #[test]
    fn test_encoder_no_timestamp() {
        let meta = EdgeSortKeyMetadata {
            width_bits: 0,
            timestamp: None,
            weight: None,
        };
        let encoder = SortKeyEncoder::new(&meta);
        assert!(!encoder.has_timestamp());
        assert_eq!(encoder.encode_lower(12345), 0);
        assert_eq!(encoder.encode_upper(12345), 0);
    }

    #[test]
    fn test_encoder_binary_search_window() {
        // Simulate binary search on a sorted edge list
        let meta = EdgeSortKeyMetadata {
            width_bits: 16,
            timestamp: Some(EdgeSortKeyTimestampMetadata {
                table: "t".into(),
                column: "c".into(),
                encoding: EdgeSortKeyTimestampEncoding {
                    kind: EdgeSortKeyTimestampKind::Int16,
                    anchor: 0,
                    step: 1,
                },
            }),
            weight: None,
        };
        let encoder = SortKeyEncoder::new(&meta);

        // Build sorted edge list with timestamps -100..100
        let edge_keys: Vec<u64> = (-100..100i64)
            .map(|ts| ordered_signed_to_u64(ts, 16))
            .collect();

        // Find edges in window [10, 50)
        let lo = encoder.encode_lower(10);
        let hi = encoder.encode_lower(50);

        let start = edge_keys.partition_point(|&k| k < lo);
        let end = edge_keys.partition_point(|&k| k < hi);

        // Should match timestamps 10..50 (40 edges)
        assert_eq!(end - start, 40, "expected 40 edges in window [10,50), got {}", end - start);
    }

    #[test]
    fn test_encoder_div_euclid_negative_delta() {
        // Ingestion uses div_euclid, encoder must match.
        // div_euclid floors toward -inf; / truncates toward zero.
        // They differ for negative dividends with step > 1:
        //   (-1000).div_euclid(86400) = -1   (ingestion: bucket -1)
        //   (-1000) / 86400            =  0   (old encoder: bucket 0) — WRONG
        let anchor = 1_704_067_200i64; // 2024-01-01
        let step = 86400i64; // 1 day

        let meta = EdgeSortKeyMetadata {
            width_bits: 32,
            timestamp: Some(EdgeSortKeyTimestampMetadata {
                table: "t".into(),
                column: "c".into(),
                encoding: EdgeSortKeyTimestampEncoding {
                    kind: EdgeSortKeyTimestampKind::Int32,
                    anchor,
                    step,
                },
            }),
            weight: None,
        };
        let encoder = SortKeyEncoder::new(&meta);
        let decoder = SortKeyDecoder::new(&meta);

        // Timestamp 1000 seconds before anchor (Dec 31 2023 23:43:20)
        let ts = anchor - 1000;

        // Ingestion puts this in bucket -1: div_euclid(-1000, 86400) = -1
        // Decoded timestamp for bucket -1 = anchor + (-1 * step) = anchor - 86400
        let encoded = encoder.encode_lower(ts);
        let (decoded, _) = decoder.decode(encoded);
        assert_eq!(
            decoded,
            Some(anchor - step),
            "encoder should use div_euclid (bucket -1), not / (bucket 0)"
        );

        // Also verify the positive case still works
        let ts_pos = anchor + 1000;
        let encoded_pos = encoder.encode_lower(ts_pos);
        let (decoded_pos, _) = decoder.decode(encoded_pos);
        // div_euclid(1000, 86400) = 0, same as 1000/86400 = 0
        assert_eq!(decoded_pos, Some(anchor));
    }

    #[test]
    fn test_window_upper_bound_with_weight_bits() {
        // Window [lo, hi) should be exclusive on hi.
        // With weight bits, encode_upper(hi) sets weight bits to max,
        // which makes edges AT timestamp hi pass the `sort_key < hi_key` check.
        // The correct upper bound is encode_lower(hi) (weight bits = 0).
        let meta = EdgeSortKeyMetadata {
            width_bits: 24, // 16-bit TS + 8-bit weight
            timestamp: Some(EdgeSortKeyTimestampMetadata {
                table: "t".into(),
                column: "c".into(),
                encoding: EdgeSortKeyTimestampEncoding {
                    kind: EdgeSortKeyTimestampKind::Int16,
                    anchor: 0,
                    step: 1,
                },
            }),
            weight: Some(EdgeSortKeyWeightMetadata {
                table: "t".into(),
                expr: "e".into(),
                encoding: EdgeSortKeyWeightEncoding {
                    dtype: EdgeSortKeyWeightKind::Byte,
                },
            }),
        };
        let encoder = SortKeyEncoder::new(&meta);

        // Build edges at timestamps 10, 20, 30, each with weight=5
        let edge_keys: Vec<u64> = [10, 20, 30]
            .iter()
            .map(|&ts| {
                let ts_enc = ordered_signed_to_u64(ts, 16);
                (ts_enc << 8) | 5 // weight = 5
            })
            .collect();

        // Window [10, 30) should include ts=10, ts=20 but NOT ts=30
        let lo = encoder.encode_lower(10);
        let hi_correct = encoder.encode_lower(30); // correct: weight bits = 0
        let hi_wrong = encoder.encode_upper(30); // wrong: weight bits = 0xFF

        let start = edge_keys.partition_point(|&k| k < lo);
        let end_correct = edge_keys.partition_point(|&k| k < hi_correct);
        let end_wrong = edge_keys.partition_point(|&k| k < hi_wrong);

        // Correct: [10, 20] = 2 edges
        assert_eq!(
            end_correct - start,
            2,
            "encode_lower(30) should exclude ts=30"
        );
        // Wrong: [10, 20, 30] = 3 edges (leaks boundary)
        assert_eq!(
            end_wrong - start,
            3,
            "encode_upper(30) incorrectly includes ts=30"
        );
    }
}
