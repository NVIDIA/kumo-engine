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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::*;

    // Ported from diskgraph-ingest logic to ensure consistency
    fn ordered_signed_to_u64(value: i64, bits: u8) -> u64 {
        let mask = 1u64 << (bits - 1);
        let raw = match bits {
            8 => (value as i8 as u8) as u64,
            16 => (value as i16 as u16) as u64,
            32 => (value as i32 as u32) as u64,
            _ => value as u64,
        };
        raw ^ mask
    }

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
}
