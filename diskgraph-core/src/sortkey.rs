use crate::metadata::{
    EdgeSortKeyTimestampKind, EdgeSortKeyWeightKind, EdgeSortKeyMetadata,
};
use half::f16;

/// Helper to decode the packed u64 sort key into its components (timestamp and weight).
pub struct SortKeyDecoder {
    pub timestamp_bits: u8,
    pub weight_bits: u8,
    pub timestamp_kind: Option<EdgeSortKeyTimestampKind>,
    pub weight_kind: Option<EdgeSortKeyWeightKind>,
    pub timestamp_anchor: i64,
    pub timestamp_step: i64,
}

impl SortKeyDecoder {
    pub fn new(meta: &EdgeSortKeyMetadata) -> Self {
        let timestamp_bits = meta.timestamp.as_ref().map(|ts| match ts.encoding.kind {
            EdgeSortKeyTimestampKind::Int8 => 8,
            EdgeSortKeyTimestampKind::Int16 => 16,
            EdgeSortKeyTimestampKind::Int32 => 32,
        }).unwrap_or(0);

        let weight_bits = meta.weight.as_ref().map(|w| match w.encoding.dtype {
            EdgeSortKeyWeightKind::Byte => 8,
            EdgeSortKeyWeightKind::UInt16 => 16,
            EdgeSortKeyWeightKind::UInt32 => 32,
            EdgeSortKeyWeightKind::Float16 => 16,
            EdgeSortKeyWeightKind::Float32 => 32,
        }).unwrap_or(0);

        let (ts_anchor, ts_step) = meta.timestamp.as_ref()
            .map(|ts| (ts.encoding.anchor, ts.encoding.step))
            .unwrap_or((0, 1));

        Self {
            timestamp_bits,
            weight_bits,
            timestamp_kind: meta.timestamp.as_ref().map(|ts| ts.encoding.kind.clone()),
            weight_kind: meta.weight.as_ref().map(|w| w.encoding.dtype.clone()),
            timestamp_anchor: ts_anchor,
            timestamp_step: ts_step,
        }
    }

    /// Unpack the timestamp and weight from the raw u64 sort key.
    /// Returns (timestamp_secs, weight_val).
    pub fn decode(&self, raw: u64) -> (Option<i64>, Option<f32>) {
        let weight_mask = if self.weight_bits > 0 {
            (1u64 << self.weight_bits) - 1
        } else {
            0
        };

        let weight_raw = if self.weight_bits > 0 { raw & weight_mask } else { 0 };
        let ts_raw = if self.weight_bits > 0 { raw >> self.weight_bits } else { raw };

        let timestamp = if self.timestamp_bits > 0 {
            let delta = decode_ordered_signed(ts_raw, self.timestamp_bits);
            // Check for "NaT" (min value)
            if delta == min_signed_for_bits(self.timestamp_bits) {
                None
            } else {
                let val = delta
                    .saturating_mul(self.timestamp_step)
                    .saturating_add(self.timestamp_anchor);
                Some(val)
            }
        } else {
            None
        };

        let weight = if self.weight_bits > 0 {
            Some(decode_weight_value(self.weight_kind.as_ref().unwrap(), weight_raw))
        } else {
            None
        };

        (timestamp, weight)
    }
}

fn decode_ordered_signed(raw: u64, bits: u8) -> i64 {
    let mask = 1u64 << (bits - 1);
    let decoded = raw ^ mask;
    match bits {
        8 => (decoded as u8 as i8) as i64,
        16 => (decoded as u16 as i16) as i64,
        32 => (decoded as u32 as i32) as i64,
        64 => decoded as i64,
        _ => decoded as i64,
    }
}

fn min_signed_for_bits(bits: u8) -> i64 {
    if bits == 0 || bits >= 63 {
        return i64::MIN;
    }
    -(1i64 << (bits - 1))
}

fn decode_weight_value(kind: &EdgeSortKeyWeightKind, raw: u64) -> f32 {
    match kind {
        EdgeSortKeyWeightKind::Byte => raw as f32,
        EdgeSortKeyWeightKind::UInt16 => raw as f32,
        EdgeSortKeyWeightKind::UInt32 => raw as f32,
        EdgeSortKeyWeightKind::Float16 => {
            ordered_f16_from_bits(raw as u16)
        },
        EdgeSortKeyWeightKind::Float32 => {
            ordered_f32_from_bits(raw as u32)
        }
    }
}

fn ordered_f32_from_bits(raw: u32) -> f32 {
    let bits = if raw & 0x8000_0000 != 0 {
        raw ^ 0x8000_0000
    } else {
        !raw
    };
    f32::from_bits(bits)
}

fn ordered_f16_from_bits(raw: u16) -> f32 {
    let bits = if raw & 0x8000 != 0 { raw ^ 0x8000 } else { !raw };
    f16::from_bits(bits).to_f32()
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
    fn test_signed_int_encoding() {
        let cases = vec![
            (0i64, 8), (-1, 8), (127, 8), (-128, 8),
            (0, 16), (-32768, 16), (32767, 16),
            (0, 32), (i32::MAX as i64, 32), (i32::MIN as i64, 32),
        ];

        for (val, bits) in cases {
            let encoded = ordered_signed_to_u64(val, bits);
            let decoded = decode_ordered_signed(encoded, bits);
            assert_eq!(val, decoded, "Failed for val={}, bits={}", val, bits);
        }
    }

    #[test]
    fn test_float_encoding() {
        let floats = vec![0.0f32, -1.0, 1.0, 123.456, -999.9, f32::MIN, f32::MAX];
        for f in floats {
            let encoded = ordered_f32_bits(f);
            let decoded = ordered_f32_from_bits(encoded);
            assert_eq!(f, decoded);
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
}
