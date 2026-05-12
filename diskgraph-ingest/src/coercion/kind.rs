//! Coercion kind determination for FK-PK type pairs.
//!
//! This module defines the `CoercionKind` enum and the `coercion_kind` function
//! that determines what kind of coercion is needed between two IngestedTypes.
//!
//! The 23×23 matrix is the single source of truth for FK-PK type compatibility.
//! Some coercions (Int/Float → Decimal) are scale-dependent.

use crate::graph_config::IngestedType;

/// What kind of coercion is needed between FK and PK types?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoercionKind {
    /// Same type, no coercion needed.
    Same,
    /// Always succeeds (e.g., Int32 → Int64 widening).
    Safe,
    /// May fail per-row (e.g., String → Int64 requires parsing).
    Conditional,
    /// Would lose information (e.g., Float64 → Int64 truncates decimals).
    Lossy,
    /// No conversion path exists (e.g., Bool → Uuid).
    Incompatible,
}

/// Parameters for scale-dependent coercion decisions.
///
/// Some coercions (Int/Float → Decimal) depend on the target's scale/precision.
/// Pass this struct to `coercion_kind` when the PK is a Decimal type.
#[derive(Debug, Clone, Copy, Default)]
pub struct CoercionParams {
    /// PK's decimal scale (number of digits after decimal point).
    /// Required for Int → Decimal128 coercion to check overflow.
    pub pk_scale: Option<i32>,
}

impl CoercionParams {
    /// Create params with the given scale.
    pub fn with_scale(scale: i32) -> Self {
        Self {
            pk_scale: Some(scale),
        }
    }
}

/// Maximum safe scale for Int → Decimal128 conversions.
/// If scale > max, the conversion may overflow and is Lossy.
///
/// Decimal128 max ≈ 10^38. For safe conversion: max_value × 10^scale < 10^38
///
/// | Type   | Max Abs Value | Digits | Max Scale | Reasoning                |
/// |--------|---------------|--------|-----------|--------------------------|
/// | Int8   | 128           | 3      | 35        | 10^3 × 10^35 < 10^38     |
/// | UInt8  | 255           | 3      | 35        | 10^3 × 10^35 < 10^38     |
/// | Int16  | 32768         | 5      | 33        | 10^5 × 10^33 < 10^38     |
/// | UInt16 | 65535         | 5      | 33        | 10^5 × 10^33 < 10^38     |
/// | Int32  | 2.1×10^9      | 10     | 28        | 10^10 × 10^28 < 10^38    |
/// | UInt32 | 4.3×10^9      | 10     | 28        | 10^10 × 10^28 < 10^38    |
/// | Int64  | 9.2×10^18     | 19     | 19        | 10^19 × 10^19 < 10^38    |
/// | UInt64 | 1.8×10^19     | 20     | 18        | 10^20 × 10^18 < 10^38    |
fn max_safe_scale_for_decimal128(fk: &IngestedType) -> Option<i32> {
    use IngestedType::*;
    match fk {
        Int8 | UInt8 => Some(35),   // max 128/255 → 3 digits
        Int16 | UInt16 => Some(33), // max 32768/65535 → 5 digits
        Int32 | UInt32 => Some(28), // max ~2×10^9/4×10^9 → 10 digits
        Int64 => Some(19),          // max ~9×10^18 → 19 digits
        UInt64 => Some(18),         // max ~1.8×10^19 → 20 digits
        _ => None,
    }
}

/// Look up the coercion kind for an FK→PK type pair.
///
/// This is the single source of truth for the 23×23 coercion matrix.
///
/// # Arguments
/// * `fk` - The foreign key's IngestedType
/// * `pk` - The primary key's IngestedType
/// * `params` - Optional parameters for scale-dependent coercions
///
/// # Returns
/// The kind of coercion needed (or error category)
///
/// # Scale-dependent coercions
///
/// Int → Decimal128 is Safe only if scale ≤ max_safe_scale:
/// - Int8/UInt8: scale ≤ 36
/// - Int16/UInt16: scale ≤ 33
/// - Int32/UInt32: scale ≤ 28
/// - Int64: scale ≤ 19
/// - UInt64: scale ≤ 18
///
/// If scale exceeds the limit, the coercion is Lossy (may overflow).
/// If params.pk_scale is None for Int → Decimal128, we assume Safe (scale=0).
pub fn coercion_kind(fk: IngestedType, pk: IngestedType, params: CoercionParams) -> CoercionKind {
    use CoercionKind::*;
    use IngestedType::*;

    if fk == pk {
        return Same;
    }

    // Match on references to avoid moving fk/pk (needed for scale check later)
    match (&fk, &pk) {
        // ===================================================================
        // Safe: Integer widening (lossless)
        // ===================================================================
        (Int8, Int16 | Int32 | Int64) => Safe,
        (Int16, Int32 | Int64) => Safe,
        (Int32, Int64) => Safe,
        (UInt8, UInt16 | UInt32 | UInt64 | Int16 | Int32 | Int64) => Safe,
        (UInt16, UInt32 | UInt64 | Int32 | Int64) => Safe,
        (UInt32, UInt64 | Int64) => Safe,

        // ===================================================================
        // Safe: Float widening
        // ===================================================================
        (Float32, Float64) => Safe,

        // ===================================================================
        // Safe: Small integers to float (fit within mantissa precision)
        // Float32 has 23-bit mantissa: Int8, Int16, UInt8, UInt16 fit
        // Float64 has 52-bit mantissa: Int8-Int32, UInt8-UInt32 fit
        // ===================================================================
        (Int8 | Int16 | UInt8 | UInt16, Float32 | Float64) => Safe,
        (Int32 | UInt32, Float64) => Safe,

        // ===================================================================
        // Lossy: Large integers to float (exceed mantissa precision)
        // Int32/UInt32 > 2^24 loses precision in Float32
        // Int64/UInt64 > 2^53 loses precision in Float64
        // ===================================================================
        (Int32 | UInt32, Float32) => Lossy,
        (Int64 | UInt64, Float64) => Lossy,

        // ===================================================================
        // Safe: Anything to String (formatting always succeeds)
        // ===================================================================
        (Int8 | Int16 | Int32 | Int64, String) => Safe,
        (UInt8 | UInt16 | UInt32 | UInt64, String) => Safe,
        (Float32 | Float64, String) => Safe,
        (Decimal128 | Decimal256, String) => Safe,
        (Bool, String) => Safe,
        (Date | Timestamp, String) => Safe,
        // NOTE: Ip → String always outputs IPv6 format (e.g., "::ffff:192.168.1.1" for IPv4).
        // IPv4 addresses are stored internally as IPv4-mapped IPv6 and formatted accordingly.
        (Uuid | Uuid256 | Ip, String) => Safe,
        (Bytes, String) => Safe, // UTF-8 decode attempt

        // ===================================================================
        // Safe: String to Bytes (always succeeds)
        // ===================================================================
        (String, Bytes) => Safe,

        // ===================================================================
        // Decimal conversions
        //
        // NOTE: `params.pk_scale` is ONLY used for Int → Decimal128 to check
        // for overflow (value × 10^scale must fit in Decimal128).
        //
        // For Float → Decimal, scale is IGNORED because the coercion kind is
        // determined by RANGE constraints (Float max vs Decimal max), not scale.
        // ===================================================================
        (Decimal128, Decimal256) => Safe,

        // Int → Decimal128: Safe if scale ≤ max_safe_scale, else Lossy (overflow)
        (Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64, Decimal128) => {
            if let Some(max_scale) = max_safe_scale_for_decimal128(&fk) {
                let scale = params.pk_scale.unwrap_or(0);
                if scale <= max_scale {
                    Safe
                } else {
                    Lossy
                }
            } else {
                Incompatible // Should not happen for Int types
            }
        }

        // Int → Decimal256: Always Safe (Decimal256 has 76 digits, plenty of room)
        (Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64, Decimal256) => Safe,

        // Float → Decimal: Determined by RANGE, not scale.
        // Float32 max (~10^38) slightly > Decimal128 max → Lossy
        // Float64 max (~10^308) >> Decimal128 max → Lossy
        (Float32 | Float64, Decimal128) => Lossy,

        // Float32 → Decimal256: Safe (Float32 max ~10^38 < Decimal256 max ~10^76)
        (Float32, Decimal256) => Safe,

        // Float64 → Decimal256: Lossy (Float64 max ~10^308 >> Decimal256 max ~10^76)
        (Float64, Decimal256) => Lossy,

        // ===================================================================
        // Safe: Date to Timestamp (no information loss)
        // ===================================================================
        (Date, Timestamp) => Safe,

        // ===================================================================
        // Lossy: Timestamp to Date (truncates time component)
        // ===================================================================
        (Timestamp, Date) => Lossy,

        // ===================================================================
        // Safe: Identifier to bytes
        // ===================================================================
        (Uuid | Uuid256 | Ip, Bytes) => Safe,

        // ===================================================================
        // Conditional: String to typed (parse required)
        // These may fail if the string doesn't parse correctly
        // ===================================================================
        (String, Int8 | Int16 | Int32 | Int64) => Conditional,
        (String, UInt8 | UInt16 | UInt32 | UInt64) => Conditional,
        (String, Float32 | Float64) => Conditional,
        (String, Decimal128 | Decimal256) => Conditional,
        (String, Bool) => Conditional,
        (String, Date | Timestamp) => Conditional,
        (String, Uuid | Uuid256 | Ip) => Conditional,

        // ===================================================================
        // Incompatible: Bytes to typed
        // If your data contains parseable values, declare proper types in schema.
        // Bytes is for opaque binary data, not for runtime parsing.
        // ===================================================================
        (Bytes, Int8 | Int16 | Int32 | Int64) => Incompatible,
        (Bytes, UInt8 | UInt16 | UInt32 | UInt64) => Incompatible,
        (Bytes, Float32 | Float64) => Incompatible,
        (Bytes, Decimal128 | Decimal256) => Incompatible,
        (Bytes, Bool) => Incompatible,
        (Bytes, Uuid | Uuid256 | Ip) => Incompatible,

        // ===================================================================
        // Lossy: Signed/Unsigned boundary (overflow/wrap is silent corruption)
        // ===================================================================
        (Int8 | Int16 | Int32 | Int64, UInt8 | UInt16 | UInt32 | UInt64) => Lossy,
        (UInt64, Int64) => Lossy, // > i64::MAX wraps silently

        // ===================================================================
        // Lossy: Float to integer (truncates decimal portion)
        // ===================================================================
        (Float32 | Float64, Int8 | Int16 | Int32 | Int64) => Lossy,
        (Float32 | Float64, UInt8 | UInt16 | UInt32 | UInt64) => Lossy,

        // ===================================================================
        // Lossy: Narrowing integers (may overflow)
        // ===================================================================
        (Int16, Int8) => Lossy,
        (Int32, Int8 | Int16) => Lossy,
        (Int64, Int8 | Int16 | Int32) => Lossy,
        (UInt16, UInt8) => Lossy,
        (UInt32, UInt8 | UInt16) => Lossy,
        (UInt64, UInt8 | UInt16 | UInt32) => Lossy,

        // ===================================================================
        // Lossy: Float narrowing
        // ===================================================================
        (Float64, Float32) => Lossy,

        // ===================================================================
        // Lossy: Decimal to float (loses precision)
        // ===================================================================
        (Decimal128, Float32 | Float64) => Lossy,
        (Decimal256, Decimal128) => Lossy,
        (Decimal256, Float32 | Float64) => Lossy,
        (Decimal256, Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64) => Lossy,

        // ===================================================================
        // Incompatible: Bool to most types (no meaningful conversion)
        // ===================================================================
        (Bool, Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64) => Incompatible,
        (Bool, Float32 | Float64 | Decimal128 | Decimal256) => Incompatible,
        (Bool, Bytes | Date | Timestamp | Uuid | Uuid256 | Ip) => Incompatible,
        (Bool, Float32Array
            | Float64Array
            | Float16Array
            | Int8Array
            | Int16Array
            | Int32Array
            | Int64Array
            | StringArray) => {
            Incompatible
        }

        // ===================================================================
        // Incompatible: Temporal to numeric
        // ===================================================================
        (Date | Timestamp, Int8 | Int16 | Int32 | UInt8 | UInt16 | UInt32 | UInt64) => Incompatible,
        (Date | Timestamp, Float32 | Float64 | Decimal128 | Decimal256) => Incompatible,
        (Date | Timestamp, Bool | Bytes | Uuid | Uuid256 | Ip) => Incompatible,
        (
            Date | Timestamp,
            Float32Array
            | Float64Array
            | Float16Array
            | Int8Array
            | Int16Array
            | Int32Array
            | Int64Array
            | StringArray,
        ) => Incompatible,
        // Note: Date/Timestamp to Int64 could be epoch conversion, but we mark as Incompatible
        // since it's a semantic change. If needed, make explicit via String intermediary.
        (Date | Timestamp, Int64) => Incompatible,

        // ===================================================================
        // Incompatible: Bytes to temporal (no standard format)
        // ===================================================================
        (Bytes, Date | Timestamp) => Incompatible,

        // ===================================================================
        // Incompatible: Identifiers to wrong types
        // ===================================================================
        (Uuid, Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64) => Incompatible,
        (Uuid, Float32 | Float64 | Decimal128 | Decimal256 | Bool | Date | Timestamp) => {
            Incompatible
        }
        (
            Uuid,
            Uuid256 | Ip | Float32Array
            | Float64Array
            | Float16Array
            | Int8Array
            | Int16Array
            | Int32Array
            | Int64Array
            | StringArray,
        ) => Incompatible,

        (Uuid256, Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64) => Incompatible,
        (Uuid256, Float32 | Float64 | Decimal128 | Decimal256 | Bool | Date | Timestamp) => {
            Incompatible
        }
        (
            Uuid256,
            Uuid | Ip | Float32Array
            | Float64Array
            | Float16Array
            | Int8Array
            | Int16Array
            | Int32Array
            | Int64Array
            | StringArray,
        ) => Incompatible,

        (Ip, Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64) => Incompatible,
        (Ip, Float32 | Float64 | Decimal128 | Decimal256 | Bool | Date | Timestamp) => Incompatible,
        (
            Ip,
            Uuid | Uuid256 | Float32Array
            | Float64Array
            | Float16Array
            | Int8Array
            | Int16Array
            | Int32Array
            | Int64Array
            | StringArray,
        ) => Incompatible,

        // ===================================================================
        // Intra-int-array widening (safe when source ≤ target width).
        // Narrowing in the other direction stays Incompatible — narrowing
        // would need a per-element bounds check at coercion time which we
        // don't model here. The encoder's `require_fits_in_signed` is the
        // backstop for the cases that do go through.
        // ===================================================================
        (Int8Array, Int16Array | Int32Array | Int64Array) => Safe,
        (Int16Array, Int32Array | Int64Array) => Safe,
        (Int32Array, Int64Array) => Safe,

        // ===================================================================
        // Incompatible: Array types
        // ===================================================================
        (Float32Array
            | Float64Array
            | Float16Array
            | Int8Array
            | Int16Array
            | Int32Array
            | Int64Array
            | StringArray, _) => Incompatible,
        (_, Float32Array
            | Float64Array
            | Float16Array
            | Int8Array
            | Int16Array
            | Int32Array
            | Int64Array
            | StringArray) => Incompatible,

        // ===================================================================
        // Default: Anything not explicitly handled is Incompatible
        // ===================================================================
        _ => Incompatible,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NO_SCALE: CoercionParams = CoercionParams { pk_scale: None };

    #[test]
    fn test_same_types() {
        let all_types = all_ingested_types();
        for t in all_types {
            assert_eq!(
                coercion_kind(t.clone(), t.clone(), NO_SCALE),
                CoercionKind::Same,
                "{:?} → {:?} should be Same",
                t,
                t
            );
        }
    }

    #[test]
    fn test_int_widening_is_safe() {
        assert_eq!(
            coercion_kind(IngestedType::Int32, IngestedType::Int64, NO_SCALE),
            CoercionKind::Safe
        );
        assert_eq!(
            coercion_kind(IngestedType::Int16, IngestedType::Int32, NO_SCALE),
            CoercionKind::Safe
        );
        assert_eq!(
            coercion_kind(IngestedType::Int8, IngestedType::Int64, NO_SCALE),
            CoercionKind::Safe
        );
    }

    #[test]
    fn test_string_to_int_is_conditional() {
        assert_eq!(
            coercion_kind(IngestedType::String, IngestedType::Int64, NO_SCALE),
            CoercionKind::Conditional
        );
    }

    #[test]
    fn test_float_to_int_is_lossy() {
        assert_eq!(
            coercion_kind(IngestedType::Float64, IngestedType::Int64, NO_SCALE),
            CoercionKind::Lossy
        );
    }

    #[test]
    fn test_bool_to_uuid_is_incompatible() {
        assert_eq!(
            coercion_kind(IngestedType::Bool, IngestedType::Uuid, NO_SCALE),
            CoercionKind::Incompatible
        );
    }

    #[test]
    fn test_anything_to_string_is_safe() {
        let all_types = all_ingested_types();
        for t in all_types {
            if t != IngestedType::String {
                let kind = coercion_kind(t.clone(), IngestedType::String, NO_SCALE);
                // Most types to String should be Safe, except arrays which are Incompatible
                match t {
                    IngestedType::Float32Array
                    | IngestedType::Float64Array
                    | IngestedType::Float16Array
                    | IngestedType::Int8Array
                    | IngestedType::Int16Array
                    | IngestedType::Int32Array
                    | IngestedType::Int64Array
                    | IngestedType::StringArray => {
                        assert_eq!(kind, CoercionKind::Incompatible, "{:?} → String", t);
                    }
                    _ => {
                        assert_eq!(kind, CoercionKind::Safe, "{:?} → String should be Safe", t);
                    }
                }
            }
        }
    }

    // =========================================================================
    // Scale-dependent Decimal coercion tests
    // =========================================================================

    #[test]
    fn test_int64_to_decimal128_scale_dependent() {
        use IngestedType::*;
        // Int64 max safe scale = 19
        assert_eq!(
            coercion_kind(Int64, Decimal128, CoercionParams::with_scale(0)),
            CoercionKind::Safe
        );
        assert_eq!(
            coercion_kind(Int64, Decimal128, CoercionParams::with_scale(19)),
            CoercionKind::Safe
        );
        assert_eq!(
            coercion_kind(Int64, Decimal128, CoercionParams::with_scale(20)),
            CoercionKind::Lossy
        );
        assert_eq!(
            coercion_kind(Int64, Decimal128, CoercionParams::with_scale(38)),
            CoercionKind::Lossy
        );
    }

    #[test]
    fn test_int8_to_decimal128_scale_dependent() {
        use IngestedType::*;
        // Int8 max safe scale = 35
        assert_eq!(
            coercion_kind(Int8, Decimal128, CoercionParams::with_scale(0)),
            CoercionKind::Safe
        );
        assert_eq!(
            coercion_kind(Int8, Decimal128, CoercionParams::with_scale(35)),
            CoercionKind::Safe
        );
        assert_eq!(
            coercion_kind(Int8, Decimal128, CoercionParams::with_scale(36)),
            CoercionKind::Lossy
        );
    }

    #[test]
    fn test_uint64_to_decimal128_scale_dependent() {
        use IngestedType::*;
        // UInt64 max safe scale = 18
        assert_eq!(
            coercion_kind(UInt64, Decimal128, CoercionParams::with_scale(0)),
            CoercionKind::Safe
        );
        assert_eq!(
            coercion_kind(UInt64, Decimal128, CoercionParams::with_scale(18)),
            CoercionKind::Safe
        );
        assert_eq!(
            coercion_kind(UInt64, Decimal128, CoercionParams::with_scale(19)),
            CoercionKind::Lossy
        );
    }

    #[test]
    fn test_int_to_decimal256_always_safe() {
        use IngestedType::*;
        // Int → Decimal256 is always Safe (plenty of room)
        for int_type in [Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64] {
            assert_eq!(
                coercion_kind(int_type.clone(), Decimal256, NO_SCALE),
                CoercionKind::Safe,
                "{:?} → Decimal256 should be Safe",
                int_type
            );
            // Even with high scale
            assert_eq!(
                coercion_kind(int_type.clone(), Decimal256, CoercionParams::with_scale(50)),
                CoercionKind::Safe,
                "{:?} → Decimal256 with scale=50 should be Safe",
                int_type
            );
        }
    }

    #[test]
    fn test_float_to_decimal_coercions() {
        use IngestedType::*;
        // Float32 → Decimal128: Lossy (range exceeds)
        assert_eq!(
            coercion_kind(Float32, Decimal128, NO_SCALE),
            CoercionKind::Lossy
        );
        // Float64 → Decimal128: Lossy (range exceeds)
        assert_eq!(
            coercion_kind(Float64, Decimal128, NO_SCALE),
            CoercionKind::Lossy
        );
        // Float32 → Decimal256: Safe (range fits, precision fits)
        assert_eq!(
            coercion_kind(Float32, Decimal256, NO_SCALE),
            CoercionKind::Safe
        );
        // Float64 → Decimal256: Lossy (range exceeds)
        assert_eq!(
            coercion_kind(Float64, Decimal256, NO_SCALE),
            CoercionKind::Lossy
        );
    }

    fn all_ingested_types() -> Vec<IngestedType> {
        vec![
            IngestedType::Int8,
            IngestedType::Int16,
            IngestedType::Int32,
            IngestedType::Int64,
            IngestedType::UInt8,
            IngestedType::UInt16,
            IngestedType::UInt32,
            IngestedType::UInt64,
            IngestedType::Float32,
            IngestedType::Float64,
            IngestedType::Decimal128,
            IngestedType::Decimal256,
            IngestedType::String,
            IngestedType::Bytes,
            IngestedType::Bool,
            IngestedType::Date,
            IngestedType::Timestamp,
            IngestedType::Uuid,
            IngestedType::Uuid256,
            IngestedType::Ip,
            IngestedType::Float32Array,
            IngestedType::Float64Array,
            IngestedType::Float16Array,
            IngestedType::Int8Array,
            IngestedType::Int16Array,
            IngestedType::Int32Array,
            IngestedType::Int64Array,
            IngestedType::StringArray,
        ]
    }
}
