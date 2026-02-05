//! Coercion kind determination for FK-PK type pairs.
//!
//! This module defines the `CoercionKind` enum and the `coercion_kind` function
//! that determines what kind of coercion is needed between two IngestedTypes.
//!
//! The 23×23 matrix is the single source of truth for FK-PK type compatibility.

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

/// Look up the coercion kind for an FK→PK type pair.
///
/// This is the single source of truth for the 23×23 coercion matrix.
///
/// # Arguments
/// * `fk` - The foreign key's IngestedType
/// * `pk` - The primary key's IngestedType
///
/// # Returns
/// The kind of coercion needed (or error category)
pub fn coercion_kind(fk: IngestedType, pk: IngestedType) -> CoercionKind {
    use CoercionKind::*;
    use IngestedType::*;

    if fk == pk {
        return Same;
    }

    match (fk, pk) {
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
        (Uuid | Uuid256 | Ip, String) => Safe,
        (Bytes, String) => Safe, // UTF-8 decode attempt

        // ===================================================================
        // Safe: String to Bytes (always succeeds)
        // ===================================================================
        (String, Bytes) => Safe,

        // ===================================================================
        // Safe: Decimal conversions
        // ===================================================================
        (Decimal128, Decimal256) => Safe,
        (Int8 | Int16 | Int32 | Int64, Decimal128) => Safe,
        (UInt8 | UInt16 | UInt32 | UInt64, Decimal128) => Safe,

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
        // Conditional: Signed/Unsigned boundary (negative values fail)
        // ===================================================================
        (Int8 | Int16 | Int32 | Int64, UInt8 | UInt16 | UInt32 | UInt64) => Conditional,
        (UInt64, Int64) => Conditional, // > i64::MAX fails

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
        (Bool, Float32Array | Float64Array | Float16Array) => Incompatible,

        // ===================================================================
        // Incompatible: Temporal to numeric
        // ===================================================================
        (Date | Timestamp, Int8 | Int16 | Int32 | UInt8 | UInt16 | UInt32 | UInt64) => Incompatible,
        (Date | Timestamp, Float32 | Float64 | Decimal128 | Decimal256) => Incompatible,
        (Date | Timestamp, Bool | Bytes | Uuid | Uuid256 | Ip) => Incompatible,
        (Date | Timestamp, Float32Array | Float64Array | Float16Array) => Incompatible,
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
        (Uuid, Uuid256 | Ip | Float32Array | Float64Array | Float16Array) => Incompatible,

        (Uuid256, Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64) => Incompatible,
        (Uuid256, Float32 | Float64 | Decimal128 | Decimal256 | Bool | Date | Timestamp) => {
            Incompatible
        }
        (Uuid256, Uuid | Ip | Float32Array | Float64Array | Float16Array) => Incompatible,

        (Ip, Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64) => Incompatible,
        (Ip, Float32 | Float64 | Decimal128 | Decimal256 | Bool | Date | Timestamp) => Incompatible,
        (Ip, Uuid | Uuid256 | Float32Array | Float64Array | Float16Array) => Incompatible,

        // ===================================================================
        // Incompatible: Array types
        // ===================================================================
        (Float32Array | Float64Array | Float16Array, _) => Incompatible,
        (_, Float32Array | Float64Array | Float16Array) => Incompatible,

        // ===================================================================
        // Default: Anything not explicitly handled is Incompatible
        // ===================================================================
        _ => Incompatible,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_types() {
        let all_types = all_ingested_types();
        for t in all_types {
            assert_eq!(
                coercion_kind(t.clone(), t.clone()),
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
            coercion_kind(IngestedType::Int32, IngestedType::Int64),
            CoercionKind::Safe
        );
        assert_eq!(
            coercion_kind(IngestedType::Int16, IngestedType::Int32),
            CoercionKind::Safe
        );
        assert_eq!(
            coercion_kind(IngestedType::Int8, IngestedType::Int64),
            CoercionKind::Safe
        );
    }

    #[test]
    fn test_string_to_int_is_conditional() {
        assert_eq!(
            coercion_kind(IngestedType::String, IngestedType::Int64),
            CoercionKind::Conditional
        );
    }

    #[test]
    fn test_float_to_int_is_lossy() {
        assert_eq!(
            coercion_kind(IngestedType::Float64, IngestedType::Int64),
            CoercionKind::Lossy
        );
    }

    #[test]
    fn test_bool_to_uuid_is_incompatible() {
        assert_eq!(
            coercion_kind(IngestedType::Bool, IngestedType::Uuid),
            CoercionKind::Incompatible
        );
    }

    #[test]
    fn test_anything_to_string_is_safe() {
        let all_types = all_ingested_types();
        for t in all_types {
            if t != IngestedType::String {
                let kind = coercion_kind(t.clone(), IngestedType::String);
                // Most types to String should be Safe, except arrays which are Incompatible
                match t {
                    IngestedType::Float32Array
                    | IngestedType::Float64Array
                    | IngestedType::Float16Array => {
                        assert_eq!(kind, CoercionKind::Incompatible, "{:?} → String", t);
                    }
                    _ => {
                        assert_eq!(kind, CoercionKind::Safe, "{:?} → String should be Safe", t);
                    }
                }
            }
        }
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
        ]
    }
}
