//! Full 23×23 FK-PK coercion matrix tests.
//!
//! This module contains the complete test matrix for all IngestedType × IngestedType
//! combinations (529 tests total).
//!
//! Each cell tests that `coercion_kind(fk_type, pk_type)` returns the expected result.

use super::{coercion_kind, CoercionKind};
use crate::graph_config::IngestedType;

/// All 23 IngestedType variants.
fn all_types() -> [IngestedType; 23] {
    use IngestedType::*;
    [
        Int8,
        Int16,
        Int32,
        Int64,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        Float32,
        Float64,
        Decimal128,
        Decimal256,
        String,
        Bytes,
        Bool,
        Date,
        Timestamp,
        Uuid,
        Uuid256,
        Ip,
        Float32Array,
        Float64Array,
        Float16Array,
    ]
}

/// Returns the expected CoercionKind for each FK→PK pair.
/// This is the single source of truth for the 23×23 matrix.
fn expected_coercion(fk: &IngestedType, pk: &IngestedType) -> CoercionKind {
    use CoercionKind::*;
    use IngestedType::*;

    if fk == pk {
        return Same;
    }

    match (fk, pk) {
        // ===== Integer widening (Safe) =====
        (Int8, Int16 | Int32 | Int64) => Safe,
        (Int16, Int32 | Int64) => Safe,
        (Int32, Int64) => Safe,
        (UInt8, UInt16 | UInt32 | UInt64 | Int16 | Int32 | Int64) => Safe,
        (UInt16, UInt32 | UInt64 | Int32 | Int64) => Safe,
        (UInt32, UInt64 | Int64) => Safe,

        // ===== Float widening (Safe) =====
        (Float32, Float64) => Safe,

        // ===== Integer/Float to larger float (Safe) =====
        (Int8 | Int16 | Int32 | UInt8 | UInt16 | UInt32, Float32 | Float64) => Safe,
        (Int64 | UInt64, Float64) => Safe,

        // ===== Anything to String (Safe) =====
        (Int8 | Int16 | Int32 | Int64, String) => Safe,
        (UInt8 | UInt16 | UInt32 | UInt64, String) => Safe,
        (Float32 | Float64, String) => Safe,
        (Decimal128 | Decimal256, String) => Safe,
        (Bool, String) => Safe,
        (Date | Timestamp, String) => Safe,
        (Uuid | Uuid256 | Ip, String) => Safe,
        (Bytes, String) => Safe,

        // ===== String to Bytes (Safe) =====
        (String, Bytes) => Safe,

        // ===== Decimal conversions (Safe) =====
        (Decimal128, Decimal256) => Safe,
        (Int8 | Int16 | Int32 | Int64, Decimal128) => Safe,
        (UInt8 | UInt16 | UInt32 | UInt64, Decimal128) => Safe,

        // ===== Temporal conversions (Safe) =====
        (Date, Timestamp) => Safe,
        (Timestamp, Date) => Safe,

        // ===== Identifier to bytes (Safe) =====
        (Uuid | Uuid256 | Ip, Bytes) => Safe,

        // ===== String to typed (Conditional) =====
        (String, Int8 | Int16 | Int32 | Int64) => Conditional,
        (String, UInt8 | UInt16 | UInt32 | UInt64) => Conditional,
        (String, Float32 | Float64) => Conditional,
        (String, Decimal128 | Decimal256) => Conditional,
        (String, Bool) => Conditional,
        (String, Date | Timestamp) => Conditional,
        (String, Uuid | Uuid256 | Ip) => Conditional,

        // ===== Bytes to typed (Incompatible - fix schema, don't parse at runtime) =====
        (Bytes, Int8 | Int16 | Int32 | Int64) => Incompatible,
        (Bytes, UInt8 | UInt16 | UInt32 | UInt64) => Incompatible,
        (Bytes, Float32 | Float64) => Incompatible,
        (Bytes, Decimal128 | Decimal256) => Incompatible,
        (Bytes, Bool) => Incompatible,
        (Bytes, Uuid | Uuid256 | Ip) => Incompatible,

        // ===== Signed/Unsigned boundary (Conditional) =====
        (Int8 | Int16 | Int32 | Int64, UInt8 | UInt16 | UInt32 | UInt64) => Conditional,
        (UInt64, Int64) => Conditional,

        // ===== Float to integer (Lossy) =====
        (Float32 | Float64, Int8 | Int16 | Int32 | Int64) => Lossy,
        (Float32 | Float64, UInt8 | UInt16 | UInt32 | UInt64) => Lossy,

        // ===== Narrowing integers (Lossy) =====
        (Int16, Int8) => Lossy,
        (Int32, Int8 | Int16) => Lossy,
        (Int64, Int8 | Int16 | Int32) => Lossy,
        (UInt16, UInt8) => Lossy,
        (UInt32, UInt8 | UInt16) => Lossy,
        (UInt64, UInt8 | UInt16 | UInt32) => Lossy,

        // ===== Float narrowing (Lossy) =====
        (Float64, Float32) => Lossy,

        // ===== Decimal to float (Lossy - loses precision) =====
        (Decimal128, Float32 | Float64) => Lossy,
        (Decimal256, Decimal128) => Lossy,
        (Decimal256, Float32 | Float64) => Lossy,
        (Decimal256, Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64) => Lossy,

        // ===== Bool to most types (Incompatible) =====
        (Bool, Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64) => Incompatible,
        (Bool, Float32 | Float64 | Decimal128 | Decimal256) => Incompatible,
        (Bool, Bytes | Date | Timestamp | Uuid | Uuid256 | Ip) => Incompatible,
        (Bool, Float32Array | Float64Array | Float16Array) => Incompatible,

        // ===== Temporal to numeric (Incompatible) =====
        (Date | Timestamp, Int8 | Int16 | Int32 | UInt8 | UInt16 | UInt32 | UInt64) => Incompatible,
        (Date | Timestamp, Int64) => Incompatible,
        (Date | Timestamp, Float32 | Float64 | Decimal128 | Decimal256) => Incompatible,
        (Date | Timestamp, Bool | Bytes | Uuid | Uuid256 | Ip) => Incompatible,
        (Date | Timestamp, Float32Array | Float64Array | Float16Array) => Incompatible,

        // ===== Bytes to temporal (Incompatible) =====
        (Bytes, Date | Timestamp) => Incompatible,

        // ===== Identifiers to wrong types (Incompatible) =====
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

        // ===== Array types (Incompatible with everything except same) =====
        (Float32Array | Float64Array | Float16Array, _) => Incompatible,
        (_, Float32Array | Float64Array | Float16Array) => Incompatible,

        // ===== Default: Incompatible =====
        _ => Incompatible,
    }
}

/// Test the entire 23×23 matrix.
#[test]
fn test_full_coercion_matrix() {
    let types = all_types();
    let mut failures = Vec::new();

    for fk in &types {
        for pk in &types {
            let expected = expected_coercion(fk, pk);
            let actual = coercion_kind(fk.clone(), pk.clone());

            if actual != expected {
                failures.push(format!(
                    "{:?} → {:?}: expected {:?}, got {:?}",
                    fk, pk, expected, actual
                ));
            }
        }
    }

    if !failures.is_empty() {
        panic!(
            "Coercion matrix mismatches ({} failures):\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
}

/// Verify we test exactly 529 combinations (23×23).
#[test]
fn test_matrix_coverage() {
    let types = all_types();
    let mut count = 0;

    for fk in &types {
        for pk in &types {
            let _ = coercion_kind(fk.clone(), pk.clone());
            count += 1;
        }
    }

    assert_eq!(count, 529, "Should test 23×23 = 529 combinations");
}

// ============================================================================
// Individual category tests for documentation and debugging
// ============================================================================

#[test]
fn test_same_types_return_same() {
    let types = all_types();
    for t in &types {
        assert_eq!(
            coercion_kind(t.clone(), t.clone()),
            CoercionKind::Same,
            "{:?} → {:?}",
            t,
            t
        );
    }
}

#[test]
fn test_integer_widening_safe() {
    use IngestedType::*;
    let cases = [
        (Int8, Int16),
        (Int8, Int32),
        (Int8, Int64),
        (Int16, Int32),
        (Int16, Int64),
        (Int32, Int64),
        (UInt8, UInt16),
        (UInt8, UInt32),
        (UInt8, UInt64),
        (UInt8, Int16),
        (UInt8, Int32),
        (UInt8, Int64),
        (UInt16, UInt32),
        (UInt16, UInt64),
        (UInt16, Int32),
        (UInt16, Int64),
        (UInt32, UInt64),
        (UInt32, Int64),
    ];

    for (fk, pk) in cases {
        assert_eq!(
            coercion_kind(fk.clone(), pk.clone()),
            CoercionKind::Safe,
            "{:?} → {:?} should be Safe",
            fk,
            pk
        );
    }
}

#[test]
fn test_integer_narrowing_lossy() {
    use IngestedType::*;
    let cases = [
        (Int64, Int32),
        (Int64, Int16),
        (Int64, Int8),
        (Int32, Int16),
        (Int32, Int8),
        (Int16, Int8),
        (UInt64, UInt32),
        (UInt64, UInt16),
        (UInt64, UInt8),
        (UInt32, UInt16),
        (UInt32, UInt8),
        (UInt16, UInt8),
    ];

    for (fk, pk) in cases {
        assert_eq!(
            coercion_kind(fk.clone(), pk.clone()),
            CoercionKind::Lossy,
            "{:?} → {:?} should be Lossy",
            fk,
            pk
        );
    }
}

#[test]
fn test_string_to_typed_conditional() {
    use IngestedType::*;
    let targets = [
        Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32, Float64, Decimal128,
        Decimal256, Bool, Date, Timestamp, Uuid, Uuid256, Ip,
    ];

    for pk in targets {
        assert_eq!(
            coercion_kind(String, pk.clone()),
            CoercionKind::Conditional,
            "String → {:?} should be Conditional",
            pk
        );
    }
}

#[test]
fn test_float_to_int_lossy() {
    use IngestedType::*;
    let floats = [Float32, Float64];
    let ints = [Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64];

    for fk in floats {
        for pk in &ints {
            assert_eq!(
                coercion_kind(fk.clone(), pk.clone()),
                CoercionKind::Lossy,
                "{:?} → {:?} should be Lossy",
                fk,
                pk
            );
        }
    }
}

#[test]
fn test_bool_mostly_incompatible() {
    use IngestedType::*;
    let types = all_types();

    for pk in &types {
        let expected = match pk {
            Bool => CoercionKind::Same,
            String => CoercionKind::Safe,
            _ => CoercionKind::Incompatible,
        };

        assert_eq!(
            coercion_kind(Bool, pk.clone()),
            expected,
            "Bool → {:?} expected {:?}",
            pk,
            expected
        );
    }
}

#[test]
fn test_array_types_incompatible() {
    use IngestedType::*;
    let arrays = [Float32Array, Float64Array, Float16Array];
    let types = all_types();

    for arr in &arrays {
        for other in &types {
            let expected = if arr == other {
                CoercionKind::Same
            } else {
                CoercionKind::Incompatible
            };

            // Array as FK
            assert_eq!(
                coercion_kind(arr.clone(), other.clone()),
                expected,
                "{:?} → {:?}",
                arr,
                other
            );

            // Array as PK (except same)
            if arr != other {
                assert_eq!(
                    coercion_kind(other.clone(), arr.clone()),
                    CoercionKind::Incompatible,
                    "{:?} → {:?}",
                    other,
                    arr
                );
            }
        }
    }
}

#[test]
fn test_to_string_safe_except_arrays() {
    use IngestedType::*;
    let types = all_types();

    for fk in &types {
        if *fk == String {
            continue;
        }

        let expected = match fk {
            Float32Array | Float64Array | Float16Array => CoercionKind::Incompatible,
            _ => CoercionKind::Safe,
        };

        assert_eq!(
            coercion_kind(fk.clone(), String),
            expected,
            "{:?} → String",
            fk
        );
    }
}
