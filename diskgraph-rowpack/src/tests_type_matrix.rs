//! Arrow DataType → IngestedType conversion tests.
//!
//! Tests all combinations of Arrow types × IngestedType targets using a loop-based approach.
//! This mirrors the FK-PK coercion matrix structure with a single source of truth.
//!
//! **Design Principle:**
//! Arrow → IngestedType conversions are user-specified in the JSON config, so we respect
//! user intent and cast as requested (similar to SQL CAST behavior):
//! - Float64 → Int64? Truncate decimals silently
//! - String → Int64? Parse the string
//! - Invalid conversions (e.g., Int64 → Uuid) still error at runtime

use super::{RowAccess, RowAccessHints, RowAccessStats};
use arrow_array::*;
use arrow_buffer::i256;
use std::sync::Arc;

// =============================================================================
// Types and Enums
// =============================================================================

/// Target IngestedType for conversion tests.
/// Maps to getter methods on RowAccess.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IngestedTarget {
    Int64,
    Int32,
    UInt64,
    UInt32,
    Float64,
    Float32,
    Bool,
    String,
    Bytes,
    Uuid,
    Uuid256,
    Ip,
    Date,
    Timestamp,
    Decimal128,
    Decimal256,
    // Array types
    Float32Array,
    Float64Array,
    Float16Array,
}

impl IngestedTarget {
    fn all() -> &'static [IngestedTarget] {
        use IngestedTarget::*;
        &[
            Int64, Int32, UInt64, UInt32, Float64, Float32, Bool,
            String, Bytes, Uuid, Uuid256, Ip, Date, Timestamp,
            Decimal128, Decimal256,
            Float32Array, Float64Array, Float16Array,
        ]
    }
}

/// Expected result of a conversion.
#[derive(Debug, Clone, PartialEq)]
enum ExpectedResult {
    /// Conversion should succeed
    Success,
    /// Conversion should error (contains expected error substring)
    Error,
}

/// Represents an Arrow type with sample test data.
struct ArrowTestData {
    name: &'static str,
    array: ArrayRef,
    hints: RowAccessHints,
}

// =============================================================================
// Test Helpers
// =============================================================================

fn default_hints() -> RowAccessHints {
    RowAccessHints {
        timestamp_format: None,
        timestamp_unit: None,
        decimal_scale: None,
    }
}

fn hints_with_scale(scale: i32) -> RowAccessHints {
    RowAccessHints {
        timestamp_format: None,
        timestamp_unit: None,
        decimal_scale: Some(scale),
    }
}

/// Creates a RowAccess with the given array and sets row 0.
fn make_access(array: &ArrayRef, hints: RowAccessHints) -> (RowAccess<'static>, Vec<ArrayRef>) {
    let arrays: Vec<ArrayRef> = vec![array.clone()];
    let stats = RowAccessStats::default();
    let access = RowAccess::new(vec![hints], stats);
    (access, arrays)
}

/// Try to convert using the specified target getter method.
fn try_convert(access: &mut RowAccess, target: IngestedTarget) -> Result<(), String> {
    match target {
        IngestedTarget::Int64 => access.get_int64(0).map(|_| ()).map_err(|e| e.to_string()),
        IngestedTarget::Int32 => access.get_int32(0).map(|_| ()).map_err(|e| e.to_string()),
        IngestedTarget::UInt64 => access.get_uint64(0).map(|_| ()).map_err(|e| e.to_string()),
        IngestedTarget::UInt32 => access.get_uint32(0).map(|_| ()).map_err(|e| e.to_string()),
        IngestedTarget::Float64 => access.get_float64(0).map(|_| ()).map_err(|e| e.to_string()),
        IngestedTarget::Float32 => access.get_float32(0).map(|_| ()).map_err(|e| e.to_string()),
        IngestedTarget::Bool => access.get_bool(0).map(|_| ()).map_err(|e| e.to_string()),
        IngestedTarget::String => access.with_string(0, |_| Ok(())).map(|_| ()).map_err(|e| e.to_string()),
        IngestedTarget::Bytes => access.with_bytes(0, |_| Ok(())).map(|_| ()).map_err(|e| e.to_string()),
        IngestedTarget::Uuid => access.get_uuid(0).map(|_| ()).map_err(|e| e.to_string()),
        IngestedTarget::Uuid256 => access.get_uuid256(0).map(|_| ()).map_err(|e| e.to_string()),
        IngestedTarget::Ip => access.get_ip(0).map(|_| ()).map_err(|e| e.to_string()),
        IngestedTarget::Date => access.get_date(0).map(|_| ()).map_err(|e| e.to_string()),
        IngestedTarget::Timestamp => access.get_timestamp(0).map(|_| ()).map_err(|e| e.to_string()),
        IngestedTarget::Decimal128 => access.get_decimal128(0).map(|_| ()).map_err(|e| e.to_string()),
        IngestedTarget::Decimal256 => access.get_decimal256(0).map(|_| ()).map_err(|e| e.to_string()),
        // Array types use float_list_view which handles List<Float*>
        IngestedTarget::Float32Array | IngestedTarget::Float16Array => {
            access.with_float32_array(0, |_| Ok(())).map(|_| ()).map_err(|e| e.to_string())
        }
        IngestedTarget::Float64Array => {
            access.with_float64_array(0, |_| Ok(())).map(|_| ()).map_err(|e| e.to_string())
        }
    }
}

// =============================================================================
// Arrow Test Data Factory
// =============================================================================

/// Creates all Arrow type test data with valid sample values.
fn all_arrow_test_data() -> Vec<ArrowTestData> {
    vec![
        // Signed integers
        ArrowTestData {
            name: "Int8",
            array: Arc::new(Int8Array::from(vec![42i8])),
            hints: default_hints(),
        },
        ArrowTestData {
            name: "Int16",
            array: Arc::new(Int16Array::from(vec![1000i16])),
            hints: default_hints(),
        },
        ArrowTestData {
            name: "Int32",
            array: Arc::new(Int32Array::from(vec![100000i32])),
            hints: default_hints(),
        },
        ArrowTestData {
            name: "Int64",
            array: Arc::new(Int64Array::from(vec![1000000i64])),
            hints: default_hints(),
        },
        // Unsigned integers
        ArrowTestData {
            name: "UInt8",
            array: Arc::new(UInt8Array::from(vec![200u8])),
            hints: default_hints(),
        },
        ArrowTestData {
            name: "UInt16",
            array: Arc::new(UInt16Array::from(vec![50000u16])),
            hints: default_hints(),
        },
        ArrowTestData {
            name: "UInt32",
            array: Arc::new(UInt32Array::from(vec![3000000000u32])),
            hints: default_hints(),
        },
        ArrowTestData {
            name: "UInt64",
            array: Arc::new(UInt64Array::from(vec![10000000000u64])),
            hints: default_hints(),
        },
        // Floats
        ArrowTestData {
            name: "Float32",
            array: Arc::new(Float32Array::from(vec![3.14f32])),
            hints: default_hints(),
        },
        ArrowTestData {
            name: "Float64",
            array: Arc::new(Float64Array::from(vec![2.718281828f64])),
            hints: default_hints(),
        },
        // Boolean
        ArrowTestData {
            name: "Boolean",
            array: Arc::new(BooleanArray::from(vec![true])),
            hints: default_hints(),
        },
        // Strings
        ArrowTestData {
            name: "Utf8",
            array: Arc::new(StringArray::from(vec!["42"])),
            hints: default_hints(),
        },
        ArrowTestData {
            name: "LargeUtf8",
            array: Arc::new(LargeStringArray::from(vec!["42"])),
            hints: default_hints(),
        },
        // Binary
        ArrowTestData {
            name: "Binary",
            array: Arc::new(BinaryArray::from_vec(vec![b"hello"])),
            hints: default_hints(),
        },
        ArrowTestData {
            name: "LargeBinary",
            array: Arc::new(LargeBinaryArray::from_vec(vec![b"hello"])),
            hints: default_hints(),
        },
        // FixedSizeBinary for UUID (16 bytes)
        ArrowTestData {
            name: "FixedSizeBinary16",
            array: Arc::new(FixedSizeBinaryArray::from(vec![Some(&[0u8; 16][..])])),
            hints: default_hints(),
        },
        // FixedSizeBinary for UUID256 (32 bytes)
        ArrowTestData {
            name: "FixedSizeBinary32",
            array: Arc::new(FixedSizeBinaryArray::from(vec![Some(&[0u8; 32][..])])),
            hints: default_hints(),
        },
        // Date
        ArrowTestData {
            name: "Date32",
            array: Arc::new(Date32Array::from(vec![19000i32])), // ~2022-01-01
            hints: default_hints(),
        },
        // Timestamps
        ArrowTestData {
            name: "TimestampSecond",
            array: Arc::new(TimestampSecondArray::from(vec![1640000000i64])),
            hints: default_hints(),
        },
        ArrowTestData {
            name: "TimestampMilli",
            array: Arc::new(TimestampMillisecondArray::from(vec![1640000000000i64])),
            hints: default_hints(),
        },
        ArrowTestData {
            name: "TimestampMicro",
            array: Arc::new(TimestampMicrosecondArray::from(vec![1640000000000000i64])),
            hints: default_hints(),
        },
        ArrowTestData {
            name: "TimestampNano",
            array: Arc::new(TimestampNanosecondArray::from(vec![1640000000000000000i64])),
            hints: default_hints(),
        },
        // Decimals
        ArrowTestData {
            name: "Decimal128",
            array: Arc::new(
                Decimal128Array::from(vec![12345i128])
                    .with_precision_and_scale(10, 2)
                    .unwrap()
            ),
            hints: hints_with_scale(2),
        },
        ArrowTestData {
            name: "Decimal256",
            array: Arc::new(
                Decimal256Array::from(vec![i256::from(12345i64)])
                    .with_precision_and_scale(10, 2)
                    .unwrap()
            ),
            hints: hints_with_scale(2),
        },
        // List types for Float arrays
        ArrowTestData {
            name: "ListFloat32",
            array: make_list_float32_array(),
            hints: default_hints(),
        },
        ArrowTestData {
            name: "ListFloat64",
            array: make_list_float64_array(),
            hints: default_hints(),
        },
    ]
}

/// Create a ListArray containing Float32 values
fn make_list_float32_array() -> ArrayRef {
    use arrow_array::builder::{Float32Builder, ListBuilder};
    let mut builder = ListBuilder::new(Float32Builder::new());
    builder.values().append_value(1.0);
    builder.values().append_value(2.0);
    builder.values().append_value(3.0);
    builder.append(true);
    Arc::new(builder.finish())
}

/// Create a ListArray containing Float64 values
fn make_list_float64_array() -> ArrayRef {
    use arrow_array::builder::{Float64Builder, ListBuilder};
    let mut builder = ListBuilder::new(Float64Builder::new());
    builder.values().append_value(1.0);
    builder.values().append_value(2.0);
    builder.values().append_value(3.0);
    builder.append(true);
    Arc::new(builder.finish())
}

// =============================================================================
// Expected Results Matrix - Single Source of Truth
// =============================================================================

/// Returns the expected result for an Arrow→IngestedType conversion.
/// This is the single source of truth for the test matrix.
fn expected_result(arrow_name: &str, target: IngestedTarget) -> ExpectedResult {
    use ExpectedResult::*;
    use IngestedTarget::*;

    match (arrow_name, target) {
        // =====================================================================
        // Integer Arrow types (Int8, Int16, Int32, Int64, UInt8-64)
        // Can convert to: all numerics, bool, string, date, timestamp
        // Cannot convert to: bytes, uuid, uuid256, ip, decimal128/256
        // =====================================================================
        ("Int8" | "Int16" | "Int32" | "Int64" | "UInt8" | "UInt16" | "UInt32" | "UInt64",
         Int64 | Int32 | UInt64 | UInt32 | Float64 | Float32 | Bool | String | Date | Timestamp) => Success,

        ("Int8" | "Int16" | "Int32" | "Int64" | "UInt8" | "UInt16" | "UInt32" | "UInt64",
         Bytes | Uuid | Uuid256 | Ip | Decimal128 | Decimal256) => Error,

        // =====================================================================
        // Float Arrow types (Float32, Float64)
        // Can convert to: all numerics, bool, string
        // Cannot convert to: bytes, uuid, uuid256, ip, date, timestamp, decimal128/256
        // =====================================================================
        ("Float32" | "Float64",
         Int64 | Int32 | UInt64 | UInt32 | Float64 | Float32 | Bool | String) => Success,

        ("Float32" | "Float64",
         Bytes | Uuid | Uuid256 | Ip | Date | Timestamp | Decimal128 | Decimal256) => Error,

        // =====================================================================
        // Boolean Arrow type
        // Can convert to: all numerics, bool, string
        // Cannot convert to: bytes, uuid, uuid256, ip, date, timestamp, decimal128/256
        // =====================================================================
        ("Boolean",
         Int64 | Int32 | UInt64 | UInt32 | Float64 | Float32 | Bool | String) => Success,

        ("Boolean",
         Bytes | Uuid | Uuid256 | Ip | Date | Timestamp | Decimal128 | Decimal256) => Error,

        // =====================================================================
        // String Arrow types (Utf8, LargeUtf8)
        // Can convert to: all types (via parsing) - depends on content
        // For test purposes with "42" content: most succeed, uuid/uuid256/ip fail
        // =====================================================================
        ("Utf8" | "LargeUtf8",
         Int64 | Int32 | UInt64 | UInt32 | Float64 | Float32 | Bool | String | Bytes) => Success,

        // String "42" cannot be parsed as valid UUID, UUID256, or IP
        ("Utf8" | "LargeUtf8", Uuid | Uuid256 | Ip) => Error,

        // String "42" can be parsed as epoch seconds → date/timestamp
        ("Utf8" | "LargeUtf8", Date | Timestamp) => Success,

        // String can be parsed as decimal
        ("Utf8" | "LargeUtf8", Decimal128 | Decimal256) => Success,

        // =====================================================================
        // Binary Arrow types (Binary, LargeBinary)
        // Can convert to: string, bytes
        // Special: can convert to uuid/uuid256/ip if correct length
        // Cannot convert to: numerics, bool, date, timestamp, decimal
        // =====================================================================
        ("Binary" | "LargeBinary", String | Bytes) => Success,

        // Binary "hello" is wrong length for uuid (needs 16), uuid256 (needs 32), ip (needs 4 or 16)
        ("Binary" | "LargeBinary", Uuid | Uuid256 | Ip) => Error,

        ("Binary" | "LargeBinary",
         Int64 | Int32 | UInt64 | UInt32 | Float64 | Float32 | Bool | Date | Timestamp | Decimal128 | Decimal256) => Error,

        // =====================================================================
        // FixedSizeBinary(16) - UUID format
        // Can convert to: bytes, string, uuid, ip (if 16 bytes = IPv6)
        // Cannot convert to: numerics, bool, date, timestamp, decimal, uuid256
        // =====================================================================
        ("FixedSizeBinary16", Bytes | String | Uuid | Ip) => Success,

        ("FixedSizeBinary16", Uuid256) => Error, // Wrong size

        ("FixedSizeBinary16",
         Int64 | Int32 | UInt64 | UInt32 | Float64 | Float32 | Bool | Date | Timestamp | Decimal128 | Decimal256) => Error,

        // =====================================================================
        // FixedSizeBinary(32) - UUID256 format
        // Can convert to: bytes, string, uuid256
        // Cannot convert to: numerics, bool, date, timestamp, decimal, uuid, ip
        // =====================================================================
        ("FixedSizeBinary32", Bytes | String | Uuid256) => Success,

        ("FixedSizeBinary32", Uuid | Ip) => Error, // Wrong size

        ("FixedSizeBinary32",
         Int64 | Int32 | UInt64 | UInt32 | Float64 | Float32 | Bool | Date | Timestamp | Decimal128 | Decimal256) => Error,

        // =====================================================================
        // Date32 Arrow type
        // Can convert to: string, date, timestamp
        // Cannot convert to: numerics, bool, bytes, uuid, uuid256, ip, decimal
        // =====================================================================
        ("Date32", String | Date | Timestamp) => Success,

        ("Date32",
         Int64 | Int32 | UInt64 | UInt32 | Float64 | Float32 | Bool | Bytes | Uuid | Uuid256 | Ip | Decimal128 | Decimal256) => Error,

        // =====================================================================
        // Timestamp Arrow types
        // Can convert to: string, date, timestamp
        // Cannot convert to: numerics, bool, bytes, uuid, uuid256, ip, decimal
        // =====================================================================
        ("TimestampSecond" | "TimestampMilli" | "TimestampMicro" | "TimestampNano",
         String | Date | Timestamp) => Success,

        ("TimestampSecond" | "TimestampMilli" | "TimestampMicro" | "TimestampNano",
         Int64 | Int32 | UInt64 | UInt32 | Float64 | Float32 | Bool | Bytes | Uuid | Uuid256 | Ip | Decimal128 | Decimal256) => Error,

        // =====================================================================
        // Decimal128 Arrow type
        // Can convert to: numerics (via float), string
        // Cannot convert to: bool, bytes, uuid, uuid256, ip, date, timestamp, decimal256
        // =====================================================================
        ("Decimal128",
         Int64 | Int32 | UInt64 | UInt32 | Float64 | Float32 | String | Decimal128) => Success,

        ("Decimal128",
         Bool | Bytes | Uuid | Uuid256 | Ip | Date | Timestamp | Decimal256) => Error,

        // =====================================================================
        // Decimal256 Arrow type
        // Can convert to: numerics, string, decimal256
        // Cannot convert to: bool, bytes, uuid, uuid256, ip, date, timestamp, decimal128
        // =====================================================================
        ("Decimal256",
         Int64 | Int32 | UInt64 | UInt32 | Float64 | Float32 | String | Decimal256) => Success,

        ("Decimal256",
         Bool | Bytes | Uuid | Uuid256 | Ip | Date | Timestamp | Decimal128) => Error,

        // =====================================================================
        // List<Float32> Arrow type (for Float32Array/Float16Array)
        // Can only convert to: Float32Array, Float16Array
        // Cannot convert to: anything else
        // =====================================================================
        ("ListFloat32", Float32Array | Float16Array) => Success,

        ("ListFloat32",
         Int64 | Int32 | UInt64 | UInt32 | Float64 | Float32 | Bool | String | Bytes |
         Uuid | Uuid256 | Ip | Date | Timestamp | Decimal128 | Decimal256 | Float64Array) => Error,

        // =====================================================================
        // List<Float64> Arrow type (for Float64Array)
        // Can convert to: Float64Array, Float32Array, Float16Array (downcast allowed)
        // Cannot convert to: non-array types
        // =====================================================================
        ("ListFloat64", Float64Array | Float32Array | Float16Array) => Success,

        ("ListFloat64",
         Int64 | Int32 | UInt64 | UInt32 | Float64 | Float32 | Bool | String | Bytes |
         Uuid | Uuid256 | Ip | Date | Timestamp | Decimal128 | Decimal256) => Error,

        // =====================================================================
        // All non-List Arrow types → Array targets = Error
        // =====================================================================
        (_, Float32Array | Float64Array | Float16Array) => Error,

        // Default: Error for any unhandled combination
        _ => Error,
    }
}

// =============================================================================
// Main Matrix Test
// =============================================================================

/// Test the full Arrow → IngestedType matrix.
/// This single test covers all ~400 combinations.
#[test]
fn test_full_arrow_to_ingested_matrix() {
    let arrow_data = all_arrow_test_data();
    let targets = IngestedTarget::all();
    let mut failures = Vec::new();
    let mut success_count = 0;
    let mut error_count = 0;

    for data in &arrow_data {
        for &target in targets {
            let (mut access, arrays) = make_access(&data.array, data.hints.clone());
            access.set_cols(&arrays);
            access.set_row(0);

            let expected = expected_result(data.name, target);
            let actual = try_convert(&mut access, target);

            match (&expected, &actual) {
                (ExpectedResult::Success, Ok(())) => {
                    success_count += 1;
                }
                (ExpectedResult::Error, Err(_)) => {
                    error_count += 1;
                }
                (ExpectedResult::Success, Err(e)) => {
                    failures.push(format!(
                        "{} → {:?}: expected Success, got Error({})",
                        data.name, target, e
                    ));
                }
                (ExpectedResult::Error, Ok(())) => {
                    failures.push(format!(
                        "{} → {:?}: expected Error, got Success",
                        data.name, target
                    ));
                }
            }
        }
    }

    if !failures.is_empty() {
        panic!(
            "Arrow→IngestedType matrix mismatches ({} failures, {} successes, {} errors):\n{}",
            failures.len(),
            success_count,
            error_count,
            failures.join("\n")
        );
    }

    // Verify we tested expected number of combinations
    let total = success_count + error_count;
    assert!(
        total >= 494,
        "Should test at least 494 combinations (26 Arrow × 19 targets), got {}",
        total
    );
}

/// Verify matrix coverage.
#[test]
fn test_matrix_coverage() {
    let arrow_count = all_arrow_test_data().len();
    let target_count = IngestedTarget::all().len();
    let total = arrow_count * target_count;

    assert_eq!(arrow_count, 26, "Should have 26 Arrow types (24 base + 2 List types)");
    assert_eq!(target_count, 19, "Should have 19 IngestedType targets (16 base + 3 array types)");
    assert_eq!(total, 494, "Should test 26×19 = 494 combinations");
}

// =============================================================================
// Category Tests (for documentation and debugging)
// =============================================================================

#[test]
fn test_all_integers_to_int64() {
    let int_arrays: Vec<(&str, ArrayRef)> = vec![
        ("Int8", Arc::new(Int8Array::from(vec![42i8]))),
        ("Int16", Arc::new(Int16Array::from(vec![1000i16]))),
        ("Int32", Arc::new(Int32Array::from(vec![100000i32]))),
        ("Int64", Arc::new(Int64Array::from(vec![1000000i64]))),
        ("UInt8", Arc::new(UInt8Array::from(vec![200u8]))),
        ("UInt16", Arc::new(UInt16Array::from(vec![50000u16]))),
        ("UInt32", Arc::new(UInt32Array::from(vec![3000000000u32]))),
        ("UInt64", Arc::new(UInt64Array::from(vec![10000000000u64]))),
    ];

    for (name, array) in int_arrays {
        let (mut access, arrays) = make_access(&array, default_hints());
        access.set_cols(&arrays);
        access.set_row(0);

        assert!(
            access.get_int64(0).is_ok(),
            "{} should convert to Int64",
            name
        );
    }
}

#[test]
fn test_string_parsing_to_numerics() {
    let array: ArrayRef = Arc::new(StringArray::from(vec!["42"]));
    let (mut access, arrays) = make_access(&array, default_hints());
    access.set_cols(&arrays);
    access.set_row(0);

    assert!(access.get_int64(0).is_ok(), "Utf8 '42' → Int64");
    assert!(access.get_int32(0).is_ok(), "Utf8 '42' → Int32");
    assert!(access.get_float64(0).is_ok(), "Utf8 '42' → Float64");
    assert!(access.get_uint64(0).is_ok(), "Utf8 '42' → UInt64");
}

#[test]
fn test_temporal_conversions() {
    // Date32 → Date (direct)
    let date_array: ArrayRef = Arc::new(Date32Array::from(vec![19000i32]));
    let (mut access, arrays) = make_access(&date_array, default_hints());
    access.set_cols(&arrays);
    access.set_row(0);
    assert!(access.get_date(0).is_ok(), "Date32 → Date");
    assert!(access.get_timestamp(0).is_ok(), "Date32 → Timestamp");

    // Timestamp → Timestamp (direct)
    let ts_array: ArrayRef = Arc::new(TimestampSecondArray::from(vec![1640000000i64]));
    let (mut access, arrays) = make_access(&ts_array, default_hints());
    access.set_cols(&arrays);
    access.set_row(0);
    assert!(access.get_timestamp(0).is_ok(), "Timestamp → Timestamp");
    assert!(access.get_date(0).is_ok(), "Timestamp → Date");
}

#[test]
fn test_uuid_from_fixed_binary() {
    // FixedSizeBinary(16) → UUID
    let uuid_bytes = [0u8; 16];
    let array: ArrayRef = Arc::new(FixedSizeBinaryArray::from(vec![Some(&uuid_bytes[..])]));
    let (mut access, arrays) = make_access(&array, default_hints());
    access.set_cols(&arrays);
    access.set_row(0);
    assert!(access.get_uuid(0).is_ok(), "FixedSizeBinary(16) → Uuid");

    // FixedSizeBinary(32) → UUID256
    let uuid256_bytes = [0u8; 32];
    let array: ArrayRef = Arc::new(FixedSizeBinaryArray::from(vec![Some(&uuid256_bytes[..])]));
    let (mut access, arrays) = make_access(&array, default_hints());
    access.set_cols(&arrays);
    access.set_row(0);
    assert!(access.get_uuid256(0).is_ok(), "FixedSizeBinary(32) → Uuid256");
}

#[test]
fn test_incompatible_conversions() {
    // Int64 → UUID should fail
    let int_array: ArrayRef = Arc::new(Int64Array::from(vec![42i64]));
    let (mut access, arrays) = make_access(&int_array, default_hints());
    access.set_cols(&arrays);
    access.set_row(0);
    assert!(access.get_uuid(0).is_err(), "Int64 → Uuid should error");

    // Boolean → Date should fail
    let bool_array: ArrayRef = Arc::new(BooleanArray::from(vec![true]));
    let (mut access, arrays) = make_access(&bool_array, default_hints());
    access.set_cols(&arrays);
    access.set_row(0);
    assert!(access.get_date(0).is_err(), "Boolean → Date should error");

    // Date32 → Int64 should fail
    let date_array: ArrayRef = Arc::new(Date32Array::from(vec![19000i32]));
    let (mut access, arrays) = make_access(&date_array, default_hints());
    access.set_cols(&arrays);
    access.set_row(0);
    assert!(access.get_int64(0).is_err(), "Date32 → Int64 should error");
}

#[test]
fn test_decimal_conversions() {
    // Decimal128 → Float64 (via integer division by scale)
    let array: ArrayRef = Arc::new(
        Decimal128Array::from(vec![12345i128])
            .with_precision_and_scale(10, 2)
            .unwrap()
    );
    let (mut access, arrays) = make_access(&array, hints_with_scale(2));
    access.set_cols(&arrays);
    access.set_row(0);
    assert!(access.get_float64(0).is_ok(), "Decimal128 → Float64");
    assert!(access.get_int64(0).is_ok(), "Decimal128 → Int64");
    assert!(access.with_string(0, |_| Ok(())).is_ok(), "Decimal128 → String");

    // Decimal256 → String
    let array: ArrayRef = Arc::new(
        Decimal256Array::from(vec![i256::from(12345i64)])
            .with_precision_and_scale(10, 2)
            .unwrap()
    );
    let (mut access, arrays) = make_access(&array, hints_with_scale(2));
    access.set_cols(&arrays);
    access.set_row(0);
    assert!(access.with_string(0, |_| Ok(())).is_ok(), "Decimal256 → String");
    assert!(access.get_decimal256(0).is_ok(), "Decimal256 → Decimal256");
}

// =============================================================================
// Roundtrip Tests: Arrow → encode → decode → verify
// Tests the full path for all IngestedTypes using loop-based approach
// =============================================================================

use super::{
    ColumnContext, ColumnEncoding, FloatStorage, IngestedType, RowPackContext, RowPackDecoder,
    RowPackEncoder, RowPackScratch, StringEncoding, ROWPACK_VERSION,
};

/// Test case for roundtrip testing
struct RoundtripTestCase {
    name: &'static str,
    logical_type: IngestedType,
    encoding: ColumnEncoding,
    array: ArrayRef,
    hints: RowAccessHints,
    /// Function to verify decoded JSON value
    verify: fn(&serde_json::Value) -> bool,
}

/// All roundtrip test cases - single source of truth
fn all_roundtrip_test_cases() -> Vec<RoundtripTestCase> {
    vec![
        // ===== Signed Integers =====
        RoundtripTestCase {
            name: "Int8",
            logical_type: IngestedType::Int8,
            encoding: ColumnEncoding::Int { signed: true, varint: false, bits: 8 },
            array: Arc::new(Int8Array::from(vec![-42i8])),
            hints: default_hints(),
            verify: |v| v["val"].as_i64() == Some(-42),
        },
        RoundtripTestCase {
            name: "Int8_max",
            logical_type: IngestedType::Int8,
            encoding: ColumnEncoding::Int { signed: true, varint: false, bits: 8 },
            array: Arc::new(Int8Array::from(vec![127i8])),
            hints: default_hints(),
            verify: |v| v["val"].as_i64() == Some(127),
        },
        RoundtripTestCase {
            name: "Int8_min",
            logical_type: IngestedType::Int8,
            encoding: ColumnEncoding::Int { signed: true, varint: false, bits: 8 },
            array: Arc::new(Int8Array::from(vec![-128i8])),
            hints: default_hints(),
            verify: |v| v["val"].as_i64() == Some(-128),
        },
        RoundtripTestCase {
            name: "Int8_varint",
            logical_type: IngestedType::Int8,
            encoding: ColumnEncoding::Int { signed: true, varint: true, bits: 8 },
            array: Arc::new(Int8Array::from(vec![-42i8])),
            hints: default_hints(),
            verify: |v| v["val"].as_i64() == Some(-42),
        },
        RoundtripTestCase {
            name: "Int16",
            logical_type: IngestedType::Int16,
            encoding: ColumnEncoding::Int { signed: true, varint: false, bits: 16 },
            array: Arc::new(Int16Array::from(vec![-1000i16])),
            hints: default_hints(),
            verify: |v| v["val"].as_i64() == Some(-1000),
        },
        RoundtripTestCase {
            name: "Int16_max",
            logical_type: IngestedType::Int16,
            encoding: ColumnEncoding::Int { signed: true, varint: false, bits: 16 },
            array: Arc::new(Int16Array::from(vec![32767i16])),
            hints: default_hints(),
            verify: |v| v["val"].as_i64() == Some(32767),
        },
        RoundtripTestCase {
            name: "Int16_min",
            logical_type: IngestedType::Int16,
            encoding: ColumnEncoding::Int { signed: true, varint: false, bits: 16 },
            array: Arc::new(Int16Array::from(vec![-32768i16])),
            hints: default_hints(),
            verify: |v| v["val"].as_i64() == Some(-32768),
        },
        RoundtripTestCase {
            name: "Int32",
            logical_type: IngestedType::Int32,
            encoding: ColumnEncoding::Int { signed: true, varint: false, bits: 32 },
            array: Arc::new(Int32Array::from(vec![-100000i32])),
            hints: default_hints(),
            verify: |v| v["val"].as_i64() == Some(-100000),
        },
        RoundtripTestCase {
            name: "Int64",
            logical_type: IngestedType::Int64,
            encoding: ColumnEncoding::Int { signed: true, varint: false, bits: 64 },
            array: Arc::new(Int64Array::from(vec![-1000000000i64])),
            hints: default_hints(),
            verify: |v| v["val"].as_i64() == Some(-1000000000),
        },
        // ===== Unsigned Integers =====
        RoundtripTestCase {
            name: "UInt8",
            logical_type: IngestedType::UInt8,
            encoding: ColumnEncoding::Int { signed: false, varint: false, bits: 8 },
            array: Arc::new(UInt8Array::from(vec![200u8])),
            hints: default_hints(),
            verify: |v| v["val"].as_u64() == Some(200),
        },
        RoundtripTestCase {
            name: "UInt8_max",
            logical_type: IngestedType::UInt8,
            encoding: ColumnEncoding::Int { signed: false, varint: false, bits: 8 },
            array: Arc::new(UInt8Array::from(vec![255u8])),
            hints: default_hints(),
            verify: |v| v["val"].as_u64() == Some(255),
        },
        RoundtripTestCase {
            name: "UInt16",
            logical_type: IngestedType::UInt16,
            encoding: ColumnEncoding::Int { signed: false, varint: false, bits: 16 },
            array: Arc::new(UInt16Array::from(vec![50000u16])),
            hints: default_hints(),
            verify: |v| v["val"].as_u64() == Some(50000),
        },
        RoundtripTestCase {
            name: "UInt16_max",
            logical_type: IngestedType::UInt16,
            encoding: ColumnEncoding::Int { signed: false, varint: false, bits: 16 },
            array: Arc::new(UInt16Array::from(vec![65535u16])),
            hints: default_hints(),
            verify: |v| v["val"].as_u64() == Some(65535),
        },
        RoundtripTestCase {
            name: "UInt32",
            logical_type: IngestedType::UInt32,
            encoding: ColumnEncoding::Int { signed: false, varint: false, bits: 32 },
            array: Arc::new(UInt32Array::from(vec![3000000000u32])),
            hints: default_hints(),
            verify: |v| v["val"].as_u64() == Some(3000000000),
        },
        RoundtripTestCase {
            name: "UInt64",
            logical_type: IngestedType::UInt64,
            encoding: ColumnEncoding::Int { signed: false, varint: false, bits: 64 },
            array: Arc::new(UInt64Array::from(vec![10000000000u64])),
            hints: default_hints(),
            verify: |v| v["val"].as_u64() == Some(10000000000),
        },
        // ===== Floats =====
        RoundtripTestCase {
            name: "Float32",
            logical_type: IngestedType::Float32,
            encoding: ColumnEncoding::Float { storage: FloatStorage::F32 },
            array: Arc::new(Float32Array::from(vec![3.14f32])),
            hints: default_hints(),
            verify: |v| (v["val"].as_f64().unwrap() - 3.14).abs() < 0.001,
        },
        RoundtripTestCase {
            name: "Float64",
            logical_type: IngestedType::Float64,
            encoding: ColumnEncoding::Float { storage: FloatStorage::F64 },
            array: Arc::new(Float64Array::from(vec![2.718281828f64])),
            hints: default_hints(),
            verify: |v| (v["val"].as_f64().unwrap() - 2.718281828).abs() < 0.0000001,
        },
        // ===== Bool =====
        RoundtripTestCase {
            name: "Bool_true",
            logical_type: IngestedType::Bool,
            encoding: ColumnEncoding::Bool,
            array: Arc::new(BooleanArray::from(vec![true])),
            hints: default_hints(),
            verify: |v| v["val"].as_bool() == Some(true),
        },
        RoundtripTestCase {
            name: "Bool_false",
            logical_type: IngestedType::Bool,
            encoding: ColumnEncoding::Bool,
            array: Arc::new(BooleanArray::from(vec![false])),
            hints: default_hints(),
            verify: |v| v["val"].as_bool() == Some(false),
        },
        // ===== String =====
        RoundtripTestCase {
            name: "String",
            logical_type: IngestedType::String,
            encoding: ColumnEncoding::String { encoding: StringEncoding::Raw },
            array: Arc::new(StringArray::from(vec!["hello world"])),
            hints: default_hints(),
            verify: |v| v["val"].as_str() == Some("hello world"),
        },
        RoundtripTestCase {
            name: "String_empty",
            logical_type: IngestedType::String,
            encoding: ColumnEncoding::String { encoding: StringEncoding::Raw },
            array: Arc::new(StringArray::from(vec![""])),
            hints: default_hints(),
            verify: |v| v["val"].as_str() == Some(""),
        },
        // ===== Bytes =====
        RoundtripTestCase {
            name: "Bytes",
            logical_type: IngestedType::Bytes,
            encoding: ColumnEncoding::Bytes { fixed_len: None },
            array: Arc::new(BinaryArray::from_vec(vec![b"binary data"])),
            hints: default_hints(),
            verify: |v| v["val"].as_str().is_some(), // Decoded as base64 or hex string
        },
        // ===== UUID =====
        RoundtripTestCase {
            name: "Uuid",
            logical_type: IngestedType::Uuid,
            encoding: ColumnEncoding::Uuid,
            array: Arc::new(FixedSizeBinaryArray::from(vec![Some(&[
                0x55, 0x0e, 0x84, 0x00, 0xe2, 0x9b, 0x41, 0xd4,
                0xa7, 0x16, 0x44, 0x66, 0x55, 0x44, 0x00, 0x00,
            ][..])])),
            hints: default_hints(),
            verify: |v| v["val"].as_str().map(|s| s.contains("550e8400")).unwrap_or(false),
        },
        // ===== UUID256 =====
        RoundtripTestCase {
            name: "Uuid256",
            logical_type: IngestedType::Uuid256,
            encoding: ColumnEncoding::Uuid256,
            array: Arc::new(FixedSizeBinaryArray::from(vec![Some(&[0xAAu8; 32][..])])),
            hints: default_hints(),
            verify: |v| v["val"].as_str().map(|s| s.to_lowercase().contains("aa")).unwrap_or(false),
        },
        // ===== IP =====
        RoundtripTestCase {
            name: "Ip_v6",
            logical_type: IngestedType::Ip,
            encoding: ColumnEncoding::Ip,
            array: Arc::new(FixedSizeBinaryArray::from(vec![Some(&[0u8; 16][..])])),
            hints: default_hints(),
            verify: |v| v["val"].as_str().is_some(),
        },
        // ===== Date =====
        RoundtripTestCase {
            name: "Date",
            logical_type: IngestedType::Date,
            encoding: ColumnEncoding::Date { varint: false, bits: 32 },
            array: Arc::new(Date32Array::from(vec![19000i32])), // ~2022-01-01
            hints: default_hints(),
            // Date may decode as number (days since epoch) or string
            verify: |v| v["val"].as_i64() == Some(19000) || v["val"].as_str().map(|s| s.contains("2022")).unwrap_or(false),
        },
        // ===== Timestamp =====
        RoundtripTestCase {
            name: "Timestamp",
            logical_type: IngestedType::Timestamp,
            encoding: ColumnEncoding::Timestamp,
            array: Arc::new(TimestampSecondArray::from(vec![1640000000i64])), // 2021-12-20
            hints: default_hints(),
            verify: |v| v["val"].as_str().map(|s| s.contains("2021")).unwrap_or(false),
        },
        // ===== Float Arrays =====
        RoundtripTestCase {
            name: "Float32Array",
            logical_type: IngestedType::Float32Array,
            encoding: ColumnEncoding::FloatArray { storage: FloatStorage::F32 },
            array: make_list_float32_array(),
            hints: default_hints(),
            verify: |v| v["val"].as_array().map(|a| a.len() == 3).unwrap_or(false),
        },
        RoundtripTestCase {
            name: "Float64Array",
            logical_type: IngestedType::Float64Array,
            encoding: ColumnEncoding::FloatArray { storage: FloatStorage::F64 },
            array: make_list_float64_array(),
            hints: default_hints(),
            verify: |v| v["val"].as_array().map(|a| a.len() == 3).unwrap_or(false),
        },
        RoundtripTestCase {
            name: "Float16Array",
            logical_type: IngestedType::Float16Array,
            encoding: ColumnEncoding::FloatArray { storage: FloatStorage::F16 },
            array: make_list_float32_array(), // Use f32 array, will be stored as f16
            hints: default_hints(),
            verify: |v| v["val"].as_array().map(|a| a.len() == 3).unwrap_or(false),
        },
        // ===== Decimals =====
        RoundtripTestCase {
            name: "Decimal128",
            logical_type: IngestedType::Decimal128,
            encoding: ColumnEncoding::Decimal {
                precision: 10,
                scale: 2,
                bytes: 16,
                storage: super::DecimalStorage::Raw,
            },
            array: Arc::new(
                Decimal128Array::from(vec![12345i128])
                    .with_precision_and_scale(10, 2)
                    .unwrap()
            ),
            hints: hints_with_scale(2),
            verify: |v| v["val"].as_str().map(|s| s.contains("123")).unwrap_or(false)
                     || v["val"].as_f64().map(|f| (f - 123.45).abs() < 0.01).unwrap_or(false),
        },
        RoundtripTestCase {
            name: "Decimal256",
            logical_type: IngestedType::Decimal256,
            encoding: ColumnEncoding::Decimal {
                precision: 10,
                scale: 2,
                bytes: 32,
                storage: super::DecimalStorage::Raw,
            },
            array: Arc::new(
                Decimal256Array::from(vec![i256::from(12345i64)])
                    .with_precision_and_scale(10, 2)
                    .unwrap()
            ),
            hints: hints_with_scale(2),
            verify: |v| v["val"].as_str().map(|s| s.contains("123")).unwrap_or(false)
                     || v["val"].as_f64().map(|f| (f - 123.45).abs() < 0.01).unwrap_or(false),
        },
    ]
}

/// Main roundtrip test - loops through all test cases
#[test]
fn test_full_roundtrip_matrix() {
    let test_cases = all_roundtrip_test_cases();
    let mut failures = Vec::new();
    let mut success_count = 0;

    for case in &test_cases {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "val".to_string(),
                logical_type: case.logical_type.clone(),
                encoding: case.encoding.clone(),
            }],
            estimated_row_bytes: 0,
        };

        let encoder = match RowPackEncoder::new(ctx.clone()) {
            Ok(e) => e,
            Err(e) => {
                failures.push(format!("{}: encoder creation failed: {}", case.name, e));
                continue;
            }
        };
        let decoder = match RowPackDecoder::new(ctx.clone()) {
            Ok(d) => d,
            Err(e) => {
                failures.push(format!("{}: decoder creation failed: {}", case.name, e));
                continue;
            }
        };
        let mut scratch = RowPackScratch::new(&ctx);

        let (mut access, arrays) = make_access(&case.array, case.hints.clone());
        access.set_cols(&arrays);
        access.set_row(0);

        // Encode
        let mut encoded = Vec::new();
        if let Err(e) = encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch) {
            failures.push(format!("{}: encode failed: {}", case.name, e));
            continue;
        }

        // Decode
        let json = match decoder.decode_json(&encoded) {
            Ok(j) => j,
            Err(e) => {
                failures.push(format!("{}: decode failed: {}", case.name, e));
                continue;
            }
        };

        // Verify
        if (case.verify)(&json) {
            success_count += 1;
        } else {
            failures.push(format!("{}: verification failed, got {:?}", case.name, json));
        }
    }

    if !failures.is_empty() {
        panic!(
            "Roundtrip matrix failures ({} failures, {} successes):\n{}",
            failures.len(),
            success_count,
            failures.join("\n")
        );
    }

    // Verify coverage - should have at least 23 (one per IngestedType)
    assert!(
        success_count >= 23,
        "Should test at least 23 roundtrip cases (one per IngestedType), got {}",
        success_count
    );
}

/// Verify roundtrip test coverage
#[test]
fn test_roundtrip_coverage() {
    let cases = all_roundtrip_test_cases();

    // Check we have tests for key types
    let type_names: Vec<&str> = cases.iter().map(|c| c.name).collect();

    assert!(type_names.iter().any(|n| n.starts_with("Int8")), "Missing Int8 tests");
    assert!(type_names.iter().any(|n| n.starts_with("Int16")), "Missing Int16 tests");
    assert!(type_names.iter().any(|n| n.starts_with("Int32")), "Missing Int32 tests");
    assert!(type_names.iter().any(|n| n.starts_with("Int64")), "Missing Int64 tests");
    assert!(type_names.iter().any(|n| n.starts_with("UInt8")), "Missing UInt8 tests");
    assert!(type_names.iter().any(|n| n.starts_with("UInt16")), "Missing UInt16 tests");
    assert!(type_names.iter().any(|n| n.starts_with("UInt32")), "Missing UInt32 tests");
    assert!(type_names.iter().any(|n| n.starts_with("UInt64")), "Missing UInt64 tests");
    assert!(type_names.iter().any(|n| n.starts_with("Float32") && !n.contains("Array")), "Missing Float32 tests");
    assert!(type_names.iter().any(|n| n.starts_with("Float64") && !n.contains("Array")), "Missing Float64 tests");
    assert!(type_names.iter().any(|n| n.starts_with("Bool")), "Missing Bool tests");
    assert!(type_names.iter().any(|n| n.starts_with("String")), "Missing String tests");
    assert!(type_names.iter().any(|n| n.starts_with("Bytes")), "Missing Bytes tests");
    assert!(type_names.iter().any(|n| n.starts_with("Uuid") && !n.contains("256")), "Missing Uuid tests");
    assert!(type_names.iter().any(|n| n.starts_with("Uuid256")), "Missing Uuid256 tests");
    assert!(type_names.iter().any(|n| n.starts_with("Ip")), "Missing Ip tests");
    assert!(type_names.iter().any(|n| n.starts_with("Date")), "Missing Date tests");
    assert!(type_names.iter().any(|n| n.starts_with("Timestamp")), "Missing Timestamp tests");
    assert!(type_names.iter().any(|n| n.contains("Float32Array")), "Missing Float32Array tests");
    assert!(type_names.iter().any(|n| n.contains("Float64Array")), "Missing Float64Array tests");
    assert!(type_names.iter().any(|n| n.contains("Float16Array")), "Missing Float16Array tests");
    assert!(type_names.iter().any(|n| n.contains("Decimal128")), "Missing Decimal128 tests");
    assert!(type_names.iter().any(|n| n.contains("Decimal256")), "Missing Decimal256 tests");

    // Test varint path
    assert!(type_names.iter().any(|n| n.contains("varint")), "Missing varint tests");

    println!("Roundtrip coverage: {} test cases", cases.len());
}

// =============================================================================
// Value Verification Tests
// These tests verify actual VALUES, not just success/error
// =============================================================================

#[test]
fn test_integer_value_verification() {
    // Test that integer conversions produce correct values
    let test_cases: Vec<(ArrayRef, i64)> = vec![
        (Arc::new(Int8Array::from(vec![42i8])), 42),
        (Arc::new(Int8Array::from(vec![-42i8])), -42),
        (Arc::new(Int16Array::from(vec![1000i16])), 1000),
        (Arc::new(Int32Array::from(vec![-100000i32])), -100000),
        (Arc::new(Int64Array::from(vec![9999999999i64])), 9999999999),
        (Arc::new(UInt8Array::from(vec![255u8])), 255),
        (Arc::new(UInt16Array::from(vec![65535u16])), 65535),
        (Arc::new(UInt32Array::from(vec![4000000000u32])), 4000000000),
    ];

    for (array, expected) in test_cases {
        let (mut access, arrays) = make_access(&array, default_hints());
        access.set_cols(&arrays);
        access.set_row(0);

        let actual = access.get_int64(0).expect("should succeed").expect("should not be null");
        assert_eq!(actual, expected, "Value mismatch for {:?}", array.data_type());
    }
}

#[test]
fn test_float_to_int_truncation() {
    // Test that Float → Int truncates toward zero (like SQL CAST)
    let test_cases: Vec<(f64, i64)> = vec![
        (3.14, 3),
        (3.99, 3),      // Truncate, not round
        (-3.14, -3),
        (-3.99, -3),    // Truncate toward zero
        (0.0, 0),
        (-0.0, 0),
        (100.001, 100),
    ];

    for (float_val, expected_int) in test_cases {
        let array: ArrayRef = Arc::new(Float64Array::from(vec![float_val]));
        let (mut access, arrays) = make_access(&array, default_hints());
        access.set_cols(&arrays);
        access.set_row(0);

        let actual = access.get_int64(0).expect("should succeed").expect("should not be null");
        assert_eq!(actual, expected_int, "Float {} should truncate to {}", float_val, expected_int);
    }
}

#[test]
fn test_string_to_int_parsing() {
    // Test that String → Int parses correctly
    let test_cases: Vec<(&str, i64)> = vec![
        ("42", 42),
        ("-42", -42),
        ("0", 0),
        ("9999999999", 9999999999),
        ("  123  ", 123),  // Whitespace handling
    ];

    for (str_val, expected) in test_cases {
        let array: ArrayRef = Arc::new(StringArray::from(vec![str_val]));
        let (mut access, arrays) = make_access(&array, default_hints());
        access.set_cols(&arrays);
        access.set_row(0);

        let actual = access.get_int64(0).expect("should succeed").expect("should not be null");
        assert_eq!(actual, expected, "String '{}' should parse to {}", str_val, expected);
    }
}

// =============================================================================
// Integer Boundary Tests
// =============================================================================

#[test]
fn test_int8_boundaries() {
    let array: ArrayRef = Arc::new(Int8Array::from(vec![i8::MIN, i8::MAX]));
    let (mut access, arrays) = make_access(&array, default_hints());
    access.set_cols(&arrays);

    access.set_row(0);
    let min_val = access.get_int64(0).expect("should succeed").expect("not null");
    assert_eq!(min_val, i8::MIN as i64, "i8::MIN should be preserved");

    access.set_row(1);
    let max_val = access.get_int64(0).expect("should succeed").expect("not null");
    assert_eq!(max_val, i8::MAX as i64, "i8::MAX should be preserved");
}

#[test]
fn test_int16_boundaries() {
    let array: ArrayRef = Arc::new(Int16Array::from(vec![i16::MIN, i16::MAX]));
    let (mut access, arrays) = make_access(&array, default_hints());
    access.set_cols(&arrays);

    access.set_row(0);
    let min_val = access.get_int64(0).expect("should succeed").expect("not null");
    assert_eq!(min_val, i16::MIN as i64);

    access.set_row(1);
    let max_val = access.get_int64(0).expect("should succeed").expect("not null");
    assert_eq!(max_val, i16::MAX as i64);
}

#[test]
fn test_int32_boundaries() {
    let array: ArrayRef = Arc::new(Int32Array::from(vec![i32::MIN, i32::MAX]));
    let (mut access, arrays) = make_access(&array, default_hints());
    access.set_cols(&arrays);

    access.set_row(0);
    let min_val = access.get_int64(0).expect("should succeed").expect("not null");
    assert_eq!(min_val, i32::MIN as i64);

    access.set_row(1);
    let max_val = access.get_int64(0).expect("should succeed").expect("not null");
    assert_eq!(max_val, i32::MAX as i64);
}

#[test]
fn test_int64_boundaries() {
    let array: ArrayRef = Arc::new(Int64Array::from(vec![i64::MIN, i64::MAX]));
    let (mut access, arrays) = make_access(&array, default_hints());
    access.set_cols(&arrays);

    access.set_row(0);
    let min_val = access.get_int64(0).expect("should succeed").expect("not null");
    assert_eq!(min_val, i64::MIN);

    access.set_row(1);
    let max_val = access.get_int64(0).expect("should succeed").expect("not null");
    assert_eq!(max_val, i64::MAX);
}

#[test]
fn test_uint64_max() {
    let array: ArrayRef = Arc::new(UInt64Array::from(vec![u64::MAX]));
    let (mut access, arrays) = make_access(&array, default_hints());
    access.set_cols(&arrays);
    access.set_row(0);

    let val = access.get_uint64(0).expect("should succeed").expect("not null");
    assert_eq!(val, u64::MAX);
}

// =============================================================================
// Null Handling Tests
// =============================================================================

#[test]
fn test_null_int64() {
    let array: ArrayRef = Arc::new(Int64Array::from(vec![Some(42), None, Some(100)]));
    let (mut access, arrays) = make_access(&array, default_hints());
    access.set_cols(&arrays);

    access.set_row(0);
    assert_eq!(access.get_int64(0).unwrap(), Some(42));

    access.set_row(1);
    assert_eq!(access.get_int64(0).unwrap(), None, "Null should return None");

    access.set_row(2);
    assert_eq!(access.get_int64(0).unwrap(), Some(100));
}

#[test]
fn test_null_string() {
    let array: ArrayRef = Arc::new(StringArray::from(vec![Some("hello"), None, Some("world")]));
    let (mut access, arrays) = make_access(&array, default_hints());
    access.set_cols(&arrays);

    access.set_row(0);
    let result = access.with_string(0, |s| Ok(s.to_string())).unwrap();
    assert_eq!(result, Some("hello".to_string()));

    access.set_row(1);
    let result = access.with_string(0, |s| Ok(s.to_string())).unwrap();
    assert_eq!(result, None, "Null string should return None");
}

#[test]
fn test_null_float64() {
    let array: ArrayRef = Arc::new(Float64Array::from(vec![Some(3.14), None]));
    let (mut access, arrays) = make_access(&array, default_hints());
    access.set_cols(&arrays);

    access.set_row(0);
    let val = access.get_float64(0).unwrap();
    assert!(val.is_some());
    assert!((val.unwrap() - 3.14).abs() < 0.001);

    access.set_row(1);
    assert_eq!(access.get_float64(0).unwrap(), None);
}

#[test]
fn test_null_bool() {
    let array: ArrayRef = Arc::new(BooleanArray::from(vec![Some(true), None, Some(false)]));
    let (mut access, arrays) = make_access(&array, default_hints());
    access.set_cols(&arrays);

    access.set_row(0);
    assert_eq!(access.get_bool(0).unwrap(), Some(true));

    access.set_row(1);
    assert_eq!(access.get_bool(0).unwrap(), None);

    access.set_row(2);
    assert_eq!(access.get_bool(0).unwrap(), Some(false));
}

// =============================================================================
// Float Special Values Tests
// =============================================================================

#[test]
fn test_float_infinity() {
    let array: ArrayRef = Arc::new(Float64Array::from(vec![
        f64::INFINITY,
        f64::NEG_INFINITY,
    ]));
    let (mut access, arrays) = make_access(&array, default_hints());
    access.set_cols(&arrays);

    access.set_row(0);
    let val = access.get_float64(0).unwrap().unwrap();
    assert!(val.is_infinite() && val.is_sign_positive());

    access.set_row(1);
    let val = access.get_float64(0).unwrap().unwrap();
    assert!(val.is_infinite() && val.is_sign_negative());
}

#[test]
fn test_float_nan() {
    let array: ArrayRef = Arc::new(Float64Array::from(vec![f64::NAN]));
    let (mut access, arrays) = make_access(&array, default_hints());
    access.set_cols(&arrays);
    access.set_row(0);

    let val = access.get_float64(0).unwrap().unwrap();
    assert!(val.is_nan(), "NaN should be preserved");
}

#[test]
fn test_float_negative_zero() {
    let array: ArrayRef = Arc::new(Float64Array::from(vec![-0.0f64]));
    let (mut access, arrays) = make_access(&array, default_hints());
    access.set_cols(&arrays);
    access.set_row(0);

    let val = access.get_float64(0).unwrap().unwrap();
    // -0.0 == 0.0 in IEEE 754
    assert_eq!(val, 0.0);
}

// =============================================================================
// String Edge Cases
// =============================================================================

#[test]
fn test_empty_string() {
    let array: ArrayRef = Arc::new(StringArray::from(vec![""]));
    let (mut access, arrays) = make_access(&array, default_hints());
    access.set_cols(&arrays);
    access.set_row(0);

    let result = access.with_string(0, |s| Ok(s.to_string())).unwrap();
    assert_eq!(result, Some("".to_string()));
}

#[test]
fn test_string_parse_failure() {
    // Unparseable string should error when converting to Int64
    let array: ArrayRef = Arc::new(StringArray::from(vec!["not_a_number"]));
    let (mut access, arrays) = make_access(&array, default_hints());
    access.set_cols(&arrays);
    access.set_row(0);

    let result = access.get_int64(0);
    assert!(result.is_err(), "Unparseable string should error");
}

#[test]
fn test_string_overflow() {
    // String with number too large for i64
    let array: ArrayRef = Arc::new(StringArray::from(vec!["99999999999999999999999"]));
    let (mut access, arrays) = make_access(&array, default_hints());
    access.set_cols(&arrays);
    access.set_row(0);

    let result = access.get_int64(0);
    assert!(result.is_err(), "Overflow string should error");
}

// =============================================================================
// Boolean Conversion Tests
// =============================================================================

#[test]
fn test_bool_to_int() {
    let array: ArrayRef = Arc::new(BooleanArray::from(vec![true, false]));
    let (mut access, arrays) = make_access(&array, default_hints());
    access.set_cols(&arrays);

    access.set_row(0);
    let val = access.get_int64(0).unwrap().unwrap();
    assert_eq!(val, 1, "true should convert to 1");

    access.set_row(1);
    let val = access.get_int64(0).unwrap().unwrap();
    assert_eq!(val, 0, "false should convert to 0");
}

// =============================================================================
// Sign Preservation Tests
// =============================================================================

#[test]
fn test_negative_int_preservation() {
    // Ensure negative values are preserved through conversions
    let test_cases: Vec<(ArrayRef, i64)> = vec![
        (Arc::new(Int8Array::from(vec![-1i8])), -1),
        (Arc::new(Int8Array::from(vec![-128i8])), -128),
        (Arc::new(Int16Array::from(vec![-32768i16])), -32768),
        (Arc::new(Int32Array::from(vec![-2147483648i32])), -2147483648),
        (Arc::new(Int64Array::from(vec![-9223372036854775808i64])), i64::MIN),
    ];

    for (array, expected) in test_cases {
        let (mut access, arrays) = make_access(&array, default_hints());
        access.set_cols(&arrays);
        access.set_row(0);

        let actual = access.get_int64(0).unwrap().unwrap();
        assert_eq!(actual, expected, "Negative value should be preserved for {:?}", array.data_type());
    }
}
