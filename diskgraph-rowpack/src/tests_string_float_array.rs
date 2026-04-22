//! Tests for string → float32_array parsing.
//!
//! When a parquet column stores embeddings as JSON-encoded strings like
//! "[-0.19, 0.27, 0.55]", diskgraph should parse these into float32 arrays
//! if the config specifies `ingested_dtype: float32_array`.
//!
//! Behavior:
//! - Valid float strings parse to Vec<f32>
//! - Null strings produce None (treated as null row)
//! - Empty strings / empty brackets produce empty arrays
//! - Malformed strings produce hard errors (Err), not silent skips

use super::*;
use crate::access::{RowAccess, RowAccessHints};
use arrow_array::*;
use std::sync::Arc;

// =============================================================================
// Helper: build RowAccess from a StringArray
// =============================================================================

fn access_from_strings(values: Vec<Option<&str>>) -> (RowAccess<'static>, Vec<ArrayRef>) {
    let array: ArrayRef = Arc::new(StringArray::from(values));
    let arrays = vec![array];
    let stats = crate::RowAccessStats::default();
    let access = RowAccess::new(vec![RowAccessHints::default()], stats);
    (access, arrays)
}

// =============================================================================
// Unit tests: parse_string_as_float32_array
// =============================================================================

#[test]
fn test_parse_basic_bracket_format() {
    let result = crate::access::parse_string_as_float32_array("[1.0, 2.0, 3.0]");
    let v = result.unwrap();
    assert_eq!(v.len(), 3);
    assert!((v[0] - 1.0).abs() < 1e-6);
    assert!((v[1] - 2.0).abs() < 1e-6);
    assert!((v[2] - 3.0).abs() < 1e-6);
}

#[test]
fn test_parse_no_brackets() {
    let result = crate::access::parse_string_as_float32_array("1.0, 2.0, 3.0");
    let v = result.unwrap();
    assert_eq!(v.len(), 3);
    assert!((v[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_parse_scientific_notation() {
    let result =
        crate::access::parse_string_as_float32_array("[-1.926282793283463e-01, 5.763512849807739e-01]");
    let v = result.unwrap();
    assert_eq!(v.len(), 2);
    assert!((v[0] - (-0.192_628_28_f32)).abs() < 1e-6);
    assert!((v[1] - 0.576_351_3_f32).abs() < 1e-6);
}

#[test]
fn test_parse_negative_values() {
    let result = crate::access::parse_string_as_float32_array("[-1.5, -2.5, -3.5]");
    let v = result.unwrap();
    assert_eq!(v.len(), 3);
    assert!((v[0] - (-1.5)).abs() < 1e-6);
}

#[test]
fn test_parse_empty_brackets() {
    let result = crate::access::parse_string_as_float32_array("[]");
    let v = result.unwrap();
    assert!(v.is_empty());
}

#[test]
fn test_parse_empty_string() {
    let result = crate::access::parse_string_as_float32_array("");
    let v = result.unwrap();
    assert!(v.is_empty());
}

#[test]
fn test_parse_whitespace_only() {
    let result = crate::access::parse_string_as_float32_array("  ");
    let v = result.unwrap();
    assert!(v.is_empty());
}

#[test]
fn test_parse_single_value() {
    let result = crate::access::parse_string_as_float32_array("[42.5]");
    let v = result.unwrap();
    assert_eq!(v.len(), 1);
    assert!((v[0] - 42.5).abs() < 1e-6);
}

#[test]
fn test_parse_extra_whitespace() {
    let result = crate::access::parse_string_as_float32_array("[  1.0 ,  2.0 ,  3.0  ]");
    let v = result.unwrap();
    assert_eq!(v.len(), 3);
    assert!((v[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_parse_high_dimensional() {
    // Simulate a real embedding (~700 dims)
    let values: Vec<String> = (0..700).map(|i| format!("{:.6e}", i as f64 * 0.001)).collect();
    let s = format!("[{}]", values.join(","));
    let result = crate::access::parse_string_as_float32_array(&s);
    let v = result.unwrap();
    assert_eq!(v.len(), 700);
    assert!((v[0]).abs() < 1e-6);
    assert!((v[699] - 0.699f32).abs() < 1e-3);
}

// =============================================================================
// Error cases: malformed strings produce hard errors
// =============================================================================

#[test]
fn test_parse_garbage_string_is_error() {
    let result = crate::access::parse_string_as_float32_array("hello world");
    assert!(result.is_err(), "garbage string should produce hard error");
}

#[test]
fn test_parse_partial_garbage_is_error() {
    let result = crate::access::parse_string_as_float32_array("[1.0, abc, 3.0]");
    assert!(result.is_err(), "partially invalid string should produce hard error");
}

#[test]
fn test_parse_nested_brackets_is_error() {
    let result = crate::access::parse_string_as_float32_array("[[1.0, 2.0], [3.0]]");
    assert!(result.is_err(), "nested brackets should produce hard error");
}

// =============================================================================
// Integration: with_float32_array on string columns
// =============================================================================

#[test]
fn test_with_float32_array_string_column() -> anyhow::Result<()> {
    let (mut access, arrays) = access_from_strings(vec![Some("[1.0, 2.0, 3.0]")]);
    access.set_cols(&arrays);
    access.set_row(0);

    let result = access.with_float32_array(0, |values| {
        assert_eq!(values.len(), 3);
        assert!((values[0] - 1.0).abs() < 1e-6);
        assert!((values[1] - 2.0).abs() < 1e-6);
        assert!((values[2] - 3.0).abs() < 1e-6);
        Ok(values.to_vec())
    })?;
    assert!(result.is_some());
    Ok(())
}

#[test]
fn test_with_float32_array_null_string() -> anyhow::Result<()> {
    let (mut access, arrays) = access_from_strings(vec![None]);
    access.set_cols(&arrays);
    access.set_row(0);

    let result = access.with_float32_array(0, |_values| Ok(()))?;
    assert!(result.is_none(), "null string should produce None");
    Ok(())
}

#[test]
fn test_with_float32_array_invalid_string_is_error() {
    let (mut access, arrays) = access_from_strings(vec![Some("not a float array")]);
    access.set_cols(&arrays);
    access.set_row(0);

    let result = access.with_float32_array(0, |_values| Ok(()));
    assert!(result.is_err(), "invalid string should produce hard error from with_float32_array");
}

#[test]
fn test_with_float32_array_scientific_notation_string() -> anyhow::Result<()> {
    let (mut access, arrays) = access_from_strings(vec![Some(
        "[-1.926282793283463e-01,-2.700167298316956e-01]",
    )]);
    access.set_cols(&arrays);
    access.set_row(0);

    let result = access.with_float32_array(0, |values| {
        assert_eq!(values.len(), 2);
        Ok(values.to_vec())
    })?;
    assert!(result.is_some());
    let v = result.unwrap();
    assert!((v[0] - (-0.19262828f32)).abs() < 1e-5);
    Ok(())
}

// =============================================================================
// Integration: float_list_view on string columns
// =============================================================================

#[test]
fn test_float_list_view_string_column() -> anyhow::Result<()> {
    let (mut access, arrays) = access_from_strings(vec![Some("[1.0, 2.0, 3.0]")]);
    access.set_cols(&arrays);
    access.set_row(0);

    let view = access.float_list_view(0)?;
    assert!(view.is_some());
    let view = view.unwrap();
    assert_eq!(view.len(), 3);

    // Verify values via for_each_f64
    let mut values = Vec::new();
    view.for_each_f64(|v| values.push(v));
    assert_eq!(values.len(), 3);
    assert!((values[0] - 1.0).abs() < 1e-6);
    assert!((values[1] - 2.0).abs() < 1e-6);
    assert!((values[2] - 3.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_float_list_view_null_string() -> anyhow::Result<()> {
    let (mut access, arrays) = access_from_strings(vec![None]);
    access.set_cols(&arrays);
    access.set_row(0);

    let view = access.float_list_view(0)?;
    assert!(view.is_none(), "null string should produce None from float_list_view");
    Ok(())
}

#[test]
fn test_float_list_view_invalid_string_is_error() {
    let (mut access, arrays) = access_from_strings(vec![Some("garbage")]);
    access.set_cols(&arrays);
    access.set_row(0);

    let result = access.float_list_view(0);
    assert!(result.is_err(), "invalid string should produce hard error from float_list_view");
}

// =============================================================================
// Full roundtrip: string → encode → decode (via encode_row_from_access)
// =============================================================================

#[test]
fn test_roundtrip_string_to_float32_array() -> anyhow::Result<()> {
    // Build a RowPackContext with a single Float32Array column
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "embedding".to_string(),
            logical_type: IngestedType::Float32Array,
            encoding: ColumnEncoding::FloatArray {
                storage: FloatStorage::F32,
            },
        }],
        estimated_row_bytes: 0,
    };

    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    // Create a StringArray with a float-list string
    let array: ArrayRef = Arc::new(StringArray::from(vec![Some("[1.5, -2.5, 3.5]")]));
    let arrays = vec![array];
    let stats = crate::RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);
    access.set_row(0);

    // Encode
    let mut encoded = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;

    // Decode to JSON and verify
    let json = decoder.decode_json(&encoded)?;
    let arr = json["embedding"].as_array().expect("should decode as array");
    assert_eq!(arr.len(), 3);
    assert!((arr[0].as_f64().unwrap() - 1.5).abs() < 1e-3);
    assert!((arr[1].as_f64().unwrap() - (-2.5)).abs() < 1e-3);
    assert!((arr[2].as_f64().unwrap() - 3.5).abs() < 1e-3);

    Ok(())
}

#[test]
fn test_roundtrip_string_to_float32_array_null() -> anyhow::Result<()> {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "embedding".to_string(),
            logical_type: IngestedType::Float32Array,
            encoding: ColumnEncoding::FloatArray {
                storage: FloatStorage::F32,
            },
        }],
        estimated_row_bytes: 0,
    };

    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array: ArrayRef = Arc::new(StringArray::from(vec![None::<&str>]));
    let arrays = vec![array];
    let stats = crate::RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut encoded = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;

    let json = decoder.decode_json(&encoded)?;
    assert!(
        json["embedding"].is_null(),
        "null string should encode as null float array"
    );

    Ok(())
}

#[test]
fn test_roundtrip_string_to_float32_array_invalid_is_error() {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "embedding".to_string(),
            logical_type: IngestedType::Float32Array,
            encoding: ColumnEncoding::FloatArray {
                storage: FloatStorage::F32,
            },
        }],
        estimated_row_bytes: 0,
    };

    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let mut scratch = RowPackScratch::new(&ctx);

    let array: ArrayRef = Arc::new(StringArray::from(vec![Some("not a float array")]));
    let arrays = vec![array];
    let stats = crate::RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut encoded = Vec::new();
    let result = encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch);
    assert!(
        result.is_err(),
        "encoding invalid string as float array should be a hard error"
    );
}

#[test]
fn test_roundtrip_string_to_float16_storage() -> anyhow::Result<()> {
    // Verify string → Float32Array with F16 storage also works
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "embedding".to_string(),
            logical_type: IngestedType::Float32Array,
            encoding: ColumnEncoding::FloatArray {
                storage: FloatStorage::F16,
            },
        }],
        estimated_row_bytes: 0,
    };

    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array: ArrayRef = Arc::new(StringArray::from(vec![Some("[1.0, 2.0, 3.0]")]));
    let arrays = vec![array];
    let stats = crate::RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut encoded = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;

    let json = decoder.decode_json(&encoded)?;
    let arr = json["embedding"].as_array().expect("should decode as array");
    assert_eq!(arr.len(), 3);
    // F16 has lower precision
    assert!((arr[0].as_f64().unwrap() - 1.0).abs() < 0.01);
    assert!((arr[1].as_f64().unwrap() - 2.0).abs() < 0.01);
    assert!((arr[2].as_f64().unwrap() - 3.0).abs() < 0.01);

    Ok(())
}

// =============================================================================
// Integration: multiple rows with mixed null/valid strings
// =============================================================================

#[test]
fn test_roundtrip_string_multiple_rows() -> anyhow::Result<()> {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "embedding".to_string(),
            logical_type: IngestedType::Float32Array,
            encoding: ColumnEncoding::FloatArray {
                storage: FloatStorage::F32,
            },
        }],
        estimated_row_bytes: 0,
    };

    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array: ArrayRef = Arc::new(StringArray::from(vec![
        Some("[1.0, 2.0]"),
        None,
        Some("[3.0]"),
        Some("[]"),
    ]));
    let arrays = vec![array];
    let stats = crate::RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);

    let mut encoded_rows = Vec::new();
    for row in 0..4 {
        access.set_row(row);
        let mut encoded = Vec::new();
        encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;
        encoded_rows.push(encoded);
    }

    // Row 0: [1.0, 2.0]
    let json = decoder.decode_json(&encoded_rows[0])?;
    let arr = json["embedding"].as_array().unwrap();
    assert_eq!(arr.len(), 2);

    // Row 1: null
    let json = decoder.decode_json(&encoded_rows[1])?;
    assert!(json["embedding"].is_null());

    // Row 2: [3.0]
    let json = decoder.decode_json(&encoded_rows[2])?;
    let arr = json["embedding"].as_array().unwrap();
    assert_eq!(arr.len(), 1);

    // Row 3: []
    let json = decoder.decode_json(&encoded_rows[3])?;
    let arr = json["embedding"].as_array().unwrap();
    assert!(arr.is_empty());

    Ok(())
}

// =============================================================================
// Type matrix: verify String → Float32Array/Float16Array/Float64Array are accepted
// =============================================================================

#[test]
fn test_type_matrix_string_to_float32_array() -> anyhow::Result<()> {
    let (mut access, arrays) = access_from_strings(vec![Some("[1.0, 2.0, 3.0]")]);
    access.set_cols(&arrays);
    access.set_row(0);

    // Float32Array target should work
    let result = access.with_float32_array(0, |v| {
        assert_eq!(v.len(), 3);
        Ok(())
    });
    assert!(result.is_ok(), "String → Float32Array should succeed");
    Ok(())
}

#[test]
fn test_type_matrix_string_to_float_list_view() -> anyhow::Result<()> {
    let (mut access, arrays) = access_from_strings(vec![Some("[1.0, 2.0, 3.0]")]);
    access.set_cols(&arrays);
    access.set_row(0);

    // float_list_view should work
    let view = access.float_list_view(0)?;
    assert!(view.is_some());
    assert_eq!(view.unwrap().len(), 3);
    Ok(())
}

// =============================================================================
// Statistics collection: verify observe_array_f32 works with string source
// =============================================================================

#[test]
fn test_stats_collection_string_source() -> anyhow::Result<()> {
    let (mut access, arrays) = access_from_strings(vec![Some("[1.0, 2.0, 3.0]")]);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut count = 0usize;
    let mut sum = 0.0f32;
    access.with_float32_array(0, |values| {
        count = values.len();
        sum = values.iter().sum();
        Ok(())
    })?;
    assert_eq!(count, 3);
    assert!((sum - 6.0).abs() < 1e-6);
    Ok(())
}
