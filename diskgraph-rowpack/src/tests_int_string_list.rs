//! Round-trip tests for Int64Array and StringArray IngestedTypes.
//!
//! Tests cover:
//! - Basic round-trip (encode → decode via decode_json)
//! - Empty lists
//! - Single-element lists
//! - NULL rows
//! - Variable-length multi-row (pointer drift test)
//! - StringArray with OOV token (tag-2 fallback)
//! - StringArray with mixed dict-hit and OOV in same list

use super::*;
use arrow_array::{
    builder::{Int64Builder, ListBuilder, StringBuilder},
    ArrayRef,
};
use std::sync::Arc;

// =============================================================================
// Helpers
// =============================================================================

/// Build a RowPackContext with a single Int64Array column using the given encoding.
fn int64_array_ctx(encoding: ColumnEncoding) -> RowPackContext {
    RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "tags".to_string(),
            logical_type: IngestedType::Int64Array,
            encoding,
        }],
        estimated_row_bytes: 0,
    }
}

/// Build a RowPackContext with a single StringArray column using the given encoding.
fn string_array_ctx(encoding: ColumnEncoding) -> RowPackContext {
    RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "tokens".to_string(),
            logical_type: IngestedType::StringArray,
            encoding,
        }],
        estimated_row_bytes: 0,
    }
}

/// Build an Arrow ListArray<Int64> from rows of Option<Vec<i64>>.
fn make_int64_list_array(rows: Vec<Option<Vec<i64>>>) -> ArrayRef {
    let mut builder = ListBuilder::new(Int64Builder::new());
    for row in rows {
        match row {
            None => builder.append_null(),
            Some(vals) => {
                builder.values().append_slice(&vals);
                builder.append(true);
            }
        }
    }
    Arc::new(builder.finish())
}

/// Build an Arrow ListArray<Utf8> from rows of Option<Vec<&str>>.
fn make_string_list_array(rows: Vec<Option<Vec<&str>>>) -> ArrayRef {
    let mut builder = ListBuilder::new(StringBuilder::new());
    for row in rows {
        match row {
            None => builder.append_null(),
            Some(vals) => {
                for v in &vals {
                    builder.values().append_value(v);
                }
                builder.append(true);
            }
        }
    }
    Arc::new(builder.finish())
}

// =============================================================================
// Int64Array tests
// =============================================================================

#[test]
fn test_int64_array_basic_roundtrip() -> anyhow::Result<()> {
    let ctx = int64_array_ctx(ColumnEncoding::Int64Array);
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array = make_int64_list_array(vec![Some(vec![11, 300, 5])]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut encoded = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;

    let json = decoder.decode_json(&encoded)?;
    let arr = json["tags"].as_array().expect("should decode as array");
    assert_eq!(arr.len(), 3);
    assert_eq!(arr[0].as_i64().unwrap(), 11);
    assert_eq!(arr[1].as_i64().unwrap(), 300);
    assert_eq!(arr[2].as_i64().unwrap(), 5);
    Ok(())
}

#[test]
fn test_int64_array_empty_list() -> anyhow::Result<()> {
    let ctx = int64_array_ctx(ColumnEncoding::Int64Array);
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array = make_int64_list_array(vec![Some(vec![])]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut encoded = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;

    let json = decoder.decode_json(&encoded)?;
    let arr = json["tags"].as_array().expect("should decode as array");
    assert_eq!(arr.len(), 0);
    Ok(())
}

#[test]
fn test_int64_array_single_element() -> anyhow::Result<()> {
    let ctx = int64_array_ctx(ColumnEncoding::Int64Array);
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array = make_int64_list_array(vec![Some(vec![42])]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut encoded = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;

    let json = decoder.decode_json(&encoded)?;
    let arr = json["tags"].as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0].as_i64().unwrap(), 42);
    Ok(())
}

#[test]
fn test_int64_array_null_row() -> anyhow::Result<()> {
    let ctx = int64_array_ctx(ColumnEncoding::Int64Array);
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array = make_int64_list_array(vec![None]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut encoded = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;

    let json = decoder.decode_json(&encoded)?;
    assert!(
        json["tags"].is_null(),
        "null row should decode as JSON null"
    );
    Ok(())
}

/// Most important: variable-length rows must not cause pointer drift.
#[test]
fn test_int64_array_variable_length_multirow() -> anyhow::Result<()> {
    let ctx = int64_array_ctx(ColumnEncoding::Int64Array);
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    // Rows with 1, 3, 7 elements.
    let rows = vec![
        Some(vec![1i64]),
        Some(vec![10i64, 20i64, 30i64]),
        Some(vec![100i64, 200i64, 300i64, 400i64, 500i64, 600i64, 700i64]),
    ];
    let array = make_int64_list_array(rows);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);

    let mut encoded_rows = Vec::new();
    for row in 0..3 {
        access.set_row(row);
        let mut encoded = Vec::new();
        encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;
        encoded_rows.push(encoded);
    }

    // Verify row 0: [1]
    let json = decoder.decode_json(&encoded_rows[0])?;
    let arr = json["tags"].as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0].as_i64().unwrap(), 1);

    // Verify row 1: [10, 20, 30]
    let json = decoder.decode_json(&encoded_rows[1])?;
    let arr = json["tags"].as_array().unwrap();
    assert_eq!(arr.len(), 3);
    assert_eq!(arr[0].as_i64().unwrap(), 10);
    assert_eq!(arr[1].as_i64().unwrap(), 20);
    assert_eq!(arr[2].as_i64().unwrap(), 30);

    // Verify row 2: [100..700]
    let json = decoder.decode_json(&encoded_rows[2])?;
    let arr = json["tags"].as_array().unwrap();
    assert_eq!(arr.len(), 7);
    assert_eq!(arr[0].as_i64().unwrap(), 100);
    assert_eq!(arr[6].as_i64().unwrap(), 700);

    Ok(())
}

/// Negative values round-trip correctly (zigzag encoding).
#[test]
fn test_int64_array_negative_values() -> anyhow::Result<()> {
    let ctx = int64_array_ctx(ColumnEncoding::Int64Array);
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array = make_int64_list_array(vec![Some(vec![-1, -100, i64::MIN, i64::MAX])]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut encoded = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;

    let json = decoder.decode_json(&encoded)?;
    let arr = json["tags"].as_array().unwrap();
    assert_eq!(arr.len(), 4);
    assert_eq!(arr[0].as_i64().unwrap(), -1);
    assert_eq!(arr[1].as_i64().unwrap(), -100);
    assert_eq!(arr[2].as_i64().unwrap(), i64::MIN);
    assert_eq!(arr[3].as_i64().unwrap(), i64::MAX);
    Ok(())
}

// =============================================================================
// StringArray tests
// =============================================================================

/// Helper: build a StringArray context with Dict encoding from given tokens.
fn string_array_dict_ctx(tokens: &[&str]) -> RowPackContext {
    let dict: Vec<Vec<u8>> = tokens.iter().map(|s| s.as_bytes().to_vec()).collect();
    string_array_ctx(ColumnEncoding::StringArray {
        encoding: StringEncoding::Dict { values: dict },
    })
}

/// Helper: build a StringArray context with Raw encoding.
fn string_array_raw_ctx() -> RowPackContext {
    string_array_ctx(ColumnEncoding::StringArray {
        encoding: StringEncoding::Raw,
    })
}

#[test]
fn test_string_array_basic_dict_roundtrip() -> anyhow::Result<()> {
    let tokens = ["kids", "outdoor", "sale"];
    let ctx = string_array_dict_ctx(&tokens);
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array = make_string_list_array(vec![Some(vec!["kids", "outdoor", "sale"])]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut encoded = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;

    let json = decoder.decode_json(&encoded)?;
    let arr = json["tokens"].as_array().expect("should decode as array");
    assert_eq!(arr.len(), 3);
    assert_eq!(arr[0].as_str().unwrap(), "kids");
    assert_eq!(arr[1].as_str().unwrap(), "outdoor");
    assert_eq!(arr[2].as_str().unwrap(), "sale");
    Ok(())
}

#[test]
fn test_string_array_raw_roundtrip() -> anyhow::Result<()> {
    let ctx = string_array_raw_ctx();
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array = make_string_list_array(vec![Some(vec!["hello", "world"])]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut encoded = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;

    let json = decoder.decode_json(&encoded)?;
    let arr = json["tokens"].as_array().unwrap();
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0].as_str().unwrap(), "hello");
    assert_eq!(arr[1].as_str().unwrap(), "world");
    Ok(())
}

#[test]
fn test_string_array_empty_list() -> anyhow::Result<()> {
    let ctx = string_array_raw_ctx();
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array = make_string_list_array(vec![Some(vec![])]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut encoded = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;

    let json = decoder.decode_json(&encoded)?;
    let arr = json["tokens"].as_array().unwrap();
    assert_eq!(arr.len(), 0);
    Ok(())
}

#[test]
fn test_string_array_single_element() -> anyhow::Result<()> {
    let ctx = string_array_raw_ctx();
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array = make_string_list_array(vec![Some(vec!["only"])]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut encoded = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;

    let json = decoder.decode_json(&encoded)?;
    let arr = json["tokens"].as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0].as_str().unwrap(), "only");
    Ok(())
}

#[test]
fn test_string_array_null_row() -> anyhow::Result<()> {
    let ctx = string_array_raw_ctx();
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array = make_string_list_array(vec![None]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut encoded = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;

    let json = decoder.decode_json(&encoded)?;
    assert!(
        json["tokens"].is_null(),
        "null row should decode as JSON null"
    );
    Ok(())
}

/// Variable-length multi-row test: pointer must not drift.
#[test]
fn test_string_array_variable_length_multirow() -> anyhow::Result<()> {
    let ctx = string_array_raw_ctx();
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    // Rows with 1, 3, 7 elements.
    let rows = vec![
        Some(vec!["a"]),
        Some(vec!["b", "c", "d"]),
        Some(vec!["e", "f", "g", "h", "i", "j", "k"]),
    ];
    let array = make_string_list_array(rows);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);

    let mut encoded_rows = Vec::new();
    for row in 0..3 {
        access.set_row(row);
        let mut encoded = Vec::new();
        encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;
        encoded_rows.push(encoded);
    }

    // Row 0: ["a"]
    let json = decoder.decode_json(&encoded_rows[0])?;
    let arr = json["tokens"].as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0].as_str().unwrap(), "a");

    // Row 1: ["b", "c", "d"]
    let json = decoder.decode_json(&encoded_rows[1])?;
    let arr = json["tokens"].as_array().unwrap();
    assert_eq!(arr.len(), 3);
    assert_eq!(arr[0].as_str().unwrap(), "b");
    assert_eq!(arr[1].as_str().unwrap(), "c");
    assert_eq!(arr[2].as_str().unwrap(), "d");

    // Row 2: ["e".."k"]
    let json = decoder.decode_json(&encoded_rows[2])?;
    let arr = json["tokens"].as_array().unwrap();
    assert_eq!(arr.len(), 7);
    assert_eq!(arr[0].as_str().unwrap(), "e");
    assert_eq!(arr[6].as_str().unwrap(), "k");

    Ok(())
}

/// OOV token: token not seen in context → tag-2 fallback → decodes correctly.
#[test]
fn test_string_array_oov_token_fallback() -> anyhow::Result<()> {
    // Dict only knows "kids" and "outdoor" but we encode "newtoken" which is OOV.
    let ctx = string_array_dict_ctx(&["kids", "outdoor"]);
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array = make_string_list_array(vec![Some(vec!["newtoken"])]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut encoded = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;

    let json = decoder.decode_json(&encoded)?;
    let arr = json["tokens"].as_array().expect("should decode as array");
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0].as_str().unwrap(), "newtoken");
    Ok(())
}

/// Mixed dict-hit and OOV tokens in the same list.
#[test]
fn test_string_array_mixed_dict_and_oov() -> anyhow::Result<()> {
    let ctx = string_array_dict_ctx(&["kids", "outdoor", "sale"]);
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    // "kids" = dict hit, "newtoken" = OOV, "outdoor" = dict hit
    let array = make_string_list_array(vec![Some(vec!["kids", "newtoken", "outdoor"])]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut encoded = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;

    let json = decoder.decode_json(&encoded)?;
    let arr = json["tokens"].as_array().expect("should decode as array");
    assert_eq!(arr.len(), 3);
    assert_eq!(arr[0].as_str().unwrap(), "kids");
    assert_eq!(arr[1].as_str().unwrap(), "newtoken");
    assert_eq!(arr[2].as_str().unwrap(), "outdoor");
    Ok(())
}

/// Mixed null and non-null rows.
#[test]
fn test_string_array_mixed_null_and_valid_rows() -> anyhow::Result<()> {
    let ctx = string_array_raw_ctx();
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let rows = vec![
        Some(vec!["hello", "world"]),
        None,
        Some(vec!["foo"]),
        Some(vec![]),
    ];
    let array = make_string_list_array(rows);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);

    let mut encoded_rows = Vec::new();
    for row in 0..4 {
        access.set_row(row);
        let mut encoded = Vec::new();
        encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;
        encoded_rows.push(encoded);
    }

    // Row 0
    let json = decoder.decode_json(&encoded_rows[0])?;
    let arr = json["tokens"].as_array().unwrap();
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0].as_str().unwrap(), "hello");

    // Row 1: null
    let json = decoder.decode_json(&encoded_rows[1])?;
    assert!(json["tokens"].is_null());

    // Row 2
    let json = decoder.decode_json(&encoded_rows[2])?;
    let arr = json["tokens"].as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0].as_str().unwrap(), "foo");

    // Row 3: empty
    let json = decoder.decode_json(&encoded_rows[3])?;
    let arr = json["tokens"].as_array().unwrap();
    assert_eq!(arr.len(), 0);

    Ok(())
}

// =============================================================================
// Context builder tests (observe → choose_encoding)
// =============================================================================

#[test]
fn test_int64_array_context_builder() -> anyhow::Result<()> {
    use crate::{RowPackColumn, RowPackConfig, RowPackContextBuilder};

    let col = RowPackColumn {
        name: "tags".to_string(),
        logical_type: IngestedType::Int64Array,
        fixed_len: None,
        precision: None,
        scale: None,
    };
    let config = RowPackConfig::default();
    let mut builder = RowPackContextBuilder::new(vec![col], config);

    // Observe some rows
    let array = make_int64_list_array(vec![Some(vec![1, 2, 3]), Some(vec![10, 20]), None]);
    let arrays: Vec<ArrayRef> = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);

    for row in 0..3 {
        access.set_row(row);
        builder.observe_row_from_access(&mut access)?;
    }

    let ctx = builder.finish()?;
    assert_eq!(ctx.columns.len(), 1);
    assert!(
        matches!(ctx.columns[0].encoding, ColumnEncoding::Int64Array),
        "expected Int64Array encoding, got {:?}",
        ctx.columns[0].encoding
    );
    Ok(())
}

#[test]
fn test_string_array_context_builder() -> anyhow::Result<()> {
    use crate::{RowPackColumn, RowPackConfig, RowPackContextBuilder};

    let col = RowPackColumn {
        name: "tokens".to_string(),
        logical_type: IngestedType::StringArray,
        fixed_len: None,
        precision: None,
        scale: None,
    };
    let config = RowPackConfig::default();
    let mut builder = RowPackContextBuilder::new(vec![col], config);

    let array = make_string_list_array(vec![Some(vec!["foo", "bar"]), Some(vec!["baz"]), None]);
    let arrays: Vec<ArrayRef> = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);

    for row in 0..3 {
        access.set_row(row);
        builder.observe_row_from_access(&mut access)?;
    }

    let ctx = builder.finish()?;
    assert_eq!(ctx.columns.len(), 1);
    assert!(
        matches!(ctx.columns[0].encoding, ColumnEncoding::StringArray { .. }),
        "expected StringArray encoding, got {:?}",
        ctx.columns[0].encoding
    );
    Ok(())
}

// =============================================================================
// String-source fallback: with_int64_array when Parquet column is Utf8
// =============================================================================
//
// Mirrors tests_string_float_array.rs for the float case.
// When a Parquet column is declared ingested_dtype=intlist but the physical
// column type is Utf8 (e.g. "[1, 2, 3]"), we expect DiskGraph to parse it.
//
// These tests use RowAccess directly (not encode/decode) — they test the
// Arrow-reading layer (access.rs::with_int64_array), not the wire format.

fn access_from_strings(values: Vec<Option<&str>>) -> (RowAccess<'static>, Vec<ArrayRef>) {
    use arrow_array::StringArray as ArrowStringArray;
    let array: ArrayRef = Arc::new(ArrowStringArray::from(values));
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let access = RowAccess::new(vec![RowAccessHints::default()], stats);
    (access, arrays)
}

#[test]
fn test_int64_array_from_string_basic() -> anyhow::Result<()> {
    let (mut access, arrays) = access_from_strings(vec![Some("[1, 2, 3]")]);
    access.set_cols(&arrays);
    access.set_row(0);
    let result = access.with_int64_array(0, |vals| {
        assert_eq!(vals, &[1i64, 2, 3]);
        Ok(())
    })?;
    assert!(result.is_some());
    Ok(())
}

#[test]
fn test_int64_array_from_string_negative_values() -> anyhow::Result<()> {
    let (mut access, arrays) = access_from_strings(vec![Some("[-1, -128, 300]")]);
    access.set_cols(&arrays);
    access.set_row(0);
    let result = access.with_int64_array(0, |vals| {
        assert_eq!(vals, &[-1i64, -128, 300]);
        Ok(())
    })?;
    assert!(result.is_some());
    Ok(())
}

#[test]
fn test_int64_array_from_string_empty() -> anyhow::Result<()> {
    let (mut access, arrays) = access_from_strings(vec![Some("[]")]);
    access.set_cols(&arrays);
    access.set_row(0);
    let result = access.with_int64_array(0, |vals| {
        assert!(vals.is_empty());
        Ok(())
    })?;
    assert!(result.is_some());
    Ok(())
}

#[test]
fn test_int64_array_from_string_null() -> anyhow::Result<()> {
    let (mut access, arrays) = access_from_strings(vec![None]);
    access.set_cols(&arrays);
    access.set_row(0);
    let result = access.with_int64_array(0, |_| Ok(()))?;
    assert!(result.is_none(), "null string should produce None");
    Ok(())
}

#[test]
fn test_int64_array_from_string_malformed_is_error() {
    let (mut access, arrays) = access_from_strings(vec![Some("not an int array")]);
    access.set_cols(&arrays);
    access.set_row(0);
    let result = access.with_int64_array(0, |_| Ok(()));
    assert!(result.is_err(), "malformed string should be a hard error");
}

// =============================================================================
// String-source fallback: with_string_array when Parquet column is Utf8
// =============================================================================
//
// When a Parquet column is declared ingested_dtype=stringlist but the physical
// column type is Utf8 containing a JSON array like '["a","b","c"]', we expect
// DiskGraph to parse it into a string list.

#[test]
fn test_string_array_from_string_basic() -> anyhow::Result<()> {
    let (mut access, arrays) = access_from_strings(vec![Some(r#"["kids", "outdoor", "sale"]"#)]);
    access.set_cols(&arrays);
    access.set_row(0);
    let result = access.with_string_array(0, |tokens| {
        assert_eq!(tokens, &["kids", "outdoor", "sale"]);
        Ok(())
    })?;
    assert!(result.is_some());
    Ok(())
}

#[test]
fn test_string_array_from_string_empty() -> anyhow::Result<()> {
    let (mut access, arrays) = access_from_strings(vec![Some("[]")]);
    access.set_cols(&arrays);
    access.set_row(0);
    let result = access.with_string_array(0, |tokens| {
        assert!(tokens.is_empty());
        Ok(())
    })?;
    assert!(result.is_some());
    Ok(())
}

#[test]
fn test_string_array_from_string_null() -> anyhow::Result<()> {
    let (mut access, arrays) = access_from_strings(vec![None]);
    access.set_cols(&arrays);
    access.set_row(0);
    let result = access.with_string_array(0, |_| Ok(()))?;
    assert!(result.is_none(), "null string should produce None");
    Ok(())
}

#[test]
fn test_string_array_from_string_malformed_is_error() {
    let (mut access, arrays) = access_from_strings(vec![Some("not json")]);
    access.set_cols(&arrays);
    access.set_row(0);
    let result = access.with_string_array(0, |_| Ok(()));
    assert!(result.is_err(), "malformed string should be a hard error");
}

#[test]
fn test_string_array_from_string_single_element() -> anyhow::Result<()> {
    let (mut access, arrays) = access_from_strings(vec![Some(r#"["solo"]"#)]);
    access.set_cols(&arrays);
    access.set_row(0);
    let result = access.with_string_array(0, |tokens| {
        assert_eq!(tokens, &["solo"]);
        Ok(())
    })?;
    assert!(result.is_some());
    Ok(())
}
