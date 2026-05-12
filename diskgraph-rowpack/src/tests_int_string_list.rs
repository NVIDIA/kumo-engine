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

/// Build an Arrow ListArray<Int64> from rows of Option<Vec<Option<i64>>>.
fn make_nullable_int64_list_array(rows: Vec<Option<Vec<Option<i64>>>>) -> ArrayRef {
    let mut builder = ListBuilder::new(Int64Builder::new());
    for row in rows {
        match row {
            None => builder.append_null(),
            Some(vals) => {
                for v in vals {
                    builder.values().append_option(v);
                }
                builder.append(true);
            }
        }
    }
    Arc::new(builder.finish())
}

/// Build an Arrow ListArray<Utf8> from rows of Option<Vec<Option<&str>>>.
fn make_nullable_string_list_array(rows: Vec<Option<Vec<Option<&str>>>>) -> ArrayRef {
    let mut builder = ListBuilder::new(StringBuilder::new());
    for row in rows {
        match row {
            None => builder.append_null(),
            Some(vals) => {
                for v in vals {
                    builder.values().append_option(v);
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

#[test]
fn test_int_list_view_drops_nulls() -> anyhow::Result<()> {
    // Inner null elements are dropped from iteration. Source list
    // [Some(7), None, Some(11)] → iterator yields [7, 11], not [7, 0, 11].
    let array = make_nullable_int64_list_array(vec![Some(vec![Some(7), None, Some(11)])]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let view = access.int_list_view(0)?.expect("expected non-null list");
    let mut values = Vec::new();
    view.for_each_i64(|v| values.push(v));
    assert_eq!(values, vec![7, 11]);
    Ok(())
}

#[test]
fn test_int64_array_null_elements_dropped_roundtrip() -> anyhow::Result<()> {
    // Inner null elements are dropped at encode time. Source list
    // [Some(7), None, Some(11)] persists as [7, 11] — length 2, not 3.
    let ctx = int64_array_ctx(ColumnEncoding::Int64Array);
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array = make_nullable_int64_list_array(vec![Some(vec![Some(7), None, Some(11)])]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut encoded = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;

    let json = decoder.decode_json(&encoded)?;
    let arr = json["tags"].as_array().unwrap();
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0].as_i64().unwrap(), 7);
    assert_eq!(arr[1].as_i64().unwrap(), 11);
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

#[test]
fn test_string_list_view_drops_nulls() -> anyhow::Result<()> {
    // Inner null elements are dropped from iteration. Source list
    // [Some("kids"), None, Some("sale")] → iterator yields ["kids", "sale"],
    // not ["kids", "", "sale"]. A legitimate empty string is NOT filtered
    // by this rule; only the null bit matters.
    let array = make_nullable_string_list_array(vec![Some(vec![Some("kids"), None, Some("sale")])]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let view = access.string_list_view(0)?.expect("expected non-null list");
    let mut values = Vec::new();
    view.for_each_str(|s| values.push(s.to_string()));
    assert_eq!(values, vec!["kids", "sale"]);
    Ok(())
}

#[test]
fn test_string_array_null_elements_dropped_roundtrip() -> anyhow::Result<()> {
    // Inner null elements are dropped at encode time. Source list
    // [Some("kids"), None, Some("sale")] persists as ["kids", "sale"] —
    // length 2, not 3.
    let ctx = string_array_raw_ctx();
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array = make_nullable_string_list_array(vec![Some(vec![Some("kids"), None, Some("sale")])]);
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
    assert_eq!(arr[0].as_str().unwrap(), "kids");
    assert_eq!(arr[1].as_str().unwrap(), "sale");
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

// =============================================================================
// Drop-null contract: additional coverage beyond the rewritten baseline tests
// =============================================================================
//
// The accessors and encoders drop inner null elements from list-typed columns.
// The rules at the encode boundary:
//   * Int64Array: skip only on validity bit; 0 is a legitimate value.
//   * StringArray: skip only on validity bit; "" is preserved as legitimate data.

#[test]
fn test_int64_array_all_null_elements_roundtrip_as_empty() -> anyhow::Result<()> {
    // A list whose every inner element is null persists as an empty list, not
    // a row-level null. The cell itself was not null — only the elements.
    let ctx = int64_array_ctx(ColumnEncoding::Int64Array);
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array = make_nullable_int64_list_array(vec![Some(vec![None, None, None])]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut encoded = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;

    let json = decoder.decode_json(&encoded)?;
    assert!(!json["tags"].is_null(), "cell was non-null, must not decode as null");
    let arr = json["tags"].as_array().expect("should decode as array");
    assert_eq!(arr.len(), 0, "all-null elements must collapse to empty list");
    Ok(())
}

#[test]
fn test_string_array_all_null_elements_roundtrip_as_empty() -> anyhow::Result<()> {
    let ctx = string_array_raw_ctx();
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array = make_nullable_string_list_array(vec![Some(vec![None, None])]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut encoded = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;

    let json = decoder.decode_json(&encoded)?;
    assert!(!json["tokens"].is_null(), "cell was non-null, must not decode as null");
    let arr = json["tokens"].as_array().expect("should decode as array");
    assert_eq!(arr.len(), 0, "all-null elements must collapse to empty list");
    Ok(())
}

#[test]
fn test_int64_array_whole_cell_null_distinct_from_all_null_elements() -> anyhow::Result<()> {
    // A row-level null and an all-null-elements list must remain
    // distinguishable after roundtrip: the former decodes to JSON null,
    // the latter decodes to an empty array.
    let ctx = int64_array_ctx(ColumnEncoding::Int64Array);
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array = make_nullable_int64_list_array(vec![
        None,                              // row 0: whole-cell null
        Some(vec![None, None]),            // row 1: all inner elements null
    ]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);

    access.set_row(0);
    let mut encoded0 = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded0, &mut scratch)?;
    let json0 = decoder.decode_json(&encoded0)?;
    assert!(json0["tags"].is_null(), "whole-cell null must decode as JSON null");

    access.set_row(1);
    let mut encoded1 = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded1, &mut scratch)?;
    let json1 = decoder.decode_json(&encoded1)?;
    assert!(!json1["tags"].is_null(), "all-null-elements row must not decode as JSON null");
    assert_eq!(json1["tags"].as_array().unwrap().len(), 0);
    Ok(())
}

#[test]
fn test_string_array_whole_cell_null_distinct_from_all_null_elements() -> anyhow::Result<()> {
    let ctx = string_array_raw_ctx();
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array = make_nullable_string_list_array(vec![
        None,
        Some(vec![None, None]),
    ]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);

    access.set_row(0);
    let mut encoded0 = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded0, &mut scratch)?;
    let json0 = decoder.decode_json(&encoded0)?;
    assert!(json0["tokens"].is_null(), "whole-cell null must decode as JSON null");

    access.set_row(1);
    let mut encoded1 = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded1, &mut scratch)?;
    let json1 = decoder.decode_json(&encoded1)?;
    assert!(!json1["tokens"].is_null(), "all-null-elements row must not decode as JSON null");
    assert_eq!(json1["tokens"].as_array().unwrap().len(), 0);
    Ok(())
}

#[test]
fn test_string_array_preserves_legitimate_empty_string() -> anyhow::Result<()> {
    // The drop-null rule on the string path filters only on the validity
    // bit. A user-provided empty string has validity=true and must survive
    // the roundtrip unchanged.
    let ctx = string_array_raw_ctx();
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array = make_string_list_array(vec![Some(vec!["a", "", "b"])]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut encoded = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;

    let json = decoder.decode_json(&encoded)?;
    let arr = json["tokens"].as_array().unwrap();
    assert_eq!(arr.len(), 3, "legitimate empty string must not be dropped");
    assert_eq!(arr[0].as_str().unwrap(), "a");
    assert_eq!(arr[1].as_str().unwrap(), "");
    assert_eq!(arr[2].as_str().unwrap(), "b");
    Ok(())
}

#[test]
fn test_string_array_mixed_null_and_legitimate_empty_string() -> anyhow::Result<()> {
    // Mixed: null element and real empty string in the same list.
    // Null is dropped, empty string is preserved.
    let ctx = string_array_raw_ctx();
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array = make_nullable_string_list_array(vec![Some(vec![
        Some("a"),
        Some(""),
        None,
        Some("b"),
    ])]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut encoded = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;

    let json = decoder.decode_json(&encoded)?;
    let arr = json["tokens"].as_array().unwrap();
    assert_eq!(arr.len(), 3, "only the null element is dropped");
    assert_eq!(arr[0].as_str().unwrap(), "a");
    assert_eq!(arr[1].as_str().unwrap(), "");
    assert_eq!(arr[2].as_str().unwrap(), "b");
    Ok(())
}

#[test]
fn test_int64_array_no_null_fast_path_preserves_values() -> anyhow::Result<()> {
    // Dense list (null_count() == 0) must exercise the fast encode path
    // and roundtrip byte-for-byte.
    let ctx = int64_array_ctx(ColumnEncoding::Int64Array);
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array = make_int64_list_array(vec![Some(vec![0, 1, 0, 2, 0, 3])]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut encoded = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;

    let json = decoder.decode_json(&encoded)?;
    let arr = json["tags"].as_array().unwrap();
    assert_eq!(arr.len(), 6, "legitimate zeros must not be dropped");
    let values: Vec<i64> = arr.iter().map(|v| v.as_i64().unwrap()).collect();
    assert_eq!(values, vec![0, 1, 0, 2, 0, 3]);
    Ok(())
}

#[test]
fn test_int64_array_context_builder_handles_inner_nulls() -> anyhow::Result<()> {
    // Non-regression: RowPackContextBuilder::observe_row_from_access must
    // accept null-bearing list rows without panic. A follow-up change to
    // the column-stats observer can assert the null-skip semantics more
    // directly; today the public stats accessor does not expose list
    // element uniques.
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

    let array = make_nullable_int64_list_array(vec![
        Some(vec![Some(1), None, Some(2)]),
        Some(vec![None, Some(3)]),
        None,
    ]);
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
    assert!(matches!(ctx.columns[0].encoding, ColumnEncoding::Int64Array));
    Ok(())
}

#[test]
fn test_string_array_context_builder_handles_inner_nulls() -> anyhow::Result<()> {
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

    let array = make_nullable_string_list_array(vec![
        Some(vec![Some("x"), None, Some("y")]),
        Some(vec![None, Some("z")]),
        None,
    ]);
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
    assert!(matches!(ctx.columns[0].encoding, ColumnEncoding::StringArray { .. }));
    Ok(())
}

// =============================================================================
// Int{8,16,32}Array tests — native narrower-width support.
// =============================================================================

use arrow_array::builder::{Int8Builder, Int16Builder, Int32Builder};

/// Build a RowPackContext with a single int-array column at the given width.
fn int_array_ctx(logical: IngestedType, encoding: ColumnEncoding) -> RowPackContext {
    RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "tags".to_string(),
            logical_type: logical,
            encoding,
        }],
        estimated_row_bytes: 0,
    }
}

fn make_int8_list_array(rows: Vec<Option<Vec<i8>>>) -> ArrayRef {
    let mut builder = ListBuilder::new(Int8Builder::new());
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

fn make_int16_list_array(rows: Vec<Option<Vec<i16>>>) -> ArrayRef {
    let mut builder = ListBuilder::new(Int16Builder::new());
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

fn make_int32_list_array(rows: Vec<Option<Vec<i32>>>) -> ArrayRef {
    let mut builder = ListBuilder::new(Int32Builder::new());
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

#[test]
fn test_int8_array_roundtrip_extremes() -> anyhow::Result<()> {
    let ctx = int_array_ctx(IngestedType::Int8Array, ColumnEncoding::Int8Array);
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array = make_int8_list_array(vec![Some(vec![i8::MIN, -1, 0, 1, i8::MAX])]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut encoded = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;

    let json = decoder.decode_json(&encoded)?;
    let arr = json["tags"].as_array().expect("should decode as array");
    assert_eq!(arr.len(), 5);
    assert_eq!(arr[0].as_i64().unwrap(), i8::MIN as i64);
    assert_eq!(arr[4].as_i64().unwrap(), i8::MAX as i64);
    Ok(())
}

#[test]
fn test_int16_array_roundtrip_extremes() -> anyhow::Result<()> {
    let ctx = int_array_ctx(IngestedType::Int16Array, ColumnEncoding::Int16Array);
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array = make_int16_list_array(vec![Some(vec![i16::MIN, -1, 0, 1, i16::MAX])]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut encoded = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;

    let json = decoder.decode_json(&encoded)?;
    let arr = json["tags"].as_array().expect("should decode as array");
    assert_eq!(arr[0].as_i64().unwrap(), i16::MIN as i64);
    assert_eq!(arr[4].as_i64().unwrap(), i16::MAX as i64);
    Ok(())
}

#[test]
fn test_int32_array_roundtrip_extremes() -> anyhow::Result<()> {
    let ctx = int_array_ctx(IngestedType::Int32Array, ColumnEncoding::Int32Array);
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array = make_int32_list_array(vec![Some(vec![i32::MIN, -1, 0, 1, i32::MAX])]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut encoded = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;

    let json = decoder.decode_json(&encoded)?;
    let arr = json["tags"].as_array().expect("should decode as array");
    assert_eq!(arr[0].as_i64().unwrap(), i32::MIN as i64);
    assert_eq!(arr[4].as_i64().unwrap(), i32::MAX as i64);
    Ok(())
}

#[test]
fn test_int64_array_accepts_int32_source_widening() -> anyhow::Result<()> {
    // Column declared as Int64Array; parquet source is List<Int32>.
    // Widening is safe — encoder should accept it.
    let ctx = int_array_ctx(IngestedType::Int64Array, ColumnEncoding::Int64Array);
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array = make_int32_list_array(vec![Some(vec![10, 20, 30])]);
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
    assert_eq!(arr[0].as_i64().unwrap(), 10);
    assert_eq!(arr[2].as_i64().unwrap(), 30);
    Ok(())
}

#[test]
fn test_int32_array_accepts_int8_source_widening() -> anyhow::Result<()> {
    // Column declared as Int32Array; parquet source is List<Int8>.
    let ctx = int_array_ctx(IngestedType::Int32Array, ColumnEncoding::Int32Array);
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array = make_int8_list_array(vec![Some(vec![-5, 0, 5])]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(vec![RowAccessHints::default()], stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut encoded = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;

    let json = decoder.decode_json(&encoded)?;
    let arr = json["tags"].as_array().expect("should decode as array");
    assert_eq!(arr[0].as_i64().unwrap(), -5);
    assert_eq!(arr[2].as_i64().unwrap(), 5);
    Ok(())
}

#[test]
fn test_int32_array_rejects_int64_overflow() -> anyhow::Result<()> {
    // Column declared as Int32Array; parquet source is List<Int64> with a
    // value that doesn't fit in i32. Encoder must reject with a clear error
    // citing the column.
    let ctx = int_array_ctx(IngestedType::Int32Array, ColumnEncoding::Int32Array);
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let overflow_val: i64 = i32::MAX as i64 + 1;
    let array = make_int64_list_array(vec![Some(vec![1, overflow_val, 3])]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(
        vec![RowAccessHints {
            column_name: Some("tags".to_string()),
            ..Default::default()
        }],
        stats,
    );
    access.set_cols(&arrays);
    access.set_row(0);

    let mut encoded = Vec::new();
    let err = encoder
        .encode_row_from_access(&mut access, &mut encoded, &mut scratch)
        .expect_err("expected overflow rejection");
    let msg = err.to_string();
    assert!(
        msg.contains("int-array overflow") && msg.contains("tags"),
        "expected overflow + column name in error, got: {}",
        msg,
    );
    Ok(())
}

#[test]
fn test_int32_array_accepts_int64_values_within_range() -> anyhow::Result<()> {
    // Same shape as the overflow test, but values fit in i32 → should succeed.
    let ctx = int_array_ctx(IngestedType::Int32Array, ColumnEncoding::Int32Array);
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array = make_int64_list_array(vec![Some(vec![1, 100, i32::MAX as i64])]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(
        vec![RowAccessHints {
            column_name: Some("tags".to_string()),
            ..Default::default()
        }],
        stats,
    );
    access.set_cols(&arrays);
    access.set_row(0);

    let mut encoded = Vec::new();
    encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;
    let json = decoder.decode_json(&encoded)?;
    let arr = json["tags"].as_array().expect("should decode as array");
    assert_eq!(arr.len(), 3);
    assert_eq!(arr[2].as_i64().unwrap(), i32::MAX as i64);
    Ok(())
}

#[test]
fn test_int8_array_rejects_i16_overflow() -> anyhow::Result<()> {
    // Declared Int8Array, source List<Int16> with a value beyond i8 range.
    let ctx = int_array_ctx(IngestedType::Int8Array, ColumnEncoding::Int8Array);
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    let array = make_int16_list_array(vec![Some(vec![1, 200, 3])]);
    let arrays = vec![array];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(
        vec![RowAccessHints {
            column_name: Some("tags".to_string()),
            ..Default::default()
        }],
        stats,
    );
    access.set_cols(&arrays);
    access.set_row(0);

    let mut encoded = Vec::new();
    let err = encoder
        .encode_row_from_access(&mut access, &mut encoded, &mut scratch)
        .expect_err("expected i8 overflow rejection");
    assert!(
        err.to_string().contains("int-array overflow"),
        "expected overflow message, got: {}",
        err,
    );
    Ok(())
}

#[test]
fn test_int32_array_arrow_output_schema_uses_int32_child() -> anyhow::Result<()> {
    use arrow_schema::DataType;
    let (field, _builder) =
        crate::make_arrow_builder("tags", IngestedType::Int32Array, 16)?;
    match field.data_type() {
        DataType::List(child) => assert_eq!(*child.data_type(), DataType::Int32),
        other => panic!("expected List<Int32>, got {:?}", other),
    }
    Ok(())
}

#[test]
fn test_int8_array_arrow_output_schema_uses_int8_child() -> anyhow::Result<()> {
    use arrow_schema::DataType;
    let (field, _) = crate::make_arrow_builder("tags", IngestedType::Int8Array, 16)?;
    match field.data_type() {
        DataType::List(child) => assert_eq!(*child.data_type(), DataType::Int8),
        other => panic!("expected List<Int8>, got {:?}", other),
    }
    Ok(())
}

#[test]
fn test_int16_array_arrow_output_schema_uses_int16_child() -> anyhow::Result<()> {
    use arrow_schema::DataType;
    let (field, _) = crate::make_arrow_builder("tags", IngestedType::Int16Array, 16)?;
    match field.data_type() {
        DataType::List(child) => assert_eq!(*child.data_type(), DataType::Int16),
        other => panic!("expected List<Int16>, got {:?}", other),
    }
    Ok(())
}
