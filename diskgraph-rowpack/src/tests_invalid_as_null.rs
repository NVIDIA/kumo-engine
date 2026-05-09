use super::*;
use arrow_array::{
    builder::{ListBuilder, StringBuilder},
    ArrayRef, Int64Array, StringArray,
};
use std::sync::Arc;

#[derive(Default)]
struct ParseFailureObserver {
    parse_failures: Vec<usize>,
    nulls: Vec<usize>,
}

impl ColumnValueObserver for ParseFailureObserver {
    fn observe_i64(&mut self, _col_idx: usize, _value: i64) {}
    fn observe_f64(&mut self, _col_idx: usize, _value: f64) {}
    fn observe_string(&mut self, _col_idx: usize, _value: &str) {}
    fn observe_timestamp(&mut self, _col_idx: usize, _secs: i64) {}
    fn observe_float_list(&mut self, _col_idx: usize, _view: &FloatListView) {}
    fn observe_int64_list(&mut self, _col_idx: usize, _view: &Int64ListView) {}
    fn observe_string_list(&mut self, _col_idx: usize, _view: &StringListView) {}
    fn observe_null(&mut self, col_idx: usize) {
        self.nulls.push(col_idx);
    }
    fn observe_parse_failure(&mut self, col_idx: usize) {
        self.parse_failures.push(col_idx);
    }
}

fn string_ctx() -> RowPackContext {
    one_column_ctx(
        IngestedType::String,
        ColumnEncoding::String {
            encoding: StringEncoding::Raw,
        },
    )
}

fn bytes_ctx() -> RowPackContext {
    one_column_ctx(
        IngestedType::Bytes,
        ColumnEncoding::Bytes { fixed_len: None },
    )
}

fn float32_array_ctx() -> RowPackContext {
    one_column_ctx(
        IngestedType::Float32Array,
        ColumnEncoding::FloatArray {
            storage: FloatStorage::F32,
        },
    )
}

fn float64_array_ctx() -> RowPackContext {
    one_column_ctx(
        IngestedType::Float64Array,
        ColumnEncoding::FloatArray {
            storage: FloatStorage::F64,
        },
    )
}

fn int64_array_ctx() -> RowPackContext {
    one_column_ctx(IngestedType::Int64Array, ColumnEncoding::Int64Array)
}

fn string_array_ctx() -> RowPackContext {
    one_column_ctx(
        IngestedType::StringArray,
        ColumnEncoding::StringArray {
            encoding: StringEncoding::Raw,
        },
    )
}

fn one_column_ctx(logical_type: IngestedType, encoding: ColumnEncoding) -> RowPackContext {
    RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "value".to_string(),
            logical_type,
            encoding,
        }],
        estimated_row_bytes: 0,
    }
}

fn list_utf8_array(values: &[&str]) -> ArrayRef {
    let mut builder = ListBuilder::new(StringBuilder::new());
    for value in values {
        builder.values().append_value(*value);
    }
    builder.append(true);
    Arc::new(builder.finish())
}

fn encode_invalid_as_null(ctx: RowPackContext, array: ArrayRef) -> anyhow::Result<()> {
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let arrays = vec![array];
    let mut access = RowAccess::new(vec![RowAccessHints::default()], RowAccessStats::default());
    access.set_cols(&arrays);
    access.set_row(0);

    let mut tags = Vec::new();
    let mut bools = Vec::new();
    let mut data = Vec::new();
    let mut f16_buf = Vec::new();
    let key_bufs = vec![Vec::new()];
    let mut observer = ParseFailureObserver::default();

    encoder.encode_row_observed(
        &mut access,
        &mut tags,
        &mut bools,
        &mut data,
        &mut f16_buf,
        &key_bufs,
        &mut observer,
    )?;

    let mut row = Vec::new();
    row.extend_from_slice(&tags);
    row.extend_from_slice(&bools);
    row.extend_from_slice(&data);

    let json = decoder.decode_json(&row)?;
    assert!(
        json["value"].is_null(),
        "invalid value should encode as null: {json}"
    );
    assert_eq!(observer.parse_failures, vec![0]);
    assert_eq!(observer.nulls, vec![0]);
    Ok(())
}

fn observe_invalid_as_null(ctx: RowPackContext, array: ArrayRef) -> anyhow::Result<()> {
    let col = RowPackColumn {
        name: "value".to_string(),
        logical_type: ctx.columns[0].logical_type,
        fixed_len: None,
        precision: None,
        scale: None,
    };
    let mut builder = RowPackContextBuilder::new(vec![col], RowPackConfig::default());
    let arrays = vec![array];
    let mut access = RowAccess::new(vec![RowAccessHints::default()], RowAccessStats::default());
    access.set_cols(&arrays);
    access.set_row(0);
    builder.observe_row_from_access(&mut access)?;
    let _ = builder.finish()?;
    Ok(())
}

#[test]
fn encode_string_destination_rejects_list_as_null() -> anyhow::Result<()> {
    encode_invalid_as_null(string_ctx(), list_utf8_array(&["a", "b"]))
}

#[test]
fn encode_bytes_destination_rejects_int_as_null() -> anyhow::Result<()> {
    encode_invalid_as_null(bytes_ctx(), Arc::new(Int64Array::from(vec![Some(42)])))
}

#[test]
fn encode_float32_array_destination_rejects_invalid_string_as_null() -> anyhow::Result<()> {
    encode_invalid_as_null(
        float32_array_ctx(),
        Arc::new(StringArray::from(vec![Some("not a float array")])),
    )
}

#[test]
fn encode_float64_array_destination_rejects_list_utf8_as_null() -> anyhow::Result<()> {
    encode_invalid_as_null(float64_array_ctx(), list_utf8_array(&["a", "b"]))
}

#[test]
fn encode_int64_array_destination_rejects_list_utf8_as_null() -> anyhow::Result<()> {
    encode_invalid_as_null(int64_array_ctx(), list_utf8_array(&["a", "b"]))
}

#[test]
fn encode_string_array_destination_rejects_int_as_null() -> anyhow::Result<()> {
    encode_invalid_as_null(
        string_array_ctx(),
        Arc::new(Int64Array::from(vec![Some(42)])),
    )
}

#[test]
fn context_observe_string_destination_rejects_list_as_null() -> anyhow::Result<()> {
    observe_invalid_as_null(string_ctx(), list_utf8_array(&["a", "b"]))
}

#[test]
fn context_observe_bytes_destination_rejects_int_as_null() -> anyhow::Result<()> {
    observe_invalid_as_null(bytes_ctx(), Arc::new(Int64Array::from(vec![Some(42)])))
}

#[test]
fn context_observe_float32_array_destination_rejects_invalid_string_as_null() -> anyhow::Result<()>
{
    observe_invalid_as_null(
        float32_array_ctx(),
        Arc::new(StringArray::from(vec![Some("not a float array")])),
    )
}

#[test]
fn context_observe_float64_array_destination_rejects_list_utf8_as_null() -> anyhow::Result<()> {
    observe_invalid_as_null(float64_array_ctx(), list_utf8_array(&["a", "b"]))
}

#[test]
fn context_observe_int64_array_destination_rejects_list_utf8_as_null() -> anyhow::Result<()> {
    observe_invalid_as_null(int64_array_ctx(), list_utf8_array(&["a", "b"]))
}

#[test]
fn context_observe_string_array_destination_rejects_int_as_null() -> anyhow::Result<()> {
    observe_invalid_as_null(
        string_array_ctx(),
        Arc::new(Int64Array::from(vec![Some(42)])),
    )
}
