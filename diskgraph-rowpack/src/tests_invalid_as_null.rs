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
    fn observe_int64_list(&mut self, _col_idx: usize, _view: &IntListView) {}
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

fn scalar_ctx(logical_type: IngestedType) -> RowPackContext {
    let encoding = match logical_type {
        IngestedType::Bool => ColumnEncoding::Bool,
        IngestedType::Int8 => ColumnEncoding::Int {
            signed: true,
            varint: false,
            bits: 8,
        },
        IngestedType::Int16 => ColumnEncoding::Int {
            signed: true,
            varint: false,
            bits: 16,
        },
        IngestedType::Int32 => ColumnEncoding::Int {
            signed: true,
            varint: false,
            bits: 32,
        },
        IngestedType::Int64 => ColumnEncoding::Int {
            signed: true,
            varint: true,
            bits: 64,
        },
        IngestedType::UInt8 => ColumnEncoding::Int {
            signed: false,
            varint: false,
            bits: 8,
        },
        IngestedType::UInt16 => ColumnEncoding::Int {
            signed: false,
            varint: false,
            bits: 16,
        },
        IngestedType::UInt32 => ColumnEncoding::Int {
            signed: false,
            varint: false,
            bits: 32,
        },
        IngestedType::UInt64 => ColumnEncoding::Int {
            signed: false,
            varint: true,
            bits: 64,
        },
        IngestedType::Float32 => ColumnEncoding::Float {
            storage: FloatStorage::F32,
        },
        IngestedType::Float64 => ColumnEncoding::Float {
            storage: FloatStorage::F64,
        },
        IngestedType::Uuid => ColumnEncoding::Uuid,
        IngestedType::Uuid256 => ColumnEncoding::Uuid256,
        IngestedType::Ip => ColumnEncoding::Ip,
        IngestedType::Date => ColumnEncoding::Date {
            varint: true,
            bits: 32,
        },
        IngestedType::Timestamp => ColumnEncoding::Timestamp,
        IngestedType::Decimal128 => ColumnEncoding::Decimal {
            precision: 10,
            scale: 2,
            bytes: decimal_raw_bytes(10).unwrap(),
            storage: DecimalStorage::Varint,
        },
        IngestedType::Decimal256 => ColumnEncoding::Decimal {
            precision: 40,
            scale: 2,
            bytes: decimal_raw_bytes(40).unwrap(),
            storage: DecimalStorage::Varint,
        },
        _ => unreachable!("not a scalar parse-failure test type"),
    };
    one_column_ctx(logical_type, encoding)
}

fn scalar_col(logical_type: IngestedType) -> RowPackColumn {
    RowPackColumn {
        name: "value".to_string(),
        logical_type,
        fixed_len: None,
        precision: if matches!(
            logical_type,
            IngestedType::Decimal128 | IngestedType::Decimal256
        ) {
            Some(10)
        } else {
            None
        },
        scale: if matches!(
            logical_type,
            IngestedType::Decimal128 | IngestedType::Decimal256
        ) {
            Some(2)
        } else {
            None
        },
    }
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

fn invalid_string_array() -> ArrayRef {
    Arc::new(StringArray::from(vec![Some("not-a-value")]))
}

fn access_for_array(array: ArrayRef) -> (RowAccess<'static>, Vec<ArrayRef>) {
    let arrays = vec![array];
    let mut access = RowAccess::new(vec![RowAccessHints::default()], RowAccessStats::default());
    access.set_cols(&arrays);
    access.set_row(0);
    (access, arrays)
}

fn assert_invalid_accessor_errors(
    access: &mut RowAccess<'_>,
    logical_type: IngestedType,
) -> anyhow::Result<()> {
    let failed = match logical_type {
        IngestedType::Bool => access.get_bool(0).is_err(),
        IngestedType::Int8 | IngestedType::Int16 | IngestedType::Int32 => {
            access.get_int32(0).is_err()
        }
        IngestedType::Int64 => access.get_int64(0).is_err(),
        IngestedType::UInt8 | IngestedType::UInt16 | IngestedType::UInt32 => {
            access.get_uint32(0).is_err()
        }
        IngestedType::UInt64 => access.get_uint64(0).is_err(),
        IngestedType::Float32 => access.get_float32(0).is_err(),
        IngestedType::Float64 => access.get_float64(0).is_err(),
        IngestedType::Uuid => access.get_uuid(0).is_err(),
        IngestedType::Uuid256 => access.get_uuid256(0).is_err(),
        IngestedType::Ip => access.get_ip(0).is_err(),
        IngestedType::Date => access.get_date(0).is_err(),
        IngestedType::Timestamp => access.get_timestamp(0).is_err(),
        IngestedType::Decimal128 => access.get_decimal128(0).is_err(),
        IngestedType::Decimal256 => access.get_decimal256(0).is_err(),
        _ => unreachable!("not a scalar parse-failure test type"),
    };
    assert!(
        failed,
        "test setup must trigger an accessor error for {logical_type:?}"
    );
    Ok(())
}

fn encode_invalid_as_null(ctx: RowPackContext, array: ArrayRef) -> anyhow::Result<()> {
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let (mut access, _arrays) = access_for_array(array);

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

fn encode_from_access_invalid_is_null(logical_type: IngestedType) -> anyhow::Result<()> {
    let ctx = scalar_ctx(logical_type);
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let decoder = RowPackDecoder::new(ctx)?;
    let (mut access, _arrays) = access_for_array(invalid_string_array());

    assert_invalid_accessor_errors(&mut access, logical_type)?;

    let mut encoded = Vec::new();
    let mut scratch = RowPackScratch::new(encoder.context());
    encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;

    let json = decoder.decode_json(&encoded)?;
    assert!(
        json["value"].is_null(),
        "invalid value should encode as null: {json}"
    );
    Ok(())
}

fn observe_context_invalid_is_null(logical_type: IngestedType) -> anyhow::Result<()> {
    let mut builder =
        RowPackContextBuilder::new(vec![scalar_col(logical_type)], RowPackConfig::default());
    let (mut access, _arrays) = access_for_array(invalid_string_array());

    assert_invalid_accessor_errors(&mut access, logical_type)?;

    builder.observe_row_from_access(&mut access)?;
    let _ = builder.finish()?;
    Ok(())
}

macro_rules! scalar_parse_failure_test {
    ($name:ident, $logical_type:expr) => {
        #[test]
        fn $name() -> anyhow::Result<()> {
            encode_invalid_as_null(scalar_ctx($logical_type), invalid_string_array())
        }
    };
}

scalar_parse_failure_test!(encode_bool_parse_failure_is_recorded, IngestedType::Bool);
scalar_parse_failure_test!(encode_int8_parse_failure_is_recorded, IngestedType::Int8);
scalar_parse_failure_test!(encode_int16_parse_failure_is_recorded, IngestedType::Int16);
scalar_parse_failure_test!(encode_int32_parse_failure_is_recorded, IngestedType::Int32);
scalar_parse_failure_test!(encode_int64_parse_failure_is_recorded, IngestedType::Int64);
scalar_parse_failure_test!(encode_uint8_parse_failure_is_recorded, IngestedType::UInt8);
scalar_parse_failure_test!(
    encode_uint16_parse_failure_is_recorded,
    IngestedType::UInt16
);
scalar_parse_failure_test!(
    encode_uint32_parse_failure_is_recorded,
    IngestedType::UInt32
);
scalar_parse_failure_test!(
    encode_uint64_parse_failure_is_recorded,
    IngestedType::UInt64
);
scalar_parse_failure_test!(
    encode_float32_parse_failure_is_recorded,
    IngestedType::Float32
);
scalar_parse_failure_test!(
    encode_float64_parse_failure_is_recorded,
    IngestedType::Float64
);
scalar_parse_failure_test!(encode_uuid_parse_failure_is_recorded, IngestedType::Uuid);
scalar_parse_failure_test!(
    encode_uuid256_parse_failure_is_recorded,
    IngestedType::Uuid256
);
scalar_parse_failure_test!(encode_ip_parse_failure_is_recorded, IngestedType::Ip);
scalar_parse_failure_test!(encode_date_parse_failure_is_recorded, IngestedType::Date);
scalar_parse_failure_test!(
    encode_timestamp_parse_failure_is_recorded,
    IngestedType::Timestamp
);
scalar_parse_failure_test!(
    encode_decimal128_parse_failure_is_recorded,
    IngestedType::Decimal128
);
scalar_parse_failure_test!(
    encode_decimal256_parse_failure_is_recorded,
    IngestedType::Decimal256
);

macro_rules! scalar_direct_context_tests {
    ($direct_name:ident, $context_name:ident, $logical_type:expr) => {
        #[test]
        fn $direct_name() -> anyhow::Result<()> {
            encode_from_access_invalid_is_null($logical_type)
        }

        #[test]
        fn $context_name() -> anyhow::Result<()> {
            observe_context_invalid_is_null($logical_type)
        }
    };
}

scalar_direct_context_tests!(
    direct_bool_invalid_is_null,
    context_bool_invalid_is_null,
    IngestedType::Bool
);
scalar_direct_context_tests!(
    direct_int8_invalid_is_null,
    context_int8_invalid_is_null,
    IngestedType::Int8
);
scalar_direct_context_tests!(
    direct_int16_invalid_is_null,
    context_int16_invalid_is_null,
    IngestedType::Int16
);
scalar_direct_context_tests!(
    direct_int32_invalid_is_null,
    context_int32_invalid_is_null,
    IngestedType::Int32
);
scalar_direct_context_tests!(
    direct_int64_invalid_is_null,
    context_int64_invalid_is_null,
    IngestedType::Int64
);
scalar_direct_context_tests!(
    direct_uint8_invalid_is_null,
    context_uint8_invalid_is_null,
    IngestedType::UInt8
);
scalar_direct_context_tests!(
    direct_uint16_invalid_is_null,
    context_uint16_invalid_is_null,
    IngestedType::UInt16
);
scalar_direct_context_tests!(
    direct_uint32_invalid_is_null,
    context_uint32_invalid_is_null,
    IngestedType::UInt32
);
scalar_direct_context_tests!(
    direct_uint64_invalid_is_null,
    context_uint64_invalid_is_null,
    IngestedType::UInt64
);
scalar_direct_context_tests!(
    direct_float32_invalid_is_null,
    context_float32_invalid_is_null,
    IngestedType::Float32
);
scalar_direct_context_tests!(
    direct_float64_invalid_is_null,
    context_float64_invalid_is_null,
    IngestedType::Float64
);
scalar_direct_context_tests!(
    direct_uuid_invalid_is_null,
    context_uuid_invalid_is_null,
    IngestedType::Uuid
);
scalar_direct_context_tests!(
    direct_uuid256_invalid_is_null,
    context_uuid256_invalid_is_null,
    IngestedType::Uuid256
);
scalar_direct_context_tests!(
    direct_ip_invalid_is_null,
    context_ip_invalid_is_null,
    IngestedType::Ip
);
scalar_direct_context_tests!(
    direct_date_invalid_is_null,
    context_date_invalid_is_null,
    IngestedType::Date
);
scalar_direct_context_tests!(
    direct_timestamp_invalid_is_null,
    context_timestamp_invalid_is_null,
    IngestedType::Timestamp
);
scalar_direct_context_tests!(
    direct_decimal128_invalid_is_null,
    context_decimal128_invalid_is_null,
    IngestedType::Decimal128
);
scalar_direct_context_tests!(
    direct_decimal256_invalid_is_null,
    context_decimal256_invalid_is_null,
    IngestedType::Decimal256
);

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
