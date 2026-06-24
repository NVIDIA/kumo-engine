use crate::parser::*;
use crate::*;
use arrow_array::{
    ArrayRef, Float32Array, Float64Array, Int32Array, Time32MillisecondArray, Time32SecondArray,
    Time64MicrosecondArray, Time64NanosecondArray,
};
use chrono::{DateTime, FixedOffset, NaiveDate, NaiveDateTime};
use num_bigint::BigInt;
use serde_json::Value as JsonValue;
use std::sync::Arc;

#[allow(dead_code)]
fn encode_row(encoder: &RowPackEncoder, row: &[ValueRef<'_>]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut scratch = RowPackScratch::new(encoder.context());
    encoder
        .encode_row_into(row, &mut out, &mut scratch)
        .unwrap();
    out
}

#[allow(dead_code)]
fn build_context() -> RowPackContext {
    RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![
            ColumnContext {
                name: "id".to_string(),
                logical_type: IngestedType::Int64,
                encoding: ColumnEncoding::Int {
                    signed: true,
                    varint: true,
                    bits: 64,
                },
            },
            ColumnContext {
                name: "name".to_string(),
                logical_type: IngestedType::String,
                encoding: ColumnEncoding::String {
                    encoding: StringEncoding::Dict {
                        values: vec![b"alice".to_vec(), b"bob".to_vec()],
                    },
                },
            },
            ColumnContext {
                name: "active".to_string(),
                logical_type: IngestedType::Bool,
                encoding: ColumnEncoding::Bool,
            },
            ColumnContext {
                name: "score".to_string(),
                logical_type: IngestedType::Float32,
                encoding: ColumnEncoding::Float {
                    storage: FloatStorage::F16,
                },
            },
            ColumnContext {
                name: "payload".to_string(),
                logical_type: IngestedType::Bytes,
                encoding: ColumnEncoding::Bytes { fixed_len: None },
            },
            ColumnContext {
                name: "created_at".to_string(),
                logical_type: IngestedType::Timestamp,
                encoding: ColumnEncoding::Timestamp,
            },
            ColumnContext {
                name: "embed".to_string(),
                logical_type: IngestedType::Float32Array,
                encoding: ColumnEncoding::FloatArray {
                    storage: FloatStorage::F32,
                },
            },
        ],
        estimated_row_bytes: 0,
    }
}

#[allow(dead_code)]
fn decimal256_bytes(value: &BigInt) -> [u8; 32] {
    let bytes = value.to_signed_bytes_le();
    let fill = if value.is_negative() { 0xff } else { 0x00 };
    let mut out = [fill; 32];
    if !bytes.is_empty() {
        out[..bytes.len()].copy_from_slice(&bytes);
    }
    out
}

#[test]
fn bool_from_numeric_ints() {
    let cols: Vec<ArrayRef> = vec![Arc::new(Int32Array::from(vec![0, 1, -1]))];
    let hints = vec![RowAccessHints {
        timestamp_format: None,
        timestamp_unit: None,
        decimal_scale: None,
        ..Default::default()
    }];
    let mut access = RowAccess::new(hints, RowAccessStats::new(None));
    access.set_cols(&cols);

    access.set_row(0);
    assert_eq!(access.get_bool(0).unwrap(), Some(false));
    access.set_row(1);
    assert_eq!(access.get_bool(0).unwrap(), Some(true));
    access.set_row(2);
    assert_eq!(access.get_bool(0).unwrap(), Some(true));
}

#[test]
fn bool_from_numeric_floats() {
    let cols: Vec<ArrayRef> = vec![Arc::new(Float64Array::from(vec![0.0, 1.0, -0.5, f64::NAN]))];
    let hints = vec![RowAccessHints {
        timestamp_format: None,
        timestamp_unit: None,
        decimal_scale: None,
        ..Default::default()
    }];
    let mut access = RowAccess::new(hints, RowAccessStats::new(None));
    access.set_cols(&cols);

    access.set_row(0);
    assert_eq!(access.get_bool(0).unwrap(), Some(false));
    access.set_row(1);
    assert_eq!(access.get_bool(0).unwrap(), Some(true));
    access.set_row(2);
    assert_eq!(access.get_bool(0).unwrap(), Some(true));
    access.set_row(3);
    assert_eq!(access.get_bool(0).unwrap(), Some(false));
}

#[test]
fn bool_from_float32_nan() {
    let cols: Vec<ArrayRef> = vec![Arc::new(Float32Array::from(vec![f32::NAN]))];
    let hints = vec![RowAccessHints {
        timestamp_format: None,
        timestamp_unit: None,
        decimal_scale: None,
        ..Default::default()
    }];
    let mut access = RowAccess::new(hints, RowAccessStats::new(None));
    access.set_cols(&cols);
    access.set_row(0);
    assert_eq!(access.get_bool(0).unwrap(), Some(false));
}

#[test]
fn parse_bool_string_variants() {
    for value in ["TRUE", "t", "Yes", "Y", "on", "enabled", "checked", "x"] {
        assert_eq!(parse_bool(value), Some(true));
    }
    for value in ["FALSE", "f", "No", "n", "off", "disabled", "unchecked", ""] {
        assert_eq!(parse_bool(value), Some(false));
    }
    for value in ["1", "-2", "0.5"] {
        assert_eq!(parse_bool(value), Some(true));
    }
    for value in ["0", "0.0", "NaN"] {
        assert_eq!(parse_bool(value), Some(false));
    }
    assert_eq!(parse_bool("\"yes\""), Some(true));
    assert_eq!(parse_bool("\"0\""), Some(false));
}

#[test]
fn parse_precompiled_timestamp_formats_match_chrono() {
    let cases: Vec<(&str, Vec<&str>)> = vec![
        ("%Y-%m-%d", vec!["1970-01-01", "2020-02-29"]),
        (
            "%Y-%m-%d %H:%M",
            vec!["1970-01-01 00:00", "2020-02-29 23:59"],
        ),
        (
            "%Y-%m-%d %H:%M:%S",
            vec!["1970-01-01 00:00:01", "2020-02-29 23:59:59"],
        ),
        (
            "%Y-%m-%d %H:%M:%S%.f",
            vec![
                "1970-01-01 00:00:00.5",
                "2020-02-29 23:59:59.123456789",
                "2020-02-29 23:59:59.000000001",
            ],
        ),
        (
            "%Y-%m-%dT%H:%M:%S",
            vec!["1970-01-01T00:00:02", "2020-02-29T23:59:59"],
        ),
        (
            "%Y-%m-%dT%H:%M:%S%.f",
            vec!["1970-01-01T00:00:03.250", "2020-02-29T23:59:59.999999999"],
        ),
        (
            "%Y-%m-%d %H:%M:%S%z",
            vec![
                "1970-01-01 00:00:00+0000",
                "1970-01-01 01:00:00+0100",
                "2020-02-29 23:59:59-0530",
            ],
        ),
        (
            "%Y-%m-%d %H:%M:%S %z",
            vec!["1970-01-01 00:00:00 +0000", "1970-01-01 01:00:00 +0100"],
        ),
        (
            "%Y-%m-%d %H:%M:%S%:z",
            vec!["1970-01-01 00:00:00+00:00", "1970-01-01 01:00:00+01:00"],
        ),
        (
            "%Y-%m-%d %H:%M:%S %:z",
            vec!["1970-01-01 00:00:00 +00:00", "1970-01-01 01:00:00 +01:00"],
        ),
        (
            "%Y-%m-%dT%H:%M:%S%z",
            vec!["1970-01-01T00:00:00+0000", "1970-01-01T01:00:00+0100"],
        ),
        (
            "%Y-%m-%dT%H:%M:%S%:z",
            vec!["1970-01-01T00:00:00+00:00", "1970-01-01T01:00:00+01:00"],
        ),
    ];

    for (fmt, values) in cases {
        let parser = TimestampFormatParser::compile(fmt).unwrap();
        for value in values {
            let fast_secs = parser.parse_timestamp(value).unwrap();
            let chrono_secs = if fmt.contains("%z") || fmt.contains("%:z") {
                let dt: DateTime<FixedOffset> = DateTime::parse_from_str(value, fmt).unwrap();
                dt.naive_utc().and_utc().timestamp()
            } else if fmt == "%Y-%m-%d" {
                let date = NaiveDate::parse_from_str(value, fmt).unwrap();
                date.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp()
            } else {
                let dt = NaiveDateTime::parse_from_str(value, fmt).unwrap();
                dt.and_utc().timestamp()
            };
            assert_eq!(fast_secs, chrono_secs);
        }
    }
}

#[test]
fn roundtrip_basic() {
    let ctx = build_context();
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let row = vec![
        ValueRef::Int64(42),
        ValueRef::String("alice"),
        ValueRef::Bool(true),
        ValueRef::Float32(3.5),
        ValueRef::Bytes(b"hello"),
        ValueRef::Timestamp(1_700_000_000),
        ValueRef::Float32Array(&[0.1, 0.2, 0.3]),
    ];
    let encoded = encode_row(&encoder, &row);
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let msg = decoder.decode_msgpack(&encoded).unwrap();
    let json = decoder.decode_json(&encoded).unwrap();
    assert!(!msg.is_empty());
    assert!(json.is_object());
}

#[test]
fn timestamp_nanos_roundtrip() {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "ts".to_string(),
            logical_type: IngestedType::Timestamp,
            encoding: ColumnEncoding::Timestamp,
        }],
        estimated_row_bytes: 0,
    };
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let row = vec![ValueRef::Timestamp(1_700_000_000)];
    let encoded = encode_row(&encoder, &row);
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let mut scratch = RowPackDecodeScratch::default();
    let val = decoder.decode_column_value(&encoded, 0, &mut scratch).unwrap();
    assert_eq!(val, ValueRef::Timestamp(1_700_000_000));
}

#[test]
fn decimal128_roundtrip() {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "amount".to_string(),
            logical_type: IngestedType::Decimal128,
            encoding: ColumnEncoding::Decimal {
                precision: 10,
                scale: 2,
                bytes: decimal_raw_bytes(10).unwrap(),
                storage: DecimalStorage::Varint,
            },
        }],
        estimated_row_bytes: 0,
    };
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let row = vec![ValueRef::Decimal128(-12345i128)];
    let encoded = encode_row(&encoder, &row);
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let json = decoder.decode_json(&encoded).unwrap();
    let JsonValue::Object(map) = json else {
        panic!("expected object");
    };
    assert_eq!(map.get("amount"), Some(&JsonValue::from("-123.45")));
}

#[test]
fn decimal256_roundtrip() {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "amount".to_string(),
            logical_type: IngestedType::Decimal256,
            encoding: ColumnEncoding::Decimal {
                precision: 50,
                scale: 4,
                bytes: decimal_raw_bytes(50).unwrap(),
                storage: DecimalStorage::Raw,
            },
        }],
        estimated_row_bytes: 0,
    };
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let big = BigInt::parse_bytes(b"123456789012345678901234567890", 10).unwrap();
    let row = vec![ValueRef::Decimal256(decimal256_bytes(&big))];
    let encoded = encode_row(&encoder, &row);
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let json = decoder.decode_json(&encoded).unwrap();
    let JsonValue::Object(map) = json else {
        panic!("expected object");
    };
    assert!(map.get("amount").and_then(|v| v.as_str()).is_some());
}

#[test]
fn decimal_raw_encoding_when_out_of_range() {
    let config = RowPackConfig::default();
    let mut builder = RowPackContextBuilder::new(
        vec![RowPackColumn {
            name: "amount".to_string(),
            logical_type: IngestedType::Decimal128,
            fixed_len: None,
            precision: Some(10),
            scale: Some(2),
        }],
        config,
    );
    builder
        .observe_row(&[ValueRef::Decimal128(i128::MAX)])
        .unwrap();
    let ctx = builder.finish().unwrap();
    match &ctx.columns[0].encoding {
        ColumnEncoding::Decimal {
            storage: DecimalStorage::Varint,
            ..
        } => {}
        other => panic!("unexpected encoding: {:?}", other),
    }
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let row = vec![ValueRef::Decimal128(i128::MAX)];
    let encoded = encode_row(&encoder, &row);
    assert_eq!(encoded[0] & 0x03, 2);
}

#[test]
fn decimal_int_encoding_scale_zero() {
    let config = RowPackConfig::default();
    let mut builder = RowPackContextBuilder::new(
        vec![RowPackColumn {
            name: "amount".to_string(),
            logical_type: IngestedType::Decimal128,
            fixed_len: None,
            precision: Some(10),
            scale: Some(0),
        }],
        config,
    );
    builder
        .observe_row(&[ValueRef::Decimal128(12345i128)])
        .unwrap();
    let ctx = builder.finish().unwrap();
    match &ctx.columns[0].encoding {
        ColumnEncoding::Decimal {
            storage: DecimalStorage::Varint,
            ..
        } => {}
        other => panic!("unexpected encoding: {:?}", other),
    }
}

#[test]
fn decimal_scale_nonzero_int_roundtrip() {
    let config = RowPackConfig::default();
    let mut builder = RowPackContextBuilder::new(
        vec![RowPackColumn {
            name: "amount".to_string(),
            logical_type: IngestedType::Decimal128,
            fixed_len: None,
            precision: Some(10),
            scale: Some(3),
        }],
        config,
    );
    builder
        .observe_row(&[ValueRef::Decimal128(123456i128)])
        .unwrap();
    let ctx = builder.finish().unwrap();
    match &ctx.columns[0].encoding {
        ColumnEncoding::Decimal {
            storage: DecimalStorage::Varint,
            ..
        } => {}
        other => panic!("unexpected encoding: {:?}", other),
    }
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let row = vec![ValueRef::Decimal128(123456i128)];
    let encoded = encode_row(&encoder, &row);
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let json = decoder.decode_json(&encoded).unwrap();
    let JsonValue::Object(map) = json else {
        panic!("expected object");
    };
    assert_eq!(map.get("amount"), Some(&JsonValue::from("123.456")));
}

#[test]
fn decimal_int_fallback_when_out_of_range() {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "amount".to_string(),
            logical_type: IngestedType::Decimal128,
            encoding: ColumnEncoding::Decimal {
                precision: 20,
                scale: 0,
                bytes: decimal_raw_bytes(20).unwrap(),
                storage: DecimalStorage::Varint,
            },
        }],
        estimated_row_bytes: 0,
    };
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let big = i128::MAX;
    let row = vec![ValueRef::Decimal128(big)];
    let encoded = encode_row(&encoder, &row);
    assert_eq!(encoded[0] & 0x03, 2);
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let json = decoder.decode_json(&encoded).unwrap();
    let JsonValue::Object(map) = json else {
        panic!("expected object");
    };
    assert!(map.get("amount").unwrap().as_str().is_some());
}

#[test]
fn decimal_varint_thresholds_all_precisions() {
    fn pow10_i128(exp: u32) -> i128 {
        let mut out = 1i128;
        for _ in 0..exp {
            out = out.saturating_mul(10);
        }
        out
    }

    for precision in 1..=38u32 {
        let bytes = decimal_raw_bytes(precision).unwrap();
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "amount".to_string(),
                logical_type: IngestedType::Decimal128,
                encoding: ColumnEncoding::Decimal {
                    precision,
                    scale: 0,
                    bytes,
                    storage: DecimalStorage::Varint,
                },
            }],
            estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let small = vec![ValueRef::Decimal128(1)];
        let encoded_small = encode_row(&encoder, &small);
        let expected_small_tag = if bytes == 1 { 2 } else { 1 };
        assert_eq!(encoded_small[0] & 0x03, expected_small_tag);

        let big = pow10_i128(precision) - 1;
        let encoded_big = encode_row(&encoder, &[ValueRef::Decimal128(big)]);
        assert_eq!(encoded_big[0] & 0x03, 2);
    }
}

#[test]
fn dict_fallback() {
    let ctx = build_context();
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let row = vec![
        ValueRef::Int64(1),
        ValueRef::String("carol"),
        ValueRef::Bool(false),
        ValueRef::Float32(1.0),
        ValueRef::Bytes(b"data"),
        ValueRef::Timestamp(0),
        ValueRef::Float32Array(&[]),
    ];
    let encoded = encode_row(&encoder, &row);
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let json = decoder.decode_json(&encoded).unwrap();
    assert!(json.to_string().contains("carol"));
}

#[test]
fn alphabet_encoding_roundtrip() {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "code".to_string(),
            logical_type: IngestedType::String,
            encoding: ColumnEncoding::String {
                encoding: StringEncoding::Alphabet {
                    alphabet: b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-".to_vec(),
                },
            },
        }],
        estimated_row_bytes: 0,
    };
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let row = vec![ValueRef::String("USER-0001")];
    let encoded = encode_row(&encoder, &row);
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let json = decoder.decode_json(&encoded).unwrap();
    let JsonValue::Object(map) = json else {
        panic!("expected object");
    };
    assert_eq!(map.get("code").and_then(|v| v.as_str()), Some("USER-0001"));
}

#[test]
fn alphabet_encoding_fallback() {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "code".to_string(),
            logical_type: IngestedType::String,
            encoding: ColumnEncoding::String {
                encoding: StringEncoding::Alphabet {
                    alphabet: b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-".to_vec(),
                },
            },
        }],
        estimated_row_bytes: 0,
    };
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let row = vec![ValueRef::String("USER-0001?")];
    let encoded = encode_row(&encoder, &row);
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let json = decoder.decode_json(&encoded).unwrap();
    let JsonValue::Object(map) = json else {
        panic!("expected object");
    };
    assert_eq!(map.get("code").and_then(|v| v.as_str()), Some("USER-0001?"));
}

#[test]
fn alphabet_encoding_selected_for_short_strings() {
    let mut builder = RowPackContextBuilder::new(
        vec![RowPackColumn {
            name: "code".to_string(),
            logical_type: IngestedType::String,
            fixed_len: None,
            precision: None,
            scale: None,
        }],
        RowPackConfig::default(),
    );
    for idx in 0..100 {
        let value = format!("USER-{:04}", idx);
        builder.observe_row(&[ValueRef::String(&value)]).unwrap();
    }
    let ctx = builder.finish().unwrap();
    match &ctx.columns[0].encoding {
        ColumnEncoding::String {
            encoding: StringEncoding::Alphabet { .. },
        } => {}
        other => panic!("unexpected encoding: {:?}", other),
    }
}

#[test]
fn alphabet_encoding_selected_at_long_threshold() {
    let config = RowPackConfig::default();
    let max_len = config.string_long_min_len.unwrap_or(64);
    let mut builder = RowPackContextBuilder::new(
        vec![RowPackColumn {
            name: "code".to_string(),
            logical_type: IngestedType::String,
            fixed_len: None,
            precision: None,
            scale: None,
        }],
        config,
    );
    let value = "ABCD".repeat(max_len.div_ceil(4))[..max_len].to_string();
    for _ in 0..50 {
        builder.observe_row(&[ValueRef::String(&value)]).unwrap();
    }
    let ctx = builder.finish().unwrap();
    match &ctx.columns[0].encoding {
        ColumnEncoding::String {
            encoding: StringEncoding::Alphabet { .. },
        } => {}
        other => panic!("unexpected encoding: {:?}", other),
    }
}

#[test]
fn bpe_encoding_roundtrip() {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "code".to_string(),
            logical_type: IngestedType::String,
            encoding: ColumnEncoding::String {
                encoding: StringEncoding::Bpe {
                    dict: vec![b"TOKEN123".to_vec()],
                },
            },
        }],
        estimated_row_bytes: 0,
    };
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let row = vec![ValueRef::String("TOKEN123TOKEN123")];
    let encoded = encode_row(&encoder, &row);
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let json = decoder.decode_json(&encoded).unwrap();
    let JsonValue::Object(map) = json else {
        panic!("expected object");
    };
    assert_eq!(
        map.get("code").and_then(|v| v.as_str()),
        Some("TOKEN123TOKEN123")
    );
}

#[test]
fn bpe_selected_over_alphabet() {
    let mut config = RowPackConfig::default();
    config.bpe_min_gain_ratio = Some(1.0);
    config.bpe_max_dict_entries = Some(64);
    config.bpe_max_dict_bytes = Some(1024);
    let mut builder = RowPackContextBuilder::new(
        vec![RowPackColumn {
            name: "code".to_string(),
            logical_type: IngestedType::String,
            fixed_len: None,
            precision: None,
            scale: None,
        }],
        config,
    );
    for _ in 0..50 {
        builder
            .observe_row(&[ValueRef::String("TOKEN123TOKEN123TOKEN123")])
            .unwrap();
    }
    let ctx = builder.finish().unwrap();
    match &ctx.columns[0].encoding {
        ColumnEncoding::String {
            encoding: StringEncoding::Bpe { .. },
        } => {}
        other => panic!("unexpected encoding: {:?}", other),
    }
}

#[test]
fn bpe_decode_rejects_token_out_of_range() {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "code".to_string(),
            logical_type: IngestedType::String,
            encoding: ColumnEncoding::String {
                encoding: StringEncoding::Bpe {
                    dict: vec![b"TOKEN".to_vec()],
                },
            },
        }],
        estimated_row_bytes: 0,
    };
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let mut tags = BitWriter::with_capacity(1);
    tags.write_bits(1, 2);
    let mut row = tags.finish();
    write_varint(1, &mut row);
    let tokens = vec![256 + 10];
    let bits = bpe_token_bits(1);
    write_tokens_packed(&mut row, &tokens, bits);
    let err = decoder.decode_json(&row).unwrap_err();
    assert!(err.to_string().contains("bpe token out of range"));
}

#[test]
fn bpe_decode_rejects_excess_tokens() {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "code".to_string(),
            logical_type: IngestedType::String,
            encoding: ColumnEncoding::String {
                encoding: StringEncoding::Bpe {
                    dict: vec![b"TOKEN".to_vec()],
                },
            },
        }],
        estimated_row_bytes: 0,
    };
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let mut tags = BitWriter::with_capacity(1);
    tags.write_bits(1, 2);
    let mut row = tags.finish();
    write_varint((MAX_ARRAY_LEN as u64) + 1, &mut row);
    let err = decoder.decode_json(&row).unwrap_err();
    assert!(err.to_string().contains("bpe token count too large"));
}

#[allow(dead_code)]
fn encode_string_size(encoding: StringEncoding, value: &str) -> usize {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "code".to_string(),
            logical_type: IngestedType::String,
            encoding: ColumnEncoding::String { encoding },
        }],
        estimated_row_bytes: 0,
    };
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let row = vec![ValueRef::String(value)];
    let encoded = encode_row(&encoder, &row);
    encoded.len()
}

#[test]
fn alphabet_beats_bpe_on_structured_strings() {
    let samples = vec![
        "2024-01-05T03:24:15Z",
        "2024-02-11T14:09:26Z",
        "2024-03-19T22:58:01Z",
        "2024-04-07T08:03:44Z",
        "2024-05-23T16:47:10Z",
        "2024-06-30T01:11:59Z",
        "2024-07-02T12:35:22Z",
        "2024-08-14T19:00:05Z",
        "2024-09-28T05:42:33Z",
        "2024-10-31T23:59:59Z",
    ];
    let mut alpha_config = RowPackConfig::default();
    alpha_config.max_dict_entries = Some(0);
    alpha_config.bpe_max_dict_entries = Some(0);
    alpha_config.bpe_max_dict_bytes = Some(0);
    let alphabet_encoding = build_string_encoding(&samples, alpha_config);
    let StringEncoding::Alphabet { alphabet } = alphabet_encoding else {
        panic!("expected alphabet encoding");
    };
    let bpe_config = RowPackConfig::default();
    let bpe_dict = train_bpe_dict(
        &samples
            .iter()
            .map(|s| s.as_bytes().to_vec())
            .collect::<Vec<_>>(),
        bpe_config.bpe_max_dict_entries(),
        bpe_config.bpe_max_dict_bytes(),
    );
    for value in samples {
        let alpha_size = encode_string_size(
            StringEncoding::Alphabet {
                alphabet: alphabet.clone(),
            },
            value,
        );
        let bpe_size = encode_string_size(
            StringEncoding::Bpe {
                dict: bpe_dict.clone(),
            },
            value,
        );
        assert!(
            alpha_size < bpe_size,
            "alphabet size {} not smaller than bpe size {} for {}",
            alpha_size,
            bpe_size,
            value
        );
        assert!(
            alpha_size < value.len(),
            "alphabet size {} not smaller than raw size {} for {}",
            alpha_size,
            value.len(),
            value
        );
    }
}

#[test]
fn bpe_beats_alphabet_on_english_sentences() {
    let samples = vec![
            "We left early for the concert and found a quiet seat near the aisle before the crowd arrived.",
            "The pilot announced a smooth flight, and the cabin lights dimmed as everyone settled in.",
            "She organized the pantry by date and labeled each jar so cooking would be quicker later.",
            "A soft breeze moved through the trees while the hikers checked the map and shared water.",
            "The mechanic explained the repair clearly, then showed the worn part to avoid confusion.",
            "We wrote a short update for the team and included a timeline for the next review.",
            "His sketchbook held dozens of city scenes, each capturing traffic, shadows, and street noise.",
            "The festival opened at noon, and vendors arranged colorful tents along the main walkway.",
            "She practiced the presentation aloud, adjusting pacing so each section felt balanced.",
            "The museum tour ended with a restored mural that had been hidden behind the wall.",
        ];
    let mut alpha_config = RowPackConfig::default();
    alpha_config.max_dict_entries = Some(0);
    alpha_config.bpe_max_dict_entries = Some(0);
    alpha_config.bpe_max_dict_bytes = Some(0);
    let alphabet_encoding = build_string_encoding(&samples, alpha_config);
    let StringEncoding::Alphabet { alphabet } = alphabet_encoding else {
        panic!("expected alphabet encoding");
    };
    let bpe_dict = train_bpe_dict(
        &samples
            .iter()
            .map(|s| s.as_bytes().to_vec())
            .collect::<Vec<_>>(),
        256,
        32 * 1024,
    );
    assert!(!bpe_dict.is_empty(), "expected bpe dict to be non-empty");
    let mut total_alpha = 0usize;
    let mut total_bpe = 0usize;
    for value in samples {
        let alpha_size = encode_string_size(
            StringEncoding::Alphabet {
                alphabet: alphabet.clone(),
            },
            value,
        );
        let bpe_size = encode_string_size(
            StringEncoding::Bpe {
                dict: bpe_dict.clone(),
            },
            value,
        );
        total_alpha += alpha_size;
        total_bpe += bpe_size;
        assert!(
            bpe_size < value.len(),
            "bpe size {} not smaller than raw size {} for {}",
            bpe_size,
            value.len(),
            value
        );
    }
    assert!(
        total_bpe < total_alpha,
        "bpe total {} not smaller than alphabet total {}",
        total_bpe,
        total_alpha
    );
}

#[allow(dead_code)]
fn build_string_encoding(samples: &[&str], config: RowPackConfig) -> StringEncoding {
    let mut builder = RowPackContextBuilder::new(
        vec![RowPackColumn {
            name: "code".to_string(),
            logical_type: IngestedType::String,
            fixed_len: None,
            precision: None,
            scale: None,
        }],
        config,
    );
    for value in samples {
        builder.observe_row(&[ValueRef::String(value)]).unwrap();
    }
    let ctx = builder.finish().unwrap();
    let ColumnEncoding::String { encoding } = ctx.columns[0].encoding.clone() else {
        panic!("expected string encoding");
    };
    encoding
}

#[test]
fn float_array_float16_encoding() {
    let mut builder = RowPackContextBuilder::new(
        vec![RowPackColumn {
            name: "embed".to_string(),
            logical_type: IngestedType::Float16Array,
            fixed_len: None,
            precision: None,
            scale: None,
        }],
        RowPackConfig::default(),
    );
    builder
        .observe_row(&[ValueRef::Float32Array(&[0.1, 0.2, 0.3])])
        .unwrap();
    let ctx = builder.finish().unwrap();
    match &ctx.columns[0].encoding {
        ColumnEncoding::FloatArray { storage } => {
            assert_eq!(*storage, FloatStorage::F16);
        }
        other => panic!("unexpected encoding: {:?}", other),
    }
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let row = vec![ValueRef::Float32Array(&[0.1, 0.2, 0.3])];
    let encoded = encode_row(&encoder, &row);
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let json = decoder.decode_json(&encoded).unwrap();
    assert!(json.to_string().contains("0.1"));
}

#[test]
fn bytes_roundtrip() {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "blob".to_string(),
            logical_type: IngestedType::Bytes,
            encoding: ColumnEncoding::Bytes { fixed_len: None },
        }],
        estimated_row_bytes: 0,
    };
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let row = vec![ValueRef::Bytes(b"xyz")];
    let encoded = encode_row(&encoder, &row);
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let json = decoder.decode_json(&encoded).unwrap();
    assert!(json.to_string().contains("eHl6"));
}

#[test]
fn bytes_fixed_len_success() {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "blob".to_string(),
            logical_type: IngestedType::Bytes,
            encoding: ColumnEncoding::Bytes { fixed_len: Some(4) },
        }],
        estimated_row_bytes: 0,
    };
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let row = vec![ValueRef::Bytes(b"ABCD")];
    let encoded = encode_row(&encoder, &row);
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let json = decoder.decode_json(&encoded).unwrap();
    assert!(json.to_string().contains("QUJDRA=="));
}

#[test]
fn bytes_fixed_len_fallback() {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "blob".to_string(),
            logical_type: IngestedType::Bytes,
            encoding: ColumnEncoding::Bytes { fixed_len: Some(4) },
        }],
        estimated_row_bytes: 0,
    };
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let row = vec![ValueRef::Bytes(b"ABCDE")];
    let encoded = encode_row(&encoder, &row);
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let json = decoder.decode_json(&encoded).unwrap();
    assert!(json.to_string().contains("QUJDREU="));
}

#[test]
fn uuid_json_hex() {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "id".to_string(),
            logical_type: IngestedType::Uuid,
            encoding: ColumnEncoding::Uuid,
        }],
        estimated_row_bytes: 0,
    };
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let value = [
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
        0xff,
    ];
    let row = vec![ValueRef::Uuid(value)];
    let encoded = encode_row(&encoder, &row);
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let json = decoder.decode_json(&encoded).unwrap();
    assert!(json
        .to_string()
        .contains("00112233445566778899aabbccddeeff"));
}

#[test]
fn decimal128_truncates_to_int() {
    let array = arrow_array::Decimal128Array::from_iter_values([12345i128])
        .with_precision_and_scale(10, 2)
        .unwrap();
    let cols: Vec<arrow_array::ArrayRef> = vec![std::sync::Arc::new(array)];
    let hints = vec![RowAccessHints {
        timestamp_format: None,
        timestamp_unit: None,
        decimal_scale: Some(2),
        ..Default::default()
    }];
    let stats = RowAccessStats::new(None);
    let mut access = RowAccess::new(hints, stats);
    access.set_cols(&cols);
    access.set_row(0);
    let value = access.get_int64(0).unwrap().unwrap();
    assert_eq!(value, 123);
}

#[test]
fn decimal128_scales_to_float() {
    let array = arrow_array::Decimal128Array::from_iter_values([12345i128])
        .with_precision_and_scale(10, 2)
        .unwrap();
    let cols: Vec<arrow_array::ArrayRef> = vec![std::sync::Arc::new(array)];
    let hints = vec![RowAccessHints {
        timestamp_format: None,
        timestamp_unit: None,
        decimal_scale: Some(2),
        ..Default::default()
    }];
    let stats = RowAccessStats::new(None);
    let mut access = RowAccess::new(hints, stats);
    access.set_cols(&cols);
    access.set_row(0);
    let value = access.get_float64(0).unwrap().unwrap();
    assert!((value - 123.45).abs() < 1e-6);
}

#[test]
fn uuid_value_ref_roundtrip() {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "id".to_string(),
            logical_type: IngestedType::Uuid,
            encoding: ColumnEncoding::Uuid,
        }],
        estimated_row_bytes: 0,
    };
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let value = [
        0xde, 0xad, 0xbe, 0xef, 0x12, 0x34, 0x56, 0x78, 0x90, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45,
        0x67,
    ];
    let row = vec![ValueRef::Uuid(value)];
    let encoded = encode_row(&encoder, &row);
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let mut scratch = RowPackDecodeScratch::default();
    let decoded = decoder
        .decode_column_value(&encoded, 0, &mut scratch)
        .unwrap();
    match decoded {
        ValueRef::Uuid(bytes) => assert_eq!(bytes, value),
        other => panic!("unexpected value ref: {:?}", other),
    }
}

#[test]
fn uuid256_json_hex() {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "id".to_string(),
            logical_type: IngestedType::Uuid256,
            encoding: ColumnEncoding::Uuid256,
        }],
        estimated_row_bytes: 0,
    };
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let mut value = [0u8; 32];
    for (idx, byte) in value.iter_mut().enumerate() {
        *byte = idx as u8;
    }
    let row = vec![ValueRef::Uuid256(value)];
    let encoded = encode_row(&encoder, &row);
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let json = decoder.decode_json(&encoded).unwrap();
    assert!(json
        .to_string()
        .contains("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"));
}

#[test]
fn uuid256_value_ref_roundtrip() {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "id".to_string(),
            logical_type: IngestedType::Uuid256,
            encoding: ColumnEncoding::Uuid256,
        }],
        estimated_row_bytes: 0,
    };
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let mut value = [0u8; 32];
    for (idx, byte) in value.iter_mut().enumerate() {
        *byte = 0xffu8.wrapping_sub(idx as u8);
    }
    let row = vec![ValueRef::Uuid256(value)];
    let encoded = encode_row(&encoder, &row);
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let mut scratch = RowPackDecodeScratch::default();
    let decoded = decoder
        .decode_column_value(&encoded, 0, &mut scratch)
        .unwrap();
    match decoded {
        ValueRef::Uuid256(bytes) => assert_eq!(bytes, value),
        other => panic!("unexpected value ref: {:?}", other),
    }
}

#[test]
fn uuid_string_parse_error() {
    use std::sync::Arc;

    let arr = arrow_array::StringArray::from(vec!["not-a-uuid"]);
    let cols: Vec<ArrayRef> = vec![Arc::new(arr)];
    let hints = vec![RowAccessHints {
        timestamp_format: None,
        timestamp_unit: None,
        decimal_scale: None,
        ..Default::default()
    }];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(hints, stats);
    access.set_cols(&cols);
    access.set_row(0);
    assert!(access.get_uuid(0).is_err());
}

#[test]
fn uuid256_string_parse_error() {
    use std::sync::Arc;

    let text = "z".repeat(64);
    let arr = arrow_array::StringArray::from(vec![text.as_str()]);
    let cols: Vec<ArrayRef> = vec![Arc::new(arr)];
    let hints = vec![RowAccessHints {
        timestamp_format: None,
        timestamp_unit: None,
        decimal_scale: None,
        ..Default::default()
    }];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(hints, stats);
    access.set_cols(&cols);
    access.set_row(0);
    assert!(access.get_uuid256(0).is_err());
}

#[test]
fn ip_string_parse_error() {
    use std::sync::Arc;

    let arr = arrow_array::StringArray::from(vec!["999.999.999.999"]);
    let cols: Vec<ArrayRef> = vec![Arc::new(arr)];
    let hints = vec![RowAccessHints {
        timestamp_format: None,
        timestamp_unit: None,
        decimal_scale: None,
        ..Default::default()
    }];
    let stats = RowAccessStats::default();
    let mut access = RowAccess::new(hints, stats);
    access.set_cols(&cols);
    access.set_row(0);
    assert!(access.get_ip(0).is_err());
}

#[test]
fn ip_json_string() {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "ip".to_string(),
            logical_type: IngestedType::Ip,
            encoding: ColumnEncoding::Ip,
        }],
        estimated_row_bytes: 0,
    };
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let ip = std::net::Ipv4Addr::new(127, 0, 0, 1)
        .to_ipv6_mapped()
        .octets();
    let row = vec![ValueRef::Ip(ip)];
    let encoded = encode_row(&encoder, &row);
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let json = decoder.decode_json(&encoded).unwrap();
    assert!(json.to_string().contains("127.0.0.1"));
}

#[test]
fn ip_value_ref_roundtrip() {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "ip".to_string(),
            logical_type: IngestedType::Ip,
            encoding: ColumnEncoding::Ip,
        }],
        estimated_row_bytes: 0,
    };
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let ip = std::net::Ipv6Addr::LOCALHOST.octets();
    let row = vec![ValueRef::Ip(ip)];
    let encoded = encode_row(&encoder, &row);
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let mut scratch = RowPackDecodeScratch::default();
    let decoded = decoder
        .decode_column_value(&encoded, 0, &mut scratch)
        .unwrap();
    match decoded {
        ValueRef::Ip(bytes) => assert_eq!(bytes, ip),
        other => panic!("unexpected value ref: {:?}", other),
    }
}

#[test]
fn small_int_encodings() {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![
            ColumnContext {
                name: "i8".to_string(),
                logical_type: IngestedType::Int8,
                encoding: ColumnEncoding::Int {
                    signed: true,
                    varint: false,
                    bits: 8,
                },
            },
            ColumnContext {
                name: "u8".to_string(),
                logical_type: IngestedType::UInt8,
                encoding: ColumnEncoding::Int {
                    signed: false,
                    varint: false,
                    bits: 8,
                },
            },
            ColumnContext {
                name: "i16".to_string(),
                logical_type: IngestedType::Int16,
                encoding: ColumnEncoding::Int {
                    signed: true,
                    varint: false,
                    bits: 16,
                },
            },
            ColumnContext {
                name: "u16".to_string(),
                logical_type: IngestedType::UInt16,
                encoding: ColumnEncoding::Int {
                    signed: false,
                    varint: false,
                    bits: 16,
                },
            },
        ],
        estimated_row_bytes: 0,
    };
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let row = vec![
        ValueRef::Int32(-7),
        ValueRef::UInt32(200),
        ValueRef::Int32(32000),
        ValueRef::UInt32(65000),
    ];
    let encoded = encode_row(&encoder, &row);
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let json = decoder.decode_json(&encoded).unwrap();
    assert!(json.to_string().contains("-7"));
    assert!(json.to_string().contains("200"));
    assert!(json.to_string().contains("32000"));
    assert!(json.to_string().contains("65000"));
}

#[test]
fn varint_int32_negative() {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "i32".to_string(),
            logical_type: IngestedType::Int32,
            encoding: ColumnEncoding::Int {
                signed: true,
                varint: true,
                bits: 32,
            },
        }],
        estimated_row_bytes: 0,
    };
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let row = vec![ValueRef::Int32(-12345)];
    let encoded = encode_row(&encoder, &row);
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let json = decoder.decode_json(&encoded).unwrap();
    assert!(json.to_string().contains("-12345"));
}

#[test]
fn lz4_encoding_selected() {
    let mut config = RowPackConfig::default();
    config.string_long_min_len = Some(8);
    config.string_long_min_ratio = Some(0.9);
    config.max_dict_entries = Some(0);
    let mut builder = RowPackContextBuilder::new(
        vec![RowPackColumn {
            name: "text".to_string(),
            logical_type: IngestedType::String,
            fixed_len: None,
            precision: None,
            scale: None,
        }],
        config,
    );
    let value = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string();
    builder.observe_row(&[ValueRef::String(&value)]).unwrap();
    let ctx = builder.finish().unwrap();
    match &ctx.columns[0].encoding {
        ColumnEncoding::String {
            encoding: StringEncoding::Lz4,
        } => {}
        other => panic!("unexpected encoding: {:?}", other),
    }
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let row = vec![ValueRef::String(&value)];
    let encoded = encode_row(&encoder, &row);
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let json = decoder.decode_json(&encoded).unwrap();
    assert!(json.to_string().contains("aaaaaaaa"));
}

#[test]
fn lz4_falls_back_to_raw() {
    let mut config = RowPackConfig::default();
    config.string_long_min_len = Some(8);
    config.string_long_min_ratio = Some(0.5);
    config.max_dict_entries = Some(0);
    let mut builder = RowPackContextBuilder::new(
        vec![RowPackColumn {
            name: "text".to_string(),
            logical_type: IngestedType::String,
            fixed_len: None,
            precision: None,
            scale: None,
        }],
        config,
    );
    let value = "x3f9A2kLmN8pQrStUvWxYz0123456789-longer-string".to_string();
    builder.observe_row(&[ValueRef::String(&value)]).unwrap();
    let ctx = builder.finish().unwrap();
    match &ctx.columns[0].encoding {
        ColumnEncoding::String {
            encoding: StringEncoding::Raw,
        } => {}
        other => panic!("unexpected encoding: {:?}", other),
    }
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let row = vec![ValueRef::String(&value)];
    let encoded = encode_row(&encoder, &row);
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let json = decoder.decode_json(&encoded).unwrap();
    assert!(json.to_string().contains("x3f9A2k"));
}

#[test]
fn dict_fallback_string() {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "name".to_string(),
            logical_type: IngestedType::String,
            encoding: ColumnEncoding::String {
                encoding: StringEncoding::Dict {
                    values: vec![b"alice".to_vec()],
                },
            },
        }],
        estimated_row_bytes: 0,
    };
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let row = vec![ValueRef::String("carol")];
    let encoded = encode_row(&encoder, &row);
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let json = decoder.decode_json(&encoded).unwrap();
    assert!(json.to_string().contains("carol"));
}

#[test]
fn null_roundtrip() {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![
            ColumnContext {
                name: "id".to_string(),
                logical_type: IngestedType::Int64,
                encoding: ColumnEncoding::Int {
                    signed: true,
                    varint: true,
                    bits: 64,
                },
            },
            ColumnContext {
                name: "note".to_string(),
                logical_type: IngestedType::String,
                encoding: ColumnEncoding::String {
                    encoding: StringEncoding::Raw,
                },
            },
        ],
        estimated_row_bytes: 0,
    };
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let row = vec![ValueRef::Null, ValueRef::Null];
    let encoded = encode_row(&encoder, &row);
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let json = decoder.decode_json(&encoded).unwrap();
    assert!(json.to_string().contains("null"));
}

#[test]
fn decode_rejects_oversize_string() {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "name".to_string(),
            logical_type: IngestedType::String,
            encoding: ColumnEncoding::String {
                encoding: StringEncoding::Raw,
            },
        }],
        estimated_row_bytes: 0,
    };
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let mut tags = BitWriter::new();
    tags.write_bits(1, 2);
    let mut row = tags.finish();
    row.push(0x80);
    row.push(0x80);
    row.push(0x80);
    row.push(0x80);
    row.push(0x10);
    let err = decoder.decode_json(&row).unwrap_err();
    assert!(err.to_string().contains("too large"));
}

#[test]
fn decode_rejects_oversize_array() {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "embed".to_string(),
            logical_type: IngestedType::Float32Array,
            encoding: ColumnEncoding::FloatArray {
                storage: FloatStorage::F32,
            },
        }],
        estimated_row_bytes: 0,
    };
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let mut tags = BitWriter::new();
    tags.write_bits(1, 2);
    let mut row = tags.finish();
    row.push(0xff);
    row.push(0xff);
    row.push(0xff);
    row.push(0xff);
    row.push(0x0f);
    let err = decoder.decode_json(&row).unwrap_err();
    assert!(err.to_string().contains("array length"));
}

#[test]
fn decode_rejects_truncated_bool_bitstream() {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "active".to_string(),
            logical_type: IngestedType::Bool,
            encoding: ColumnEncoding::Bool,
        }],
        estimated_row_bytes: 0,
    };
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let mut tags = BitWriter::new();
    tags.write_bits(1, 2);
    let row = tags.finish();
    let err = decoder.decode_json(&row).unwrap_err();
    assert!(err.to_string().contains("truncated"));
}

#[test]
fn decode_rejects_invalid_tag() {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "id".to_string(),
            logical_type: IngestedType::Int64,
            encoding: ColumnEncoding::Int {
                signed: true,
                varint: true,
                bits: 64,
            },
        }],
        estimated_row_bytes: 0,
    };
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let mut tags = BitWriter::new();
    tags.write_bits(3, 2);
    let row = tags.finish();
    let err = decoder.decode_json(&row).unwrap_err();
    assert!(err.to_string().contains("invalid tag"));
}

#[test]
fn decode_rejects_corrupt_lz4() {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        columns: vec![ColumnContext {
            name: "text".to_string(),
            logical_type: IngestedType::String,
            encoding: ColumnEncoding::String {
                encoding: StringEncoding::Lz4,
            },
        }],
        estimated_row_bytes: 0,
    };
    let decoder = RowPackDecoder::new(ctx).unwrap();
    let mut tags = BitWriter::new();
    tags.write_bits(1, 2);
    let mut row = tags.finish();
    row.push(0x05);
    row.push(0x05);
    row.extend_from_slice(&[0x00, 0x01, 0x02, 0x03, 0x04]);
    let err = decoder.decode_json(&row).unwrap_err();
    assert!(err.to_string().contains("lz4"));
}

// ---------------------------------------------------------------------------
// get_timestamp with timestamp_format on integer columns (YYYYMMDD support)
// ---------------------------------------------------------------------------

/// Helper: build a RowAccess with one column, set the given Arrow array, and
/// call `get_timestamp(0)`.
fn get_timestamp_from_array(
    array: ArrayRef,
    format: Option<&str>,
    unit: Option<arrow_schema::TimeUnit>,
) -> anyhow::Result<Option<i64>> {
    let hints = vec![RowAccessHints {
        timestamp_format: format.map(|s| s.to_string()),
        timestamp_unit: unit,
        ..Default::default()
    }];
    let stats = RowAccessStats::new(None);
    let mut access = RowAccess::new(hints, stats);
    let cols: Vec<ArrayRef> = vec![array];
    access.set_cols(&cols);
    access.set_row(0);
    access.get_timestamp(0)
}

#[test]
fn int32_yyyymmdd_with_format_hint() {
    // 20240301 as Int32 with format "%Y%m%d" should parse to 2024-03-01 00:00:00 UTC
    let arr: ArrayRef = Arc::new(Int32Array::from(vec![20240301]));
    let secs = get_timestamp_from_array(arr, Some("%Y%m%d"), None)
        .unwrap()
        .unwrap();
    let expected = chrono::NaiveDate::from_ymd_opt(2024, 3, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc()
        .timestamp();
    assert_eq!(secs, expected);
}

#[test]
fn int64_yyyymmdd_with_format_hint() {
    // Same test but with Int64 backing
    let arr: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![19930715i64]));
    let secs = get_timestamp_from_array(arr, Some("%Y%m%d"), None)
        .unwrap()
        .unwrap();
    let expected = chrono::NaiveDate::from_ymd_opt(1993, 7, 15)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc()
        .timestamp();
    assert_eq!(secs, expected);
}

#[test]
fn int32_epoch_without_format_hint() {
    // Without a format hint, integer values should be treated as epoch seconds
    // (the original behavior).
    let epoch_secs = 1709251200i32; // 2024-03-01 00:00:00 UTC
    let arr: ArrayRef = Arc::new(Int32Array::from(vec![epoch_secs]));
    let secs = get_timestamp_from_array(arr, None, None)
        .unwrap()
        .unwrap();
    assert_eq!(secs, epoch_secs as i64);
}

#[test]
fn int64_epoch_millis_without_format_hint() {
    // Int64 with unit=Millisecond and no format → divide by 1000
    let epoch_ms = 1_709_251_200_000_i64;
    let arr: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![epoch_ms]));
    let secs = get_timestamp_from_array(arr, None, Some(arrow_schema::TimeUnit::Millisecond))
        .unwrap()
        .unwrap();
    assert_eq!(secs, 1709251200);
}

#[test]
fn time64_microsecond_as_timestamp() {
    // Arrow Time64 is a time-of-day physical type. When Kumo config declares
    // the column as a timestamp, DiskGraph should coerce it to an epoch-second
    // offset instead of treating it as an unreadable timestamp and dropping the
    // row as null.
    let arr: ArrayRef = Arc::new(Time64MicrosecondArray::from(vec![1_000_000_i64]));
    let secs = get_timestamp_from_array(arr, None, None)
        .unwrap()
        .unwrap();
    assert_eq!(secs, 1);
}

#[test]
fn arrow_time_units_as_timestamp() {
    let cases: Vec<(ArrayRef, i64)> = vec![
        (Arc::new(Time32SecondArray::from(vec![42_i32])), 42),
        (Arc::new(Time32MillisecondArray::from(vec![1_500_i32])), 2),
        (
            Arc::new(Time64MicrosecondArray::from(vec![1_500_000_i64])),
            2,
        ),
        (
            Arc::new(Time64NanosecondArray::from(vec![1_500_000_000_i64])),
            2,
        ),
    ];
    for (arr, expected) in cases {
        let secs = get_timestamp_from_array(arr, None, None)
            .unwrap()
            .unwrap();
        assert_eq!(secs, expected);
    }
}

#[test]
fn int32_yyyymmdd_format_takes_priority_over_unit() {
    // When both format and unit are set, format should win for YYYYMMDD values.
    let arr: ArrayRef = Arc::new(Int32Array::from(vec![20240301]));
    let secs = get_timestamp_from_array(
        arr,
        Some("%Y%m%d"),
        Some(arrow_schema::TimeUnit::Second),
    )
    .unwrap()
    .unwrap();
    let expected = chrono::NaiveDate::from_ymd_opt(2024, 3, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc()
        .timestamp();
    assert_eq!(secs, expected);
}

// ---- detect_encoded_int_format tests ----

#[test]
fn test_detect_encoded_int_format_yyyymmdd() {
    assert_eq!(detect_encoded_int_format(20230408), Some("%Y%m%d"));
}

#[test]
fn test_detect_encoded_int_format_yyyymmddhhmmss() {
    assert_eq!(
        detect_encoded_int_format(20230408120530),
        Some("%Y%m%d%H%M%S")
    );
}

#[test]
fn test_detect_encoded_int_format_yyyy() {
    assert_eq!(detect_encoded_int_format(2023), Some("%Y"));
}

#[test]
fn test_detect_encoded_int_format_yyyymm() {
    assert_eq!(detect_encoded_int_format(202304), Some("%Y%m"));
}

#[test]
fn test_detect_encoded_int_format_not_date() {
    // 99999999 → invalid month (99) and day (99), should not match
    assert_eq!(detect_encoded_int_format(99999999), None);
    // epoch seconds value — 10 digits, no matching pattern
    assert_eq!(detect_encoded_int_format(1709251200), None);
    // 5-digit number — no matching pattern
    assert_eq!(detect_encoded_int_format(12345), None);
}

#[test]
fn test_detect_encoded_int_format_boundary() {
    // Earliest valid date: 1900-01-01
    assert_eq!(detect_encoded_int_format(19000101), Some("%Y%m%d"));
    // Latest valid date: 2100-12-31
    assert_eq!(detect_encoded_int_format(21001231), Some("%Y%m%d"));
    // Year out of range
    assert_eq!(detect_encoded_int_format(18991231), None);
    assert_eq!(detect_encoded_int_format(21010101), None);
    // Invalid month
    assert_eq!(detect_encoded_int_format(20231301), None);
    assert_eq!(detect_encoded_int_format(20230001), None);
    // Invalid day
    assert_eq!(detect_encoded_int_format(20230100), None);
    assert_eq!(detect_encoded_int_format(20230132), None);
    // Valid 14-digit boundary
    assert_eq!(
        detect_encoded_int_format(19000101000000),
        Some("%Y%m%d%H%M%S")
    );
    assert_eq!(
        detect_encoded_int_format(21001231235959),
        Some("%Y%m%d%H%M%S")
    );
    // Invalid time components in 14-digit
    assert_eq!(detect_encoded_int_format(20230408240000), None); // hour 24
    assert_eq!(detect_encoded_int_format(20230408006000), None); // minute 60
    assert_eq!(detect_encoded_int_format(20230408000060), None); // second 60
}

// ---- compact format parsing tests ----

#[test]
fn test_parse_yyyymmdd_fast() {
    let expected = chrono::NaiveDate::from_ymd_opt(2023, 4, 8)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc()
        .timestamp();
    assert_eq!(parse_compact_ymd_fast(b"20230408"), Some(expected));

    // 1970-01-01
    let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc()
        .timestamp();
    assert_eq!(parse_compact_ymd_fast(b"19700101"), Some(epoch));

    // Wrong length
    assert_eq!(parse_compact_ymd_fast(b"2023040"), None);
    assert_eq!(parse_compact_ymd_fast(b"202304081"), None);

    // Non-digits
    assert_eq!(parse_compact_ymd_fast(b"2023-4-8"), None);
}

#[test]
fn test_parse_yyyymmddhhmmss_fast() {
    let expected = chrono::NaiveDate::from_ymd_opt(2023, 4, 8)
        .unwrap()
        .and_hms_opt(12, 5, 30)
        .unwrap()
        .and_utc()
        .timestamp();
    assert_eq!(
        parse_compact_ymd_hms_fast(b"20230408120530"),
        Some(expected)
    );

    // Midnight
    let midnight = chrono::NaiveDate::from_ymd_opt(2024, 1, 15)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc()
        .timestamp();
    assert_eq!(
        parse_compact_ymd_hms_fast(b"20240115000000"),
        Some(midnight)
    );

    // Wrong length
    assert_eq!(parse_compact_ymd_hms_fast(b"2023040812053"), None);
    assert_eq!(parse_compact_ymd_hms_fast(b"202304081205300"), None);
}

// ---- non-ISO format detection tests ----

#[test]
fn test_detect_non_iso_formats() {
    // US date: month/day/year — chrono fallback since no ISO fast-path matches
    assert_eq!(
        detect_timestamp_format("01/15/2024"),
        Some("%m/%d/%Y")
    );
    // US date with time
    assert_eq!(
        detect_timestamp_format("01/15/2024 10:30:00"),
        Some("%m/%d/%Y %H:%M:%S")
    );
    // EU date (note: 15 > 12 so only d/m/Y can match)
    assert_eq!(
        detect_timestamp_format("15/01/2024"),
        Some("%d/%m/%Y")
    );
    // Slash-separated ISO-like: the fast parser for "%Y-%m-%d" is separator-agnostic,
    // so "2024/01/15" matches "%Y-%m-%d" first (higher priority). This is acceptable
    // since the parsed timestamp is identical.
    assert_eq!(
        detect_timestamp_format("2024/01/15"),
        Some("%Y-%m-%d")
    );
    // Slash-separated ISO with time: same — fast parser is separator-agnostic
    assert_eq!(
        detect_timestamp_format("2024/01/15 10:30:00"),
        Some("%Y-%m-%d %H:%M:%S")
    );
    // Verify the non-ISO formats still parse correctly via parse_timestamp_text
    assert!(parse_timestamp_text("01/15/2024", Some("%m/%d/%Y")).is_some());
    assert!(parse_timestamp_text("15/01/2024", Some("%d/%m/%Y")).is_some());
    assert!(parse_timestamp_text("15/01/2024 10:30:00", Some("%d/%m/%Y %H:%M:%S")).is_some());
}

#[test]
fn test_detect_non_iso_hour_minute_no_seconds() {
    // US date with HH:MM (no seconds), unpadded month/day/hour — the exact shape
    // of real data like customerltv's `InvoiceDate` (`12/1/2010 8:26`). Without
    // the `%m/%d/%Y %H:%M` candidate, detection finds nothing and the column
    // parses to all-null.
    assert_eq!(
        detect_timestamp_format("12/1/2010 8:26"),
        Some("%m/%d/%Y %H:%M")
    );
    // And it parses to the expected naive datetime.
    let ts = parse_timestamp_text("12/1/2010 8:26", Some("%m/%d/%Y %H:%M")).unwrap();
    let expected = chrono::NaiveDate::from_ymd_opt(2010, 12, 1)
        .unwrap()
        .and_hms_opt(8, 26, 0)
        .unwrap()
        .and_utc()
        .timestamp();
    assert_eq!(ts, expected);

    // A seconds-bearing value still resolves to the more-specific `%H:%M:%S`
    // format — the new `%H:%M` candidate must not shadow it (chrono's
    // `parse_from_str` is whole-string/exact, so `%H:%M` rejects a `:SS` value).
    assert_eq!(
        detect_timestamp_format("12/1/2010 8:26:30"),
        Some("%m/%d/%Y %H:%M:%S")
    );

    // EU (d/m/Y) with HH:MM resolves (month 15 > 12 rules out m/d/Y).
    assert_eq!(
        detect_timestamp_format("15/01/2024 10:30"),
        Some("%d/%m/%Y %H:%M")
    );
    // Slash-separated ISO with HH:MM: the fast parser for "%Y-%m-%d %H:%M" is
    // separator-agnostic, so it matches "%Y-%m-%d %H:%M" first (higher priority).
    // Acceptable — the parsed timestamp is identical — and the dedicated
    // "%Y/%m/%d %H:%M" candidate still parses it directly when hinted.
    assert_eq!(
        detect_timestamp_format("2024/01/15 10:30"),
        Some("%Y-%m-%d %H:%M")
    );
    assert!(parse_timestamp_text("2024/01/15 10:30", Some("%Y/%m/%d %H:%M")).is_some());
}

// ---- mixed timezone fallback tests ----

#[test]
fn test_parse_timestamp_text_no_tz_format_rejects_tz_value() {
    // Format has no tz, value has tz offset → should NOT silently parse
    // (detection should have picked a tz-aware format if the column has tz values)
    let result =
        parse_timestamp_text("2024-01-15 10:30:00+05:30", Some("%Y-%m-%d %H:%M:%S"));
    assert!(result.is_none());
}

#[test]
fn test_parse_timestamp_text_tz_format_no_tz_value() {
    // Format expects tz, but value has none → should still parse (as naive UTC)
    let result =
        parse_timestamp_text("2024-01-15 10:30:00", Some("%Y-%m-%d %H:%M:%S%:z"));
    assert!(result.is_some());
    let expected = chrono::NaiveDate::from_ymd_opt(2024, 1, 15)
        .unwrap()
        .and_hms_opt(10, 30, 0)
        .unwrap()
        .and_utc()
        .timestamp();
    assert_eq!(result.unwrap(), expected);
}
