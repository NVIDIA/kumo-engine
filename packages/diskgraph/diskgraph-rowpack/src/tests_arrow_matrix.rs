// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use anyhow::Result;
use arrow_array::*;
use std::sync::Arc;

// Helper to encode a single value and decode it back to Arrow
fn roundtrip_value<F>(
    logical_type: IngestedType,
    value: ValueRef,
    _col_idx: usize,
    check: F,
    precision: Option<u32>,
    scale: Option<i32>,
) -> Result<()>
where
    F: FnOnce(&Arc<dyn Array>),
{
    // 1. Setup Context
    let col = RowPackColumn {
        name: "col".into(),
        logical_type,
        fixed_len: None,
        precision,
        scale,
    };
    let mut builder = RowPackContextBuilder::new(vec![col], RowPackConfig::default());

    // Train with the value (twice to avoid single-value optimizations if any)
    builder.observe_row(&[value.clone()])?;
    builder.observe_row(&[value.clone()])?;

    let ctx = builder.finish()?;
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    // 2. Encode
    let mut encoded_row = Vec::new();
    encoder.encode_row_into(&[value], &mut encoded_row, &mut scratch)?;

    // 3. Decode
    let decoder = RowPackDecoder::new(ctx)?;
    let batch = decoder.decode_batch_to_arrow(&[&encoded_row], &[0])?;

    // 4. Check
    assert_eq!(batch.num_rows(), 1);
    check(batch.column(0));

    Ok(())
}

#[test]
fn test_matrix_bool() -> Result<()> {
    roundtrip_value(
        IngestedType::Bool,
        ValueRef::Bool(true),
        0,
        |arr| {
            let a = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
            assert!(a.value(0));
        },
        None,
        None,
    )?;

    roundtrip_value(
        IngestedType::Bool,
        ValueRef::Bool(false),
        0,
        |arr| {
            let a = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
            assert!(!a.value(0));
        },
        None,
        None,
    )?;

    // Null
    roundtrip_value(
        IngestedType::Bool,
        ValueRef::Null,
        0,
        |arr| {
            assert!(arr.is_null(0));
        },
        None,
        None,
    )?;

    Ok(())
}

#[test]
fn test_matrix_int() -> Result<()> {
    // Int64
    roundtrip_value(
        IngestedType::Int64,
        ValueRef::Int64(i64::MIN),
        0,
        |arr| {
            let a = arr.as_any().downcast_ref::<Int64Array>().unwrap();
            assert_eq!(a.value(0), i64::MIN);
        },
        None,
        None,
    )?;

    // Int32
    roundtrip_value(
        IngestedType::Int32,
        ValueRef::Int32(i32::MAX),
        0,
        |arr| {
            let a = arr.as_any().downcast_ref::<Int32Array>().unwrap();
            assert_eq!(a.value(0), i32::MAX);
        },
        None,
        None,
    )?;

    // Int16
    roundtrip_value(
        IngestedType::Int16,
        ValueRef::Int32(30000),
        0,
        |arr| {
            let a = arr.as_any().downcast_ref::<Int16Array>().unwrap();
            assert_eq!(a.value(0), 30000);
        },
        None,
        None,
    )?;

    // Int8
    roundtrip_value(
        IngestedType::Int8,
        ValueRef::Int32(-128),
        0,
        |arr| {
            let a = arr.as_any().downcast_ref::<Int8Array>().unwrap();
            assert_eq!(a.value(0), -128);
        },
        None,
        None,
    )?;

    Ok(())
}

#[test]
fn test_matrix_uint() -> Result<()> {
    // UInt64
    roundtrip_value(
        IngestedType::UInt64,
        ValueRef::UInt64(u64::MAX),
        0,
        |arr| {
            let a = arr.as_any().downcast_ref::<UInt64Array>().unwrap();
            assert_eq!(a.value(0), u64::MAX);
        },
        None,
        None,
    )?;

    // UInt8
    roundtrip_value(
        IngestedType::UInt8,
        ValueRef::UInt32(255),
        0,
        |arr| {
            let a = arr.as_any().downcast_ref::<UInt8Array>().unwrap();
            assert_eq!(a.value(0), 255);
        },
        None,
        None,
    )?;

    Ok(())
}

#[test]
fn test_matrix_float() -> Result<()> {
    // Float32
    roundtrip_value(
        IngestedType::Float32,
        ValueRef::Float32(1.5),
        0,
        |arr| {
            let a = arr.as_any().downcast_ref::<Float32Array>().unwrap();
            assert_eq!(a.value(0), 1.5);
        },
        None,
        None,
    )?;

    // Float64
    roundtrip_value(
        IngestedType::Float64,
        ValueRef::Float64(std::f64::consts::PI),
        0,
        |arr| {
            let a = arr.as_any().downcast_ref::<Float64Array>().unwrap();
            assert_eq!(a.value(0), std::f64::consts::PI);
        },
        None,
        None,
    )?;

    Ok(())
}

#[test]
fn test_matrix_string_bytes() -> Result<()> {
    roundtrip_value(
        IngestedType::String,
        ValueRef::String("hello"),
        0,
        |arr| {
            let a = arr.as_any().downcast_ref::<StringArray>().unwrap();
            assert_eq!(a.value(0), "hello");
        },
        None,
        None,
    )?;

    roundtrip_value(
        IngestedType::Bytes,
        ValueRef::Bytes(b"\xDE\xAD\xBE\xEF"),
        0,
        |arr| {
            let a = arr.as_any().downcast_ref::<BinaryArray>().unwrap();
            assert_eq!(a.value(0), b"\xDE\xAD\xBE\xEF");
        },
        None,
        None,
    )?;

    Ok(())
}

#[test]
fn test_matrix_uuid() -> Result<()> {
    let u_bytes = [0x55u8; 16]; // 5555...
    let u_str = "55555555-5555-5555-5555-555555555555";

    roundtrip_value(
        IngestedType::Uuid,
        ValueRef::Uuid(u_bytes),
        0,
        |arr| {
            let a = arr.as_any().downcast_ref::<StringArray>().unwrap();
            assert_eq!(a.value(0), u_str);
        },
        None,
        None,
    )?;

    Ok(())
}

#[test]
fn test_matrix_ip() -> Result<()> {
    let _ipv4_bytes = [192, 168, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]; // Padded?
                                                                            // RowPack::Ip expects 16 bytes.
                                                                            // If it's IPv4, usually stored as IPv4-mapped IPv6 or just bytes.
                                                                            // Use 2001:db8::1 to ensure it stays IPv6 string
    let ipv6_bytes = [0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];

    roundtrip_value(
        IngestedType::Ip,
        ValueRef::Ip(ipv6_bytes),
        0,
        |arr| {
            let a = arr.as_any().downcast_ref::<StringArray>().unwrap();
            assert_eq!(a.value(0), "2001:db8::1");
        },
        None,
        None,
    )?;

    Ok(())
}

#[test]
fn test_matrix_timestamp() -> Result<()> {
    // 2023-01-01 00:00:00 UTC = 1672531200 secs
    let ts = 1672531200i64;

    roundtrip_value(
        IngestedType::Timestamp,
        ValueRef::Timestamp(ts),
        0,
        |arr| {
            let a = arr
                .as_any()
                .downcast_ref::<TimestampSecondArray>()
                .unwrap();
            assert_eq!(a.value(0), ts);
        },
        None,
        None,
    )?;

    // Test large value (no overflow concern — seconds stored as i64 directly)
    let max_secs = i64::MAX;

    roundtrip_value(
        IngestedType::Timestamp,
        ValueRef::Timestamp(max_secs),
        0,
        |arr| {
            let a = arr
                .as_any()
                .downcast_ref::<TimestampSecondArray>()
                .unwrap();
            assert_eq!(a.value(0), i64::MAX);
        },
        None,
        None,
    )?;

    Ok(())
}

#[test]
fn test_matrix_decimal() -> Result<()> {
    // Decimal128: 123.456 (Scale 3, Val 123456)
    roundtrip_value(
        IngestedType::Decimal128,
        ValueRef::Decimal128(123456),
        0,
        |arr| {
            let a = arr.as_any().downcast_ref::<StringArray>().unwrap();
            assert_eq!(a.value(0), "123.456");
        },
        Some(10),
        Some(3),
    )?;

    // Negative
    roundtrip_value(
        IngestedType::Decimal128,
        ValueRef::Decimal128(-123456),
        0,
        |arr| {
            let a = arr.as_any().downcast_ref::<StringArray>().unwrap();
            assert_eq!(a.value(0), "-123.456");
        },
        Some(10),
        Some(3),
    )?;

    Ok(())
}

#[test]
fn test_matrix_date() -> Result<()> {
    // Day 0 = 1970-01-01
    roundtrip_value(
        IngestedType::Date,
        ValueRef::Date(0),
        0,
        |arr| {
            let a = arr.as_any().downcast_ref::<Date32Array>().unwrap();
            assert_eq!(a.value(0), 0);
        },
        None,
        None,
    )?;

    roundtrip_value(
        IngestedType::Date,
        ValueRef::Date(19358),
        0,
        |arr| {
            // 2023-01-01
            let a = arr.as_any().downcast_ref::<Date32Array>().unwrap();
            assert_eq!(a.value(0), 19358);
        },
        None,
        None,
    )?;

    Ok(())
}

#[test]
fn test_matrix_list() -> Result<()> {
    let floats = vec![1.0f32, 2.0, 3.0];
    roundtrip_value(
        IngestedType::Float32Array,
        ValueRef::Float32Array(&floats),
        0,
        |arr| {
            let list_arr = arr.as_any().downcast_ref::<ListArray>().unwrap();
            let values = list_arr
                .values()
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap();
            assert_eq!(values.len(), 3);
            assert_eq!(values.value(0), 1.0);
            assert_eq!(values.value(2), 3.0);
        },
        None,
        None,
    )?;

    Ok(())
}
