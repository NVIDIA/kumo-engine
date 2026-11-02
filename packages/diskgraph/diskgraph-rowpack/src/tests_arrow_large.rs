// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use arrow_array::{LargeBinaryArray, LargeListArray, LargeStringArray};
use arrow_schema::DataType;

fn encode(ctx: &RowPackContext, rows: &[Vec<ValueRef<'_>>]) -> Result<Vec<Vec<u8>>> {
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(ctx);
    rows.iter()
        .map(|row| {
            let mut bytes = Vec::new();
            encoder.encode_row_into(row, &mut bytes, &mut scratch)?;
            Ok(bytes)
        })
        .collect()
}

#[test]
fn large_offsets_preserve_values_nulls_and_projection() -> Result<()> {
    let types = [
        IngestedType::String,
        IngestedType::Bytes,
        IngestedType::Uuid,
        IngestedType::Uuid256,
        IngestedType::Ip,
        IngestedType::Decimal128,
        IngestedType::Decimal256,
        IngestedType::StringArray,
        IngestedType::Int8Array,
        IngestedType::Int16Array,
        IngestedType::Int32Array,
        IngestedType::Int64Array,
        IngestedType::Float16Array,
        IngestedType::Float32Array,
        IngestedType::Float64Array,
    ];
    let columns = types
        .iter()
        .enumerate()
        .map(|(i, &logical_type)| RowPackColumn {
            name: format!("c{i}"),
            logical_type,
            fixed_len: None,
            precision: match logical_type {
                IngestedType::Decimal128 => Some(38),
                IngestedType::Decimal256 => Some(76),
                _ => None,
            },
            scale: matches!(
                logical_type,
                IngestedType::Decimal128 | IngestedType::Decimal256
            )
            .then_some(2),
        })
        .collect();
    let values = vec![
        ValueRef::String("café💳"),
        ValueRef::Bytes(b"\x00\xffpayload"),
        ValueRef::Uuid([7; 16]),
        ValueRef::Uuid256([9; 32]),
        ValueRef::Ip([0; 16]),
        ValueRef::Decimal128(-12345),
        ValueRef::Decimal256([0; 32]),
        ValueRef::StringArray(&["a", "café💳", ""]),
        ValueRef::Int8Array(&[-8, 7]),
        ValueRef::Int16Array(&[-300, 300]),
        ValueRef::Int32Array(&[-100000, 100000]),
        ValueRef::Int64Array(&[i64::MIN, i64::MAX]),
        ValueRef::Float32Array(&[0.5, -1.0]),
        ValueRef::Float32Array(&[1.25, -2.5]),
        ValueRef::Float64Array(&[1.5, -2.0]),
    ];
    let rows = vec![values.clone(), vec![ValueRef::Null; types.len()], values];
    let mut builder = RowPackContextBuilder::new(columns, RowPackConfig::default());
    for row in &rows {
        builder.observe_row(row)?;
    }
    let ctx = builder.finish()?;
    let encoded = encode(&ctx, &rows)?;
    let decoder = RowPackDecoder::new(ctx)?;
    // A reordered projection with duplicates must not change values or null positions.
    let projection: Vec<_> = (0..types.len()).rev().chain([0, 1]).collect();
    let small = decoder.decode_batch_to_arrow(&encoded, &projection)?;
    let large = decoder.decode_batch_to_arrow_large(&encoded, &projection)?;
    assert_eq!(large.num_rows(), 3);
    assert_eq!(large.num_columns(), projection.len());
    for (small, large) in small.columns().iter().zip(large.columns()) {
        assert!(large.is_null(1));
        let restored = arrow_cast::cast(large, small.data_type())?;
        assert_eq!(restored.to_data(), small.to_data());
        match small.data_type() {
            DataType::Utf8 => assert_eq!(large.data_type(), &DataType::LargeUtf8),
            DataType::Binary => assert_eq!(large.data_type(), &DataType::LargeBinary),
            DataType::List(_) => assert!(matches!(large.data_type(), DataType::LargeList(_))),
            _ => unreachable!(),
        }
    }
    let strings = large
        .column(projection.len() - 2)
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .unwrap();
    assert_eq!(strings.value(0), "café💳");
    assert_eq!(strings.value(2), "café💳");
    let bytes = large
        .column(projection.len() - 1)
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .unwrap();
    assert_eq!(bytes.value(0), b"\x00\xffpayload");
    let list = large
        .column(types.len() - 1 - 7)
        .as_any()
        .downcast_ref::<LargeListArray>()
        .unwrap();
    assert_eq!(list.values().data_type(), &DataType::LargeUtf8);
    assert_eq!(list.value_offsets(), &[0, 3, 3, 6]);
    Ok(())
}

#[test]
fn large_offsets_keep_dictionary_fallback_and_empty_values() -> Result<()> {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        estimated_row_bytes: 0,
        columns: vec![
            ColumnContext {
                name: "text".into(),
                logical_type: IngestedType::String,
                encoding: ColumnEncoding::String {
                    encoding: StringEncoding::Dict {
                        values: vec![b"known".to_vec()],
                    },
                },
            },
            ColumnContext {
                name: "tokens".into(),
                logical_type: IngestedType::StringArray,
                encoding: ColumnEncoding::StringArray {
                    encoding: StringEncoding::Dict {
                        values: vec![b"known".to_vec()],
                    },
                },
            },
        ],
    };
    let rows = vec![
        vec![
            ValueRef::String("known"),
            ValueRef::StringArray(&["known", "unseen💳"]),
        ],
        vec![ValueRef::String(""), ValueRef::StringArray(&[])],
        vec![ValueRef::Null, ValueRef::Null],
        vec![ValueRef::String("unseen💳"), ValueRef::StringArray(&[""])],
    ];
    let encoded = encode(&ctx, &rows)?;
    let decoder = RowPackDecoder::new(ctx)?;
    let actual = decoder.decode_batch_to_arrow_large(&encoded, &[0, 1])?;
    let text = actual
        .column(0)
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .unwrap();
    assert_eq!(
        text.iter().collect::<Vec<_>>(),
        vec![Some("known"), Some(""), None, Some("unseen💳")]
    );
    let tokens = actual
        .column(1)
        .as_any()
        .downcast_ref::<LargeListArray>()
        .unwrap();
    assert_eq!(tokens.value_offsets(), &[0, 2, 2, 2, 3]);
    assert!(!tokens.is_null(1));
    assert!(tokens.is_null(2));
    let values = tokens
        .values()
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .unwrap();
    assert_eq!(
        values.iter().collect::<Vec<_>>(),
        vec![Some("known"), Some("unseen💳"), Some("")]
    );
    let empty = decoder.decode_batch_to_arrow_large::<&[u8]>(&[], &[0, 1])?;
    assert_eq!(empty.num_rows(), 0);
    assert_eq!(empty.schema(), actual.schema());
    assert!(decoder.decode_batch_to_arrow_large(&encoded, &[2]).is_err());
    assert!(decoder
        .decode_batch_to_arrow_large(&[&[][..]], &[0, 1])
        .is_err());
    Ok(())
}

#[test]
#[ignore = "allocates more than 2 GiB; run explicitly on a host with at least 6 GiB free"]
fn large_offsets_cross_i32_byte_boundary() -> Result<()> {
    let ctx = RowPackContext {
        version: ROWPACK_VERSION,
        estimated_row_bytes: 0,
        columns: vec![ColumnContext {
            name: "text".into(),
            logical_type: IngestedType::String,
            encoding: ColumnEncoding::String {
                encoding: StringEncoding::Raw,
            },
        }],
    };
    let value = "x".repeat(1024 * 1024);
    let encoded = encode(&ctx, &[vec![ValueRef::String(&value)]])?;
    // Reuse the encoded row bytes so the input stays small. The cumulative
    // output byte count is what overflows a StringBuilder's i32 offsets.
    let rows = vec![encoded[0].as_slice(); 2049];
    let decoder = RowPackDecoder::new(ctx)?;
    let old_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        decoder.decode_batch_to_arrow(&rows, &[0])
    }));
    assert!(
        old_result.is_err(),
        "the legacy i32 builder must hit the reproduced limit"
    );
    let batch = decoder.decode_batch_to_arrow_large(&rows, &[0])?;
    let strings = batch
        .column(0)
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .unwrap();
    assert_eq!(strings.len(), rows.len());
    assert_eq!(
        strings.value_offsets().last().copied(),
        Some(2049 * 1024 * 1024)
    );
    assert!(strings.value_offsets().last().copied().unwrap() > i32::MAX as i64);
    for index in 0..strings.len() {
        assert_eq!(strings.value(index), value);
    }
    Ok(())
}

#[test]
fn large_offsets_preserve_timestamp_clocks() -> Result<()> {
    let columns = vec![RowPackColumn {
        name: "time".into(),
        logical_type: IngestedType::Timestamp,
        fixed_len: None,
        precision: None,
        scale: None,
    }];
    let rows = vec![vec![ValueRef::Timestamp(123456789)], vec![ValueRef::Null]];
    let mut builder = RowPackContextBuilder::new(columns, RowPackConfig::default());
    for row in &rows {
        builder.observe_row(row)?;
    }
    let ctx = builder.finish()?;
    let encoded = encode(&ctx, &rows)?;
    for clock in [
        TimeUnit::Seconds,
        TimeUnit::Milliseconds,
        TimeUnit::Microseconds,
        TimeUnit::Nanoseconds,
    ] {
        let decoder = RowPackDecoder::new(ctx.clone())?.with_clock(clock);
        let small = decoder.decode_batch_to_arrow(&encoded, &[0])?;
        let large = decoder.decode_batch_to_arrow_large(&encoded, &[0])?;
        assert_eq!(small, large);
    }
    Ok(())
}
