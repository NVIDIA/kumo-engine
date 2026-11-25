// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use anyhow::Result;
use arrow_array::{BooleanArray, Float32Array, Int64Array, StringArray};

#[test]
fn test_decode_batch_simple() -> Result<()> {
    // 1. Setup Schema
    let cols = vec![
        RowPackColumn {
            name: "id".into(),
            logical_type: IngestedType::Int64,
            fixed_len: None,
            precision: None,
            scale: None,
        },
        RowPackColumn {
            name: "score".into(),
            logical_type: IngestedType::Float32,
            fixed_len: None,
            precision: None,
            scale: None,
        },
        RowPackColumn {
            name: "label".into(),
            logical_type: IngestedType::String,
            fixed_len: None,
            precision: None,
            scale: None,
        },
        RowPackColumn {
            name: "active".into(),
            logical_type: IngestedType::Bool,
            fixed_len: None,
            precision: None,
            scale: None,
        },
    ];

    let mut builder = RowPackContextBuilder::new(cols, RowPackConfig::default());

    // 2. Observe some data to train context
    let rows = vec![
        vec![
            ValueRef::Int64(1),
            ValueRef::Float32(0.5),
            ValueRef::String("A"),
            ValueRef::Bool(true),
        ],
        vec![
            ValueRef::Int64(2),
            ValueRef::Float32(1.5),
            ValueRef::String("B"),
            ValueRef::Bool(false),
        ],
        vec![
            ValueRef::Int64(3),
            ValueRef::Null,
            ValueRef::String("C"),
            ValueRef::Bool(true),
        ], // Null float
    ];

    for r in &rows {
        builder.observe_row(r)?;
    }

    let ctx = builder.finish()?;
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    // 3. Encode Rows
    let mut encoded_rows = Vec::new();
    for r in &rows {
        let mut out = Vec::new();
        encoder.encode_row_into(r, &mut out, &mut scratch)?;
        encoded_rows.push(out);
    }

    let encoded_slices: Vec<&[u8]> = encoded_rows.iter().map(|v| v.as_slice()).collect();

    // 4. Decode Batch
    let decoder = RowPackDecoder::new(ctx)?;

    // Decode all columns
    let batch = decoder.decode_batch_to_arrow(&encoded_slices, &[0, 1, 2, 3])?;

    assert_eq!(batch.num_rows(), 3);
    assert_eq!(batch.num_columns(), 4);

    // 5. Verify Data
    let ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(ids.value(0), 1);
    assert_eq!(ids.value(1), 2);
    assert_eq!(ids.value(2), 3);

    let scores = batch
        .column(1)
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    assert_eq!(scores.value(0), 0.5);
    assert_eq!(scores.value(1), 1.5);
    assert!(scores.is_null(2));

    let labels = batch
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(labels.value(0), "A");
    assert_eq!(labels.value(1), "B");
    assert_eq!(labels.value(2), "C");

    let active = batch
        .column(3)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();
    assert!(active.value(0));
    assert!(!active.value(1));
    assert!(active.value(2));

    Ok(())
}

#[test]
fn test_decode_batch_subset() -> Result<()> {
    // Setup same data...
    let cols = vec![
        RowPackColumn {
            name: "id".into(),
            logical_type: IngestedType::Int64,
            fixed_len: None,
            precision: None,
            scale: None,
        },
        RowPackColumn {
            name: "score".into(),
            logical_type: IngestedType::Float32,
            fixed_len: None,
            precision: None,
            scale: None,
        },
    ];
    let mut builder = RowPackContextBuilder::new(cols, RowPackConfig::default());
    let rows = vec![
        vec![ValueRef::Int64(10), ValueRef::Float32(10.0)],
        vec![ValueRef::Int64(20), ValueRef::Float32(20.0)],
    ];
    for r in &rows {
        builder.observe_row(r)?;
    }
    let ctx = builder.finish()?;
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);
    let mut encoded_rows = Vec::new();
    for r in &rows {
        let mut out = Vec::new();
        encoder.encode_row_into(r, &mut out, &mut scratch)?;
        encoded_rows.push(out);
    }
    let encoded_slices: Vec<&[u8]> = encoded_rows.iter().map(|v| v.as_slice()).collect();
    let decoder = RowPackDecoder::new(ctx)?;

    // Decode only "score" (column index 1)
    let batch = decoder.decode_batch_to_arrow(&encoded_slices, &[1])?;
    assert_eq!(batch.num_columns(), 1);
    let scores = batch
        .column(0)
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    assert_eq!(scores.value(0), 10.0);
    assert_eq!(scores.value(1), 20.0);
    assert_eq!(batch.schema().field(0).name(), "score");

    Ok(())
}

#[test]
fn wide_float_arrays_decode_to_arrow_across_storage_widths() -> Result<()> {
    use arrow_array::{Float64Array, ListArray};

    let mut values: Vec<f32> = (0..1152).map(|i| (i - 576) as f32 / 8.0).collect();
    values[..6].copy_from_slice(&[
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0xffc1_2000),
        f32::from_bits(0x7f91_2000),
        0.0,
    ]);
    let wide: Vec<f64> = values.iter().copied().map(f64::from).collect();
    for storage in [FloatStorage::F16, FloatStorage::F32, FloatStorage::F64] {
        for (logical, value) in [
            (IngestedType::Float32Array, ValueRef::Float32Array(&values)),
            (IngestedType::Float64Array, ValueRef::Float64Array(&wide)),
        ] {
            for encoding in [
                ColumnEncoding::FixedFloatArray {
                    storage,
                    len: values.len(),
                },
                ColumnEncoding::FloatArray { storage },
            ] {
                let ctx = RowPackContext {
                    version: ROWPACK_VERSION,
                    columns: vec![ColumnContext {
                        name: "embedding".into(),
                        logical_type: logical,
                        encoding,
                    }],
                    estimated_row_bytes: 0,
                };
                let mut scratch = RowPackScratch::new(&ctx);
                let encoder = RowPackEncoder::new(ctx.clone())?;
                let decoder = RowPackDecoder::new(ctx)?;
                let mut row = Vec::new();
                encoder.encode_row_into(std::slice::from_ref(&value), &mut row, &mut scratch)?;
                let stride = float_storage_bytes(storage);
                let start = row.len() - values.len() * stride;
                let mut finite = row.clone();
                finite[start..start + 6 * stride].fill(0);
                let batch = decoder.decode_batch_to_arrow(
                    &[&row[..], decoder.row_or_null(&[]), &finite[..]],
                    &[0],
                )?;
                let list = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .unwrap();
                assert!(list.is_null(1));
                for (i, row) in [(0, &row), (2, &finite)] {
                    let mut input = &row[start..];
                    let a = list.value(i);
                    let expected_len = values.len();
                    assert_eq!(a.len(), expected_len);
                    if logical == IngestedType::Float32Array {
                        let actual = a.as_any().downcast_ref::<Float32Array>().unwrap();
                        // Match the original Arrow decoder: narrow each decoded value inside
                        // the loop. Materializing an intermediate f64 array can quiet a
                        // signaling f32 NaN where an immediate round trip preserves its bits.
                        let mut expected = Vec::<f32>::with_capacity(expected_len);
                        for _ in 0..expected_len {
                            expected.push(decode_float(&mut input, storage)? as f32);
                        }
                        for (j, (v, e)) in actual.values().iter().zip(&expected).enumerate() {
                            assert_eq!(
                                v.to_bits(),
                                e.to_bits(),
                                "{logical:?} {storage:?} row={i} index={j}"
                            );
                        }
                    } else {
                        let actual = a.as_any().downcast_ref::<Float64Array>().unwrap();
                        for v in actual.values() {
                            assert_eq!(v.to_bits(), decode_float(&mut input, storage)?.to_bits());
                        }
                    }
                }
                assert!(decoder
                    .decode_batch_to_arrow(&[&row[..row.len() - 1]], &[0])
                    .is_err());
            }
        }
    }
    Ok(())
}
