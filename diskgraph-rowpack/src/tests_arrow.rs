use crate::*;
use arrow_array::{Int64Array, Float32Array, StringArray, BooleanArray};
use anyhow::Result;

#[test]
fn test_decode_batch_simple() -> Result<()> {
    // 1. Setup Schema
    let cols = vec![
        RowPackColumn {
            name: "id".into(),
            logical_type: IngestedType::Int64,
            fixed_len: None, precision: None, scale: None
        },
        RowPackColumn {
            name: "score".into(),
            logical_type: IngestedType::Float32,
            fixed_len: None, precision: None, scale: None
        },
        RowPackColumn {
            name: "label".into(),
            logical_type: IngestedType::String,
            fixed_len: None, precision: None, scale: None
        },
        RowPackColumn {
            name: "active".into(),
            logical_type: IngestedType::Bool,
            fixed_len: None, precision: None, scale: None
        },
    ];

    let mut builder = RowPackContextBuilder::new(cols, RowPackConfig::default());

    // 2. Observe some data to train context
    let rows = vec![
        vec![ValueRef::Int64(1), ValueRef::Float32(0.5), ValueRef::String("A"), ValueRef::Bool(true)],
        vec![ValueRef::Int64(2), ValueRef::Float32(1.5), ValueRef::String("B"), ValueRef::Bool(false)],
        vec![ValueRef::Int64(3), ValueRef::Null, ValueRef::String("C"), ValueRef::Bool(true)], // Null float
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
    let ids = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(ids.value(0), 1);
    assert_eq!(ids.value(1), 2);
    assert_eq!(ids.value(2), 3);

    let scores = batch.column(1).as_any().downcast_ref::<Float32Array>().unwrap();
    assert_eq!(scores.value(0), 0.5);
    assert_eq!(scores.value(1), 1.5);
    assert!(scores.is_null(2));

    let labels = batch.column(2).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(labels.value(0), "A");
    assert_eq!(labels.value(1), "B");
    assert_eq!(labels.value(2), "C");

    let active = batch.column(3).as_any().downcast_ref::<BooleanArray>().unwrap();
    assert_eq!(active.value(0), true);
    assert_eq!(active.value(1), false);
    assert_eq!(active.value(2), true);

    Ok(())
}

#[test]
fn test_decode_batch_subset() -> Result<()> {
    // Setup same data...
     let cols = vec![
        RowPackColumn {
            name: "id".into(),
            logical_type: IngestedType::Int64,
            fixed_len: None, precision: None, scale: None
        },
        RowPackColumn {
            name: "score".into(),
            logical_type: IngestedType::Float32,
            fixed_len: None, precision: None, scale: None
        },
    ];
    let mut builder = RowPackContextBuilder::new(cols, RowPackConfig::default());
    let rows = vec![
        vec![ValueRef::Int64(10), ValueRef::Float32(10.0)],
        vec![ValueRef::Int64(20), ValueRef::Float32(20.0)],
    ];
    for r in &rows { builder.observe_row(r)?; }
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
    let scores = batch.column(0).as_any().downcast_ref::<Float32Array>().unwrap();
    assert_eq!(scores.value(0), 10.0);
    assert_eq!(scores.value(1), 20.0);
    assert_eq!(batch.schema().field(0).name(), "score");

    Ok(())
}
