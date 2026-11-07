// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use diskgraph_rowpack::*;
use std::{hint::black_box, time::Instant};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let kind = &args[1];
    let n: usize = args[2].parse()?;
    let repeats: usize = args[3].parse()?;
    let text = "abcdefgh".repeat(if kind == "long" { 128 } else { 1 });
    let tokens = vec!["a", "bb", "ccc", "dddd"];
    let (dtype, value) = match kind.as_str() {
        "numeric" => (IngestedType::Int64, ValueRef::Int64(42)),
        "list" => (IngestedType::StringArray, ValueRef::StringArray(&tokens)),
        "binary" => (IngestedType::Bytes, ValueRef::Bytes(text.as_bytes())),
        _ => (IngestedType::String, ValueRef::String(&text)),
    };
    let columns = vec![RowPackColumn {
        name: "value".into(),
        logical_type: dtype,
        fixed_len: None,
        precision: None,
        scale: None,
    }];
    let mut cb = RowPackContextBuilder::new(columns, RowPackConfig::default());
    cb.observe_row(&[value.clone()])?;
    cb.observe_row(&[ValueRef::Null])?;
    let ctx = cb.finish()?;
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);
    let mut encoded = Vec::new();
    encoder.encode_row_into(&[value], &mut encoded, &mut scratch)?;
    let mut null = Vec::new();
    encoder.encode_row_into(&[ValueRef::Null], &mut null, &mut scratch)?;
    let rows: Vec<&[u8]> = (0..n)
        .map(|i| {
            if i % 17 == 0 {
                null.as_slice()
            } else {
                encoded.as_slice()
            }
        })
        .collect();
    let decoder = RowPackDecoder::new(ctx)?;
    for _ in 0..3 {
        black_box(decoder.decode_batch_to_arrow_large(&rows, &[0])?);
    }
    let start = Instant::now();
    let mut bytes = 0;
    for _ in 0..repeats {
        let batch = decoder.decode_batch_to_arrow_large(black_box(&rows), &[0])?;
        assert_eq!(batch.num_rows(), n);
        bytes = batch.column(0).get_buffer_memory_size();
        black_box(batch);
    }
    println!(
        "{{\"case\":\"{}\",\"rows\":{},\"repeats\":{},\"seconds\":{},\"output_buffer_bytes\":{}}}",
        kind,
        n,
        repeats,
        start.elapsed().as_secs_f64(),
        bytes
    );
    Ok(())
}
