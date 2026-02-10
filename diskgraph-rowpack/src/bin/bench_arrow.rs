use anyhow::Result;
use arrow_array::builder::*;
use arrow_array::Array;
use arrow_schema::{DataType, Field, Schema};
use diskgraph_rowpack::*;
use rand::Rng;
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<()> {
    let num_rows = 100_000;
    let num_cols = 50;

    println!("Generating {} rows with {} columns...", num_rows, num_cols);

    // 1. Generate Schema & Data
    let mut columns = Vec::new();
    for i in 0..num_cols {
        let logical_type = match i % 4 {
            0 => IngestedType::Int64,
            1 => IngestedType::Float32,
            2 => IngestedType::String,
            _ => IngestedType::Bool,
        };
        columns.push(RowPackColumn {
            name: format!("col_{}", i),
            logical_type,
            fixed_len: None,
            precision: None,
            scale: None,
        });
    }

    let mut builder = RowPackContextBuilder::new(columns.clone(), RowPackConfig::default());

    // Generate raw values to train context
    // We only train on a sample to be fast
    let sample_rows = 1000;
    let mut rng = rand::thread_rng();

    for _ in 0..sample_rows {
        let row = generate_random_row(num_cols, &mut rng);
        builder.observe_row(&row)?;
    }

    let ctx = builder.finish()?;
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    println!("Encoding data...");
    let mut encoded_rows = Vec::with_capacity(num_rows);
    let mut encoded_slices = Vec::with_capacity(num_rows);

    for _ in 0..num_rows {
        let row = generate_random_row(num_cols, &mut rng);
        let mut out = Vec::new();
        encoder.encode_row_into(&row, &mut out, &mut scratch)?;
        encoded_rows.push(out);
    }

    // Create slice references
    for r in &encoded_rows {
        encoded_slices.push(r.as_slice());
    }

    println!(
        "Data ready. Size: {:.2} MB",
        encoded_rows.iter().map(|v| v.len()).sum::<usize>() as f64 / 1024.0 / 1024.0
    );

    let decoder = RowPackDecoder::new(ctx.clone())?;
    let col_indices: Vec<usize> = (0..num_cols).collect();

    // Simulate Scatter-Gather: Select random subset
    let subset_size = 10_000;
    println!("Selecting random subset of {} rows...", subset_size);
    let mut indices: Vec<usize> = (0..num_rows).collect();
    // shuffle
    for i in (0..indices.len()).rev() {
        let j = rng.gen_range(0..=i);
        indices.swap(i, j);
    }
    indices.truncate(subset_size);

    let mut subset_slices = Vec::with_capacity(subset_size);
    for idx in &indices {
        subset_slices.push(encoded_rows[*idx].as_slice());
    }

    // 2. Benchmark Fully Batched (Rows + Cols)
    println!("\n--- Benchmarking Fully Batched (Rows + Cols) ---");
    let start = Instant::now();
    let batch = decoder.decode_batch_to_arrow(&subset_slices, &col_indices)?;
    let duration_batched = start.elapsed();
    println!("Time: {:?}", duration_batched);
    println!(
        "Throughput: {:.2} rows/sec",
        subset_size as f64 / duration_batched.as_secs_f64()
    );

    assert_eq!(batch.num_rows(), subset_size);

    // 3. Benchmark Row-by-Row (Cols Batched)
    println!("\n--- Benchmarking Row-by-Row (Cols Batched) ---");
    let mut row_by_row_builders: Vec<Box<dyn ArrayBuilder>> = Vec::new();
    for col in &columns {
        let b: Box<dyn ArrayBuilder> = match col.logical_type {
            IngestedType::Int64 => Box::new(Int64Builder::with_capacity(subset_size)),
            IngestedType::Float32 => Box::new(Float32Builder::with_capacity(subset_size)),
            IngestedType::String => {
                Box::new(StringBuilder::with_capacity(subset_size, subset_size * 10))
            }
            IngestedType::Bool => Box::new(BooleanBuilder::with_capacity(subset_size)),
            _ => panic!("unsupported in bench"),
        };
        row_by_row_builders.push(b);
    }
    let requested_mask = vec![true; num_cols];
    let mut row_by_row_scratch = RowPackDecodeScratch::default();

    let start = Instant::now();
    for slice in &subset_slices {
        decoder.decode_row_to_builders(
            slice,
            &mut row_by_row_builders,
            &requested_mask,
            &mut row_by_row_scratch,
        )?;
    }
    let duration_row_batched = start.elapsed();
    println!("Time: {:?}", duration_row_batched);
    println!(
        "Throughput: {:.2} rows/sec",
        subset_size as f64 / duration_row_batched.as_secs_f64()
    );

    // Finish to ensure work is done
    let _ = row_by_row_builders
        .into_iter()
        .map(|mut b| b.finish())
        .collect::<Vec<_>>();

    // 4. Benchmark Single-Col (Baseline)
    println!("\n--- Benchmarking Single-Col Loop (No Batching) ---");
    let start = Instant::now();
    let _ = decode_single_col_loop(&decoder, &subset_slices, &columns)?;
    let duration_single = start.elapsed();
    println!("Time: {:?}", duration_single);
    println!(
        "Throughput: {:.2} rows/sec",
        subset_size as f64 / duration_single.as_secs_f64()
    );

    println!("\n--- Speedups ---");
    println!(
        "Fully Batched vs Single-Col: {:.2}x",
        duration_single.as_secs_f64() / duration_batched.as_secs_f64()
    );
    println!(
        "Row-by-Row vs Single-Col:    {:.2}x",
        duration_single.as_secs_f64() / duration_row_batched.as_secs_f64()
    );
    println!(
        "Fully Batched vs Row-by-Row: {:.2}x",
        duration_row_batched.as_secs_f64() / duration_batched.as_secs_f64()
    );

    Ok(())
}

fn generate_random_row(num_cols: usize, rng: &mut impl Rng) -> Vec<ValueRef<'static>> {
    let mut row = Vec::with_capacity(num_cols);
    for i in 0..num_cols {
        let val = match i % 4 {
            0 => ValueRef::Int64(rng.gen()),
            1 => ValueRef::Float32(rng.gen()),
            2 => ValueRef::String("sample_string"), // Keep it simple to avoid lifetime hell in bench
            _ => ValueRef::Bool(rng.gen()),
        };
        row.push(val);
    }
    row
}

fn decode_single_col_loop(
    decoder: &RowPackDecoder,
    rows: &[&[u8]],
    columns: &[RowPackColumn],
) -> Result<arrow_array::RecordBatch> {
    // Setup builders
    let mut builders: Vec<Box<dyn ArrayBuilder>> = Vec::new();
    let mut fields = Vec::new();

    // We assume we know the types (Int64, Float32, String, Bool) for simplicity of the bench harness
    // In reality, we'd use a helper to create them.
    for col in columns {
        let (dtype, b): (DataType, Box<dyn ArrayBuilder>) = match col.logical_type {
            IngestedType::Int64 => (
                DataType::Int64,
                Box::new(Int64Builder::with_capacity(rows.len())),
            ),
            IngestedType::Float32 => (
                DataType::Float32,
                Box::new(Float32Builder::with_capacity(rows.len())),
            ),
            IngestedType::String => (
                DataType::Utf8,
                Box::new(StringBuilder::with_capacity(rows.len(), rows.len() * 10)),
            ),
            IngestedType::Bool => (
                DataType::Boolean,
                Box::new(BooleanBuilder::with_capacity(rows.len())),
            ),
            _ => panic!("unsupported in bench"),
        };
        builders.push(b);
        fields.push(Field::new(&col.name, dtype, true));
    }

    let mut scratch = RowPackDecodeScratch::default();

    for row in rows {
        for (i, builder) in builders.iter_mut().enumerate() {
            let val = decoder.decode_column_value(row, i, &mut scratch)?;
            // Append
            match val {
                ValueRef::Int64(v) => builder
                    .as_any_mut()
                    .downcast_mut::<Int64Builder>()
                    .unwrap()
                    .append_value(v),
                ValueRef::Float32(v) => builder
                    .as_any_mut()
                    .downcast_mut::<Float32Builder>()
                    .unwrap()
                    .append_value(v),
                ValueRef::String(v) => builder
                    .as_any_mut()
                    .downcast_mut::<StringBuilder>()
                    .unwrap()
                    .append_value(v),
                ValueRef::Bool(v) => builder
                    .as_any_mut()
                    .downcast_mut::<BooleanBuilder>()
                    .unwrap()
                    .append_value(v),
                ValueRef::Null => {
                    // simplified null handling
                    if let Some(b) = builder.as_any_mut().downcast_mut::<Int64Builder>() {
                        b.append_null();
                    } else if let Some(b) = builder.as_any_mut().downcast_mut::<Float32Builder>() {
                        b.append_null();
                    } else if let Some(b) = builder.as_any_mut().downcast_mut::<StringBuilder>() {
                        b.append_null();
                    } else if let Some(b) = builder.as_any_mut().downcast_mut::<BooleanBuilder>() {
                        b.append_null();
                    }
                }
                _ => panic!("unexpected type"),
            }
        }
    }

    let arrays: Vec<Arc<dyn Array>> = builders.into_iter().map(|mut b| b.finish()).collect();
    let schema = Schema::new(fields);
    Ok(arrow_array::RecordBatch::try_new(Arc::new(schema), arrays)?)
}
