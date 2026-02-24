//! Parquet shard writer for TT-gen output.
//!
//! Each thread gets its own `ParquetShardWriter` that buffers rows and
//! writes batched Arrow RecordBatches to a Parquet file.
//!
//! Three output modes are supported:
//! - **Node**: scalar f64 target (one row per entity/timestamp)
//! - **LinkExploded**: one row per distinct target value (string target + i64 frequency)
//! - **LinkArray**: one row per entity/timestamp with list columns

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow_array::builder::{Int64Builder, ListBuilder, StringBuilder};
use arrow_array::{
    BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray,
    TimestampNanosecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use diskgraph_core::tt_builder::Value;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

/// Batch size: flush to Parquet every N rows.
/// 64K rows × 3-4 columns × 8 bytes = ~1.5-2MB per batch.
const BATCH_SIZE: usize = 65536;

/// Which output format the writer produces.
#[derive(Debug, Clone)]
pub enum OutputMode {
    /// Node prediction: scalar target (f64 or bool depending on target_dtype).
    Node { target_dtype: DataType },
    /// Link + explode: one row per distinct target + i64 frequency.
    LinkExploded { include_frequency: bool, target_dtype: DataType },
    /// Link + no explode: one row per entity/ts with list columns.
    LinkArray { include_frequency: bool, target_dtype: DataType },
}

/// Per-thread Parquet writer that buffers rows and writes Arrow batches.
pub struct ParquetShardWriter {
    writer: ArrowWriter<BufWriter<File>>,
    schema: Arc<Schema>,
    mode: OutputMode,
    has_include_bit: bool,
    entity_dtype: DataType,

    // Common buffers
    entity_buf: Vec<Value>,
    timestamp_buf: Vec<i64>,
    include_bit_buf: Vec<bool>,

    // Node mode
    target_f64_buf: Vec<f64>,
    target_bool_buf: Vec<bool>,

    // LinkExploded mode
    target_value_buf: Vec<Value>,
    frequency_buf: Vec<i64>,

    // LinkArray mode
    target_value_list_buf: Vec<Vec<Value>>,
    frequency_list_buf: Vec<Vec<i64>>,

    rows_written: u64,
}

impl ParquetShardWriter {
    /// Create a new shard writer at the given path.
    ///
    /// `entity_dtype` is the Arrow data type for the ENTITY column, matching
    /// the ingested type of the entity table's PK column.
    ///
    /// When `include_entity_bit` is true, an extra `ENTITY_INCLUDE_BIT` boolean
    /// column is added to the schema. This is used when `lag_timesteps > 0` to
    /// mark rows where the entity filter failed (kept for autoregressive features).
    pub fn new(
        path: &Path,
        mode: OutputMode,
        include_entity_bit: bool,
        entity_dtype: DataType,
    ) -> Result<Self> {
        let mut fields = vec![
            Field::new("ENTITY", entity_dtype.clone(), false),
            Field::new("TIMESTAMP", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        ];

        match &mode {
            OutputMode::Node { target_dtype } => {
                fields.push(Field::new("TARGET", target_dtype.clone(), true));
            }
            OutputMode::LinkExploded { include_frequency, target_dtype } => {
                fields.push(Field::new("TARGET", target_dtype.clone(), true));
                if *include_frequency {
                    fields.push(Field::new("FREQUENCY", DataType::Int64, false));
                }
            }
            OutputMode::LinkArray { include_frequency, target_dtype } => {
                fields.push(Field::new(
                    "TARGET",
                    DataType::List(Arc::new(Field::new("item", target_dtype.clone(), true))),
                    true,
                ));
                if *include_frequency {
                    fields.push(Field::new(
                        "FREQUENCY",
                        DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
                        false,
                    ));
                }
            }
        }

        if include_entity_bit {
            fields.push(Field::new("ENTITY_INCLUDE_BIT", DataType::Boolean, false));
        }
        let schema = Arc::new(Schema::new(fields));

        let file = File::create(path)
            .with_context(|| format!("creating shard file {}", path.display()))?;
        let buf_writer = BufWriter::with_capacity(8 * 1024 * 1024, file); // 8MB buffer

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let writer = ArrowWriter::try_new(buf_writer, schema.clone(), Some(props))
            .context("creating Arrow writer")?;

        Ok(Self {
            writer,
            schema,
            mode,
            has_include_bit: include_entity_bit,
            entity_dtype,
            entity_buf: Vec::with_capacity(BATCH_SIZE),
            timestamp_buf: Vec::with_capacity(BATCH_SIZE),
            include_bit_buf: Vec::with_capacity(if include_entity_bit { BATCH_SIZE } else { 0 }),
            target_f64_buf: Vec::new(),
            target_bool_buf: Vec::new(),
            target_value_buf: Vec::new(),
            frequency_buf: Vec::new(),
            target_value_list_buf: Vec::new(),
            frequency_list_buf: Vec::new(),
            rows_written: 0,
        })
    }

    /// Buffer a single node-prediction output row.
    ///
    /// The target value is dispatched by the writer's target dtype:
    /// - Boolean: stored in `target_bool_buf`
    /// - Float64 (default): converted via `value_to_f64` into `target_f64_buf`
    #[inline]
    pub fn write_row(
        &mut self,
        entity: Value,
        timestamp: i64,
        target: &Value,
        entity_include_bit: bool,
    ) -> Result<()> {
        self.entity_buf.push(entity);
        self.timestamp_buf.push(timestamp);
        match &self.mode {
            OutputMode::Node { target_dtype } if *target_dtype == DataType::Boolean => {
                let b = match target {
                    Value::Bool(b) => *b,
                    Value::Int(i) => *i != 0,
                    Value::Float(f) => *f != 0.0,
                    _ => false,
                };
                self.target_bool_buf.push(b);
            }
            _ => {
                self.target_f64_buf.push(value_to_f64(target));
            }
        }
        if self.has_include_bit {
            self.include_bit_buf.push(entity_include_bit);
        }
        if self.entity_buf.len() >= BATCH_SIZE {
            self.flush_batch()?;
        }
        Ok(())
    }

    /// Buffer a single exploded link-prediction row (one per distinct target).
    #[inline]
    pub fn write_row_exploded(
        &mut self,
        entity: Value,
        timestamp: i64,
        target: Value,
        frequency: i64,
        entity_include_bit: bool,
    ) -> Result<()> {
        self.entity_buf.push(entity);
        self.timestamp_buf.push(timestamp);
        self.target_value_buf.push(target);
        self.frequency_buf.push(frequency);
        if self.has_include_bit {
            self.include_bit_buf.push(entity_include_bit);
        }
        if self.entity_buf.len() >= BATCH_SIZE {
            self.flush_batch()?;
        }
        Ok(())
    }

    /// Buffer a single array link-prediction row (one per entity/timestamp).
    #[inline]
    pub fn write_row_array(
        &mut self,
        entity: Value,
        timestamp: i64,
        targets: Vec<Value>,
        frequencies: &[i64],
        entity_include_bit: bool,
    ) -> Result<()> {
        self.entity_buf.push(entity);
        self.timestamp_buf.push(timestamp);
        self.target_value_list_buf.push(targets);
        self.frequency_list_buf.push(frequencies.to_vec());
        if self.has_include_bit {
            self.include_bit_buf.push(entity_include_bit);
        }
        if self.entity_buf.len() >= BATCH_SIZE {
            self.flush_batch()?;
        }
        Ok(())
    }

    /// Flush buffered rows as an Arrow RecordBatch.
    fn flush_batch(&mut self) -> Result<()> {
        if self.entity_buf.is_empty() {
            return Ok(());
        }

        let entity_arr = flush_entity_buf(&mut self.entity_buf, &self.entity_dtype);
        // Convert epoch seconds → nanoseconds to match Kumo's datetime64[ns] convention.
        let ts_nanos: Vec<i64> = std::mem::take(&mut self.timestamp_buf)
            .into_iter()
            .map(|s| s.saturating_mul(1_000_000_000))
            .collect();
        let ts_arr = TimestampNanosecondArray::from(ts_nanos);

        let mut columns: Vec<Arc<dyn arrow_array::Array>> = vec![
            entity_arr,
            Arc::new(ts_arr),
        ];

        match &self.mode {
            OutputMode::Node { target_dtype } if *target_dtype == DataType::Boolean => {
                let target_arr = BooleanArray::from(std::mem::take(&mut self.target_bool_buf));
                columns.push(Arc::new(target_arr));
            }
            OutputMode::Node { .. } => {
                let target_arr = Float64Array::from(std::mem::take(&mut self.target_f64_buf));
                columns.push(Arc::new(target_arr));
            }
            OutputMode::LinkExploded { include_frequency, target_dtype } => {
                let target_arr = flush_value_buf(&mut self.target_value_buf, target_dtype);
                columns.push(target_arr);

                if *include_frequency {
                    let freq_arr = Int64Array::from(std::mem::take(&mut self.frequency_buf));
                    columns.push(Arc::new(freq_arr));
                } else {
                    self.frequency_buf.clear();
                }
            }
            OutputMode::LinkArray { include_frequency, target_dtype } => {
                let lists = std::mem::take(&mut self.target_value_list_buf);
                let target_arr = flush_value_list_buf(lists, target_dtype);
                columns.push(target_arr);

                if *include_frequency {
                    let freq_lists = std::mem::take(&mut self.frequency_list_buf);
                    let mut freq_builder =
                        ListBuilder::new(Int64Builder::new());
                    for list in &freq_lists {
                        for &f in list {
                            freq_builder.values().append_value(f);
                        }
                        freq_builder.append(true);
                    }
                    columns.push(Arc::new(freq_builder.finish()));
                } else {
                    self.frequency_list_buf.clear();
                }
            }
        }

        if self.has_include_bit {
            let bit_arr = BooleanArray::from(std::mem::take(&mut self.include_bit_buf));
            columns.push(Arc::new(bit_arr));
        }

        let batch = RecordBatch::try_new(self.schema.clone(), columns)
            .context("creating RecordBatch")?;

        self.rows_written += batch.num_rows() as u64;
        self.writer.write(&batch).context("writing RecordBatch")?;

        // Re-allocate common buffers with capacity
        self.entity_buf = Vec::with_capacity(BATCH_SIZE);
        self.timestamp_buf = Vec::with_capacity(BATCH_SIZE);
        if self.has_include_bit {
            self.include_bit_buf = Vec::with_capacity(BATCH_SIZE);
        }

        Ok(())
    }

    /// Flush remaining rows and close the Parquet file.
    pub fn finish(mut self) -> Result<u64> {
        self.flush_batch()?;
        self.writer.close().context("closing Parquet writer")?;
        Ok(self.rows_written)
    }

    /// Number of rows written so far (including buffered).
    pub fn rows_buffered(&self) -> u64 {
        self.rows_written + self.entity_buf.len() as u64
    }
}

fn value_to_f64(v: &Value) -> f64 {
    match v {
        Value::Int(i) => *i as f64,
        Value::Float(f) => *f,
        Value::Bool(b) => if *b { 1.0 } else { 0.0 },
        Value::Str(_) => 0.0,
        Value::List(v) => v.len() as f64,
    }
}

/// Convert a flat buffer of `Value` to the correct Arrow array based on dtype.
/// Used for both ENTITY and LP TARGET columns.
fn flush_value_buf(buf: &mut Vec<Value>, dtype: &DataType) -> Arc<dyn arrow_array::Array> {
    flush_entity_buf(buf, dtype)
}

/// Convert a list-of-lists `Value` buffer to a List Arrow array based on dtype.
/// Used for LinkArray TARGET columns.
fn flush_value_list_buf(
    lists: Vec<Vec<Value>>,
    dtype: &DataType,
) -> Arc<dyn arrow_array::Array> {
    match dtype {
        DataType::Int64 => {
            let mut builder = ListBuilder::new(Int64Builder::new());
            for list in &lists {
                for v in list {
                    let i = match v {
                        Value::Int(i) => *i,
                        Value::Float(f) => *f as i64,
                        _ => 0,
                    };
                    builder.values().append_value(i);
                }
                builder.append(true);
            }
            Arc::new(builder.finish())
        }
        DataType::Float64 => {
            let mut builder = ListBuilder::new(arrow_array::builder::Float64Builder::new());
            for list in &lists {
                for v in list {
                    let f = match v {
                        Value::Float(f) => *f,
                        Value::Int(i) => *i as f64,
                        _ => 0.0,
                    };
                    builder.values().append_value(f);
                }
                builder.append(true);
            }
            Arc::new(builder.finish())
        }
        _ => {
            // Utf8 and all other types: convert to string
            let mut builder = ListBuilder::new(StringBuilder::new());
            for list in &lists {
                for v in list {
                    let s = match v {
                        Value::Str(s) => s.clone(),
                        Value::Int(i) => i.to_string(),
                        Value::Float(f) => f.to_string(),
                        Value::Bool(b) => b.to_string(),
                        Value::List(_) => String::new(),
                    };
                    builder.values().append_value(&s);
                }
                builder.append(true);
            }
            Arc::new(builder.finish())
        }
    }
}

/// Convert a buffer of `Value` entities to the correct Arrow array based on dtype.
fn flush_entity_buf(buf: &mut Vec<Value>, dtype: &DataType) -> Arc<dyn arrow_array::Array> {
    let entities = std::mem::take(buf);
    match dtype {
        DataType::Int64 => {
            let values: Vec<i64> = entities
                .into_iter()
                .map(|v| match v {
                    Value::Int(i) => i,
                    Value::Float(f) => f as i64,
                    _ => 0,
                })
                .collect();
            Arc::new(Int64Array::from(values))
        }
        DataType::Float64 => {
            let values: Vec<f64> = entities
                .into_iter()
                .map(|v| match v {
                    Value::Float(f) => f,
                    Value::Int(i) => i as f64,
                    _ => 0.0,
                })
                .collect();
            Arc::new(Float64Array::from(values))
        }
        // Utf8 and all other types: convert to string representation
        _ => {
            let values: Vec<String> = entities
                .into_iter()
                .map(|v| match v {
                    Value::Str(s) => s,
                    Value::Int(i) => i.to_string(),
                    Value::Float(f) => f.to_string(),
                    Value::Bool(b) => b.to_string(),
                    Value::List(_) => String::new(),
                })
                .collect();
            Arc::new(StringArray::from(
                values.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_and_read_parquet() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("shard_0.parquet");

        // Write some rows (no include bit, Int64 entity)
        let mut writer =
            ParquetShardWriter::new(&path, OutputMode::Node { target_dtype: DataType::Float64 }, false, DataType::Int64)?;
        for i in 0..100i64 {
            writer.write_row(Value::Int(i), i * 100, &Value::Float(i as f64 * 1.5), true)?;
        }
        let rows = writer.finish()?;
        assert_eq!(rows, 100);

        // Read back and verify
        let file = File::open(&path)?;
        let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)?
            .build()?;

        let mut total_rows = 0;
        for batch in reader {
            let batch = batch?;
            total_rows += batch.num_rows();
            assert_eq!(batch.num_columns(), 3); // No include bit column

            let entities = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let timestamps = batch
                .column(1)
                .as_any()
                .downcast_ref::<TimestampNanosecondArray>()
                .unwrap();
            let targets = batch
                .column(2)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();

            // Verify first row (timestamps stored as nanoseconds)
            assert_eq!(entities.value(0), 0);
            assert_eq!(timestamps.value(0), 0); // 0 seconds * 1e9 = 0 ns
            assert_eq!(targets.value(0), 0.0);
        }
        assert_eq!(total_rows, 100);

        Ok(())
    }

    #[test]
    fn test_write_with_include_bit() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("shard_bit.parquet");

        let mut writer =
            ParquetShardWriter::new(&path, OutputMode::Node { target_dtype: DataType::Float64 }, true, DataType::Int64)?;
        writer.write_row(Value::Int(0), 100, &Value::Float(1.0), true)?;
        writer.write_row(Value::Int(1), 100, &Value::Float(2.0), false)?;
        writer.write_row(Value::Int(2), 100, &Value::Float(3.0), true)?;
        let rows = writer.finish()?;
        assert_eq!(rows, 3);

        let file = File::open(&path)?;
        let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)?
            .build()?;

        for batch in reader {
            let batch = batch?;
            assert_eq!(batch.num_columns(), 4);
            let bits = batch
                .column(3)
                .as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap();
            assert!(bits.value(0));   // entity 0: true
            assert!(!bits.value(1));  // entity 1: false
            assert!(bits.value(2));   // entity 2: true
        }

        Ok(())
    }

    #[test]
    fn test_batch_flush() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("shard_batch.parquet");

        let mut writer =
            ParquetShardWriter::new(&path, OutputMode::Node { target_dtype: DataType::Float64 }, false, DataType::Int64)?;

        // Write more than BATCH_SIZE rows to trigger auto-flush
        for i in 0..(BATCH_SIZE as i64 + 100) {
            writer.write_row(Value::Int(i), i, &Value::Float(i as f64), true)?;
        }
        let rows = writer.finish()?;
        assert_eq!(rows, BATCH_SIZE as u64 + 100);

        Ok(())
    }

    #[test]
    fn test_write_exploded() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("shard_exploded.parquet");

        let mut writer = ParquetShardWriter::new(
            &path,
            OutputMode::LinkExploded { include_frequency: true, target_dtype: DataType::Utf8 },
            false,
            DataType::Int64,
        )?;
        writer.write_row_exploded(Value::Int(0), 100, Value::Str("Nike".into()), 3, true)?;
        writer.write_row_exploded(Value::Int(0), 100, Value::Str("Adidas".into()), 1, true)?;
        writer.write_row_exploded(Value::Int(1), 200, Value::Str("Puma".into()), 2, true)?;
        let rows = writer.finish()?;
        assert_eq!(rows, 3);

        let file = File::open(&path)?;
        let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)?
            .build()?;

        for batch in reader {
            let batch = batch?;
            assert_eq!(batch.num_columns(), 4); // ENTITY, TIMESTAMP, TARGET, FREQUENCY
            let targets = batch
                .column(2)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            assert_eq!(targets.value(0), "Nike");
            assert_eq!(targets.value(1), "Adidas");
            assert_eq!(targets.value(2), "Puma");

            let freqs = batch
                .column(3)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            assert_eq!(freqs.value(0), 3);
            assert_eq!(freqs.value(1), 1);
            assert_eq!(freqs.value(2), 2);
        }

        Ok(())
    }

    #[test]
    fn test_write_exploded_no_frequency() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("shard_exploded_nf.parquet");

        let mut writer = ParquetShardWriter::new(
            &path,
            OutputMode::LinkExploded { include_frequency: false, target_dtype: DataType::Utf8 },
            false,
            DataType::Int64,
        )?;
        writer.write_row_exploded(Value::Int(0), 100, Value::Str("Nike".into()), 0, true)?;
        writer.write_row_exploded(Value::Int(0), 100, Value::Str("Adidas".into()), 0, true)?;
        let rows = writer.finish()?;
        assert_eq!(rows, 2);

        let file = File::open(&path)?;
        let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)?
            .build()?;

        for batch in reader {
            let batch = batch?;
            assert_eq!(batch.num_columns(), 3); // ENTITY, TIMESTAMP, TARGET (no FREQUENCY)
        }

        Ok(())
    }

    #[test]
    fn test_write_array() -> Result<()> {
        use arrow_array::{Array, ListArray};

        let dir = tempfile::tempdir()?;
        let path = dir.path().join("shard_array.parquet");

        let mut writer = ParquetShardWriter::new(
            &path,
            OutputMode::LinkArray { include_frequency: true, target_dtype: DataType::Utf8 },
            false,
            DataType::Int64,
        )?;
        writer.write_row_array(
            Value::Int(0),
            100,
            vec![Value::Str("Nike".into()), Value::Str("Adidas".into())],
            &[3, 1],
            true,
        )?;
        writer.write_row_array(Value::Int(1), 200, vec![Value::Str("Puma".into())], &[2], true)?;
        let rows = writer.finish()?;
        assert_eq!(rows, 2);

        let file = File::open(&path)?;
        let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)?
            .build()?;

        for batch in reader {
            let batch = batch?;
            assert_eq!(batch.num_columns(), 4); // ENTITY, TIMESTAMP, TARGET(list), FREQUENCY(list)

            let target_list = batch
                .column(2)
                .as_any()
                .downcast_ref::<ListArray>()
                .unwrap();
            assert_eq!(target_list.len(), 2); // 2 rows

            let freq_list = batch
                .column(3)
                .as_any()
                .downcast_ref::<ListArray>()
                .unwrap();
            assert_eq!(freq_list.len(), 2);
        }

        Ok(())
    }
}
