use crate::config::{IdKind, TimestampKind};
use anyhow::{anyhow, Result};
use arrow_array::{
    Array, ArrayRef, Float32Array, Float64Array, Int64Array, ListArray, RecordBatch,
    StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

pub struct ParquetStream {
    reader: ParquetRecordBatchReader,
    id_idx: usize,
    ts_idx: usize,
    pred_idx: usize,
    emb_idx: usize,
    id_kind: IdKind,
    ts_kind: TimestampKind,
}

impl ParquetStream {
    pub fn open(path: &Path, id: &str, ts: &str, pred: &str, emb: &str, id_kind: IdKind, ts_kind: TimestampKind) -> Result<Self> {
        let file = File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let schema = builder.schema();
        let id_idx = schema
            .index_of(id)
            .map_err(|_| anyhow!("missing id column {}", id))?;
        let ts_idx = schema
            .index_of(ts)
            .map_err(|_| anyhow!("missing timestamp column {}", ts))?;
        let pred_idx = schema
            .index_of(pred)
            .map_err(|_| anyhow!("missing prediction column {}", pred))?;
        let emb_idx = schema
            .index_of(emb)
            .map_err(|_| anyhow!("missing embedding column {}", emb))?;
        let reader = builder.build()?;
        Ok(Self { reader, id_idx, ts_idx, pred_idx, emb_idx, id_kind, ts_kind })
    }

    pub fn next_batch(&mut self) -> Option<Result<BatchView>> {
        self.reader.next().map(|res| {
            res.map_err(|e| e.into()).map(|batch| {
            let ids = batch.column(self.id_idx).clone();
            let ts = batch.column(self.ts_idx).clone();
            let preds = batch.column(self.pred_idx).clone();
            let embs = batch.column(self.emb_idx).clone();
            BatchView { batch, ids, ts, preds, embs, id_kind: self.id_kind.clone(), ts_kind: self.ts_kind.clone() }
        })
        })
    }
}

pub struct BatchView {
    pub batch: RecordBatch,
    pub ids: ArrayRef,
    pub ts: ArrayRef,
    pub preds: ArrayRef,
    pub embs: ArrayRef,
    id_kind: IdKind,
    ts_kind: TimestampKind,
}

impl BatchView {
    pub fn len(&self) -> usize {
        self.batch.num_rows()
    }

    pub fn id_as_string(&self, idx: usize) -> Result<String> {
        match self.id_kind {
            IdKind::Int64 => {
                let arr = self.ids.as_any().downcast_ref::<Int64Array>().ok_or_else(|| anyhow!("id column not int64"))?;
                Ok(arr.value(idx).to_string())
            }
            IdKind::Utf8 => {
                let arr = self.ids.as_any().downcast_ref::<StringArray>().ok_or_else(|| anyhow!("id column not utf8"))?;
                Ok(arr.value(idx).to_string())
            }
        }
    }

    pub fn timestamp_as_i64(&self, idx: usize) -> Result<i64> {
        match self.ts_kind {
            TimestampKind::Int64 => {
                let arr = self.ts.as_any().downcast_ref::<Int64Array>().ok_or_else(|| anyhow!("timestamp not int64"))?;
                Ok(arr.value(idx))
            }
            TimestampKind::TimestampMillis => {
                let arr = self.ts.as_any().downcast_ref::<Int64Array>().ok_or_else(|| anyhow!("timestamp millis not int64"))?;
                Ok(arr.value(idx) / 1000)
            }
            TimestampKind::TimestampMicros => {
                let arr = self.ts.as_any().downcast_ref::<Int64Array>().ok_or_else(|| anyhow!("timestamp micros not int64"))?;
                Ok(arr.value(idx) / 1_000_000)
            }
            TimestampKind::TimestampNanos => {
                let arr = self.ts.as_any().downcast_ref::<Int64Array>().ok_or_else(|| anyhow!("timestamp nanos not int64"))?;
                Ok(arr.value(idx) / 1_000_000_000)
            }
        }
    }

    pub fn prediction_as_f64(&self, idx: usize) -> Result<f64> {
        if let Some(arr) = self.preds.as_any().downcast_ref::<Float64Array>() {
            return Ok(arr.value(idx));
        }
        if let Some(arr) = self.preds.as_any().downcast_ref::<Float32Array>() {
            return Ok(arr.value(idx) as f64);
        }
        Err(anyhow!("prediction column must be float"))
    }

    pub fn embedding_as_f32(&self, idx: usize) -> Result<Vec<f32>> {
        if let Some(arr) = self.embs.as_any().downcast_ref::<ListArray>() {
            let values = arr.value(idx);
            if let Some(f32s) = values.as_any().downcast_ref::<Float32Array>() {
                return Ok((0..f32s.len()).map(|i| f32s.value(i)).collect());
            }
            if let Some(f64s) = values.as_any().downcast_ref::<Float64Array>() {
                return Ok((0..f64s.len()).map(|i| f64s.value(i) as f32).collect());
            }
        }
        Err(anyhow!("embedding column must be list<float>"))
    }
}

pub fn write_parquet(path: &Path, schema: Schema, batches: Vec<RecordBatch>) -> Result<()> {
    let file = File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, Arc::new(schema), Some(props))?;
    for batch in batches {
        writer.write(&batch)?;
    }
    writer.close()?;
    Ok(())
}

pub fn explanation_schema() -> Schema {
    Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("prediction", DataType::Float32, false),
        Field::new(
            "embedding",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, false))),
            false,
        ),
        Field::new("cohort_id", DataType::Int32, false),
        Field::new("cohort_mean_pred", DataType::Float32, false),
        Field::new("cohort_std_pred", DataType::Float32, false),
        Field::new("global_mean_pred", DataType::Float32, false),
        Field::new("local_lift", DataType::Float32, false),
        Field::new("contrast_cohort_id", DataType::Int32, false),
        Field::new("top_features", DataType::Utf8, false),
        Field::new("top_events", DataType::Utf8, false),
    ])
}

pub fn trends_schema() -> Schema {
    Schema::new(vec![
        Field::new("artifact_type", DataType::Utf8, false),
        Field::new("cohort_id", DataType::Int32, true),
        Field::new("time_bucket", DataType::Int32, true),
        Field::new("count", DataType::Int64, true),
        Field::new("mean_pred", DataType::Float32, true),
        Field::new("std_pred", DataType::Float32, true),
        Field::new("p10", DataType::Float32, true),
        Field::new("p50", DataType::Float32, true),
        Field::new("p90", DataType::Float32, true),
        Field::new("feature_key", DataType::Utf8, true),
        Field::new("lift", DataType::Float32, true),
        Field::new("support", DataType::Int64, true),
        Field::new("notes", DataType::Utf8, true),
    ])
}
