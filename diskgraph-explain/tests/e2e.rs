use anyhow::Result;
use arrow_array::{Array, ArrayRef, Float32Array, Int64Array, ListArray, RecordBatch, StringArray};
use arrow_buffer::OffsetBuffer;
use arrow_schema::{DataType, Field, Schema};
use diskgraph_explain::ExplainConfig;
use diskgraph_explain::run;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;

fn write_input_parquet(path: &PathBuf) -> Result<()> {
    let ids = (0..20).map(|i| format!("user_{i}")).collect::<Vec<_>>();
    let timestamps = (0..20).map(|i| 1_700_000_000i64 + i as i64).collect::<Vec<_>>();

    let mut preds = Vec::with_capacity(20);
    let mut values: Vec<f32> = Vec::with_capacity(20 * 2);
    let mut offsets: Vec<i32> = Vec::with_capacity(21);
    offsets.push(0);

    for i in 0..20 {
        if i < 10 {
            preds.push(0.2 + (i as f32) * 0.01);
            values.push(0.1 + (i as f32) * 0.01);
            values.push(0.2 + (i as f32) * 0.01);
        } else {
            preds.push(0.8 + (i as f32) * 0.005);
            values.push(5.0 + (i as f32) * 0.01);
            values.push(5.1 + (i as f32) * 0.01);
        }
        offsets.push(values.len() as i32);
    }

    let list_values = Arc::new(Float32Array::from(values)) as ArrayRef;
    let offsets = OffsetBuffer::new(offsets.into());
    let list_array = ListArray::new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        offsets,
        list_values,
        None,
    );

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("prediction", DataType::Float32, false),
        Field::new("embedding", DataType::List(Arc::new(Field::new("item", DataType::Float32, false))), false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(Int64Array::from(timestamps)),
            Arc::new(Float32Array::from(preds)),
            Arc::new(list_array),
        ],
    )?;

    let file = File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

#[test]
fn e2e_explain() -> Result<()> {
    let tmp = TempDir::new()?;
    let input = tmp.path().join("input.parquet");
    let explanations = tmp.path().join("explanations.parquet");
    let trends = tmp.path().join("trends.parquet");

    write_input_parquet(&input)?;

    let config: ExplainConfig = serde_json::from_value(serde_json::json!({
        "input_parquet": input,
        "output_explanations_parquet": explanations,
        "output_global_trends_parquet": trends,
        "id_column": "id",
        "timestamp_column": "timestamp",
        "prediction_column": "prediction",
        "embedding_column": "embedding",
        "id_kind": "utf8",
        "timestamp_kind": "int64",
        "num_clusters": 2,
        "kmeans_passes": 2,
        "minibatch_size": 8,
        "knn_k": 5,
        "knn_sample_size": 10,
        "cluster_sample_size": 10,
        "ridge_lambda": 1e-3,
        "top_factors": 3,
        "time_bucket_secs": 1000,
        "max_time_buckets": 5
    }))?;

    run(config)?;

    let expl_file = File::open(&explanations)?;
    let expl_reader = ParquetRecordBatchReaderBuilder::try_new(expl_file)?.build()?;
    let mut total_rows = 0;
    for batch in expl_reader {
        let batch = batch?;
        total_rows += batch.num_rows();
        assert!(batch.schema().field_with_name("cluster_id").is_ok());
        assert!(batch.schema().field_with_name("top_factors").is_ok());
    }
    assert_eq!(total_rows, 20);

    let trend_file = File::open(&trends)?;
    let trend_reader = ParquetRecordBatchReaderBuilder::try_new(trend_file)?.build()?;
    let mut cohort_means: Vec<f32> = Vec::new();
    let mut has_feature_importance = false;
    for batch in trend_reader {
        let batch = batch?;
        let types = batch.column(batch.schema().index_of("artifact_type")?).as_any()
            .downcast_ref::<StringArray>().unwrap();
        let mean_pred = batch.column(batch.schema().index_of("mean_pred")?).as_any()
            .downcast_ref::<Float32Array>().unwrap();
        for i in 0..batch.num_rows() {
            let t = types.value(i);
            if t == "cohort_summary" {
                if mean_pred.is_valid(i) {
                    cohort_means.push(mean_pred.value(i));
                }
            }
            if t == "feature_importance" {
                has_feature_importance = true;
            }
        }
    }
    assert!(has_feature_importance);
    assert!(cohort_means.iter().any(|v| *v > 0.6));
    assert!(cohort_means.iter().any(|v| *v < 0.4));

    Ok(())
}
