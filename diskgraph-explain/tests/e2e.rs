use anyhow::Result;
use arrow_array::{Array, ArrayRef, Float32Array, Int64Array, ListArray, RecordBatch, StringArray, TimestampSecondArray};
use arrow_buffer::OffsetBuffer;
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use diskgraph_explain::ExplainConfig;
use diskgraph_explain::run;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;

fn write_parquet(path: &PathBuf, schema: Arc<Schema>, batch: RecordBatch) -> Result<()> {
    let file = File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn write_users(path: &PathBuf) -> Result<()> {
    let ids = vec!["u1", "u2", "u3", "u4", "u5", "u6"];
    let countries = vec!["us", "us", "us", "ca", "ca", "ca"];
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("country", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(ids)) as ArrayRef,
            Arc::new(StringArray::from(countries)),
        ],
    )?;
    write_parquet(path, schema, batch)
}

fn write_devices(path: &PathBuf) -> Result<()> {
    let ids = vec!["d1", "d2", "d3", "d4"];
    let os = vec!["android", "android", "ios", "ios"];
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("os", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(ids)) as ArrayRef,
            Arc::new(StringArray::from(os)),
        ],
    )?;
    write_parquet(path, schema, batch)
}

fn write_edges(path: &PathBuf) -> Result<()> {
    let user_ids = vec!["u1", "u2", "u3", "u4", "u5", "u6"];
    let device_ids = vec!["d1", "d2", "d1", "d3", "d4", "d3"];
    let ts = vec![1_700_000_000i64; 6];
    let schema = Arc::new(Schema::new(vec![
        Field::new("user_id", DataType::Utf8, false),
        Field::new("device_id", DataType::Utf8, false),
        Field::new("ts", DataType::Timestamp(TimeUnit::Second, None), false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(user_ids)) as ArrayRef,
            Arc::new(StringArray::from(device_ids)),
            Arc::new(TimestampSecondArray::from(ts)),
        ],
    )?;
    write_parquet(path, schema, batch)
}

fn write_input(path: &PathBuf) -> Result<()> {
    let ids = vec!["u1", "u2", "u3", "u4", "u5", "u6"];
    let ts = vec![1_700_000_100i64; 6];
    let preds = vec![0.9, 0.85, 0.88, 0.1, 0.2, 0.15];

    let mut values: Vec<f32> = Vec::new();
    let mut offsets: Vec<i32> = Vec::with_capacity(ids.len() + 1);
    offsets.push(0);
    for (i, _) in ids.iter().enumerate() {
        if i < 3 {
            values.push(5.0 + i as f32);
            values.push(5.5 + i as f32);
        } else {
            values.push(0.5 + i as f32);
            values.push(0.7 + i as f32);
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
            Arc::new(StringArray::from(ids)) as ArrayRef,
            Arc::new(Int64Array::from(ts)),
            Arc::new(Float32Array::from(preds)),
            Arc::new(list_array),
        ],
    )?;
    write_parquet(path, schema, batch)
}

fn write_config(path: &PathBuf, users: &PathBuf, devices: &PathBuf, edges: &PathBuf) -> Result<()> {
    let config = serde_json::json!({
        "dataset_name": "explain_test",
        "num_shards": 1,
        "edge_format": "binary",
        "tables": [
            {
                "name": "users",
                "format": "parquet",
                "paths": [users],
                "columns": [
                    {"name": "id", "ingested_dtype": "string"},
                    {"name": "country", "ingested_dtype": "string"}
                ],
                "primary_key": "id"
            },
            {
                "name": "devices",
                "format": "parquet",
                "paths": [devices],
                "columns": [
                    {"name": "id", "ingested_dtype": "string"},
                    {"name": "os", "ingested_dtype": "string"}
                ],
                "primary_key": "id"
            },
            {
                "name": "user_device",
                "format": "parquet",
                "paths": [edges],
                "columns": [
                    {"name": "user_id", "ingested_dtype": "string"},
                    {"name": "device_id", "ingested_dtype": "string"},
                    {"name": "ts", "ingested_dtype": "timestamp"}
                ],
                "edge_table": {
                    "src_fk": {"column": "user_id", "ref_table": "users", "ref_column": "id"},
                    "dst_fk": {"column": "device_id", "ref_table": "devices", "ref_column": "id"}
                }
            }
        ],
        "edge_sort_keys": {
            "edge_tables": {
                "user_device": {
                    "src_to_dst": {
                        "timestamp": {
                            "table": "user_device",
                            "column": "ts",
                            "encoding": {"kind": "int32", "anchor": 0, "step": 1}
                        }
                    },
                    "dst_to_src": null
                }
            }
        }
    });
    std::fs::write(path, serde_json::to_vec_pretty(&config)?)?;
    Ok(())
}

#[test]
fn e2e_explain_with_graph_features() -> Result<()> {
    let tmp = TempDir::new()?;
    let users = tmp.path().join("users.parquet");
    let devices = tmp.path().join("devices.parquet");
    let edges = tmp.path().join("edges.parquet");
    let config_path = tmp.path().join("diskgraph_config.json");
    let index_path = tmp.path().join("index");
    let input = tmp.path().join("input.parquet");
    let explanations = tmp.path().join("explanations.parquet");
    let trends = tmp.path().join("trends.parquet");

    write_users(&users)?;
    write_devices(&devices)?;
    write_edges(&edges)?;
    write_config(&config_path, &users, &devices, &edges)?;
    write_input(&input)?;

    let options = diskgraph_ingest::IngestOptions::default();
    diskgraph_ingest::ingest(&config_path, index_path.to_str().unwrap(), diskgraph_ingest::Dest::Local, options)?;

    let config: ExplainConfig = serde_json::from_value(serde_json::json!({
        "input_parquet": input,
        "output_explanations_parquet": explanations,
        "output_global_trends_parquet": trends,
        "index_path": index_path,
        "root_table": "users",
        "id_column": "id",
        "timestamp_column": "timestamp",
        "prediction_column": "prediction",
        "embedding_column": "embedding",
        "id_kind": "utf8",
        "id_lookup": "primary_key",
        "timestamp_kind": "int64",
        "num_clusters": 2,
        "kmeans_passes": 2,
        "relation_fanout": 4,
        "max_nodes": 50,
        "feature_top_k": 3,
        "event_top_k": 2,
        "min_support": 1,
        "attributes": [
            {"table": "devices", "columns": [{"name": "os", "kind": "categorical"}]}
        ]
    }))?;

    run(config)?;

    let expl_file = File::open(&explanations)?;
    let expl_reader = ParquetRecordBatchReaderBuilder::try_new(expl_file)?.build()?;
    let mut rows = 0;
    let mut has_android = false;
    for batch in expl_reader {
        let batch = batch?;
        rows += batch.num_rows();
        let top_features = batch
            .column(batch.schema().index_of("top_features")?)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..batch.num_rows() {
            let s = top_features.value(i);
            if s.contains("devices.os=android") {
                has_android = true;
            }
        }
    }
    assert_eq!(rows, 6);
    assert!(has_android);

    let trend_file = File::open(&trends)?;
    let trend_reader = ParquetRecordBatchReaderBuilder::try_new(trend_file)?.build()?;
    let mut has_feature_lift = false;
    for batch in trend_reader {
        let batch = batch?;
        let types = batch
            .column(batch.schema().index_of("artifact_type")?)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let feature_key = batch
            .column(batch.schema().index_of("feature_key")?)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..batch.num_rows() {
            if types.value(i) == "cohort_feature_lift" && feature_key.is_valid(i) {
                if feature_key.value(i).contains("devices.os=android") {
                    has_feature_lift = true;
                }
            }
        }
    }
    assert!(has_feature_lift);

    Ok(())
}
