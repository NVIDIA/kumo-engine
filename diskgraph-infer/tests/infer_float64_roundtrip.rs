use anyhow::{anyhow, Result};
use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use diskgraph_index::graph::GraphIndexReader;
use diskgraph_infer::{infer_source_metadata, parse_root_source, InferOptions};
use diskgraph_ingest::graph_config::{DataSource, IngestedType};
use diskgraph_ingest::{ingest, Dest, IngestOptions};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tempfile::tempdir;

fn write_parquet(path: &Path, schema: Schema, columns: Vec<ArrayRef>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let props = WriterProperties::builder()
        .set_max_row_group_size(1024)
        .build();
    let file = fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, Arc::new(schema.clone()), Some(props))?;
    let batch = RecordBatch::try_new(Arc::new(schema), columns)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

// 300_000_000 is well above f32's 2^24 ≈ 16.8M exact-integer threshold,
// so the fractional ".375" is silently lost on downcast to f32.
const PRECISE_F64: f64 = 300_000_000.375_f64;

#[test]
fn ingest_preserves_float64_precision_above_16m() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");
    fs::create_dir_all(&input_root)?;
    fs::create_dir_all(&output_root)?;

    write_parquet(
        &input_root.join("Budget.parquet"),
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("production_budget", DataType::Float64, false),
        ]),
        vec![
            Arc::new(Int64Array::from(vec![0_i64])) as ArrayRef,
            Arc::new(Float64Array::from(vec![PRECISE_F64])) as ArrayRef,
        ],
    )?;

    let root = parse_root_source(input_root.to_str().unwrap())?;
    let result = infer_source_metadata(&root, InferOptions::default())?;
    let mut graph_config = result.metadata;

    let budget = graph_config
        .tables
        .iter()
        .find(|t| t.name == "Budget")
        .ok_or_else(|| anyhow!("Budget table missing from inferred config"))?;
    let col = budget
        .columns
        .iter()
        .find(|c| c.name == "production_budget")
        .ok_or_else(|| anyhow!("production_budget column missing"))?;
    assert!(
        matches!(col.ingested_dtype, IngestedType::Float64),
        "inferred ingested_dtype for production_budget should be Float64, got {:?}",
        col.ingested_dtype
    );

    for table in graph_config.tables.iter_mut() {
        if let DataSource::Parquet { paths } = &mut table.source {
            for p in paths.iter_mut() {
                let abs: PathBuf = input_root.join(&*p);
                *p = abs.to_string_lossy().to_string();
            }
        }
    }

    let source_path = dir.path().join("diskgraph_config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&graph_config)?)?;

    ingest(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    let reader =
        GraphIndexReader::open(&output_root.join("index_metadata.json"), &output_root)?;
    let table = reader
        .table("Budget")
        .ok_or_else(|| anyhow!("Budget table missing from index reader"))?;
    let view = table.edges_for_node(0)?;
    let value = table
        .decode_node_data_json(&view)?
        .ok_or_else(|| anyhow!("no decoded row for node 0"))?;
    let roundtripped = value["production_budget"]
        .as_f64()
        .ok_or_else(|| anyhow!("production_budget not a number in decoded row"))?;
    assert_eq!(
        roundtripped, PRECISE_F64,
        "expected Float64 value preserved through infer+ingest round-trip"
    );
    Ok(())
}
