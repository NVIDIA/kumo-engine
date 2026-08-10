//! The context sample steps between partitions, and each partition it steps off
//! is handed back to the encode phase. Both halves can go wrong silently: a
//! dropped partition loses rows, and a sampled batch carried forward encodes
//! them twice. Row counts are what catch either.

use anyhow::Result;
use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use diskgraph_ingest::{ingest_with_stats, Dest, IngestOptions};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs;
use std::sync::Arc;
use tempfile::tempdir;

const FILES: i64 = 8;
const ROWS_PER_FILE: i64 = 5_000;

#[test]
fn strided_context_sample_reads_every_row_exactly_once() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    fs::create_dir_all(&input_root)?;

    // Each file holds its own slice of the `tag` vocabulary, which is the shape
    // that makes a single-partition sample unrepresentative.
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("tag", DataType::Utf8, false),
    ]);
    let mut paths = Vec::new();
    for file in 0..FILES {
        let ids: Vec<i64> = (0..ROWS_PER_FILE).map(|r| file * ROWS_PER_FILE + r).collect();
        let tags: Vec<String> = (0..ROWS_PER_FILE)
            .map(|r| format!("f{file}-tag{}", r % 50))
            .collect();
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringArray::from(tags)),
        ];
        let path = input_root.join(format!("part-{file}.parquet"));
        let props = WriterProperties::builder()
            .set_max_row_group_size(1024)
            .build();
        let writer_file = fs::File::create(&path)?;
        let mut writer =
            ArrowWriter::try_new(writer_file, Arc::new(schema.clone()), Some(props))?;
        writer.write(&RecordBatch::try_new(
            Arc::new(schema.clone()),
            columns,
        )?)?;
        writer.close()?;
        paths.push(path.to_string_lossy().to_string());
    }

    let source = serde_json::json!({
        "dataset_name": "context_stride",
        "tables": [{
            "name": "Item",
            "source": {"type": "parquet", "paths": paths},
            "primary_key": "id",
            "columns": [
                {"name": "id", "ingested_dtype": "int64"},
                {"name": "tag", "ingested_dtype": "string"}
            ]
        }]
    });
    let source_path = dir.path().join("diskgraph_config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let (_metadata, stats) = ingest_with_stats(
        &source_path,
        dir.path().join("out").to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    assert_eq!(
        *stats.node_counts.get("Item").unwrap_or(&0),
        (FILES * ROWS_PER_FILE) as u64,
        "every source row must be encoded exactly once"
    );
    Ok(())
}
