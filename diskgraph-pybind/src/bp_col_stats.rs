//! Compute Kumo-format column statistics over an arbitrary parquet output
//! (e.g. batch-prediction `predictions.parquet`), reusing the col-stats
//! machinery from `diskgraph-ingest`.
//!
//! This is the post-hoc read-back analog of `tt_executor.rs`'s
//! `compute_train_table_col_stats_payload` — it lives at the binding layer
//! (not in the ingest crate) because it's orchestration over a finished
//! parquet, not part of the graph-ingest pipeline. It reuses ingest's
//! `FileColStats` / `TableCategoryCounts` / `to_kumo_json` primitives.
//!
//! Design notes:
//!   - **Row-group parallelism**: each parquet row group is processed by a
//!     rayon worker into its own `FileColStats`, all sharing one
//!     `Arc<TableCategoryCounts>`. So a single large `predictions.parquet`
//!     (one file, many row groups) parallelizes fully — parallelism is a
//!     function of total row groups, not file count.
//!   - **Memory-bounded category counts**: we use the Hybrid strategy
//!     (`MAX_CATEGORIES = 1M` cap, then abort → HLL fallback). A naive
//!     `HashMap`/`HashSet` of distinct values would OOM on a billions-of-rows
//!     high-cardinality column (e.g. the entity id); the cap prevents that.
//!   - **Single read**: `to_kumo_json` drains the shared category counts
//!     (destructive), so we serialize once to the full map, then derive the
//!     min variant via `minify_kumo_stats` — no second read, no double-drain.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use arrow_array::{
    Array, BooleanArray, Float64Array, Int64Array, StringArray, TimestampNanosecondArray,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use serde_json::{Map as JsonMap, Value as JsonValue};

use diskgraph_core::accumulators::CategoryCountStrategy;
use diskgraph_ingest::col_stats::{minify_kumo_stats, FileColStats, TableCategoryCounts};
use diskgraph_ingest::graph_config::{ColumnSpec, IngestedType};
use diskgraph_rowpack::ColumnValueObserver;

/// CATEGORY_COUNTS truncation limit for the "min" variant (matches ingest).
const MIN_COL_STATS_CATEGORY_LIMIT: usize = 20;

/// Drive a `ColumnValueObserver` from one Arrow column for one batch.
///
/// Handles the Arrow array types BP-output parquet actually produces (the
/// same set vetted by `compute_train_table_col_stats_payload`). Unexpected
/// types are no-ops: that column's stats stay empty rather than wrong.
fn drive_from_arrow_column<O: ColumnValueObserver>(
    col_idx: usize,
    array: &dyn Array,
    ingested: &IngestedType,
    observer: &mut O,
) {
    match ingested {
        IngestedType::Int64
        | IngestedType::Int32
        | IngestedType::Int16
        | IngestedType::Int8
        | IngestedType::UInt64
        | IngestedType::UInt32
        | IngestedType::UInt16
        | IngestedType::UInt8 => {
            if let Some(a) = array.as_any().downcast_ref::<Int64Array>() {
                for i in 0..a.len() {
                    if a.is_null(i) {
                        observer.observe_null(col_idx);
                    } else {
                        observer.observe_i64(col_idx, a.value(i));
                    }
                }
            }
        }
        IngestedType::Float64 | IngestedType::Float32 => {
            if let Some(a) = array.as_any().downcast_ref::<Float64Array>() {
                for i in 0..a.len() {
                    if a.is_null(i) {
                        observer.observe_null(col_idx);
                    } else {
                        observer.observe_f64(col_idx, a.value(i));
                    }
                }
            }
        }
        IngestedType::Bool => {
            if let Some(a) = array.as_any().downcast_ref::<BooleanArray>() {
                for i in 0..a.len() {
                    if a.is_null(i) {
                        observer.observe_null(col_idx);
                    } else {
                        observer.observe_i64(col_idx, i64::from(a.value(i)));
                    }
                }
            }
        }
        IngestedType::String => {
            if let Some(a) = array.as_any().downcast_ref::<StringArray>() {
                for i in 0..a.len() {
                    if a.is_null(i) {
                        observer.observe_null(col_idx);
                    } else {
                        observer.observe_string(col_idx, a.value(i));
                    }
                }
            }
        }
        IngestedType::Timestamp => {
            if let Some(a) = array.as_any().downcast_ref::<TimestampNanosecondArray>() {
                for i in 0..a.len() {
                    if a.is_null(i) {
                        observer.observe_null(col_idx);
                    } else {
                        observer.observe_timestamp(col_idx, a.value(i) / 1_000_000_000);
                    }
                }
            } else if let Some(a) = array.as_any().downcast_ref::<Int64Array>() {
                for i in 0..a.len() {
                    if a.is_null(i) {
                        observer.observe_null(col_idx);
                    } else {
                        observer.observe_timestamp(col_idx, a.value(i));
                    }
                }
            }
        }
        // BP output (per get_bp_columns_by_task_type) doesn't emit list /
        // array / decimal types. Extend here if a future schema needs them.
        _ => {}
    }
}

/// Process a single row group of one parquet file into its own `FileColStats`,
/// flushing category counts into the shared `Arc<TableCategoryCounts>`.
fn stats_for_row_group(
    path: &Path,
    row_group: usize,
    columns: &[ColumnSpec],
    table_cat: &Arc<TableCategoryCounts>,
) -> Result<FileColStats> {
    let mut fcs = FileColStats::new_with_categories(columns, Arc::clone(table_cat));
    let file = File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_row_groups(vec![row_group])
        .build()?;
    for batch_res in reader {
        let batch = batch_res?;
        for (i, col) in columns.iter().enumerate() {
            drive_from_arrow_column(i, batch.column(i).as_ref(), &col.ingested_dtype, &mut fcs);
        }
    }
    // Push this worker's local category counts into the shared (concurrent)
    // state — the same pattern ingest uses from its parallel file workers.
    fcs.flush_categories();
    Ok(fcs)
}

/// Compute Kumo-format column statistics over one or more parquet files,
/// returning `(full_json, min_json)`.
///
/// `columns` must match the parquet schema column-for-column (same order,
/// same count).
pub fn compute_parquet_col_stats(
    parquet_paths: &[&Path],
    columns: &[ColumnSpec],
    data_version: &str,
) -> Result<(String, String)> {
    let names: Vec<String> = columns.iter().map(|c| c.name.clone()).collect();
    let types: Vec<IngestedType> = columns.iter().map(|c| c.ingested_dtype.clone()).collect();
    let stypes: Vec<_> = columns.iter().map(|c| c.semantic_type).collect();

    // Shared, memory-bounded category counts. Hybrid + no row subsampling:
    // exact counts up to MAX_CATEGORIES, then abort → HLL fallback for
    // num_unique. Bounds memory on high-cardinality columns (entity id) at
    // billions of rows where a plain hashmap would OOM.
    let table_cat = Arc::new(TableCategoryCounts::new(
        CategoryCountStrategy::Hybrid,
        &stypes,
        /*row_subsample=*/ false,
    ));

    // Work list: (path_idx, row_group_idx) across all input files.
    let mut work: Vec<(usize, usize)> = Vec::new();
    for (pi, path) in parquet_paths.iter().enumerate() {
        let file = File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let num_row_groups = builder.metadata().num_row_groups();
        for rg in 0..num_row_groups {
            work.push((pi, rg));
        }
    }

    let mut acc = if work.is_empty() {
        // Empty / no-row-group input: still emit a zero-row stats payload.
        let mut fcs = FileColStats::new_with_categories(columns, Arc::clone(&table_cat));
        fcs.flush_categories();
        fcs
    } else {
        // Parallelize across row groups; each yields a FileColStats whose
        // category counts are already flushed to the shared `table_cat`.
        let partials: Vec<FileColStats> = work
            .par_iter()
            .map(|&(pi, rg)| stats_for_row_group(parquet_paths[pi], rg, columns, &table_cat))
            .collect::<Result<Vec<_>>>()?;

        // Merge the per-worker scalar accumulators (welford / hll / tdigest /
        // counts). Category counts already live in the shared `table_cat`.
        let mut iter = partials.into_iter();
        let mut acc = iter.next().expect("work is non-empty");
        for mut p in iter {
            acc.merge(&mut p);
        }
        acc
    };

    // Single serialization: drains the shared category counts (destructive),
    // so we build the full map once and derive min from it in-memory.
    let full = acc.to_kumo_json(&names, &types, Some(&table_cat), None, data_version);
    let min = minify_kumo_stats(&full, MIN_COL_STATS_CATEGORY_LIMIT);
    Ok((to_json_string(&full)?, to_json_string(&min)?))
}

fn to_json_string(map: &JsonMap<String, JsonValue>) -> Result<String> {
    Ok(serde_json::to_string(map)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::Arc as StdArc;

    use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use diskgraph_ingest::graph_config::SemanticType;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use tempfile::tempdir;

    /// Write a parquet with an explicit max rows-per-row-group so the test
    /// exercises the multi-row-group parallel path on a single file.
    fn write_parquet(path: &Path, schema: Schema, columns: Vec<ArrayRef>, max_rows_per_rg: usize) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let file = fs::File::create(path).unwrap();
        let batch = RecordBatch::try_new(StdArc::new(schema), columns).unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_size(max_rows_per_rg)
            .build();
        let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    fn col(name: &str, ingested: IngestedType, stype: SemanticType) -> ColumnSpec {
        ColumnSpec {
            name: name.to_string(),
            source_dtype: None,
            ingested_dtype: ingested,
            fixed_len: None,
            timestamp_format: None,
            timestamp_unit: None,
            precision: None,
            scale: None,
            semantic_type: Some(stype),
            multicategorical_delimiter: None,
        }
    }

    fn val(full: &str, key: &str) -> JsonValue {
        let m: JsonMap<String, JsonValue> = serde_json::from_str(full).unwrap();
        m[key]["value"].clone()
    }

    /// Multi-row-group single file → parallel workers + merge must equal truth.
    #[test]
    fn parallel_row_groups_match_truth() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("predictions.parquet");
        let n = 1000usize;
        let schema = Schema::new(vec![
            Field::new("entity", DataType::Utf8, false),
            Field::new("SCORE", DataType::Float64, true),
            Field::new("CLASS", DataType::Utf8, true),
        ]);
        let entity: Vec<String> = (0..n).map(|i| format!("u{i}")).collect();
        let score: Vec<Option<f64>> = (0..n).map(|i| Some(i as f64 / 1000.0)).collect();
        let class: Vec<Option<&str>> =
            (0..n).map(|i| Some(["A", "B", "C"][i % 3])).collect();
        write_parquet(
            &path,
            schema,
            vec![
                StdArc::new(StringArray::from(entity)) as ArrayRef,
                StdArc::new(Float64Array::from(score)) as ArrayRef,
                StdArc::new(StringArray::from(class)) as ArrayRef,
            ],
            // 128 rows/group → ~8 row groups → real parallelism.
            128,
        );

        let columns = vec![
            col("entity", IngestedType::String, SemanticType::Id),
            col("SCORE", IngestedType::Float64, SemanticType::Numerical),
            col("CLASS", IngestedType::String, SemanticType::Categorical),
        ];
        let (full, min) =
            compute_parquet_col_stats(&[path.as_path()], &columns, "v1").unwrap();

        // COUNT == n across columns (merge across row groups correct).
        assert_eq!(val(&full, "entity__kumo.ID__COUNT__False"), n as u64);
        assert_eq!(val(&full, "SCORE__kumo.numerical__COUNT__False"), n as u64);
        assert_eq!(val(&full, "CLASS__kumo.categorical__COUNT__False"), n as u64);

        // SCORE min/max exact across the whole file.
        let mn = val(&full, "SCORE__kumo.numerical__MIN__False").as_f64().unwrap();
        let mx = val(&full, "SCORE__kumo.numerical__MAX__False").as_f64().unwrap();
        assert!((mn - 0.0).abs() < 1e-9, "min={mn}");
        assert!((mx - 0.999).abs() < 1e-9, "max={mx}");

        // CLASS num_unique == 3 EXACT (Hybrid map complete, overrides HLL).
        assert_eq!(val(&full, "CLASS__kumo.categorical__NUM_UNIQUE__False"), 3u64);

        // entity num_unique == n EXACT (1000 distinct, well under the 1M cap).
        assert_eq!(val(&full, "entity__kumo.ID__NUM_UNIQUE__False"), n as u64);

        // CLASS category counts populated (new_with_categories).
        let mfull: JsonMap<String, JsonValue> = serde_json::from_str(&full).unwrap();
        assert!(mfull.contains_key("CLASS__kumo.categorical__CATEGORY_COUNTS__False"));

        // min variant parses and keeps the scalar COUNT.
        let mmin: JsonMap<String, JsonValue> = serde_json::from_str(&min).unwrap();
        assert_eq!(mmin["CLASS__kumo.categorical__COUNT__False"]["value"], 1000u64);
    }

    /// Multiple part files merge correctly.
    #[test]
    fn merges_multiple_files() {
        let dir = tempdir().unwrap();
        let p1 = dir.path().join("part-0.parquet");
        let p2 = dir.path().join("part-1.parquet");
        let schema = || Schema::new(vec![Field::new("v", DataType::Int64, false)]);
        write_parquet(
            &p1,
            schema(),
            vec![StdArc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef],
            2,
        );
        write_parquet(
            &p2,
            schema(),
            vec![StdArc::new(Int64Array::from(vec![4, 5, 6])) as ArrayRef],
            2,
        );
        let columns = vec![col("v", IngestedType::Int64, SemanticType::Numerical)];
        let (full, _min) =
            compute_parquet_col_stats(&[p1.as_path(), p2.as_path()], &columns, "v1").unwrap();
        assert_eq!(val(&full, "v__kumo.numerical__COUNT__False"), 6u64);
        let mx = val(&full, "v__kumo.numerical__MAX__False").as_f64().unwrap();
        assert!((mx - 6.0).abs() < 1e-9);
    }
}
