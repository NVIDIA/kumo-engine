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
//!   - **SUMMARY_STATS_BY_CLASS**: the post-cutover replacement for Spark's
//!     group-by — SCORE grouped by CLASS into `{class: {min, max, mean}}` for
//!     multiclass/multilabel/ranking/link-pred output. It's the one
//!     cross-column stat, so it lives here (not in per-column `FileColStats`),
//!     accumulated per-row-group into thread-local `HashMap<class, Welford>`s
//!     that merge exactly. It is `EXACT_ONLY` (no cap / no truncation), so the
//!     map is bounded by the distinct-class count (the label/candidate-item
//!     set), NOT row count — safe for multiclass, and assumed bounded for
//!     ranking's candidate set. The identical entry is written to BOTH the full
//!     and min variants because `BP_MULTICLASS_CLASSIFICATION` maps to the MIN
//!     refresh stage on the kumo side (the `/prediction_stats` cache-fallback
//!     reads min), and it can't be truncated like CATEGORY_COUNTS.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use arrow_array::{
    Array, BooleanArray, Float32Array, Float64Array, Int64Array, RecordBatch, StringArray,
    TimestampNanosecondArray,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use serde_json::{json, Map as JsonMap, Value as JsonValue};

use diskgraph_core::accumulators::{
    kumo_entry, kumo_key, CategoryCountStrategy, WelfordAccumulator,
};
use diskgraph_ingest::col_stats::{minify_kumo_stats, FileColStats, TableCategoryCounts};
use diskgraph_ingest::graph_config::{ColumnSpec, IngestedType, SemanticType, TimestampUnit};
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
    col: &ColumnSpec,
    observer: &mut O,
) {
    match &col.ingested_dtype {
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
            } else if let Some(a) = array.as_any().downcast_ref::<Float32Array>() {
                // Model outputs (probabilities / scores) are frequently f32;
                // widen to f64 so numerical stats aren't silently dropped.
                for i in 0..a.len() {
                    if a.is_null(i) {
                        observer.observe_null(col_idx);
                    } else {
                        observer.observe_f64(col_idx, a.value(i) as f64);
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
                // Int64 epoch carries no Arrow TimeUnit, so scale to seconds
                // before `observe_timestamp`. Explicit `timestamp_unit` wins,
                // else infer per value from magnitude. Epoch-int only — no
                // encoded-date (`YYYYMMDD`) detection.
                for i in 0..a.len() {
                    if a.is_null(i) {
                        observer.observe_null(col_idx);
                    } else {
                        let raw = a.value(i);
                        let secs = match col
                            .timestamp_unit
                            .or_else(|| TimestampUnit::from_epoch_magnitude(raw))
                        {
                            Some(unit) => unit.scale_to_secs(raw),
                            None => raw,
                        };
                        observer.observe_timestamp(col_idx, secs);
                    }
                }
            }
        }
        // BP output (per get_bp_columns_by_task_type) doesn't emit list /
        // array / decimal types. Extend here if a future schema needs them.
        _ => {}
    }
}

/// Per-class accumulators over SCORE, keyed by the CLASS string value.
/// Backs `SUMMARY_STATS_BY_CLASS` (the post-cutover replacement for Spark's
/// group-by). Merge is exact and associative, so each row-group worker keeps a
/// thread-local map that's merged at the end — no shared map, no lock
/// contention on the handful of hot classes a multiclass job has.
type ByClassStats = HashMap<String, WelfordAccumulator>;

/// Locate the `(SCORE, CLASS)` column indices, if both are present and SCORE is
/// numerical. Only multiclass / multilabel / ranking / link-pred BP output
/// carries both (per `get_bp_columns_by_task_type`) — exactly when the kumo
/// side requests `ColStatsSet.BP_MULTICLASS_CLASSIFICATION`. Returns `None`
/// otherwise (regression/binary), so no by-class stat is produced.
fn find_score_class(columns: &[ColumnSpec]) -> Option<(usize, usize)> {
    let score = columns
        .iter()
        .position(|c| c.name == "SCORE" && c.semantic_type == Some(SemanticType::Numerical))?;
    let class = columns
        .iter()
        .position(|c| c.name == "CLASS" && c.semantic_type == Some(SemanticType::Categorical))?;
    Some((score, class))
}

/// Group SCORE (numerical) by CLASS (string) for one record batch, feeding each
/// finite SCORE into the per-class Welford accumulator. `class` only selects the
/// bucket; SCORE is the value being summarized — the one genuinely cross-column
/// stat. Rows with a null SCORE or null CLASS are skipped (valid-only
/// semantics); `Welford::observe` already drops non-finite scores.
fn accumulate_by_class(
    batch: &RecordBatch,
    score_idx: usize,
    class_idx: usize,
    out: &mut ByClassStats,
) {
    // CLASS must be plain Utf8 (matches the CATEGORY_COUNTS path's assumption;
    // BP writes CLASS as Dtype.string). Dictionary-encoded / non-string CLASS
    // is skipped rather than mis-read.
    let classes = match batch.column(class_idx).as_any().downcast_ref::<StringArray>() {
        Some(c) => c,
        None => return,
    };
    // SCORE may be Float64 or Float32 (model scores are often f32); widen f32.
    // Neither → skip (consistent with `drive_from_arrow_column`).
    let score_col = batch.column(score_idx);
    let scores_f64 = score_col.as_any().downcast_ref::<Float64Array>();
    let scores_f32 = score_col.as_any().downcast_ref::<Float32Array>();
    if scores_f64.is_none() && scores_f32.is_none() {
        return;
    }
    // Within a RecordBatch all columns share length (Arrow invariant).
    for i in 0..classes.len() {
        if classes.is_null(i) {
            continue;
        }
        let score = if let Some(a) = scores_f64 {
            if a.is_null(i) {
                continue;
            }
            a.value(i)
        } else {
            let a = scores_f32.expect("checked non-none above");
            if a.is_null(i) {
                continue;
            }
            a.value(i) as f64
        };
        // `Welford::observe` drops non-finite scores.
        out.entry(classes.value(i).to_string())
            .or_default()
            .observe(score);
    }
}

/// Process a single row group of one parquet file into its own `FileColStats`
/// (flushing category counts into the shared `Arc<TableCategoryCounts>`) plus a
/// thread-local by-class map when `score_class` is set.
fn stats_for_row_group(
    path: &Path,
    row_group: usize,
    columns: &[ColumnSpec],
    table_cat: &Arc<TableCategoryCounts>,
    score_class: Option<(usize, usize)>,
) -> Result<(FileColStats, ByClassStats)> {
    let mut fcs = FileColStats::new_with_categories(columns, Arc::clone(table_cat));
    let mut by_class = ByClassStats::new();
    let file = File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_row_groups(vec![row_group])
        .build()?;
    for batch_res in reader {
        let batch = batch_res?;
        for (i, col) in columns.iter().enumerate() {
            drive_from_arrow_column(i, batch.column(i).as_ref(), col, &mut fcs);
        }
        if let Some((si, ci)) = score_class {
            accumulate_by_class(&batch, si, ci, &mut by_class);
        }
    }
    // Push this worker's local category counts into the shared (concurrent)
    // state — the same pattern ingest uses from its parallel file workers.
    fcs.flush_categories();
    Ok((fcs, by_class))
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

    // SCORE-by-CLASS detection (multiclass/multilabel/ranking/link-pred only).
    let score_class = find_score_class(columns);

    let (mut acc, by_class) = if work.is_empty() {
        // Empty / no-row-group input: still emit a zero-row stats payload.
        let mut fcs = FileColStats::new_with_categories(columns, Arc::clone(&table_cat));
        fcs.flush_categories();
        (fcs, ByClassStats::new())
    } else {
        // Parallelize across row groups; each yields a FileColStats (whose
        // category counts are already flushed to the shared `table_cat`) plus a
        // thread-local by-class map.
        let partials: Vec<(FileColStats, ByClassStats)> = work
            .par_iter()
            .map(|&(pi, rg)| {
                stats_for_row_group(parquet_paths[pi], rg, columns, &table_cat, score_class)
            })
            .collect::<Result<Vec<_>>>()?;

        // Merge the per-worker scalar accumulators (welford / hll / tdigest /
        // counts) and the by-class maps. Category counts already live in the
        // shared `table_cat`.
        let mut iter = partials.into_iter();
        let (mut acc, mut by_class) = iter.next().expect("work is non-empty");
        for (mut p, pbc) in iter {
            acc.merge(&mut p);
            for (cls, w) in pbc {
                by_class.entry(cls).or_default().merge(&w);
            }
        }
        (acc, by_class)
    };

    // Single serialization: drains the shared category counts (destructive),
    // so we build the full map once and derive min from it in-memory.
    let mut full = acc.to_kumo_json(&names, &types, Some(&table_cat), None, data_version);
    let mut min = minify_kumo_stats(&full, MIN_COL_STATS_CATEGORY_LIMIT);
    // SUMMARY_STATS_BY_CLASS goes into BOTH variants. `BP_MULTICLASS_CLASSIFICATION`
    // maps to the MIN refresh stage on the kumo side, so the `/prediction_stats`
    // cache-fallback reads the min variant while the primary per-job path reads
    // the full one — both must carry it. It's EXACT_ONLY (can't be truncated like
    // CATEGORY_COUNTS), so the identical entry is written to each.
    if let Some((key, entry)) =
        build_summary_stats_by_class(columns, score_class, &by_class, data_version)
    {
        full.insert(key.clone(), entry.clone());
        min.insert(key, entry);
    }
    Ok((to_json_string(&full)?, to_json_string(&min)?))
}

/// Build the `SUMMARY_STATS_BY_CLASS` entry for the SCORE column —
/// `{class: {min, max, mean}}` (SCORE grouped by CLASS) — as a `(key, entry)`
/// pair, or `None` when there's no SCORE/CLASS pair or no observed values. This
/// cross-column stat can't be produced by `FileColStats`/`to_kumo_json` (they're
/// strictly per-column), so it's assembled here from the merged by-class map and
/// inserted into both the full and min variants by the caller.
fn build_summary_stats_by_class(
    columns: &[ColumnSpec],
    score_class: Option<(usize, usize)>,
    by_class: &ByClassStats,
    data_version: &str,
) -> Option<(String, JsonValue)> {
    let (score_idx, _class_idx) = score_class?;
    let mut total: u64 = 0;
    let mut classes = JsonMap::new();
    for (cls, w) in by_class {
        if w.count == 0 {
            continue;
        }
        total += w.count;
        classes.insert(
            cls.clone(),
            json!({"min": w.min, "max": w.max, "mean": w.mean()}),
        );
    }
    if classes.is_empty() {
        return None;
    }
    let score_name = &columns[score_idx].name;
    Some((
        kumo_key(score_name, "kumo.numerical", "SUMMARY_STATS_BY_CLASS", false),
        kumo_entry(JsonValue::Object(classes), false, data_version, total),
    ))
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
        let class: Vec<Option<&str>> = (0..n).map(|i| Some(["A", "B", "C"][i % 3])).collect();
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
        let (full, min) = compute_parquet_col_stats(&[path.as_path()], &columns, "v1").unwrap();

        // COUNT == n across columns (merge across row groups correct).
        assert_eq!(val(&full, "entity__kumo.ID__COUNT__False"), n as u64);
        assert_eq!(val(&full, "SCORE__kumo.numerical__COUNT__False"), n as u64);
        assert_eq!(
            val(&full, "CLASS__kumo.categorical__COUNT__False"),
            n as u64
        );

        // SCORE min/max exact across the whole file.
        let mn = val(&full, "SCORE__kumo.numerical__MIN__False")
            .as_f64()
            .unwrap();
        let mx = val(&full, "SCORE__kumo.numerical__MAX__False")
            .as_f64()
            .unwrap();
        assert!((mn - 0.0).abs() < 1e-9, "min={mn}");
        assert!((mx - 0.999).abs() < 1e-9, "max={mx}");

        // CLASS num_unique == 3 EXACT (Hybrid map complete, overrides HLL).
        assert_eq!(
            val(&full, "CLASS__kumo.categorical__NUM_UNIQUE__False"),
            3u64
        );

        // entity num_unique == n EXACT (1000 distinct, well under the 1M cap).
        assert_eq!(val(&full, "entity__kumo.ID__NUM_UNIQUE__False"), n as u64);

        // CLASS category counts populated (new_with_categories).
        let mfull: JsonMap<String, JsonValue> = serde_json::from_str(&full).unwrap();
        assert!(mfull.contains_key("CLASS__kumo.categorical__CATEGORY_COUNTS__False"));

        // SUMMARY_STATS_BY_CLASS: SCORE grouped by CLASS, merged correctly
        // across the ~8 row groups. Class A = {i : i%3==0}, score=i/1000.
        let bc = val(
            &full,
            "SCORE__kumo.numerical__SUMMARY_STATS_BY_CLASS__False",
        );
        assert!((bc["A"]["min"].as_f64().unwrap() - 0.0).abs() < 1e-9);
        assert!((bc["A"]["max"].as_f64().unwrap() - 0.999).abs() < 1e-9);
        assert!((bc["A"]["mean"].as_f64().unwrap() - 0.4995).abs() < 1e-6);
        // Three classes, no more.
        assert_eq!(bc.as_object().unwrap().len(), 3);

        // min variant parses and keeps the scalar COUNT.
        let mmin: JsonMap<String, JsonValue> = serde_json::from_str(&min).unwrap();
        assert_eq!(
            mmin["CLASS__kumo.categorical__COUNT__False"]["value"],
            1000u64
        );
        // by-class lives in BOTH variants: BP_MULTICLASS_CLASSIFICATION maps to
        // the MIN refresh stage, so the cache-fallback read path reads min.
        assert!(mmin.contains_key("SCORE__kumo.numerical__SUMMARY_STATS_BY_CLASS__False"));
        assert_eq!(
            mmin["SCORE__kumo.numerical__SUMMARY_STATS_BY_CLASS__False"]["value"],
            mfull["SCORE__kumo.numerical__SUMMARY_STATS_BY_CLASS__False"]["value"],
        );
    }

    /// Float32 SCORE must still produce numerical + by-class stats (model
    /// scores are frequently f32 — must not silently zero out).
    #[test]
    fn float32_score_by_class() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("predictions.parquet");
        let schema = Schema::new(vec![
            Field::new("SCORE", DataType::Float32, true),
            Field::new("CLASS", DataType::Utf8, true),
        ]);
        write_parquet(
            &path,
            schema,
            vec![
                StdArc::new(arrow_array::Float32Array::from(vec![
                    Some(1.0f32),
                    Some(3.0),
                    Some(5.0),
                ])) as ArrayRef,
                StdArc::new(StringArray::from(vec![Some("A"), Some("A"), Some("B")])) as ArrayRef,
            ],
            1024,
        );
        let columns = vec![
            col("SCORE", IngestedType::Float32, SemanticType::Numerical),
            col("CLASS", IngestedType::String, SemanticType::Categorical),
        ];
        let (full, _min) = compute_parquet_col_stats(&[path.as_path()], &columns, "v1").unwrap();
        // Numerical scalar stats populated (not empty-Welford defaults).
        let mx = val(&full, "SCORE__kumo.numerical__MAX__False")
            .as_f64()
            .unwrap();
        assert!((mx - 5.0).abs() < 1e-6, "max={mx}");
        let bc = val(&full, "SCORE__kumo.numerical__SUMMARY_STATS_BY_CLASS__False");
        assert!((bc["A"]["mean"].as_f64().unwrap() - 2.0).abs() < 1e-6);
        assert!((bc["B"]["min"].as_f64().unwrap() - 5.0).abs() < 1e-6);
    }

    /// Null SCORE / null CLASS rows are skipped; remaining valid pairs still
    /// produce correct per-class stats and `num_entries`.
    #[test]
    fn by_class_skips_nulls() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("predictions.parquet");
        let schema = Schema::new(vec![
            Field::new("SCORE", DataType::Float64, true),
            Field::new("CLASS", DataType::Utf8, true),
        ]);
        write_parquet(
            &path,
            schema,
            vec![
                // (A,2.0) (A,null-score→skip) (null-class,9.0→skip) (B,4.0)
                StdArc::new(Float64Array::from(vec![Some(2.0), None, Some(9.0), Some(4.0)]))
                    as ArrayRef,
                StdArc::new(StringArray::from(vec![Some("A"), Some("A"), None, Some("B")]))
                    as ArrayRef,
            ],
            1024,
        );
        let columns = vec![
            col("SCORE", IngestedType::Float64, SemanticType::Numerical),
            col("CLASS", IngestedType::String, SemanticType::Categorical),
        ];
        let (full, _min) = compute_parquet_col_stats(&[path.as_path()], &columns, "v1").unwrap();
        let bc = val(&full, "SCORE__kumo.numerical__SUMMARY_STATS_BY_CLASS__False");
        assert!((bc["A"]["mean"].as_f64().unwrap() - 2.0).abs() < 1e-9);
        assert!((bc["B"]["mean"].as_f64().unwrap() - 4.0).abs() < 1e-9);
        // null-CLASS row (score 9.0) contributed to no bucket.
        assert_eq!(bc.as_object().unwrap().len(), 2);
        let mfull: JsonMap<String, JsonValue> = serde_json::from_str(&full).unwrap();
        assert_eq!(
            mfull["SCORE__kumo.numerical__SUMMARY_STATS_BY_CLASS__False"]["num_entries"],
            2u64,
        );
    }

    /// Exact per-class min/max/mean for a hand-checked tiny input.
    #[test]
    fn summary_stats_by_class_exact() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("predictions.parquet");
        let schema = Schema::new(vec![
            Field::new("SCORE", DataType::Float64, true),
            Field::new("CLASS", DataType::Utf8, true),
        ]);
        write_parquet(
            &path,
            schema,
            vec![
                StdArc::new(Float64Array::from(vec![Some(1.0), Some(3.0), Some(5.0)])) as ArrayRef,
                StdArc::new(StringArray::from(vec![Some("A"), Some("A"), Some("B")])) as ArrayRef,
            ],
            1024,
        );
        let columns = vec![
            col("SCORE", IngestedType::Float64, SemanticType::Numerical),
            col("CLASS", IngestedType::String, SemanticType::Categorical),
        ];
        let (full, _min) = compute_parquet_col_stats(&[path.as_path()], &columns, "v1").unwrap();
        let bc = val(
            &full,
            "SCORE__kumo.numerical__SUMMARY_STATS_BY_CLASS__False",
        );
        // A: [1,3] -> min 1, max 3, mean 2 ; B: [5] -> min/max/mean 5.
        assert!((bc["A"]["min"].as_f64().unwrap() - 1.0).abs() < 1e-9);
        assert!((bc["A"]["max"].as_f64().unwrap() - 3.0).abs() < 1e-9);
        assert!((bc["A"]["mean"].as_f64().unwrap() - 2.0).abs() < 1e-9);
        assert!((bc["B"]["min"].as_f64().unwrap() - 5.0).abs() < 1e-9);
        assert!((bc["B"]["max"].as_f64().unwrap() - 5.0).abs() < 1e-9);
        assert!((bc["B"]["mean"].as_f64().unwrap() - 5.0).abs() < 1e-9);
    }

    /// No CLASS column (e.g. regression/binary BP) → no by-class stat emitted.
    #[test]
    fn no_summary_stats_by_class_without_class_column() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("predictions.parquet");
        let schema = Schema::new(vec![Field::new("TARGET_PRED", DataType::Float64, true)]);
        write_parquet(
            &path,
            schema,
            vec![StdArc::new(Float64Array::from(vec![Some(1.0), Some(2.0)])) as ArrayRef],
            1024,
        );
        let columns = vec![col(
            "TARGET_PRED",
            IngestedType::Float64,
            SemanticType::Numerical,
        )];
        let (full, _min) = compute_parquet_col_stats(&[path.as_path()], &columns, "v1").unwrap();
        let mfull: JsonMap<String, JsonValue> = serde_json::from_str(&full).unwrap();
        assert!(!mfull.keys().any(|k| k.contains("SUMMARY_STATS_BY_CLASS")));
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
        let mx = val(&full, "v__kumo.numerical__MAX__False")
            .as_f64()
            .unwrap();
        assert!((mx - 6.0).abs() < 1e-9);
    }

    /// Build a ColumnSpec with an explicit `timestamp_unit`.
    fn ts_col(name: &str, unit: Option<TimestampUnit>) -> ColumnSpec {
        ColumnSpec {
            timestamp_unit: unit,
            ..col(name, IngestedType::Timestamp, SemanticType::Timestamp)
        }
    }

    /// Int64 epoch-ms timestamp scales to seconds (inferred and explicit unit).
    #[test]
    fn int64_epoch_ms_timestamp_scales_to_correct_year() {
        let dir = tempdir().unwrap();
        // Two equal-instant rows so MIN == MAX and the assertion is exact.
        let raw_ms = vec![1_659_304_801_388_i64, 1_659_304_801_388_i64];
        let expected = "kumo_timestamp_2022-07-31 22:00:02";

        for (label, unit) in [("inferred", None), ("explicit", Some(TimestampUnit::Millis))] {
            let path = dir.path().join(format!("preds_{label}.parquet"));
            let schema = Schema::new(vec![Field::new("ts", DataType::Int64, true)]);
            write_parquet(
                &path,
                schema,
                vec![StdArc::new(Int64Array::from(raw_ms.clone())) as ArrayRef],
                64,
            );
            let columns = vec![ts_col("ts", unit)];
            let (full, _min) =
                compute_parquet_col_stats(&[path.as_path()], &columns, "v1").unwrap();
            assert_eq!(
                val(&full, "ts__kumo.timestamp__MIN__False"),
                expected,
                "MIN wrong for {label} unit"
            );
            assert_eq!(
                val(&full, "ts__kumo.timestamp__MAX__False"),
                expected,
                "MAX wrong for {label} unit"
            );
        }
    }

    /// Int64 epoch-seconds (no explicit unit) is left as-is, not over-scaled.
    #[test]
    fn int64_epoch_seconds_timestamp_unaffected() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("preds.parquet");
        let raw_s = vec![1_659_304_801_i64, 1_659_304_801_i64];
        let schema = Schema::new(vec![Field::new("ts", DataType::Int64, true)]);
        write_parquet(
            &path,
            schema,
            vec![StdArc::new(Int64Array::from(raw_s)) as ArrayRef],
            64,
        );
        let columns = vec![ts_col("ts", None)];
        let (full, _min) = compute_parquet_col_stats(&[path.as_path()], &columns, "v1").unwrap();
        assert_eq!(
            val(&full, "ts__kumo.timestamp__MIN__False"),
            "kumo_timestamp_2022-07-31 22:00:01"
        );
    }

    /// Arrow Timestamp(ns) stays on the /1e9 path, not the Int64 unit logic.
    #[test]
    fn arrow_nanosecond_timestamp_unaffected() {
        use arrow_array::TimestampNanosecondArray;
        let dir = tempdir().unwrap();
        let path = dir.path().join("preds.parquet");
        // 1659304801 s expressed in ns.
        let raw_ns = vec![1_659_304_801_000_000_000_i64, 1_659_304_801_000_000_000_i64];
        let schema = Schema::new(vec![Field::new(
            "ts",
            DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, None),
            true,
        )]);
        write_parquet(
            &path,
            schema,
            vec![StdArc::new(TimestampNanosecondArray::from(raw_ns)) as ArrayRef],
            64,
        );
        let columns = vec![ts_col("ts", None)];
        let (full, _min) = compute_parquet_col_stats(&[path.as_path()], &columns, "v1").unwrap();
        assert_eq!(
            val(&full, "ts__kumo.timestamp__MIN__False"),
            "kumo_timestamp_2022-07-31 22:00:01"
        );
    }
}
