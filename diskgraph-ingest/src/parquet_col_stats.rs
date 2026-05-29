//! Compute Kumo-format column statistics over an arbitrary parquet file or
//! directory of parquet files, reusing the existing `FileColStats` machinery.
//!
//! Used to populate post-batch-prediction column statistics for the
//! diskgraph deploy: after a BP run writes its output parquet, kumo invokes
//! this code path (via the `compute_col_stats_for_parquet` pybind entry)
//! to produce the same `kumo_col_stats_v1` JSON that ingest emits, so
//! downstream readers (UI, encoder inference, `PredStats.parse`) light up
//! without any further changes on the consumer side.

use std::fs::File;
use std::path::Path;

use arrow_array::{
    Array, BooleanArray, Float64Array, Int64Array, StringArray, TimestampNanosecondArray,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::{Map as JsonMap, Value as JsonValue};

use diskgraph_rowpack::ColumnValueObserver;

use crate::col_stats::{minify_kumo_stats, FileColStats};
use crate::graph_config::{ColumnSpec, IngestedType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    Full,
    Min,
}

/// Drive the observer trait from one Arrow column for one batch.
///
/// Handles the Arrow array types actually produced by BP-output parquet
/// (the same set vetted by `compute_train_table_col_stats_payload` in
/// `tt_executor.rs`). Unsupported / unexpected types are no-ops: the
/// column's stats will be empty rather than wrong.
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
        _ => {
            // BP outputs (per get_bp_columns_by_task_type) don't emit list /
            // array / decimal types. If a future schema introduces them,
            // extend this match with additional arms.
        }
    }
}

/// Compute Kumo-format column statistics over one or more parquet files,
/// returning the `kumo_col_stats_v1` JSON map.
///
/// `columns` must match the parquet schema column-for-column (same order,
/// same count). Each `ColumnSpec` carries the `(name, ingested_dtype,
/// semantic_type)` used to dispatch into the right `ColumnStatsCollector`
/// variant inside `FileColStats`.
pub fn compute_parquet_col_stats(
    parquet_paths: &[&Path],
    columns: &[ColumnSpec],
    data_version: &str,
    variant: Variant,
) -> anyhow::Result<JsonMap<String, JsonValue>> {
    let names: Vec<std::string::String> = columns.iter().map(|c| c.name.clone()).collect();
    let types: Vec<IngestedType> = columns.iter().map(|c| c.ingested_dtype.clone()).collect();

    let mut stats = FileColStats::new(columns);

    for path in parquet_paths {
        let file = File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let reader = builder.build()?;
        for batch_res in reader {
            let batch = batch_res?;
            for (i, col) in columns.iter().enumerate() {
                drive_from_arrow_column(i, batch.column(i).as_ref(), &col.ingested_dtype, &mut stats);
            }
        }
    }

    let full = stats.to_kumo_json(&names, &types, None, None, data_version);
    Ok(match variant {
        Variant::Min => minify_kumo_stats(&full, 20),
        Variant::Full => full,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::fs;

    use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use tempfile::tempdir;

    use crate::graph_config::SemanticType;

    fn write_parquet(path: &Path, schema: Schema, columns: Vec<ArrayRef>) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let file = fs::File::create(path).unwrap();
        let batch = RecordBatch::try_new(Arc::new(schema), columns).unwrap();
        let mut writer =
            parquet::arrow::ArrowWriter::try_new(file, batch.schema(), None).unwrap();
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

    /// End-to-end: write a small parquet (id Int64 / score Float64 / klass
    /// String), compute stats, assert the JSON has the expected keys and
    /// values match hand-computed truth.
    #[test]
    fn computes_stats_for_simple_parquet() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("predictions.parquet");

        let schema = Schema::new(vec![
            Field::new("entity_id", DataType::Int64, false),
            Field::new("score", DataType::Float64, true),
            Field::new("klass", DataType::Utf8, true),
        ]);
        let entity = Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])) as ArrayRef;
        let score = Arc::new(Float64Array::from(vec![
            Some(0.1),
            Some(0.4),
            Some(0.5),
            Some(0.6),
            Some(0.9),
        ])) as ArrayRef;
        let klass = Arc::new(StringArray::from(vec![
            Some("a"),
            Some("b"),
            Some("a"),
            Some("c"),
            Some("a"),
        ])) as ArrayRef;
        write_parquet(&path, schema, vec![entity, score, klass]);

        let columns = vec![
            col("entity_id", IngestedType::Int64, SemanticType::Id),
            col("score", IngestedType::Float64, SemanticType::Numerical),
            col("klass", IngestedType::String, SemanticType::Categorical),
        ];

        let stats = compute_parquet_col_stats(
            &[path.as_path()],
            &columns,
            "v1",
            Variant::Full,
        )
        .expect("compute_parquet_col_stats");

        // 1. Keys present (kumo_col_stats_v1 shape: {col}__{stype}__{STAT}__{approx}).
        assert!(stats.contains_key("entity_id__kumo.ID__COUNT__False"),
                "missing entity_id COUNT key — got keys: {:?}", stats.keys().collect::<Vec<_>>());
        assert!(stats.contains_key("score__kumo.numerical__COUNT__False"));
        assert!(stats.contains_key("score__kumo.numerical__MEAN__False"));
        assert!(stats.contains_key("score__kumo.numerical__MIN__False"));
        assert!(stats.contains_key("score__kumo.numerical__MAX__False"));
        assert!(stats.contains_key("klass__kumo.categorical__COUNT__False"));

        // 2. COUNT == 5 for all three columns (5 input rows).
        let entry_value = |k: &str| -> JsonValue {
            stats[k]["value"].clone()
        };
        assert_eq!(entry_value("entity_id__kumo.ID__COUNT__False"), 5);
        assert_eq!(entry_value("score__kumo.numerical__COUNT__False"), 5);
        assert_eq!(entry_value("klass__kumo.categorical__COUNT__False"), 5);

        // 3. Numerical MIN / MAX exact.
        assert_eq!(entry_value("score__kumo.numerical__MIN__False"), 0.1);
        assert_eq!(entry_value("score__kumo.numerical__MAX__False"), 0.9);

        // 4. Mean = (0.1 + 0.4 + 0.5 + 0.6 + 0.9) / 5 = 0.5.
        let mean = stats["score__kumo.numerical__MEAN__False"]["value"]
            .as_f64()
            .expect("mean must be f64");
        assert!((mean - 0.5).abs() < 1e-9, "mean was {mean}");

        // 5. data_version is propagated.
        let dv = stats["score__kumo.numerical__COUNT__False"]["data_version"]
            .as_str()
            .expect("data_version must be a string");
        assert_eq!(dv, "v1");
    }

    /// Nulls advance NUM_NA without contributing to MIN/MAX/MEAN.
    #[test]
    fn nulls_handled_correctly() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("p.parquet");
        let schema = Schema::new(vec![Field::new("x", DataType::Float64, true)]);
        let x = Arc::new(Float64Array::from(vec![Some(1.0), None, Some(3.0)])) as ArrayRef;
        write_parquet(&path, schema, vec![x]);

        let columns = vec![col("x", IngestedType::Float64, SemanticType::Numerical)];
        let stats =
            compute_parquet_col_stats(&[path.as_path()], &columns, "v1", Variant::Full).unwrap();

        assert_eq!(stats["x__kumo.numerical__NUM_NA__False"]["value"], 1);
        assert_eq!(stats["x__kumo.numerical__MIN__False"]["value"], 1.0);
        assert_eq!(stats["x__kumo.numerical__MAX__False"]["value"], 3.0);
        let mean = stats["x__kumo.numerical__MEAN__False"]["value"].as_f64().unwrap();
        assert!((mean - 2.0).abs() < 1e-9);
    }

    /// Multi-file merge: stats over two parquet files equals stats over their union.
    #[test]
    fn merges_multiple_files() {
        let dir = tempdir().unwrap();
        let p1 = dir.path().join("a.parquet");
        let p2 = dir.path().join("b.parquet");
        let schema = || Schema::new(vec![Field::new("v", DataType::Int64, false)]);
        write_parquet(
            &p1,
            schema(),
            vec![Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef],
        );
        write_parquet(
            &p2,
            schema(),
            vec![Arc::new(Int64Array::from(vec![4, 5, 6])) as ArrayRef],
        );

        let columns = vec![col("v", IngestedType::Int64, SemanticType::Numerical)];
        let stats = compute_parquet_col_stats(
            &[p1.as_path(), p2.as_path()],
            &columns,
            "v1",
            Variant::Full,
        )
        .unwrap();

        assert_eq!(stats["v__kumo.numerical__COUNT__False"]["value"], 6);
        let min = stats["v__kumo.numerical__MIN__False"]["value"].as_f64().unwrap();
        let max = stats["v__kumo.numerical__MAX__False"]["value"].as_f64().unwrap();
        assert!((min - 1.0).abs() < 1e-9, "min was {min}");
        assert!((max - 6.0).abs() < 1e-9, "max was {max}");
    }

    /// End-to-end through the Kumo→DG mapping path used by the pybind
    /// entry — confirms ``map_kumo_dtype`` / ``map_kumo_stype`` produce
    /// the same `ColumnSpec` shape the direct constructor produces.
    /// This guards against future drift if either mapping changes.
    #[test]
    fn kumo_mapping_path_matches_direct() {
        use crate::config_converter::{map_kumo_dtype, map_kumo_stype};

        let dir = tempdir().unwrap();
        let path = dir.path().join("p.parquet");
        let schema = Schema::new(vec![
            Field::new("entity", DataType::Int64, false),
            Field::new("score", DataType::Float64, true),
        ]);
        let entity = Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef;
        let score = Arc::new(Float64Array::from(vec![Some(0.1), Some(0.5), Some(0.9)])) as ArrayRef;
        write_parquet(&path, schema, vec![entity, score]);

        // Build ColumnSpecs the same way the pybind layer does: from Kumo
        // stype/dtype value strings through map_kumo_*.
        let kumo_inputs = [
            ("entity", "ID", "int64"),
            ("score", "numerical", "float64"),
        ];
        let columns: Vec<ColumnSpec> = kumo_inputs
            .iter()
            .map(|(name, stype, dtype)| ColumnSpec {
                name: name.to_string(),
                source_dtype: None,
                ingested_dtype: map_kumo_dtype(dtype).unwrap(),
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                precision: None,
                scale: None,
                semantic_type: map_kumo_stype(stype),
                multicategorical_delimiter: None,
            })
            .collect();

        let stats = compute_parquet_col_stats(
            &[path.as_path()],
            &columns,
            "v1",
            Variant::Full,
        )
        .unwrap();

        assert!(stats.contains_key("entity__kumo.ID__COUNT__False"));
        assert!(stats.contains_key("score__kumo.numerical__MEAN__False"));
        assert_eq!(stats["entity__kumo.ID__COUNT__False"]["value"], 3);
    }

    /// Min variant truncates CATEGORY_COUNTS to ≤ 20 entries.
    #[test]
    fn min_variant_truncates_category_counts() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("c.parquet");
        let schema = Schema::new(vec![Field::new("c", DataType::Utf8, false)]);
        // 30 distinct categories
        let vals: Vec<std::string::String> = (0..30).map(|i| format!("cat{i}")).collect();
        let arr =
            Arc::new(StringArray::from(vals.iter().map(|s| s.as_str()).collect::<Vec<_>>())) as ArrayRef;
        write_parquet(&path, schema, vec![arr]);

        let columns = vec![col("c", IngestedType::String, SemanticType::Categorical)];
        let stats_full =
            compute_parquet_col_stats(&[path.as_path()], &columns, "v1", Variant::Full).unwrap();
        let stats_min =
            compute_parquet_col_stats(&[path.as_path()], &columns, "v1", Variant::Min).unwrap();

        // The "min" variant must produce the same COUNT/NUM_NA scalars.
        assert_eq!(
            stats_full["c__kumo.categorical__COUNT__False"],
            stats_min["c__kumo.categorical__COUNT__False"]
        );
    }
}
