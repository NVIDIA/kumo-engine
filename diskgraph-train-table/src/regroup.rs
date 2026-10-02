// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Empty-target row drop for array-form Link Prediction tasks.
//!
//! The fused pass-B router (`crate::route` + `crate::pass_b`) now performs
//! the per-link stage assignment and `(row, stage)` regrouping directly,
//! writing per-stage multi-part parquets. The standalone regroup PASS that
//! used to live here (explode → per-link stage hash → regroup into an inline
//! `__internal_kumo_split_id_col__` column) is gone.
//!
//! What remains is the split-kind-AGNOSTIC empty-target drop used for
//! array-LP (`PredictionType::Link`) on the no-split (prediction) path.
//! Split LP rows are handled implicitly by the fused pass-B router (an
//! empty-target anchor row produces zero sub-batches and is simply not
//! written). Only the no-split prediction path can reach this module.

use std::fs::File;
use std::path::Path;

#[cfg(test)]
use arrow_array::RecordBatch;
use arrow_array::{Array, BooleanArray, ListArray, RecordBatchReader};
use arrow_schema::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;

use crate::errors::{TTGenError, TTGenResult};
use crate::parquet_dataset::ordered_tt_dataset_parts;

/// Drop empty-target anchor rows from an array-LP `tt.parquet` in place,
/// returning the per-row keep mask (`keep[i] == true` ⇔ row `i` survived).
///
/// Used only on the no-split (prediction) LP path. Split LP is handled
/// implicitly by the fused pass-B router — an anchor with an empty TARGET
/// list produces zero sub-batches and is never written to any stage part.
/// The no-split path writes the `tt.parquet` dataset before routing, so
/// empty-target rows must be pruned here after pass A completes.
///
/// `TARGET1` must be a `List`-typed column. Returns the keep mask (length =
/// original row count). The rewritten dataset has `keep.iter().filter(|k|
/// **k).count()` rows. Empty input → empty mask, dataset left as-is.
///
/// MUST only be called for LP array tasks — non-LP array tasks (e.g.
/// multilabel/forecasting) may legitimately carry empty list cells.
pub(crate) fn drop_empty_target_rows_in_place(tt_parquet_path: &Path) -> TTGenResult<Vec<bool>> {
    use arrow_select::filter::filter_record_batch;

    let mut keep: Vec<bool> = Vec::new();
    for part_path in ordered_tt_dataset_parts(tt_parquet_path)? {
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&part_path)?)?.build()?;
        let schema = reader.schema();
    let target_idx = schema.index_of("TARGET1").map_err(|_| {
        TTGenError::SystemError("drop_empty_target_rows: requires TARGET1 column".to_string())
    })?;
    if !matches!(schema.field(target_idx).data_type(), DataType::List(_)) {
        return Err(TTGenError::SystemError(
            "drop_empty_target_rows: TARGET1 must be List type".to_string(),
        ));
    }

        let tmp_path = part_path.with_extension("parquet.tmp");
        let mut writer = ArrowWriter::try_new(File::create(&tmp_path)?, schema.clone(), None)?;
        for batch_res in reader {
            let batch = batch_res.map_err(|e| {
                TTGenError::Parquet(format!("drop_empty_target_rows: reading tt.parquet: {e}"))
            })?;
        let target_list = batch
            .column(target_idx)
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| {
                TTGenError::SystemError(
                    "drop_empty_target_rows: TARGET1 must be ListArray".to_string(),
                )
            })?;
        let offsets = target_list.value_offsets();
        let mask_vals: Vec<Option<bool>> = (0..batch.num_rows())
            .map(|i| {
                    let non_empty = !target_list.is_null(i) && (offsets[i + 1] - offsets[i]) > 0;
                keep.push(non_empty);
                Some(non_empty)
            })
            .collect();
            let filtered =
                filter_record_batch(&batch, &BooleanArray::from(mask_vals)).map_err(|e| {
            TTGenError::SystemError(format!("drop_empty_target_rows: filter failed: {e}"))
        })?;
            if filtered.num_rows() > 0 {
                writer.write(&filtered)?;
        }
    }
    writer.close()?;
        std::fs::rename(tmp_path, part_path)?;
    }

    Ok(keep)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::type_complexity)]
mod tests {
    use super::*;
    use arrow_array::builder::{
        Int64Builder, ListBuilder, StringBuilder, TimestampNanosecondBuilder,
    };
    use arrow_array::Int64Array;
    use arrow_schema::{Field, Schema, TimeUnit};
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc as StdArc;

    /// Synthesize a tiny LP-array tt.parquet at `path`:
    /// columns = (ENTITY1:Int64, TIMESTAMP:Timestamp(ns), TARGET1:List<Utf8>,
    ///            __kumo_internal_frequency_column__:List<Int64>).
    /// Each row has the given list of (target, freq) pairs.
    fn write_lp_array_parquet(path: &Path, rows: &[(i64, i64, Vec<(&str, i64)>)]) {
        std::fs::create_dir(path).unwrap();
        let mut entity = Int64Builder::new();
        let mut ts_b = TimestampNanosecondBuilder::new();
        let mut target_lb = ListBuilder::new(StringBuilder::new());
        let mut freq_lb = ListBuilder::new(Int64Builder::new());
        for (e, t, items) in rows {
            entity.append_value(*e);
            ts_b.append_value(*t * 1_000_000_000);
            for (s, f) in items {
                target_lb.values().append_value(*s);
                freq_lb.values().append_value(*f);
            }
            target_lb.append(true);
            freq_lb.append(true);
        }
        let schema = StdArc::new(Schema::new(vec![
            Field::new("ENTITY1", DataType::Int64, false),
            Field::new(
                "TIMESTAMP",
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                false,
            ),
            Field::new(
                "TARGET1",
                DataType::List(StdArc::new(Field::new("item", DataType::Utf8, true))),
                true,
            ),
            Field::new(
                "__kumo_internal_frequency_column__",
                DataType::List(StdArc::new(Field::new("item", DataType::Int64, true))),
                false,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                StdArc::new(entity.finish()),
                StdArc::new(ts_b.finish()),
                StdArc::new(target_lb.finish()),
                StdArc::new(freq_lb.finish()),
            ],
        )
        .unwrap();
        let f = File::create(path.join(crate::parquet_dataset::part_name(0))).unwrap();
        let mut w = ArrowWriter::try_new(f, schema, None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    fn read_back(path: &Path) -> Vec<RecordBatch> {
        crate::parquet_dataset::ordered_tt_dataset_parts(path)
            .unwrap()
            .into_iter()
            .flat_map(|part| {
                ParquetRecordBatchReaderBuilder::try_new(File::open(part).unwrap())
            .unwrap()
            .build()
                    .unwrap()
                    .map(|batch| batch.unwrap())
            })
            .collect()
    }

    // ── Empty-target drop (split-kind-agnostic; temporal-LP path) ──────────

    /// Helper: read TARGET1 list lengths per surviving row from a parquet.
    fn read_target_list_lens(path: &Path) -> Vec<usize> {
        let batches = read_back(path);
        let mut lens = Vec::new();
        for b in &batches {
            let ti = b.schema().index_of("TARGET1").unwrap();
            let tl = b.column(ti).as_any().downcast_ref::<ListArray>().unwrap();
            for i in 0..b.num_rows() {
                let s = tl.value_offsets()[i] as usize;
                let e = tl.value_offsets()[i + 1] as usize;
                lens.push(e - s);
            }
        }
        lens
    }

    #[test]
    fn test_drop_empty_target_rows_drops_only_empties() {
        // Mixed rows: non-empty, EMPTY, non-empty, EMPTY, non-empty.
        // Mirrors the array-LP + DateOffsetSplit case where most anchors'
        // LIST_DISTINCT cell is empty after the temporal cut.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tt.parquet");
        write_lp_array_parquet(
            &path,
            &[
                (1_i64, 1000, vec![("a", 1), ("b", 2)]),
                (2, 1000, vec![]), // empty target → must be dropped
                (3, 1000, vec![("c", 1)]),
                (4, 1000, vec![]), // empty target → must be dropped
                (5, 1000, vec![("d", 1), ("e", 1), ("f", 1)]),
            ],
        );

        let keep = drop_empty_target_rows_in_place(&path).unwrap();
        assert_eq!(keep, vec![true, false, true, false, true]);

        // Rewritten parquet: 3 surviving rows, none empty.
        let lens = read_target_list_lens(&path);
        assert_eq!(lens, vec![2, 1, 3]);
        // ENTITY1 of survivors preserved in order: 1, 3, 5.
        let batches = read_back(&path);
        let mut ents = Vec::new();
        for b in &batches {
            let ei = b.schema().index_of("ENTITY1").unwrap();
            let ea = b.column(ei).as_any().downcast_ref::<Int64Array>().unwrap();
            for i in 0..b.num_rows() {
                ents.push(ea.value(i));
            }
        }
        assert_eq!(ents, vec![1, 3, 5]);
    }

    #[test]
    fn test_drop_empty_target_rows_noop_when_all_nonempty() {
        // When every row has a non-empty target (e.g. legacy anchor table),
        // the drop is a no-op: all rows survive in order.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tt.parquet");
        write_lp_array_parquet(
            &path,
            &[
                (1_i64, 1000, vec![("a", 1)]),
                (2, 1000, vec![("b", 1), ("c", 1)]),
            ],
        );
        let keep = drop_empty_target_rows_in_place(&path).unwrap();
        assert_eq!(keep, vec![true, true]);
        assert_eq!(read_target_list_lens(&path), vec![1, 2]);
    }

    #[test]
    fn test_drop_empty_target_rows_all_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tt.parquet");
        write_lp_array_parquet(&path, &[(1_i64, 1000, vec![]), (2, 1000, vec![])]);
        let keep = drop_empty_target_rows_in_place(&path).unwrap();
        assert_eq!(keep, vec![false, false]);
        assert_eq!(read_target_list_lens(&path), Vec::<usize>::new());
    }
}
