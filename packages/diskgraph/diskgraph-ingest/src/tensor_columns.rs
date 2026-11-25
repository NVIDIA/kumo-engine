// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::graph_config::{ColumnSpec, IngestedType};
use crate::PublicIngestError;
use anyhow::{anyhow, ensure, Result};
use arrow_array::{
    make_array, Array, ArrayRef, FixedSizeListArray, GenericListArray, LargeListArray, ListArray,
    OffsetSizeTrait, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

/// Check tensor rows before rowpack's ordinary invalid-as-null coercions. Only
/// fixed-list offsets and signed views of unsigned children need adaptation.
pub(super) fn prepare(table: &str, columns: &[ColumnSpec], batch: &mut RecordBatch) -> Result<()> {
    let schema = batch.schema();
    let mut replacement = None;
    for column in columns {
        let i = schema.index_of(&column.name)?;
        let input = batch.column(i);
        let width = column.fixed_len.filter(|w| *w != 0).ok_or_else(|| {
            anyhow!(
                "{table}.{}: tensor columns require a positive fixed_len",
                column.name
            )
        })?;
        let result = if let Some(rows) = input.as_any().downcast_ref::<ListArray>() {
            normalize(rows, width, column)
        } else if let Some(rows) = input.as_any().downcast_ref::<LargeListArray>() {
            normalize(rows, width, column)
        } else if let Some(rows) = input.as_any().downcast_ref::<FixedSizeListArray>() {
            let rows = LargeListArray::from(rows.clone());
            normalize(&rows, width, column)
                .map(|child| Some(child.unwrap_or_else(|| Arc::new(rows))))
        } else {
            Err(anyhow!(
                "expected a numeric list, got {}",
                input.data_type()
            ))
        }
        .map_err(|e| PublicIngestError::new(format!("{table}.{}: {e}", column.name)))?;
        if let Some(array) = result {
            let (arrays, fields) = replacement
                .get_or_insert_with(|| (batch.columns().to_vec(), schema.fields().to_vec()));
            fields[i] = Arc::new(
                fields[i]
                    .as_ref()
                    .clone()
                    .with_data_type(array.data_type().clone()),
            );
            arrays[i] = array;
        }
    }
    if let Some((arrays, fields)) = replacement {
        *batch = RecordBatch::try_new(
            Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone())),
            arrays,
        )?;
    }
    Ok(())
}

fn normalize<O: OffsetSizeTrait>(
    rows: &GenericListArray<O>,
    width: usize,
    column: &ColumnSpec,
) -> Result<Option<ArrayRef>> {
    let values = rows.values();
    let physical = values.data_type();
    let stored = match (&column.ingested_dtype, physical) {
        (IngestedType::Int8Array, DataType::UInt8 | DataType::Int8) => DataType::Int8,
        (IngestedType::Int16Array, DataType::UInt16 | DataType::Int16) => DataType::Int16,
        (IngestedType::Int32Array, DataType::UInt32 | DataType::Int32) => DataType::Int32,
        (IngestedType::Int64Array, DataType::UInt64 | DataType::Int64) => DataType::Int64,
        (IngestedType::Float32Array, DataType::Float32) => DataType::Float32,
        (IngestedType::Float64Array, DataType::Float64) => DataType::Float64,
        _ => {
            return Err(anyhow!(
                "unsupported tensor element type {physical} for {:?}",
                column.ingested_dtype
            ));
        }
    };
    let validity = values.nulls().filter(|v| v.null_count() != 0);
    for (row, offsets) in rows.value_offsets().windows(2).enumerate() {
        if rows.is_null(row) {
            continue;
        }
        let (start, end) = (offsets[0].as_usize(), offsets[1].as_usize());
        ensure!(
            end - start == width,
            "expected {width} elements, got {} in row {row}",
            end - start
        );
        if let Some(validity) = validity {
            ensure!(
                validity.inner().slice(start, width).count_set_bits() == width,
                "null element in tensor row {row}"
            );
        }
    }
    if *physical == stored {
        return Ok(None);
    }
    // Rowpack has signed integer list codecs. Their equal-width signed view
    // preserves every unsigned bit, including values above i64::MAX.
    let signed = make_array(
        values
            .to_data()
            .into_builder()
            .data_type(stored.clone())
            .build()?,
    );
    Ok(Some(Arc::new(GenericListArray::<O>::try_new(
        Arc::new(Field::new("item", stored, true)),
        rows.offsets().clone(),
        signed,
        rows.nulls().cloned(),
    )?)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::types::{Float32Type, UInt64Type};

    fn tensor(dtype: IngestedType) -> ColumnSpec {
        serde_json::from_value(serde_json::json!({
            "name": "vector", "ingested_dtype": dtype, "semantic_type": "tensor", "fixed_len": 2
        }))
        .unwrap()
    }

    #[test]
    fn flat_native_tensor_reuses_input_and_skips_null_rows() {
        let input = ListArray::from_iter_primitive::<Float32Type, _, _>([
            Some(vec![Some(1.0), Some(-3.0)]),
            None,
        ]);
        let keep: ArrayRef = Arc::new(input);
        let mut batch = RecordBatch::try_from_iter([("vector", keep.clone())]).unwrap();
        prepare("pins", &[tensor(IngestedType::Float32Array)], &mut batch).unwrap();
        assert!(Arc::ptr_eq(batch.column(0), &keep));
    }

    #[test]
    fn nullable_fixed_list_preserves_unsigned_bits_in_sliced_batches() {
        let input = FixedSizeListArray::from_iter_primitive::<UInt64Type, _, _>(
            [
                Some(vec![Some(11), Some(22)]),
                None,
                Some(vec![Some(u64::MAX), Some(1 << 63)]),
            ],
            2,
        );
        let sliced: ArrayRef = Arc::new(input.slice(1, 2));
        let mut batch = RecordBatch::try_from_iter([("vector", sliced)]).unwrap();
        prepare("pins", &[tensor(IngestedType::Int64Array)], &mut batch).unwrap();
        let out = batch
            .column(0)
            .as_any()
            .downcast_ref::<LargeListArray>()
            .unwrap();
        assert!(out.is_null(0));
        let row = out.value(1);
        assert_eq!(
            row.as_any()
                .downcast_ref::<arrow_array::Int64Array>()
                .unwrap()
                .values(),
            &[-1, i64::MIN]
        );
    }

    #[test]
    fn malformed_rows_raise_source_validation_errors_before_encoding() {
        let spec = tensor(IngestedType::Float32Array);
        for row in [vec![Some(3.0)], vec![None, Some(3.0)]] {
            let input = ListArray::from_iter_primitive::<Float32Type, _, _>([Some(row)]);
            let mut batch =
                RecordBatch::try_from_iter([("vector", Arc::new(input) as ArrayRef)]).unwrap();
            let error = prepare("pins", std::slice::from_ref(&spec), &mut batch).unwrap_err();
            assert!(error.is::<PublicIngestError>());
            assert!(error.to_string().contains("pins.vector"));
        }
    }
}
