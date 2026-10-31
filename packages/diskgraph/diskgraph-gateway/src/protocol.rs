// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Versioned Arrow envelopes shared with the Python Flight client.
use arrow_array::{
    builder::{BinaryBuilder, ListBuilder},
    Array, BinaryArray, ListArray, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use serde_json::Value;
use std::sync::Arc;
use tonic::Status;

pub(crate) const VERSION: u32 = 4;
pub(crate) const WINDOW: u64 = 2;
pub(crate) const CHUNK_BYTES: usize = 1024 * 1024;
pub(crate) const MAX_BATCH_BYTES: usize = 64 * 1024 * 1024;

pub(crate) fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("metadata", DataType::Binary, false),
        Field::new(
            "components",
            DataType::List(Arc::new(Field::new("item", DataType::Binary, true))),
            false,
        ),
    ]))
}

pub(crate) fn envelope(metadata: Value, parts: Vec<Vec<u8>>) -> Result<RecordBatch, Status> {
    let metadata = metadata.to_string();
    if metadata.len() + parts.iter().map(Vec::len).sum::<usize>() > MAX_BATCH_BYTES {
        return Err(Status::resource_exhausted(
            "Task batch exceeds 64 MiB; reduce batch size or fanout",
        ));
    }
    let mut builder = ListBuilder::new(BinaryBuilder::new());
    for part in parts {
        builder.values().append_value(part);
    }
    builder.append(true);
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(BinaryArray::from(vec![metadata.as_bytes()])),
            Arc::new(builder.finish()),
        ],
    )
    .map_err(|error| Status::internal(error.to_string()))
}

pub(crate) fn decode(batch: &RecordBatch) -> Result<(Value, Vec<&[u8]>), Status> {
    if batch.num_rows() != 1 || batch.schema() != schema() {
        return Err(Status::invalid_argument("Invalid task stream envelope"));
    }
    let metadata = batch
        .column(0)
        .as_any()
        .downcast_ref::<BinaryArray>()
        .unwrap();
    let parts = batch
        .column(1)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    let values = parts
        .values()
        .as_any()
        .downcast_ref::<BinaryArray>()
        .unwrap();
    if metadata.is_null(0) || parts.is_null(0) {
        return Err(Status::invalid_argument("Null task envelope"));
    }
    let offsets = parts.value_offsets();
    let mut result = Vec::new();
    let mut size = metadata.value(0).len();
    for index in offsets[0] as usize..offsets[1] as usize {
        if values.is_null(index) {
            return Err(Status::invalid_argument("Null task component"));
        }
        let value = values.value(index);
        size += value.len();
        result.push(value);
    }
    if size > MAX_BATCH_BYTES {
        return Err(Status::resource_exhausted("Task frame exceeds 64 MiB"));
    }
    let metadata: Value = serde_json::from_slice(metadata.value(0))
        .map_err(|error| Status::invalid_argument(error.to_string()))?;
    if !metadata.is_object() {
        return Err(Status::invalid_argument("Task metadata must be an object"));
    }
    Ok((metadata, result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn envelopes_preserve_metadata_and_borrow_binary_components() {
        let batch = envelope(json!({"kind":"data"}), vec![vec![1, 2, 3]]).unwrap();
        let (metadata, parts) = decode(&batch).unwrap();
        assert_eq!(metadata, json!({"kind":"data"}));
        assert_eq!(parts, vec![&[1, 2, 3][..]]);
    }
    #[test]
    fn rejects_wrong_shapes_and_non_object_metadata() {
        assert!(decode(&envelope(json!(3), vec![]).unwrap()).is_err());
        let batch = RecordBatch::new_empty(Arc::new(Schema::empty()));
        assert!(decode(&batch).is_err());
    }
}
