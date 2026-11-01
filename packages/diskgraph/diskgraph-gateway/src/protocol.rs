// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Versioned Arrow envelopes shared with the Python Flight client.
use arrow_array::{
    builder::{BinaryBuilder, ListBuilder},
    BinaryArray, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use serde_json::Value;
use std::sync::Arc;
use tonic::Status;

pub(crate) const VERSION: u32 = 5;
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn rejects_payloads_that_exceed_the_frame_budget() {
        let error = envelope(json!({"kind":"batch"}), vec![vec![0; MAX_BATCH_BYTES]]).unwrap_err();
        assert_eq!(error.code(), tonic::Code::ResourceExhausted);
    }
}
