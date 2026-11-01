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
use std::{collections::HashMap, sync::Arc};
use tonic::Status;

pub(crate) const VERSION: u32 = 6;
pub(crate) const STREAM_METADATA_KEY: &str = "kumo.sampling";
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

pub(crate) fn stream_schema(num_samples: u64, num_batches: u64) -> Arc<Schema> {
    let metadata = serde_json::json!({
        "version": VERSION, "num_samples": num_samples, "num_batches": num_batches
    });
    Arc::new(schema().as_ref().clone().with_metadata(HashMap::from([(
        STREAM_METADATA_KEY.into(),
        metadata.to_string(),
    )])))
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
    fn stream_schema_carries_counts_without_changing_batch_fields() {
        let result = stream_schema(17, 3);
        assert_eq!(result.fields(), schema().fields());
        let metadata: Value =
            serde_json::from_str(&result.metadata()[STREAM_METADATA_KEY]).unwrap();
        assert_eq!(
            metadata,
            json!({"version":VERSION,"num_samples":17,"num_batches":3})
        );
        let empty = stream_schema(0, 0);
        let metadata: Value = serde_json::from_str(&empty.metadata()[STREAM_METADATA_KEY]).unwrap();
        assert_eq!(metadata["num_batches"], 0);
    }
    #[test]
    fn rejects_payloads_that_exceed_the_frame_budget() {
        let error = envelope(json!({"kind":"batch"}), vec![vec![0; MAX_BATCH_BYTES]]).unwrap_err();
        assert_eq!(error.code(), tonic::Code::ResourceExhausted);
    }
}
