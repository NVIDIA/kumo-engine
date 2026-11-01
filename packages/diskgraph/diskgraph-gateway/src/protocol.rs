// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native Flight schema changes and per-component metadata.
use arrow_schema::Schema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, sync::Arc};

pub const VERSION: u32 = 7;
pub(crate) const STREAM_METADATA_KEY: &str = "kumo.sampling";
pub(crate) const MAX_BATCH_BYTES: usize = 64 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StreamMetadata {
    pub version: u32,
    pub num_samples: u64,
    pub num_batches: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ComponentMetadata {
    pub batch_id: u64,
    pub component_id: usize,
    pub num_components: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<Value>,
}

pub(crate) fn stream_schema(num_samples: u64, num_batches: u64) -> Arc<Schema> {
    let metadata = StreamMetadata {
        version: VERSION,
        num_samples,
        num_batches,
    };
    Arc::new(Schema::empty().with_metadata(HashMap::from([(
        STREAM_METADATA_KEY.into(),
        serde_json::to_string(&metadata).unwrap(),
    )])))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stream_schema_carries_counts_without_fake_data_columns() {
        let result = stream_schema(17, 3);
        assert!(result.fields().is_empty());
        let metadata: StreamMetadata =
            serde_json::from_str(&result.metadata()[STREAM_METADATA_KEY]).unwrap();
        assert_eq!(metadata.version, VERSION);
        assert_eq!(metadata.num_samples, 17);
        assert_eq!(metadata.num_batches, 3);
    }
}
