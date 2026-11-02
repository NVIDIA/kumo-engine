// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stream, minibatch and component metadata carried by FlightData envelopes.
use arrow_flight::{FlightData, SchemaAsIpc};
use arrow_ipc::writer::IpcWriteOptions;
use arrow_schema::Schema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const VERSION: u32 = 9;
pub(crate) const MAX_BATCH_BYTES: usize = 64 * 1024 * 1024;

#[derive(Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ControlKind {
    Stream,
    Minibatch,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StreamMetadata {
    pub kind: ControlKind,
    pub version: u32,
    pub num_samples: u64,
    pub num_batches: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BatchMetadata {
    pub kind: ControlKind,
    pub batch_id: u64,
    pub num_components: usize,
    pub layout: Value,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ComponentMetadata {
    pub batch_id: u64,
    pub component_id: usize,
}

pub(crate) fn stream_header(num_samples: u64, num_batches: u64) -> FlightData {
    let mut message: FlightData =
        SchemaAsIpc::new(&Schema::empty(), &IpcWriteOptions::default()).into();
    message.app_metadata = serde_json::to_vec(&StreamMetadata {
        kind: ControlKind::Stream,
        version: VERSION,
        num_samples,
        num_batches,
    })
    .unwrap()
    .into();
    message
}

pub(crate) fn batch_header(metadata: &BatchMetadata) -> FlightData {
    // A valid IPC NONE header lets FlightDataDecoder surface metadata-only messages.
    let mut builder = Default::default();
    let message = arrow_ipc::Message::create(
        &mut builder,
        &arrow_ipc::MessageArgs {
            version: arrow_ipc::MetadataVersion::V5,
            ..Default::default()
        },
    );
    builder.finish(message, None);
    FlightData {
        data_header: builder.finished_data().to_vec().into(),
        app_metadata: serde_json::to_vec(metadata).unwrap().into(),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stream_header_carries_counts_in_application_metadata() {
        let message = stream_header(17, 3);
        let schema = Schema::try_from(&message).unwrap();
        assert!(schema.fields().is_empty());
        assert!(schema.metadata().is_empty());
        let metadata: StreamMetadata = serde_json::from_slice(&message.app_metadata).unwrap();
        assert_eq!(metadata.version, VERSION);
        assert_eq!(metadata.num_samples, 17);
        assert_eq!(metadata.num_batches, 3);
    }
    #[test]
    fn batch_header_has_metadata_without_arrow_data() {
        let message = batch_header(&BatchMetadata {
            kind: ControlKind::Minibatch,
            batch_id: 0,
            num_components: 2,
            layout: serde_json::json!({}),
        });
        assert!(message.data_body.is_empty());
        assert_eq!(
            arrow_ipc::root_as_message(&message.data_header)
                .unwrap()
                .header_type(),
            arrow_ipc::MessageHeader::NONE
        );
        let metadata: BatchMetadata = serde_json::from_slice(&message.app_metadata).unwrap();
        assert_eq!((metadata.batch_id, metadata.num_components), (0, 2));
    }
}
