// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire format for the sampling server's Arrow Flight DoGet response.
//!
//! # Request and response scope
//!
//! The client serializes a `TaskRequest` into `Ticket.ticket` and calls DoGet
//! directly; GetFlightInfo is not required. One request selects one epoch shard
//! of a prepared training-table extension. Its response streams all minibatches
//! in that shard, rather than returning only one minibatch per RPC.
//!
//! A native minibatch contains seed/target arrays, sampled graph structure and
//! feature record batches. These can have different row counts and schemas, so
//! the server encodes one minibatch as several Arrow record-batch *components*.
//! Each component has one schema and one record batch; there is no single schema
//! shared by the whole response. The schema-aware client in `client.rs` handles
//! these changes instead of using a fixed-schema Flight record-batch reader.
//!
//! # FlightData message order
//!
//! ```text
//! DoGet(Ticket { ticket: serialized TaskRequest })
//!   -> Schema(empty)             app_metadata = StreamMetadata
//!   -> IPC NONE                  app_metadata = BatchMetadata(batch_id = 0)
//!      -> Schema(component 0)    app_metadata = ComponentMetadata(0, 0)
//!      -> DictionaryBatch(s)     app_metadata = ComponentMetadata(0, 0), if needed
//!      -> RecordBatch            app_metadata = ComponentMetadata(0, 0)
//!      -> Schema(component 1)    app_metadata = ComponentMetadata(0, 1)
//!      -> DictionaryBatch(s)     app_metadata = ComponentMetadata(0, 1), if needed
//!      -> RecordBatch            app_metadata = ComponentMetadata(0, 1)
//!      ... remaining components of minibatch 0 ...
//!   -> IPC NONE                  app_metadata = BatchMetadata(batch_id = 1)
//!      ... components of minibatch 1 ...
//!   ... remaining minibatches in the requested shard ...
//!   -> EOF and final gRPC status
//! ```
//!
//! `data_header` identifies an Arrow IPC Schema, DictionaryBatch, RecordBatch or
//! NONE message; `data_body` contains the corresponding Arrow buffers. All three
//! metadata types below are JSON in the envelope's `app_metadata` bytes. They
//! are separate from Arrow schema/field metadata, which remains part of the IPC
//! schema. Arrow payloads are sent directly as Flight IPC messages, without
//! wrapping serialized record batches in binary-valued Arrow columns.
//!
//! The initial empty schema is a stream descriptor: it advertises the protocol
//! version and native iterator counts, not the schema of a data component. Each
//! minibatch then starts with a separate metadata-only IPC NONE message carrying
//! its layout and component count. Component schemas, dictionary messages and
//! record batches all carry the same `(batch_id, component_id)` coordinates.
//! Dictionaries belong to that component's schema; the sender starts a fresh
//! dictionary tracker for each component.
//!
//! # Grouping and reconstructing a minibatch
//!
//! `encoding.rs` groups fields that share a row domain: seed fields and target
//! columns; node IDs and seed assignments for one node type; source/destination
//! endpoints for one edge type; and compatible flattened positive-label fields.
//! Other domains, such as negative masks, per-hop counts and flattened exclusions,
//! remain separate. Each original feature record-batch chunk is a component, so
//! feature banks can span multiple components and different node types can have
//! different feature schemas. Grouping does not pad or repeat rows to make
//! unrelated arrays fit one record batch.
//!
//! `BatchMetadata.layout` has exactly three top-level keys: `task`, `graphs` and
//! `raw_features`. Containers and scalar values retain their native structure;
//! Arrow values are replaced by positional references into the component list:
//!
//! - `{"component": 0, "column": 2}` selects one Arrow array.
//! - `{"component": 0, "columns": [3, 5]}` selects columns 3 and 4 as a
//!   record batch. Ranges are half-open, and even a one-column range returns a
//!   record batch. Targets use ranges, so their original columns are appended
//!   directly rather than packed into a struct array; no `as` marker is needed.
//!
//! For example, one possible minibatch has these components. The order is
//! illustrative: clients must follow the layout, not infer IDs from field names
//! or node types. Optional fields, graph channels and feature chunks change the
//! number and schemas of components between minibatches.
//!
//! ```text
//! component  rows  columns
//!     0        2   input_ids, tt_stage_node_ids, seed_node_ids, target_a, target_b
//!     1        0   value (empty negative mask)
//!     2        5   node_ids, batch_dict                       [users]
//!     3       20   node_ids, batch_dict                       [orders]
//!     4        5   age, country                              [users features]
//!     5       20   price, quantity, category                 [orders features]
//!
//! Excerpts from BatchMetadata.layout for those components:
//!   task.seed_node_ids                 = {"component": 0, "column": 2}
//!   task.target                        = {"component": 0, "columns": [3, 5]}
//!   task.neg_mask                      = {"component": 1, "column": 0}
//!   graphs.main.node_ids.users         = {"component": 2, "column": 0}
//!   graphs.main.batch_dict.users       = {"component": 2, "column": 1}
//!   graphs.main.node_ids.orders        = {"component": 3, "column": 0}
//!   graphs.main.batch_dict.orders      = {"component": 3, "column": 1}
//!   raw_features.users.record_batches  = [{"component": 4, "columns": [0, 2]}]
//!   raw_features.orders.record_batches = [{"component": 5, "columns": [0, 3]}]
//! ```
//!
//! `layout.rs` validates these references and projects the selected arrays or
//! record batches while sharing their Arrow buffers. The PyO3 client recursively
//! replaces references with PyArrow objects through the Arrow C Data Interface,
//! restoring the native `task`/`graphs`/`raw_features` structure for Python.
//!
//! # Completion, errors and flow control
//!
//! Batch IDs start at zero and increase within this response; component IDs start
//! at zero for each minibatch. Components are sent contiguously and in order,
//! without interleaving minibatches. A minibatch is complete after exactly
//! `num_components` record batches, not after that many FlightData messages:
//! schemas and dictionaries do not count as components. There is no minibatch-end
//! record batch or end sentinel; the next metadata-only header starts the next
//! minibatch. Successful EOF must match the descriptor's `num_batches`, and the
//! client must observe the final RPC status to detect errors after the last batch.
//! Missing components, invalid references, premature EOF and non-OK status fail
//! the stream rather than producing a partial minibatch.
//!
//! The byte budget below applies cumulatively to one minibatch's metadata, IPC
//! headers, dictionaries and Arrow bodies, not independently to each component.
//! Splitting features into components does not bypass that budget. `lib.rs` sends
//! through a bounded one-slot queue and reserves a slot before sampling another
//! minibatch. A slow consumer eventually stalls production through that queue and
//! gRPC transport backpressure; transport buffers and native working memory are
//! additional. Cancellation closes the stream; there are no credit/ack messages.

use arrow_flight::{FlightData, SchemaAsIpc};
use arrow_ipc::writer::IpcWriteOptions;
use arrow_schema::Schema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Wire compatibility version checked against both the ticket and stream header.
pub const VERSION: u32 = 9;
/// Maximum encoded bytes for one complete minibatch (also used to bound tickets).
pub(crate) const MAX_BATCH_BYTES: usize = 64 * 1024 * 1024;

#[derive(Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ControlKind {
    Stream,
    Minibatch,
}

/// JSON attached to the initial empty-schema FlightData message.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StreamMetadata {
    pub kind: ControlKind,
    pub version: u32,
    /// Native iterator count: samples in ordinary mode, batches in grouped mode.
    /// This is not the number of feature rows or Arrow components.
    pub num_samples: u64,
    /// Exact minibatch count for the requested worker shard, including zero.
    pub num_batches: u64,
}

/// JSON on the metadata-only message immediately before a minibatch's components.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BatchMetadata {
    pub kind: ControlKind,
    /// Zero-based minibatch position within this DoGet response.
    pub batch_id: u64,
    /// Number of record-batch components; excludes schemas and dictionaries.
    pub num_components: usize,
    /// Native output shape with Arrow values replaced by column/range references.
    pub layout: Value,
}

/// JSON repeated on a component's schema, dictionaries and record-batch message.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ComponentMetadata {
    pub batch_id: u64,
    /// Position in this minibatch's component list, not a semantic field/type ID.
    pub component_id: usize,
}

/// Build the descriptor schema; actual component schemas follow minibatch headers.
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

/// Announce a minibatch without introducing an Arrow schema or record batch.
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
