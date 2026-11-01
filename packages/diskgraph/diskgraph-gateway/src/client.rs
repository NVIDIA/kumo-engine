// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reusable native Flight client with schema-aware component assembly.
use crate::protocol::{
    ComponentMetadata, StreamMetadata, MAX_BATCH_BYTES, STREAM_METADATA_KEY, VERSION,
};
use anyhow::{anyhow, ensure, Context};
use arrow_array::RecordBatch;
use arrow_flight::{
    decode::{DecodedFlightData, DecodedPayload, FlightDataDecoder},
    error::FlightError,
    flight_service_client::FlightServiceClient,
    FlightData, Ticket,
};
use futures::StreamExt;
use parking_lot::Mutex;
use serde_json::Value;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Duration;
use tokio::{runtime::Runtime, sync::watch};
use tonic::{
    transport::{Channel, Endpoint},
    Request, Status,
};

struct ClientRuntime(Option<Runtime>);
impl Drop for ClientRuntime {
    fn drop(&mut self) {
        if let Some(runtime) = self.0.take() {
            // A Python finalizer may hold the GIL while native loggers need it.
            std::thread::spawn(move || runtime.shutdown_timeout(Duration::from_secs(5)));
        }
    }
}
#[derive(Clone)]
struct Connection {
    runtime: Arc<ClientRuntime>,
    client: FlightServiceClient<Channel>,
}

/// A cached native connection shared by independent epoch shard streams.
pub struct SamplingClient {
    connection: Mutex<Option<Connection>>,
    stop: watch::Sender<bool>,
}

/// A complete logical minibatch, using native Arrow components and a small layout.
pub struct DecodedBatch {
    pub layout: Value,
    pub components: Vec<RecordBatch>,
}

struct Reader {
    connection: Connection,
    decoder: FlightDataDecoder,
    budget: Arc<AtomicUsize>,
    batch_id: u64,
    deadline: tokio::time::Instant,
}

/// Cancel-safe blocking interface to a schema-changing native Flight stream.
pub struct SamplingStream {
    reader: Mutex<Option<Reader>>,
    stop: watch::Sender<bool>,
    client_stop: watch::Sender<bool>,
    pub num_samples: u64,
    pub num_batches: u64,
}

fn flight_error(error: FlightError) -> anyhow::Error {
    match error {
        FlightError::Tonic(status) => (*status).into(),
        other => anyhow!(other),
    }
}

fn measure_message(message: &FlightData, budget: &AtomicUsize) -> Result<(), FlightError> {
    let size = message.data_header.len() + message.data_body.len() + message.app_metadata.len();
    let previous = budget.fetch_add(size, Ordering::Relaxed);
    if previous.saturating_add(size) > MAX_BATCH_BYTES {
        return Err(FlightError::protocol("Sampling batch exceeds 64 MiB"));
    }
    Ok(())
}

fn stream_metadata(payload: DecodedPayload) -> anyhow::Result<StreamMetadata> {
    let DecodedPayload::Schema(schema) = payload else {
        return Err(anyhow!("Missing sampling stream schema"));
    };
    ensure!(
        schema.fields().is_empty(),
        "Invalid sampling descriptor schema"
    );
    let metadata: StreamMetadata = serde_json::from_str(
        schema
            .metadata()
            .get(STREAM_METADATA_KEY)
            .context("Missing sampling schema metadata")?,
    )?;
    ensure!(
        metadata.version == VERSION,
        "Unsupported sampling schema version"
    );
    Ok(metadata)
}

async fn next_message(
    reader: &mut Reader,
    stop: &mut watch::Receiver<bool>,
    client_stop: &mut watch::Receiver<bool>,
) -> anyhow::Result<Option<DecodedFlightData>> {
    if *stop.borrow() || *client_stop.borrow() {
        return Err(Status::cancelled("Sampling stream closed").into());
    }
    tokio::select! {
        biased;
        _ = stop.wait_for(|stop| *stop) => Err(Status::cancelled("Sampling stream closed").into()),
        _ = client_stop.wait_for(|stop| *stop) => Err(Status::cancelled("Sampling client closed").into()),
        _ = tokio::time::sleep_until(reader.deadline) => Err(Status::deadline_exceeded("Sampling stream deadline exceeded").into()),
        message = reader.decoder.next() => message.transpose().map_err(flight_error),
    }
}

impl SamplingClient {
    /// Create a lazy reusable Flight connection without starting any sampling.
    pub fn new(endpoint: &str) -> anyhow::Result<Self> {
        let uri = endpoint
            .strip_prefix("grpc://")
            .map(|host| format!("http://{host}"))
            .unwrap_or_else(|| endpoint.to_owned());
        let endpoint = Endpoint::from_shared(uri)?.connect_timeout(Duration::from_secs(10));
        let runtime = Arc::new(ClientRuntime(Some(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .thread_name("diskgraph-flight-client")
                .enable_all()
                .build()?,
        )));
        let client = {
            let _guard = runtime.0.as_ref().unwrap().enter();
            FlightServiceClient::new(endpoint.connect_lazy())
                .max_decoding_message_size(128 * 1024 * 1024)
                .max_encoding_message_size(128 * 1024 * 1024)
        };
        let (stop, _) = watch::channel(false);
        Ok(Self {
            connection: Mutex::new(Some(Connection { runtime, client })),
            stop,
        })
    }

    /// Open one DoGet epoch request and validate the initial descriptor schema.
    pub fn open(&self, request: Vec<u8>) -> anyhow::Result<SamplingStream> {
        ensure!(
            request.len() <= MAX_BATCH_BYTES,
            "Epoch ticket exceeds the byte budget"
        );
        let connection = self
            .connection
            .lock()
            .as_ref()
            .cloned()
            .ok_or_else(|| anyhow!(Status::cancelled("Sampling client closed")))?;
        let runtime = connection.runtime.clone();
        runtime.0.as_ref().unwrap().block_on(async {
            let mut client_stop = self.stop.subscribe();
            let mut ticket = Request::new(Ticket { ticket: request.into() });
            ticket.set_timeout(Duration::from_secs(3600));
            let mut client = connection.client.clone();
            let response = tokio::select! {
                _ = client_stop.wait_for(|stop| *stop) => return Err(Status::cancelled("Sampling client closed").into()),
                response = client.do_get(ticket) => response?,
            };
            let budget = Arc::new(AtomicUsize::new(0));
            let measured = budget.clone();
            let stream = response.into_inner().map(move |message| {
                let message = message.map_err(FlightError::from)?;
                measure_message(&message, &measured)?;
                Ok(message)
            });
            let (stop, _) = watch::channel(false);
            let mut local_stop = stop.subscribe();
            let mut reader = Reader {
                connection, decoder: FlightDataDecoder::new(stream), budget,
                batch_id: 0, deadline: tokio::time::Instant::now() + Duration::from_secs(3600),
            };
            let header = next_message(&mut reader, &mut local_stop, &mut client_stop).await?
                .context("Missing sampling stream schema")?;
            let metadata = stream_metadata(header.payload)?;
            Ok(SamplingStream {
                reader: Mutex::new(Some(reader)), stop, client_stop: self.stop.clone(),
                num_samples: metadata.num_samples, num_batches: metadata.num_batches,
            })
        })
    }

    /// Cancel all owned streams and release the cached connection.
    pub fn close(&self) {
        self.stop.send_replace(true);
        self.connection.lock().take();
    }
}
impl Drop for SamplingClient {
    fn drop(&mut self) {
        self.close();
    }
}

impl SamplingStream {
    /// Read one complete minibatch; successful EOF must match the advertised count.
    pub fn next_batch(&self) -> anyhow::Result<Option<DecodedBatch>> {
        let mut state = self.reader.lock();
        let Some(reader) = state.as_mut() else {
            return Ok(None);
        };
        let runtime = reader.connection.runtime.clone();
        let mut stop = self.stop.subscribe();
        let mut client_stop = self.client_stop.subscribe();
        let result = runtime.0.as_ref().unwrap().block_on(async {
            reader.budget.store(0, Ordering::Relaxed);
            let mut parts = Vec::new();
            let mut layout = None;
            let mut count = None;
            let mut pending_schema = false;
            loop {
                let Some(message) = next_message(reader, &mut stop, &mut client_stop).await? else {
                    ensure!(
                        parts.is_empty() && !pending_schema && reader.batch_id == self.num_batches,
                        "Incomplete sampling stream: expected {} batches, received {}",
                        self.num_batches,
                        reader.batch_id
                    );
                    return Ok(None);
                };
                ensure!(
                    reader.batch_id < self.num_batches,
                    "Unexpected data after advertised batch count"
                );
                let tag: ComponentMetadata = serde_json::from_slice(&message.inner.app_metadata)
                    .context("Invalid sampling component metadata")?;
                ensure!(
                    tag.batch_id == reader.batch_id
                        && tag.component_id == parts.len()
                        && tag.num_components > 0,
                    "Invalid sampling component sequence"
                );
                ensure!(
                    count.is_none_or(|count| count == tag.num_components),
                    "Inconsistent component count"
                );
                count = Some(tag.num_components);
                match message.payload {
                    DecodedPayload::Schema(_) => {
                        ensure!(
                            !pending_schema && tag.value.is_none(),
                            "Unexpected component schema"
                        );
                        pending_schema = true;
                    }
                    DecodedPayload::RecordBatch(batch) => {
                        ensure!(pending_schema, "Component batch has no matching schema");
                        if parts.is_empty() {
                            let value = tag.value.context("Missing minibatch layout")?;
                            ensure!(
                                value.as_object().is_some_and(|value| value.len() == 3
                                    && value.contains_key("task")
                                    && value.contains_key("graphs")
                                    && value.contains_key("raw_features")),
                                "Invalid minibatch layout"
                            );
                            layout = Some(value);
                        } else {
                            ensure!(tag.value.is_none(), "Repeated minibatch layout");
                        }
                        parts.push(batch);
                        pending_schema = false;
                        if parts.len() == tag.num_components {
                            let layout = layout.unwrap();
                            validate_layout(&layout, &parts)?;
                            reader.batch_id += 1;
                            return Ok(Some(DecodedBatch {
                                layout,
                                components: parts,
                            }));
                        }
                    }
                    _ => return Err(anyhow!("Unexpected metadata-only sampling message")),
                }
            }
        });
        if !matches!(&result, Ok(Some(_))) {
            state.take();
        }
        result
    }

    /// Wake blocked reads and discard partial components without waiting for the GIL.
    pub fn close(&self) {
        self.stop.send_replace(true);
        if let Some(mut state) = self.reader.try_lock() {
            state.take();
        }
    }
}
impl Drop for SamplingStream {
    fn drop(&mut self) {
        self.close();
    }
}

fn validate_layout(value: &Value, components: &[RecordBatch]) -> anyhow::Result<()> {
    match value {
        Value::Object(values)
            if values.len() == 2
                && values.contains_key("__arrow__")
                && values.get("array").is_some_and(Value::is_boolean) =>
        {
            let slot = values["__arrow__"]
                .as_u64()
                .and_then(|slot| usize::try_from(slot).ok())
                .context("Invalid Arrow component reference")?;
            let batch = components
                .get(slot)
                .context("Invalid Arrow component reference")?;
            ensure!(
                values["array"] != true || batch.num_columns() == 1,
                "Array component must contain one column"
            );
        }
        Value::Object(values) => {
            for value in values.values() {
                validate_layout(value, components)?;
            }
        }
        Value::Array(values) => {
            for value in values {
                validate_layout(value, components)?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{types::Int32Type, ArrayRef, DictionaryArray, Int64Array, StringArray};
    use arrow_flight::FlightData;
    use arrow_ipc::writer::{
        CompressionContext, DictionaryTracker, IpcDataGenerator, IpcWriteOptions,
    };
    use futures::{stream, Stream};
    use serde_json::json;

    fn layout() -> Value {
        json!({"task": {"target": {"__arrow__": 0, "array": false}}, "graphs": {}, "raw_features": {}})
    }
    pub(super) fn batch() -> RecordBatch {
        RecordBatch::try_from_iter([("target", Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef)])
            .unwrap()
    }
    fn frames(
        batch_id: u64,
        component_id: usize,
        count: usize,
        batch: &RecordBatch,
        value: Option<Value>,
    ) -> Vec<Result<FlightData, FlightError>> {
        let tag = ComponentMetadata {
            batch_id,
            component_id,
            num_components: count,
            value: None,
        };
        let metadata = serde_json::to_vec(&tag).unwrap();
        let options = IpcWriteOptions::default();
        let generator = IpcDataGenerator::default();
        let mut tracker = DictionaryTracker::new(false);
        let mut schema: FlightData = generator
            .schema_to_bytes_with_dictionary_tracker(&batch.schema(), &mut tracker, &options)
            .into();
        schema.app_metadata = metadata.clone().into();
        let (dictionaries, data) = generator
            .encode(
                batch,
                &mut tracker,
                &options,
                &mut CompressionContext::default(),
            )
            .unwrap();
        let mut result = vec![Ok(schema)];
        for dictionary in dictionaries {
            let mut frame: FlightData = dictionary.into();
            frame.app_metadata = metadata.clone().into();
            result.push(Ok(frame));
        }
        let mut frame: FlightData = data.into();
        frame.app_metadata = serde_json::to_vec(&ComponentMetadata { value, ..tag })
            .unwrap()
            .into();
        result.push(Ok(frame));
        result
    }
    fn reader(
        stream: impl Stream<Item = Result<FlightData, FlightError>> + Send + 'static,
        num_batches: u64,
    ) -> SamplingStream {
        let client = SamplingClient::new("grpc://127.0.0.1:1").unwrap();
        let connection = client.connection.lock().as_ref().unwrap().clone();
        let (stop, _) = watch::channel(false);
        SamplingStream {
            reader: Mutex::new(Some(Reader {
                connection,
                decoder: FlightDataDecoder::new(stream),
                budget: Arc::new(AtomicUsize::new(0)),
                batch_id: 0,
                deadline: tokio::time::Instant::now() + Duration::from_secs(10),
            })),
            stop,
            client_stop: watch::channel(false).0,
            num_samples: 2,
            num_batches,
        }
    }
    #[test]
    fn native_components_change_schemas_row_counts_and_dictionaries() {
        let targets = batch();
        let first = RecordBatch::try_from_iter([(
            "category",
            Arc::new(DictionaryArray::<Int32Type>::from_iter([
                Some("a"),
                None,
                Some("b"),
            ])) as ArrayRef,
        )])
        .unwrap();
        let second = RecordBatch::try_from_iter([(
            "category",
            Arc::new(DictionaryArray::<Int32Type>::from_iter([Some("new")])) as ArrayRef,
        )])
        .unwrap();
        let text = RecordBatch::try_from_iter([(
            "text",
            Arc::new(StringArray::from(vec!["x"])) as ArrayRef,
        )])
        .unwrap();
        let mut wire = frames(0, 0, 4, &targets, Some(layout()));
        wire.extend(frames(0, 1, 4, &first, None));
        wire.extend(frames(0, 2, 4, &second, None));
        wire.extend(frames(0, 3, 4, &text, None));
        let stream = reader(stream::iter(wire), 1);
        let decoded = stream.next_batch().unwrap().unwrap();
        assert_eq!(decoded.components, vec![targets, first, second, text]);
        assert!(stream.next_batch().unwrap().is_none());
    }
    #[test]
    fn malformed_or_truncated_components_never_yield_partial_batches() {
        let cases = [
            (vec![], 1, "Incomplete"),
            (frames(0, 0, 2, &batch(), Some(layout())), 1, "Incomplete"),
            (frames(0, 1, 1, &batch(), Some(layout())), 1, "sequence"),
            (frames(1, 0, 1, &batch(), Some(layout())), 1, "sequence"),
            (
                frames(0, 0, 1, &batch(), None),
                1,
                "Missing minibatch layout",
            ),
            (
                frames(0, 0, 1, &batch(), Some(json!({}))),
                1,
                "Invalid minibatch layout",
            ),
            (
                frames(
                    0,
                    0,
                    1,
                    &batch(),
                    Some(
                        json!({"task":{"__arrow__":9,"array":true},"graphs":{},"raw_features":{}}),
                    ),
                ),
                1,
                "component reference",
            ),
            (
                frames(0, 0, 1, &batch(), Some(layout())),
                0,
                "Unexpected data",
            ),
        ];
        for (wire, count, message) in cases {
            let stream = reader(stream::iter(wire), count);
            assert!(
                stream
                    .next_batch()
                    .err()
                    .unwrap()
                    .to_string()
                    .contains(message),
                "{message}"
            );
            assert!(stream.next_batch().unwrap().is_none());
        }
        let mut wire = frames(0, 0, 1, &batch(), Some(layout()));
        wire.pop();
        assert!(reader(stream::iter(wire), 1)
            .next_batch()
            .err()
            .unwrap()
            .to_string()
            .contains("Incomplete"));
    }
    #[test]
    fn final_rpc_error_is_observed_even_after_all_advertised_batches() {
        for count in [0, 1] {
            let mut wire = if count == 1 {
                frames(0, 0, 1, &batch(), Some(layout()))
            } else {
                vec![]
            };
            wire.push(Err(Status::unavailable("connection lost").into()));
            let stream = reader(stream::iter(wire), count);
            if count == 1 {
                assert!(stream.next_batch().unwrap().is_some());
            }
            let error = stream.next_batch().err().unwrap();
            assert_eq!(
                error.downcast_ref::<Status>().unwrap().code(),
                tonic::Code::Unavailable
            );
        }
        assert!(reader(stream::empty(), 0).next_batch().unwrap().is_none());
    }
    #[test]
    fn close_wakes_blocked_reader_and_client_shutdown_cancels_streams() {
        for close_client in [false, true] {
            let stream = Arc::new(reader(stream::pending(), 1));
            let worker = stream.clone();
            let (started, waiting) = std::sync::mpsc::channel();
            let thread = std::thread::spawn(move || {
                started.send(()).unwrap();
                worker.next_batch().err().unwrap()
            });
            waiting.recv_timeout(Duration::from_secs(2)).unwrap();
            if close_client {
                stream.client_stop.send_replace(true);
            } else {
                stream.stop.send_replace(true);
            }
            assert_eq!(
                thread
                    .join()
                    .unwrap()
                    .downcast_ref::<Status>()
                    .unwrap()
                    .code(),
                tonic::Code::Cancelled
            );
        }
    }
    #[test]
    fn layout_reference_names_can_be_node_type_names() {
        let marker = json!({"__arrow__":0,"array":true});
        validate_layout(&json!({"__arrow__": marker, "array": marker}), &[batch()]).unwrap();
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;
    use arrow_schema::Schema;
    use serde_json::json;

    #[test]
    fn descriptor_rejects_incompatible_versions_and_invalid_counts() {
        for metadata in [
            json!({}),
            json!([]),
            json!(null),
            json!({"version":0,"num_samples":3,"num_batches":1}),
            json!({"version":VERSION,"num_samples":-1,"num_batches":1}),
            json!({"version":VERSION,"num_samples":true,"num_batches":1}),
            json!({"version":VERSION,"num_samples":3,"num_batches":-1}),
            json!({"version":VERSION,"num_samples":3,"num_batches":false}),
            json!({"version":7.0,"num_samples":3,"num_batches":1}),
            json!({"version":VERSION,"num_samples":3,"num_batches":1,"extra":0}),
        ] {
            let schema = Schema::empty()
                .with_metadata([(STREAM_METADATA_KEY.to_owned(), metadata.to_string())].into());
            assert!(stream_metadata(DecodedPayload::Schema(Arc::new(schema))).is_err());
        }
        assert!(stream_metadata(DecodedPayload::Schema(Arc::new(Schema::empty()))).is_err());
        assert!(stream_metadata(DecodedPayload::RecordBatch(super::tests::batch())).is_err());
        let metadata =
            stream_metadata(DecodedPayload::Schema(crate::protocol::stream_schema(3, 2))).unwrap();
        assert_eq!((metadata.num_samples, metadata.num_batches), (3, 2));
    }
    #[test]
    fn wire_budget_includes_headers_metadata_and_all_component_bodies() {
        let budget = AtomicUsize::new(MAX_BATCH_BYTES - 6);
        let message = FlightData {
            data_header: vec![1, 2].into(),
            data_body: vec![3, 4].into(),
            app_metadata: vec![5, 6].into(),
            ..Default::default()
        };
        measure_message(&message, &budget).unwrap();
        assert!(measure_message(&message, &budget)
            .unwrap_err()
            .to_string()
            .contains("64 MiB"));
        budget.store(0, Ordering::Relaxed);
        measure_message(&message, &budget).unwrap();
    }
}
