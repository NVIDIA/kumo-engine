// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native task registration and credit-controlled Arrow Flight batch serving.
use arrow_array::RecordBatch;
use arrow_flight::{
    decode::FlightRecordBatchStream, error::FlightError, flight_service_server::FlightService,
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, PollInfo, PutResult, SchemaAsIpc, SchemaResult, Ticket,
};
use arrow_ipc::writer::{CompressionContext, DictionaryTracker, IpcDataGenerator, IpcWriteOptions};
use diskgraph_runtime::RuntimeIndex;
use futures::{Stream, StreamExt};
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    path::PathBuf,
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};
use tokio::sync::{mpsc, watch, OwnedSemaphorePermit, Semaphore};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

mod encoding;
mod protocol;
mod runtime;
mod server;
mod storage;
use encoding::BatchEncoder;
use protocol::{decode, envelope, MAX_BATCH_BYTES, VERSION, WINDOW};
use runtime::{runtime_error, TaskRequest};
pub use server::{SamplingServer, ServerConfig};
use storage::{hash_task, task_path, Upload};

type RpcStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;
type IndexKey = (String, Option<u64>, usize);

struct State {
    root: PathBuf,
    max_threads: usize,
    port: u16,
    stop: watch::Sender<bool>,
    permits: Arc<Semaphore>,
    indexes: Mutex<HashMap<IndexKey, Arc<RuntimeIndex>>>,
    tasks: Mutex<HashMap<String, String>>,
    uploads: Arc<tokio::sync::Mutex<()>>,
    active: AtomicU64,
    requests: AtomicU64,
    batches: AtomicU64,
}

/// Own immutable native indexes and serve complete worker-owned epoch shards.
#[derive(Clone)]
pub struct SamplerGateway {
    state: Arc<State>,
}
impl SamplerGateway {
    /// Validate the graph root and positive resource budgets without opening an index.
    pub fn new(
        root: PathBuf,
        max_streams: usize,
        max_threads: usize,
        port: u16,
    ) -> Result<Self, Status> {
        let root = root.canonicalize().map_err(storage::io_error)?;
        if !root.join("index_metadata.json").is_file() || max_streams == 0 || max_threads == 0 {
            return Err(Status::invalid_argument(
                "A completed graph index and positive sampling budgets are required",
            ));
        }
        let (stop, _) = watch::channel(false);
        Ok(Self {
            state: Arc::new(State {
                root,
                max_threads,
                port,
                stop,
                permits: Arc::new(Semaphore::new(max_streams)),
                indexes: Mutex::new(HashMap::new()),
                tasks: Mutex::new(HashMap::new()),
                uploads: Arc::new(tokio::sync::Mutex::new(())),
                active: AtomicU64::new(0),
                requests: AtomicU64::new(0),
                batches: AtomicU64::new(0),
            }),
        })
    }

    /// Cancel waiting producers and reject subsequent work before shutting down the listener.
    pub fn shutdown(&self) {
        self.state.stop.send_replace(true);
    }

    fn health(&self) -> Value {
        let requests = self.state.requests.load(Ordering::Relaxed);
        json!({"ready": !*self.state.stop.borrow(), "pid":std::process::id(), "port":self.state.port,
            "has_training_tables":self.state.root.join("train_tables").exists(),
            "active_streams":self.state.active.load(Ordering::Relaxed), "requests":requests,
            "task_requests":requests, "exchange_streams":requests,
            "batches":self.state.batches.load(Ordering::Relaxed)})
    }
}

fn command<T: serde::de::DeserializeOwned>(
    descriptor: Option<&FlightDescriptor>,
) -> Result<T, Status> {
    let data = descriptor.ok_or_else(|| Status::invalid_argument("Missing Flight descriptor"))?;
    if data.cmd.len() > MAX_BATCH_BYTES {
        return Err(Status::resource_exhausted(
            "Descriptor exceeds the byte budget",
        ));
    }
    serde_json::from_slice(&data.cmd).map_err(|error| Status::invalid_argument(error.to_string()))
}

fn input(first: FlightData, rest: tonic::Streaming<FlightData>) -> FlightRecordBatchStream {
    FlightRecordBatchStream::new_from_flight_data(
        futures::stream::iter(
            (!first.data_header.is_empty() || !first.data_body.is_empty()).then_some(Ok(first)),
        )
        .chain(rest.map(|message| message.map_err(FlightError::from))),
    )
}

struct Output {
    tx: mpsc::Sender<Result<FlightData, Status>>,
    stop: watch::Receiver<bool>,
}
impl Output {
    fn send(&mut self, data: FlightData) -> Result<(), Status> {
        tokio::runtime::Handle::current().block_on(async {
            tokio::select! {
                _ = self.stop.wait_for(|stop| *stop) => Err(Status::cancelled("Gateway is stopping")),
                sent = self.tx.send(Ok(data)) => sent.map_err(|_| Status::cancelled("Consumer closed")),
            }
        })
    }
    fn batch(&mut self, batch: RecordBatch) -> Result<(), Status> {
        let (dictionaries, data) = IpcDataGenerator::default()
            .encode(
                &batch,
                &mut DictionaryTracker::new(false),
                &IpcWriteOptions::default(),
                &mut CompressionContext::default(),
            )
            .map_err(|error| Status::internal(error.to_string()))?;
        for message in dictionaries.into_iter().chain(std::iter::once(data)) {
            self.send(message.into())?;
        }
        Ok(())
    }
    fn receive(
        &mut self,
        reader: &mut FlightRecordBatchStream,
    ) -> Result<Option<RecordBatch>, Status> {
        tokio::runtime::Handle::current().block_on(async {
            tokio::select! {
                _ = self.stop.wait_for(|stop| *stop) => Err(Status::cancelled("Gateway is stopping")),
                _ = self.tx.closed() => Err(Status::cancelled("Consumer closed")),
                batch = reader.next() => batch.transpose().map_err(Status::from),
            }
        })
    }
}

struct Lease {
    state: Arc<State>,
    _permit: OwnedSemaphorePermit,
    id: u64,
    started: Instant,
    count: u64,
    complete: bool,
}
impl Drop for Lease {
    fn drop(&mut self) {
        self.state.active.fetch_sub(1, Ordering::Relaxed);
        tracing::info!(
            stream_id = self.id,
            batches = self.count,
            seconds = self.started.elapsed().as_secs_f64(),
            complete = self.complete,
            "SAMPLING_STREAM_CLOSED"
        );
    }
}

fn epoch(
    state: Arc<State>,
    request: TaskRequest,
    mut reader: FlightRecordBatchStream,
    tx: mpsc::Sender<Result<FlightData, Status>>,
    permit: OwnedSemaphorePermit,
) {
    state.active.fetch_add(1, Ordering::Relaxed);
    let id = state.requests.fetch_add(1, Ordering::Relaxed) + 1;
    let mut lease = Lease {
        state: state.clone(),
        _permit: permit,
        id,
        started: Instant::now(),
        count: 0,
        complete: false,
    };
    let mut output = Output {
        tx,
        stop: state.stop.subscribe(),
    };
    tracing::info!(
        stream_id = id,
        pid = std::process::id(),
        "SAMPLING_STREAM_OPEN"
    );
    let result = (|| -> Result<(), Status> {
        if request.kind != "push_iter" || request.version != VERSION {
            return Err(Status::invalid_argument(
                "Unsupported push sampling protocol",
            ));
        }
        let threads = request.num_threads.unwrap_or(1);
        if threads == 0 || threads > state.max_threads {
            return Err(Status::invalid_argument(
                "Requested native threads exceed the service budget",
            ));
        }
        let extension = task_path(&state.root, &request.extension_ref)?
            .canonicalize()
            .map_err(storage::io_error)?;
        let key = (request.extension_ref.clone(), request.engine_seed, threads);
        let index = {
            let mut indexes = state.indexes.lock();
            if let Some(index) = indexes.get(&key) {
                index.clone()
            } else {
                if indexes.len() >= 4 {
                    return Err(Status::resource_exhausted("Task index cache is full"));
                }
                let index = Arc::new(
                    RuntimeIndex::new(
                        state.root.to_string_lossy().into_owned(),
                        request.extension_ref,
                        request.engine_seed,
                        Some(threads),
                    )
                    .map_err(runtime_error)?,
                );
                indexes.insert(key, index.clone());
                index
            }
        };
        let mut iterator = request.options.open(index, &extension)?;
        let expected = iterator.worker_batch_count() as u64;
        output.send(SchemaAsIpc::new(&protocol::schema(), &IpcWriteOptions::default()).into())?;
        output.batch(envelope(json!({"kind":"iterator","version":VERSION,"num_samples":iterator.num_samples(),"num_batches":expected}), vec![])?)?;
        while lease.count < expected {
            let frame = output.receive(&mut reader)?.ok_or_else(|| {
                Status::invalid_argument("Push stream ended before its shard completed")
            })?;
            let (metadata, parts) = decode(&frame)?;
            let credit = metadata.get("batches").and_then(Value::as_u64).unwrap_or(0);
            if metadata.as_object().unwrap().len() != 2
                || metadata["kind"] != "credit"
                || credit == 0
                || credit > WINDOW
                || credit > expected - lease.count
                || !parts.is_empty()
            {
                return Err(Status::invalid_argument("Invalid push stream credits"));
            }
            for _ in 0..credit {
                if output.tx.is_closed() || *output.stop.borrow() {
                    return Err(Status::cancelled("Consumer closed"));
                }
                let batch = iterator
                    .next_batch()
                    .map_err(runtime_error)?
                    .ok_or_else(|| {
                        Status::internal("Native shard ended before its declared batch count")
                    })?;
                let mut encoder = BatchEncoder::default();
                let value = encoder
                    .batch(batch)
                    .map_err(|error| Status::resource_exhausted(error.to_string()))?;
                output.batch(envelope(
                    json!({"kind":"batch","batch_id":lease.count,"value":value}),
                    encoder.parts,
                )?)?;
                lease.count += 1;
                let total = state.batches.fetch_add(1, Ordering::Relaxed) + 1;
                if total == 1 || total.is_multiple_of(100) {
                    tracing::info!(
                        batches = total,
                        task_requests = state.requests.load(Ordering::Relaxed),
                        "SAMPLING_PROGRESS"
                    );
                }
            }
        }
        if iterator.next_batch().map_err(runtime_error)?.is_some() {
            return Err(Status::internal(
                "Native shard exceeded its declared batch count",
            ));
        }
        output.batch(envelope(
            json!({"kind":"end","batches":lease.count}),
            vec![],
        )?)?;
        if output.receive(&mut reader)?.is_some() {
            return Err(Status::invalid_argument(
                "Unexpected credits after shard completion",
            ));
        }
        lease.complete = true;
        Ok(())
    })();
    if let Err(error) = result {
        if error.code() != tonic::Code::Cancelled {
            tracing::error!(stream_id=id, code=?error.code(), "SAMPLING_STREAM_FAILED");
        }
        // Preserve the RPC error even when a previous batch fills the output slot.
        tokio::runtime::Handle::current().block_on(async {
            tokio::select! {
                _ = output.stop.wait_for(|stop| *stop) => {},
                _ = output.tx.closed() => {},
                _ = output.tx.send(Err(error)) => {},
            }
        });
    }
    drop(lease);
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TaskIdentity {
    extension_ref: String,
    sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Prepare {
    kind: String,
    version: u32,
    extension_ref: String,
    sha256: String,
}

#[tonic::async_trait]
impl FlightService for SamplerGateway {
    type HandshakeStream = RpcStream<HandshakeResponse>;
    type ListFlightsStream = RpcStream<FlightInfo>;
    type DoGetStream = RpcStream<FlightData>;
    type DoPutStream = RpcStream<PutResult>;
    type DoActionStream = RpcStream<arrow_flight::Result>;
    type ListActionsStream = RpcStream<ActionType>;
    type DoExchangeStream = RpcStream<FlightData>;

    async fn handshake(
        &self,
        _: Request<tonic::Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        Err(Status::unimplemented("Handshake"))
    }
    async fn list_flights(
        &self,
        _: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        Err(Status::unimplemented("ListFlights"))
    }
    async fn get_flight_info(
        &self,
        _: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        Err(Status::unimplemented("GetFlightInfo"))
    }
    async fn poll_flight_info(
        &self,
        _: Request<FlightDescriptor>,
    ) -> Result<Response<PollInfo>, Status> {
        Err(Status::unimplemented("PollFlightInfo"))
    }
    async fn get_schema(
        &self,
        _: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        Err(Status::unimplemented("GetSchema"))
    }
    async fn do_get(&self, _: Request<Ticket>) -> Result<Response<Self::DoGetStream>, Status> {
        Err(Status::unimplemented(
            "Use DoExchange for worker epoch shards",
        ))
    }

    async fn list_actions(
        &self,
        _: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        Ok(Response::new(Box::pin(futures::stream::iter([
            Ok(ActionType {
                r#type: "task_status".into(),
                description: "Check immutable task identity".into(),
            }),
            Ok(ActionType {
                r#type: "health".into(),
                description: "Read worker readiness and counters".into(),
            }),
        ]))))
    }
    async fn do_action(
        &self,
        request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        let action = request.into_inner();
        let body = match action.r#type.as_str() {
            "health" => serde_json::to_vec(&self.health()).unwrap(),
            "task_status" => {
                if *self.state.stop.borrow() {
                    return Err(Status::unavailable("Gateway is stopping"));
                }
                if action.body.len() > MAX_BATCH_BYTES {
                    return Err(Status::resource_exhausted(
                        "Task request exceeds the byte budget",
                    ));
                }
                let request: TaskIdentity = serde_json::from_slice(&action.body)
                    .map_err(|error| Status::invalid_argument(error.to_string()))?;
                let _lock = self.state.uploads.lock().await;
                let state = self.state.clone();
                tokio::task::spawn_blocking(move || {
                    let path = task_path(&state.root, &request.extension_ref)?;
                    let existing = state.tasks.lock().get(&request.extension_ref).cloned();
                    let hash = match existing {
                        Some(hash) => Some(hash),
                        None if path.exists() => Some(hash_task(&path)?),
                        None => None,
                    };
                    if let Some(hash) = hash {
                        if hash != request.sha256 {
                            return Err(Status::invalid_argument(
                                "Task extension identity mismatch",
                            ));
                        }
                        state
                            .tasks
                            .lock()
                            .insert(request.extension_ref, hash.clone());
                        Ok(hash.into_bytes())
                    } else {
                        Ok(vec![])
                    }
                })
                .await
                .map_err(|error| Status::internal(error.to_string()))??
            }
            _ => return Err(Status::unimplemented("Unknown sampling action")),
        };
        Ok(Response::new(Box::pin(futures::stream::once(async move {
            Ok(arrow_flight::Result { body: body.into() })
        }))))
    }

    async fn do_exchange(
        &self,
        request: Request<tonic::Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        if *self.state.stop.borrow() {
            return Err(Status::unavailable("Gateway is stopping"));
        }
        let mut stream = request.into_inner();
        let first = stream
            .message()
            .await?
            .ok_or_else(|| Status::invalid_argument("Missing epoch request"))?;
        let request: TaskRequest = command(first.flight_descriptor.as_ref())?;
        let permit = self
            .state
            .permits
            .clone()
            .try_acquire_owned()
            .map_err(|_| Status::resource_exhausted("Sampling service is busy"))?;
        let (tx, rx) = mpsc::channel(1);
        let state = self.state.clone();
        tokio::task::spawn_blocking(move || {
            epoch(state, request, input(first, stream), tx, permit)
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }

    async fn do_put(
        &self,
        request: Request<tonic::Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        if *self.state.stop.borrow() {
            return Err(Status::unavailable("Gateway is stopping"));
        }
        let mut stream = request.into_inner();
        let first = stream
            .message()
            .await?
            .ok_or_else(|| Status::invalid_argument("Missing task preparation request"))?;
        let request: Prepare = command(first.flight_descriptor.as_ref())?;
        if request.kind != "prepare_task" || request.version != VERSION {
            return Err(Status::invalid_argument(
                "Unsupported task preparation protocol",
            ));
        }
        let lock = self.state.uploads.clone().lock_owned().await;
        let state = self.state.clone();
        let (tx, rx) = mpsc::channel(1);
        tokio::task::spawn_blocking(move || {
            let _lock = lock;
            let mut reader = input(first, stream);
            let mut stop = state.stop.subscribe();
            let result = (|| -> Result<(), Status> {
                let mut upload = Upload::new(
                    task_path(&state.root, &request.extension_ref)?,
                    &request.sha256,
                )?;
                loop {
                    let batch=tokio::runtime::Handle::current().block_on(async {
                        tokio::select! {
                            _=stop.wait_for(|stop|*stop)=>Err(Status::cancelled("Gateway is stopping")),
                            _=tx.closed()=>Err(Status::cancelled("Consumer closed")),
                            frame=reader.next()=>frame.transpose().map_err(Status::from),
                        }
                    })?.ok_or_else(||Status::invalid_argument("Task transfer ended without completion"))?;
                    let (metadata, parts) = decode(&batch)?;
                    match metadata["kind"].as_str() {
                        Some("file")
                            if metadata.as_object().unwrap().len() == 3 && parts.is_empty() =>
                        {
                            let name = metadata["path"].as_str().ok_or_else(|| {
                                Status::invalid_argument("Invalid task file path")
                            })?;
                            let size = metadata["size"].as_u64().ok_or_else(|| {
                                Status::invalid_argument("Invalid task file size")
                            })?;
                            upload.header(name.into(), size)?;
                        }
                        Some("data")
                            if metadata.as_object().unwrap().len() == 1 && parts.len() == 1 =>
                        {
                            upload.data(parts[0])?
                        }
                        Some("end")
                            if metadata.as_object().unwrap().len() == 1 && parts.is_empty() =>
                        {
                            let extra=tokio::runtime::Handle::current().block_on(async {
                                tokio::select! {
                                    _=stop.wait_for(|stop|*stop)=>Err(Status::cancelled("Gateway is stopping")),
                                    _=tx.closed()=>Err(Status::cancelled("Consumer closed")),
                                    frame=reader.next()=>frame.transpose().map_err(Status::from),
                                }
                            })?;
                            if extra.is_some() {
                                return Err(Status::invalid_argument(
                                    "Invalid task transfer completion",
                                ));
                            }
                            upload.finish(&request.sha256)?;
                            state
                                .tasks
                                .lock()
                                .insert(request.extension_ref, request.sha256.clone());
                            tx.blocking_send(Ok(PutResult {
                                app_metadata: request.sha256.into_bytes().into(),
                            }))
                            .map_err(|_| Status::cancelled("Consumer closed"))?;
                            return Ok(());
                        }
                        _ => return Err(Status::invalid_argument("Invalid task upload frame")),
                    }
                }
            })();
            if let Err(error) = result {
                let _ = tx.try_send(Err(error));
            }
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}
