// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native epoch sampling over prepared worker-local graph and task artifacts.
use arrow_array::RecordBatch;
use arrow_flight::{
    flight_service_server::FlightService, Action, ActionType, Criteria, Empty, FlightData,
    FlightDescriptor, FlightInfo, HandshakeRequest, HandshakeResponse, PollInfo, PutResult,
    SchemaAsIpc, SchemaResult, Ticket,
};
use arrow_ipc::writer::{CompressionContext, DictionaryTracker, IpcDataGenerator, IpcWriteOptions};
use diskgraph_runtime::RuntimeIndex;
use futures::Stream;
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    path::PathBuf,
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, watch, OwnedSemaphorePermit, Semaphore};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

mod encoding;
mod paths;
mod protocol;
mod runtime;
mod server;
use encoding::BatchEncoder;
use paths::task_path;
use protocol::{envelope, MAX_BATCH_BYTES, VERSION};
use runtime::{runtime_error, TaskRequest};
pub use server::{SamplingServer, ServerConfig};

type RpcStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;
type IndexKey = (String, Option<u64>, usize);

struct State {
    root: PathBuf,
    max_threads: usize,
    port: u16,
    stop: watch::Sender<bool>,
    permits: Arc<Semaphore>,
    indexes: Mutex<HashMap<IndexKey, Arc<RuntimeIndex>>>,
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
        let root = root.canonicalize().map_err(paths::io_error)?;
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
            "task_requests":requests, "get_streams":requests,
            "batches":self.state.batches.load(Ordering::Relaxed)})
    }
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
    fn reserve(&mut self) -> Result<mpsc::OwnedPermit<Result<FlightData, Status>>, Status> {
        let permit = tokio::runtime::Handle::current().block_on(async {
            tokio::select! {
                _ = self.stop.wait_for(|stop| *stop) => Err(Status::cancelled("Gateway is stopping")),
                permit = self.tx.clone().reserve_owned() => permit.map_err(|_| Status::cancelled("Consumer closed")),
            }
        })?;
        if self.tx.is_closed() || *self.stop.borrow() {
            return Err(Status::cancelled("Sampling stream is stopping"));
        }
        Ok(permit)
    }
    fn reserved_batch(
        &self,
        permit: mpsc::OwnedPermit<Result<FlightData, Status>>,
        batch: RecordBatch,
    ) -> Result<(), Status> {
        if self.tx.is_closed() || *self.stop.borrow() {
            return Err(Status::cancelled("Consumer closed"));
        }
        let (dictionaries, data) = IpcDataGenerator::default()
            .encode(
                &batch,
                &mut DictionaryTracker::new(false),
                &IpcWriteOptions::default(),
                &mut CompressionContext::default(),
            )
            .map_err(|error| Status::internal(error.to_string()))?;
        if !dictionaries.is_empty() {
            return Err(Status::internal(
                "Task envelopes must not contain dictionaries",
            ));
        }
        permit.send(Ok(data.into()));
        Ok(())
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
        if request.kind != "epoch" || request.version != VERSION {
            return Err(Status::invalid_argument("Unsupported sampling protocol"));
        }
        let threads = request.num_threads.unwrap_or(1);
        if threads == 0 || threads > state.max_threads {
            return Err(Status::invalid_argument(
                "Requested native threads exceed the service budget",
            ));
        }
        let extension = task_path(&state.root, &request.extension_ref)?
            .canonicalize()
            .map_err(paths::io_error)?;
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
        let schema = protocol::stream_schema(iterator.num_samples() as u64, expected);
        output.send(SchemaAsIpc::new(&schema, &IpcWriteOptions::default()).into())?;
        while lease.count < expected {
            // Reserve before sampling: a full queue must not create another native batch.
            let permit = output.reserve()?;
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
            output.reserved_batch(
                permit,
                envelope(
                    json!({"kind":"batch","batch_id":lease.count,"value":value}),
                    encoder.parts,
                )?,
            )?;
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
        if iterator.next_batch().map_err(runtime_error)?.is_some() {
            return Err(Status::internal(
                "Native shard exceeded its declared batch count",
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
    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        if *self.state.stop.borrow() {
            return Err(Status::unavailable("Gateway is stopping"));
        }
        let ticket = request.into_inner().ticket;
        if ticket.len() > MAX_BATCH_BYTES {
            return Err(Status::resource_exhausted(
                "Epoch ticket exceeds the byte budget",
            ));
        }
        let request: TaskRequest = serde_json::from_slice(&ticket)
            .map_err(|error| Status::invalid_argument(error.to_string()))?;
        let mut stopping = self.state.stop.subscribe();
        // A cancelled reader can close before its producer releases the permit.
        let permit = tokio::select! {
            _ = stopping.wait_for(|stop| *stop) => {
                return Err(Status::unavailable("Gateway is stopping"));
            }
            permit = tokio::time::timeout(
                Duration::from_secs(1), self.state.permits.clone().acquire_owned()
            ) => {
                permit.map_err(|_| Status::resource_exhausted("Sampling service is busy"))?
                    .map_err(|_| Status::unavailable("Gateway is stopping"))?
            }
        };
        let (tx, rx) = mpsc::channel(1);
        let state = self.state.clone();
        tokio::task::spawn_blocking(move || epoch(state, request, tx, permit));
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }

    async fn list_actions(
        &self,
        _: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        Ok(Response::new(Box::pin(futures::stream::once(async {
            Ok(ActionType {
                r#type: "health".into(),
                description: "Read worker readiness and counters".into(),
            })
        }))))
    }

    async fn do_action(
        &self,
        request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        let action = request.into_inner();
        if action.r#type != "health" {
            return Err(Status::unimplemented("Unknown sampling action"));
        }
        let body = serde_json::to_vec(&self.health()).unwrap();
        Ok(Response::new(Box::pin(futures::stream::once(async move {
            Ok(arrow_flight::Result { body: body.into() })
        }))))
    }

    async fn do_exchange(
        &self,
        _: Request<tonic::Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("Use DoGet for worker epoch shards"))
    }

    async fn do_put(
        &self,
        _: Request<tonic::Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        Err(Status::unimplemented(
            "Materialize graph and training-table artifacts before sampling",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use std::time::Duration;

    #[tokio::test]
    async fn admission_waits_for_a_released_slot_and_wakes_on_shutdown() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("index_metadata.json"), b"{}").unwrap();
        for shutdown in [false, true] {
            let gateway = SamplerGateway::new(root.path().to_path_buf(), 1, 1, 0).unwrap();
            let held = gateway.state.permits.clone().acquire_owned().await.unwrap();
            let request = Request::new(Ticket {
                ticket: serde_json::to_vec(&json!({
                    "kind":"epoch", "version":VERSION,
                    "extension_ref":"train_tables/unprepared",
                    "options":{
                        "stage_table":"seeds", "sampling_plan_json":"{\"layers\":[]}",
                        "num_workers":1, "worker_id":0, "shuffle":false,
                        "epoch":0, "batch_size":1
                    }
                }))
                .unwrap()
                .into(),
            });
            let waiting = gateway.do_get(request);
            tokio::pin!(waiting);
            assert!(
                tokio::time::timeout(Duration::from_millis(50), &mut waiting)
                    .await
                    .is_err()
            );
            if shutdown {
                gateway.shutdown();
                let error = waiting.await.err().unwrap();
                assert_eq!(error.code(), tonic::Code::Unavailable);
            } else {
                drop(held);
                let mut stream = waiting.await.unwrap().into_inner();
                let error = stream.next().await.unwrap().unwrap_err();
                assert_eq!(error.code(), tonic::Code::FailedPrecondition);
            }
        }
    }

    #[tokio::test]
    async fn full_output_queue_reserves_capacity_before_production() {
        let (tx, mut rx) = mpsc::channel(1);
        tx.try_send(Ok(FlightData::default())).unwrap();
        let (stop, receiver) = watch::channel(false);
        let (produced, mut production) = tokio::sync::oneshot::channel();
        let mut output = Output { tx, stop: receiver };
        let producer = tokio::task::spawn_blocking(move || {
            let permit = output.reserve().unwrap();
            produced.send(()).unwrap();
            output
                .reserved_batch(permit, envelope(json!({"kind":"batch"}), vec![]).unwrap())
                .unwrap();
        });
        assert!(
            tokio::time::timeout(Duration::from_millis(50), &mut production)
                .await
                .is_err()
        );
        rx.recv().await.unwrap().unwrap();
        tokio::time::timeout(Duration::from_secs(2), production)
            .await
            .unwrap()
            .unwrap();
        rx.recv().await.unwrap().unwrap();
        producer.await.unwrap();
        drop(stop);
    }

    #[tokio::test]
    async fn blocked_producer_wakes_on_shutdown_or_reader_cancellation() {
        for shutdown in [false, true] {
            let (tx, rx) = mpsc::channel(1);
            tx.try_send(Ok(FlightData::default())).unwrap();
            let (stop, receiver) = watch::channel(false);
            let mut output = Output { tx, stop: receiver };
            let producer = tokio::task::spawn_blocking(move || output.reserve().err().unwrap());
            if shutdown {
                stop.send_replace(true);
            } else {
                drop(rx);
            }
            let error = tokio::time::timeout(Duration::from_secs(2), producer)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(error.code(), tonic::Code::Cancelled);
        }
    }

    #[tokio::test]
    async fn available_output_capacity_does_not_restart_a_cancelled_producer() {
        for shutdown in [false, true] {
            let (tx, rx) = mpsc::channel(1);
            let (stop, receiver) = watch::channel(false);
            if shutdown {
                stop.send_replace(true);
            } else {
                drop(rx);
            }
            let mut output = Output { tx, stop: receiver };
            let error = tokio::task::spawn_blocking(move || output.reserve().err().unwrap())
                .await
                .unwrap();
            assert_eq!(error.code(), tonic::Code::Cancelled);
        }
    }
}
