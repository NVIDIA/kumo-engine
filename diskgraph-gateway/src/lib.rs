use arrow_flight::{
    flight_service_server::FlightService, Action, ActionType, Criteria, FlightData,
    FlightDescriptor, FlightInfo, HandshakeRequest, HandshakeResponse, PutResult, SchemaResult,
    Ticket,
};
use futures::Stream;
use std::pin::Pin;
use std::sync::Arc;
use tonic::{Request, Response, Status};

use arrow_array::RecordBatch;
use arrow_ipc::writer::IpcWriteOptions;
use diskgraph_core::features::BatchFeatureRequest;
use diskgraph_core::sampling::BatchSamplingRequest;
use diskgraph_engine::engine::DiskGraphEngine;
use diskgraph_engine::state::StatePool;
use std::collections::HashMap;
use tokio::sync::mpsc;

mod handler;
mod registry;
mod sender;
use handler::ArrowFlightOutput;
pub use registry::TicketRegistry;
use sender::FlightSender;

fn get_env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

pub struct SamplerGateway {
    sampler: Arc<DiskGraphEngine>,
    pool: StatePool,
    registry: TicketRegistry,
    channel_capacity: usize,
    edge_batch_size: usize,
}

impl SamplerGateway {
    pub fn new(sampler: Arc<DiskGraphEngine>, pool: StatePool) -> Self {
        Self {
            sampler,
            pool,
            registry: TicketRegistry::new(),
            channel_capacity: get_env_usize("CHANNEL_CAPACITY", 100),
            edge_batch_size: get_env_usize("EDGE_BATCH_SIZE", 64 * 1024),
        }
    }
}

#[tonic::async_trait]
impl FlightService for SamplerGateway {
    type HandshakeStream =
        Pin<Box<dyn Stream<Item = Result<HandshakeResponse, Status>> + Send + 'static>>;
    type ListFlightsStream =
        Pin<Box<dyn Stream<Item = Result<FlightInfo, Status>> + Send + 'static>>;
    type DoGetStream = Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send + 'static>>;
    type DoPutStream = Pin<Box<dyn Stream<Item = Result<PutResult, Status>> + Send + 'static>>;
    type DoActionStream =
        Pin<Box<dyn Stream<Item = Result<arrow_flight::Result, Status>> + Send + 'static>>;
    type ListActionsStream =
        Pin<Box<dyn Stream<Item = Result<ActionType, Status>> + Send + 'static>>;
    type DoExchangeStream =
        Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send + 'static>>;

    async fn handshake(
        &self,
        _request: Request<tonic::Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        Err(Status::unimplemented("Handshake not implemented"))
    }

    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        Err(Status::unimplemented("ListFlights not implemented"))
    }

    async fn get_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        Err(Status::unimplemented("GetFlightInfo not implemented"))
    }

    async fn poll_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<arrow_flight::PollInfo>, Status> {
        Err(Status::unimplemented("PollFlightInfo not implemented"))
    }

    async fn get_schema(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        Err(Status::unimplemented("GetSchema not implemented"))
    }

    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let ticket = request.into_inner();

        // 1. Resolve Side-Channel Feature Tickets (Plain string ticket)
        if ticket.ticket.len() == 36 {
            if let Ok(ticket_str) = std::str::from_utf8(&ticket.ticket) {
                if let Some(rx) = self.registry.take(ticket_str) {
                    return self.handle_ticket_stream(rx).await;
                }
            }
        }

        // 2. Resolve Standard Requests
        if let Ok(req) = serde_json::from_slice::<BatchSamplingRequest>(&ticket.ticket) {
            return self.handle_sampling(req).await;
        }
        if let Ok(req) = serde_json::from_slice::<BatchFeatureRequest>(&ticket.ticket) {
            return self.handle_features(req).await;
        }
        if let Ok(req) = bincode::deserialize::<BatchSamplingRequest>(&ticket.ticket) {
            return self.handle_sampling(req).await;
        }

        Err(Status::invalid_argument("Unknown request type in ticket"))
    }

    async fn do_action(
        &self,
        request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        let action = request.into_inner();
        if action.r#type == "get_type_name_mapping" {
            let mapping = self.sampler.get_type_name_mapping();
            let body = serde_json::to_vec(&mapping)
                .map_err(|e| Status::internal(format!("Failed to serialize mapping: {}", e)))?;

            let result = arrow_flight::Result { body: body.into() };
            let stream = futures::stream::iter(vec![Ok(result)]);
            return Ok(Response::new(Box::pin(stream)));
        }
        Err(Status::invalid_argument("Unknown action type"))
    }

    async fn list_actions(
        &self,
        _request: Request<arrow_flight::Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        let actions = vec![
            ActionType {
                r#type: "get_type_name_mapping".to_string(),
                description: "Returns JSON object mapping internal IDs to names for node_types and edge_relations.".to_string(),
            }
        ];
        let stream = futures::stream::iter(actions.into_iter().map(Ok));
        Ok(Response::new(Box::pin(stream)))
    }

    async fn do_put(
        &self,
        _request: Request<tonic::Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        Err(Status::unimplemented("DoPut not implemented"))
    }

    async fn do_exchange(
        &self,
        _request: Request<tonic::Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("DoExchange not implemented"))
    }
}

impl SamplerGateway {
    async fn handle_sampling(
        &self,
        req: BatchSamplingRequest,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        let sampler = Arc::clone(&self.sampler);
        let pool = self.pool.clone();
        let registry = self.registry.clone();
        let capacity = self.channel_capacity;
        let edge_batch_size = self.edge_batch_size;
        let (tx, rx) = mpsc::channel(capacity);

        tokio::task::spawn_blocking(move || {
            let mut state = pool.acquire();

            // Side-Channel Setup
            let mut feature_senders = HashMap::new();
            let mut tickets = HashMap::new();

            if req.flags.fused_feature_fetching {
                for (type_id, _) in sampler.id_to_node_type.iter().enumerate() {
                    let (ftx, frx) = mpsc::channel(capacity);
                    let ticket_id = registry.register(frx);
                    tickets.insert(type_id, ticket_id);
                    feature_senders.insert(type_id, ftx);
                }
            }

            let mut output = ArrowFlightOutput::with_config(tx.clone(), feature_senders, edge_batch_size);
            let options = IpcWriteOptions::default();

            // Message 1: Edges Schema (No metadata)
            let schema_msg: arrow_flight::FlightData =
                arrow_flight::SchemaAsIpc::new(&output.schema(), &options).into();
            if let Err(_) = tx.blocking_send(Ok(schema_msg)) {
                return;
            }

            // Injected tickets metadata into output handler
            if let Err(e) = output.send_ticket_batch(tickets) {
                let _ = tx.blocking_send(Err(Status::internal(e.to_string())));
                return;
            }

            if let Err(e) = sampler.sample(&req, &mut state, &mut output) {
                let _ = tx.blocking_send(Err(Status::internal(e.to_string())));
            }
        });

        let output_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(output_stream)))
    }

    async fn handle_features(
        &self,
        req: BatchFeatureRequest,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        let sampler = Arc::clone(&self.sampler);
        let capacity = self.channel_capacity;
        let edge_batch_size = self.edge_batch_size;
        let (tx, rx) = mpsc::channel(capacity);

        tokio::task::spawn_blocking(move || {
            let mut output = ArrowFlightOutput::with_config(tx.clone(), HashMap::new(), edge_batch_size);
            for feature_req in req.batch {
                let node_type_id = match sampler.node_type_to_id.get(&feature_req.node_type) {
                    Some(&id) => id,
                    None => {
                        let _ = tx.blocking_send(Err(Status::invalid_argument(format!(
                            "Unknown node type: {}",
                            feature_req.node_type
                        ))));
                        return;
                    }
                };
                if let Err(e) = sampler.fetch_features_for_type(
                    node_type_id,
                    &feature_req.node_ids,
                    &feature_req.feature_names,
                    &mut output,
                ) {
                    let _ = tx.blocking_send(Err(Status::internal(e.to_string())));
                    return;
                }
            }
        });

        let output_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(output_stream)))
    }

    async fn handle_ticket_stream(
        &self,
        mut rx: mpsc::Receiver<RecordBatch>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        let capacity = self.channel_capacity;
        let (tx, response_rx) = mpsc::channel(capacity);

        tokio::spawn(async move {
            let sender = FlightSender::new(tx);
            let mut schema_sent = false;

            while let Some(batch) = rx.recv().await {
                if !schema_sent {
                    // Send schema derived from first batch
                    if let Err(e) = sender.send_schema_async(batch.schema()).await {
                        eprintln!("Failed to send side channel schema: {}", e);
                        return;
                    }
                    schema_sent = true;
                }

                if let Err(e) = sender.send_batch_async(batch).await {
                    eprintln!("Failed to send side channel batch: {}", e);
                    return;
                }
            }
        });

        let output_stream = tokio_stream::wrappers::ReceiverStream::new(response_rx);
        Ok(Response::new(Box::pin(output_stream)))
    }
}
