use std::pin::Pin;
use std::sync::Arc;
use futures::Stream;
use tonic::{Request, Response, Status};
use arrow_flight::{
    flight_service_server::FlightService, Ticket, FlightData,
    SchemaResult, Action, ActionType,
    FlightDescriptor, FlightInfo, HandshakeRequest, HandshakeResponse,
    PutResult, Criteria,
};

use arrow_ipc::writer::IpcWriteOptions;
use diskgraph_sampler::sampler::Sampler;
use diskgraph_sampler::state::StatePool;
use diskgraph_core::sampling::BatchSamplingRequest;
use tokio::sync::mpsc;

mod handler;
use handler::ArrowFlightOutput;

pub struct SamplerGateway {
    sampler: Arc<Sampler>,
    pool: StatePool,
}

impl SamplerGateway {
    pub fn new(sampler: Arc<Sampler>, pool: StatePool) -> Self {
        Self { sampler, pool }
    }
}

#[tonic::async_trait]
impl FlightService for SamplerGateway {
    type HandshakeStream = Pin<Box<dyn Stream<Item = Result<HandshakeResponse, Status>> + Send + 'static>>;
    type ListFlightsStream = Pin<Box<dyn Stream<Item = Result<FlightInfo, Status>> + Send + 'static>>;
    type DoGetStream = Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send + 'static>>;
    type DoPutStream = Pin<Box<dyn Stream<Item = Result<PutResult, Status>> + Send + 'static>>;
    type DoActionStream = Pin<Box<dyn Stream<Item = Result<arrow_flight::Result, Status>> + Send + 'static>>;
    type ListActionsStream = Pin<Box<dyn Stream<Item = Result<ActionType, Status>> + Send + 'static>>;
    type DoExchangeStream = Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send + 'static>>;

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

        // Deserialize request from ticket
        let sampling_req: BatchSamplingRequest = bincode::deserialize(&ticket.ticket)
            .or_else(|_| serde_json::from_slice(&ticket.ticket))
            .map_err(|e| Status::invalid_argument(format!("Failed to deserialize request (tried Bincode & JSON): {}", e)))?;

        let sampler = Arc::clone(&self.sampler);
        let pool = self.pool.clone();

        // Channel for streaming results
        let (tx, rx) = mpsc::channel(100);

        // Spawn blocking sampling task
        tokio::task::spawn_blocking(move || {
            let mut state = pool.acquire();
            let mut output = ArrowFlightOutput::new(tx.clone());

            // Explicitly send schema as the first message
            let options = IpcWriteOptions::default();
            let schema_flight_data = arrow_flight::SchemaAsIpc::new(&output.schema(), &options).into();
            if let Err(_) = tx.blocking_send(Ok(schema_flight_data)) {
                 return;
            }

            if let Err(e) = sampler.sample(&sampling_req, &mut state, &mut output) {
                // If sampling fails, try to send error status
                // We ignore send error here (client might have disconnected)
                let _ = tx.blocking_send(Err(Status::internal(e.to_string())));
            }
        });

        // Convert receiver to stream
        let output_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(output_stream)))
    }

    async fn do_put(
        &self,
        _request: Request<tonic::Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        Err(Status::unimplemented("DoPut not implemented"))
    }

    async fn do_action(
        &self,
        request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        let action = request.into_inner();
        if action.r#type == "get_schema_mapping" {
            let mapping = self.sampler.get_schema_mapping();
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
                r#type: "get_schema_mapping".to_string(),
                description: "Returns JSON object mapping internal IDs to names for node_types and edge_relations.".to_string(),
            }
        ];
        let stream = futures::stream::iter(actions.into_iter().map(Ok));
        Ok(Response::new(Box::pin(stream)))
    }

    async fn do_exchange(
        &self,
        _request: Request<tonic::Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("DoExchange not implemented"))
    }
}
