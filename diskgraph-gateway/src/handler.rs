use crate::registry::FeatureSender;
use crate::sender::FlightSender;
use arrow_array::builder::{ArrayBuilder, UInt64Builder};
use arrow_array::{ArrayRef, Int8Array, RecordBatch, UInt64Array};
use arrow_flight::FlightData;
use arrow_schema::{DataType, Field, Schema};
use diskgraph_engine::errors::EngineError;
use diskgraph_engine::output::OutputHandler;
use diskgraph_engine::state::SamplingState;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use tonic::Status;

const CHUNK_SIZE: usize = 64 * 1024; // 64k edges per batch

pub struct ArrowFlightOutput {
    flight_sender: FlightSender,
    /// Map of rel_id -> (src_builder, dst_builder)
    edge_builders: HashMap<u16, (UInt64Builder, UInt64Builder)>,
    // Map of node_type_id -> (Sender, Buffer)
    feature_senders: HashMap<usize, (FeatureSender, Vec<RecordBatch>)>,
    tickets: Option<HashMap<usize, String>>,
    tickets_sent: bool,
}

impl ArrowFlightOutput {
    pub fn new(sender: mpsc::Sender<Result<FlightData, Status>>) -> Self {
        Self {
            flight_sender: FlightSender::new(sender),
            edge_builders: HashMap::new(),
            feature_senders: HashMap::new(),
            tickets: None,
            tickets_sent: false,
        }
    }

    pub fn with_feature_senders(
        sender: mpsc::Sender<Result<FlightData, Status>>,
        feature_senders: HashMap<usize, FeatureSender>,
    ) -> Self {
        let feature_senders = feature_senders
            .into_iter()
            .map(|(k, v)| (k, (v, Vec::new())))
            .collect();
        Self {
            flight_sender: FlightSender::new(sender),
            edge_builders: HashMap::new(),
            feature_senders,
            tickets: None,
            tickets_sent: false,
        }
    }

    pub fn set_tickets(&mut self, tickets: HashMap<usize, String>) {
        if !tickets.is_empty() {
            self.tickets = Some(tickets);
        }
    }

    fn flush_edge_builder(&mut self, rel_id: u16) -> Result<(), EngineError> {
        let (src_b, dst_b) = if let Some(b) = self.edge_builders.get_mut(&rel_id) {
            b
        } else {
            return Ok(());
        };

        if src_b.is_empty() {
            return Ok(());
        }

        let len = src_b.len();
        let src_array = src_b.finish();
        let dst_array = dst_b.finish();

        let row_type = Int8Array::from(vec![0; len]);
        let meta_type = UInt64Array::from(vec![rel_id as u64; len]);
        let meta_val = UInt64Array::from(vec![0; len]);

        let batch = RecordBatch::try_new(
            self.schema(),
            vec![
                Arc::new(row_type) as ArrayRef,
                Arc::new(src_array) as ArrayRef,
                Arc::new(dst_array) as ArrayRef,
                Arc::new(meta_type) as ArrayRef,
                Arc::new(meta_val) as ArrayRef,
            ],
        )
        .map_err(|e| EngineError::Internal(e.to_string()))?;

        self.send_batch(batch)
    }

    pub fn schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("row_type", DataType::Int8, false),
            Field::new("id_a", DataType::UInt64, false),
            Field::new("id_b", DataType::UInt64, false),
            Field::new("meta_type", DataType::UInt64, false),
            Field::new("meta_val", DataType::UInt64, true),
        ]))
    }

    fn send_batch(&mut self, batch: RecordBatch) -> Result<(), EngineError> {
        if !self.tickets_sent {
            if let Some(tickets) = &self.tickets {
                let tickets_bytes = serde_json::to_vec(tickets)
                    .map_err(|e| EngineError::Internal(format!("failed to serialize tickets: {}", e)))?;

                self.flight_sender
                    .send_batch_with_metadata(batch, tickets_bytes.into())
                    .map_err(|e| EngineError::Output(e.to_string()))?;

                self.tickets_sent = true;
                return Ok(());
            }
        }

        self.flight_sender
            .send_batch(batch)
            .map_err(|e| EngineError::Output(e.to_string()))
    }
}

impl OutputHandler for ArrowFlightOutput {
    fn handle_edge(
        &mut self,
        _hop: usize,
        rel_id: u16,
        src_local_id: u32,
        dst_local_id: u32,
    ) -> Result<(), EngineError> {
        let (src_b, dst_b) = self.edge_builders.entry(rel_id).or_insert_with(|| {
            (
                UInt64Builder::with_capacity(CHUNK_SIZE),
                UInt64Builder::with_capacity(CHUNK_SIZE),
            )
        });

        src_b.append_value(src_local_id as u64);
        dst_b.append_value(dst_local_id as u64);

        if src_b.len() >= CHUNK_SIZE {
            self.flush_edge_builder(rel_id)?;
        }
        Ok(())
    }

    fn handle_schema(&mut self, schema: Arc<Schema>) -> Result<(), EngineError> {
        self.flight_sender
            .send_schema(schema)
            .map_err(|e| EngineError::Output(e.to_string()))
    }

    fn handle_feature_batch(&mut self, batch: RecordBatch) -> Result<(), EngineError> {
        self.send_batch(batch)
    }

    fn handle_fused_features(
        &mut self,
        type_id: usize,
        batch: RecordBatch,
    ) -> Result<(), EngineError> {
        if let Some((sender, _)) = self.feature_senders.get_mut(&type_id) {
            sender
                .blocking_send(batch)
                .map_err(|_| EngineError::Output("side channel closed".into()))?;
        } else {
            // Fallback: Send to main stream (for Baseline Feature Fetching)
            self.send_batch(batch)?;
        }
        Ok(())
    }

    fn finalize(&mut self, state: &SamplingState) -> Result<(), EngineError> {
        let rel_ids: Vec<u16> = self.edge_builders.keys().cloned().collect();
        for rel_id in rel_ids {
            self.flush_edge_builder(rel_id)?;
        }

        for (type_id, type_state) in &state.type_states {
            if type_state.nodes.is_empty() {
                continue;
            }
            let len = type_state.nodes.len();
            let row_type = Int8Array::from(vec![1; len]);
            let id_a = UInt64Array::from(
                type_state
                    .nodes
                    .iter()
                    .map(|n| n.local_id as u64)
                    .collect::<Vec<_>>(),
            );
            let id_b = UInt64Array::from(
                type_state
                    .nodes
                    .iter()
                    .map(|n| n.global_id)
                    .collect::<Vec<_>>(),
            );
            let meta_type = UInt64Array::from(vec![*type_id as u64; len]);
            let meta_val = UInt64Array::from(
                type_state
                    .nodes
                    .iter()
                    .map(|n| n.seed_index as u64)
                    .collect::<Vec<_>>(),
            );

            let batch = RecordBatch::try_new(
                self.schema(),
                vec![
                    Arc::new(row_type) as ArrayRef,
                    Arc::new(id_a) as ArrayRef,
                    Arc::new(id_b) as ArrayRef,
                    Arc::new(meta_type) as ArrayRef,
                    Arc::new(meta_val) as ArrayRef,
                ],
            )
            .map_err(|e| EngineError::Internal(e.to_string()))?;

            self.send_batch(batch)?;
        }

        let stats_bytes = serde_json::to_vec(&state.stats)
            .map_err(|e| EngineError::Internal(format!("failed to serialize stats: {}", e)))?;

        let metadata_msg = FlightData {
            app_metadata: stats_bytes.into(),
            data_header: Default::default(),
            data_body: Default::default(),
            flight_descriptor: None,
        };

        self.flight_sender
            .send_raw(metadata_msg)
            .map_err(|_| EngineError::Output("channel closed".into()))?;

        Ok(())
    }
}
