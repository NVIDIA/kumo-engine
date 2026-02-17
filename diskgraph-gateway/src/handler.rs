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

pub struct ArrowFlightOutput {
    flight_sender: FlightSender,
    /// Map of rel_id -> (src_builder, dst_builder)
    edge_builders: HashMap<u16, (UInt64Builder, UInt64Builder)>,
    /// Map of node_type_id -> Sender
    feature_senders: HashMap<usize, FeatureSender>,
    /// Edge batch target chunk size in num rows
    chunk_size: usize,
    // Relation ID -> source node type ID
    relation_src_type: Vec<usize>,
    // Relation ID -> destination node type ID
    relation_dst_type: Vec<usize>,
    // Track max src/dst local-IDs per relation for validation.
    edge_max_ids: HashMap<u16, (u32, u32)>,
}

impl ArrowFlightOutput {
    pub fn with_config(
        sender: mpsc::Sender<Result<FlightData, Status>>,
        feature_senders: HashMap<usize, FeatureSender>,
        chunk_size: usize,
        relation_src_type: Vec<usize>,
        relation_dst_type: Vec<usize>,
    ) -> Self {
        Self {
            flight_sender: FlightSender::new(sender),
            edge_builders: HashMap::new(),
            feature_senders,
            chunk_size,
            relation_src_type,
            relation_dst_type,
            edge_max_ids: HashMap::new(),
        }
    }

    pub fn send_ticket_batch(&self, tickets: HashMap<usize, String>) -> Result<(), EngineError> {
        if tickets.is_empty() {
            return Ok(());
        }
        // Send an empty batch with the tickets metadata
        let batch = RecordBatch::new_empty(self.schema());
        let tickets_bytes = serde_json::to_vec(&tickets)
            .map_err(|e| EngineError::Internal(format!("failed to serialize tickets: {}", e)))?;

        self.flight_sender
            .send_batch_with_metadata(batch, tickets_bytes.into())
            .map_err(|e| EngineError::Output(e.to_string()))?;
        Ok(())
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

        let batch = RecordBatch::try_new(
            self.schema(),
            vec![
                Arc::new(row_type) as ArrayRef,
                Arc::new(src_array) as ArrayRef,
                Arc::new(dst_array) as ArrayRef,
                Arc::new(meta_type) as ArrayRef,
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
        ]))
    }

    fn send_batch(&mut self, batch: RecordBatch) -> Result<(), EngineError> {
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
        // Track max IDs for validation
        let maxes = self.edge_max_ids.entry(rel_id).or_insert((0, 0));
        maxes.0 = maxes.0.max(src_local_id);
        maxes.1 = maxes.1.max(dst_local_id);

        let chunk_size = self.chunk_size;
        let (src_b, dst_b) = self.edge_builders.entry(rel_id).or_insert_with(|| {
            (
                UInt64Builder::with_capacity(chunk_size),
                UInt64Builder::with_capacity(chunk_size),
            )
        });

        src_b.append_value(src_local_id as u64);
        dst_b.append_value(dst_local_id as u64);

        if src_b.len() >= chunk_size {
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
        if let Some(sender) = self.feature_senders.get_mut(&type_id) {
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
        // Validate edge local-ID bounds before flushing.
        for (&rel_id, &(max_src, max_dst)) in &self.edge_max_ids {
            let rel_idx = rel_id as usize;
            if let (Some(&src_type_id), Some(&dst_type_id)) = (
                self.relation_src_type.get(rel_idx),
                self.relation_dst_type.get(rel_idx),
            ) {
                let src_count = state
                    .type_states
                    .get(&src_type_id)
                    .map(|s| s.nodes.len())
                    .unwrap_or(0);
                let dst_count = state
                    .type_states
                    .get(&dst_type_id)
                    .map(|s| s.nodes.len())
                    .unwrap_or(0);
                if src_count > 0 && (max_src as usize) >= src_count {
                    return Err(EngineError::DataCorruption(format!(
                        "invalid src_local_id for rel_id={}: max_src={} >= src_count={}",
                        rel_id, max_src, src_count
                    )));
                }
                if dst_count > 0 && (max_dst as usize) >= dst_count {
                    return Err(EngineError::DataCorruption(format!(
                        "invalid dst_local_id for rel_id={}: max_dst={} >= dst_count={}",
                        rel_id, max_dst, dst_count
                    )));
                }
            }
        }

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
                    .map(|n| n.seed_index as u64)
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

            let batch = RecordBatch::try_new(
                self.schema(),
                vec![
                    Arc::new(row_type) as ArrayRef,
                    Arc::new(id_a) as ArrayRef,
                    Arc::new(id_b) as ArrayRef,
                    Arc::new(meta_type) as ArrayRef,
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
