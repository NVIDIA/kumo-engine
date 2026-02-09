use diskgraph_sampler::output::OutputHandler;
use diskgraph_sampler::state::SamplingState;
use diskgraph_sampler::errors::SamplerError;
use arrow_array::{RecordBatch, UInt64Array, Int8Array, ArrayRef};
use arrow_schema::{DataType, Field, Schema};
use arrow_flight::FlightData;
use arrow_ipc::writer::{IpcDataGenerator, DictionaryTracker};
use arrow_ipc::writer::IpcWriteOptions;
use tonic::Status;
use tokio::sync::mpsc;
use std::sync::Arc;
use std::collections::HashMap;

const CHUNK_SIZE: usize = 64 * 1024; // 64k edges per batch

pub struct ArrowFlightOutput {
    sender: mpsc::Sender<Result<FlightData, Status>>,
    edge_buffers: HashMap<u16, EdgeBuffer>,
    options: IpcWriteOptions,
}

struct EdgeBuffer {
    src: Vec<u32>,
    dst: Vec<u32>,
    layer_idx: Vec<u32>, // Optional: if we want to track hop in the edge list
}

impl EdgeBuffer {
    fn new() -> Self {
        Self {
            src: Vec::with_capacity(CHUNK_SIZE),
            dst: Vec::with_capacity(CHUNK_SIZE),
            layer_idx: Vec::with_capacity(CHUNK_SIZE),
        }
    }

    fn push(&mut self, src: u32, dst: u32, layer: u32) {
        self.src.push(src);
        self.dst.push(dst);
        self.layer_idx.push(layer);
    }

    fn len(&self) -> usize {
        self.src.len()
    }

    fn clear(&mut self) {
        self.src.clear();
        self.dst.clear();
        self.layer_idx.clear();
    }
}

impl ArrowFlightOutput {
    pub fn new(sender: mpsc::Sender<Result<FlightData, Status>>) -> Self {
        Self {
            sender,
            edge_buffers: HashMap::new(),
            options: IpcWriteOptions::default(),
        }
    }

    fn flush_edge_buffer(&mut self, rel_id: u16) -> Result<(), SamplerError> {
        let mut buffer_contents = EdgeBuffer::new();
        if let Some(buffer) = self.edge_buffers.get_mut(&rel_id) {
            if buffer.len() == 0 { return Ok(()); }
            // Swap buffer out to release borrow on self
            std::mem::swap(buffer, &mut buffer_contents);
        } else {
            return Ok(());
        }

        let len = buffer_contents.len();

        let row_type = Int8Array::from(vec![0; len]);
        let id_a = UInt64Array::from(buffer_contents.src.iter().map(|&x| x as u64).collect::<Vec<_>>());
        let id_b = UInt64Array::from(buffer_contents.dst.iter().map(|&x| x as u64).collect::<Vec<_>>());
        let meta_type = UInt64Array::from(vec![rel_id as u64; len]);
        let meta_val = UInt64Array::from(vec![0; len]);

        let batch = RecordBatch::try_new(
            self.schema(),
            vec![
                Arc::new(row_type) as ArrayRef,
                Arc::new(id_a) as ArrayRef,
                Arc::new(id_b) as ArrayRef,
                Arc::new(meta_type) as ArrayRef,
                Arc::new(meta_val) as ArrayRef,
            ],
        ).map_err(|e| SamplerError::Internal(e.to_string()))?;

        self.send_batch(batch)?;

        // We could reuse the buffer capacity by swapping back, but for simplicity we just let it drop.
        // Or better: put it back.
        if let Some(buffer) = self.edge_buffers.get_mut(&rel_id) {
             buffer_contents.clear();
             *buffer = buffer_contents;
        }
        Ok(())
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

    fn send_batch(&mut self, batch: RecordBatch) -> Result<(), SamplerError> {
        let generator = IpcDataGenerator::default();
        let mut tracker = DictionaryTracker::new(false);
        let (_dictionaries, encoded_data) = generator.encoded_batch(&batch, &mut tracker, &self.options)
             .map_err(|e: arrow_schema::ArrowError| SamplerError::Internal(e.to_string()))?;

        let flight_data: FlightData = encoded_data.into();
        self.sender.blocking_send(Ok(flight_data))
            .map_err(|_| SamplerError::Output("channel closed".into()))?;
        Ok(())
    }
}

impl OutputHandler for ArrowFlightOutput {
    fn handle_edge(
        &mut self,
        hop: usize,
        rel_id: u16,
        src_local_id: u32,
        dst_local_id: u32,
    ) -> Result<(), SamplerError> {
        let buffer = self.edge_buffers.entry(rel_id).or_insert_with(EdgeBuffer::new);
        buffer.push(src_local_id, dst_local_id, hop as u32);

        if buffer.len() >= CHUNK_SIZE {
            // Need to drop borrow of buffer before flushing which borrows self
            // We can check length, but calling flush needs self.
            // But we can't call flush while `buffer` ref exists.
            // So we return early and call flush?
        }
        // Correct logic:
        let needs_flush = buffer.len() >= CHUNK_SIZE;
        if needs_flush {
             // Drop borrow
        } else {
             return Ok(());
        }

        // Flush
        self.flush_edge_buffer(rel_id)?;
        Ok(())
    }

    fn finalize(&mut self, state: &SamplingState) -> Result<(), SamplerError> {
        // 1. Flush remaining edges
        let rel_ids: Vec<u16> = self.edge_buffers.keys().cloned().collect();
        for rel_id in rel_ids {
            self.flush_edge_buffer(rel_id)?;
        }

        // 2. Send Node Mappings (Global IDs)
        // state.type_states maps type_id -> nodes[global_id]
        for (type_id, type_state) in &state.type_states {
            if type_state.nodes.is_empty() { continue; }

            let len = type_state.nodes.len();
            let row_type = Int8Array::from(vec![1; len]);
            let id_a = UInt64Array::from(vec![0; len]); // Not used for nodes
            let id_b = UInt64Array::from(type_state.nodes.iter().map(|n| n.global_id).collect::<Vec<_>>()); // Global ID
            let meta_type = UInt64Array::from(vec![*type_id as u64; len]);
            let meta_val = UInt64Array::from(vec![0; len]);

            let batch = RecordBatch::try_new(
                self.schema(),
                vec![
                    Arc::new(row_type) as ArrayRef,
                    Arc::new(id_a) as ArrayRef,
                    Arc::new(id_b) as ArrayRef,
                    Arc::new(meta_type) as ArrayRef,
                    Arc::new(meta_val) as ArrayRef,
                ],
            ).map_err(|e| SamplerError::Internal(e.to_string()))?;

            self.send_batch(batch)?;
        }

        // 3. Serialize Stats (Metadata)
        let stats_bytes = bincode::serialize(&state.stats)
            .map_err(|e| SamplerError::Internal(format!("failed to serialize stats: {}", e)))?;

        let metadata_msg = FlightData {
            app_metadata: stats_bytes.into(),
            data_header: Default::default(),
            data_body: Default::default(),
            flight_descriptor: None,
        };

        self.sender.blocking_send(Ok(metadata_msg))
            .map_err(|_| SamplerError::Output("channel closed".into()))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use diskgraph_sampler::state::{SamplingState, SamplingStats, TypeState};
    use gxhash::HashMap as GxHashMap;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_edge_buffering() {
        let (tx, mut rx) = mpsc::channel(10);
        
        tokio::task::spawn_blocking(move || {
            let mut handler = ArrowFlightOutput::new(tx);

            handler.handle_edge(1, 1, 10, 20).unwrap();
            handler.handle_edge(1, 1, 11, 21).unwrap();
            handler.handle_edge(2, 2, 30, 40).unwrap();

            assert_eq!(handler.edge_buffers.get(&1).unwrap().len(), 2);
            assert_eq!(handler.edge_buffers.get(&2).unwrap().len(), 1);

            let state = SamplingState {
                type_states: GxHashMap::default(),
                dst_scratch: Vec::new(),
                key_scratch: Vec::new(),
                reservoir_scratch: Vec::new(),
                stats: SamplingStats::default(),
            };
            handler.finalize(&state).unwrap();
        });

        // Expect 2 edges batches + 1 stats msg
        let msg1 = rx.recv().await.unwrap().unwrap();
        let msg2 = rx.recv().await.unwrap().unwrap();
        let msg3 = rx.recv().await.unwrap().unwrap();
        
        assert!(msg1.data_header.len() > 0);
        assert!(msg2.data_header.len() > 0);
        assert!(msg3.app_metadata.len() > 0);
    }

    #[tokio::test]
    async fn test_node_flushing() {
        let (tx, mut rx) = mpsc::channel(10);
        
        tokio::task::spawn_blocking(move || {
            let mut handler = ArrowFlightOutput::new(tx);

            let mut type_states = GxHashMap::default();
            let mut t0 = TypeState::new();
            t0.nodes.push(diskgraph_sampler::state::NodeRecord { global_id: 100, local_id: 0, seed_index: 0 });
            t0.nodes.push(diskgraph_sampler::state::NodeRecord { global_id: 101, local_id: 1, seed_index: 0 });
            type_states.insert(0, t0);

            let state = SamplingState {
                type_states,
                dst_scratch: Vec::new(),
                key_scratch: Vec::new(),
                reservoir_scratch: Vec::new(),
                stats: SamplingStats::default(),
            };

            handler.finalize(&state).unwrap();
        });

        // Expect: 1 node batch + 1 stats msg
        let msg1 = rx.recv().await.unwrap().unwrap();
        let msg2 = rx.recv().await.unwrap().unwrap();

        assert!(msg1.data_header.len() > 0);
        assert!(msg2.app_metadata.len() > 0);
    }
}

