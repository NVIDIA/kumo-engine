use crate::errors::Result;
use crate::state::SamplingState;

/// Trait for handling the output stream of the sampler.
pub trait OutputHandler {
    /// Called when an edge is traversed.
    fn handle_edge(
        &mut self,
        hop: usize,
        rel_id: u16,
        src_local_id: u32,
        dst_local_id: u32,
    ) -> Result<()>;

    /// Called before sending feature batches to define schema.
    fn handle_schema(&mut self, schema: arrow_schema::SchemaRef) -> Result<()>;

    /// Called when a batch of features is decoded.
    fn handle_feature_batch(&mut self, batch: arrow_array::RecordBatch) -> Result<()>;

    /// Called during sampling when a batch of fused features is ready for a type.
    fn handle_fused_features(
        &mut self,
        type_id: usize,
        batch: arrow_array::RecordBatch,
    ) -> Result<()>;

    /// Called at the end of sampling to flush nodes and stats.
    fn finalize(&mut self, state: &SamplingState) -> Result<()>;
}

/// A simple in-memory accumulator for testing/benchmarking.
pub struct InMemoryOutput {
    pub edges: Vec<(usize, u16, u32, u32)>, // (hop, rel, src, dst)
    pub feature_batches: Vec<arrow_array::RecordBatch>,
    pub schema: Option<arrow_schema::SchemaRef>,
    pub fused_batches: Vec<(usize, arrow_array::RecordBatch)>,
}

impl Default for InMemoryOutput {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryOutput {
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            feature_batches: Vec::new(),
            schema: None,
            fused_batches: Vec::new(),
        }
    }
}

impl OutputHandler for InMemoryOutput {
    fn handle_edge(
        &mut self,
        hop: usize,
        rel_id: u16,
        src_local_id: u32,
        dst_local_id: u32,
    ) -> Result<()> {
        self.edges.push((hop, rel_id, src_local_id, dst_local_id));
        Ok(())
    }

    fn handle_schema(&mut self, schema: arrow_schema::SchemaRef) -> Result<()> {
        self.schema = Some(schema);
        Ok(())
    }

    fn handle_feature_batch(&mut self, batch: arrow_array::RecordBatch) -> Result<()> {
        self.feature_batches.push(batch);
        Ok(())
    }

    fn handle_fused_features(
        &mut self,
        type_id: usize,
        batch: arrow_array::RecordBatch,
    ) -> Result<()> {
        self.fused_batches.push((type_id, batch));
        Ok(())
    }

    fn finalize(&mut self, _state: &SamplingState) -> Result<()> {
        Ok(())
    }
}
