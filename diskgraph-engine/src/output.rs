

use crate::state::SamplingState;
use crate::errors::{Result, EngineError};

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

    /// Called at the end of sampling to flush nodes and stats.
    fn finalize(&mut self, state: &SamplingState) -> Result<()>;
}

/// A simple in-memory accumulator for testing/benchmarking.
pub struct InMemoryOutput {
    pub edges: Vec<(usize, u16, u32, u32)>,        // (hop, rel, src, dst)
}

impl InMemoryOutput {
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
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

    fn finalize(&mut self, _state: &SamplingState) -> Result<()> {
        Ok(())
    }
}
