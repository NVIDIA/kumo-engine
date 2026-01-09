use std::path::PathBuf;

/// Edge shard input bundle for a single source shard index.
#[derive(Debug, Clone)]
pub struct EdgeShardInput {
    pub relation: String,
    pub source_shard_index: usize,
    pub paths: Vec<PathBuf>,
}
