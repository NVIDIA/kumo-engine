use std::path::PathBuf;

/// Abstraction over input Parquet sources. This will later support remote
/// object storage; for now, it captures a local path for streaming reads.
#[derive(Debug, Clone)]
pub enum ParquetSource {
    LocalFile(PathBuf),
    // Future: S3, GCS, ADLS, in-memory buffers, etc.
}

/// Edge shard input bundle for a single source node shard.
#[derive(Debug, Clone)]
pub struct EdgeShardInput {
    pub relation: String,
    pub source_shard_index: usize,
    pub parquet: ParquetSource,
    /// Whether this edge shard carries dest_sort_key in its schema.
    pub has_dest_sort_key: bool,
}

/// Node shard input bundle.
#[derive(Debug, Clone)]
pub struct NodeShardInput {
    pub node_type: String,
    pub shard_index: usize,
    pub parquet: ParquetSource,
}
