use std::path::PathBuf;

/// Abstraction over input Parquet sources. This will later support remote
/// object storage; for now, it captures a local path for streaming reads.
#[derive(Debug, Clone)]
pub enum ParquetSource {
    LocalFile(PathBuf),
    S3(S3Location),
    // Future: GCS, ADLS, in-memory buffers, etc.
}

/// S3 location and optional configuration for fetching Parquet.
#[derive(Debug, Clone)]
pub struct S3Location {
    pub bucket: String,
    pub key: String,
    pub region: Option<String>,
    pub endpoint: Option<String>,
    pub access_key_id: Option<String>,
    pub secret_access_key: Option<String>,
    pub session_token: Option<String>,
    pub virtual_hosted_style_request: Option<bool>,
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
