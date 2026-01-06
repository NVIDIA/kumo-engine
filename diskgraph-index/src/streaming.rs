use std::path::PathBuf;
use diskgraph_core::metadata::EdgeStorageFormat;

/// Abstraction over input Parquet sources. This only supports local files.
#[derive(Debug, Clone)]
pub enum ParquetSource {
    LocalFile(PathBuf),
}

/// Edge shard input bundle for a single source shard index.
#[derive(Debug, Clone)]
pub struct EdgeShardInput {
    pub relation: String,
    pub source_shard_index: usize,
    pub parquet: ParquetSource,
    /// Whether this edge shard carries dest_sort_key in its schema.
    pub has_dest_sort_key: bool,
    pub format: EdgeStorageFormat,
}
