use std::path::PathBuf;
use diskgraph_core::metadata::EdgeStorageFormat;

/// Abstraction over input Parquet sources. This supports local files and remote
/// object storage backends.
#[derive(Debug, Clone)]
pub enum ParquetSource {
    LocalFile(PathBuf),
    S3(S3Location),
    Gcs(GcsLocation),
    Adls(AdlsLocation),
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

/// GCS location and optional configuration.
#[derive(Debug, Clone)]
pub struct GcsLocation {
    pub bucket: String,
    pub key: String,
    pub service_account_path: Option<String>,
    pub service_account_key: Option<String>,
    pub application_credentials_path: Option<String>,
}

/// ADLS v2 / Azure Blob location and optional configuration.
#[derive(Debug, Clone)]
pub struct AdlsLocation {
    pub account: String,
    pub container: String,
    pub key: String,
    pub access_key: Option<String>,
    pub sas_key: Option<String>,
    pub bearer_token: Option<String>,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub tenant_id: Option<String>,
    pub endpoint: Option<String>,
}
