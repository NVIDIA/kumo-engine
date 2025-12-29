use thiserror::Error;

pub type Result<T> = std::result::Result<T, DiskgraphError>;

/// Shared error type for the service and library surface.
#[derive(Debug, Error)]
pub enum DiskgraphError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("invalid metadata: {0}")]
    InvalidMetadata(String),
    #[error("index_metadata.json missing or unreadable")]
    MetadataMissing,
    #[error("dataset not found: {0}")]
    DatasetNotFound(String),
    #[error("dataset is busy: {0}")]
    DatasetBusy(String),
    #[error("insufficient disk space: required {required} bytes, available {available} bytes")]
    InsufficientDiskSpace { required: u64, available: u64 },
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),
}
