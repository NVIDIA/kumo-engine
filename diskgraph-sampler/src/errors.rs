use thiserror::Error;

#[derive(Error, Debug)]
pub enum SamplerError {
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Data corruption or missing artifacts: {0}")]
    DataCorruption(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Output error: {0}")]
    Output(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, SamplerError>;
