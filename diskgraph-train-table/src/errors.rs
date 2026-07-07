use thiserror::Error;

#[derive(Error, Debug)]
pub enum TTGenError {
    #[error("Schema error: {0}")]
    SchemaError(String),

    #[error("Invalid plan: {0}")]
    InvalidPlan(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parquet error: {0}")]
    Parquet(String),

    #[error("System error: {0}")]
    SystemError(String),
}

impl From<parquet::errors::ParquetError> for TTGenError {
    fn from(err: parquet::errors::ParquetError) -> Self {
        TTGenError::Parquet(err.to_string())
    }
}

pub type TTGenResult<T> = std::result::Result<T, TTGenError>;
