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

impl TTGenError {
    /// Add worker context without changing whether the failure is caused by
    /// user data, configuration, or the system.
    pub fn with_context(self, context: impl AsRef<str>) -> Self {
        let context = context.as_ref();
        match self {
            Self::SchemaError(message) => Self::SchemaError(format!("{context}: {message}")),
            Self::InvalidPlan(message) => Self::InvalidPlan(format!("{context}: {message}")),
            Self::Io(error) => Self::Io(std::io::Error::new(
                error.kind(),
                format!("{context}: {error}"),
            )),
            Self::Parquet(message) => Self::Parquet(format!("{context}: {message}")),
            Self::SystemError(message) => Self::SystemError(format!("{context}: {message}")),
        }
    }
}

pub type TTGenResult<T> = std::result::Result<T, TTGenError>;
