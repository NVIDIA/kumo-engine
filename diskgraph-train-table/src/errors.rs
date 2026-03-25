use thiserror::Error;

#[derive(Error, Debug)]
pub enum TTGenError {
    #[error("Schema error: {0}")]
    SchemaError(String),

    #[error("Invalid plan: {0}")]
    InvalidPlan(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("System error: {0}")]
    SystemError(String),
}

pub type TTGenResult<T> = std::result::Result<T, TTGenError>;
