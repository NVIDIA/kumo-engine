//! Centralized error types for diskgraph-ingest.
//!
//! Error types are organized by domain:
//! - `key`: Primary/foreign key parsing errors
//! - `coercion`: FK-PK type coercion errors
//! - `convert`: Kumo config conversion errors

use thiserror::Error;

use crate::coercion::CoercionKind;
use crate::graph_config::IngestedType;

// ============================================================================
// Key Parsing Errors (runtime)
// ============================================================================

/// Errors during primary/foreign key value parsing at runtime.
#[derive(Clone, Copy, Debug, Error)]
pub enum KeyParseError {
    #[error("type mismatch")]
    TypeMismatch,
    #[error("unsupported type")]
    UnsupportedType,
    #[error("length mismatch")]
    LengthMismatch,
    #[error("empty")]
    Empty,
    #[error("too large")]
    TooLarge,
    #[error("not integer")]
    NotInteger,
    #[error("not base 10")]
    NotBase10,
    #[error("not utf8")]
    NotUtf8,
    #[error("invalid uuid")]
    InvalidUuid,
    #[error("invalid uuid 256")]
    InvalidUuid256,
    #[error("invalid ip")]
    InvalidIp,
}

pub type KeyParseResult<T> = std::result::Result<T, KeyParseError>;

// ============================================================================
// Coercion Errors (config validation)
// ============================================================================

/// Errors during FK-PK type coercion validation.
#[derive(Debug, Error)]
pub enum CoercionError {
    #[error(
        "type mismatch in strict mode: {fk_table}.{fk_col} ({fk_type:?}) -> {pk_table}.{pk_col} ({pk_type:?})\n\
         Hint: Use coercion_mode: \"lossless\" to allow safe coercions"
    )]
    StrictModeMismatch {
        fk_table: String,
        fk_col: String,
        fk_type: IngestedType,
        pk_table: String,
        pk_col: String,
        pk_type: IngestedType,
    },

    #[error(
        "lossy coercion not allowed: {fk_table}.{fk_col} ({fk_type:?}) -> {pk_table}.{pk_col} ({pk_type:?})\n\
         Reason: {reason}"
    )]
    LossyCoercion {
        fk_table: String,
        fk_col: String,
        fk_type: IngestedType,
        pk_table: String,
        pk_col: String,
        pk_type: IngestedType,
        reason: String,
    },

    #[error(
        "incompatible types: {fk_table}.{fk_col} ({fk_type:?}) -> {pk_table}.{pk_col} ({pk_type:?})\n\
         No conversion path exists between {fk_type:?} and {pk_type:?}"
    )]
    IncompatibleTypes {
        fk_table: String,
        fk_col: String,
        fk_type: IngestedType,
        pk_table: String,
        pk_col: String,
        pk_type: IngestedType,
    },
}

/// Warning for conditional coercions (not an error, but logged).
#[derive(Debug)]
pub struct CoercionWarning {
    pub fk_table: String,
    pub fk_col: String,
    pub fk_type: IngestedType,
    pub pk_table: String,
    pub pk_col: String,
    pub pk_type: IngestedType,
    pub kind: CoercionKind,
}

impl std::fmt::Display for CoercionWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "conditional coercion: {}.{} ({:?}) to {}.{} ({:?}) - invalid values will skip edge creation",
            self.fk_table, self.fk_col, self.fk_type, self.pk_table, self.pk_col, self.pk_type
        )
    }
}

// ============================================================================
// Config Conversion Errors
// ============================================================================

/// Errors during Kumo config to DiskGraph config conversion.
#[derive(Debug, Error)]
pub enum ConvertError {
    #[error("table '{table}' is missing required field: {field}")]
    MissingField { table: String, field: String },

    #[error("table '{table}' column '{column}' has unknown dtype: {dtype}")]
    UnknownDtype {
        table: String,
        column: String,
        dtype: String,
    },

    #[error("table '{table}' has no columns")]
    NoColumns { table: String },
}
