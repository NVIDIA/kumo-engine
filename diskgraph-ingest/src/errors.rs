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

    #[error(
        "table '{table}' column '{column}': dtype '{dtype}' with stype 'sequence' is not supported. \
         The column likely contains JSON-encoded float arrays (e.g. \"[0.1, -0.2, ...]\") — \
         use dtype 'floatlist' instead"
    )]
    StringSequenceUnsupported {
        table: String,
        column: String,
        dtype: String,
    },

    #[error(
        "table '{table}' column '{column}': multicategorical columns with dtype '{dtype:?}' \
         are not supported. Use a string (comma/pipe-delimited), list<utf8>, list<int64>, \
         or list<float32/float64/float16> dtype."
    )]
    MulticategoricalOnUnsupportedInput {
        table: String,
        column: String,
        dtype: IngestedType,
    },

    #[error(
        "table '{table}' column '{column}': declared multicategorical_delimiter \
         {declared:?} does not match the source data — only {hits} of {total} \
         sampled non-empty rows contain it (need at least {required}). \
         Sample row that did not contain the delimiter: {sample:?}. \
         Either fix the delimiter on the column config or omit it to let \
         ingest auto-detect from the data."
    )]
    MulticategoricalDelimiterMismatch {
        table: String,
        column: String,
        declared: String,
        hits: usize,
        total: usize,
        required: usize,
        sample: String,
    },

    #[error("{0}")]
    InferenceError(String),

    #[error("table '{table}': no source files found: {message}")]
    SourcePaths { table: String, message: String },

    #[error("table '{table}': source path I/O or transport error: {message}")]
    SourcePathsIO { table: String, message: String },

    #[error(
        "table '{table}' column '{column}': conflicting `timestamp_format` values \
         declared in different positions of the table config: {first:?} vs {second:?}. \
         Production wire format puts the same time column in both `cols[]` and \
         `time_col` (or `end_time_col`); when both carry a `timestamp_format` the \
         values must match (one side may be omitted)."
    )]
    TimestampFormatConflict {
        table: String,
        column: String,
        first: String,
        second: String,
    },

    #[error(
        "table '{table}' column '{column}': declared `timestamp_format` {format:?} \
         is not a valid chrono format string. Use a recognized strftime specifier \
         (e.g. \"%Y-%m-%d %H:%M:%S\", \"%y%m%d %H:%M:%S\", \"%d-%b-%Y\")."
    )]
    InvalidTimestampFormat {
        table: String,
        column: String,
        format: String,
    },
}
