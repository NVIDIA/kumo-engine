//! Type mapping functions for Kumo to DiskGraph conversion.

use crate::graph_config::{IngestedType, SemanticType, TimestampUnit};

/// Map a Kumo dtype string to a DiskGraph IngestedType.
///
/// Returns None for unknown or unsupported types.
pub fn map_kumo_dtype(dtype: &str) -> Option<IngestedType> {
    match dtype {
        // Boolean
        "bool" => Some(IngestedType::Bool),

        // Signed integers
        "int" | "int64" => Some(IngestedType::Int64),
        "int32" => Some(IngestedType::Int32),
        "int16" => Some(IngestedType::Int16),

        // Unsigned integers (kumo uses "byte" for uint8)
        "byte" => Some(IngestedType::UInt8),

        // Floating point
        "float" | "float32" => Some(IngestedType::Float32),
        "float64" => Some(IngestedType::Float64),

        // String
        "string" => Some(IngestedType::String),

        // Date/Time
        "date" => Some(IngestedType::Date),
        "time" => Some(IngestedType::Timestamp),

        // Time delta stored as int64 nanoseconds
        "timedelta" => Some(IngestedType::Int64),

        // Array types
        "floatlist" => Some(IngestedType::Float32Array),
        // intlist and stringlist don't have native array support, fallback to string
        "intlist" => Some(IngestedType::String),
        "stringlist" => Some(IngestedType::String),

        // Binary
        "binary" => Some(IngestedType::Bytes),

        // Unsupported or unknown
        "unsupported" => None,
        _ => None,
    }
}

/// Map a Kumo stype string to a DiskGraph SemanticType.
///
/// Returns None for unknown stypes (let inference decide).
pub fn map_kumo_stype(stype: &str) -> Option<SemanticType> {
    match stype {
        "numerical" => Some(SemanticType::Numerical),
        "categorical" => Some(SemanticType::Categorical),
        "multicategorical" => Some(SemanticType::Multicategorical),
        "ID" => Some(SemanticType::Id),
        "text" => Some(SemanticType::Text),
        "timestamp" => Some(SemanticType::Timestamp),
        "sequence" => Some(SemanticType::Sequence),
        "image" => Some(SemanticType::Text), // image URLs stored as text
        "unsupported" => Some(SemanticType::Unsupported),
        _ => None,
    }
}

/// Map a Kumo timestamp unit string to a DiskGraph TimestampUnit.
///
/// Returns None for unknown units.
pub fn map_timestamp_unit(unit: &str) -> Option<TimestampUnit> {
    match unit {
        "s" | "seconds" | "sec" => Some(TimestampUnit::Seconds),
        "ms" | "millis" | "milliseconds" => Some(TimestampUnit::Millis),
        "us" | "micros" | "microseconds" => Some(TimestampUnit::Micros),
        "ns" | "nanos" | "nanoseconds" => Some(TimestampUnit::Nanos),
        _ => None,
    }
}
