// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Type mapping functions for Kumo to DiskGraph conversion.

use crate::graph_config::{IngestedType, SemanticType, TimestampUnit};

/// Map a Kumo dtype string to a DiskGraph IngestedType.
///
/// Returns `Ok(IngestedType)` for supported types, `Err(dtype)` for
/// unsupported or unknown types that should be reported as errors.
pub fn map_kumo_dtype(dtype: &str) -> Result<IngestedType, &str> {
    match dtype {
        // Boolean
        "bool" => Ok(IngestedType::Bool),

        // Signed integers
        "int" | "int64" => Ok(IngestedType::Int64),
        "int32" => Ok(IngestedType::Int32),
        "int16" => Ok(IngestedType::Int16),

        // Unsigned integers (kumo uses "byte" for uint8)
        "byte" => Ok(IngestedType::UInt8),

        // Floating point
        "float" | "float32" => Ok(IngestedType::Float32),
        "float64" => Ok(IngestedType::Float64),

        // String
        "string" => Ok(IngestedType::String),

        // Date/Time
        "date" => Ok(IngestedType::Timestamp),
        "time" => Ok(IngestedType::Timestamp),

        // Time delta stored as int64 nanoseconds
        "timedelta" => Ok(IngestedType::Int64),

        // Array types. "intlist" stays mapped to int64_array for backward
        // compat with existing Kumo configs; the narrower widths are opt-in.
        "floatlist" => Ok(IngestedType::Float32Array),
        "intlist" | "int64_list" => Ok(IngestedType::Int64Array),
        "int32_list" => Ok(IngestedType::Int32Array),
        "int16_list" => Ok(IngestedType::Int16Array),
        "int8_list" => Ok(IngestedType::Int8Array),
        "stringlist" => Ok(IngestedType::StringArray),

        // Binary
        "binary" => Ok(IngestedType::Bytes),

        // Unsupported types — error out so the caller knows
        _ => Err(dtype),
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
        "latlong" => Some(SemanticType::Latlong),
        "image" => Some(SemanticType::ImageUrl),
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
