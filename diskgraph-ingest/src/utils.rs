//! Shared utility functions for graph configuration processing.

use std::collections::HashSet;

use crate::graph_config::{ColumnSpec, IngestedType};

/// Check if an IngestedType is numeric (suitable for edge weight columns).
/// Includes all integer, float, and decimal types.
pub fn is_numeric_weight_dtype(dtype: &IngestedType) -> bool {
    matches!(
        dtype,
        IngestedType::Int8
            | IngestedType::Int16
            | IngestedType::Int32
            | IngestedType::Int64
            | IngestedType::UInt8
            | IngestedType::UInt16
            | IngestedType::UInt32
            | IngestedType::UInt64
            | IngestedType::Float32
            | IngestedType::Float64
            | IngestedType::Decimal128
            | IngestedType::Decimal256
    )
}

/// Check if a column has a weight-like name AND numeric type.
/// Weight-like names: "weight", "score", "strength", "prob", "probability"
pub fn is_weight_like_named(col: &ColumnSpec) -> bool {
    if !is_numeric_weight_dtype(&col.ingested_dtype) {
        return false;
    }
    let name = col.name.to_lowercase();
    matches!(
        name.as_str(),
        "weight" | "score" | "strength" | "prob" | "probability"
    )
}

/// Check if a table's columns are compatible with edge table classification.
///
/// Edge table criteria:
/// - All non-FK columns must be either:
///   - The create_time column (at most ONE)
///   - A numeric column (at most ONE, preferring weight-like names)
///   - Nothing else
///
/// # Arguments
/// * `columns` - All columns in the table
/// * `fkey_col_names` - Names of foreign key columns
/// * `create_time_col` - Name of the create_time column, if any
///
/// # Returns
/// `true` if the columns are compatible with edge table classification
pub fn is_edge_table_compatible(
    columns: &[ColumnSpec],
    fkey_col_names: &HashSet<&str>,
    create_time_col: Option<&str>,
) -> bool {
    let mut seen_time = false;
    let mut seen_weight = false;

    for col in columns {
        // Skip FK columns
        if fkey_col_names.contains(col.name.as_str()) {
            continue;
        }

        // Check create_time column (allow at most ONE)
        if create_time_col == Some(col.name.as_str()) {
            if seen_time {
                return false;
            }
            seen_time = true;
            continue;
        }

        // Check weight-like named columns first (at most ONE numeric total)
        if is_weight_like_named(col) {
            if seen_weight {
                return false;
            }
            seen_weight = true;
            continue;
        }

        // Check any numeric column (at most ONE numeric total)
        if is_numeric_weight_dtype(&col.ingested_dtype) {
            if seen_weight {
                return false;
            }
            seen_weight = true;
            continue;
        }

        // Non-FK, non-time, non-numeric column found -> not an edge table
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_col(name: &str, dtype: IngestedType) -> ColumnSpec {
        ColumnSpec {
            name: name.to_string(),
            source_dtype: None,
            ingested_dtype: dtype,
            fixed_len: None,
            timestamp_format: None,
            timestamp_unit: None,
            precision: None,
            scale: None,
            semantic_type: None,
        }
    }

    #[test]
    fn test_is_numeric_weight_dtype() {
        assert!(is_numeric_weight_dtype(&IngestedType::Int64));
        assert!(is_numeric_weight_dtype(&IngestedType::Float32));
        assert!(is_numeric_weight_dtype(&IngestedType::Decimal128));
        assert!(!is_numeric_weight_dtype(&IngestedType::String));
        assert!(!is_numeric_weight_dtype(&IngestedType::Bool));
        assert!(!is_numeric_weight_dtype(&IngestedType::Timestamp));
    }

    #[test]
    fn test_is_weight_like_named() {
        let weight_col = make_col("weight", IngestedType::Float64);
        let score_col = make_col("Score", IngestedType::Float32);
        let amount_col = make_col("amount", IngestedType::Float64);
        let name_col = make_col("weight", IngestedType::String);

        assert!(is_weight_like_named(&weight_col));
        assert!(is_weight_like_named(&score_col));
        assert!(!is_weight_like_named(&amount_col)); // numeric but not weight-named
        assert!(!is_weight_like_named(&name_col)); // weight-named but not numeric
    }

    #[test]
    fn test_edge_table_compatible_simple() {
        // Simple edge table: 2 FKs + timestamp
        let columns = vec![
            make_col("user_id", IngestedType::Int64),
            make_col("item_id", IngestedType::Int64),
            make_col("timestamp", IngestedType::Timestamp),
        ];
        let fkeys: HashSet<&str> = ["user_id", "item_id"].into_iter().collect();

        assert!(is_edge_table_compatible(
            &columns,
            &fkeys,
            Some("timestamp")
        ));
    }

    #[test]
    fn test_edge_table_compatible_with_weight() {
        // Edge table with weight column
        let columns = vec![
            make_col("user_id", IngestedType::Int64),
            make_col("item_id", IngestedType::Int64),
            make_col("timestamp", IngestedType::Timestamp),
            make_col("weight", IngestedType::Float32),
        ];
        let fkeys: HashSet<&str> = ["user_id", "item_id"].into_iter().collect();

        assert!(is_edge_table_compatible(
            &columns,
            &fkeys,
            Some("timestamp")
        ));
    }

    #[test]
    fn test_edge_table_incompatible_two_numeric() {
        // Not an edge table: 2 FKs + 2 numeric columns
        let columns = vec![
            make_col("user_id", IngestedType::Int64),
            make_col("item_id", IngestedType::Int64),
            make_col("amount", IngestedType::Float64),
            make_col("quantity", IngestedType::Int32),
        ];
        let fkeys: HashSet<&str> = ["user_id", "item_id"].into_iter().collect();

        assert!(!is_edge_table_compatible(&columns, &fkeys, None));
    }

    #[test]
    fn test_edge_table_incompatible_string_column() {
        // Not an edge table: has a string column
        let columns = vec![
            make_col("user_id", IngestedType::Int64),
            make_col("item_id", IngestedType::Int64),
            make_col("note", IngestedType::String),
        ];
        let fkeys: HashSet<&str> = ["user_id", "item_id"].into_iter().collect();

        assert!(!is_edge_table_compatible(&columns, &fkeys, None));
    }
}
