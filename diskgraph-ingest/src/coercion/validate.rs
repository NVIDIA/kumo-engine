//! Config-time validation of FK-PK type compatibility.
//!
//! This module validates FK-PK type pairs at config time, before any data is read.
//! It catches incompatible types early and provides clear error messages.

use super::{coercion_kind, CoercionKind, CoercionMode};
use crate::errors::{CoercionError, CoercionWarning};
use crate::graph_config::IngestedType;

/// Result of validating an FK-PK type pair.
#[derive(Debug)]
pub struct ValidationResult {
    /// The FK column's IngestedType.
    pub fk_type: IngestedType,
    /// The PK column's IngestedType.
    pub pk_type: IngestedType,
    /// The kind of coercion determined from the type pair.
    pub kind: CoercionKind,
    /// Whether this type pair is allowed under the given mode.
    pub allowed: bool,
    /// Warning for conditional coercions (if any).
    pub warning: Option<CoercionWarning>,
    /// Error for disallowed coercions (if any).
    pub error: Option<CoercionError>,
}

/// Validate FK→PK type compatibility based on coercion mode.
///
/// # Arguments
/// * `fk_type` - The FK column's IngestedType
/// * `pk_type` - The PK column's IngestedType
/// * `fk_table` - Name of the FK table (for error messages)
/// * `fk_column` - Name of the FK column (for error messages)
/// * `pk_table` - Name of the PK table (for error messages)
/// * `pk_column` - Name of the PK column (for error messages)
/// * `mode` - The coercion mode to use
///
/// # Returns
/// A `ValidationResult` with allowed status and any warnings/errors.
pub fn validate_fk_pk(
    fk_type: IngestedType,
    pk_type: IngestedType,
    fk_table: &str,
    fk_column: &str,
    pk_table: &str,
    pk_column: &str,
    mode: CoercionMode,
) -> ValidationResult {
    let kind = coercion_kind(fk_type.clone(), pk_type.clone());

    let (allowed, warning, error) = match (mode, kind) {
        // Same type: always OK
        (_, CoercionKind::Same) => (true, None, None),

        // Strict mode: any difference is error
        (CoercionMode::Strict, _) => (
            false,
            None,
            Some(CoercionError::StrictModeMismatch {
                fk_table: fk_table.to_string(),
                fk_col: fk_column.to_string(),
                fk_type: fk_type.clone(),
                pk_table: pk_table.to_string(),
                pk_col: pk_column.to_string(),
                pk_type: pk_type.clone(),
            }),
        ),

        // Lossless mode: safe is OK
        (CoercionMode::Lossless, CoercionKind::Safe) => (true, None, None),

        // Lossless mode: conditional is OK with warning
        (CoercionMode::Lossless, CoercionKind::Conditional) => (
            true,
            Some(CoercionWarning {
                fk_table: fk_table.to_string(),
                fk_col: fk_column.to_string(),
                fk_type: fk_type.clone(),
                pk_table: pk_table.to_string(),
                pk_col: pk_column.to_string(),
                pk_type: pk_type.clone(),
                kind,
            }),
            None,
        ),

        // Lossy: always error
        (_, CoercionKind::Lossy) => (
            false,
            None,
            Some(CoercionError::LossyCoercion {
                fk_table: fk_table.to_string(),
                fk_col: fk_column.to_string(),
                fk_type: fk_type.clone(),
                pk_table: pk_table.to_string(),
                pk_col: pk_column.to_string(),
                pk_type: pk_type.clone(),
                reason: lossy_reason(&fk_type, &pk_type),
            }),
        ),

        // Incompatible: always error
        (_, CoercionKind::Incompatible) => (
            false,
            None,
            Some(CoercionError::IncompatibleTypes {
                fk_table: fk_table.to_string(),
                fk_col: fk_column.to_string(),
                fk_type: fk_type.clone(),
                pk_table: pk_table.to_string(),
                pk_col: pk_column.to_string(),
                pk_type: pk_type.clone(),
            }),
        ),
    };

    ValidationResult {
        fk_type,
        pk_type,
        kind,
        allowed,
        warning,
        error,
    }
}

/// Returns a human-readable reason for why a coercion is lossy.
fn lossy_reason(fk: &IngestedType, pk: &IngestedType) -> String {
    use IngestedType::*;

    match (fk, pk) {
        (Float32 | Float64, Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64) => {
            "float to integer truncates decimal portion".into()
        }
        (Int64, Int8 | Int16 | Int32) | (Int32, Int8 | Int16) | (Int16, Int8) => {
            "integer narrowing may overflow".into()
        }
        (UInt64, UInt8 | UInt16 | UInt32) | (UInt32, UInt8 | UInt16) | (UInt16, UInt8) => {
            "unsigned integer narrowing may overflow".into()
        }
        (Float64, Float32) => "float64 to float32 loses precision".into(),
        (Decimal256, Decimal128) => "decimal256 to decimal128 may overflow".into(),
        (Decimal256, _) => "decimal256 to smaller type may overflow".into(),
        _ => "conversion may lose data".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_type_allowed() {
        let result = validate_fk_pk(
            IngestedType::Int64,
            IngestedType::Int64,
            "orders",
            "customer_id",
            "customers",
            "id",
            CoercionMode::Strict,
        );
        assert!(result.allowed);
        assert!(result.warning.is_none());
        assert!(result.error.is_none());
        assert_eq!(result.kind, CoercionKind::Same);
    }

    #[test]
    fn test_safe_coercion_in_lossless() {
        let result = validate_fk_pk(
            IngestedType::Int32,
            IngestedType::Int64,
            "orders",
            "customer_id",
            "customers",
            "id",
            CoercionMode::Lossless,
        );
        assert!(result.allowed);
        assert!(result.warning.is_none());
        assert!(result.error.is_none());
        assert_eq!(result.kind, CoercionKind::Safe);
    }

    #[test]
    fn test_safe_coercion_rejected_in_strict() {
        let result = validate_fk_pk(
            IngestedType::Int32,
            IngestedType::Int64,
            "orders",
            "customer_id",
            "customers",
            "id",
            CoercionMode::Strict,
        );
        assert!(!result.allowed);
        assert!(result.warning.is_none());
        assert!(result.error.is_some());
        assert!(matches!(
            result.error,
            Some(CoercionError::StrictModeMismatch { .. })
        ));
    }

    #[test]
    fn test_conditional_coercion_warning() {
        let result = validate_fk_pk(
            IngestedType::String,
            IngestedType::Int64,
            "orders",
            "customer_id",
            "customers",
            "id",
            CoercionMode::Lossless,
        );
        assert!(result.allowed);
        assert!(result.warning.is_some());
        assert!(result.error.is_none());
        assert_eq!(result.kind, CoercionKind::Conditional);
    }

    #[test]
    fn test_lossy_coercion_error() {
        let result = validate_fk_pk(
            IngestedType::Float64,
            IngestedType::Int64,
            "orders",
            "amount",
            "products",
            "id",
            CoercionMode::Lossless,
        );
        assert!(!result.allowed);
        assert!(result.warning.is_none());
        assert!(result.error.is_some());
        assert!(matches!(
            result.error,
            Some(CoercionError::LossyCoercion { .. })
        ));
    }

    #[test]
    fn test_incompatible_types_error() {
        let result = validate_fk_pk(
            IngestedType::Bool,
            IngestedType::Uuid,
            "orders",
            "flag",
            "items",
            "id",
            CoercionMode::Lossless,
        );
        assert!(!result.allowed);
        assert!(result.warning.is_none());
        assert!(result.error.is_some());
        assert!(matches!(
            result.error,
            Some(CoercionError::IncompatibleTypes { .. })
        ));
    }
}
