//! FK-PK type coercion module.
//!
//! This module handles type compatibility between foreign key and primary key columns
//! when building graph edges.
//!
//! # Overview
//!
//! When an FK column references a PK column, their types must be compatible for hash
//! lookup to work. This module:
//!
//! 1. Classifies FK→PK type pairs into coercion categories
//! 2. Validates type compatibility at config load time
//! 3. Enables automatic FK type resolution to match PK types
//!
//! # Components
//!
//! - `CoercionKind`: Classification of conversions (Same, Safe, Conditional, Lossy, Incompatible)
//! - `CoercionMode`: User config option (Strict or Lossless)
//! - `validate_fk_pk`: Config-time validation function
//!
//! # Coercion Categories
//!
//! | Category | Example | Description |
//! |----------|---------|-------------|
//! | Same | Int64 → Int64 | No conversion needed |
//! | Safe | Int32 → Int64 | Widening, always succeeds |
//! | Conditional | String → Int64 | May fail per-row (parsing) |
//! | Lossy | Float64 → Int64 | Would lose data (truncation) |
//! | Incompatible | Bool → Uuid | No conversion path exists |
//!
//! # Coercion Modes
//!
//! | Mode | Same | Safe | Conditional | Lossy | Incompatible |
//! |------|------|------|-------------|-------|--------------|
//! | Strict | ✓ | ✗ | ✗ | ✗ | ✗ |
//! | Lossless | ✓ | ✓ | ✓ (warn) | ✗ | ✗ |

mod kind;
mod mode;
mod validate;
#[cfg(test)]
mod tests_matrix;

pub use kind::{coercion_kind, CoercionKind, CoercionParams};
pub use mode::CoercionMode;
pub use validate::{validate_fk_pk, ValidationResult};
