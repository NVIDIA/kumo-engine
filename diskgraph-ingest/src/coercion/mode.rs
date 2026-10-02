// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coercion mode configuration.
//!
//! Defines the two coercion modes: Strict and Lossless.

use serde::{Deserialize, Serialize};

/// Coercion mode for FK-PK type compatibility.
///
/// This controls how strictly type differences are handled:
/// - `Strict`: FK type must exactly match PK type (no coercion allowed)
/// - `Lossless`: Safe and conditional coercions allowed (default)
///
/// Lossy coercions (e.g., Float64 → Int64) always error regardless of mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoercionMode {
    /// No type differences allowed.
    /// FK IngestedType must exactly equal PK IngestedType.
    Strict,

    /// Safe and conditional coercions allowed.
    /// Lossy coercions always error.
    #[default]
    Lossless,
}

impl CoercionMode {
    /// Returns true if this mode allows safe coercions (e.g., Int32 → Int64).
    pub fn allows_safe(&self) -> bool {
        match self {
            CoercionMode::Strict => false,
            CoercionMode::Lossless => true,
        }
    }

    /// Returns true if this mode allows conditional coercions (e.g., String → Int64).
    pub fn allows_conditional(&self) -> bool {
        match self {
            CoercionMode::Strict => false,
            CoercionMode::Lossless => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_lossless() {
        assert_eq!(CoercionMode::default(), CoercionMode::Lossless);
    }

    #[test]
    fn test_strict_disallows_all() {
        assert!(!CoercionMode::Strict.allows_safe());
        assert!(!CoercionMode::Strict.allows_conditional());
    }

    #[test]
    fn test_lossless_allows_safe_and_conditional() {
        assert!(CoercionMode::Lossless.allows_safe());
        assert!(CoercionMode::Lossless.allows_conditional());
    }

    #[test]
    fn test_serde_roundtrip() {
        let strict = CoercionMode::Strict;
        let json = serde_json::to_string(&strict).unwrap();
        assert_eq!(json, r#""strict""#);

        let lossless = CoercionMode::Lossless;
        let json = serde_json::to_string(&lossless).unwrap();
        assert_eq!(json, r#""lossless""#);

        let parsed: CoercionMode = serde_json::from_str(r#""strict""#).unwrap();
        assert_eq!(parsed, CoercionMode::Strict);

        let parsed: CoercionMode = serde_json::from_str(r#""lossless""#).unwrap();
        assert_eq!(parsed, CoercionMode::Lossless);
    }
}
