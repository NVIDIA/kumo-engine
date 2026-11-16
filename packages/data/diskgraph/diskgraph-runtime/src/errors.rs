// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use diskgraph_engine::errors::EngineError;
use diskgraph_train_table::errors::TTGenError;

/// Runtime failures classified independently of any language binding.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("{0}")]
    /// Invalid iterator or sampling configuration.
    Config(String),
    #[error("{0}")]
    /// Invalid dataset or sidecar contents.
    Data(String),
    #[error("{0}")]
    /// Storage, engine, or internal failure.
    System(String),
    #[error("{0}")]
    /// Training-table target categories exceed the supported limit.
    TargetCategoryLimit(String),
}
impl RuntimeError {
    /// Classify a configuration failure while retaining its message.
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config(message.into())
    }
    /// Classify invalid data while retaining its message.
    pub fn data(message: impl Into<String>) -> Self {
        Self::Data(message.into())
    }
    /// Classify a system failure while retaining its message.
    pub fn system(message: impl Into<String>) -> Self {
        Self::System(message.into())
    }
}
impl From<EngineError> for RuntimeError {
    fn from(error: EngineError) -> Self {
        match error {
            EngineError::InvalidRequest(msg) | EngineError::Configuration(msg) => Self::Config(msg),
            EngineError::DataCorruption(msg)
            | EngineError::Output(msg)
            | EngineError::Internal(msg) => Self::System(msg),
            EngineError::Io(error) => Self::System(error.to_string()),
        }
    }
}
impl From<TTGenError> for RuntimeError {
    fn from(error: TTGenError) -> Self {
        match error {
            TTGenError::SchemaError(msg) => Self::Data(msg),
            TTGenError::InvalidPlan(msg) => Self::Config(msg),
            TTGenError::TargetCategoryLimit(msg) => Self::TargetCategoryLimit(msg),
            TTGenError::SystemError(msg) | TTGenError::Parquet(msg) => Self::System(msg),
            TTGenError::Io(error) => Self::System(error.to_string()),
        }
    }
}
/// Result returned by index construction and native batch iteration.
pub type Result<T> = std::result::Result<T, RuntimeError>;
