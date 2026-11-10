// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context;
use diskgraph_core::metadata::DatasetMetadata;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use thiserror::Error;

use crate::graph_config::{GraphConfig, DISKGRAPH_CONFIG_FILENAME};
use crate::snapshot_mirror::MANIFEST_FILENAME;

pub const INDEX_METADATA_FILENAME: &str = "index_metadata.json";

#[derive(Debug, Error)]
pub enum IndexMetadataError {
    #[error("invalid index metadata JSON: {0}")]
    InvalidPayload(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub fn read_index_metadata(index_path: &Path) -> Result<String, IndexMetadataError> {
    let metadata_path = index_path.join(INDEX_METADATA_FILENAME);
    let raw = fs::read_to_string(&metadata_path)?;
    let _ = serde_json::from_str::<Value>(&raw)?;
    Ok(raw)
}

pub fn read_dataset_metadata(index_path: &Path) -> anyhow::Result<DatasetMetadata> {
    let path = index_path.join(INDEX_METADATA_FILENAME);
    let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

/// The id ingest records: the resolved config, every source file with its
/// size, and the DiskGraph version.
pub(crate) fn ingest_index_id(config: &GraphConfig) -> anyhow::Result<String> {
    let sources: BTreeSet<(&str, Option<u64>)> = config
        .tables
        .iter()
        .flat_map(|table| table.source_paths())
        .map(|path| (path, fs::metadata(path).ok().map(|m| m.len())))
        .collect();
    let identity = json!({
        "config": serde_json::to_value(config)?,
        "sources": sources,
        "diskgraph": env!("CARGO_PKG_VERSION"),
    });
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&identity)?)
    ))
}

/// The recorded id, or for an index ingested before ids existed, a hash of
/// its config, metadata and manifest files.
pub fn index_id(index_path: &Path, metadata: &DatasetMetadata) -> anyhow::Result<String> {
    if let Some(id) = &metadata.index_id {
        return Ok(id.clone());
    }
    let mut hasher = Sha256::new();
    for name in [
        DISKGRAPH_CONFIG_FILENAME,
        INDEX_METADATA_FILENAME,
        MANIFEST_FILENAME,
    ] {
        let bytes = match fs::read(index_path.join(name)) {
            Ok(bytes) => bytes,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
            Err(err) => return Err(err).with_context(|| format!("read {name}")),
        };
        hasher.update(name.as_bytes());
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_index_metadata_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let payload = serde_json::json!({
            "node_types": [{"name": "users", "max_id": 7}],
            "edge_relations": [],
        })
        .to_string();
        fs::write(dir.path().join(INDEX_METADATA_FILENAME), &payload).unwrap();

        let read_back = read_index_metadata(dir.path()).unwrap();

        assert_eq!(read_back, payload);
    }

    #[test]
    fn test_read_index_metadata_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let err = read_index_metadata(dir.path()).unwrap_err();
        assert!(matches!(err, IndexMetadataError::Io(_)));
    }

    #[test]
    fn test_read_index_metadata_invalid_json() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join(INDEX_METADATA_FILENAME), "{not json").unwrap();

        let err = read_index_metadata(dir.path()).unwrap_err();
        assert!(matches!(err, IndexMetadataError::InvalidPayload(_)));
    }
}
