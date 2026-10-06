// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Information about a single index directory on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexInfo {
    /// Full path to the index directory.
    pub path: PathBuf,
    /// Graph key (parent directory name).
    pub graph_key: String,
    /// Graph version (directory name).
    pub graph_version: String,
    /// Size in bytes of the entire directory.
    pub size_bytes: u64,
    /// Whether a `_SUCCESS` marker exists (valid, completed ingest).
    pub has_success_marker: bool,
    /// mtime of the `_SUCCESS` file (epoch seconds), used for LRU.
    /// None if no `_SUCCESS` marker.
    pub success_mtime_secs: Option<u64>,
}

impl IndexInfo {
    /// Returns true if this is an orphaned directory (no _SUCCESS marker).
    pub fn is_orphan(&self) -> bool {
        !self.has_success_marker
    }
}

/// Disk usage statistics for the filesystem containing the data root.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskUsage {
    /// Total filesystem capacity in bytes.
    pub total_bytes: u64,
    /// Available bytes on the filesystem.
    pub available_bytes: u64,
    /// Used bytes on the filesystem.
    pub used_bytes: u64,
}

/// Request to ensure enough space for an ingest operation.
#[derive(Debug, Clone)]
pub struct EnsureSpaceRequest {
    /// Bytes needed for the new ingest.
    pub needed_bytes: u64,
    /// Graph key of the current ingest target (will not be evicted).
    pub current_graph_key: String,
    /// Graph version of the current ingest target (will not be evicted).
    pub current_graph_version: String,
}

/// Result of an ensure_space operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsureSpaceResult {
    /// Indexes that were evicted to make space.
    pub evicted: Vec<EvictedIndex>,
    /// Available bytes after eviction.
    pub available_bytes: u64,
    /// Bytes that were needed.
    pub needed_bytes: u64,
    /// Total bytes freed by eviction.
    pub freed_bytes: u64,
}

/// Record of an evicted index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvictedIndex {
    pub path: PathBuf,
    pub graph_key: String,
    pub graph_version: String,
    pub size_bytes: u64,
    pub was_orphan: bool,
}
