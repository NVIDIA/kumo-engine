// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::errors::{DiskgraphError, Result};
use crate::metadata::DatasetMetadata;
use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Unique identifier for a dataset once it has been registered with the service.
pub type DatasetId = Uuid;

/// Lifecycle states for a dataset managed by the service.
///
/// The HTTP layer mirrors these to clients, and the sampler uses them to refuse
/// queries when the underlying storage is not ready.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DatasetState {
    NotPresent,
    Loading,
    LoadedActive,
    LoadedInactive,
    Unloading,
    Failed,
}

/// Admission control and status details used by the disk usage guard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetReservation {
    /// Bytes reserved for the dataset during admission control.
    pub reserved_bytes: u64,
    /// Estimation of the final on-disk footprint based on metadata heuristics.
    pub estimated_bytes: u64,
    /// Optional note on why the reservation was made (e.g., eviction policy).
    pub rationale: Option<String>,
}

/// Snapshot of a dataset being tracked by the runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetHandle {
    pub id: DatasetId,
    pub root_uri: String,
    pub metadata: DatasetMetadata,
    pub state: DatasetState,
    pub reservation: Option<DatasetReservation>,
}

impl DatasetHandle {
    /// Returns true if the dataset is available for sampling queries.
    pub fn is_queryable(&self) -> bool {
        matches!(self.state, DatasetState::LoadedActive)
    }
}

/// Descriptor used when registering a dataset before ingestion starts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetDescriptor {
    pub root_uri: String,
    pub metadata: DatasetMetadata,
}

/// User-facing status payload. The service can attach a progress message to
/// surface ingestion phases without forcing the ingest crate to leak types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStatus {
    pub handle: DatasetHandle,
    pub status_message: Option<String>,
}

/// Contract for dataset lifecycle management. Implementations may keep their
/// state purely in-memory or persist it on disk alongside the graph data.
#[async_trait]
pub trait DatasetRegistry: Send + Sync {
    /// Insert or update a dataset handle. Typically invoked at the beginning of
    /// ingestion after parsing index_metadata.json.
    async fn upsert(
        &self,
        descriptor: DatasetDescriptor,
        reservation: DatasetReservation,
    ) -> Result<DatasetHandle>;

    /// Transition a dataset to the next state (e.g., Loading -> LoadedActive).
    async fn update_state(&self, id: DatasetId, next: DatasetState) -> Result<DatasetHandle>;

    /// Remove the dataset from the registry once the on-disk data has been
    /// purged. Implementations may choose to retain tombstones.
    async fn remove(&self, id: DatasetId) -> Result<()>;

    /// List all datasets and their states. Used by HTTP status endpoints.
    async fn list(&self) -> Result<Vec<DatasetHandle>>;

    /// Fetch a single dataset handle if it exists.
    async fn get(&self, id: DatasetId) -> Result<Option<DatasetHandle>>;
}

/// A minimal in-memory registry suitable for bootstrapping the service and for
/// integration tests. It does not persist state across restarts.
pub struct InMemoryDatasetRegistry {
    inner: RwLock<HashMap<DatasetId, DatasetHandle>>,
}

impl Default for InMemoryDatasetRegistry {
    fn default() -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
        }
    }
}

impl InMemoryDatasetRegistry {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }
}

#[async_trait]
impl DatasetRegistry for InMemoryDatasetRegistry {
    async fn upsert(
        &self,
        descriptor: DatasetDescriptor,
        reservation: DatasetReservation,
    ) -> Result<DatasetHandle> {
        let mut guard = self.inner.write();
        let id = Uuid::new_v4();
        let handle = DatasetHandle {
            id,
            root_uri: descriptor.root_uri,
            metadata: descriptor.metadata,
            state: DatasetState::Loading,
            reservation: Some(reservation),
        };
        guard.insert(id, handle.clone());
        Ok(handle)
    }

    async fn update_state(&self, id: DatasetId, next: DatasetState) -> Result<DatasetHandle> {
        let mut guard = self.inner.write();
        let handle = guard
            .get_mut(&id)
            .ok_or_else(|| DiskgraphError::DatasetNotFound(id.to_string()))?;
        handle.state = next;
        Ok(handle.clone())
    }

    async fn remove(&self, id: DatasetId) -> Result<()> {
        let mut guard = self.inner.write();
        guard
            .remove(&id)
            .ok_or_else(|| DiskgraphError::DatasetNotFound(id.to_string()))?;
        Ok(())
    }

    async fn list(&self) -> Result<Vec<DatasetHandle>> {
        let guard = self.inner.read();
        Ok(guard.values().cloned().collect())
    }

    async fn get(&self, id: DatasetId) -> Result<Option<DatasetHandle>> {
        let guard = self.inner.read();
        Ok(guard.get(&id).cloned())
    }
}
