//! Ingestion pipeline for diskgraph.
//!
//! This crate orchestrates three phases:
//! 1. Metadata discovery (reading metadata.json and validating schemas).
//! 2. Admission control (disk space reservation and layout planning).
//! 3. Materialization (stream Parquet shards, encode adjacency, commit to disk).
//!
//! Only phases 1 and 2 are partially implemented here to keep the project
//! compiling. The materialization phase intentionally leaves compression and
//! layout unimplemented until the core algorithms are chosen.

use async_trait::async_trait;
use diskgraph_core::dataset::{
    DatasetDescriptor, DatasetHandle, DatasetId, DatasetRegistry, DatasetReservation, DatasetState,
};
use diskgraph_core::errors::{DiskgraphError, Result};
use diskgraph_core::metadata::DatasetMetadata;
use diskgraph_core::storage::GraphStorage;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs;
use tracing::info;

/// Progress snapshot emitted by the ingest coordinator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestProgress {
    pub dataset_id: DatasetId,
    pub phase: IngestPhase,
    /// Fractional progress within the current phase (0.0-1.0).
    pub progress: f32,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IngestPhase {
    MetadataDiscovery,
    Reservation,
    Materialization,
    Finalizing,
}

/// Interface for components that want to receive ingestion progress events.
#[async_trait]
pub trait IngestObserver: Send + Sync {
    async fn on_progress(&self, progress: IngestProgress);
}

/// Coordinates metadata discovery, admission control, and delegating the heavy
/// lifting to the storage backend.
pub struct IngestCoordinator {
    storage: Arc<dyn GraphStorage>,
    registry: Arc<dyn DatasetRegistry>,
    observer: Option<Arc<dyn IngestObserver>>,
}

impl IngestCoordinator {
    pub fn new(
        storage: Arc<dyn GraphStorage>,
        registry: Arc<dyn DatasetRegistry>,
        observer: Option<Arc<dyn IngestObserver>>,
    ) -> Self {
        Self {
            storage,
            registry,
            observer,
        }
    }

    /// Load metadata.json from the dataset root. Remote backends are not yet
    /// supported; those callers will receive a NotImplemented error.
    pub async fn load_metadata(&self, root_uri: &str) -> Result<DatasetMetadata> {
        if root_uri.starts_with("s3://")
            || root_uri.starts_with("gs://")
            || root_uri.starts_with("abfs://")
        {
            return Err(DiskgraphError::NotImplemented(
                "remote object store metadata fetching",
            ));
        }
        let mut path = PathBuf::from(root_uri);
        path.push("metadata.json");
        let data = fs::read(&path).await?;
        let metadata: DatasetMetadata = serde_json::from_slice(&data)?;
        Ok(metadata)
    }

    /// Main entry point invoked by the HTTP service. This function wires
    /// together admission control and hands off to the storage engine to
    /// materialize data. The algorithmic heavy lifting is stubbed out.
    pub async fn ingest_dataset(&self, root_uri: String) -> Result<DatasetHandle> {
        self.emit_progress(None, IngestPhase::MetadataDiscovery, 0.0, Some("loading metadata"));
        let metadata = self.load_metadata(&root_uri).await?;
        self.emit_progress(None, IngestPhase::MetadataDiscovery, 1.0, None);

        let reservation = self.storage.reserve(&metadata).await?;
        self.emit_progress(None, IngestPhase::Reservation, 1.0, None);

        let descriptor = DatasetDescriptor {
            root_uri: root_uri.clone(),
            metadata: metadata.clone(),
        };
        let handle = self.registry.upsert(descriptor, reservation).await?;

        self.emit_progress(
            Some(handle.id),
            IngestPhase::Materialization,
            0.0,
            Some("streaming shards into on-disk layout"),
        );
        // Placeholder: real implementation would stream shards, compress, and
        // build indices. For now we swap the dataset into an active state.
        let ingested_handle = self.storage.ingest(handle.clone()).await?;
        let activated = self
            .registry
            .update_state(ingested_handle.id, DatasetState::LoadedActive)
            .await?;
        self.emit_progress(
            Some(ingested_handle.id),
            IngestPhase::Materialization,
            1.0,
            Some("materialization stub complete"),
        );
        self.emit_progress(
            Some(ingested_handle.id),
            IngestPhase::Finalizing,
            1.0,
            Some("dataset ready for queries"),
        );
        info!("ingested dataset {} from {}", activated.id, root_uri);
        Ok(activated)
    }

    fn emit_progress(
        &self,
        id: Option<DatasetId>,
        phase: IngestPhase,
        progress: f32,
        message: Option<&str>,
    ) {
        if let Some(observer) = &self.observer {
            if let Some(dataset_id) = id {
                let progress = IngestProgress {
                    dataset_id,
                    phase,
                    progress,
                    message: message.map(str::to_string),
                };
                let observer = Arc::clone(observer);
                tokio::spawn(async move {
                    observer.on_progress(progress).await;
                });
            }
        }
    }
}

/// Helper to validate the dataset directory exists and contains a metadata.json
/// file. Remote datasets should be validated by the transport-specific loader.
pub fn validate_dataset_root(path: &Path) -> Result<()> {
    if !path.exists() {
        return Err(DiskgraphError::InvalidMetadata(format!(
            "dataset root {} does not exist",
            path.display()
        )));
    }
    let metadata = path.join("metadata.json");
    if !metadata.exists() {
        return Err(DiskgraphError::MetadataMissing);
    }
    Ok(())
}
