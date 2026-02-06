use crate::dataset::{DatasetHandle, DatasetId, DatasetReservation, DatasetState};
use crate::errors::{DiskgraphError, Result};
use crate::metadata::DatasetMetadata;
use crate::sampling::BatchSamplingRequest;
use crate::features::BatchFeatureRequest;
use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use arrow_array::RecordBatch;

pub mod compression;

/// Basic storage settings shared between ingestion and runtime lookup.
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Root directory for on-disk graph artifacts.
    pub data_path: PathBuf,
    /// Maximum fraction of filesystem usage allowed for datasets (e.g., 0.95).
    pub max_disk_fraction: f64,
    /// Optional bound on in-memory cache usage.
    pub cache_bytes: Option<u64>,
}

impl StorageConfig {
    pub fn new<P: AsRef<Path>>(data_path: P) -> Self {
        Self {
            data_path: data_path.as_ref().to_path_buf(),
            max_disk_fraction: 0.95,
            cache_bytes: None,
        }
    }
}

/// Contract implemented by storage backends (local disk, remote cache, etc.).
#[async_trait]
pub trait GraphStorage: Send + Sync {
    /// Reserve disk space for a dataset. Implementations must enforce the
    /// max_disk_fraction guard and fail fast if the reservation would exceed
    /// the limit. This is called prior to ingestion.
    async fn reserve(&self, metadata: &DatasetMetadata) -> Result<DatasetReservation>;

    /// Persist the ingested graph to disk and return an updated handle. The
    /// caller is responsible for loading Parquet shards; storage handles layout
    /// and compression.
    async fn ingest(&self, handle: DatasetHandle) -> Result<DatasetHandle>;

    /// Unload a dataset and free any disk space and caches tied to it.
    async fn unload(&self, id: DatasetId) -> Result<()>;

    /// Perform a sampling request against the on-disk structure.
    async fn sample(&self, request: BatchSamplingRequest) -> Result<RecordBatch>;

    /// Retrieve node features for a batch of IDs.
    async fn get_features(&self, request: BatchFeatureRequest) -> Result<RecordBatch>;
}

/// Skeleton storage engine that manages reservations and tracks dataset state.
/// The compression and on-disk representation are intentionally left
/// unimplemented; this struct wires together the control plane.
pub struct LocalGraphStorage {
    pub config: StorageConfig,
    datasets: RwLock<HashMap<DatasetId, DatasetHandle>>,
}

impl LocalGraphStorage {
    pub fn new(config: StorageConfig) -> Arc<Self> {
        Arc::new(Self {
            config,
            datasets: RwLock::new(HashMap::new()),
        })
    }

    fn current_disk_usage_bytes(&self) -> Result<u64> {
        // Placeholder: actual implementation should walk self.config.data_path
        // and sum directory sizes using fallocate/stat to reflect sparse files.
        Ok(0)
    }

    fn total_capacity_bytes(&self) -> Result<u64> {
        // Placeholder: on Unix, use statvfs; on Windows, GetDiskFreeSpaceEx.
        Ok(u64::MAX)
    }
}

#[async_trait]
impl GraphStorage for LocalGraphStorage {
    async fn reserve(&self, metadata: &DatasetMetadata) -> Result<DatasetReservation> {
        let estimate = metadata.estimate_ondisk_bytes()?;
        let used = self.current_disk_usage_bytes()?;
        let capacity = self.total_capacity_bytes()?;
        let allowed = (capacity as f64 * self.config.max_disk_fraction) as u64;
        if used.saturating_add(estimate) > allowed {
            return Err(DiskgraphError::InsufficientDiskSpace {
                required: used.saturating_add(estimate),
                available: allowed,
            });
        }
        Ok(DatasetReservation {
            reserved_bytes: estimate,
            estimated_bytes: estimate,
            rationale: Some("pre-ingestion reservation".into()),
        })
    }

    async fn ingest(&self, mut handle: DatasetHandle) -> Result<DatasetHandle> {
        // The ingestion pipeline writes to disk and swaps the dataset into an
        // active state. Here we only update bookkeeping so that the HTTP layer
        // can return useful status until the full implementation is added.
        handle.state = DatasetState::LoadedActive;
        let mut guard = self.datasets.write();
        guard.insert(handle.id, handle.clone());
        Ok(handle)
    }

    async fn unload(&self, id: DatasetId) -> Result<()> {
        let mut guard = self.datasets.write();
        guard
            .remove(&id)
            .ok_or_else(|| DiskgraphError::DatasetNotFound(id.to_string()))?;
        // TODO: remove on-disk files under config.data_path once layout exists.
        Ok(())
    }

    async fn sample(&self, request: BatchSamplingRequest) -> Result<RecordBatch> {
        let guard = self.datasets.read();
        let handle = guard
            .get(&request.dataset_id)
            .ok_or_else(|| DiskgraphError::DatasetNotFound(request.dataset_id.to_string()))?;
        if !handle.is_queryable() {
            return Err(DiskgraphError::DatasetBusy(format!(
                "dataset {} is not active",
                request.dataset_id
            )));
        }
        // Placeholder return empty batch
        Ok(RecordBatch::new_empty(Arc::new(arrow_schema::Schema::empty())))
    }

    async fn get_features(&self, request: BatchFeatureRequest) -> Result<RecordBatch> {
        let guard = self.datasets.read();
        let handle = guard
            .get(&request.dataset_id)
            .ok_or_else(|| DiskgraphError::DatasetNotFound(request.dataset_id.to_string()))?;
        if !handle.is_queryable() {
            return Err(DiskgraphError::DatasetBusy(format!(
                "dataset {} is not active",
                request.dataset_id
            )));
        }
        // Placeholder return empty batch
        Ok(RecordBatch::new_empty(Arc::new(arrow_schema::Schema::empty())))
    }
}