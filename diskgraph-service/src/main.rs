//! HTTP service entrypoint for diskgraph. This server exposes dataset lifecycle
//! management endpoints and routes sampling requests to the storage backend.
//!
//! The sampling pipeline itself is stubbed; handlers focus on validating input,
//! reserving disk space, and wiring together the ingest coordinator.

use anyhow::Result;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use diskgraph_core::dataset::{DatasetHandle, DatasetRegistry, DatasetState, InMemoryDatasetRegistry};
use diskgraph_core::errors::{DiskgraphError, Result as CoreResult};
use diskgraph_core::sampling::{SampledNeighborhood, SamplingRequest};
use diskgraph_core::storage::{GraphStorage, LocalGraphStorage, StorageConfig};
use diskgraph_ingest::IngestCoordinator;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::signal;
use tracing::{error, info};
use uuid::Uuid;

#[derive(Clone)]
struct AppState {
    ingest: Arc<IngestCoordinator>,
    registry: Arc<dyn DatasetRegistry>,
    storage: Arc<dyn GraphStorage>,
}

#[derive(Debug, Deserialize)]
struct LoadDatasetRequest {
    dataset_root: String,
}

#[derive(Debug, Deserialize)]
struct UnloadDatasetRequest {
    purge: Option<bool>,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
}

#[derive(Debug)]
struct ApiError(DiskgraphError);

impl From<DiskgraphError> for ApiError {
    fn from(err: DiskgraphError) -> Self {
        Self(err)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self.0 {
            DiskgraphError::DatasetNotFound(_) => StatusCode::NOT_FOUND,
            DiskgraphError::MetadataMissing | DiskgraphError::InvalidMetadata(_) => StatusCode::BAD_REQUEST,
            DiskgraphError::InsufficientDiskSpace { .. } => StatusCode::INSUFFICIENT_STORAGE,
            DiskgraphError::DatasetBusy(_) => StatusCode::CONFLICT,
            DiskgraphError::NotImplemented(_) => StatusCode::NOT_IMPLEMENTED,
            DiskgraphError::Io(_) | DiskgraphError::Serde(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, self.0.to_string()).into_response()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let config = StorageConfig::new(default_data_path());
    let storage: Arc<dyn GraphStorage> = LocalGraphStorage::new(config);
    let registry: Arc<dyn DatasetRegistry> = InMemoryDatasetRegistry::new();
    let ingest = Arc::new(IngestCoordinator::new(
        storage.clone(),
        registry.clone(),
        None,
    ));
    let state = AppState {
        ingest,
        registry: registry.clone(),
        storage: storage.clone(),
    };

    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/datasets", get(list_datasets))
        .route("/datasets/load", post(load_dataset))
        .route("/datasets/:id/unload", post(unload_dataset))
        .route("/datasets/:id/sample", post(sample_neighborhood))
        .with_state(state);

    let addr: SocketAddr = ([0, 0, 0, 0], 8080).into();
    info!("listening on {}", addr);
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

fn default_data_path() -> PathBuf {
    std::env::var_os("DISKGRAPH_DATA_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data"))
}

async fn shutdown_signal() {
    let _ = signal::ctrl_c().await;
    info!("shutdown signal received");
}

async fn healthz() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

async fn list_datasets(State(state): State<AppState>) -> CoreResult<Json<Vec<DatasetHandle>>> {
    let datasets = state.registry.list().await?;
    Ok(Json(datasets))
}

async fn load_dataset(
    State(state): State<AppState>,
    Json(req): Json<LoadDatasetRequest>,
) -> Result<Json<DatasetHandle>, ApiError> {
    state.ingest.ingest_dataset(req.dataset_root).await.map(Json).map_err(ApiError::from)
}

async fn unload_dataset(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<UnloadDatasetRequest>,
) -> Result<StatusCode, ApiError> {
    let dataset_id =
        Uuid::parse_str(&id).map_err(|_| ApiError(DiskgraphError::InvalidMetadata("invalid dataset id".into())))?;
    // Transition state and purge if requested.
    state.storage.unload(dataset_id).await.map_err(ApiError::from)?;
    state.registry.remove(dataset_id).await.map_err(ApiError::from)?;
    if req.purge.unwrap_or(false) {
        info!("purge requested for dataset {}", dataset_id);
        // Actual purging of on-disk files will be implemented alongside layout.
    }
    Ok(StatusCode::NO_CONTENT)
}

async fn sample_neighborhood(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(mut request): Json<SamplingRequest>,
) -> Result<Json<SampledNeighborhood>, ApiError> {
    let dataset_id =
        Uuid::parse_str(&id).map_err(|_| ApiError(DiskgraphError::InvalidMetadata("invalid dataset id".into())))?;
    request.dataset_id = dataset_id;
    state.storage.sample(request).await.map(Json).map_err(ApiError::from)
}
