# DiskGraph integration

DiskGraph provides native graph ingestion, training/prediction table generation,
and indexed data access for Kumo. Kumo services coordinate jobs and storage;
`kumo-core` supplies the shared training runtime.

## Pipeline

1. Graph refresh builds or restores a snapshot on a pool worker.
2. Table generation publishes a durable bundle containing the generated table
   and the inputs needed to ingest it into a graph snapshot.
3. Materialization ingests the selected bundle into the job's graph and
   prepares its task dataset.
4. Training and prediction consume native dataloader batches through
   `DiskgraphNestedTrainer`.
5. Prediction workers write final-schema artifacts for export.

See the [architecture map](../ARCHITECTURE.md) for component boundaries and the
[train-table ingest contract](ttng-train-table-ingest-and-cross-graph-reuse.md)
for bundle ownership and cross-graph behavior.

## Configuration

### DiskgraphConfig

`DiskgraphConfig` is the data-compute config used by live data pipelines. Its
fields, admission semantics, and current defaults are defined authoritatively
in `service_framework/_raw_deploy_config.py`.

### Mode Detection

Deployment-specific code can inspect the configured data backend with:

```python
from service_framework.deploy_env_config import deploy_env

if deploy_env().is_diskgraph:
    # diskgraph path
```

`is_diskgraph` is a property on `DeployEnvConfig` in
`service_framework/deploy_env_config.py` that checks whether `data_compute` is
a `DiskgraphConfig`. It is not a universal pipeline gate: current ML table
generation requires a DiskGraph index.

### Index Path

The diskgraph index for a specific graph lives at:

```
{data_root}/{graph_key}/{graph_data_version}/
```

Resolved from a graph handle by `DiskgraphService.local_snapshot_path()` in
`kumo/diskgraph/service.py`. ML jobs obtain their bound path through
`get_diskgraph_index_path_for_job()` in `statedb/config/diskgraph.py`.

### Running Locally

Use the current [devserver guide](devserver.md) for coordinator commands, resource-policy selection, and the local Temporal variant. The [DiskGraph build instructions](../packages/diskgraph/README.md#python-wheel-builds) describe how the native dependency is supplied.

## Stage Details

### Stage 1: Graph Refresh (Ingestion)

`GraphRefreshWorkflow` dispatches snapshot ingestion through the DiskGraph pool.

**Flow:**

```
GraphRefreshWorkflow
  └─ DiskgraphIngestWorkflow (child workflow)
       ├─ route snapshot session to a pool worker
       ├─ SnapshotActivities.ensure_snapshot
       │    ├─ restore a valid dataplane mirror on local cache miss, or
       │    └─ fresh Rust ingest
       │         ├─ build the index on pool NVMe
       │         ├─ stream-mirror every output inline to the dataplane
       │         └─ write _MANIFEST.json + _SUCCESS last on both
       ├─ cleanup source parquet files
       └─ diskgraph_ingest_stats_activity (stats bridge)
```

**Key behaviors:**
- **Pool routing and disk management**: `snapshot_placement_session_ctx` routes the request to a pool worker and exposes the selected capacity tier. The controller estimates fresh-ingest demand and admits it against the placement strategy's reported `data_root` (NVMe) capacity. After leases are released, controller retention sweeps dispatch local deletion under disk pressure or when the force-eviction window expires.
- **Inline dataplane mirror**: Rust writes every snapshot artifact to the dataplane during ingest, not in a later workflow. `_MANIFEST.json` and `_SUCCESS` are finalized last on both destinations, so a mirror write failure is fatal to the build.
- **Local dataplane capacity**: Before provisioning a fresh `MISSING` placement, the controller admits its mirror estimate against capacity reported by the selected durable writer. It subtracts estimates for other leased, non-READY snapshots on that writer that do not yet have dataplane backups; same-snapshot fan-out is charged once unless routing moves to another writer. If `data_root` and `data_plane_storage_root` share a filesystem, admission uses one free-space budget and accounts for both output copies. Object-storage roots and unknown capacity or estimates bypass this check for compatibility. Because heartbeats and estimates are not exact runtime reservations, operators must retain and monitor headroom. If the volume still fills, free or expand it, or move the dataplane root to supported object storage; re-placing the ingest on a larger DiskGraph node does not enlarge a separate dataplane volume.
- **Stats bridge**: After ingest, `diskgraph_ingest_stats_activity` copies Rust-produced col stats and edge stats from the index directory into the Kumo data backend cache. This allows downstream code (encoders, validation) to work unchanged.
- **Edge stats skipped**: Edge stats computation is handled natively by the Rust ingest, not by a separate Spark job.

**Key files:**

| File | Purpose |
|------|---------|
| `rest/backends/data_backend/workflows/graph_refresh_workflow.py` | Top-level workflow and DiskGraph dispatch |
| `rest/backends/data_backend/workflows/diskgraph_ingest_workflow.py` | Child workflow: pool session, ensure snapshot, cleanup, stats |
| `pool/wf_coordinator.py` | Snapshot-session routing to a pool task queue |
| `pool/activities.py` | Snapshot restore-or-ingest activity |
| `kumo/diskgraph/service.py` | Local/dataplane path layout and mirror contract |
| `kumo/diskgraph/ingest.py` | Config conversion and subprocess invocation |
| `packages/kumo-core/src/kumo_core/diskgraph/runner.py` | Subprocess retry and fatal disk classification |
| `rest/backends/data_backend/diskgraph_stats_bridge.py` | Col/edge stats → Kumo cache copy |

### Stage 2: Table Generation

DiskGraph generates training and prediction tables natively in Rust via the `diskgraph` Python extension.

**Training tables:**
- `diskgraph_train_table.py` prepares generation inputs and delegates to
  `kumo_core.training.table.generate_train_table()`, which invokes the native
  compiler and TT-NG generator.
- `generate_train_table.py` publishes the immutable durable bundle and returns
  its physical `tt.parquet` path. Training later ingests that bundle into the
  selected graph; see the [current ingest contract](ttng-train-table-ingest-and-cross-graph-reuse.md).

**Prediction tables:**
- `diskgraph_batch_pred.py` compiles and generates node, link, and forecasting
  prediction tables against the required index.
- `generate_current_pred_table.py` and `PredictionTableGeneratorV2` coordinate
  prediction-table generation; there is no Spark fallback.

**Workflow wiring:**
- The training and prediction activities in
  `rest/backends/ml_backend/activities/generate_target_table.py` obtain
  `diskgraph_index_path` through `get_diskgraph_index_path_for_job()`.

**Key files:**

| File | Purpose |
|------|---------|
| `kumo/data/generic/diskgraph_train_table.py` | Service generation adapter and durable bundle publication helpers |
| `packages/kumo-core/src/kumo_core/training/table.py` | Shared native training-table generation |
| `kumo/data/generic/diskgraph_batch_pred.py` | Native prediction-table generation |
| `kumo/data/generic/generate_train_table.py` | Training-table validation and publication orchestration |
| `kumo/data/generic/generate_current_pred_table.py` | Prediction-table orchestration |
| `kumo/pquery/batch_pred_table_generator.py` | `PredictionTableGeneratorV2` |
| `rest/backends/ml_backend/activities/generate_target_table.py` | Workflow activities and graph-index binding |

### Stage 3: Task-dataset materialization

The training and prediction materialization activities ingest the selected
bundle into the job's graph snapshot and prepare a task dataset. Consumers
read the published `train_tables/<tt_id>/` extension; the durable generation
bundle supplies the inputs for ingest or recovery. See the
[ingest contract](ttng-train-table-ingest-and-cross-graph-reuse.md).

- [Training materialization](../rest/backends/ml_backend/activities/materialize_training_task.py)
- [Prediction materialization](../rest/backends/ml_backend/activities/materialize_prediction_task.py)
- [Task-dataset service](../kumo/diskgraph/service.py)

### Stage 4: Training and prediction

[AutoTrainer](../packages/kumo-core/src/kumo_core/automl/auto_trainer.py)
dispatches training trials to
[DiskgraphNestedTrainer](../packages/kumo-core/src/kumo_core/train/diskgraph_nested_trainer.py).
That trainer constructs `DiskgraphLightningDataModule`, whose
[runtime adapter](../packages/kumo-core/src/kumo_core/diskgraph/runtime_adapter.py)
consumes native dataloader batches from the task dataset.
For two-tower link prediction, that adapter supplies positive/negative indices
and a negative mask to the
[link GNN module](../packages/kumo-core/src/kumo_core/train/link_gnn_module.py).
The [engine-side negative-sampling design](../packages/diskgraph/docs/tt-ng-link-prediction-negative-sampling.md)
explains the sampling contract.

The service entry points are `bin/kumo-auto` and `bin/kumo-pred`.
`run_ttng_batch_prediction` in [kumo-pred](../bin/kumo-pred) loads the trainer
with the prediction task dataset and runs inference.

### Stage 5: Batch Prediction Artifact Export

The prediction process writes each requested artifact in its final customer
schema. Prediction shards contain `ENTITY`, `TARGET_PRED`, and requested
metadata such as `JOB_TIMESTAMP`; embedding shards contain `ENTITY`,
`EMBEDDING`, and the requested metadata. There is no whole-artifact
post-processing or second Parquet rewrite.

After inference:

- **Artifact collection** checks that each producer-finalized Parquet dataset
  exists and is non-empty. It does not read and rewrite payload rows.
- **Shared metadata** is computed once and reused for every configured output
  destination.
- **Provider-native Databricks export** has every prediction worker upload its
  final local Parquet shards directly to the connector schema's managed Unity
  Catalog volume with parallel `PUT INTO` operations. Databricks-bound
  producers rotate files at a conservative 128 MiB uncompressed target so
  uploads remain memory-bounded and expose enough parallel streams to
  saturate the worker network. One coordinator issues one `COPY INTO`, then
  removes the exact staged files. No prediction payload passes through the
  control plane, and the path has no S3/ADLS credential or deployment staging
  configuration.
- **Staging cleanup**: The `cleanup_diskgraph_staging` activity in
  `batch_prediction_workflow.py` removes local DiskGraph staging directories
  after the BP workflow completes.

**Key files:**

| File | Purpose |
|------|---------|
| `packages/kumo-core/src/kumo_core/train/output.py` | Streaming final-schema prediction and embedding writers |
| `rest/backends/ml_backend/activities/parquet_io_activities.py` | Artifact collection, metadata, and output dispatch |
| `rest/backends/ml_backend/artifact_export.py` | Provider-specific artifact export |
| `rest/backends/ml_backend/workflows/batch_prediction_workflow.py` | BP workflow, staging cleanup |
