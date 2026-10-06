# Kumo architecture

This monorepo contains Kumo's graph machine-learning application, Python packages, local execution stack, and deployment artifacts.

## Directory structure

| Area | Location | Purpose |
|------|----------|---------|
| REST service | `rest/` | FastAPI routes, services, persistence, and job orchestration |
| Web application | `ui/` | React user interface and generated API client |
| Public API package | `packages/kumo-api/` | Shared schemas, PQL types, and API contracts |
| Service Python SDK | `packages/kumo-sdk/` | Python client for a configured Kumo service (`kumoai_studio`) |
| Local Python SDK | `packages/kumo-engine/` | In-process SDK surface (`kumo.engine`) for local execution with local or remote data sources |
| Shared runtime | `packages/kumo-core/` | Service-independent data processing, training, evaluation, and prediction used by the service and local SDK |
| Connectors | `kumo/connector/` | Data-warehouse and file-source integrations |
| Data pipeline | `kumo/data/` | Export, cache, and DiskGraph-backed table generation |
| Query language | `packages/kumo-pql/`, `kumo/pquery/` | Parser and semantic validation package; service query and table-generation orchestration |
| Service training adapters | `kumo/train/`, `kumo/task/` | Service task materialization and training/prediction adapters around the shared runtime |
| Graph engine | `kumo/diskgraph/` | Service integration for DiskGraph ingest and snapshot management |
| Workflows | `temporalio_commons/`, `rest/backends/` | Temporal workflows and activities |
| Deployment | `docker/`, `helm/` | Container images, Compose topology, and Helm charts |
| Shared infrastructure | `service_framework/`, `io_utils/`, `async_utils/`, `common_utils/` | Deployment, storage, concurrency, and general utilities |

Package-owned documentation, including the migrated parser, catalog, graph engine,
and source providers, is indexed in [docs/README.md](docs/README.md#package-documentation).

## Execution pipeline

The service uses a DiskGraph-backed execution pipeline:

1. A connector exports source tables to Parquet in blob storage.
2. `DiskgraphService` ingests those files and produces an on-disk index plus column statistics.
3. `DiskGraphPlanCompiler` compiles a predictive query and materializes training or batch-prediction tables against the index.
4. Task materialization and the core DiskGraph data module/runtime adapter feed the resulting graph data into GNN training or prediction.

The REST service coordinates these stages through Temporal. The local SDK runs the core pipeline in-process through `kumo_core`.

## User flow

```text
Connector -> Tables -> Graph -> Predictive query -> Training -> Model -> Prediction
```

Users configure data sources, tables, graph relationships, and a predictive query through the UI or SDK. Training produces a model that supports batch prediction, embedding export, and explanation workflows.

## Important boundaries

- `rest/` owns service APIs and orchestration; it does not implement graph storage or model training.
- `diskgraph` is a Rust-backed wheel built from the synchronized source in `packages/diskgraph/`. The shared runtime in `packages/kumo-core/` and service adapters in `kumo/diskgraph/` and `kumo/data/generic/` integrate its ingest, compilation, and sampling APIs. Changes to the mirror follow its [upstream ownership boundary](docs/development/generated-artifacts.md#diskgraph-source-mirror).
- `kumoapi` contains contracts shared by the service, SDK, and UI generator.
- `kumo_core` implements shared training, AutoML, evaluation, explanation, prediction, and DiskGraph integration without service orchestration dependencies.
- `docker/` defines the images built from this repository; `helm/` defines how the service is installed on Kubernetes.

## Service layering

The main REST path follows `router -> service -> DAO/driver -> workflow or activity`. Long-running ingestion, training, and prediction operations run as Temporal workflows. Deployment-specific configuration is provided through `service_framework` and the Helm or Compose environment.

## Entry points

| Task | Entry point |
|------|-------------|
| REST development server | `rest/main.py` via `make devserver` |
| UI development server | `ui/` |
| Service Python SDK | `packages/kumo-sdk/` |
| Local Python SDK | `packages/kumo-engine/` |
| Shared runtime | `packages/kumo-core/` |
| Tests | `test/` and package-local test directories |
| Image builds | `scripts/release/build_kumo_images.py` and Dockerfiles under `docker/` |
| Helm packaging | `helm/` |

## Generated artifacts

- The UI OpenAPI client is generated under `ui/src/gen/`.
- Python dependency lockfiles are generated from `3rdparty/BUILD` and `setup.py`.
- Third-party attribution files are generated from resolved dependency metadata.
