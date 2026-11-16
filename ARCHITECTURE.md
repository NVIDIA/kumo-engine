# Kumo architecture

This monorepo contains Kumo's graph machine-learning application, Python packages, local execution stack, and deployment artifacts.

## Directory structure

| Area | Location | Purpose |
| --- | --- | --- |
| Applications | `apps/studio/` | Studio backend, frontend, utilities and deployment assets |
| Reusable packages | `packages/` | Contracts, data engine, shared runtime, integrations, clients and UI library |
| Development | `development/` | Shared tooling, cross-component tests, benchmarks and configuration |
| Documentation | `docs/` | Shared architecture, contributor guidance and system designs |

See [package ownership](packages/README.md), [development commands](development/README.md),
and the [layout design](docs/designs/repository-layout.md). Directory groups do
not change Python distribution names or import paths. Studio is packaged from
`apps/studio`; the root Python manifest defines the workspace.

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

- `apps/studio/rest/` owns service APIs and orchestration; it does not implement graph storage or model training.
- `diskgraph` is a Rust-backed wheel built from the source in `packages/data/diskgraph/`. The shared runtime in `packages/runtime/kumo-core/` and service adapters in `apps/studio/kumo/diskgraph/` and `apps/studio/kumo/data/generic/` integrate its ingest, compilation, and sampling APIs. This repository is the sole source of truth; make engine changes directly in `packages/data/diskgraph/`.
- `kumoapi` contains contracts shared by the service, SDK, and UI generator.
- `kumo_core` implements shared training, AutoML, evaluation, explanation, prediction, and DiskGraph integration without service orchestration dependencies.
- `apps/studio/docker/` defines the images built from this repository; `apps/studio/helm/` defines how the service is installed on Kubernetes.

## Service layering

The main REST path follows `router -> service -> DAO/driver -> workflow or activity`. Long-running ingestion, training, and prediction operations run as Temporal workflows. Deployment-specific configuration is provided through `service_framework` and the Helm or Compose environment.

## Entry points

| Task | Entry point |
|------|-------------|
| REST development server | `apps/studio/rest/main.py` via `make devserver` |
| UI development server | `apps/studio/ui/` |
| Service Python SDK | `packages/clients/kumo-sdk/` |
| Local Python SDK | `packages/clients/kumo-engine/` |
| Shared runtime | `packages/runtime/kumo-core/` |
| Tests | `development/testing/test/` and package-local test directories |
| Image builds | `development/tooling/scripts/release/build_kumo_images.py` and Dockerfiles under `apps/studio/docker/` |
| Helm packaging | `apps/studio/helm/` |

## Generated artifacts

- The UI OpenAPI client is generated under `apps/studio/ui/src/gen/`.
- `uv.lock` is generated from `apps/studio/setup.py`, the workspace `pyproject.toml` files.
- Third-party attribution files are generated from resolved dependency metadata.
