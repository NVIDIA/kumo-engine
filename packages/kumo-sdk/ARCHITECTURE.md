<!--
ARCHITECTURE_META
last_updated: 2026-07-09T07:30:00Z
commit: b7a0610fabf1da8776f7639e8a4414b469c1d335
-->

# kumo-sdk

Python SDK for the Kumo machine learning platform, enabling declarative ML task definition via Predictive Query Language (PQL).

## Overview

The Kumo SDK (`kumoai_studio` package) provides a pythonic interface for defining relational ML problems and orchestrating training/prediction jobs on a configured Kumo service via `Trainer`.

Primary consumers are data scientists and ML engineers building predictive models on relational data.

**Repository context:** [Kumo overview](../../README.md). The legacy
standalone SDK site is being reviewed separately.

## Directory Structure

```
kumoai_studio/
├── __init__.py           # Package entry, init(), GlobalState singleton, project/identity helpers
├── _logging.py           # Logging configuration
├── _singleton.py         # Singleton pattern implementation
├── _version.py           # Version information
├── artifact_export/      # Model artifact export functionality
│   ├── config.py         # Export configuration options
│   └── job.py            # Export job handling
├── client/               # HTTP client + typed API classes per resource
│   ├── client.py         # KumoClient main class
│   ├── connector.py      # Connector API
│   ├── endpoints.py      # API endpoint definitions
│   ├── graph.py          # Graph API
│   ├── groups.py         # Groups API
│   ├── identity.py       # Identity API
│   ├── jobs.py           # Job management APIs
│   ├── pquery.py         # Predictive query API
│   ├── projects.py       # Projects API
│   ├── source_table.py   # Source table API
│   ├── table.py          # Table API
│   └── utils.py          # Client utility functions
├── codegen/              # Code generation utilities
│   ├── cli.py            # CLI interface for codegen
│   ├── context.py        # Generation context
│   ├── edits.py          # Code edit operations
│   ├── exceptions.py     # Codegen-specific exceptions
│   ├── generate.py       # Main generation logic
│   └── handlers/         # Entity-specific handlers
├── connector/            # Data source connectors (S3, GCS, ADLS, Snowflake, BigQuery, Databricks, file upload)
├── databricks.py         # Databricks output table name mangling
├── encoder/              # Re-exports of kumoapi encoders for ModelPlan
├── exceptions.py         # HTTPException, ProjectNotSetError
├── formatting.py         # Job error-detail pretty printing
├── futures.py            # KumoFuture, async job monitoring
├── graph/                # Table, Column, Edge, Graph relational metadata
├── identity.py           # Credential-free Identity value object
├── jobs.py               # JobInterface + job tag search
├── mixin.py              # CastMixin for flexible object construction
├── pquery/               # PredictiveQuery, TrainingTable, PredictionTable
├── project.py            # Group, Project value objects
├── session.py            # Session/ComputeType context manager
├── testing/              # Test helpers
├── trainer/              # Trainer, ModelPlan, and job execution
└── utils/                # ForecastVisualizer, RelBench upload helper
examples/
├── datasets/             # Dataset loading utilities
│   ├── customerltv_mini.py
│   ├── from_fixture.py
│   ├── hm_churn.py
│   ├── rel-stack.py
│   └── weighted_train_table.py
├── flows/                # End-to-end workflow examples
│   ├── data_preparation/ # Data prep examples (file_upload, snowflake)
│   ├── model_development/# Training workflows (champion_challenger, retrain, combined_pq)
│   ├── production/       # Production deployment (airflow)
│   └── setup/            # Setup examples (s3, snowflake)
└── paper/                # Benchmarking framework (enterprise, single-table, forecasting)
test/                     # Tests mirroring source structure
```

## Core Abstractions

### Type System

| Type    | Location         | Values                                                                                                                                                                                | Purpose              |
| ------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| `Dtype` | `kumoapi.typing` | `bool`, `int`, `byte`, `int16`, `int32`, `int64`, `float`, `float32`, `float64`, `string`, `binary`, `date`, `time`, `timedelta`, `floatlist`, `intlist`, `stringlist`, `unsupported` | Data type            |
| `Stype` | `kumoapi.typing` | `numerical`, `categorical`, `multicategorical`, `ID`, `text`, `timestamp`, `sequence`, `image`, `unsupported`                                                                         | Semantic type for ML |

Dtype determines storage/parsing; Stype determines ML encoding. A column needs both.

### Relational Model Hierarchy

```
Connector                    # Connection to data warehouse/lake
    └── SourceTable          # Raw table reference (lazy, no schema)
        └── Table            # Kumo table with Column configs
            └── Graph        # Collection of Tables + Edges
```

| Class         | Location                    | Key Attributes                                          | Key Methods                                         |
| ------------- | --------------------------- | ------------------------------------------------------- | --------------------------------------------------- |
| `Connector`   | `connector/base.py`         | `source_type`, `name`                                   | `table(name)`, `table_names()`                      |
| `SourceTable` | `connector/source_table.py` | `name`, `connector`, `column_dict`                      | `_to_api_source_table()`                            |
| `Column`      | `graph/column.py`           | `name`, `dtype`, `stype`, `timestamp_format`            | `update()`, `CastMixin._cast()`                     |
| `Table`       | `graph/table.py`            | `source_table`, `columns`, `primary_key`, `time_column` | `infer_metadata()`, `validate()`, `snapshot()`      |
| `Edge`        | `graph/graph.py`            | `src_table`, `fkey`, `dst_table`                        | Bidirectional FK→PK relationship                    |
| `Graph`       | `graph/graph.py`            | `tables: Dict[str, Table]`, `edges: List[Edge]`         | `link()`, `validate()`, `snapshot()`, `visualize()` |

### PQL (Predictive Query Language)

PQL is a SQL-like DSL for specifying ML tasks declaratively:

```sql
PREDICT <target_column>.<aggregation>
FOR EACH <entity_table>.<entity_key>
[WHERE <conditions>]
[OVER <time_window>]
```

| Class             | Location                     | Purpose                                         |
| ----------------- | ---------------------------- | ----------------------------------------------- |
| `PredictiveQuery` | `pquery/predictive_query.py` | Parses PQL, validates against graph schema      |
| `TrainingTable`   | `pquery/training_table.py`   | Labeled examples generated from historical data |
| `PredictionTable` | `pquery/prediction_table.py` | Entity list for inference (future predictions)  |

Key PredictiveQuery methods:

- `generate_training_table()` → `TrainingTableJob` → `TrainingTable`
- `generate_prediction_table()` → `PredictionTableJob` → `PredictionTable`
- `suggest_model_plan()` → `ModelPlan` (hyperparameters, architecture)

## Training & Prediction Flow

```
PredictiveQuery
    ├── .generate_training_table() → TrainingTable
    ├── .suggest_model_plan() → ModelPlan
    │
Trainer(model_plan)
    ├── .fit(graph, training_table) → TrainingJob → TrainingJobResult
    │       └── GNN training on Kumo cloud infrastructure
    └── .predict(graph, prediction_table) → BatchPredictionJob → Results
```

| Component            | Location             | Responsibility                             |
| -------------------- | -------------------- | ------------------------------------------ |
| `Trainer`            | `trainer/trainer.py` | Orchestrates fit/predict, holds model plan |
| `TrainingJob`        | `trainer/job.py`     | Job handle with status polling, `attach()` |
| `BatchPredictionJob` | `trainer/job.py`     | Prediction job handle, output config       |
| `ModelPlan`          | `kumoapi.model_plan` | Architecture, hyperparameters, encodings   |

### Artifact Export

The SDK supports exporting trained model artifacts for external use:

| Component              | Location                    | Responsibility                   |
| ---------------------- | --------------------------- | -------------------------------- |
| `ArtifactExportConfig` | `artifact_export/config.py` | Export configuration options     |
| `ArtifactExportJob`    | `artifact_export/job.py`    | Job handle for export operations |

### Snapshotting

Tables and Graphs support **snapshotting** to freeze data for reproducible ML:

```python
table.snapshot(force_refresh=False, non_blocking=False)  # Returns TableSnapshotID
graph.snapshot(force_refresh=False, non_blocking=False)  # Returns GraphSnapshotID
```

- Snapshots enable iterative development without re-ingesting data
- `force_refresh=True` creates new snapshot even if one exists
- `non_blocking=True` returns immediately (job proceeds in background)
- Snapshots are **not** time-travel; only one version per table exists

## Async Job System

Long-running operations return `KumoFuture` objects:

```python
# Non-blocking
job = trainer.fit(graph, train_table, non_blocking=True)  # Returns TrainingJob
job.done()      # Check if complete
job.attach()    # Block until done, show progress
job.result()    # Block until done, return result

# Blocking (default)
result = trainer.fit(graph, train_table)  # Returns TrainingJobResult directly
```

Implementation in `futures.py`:

- Background event loop (`_KUMO_EVENT_LOOP`) runs in daemon thread
- `create_future(coro)` schedules coroutines in this loop
- `KumoProgressFuture` adds `attach()` for progress tracking

## Client Architecture

```
GlobalState.client: KumoClient
    ├── graph_api: GraphAPI                    # create, get, validate, snapshot graphs
    ├── table_api: TableAPI                    # create, get, validate, snapshot tables
    ├── source_table_api: SourceTableAPI
    ├── connector_api: ConnectorAPI
    ├── identity_api: IdentityAPI
    ├── pquery_api: PQueryAPI                  # Predictive query operations
    ├── training_job_api: TrainingJobAPI
    ├── batch_prediction_job_api: BatchPredictionJobAPI
    ├── generate_train_table_job_api / generate_prediction_table_job_api
    ├── artifact_export_api: ArtifactExportJobAPI
    ├── groups_api: GroupsAPI
    └── projects_api: ProjectsAPI
```

Each `*API` class handles CRUD for one resource type. `KumoClient` provides:

- Retry logic with exponential backoff
- Request/response serialization via `kumoapi` dataclasses
- Authentication header injection

## Code Generation

The SDK includes a code generation system for scaffolding Kumo configurations:

```
kumoai_studio/codegen/
├── cli.py            # CLI entry point
├── context.py        # Generation context and state
├── edits.py          # Code modification operations
├── exceptions.py     # Codegen-specific errors
├── generate.py       # Main generation orchestration
└── handlers/
    ├── connector.py  # Generate connector configurations
    └── graph.py      # Generate graph definitions
```

## Integration Points

| Dependency                 | Type            | Purpose                                       |
| -------------------------- | --------------- | --------------------------------------------- |
| `packages/kumo-api`        | In-tree package | Shared types and request/response dataclasses |
| `rest/` and `kumo/pquery/` | In-tree service | Coordinate API and SDK changes in one Kumo PR |
| Snowflake                  | Connector       | `SnowflakeConnector`, SPCS native app support |
| AWS S3                     | Connector       | `S3Connector` for data lake access            |
| BigQuery                   | Connector       | `BigQueryConnector` for GCP                   |
| Databricks                 | Connector       | `DatabricksConnector`, Unity Catalog          |

## Key Patterns

- **CastMixin**: Enables `Edge._cast((src, fkey, dst))` or `Edge._cast({'src_table': ...})` - flexible construction from tuples/dicts/dataclasses
- **GlobalState singleton**: Thread-local `kumoai_studio.global_state.client` stores authenticated client (see `_singleton.py`)
- **Environment auto-init**: `KUMO_API_KEY` + `KUMO_API_ENDPOINT` → automatic initialization on import
- **Lazy validation**: Tables/Graphs validate on-demand via `.validate()`, not at construction
- **Metadata inference**: `table.infer_metadata()` calls backend to detect dtypes/stypes from sample data
- **Optional dependencies**: The Snowflake/SPCS integration is an optional extra in `pyproject.toml`

## Entry Points

| Task                     | Start Here                                                                      |
| ------------------------ | ------------------------------------------------------------------------------- |
| Initialize SDK           | `kumoai_studio.init(url, api_key)` or set `KUMO_API_KEY` env var                |
| Connect to data          | `kumoai_studio.S3Connector()`, `SnowflakeConnector()`, etc.                     |
| Define relational schema | `kumoai_studio.Table.from_source_table()`, `kumoai_studio.Graph(tables, edges)` |
| Write ML task (PQL)      | `kumoai_studio.PredictiveQuery(graph, "PREDICT ... FOR EACH ...")`              |
| Train model              | `trainer.fit(graph, training_table)`                                            |
| Run predictions          | `trainer.predict(graph, prediction_table, output_config=...)`                   |
| Export model artifacts   | `ArtifactExportJob` with `ArtifactExportConfig`                                 |
| Load saved artifacts     | `Graph.load(id)`, `Table.load(id)`, `Trainer.load(job_id)`                      |
| Generate code scaffolds  | `kumoai_studio.codegen` module or CLI                                           |
| Run tests                | `python -m pytest test/`                                                        |
