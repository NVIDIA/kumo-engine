<p align="center">
  <img height="180" src="https://kumo-sdk-public.s3.us-west-2.amazonaws.com/kumo-logo-pink.svg" />
</p>

______________________________________________________________________

`kumo.engine` (local) is a local-only, in-process implementation of the Kumo
relational deep learning pipeline: define tables and graphs, materialize a
snapshot with `kumo-core`, and train/predict with a GNN — all without a Kumo
account or a network call. It has no notion of a remote backend, a project, or
a hosted job.

For a configured Kumo service (`kumoai_studio.init(url=..., api_key=...)`
and service-managed training), use the separate `kumo-sdk` package instead.

## Object model

Pass connectors, tables, graphs, and trainers directly as Python objects;
there is no backend selection or hosted project/job registry. A `Table`
holds its source and schema, and a `Graph` holds tables and edges.
`Table.from_source_table()` reads source metadata and may sample rows to
infer column types; `Graph.snapshot()` performs graph ingestion.

`Graph.snapshot()`, `Trainer.fit()`, and `Trainer.predict()` are synchronous.
Training and prediction require an explicit `snapshot=`; retained table rows
can be used with another snapshot as described under
[local storage and results](#local-storage-and-results).

Temporal PQL windows support calendar months, for example
`PREDICT COUNT(events.*, 0, 1, months) FOR EACH users.id`. Month windows and
integer time options use calendar offsets, including month-end and leap-year
boundaries, through the same timestamp generator as Kumo Studio.

The package reuses `kumoapi` model plans and connector plugin contracts.
Its [output configurations](kumo/engine/trainer/output_config.py) hold live
connector objects rather than the connector IDs used by service requests.

## Installation

```
pip install kumo-engine
```

Connector plugins (Snowflake, S3, GCS, BigQuery, Databricks) are installed as
extras, mirroring `kumo-sdk`:

```
pip install "kumo-engine[snowflake]"
```

## Text columns

Text columns are encoded with word vectors. The default model, `glove.6B-50d`,
ships as the `text` extra:

```
pip install "kumo-engine[text]"
```

For any other model, register it in a store and pass the store to the trainer:

```python
from kumo.engine import EmbeddingModelStore, GloVeVectors, Trainer
from kumoapi.encoder import TextEmbedding

store = EmbeddingModelStore()
store.add("desc", GloVeVectors("~/Downloads/glove.42B.300d.zip"))
plan.column_processing.encoder_overrides = {
    "articles.detail_desc": TextEmbedding(model="desc"),
}
trainer = Trainer(plan, embedding_models=store)
```

`GloVeVectors` takes an official Stanford release `.zip`
(https://nlp.stanford.edu/projects/glove/), converted once on first use, or a
directory of converted files; pass `dim=` for zips that hold several sizes.
Load a trained model with `Trainer.load(bundle, embedding_models=store)`;
it finds the vectors by content, whatever name they are registered under.
Custom models implement `kumo.engine.TokenEmbeddingModel`. Kumo never downloads word
vectors itself.

## Local files

Read CSV or Parquet below a filesystem root with `FileConnector`; no credentials
or connector extra are required. For `restaurant-data/users.csv` with an `id`
column:

```python
from kumo.engine import FileConnector, Graph, Table, configure

configure(scratch_dir="/mnt/nvme/kumo-scratch")
files = FileConnector(root_dir="/mnt/nvme/restaurant-data")
users = Table.from_source_table(files["users"], primary_key="id")
snapshot = Graph(tables=[users], edges=[]).snapshot()
```

Use a table mapping when graph names differ from source names, such as
`Graph(tables={"users": users})` for a table read from `users.parquet`.
Edges and PQL use the mapping keys; connector reads keep the original source names.


Table names are relative paths, optionally with a `.csv`/`.parquet` suffix,
or directories of parts in one format. Absolute paths, URIs, traversal and
globs are unsupported. For CSV timestamps, set `dtype=Dtype.time` (from
`kumoapi.typing`) and `timestamp_format` with `Table.with_column()`.

The root and scratch directory must not overlap. Directory symlinks and links
escaping the root are rejected; file symlinks within the root are allowed.
Keep sources and links unchanged during reads. Versions use paths, sizes and
modification times, not content hashes; do not restore timestamps after edits.
Source identities use canonical paths and are not portable across machines.

Use a separate output root for artifacts. Single-file outputs overwrite; use
an empty directory for partitioned outputs. Append is unsupported. Model
publication copies bundles to an empty destination and rejects symlinks and
overlap. It does not register models; `fetch_model` is unsupported.

## Local storage and results

Snapshots, generated tables, trained model bundles, and unexported predictions
are written below the configured scratch root. It defaults to
`~/.cache/kumo-engine`; call `configure(scratch_dir=...)` once, before creating
graphs or trainers, to place it on a larger local disk. `Graph.snapshot()`
warns if it is on network-backed storage such as EBS. Each `fit()` and
`predict()` call uses a separate run directory, so a later call does not
overwrite an earlier result.

Snapshot indexes are keyed by the graph definition and connector-reported
source versions. A completed index with the same key is reused; a changed
version gets a separate directory. Snapshot builds for the same graph are
serialized with a filesystem lock. See [the implementation](kumo/engine/graph.py)
for the fingerprint and locking rules.

The return values expose the retained files directly:

```python
from kumo.engine import configure

configure(scratch_dir="/mnt/nvme/kumo-scratch")

fit_result = trainer.fit(training_table, snapshot=snapshot)
print(fit_result.bundle_dir)       # pass this path to Trainer.load(...)

prediction_result = trainer.predict(prediction_table, snapshot=snapshot)
print(prediction_result.files)     # selected Parquet files
predictions = prediction_result.frame  # collect them into a pandas DataFrame
```

`fit()` requires an explicit graph snapshot. To keep an existing table's rows
and labels but train with updated graph features, pass the updated snapshot:

```python
updated_snapshot = graph.snapshot()
fit_result = trainer.fit(training_table, snapshot=updated_snapshot)
```

Each fit builds its training task from the selected snapshot and the saved
training-table artifacts. The table retains its rows, labels, splits, and
generation settings; it does not retain a graph-bound training plan. Fitting
does not regenerate or mutate the table.

### Caller-supplied training and prediction rows

A connector table can supply labeled training rows directly. The table uses
the standard `ENTITY1`, `TARGET1` and, for temporal queries, `TIMESTAMP`
columns. A named split column contains `0` for train, `1` for validation and
`2` for test; a weight column carries per-row loss weights:

```python
from kumo.engine import TrainingTable
from kumoapi.train import TrainingTableSpec

training_table = TrainingTable.from_source_table(
    files["curated_training.parquet"],
    TrainingTableSpec(weight_col="WEIGHT", split_col="SPLIT"),
    pq=pquery,
    snapshot=snapshot,
)
result = trainer.fit(training_table, snapshot=snapshot)
```

When the source has no split column, the task's normal inferred split assigns
the supplied rows. Pass `TrainingTableGenerationPlan(split=...)` to select a
different split explicitly.

A custom prediction table similarly supplies the exact entities and anchor
times to score. It contains `ENTITY1` and, for a temporal query, `TIMESTAMP`;
no target column is needed:

```python
from kumo.engine import PredictionTable

prediction_table = PredictionTable.from_source_table(
    files["curated_population.parquet"],
    pq=pquery,
    snapshot=snapshot,
    plan=trainer.training_table_generation_plan,
)
predictions = trainer.predict(prediction_table, snapshot=snapshot)
```

Both constructors stage and validate their source rows immediately. Their
results retain local artifact paths and query/generation metadata, without a
connector registry, backend IDs, or a graph-bound plan. The later `fit` or
`predict` call selects its graph data independently with `snapshot=`.

Keep the scratch directory for as long as those paths must remain usable.
`ModelOutputConfig` and `PredictionOutputConfig` can additionally copy results
through a connector when they need storage outside the scratch root.

For forecasting, custom rows must also carry the precomputed sequence columns
that the corresponding generated table would contain: `future_labels` for
training, `autoregression_labels` when lag features are enabled, and
`year_over_year_labels` when year-over-year features are enabled. The custom
constructors preserve these columns; they do not derive historical or future
sequences from `ENTITY1`, `TARGET1`, and `TIMESTAMP` alone.

## BigQuery staged ingestion

BigQuery streams through the Storage Read API by default. For large tables,
configure an owned GCS prefix to export Parquet and download it locally:

```python
from kumo.engine import BigQueryConnector

events = BigQueryConnector(
    project_id="data-project",
    dataset_id="analytics",
    staging_uri="gs://data-staging/kumo/run-123",
    require_staged=True,
)
```

`require_staged=True` fails if that prefix cannot be used; it never silently
falls back to streaming. Supplying `staging_uri` without `require_staged`
keeps staged-first behavior with the existing read-time streaming fallback.

## Databricks staged ingestion

Databricks streams through CloudFetch by default. To unload Parquet through
an existing Unity Catalog volume and download it locally:

```python
from kumo.engine import DatabricksConnector, DatabricksIdentity

identity = DatabricksIdentity(
    host="https://workspace.cloud.databricks.com",
    client_id="service-principal-id",
    client_secret="...",
)
events = DatabricksConnector(
    catalog="main",
    schema_name="analytics",
    warehouse_id="warehouse-id",
    identity=identity,
    staging_volume_path="/Volumes/main/kumo/staging",
    require_staged=True,
)
```

`require_staged=True` propagates staging failures without streaming fallback.
Supplying only `staging_volume_path` keeps the backend's staged-first behavior
and existing fallback policy. The configured identity needs access to the
volume as well as the source table.

## Model-plan suggestions

Given a `pquery = PredictiveQuery(graph, query)` and its existing
`snapshot = graph.snapshot()`, suggest a plan locally, inspect it and customize
it before constructing a trainer:

```python
from kumo.engine import Trainer
from kumoapi.model_plan import RunMode

plan = pquery.suggest_model_plan(snapshot, run_mode=RunMode.FAST)
print(plan.training_job.metrics)
plan.optimization.max_epochs = 5
trainer = Trainer(plan)
```

Suggestions use graph/query metadata without starting training or generating
a training table.
Inferred settings are resolved during training.

## Status

The core pipeline is implemented and runs real `kumo-core`/DiskGraph
training and batch prediction, in-process: `Table`/`Graph`/`Graph.snapshot()`,
`PredictiveQuery.validate`/`get_task_type`/`suggest_model_plan`/
`generate_training_table`/`generate_prediction_table`, and
`Trainer.fit`/`predict`/`load` (direct local bundle reload) and
`load_from_connector` (model reload from Snowflake/S3/GCS storage).

Not yet implemented, or explicitly out of scope for this package:

- `PredictiveQuery.suggest_prediction_table_plan`/`suggest_distilled_model_plan`
  -- deferred by design, not merely unimplemented.
- Custom SQL tables -- `Table.from_source_table()` reads a table's schema
  as-is (which may sample source data to resolve ambiguous column types);
  there is no `sql=` query-table support the cloud SDK has.
- BigQuery is read-only: no sink, so it can't be an `output_connector`,
  `export_config` connector, or `Trainer.load_from_connector(...)` target.
- Scoped per-table ingest credentials, and a table staged across multiple
  directories, both raise `NotImplementedError` in `Graph.snapshot()`.

See inline `TODO(local-sdk)` markers for the specifics of each gap.

### Reloading a local model

Every successful `Trainer.fit(training_table, snapshot=snapshot)` returns a
result whose `bundle_dir` already exists on local disk, with the training metadata needed
for reload. No export is required:

```python
result = trainer.fit(training_table, snapshot=snapshot)
# In a fresh Python process:
restored = Trainer.load(result.bundle_dir)  # pass the recorded path
```

For an exported model, use
`Trainer.load_from_connector(connector, path="model/version")`, or
`location=ModelRegistryLocation(...)` for Snowflake Model Registry. Rebuild
the live graph and snapshot before generating prediction inputs. Bundles
contain trained artifacts, not source data or graph storage.

When `fit(..., export_config=...)` publishes a model, its result also exposes
`published_model`, the connector's `PublishedModel` with the actual name,
version and URI. Without an export this is `None`. Training metrics and a
small projection of the training metadata are passed to the connector;
Snowflake records them for Kumo model discovery and registry comparison.
For a registry-assigned version, retain that exact version for later reload:

```python
result = trainer.fit(
    training_table,
    snapshot=snapshot,
    export_config=ModelOutputConfig(
        snowflake, location=ModelRegistryLocation("CHURN_GNN")),
)
published = result.published_model
restored = Trainer.load_from_connector(snowflake, location=ModelRegistryLocation(
    published.name, version=published.version))
```

S3, GCS and file connectors expose their published path/URI through the same
result; this does not add registry versioning to those destinations.

## Prediction and embedding destinations

Request both artifacts with separate destinations. Mapping keys select which
artifacts to generate; paths receive no artifact-name suffix. Use a trailing
`/` for a dataset directory, regardless of the number of output parts.

```python
from kumo.engine import PredictionOutputConfig, S3Connector

outputs = PredictionOutputConfig(
    connector=S3Connector(),
    output_paths={
        "predictions": "s3://my-bucket/predictions/project/",
        "embeddings": "s3://my-bucket/embeddings/project/",
    },
)
result = trainer.predict(prediction_table, snapshot=snapshot,
                         output_config=outputs)
```

Use the connector's existing identity configuration, or its ambient credential
chain. Engine supports this interface with S3, GCS, and local file connectors;
ADLS is currently available through the hosted SDK only. Rooted connectors also
accept relative destinations. Do not combine `output_paths` with `artifact_type`
or `table_name`. Existing single-artifact configurations remain supported.

`result.files`, `result.frame`, and `result.num_rows` refer to predictions when
both artifacts were requested, or embeddings for an embeddings-only request.
`result.export(outputs)` exports existing artifacts without rerunning inference;
it cannot generate artifacts absent from the original run, and rejects
`extra_fields` (pass them to `Trainer.predict(output_config=...)`). Directory exports
retain the existing ownership checks: retries can reuse their own output, but
a different export requires an empty destination. Explicit `.parquet`/`.csv`
paths retain existing single-file/multipart export behavior.

Prediction and embedding destinations are exported independently. If a later
output fails, earlier outputs may already be complete and the failed directory
may contain partial parts without `_SUCCESS`. Retry the retained result's
`export()` with the same destinations. Directory retries remove an existing
`_SUCCESS` before copying and publish it again only after that destination
completes; this requires permission to delete that marker. Readers should wait
for `_SUCCESS` before consuming a directory. Exports do not provide a
transaction across destinations, atomic single-file replacement, or isolation
from concurrent writers or readers.
