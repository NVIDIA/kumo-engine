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

Training-table generation rejects classification targets with fewer than two
classes or more than the configured class limit, including custom training
tables. Temporal training requires `train_end_offset < train_start_offset`
when both are supplied. Prediction anchors before the query's source-data
range are rejected; future anchors are allowed. These checks share the same
validation rules as Kumo Studio.


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

Integer and string columns with identifier-like names, such as `user_id`, are
automatically assigned the `ID` semantic type, including columns that are not
primary keys. Floating-point columns without a matching name are inferred from
sampled values.
Names that indicate numerical quantities, such as `num_items`, `price`, and
`score`, assign the numerical semantic type to integer and floating-point
columns, even when the sample contains only a few distinct values.
Types supplied by the source or set with `with_column()` take precedence over
automatic inference.

Columns used as foreign keys in supported `Edge` relationships have the `ID`
semantic type in the graph, even if the source table inferred them as numerical
or categorical. The source `Table` object retains its original column types.
Omitted self-relations leave their columns' semantic types unchanged.

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
overlap. It does not register models; reload a published bundle by path with
`Trainer.load_from_connector(connector, path=...)`.

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

### Holdout predictions

`fit_result.holdout_df()` reads the test-split predictions as a pandas
DataFrame, like `holdout_df()` on a hosted training job. The file is
`fit_result.holdout_path`; it is not part of the published model bundle.
Link-prediction tasks produce no holdout: `holdout_path` is `None` and
`holdout_df()` raises `FileNotFoundError`. The same holds for results from
`Trainer.load`. Holdouts above 1 GiB in memory are cut to fit, and
`fit_result.holdout_truncated` is then `True` (`None` if the size record
is unreadable; `fit()` still succeeds).

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

### Warm-start training

Pass a fitted or restored trainer to start from its selected checkpoint:

```python
previous = Trainer.load("/models/previous/published-model")
# Or: Trainer.load_from_connector(registry, location=location)
trainer = Trainer(model_plan)
result = trainer.fit(training_table, snapshot=snapshot, warm_start=previous)
```

Warm start reuses weights, optimizer state, fitted encoders, architecture, and
neighbor sampling. The destination plan still controls training and refits.
Its configured `model_plan` remains unchanged for subsequent ordinary fits;
the effective warm-start plan is saved in the result's bundle. New bundles
select the same checkpoint as hosted warm start: full refit, then train/validation
refit, then the original fit, according to which modes completed.
The bundle's persisted selected plan is authoritative, even if the source
trainer's public `model_plan` was subsequently edited. Task types must match;
forecasting warm starts are unsupported. Graph data is supplied by `snapshot`.
The source bundle remains unchanged; the result is a separate reloadable bundle.

If the source checkpoint was refitted on train/validation or all rows,
warm-starting on overlapping data can reuse rows that the destination treats
as held out. Validation or test metrics on those rows can be optimistic.
For independent evaluation, use held-out rows that were not used to train
or refit the source model.

Bundles produced before selected trial plans were persisted may contain an
AutoML search space instead of the fitted architecture; those bundles are
still loadable for prediction but are not reliable warm-start sources.

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

### Exporting a generated training table

To train with per-row weights, export a generated table, add a weight column,
and load it back:

```python
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from kumo.engine import FileConnector, TrainingTable, TrainingTableOutputConfig
from kumoapi.train import TrainingTableSpec

training_table = pquery.generate_training_table(snapshot)
exports = FileConnector(root_dir="/data/exports")
training_table.export(TrainingTableOutputConfig(exports, "training_table"))

weighted_dir = Path("/data/exports/weighted")
weighted_dir.mkdir()
for part in sorted(Path("/data/exports/training_table").glob("*.parquet")):
    source = pq.ParquetFile(part)
    schema = source.schema_arrow.append(pa.field("WEIGHT", pa.float64()))
    with pq.ParquetWriter(weighted_dir / part.name, schema) as writer:
        for batch in source.iter_batches():
            weight = pa.repeat(pa.scalar(1.0), batch.num_rows)
            writer.write_batch(batch.append_column("WEIGHT", weight))

weighted = TrainingTable.from_source_table(
    exports["weighted"],
    TrainingTableSpec(weight_col="WEIGHT", split_col="SPLIT"),
    pq=pquery,
    snapshot=snapshot,
)
trainer.fit(weighted, snapshot=snapshot)
```

Replace the constant `1.0` with your own per-row weights. Each exported part
is read and rewritten one batch at a time, so the table never has to fit in
memory. When the export goes to a Snowflake or Databricks
table, add the `WEIGHT` column in the warehouse instead and load that table.

The export holds exactly the rows of the generated train, validation and test
stages: `ENTITY1`, `TIMESTAMP`, `TARGET1`, any sequence columns the task needs
(link-prediction target frequencies, forecasting labels), and `SPLIT` (`0`
train, `1` validation, `2` test). Loading it with `split_col="SPLIT"` keeps
every row in its original stage. File and object-store destinations (local,
S3, GCS) are written as a directory of Parquet parts; exporting the same table
again replaces it, and a different table needs an empty directory. Snowflake
and Databricks destinations are tables, with `write_mode` choosing overwrite
or append.

- For link prediction, `WEIGHT` must be a list with one weight per element of
  `TARGET1`.
- Forecasting tables are loaded back without `split_col`; their split is
  recomputed from the timestamps.
- A static link-prediction table that was split per link loses the rule that
  excludes a user's training links from validation and test ranking, so its
  validation metrics can differ from the original.
- Snowflake stores list columns as JSON strings, so export link-prediction
  and forecasting tables to files.

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

## Custom SQL tables

A Snowflake, Databricks or BigQuery connector can define a table from a
read-only `SELECT`. The query runs in the warehouse and only its result is
ingested, through the same reader (and staging options) as a named table:

```python
from kumo.engine import Graph, Table
from kumoapi.graph import Edge

active_users = Table.from_sql(
    events,
    "SELECT user_id, country, signup_ts FROM users WHERE active",
    primary_key="user_id",
    time_column="signup_ts",
)
graph = Graph({"active_users": active_users, "orders": orders},
              edges=[Edge(src_table="orders", fkey="user_id",
                          dst_table="active_users")])
snapshot = graph.snapshot()
```

`events.sql(query)` returns the `SQLSourceTable` to pass to
`Table.from_source_table` instead. The query must be one statement with
named output columns and may only read tables inside the connector's own
database/catalog/project and schema/dataset; these checks are shared with
the cloud platform. Schema inference samples the query's result, and on
BigQuery a query estimated to scan more than 10 GiB is not sampled. To skip
inference, pass `Table(events, name, definition=kumoapi.table.TableDefinition(
cols=..., source_table=events.sql(query).to_source_table(), ...))`.

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
`load_from_connector` (model reload from local/S3/GCS paths or Snowflake).

Not yet implemented, or explicitly out of scope for this package:

- `PredictiveQuery.suggest_prediction_table_plan`/`suggest_distilled_model_plan`
  -- deferred by design, not merely unimplemented.
- BigQuery stores predictions and embeddings only: it can't be an
  `export_config` connector or a `Trainer.load_from_connector(...)` target.
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

## BigQuery prediction output

A `BigQueryConnector` writes predictions or embeddings to a table in its own
dataset, either during `predict` or later through `result.export`:

```python
from kumo.engine import (
    BigQueryConnector,
    BigQueryIdentity,
    PredictionOutputConfig,
)
from kumoapi.jobs import PredictionArtifactType, WriteMode

output = BigQueryConnector(
    identity=BigQueryIdentity(),  # application default credentials
    project_id="data-project",
    dataset_id="predictions",
    staging_uri="gs://data-staging/kumo",
)
result = trainer.predict(
    prediction_table,
    snapshot=snapshot,
    binary_classification_threshold=0.5,
    output_config=PredictionOutputConfig(
        connector=output,
        artifact_type=PredictionArtifactType.PREDICTIONS,
        table_name="user_scores",
    ),
)
# Later, without rerunning inference:
result.export(PredictionOutputConfig(
    connector=output,
    artifact_type=PredictionArtifactType.PREDICTIONS,
    table_name="user_scores_history",
    write_mode=WriteMode.APPEND,
))
```

- `table_name` is an unqualified table ID. Each configuration writes one
  artifact; `output_paths` is for file and object-store connectors.
- `OVERWRITE` replaces the table's rows and schema. `APPEND` adds rows and is
  strict: every output column must be a column of the existing table with the
  same type (integers may also go into `FLOAT` or `NUMERIC` columns, and an
  all-null column fits any type); otherwise it fails before writing and names
  each mismatched column and its types.
- Every export commits at once, so a failed export leaves the table's rows
  unchanged. Rerunning an `APPEND` adds the rows again; an `APPEND` to a
  missing table creates it first.
- Output is written the way BigQuery input is read. With a usable
  `staging_uri`, output files are uploaded under
  `<staging_uri>/_kumo.export/<id>/`, loaded by one load job and deleted.
  Otherwise they are streamed through the BigQuery Storage Write API into
  pending streams committed in one call; `OVERWRITE` commits into an expiring
  `_kumo_export_*` table and replaces the destination with one
  `WRITE_TRUNCATE` query over it. `require_staged=True` fails instead of
  streaming when staging is unusable. A bucket outside the dataset's location
  is not used for staging.
- Cost: load jobs are free; the Storage Write API is billed per GB beyond its
  monthly free tier, and a streamed `OVERWRITE` also bills the query over the
  output. Prefer staging for large or frequent exports.
- Rows written through the Storage Write API are queryable immediately, but
  table metadata such as `num_rows` lags, and a copy job run right after may
  miss them.
- `OVERWRITE` keeps the table's partitioning and clustering, and the
  descriptions and policy tags of columns that keep their name and type. A
  streamed `OVERWRITE` restores them just after replacing the rows.
- Embeddings land as `REPEATED FLOAT` columns, timestamps as `TIMESTAMP`.
- `predict` checks the destination before scoring: the dataset exists in the
  connector's `location`, the identity can create or update the table and run
  load jobs or write streams, and staging is usable when required. The
  identity needs `roles/bigquery.dataEditor` on the dataset,
  `roles/bigquery.jobUser` on the billing project and, with staging,
  `roles/storage.objectAdmin` on the bucket.
