<p align="center">
  <img height="180" src="https://raw.githubusercontent.com/NVIDIA/kumo-engine/main/ui/src/images/kumo_logo_green.svg" alt="Kumo" />
</p>

______________________________________________________________________

`kumo.engine` runs relational deep learning in-process: define tables and graphs,
materialize a snapshot, and train or predict with a GNN.

## Object model

Pass connectors, tables, graphs, and trainers directly as Python objects.
A `Table` holds its source and schema, and a `Graph` holds tables and edges.
Import these types with `from kumo.engine import Column, Edge, Graph, Table`.
`Column` and `Edge` come from `kumoapi`. `Table.from_source_table()` reads metadata
and may sample rows to infer column types; `Graph.snapshot()` ingests the graph.

`Graph.snapshot()`, `Trainer.fit()`, and `Trainer.predict()` are synchronous.
Training and prediction require `snapshot=`; reuse table rows with another
snapshot as described under
[local storage and results](#local-storage-and-results).

Classification targets must have between two and the configured maximum number
of classes, including custom training tables. Temporal training requires
`train_end_offset < train_start_offset` when both are supplied. Prediction
anchors may be in the future, but cannot precede the query's source-data range.

Temporal PQL windows support calendar months, for example
`PREDICT COUNT(events.*, 0, 1, months) FOR EACH users.id`. Month windows and
integer time options respect month-end and leap-year boundaries.

The package uses `kumoapi` model plans and connector plugin contracts.
[Output configurations](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-engine/kumo/engine/trainer/output_config.py)
take connector objects directly.

### Training device

Choose the accelerator when constructing a trainer:

```python
trainer = Trainer(model_plan, accelerator="cpu")
```

Accepted values are `auto` (default), `cpu`, `gpu`, and `mps`, using Lightning's
device selection. The choice applies to training, refits, explanations, and
prediction. An explicitly requested accelerator must be available on the host.

Device selection is local to the instance and is not saved in model bundles.
Select it again when loading, for example
`Trainer.load(bundle_path, accelerator="cpu")` or
`Trainer.load_from_connector(registry, path="model", accelerator="cpu")`.

## Installation

Requires Linux x86_64, CPython 3.11 or newer, and glibc 2.34 or newer.
For ARM64, see the
[DGX Spark installation guide](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-engine/dgx-spark.md).

```
pip install kumo-engine
```

Install connector dependencies with the `snowflake`, `s3`, `gcs`, `bigquery`,
or `databricks` extra:

```
pip install "kumo-engine[snowflake]"
```

## Text columns

Text columns use word vectors. Install the default model, `glove.6B-50d`, with:

```
pip install "kumo-engine[text]"
```

Register other models in a store and pass it to the trainer:

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

`GloVeVectors` accepts an [official Stanford ZIP](https://nlp.stanford.edu/projects/glove/)
or converted directory. ZIPs are converted on first use; pass `dim=` for archives
with several sizes. Reload with `Trainer.load(bundle, embedding_models=store)`;
vectors are matched by content. Custom models implement `TokenEmbeddingModel`.
Kumo does not download vectors automatically.

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

Identifier names such as `user_id` give integer and string columns the `ID`
semantic type, even for non-primary keys. Quantity names such as `num_items`,
`price`, and `score` give integer and floating-point columns the numerical type,
even with few distinct values. Other floating-point columns use sampled values.
Source types and `with_column()` overrides take precedence over inference.

Foreign keys in supported `Edge` relationships receive the `ID` type in the graph;
the source `Table` keeps its original types. Omitted self-relations leave types
unchanged.

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
overlap. Reload a published bundle with
`Trainer.load_from_connector(connector, path=...)`.

## Visualize table relationships

Display a graph's declared tables and key relationships in a notebook, or save the same view as an SVG:

```python
graph.visualize()                        # inline notebook display
svg = graph.visualize("relationships.svg", show_cols=True)
graph.visualize(layout="compact")        # balance crowded groups
```

The return value is SVG text with notebook display support.
Choose Standard or Compact from the notebook's layout selector.
The default `layout="auto"` chooses an initial arrangement from graph structure, independently of viewport size.
Use `layout="standard"` to retain the layered relationship view.
Each arrangement is computed once; switching layouts does not read source data.
Fullscreen fits the selected arrangement into the larger viewport without moving tables or rerouting relationships.
Use the zoom controls or Ctrl/Cmd+wheel (including trackpad pinch) to zoom, and Fit to show the whole graph again.
Drag the background or a table to pan the whole graph; individual tables stay fixed relative to each other.
Ordinary scrolling moves through the notebook.
Fit reserves tooltip space beside the graph when the viewport allows it.
Hover may pan the view to make room for details while retaining the hovered item; extreme zoom can require overlap.
Connected-table lists use up to two columns of 15 single-line entries, with ellipsis for long names.
Fit changes only the view, and toolbar controls keep their size during graph zoom.
Interactive controls require trusted HTML output and fullscreen requires permission from the notebook host.
SVG export retains the initial selected arrangement at its full dimensions.
By default, visualization uses declared metadata without reading source rows or requiring Graphviz or other rendering dependencies.
Hover or keyboard-focus a header or column for schema details; static SVG and restricted notebook output retain native tooltips.
Optionally use `graph.visualize(sample_rows=3)` to fetch up to 30 preview rows per table once and show up to three non-null examples independently for each visible column.
Samples have no guaranteed ordering, and previews can run warehouse queries and incur costs; the row limit does not bound bytes scanned.
Unavailable previews and columns with no non-null values in the preview are labeled, and long values are abbreviated.
Samples become part of saved notebook HTML; SVG text and exports always contain metadata only.
The default view shows primary keys, relationship foreign keys, and the declared time column (`TS`).
`show_cols=True` adds columns up to 10 rows per table, or 5 for graphs with more than 10 tables; keys and the declared time column always remain visible, even above that limit.
A footer counts hidden columns.
Visualization accepts at most 30 tables; larger graphs raise an error so no tables or relationships are silently omitted.
Self-relations are rejected because graph snapshots do not support them.

### Developing another notebook widget

Internal widgets share `kumo.engine._widget` for theme values, HTML chrome, buttons, safe JSON payloads and instance cleanup.
Keep feature data, rendering and interactions in the feature module; a chart does not need graph tables, layouts or camera controls.
Scope feature-specific CSS under a feature root class; the shared toolbar and button defaults use lower specificity so later notebook outputs do not override feature styling.
This synthetic example uses the same shell and button styling as graph visualization:

```python
from html import escape
from IPython.display import HTML, display
from kumo.engine._widget import THEME, button, render_widget

label = "Synthetic contribution: 75%"
body = (
    '<div class="kumo-widget-toolbar">'
    + button("chart-values", "Toggle values", text="#") + '</div>'
    '<svg width="320" height="64" viewBox="0 0 320 64" role="img" '
    f'aria-label="{escape(label)}" font-family="{THEME["font"]}">'
    f'<rect x="0" y="8" width="225" height="24" fill="{THEME["accent"]}"/>'
    f'<text class="chart-value" x="0" y="55" fill="{THEME["foreground"]}">'
    '75%</text></svg>'
)
script = """
const toggle=host.querySelector('.chart-values');
toggle.addEventListener('click',()=>{
  host.querySelector('.chart-value').toggleAttribute('hidden');
},{signal});
toggle.disabled=false;
"""
display(HTML(render_widget(body, script=script)))
```

These helpers are internal implementation interfaces, not a public explainability API.
`body`, `style` and `script` accept trusted feature-authored code; escape user text with `html.escape`, pass structured values through `data=`, and read the decoded `data` in the initializer.
The initializer also receives `host`, a unique `widgetId`, `signal` and `onDispose(callback)`; scope DOM queries to `host`, bind listeners with `signal`, and register timers or observers for cleanup when the output is removed.
Each render creates a new instance; render again after removal rather than reinserting disposed HTML.
Keep meaningful static content and enable controls only after their handlers are initialized.
Interactive output requires a notebook host that executes trusted inline scripts; the shell adds no network dependency and does not mount React components or load chart data.

## Local storage and results

Snapshots, generated tables, model bundles, and unexported predictions live under
`~/.cache/kumo-engine`. Call `configure(scratch_dir=...)` before creating graphs
or trainers to choose another local disk. `Graph.snapshot()` warns about
network-backed storage such as EBS. Each `fit()` and `predict()` uses a separate
run directory.

Snapshot indexes are reused by graph definition and source version; changed
versions get separate directories. Builds for the same graph use a filesystem
lock. See the [implementation](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-engine/kumo/engine/graph.py).

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

`fit_result.holdout_df()` reads test predictions from `holdout_path` into a pandas
DataFrame. Holdouts are excluded from published bundles. Link-prediction tasks
and loaded trainers have no holdout; `holdout_df()` raises `FileNotFoundError`.
Holdouts are capped at 1 GiB in memory. `holdout_truncated` is `True` if rows were
dropped, or `None` if the size record is unreadable; training still succeeds.

To reuse table rows and labels with updated graph features, pass a new snapshot:

```python
updated_snapshot = graph.snapshot()
fit_result = trainer.fit(training_table, snapshot=updated_snapshot)
```

Each fit uses the selected snapshot and saved table artifacts, preserving the
table's rows, labels, splits, and generation settings.

### Warm-start training

Pass a fitted or restored trainer to start from its selected checkpoint:

```python
previous = Trainer.load("/models/previous/published-model")
# Or: Trainer.load_from_connector(registry, location=location)
trainer = Trainer(model_plan)
result = trainer.fit(training_table, snapshot=snapshot, warm_start=previous)
```

Warm start reuses weights, optimizer state, fitted encoders, architecture, and
neighbor sampling. The destination plan controls training and refits; its
`model_plan` stays unchanged. The effective plan is saved in a new bundle, leaving
the source intact. Checkpoint priority is full refit, train/validation refit, then
original fit, depending on which completed. The saved plan takes precedence over
later edits to the source trainer's plan. Task types must match; forecasting warm
starts are unsupported.

If source training or refitting included the destination's held-out rows,
validation and test metrics may be optimistic. Use independent held-out data
for evaluation.

Older bundles may contain an AutoML search space instead of the fitted
architecture. They support prediction but are unreliable warm-start sources.

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

Without a split column, rows use the task's inferred split. Override it with
`TrainingTableGenerationPlan(split=...)`.

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

Replace `1.0` with per-row weights. The example processes one batch at a time.
For Snowflake or Databricks exports, add `WEIGHT` in the warehouse and reload.

Exports preserve generated train, validation, and test rows, including entity,
target, timestamp, task-specific columns, and `SPLIT`. Reload with
`split_col="SPLIT"` to preserve each stage. File, S3, and GCS exports use Parquet
directories: the same table can replace its export; a different table needs an
empty directory. Warehouse exports use `write_mode` for overwrite or append.

- For link prediction, `WEIGHT` must be a list with one weight per element of
  `TARGET1`.
- Forecasting tables are loaded back without `split_col`; their split is
  recomputed from the timestamps.
- A static link-prediction table that was split per link loses the rule that
  excludes a user's training links from validation and test ranking, so its
  validation metrics can differ from the original.
- Snowflake stores list columns as JSON strings, so export link-prediction
  and forecasting tables to files.

A custom prediction table supplies `ENTITY1` and, for temporal queries,
`TIMESTAMP`, with no target column:

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

Both constructors stage and validate rows immediately, retaining local paths and
query/generation metadata. Select graph data at `fit` or `predict` with `snapshot=`.

Keep the scratch directory while those paths are needed. Use
`PredictionOutputConfig` with `Trainer.predict(..., output_config=...)` or
`result.export(...)` to export predictions. Use `ModelOutputConfig` with
`Trainer.fit(..., export_config=...)` to publish the fitted model bundle.

Custom forecasting rows must include precomputed `future_labels` for training,
`autoregression_labels` for lag features, and `year_over_year_labels` for
year-over-year features when enabled. Constructors preserve these sequences;
they do not derive them from entity, target, and timestamp columns.

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

`require_staged=True` fails if staging is unavailable. With only `staging_uri`,
reads try staging first and can fall back to streaming.

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

`require_staged=True` disables streaming fallback. With only
`staging_volume_path`, reads try staging first and retain streaming fallback.
The identity needs access to both the volume and source table.

## Custom SQL tables

Snowflake, Databricks, and BigQuery connectors accept read-only `SELECT` queries.
Results use the same ingestion and staging options as named tables:

```python
from kumo.engine import Edge, Graph, Table

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

Alternatively, pass `events.sql(query)` to `Table.from_source_table`. Queries must
be one statement with named output columns, restricted to the connector's
database/catalog/project and schema/dataset. Inference samples query results;
BigQuery skips sampling queries estimated to scan more than 10 GiB. To skip
inference, pass `Table(events, name, definition=kumoapi.table.TableDefinition(
cols=..., source_table=events.sql(query).to_source_table(), ...))`.

## Model-plan suggestions

Given `pquery = PredictiveQuery(graph, query)` and `snapshot = graph.snapshot()`,
suggest and customize a plan before training:

```python
from kumo.engine import Trainer
from kumoapi.model_plan import RunMode

plan = pquery.suggest_model_plan(snapshot, run_mode=RunMode.FAST)
print(plan.training_job.metrics)
plan.optimization.max_epochs = 5
trainer = Trainer(plan)
```

Suggestions use graph/query metadata. Inferred settings resolve during training.

## Status

The pipeline supports graph ingestion, query validation, plan suggestions,
training/prediction table generation, GNN training, batch prediction, and model
reload from local files, S3, GCS, or Snowflake.

Limitations:

- `PredictiveQuery.suggest_prediction_table_plan`/`suggest_distilled_model_plan`
  are deferred.
- BigQuery cannot publish or reload model bundles.
- Scoped per-table ingest credentials, and a table staged across multiple
  directories, both raise `NotImplementedError` in `Graph.snapshot()`.

### Reloading a local model

Every successful fit writes a reloadable bundle to `result.bundle_dir`:

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

With `export_config`, `result.published_model` records the assigned name, version,
and URI; otherwise it is `None`. Snowflake also records training metrics and
metadata for model discovery and comparison. Retain the assigned version for reload:

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

S3, GCS, and file connectors expose the published path or URI in the same result.

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

S3, GCS, and file connectors support `output_paths` using configured or ambient
credentials. Rooted connectors accept relative paths. Do not combine
`output_paths` with `artifact_type` or `table_name`. ADLS is unsupported.

`result.files`, `result.frame`, and `result.num_rows` select predictions when both
artifacts are requested, or embeddings for embeddings-only requests.
`result.export(outputs)` copies existing artifacts without rerunning inference.
Choose `extra_fields` during `predict`; `export()` rejects them. Directory retries
can reuse their own output; different exports need an empty destination. Explicit
`.parquet`/`.csv` paths retain single-file/multipart behavior.

Destinations export independently; failures may leave completed outputs and partial
directories. Retry `result.export()` with the same destinations. Directory retries
delete `_SUCCESS` before copying and restore it on completion, so delete permission
is required. Readers should wait for `_SUCCESS`. Exports are not transactional
across destinations, atomic for single files, or isolated from concurrent access.

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
- `OVERWRITE` replaces rows and schema. `APPEND` requires matching destination
  columns and types; integers also fit `FLOAT`/`NUMERIC`, and all-null columns fit
  any type. Schema mismatches fail before writing.
- Rows commit together. Retrying `APPEND` duplicates rows; appending to a missing
  table creates it.
- With a usable `staging_uri`, files are uploaded under
  `<staging_uri>/_kumo.export/<id>/`, loaded in one job, then deleted. Otherwise,
  the Storage Write API commits pending streams together. Streamed `OVERWRITE`
  uses an expiring table and a `WRITE_TRUNCATE` query. `require_staged=True`
  disables streaming fallback; staging requires a colocated bucket.
- Prefer staging for large exports. Streaming and the overwrite query can incur
  charges. Streamed rows are queryable immediately, but table metadata can lag
  and immediate copy jobs may miss rows.
- `OVERWRITE` preserves partitioning, clustering, and column descriptions and
  policy tags for unchanged names/types. Streamed overwrites restore column
  metadata after replacing rows.
- Embeddings land as `REPEATED FLOAT` columns, timestamps as `TIMESTAMP`.
- Before scoring, `predict` checks dataset location, write access, and required
  staging. Grant `roles/bigquery.dataEditor` on the dataset,
  `roles/bigquery.jobUser` on the billing project, and
  `roles/storage.objectAdmin` on the staging bucket when used.
