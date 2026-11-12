# Host Runtime — Generic coding agent

Read at Step 2, when generating the artifact: where it runs, the config cell, the connect step,
credentials, destinations and read-back. Intake is in `adapter.md`.

---

## 0. Where the code runs

Unknown to you, and it does not matter: the artifact must run unchanged in any Python environment
where `import kumo.engine` works and the source's credentials resolve — a workstation, a VM, a
hosted notebook. So it assumes no platform:

- no Snowpark session from `get_active_session()`, no `dbutils`, no SPCS token, no notebook
  magics, no platform secret store;
- no path outside the ones confirmed at intake, except Kumo's scratch default
  (`kumoai-reference/references/kumo_finetuning_sdk.md` §2.1).

Your own session's access (`adapter.md` §2) says nothing about the runtime's. The connect step
resolves credentials where the code runs.

Write the artifact to the path the user names, else to the current working directory, named after
the task (`churn_30d_train.ipynb`). Write only that file. Creating it is not running it: Step 3
still waits for an explicit request.

---

## 1. Config cell

The first code cell holds the imports and every location as a named constant: the confirmed
source, the confirmed destination, and any population table — each written verbatim. A
population also gets `POPULATION_CONFIRMED`, its name exactly as Node 1d confirmed it, which the
staging step checks before replacing anything (§3.1).

Values only the runtime knows — a Snowflake warehouse or `connections.toml` entry, a Databricks
SQL warehouse or cluster — are constants here too. S3, GCS and BigQuery have none: the ambient
chain resolves credentials and region where the code runs (an instance or job role, the
environment, `aws configure`, `gcloud`), and BigQuery bills queries to the confirmed dataset's own
project. So add no region, profile, project or billing constant for them beyond the confirmed
location, and pass `region=` or `billing_project_id=` only when the user named one. A BigQuery
dataset's location is not a constant either: the notebook reads it from the dataset (§2). For the
values that are constants, write the user's value if they gave one; otherwise a placeholder of
the form `'<DATABRICKS_WAREHOUSE_ID>'`, and list each placeholder in your reply as something to
fill in before running. A placeholder never stands in for a confirmed
value, and never for a secret.

End the cell with a guard, so a run with an unfilled value stops before reading anything:

```python
CONFIG = dict(SOURCE_DIR=SOURCE_DIR, OUTPUT_DIR=OUTPUT_DIR)   # every constant in this cell
unfilled = [k for k, v in CONFIG.items() if isinstance(v, str) and v.startswith('<')]
assert not unfilled, f'Fill in before running: {unfilled}'
```

---

## 2. Connect step

One connector per system, built from the confirmed `source_scope` with ambient credentials
(`kumoai-reference/references/kumo_finetuning_sdk.md` §3, §3.4). Signatures are there; do not
add arguments the user did not confirm.

```python
# Local files
connector = kumo.FileConnector(root_dir=SOURCE_DIR)
# S3 / GCS — rooted at the confirmed prefix, so table_names() works
connector = kumo.S3Connector(root_dir=SOURCE_URI)            # 's3://bucket/prefix/'; no region=
connector = kumo.GCSConnector(root_dir=SOURCE_URI)           # 'gs://bucket/prefix/'
# BigQuery — PROJECT_ID and DATASET_ID are the two parts of the confirmed `project.dataset` scope,
# never placeholders. The identity is required; with no arguments it is the ambient Google
# credential. No billing_project_id (§1). location= is the dataset's own, read by bq_location below.
connector = kumo.BigQueryConnector(project_id=PROJECT_ID, dataset_id=DATASET_ID,
                                   identity=kumo.BigQueryIdentity(),
                                   location=bq_location(PROJECT_ID, DATASET_ID))
# Databricks — exactly one of warehouse_id / cluster_id
connector = kumo.DatabricksConnector(
    catalog=CATALOG, schema_name=SCHEMA, identity=kumo.DatabricksIdentity(),
    warehouse_id=DATABRICKS_WAREHOUSE_ID)
# Snowflake — outside Snowflake's own runtime, through a connections.toml entry
connector = kumo.SnowflakeConnector.from_connection_toml(
    connection_name=SNOWFLAKE_CONNECTION, warehouse=SNOWFLAKE_WAREHOUSE,
    database=DATABASE, schema_name=SCHEMA)                   # lower case: keep the quotes, §3

assert connector.table_names(), 'Connector reached no tables.'
```

Emit only the line for the confirmed system. Never emit a rootless S3 or GCS connector for a
source: it cannot list tables.

**BigQuery location.** `BigQueryConnector` defaults to `location='US'`, and a dataset anywhere
else — `EU`, or a single region such as `us-central1`, which is not `US` — fails: reads with
not-found, and the prediction write only after training, at the sink's location check. Whenever
any connector in the notebook is BigQuery — source, population or destination — define this
once, in the connect step before the first connector, and pass
`location=bq_location(<its project>, <its dataset>)` to every `BigQueryConnector`:

```python
from google.cloud import bigquery

def bq_location(project, dataset):
    """The location a dataset lives in; the dataset must already exist."""
    return bigquery.Client(project=project).get_dataset(f'{project}.{dataset}').location
```

Each BigQuery location keeps the project it was confirmed with: `PROJECT_ID` for the source,
`OUTPUT_PROJECT` and `POP_PROJECT` for the destination and the population, equal to `PROJECT_ID`
only when the confirmed names say so. Use that project for its connector, location, staging query
and read-back.

Name file and object-store tables without their suffix — `connector['accounts']` for
`accounts.parquet` — and use those names in the tables, edges and PQL (`adapter.md` §3).

**Credentials.** The ambient forms above name no secret. Use an explicit identity (§3.4) only
when the user asks for one, and then read each secret at run time from an environment variable
whose **name** is a config-cell constant — never a literal, never a value asked for in chat:

```python
identity = kumo.S3Identity(access_key_id=os.environ[AWS_KEY_ID_VAR],
                           secret_access_key=os.environ[AWS_SECRET_VAR])
```

Arrow does not see that identity. Every `pyarrow` read or write of S3 in §3 and §3.1 then takes
the same keys, with the path written without `s3://`:

```python
import pyarrow.fs
s3 = pyarrow.fs.S3FileSystem(access_key=os.environ[AWS_KEY_ID_VAR],
                             secret_key=os.environ[AWS_SECRET_VAR])
written = ds.dataset(OUTPUT_LOCATION.removeprefix('s3://'), format=OUTPUT_FORMAT,
                     filesystem=s3).count_rows()
```

Without an explicit identity, emit `s3 = None`: §3.1's population staging then lets Arrow resolve
the ambient credentials itself.

The Snowflake factory `from_spcs_env` and Snowflake's notebook secret store belong to
Snowflake's own runtime. Emit them only when the user says the code runs there.

### 2.1 Storage

Kumo's scratch directory defaults to `~/.cache/kumo-engine`. Emit `kumo.configure(scratch_dir=...)`
only for a scratch path the user named. A `FileConnector` source or destination must not overlap
it (§3.4). Scratch is disposable: anything that must outlive the run is written to the confirmed
destination.

---

## 3. Destinations and read-back

Build the output connector for the destination's own system, and the `PredictionOutputConfig` in
that connector's form (`kumoai-reference/references/kumo_finetuning_sdk.md` §9.2). Then read the
destination back and compare its row count to `prediction_result.num_rows` — the only proof the
write worked.

**A path destination is written at exactly the confirmed path**, never inside it. Root the output
connector at the destination's **parent** and name the destination's last segment in
`output_paths`, keeping its trailing `/` for a directory:

```python
# confirmed destination: s3://bucket/prediction_outputs/churn_scores/
OUTPUT_PARENT, OUTPUT_LEAF = 's3://bucket/prediction_outputs/', 'churn_scores/'
output_connector = kumo.S3Connector(root_dir=OUTPUT_PARENT)    # FileConnector / GCSConnector alike
output_config = kumo.PredictionOutputConfig(
    connector=output_connector, output_paths={'predictions': OUTPUT_LEAF})
OUTPUT_LOCATION = OUTPUT_PARENT + OUTPUT_LEAF                   # the confirmed path, for read-back
```

A BigQuery destination's connector is scoped to `OUTPUT_PROJECT` and `OUTPUT_DATASET` and takes
`location=bq_location(OUTPUT_PROJECT, OUTPUT_DATASET)` (§2), which may differ from the source's.
A Databricks destination's connector is scoped to `OUTPUT_CATALOG` and `OUTPUT_SCHEMA`.

Rooting the connector at the destination itself and passing a `table_name` writes one level
below it, to a path the user never confirmed. A directory destination must be empty or absent;
the sink creates it.

`OUTPUT_LEAF` ends in `/`, `.parquet` or `.csv` (`adapter.md` §4). File, S3 and GCS sinks alike
add `.parquet` to any other leaf, and the read-back would then miss the file. The suffix also
sets the format the sink writes, so read back in that format.

```python
# Local files, S3, GCS — directory or single-file output
import pyarrow.dataset as ds
OUTPUT_FORMAT = 'csv' if OUTPUT_LOCATION.rstrip('/').lower().endswith('.csv') else 'parquet'
written = ds.dataset(OUTPUT_LOCATION, format=OUTPUT_FORMAT).count_rows()  # full path or URI

# BigQuery
from google.cloud import bigquery
written = next(iter(bigquery.Client(project=OUTPUT_PROJECT).query(
    f'SELECT COUNT(*) FROM `{OUTPUT_PROJECT}.{OUTPUT_DATASET}.{OUTPUT_TABLE}`').result()))[0]

# Databricks — with the confirmed SQL warehouse; wait_timeout is at most '50s', so poll
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

def q(*parts):
    """`catalog`.`schema`.`table`: Unity Catalog names may contain hyphens."""
    return '.'.join('`' + p.replace('`', '``') + '`' for p in parts)

ws = WorkspaceClient()
r = ws.statement_execution.execute_statement(
    warehouse_id=DATABRICKS_WAREHOUSE_ID, wait_timeout='50s',
    statement=f'SELECT COUNT(*) FROM {q(OUTPUT_CATALOG, OUTPUT_SCHEMA, OUTPUT_TABLE)}')
while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
    time.sleep(5)
    r = ws.statement_execution.get_statement(r.statement_id)
assert r.status.state == StatementState.SUCCEEDED, r.status
written = int(r.result.data_array[0][0])

# Snowflake — through the same connections.toml entry and warehouse. Kumo keeps the table
# name's case, so quote it (kumo_finetuning_sdk.md §9.1); database and schema as in §2.
import snowflake.connector
with snowflake.connector.connect(connection_name=SNOWFLAKE_CONNECTION,
                                 warehouse=SNOWFLAKE_WAREHOUSE) as conn:
    table = '"' + OUTPUT_TABLE.replace('"', '""') + '"'
    written = conn.cursor().execute(
        f'SELECT COUNT(*) FROM {OUTPUT_DATABASE}.{OUTPUT_SCHEMA}.{table}').fetchone()[0]

assert written - before == prediction_result.num_rows, (written, before, prediction_result.num_rows)
```

`before` is 0 for OVERWRITE. For APPEND (`kumoai-reference/references/outputs.md`) it is the same
count taken before `predict` — 0 when the destination does not exist yet, since the first APPEND
creates it — so the check compares the rows this run added. The delta holds only while nothing
else writes to the destination; when the user says others do, confirm a run-specific destination
instead.

- `OUTPUT_LOCATION` for files, S3 and GCS is the confirmed destination itself.
- A Databricks destination is passed to Kumo as `'<schema>__kumo__<table>'` and read back as
  `<catalog>.<schema>.<table>`, in its own catalog. With a cluster and no SQL warehouse, there is no read-back here:
  say so in the artifact and in your reply.
- Report the destination in its own form, unchanged — `DB.SCHEMA.TABLE`, or the path.

> **Not verified:** none of these read-backs, and no connector other than Snowflake, has been run
> end to end with this skill. Say so when you hand over an artifact for one of them.

### 3.1 Static scoring population

When Node 1d confirmed a population table (`kumoai-reference/references/kumo_finetuning_sdk.md`
§7.1), stage it in the **source's** system — where the entity table is read from — whatever the
destination. Each recipe below reads the entity table and writes the population inside that one
system; the destination may be in another system (`adapter.md` §4) and plays no part here.

- **Files, S3, GCS source:** read the entity key — and, when the entity table has a time column, that
  column too — from the source with `pyarrow.dataset`; name them `ENTITY1` and `TIMESTAMP`; write
  one Parquet file inside the confirmed population path, creating the directory first for local
  files; and build the `PredictionTable` from a connector rooted there. Decide from the entity's
  `Table` as you built it: if it has a `time_column`, the population **must** carry it as
  `TIMESTAMP`, **cast to a timestamp**. Kumo rejects a population without it, even for a static
  query, whatever the §7.1 Snowflake example shows (`ValueError: ... missing TIMESTAMP`), and
  reads a `DATE` anchor as null (`Column 'TIMESTAMP' is date32[day]`): a date column is cast.

  ```python
  import os
  import pyarrow as pa
  import pyarrow.compute as pc
  import pyarrow.dataset as ds
  import pyarrow.fs
  import pyarrow.parquet as papq

  def arrow_fs(uri):
      """Arrow's filesystem and path for uri; S3 with an explicit S3Identity takes §2's `s3`."""
      if uri.startswith('s3://') and s3 is not None:         # s3 = None under ambient credentials
          return s3, uri.removeprefix('s3://')
      return pyarrow.fs.FileSystem.from_uri(uri if '://' in uri else os.path.abspath(uri))

  if POPULATION_PATH != POPULATION_CONFIRMED:                # Node 1d approved replacing this path only
      raise RuntimeError('POPULATION_PATH changed after Node 1d: confirm the new path first.')
  keep = {ENTITY_KEY: 'ENTITY1'}
  if ENTITY_TIME:                                             # the entity Table's time_column, else None
      keep[ENTITY_TIME] = 'TIMESTAMP'
  source_fs, source_path = arrow_fs(ENTITY_SOURCE)
  scanner = ds.dataset(source_path, format=ENTITY_FORMAT,
                       filesystem=source_fs).scanner(columns=list(keep))

  def anchor_type(t):            # a timestamp keeps its unit; a DATE anchor reads as null: cast it
      return t if pa.types.is_timestamp(t) else pa.timestamp('us')

  schema = pa.schema([(keep[f.name], anchor_type(f.type) if keep[f.name] == 'TIMESTAMP' else f.type)
                      for f in scanner.projected_schema])
  pop_fs, pop_dir = arrow_fs(POPULATION_PATH)
  if isinstance(pop_fs, pyarrow.fs.LocalFileSystem):
      pop_fs.create_dir(pop_dir)                              # object stores need no directory
  final = pop_dir.rstrip('/') + '/population.parquet'
  with papq.ParquetWriter(final + '.partial', schema, filesystem=pop_fs) as writer:
      for batch in scanner.to_batches():                     # in batches: bounded memory
          writer.write_batch(pa.RecordBatch.from_arrays(
              [pc.cast(batch[f.name], t.type) for f, t in zip(batch.schema, schema)],
              schema=schema))
  pop_fs.move(final + '.partial', final)                     # replace only once every batch is in
  if POPULATION_PATH.startswith('s3://'):                     # the source's kind, in the §2 forms
      population_connector = kumo.S3Connector(root_dir=POPULATION_PATH)
  elif POPULATION_PATH.startswith('gs://'):
      population_connector = kumo.GCSConnector(root_dir=POPULATION_PATH)
  else:
      population_connector = kumo.FileConnector(root_dir=POPULATION_PATH)
  prediction_table = kumo.PredictionTable.from_source_table(
      population_connector['population'], pq=pquery, snapshot=snapshot)
  ```

  `ENTITY_SOURCE` is the entity table's file or directory as preflight listed it, and
  `ENTITY_FORMAT` its format as preflight found it, `'csv'` or `'parquet'`. The population is
  written as Parquet either way. `POPULATION_PATH` is the confirmed path, ending in `/`, in the
  source's file system or bucket, and `POPULATION_CONFIRMED` the same path as Node 1d confirmed
  it, both in the config cell. Each run replaces its `population.parquet`, as Node 1d says. The
  rows go to `population.parquet.partial` first, so a scan or cast that fails leaves the previous
  population in place. A source timestamp keeps its own unit: a `timestamp[ns]` with
  sub-microsecond values would fail a safe cast to microseconds.
- **Snowflake source:** the §7.1 SQL, run through
  `snowflake.connector.connect(connection_name=SNOWFLAKE_CONNECTION, warehouse=SNOWFLAKE_WAREHOUSE)`
  as in the read-back above,
  and the `PredictionTable` built from a `SnowflakeConnector` made as in §2 but scoped to the
  population's database and schema.
- **Databricks source:** create the table through the source's SQL warehouse with the statement
  API, wait for it, and build the `PredictionTable` from a `DatabricksConnector` scoped to the
  population's catalog and schema, `POP_CATALOG` and `POP_SCHEMA`. A source reached through a cluster has no SQL warehouse: say at Node 1d
  that staging is not covered there, and offer the rows whose target is NULL instead. The same `TIMESTAMP` rule applies: project `CAST(<time> AS TIMESTAMP) AS
  TIMESTAMP` only when the entity table has a time column, and drop it otherwise. `wait_timeout`
  is at most `'50s'` (Databricks rejects more), so poll until the statement finishes:

  ```python
  import time
  from databricks.sdk import WorkspaceClient
  from databricks.sdk.service.sql import StatementState

  def q(*parts):
      """`catalog`.`schema`.`table`: Unity Catalog names may contain hyphens."""
      return '.'.join('`' + p.replace('`', '``') + '`' for p in parts)

  if f'{POP_CATALOG}.{POP_SCHEMA}.{POP_TABLE}' != POPULATION_CONFIRMED:   # approved at Node 1d
      raise RuntimeError('The population table changed after Node 1d: confirm the new one first.')
  ws = WorkspaceClient()
  time_col = f', CAST({q(ENTITY_TIME)} AS TIMESTAMP) AS TIMESTAMP' if ENTITY_TIME else ''
  r = ws.statement_execution.execute_statement(
      warehouse_id=DATABRICKS_WAREHOUSE_ID, wait_timeout='50s',
      statement=(f'CREATE OR REPLACE TABLE {q(POP_CATALOG, POP_SCHEMA, POP_TABLE)} AS '
                 f'SELECT {q(ENTITY_KEY)} AS ENTITY1{time_col} '
                 f'FROM {q(CATALOG, SCHEMA, ENTITY_TABLE)}'))
  while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
      time.sleep(5)
      r = ws.statement_execution.get_statement(r.statement_id)
  assert r.status.state == StatementState.SUCCEEDED, r.status
  ```
- **BigQuery source:** create the table through the BigQuery client with `CREATE OR REPLACE TABLE
  ... AS SELECT`, wait for the job, and build the `PredictionTable` from a `BigQueryConnector`
  scoped to the population's dataset. The same `TIMESTAMP` rule applies: project
  `CAST(<time> AS TIMESTAMP) AS TIMESTAMP` only when the entity table has a time column, and drop
  it otherwise. The population's dataset must already exist; `CREATE TABLE` does not create it.
  It must be in the source dataset's location, because one BigQuery query cannot read and write
  across locations.

  ```python
  from google.cloud import bigquery

  if f'{POP_PROJECT}.{POP_DATASET}.{POP_TABLE}' != POPULATION_CONFIRMED:   # approved at Node 1d
      raise RuntimeError('The population table changed after Node 1d: confirm the new one first.')
  time_col = f', CAST({ENTITY_TIME} AS TIMESTAMP) AS `TIMESTAMP`' if ENTITY_TIME else ''
  bigquery.Client(project=POP_PROJECT).query(
      f'CREATE OR REPLACE TABLE `{POP_PROJECT}.{POP_DATASET}.{POP_TABLE}` AS '
      f'SELECT {ENTITY_KEY} AS ENTITY1{time_col} '
      f'FROM `{PROJECT_ID}.{DATASET_ID}.{ENTITY_TABLE}`').result()   # waits; raises on failure
  population_connector = kumo.BigQueryConnector(project_id=POP_PROJECT, dataset_id=POP_DATASET,
                                                 identity=kumo.BigQueryIdentity(),
                                                 location=bq_location(POP_PROJECT, POP_DATASET))
  prediction_table = kumo.PredictionTable.from_source_table(
      population_connector[POP_TABLE], pq=pquery, snapshot=snapshot)
  ```

---

## 4. Step 3 — running

Run only on explicit request (router Step 3), in this session's own Python, and only if
`import kumo.engine` works here and the connect step resolves credentials. If either fails, report
it and stop: making this session able to run Kumo is not your job. `fit` blocks for as long as
training takes — tell the user before starting.

## Graph visualization

Notebook calls display an interactive HTML view.
For scripts, save an SVG and report its path on the Python runtime, which may be remote.
Retrieve exports through that environment's file tools.

Source: shipped package (`kumo.engine.graph.Graph.visualize`, `kumo.engine._visualization.SVG`)
