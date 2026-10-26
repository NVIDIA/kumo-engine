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
source, the confirmed destination, and any population table — each written verbatim.

Values only the runtime knows — a Snowflake warehouse or `connections.toml` entry, a Databricks
SQL warehouse or cluster — are constants here too. S3 and GCS have none: the ambient chain
resolves credentials and region where the code runs (an instance or job role, the environment,
`aws configure`), so add no region, profile or project constant for them, and pass `region=` only
when the user named one. Write the user's value if they gave one;
otherwise a placeholder of the form `'<DATABRICKS_WAREHOUSE_ID>'`, and list each placeholder in
your reply as something to fill in before running. A placeholder never stands in for a confirmed
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
# BigQuery — the identity is required; with no arguments it is the ambient Google credential
connector = kumo.BigQueryConnector(project_id=PROJECT_ID, dataset_id=DATASET_ID,
                                   identity=kumo.BigQueryIdentity())
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

Name file and object-store tables without their suffix — `connector['accounts']` for
`accounts.parquet` — and use those names in the tables, edges and PQL (`adapter.md` §3).

**Credentials.** The ambient forms above name no secret. Use an explicit identity (§3.4) only
when the user asks for one, and then read each secret at run time from an environment variable
whose **name** is a config-cell constant — never a literal, never a value asked for in chat:

```python
identity = kumo.S3Identity(access_key_id=os.environ[AWS_KEY_ID_VAR],
                           secret_access_key=os.environ[AWS_SECRET_VAR])
```

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
written = next(iter(bigquery.Client(project=PROJECT_ID).query(
    f'SELECT COUNT(*) FROM `{PROJECT_ID}.{OUTPUT_DATASET}.{OUTPUT_TABLE}`').result()))[0]

# Databricks — with the confirmed SQL warehouse; wait_timeout is at most '50s'
from databricks.sdk import WorkspaceClient
response = WorkspaceClient().statement_execution.execute_statement(
    warehouse_id=DATABRICKS_WAREHOUSE_ID, wait_timeout='30s',
    statement=f'SELECT COUNT(*) FROM {CATALOG}.{OUTPUT_SCHEMA}.{OUTPUT_TABLE}')
written = int(response.result.data_array[0][0])

# Snowflake — through the same connections.toml entry
import snowflake.connector
with snowflake.connector.connect(connection_name=SNOWFLAKE_CONNECTION) as conn:
    written = conn.cursor().execute(
        f'SELECT COUNT(*) FROM {OUTPUT_DATABASE}.{OUTPUT_SCHEMA}.{OUTPUT_TABLE}').fetchone()[0]

assert written == prediction_result.num_rows, (written, prediction_result.num_rows)
```

- `OUTPUT_LOCATION` for files, S3 and GCS is the confirmed destination itself.
- A Databricks destination is passed to Kumo as `'<schema>__kumo__<table>'` and read back as
  `<catalog>.<schema>.<table>`. With a cluster and no SQL warehouse, there is no read-back here:
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
  import pyarrow.parquet as papq

  keep = {ENTITY_KEY: 'ENTITY1'}
  if ENTITY_TIME:                                             # the entity Table's time_column, else None
      keep[ENTITY_TIME] = 'TIMESTAMP'
  population = ds.dataset(ENTITY_SOURCE, format=ENTITY_FORMAT).to_table(columns=list(keep))
  population = population.rename_columns([keep[c] for c in population.column_names])
  if 'TIMESTAMP' in population.column_names:                  # a DATE anchor reads as null
      i = population.column_names.index('TIMESTAMP')
      population = population.set_column(i, 'TIMESTAMP',
                                         pc.cast(population['TIMESTAMP'], pa.timestamp('us')))
  if '://' not in POPULATION_PATH:
      os.makedirs(POPULATION_PATH, exist_ok=True)              # object stores need no directory
  papq.write_table(population, POPULATION_PATH + 'population.parquet')
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
  source's file system or bucket.
- **Snowflake source:** the §7.1 SQL, run through
  `snowflake.connector.connect(connection_name=SNOWFLAKE_CONNECTION)` as in the read-back above,
  and the `PredictionTable` built from a `SnowflakeConnector` made as in §2 but scoped to the
  population's database and schema.
- **Databricks source:** create the table through the source's SQL warehouse with the statement
  API, wait for it, and build the `PredictionTable` from a `DatabricksConnector` scoped to the
  population's schema. A source reached through a cluster has no SQL warehouse: say at Node 1d
  that staging is not covered there, and offer the rows whose target is NULL instead. The same `TIMESTAMP` rule applies: project `CAST(<time> AS TIMESTAMP) AS
  TIMESTAMP` only when the entity table has a time column, and drop it otherwise. `wait_timeout`
  is at most `'50s'` (Databricks rejects more), so poll until the statement finishes:

  ```python
  import time
  from databricks.sdk import WorkspaceClient
  from databricks.sdk.service.sql import StatementState

  ws = WorkspaceClient()
  time_col = f', CAST({ENTITY_TIME} AS TIMESTAMP) AS TIMESTAMP' if ENTITY_TIME else ''
  r = ws.statement_execution.execute_statement(
      warehouse_id=DATABRICKS_WAREHOUSE_ID, wait_timeout='50s',
      statement=(f'CREATE OR REPLACE TABLE {CATALOG}.{POP_SCHEMA}.{POP_TABLE} AS '
                 f'SELECT {ENTITY_KEY} AS ENTITY1{time_col} '
                 f'FROM {CATALOG}.{SCHEMA}.{ENTITY_TABLE}'))
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

  ```python
  from google.cloud import bigquery

  time_col = f', CAST({ENTITY_TIME} AS TIMESTAMP) AS `TIMESTAMP`' if ENTITY_TIME else ''
  bigquery.Client(project=PROJECT_ID).query(
      f'CREATE OR REPLACE TABLE `{PROJECT_ID}.{POP_DATASET}.{POP_TABLE}` AS '
      f'SELECT {ENTITY_KEY} AS ENTITY1{time_col} '
      f'FROM `{PROJECT_ID}.{DATASET_ID}.{ENTITY_TABLE}`').result()   # waits; raises on failure
  population_connector = kumo.BigQueryConnector(project_id=PROJECT_ID, dataset_id=POP_DATASET,
                                                 identity=kumo.BigQueryIdentity())
  prediction_table = kumo.PredictionTable.from_source_table(
      population_connector[POP_TABLE], pq=pquery, snapshot=snapshot)
  ```

---

## 4. Step 3 — running

Run only on explicit request (router Step 3), in this session's own Python, and only if
`import kumo.engine` works here and the connect step resolves credentials. If either fails, report
it and stop: making this session able to run Kumo is not your job. `fit` blocks for as long as
training takes — tell the user before starting.
