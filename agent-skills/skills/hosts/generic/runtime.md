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
SQL warehouse or cluster — are constants here too. Write the user's value if they gave one;
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
connector = kumo.S3Connector(root_dir=SOURCE_URI)            # 's3://bucket/prefix/'
connector = kumo.GCSConnector(root_dir=SOURCE_URI)           # 'gs://bucket/prefix/'
# BigQuery
connector = kumo.BigQueryConnector(project_id=PROJECT_ID, dataset_id=DATASET_ID)
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

```python
# Local files, S3, GCS — directory or single-file output
import pyarrow.dataset as ds
written = ds.dataset(OUTPUT_LOCATION, format='parquet').count_rows()  # full path or URI

# BigQuery
from google.cloud import bigquery
written = next(iter(bigquery.Client(project=PROJECT_ID).query(
    f'SELECT COUNT(*) FROM `{PROJECT_ID}.{OUTPUT_DATASET}.{OUTPUT_TABLE}`').result()))[0]

# Databricks — with the confirmed SQL warehouse
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

- `OUTPUT_LOCATION` for files is the destination as written: a rooted connector's relative path
  joined to its root.
- A Databricks destination is passed to Kumo as `'<schema>__kumo__<table>'` and read back as
  `<catalog>.<schema>.<table>`. With a cluster and no SQL warehouse, there is no read-back here:
  say so in the artifact and in your reply.
- Report the destination in its own form, unchanged — `DB.SCHEMA.TABLE`, or the path.

> **Not verified:** none of these read-backs, and no connector other than Snowflake, has been run
> end to end with this skill. Say so when you hand over an artifact for one of them.

### 3.1 Static scoring population

When Node 1d confirmed a population table (`kumoai-reference/references/kumo_finetuning_sdk.md`
§7.1), stage it in the destination's system:

- **Files, S3, GCS:** read the entity key column from the source with `pyarrow.dataset`, rename it
  `ENTITY1`, and write one Parquet file at the confirmed population path. Build the
  `PredictionTable` from a connector rooted at that path's directory.
- **Snowflake:** the §7.1 SQL, run through the read-back connection above.
- **BigQuery, Databricks:** not covered on this host. Say so at Node 1d and offer the rows whose
  target is NULL instead.

---

## 4. Step 3 — running

Run only on explicit request (router Step 3), in this session's own Python, and only if
`import kumo.engine` works here and the connect step resolves credentials. If either fails, report
it and stop: making this session able to run Kumo is not your job. `fit` blocks for as long as
training takes — tell the user before starting.
