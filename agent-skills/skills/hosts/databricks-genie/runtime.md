# Host Runtime — Databricks Genie Code

Read at Step 2, when generating the notebook: where it runs, the config cell, the connect step,
destinations, read-back and running it. Intake is in `adapter.md`.

---

## 0. Where the code runs

A Databricks notebook in the user's workspace, attached to a **classic GPU cluster** on which
`import kumo.engine` works. Kumo trains in-process on the cluster's GPU and reads and writes Unity
Catalog through the confirmed SQL warehouse; it does not use Spark. Genie Code's own serverless
execution has neither a GPU nor Kumo: it serves intake, never the notebook.

Create **one new notebook**, named after the task (`kumo_churn_30d`), next to the notebook the user
is in, else in their home folder. Never overwrite or append to an existing notebook. Each step is
its own code cell. Show the user where the notebook is. Creating it is not running it: Step 3
still waits for an explicit request.

Kumo's scratch directory default works on the cluster. Emit `kumo.configure(scratch_dir=...)` only
for a scratch path the user named; it must be a local path, never `/Volumes/...` or `/dbfs/...`.
Scratch is lost when the cluster terminates: anything that must outlive the run is written to the
confirmed destination.

---

## 1. Config cell

The first code cell holds the imports and every location as a named constant, each written
verbatim from intake: the source catalog and schema, the SQL warehouse id, the destination's
catalog, schema and table, and any population table.

The warehouse is the one confirmed with the graph (`adapter.md` §2). If none was confirmed, write
the placeholder `'<DATABRICKS_WAREHOUSE_ID>'` and list it in your reply as something to fill in
before running. A placeholder never stands in for a confirmed value, and never for a secret.

End the cell with a guard, so a run with an unfilled value stops before reading anything:

```python
CONFIG = dict(CATALOG=CATALOG, SCHEMA=SCHEMA, WAREHOUSE_ID=WAREHOUSE_ID)   # every constant here
unfilled = [k for k, v in CONFIG.items() if isinstance(v, str) and v.startswith('<')]
assert not unfilled, f'Fill in before running: {unfilled}'
```

---

## 2. Connect step

The notebook runs as the user who runs it. `DatabricksIdentity()` with no arguments takes the
notebook's own credentials from its execution context
(`kumoai-reference/references/kumo_finetuning_sdk.md` §3, §3.4) — no secret, no token, no secret
scope:

```python
identity = kumo.DatabricksIdentity()
connector = kumo.DatabricksConnector(
    catalog=CATALOG, schema_name=SCHEMA, identity=identity, warehouse_id=WAREHOUSE_ID)
assert connector.table_names(), 'Connector reached no tables.'
```

- `catalog` and `schema_name` are plain strings: no quotes or backticks inside them.
- Pass `warehouse_id` and not `cluster_id`. Omit `staging_volume_path` unless the user asks for it.
- Use table names exactly as preflight returned them: `connector['<table>']`.
- One connector reads one schema. Build one more, with the same `identity`, for each other schema
  the notebook writes to — the destination, a population.

**A service principal** only when the user asks for one: `DatabricksIdentity(host=...,
client_id=..., client_secret=...)`, with the id and secret read at run time by
`dbutils.secrets.get(scope=SECRET_SCOPE, key=...)` and the scope and key names as config-cell
constants the user confirmed. Never a literal, never a value asked for in chat, and never call
`dbutils.secrets.get` yourself during intake.

---

## 3. SQL through the warehouse

Population staging and read-back run SQL on the confirmed warehouse with the statement API, which
the cluster's Databricks SDK provides. Define the helper once, after the config cell.
`wait_timeout` is at most `'50s'` — Databricks rejects more — so it polls until the statement
finishes. Every identifier in a statement goes through `q`; never interpolate a bare name:

```python
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

ws = WorkspaceClient()

def q(*parts):
    """`catalog`.`schema`.`table`: Unity Catalog names may contain hyphens."""
    return '.'.join('`' + p.replace('`', '``') + '`' for p in parts)

def run_sql(statement):
    r = ws.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID, statement=statement, wait_timeout='50s')
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(5)
        r = ws.statement_execution.get_statement(r.statement_id)
    assert r.status.state == StatementState.SUCCEEDED, r.status
    return r.result.data_array if r.result else None
```

---

## 4. Destinations and read-back

A confirmed destination `catalog.schema.table` is written through a connector on its own catalog
and schema, with `table_name='<schema>__kumo__<table>'`
(`kumoai-reference/references/kumo_finetuning_sdk.md` §9.2). The schema part must equal the output
connector's `schema_name`; a bare table name fails in the sink. The table lands as
`catalog.schema.table`, exactly that name.

```python
output_connector = kumo.DatabricksConnector(
    catalog=OUTPUT_CATALOG, schema_name=OUTPUT_SCHEMA, identity=identity,
    warehouse_id=WAREHOUSE_ID)
output_config = kumo.PredictionOutputConfig(
    connector=output_connector,
    artifact_type=PredictionArtifactType.PREDICTIONS,
    table_name=f'{OUTPUT_SCHEMA}__kumo__{OUTPUT_TABLE}',
    write_mode=WriteMode.OVERWRITE)
```

The connector stages the rows through the volume `<OUTPUT_CATALOG>.<OUTPUT_SCHEMA>._kumo_staging`
and creates it on first use (`adapter.md` §4 has the grants). When it does not exist yet, the
Node 1d output line says so — `Output: <catalog.schema.table>; creates the staging volume
<catalog.schema>._kumo_staging` — so the user confirms that write too.

`PredictionArtifactType` and `WriteMode` come from `kumoapi.jobs`. Then read the destination back
and compare its row count to `prediction_result.num_rows` — the only proof the write worked:

```python
written = int(run_sql(f'SELECT COUNT(*) FROM {q(OUTPUT_CATALOG, OUTPUT_SCHEMA, OUTPUT_TABLE)}')[0][0])
assert written == prediction_result.num_rows, (written, prediction_result.num_rows)
```

Report the destination as `catalog.schema.table`, unchanged.

### 4.1 Populations

When Node 1d confirmed a population table (`kumoai-reference/references/kumo_finetuning_sdk.md`
§7.1), create it with `run_sql` and build the `PredictionTable` from a connector on the
population's schema. Decide from the entity's `Table` as you built it: if it has a `time_column`,
the population **must** carry it as `TIMESTAMP`, **cast to a timestamp**. Kumo rejects a population
without it, even for a static query, whatever the §7.1 Snowflake example shows, and reads a `DATE`
anchor as null. Without a time column, select `ENTITY1` alone.

```python
run_sql(f'CREATE OR REPLACE TABLE {q(POP_CATALOG, POP_SCHEMA, POP_TABLE)} AS '
        f'SELECT {q(ENTITY_KEY)} AS ENTITY1, CAST({q(ENTITY_TIME)} AS TIMESTAMP) AS TIMESTAMP '
        f'FROM {q(CATALOG, SCHEMA, ENTITY_TABLE)}')
population_connector = kumo.DatabricksConnector(
    catalog=POP_CATALOG, schema_name=POP_SCHEMA, identity=identity, warehouse_id=WAREHOUSE_ID)
prediction_table = kumo.PredictionTable.from_source_table(
    population_connector[POP_TABLE], pq=pquery, snapshot=snapshot)
```

`ENTITY_TIME` is the entity `Table`'s `time_column`; `POP_CATALOG` and `POP_SCHEMA` are the
population's confirmed location, which need not be the source's.

**A population the user supplies** — a table of exact scoring rows, or curated labels, weights and
splits (`kumo-custom-populations/SKILL.md`) — is read through a `DatabricksConnector` on that
table's own catalog and schema, built with the same `identity` and warehouse. Kumo stages it to
local Parquet exactly as it does a source table. `PredictionTable.from_source_table` has run this
way on this host; `TrainingTable.from_source_table` uses the same staging but has not been run
end to end here.

---

## 5. Model registry — Unity Catalog

A published model is a Unity Catalog **registered model**, `catalog.schema.model`. Kumo reaches it
through a `DatabricksConnector` on that catalog and schema; `ModelRegistryLocation.model_name` is
the bare model name.

```python
registry_connector = kumo.DatabricksConnector(
    catalog=REGISTRY_CATALOG, schema_name=REGISTRY_SCHEMA, identity=identity,
    warehouse_id=WAREHOUSE_ID)
```

The user needs `USE CATALOG`, `USE SCHEMA` and `CREATE MODEL` on that schema to publish, and
`EXECUTE` on the model to load it. Publishing creates the registered model when it does not exist.

### 5.1 Publish during the requested fit

**Unity Catalog assigns the version** — the next integer, `1` for a new model — and Kumo refuses a
`ModelRegistryLocation` that names one when publishing. So a version the user asks to publish
*as* ("as V1") cannot be honoured, and an existing version is never replaced: publishing always
adds a new one. Say so at the gate and restate it — `Publish: <catalog.schema.model>, version
assigned by Unity Catalog [default]` — rather than asking for a version.

```python
location = kumo.ModelRegistryLocation(model_name=MODEL_NAME)      # no version= here
training_result = trainer.fit(
    training_table, snapshot=snapshot,
    export_config=kumo.ModelOutputConfig(connector=registry_connector, location=location))
published = training_result.published_model
if published is None:
    raise RuntimeError('Fit returned no publication record')
print(published.name, published.version, published.uri)   # the version Unity Catalog assigned
```

Report the assigned version; it is the one every later load names.

### 5.2 Pick the version and load it

Versions are positive integers. A user's `v2` or `V2` is version `2`; pass it as the string `'2'`.
`version=None` loads the newest *ready Kumo* version, not a registry default — still pass the
version explicitly, chosen as `kumo-model-persistence/references/reuse.md` says. To list the
loadable versions (the counterpart of `SHOW VERSIONS`), ask the registry connector, which keeps
only ready versions Kumo published — Unity Catalog's own version listing carries no such marker,
and a version without it does not load (Source: shipped package (`kumo_databricks.registry`)):

```python
for m in registry_connector.sink_backend.list_models(registry_connector):
    if m.name == MODEL_NAME:
        print(m.name, m.version, m.uri)
```

"Latest" is the highest version listed; say which one that is and pass it.

```python
loaded = kumo.Trainer.load_from_connector(
    registry_connector,
    location=kumo.ModelRegistryLocation(model_name=MODEL_NAME, version=MODEL_VERSION))
```

Then rebuild the graph and score exactly as `reuse.md` says, writing to the §4 destination.
Never delete a model version, and never publish to stand in for one that was deleted.

> **Not verified:** publishing to and loading from Unity Catalog have not been run with this skill.

---

## 6. Retained artifacts — a volume

What `kumo-model-persistence/references/retention.md` puts on a Snowflake stage goes to a Unity
Catalog volume the user names: `/Volumes/<catalog>/<schema>/<volume>/<run prefix>/`. The cluster
reads and writes volumes as ordinary paths. The volume must exist; if the user names none, ask.
Creating one is a write that needs their confirmation.

Build the archive with the packaged helper exactly as `retention.md` says, then copy it — never
over an existing object — and check the copy's digest:

```python
import hashlib
import shutil
from pathlib import Path

target = Path(VOLUME_PREFIX) / Path(ARCHIVE_PATH).name
if target.exists():
    raise FileExistsError(f'{target} exists; choose a new run prefix')
target.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(ARCHIVE_PATH, target)
copied = hashlib.sha256(target.read_bytes()).hexdigest()
if copied != digest:
    raise RuntimeError(f'Retained archive digest mismatch: {copied} != {digest}')
print('retained:', target, digest)
```

To retrieve, copy the exact volume object into a fresh empty directory and pass that file to
`restore_archive` with the recorded digest, as `retention.md` says.

> **Not verified:** retention to a volume has not been run with this skill.

---

## 7. Step 3 — running

Run only on explicit request (router Step 3).

- Check the notebook's attached compute first. If it is serverless, or a cluster without a GPU,
  say so and stop; do not change the compute yourself, and do not create a cluster.
- `fit` blocks for as long as training takes — tell the user before starting.
- Run the cells in order. If a cell fails, report the cell and the error, quoted. Fix only the
  code that failed; never change the confirmed task, graph, population or destination to make it
  pass.
- On success, report the destination as `catalog.schema.table` and the read-back row count from
  the notebook's own check.

> **Not verified:** how long a Genie Code session stays attached through a long training run.
