# Host Adapter — Databricks Genie Code

For **Genie Code** (agent mode) inside a Databricks workspace. You run as the signed-in user. With
the user's approval you can search Unity Catalog, run SQL and Python, create and edit notebooks,
and read cell output. The generated notebook runs in the same workspace, on a GPU cluster, reading
and writing Unity Catalog through `DatabricksConnector`.

Kumo is already present on the GPU cluster that runs the notebook; never raise environment setup
with the user. Generating the notebook — connect step, destinations, read-back and running it —
is in `runtime.md` in this directory; read it at Step 2.

Running the artifact as a Databricks job — when to, the job notebook, submission, run output,
schedules and cost — is in `job.md`; read it only when the user asks for a job.

---

## 1. Source scope

`source_scope` is a Unity Catalog **catalog and schema**: `catalog.schema`, or a schema with the
catalog named elsewhere in the conversation. Tables are `catalog.schema.table`.

- A bare schema name with no catalog in any turn: look for it with the §2 scope query across the
  catalogs you can see. One match settles it; several, ask which catalog — that one question.
- A name written with a hyphen in chat (`my-catalog`) is the same object as its underscore form
  (`my_catalog`) — resolve, do not ask. Unity Catalog names are lower case.
- A Unity Catalog **metric view** (`catalog.schema.view`) is this host's semantic view. It
  resolves through `kumo-data-science/accept/semantic_view.md`, read as §1.1 says.
- A Snowflake database or semantic view, an `s3://` or `gs://` path, or a BigQuery dataset is not
  a source on this host. Say so, and say that the data must be in a Unity Catalog schema for this
  host to read it.
- A connector name (`benchmark_sf`) is not a location here: this host keeps no registry of named
  connectors. Ask which catalog and schema it points at.
- The SQL warehouse is not part of `source_scope` and is never asked for at intake. §2 resolves
  it.

### 1.1 Metric views

`kumo-data-science/accept/semantic_view.md` is written for Snowflake's `DESCRIBE SEMANTIC VIEW`.
Every rule in it holds here; only the reading differs. Read the definition, never query the view:

```sql
SELECT view_definition FROM <catalog>.information_schema.views
WHERE table_schema = lower('<schema>') AND table_name = lower('<view>');
```

A metric view's definition is YAML. A view whose definition is plain SQL is an ordinary view, not
a metric view: treat it as a query-defined table (the semantic-view rule for a logical table with
no base table). Map the YAML onto that file's `TABLE` and `RELATIONSHIP` records:

| Metric view YAML | `semantic_view.md` reads it as |
|---|---|
| `source: catalog.schema.table` | a `TABLE` with that base table |
| `source:` a SQL query | a logical table defined by a query → ASK BACK, as that file says |
| each entry of `joins:` (and of a join's own nested `joins:`) with `source: catalog.schema.table` | a `TABLE` with that base table; its `name` is a logical name |
| that join's `on: <parent>.<fk> = <name>.<key>`, or `using: [<col>]` | a `RELATIONSHIP` from the parent (`source` for a top-level join, else the enclosing join) to this table: `FOREIGN_KEY` `<fk>`, `REF_KEY` `<key>` |
| an `on:` that is anything but one column-to-column equality | a non-equality relationship → ASK BACK, as that file says |
| `dimensions:`, `measures:`, `filter:` | the view's dimensions, facts and metrics: ignored, as that file says |

A metric view declares no keys of its own. A base table's `PRIMARY_KEY` is its Unity Catalog
primary-key constraint, read with the §3 declared-keys query, and tagged `[declared]`. Every rule
of that file about primary keys then applies as written — a composite key, a relationship whose
`REF_KEY` is not the table's primary key, two relationships referencing two columns of one
table. A base table with no primary-key constraint has no declared key: its key comes from
preflight §3, as for a view table with no `PRIMARY_KEY`. The one-schema rule, physical names and
`[declared]` relationships apply unchanged, with `catalog.schema` in place of `DB.SCHEMA`.

> **Not verified:** no metric view has been read by this skill in a workspace.

---

## 2. Access

Implements `kumo-data-science/accept/preflight.md` §0. The read tool is Genie Code's own SQL,
run as the signed-in user. The check is the scope query, against `source_scope` itself:

```sql
SELECT schema_name FROM <catalog>.information_schema.schemata WHERE schema_name = lower('<schema>');
```

- A row back: you can read the scope. Name the tool in the proposal's `Source scope` line
  (`Genie Code SQL, as <user>`).
- No row, or `CATALOG_NOT_FOUND` / `SCHEMA_NOT_FOUND`: the scope may not exist, or this user may
  not see it — say both.
- `PERMISSION_DENIED` or `INSUFFICIENT_PERMISSIONS`: the scope exists and this user cannot read it.

When nothing reads the scope, send the preflight §0 ask-back. State what you ran and what came
back, quoting the error, then offer:

1. **Access**: `USE CATALOG`, `USE SCHEMA` and `SELECT` on the tables for this user, granted by
   the user or a workspace administrator; tell you when it is done. Never run a `GRANT` yourself.
2. **Details**: paste the structure. No rows are needed. Useful forms: `DESCRIBE TABLE` output,
   `SHOW CREATE TABLE`, the Catalog Explorer columns — with any declared keys.

**SQL warehouse.** In the same pass, resolve the warehouse the notebook will read through: list
them read-only (`WorkspaceClient().warehouses.list()` from Python), prefer one that is running,
and show it in the proposal as `Warehouse: <name> (<id>) [inferred]`. The user confirms it with
the graph. A warehouse the user names is `[supplied]`. If none is visible, say so; the notebook
then carries the placeholder in `runtime.md` §1.

**What a check may do.** Read catalog metadata and run the preflight §3 probe — nothing else.
Never create, alter, drop or write anything during intake, and never change compute, permissions
or workspace settings.

---

## 3. Preflight

Answer `kumo-data-science/accept/preflight.md` §1-§6 with SQL on the warehouse (§2). Write table
and column names exactly as `information_schema` returns them; the notebook must use the same
spelling (`kumoai-reference/references/kumo_finetuning_sdk.md` §3.2). Read no rows beyond
`information_schema` except the §3 probe, and never call the Kumo SDK during preflight. Probe
results are evidence for the proposal only: never written, never saved, never fed to the notebook.

```sql
-- tables in scope (enumerate only here)
SELECT table_name FROM <catalog>.information_schema.tables WHERE table_schema = lower('<schema>');
-- columns and types
SELECT table_name, column_name, data_type FROM <catalog>.information_schema.columns
WHERE table_schema = lower('<schema>') ORDER BY table_name, ordinal_position;
-- declared keys, if any (informational constraints)
SELECT tc.table_name, tc.constraint_type, kcu.column_name
FROM <catalog>.information_schema.table_constraints tc
JOIN <catalog>.information_schema.key_column_usage kcu
  ON tc.constraint_catalog = kcu.constraint_catalog AND tc.constraint_schema = kcu.constraint_schema
 AND tc.constraint_name = kcu.constraint_name
WHERE tc.table_schema = lower('<schema>');
```

- Row counts are not in `information_schema` and are not needed for the graph proposal.
- Unity Catalog types: `STRING`, `BIGINT` (listed as `LONG`), `INT`, `DOUBLE`, `DECIMAL(p,s)`,
  `BOOLEAN`, `DATE`, `TIMESTAMP`. Time columns are `DATE` or `TIMESTAMP`; preflight §4 applies
  unchanged.
- The confirmed `source_scope` becomes the connector's `catalog` and `schema_name`, and the
  confirmed warehouse its `warehouse_id` (`runtime.md` §2).

---

## 4. Destinations

Predictions and embeddings go to a Unity Catalog table, `catalog.schema.table`, in any schema the
user can create tables in. It is written through a `DatabricksConnector` on that catalog and schema
(`kumoai-reference/references/kumo_finetuning_sdk.md` §9.2). Not destinations for predictions
here: a path in a volume, S3, GCS or local files, and any table outside Unity Catalog.

A model is published to a Unity Catalog registered model (`runtime.md` §5), and retained run
files go to a volume path (`runtime.md` §6). Neither is a prediction destination.

---

## 5. Runtime column constraints

None known for this host. Do not exclude text columns pre-emptively. If a run fails while loading
a text encoder, report the error and the column; that finding belongs at Node 1c.

---

## 6. Capability skills

The capability skills and `kumo-data-science/accept/batch_inference.md` were written for
Snowflake. Their Kumo calls hold for `DatabricksConnector`; read their Snowflake steps with these
substitutions, which `runtime.md` §4.1, §5 and §6 spell out:

| They say (Snowflake) | On this host |
|---|---|
| the Snowflake model registry in `DB.SCHEMA`, a registry connector | a Unity Catalog registered model `catalog.schema.model`, through a `DatabricksConnector` on that catalog and schema (`runtime.md` §5) |
| a model version you choose (`V1`) when publishing | **Unity Catalog assigns the version**, the next integer; it cannot be named or replaced (`runtime.md` §5.1) |
| `SHOW VERSIONS`; `version=None` loads the registry default | the version listing in `runtime.md` §5.2; `version=None` loads the newest ready Kumo version — still pass an explicit one |
| an internal stage for retained artifacts | a Unity Catalog volume path (`runtime.md` §6) |
| the Snowpark `session`, Snowflake SQL | `run_sql` (`runtime.md` §3) |
| a Snowflake connector scoped to a population table | a `DatabricksConnector` on the population's catalog and schema (`runtime.md` §4.1) |
| Snowflake `ARRAY` / `VARIANT` columns arrive as strings | not verified that a Unity Catalog `ARRAY` arrives as a typed list: treat list-typed labels, link weights and forecast sequences as the same supplied-shape limitation |
| a Snowflake ML Job | a Databricks job run on a GPU cluster (`job.md`) |
| "the existing Snowflake notebook" | the existing Databricks notebook; run files are on the cluster's local disk |

A step that has no row here and names a Snowflake object is not covered on this host: say so; do
not adapt it.

---

## 7. Other deployment modes — out of scope

Hosted Kumo — `kumo.init(...)`, an API key, `KUMO_API_KEY` / `KUMO_API_ENDPOINT`, Kumo Studio — is
not this SDK, and neither is Kumo through Spark (a Spark DataFrame handed to Kumo; no such
connector exists). If the user is on one of them, say so and stop. Never blend their setup into
generated code. A Model Serving endpoint is not something this skill creates: a request to deploy
a Kumo model as a live endpoint is out of scope.
