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

## 0. Nothing is written before the gate

Genie Code can act on the workspace, and a request that names a deliverable — "make the notebook",
"write a job", "score every account", "fit on this table" — reads like permission to act. It is
not: it starts intake. For a request that has a Node 1d gate — a new model, batch scoring, a job —
until the user has confirmed the restatement, in a later turn:

- do not create, import, upload or edit a notebook or any file, in the workspace or on local disk;
- do not submit a run, create a job, or create a table, volume or model;
- do not put the code in your reply.

An inspection request has no Node 1d — explanations, training diagnostics, graph visualization.
Answer it directly, code in the reply included; it still writes, imports and runs nothing in the
workspace unasked.

Preflight reads — the SQL in §2 and §3, reading clusters, registered models and runs — are not
writes and need no confirmation. A reply that reaches a gate ends with the proposal or the
restatement and its confirmation question.

Confirmation is the user's answer to a restatement you showed in an earlier reply of this
conversation. "Looks good, build it" when no restatement has been shown is not confirmation:
show the gates the operation has — for a new model the graph proposal, then the restatement; for
§6.1 the single combined restatement — and stop at each.

This holds for a follow-up in a session that already has a notebook, and for a request routed to
a capability skill — custom labels, weights, splits or scoring rows; publishing, reloading or
retaining a model; tuning. A change to the population, labels, weights, splits,
task or destination goes back through the gates: answer with the restatement, or with the
ask-back that names the offending field, never with new or edited code. A field the capability
skill says not to transform — a split column holding strings, weights to normalize, a list
target — is reported and asked about, not converted in a file you write.

A job, a run or a cluster you find in the workspace is not a request to change it. Creating,
editing, deploying, scheduling, unpausing or pausing a job, submitting or repairing a run, and
starting a cluster are writes, each confirmed on its own (`job.md` §3, §7, §8). Reading a job's
settings, a run's state and output, or a cluster's state is a read.

After confirmation, Step 2 creates the one notebook (`runtime.md` §0). Running it, or submitting
it as a job, is Step 3 and waits for its own explicit request (`job.md` §3).

**Tools.** Preflight reads Unity Catalog through SQL on a warehouse (§2, §3) — Genie Code's own
SQL, the SQL statement API (`WorkspaceClient().statement_execution`), or
`databricks api post /api/2.0/sql/statements`. A minimum CLI version another skill asks for (for
`genie ask`, say) is not needed for that: never stop intake to ask the user to upgrade or add a
tool; use a SQL path that works.

---

## 1. Source scope

`source_scope` is a Unity Catalog **catalog and schema**: `catalog.schema`, or a schema with the
catalog named elsewhere in the conversation. Tables are `catalog.schema.table`.

- A bare schema name with no catalog in any turn: look for it with the §2 scope query across the
  catalogs you can see. One match settles it; several, ask which catalog — that one question.
- Unity Catalog names are lower case, and may contain hyphens (`dev-ws`). A hyphenated name in
  chat is not rewritten: look it up as written and, if absent, in its underscore form, and use
  whichever `information_schema` returns. Backtick-quote every identifier you put in SQL
  (`` `dev-ws`.`sales`.`orders` ``); the connector quotes its own.
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
DESCRIBE TABLE EXTENDED <catalog>.<schema>.<view> AS JSON;
```

The result is one JSON document. Its `type` is `METRIC_VIEW` for a metric view, and its
`view_text` is the YAML definition. Do not read `information_schema.views` for this: it does not
list metric views, and its `view_definition` is NULL for anyone but a view's owner.

- `type` `VIEW`: an ordinary view, not a metric view. Treat it as a query-defined table (the
  semantic-view rule for a logical table with no base table).
- `type` `METRIC_VIEW` with no `view_text`, or an error: you cannot read the definition. Say so,
  quoting what came back, and ask for the YAML (or the base tables and joins) — the §2 *Details*
  option. Never read a missing definition as an ordinary view.

Every three-part `source` names a table, a view or another metric view. Check which before you
map it, in one query per catalog:

```sql
SELECT table_schema, table_name, table_type FROM <catalog>.information_schema.tables
WHERE (table_schema, table_name) IN ((lower('<schema>'), lower('<table>')), ...);
```

`MANAGED`, `EXTERNAL`, `MANAGED_SHALLOW_CLONE` and `EXTERNAL_SHALLOW_CLONE` are base (Delta)
tables. Any other `table_type` — `VIEW`, `METRIC_VIEW`,
`MATERIALIZED_VIEW`, `STREAMING_TABLE`, `FOREIGN` — is not one: read it as a logical table defined
by a query. Map the YAML onto that file's `TABLE` and `RELATIONSHIP` records:

| Metric view YAML | `semantic_view.md` reads it as |
|---|---|
| `source: catalog.schema.table`, a base table | a `TABLE` with that base table |
| `source:` a SQL query, or a three-part name that is not a base table | a logical table defined by a query → ASK BACK, as that file says |
| each entry of `joins:` (and of a join's own nested `joins:`) whose `source` is a base table | a `TABLE` with that base table; its `name` is a logical name |
| a join whose `source` is a SQL query or not a base table | a logical table defined by a query → ASK BACK, as that file says |
| that join's `on: <parent>.<fk> = <name>.<key>`, or `using: [<col>]`, when its `cardinality` is absent or `many_to_one` | a `RELATIONSHIP` from the parent (`source` for a top-level join, else the enclosing join) to this table: `FOREIGN_KEY` `<fk>`, `REF_KEY` `<key>` |
| an `on:` that is anything but one column-to-column equality | a non-equality relationship → ASK BACK, as that file says |
| a join with any other `cardinality` (`one_to_many`, `one_to_one`, …) | → ASK BACK: name the join and its `cardinality`, say the skill reads only many-to-one joins, and ask which table holds the foreign key and which column it references. Never swap the keys yourself |
| `dimensions:`, `measures:`, `filter:` | the view's dimensions, facts and metrics: ignored, as that file says |

A metric view declares no keys of its own. A base table's `PRIMARY_KEY` is its Unity Catalog
primary-key constraint, read with the §3 declared-keys query, and tagged `[declared]`. Every rule
of that file about primary keys then applies as written — a composite key, a relationship whose
`REF_KEY` is not the table's primary key, two relationships referencing two columns of one
table. A base table with no primary-key constraint has no declared key: its key comes from
preflight §3, as for a view table with no `PRIMARY_KEY`. The one-schema rule, physical names and
`[declared]` relationships apply unchanged, with `catalog.schema` in place of `DB.SCHEMA`.

> **Verified 2026-10-27**, as a user who does not own the view: `DESCRIBE TABLE EXTENDED … AS
> JSON` returns `type` and `view_text`, and `information_schema.views` has no row for a metric
> view. **Not verified:** no metric view has been taken through intake by this skill.

---

## 2. Access

Implements `kumo-data-science/accept/preflight.md` §0. The read tool is Genie Code's own SQL,
run as the signed-in user. The check is the scope query, against `source_scope` itself:

```sql
SELECT schema_name FROM <catalog>.information_schema.schemata WHERE schema_name = lower('<schema>');
```

- A row back: you can see the scope. That is not yet reading it — continue with the read probe
  below.
- No row, or `CATALOG_NOT_FOUND` / `SCHEMA_NOT_FOUND`: name `source_scope` as not found and ask
  for a valid catalog and schema, adding that the user may also lack access to it. Do not
  suggest a similar name, and do not ask for its structure.
- `PERMISSION_DENIED` or `INSUFFICIENT_PERMISSIONS`: the scope exists and this user cannot read it.

**Read probe.** Unity Catalog lists a schema's tables and columns to a user who holds no `SELECT`
on them, so the scope query, and every `information_schema` read in §3, can succeed for tables
the notebook will then fail to read. Once §3 has listed the tables, probe each one the notebook
will read:

```sql
SELECT 1 FROM <catalog>.<schema>.<table> LIMIT 0;   -- no rows, nothing scanned; needs SELECT
```

Quote each name with backticks, as `runtime.md` §3 does. Every probe succeeding is what makes the
scope readable: name the tool in the proposal's `Source scope` line (`Genie Code SQL, as <user>;
<n> tables read-checked`). `INSUFFICIENT_PERMISSIONS` on any probe is the ask-back below, naming
those tables. The probe checks the signed-in user only: when the notebook will run as a service
principal (`runtime.md` §2), which its SQL then runs as too, say in the proposal that the
service principal's access, to these tables and the warehouse, is unchecked.

Run the scope query before any ask-back that names a source: a scope that does not resolve is
listed in that same ask-back, with the other missing fields. When the scope exists but nothing
reads it (`PERMISSION_DENIED`), or a read probe fails, send the preflight §0 ask-back. State what
you ran and what came back, quoting the error, then offer:

1. **Access**: `USE CATALOG`, `USE SCHEMA` and `SELECT` on the tables for this user, granted by
   the user or a workspace administrator; tell you when it is done. Never run a `GRANT` yourself.
2. **Details**: paste the structure. No rows are needed. Useful forms: `DESCRIBE TABLE` output,
   `SHOW CREATE TABLE`, the Catalog Explorer columns — with any declared keys.

**SQL warehouse.** In the same pass, resolve the warehouse the notebook will read through: list
them read-only (`WorkspaceClient().warehouses.list()` from Python), prefer one that is running,
and show it in the proposal as `Warehouse: <name> (<id>) [inferred]`. The user confirms it with
the graph. A warehouse the user names is `[supplied]`. If none is visible, say so; the notebook
then carries the placeholder in `runtime.md` §1.

**What a check may do.** Read catalog metadata, run the read probe above and the preflight §3
probe — nothing else.
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
SELECT table_name, column_name, data_type, full_data_type FROM <catalog>.information_schema.columns
WHERE table_schema = lower('<schema>') ORDER BY table_name, ordinal_position;
-- declared keys, if any (informational constraints); a FOREIGN KEY row names its target
SELECT tc.table_name, tc.constraint_name, tc.constraint_type, kcu.column_name,
       rc.unique_constraint_catalog AS referenced_catalog,
       rc.unique_constraint_schema AS referenced_schema,
       ref.table_name AS referenced_table, ref.column_name AS referenced_column
FROM <catalog>.information_schema.table_constraints tc
JOIN <catalog>.information_schema.key_column_usage kcu
  ON tc.constraint_catalog = kcu.constraint_catalog AND tc.constraint_schema = kcu.constraint_schema
 AND tc.constraint_name = kcu.constraint_name
LEFT JOIN <catalog>.information_schema.referential_constraints rc
  ON tc.constraint_catalog = rc.constraint_catalog AND tc.constraint_schema = rc.constraint_schema
 AND tc.constraint_name = rc.constraint_name
LEFT JOIN <catalog>.information_schema.key_column_usage ref
  ON rc.unique_constraint_catalog = ref.constraint_catalog
 AND rc.unique_constraint_schema = ref.constraint_schema
 AND rc.unique_constraint_name = ref.constraint_name
 AND kcu.position_in_unique_constraint = ref.ordinal_position
WHERE tc.table_schema = lower('<schema>')
ORDER BY tc.table_name, tc.constraint_name, kcu.ordinal_position;
```

- A FOREIGN KEY row links two tables in scope only when its `referenced_catalog` and
  `referenced_schema` are the confirmed `source_scope`, compared lowercased. A key into another
  schema or catalog (there `referenced_table` may be empty) is not a link: never propose it, and
  name it in the proposal as a reference outside the scope.
- Row counts are not in `information_schema` and are not needed for the graph proposal.
- Unity Catalog types: `STRING`, `BIGINT`, `INT`, `DOUBLE`, `DECIMAL(p,s)`, `BOOLEAN`, `DATE`,
  `TIMESTAMP`, `TIMESTAMP_NTZ`, `ARRAY<...>`. Read the type from `full_data_type`: `data_type`
  reports `BIGINT` as `LONG` and every array as plain `ARRAY`, so only `full_data_type`
  (`array<double>`, `bigint`) tells a supported array from an unsupported one (§6). Time columns
  are `DATE`, `TIMESTAMP` or `TIMESTAMP_NTZ`; preflight §4 applies unchanged.
- The confirmed `source_scope` becomes the connector's `catalog` and `schema_name`, and the
  confirmed warehouse its `warehouse_id` (`runtime.md` §2).

---

## 4. Destinations

Predictions and embeddings go to a Unity Catalog table, `catalog.schema.table`. It is written
through a `DatabricksConnector` on that catalog and schema
(`kumoai-reference/references/kumo_finetuning_sdk.md` §9.2), which stages the rows through the
volume `<catalog>.<schema>._kumo_staging` and creates that volume on first use. So writing needs,
for the user who runs the notebook: `USE CATALOG`, `USE SCHEMA`, `CREATE TABLE`, and either
`CREATE VOLUME` on the schema or `READ VOLUME` and `WRITE VOLUME` on an existing `_kumo_staging`.
A new volume is a write the user confirms: name it on its own restatement line, after the
verbatim `Output:` line (`runtime.md` §4). Not destinations for predictions here: a path in a volume, S3, GCS or local
files, and any table outside Unity Catalog.

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
| Snowflake `ARRAY` / `VARIANT` columns arrive as strings | by `full_data_type` (§3), the connector types `ARRAY<FLOAT>`, `ARRAY<DOUBLE>`, `ARRAY<INT>`, `ARRAY<BIGINT>` and `ARRAY<STRING>` as lists (Source: shipped package (`kumo_databricks.types`)); other `ARRAY`s, `MAP`s and `STRUCT`s are not. Supplied list targets, link weights and forecast sequences have not been run on this host: say so at the gate rather than promise them |
| a Snowflake ML Job | a Databricks job run on a GPU cluster (`job.md`) |
| "the existing Snowflake notebook" | the existing Databricks notebook; run files are on the cluster's local disk |

A step that has no row here and names a Snowflake object is not covered on this host: say so; do
not adapt it.

### 6.1 Scoring a registered model

This refines `batch_inference.md` Node 1c and 1d for this host. For `batch_inference` from a Unity
Catalog registered model, Node 1c reads the version's training
record — the version's comment and tags (`databricks model-versions get`), then the files they
point to in its run (`databricks experiments list-artifacts --run-id`). Read a file with
`databricks api get "/api/2.0/mlflow/artifacts/credentials-for-read?run_id=<id>&path=<path>"`,
then `curl` its `signed_uri`. Never read a credential file or pass a token yourself. That record is the
graph: nothing in it is proposed or inferred, so 1c and 1d are **one gate**. When the record is
found and the live source still has every recorded column, reply with the
`kumo-data-science/accept/batch_inference.md` Node 1d restatement, one `Graph:` line added, and
stop for confirmation:

```
- Model: <catalog.schema.model> version <N> — trained task reused unchanged
- Source: <source in the record> [recorded]
- Graph: rebuilt from the version's training record — <n> tables, the recorded column lists, keys and links [recorded]
- Population: all <entity> as of latest snapshot timestamp [default]
- Output: <catalog.schema.table, verbatim>
- Staging volume: creates <catalog.schema>._kumo_staging   (only when it does not exist)
- Embeddings: <included, if explicitly requested>
- Scoring options: <as batch_inference.md gives them by task type>
```

`[recorded]` marks a value read from the version's training record.

For a job, add the `job.md` §3 lines. Do not restate the task (entity, outcome, window) or list
tables and columns beyond that line:
the task is reused, not reconfirmed. A recorded column the live source lacks, or a record that
names no source, is the 1c finding to report instead.

When the version has **no** training record, say that no column selection or graph definition
is recorded for that version, that scoring must use the columns it was trained on, and ask for
the training-time graph definition — the graph-building code or the per-table column lists.
Ask only for that: not the model, version, source or destination. Never propose retraining or
publishing a new model in its place; that is a new request for the user to make.

---

## 7. Other deployment modes — out of scope

Hosted Kumo — `kumo.init(...)`, an API key, `KUMO_API_KEY` / `KUMO_API_ENDPOINT`, Kumo Studio — is
not this SDK, and neither is Kumo through Spark (a Spark DataFrame handed to Kumo; no such
connector exists). If the user is on one of them, say so and stop. Never blend their setup into
generated code. A Model Serving endpoint is not something this skill creates: a request to deploy
a Kumo model as a live endpoint is out of scope.
