# Host Adapter — Databricks Genie Code

For **Genie Code** (agent mode) inside a Databricks workspace. You run as the signed-in user. With
the user's approval you can search Unity Catalog, run SQL and Python, create and edit notebooks,
and read cell output. The generated notebook runs in the same workspace, on a GPU cluster, reading
and writing Unity Catalog through `DatabricksConnector`.

Kumo is already present on the GPU cluster that runs the notebook; never raise environment setup
with the user. Generating the notebook — connect step, destinations, read-back and running it —
is in `runtime.md` in this directory; read it at Step 2.

This host has no job file: it does not submit, schedule or monitor jobs. When asked for one, say
so and offer the notebook.

---

## 1. Source scope

`source_scope` is a Unity Catalog **catalog and schema**: `catalog.schema`, or a schema with the
catalog named elsewhere in the conversation. Tables are `catalog.schema.table`.

- A bare schema name with no catalog in any turn: look for it with the §2 scope query across the
  catalogs you can see. One match settles it; several, ask which catalog — that one question.
- A name written with a hyphen in chat (`my-catalog`) is the same object as its underscore form
  (`my_catalog`) — resolve, do not ask. Unity Catalog names are lower case.
- A semantic view, a Snowflake database, an `s3://` or `gs://` path, or a BigQuery dataset is not
  a source on this host. Say so, and say that the data must be in a Unity Catalog schema for this
  host to read it.
- The SQL warehouse is not part of `source_scope` and is never asked for at intake. §2 resolves
  it.

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

A Unity Catalog table, `catalog.schema.table`, in any schema the user can create tables in. It is
written through a `DatabricksConnector` on that catalog and schema
(`kumoai-reference/references/kumo_finetuning_sdk.md` §9.2). Not destinations here: a path in a
volume, S3, GCS or local files, and any table outside Unity Catalog.

---

## 5. Runtime column constraints

None known for this host. Do not exclude text columns pre-emptively. If a run fails while loading
a text encoder, report the error and the column; that finding belongs at Node 1c.

---

## 6. Capability skills

The capability skills were written for, and measured on, Snowflake. Their Kumo calls hold for
`DatabricksConnector`. Where a step names a Snowflake registry, stage, session or table, say that
the step is not covered on this host; do not adapt it — including publishing a model to Unity
Catalog, which no run of this skill has exercised.

---

## 7. Other deployment modes — out of scope

Hosted Kumo — `kumo.init(...)`, an API key, `KUMO_API_KEY` / `KUMO_API_ENDPOINT`, Kumo Studio — is
not this SDK, and neither is Kumo through Spark (a Spark DataFrame handed to Kumo; no such
connector exists). If the user is on one of them, say so and stop. Never blend their setup into
generated code.
