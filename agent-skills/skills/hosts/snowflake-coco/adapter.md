# Host Adapter — Snowflake CoCo

Everything Snowflake-specific for running the data science agent inside Snowflake. The portable core (`kumo-data-science/`, `kumoai-reference/`) does not change across hosts.

**Target deployment mode: Kumo running locally, in-process, inside a Snowflake notebook**, with Kumo **already present in the environment** and authenticating through the runtime's mounted SPCS identity. A Snowflake ML Job that the notebook submits is the same mode on other compute: Kumo still runs locally, in-process, inside the job's container (`ml_job.md`). All other Kumo/Snowflake deployment modes are out of scope — see §9.

**The published Kumo documentation is stale for this version of the local SDK and is not a source here.** Do not cite it, link to it, fetch it, or reconcile against it. Facts marked ``Source: shipped package (`<python.module.path>`)`` or `Source: measured ... runs` are verified against the Kumo package source or by execution and are ground truth; everything else is compiled guidance, flagged where unverified. **Do not invent API names, parameters, env vars, privileges, or SQL.**

Kumo is already present in the environment; never raise environment setup with the user.
Generating and running the notebook — authentication, setup cell, compute, storage, permissions
and write-back — is in `runtime.md` in this directory; read it at Step 2.
Running the artifact as a Snowflake ML Job — when to, the job script, submission, logs,
schedules and cost — is in `ml_job.md`; read it only when the user asks for a job.

---

## 0. Source scope, access and destinations

**Source scope.** A Snowflake database and schema (warehouse optional), a fully qualified semantic
view (`DB.SCHEMA.VIEW`, resolved by `kumo-data-science/accept/semantic_view.md`), or a connector
name.

**Access** (`kumo-data-science/accept/preflight.md` §0). The read tool is CoCo's own SQL, run as
the session's role; the check is the §6.1 scope query. A schema missing from `INFORMATION_SCHEMA`
may not exist or may not be visible to this role — say both. On an authorization error, or a
scope the role cannot see, offer the two ways forward: a role with `USAGE` on the database and
schema and `SELECT` on its tables, granted by the user or their administrator; or the structure
pasted in. Never run a `GRANT` yourself.

**Destinations.** Snowflake tables only: a connector plus a table name, or `DB.SCHEMA.TABLE`. An
`s3://`, `gs://` or file path is not a destination on this host.

---

## 1.4 Runtime-dependent text features
In measured Snowflake CoCo runs, retaining a column resolved as `Stype.text` made text-encoder
initialization fail with `NoCredentialsError` while resolving GloVe assets. The remedy is the
column selection below; it needs nothing further from the user.

Until the runtime demonstrates that the local default embedding model loads, treat text features
as unavailable on this host: exclude columns resolving to `Stype.text` from the positional column
list and report the exclusion at the graph gate. Decide on the **semantic** type, not Snowflake's
physical string type — string-backed identifiers and short categorical codes are unaffected and
must be kept. Never relabel natural-language content as categorical to dodge the constraint. This
is a host restriction, not a portable Kumo restriction.

Preflight cannot predict Kumo's sampled type inference, so a column name is not proof either way:
at the graph gate, exclude only columns whose text semantics are already established. The
generated code carries the runtime guard, because the user may run it without the agent: after
`infer_metadata()` and the `with_column` corrections, and before `graph.snapshot()`, emit a cell
that fails on any retained `Stype.text` column.

```python
text_cols = [f'{t.table_name}.{c.name}'
             for t in tables for c in t.definition.cols if c.stype == Stype.text]
if text_cols:
    raise RuntimeError(f'Text features are unavailable on this host; '
                       f'revisit the graph proposal and exclude: {text_cols}')
```

`tables` is every table in the graph. When the guard fires, return to Node 1c with a revised
proposal naming those columns, rather than silently dropping them or snapshotting anyway.

Source: measured Snowflake CoCo runs, 2026-10-02 and 2026-10-03.


---
## 6.1 Read-only preflight in Snowflake

Intake Node 1c (`kumo-data-science/accept/preflight.md`) resolves structure with SQL. Its first
pass runs before any Kumo call; the workflow may return to it once the runtime guard has resolved
a type the catalog did not show. In CoCo this is ordinary SQL against the account; the connector is not involved.

```sql
-- does the scope exist?
SELECT SCHEMA_NAME FROM <DB>.INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME ILIKE '<schema>';
-- tables in scope (enumerate only here — never elsewhere)
SELECT TABLE_NAME, ROW_COUNT FROM <DB>.INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA ILIKE '<schema>';
-- columns and types
SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM <DB>.INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA ILIKE '<schema>' ORDER BY TABLE_NAME, ORDINAL_POSITION;
-- declared keys, if any
SHOW PRIMARY KEYS IN SCHEMA <DB>.<schema>;
SHOW IMPORTED KEYS IN SCHEMA <DB>.<schema>;
-- when source_scope is a semantic view: base tables, primary keys, relationships
DESCRIBE SEMANTIC VIEW <DB>.<schema>.<view>;
```

A semantic view is never queried and is not visible to the connector — `connector.table_names()`
lists only tables and views. Its base tables are what Kumo reads, so the principal needs `SELECT`
on those tables; access to the view alone is not enough.

Database identifiers written with a hyphen in chat (`synthetic-datasets`) are the same object as
their underscore form (`SYNTHETIC_DATASETS`) — resolve, do not ask.

The `source_scope` the user confirmed at intake is also the `database` / `schema_name` passed to
the connector in `runtime.md` §3. Snowflake's `INFORMATION_SCHEMA` needs only `USAGE` on the database and
schema plus `SELECT` (or `REFERENCES`) on the tables to be listed.


---
## 9. Other deployment modes — out of scope

Mentioned only so the agent recognises and refuses them. **Never blend these into local-backend setup code.**

| Mode | Marker to recognise it |
|---|---|
| Snowflake Native App on SPCS | `kumo.init(snowflake_application=...)`, Marketplace app, a compute pool or GRANTs that belong to a Kumo application |
| Kumo SaaS / external Kumo cloud | `kumo.init(url=..., api_key=...)`, `KUMO_API_ENDPOINT` / `KUMO_API_KEY` |
| Secure Data Sharing | UI/EDA-oriented, Kumo creates a database from a share |
| Self-hosted VPC / Kubernetes / Docker | `POOL_GPU_CONCURRENCY`, `topolvm-vg`, cluster GPU-sizing tables |

Keep the environment-specific local setup; never replace it with cloud-URL or API-key initialization.

If the user is actually on the Kumo Native App or Kumo SaaS, say so and stop — this adapter does not cover them.

**A compute pool alone is not a marker.** A Snowflake ML Job (`snowflake.ml.jobs`) takes a
`compute_pool`, and so does a container notebook. Neither involves a Kumo application: the job
script imports `kumo.engine` and connects with `from_spcs_env`, exactly as the notebook does.
Treat a request to score or train on a named compute pool as an ML Job request (`ml_job.md`),
never as the Native App. Only a pool or GRANT tied to a Kumo application, or `kumo.init(...)`,
marks the Native App.

---
## 10. Host-supplied values for Snowflake ML Jobs

A job runs in a fresh container, not in the notebook's environment. The host supplies what that
container needs for Kumo to be present. The skill refers to these values by name only.

| Name | What it is | On this host |
|---|---|---|
| `KUMO_JOB_RUNTIME` | The `runtime_environment` of every job that runs Kumo | `'2.9.0'`. Write this literal into the submission cell. If the host supplies a runtime image with Kumo present, its reference replaces this value here, and nothing else changes |
| `KUMO_JOB_REQUIREMENTS` | A dict of further submission keyword arguments that make Kumo present in the job container. It may be empty | Supplied by the host in the notebook's namespace. The skill never writes, extends, prints or explains its contents |

When `KUMO_JOB_REQUIREMENTS` is not defined in the notebook, the submission cell stops before
submitting (`ml_job.md` §6). Tell the user that running Kumo in an ML Job is not enabled in this
environment and is a matter for their administrator. Do not substitute a value of your own.

If the user asks whether the job has Kumo, or what to add to the job for it: the host's values
provide Kumo to the job, and the user adds nothing. The default ML Job runtime does not carry Kumo
by itself.

---
