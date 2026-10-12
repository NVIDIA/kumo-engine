# Host Adapter — Snowflake CoCo

Everything Snowflake-specific for running the data science agent inside Snowflake. The portable core (`kumo-data-science/`, `kumoai-reference/`) does not change across hosts.

**Target deployment mode: Kumo running locally, in-process, inside a Snowflake notebook**, with Kumo **already present in the environment** and authenticating through the runtime's mounted SPCS identity. All other Kumo/Snowflake deployment modes are out of scope — see §9.

**The published Kumo documentation is stale for this version of the local SDK and is not a source here.** Do not cite it, link to it, fetch it, or reconcile against it. Facts marked ``Source: shipped package (`<python.module.path>`)`` or `Source: measured ... runs` are verified against the Kumo package source or by execution and are ground truth; everything else is compiled guidance, flagged where unverified. **Do not invent API names, parameters, env vars, privileges, or SQL.**

Kumo is already present in the environment; never raise environment setup with the user.
Generating and running the notebook — authentication, setup cell, compute, storage, permissions
and write-back — is in `runtime.md` in this directory; read it at Step 2.

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
| Snowflake Native App on SPCS | `kumo.init(snowflake_application=...)`, Marketplace app, compute pools, app-level GRANTs |
| Kumo SaaS / external Kumo cloud | `kumo.init(url=..., api_key=...)`, `KUMO_API_ENDPOINT` / `KUMO_API_KEY` |
| Secure Data Sharing | UI/EDA-oriented, Kumo creates a database from a share |
| Self-hosted VPC / Kubernetes / Docker | `POOL_GPU_CONCURRENCY`, `topolvm-vg`, cluster GPU-sizing tables |

Keep the environment-specific local setup; never replace it with cloud-URL or API-key initialization.

If the user is actually on SPCS or SaaS, say so and stop — this adapter does not cover them.

---
