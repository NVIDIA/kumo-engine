# Host Adapter — Snowflake CoCo

Everything Snowflake-specific for running the data science agent inside Snowflake. The portable core (`kumo-data-science/`, `kumoai-reference/`) does not change across hosts.

**Target deployment mode: Kumo running locally, in-process, inside a Snowflake notebook**, with Kumo **already present in the environment** and authenticating through the runtime's mounted SPCS identity. All other Kumo/Snowflake deployment modes are out of scope — see §9.

**The published Kumo documentation is stale for this version of the local SDK and is not a source here.** Do not cite it, link to it, fetch it, or reconcile against it. Facts marked ``Source: shipped package (`<python.module.path>`)`` or `Source: measured ... runs` are verified against the Kumo package source or by execution and are ground truth; everything else is compiled guidance, flagged where unverified. **Do not invent API names, parameters, env vars, privileges, or SQL.**

---

## 0. Blocking risk — read first

This path assumes (a) Kumo is already present in the execution runtime and (b) that runtime can authenticate to Snowflake.

**Assume Kumo is present.** Setting up the environment is not this skill's job — never raise it with the user, and never generate a cell that does it. If `import kumo.engine` raises, the environment is not ready: report that and stop.

**Authenticate in this order of priority** (§3): (1) the runtime's mounted OAuth token, through `from_spcs_env()`; (2) a `connections.toml` entry, through `from_connection_toml()`; (3) a key pair read from Snowflake secrets attached to the notebook, through `SnowflakeIdentity`. Generated code tries them in that order at run time.

**If the CoCo runtime does not have Kumo present, or none of the three is available, this deployment mode is blocked** — report it and stop. Confirm the runtime with the CoCo team before building on it. See §10.

An active Snowpark session alone is **not** a substitute for any of the three.

---

## 1. Runtime prerequisites

### 1.1 Kumo is already present

The agent's first Kumo line is `import kumo.engine as kumo` (§0).

Confirm the environment rather than assuming it:

```python
import kumo.engine as kumo
print('kumo.engine:', kumo.__version__)
```

If that import raises, stop and report that the environment is not ready.

Notebook kernel observed: Python 3.12.

### 1.2 Execution storage

Scratch and artifacts can live under `/tmp` in a Snowflake Notebook container; training and prediction complete there.

- Both the scratch directory and the artifact directory must exist and be writable.
- Space is needed for intermediate data as well as saved outputs.
- **Local NVMe SSD is a production scaling concern, not a gate.** Do not block a small run on it. An `/dev/nvme*` device name is not proof of one — AWS exposes EBS volumes through an NVMe interface. `graph.snapshot()` itself warns when the scratch directory is on network storage; treat that as advice, not a failure.

> Not verified: the storage class and throughput Kumo actually needs at full data scale on this path.

Source: shipped package (`kumo.engine.config`, `kumo.engine.graph`)

### 1.3 Network access

The runtime executing Python must reach: Snowflake authentication services including credential-refresh endpoints; and the data, query, and storage services used by the connector and output destination.

> Notebook network policies must apply to the runtime where the code executes, not just to your browser.

### 1.4 Runtime-dependent text features

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

## 2. Notebook compute and GPU sizing

- Compute must be compatible with the Kumo build that is present. For GPU execution the GPU must be **visible to the Python process**, with a driver compatible with the build's CUDA and PyTorch dependencies. Do not infer GPU or driver version from the Python version.
- Memory, GPU memory, and disk depend on data and workload. There is **no validated universal minimum** for this path. Confirm sizing before provisioning compute or starting a large job.

> Not verified: GPU model, VRAM, CPU, system-memory, storage, and scaling limits for this path. Do not substitute numbers from the self-hosted Kubernetes/Docker or SPCS native-app modes — different deployment modes, different figures.

> Not verified: the required notebook runtime, and whether a warehouse-backed or container-backed runtime is needed. Training and prediction have run in a Snowflake Notebook container; confirm with the CoCo team (§10).

---

## 3. Setup cell — scratch directory and connector

There is **no backend object**. Two calls get you connected.

```python
import os
from pathlib import Path

import kumo.engine as kumo
from kumo_snowflake.spcs import SPCS_TOKEN_PATH

scratch = Path('/tmp/kumo-run/scratch')
scratch.mkdir(parents=True, exist_ok=True)
kumo.configure(scratch_dir=scratch)          # optional; once, before anything else

TARGET = dict(warehouse=WAREHOUSE, database=DATABASE, schema_name=SCHEMA)
if os.path.exists(SPCS_TOKEN_PATH):                                  # 1. runtime token
    connector = kumo.SnowflakeConnector.from_spcs_env(**TARGET)
else:
    try:                                                             # 2. connections.toml
        connector = kumo.SnowflakeConnector.from_connection_toml(
            connection_name=CONNECTION_NAME, **TARGET)
    except ValueError as err:
        if 'connections.toml has no connection' not in str(err):
            raise
        from snowflake.snowpark.secrets import get_generic_secret_string
        identity = kumo.SnowflakeIdentity(                           # 3. key pair from secrets
            account=ACCOUNT, username=USERNAME,
            private_key=get_generic_secret_string(PRIVATE_KEY_SECRET),
            private_key_passphrase=get_generic_secret_string(PASSPHRASE_SECRET))
        connector = kumo.SnowflakeConnector(identity=identity, **TARGET)

assert connector.table_names(), 'Connector reached no tables.'
```

- `DATABASE` and `SCHEMA` are declared Snowflake identifiers. A lower-case name keeps its double
  quotes inside the string (`SCHEMA = '"my_schema"'`); unquoted, it is folded to upper case and
  the connector reaches no tables (`kumoai-reference/references/kumo_finetuning_sdk.md` §3).
- Take the Snowpark session with `get_active_session()` once, before this block, and reuse it.
  Kumo opens its own session, so a later call raises (`kumoai-reference/references/kumo_finetuning_sdk.md` §3.3).
- `kumo.configure(scratch_dir=...)` is the **only** process-wide setup call, and it is optional
  (default `~/.cache/kumo-engine`). Call it once, early, before any `Graph` or `Trainer`.
- `from_spcs_env()` builds the connector **and its identity** from the runtime's mounted SPCS
  credentials; `from_connection_toml()` does the same from the named entry. Neither puts any
  credential in a cell.
- The key-pair branch reads the key and its passphrase from Snowflake secrets attached to the
  notebook; the code holds only the secret **names**. `ACCOUNT` and `USERNAME` are the Snowflake
  account identifier and the key's user. Ask the user for these names only when the first two
  options are unavailable in their runtime — never for a secret's value.
- Always pass `warehouse`, `database`, `schema_name` explicitly — they default to `None`.
- `SPCS_TOKEN_PATH` is imported, never hardcoded.
- Full signatures in `kumoai-reference/references/kumo_finetuning_sdk.md` §3.

**`kumo.set_backend`, `kumo.LocalBackend`, `kumo.Storage`, `kumo.FilesystemArtifactLocation`,
`kumo.create_identity` and `kumo.test_identity_credentials` no longer exist.** If a user's
notebook contains them it targets a superseded build — say so and rewrite it.

### 3.1 Storage lifetime — spell this out to the user

Everything under the scratch directory is **session-scoped and disposable**. `/tmp` in a notebook
container does not survive a restart, and nothing about `configure()` makes it persistent.

Anything that must outlive the session — predictions above all — has to be written to a durable
Snowflake destination before the session ends. Never describe a scratch path as durable, and never
plan a workflow that reads an artifact back in a later session without first confirming storage
lifetime.

---

## 4. What the Snowflake runtime provides

| Item | Value |
|---|---|
| Account | env var `SNOWFLAKE_ACCOUNT` |
| Host | env var `SNOWFLAKE_HOST` |
| OAuth token | file at `/snowflake/session/token` (`SPCS_TOKEN_PATH`) |

`from_spcs_env()` reads these for you. You should not need to touch them directly.

**Hard rule for generated code: no password, private key, API key, or token literal may ever
appear.** Never prompt the user for a credential's value and never inline one. A key pair is
read from a Snowflake secret by name, never pasted.

An active Snowpark session alone is **not** sufficient. Without the token file, fall back to
`connections.toml`, then to the key pair from secrets (§0) — never to pasted credentials.

Creating a connector grants nothing in Snowflake. The identity authenticates as the principal
behind the runtime token, and that principal needs its own privileges (§6).

> Not verified: the key-pair branch (`SnowflakeIdentity` built by hand) against this runtime.
> It is the last resort; prefer the factories (`from_spcs_env`, `from_connection_toml`).

---

## 6. Snowflake-side permissions

The identity authenticates as the Snowflake principal behind the runtime token. **Creating a Kumo identity does not create a Snowflake user and grants no permissions.**

The identity needs access to the warehouse and source data. Query, staging, export, and output permissions depend on the connector and workflow and must be confirmed with a Snowflake administrator. Writing predictions requires access to the chosen output destination.

> Not verified: the concrete GRANT statements for the local-backend path. Do not emit a grant set borrowed from the native-app or Kumo-cloud modes as if it were required here.

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
the connector in §3. Snowflake's `INFORMATION_SCHEMA` needs only `USAGE` on the database and
schema plus `SELECT` (or `REFERENCES`) on the tables to be listed.

---

## 7. Writing predictions back to Snowflake

Write-back is the reason this skill exists, and it is the only durable home for results (§3.1).

The verified write-back form — `kumo.PredictionOutputConfig(connector=, artifact_type=, table_name=, write_mode=)` passed to `trainer.predict(..., snapshot=snapshot, output_config=...)` — is in `kumoai-reference/references/kumo_finetuning_sdk.md` §9. Emit that and nothing else. **The table is written with exactly the name you pass**; read it back and compare row counts.

Two further facts: the target destination must be writable by the identity's principal, and a passing credential check does not prove it. Verify with a trivial write before a long job.

---

## 8. Setup sequence (canonical order)

1. Confirm `import kumo.engine` succeeds and the runtime can authenticate (§0, §1.1). If not — stop.
2. Create a writable scratch directory (§1.2).
3. `kumo.configure(scratch_dir=...)`, then the connector, in the §3 order: `from_spcs_env`,
   `from_connection_toml`, key pair from secrets.
4. Assert `connector.table_names()` (§3).
5. Tables → graph → `validate()` → predictive query → `snapshot()` → train → predict
   (portable core; `snapshot=` is required on `fit` and `predict`).
6. Write results to a durable Snowflake destination before the session ends (§3.1, §7).

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

## 10. Still needs the CoCo team

Not answerable from anything on hand. Do not guess.

**Blocking (answer these first):**

- **Is Kumo present in the CoCo runtime — does `import kumo.engine` succeed?** If not, this path is dead (§0, §1.1).
- **Does the CoCo runtime expose `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_HOST`, and `/snowflake/session/token`?** A Snowpark session alone is explicitly insufficient. If no, generated code falls back to `connections.toml`, then to a key pair from secrets (§0, §3).
- **Which paths in a CoCo session are writable, and what is backed by fast local storage?** (§1.2)

**Non-blocking but required before shipping:**

- CoCo skill packaging format and manifest schema.
- Notebook/runtime compute and GPU selection available to a CoCo session, and whether a GPU is visible to the Python process (§2).
- Lifetime of CoCo session storage across restarts — determines how aggressively results must be flushed to Snowflake (§3.1).
- Whether CoCo imposes conventions on the prediction output table.
- CoCo-level context window, token budget, latency, and rate limits.
- Long-running training, for the opt-in Run step only: whether a CoCo session stays alive for the whole of a blocking `fit`, which can run for hours. There is no job to poll; if the session cannot outlive training, the Run step is not viable on this host.

**Needed from Kumo (not CoCo):**

- The GRANT set for this path (§6).
- Validated GPU/memory/storage sizing for this path (§2).

---

## 11. Integration checklist

- [ ] `import kumo.engine` succeeds in the CoCo runtime; version printed.
- [ ] Generated code begins at `import kumo.engine`; no environment-setup step anywhere in it.
- [ ] Scratch and artifact directories exist and are writable.
- [ ] One authentication option available, in priority order: the SPCS token (`SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_HOST`, `/snowflake/session/token`), a `connections.toml` entry, or key-pair secrets attached to the notebook.
- [ ] `kumo.configure(scratch_dir=...)` called once, before any `Graph` or `Trainer`.
- [ ] Connector built by `from_spcs_env()`, else `from_connection_toml()`, else `SnowflakeIdentity` from secrets — with explicit warehouse, database, schema_name.
- [ ] `connector.table_names()` asserted — then separately verify the output write.
- [ ] No call from `kumoai-reference/references/kumo_finetuning_sdk.md` §11.1 appears anywhere.
- [ ] No credential literal anywhere in generated code.
- [ ] Every result the user needs later is written to a durable Snowflake destination, not left in artifacts.
- [ ] Preflight SQL (§6.1) runs against the user's `source_scope` from a CoCo session.
- [ ] End-to-end test: run the canonical good example from `validate/examples/good_request.md` unmodified in the target environment.
