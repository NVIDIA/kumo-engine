# Host Runtime — Snowflake CoCo

Read at Step 2, when generating the notebook: authentication, setup cell, compute, storage,
permissions and write-back. Section numbers continue those in `adapter.md` (§1.4, §6.1 and §9
are there).

---

## 0. Blocking risk — read first

This path assumes (a) Kumo is already present in the execution runtime and (b) that runtime can authenticate to Snowflake.

**Assume Kumo is present.** Setting up the environment is not this skill's job — never raise it with the user, and never generate a cell that does it. If `import kumo.engine` raises, the environment is not ready: report that and stop.

**Share the runtime's Snowpark session** (§3) with `SnowflakeConnector.from_session()`. It uses the notebook's warehouse and role. For a separate session, use `from_spcs_env()`, `from_connection_toml()`, or a `SnowflakeIdentity` from secrets.

**If Kumo is absent or no session or identity is available, this deployment mode is blocked** — report it and stop. Confirm the runtime with the CoCo team before building on it (`INTEGRATION.md` §10).


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

A Snowflake ML Job runs in its own container, not in this one. How Kumo is present there is in
`ml_job.md` and `adapter.md` §10.

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

---

## 2. Notebook compute and GPU sizing

- Compute must be compatible with the Kumo build that is present. For GPU execution the GPU must be **visible to the Python process**, with a driver compatible with the build's CUDA and PyTorch dependencies. Do not infer GPU or driver version from the Python version.
- Memory, GPU memory, and disk depend on data and workload. There is **no validated universal minimum** for this path. Confirm sizing before provisioning compute or starting a large job.

> Not verified: GPU model, VRAM, CPU, system-memory, storage, and scaling limits for this path. Do not substitute numbers from the self-hosted Kubernetes/Docker or SPCS native-app modes — different deployment modes, different figures.

> Not verified: the required notebook runtime, and whether a warehouse-backed or container-backed runtime is needed. Training and prediction have run in a Snowflake Notebook container; confirm with the CoCo team (`INTEGRATION.md` §10).


---
## 3. Setup cell — scratch directory and connector

There is **no backend object**. Configure scratch storage, then share the runtime's session.

```python
from pathlib import Path

import kumo.engine as kumo
from snowflake.snowpark.context import get_active_session

scratch = Path('/tmp/kumo-run/scratch')
scratch.mkdir(parents=True, exist_ok=True)
kumo.configure(scratch_dir=scratch)

session = get_active_session()
connector = kumo.SnowflakeConnector.from_session(
    session, database=DATABASE, schema_name=SCHEMA)
assert connector.table_names(), 'Connector reached no tables.'
```

- `DATABASE` and `SCHEMA` are declared Snowflake identifiers. Keep double quotes inside lower-case or mixed-case names (`SCHEMA = '"my_schema"'`); unquoted names fold to upper case.
- `from_session()` uses the session's warehouse and role. Confirm they are the intended ones. It preserves the session's context and lifetime; `get_active_session()` remains usable when this is the only session.
- Database and schema overrides support table access. SQL sources require matching session context; call `session.use_database(...)` and `session.use_schema(...)`, or omit the overrides.
- Kumo's DDL commits any open transaction. Kumo propagates a lapsed shared-session token rather than reconnecting it.
- `kumo.configure(scratch_dir=...)` is optional (default `~/.cache/kumo-engine`); call it before constructing a `Graph` or `Trainer`.
- For a separate session, use `from_spcs_env()` (user's default role unless `role=` is passed), `from_connection_toml()`, or `SnowflakeIdentity` from secrets. Remote execution, including Ray workers, needs `identity=`; a carried session is dropped when pickled.
- Full signatures and credential examples: `kumoai-reference/references/kumo_finetuning_sdk.md` §3.

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

Use the runtime's active Snowpark session through `from_session()`. It supplies the authenticated user, role and warehouse; no token file or separate identity is needed. The shared session's role needs the source and destination privileges (§6).

For a separate session, `from_spcs_env()` reads `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_HOST` and the mounted OAuth token at `SPCS_TOKEN_PATH`. Use `from_connection_toml()` or a `SnowflakeIdentity` from secrets when those runtime credentials are unavailable (§3).

**No password, private key, API key, or token literal may appear in generated code.** Never prompt the user for a credential's value. Read a key pair from a Snowflake secret by name.

Creating a connector or identity grants no Snowflake permissions.

> Not verified: the key-pair branch (`SnowflakeIdentity` built by hand) against this runtime.


---
## 6. Snowflake-side permissions

The shared session uses its current Snowflake role. A separate session uses the role configured by its identity factory or connector.

The session's role needs access to the warehouse and source data. Query, staging, export, and output permissions depend on the connector and workflow and must be confirmed with a Snowflake administrator. Writing predictions requires access to the chosen output destination.

> Not verified: the concrete GRANT statements for the local-backend path. Do not emit a grant set borrowed from the native-app or Kumo-cloud modes as if it were required here.


---
## 7. Writing predictions back to Snowflake

Write-back is the reason this skill exists, and it is the only durable home for results (§3.1).

The verified write-back form — `kumo.PredictionOutputConfig(connector=, artifact_type=, table_name=, write_mode=)` passed to `trainer.predict(..., snapshot=snapshot, output_config=...)` — is in `kumoai-reference/references/kumo_finetuning_sdk.md` §9. Emit that and nothing else. **The table is written with exactly the name you pass**; read it back and compare row counts. An APPEND write adds the run's columns, as `kumoai-reference/references/outputs.md` *Write predictions* says.

The session's role must be able to write to the destination; a successful connection does not prove it. Verify with a trivial write before a long job.


---
## 8. Setup sequence (canonical order)

1. Confirm `import kumo.engine` succeeds and the runtime can authenticate (§0, §1.1). If not — stop.
2. Create a writable scratch directory (§1.2).
3. `kumo.configure(scratch_dir=...)`, then `SnowflakeConnector.from_session()` (§3).
4. Assert `connector.table_names()` (§3).
5. Tables → graph → `validate()` → predictive query → `snapshot()` → train → predict
   (portable core; `snapshot=` is required on `fit` and `predict`).
6. Write results to a durable Snowflake destination before the session ends (§3.1, §7).

---

## Graph visualization

> **Not verified:** interactive rendering has not been tested in Snowflake Workspace notebooks.
The SDK selects a widget in Workspace, requiring `anywidget>=0.9,<1` and `ipywidgets`; if these dependencies cannot be imported, it displays a static SVG instead.
Other notebook hosts use HTML.
Retrieve exported diagrams from the notebook filesystem through the Workspace file browser.

Source: shipped package (`kumo.engine.graph.Graph.visualize`, `kumo.engine._visualization.SVG`)
