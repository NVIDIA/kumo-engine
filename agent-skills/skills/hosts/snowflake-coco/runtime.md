# Host Runtime — Snowflake CoCo

Read at Step 2, when generating the notebook: authentication, setup cell, compute, storage,
permissions and write-back. Section numbers continue those in `adapter.md` (§1.4, §6.1 and §9
are there).

---

## 0. Blocking risk — read first

This path assumes (a) Kumo is already present in the execution runtime and (b) that runtime can authenticate to Snowflake.

**Assume Kumo is present.** Setting up the environment is not this skill's job — never raise it with the user, and never generate a cell that does it. If `import kumo.engine` raises, the environment is not ready: report that and stop.

**Authenticate in this order of priority** (§3): (1) the runtime's mounted OAuth token, through `from_spcs_env()`; (2) a `connections.toml` entry, through `from_connection_toml()`; (3) a key pair read from Snowflake secrets attached to the notebook, through `SnowflakeIdentity`. Generated code tries them in that order at run time.

**If the CoCo runtime does not have Kumo present, or none of the three is available, this deployment mode is blocked** — report it and stop. Confirm the runtime with the CoCo team before building on it (`INTEGRATION.md` §10).

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
