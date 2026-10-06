# On-Demand SDK E2E Harnesses

This directory contains manually run dataset generators and SDK harnesses for
fast end-to-end checks. They are not pytest tests and are not a replacement for
the broader `kumo-tests` suite.

## Why This Exists

The existing SDK E2E coverage depends on S3-backed `kumo-tests` fixtures. That
is expensive for agent-driven debugging and local connector work because an
Azure, GCP, or local Compose environment has to reach across clouds, carry the
right object-store credentials, and reuse shared fixture state.

These harnesses create deterministic datasets on demand, upload them to storage
that is local to the environment under test, and then exercise realistic SDK
flows against a Kumo API. The main loop is:

1. Generate small smoke data in the same cloud as the test environment.
2. Register SDK tables, graphs, and predictive queries against local Compose or
   another target API.
3. Optionally run training and batch prediction, then verify prediction and
   embedding outputs were written.

## Layout

- `datasets/generate.py` builds deterministic graph datasets and uploads them.
- `harnesses/success/` contains SDK scenarios that should validate, train, and
  batch predict successfully.
- `harnesses/expected_failure/` contains SDK scenarios that should be rejected
  cleanly.
- `harnesses/run_success_scenarios.py` and
  `harnesses/run_expected_failure_scenarios.py` run the scenario groups.
- `lib/config.py` and `lib/storage.py` handle cloud-local storage defaults and
  upload/download helpers.

## Storage Selection

The current metadata-selected defaults refer to internal test storage. Set
an explicit operator-controlled object-store root (`s3://`, `gs://`, or
`abfss://`) for service harness runs:

```bash
export KUMO_E2E_STORAGE_ROOT=<object-store-root>
```

The dataset generator also accepts local paths, but the service harness currently
routes those paths to an S3 connector, which rejects them. Use local paths only
for generation, not for the SDK scenarios below.

The root must be writable by the generator and readable by the Kumo workers that
will load the SDK connector tables. Provider-specific overrides are also
supported: `KUMO_E2E_ADLS_STORAGE_ROOT`, `KUMO_E2E_S3_STORAGE_ROOT`, and
`KUMO_E2E_GCS_STORAGE_ROOT`.

Generated data is written under:

```text
<storage-root>/<suite-name>/<dataset>/<dataset-version>/<variant>/
```

Batch prediction outputs default to `<storage-root>/outputs`.

## Generate Data

Run commands from the repo root with the normal Kumo dev environment active.
Use one `dataset-version` value for both generation and harness execution.

Generate every dataset in both supported variants:

```bash
DATASET_VERSION=$(date -u +%Y%m%dT%H%M%SZ)

python -m tools.e2e.datasets.generate \
  --dataset all \
  --variant all \
  --scale smoke \
  --dataset-version "${DATASET_VERSION}" \
  --overwrite
```

Useful options:

- `--scale smoke` creates small data for local and agent loops.
- `--scale standard` creates larger data for heavier manual checks.
- `--variant parquet_typed` writes typed Parquet tables.
- `--variant string_encoded` writes CSV plus all-string Parquet mirrors.
- `--storage-root` overrides `KUMO_E2E_STORAGE_ROOT` for a single run.
- `--local-build-root` controls where temporary files are assembled before
  upload.

## Run SDK Scenarios

For local Docker Compose, use the nginx gateway. It routes SDK traffic through
the same public entrypoint shape as deployed environments:

```bash
python -m tools.e2e.harnesses.run_success_scenarios \
  --url http://localhost:8888/api \
  --api-key test:DISABLED \
  --dataset-version "${DATASET_VERSION}" \
  --variant parquet_typed
```

Expected-failure scenarios intentionally return non-zero from their individual
modules. The group runner succeeds only when each scenario fails:

```bash
python -m tools.e2e.harnesses.run_expected_failure_scenarios \
  --url http://localhost:8888/api \
  --api-key test:DISABLED \
  --dataset-version "${DATASET_VERSION}" \
  --variant parquet_typed
```

Run one scenario directly when iterating on a narrower path:

```bash
python -m tools.e2e.harnesses.success.customer_churn_binary_implicit \
  --url http://localhost:8888/api \
  --api-key test:DISABLED \
  --dataset-version "${DATASET_VERSION}" \
  --variant parquet_typed \
  --stop-after validate
```

`--stop-after` controls how far a success scenario goes:

- `save`: initialize the SDK, infer metadata, save tables, graph, and pquery.
- `validate`: also suggest a model plan. This is the default.
- `train`: run training and stop after a successful training job.
- `bp`: run training plus batch prediction and verify output artifacts.

Use `--training-job-id <id>` with `--stop-after bp` to reuse an existing
training job for batch prediction checks.

For `string_encoded` data, choose whether SDK table sources read CSV or the
all-string Parquet mirror:

```bash
python -m tools.e2e.harnesses.run_success_scenarios \
  --dataset-version "${DATASET_VERSION}" \
  --variant string_encoded \
  --string-physical-format csv
```

## Scenario Coverage

Success scenarios cover:

- binary classification on an omnichannel customer graph
- multiclass classification on the same customer graph
- customer spend regression
- marketplace static link prediction and ranking
- temporal account balance regression
- reserved-name output collision handling

Expected-failure scenarios cover:

- regression over a categorical target
- graph edges that point at a missing destination table
- unsupported struct feature handling
- invalid implicit target distribution with no null holdout rows

The generated datasets intentionally include duplicates, nulls, dangling
foreign keys, schema edge cases, reserved names, strings with unusual
characters, temporal windows, and typed-storage variants. This gives SDK and
backend changes a quick way to exercise realistic bad-data handling without
depending on shared S3 fixtures.

## StateDB Final Gate Manifest

The StateDB final gate is ordered so heavyweight and mutating checks
do not mask lower-level failures:

1. Black-box REST tests, parallel when pytest-xdist is available.
2. SQLite SDK API-key train plus batch-prediction e2e.
3. Postgres SDK API-key train plus batch-prediction e2e.
4. Migrated impacted unit/static tests.
5. UI Playwright last.

Print the on-demand manifest without running the gates:

```bash
python -m tools.e2e.final_gate
python -m tools.e2e.final_gate --format shell
python -m tools.e2e.final_gate --phase sdk-sqlite
```

Run a selected phase only against a test environment you control:

```bash
python -m tools.e2e.final_gate --phase unit-static --execute
```

The SDK e2e entries use
`tools.e2e.harnesses.success.customer_churn_binary_implicit`. Use the manifest
to inspect command arguments and gate order before execution.

Useful environment overrides:

- `KUMO_FINAL_GATE_BLACK_BOX_WORKERS`, unset by default. When unset, the
  black-box gate uses `-n auto` if pytest-xdist is installed and otherwise
  falls back to serial pytest. Set this after installing pytest-xdist to force
  a specific worker count such as `auto` or `4`.
- `KUMO_FINAL_GATE_DATASET_VERSION`, or storage-specific
  `KUMO_FINAL_GATE_SQLITE_DATASET_VERSION` /
  `KUMO_FINAL_GATE_POSTGRES_DATASET_VERSION`
- `KUMO_FINAL_GATE_SQLITE_URL`, `KUMO_FINAL_GATE_SQLITE_API_KEY`,
  `KUMO_FINAL_GATE_SQLITE_STORAGE_ROOT`,
  `KUMO_FINAL_GATE_SQLITE_OUTPUT_ROOT`
- `KUMO_FINAL_GATE_POSTGRES_URL`, `KUMO_FINAL_GATE_POSTGRES_API_KEY`,
  `KUMO_FINAL_GATE_POSTGRES_STORAGE_ROOT`,
  `KUMO_FINAL_GATE_POSTGRES_OUTPUT_ROOT`
- `KUMO_FINAL_GATE_POSTGRES_POSTGRES_MODE`, default `bundled`

## Running From a Compose Network Namespace

When the API is only reachable from inside the Docker Compose network, run the
harness from a one-off container that shares the nginx container network. Adjust
the image, container name, and mounted SDK/API checkout paths for your local
project:

```bash
env UV_CACHE_DIR=/tmp/uv-cache \
  uv pip install --python .venv/bin/python \
  --target /tmp/kumo-e2e-sdk-deps rich

docker run --rm --network container:<nginx-container> \
  -v /tmp/kumo-e2e-sdk-deps:/sdk-deps:ro \
  -v "$PWD":/workspace \
  -v "$PWD/packages/kumo-sdk":/kumo-sdk \
  -v "$PWD/packages/kumo-api":/kumo-api \
  -w /workspace \
  -e PYTHONPATH=/sdk-deps:/workspace:/kumo-sdk:/kumo-api:/opt/app \
  -e KUMO_E2E_STORAGE_ROOT="${KUMO_E2E_STORAGE_ROOT}" \
  <backend-image> \
  python -m tools.e2e.harnesses.run_success_scenarios \
    --url http://127.0.0.1:8888/api \
    --api-key test:DISABLED \
    --dataset-version "${DATASET_VERSION}" \
    --variant parquet_typed
```
