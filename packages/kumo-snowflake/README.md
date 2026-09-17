# kumo-snowflake

Reads a Snowflake table into local Parquet that DiskGraph can ingest.

The reader implements `kumo_core.sources.protocol.SourceReader` in
`ReadMode.STAGED` only: it unloads through a temporary stage and downloads the
resulting files over presigned URLs. A temporary stage needs no grant beyond
the `USAGE` and `SELECT` a read-only analyst already holds.

From the repository root, run the public tests without cloud credentials:

```sh
python3 scripts/test_public.py --group snowflake
```

See the [public test guide](../../docs/development/public-tests.md) for Docker
prerequisites, Python-only setup, and results. The
[test matrix](../../docs/connector-test-matrix.md#reader-groups) lists coverage
and limits. These tests use fake SQL and HTTP responses. Snowpark supports the
backend tests.

Run the complete non-live tier from `packages/kumo-snowflake` in a prepared
API/core environment with this package's runtime and `[test,sdk]` dependencies
installed:

```sh
KUMO_SNOWFLAKE_LIVE=0 python -m pytest tests -q -rs
```

This suite includes Snowpark session/sink imports and a native-ingestion case
that needs working DiskGraph.

The public profile's live-activation tests can also run alone from this directory.
They need Python, pytest, and this checkout's `sources.live_opt_in` helper,
without installed provider packages or credentials:

```sh
python -m pytest tests/test_live_opt_in.py -q
```

The live module skips before credential-file access and API/core/provider
imports unless `KUMO_SNOWFLAKE_LIVE` is exactly `1`. An enabled run requires an
explicit `KUMO_SNOWFLAKE_CREDS` path to a readable UTF-8 JSON object containing
nonempty string fields `account`, `user` and `password`. There is no default
credential path. Missing or invalid configuration fails collection; connection
and authentication failures during execution propagate instead of skipping.

An authorized live run from the package directory uses:

```sh
KUMO_SNOWFLAKE_LIVE=1 \
KUMO_SNOWFLAKE_CREDS=/path/to/test-credentials.json \
python -m pytest tests/live/test_live_conformance.py -q -rs
```

This command needs API/core, provider runtime/test dependencies, and the matching
`sources.conformance` test helper. Prepare the role, warehouse, table, and views
listed in the [cloud test inventory](../../docs/credentialed-test-inventory.md#inputs-to-separate-from-assertions).

## Explicit Snowflake ML Job submission

`submit_ml_job` is a thin execution boundary, not a Kumo backend. The caller
writes the complete function that runs remotely and chooses the training or
prediction API used inside that function.

```python
import kumo_snowflake
from kumoapi.plugin import PublishedModel


def train_and_publish() -> PublishedModel:
    # Construct the runtime API and Registry connector inside the job.
    # Train, publish through that API, and return its PublishedModel.
    ...


def restore_and_predict(model: PublishedModel):
    # Construct the runtime API inside this fresh job, restore model through
    # that API, then run prediction.
    ...


training_job = kumo_snowflake.submit_ml_job(
    train_and_publish,
    compute_pool="KUMO_GPU_POOL",
    stage_name="@DB.SCHEMA.KUMO_JOBS",
)
model = training_job.attach().raise_for_failure().value
assert isinstance(model, PublishedModel)

prediction_job = kumo_snowflake.submit_ml_job(
    restore_and_predict,
    compute_pool="KUMO_GPU_POOL",
    stage_name="@DB.SCHEMA.KUMO_JOBS",
    args=[model],
)
prediction_job.attach().raise_for_failure()
```

The helper preflights the GPU pool and internal stage, discovers the installed
Kumo fat wheel, verifies its bundle manifest and declared installation
requirements, stages it by content digest, and asks Snowflake ML Jobs to
install exactly that wheel. Normal `Requires-Dist` dependencies are resolved
by the remote environment; pass `artifact_repositories=` when they must come
from a Snowflake Artifact Repository. The helper does not impose a Container
Runtime version. Pass `runtime_environment=` when you want to select one;
compatible runtime versions are maintained outside this package.

Function dispatch uses Snowflake ML Jobs' supported CloudPickle mechanism, but
Kumo deliberately accepts only a top-level module or notebook function plus
JSON-safe `args=`/`kwargs=` or an existing `PublishedModel`. Never capture or
pass a Snowpark session, credential, lock, open file, connector, graph, or
runtime client. Construct those inside the submitted function, and keep
application data in durable, importable locations.

`PublishedModel` is the only model handoff contract owned by this package.
`kumo-snowflake` serializes and validates that shared `kumoapi` value across
the ML Job boundary; it does not define how a training SDK publishes a model
or how a prediction SDK restores it. Those operations stay with the runtime
API and its existing Registry connector implementation.

The supported terminal value is deliberately narrow: JSON values, dataclasses
containing JSON values, paths, or `PublishedModel`. Models and large outputs
should be published to durable storage and returned by reference.
`job.result()` returns a `SnowflakeMLJobResult` whose `value` preserves that
return value. When the top-level return value is a `PublishedModel`, the same
value is also exposed as `result.published_model` for convenient discovery.

Native Kumo training and prediction progress reaches the durable ML Job
progress sidecar through the shared `kumo_core.runtime_progress` observer.
The submission layer does not import or freeze either SDK's public API.
A submitted function may run at most one training operation and at most one
prediction operation, sequentially. Training followed by prediction is
supported; repeated or overlapping operations of either kind fail the durable
job result.

Handles are not thread-safe. Serialize calls on one handle. Independent jobs
may be observed concurrently when callers supply distinct Snowpark sessions.
Passing the same explicit or active `session` to multiple handles borrows and
shares that session, so those calls must also be serialized. Job handles never
close borrowed sessions. After a notebook restart, reconstruct an observer
from durable stage metadata:

```python
job = kumo_snowflake.get_ml_job(
    service_id,
    stage_name="@DB.SCHEMA.KUMO_JOBS",
)
```
