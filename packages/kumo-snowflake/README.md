# kumo-snowflake

Reads a Snowflake table into local Parquet that DiskGraph can ingest.

The reader implements `kumo_core.sources.protocol.SourceReader` in
`ReadMode.STAGED` only: it unloads through a temporary stage and downloads the
resulting files over presigned URLs. A temporary stage needs no grant beyond
the `USAGE` and `SELECT` a read-only analyst already holds.

`SnowflakeSinkBackend` also holds files in a named internal stage
(`@DB.SCHEMA.STAGE/prefix`). Output configs take stage paths, such as
`ModelOutputConfig(snowflake_connector, path="@ML.ARTIFACTS.KUMO/models/m1")`
or `PredictionOutputConfig(snowflake_connector,
output_paths={"predictions": "@ML.ARTIFACTS.KUMO/scores/"})`, and
`Trainer.load_from_connector` reads a model back from one. For an MLOps
artifact root on a stage, see `packages/kumo-engine/docs/mlops.md`. File names may
use only `A-Z a-z 0-9 _ $ = + . , / -`.

A file of up to 4 GiB is one staged object. A larger one is stored as numbered
parts under `<name>.chunks/`, which only kumo reads back. To load exported
prediction Parquet into a table, skip the marker and manifest files:

```sql
COPY INTO SCORES FROM @ML.ARTIFACTS.KUMO/scores/
  PATTERN = '.*[.]parquet' FILE_FORMAT = (TYPE = PARQUET)
  MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;
```

The role needs `USAGE` on the database and schema and `READ` and `WRITE` on the
stage. No warehouse runs: `PUT`, `LIST`, `GET`, `REMOVE` and
`GET_PRESIGNED_URL` complete while the session's warehouse stays suspended.
Create the stage with server-side encryption so that reads fetch byte ranges
over presigned URLs:

```sql
CREATE STAGE ML.ARTIFACTS.KUMO ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
```

A stage with the default `SNOWFLAKE_FULL` (client-side) encryption is written
to, but reading from it raises: its `LIST` reports encrypted sizes (padded to
16 bytes), so files cannot be read back exactly. With a role that cannot write
the probe file, each file is downloaded once with `GET` into the spool
directory instead, and a warning says so.

`publish_model` registers a bundle in the Snowflake Model Registry. When an
installed host registers a model builder under the
`kumo_snowflake.registry_models` entry-point group, as kumo-engine does, the
version is a runnable custom model that scores in Snowpark Container Services
(`ModelVersion.run_batch`); otherwise it carries the bundle and refuses to
score. A bundle whose `kumo_bundle_id` is already a Kumo version of the model
returns that version rather than logging a new one. See "Batch inference in
Snowflake" in `packages/kumo-engine/docs/mlops.md`.

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
