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
