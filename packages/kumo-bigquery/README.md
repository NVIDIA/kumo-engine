# kumo-bigquery

Reads a BigQuery table into local Parquet that DiskGraph can ingest, and
writes prediction output back to BigQuery.

The reader implements `kumo_core.sources.protocol.SourceReader` in both modes.
`ReadMode.STREAMED` needs no bucket: it opens a Storage Read API session and
consumes every stream the server planned at once, a client and so a connection
per worker. `ReadMode.STAGED` runs `EXPORT DATA` to a configured GCS prefix and
fetches every object whole, so the client checksums it against the crc32c the
bucket holds.

Staged leads where it is available. `probe_staging()` decides that: it exports
one row to the configured prefix and reports ALLOWED, DENIED or UNKNOWN, so a
user with no bucket or no grant gets a streamed read rather than a failure.

`kumo_bigquery.write` writes local Parquet output into a table, for hosted
Kumo and `BigQuerySinkBackend` alike, as reads in reverse. `STAGED` uploads to
a unique prefix and runs one load job over its wildcard (hosted uses its
transfer bucket, Engine its `staging_uri`). `STREAMED` appends through
Storage Write API pending streams, one connection per worker, and commits
them in one call; resent requests carry their offsets, so a retry never
duplicates rows. Shards are first conformed to one nullable schema with
microsecond UTC timestamps, since a multi-file load autodetects from a single
file and nanosecond timestamps load as integers. Modes are offered
staged-first after a probe, as for reads.

From this package directory, in an environment with the package's test/runtime
dependencies and the matching core/API source available, run the unit tier:

    KUMO_BIGQUERY_LIVE=0 python -m pytest tests -q

From the repository root, run the public tests without cloud credentials:

```sh
python3 scripts/test_public.py --group bigquery
```

See the [public test guide](../../docs/development/public-tests.md) for setup.

Both live modules skip before credential discovery unless
`KUMO_BIGQUERY_LIVE=1`. Existing application default credentials (ADC) alone
cannot activate it. The collection-boundary regression itself only needs pytest:

    python -m pytest tests/test_live_opt_in.py -q

To run the live tier against authorized resources, explicitly supply both the
billing project and a staging GCS bucket or prefix; neither has a default:

```sh
KUMO_BIGQUERY_LIVE=1 \
KUMO_BIGQUERY_PROJECT=your-test-project \
KUMO_BIGQUERY_STAGING=gs://your-test-bucket/connector-tests \
python -m pytest tests/live/test_live_conformance.py tests/live/test_live_sink.py -rs
```

Provision ADC before running this command. Once enabled, missing configuration
or an ADC discovery error fails collection instead of skipping. The principal
needs query and Storage Read API access to `bigquery-public-data.usa_names.usa_1910_2013`,
permission to create/delete a temporary dataset and view in the billing
project, and list/read/write/delete access to the staging prefix. Use compatible
dataset/bucket locations. The suite creates billable jobs and temporary cloud
resources; it does not provision the project, bucket or IAM permissions.

The live table/view/query assertions and existing cleanup behavior are
unchanged. Partial-setup and teardown cleanup gaps remain tracked in the
[credentialed-suite inventory](../../docs/credentialed-test-inventory.md).
