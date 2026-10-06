# kumo-bigquery

Reads a BigQuery table into local Parquet that DiskGraph can ingest.

The reader implements `kumo_core.sources.protocol.SourceReader` in both modes.
`ReadMode.STREAMED` needs no bucket: it opens a Storage Read API session and
consumes every stream the server planned at once, a client and so a connection
per worker. `ReadMode.STAGED` runs `EXPORT DATA` to a configured GCS prefix and
fetches every object whole, so the client checksums it against the crc32c the
bucket holds.

Staged leads where it is available. `probe_staging()` decides that: it exports
one row to the configured prefix and reports ALLOWED, DENIED or UNKNOWN, so a
user with no bucket or no grant gets a streamed read rather than a failure.

From the monorepo root, run the unit tier with no credentials:

    pytest packages/kumo-bigquery/tests -q

The live tier resolves application default credentials and skips with a stated
reason when they are absent.
