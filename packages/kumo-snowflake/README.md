# kumo-snowflake

Reads a Snowflake table into local Parquet that DiskGraph can ingest.

The reader implements `kumo_core.sources.protocol.SourceReader` in
`ReadMode.STAGED` only: it unloads through a temporary stage and downloads the
resulting files over presigned URLs. A temporary stage needs no grant beyond
the `USAGE` and `SELECT` a read-only analyst already holds.

From the monorepo root, run the unit tier with no credentials:

    pytest packages/kumo-snowflake/tests -q

The live tier reads `KUMO_SNOWFLAKE_CREDS` and skips with a stated reason when
that file is absent.
