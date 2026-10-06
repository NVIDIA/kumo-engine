# kumo-s3

Shared Amazon S3 source access for Kumo services and the local `kumoai` workflow.
The provider supplies an authenticated Arrow metadata filesystem and SDK
object downloads to `kumo_core.sources.ObjectStoreSourceReader`. Downloads
retain the resolved credential or session for refresh; core handles source
layout, CSV/Parquet conversion and local output.

The service resolves credentials through its authorization broker. The local
`kumoai` connector resolves its configured identity. Both call the same
provider source-access factory.
Dataplane/L2 transfer and publication remain runtime responsibilities.

`backend.py` and `sink.py` implement the local `SourceBackend` and
`ArtifactSink` contracts using shared object-store machinery in `kumo-core`.
Installing this distribution registers its source, sink and identity plugins.

From the monorepo root, run the unit tests with no credentials:

    pytest packages/kumo-s3/tests -q
