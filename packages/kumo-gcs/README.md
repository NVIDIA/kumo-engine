# kumo-gcs

Shared Google Cloud Storage source access for Kumo services and the local `kumoai` workflow.
The provider supplies an authenticated Arrow filesystem and the shared GCS
downloader to `kumo_core.sources.ObjectStoreSourceReader`. Ordinary objects
use native transfers with token renewal between files; gzip and billed transfers
retain the SDK route. Core handles layout, CSV/Parquet conversion and local output.

The service resolves credentials through its authorization broker. The local
`kumoai` connector resolves its configured identity. Both call the same
provider source-access factory.
Dataplane/L2 transfer and publication remain runtime responsibilities.

`backend.py` and `sink.py` implement the local `SourceBackend` and
`ArtifactSink` contracts using shared object-store machinery in `kumo-core`.
Installing this distribution registers its source, sink and identity plugins.

From the repository root, run the public tests without cloud credentials:

```bash
python3 scripts/test_public.py --group gcs
```

See the [public test guide](../../docs/development/public-tests.md) for setup
and the [test matrix](../../docs/connector-test-matrix.md#reader-groups) for coverage.
