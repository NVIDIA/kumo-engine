# kumo-adls

Azure Data Lake Storage Gen2 source and sink provider for `kumo-engine`.
The provider supplies an authenticated Arrow filesystem and conditional SDK
file transfers to the shared object-store machinery in `kumo-core`. Core owns
CSV/Parquet layout, conversion, local staging, and output.

`backend.py` and `sink.py` implement the local `SourceBackend` and
`ArtifactSink` contracts. Installing this distribution registers source, sink,
and identity plugins. Engine imports the provider lazily through its extra.

## Local workflow

```sh
pip install "kumo-engine[adls]"
```

Use a wheel or source checkout containing this provider; this does not establish
availability in an older published release.

```python
from kumo.engine import ADLSConnector, Graph, Table

lake = ADLSConnector(
    root_dir="abfss://curated@companylake.dfs.core.windows.net/orders/",
)
orders = Table.from_source_table(lake["orders.parquet"], primary_key="id")
snapshot = Graph(tables=[orders]).snapshot()
```

Reads use shared `ObjectStoreSourceReader.prepare` in `ReadMode.STREAMED`,
including local file staging. `Graph.snapshot()` passes the staged Parquet to
DiskGraph through the existing ingestion interface; DiskGraph needs no Azure
credentials for those local files.

A rooted connector discovers CSV/Parquet tables and writes predictions,
embeddings, and model bundles beneath its filesystem/prefix. A rootless
connector reads exact canonical `abfss://` URIs; discovery and output are
unavailable. SQL, warehouse-style `ReadMode.STAGED`, and append output are
unsupported. Prepared exports and generated output use native Arrow streams,
matching S3/GCS; consumers must wait for export completion.

## Authentication and resources

Without an identity, requests use Azure's ambient credential chain. An explicit
`ADLSIdentity(tenant_id=..., client_id=..., cloud=...)` uses workload federation
with `AZURE_FEDERATED_TOKEN_FILE`. Secrets, storage keys, and SAS tokens are not
connector arguments. The identity cloud must match the public, US Government,
or China URI endpoint. Native writes require the process's `AZURE_TENANT_ID`
and `AZURE_CLIENT_ID` to match an explicit identity and
`AZURE_FEDERATED_TOKEN_FILE` to name its projected token. Other per-connector
identities remain read-only. Native writes also require `AZURE_AUTHORITY_HOST`
to match the cloud (public is the default). Unsupported configurations fail
explicitly; writes do not switch identity or fall back to SDK uploads.
Probing checks Python token acquisition only; it does not attest native-write
compatibility or storage permissions.

Set a positive `KUMO_ADLS_TRANSFER_WORKERS` before the provider's first import
to lower SDK read concurrency and HTTP pooling. The connection pool and transfer
slots are fixed at import; changing the variable afterward has no effect until
the process restarts. `KUMO_SOURCE_COPY_WORKERS` lowers the shared file-copy
ceiling and its contribution to the ADLS limit is also captured at import.
These controls do not bound native Arrow uploads or their background
buffers. Native writes use Arrow's C++ credentials and upload scheduling;
ambient credential selection can differ from the Python SDK read chain.
Directory exports use the shared eight-file scheduler, while model bundles
upload files sequentially. Reserve memory headroom for native uploads.

See the [storage contract](../../docs/cloud-storage/adls.md)
and [implementation and resource controls](docs/implementation.md) for
authentication scope, consistency, memory estimates, and export behavior.

## Tests

From the repository root, run public tests without cloud credentials:

```sh
python3 scripts/test_public.py --group adls
```

See the [public test guide](../../docs/development/public-tests.md) for setup
and the [test matrix](../../docs/connector-test-matrix.md#reader-groups) for coverage.

Read-only live filesystem checks require an existing HNS directory and ambient
credentials; they do not create or delete cloud paths:

```sh
KUMO_ADLS_LIVE_ROOT='abfss://filesystem@account.dfs.core.windows.net/existing-directory/' \
python -m pytest packages/kumo-adls/tests/live --confcutdir=packages/kumo-adls/tests
```

Live token renewal, RBAC/ACL propagation, sovereign endpoints, and full graph
workloads require separate validation. Offline tests and raw-copy measurements
do not establish end-to-end graph performance or live cloud support.
