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
unsupported. Prepared exports use parallel SDK uploads; publication is not
atomic, so consumers must wait for export completion.

## Authentication and resources

Without an identity, requests use Azure's ambient credential chain. An explicit
`ADLSIdentity(tenant_id=..., client_id=..., cloud=...)` uses workload federation
with `AZURE_FEDERATED_TOKEN_FILE`. Secrets, storage keys, and SAS tokens are not
connector arguments. The identity cloud must match the public, US Government,
or China URI endpoint. Probing checks token acquisition only; list or read the
configured source to verify storage permissions and connectivity.

Set a positive `KUMO_ADLS_TRANSFER_WORKERS` before the provider's first import
to lower ADLS transfer concurrency and HTTP pooling. Restart existing processes
after changing it. `KUMO_SOURCE_COPY_WORKERS` lowers the shared file-copy
ceiling. Both controls limit concurrency rather than total memory; SDK and
projected-read buffers need additional headroom.

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
