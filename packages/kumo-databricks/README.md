# kumo-databricks

`kumo-databricks` is Kumo's Databricks provider package. It contains the
Databricks source reader, local source/sink backends, identity lifecycle,
Unity Catalog staging, CloudFetch streaming, and Delta-table output support.

It does not define a second Kumo workflow API. End-to-end ingest, training,
and prediction are owned by `kumo-engine`. The service connector uses
`DatabricksSourceReader`; the service SDK sends connector configuration to
the service.

## Provider interfaces

| Interface | Purpose |
| --- | --- |
| `DatabricksSourceReader` | Low-level `kumo_core` reader used by the service |
| `DatabricksSourceBackend` | `kumoapi.plugin.SourceBackend` for local workflows |
| `DatabricksSinkBackend` | Table and file output to Delta tables |
| `kumo_databricks.session` | Identity probing and connection creation |

The local SDK constructs the provider backends directly. The provider imports
neither SDK. Backend/session exports are lazy; `DatabricksSourceReader` is
imported eagerly.

## Reading data

The reader implements `kumo_core.sources.protocol.SourceReader` in both modes:

- `ReadMode.STREAMED` reads through CloudFetch and needs only `CAN USE` on the
  warehouse plus `SELECT` on the table.
- `ReadMode.STAGED` unloads Parquet through a configured Unity Catalog volume.
  A denied staged read falls back to streaming unless the caller requires
  staged mode.

Neither mode gets a row count from the warehouse. CloudFetch downloads use the
driver's own pool, sized by `max_download_threads` when the connection opens.

## Local workflow

Install it through the local SDK extra:

```text
pip install "kumo-engine[databricks]"
```

The public workflow remains the `kumo.engine` workflow:

```python
from kumo.engine import DatabricksConnector, DatabricksIdentity, Graph, Table

identity = DatabricksIdentity(
    host="https://workspace.cloud.databricks.com",
    client_id="service-principal-id",
    client_secret="...",
)
databricks = DatabricksConnector(
    catalog="main",
    schema_name="analytics",
    warehouse_id="warehouse-id",
    identity=identity,
)
events = Table.from_source_table(
    databricks["events"],
    primary_key="id",
)
snapshot = Graph(tables=[events]).snapshot()
```

From the repository root, run the public tests without cloud credentials:

```sh
python3 scripts/test_public.py --group databricks
```

See the [public test guide](../../docs/development/public-tests.md) for Docker
prerequisites, Python-only setup, and results. The
[test matrix](../../docs/connector-test-matrix.md#reader-groups) lists coverage
and limits. These tests use generated data and fake clients, without a warehouse
or staging volume.

Run the complete non-live tier from `packages/kumo-databricks` in a prepared
API/core environment with this package's runtime and `[test]` dependencies installed.
Use the versions in [`pyproject.toml`](pyproject.toml), plus pandas and Arrow:

```sh
KUMO_DATABRICKS_LIVE=0 python -m pytest tests -q -rs
```

The live-activation tests can also run alone from the package directory.
They need Python, pytest, and this checkout's `sources.live_opt_in` helper,
without installed provider packages or credentials:

```sh
python -m pytest tests/test_live_opt_in.py -q
```

The live module skips before credential-file access and API/core/provider
imports unless `KUMO_DATABRICKS_LIVE` is exactly `1`. An enabled run requires an
explicit `KUMO_DATABRICKS_CREDS` path to a readable UTF-8 JSON object containing
nonempty string fields `host`, `client_id`, `client_secret` and `warehouse_id`.
There is no default credential path. Missing or invalid configuration fails
collection; connection and authentication failures during execution propagate
instead of skipping.

An authorized live run from the package directory uses:

```sh
KUMO_DATABRICKS_LIVE=1 \
KUMO_DATABRICKS_CREDS=/path/to/test-credentials.json \
python -m pytest tests/live/test_live_conformance.py -q -rs
```

This command also needs the matching `sources.conformance` test helper.
Prepare the warehouse, Delta history, staging resources, and permissions listed
in the [cloud test inventory](../../docs/credentialed-test-inventory.md#inputs-to-separate-from-assertions).
