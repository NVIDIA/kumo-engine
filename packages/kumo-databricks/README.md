# kumo-databricks

`kumo-databricks` is Kumo's Databricks provider package. It contains the
Databricks source reader, local source/sink backends, identity lifecycle,
Unity Catalog staging, CloudFetch streaming, Delta-table output, and Unity
Catalog model registry support.

It does not define a second Kumo workflow API. End-to-end ingest, training,
and prediction are owned by `kumo-engine`. The service connector uses
`DatabricksSourceReader`; the service SDK sends connector configuration to
the service.

## Provider interfaces

| Interface | Purpose |
| --- | --- |
| `DatabricksSourceReader` | Low-level `kumo_core` reader used by the service |
| `DatabricksSourceBackend` | `kumoapi.plugin.SourceBackend` for local workflows |
| `DatabricksSinkBackend` | Delta-table output and model bundle publish/fetch/list |
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

## Model registry

The sink publishes Kumo bundles directly to Unity Catalog using the connector's
WorkspaceClient. No MLflow installation or tracking experiment is required.
Like the Snowflake integration, the registered version carries artifacts;
prediction requires reloading the bundle into Kumo and supplying a graph.
These bundles do not provide a Databricks Model Serving flavor.

Use the existing local training export API:

```python
from kumo.engine import ModelOutputConfig, ModelRegistryLocation, Trainer

result = trainer.fit(
    training_table,
    snapshot=snapshot,
    export_config=ModelOutputConfig(
        databricks, location=ModelRegistryLocation('CHURN_GNN')),
)
published = result.published_model
restored = Trainer.load_from_connector(
    databricks,
    location=ModelRegistryLocation(published.name, version=published.version),
)
models = databricks.sink_backend.list_models(databricks)
```

Model names resolve within the connector's catalog and schema; a qualified
name must match both. Publishing requires an unset `version`: Databricks
assigns numeric versions, returned in `PublishedModel` with a
`models:/catalog.schema.model/version` URI. Fetching without a version selects
the latest ready Kumo version. Listing follows pagination and excludes other
models and versions that are not ready.

A single `kumo_engine=1` tag identifies engine bundles. Publishing uploads the
original bundle unchanged. Fetching reads training metadata from
`training-spec.json` and metrics from `metrics.json`; listing returns model
identity with empty metadata and metrics. Additional caller-supplied metadata
is not persisted.
The connector identity needs the
[Unity Catalog model privileges](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/).

Publishing and fetching verify bundle integrity. A failed download or local
replacement preserves the previous destination. Hosted service registry export
and import remain unavailable.

This integration requires workspace support for native UC create/finalize
endpoints and the SDK Files API `/Models` route. There is no legacy endpoint
fallback. Local HTTP tests cover publish/list/fetch and credential isolation;
live checks verified these endpoints on one workspace.

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
