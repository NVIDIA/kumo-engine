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

## Testing

Run the unit tier without credentials:

```text
pytest packages/kumo-databricks/tests -q
```

The live tier reads `KUMO_DATABRICKS_CREDS` and skips with a stated reason when
that file is absent.
