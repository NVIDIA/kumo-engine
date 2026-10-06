# Running Devserver with a Databricks Connector

Start the [local devserver](devserver.md) in a prepared environment:

```bash
UNFENCED=1 NUMS=1 make devserver-local
```

Create a Databricks connector through the service UI or API with its workspace
host, compute IDs, Unity Catalog catalog/schema, and the selected authentication
configuration. The service connector accepts `host`, `cluster_id`,
`warehouse_id`, `catalog`, and `schema_name`; see
[`DatabricksConnector`](../kumo/connector/databricks_connector.py).
Connections use PAT or OAuth credentials; identity-backed connectors require
explicit OAuth credentials.

## Deployment settings are separate from connector configuration

Setting `DATABRICKS_HOST` in `rest/devserver/.env` enables the
`IS_DATABRICKS_DEPLOY` UI/deployment flag. It does not create or configure a
connector. `DATABRICKS_CLUSTER_ID` and `DATABRICKS_WAREHOUSE_ID` environment
variables likewise do not populate persisted connector settings in the current
implementation. Removing `DATABRICKS_HOST` clears the deployment flag after a restart, but
does not remove existing connectors.

For the in-process Python SDK, use the separate
[Databricks provider workflow](../packages/kumo-databricks/README.md#local-workflow).
