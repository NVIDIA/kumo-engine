# Documentation

Repository Markdown is the authored source for engineering guidance. The
website is maintained and published separately from
[kumo-ai/docs](https://github.com/kumo-ai/docs).
Keep one maintained explanation per topic and link to it from agent guidance.

| Need | Start here |
| --- | --- |
| Industry notebooks and standalone examples | [Example collection](../packages/clients/kumo-engine/examples/kumo-example-notebooks/README.md) |
| Product and available source interfaces | [Repository overview](../README.md) |
| System topology and execution boundaries | [Architecture](../ARCHITECTURE.md) |
| Development prerequisites and setup side effects | [Setup](development/setup.md) |
| Local StateDB inspection and reset behavior | [SQLite](development/sqlite.md) |
| Run tests without cloud credentials | [Public tests](development/public-tests.md) |
| Choose a validation tier | [Testing](development/testing.md) |
| Regenerate derived files | [Generated artifacts](development/generated-artifacts.md) |
| UI conventions and shared components | [UI development](development/ui.md) |
| Query and training review | [ML development](development/ml.md) |
| Review and PR requirements | [Review guide](development/review.md) |
| Publish the `kumo-engine` wheel | [Releasing kumo-engine](development/releasing-kumo-engine.md) |
| Use a coding agent | [Agent setup](development/agents.md) |
| Deploy the application or inspect infrastructure examples | [Helm boundary and index](../apps/studio/helm/README.md) |
| Cloud storage integration | [Storage contract](cloud-storage/README.md) |
| Maintain documentation | [Maintenance](maintenance/README.md) |

## Package documentation

Use the [architecture map](../ARCHITECTURE.md) for package roles and these
guides for their interfaces and development workflows.

| Package | Existing entry point |
| --- | --- |
| API schemas | [kumo-api](../packages/contracts/kumo-api/README.md) |
| Service Python client | [kumo-sdk](../packages/clients/kumo-sdk/README.md) |
| In-process Python interface | [kumo-engine](../packages/clients/kumo-engine/README.md) |
| Shared execution runtime | [kumo-core](../packages/runtime/kumo-core/README.md) |
| Query language | [kumo-pql](../packages/contracts/kumo-pql/README.md) and [existing instructions](../packages/contracts/kumo-pql/AGENTS.md) |
| UI component library | [catalog](../packages/ui/kumo-ui-catalog/README.md) and [existing component guide](../packages/ui/kumo-ui-catalog/CLAUDE.md) |
| Native graph engine | [DiskGraph](../packages/data/diskgraph/README.md); maintained directly in this repository |
| Source providers | [S3](../packages/integrations/kumo-s3/README.md), [GCS](../packages/integrations/kumo-gcs/README.md), [ADLS](../packages/integrations/kumo-adls/README.md), [BigQuery](../packages/integrations/kumo-bigquery/README.md), [Databricks](../packages/integrations/kumo-databricks/README.md), [Snowflake](../packages/integrations/kumo-snowflake/README.md) |
