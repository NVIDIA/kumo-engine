# Real-provider tests: setup and cleanup

This inventory covers six live modules for S3, Snowflake, Databricks, and
BigQuery, including the BigQuery sink. It records source inspection and local
collection checks, not live-provider results. The
[test matrix](connector-test-matrix.md#real-cloud-tests-on-kumo-resources)
covers the other hosted and provider suites.

## Current entry points and activation

Run these commands from the stated directory with provider, core, and API test
dependencies installed. They need an authorized cloud account and the resources
listed below. A skipped module does not mean that a live suite passed.

| Suite / working directory | Current selection | Activation and credential timing |
| --- | --- | --- |
| [S3 binding](../packages/kumo-s3/tests/live/test_live_conformance.py), `packages/kumo-s3` | `KUMO_S3_LIVE=1 python -m pytest tests/live/test_live_conformance.py -rs` | Module skips unless the flag is nonempty; ambient boto3 credentials and region resolution are used when tests run |
| [S3 reader](../packages/kumo-core/tests/sources/live/test_live_object_store.py), `packages/kumo-core` | `KUMO_S3_LIVE=1 python -m pytest tests/sources/live/test_live_object_store.py -rs` | Same flag; native Arrow S3 filesystem resolves credentials at test execution |
| [Snowflake](../packages/kumo-snowflake/tests/live/test_live_conformance.py), `packages/kumo-snowflake` | `KUMO_SNOWFLAKE_LIVE=1 python -m pytest tests/live/test_live_conformance.py -rs` with explicit `KUMO_SNOWFLAKE_CREDS` | Skips before credential-file access or API/core/provider imports unless the flag is exactly `1`. Enabled collection reads and validates the explicit JSON file; missing or invalid configuration fails rather than skips |
| [Databricks](../packages/kumo-databricks/tests/live/test_live_conformance.py), `packages/kumo-databricks` | `KUMO_DATABRICKS_LIVE=1 python -m pytest tests/live/test_live_conformance.py -rs` with explicit `KUMO_DATABRICKS_CREDS` | Skips before credential-file access or API/core/provider imports unless the flag is exactly `1`. Enabled collection reads and validates the explicit JSON file; missing or invalid configuration fails rather than skips |
| [BigQuery](../packages/kumo-bigquery/tests/live/test_live_conformance.py), `packages/kumo-bigquery` | `KUMO_BIGQUERY_LIVE=1 python -m pytest tests/live/test_live_conformance.py -rs` with explicit project/staging environment variables | Skips before provider imports or ADC unless the flag is exactly `1`. Enabled runs require nonempty `KUMO_BIGQUERY_PROJECT` and `KUMO_BIGQUERY_STAGING` before resolving ADC; missing configuration or failed ADC is a collection error |
| [BigQuery sink](../packages/kumo-bigquery/tests/live/test_live_sink.py), `packages/kumo-bigquery` | `KUMO_BIGQUERY_LIVE=1 python -m pytest tests/live/test_live_sink.py -rs` with explicit project/staging environment variables | Same opt-in, configuration and ADC boundary as the BigQuery reader |

The existing [`testing.yml` packages job](../.github/workflows/testing.yml)
discovers package tests but does not set the warehouse live flags. All four
warehouse live modules require explicit opt-in; credential presence alone
does not activate them. The public runner discovers group directories and
excludes `live/` before import. The package READMEs for
[Snowflake](../packages/kumo-snowflake/README.md),
[Databricks](../packages/kumo-databricks/README.md) and
[BigQuery](../packages/kumo-bigquery/README.md) document both execution modes
and pytest-only collection-boundary regressions. Enabled provider connection
or authentication failures remain failures, not skips.

## Inputs to separate from assertions

Internal resource values stay in the linked source pending remediation; this
document names the required fields rather than reproducing account identifiers
or personal credential paths. The reader suites do not bootstrap their full
source datasets. The BigQuery sink generates local input data and creates its
destination dataset and tables, but needs a prepared project and staging bucket.

| Provider | Inputs available today | Fixed assumptions that need an explicit runtime/fixture contract |
| --- | --- | --- |
| S3 | Live flag and ambient AWS credential chain | Bucket, source prefix, region, table/object names, schemas and independently established row counts. The reader also asserts an exact table listing containing output tables. A bucket being named public does not make its current signed access path credential-free |
| Snowflake | `KUMO_SNOWFLAKE_LIVE=1` and explicit `KUMO_SNOWFLAKE_CREDS`; JSON `account`, `user`, `password` must be nonempty strings; no default file path | Role, warehouse, database, schema, source table, ordinary view, materialized view, five-column schema and expected row count; source/view provisioning and permissions are external prerequisites |
| Databricks | `KUMO_DATABRICKS_LIVE=1` and explicit `KUMO_DATABRICKS_CREDS`; JSON `host`, `client_id`, `client_secret`, `warehouse_id` must be nonempty strings; no default file path | Source catalog/schema/table, writable staging catalog/schema, five-column schema and expected row count; permission to create/revoke tokens and create/delete volumes, views and tables |
| BigQuery | `KUMO_BIGQUERY_LIVE=1`, ADC, required `KUMO_BIGQUERY_PROJECT` and `KUMO_BIGQUERY_STAGING`; no project/bucket defaults | Reader source project/dataset/table, projected schema, expected rows and compatible locations. Execution needs query/Read API access, dataset/view creation and deletion, and scoped object list/read/write/delete rights. The sink also needs table creation/update and Storage Write API access, a US destination dataset, and denied-write behavior on its fixed public dataset/bucket probes |

Moving coordinates into configuration must preserve the fixture's behavior:
Snowflake needs both view kinds; Databricks needs a Delta table with readable
version/commit history, a view exercising the hour-window fallback, staged and
streamed reads, and its decimal case. BigQuery needs table, view and query paths. A
smaller generated fixture needs independent expected results, not assertions
weakened to accept whatever the reader returns.

BigQuery currently reads `bigquery-public-data.usa_names.usa_1910_2013`, expects
5,552,452 rows and projects `state:string` and `number:int64`. Its projection
test also requires scanned bytes below 75% of the source table's size. A
replacement fixture must preserve meaningful unselected columns as well as the
row/schema oracle; a two-column-only fixture would lose that coverage.
The sink adds staged and streamed writes, overwrite/append, schema changes,
failed-write preservation, timestamp and embedding values, partitioned tables,
and permission preflight checks using generated Parquet and DataFrames.

## Resources and cleanup before extraction

| Provider | Current lifecycle | Readiness gap |
| --- | --- | --- |
| S3 | Selected live modules read existing source objects and write pytest-managed local outputs; they do not create cloud fixtures | Record dataset provenance, fixture/version owner and reproducible bootstrap. The exact bucket listing can change independently of source code |
| Snowflake | Reader creates a temporary encrypted stage with per-attempt prefixes; normal reads clear prefixes and the module closes its connection | Source table/views are not provisioned. Prefix cleanup is best effort and stage lifetime depends on the session; record timeout/cancellation behavior and recovery ownership |
| Databricks | Suite creates a short-lived token and managed volume; individual cases create a view/table. Teardown closes the connection, deletes the volume and revokes the token | Cleanup actions are sequential, so an earlier exception prevents later cleanup. Failed DROP can also skip cursor close. No durable resource manifest or hard-cancellation cleanup owner is defined |
| BigQuery reader | Module fixture creates a uniquely named dataset/view and uses a run-specific staging prefix; teardown deletes the dataset and staged objects | Dataset and view creation happen before the fixture's `try/finally`; partial setup can leak a dataset. A dataset-delete failure prevents object cleanup. Record resource handles as created and define cancellation/orphan cleanup |
| BigQuery sink | Module fixture creates a local identity and a uniquely named dataset; tests create destination tables and use a run-specific staging prefix. Teardown deletes the dataset and staged objects | Dataset creation precedes `try/finally`; a dataset-delete failure prevents object cleanup. The local identity is not removed. Define cleanup after partial setup and cancellation, and recovery for leftover cloud resources |

These are observed control-flow gaps, not provider reproductions. A protected
run must verify successful cleanup and injected partial failures before an
extracted suite is treated as operationally maintained.

## Dependencies

All six modules need API/core source or installed packages, pytest and Arrow.
The warehouse suites additionally use provider packages and their declared
runtime dependencies: [Snowflake](../packages/kumo-snowflake/pyproject.toml),
[Databricks](../packages/kumo-databricks/pyproject.toml), and
[BigQuery](../packages/kumo-bigquery/pyproject.toml). The S3 binding also needs
[boto3](../packages/kumo-s3/pyproject.toml). Normal core installation still has
the broader dependency closure recorded in the public guide; do not assume
these commands work in the narrow public CSV/S3 environment.

The pytest-only Snowflake and Databricks activation regressions use this source
checkout's `sources.live_opt_in` helper; they do not need installed API/core or
provider packages. The reader conformance suites share
[`conformance.py`](../packages/kumo-core/tests/sources/conformance.py)
from the core test tree. Core uses `tests.sources.conformance`; warehouse
suites use `sources.conformance`. An external runner needs an explicit way to obtain
that helper at a matching revision; installing the production wheel alone does
not provide this test-tree import. Consume a reviewed source checkout or
artifact without maintaining a second full monorepo.

See the [test matrix](connector-test-matrix.md) for public coverage.
[Issue #31132](https://github.com/kumo-ai/kumo/issues/31132) tracks the remaining
setup, publication, ownership, and private-repository decisions.
