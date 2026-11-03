# Connector tests: coverage and setup

Use the [test guide](development/public-tests.md) for installation, version choice, and running
without network access. Use code and tests from the same checkout.

## Reader groups

These groups run real Kumo code with generated data and fake cloud responses.
They need no cloud accounts or Kumo credentials.

Run `python3 scripts/test_public.py --group <name>` from the repository root.
Use `all` for the 15 groups, or `connectors` for the nine reader groups below.
The [runner](../scripts/test_public.py) discovers tests in the directories below.

The `core` group also runs the discovery and socket regressions in
[`tests/public`](../tests/public). Staging means storing data temporarily
during a transfer.

| Test group | Test directory | Main local coverage |
| --- | --- | --- |
| `core` | [`packages/kumo-core/tests/sources/unit`](../packages/kumo-core/tests/sources/unit) | Covers shared source errors, identities, transfer cleanup, staging, query contracts, and resource limits. |
| `csv` | [`packages/kumo-core/tests/sources/csv`](../packages/kumo-core/tests/sources/csv) | Covers missing values, numeric limits, type detection, invalid or multipart input, dates, and output staging. |
| `parquet` | [`packages/kumo-core/tests/sources/parquet`](../packages/kumo-core/tests/sources/parquet) | Covers reads, writes, columns, row counts, versions, partitions, cleanup after failure, and limits on file paths. |
| `s3` | [`packages/kumo-s3/tests`](../packages/kumo-s3/tests) | Covers credentials, errors, sessions, regions, service addresses, connection probes, imports, and downloads that preserve files on failure. |
| `gcs` | [`packages/kumo-gcs/tests`](../packages/kumo-gcs/tests) | Covers downloads, object versions, credential renewal, concurrent transfers, cleanup, billing configuration, and imports. |
| `adls` | [`packages/kumo-adls/tests`](../packages/kumo-adls/tests), [`test/io_utils/fsutil/adls`](../test/io_utils/fsutil/adls) | Provider identity binding, CSV/Parquet ingestion, rooted output, version checks, URI fencing, and filesystem preview behavior. |
| `bigquery` | [`packages/kumo-bigquery/tests`](../packages/kumo-bigquery/tests) | Covers queries, streamed and staged reads, versions, row counts, errors, cleanup, and rules that enable cloud tests. |
| `databricks` | [`packages/kumo-databricks/tests`](../packages/kumo-databricks/tests) | Covers catalogs, versions, queries, reads, write helpers, and limits on cloud file downloads called CloudFetch. |
| `snowflake` | [`packages/kumo-snowflake/tests`](../packages/kumo-snowflake/tests) | Covers catalogs, types, queries, staged reads, concurrent downloads, retries, cleanup, write helpers, and hiding sensitive values. |

These tests do not prove that real cloud login, permissions, or transfers work.
Installed Kumo packages, snapshots, loading data into DiskGraph, and training
need separate tests.

The selected tests also have these limits:

- Snowflake tests use fake download bytes without decoding them as Parquet. It includes local output-writer and Snowflake Container Services (SPCS) tests. Native ingestion runs in `shared-sql`.
- ADLS provider tests use offline Azure responses for identities, reads, and writes. Filesystem tests cover preview and addressing; live credentials and Azure permissions remain separate checks.
- Parquet tests do not prove detection of corrupt Parquet data pages or every timestamp conversion.
- BigQuery tests use fake clients to test rules that enable real cloud tests. Pytest loads the cloud test file, but does not run its cloud tests.
- The BigQuery, Snowflake, and Databricks groups test their live-test gates with fake credentials and blocked provider imports.

## SDK, graph, and local-service groups

These six groups also run through the public command. They use source and public
dependencies. The SDK is a library for calling Kumo from code.
DiskGraph is the native library that builds and reads graph snapshots.
The runner builds it from this checkout for every group except `hosted-sdk`.

| Group | Test directory | What it tests | Local setup |
| --- | --- | --- | --- |
| `hosted-sdk` | [`packages/kumo-sdk/test/connector/local`](../packages/kumo-sdk/test/connector/local) | Upload, delete, connector configuration, and SQL API requests | API models and fake HTTP responses. No Kumo server. |
| `local-sdk` | [`packages/kumo-engine/test/connector`](../packages/kumo-engine/test/connector) | File and custom sources, including graph snapshots | CPU core packages and native DiskGraph. |
| `mixed-files` | [`test/connector/local`](../test/connector/local), [`test/io_utils/fsutil/local`](../test/io_utils/fsutil/local) | File and warehouse connectors, filesystem operations, timestamps, and credential boundaries | Hosted Python modules, bundled files, generated data, and fake cloud operations. |
| `pandas` | [`test/graph/local`](../test/graph/local) | Table and column behavior, encoders, and graph metadata | Hosted Python modules, local statistics, and bundled data. |
| `minio` | [`test/io_utils/fsutil/minio`](../test/io_utils/fsutil/minio) | S3-compatible storage, permissions, and TLS failures | Local MinIO with fake accounts. The runner creates and removes it. |
| `shared-sql` | [`packages/kumo-core/tests/sources/sql`](../packages/kumo-core/tests/sources/sql) | Common SQL helpers, API types, and native Snowflake type precision | Core and API packages plus DiskGraph. Also discovers [`tests/native`](../packages/kumo-snowflake/tests/native) in the Snowflake package. |

## Test tiers and product surfaces

The public command covers the source groups above. Released packages, GPU
workloads, real providers, and deployments need separate validation.

| Tier and surface | Command or entry point | Prerequisites | CI trigger and blocking status |
| --- | --- | --- | --- |
| Public readers and source helpers | `python3 scripts/test_public.py --group connectors` | Public reader image | Branch or fork PR, master push, merge queue event, or manual run. A failure fails its job; required-check policy remains under #31132. |
| SDK, graph, SQL, local storage | `python3 scripts/test_public.py --group all` | Public images, source-built DiskGraph, local MinIO | Same public workflow and blocking status. |
| Real provider integration | [Provider commands](credentialed-test-inventory.md) and hosted workflows below | Approved cloud account, fixtures, permissions, cleanup | Hosted jobs run conditionally as described below. Protected package-suite execution and release gates remain separate work. |
| GPU training and prediction | `make test-kumo-core-gpu`; [workflow](../.github/workflows/kumo-core-gpu.yml) | Prepared CUDA runtime and a trusted GPU runner | Manual run, or a team PR with the `ml-algorithms` label on labeled/synchronize/reopened events. A failure fails its job. Required-check policy remains under #31132. |
| Full service and deployment | [Component test guide](development/testing.md), [Helm tests](../helm/README.md) | Running services and deployment resources | Separate workflows. This PR does not change their gates. |
| Released package and release qualification | [Install prerequisites](development/setup.md) | Built release artifacts and target environment | Not covered by source tests. Release evidence and blocking policy remain separate work. |

## CI behavior

The [public workflow](../.github/workflows/public-connectors.yml) uses the triggers
listed above. Depending on repository settings, a maintainer must
[approve a fork's run](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/approve-runs-from-forks).
Reader groups run on x86_64 and ARM64. The other groups run on x86_64.
All jobs use GitHub-hosted runners and read-only repository permissions.
No group uses Kumo cloud credentials. Tests cannot access the external network.
A newer PR update cancels that PR's earlier run. Master and manual runs remain
independent. A failed test fails its job. Repository rules separately determine
which checks must pass before merging.

The existing hosted workflows have different triggers and failure policies:

| Workflow / coverage | When tests run | Failure behavior |
| --- | --- | --- |
| [Testing and Coverage](../.github/workflows/testing.yml): hosted S3, ADLS and GCP tests | PR, master push, merge queue, manual or reusable-workflow invocation, subject to test-file filters and selected shards. ADLS/GCP authentication is limited to configured repository/shard and same-repository PR conditions. | The hosted `pytest` job has `continue-on-error: true`. Package-suite failures fail the separate `packages` job. |
| [GCS Testing](../.github/workflows/testing_gcs.yml): hosted GCS and BigQuery end-to-end tests | PR, master push, merge queue or manual run; test steps require matching changed files. GCP authentication also requires the configured repository and a same-repository PR for PR events. | Test job has `continue-on-error: true`. |
| [Snowflake Testing](../.github/workflows/testing_snowflake.yml): hosted Snowflake tests | PR, master push, merge queue or manual run; each shard's test steps require matching changed files. | Test job has `continue-on-error: true`. |
| [Databricks Testing](../.github/workflows/testing_databricks.yml): hosted REST and DiskGraph tests | PR, master push, merge queue, manual or reusable-workflow invocation; test steps apply changed-file and full/minimal-mode conditions. Durations recording can force a minimal run. | Test job has `continue-on-error: true`. |

These workflows exclude pushes that only change `ui/` or `tools/`.
Their existence does not mean every PR runs cloud tests. Jobs marked
`continue-on-error` are informational for workflow failure propagation, not
evidence of a required merge gate. The package job does not enable the
Snowflake, Databricks or BigQuery live flags. Their source-reader/sink suites
remain distinct from the hosted workflows above. Access approvals and merge
rules are separate from these workflow triggers.

## Real cloud tests on Kumo resources

Maintainers run these tests with prepared data, credentials, and cloud permissions.
ADC means Application Default Credentials, Google's automatic credential lookup.

| Area | Required resources and remaining work |
| --- | --- |
| S3 | The [hosted gate](../kumo/testing/s3.py) accepts an access/secret key pair in AWS environment variables, web-identity role/token variables, container-credential URI variables, or an existing shared credential file. It does not validate credentials during collection. Needs a bucket and object prefix, a region, and an AWS identity. Older version tests assume fixed timestamps. Any nonempty `KUMO_S3_LIVE` value enables the package's [cloud tests](../packages/kumo-s3/tests/live/test_live_conformance.py). |
| GCS | Hosted tests can start when a configured credential file or `~/.config/gcloud/application_default_credentials.json` exists. Needs a project, identity or ADC, source and staging buckets, and read/write/delete permissions. `GCS_TEST_BUCKET` does not create data or grant permissions ([test setup](../test/rest/integ/gcs_connector_helpers.py)). |
| ADLS | Needs an Azure tenant, client, subscription, storage account, container, and storage permissions. Existing Azure environment variables or a credential file can enable the [cloud test](../test/io_utils/fsutil/test_adls_arrow_live.py). `KUMO_RUN_ADLS_TESTS=0` does not guarantee that the test stays disabled. |
| Snowflake | Needs an account, database, schema, warehouse, prepared input data, and permission to write output. Requires `KUMO_SNOWFLAKE_LIVE=1`. Set `KUMO_SNOWFLAKE_CREDS` to the path of a JSON file with credentials. There is no default credential path ([tests](../packages/kumo-snowflake/tests/live/test_live_conformance.py)). |
| Databricks | Needs a workspace, warehouse, catalogs, schemas, prepared data, and storage for staging. Needs permission to manage tokens, volumes, views, and tables. Requires `KUMO_DATABRICKS_LIVE=1`. Set `KUMO_DATABRICKS_CREDS` to the path of a JSON file with credentials ([tests](../packages/kumo-databricks/tests/live/test_live_conformance.py)). |
| BigQuery | Needs ADC, a billing project, reader source data, GCS staging, and permissions for datasets, tables, views, and objects. The sink creates destination tables and also uses the Storage Write API. Some hosted tests need identities with restricted permissions. Both package [reader](../packages/kumo-bigquery/tests/live/test_live_conformance.py) and [sink](../packages/kumo-bigquery/tests/live/test_live_sink.py) require `KUMO_BIGQUERY_LIVE=1`, `KUMO_BIGQUERY_PROJECT`, and `KUMO_BIGQUERY_STAGING` before credential lookup. |
| Hosted upload from start to finish | Needs a running API, writable storage, and generated files. Uses `RUN_KUMO_INTEG`, `KUMO_API_URL`, `KUMO_API_TOKEN`, and `KUMO_DEST` ([tests](../packages/kumo-sdk/test/connector/test_kumo_integration_e2e.py)). |

The Snowflake, Databricks, and BigQuery packages read credentials only after you
enable their cloud tests. Invalid configuration then fails the tests.

The [shared hosted setup](../test/conftest.py) imports warehouse libraries and
training packages. Live cloud tests also need credentials, cloud resources,
and prepared datasets. Use the [public test command](development/public-tests.md)
for the supported tests that run without cloud credentials.
The [financial helpers](../kumo/testing/data.py) read files from
[`test/csv_data/FINANCIAL`](../test/csv_data/FINANCIAL) and
[`test/parquet_data/FINANCIAL`](../test/parquet_data/FINANCIAL) in this checkout.
These local files do not require S3 credentials. Separate
[CSV](../test/connector/test_csv_connector.py) and
[Parquet](../test/connector/test_parquet_connector.py) tests read S3 datasets and
require AWS credentials. Passing tests do not establish permission to publish
the bundled datasets.

See the [cloud test inventory](credentialed-test-inventory.md)
for commands, required resources, and cleanup gaps.
[Issue #31132](https://github.com/kumo-ai/kumo/issues/31132) tracks fixture publication,
provider setup and cleanup ownership, test tiers required before merge or release,
and further public coverage. Protected execution is coordinated with
[#31074](https://github.com/kumo-ai/kumo/issues/31074).
