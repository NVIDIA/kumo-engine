# Testing

Depending on the behavior changed, run the relevant checks below. A passing
local check does not establish a supported deployment or GPU profile.

| Tier | Examples | Prerequisites |
| --- | --- | --- |
| Documentation | [Reference, agent, and catalog checks](../maintenance/README.md#validation) | Python 3; Bash and standard shell tools |
| Focused Python | `.venv/bin/pytest path/to/test.py -k test_name`; package-local tests | Matching installed environment and test dependencies |
| Lint and types | `.venv/bin/flake8 <path>`, `.venv/bin/isort --check <path>`, `.venv/bin/pyright -p pyrightconfig.json` | `uv sync --extra dev --extra cpu` (or `--extra cu130` for GPU) environment |
| UI | `npm --prefix ui run test -- --run <test-path>`, `npm --prefix ui run type-check` | Installed UI and in-tree catalog dependencies |
| Catalog | `npm --prefix packages/kumo-ui-catalog run test:ci` | Catalog dependencies and configured Playwright browser |
| Rust | `cargo test --manifest-path packages/diskgraph/Cargo.toml -p diskgraph-train-table` | Rust toolchain, native build prerequisites, and crate dependencies |
| Helm contract | [Helm CI checks](../../.github/workflows/kumo-helm-ci.yml) | Helm, Python, and each check's declared tooling |
| Integration/GPU | `make test-kumo-core-gpu` for CUDA behavior; selected storage, training, or deployment tests | CUDA-capable GPU with matching PyTorch/CUDA/Faiss dependencies for GPU tests; explicit services, data, and credentials where required |

Shared test setup loads Snowflake credentials only when a selected test requests
them. Live Snowflake tests still need credentials and access to the service.
In `test` and `dev_local` deployment modes, instance discovery uses the hostname
without querying the EC2 metadata service.

Run from the repository root unless a component guide says otherwise.
Python package test suites have different roots and fixtures; inspect the
package's configuration before invoking pytest over multiple packages.

For connector changes, start with the [public test command](public-tests.md).
It builds its prerequisites and runs without cloud credentials. The
[test matrix](../connector-test-matrix.md) lists coverage and CI limits.

## Credential profiles

The opt-in [profile registry and launcher](../../tools/ci_test_profiles.py)
separate test selection from isolated execution. GCP profiles start pytest
with an allowlisted environment and temporary credential/configuration
directories. General and Snowflake profiles currently support only collection
through this launcher: their remaining Azure and Databricks dependencies have
not been partitioned, so isolated execution fails before setup rather than
silently removing credentials and losing live coverage.

| Profile | Credentials and ownership |
| --- | --- |
| `general` | Non-GCP selection; isolated launcher is collection-only during migration |
| `snowflake` | Non-GCP selection with `--runsnowflake`; isolated launcher is collection-only during migration |
| `gcp` | GCS/BigQuery tests with federated ADC; local storage for REST integration tests |
| `gcp-aws` | BigQuery training integration test with federated ADC and the S3 dataplane |

Run from the checkout root with its test dependencies installed:

```bash
.venv/bin/python -m tools.ci_test_profiles run gcp \
  test/io_utils/fsutil/test_gcs.py -q
.venv/bin/python -m tools.ci_test_profiles run gcp-aws
```

During migration, general and Snowflake execution uses pytest directly with
the selection plugin, preserving the existing credential supply:

```bash
.venv/bin/python -m pytest -p tools.pytest_test_profiles \
  --test-profile general test/connector/
.venv/bin/python -m pytest -p tools.pytest_test_profiles \
  --test-profile snowflake --runsnowflake test/connector/
```

These direct commands select tests but do not scrub the environment or isolate
credential files. They are a transitional coverage-preserving path, not a
claim that general and Snowflake processes already have minimal credentials.

GCP execution requires an explicit `GOOGLE_APPLICATION_CREDENTIALS` file of
type `external_account`; ambient user credentials and stored service-account
keys do not activate these profiles. Set `GCP_TEST_SERVICE_ACCOUNT` and
`GCP_PROJECT_ID` for identity-backed REST tests, plus
`GCP_BIGQUERY_NO_TABLE_SERVICE_ACCOUNT` and
`GCP_BIGQUERY_NO_JOB_SERVICE_ACCOUNT` for BigQuery permission tests.
Missing required credentials fail setup. The
launcher copies ADC into its temporary directory and preserves its external
token-source references, which must remain accessible for the run.

Direct local BigQuery tests also require an explicit
`GOOGLE_APPLICATION_CREDENTIALS` file. Supported local ADC types are
`service_account`, `authorized_user`, `impersonated_service_account`, and
`external_account`. Enabling `KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED` requires
`external_account` even outside a profile. BigQuery test setup does not fetch
AWS secrets, search home-directory credentials, or write credential files.
Missing or invalid inputs fail before dependent resources start.

For local restricted-permission tests using a service-account key, provide
separate key files through `GCP_BIGQUERY_NO_TABLE_CREDENTIALS` and
`GCP_BIGQUERY_NO_JOB_CREDENTIALS`. For user, impersonated, or federated ADC,
provide the two `GCP_BIGQUERY_NO_*_SERVICE_ACCOUNT` targets instead; the source
identity needs permission to impersonate them. The test identities must have
the intended restricted permissions. The helper validates the credential type
and explicit configuration; the live tests verify the actual permissions.

Use `@withBigQueryCredentials` for the default identity, or
`@withBigQueryCredentials(identity=BigQueryIdentity.NO_TABLE)` /
`BigQueryIdentity.NO_JOB` for restricted tests. These decorators only declare
requirements. Shared pytest fixtures prepare the base identity before session
resources and scope each test's identity until its dependent fixtures finish.
Explicit input files remain unchanged, and environment changes are restored
after success, failure, or skip.

The GCP-only runtime removes AWS secret fallback and uses a local artifact
directory. The GCP+AWS profile retains S3 because the training test actually
uses it. Credentials are prepared before worker startup; function-specific
identity changes are restored during teardown. These profiles constrain
credential inputs, not network access. Use the public Docker suites when
network isolation is required.

For a collection audit, `--manifest` saves the selected node IDs without
running fixtures:

```bash
.venv/bin/python -m tools.ci_test_profiles run gcp \
  --manifest /tmp/gcp-nodes.json --collect-only
```

For the unpartitioned baseline, run pytest over the same source revision and
target scope with `-p tools.pytest_test_profiles --collect-only
--test-profile-manifest` and no profile. Its output contains a
`KUMO_TEST_PROFILE_MANIFEST=` JSON record. Compare its `nodeids` with the union
of the `general`, `gcp`, and `gcp-aws` manifests, checking both missing nodes
and overlaps. Audit the Snowflake mode separately with `--runsnowflake`;
existing Snowflake skip rules are independent of GCP ownership. Collection
still imports shared conftests and requires their dependencies. A collection
manifest or local plugin test does not verify live provider access.

## Report evidence

Record the command, environment, result, and any untested dependency. Keep
credentials, customer data, and private logs out of committed test reports.
For a behavior change, add a regression test at the relevant boundary; pure
prose edits need reference checks and a factual review rather than tests that
repeat the prose.
