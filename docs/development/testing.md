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
| Integration/GPU | Selected workflow, storage, training, or deployment tests | Explicit services, data, GPU/runtime resources, and operator credentials where required |

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

## Finished Kumo Engine wheel

After each manual or release build, `wheel-smoke` downloads the finished x86_64
wheel from that run. It uses Python 3.12 and an existing T4 GPU runner. It does
not rebuild the wheel on the GPU. Failure blocks release attachment and Kitmaker
publication. The GPU job has a 22-minute limit.

The job uses the same [GPU runner](../../packages/kumo-engine/test/wheel_gpu_smoke.py)
and [setup action](../../.github/actions/engine-gpu-smoke/action.yml) as the PR
check. The action installs the selected wheel in a fresh environment outside the
checkout. It uses public PyPI and ignores host pip settings. The installed version
and SHA-256 (file fingerprint) must match the downloaded wheel. Dependencies must
pass `pip check`.

The eight task types each train one model with seed zero, predict, and reload the
saved model on CUDA. Outputs must be finite and complete. Restored model weights
must match exactly. Prediction comparisons use the shared runner's GPU limits.
Learning-quality scores remain informational.

Prepared PR 172 inputs come from the S3 object and checksum in the
[input manifest](../../packages/kumo-engine/test/pr172/inputs.json). The workflow
uses its existing AWS test role. It does not read a private GitHub repository.
The job preserves GPU results and logs in the `kumo-engine-wheel-smoke-*` artifact.

To test a new wheel without publication, run Kumo Engine - Build and publish
wheel manually at the selected commit. To test an existing release, run
Kumo Engine - Wheel smoke and set `release_tag` to its `kumo-engine-v<version>`
tag. Candidate PR feedback runs separately through the trusted PR GPU workflow.

## Report evidence

Record the command, environment, result, and any untested dependency. Keep
credentials, customer data, and private logs out of committed test reports.
For a behavior change, add a regression test at the relevant boundary; pure
prose edits need reference checks and a factual review rather than tests that
repeat the prose.
