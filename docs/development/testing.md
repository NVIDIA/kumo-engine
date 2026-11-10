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

kumo-engine and kumo-core tests load the DiskGraph native module
(`diskgraph/_c.abi3.so`) and its `bin/ingest` binary from the installed
`diskgraph` package. Both must be built from the same tree as the Python code
under test: a build from an older checkout fails with missing attributes or
arguments (for example `RuntimeIndex.iter(projections=...)` or
`compute_col_stats_for_parquet(percentiles=...)`). After pulling DiskGraph
changes, rebuild with `make install-diskgraph`, which builds and installs both.

For connector changes, start with the [public test command](public-tests.md).
It builds its prerequisites and runs without cloud credentials. The
[test matrix](../connector-test-matrix.md) lists coverage and CI limits.

The [Engine PR check](engine-pr-ci.md) builds the candidate wheel on CPU and
runs eight small task checks on CUDA through one shared smoke action. Repeated
and reloaded predictions must match within `0.005` absolute tolerance and
zero relative tolerance for every task. Labels and ranks can change only for
score ambiguities within that tolerance. Complete valid outputs and completed
CUDA work remain strict. Training metrics report separately and set
no learning-quality requirement. Source and package workflows remain separate.

## Finished Kumo Engine wheel

After each manual or release build, `wheel-smoke` downloads the finished Linux
x86_64 wheel from that run. It uses Python 3.12 and the authorized RTX PRO 6000 runner.
It does not rebuild the wheel. Failure blocks release attachment and Kitmaker
publication. The macOS wheel has no GPU smoke check. The GPU job has a 22-minute
limit.

The platform CPU matrix also qualifies the finished macOS ARM64 wheel on
`macos-15` with Python 3.11–3.14. It uses the same clean-install action and
train/predict fixture as Linux ARM64 and Windows: verify wheel provenance and
dependencies, observe four optimizer steps, predict 100 held-out entities, and
reload the saved model. The manual **Kumo Engine - macOS ARM64 smoke** workflow
builds and tests just this platform on a selected Python version.

The job uses the same [GPU test script](../../packages/kumo-engine/test/wheel_gpu_smoke.py)
and [setup action](../../.github/actions/engine-gpu-smoke/action.yml) as the PR
check. The action installs the selected wheel in a fresh environment outside the
checkout. It uses public PyPI and ignores host pip settings. The installed version
must match the selected wheel. Dependencies must pass `pip check`.

Each of the eight task types trains one model with seed zero, predicts, and
reloads the saved model on CUDA. Outputs must be finite and complete.
Repeat and reload scores must satisfy `abs(before - after) <= 0.005` at every
coordinate, with zero relative tolerance. Labels and ranks can change only when
their scores justify the change within that bound. These comparisons use the
same retained model.

The [input manifest](../../packages/kumo-engine/test/pr172/inputs.json) pins
the prepared S3 data and its checksum. The workflow uses its existing AWS test role.
The job preserves GPU results and logs in the `kumo-engine-wheel-smoke-*` artifact.

To test a new wheel without publication, run Kumo Engine - Build and publish
wheel manually at the selected commit. Candidate PR feedback runs separately
through the trusted PR GPU workflow.

## Report evidence

Record the command, environment, result, and any untested dependency. Keep
credentials, customer data, and private logs out of committed test reports.
For a behavior change, add a regression test at the relevant boundary; pure
prose edits need reference checks and a factual review rather than tests that
repeat the prose.

## Local PR feedback

The [Engine PR guide](engine-pr-ci.md#local-feedback) gives one pytest command
for the small pure output and workflow checks. The existing OSS Readiness job
runs the same tests. They need no GPU or cloud access. The GPU job runs the
eight actual training and save/load cases.
