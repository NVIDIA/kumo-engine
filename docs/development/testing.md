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

The [Engine PR check](engine-pr-ci.md) builds the candidate wheel and runs eight
small task checks on CUDA through one shared smoke action. It checks completed
work, output structure, saved model state, and reloaded predictions. It sets no
learning-quality score requirement. Source and package workflows remain separate.
Repeat and reload scores use the shared absolute tolerance of `1e-5` with
zero relative tolerance. Weights, labels, ranks, valid outputs, and CUDA
execution remain strict.

## Finished Kumo Engine wheel

After each manual or release build, `wheel-smoke` downloads the finished Linux
x86_64 wheel from that run. It uses Python 3.12 and the existing T4 GPU runner.
It does not rebuild the wheel. Failure blocks release attachment and Kitmaker
publication. The macOS wheel has no GPU smoke check. The GPU job has a 22-minute
limit.

The job uses the same [GPU test script](../../packages/kumo-engine/test/wheel_gpu_smoke.py)
and [setup action](../../.github/actions/engine-gpu-smoke/action.yml) as the PR
check. The action installs the selected wheel in a fresh environment outside the
checkout. It uses public PyPI and ignores host pip settings. The installed version
and SHA-256 (file fingerprint) must match the downloaded wheel. Dependencies must
pass `pip check`.

Each of the eight task types trains one model with seed zero, predicts, and
reloads the saved model on CUDA. Outputs must be finite and complete. Restored
model weights, predicted labels, and rank order must match exactly. Repeat and
reload scores must satisfy `abs(before - after) <= 1e-5` at every coordinate,
with zero relative tolerance. These comparisons use the same retained model.

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

## Agent-local PR feedback

Vedaant Jain requested a marked local selection that finishes in strictly less
than five minutes. See [Engine PR checks](engine-pr-ci.md#agent-local-feedback)
for one-time environment setup, the single command, selected coverage and timing
evidence. The local command runs only the small pure output tests. It needs no
GPU or cloud access. Normal GPU CI fits eight models. The separate numeric
qualification fits eight models and repeats prediction and reload three times
on each model. It adds no fits for those repeats. Source and package workflows
keep their existing tests and failure policies. The preferred full-CI timeline
is 20 minutes. The accepted budget is about 22 minutes, with a strict
under-25-minute guardrail. The first normal run reached the 22-minute job limit
after a cold 14m22s build. Five task checks passed, but the run remained
incomplete. A complete normal runtime is still unverified.
