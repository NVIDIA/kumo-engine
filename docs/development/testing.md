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

The wheel smoke test runs after each manual or release wheel build.
A failure blocks release attachment and Kitmaker publication.
This test uses Linux x64, Python 3.12, and CPU execution.
It installs the selected wheel with dependencies from public PyPI in a fresh
virtual environment outside the source checkout.
Current dependencies include CUDA libraries, but this test requires no GPU.

From the repository root, use Python 3.12 and a new output directory:

```bash
python3.12 packages/kumo-engine/scripts/smoke_wheel.py /absolute/path/to/kumo_engine-SELECTED.whl \
  --output-dir /absolute/path/to/new-evidence
```

Success requires all of these results:

1. The installed version and SHA-256 fingerprint match the selected wheel.
   SHA-256 identifies the exact file contents. Dependency checks also pass.
2. The graph contains 50 entities, 3,000 events, and 6,000 directed edges.
   Each event links to one entity. The graph stores both edge directions.
3. Training completes and prediction returns each entity exactly once.
   Probabilities are finite, range from zero to one, and sum to one.
   Each label matches its probability at the 0.5 threshold.
4. A saved model loads and returns the same entity IDs and labels.
   Each probability difference satisfies `abs(a-b) <= 1e-6 + 1e-5 * abs(b)`.
5. The command exits with zero. Every failed assertion returns a nonzero exit code.

The fixture uses seed zero and generated data without a learning-quality target.
Training uses one epoch and two steps per epoch for each trial and refit.
This checks the workflow. It does not measure learning quality or performance
regressions, or establish support for other platforms.

The output directory contains `summary.json`, the pip installation report,
dependency checks, installed package versions, and the full workflow log.
`workflow-result.json` records graph counts, prediction checks, and stage times.
Prediction files preserve values before and after reload.
CI uploads these files after success or failure.
The job has a 20-minute limit. Measure its duration in CI before accepting the
under-20-minute target. Stage times are diagnostic, not a performance baseline.

Changes to this harness also run against the released `kumo-engine-v1.0.0`
wheel on pull requests. That run tests the harness, not the wheel built from
the pull request source. The manual `Kumo Engine - Wheel smoke` workflow
accepts an existing release tag and does not publish anything.

Run the small runner tests without engine dependencies:

```bash
python3.12 -m unittest discover -s test/scripts -p test_engine_wheel_smoke.py
```

## Report evidence

Record the command, environment, result, and any untested dependency. Keep
credentials, customer data, and private logs out of committed test reports.
For a behavior change, add a regression test at the relevant boundary; pure
prose edits need reference checks and a factual review rather than tests that
repeat the prose.
