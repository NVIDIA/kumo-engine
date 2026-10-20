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

The wheel test runs after every manual or release build on Linux x64, Python 3.12, CPU.
Failure blocks release attachment and Kitmaker publication. The limit is 20 minutes.
It installs in a fresh environment outside the checkout, uses PyPI, and ignores
host pip settings. CUDA libraries still install, but no GPU is required.

From the repository root, use a new output directory:

```bash
python3.12 packages/kumo-engine/scripts/smoke_wheel.py /absolute/path/to/kumo_engine-SELECTED.whl \
  --output-dir /absolute/path/to/new-evidence
```

Exit code zero requires all of these results:

- The installed version and SHA-256 (file fingerprint) match the selected wheel. Dependencies pass `pip check`.
- The graph has 50 entities, 3,000 events, and 6,000 directed edges.
- Training completes. Prediction returns each of the 50 entity IDs once. Probabilities are finite, within
  `[0, 1]`, and sum to one. Labels match the `> 0.5` probability threshold.
- Reloaded IDs and labels match. Probability differences satisfy
  `abs(a-b) <= 1e-6 + 1e-5 * abs(b)`.

[Settings and the approved reference](../../packages/kumo-engine/test/wheel_workflow.json)
fix the data, query, seed `0`, and one epoch with two steps per trial and refit.
This checks workflow correctness. Learning quality, speed limits, and other
platforms need separate tests.

CI saves `summary.json`, `workflow-result.json`, settings, installation and
dependency reports, model configuration, graph statistics, predictions, and logs
after success or failure.

Add `--reference` to require the approved wheel and dependency versions.
PRs test both modes against the approved release. To replace the reference,
first pass the ordinary test and save its evidence. Obtain the release owner's
approval. Update the settings file. Repeat with `--reference`.

Run the runner tests without engine dependencies:

```bash
python3.12 -m unittest discover -s test/scripts -p test_engine_wheel_smoke.py
```

Add `--learning` to train two fresh models per seed on the same installed wheel.
One case has a known relationship. The other keeps the graph but uses independent random labels.
Both use seeds `0–4`, 1,200 training rows, 400 validation rows, and 400 evaluation rows.
Evaluation rows stay out of training. The target column stays out of model features.

The test computes AUROC, a ranking score, from saved probabilities. Constant scores give `0.50`.
Provisional limits require mean AUROC of at least `0.75` for learnable labels
and `0.45–0.55` for independent labels. These limits need release-owner approval.
This option does not change release gates. It does not compare a candidate with a quality reference yet.
Results under `learning/` include every score, split, probability, model plan, dependency version, and source fingerprint.

Run the learning unit tests in an environment with pandas, NumPy, and scikit-learn:

```bash
python3.12 -m unittest discover -s test/scripts -p test_engine_wheel_learning.py
```

## Report evidence

Record the command, environment, result, and any untested dependency. Keep
credentials, customer data, and private logs out of committed test reports.
For a behavior change, add a regression test at the relevant boundary; pure
prose edits need reference checks and a factual review rather than tests that
repeat the prose.
