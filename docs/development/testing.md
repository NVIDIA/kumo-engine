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

To report learning scores, compare a candidate wheel with the reference:

```bash
python3.12 packages/kumo-engine/scripts/smoke_wheel.py /path/to/candidate.whl \
  --compare-with /path/to/reference.whl --report-learning \
  --output-dir /path/to/new-comparison
```

Each wheel gets a fresh environment with the same fixed dependencies.
Each trains a fresh graph neural network for seeds `0–4` on two cases.
One case has a known relationship. The other uses independent random labels on the same graph.
Each case has 1,200 training rows, 400 validation rows, and 400 evaluation rows.
Evaluation rows stay out of training. Labels stay out of model features.

The fixed recipe uses 16 channels, batch size 64, Adam, learning rate `0.01`,
and at most ten epochs of twenty steps. Early stopping can shorten training.
The test rejects changes to requested settings in the selected training plan.

AUROC measures how well probabilities rank the correct answers. Constant scores give `0.50`.
The settings file records limits from repeated reference runs.
The report lists missed limits for learnable mean AUROC, individual seeds,
the reference learning floor, and independent-label means near `0.50`.
Its `comparison` field contains both means, signed score drops, limits, and `failed_limits`.
Positive drops mean that the candidate scored lower than the reference.
The command completes both wheels when only these provisional limits fail.
It returns zero with `status: MEASURED` and `release_approved: false`.
The report's `quality_status` records whether the provisional limits passed.
Installation, training, prediction, reload, and evidence failures still return nonzero.
Missing or unfinished results cannot pass. Data, plan, and dependency mismatches still fail.
Omit `--report-learning` to enforce the provisional limits in the strict comparison.
Shuffled-label tests reject suspicious associations in the independent cases.
These checks remain strict in reporting mode.
The fixed policy uses 999 shuffles, seed `0`, and `p <= 0.01`.
Calibration shares each label shuffle across arms that repeat the same fixture.
This policy is separate from measured repeat limits.
Evidence includes saved probabilities, splits, plans, dependency versions, and source fingerprints.
Mismatched data, plans, dependencies, or worker source fail the comparison.

The manual `Kumo Engine - Wheel smoke` workflow exposes `learning_comparison` in reporting mode.
This experiment does not block publication. Its limits cover one VM and five fixtures of one generated graph pattern.
They do not establish a false-alarm rate or broader model quality.

A subsequent full same-version check rejected the unchanged reference.
Its mean AUROC was `0.891132`, below the frozen floor of `0.891182`.
This profile needs further review before it can block releases.
The runner preserves a failed learning report and its specific error in the summary.

Run calibration in the wheel environment with NumPy and SciPy installed.
Collect six pairs with `--reference --measure-learning` under `calibration/pair-N/{a,b}`.
Freeze the limits. Then collect two new pairs under `reserved/pair-N/{a,b}`.

```bash
python3.12 packages/kumo-engine/test/wheel_learning.py --calibrate calibration --output-dir frozen
# After collecting reserved runs:
python3.12 packages/kumo-engine/test/wheel_learning.py --calibrate calibration \
  --validate-with reserved --output-dir checked
```

Only copy `checked/learning-baseline.json` into the settings file's `learning` field after reserved runs pass.
Keep failed evidence. Do not widen limits using reserved results.

Run the learning unit tests in the installed wheel environment:

```bash
python3.12 -m unittest discover -s test/scripts -p test_engine_wheel_learning.py
```

## Report evidence

Record the command, environment, result, and any untested dependency. Keep
credentials, customer data, and private logs out of committed test reports.
For a behavior change, add a regression test at the relevant boundary; pure
prose edits need reference checks and a factual review rather than tests that
repeat the prose.
