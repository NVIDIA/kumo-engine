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
A failed or missing result blocks release attachment and Kitmaker publication.
It installs in a fresh environment outside the checkout, uses PyPI, and ignores
host pip settings. CUDA libraries still install, but no GPU is required.

From the repository root, use a new output directory:

```bash
python3.12 packages/kumo-engine/scripts/smoke_wheel.py /absolute/path/to/kumo_engine-SELECTED.whl \
  --output-dir /absolute/path/to/new-evidence
```

The default smoke check passes only when:

- The installed version and SHA-256 (file fingerprint) match the selected wheel. Dependencies pass `pip check`.
- The graph has 50 entities, 3,000 events, and 6,000 directed edges.
- Training completes. Prediction returns each of the 50 entity IDs once. Probabilities are finite, within
  `[0, 1]`, and sum to one. Labels match the `> 0.5` probability threshold.
- Reloaded IDs and labels match. Probability differences satisfy
  `abs(a-b) <= 1e-6 + 1e-5 * abs(b)`.

[Settings and the approved reference](../../packages/kumo-engine/test/wheel_workflow.json)
fix the data, query, seed `0`, and one epoch with two steps per trial and refit.
The default smoke check covers workflow correctness. The paired learning check
below covers the release quality rule. Speed limits and other platforms need separate tests.

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

To enforce the release learning rule, compare a candidate wheel with the reference:

```bash
python3.12 packages/kumo-engine/scripts/smoke_wheel.py /path/to/candidate.whl \
  --compare-with /path/to/reference.whl \
  --output-dir /path/to/new-comparison
```

Each wheel gets a fresh environment with the same fixed dependencies.
Each trains a fresh graph neural network for the same five seeds, `0–4`, on each of two cases.
One case has a known relationship. The other uses independent random labels on the same graph.
Each case has 1,200 training rows, 400 validation rows, and 400 evaluation rows.
Evaluation rows stay out of training. Labels stay out of model features.

The fixed recipe uses 16 channels, batch size 64, Adam, learning rate `0.01`,
and at most ten epochs of twenty steps. Early stopping can shorten training.
The test rejects changes to requested settings in the selected training plan.

AUROC measures how well probabilities rank the correct answers. Constant scores give `0.50`.
For the case with a known relationship, subtract the candidate mean from the reference mean across all five matching seeds.
An average drop greater than `0.001` blocks release. A drop equal to `0.001` passes.
Improvements give negative drops.
The profile sets `learning.release_max_mean_drop` to `0.001` AUROC.

The `comparison` result keeps both means, individual scores, signed drops, and the decision in `mean_drop_alert`.
Historical `limits`, `failed_limits`, and `quality_status` remain diagnostics, even when they report a failure.
The old score floor, per-seed limits, and measured mean-drop limit cannot override the release rule.
Frozen calibration evidence stays unchanged, including its historical `learning.release_approved: false` field.
The current release decision is `comparison.release_approved`.
Strict runs return `status: PASS` with `comparison.release_approved: true` only after all required checks pass.

Add `--report-learning` to report the same decision without blocking on its average drop.
Reporting returns zero with `status: MEASURED` and `comparison.release_approved: false` when execution and evidence checks pass.
Installation, training, prediction, reload, and evidence failures return nonzero in both modes.
Missing or unfinished results cannot pass. Data, plan, dependency, and seed mismatches fail.
Shuffled-label tests reject suspicious associations in the independent cases.
These checks remain strict in reporting mode.
The control uses 999 shuffles and seed `0`. It fails when `p <= 0.01`.
Calibration shares each label shuffle across arms that repeat the same fixture.
This policy is separate from measured repeat limits.
Evidence includes saved probabilities, splits, plans, dependency versions, and source fingerprints.
Mismatched data, plans, dependencies, or worker source fail the comparison.

The build workflow requires the strict comparison before release attachment or publication.
The manual `Kumo Engine - Wheel smoke` workflow exposes `learning_comparison` in reporting mode.
The rule covers five fixtures of one generated graph pattern. It does not establish a false-alarm rate or broader model quality.
An unchanged reference previously missed the historical score floor. That floor remains a diagnostic.
The runner preserves failed learning evidence and its specific error in the summary.

Run calibration in the wheel environment with NumPy and SciPy installed.
Collect six pairs with `--reference --measure-learning` under `calibration/pair-N/{a,b}`.
Freeze the limits. Then collect two new pairs under `reserved/pair-N/{a,b}`.

```bash
python3.12 packages/kumo-engine/test/wheel_learning.py --calibrate calibration --output-dir frozen
# After collecting reserved runs:
python3.12 packages/kumo-engine/test/wheel_learning.py --calibrate calibration \
  --validate-with reserved --output-dir checked
```

These calibration commands preserve the historical repeat limits. They do not set the release mean-drop rule.
Only replace the recorded baseline after reserved runs pass and the reference update is approved.
Keep failed evidence. Do not widen limits using reserved results.

Run the learning unit tests in the installed wheel environment:

```bash
python3.12 -m unittest discover -s test/scripts -p test_engine_wheel_learning.py
```

### Measure prediction variation

On Linux x64 with a working CUDA GPU, run:

```bash
python3.12 packages/kumo-engine/scripts/smoke_wheel.py /path/to/selected.whl \
  --measure-variation --device cuda --runs 10 --output-dir /path/to/new-evidence
```

The installer runs once. Each fresh process trains a new model with the same
data, seed, and settings. `variation.json` separates repeat prediction, reload,
and differences between fresh models. It records probability differences and
label changes. Each `run-NNN` directory keeps predictions, metrics, device
observations, and logs. The summary configuration records the requested device;
`requirements.json` preserves the base CPU profile. The worker verifies the
actual device at training and prediction. Missing CUDA or CPU fallback fails.

This returns `MEASURED`. It approves no GPU tolerance and changes no publication
gate. Functional checks remain required. Reload differences are measured without
the CPU agreement limits. Ten runs are a sample, not a statistical guarantee.
Use `--device cpu` for CPU measurements. Run learning and wheel comparisons
separately; these options cannot be combined with variation measurements.

## Report evidence

Record the command, environment, result, and any untested dependency. Keep
credentials, customer data, and private logs out of committed test reports.
For a behavior change, add a regression test at the relevant boundary; pure
prose edits need reference checks and a factual review rather than tests that
repeat the prose.
