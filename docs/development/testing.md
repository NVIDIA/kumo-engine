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

Each wheel installs once in its own fresh environment with the same fixed dependencies.
The comparison runs the basic workflow, frozen binary checks, and seven additional tasks in that environment.
Both wheels use the same five seeds, `0–4`, with fresh models for each task and case.
Every task has data with a known pattern and a matching case without that pattern.
Evaluation answers stay out of training and model features.

The binary cases each have 1,200 training rows, 400 validation rows, and 400 evaluation rows.
The fixed binary recipe uses 16 channels, batch size 64, Adam, learning rate `0.01`,
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
Strict comparisons return `status: PASS` with `comparison.release_approved: true` only after all required checks pass.

The additional tasks report these metrics and simple baselines:

| Task | Metric | Baseline |
| --- | --- | --- |
| Multiclass classification | Fraction of correct classes | Most frequent training class |
| Multilabel classification | Average precision, averaged across labels | Training frequency of each label |
| Multilabel ranking | Fraction of true labels in the top five | Training frequency of each label |
| Regression | Mean absolute error | Median training answer |
| Forecasting | Mean absolute error across 12 future steps | Last observed value |
| Static link prediction | Fraction of true links in the top five | Observed item frequency, excluding known links |
| Temporal link prediction | Fraction of true links in the top five | Observed item frequency |

These seven quality limits remain unvalidated. Strict release comparisons block until qualification passes.
`packages/kumo-engine/test/wheel_task_quality.json` records calculated limits with `status: PENDING` until fresh validation passes.
`comparison.additional_tasks` records each seed's reference, candidate, baseline, and signed drop, plus each task's average drop.
A positive drop means worse quality, including increased error for regression and forecasting.
No additional task uses the binary `0.001` limit.
An unvalidated result has `quality_limits: null` and `quality_gate_approved: false`.
Fresh fits of the same wheel can produce different scores. New limits need separate repeat measurements and reserved validation.

Each task compares the reference average with the candidate average across five matching seeds.
Improved seeds offset weaker seeds. Individual drops remain diagnostics.
Accuracy, average precision, and recall use fractions. Error uses the generated target's units.
A drop equal to its validated limit passes. Any greater drop blocks release, without a rounding allowance.
An exact decimal string can preserve a calculated limit that a JSON floating-point number cannot represent.

Qualification requires at least six independent unchanged-wheel trials and six actual weaker-training trials.
These calibration trials measure variation and sensitivity. They do not establish reliability by themselves.
For each task, let `n` be the largest absolute unchanged-wheel average difference in calibration.
Let `m` be its justified smallest blockworthy loss. Require `n < m`.
The proposed limit is `(n + m) / 2`. This gives equal space from observed variation and the blockworthy loss.
That midpoint maximizes the smaller of those two gaps. Actual weaker-training trials must show losses of at least `m`.
The qualification report preserves both calibration counts, observed variation, actual training losses, and the exact calculation.

Calibration of the revised fixture on October 23, 2026 used six unchanged-wheel comparisons and six actual reduced-training comparisons.
Each comparison used five matching seeds per wheel. The independent audits checked 1,680 additional-task models and 240 binary fits.
Every comparison has distinct model, measurement, and evidence identities.
The following values are rounded for display. The pending profile preserves exact fractional limits and all twelve audited calibration records.
The profile was frozen at `2026-10-23T14:22:53.799324Z`, before any new reserved trial.
Its canonical fingerprint is `b4ce4c903aba0dc0c24338759cde0504c8c01fa6602622ecb3099815388076c8`.
All 198 planned validation IDs were unused at the native preflight. Strict release approval remains blocked.

| Task | Largest normal difference `n` | Meaningful loss `m` | Frozen limit `(n + m) / 2` | Smallest actual weakened loss |
| --- | ---: | ---: | ---: | ---: |
| Multiclass accuracy | 0.022500000000 | 0.044000000000 | 0.033250000000 | 0.526000000000 |
| Multilabel average precision | 0.001933584626 | 0.022155904930 | 0.012044744778 | 0.497832908903 |
| Multilabel Recall@5 | 0.000000000000 | 0.047500000000 | 0.023750000000 | 0.445666666667 |
| Regression error | 0.005271614463 | 0.105263157895 | 0.055267386179 | 0.517396029036 |
| Forecast error | 0.032155811522 | 0.250000000000 | 0.141077905761 | 5.556548266212 |
| Static-link Recall@5 | 0.019333333333 | 0.033333333333 | 0.026333333333 | 0.780000000000 |
| Temporal-link Recall@5 | 0.006000000000 | 0.010000000000 | 0.008000000000 | 0.444000000000 |

All seven normal differences remain below their fixed meaningful losses. All six weakened comparisons meet those losses for every task.
The frozen fault limits candidate training to one epoch and two steps per epoch. Reference and binary training remain unchanged.
Detection claims cover that fixed fault and fixture. These results do not establish detection rates for every regression of the same size.
The twelve measured durations estimate 46.08 hours for 198 comparisons at four workers, before audits, archiving, delays, failures, and contention.
Compressed evidence samples estimate 30.58 GB. Verify each complete archive before removing task-owned original comparison files when space requires it.
Calibration alone does not approve release. The profile freezes all 198 trial IDs before any reserved comparison starts.

The previous study failed unchanged static-link trial 005: its recall loss was `0.0377777777777778`, above its fixed `2/75` limit.
That study cannot approve release. Its frozen profile and all fourteen claimed results remain preserved. Its remaining IDs are retired.

The pending profile defines proposed minimum losses from complete group or label-pattern failures and natural target steps.
Its formulas use fixed fixture counts and target structure, not candidate training scores.
The revised static-link fixture keeps the `1/30` policy loss.
Missing one link per entity in a ten-entity interest cohort gives `10/(100*3) = 1/30` recall loss.
Counterfactual metric calculations check these policy units. They do not count as actual weaker-training or reliability evidence.
The revised static-link fixture uses 100 entities and ten interest groups.
Each item has six observed links and three reserved answers across its group. No entity observes its reserved links.
This removes absent items and uneven observed coverage from the learnable fixture. Independent-label answers stay random.
Six unchanged comparisons measured static-link variation of `0.01933333333333336`, below the fixed `1/30` meaningful loss.
Fresh validation must still establish reliability.
The temporal-link fixture uses 100 entities and 100 items, with ten-item interest groups and three hidden future links per entity.
Each ten-entity cohort has equal hidden item counts. Every item in its interest group appears three times.
A shared correct-group ranking then scores `0.5` Recall@5 under any ordering of those ten items.
One outside-group replacement for that cohort loses three hits, or `3/(100*3) = 1/100` recall.
The independent-label control stays random. The metric, chronological splits, and training recipe stay unchanged.
This balances measurement noise. Fresh native trials must still establish actual variation, learning sensitivity, and reliability.
If normal variation hides that loss, improve the test instead of accepting a larger loss.
Freeze the limits, minimum losses, weaker-training recipe, source, and dependencies before reserved validation.

The reliability policy permits at most 5% false rejection across all seven tasks.
Each task must detect the frozen reduced-training fault at least 95% of the time.
The evidence must establish these eight claims together with 95% confidence.
A fixed zero-error design needs 99 fresh unchanged-wheel comparisons and 99 fresh weaker-training comparisons.
The count is `ceil(log(0.05 / 8) / log(0.95)) = 99` independent comparison trials, not individual seeds.
All reserved trials must pass the frozen rules. Do not reuse calibration trials or widen limits after validation starts.
Freeze the 99 trial IDs for each treatment in `validation_plan` before the first reserved trial.
Every planned trial must have complete passing evidence. A missing or failed trial keeps qualification unapproved.
Do not replace a failed planned trial with a new success.
Saved trial records include dates, unique run identities, evidence fingerprints, audited results, and all seven metrics.
Each trial lists 140 distinct model identities for the additional tasks and four measurement identities across both wheels.
Those four identities cover binary and additional-task measurements. No model or measurement identity can repeat across study trials.
The release guard repeats the calculation from the saved scores. Claimed approval alone cannot authorize publication.

Before validation, freeze the profile with `status: PENDING` and an empty `validation` list.
Keep every other field unchanged after that freeze.
Each audit record must carry `quality_profile_sha256` from the executed comparison.
The auditor must match that fingerprint against the profile saved with the source for that run.

After all planned trials pass, append their records to `validation`.
Change `status` to `VALIDATED`.
The release guard reconstructs the frozen pending profile and requires every validation fingerprint to match.
This blocks rewritten trial plans and wider limits that reuse earlier validation results.

To measure a real training weakness, use the reporting-only calibration mode:

```bash
python3.12 packages/kumo-engine/scripts/smoke_wheel.py /path/to/candidate.whl \
  --compare-with /path/to/reference.whl --report-learning \
  --calibration-budget 1 2 --output-dir /path/to/new-sensitivity-evidence
```

This example limits each candidate fit for the seven additional tasks to one epoch and two steps per epoch.
It does not change binary training or the reference fits. Both wheels still use five matching seeds and fresh installations.
This is an initial strong sensitivity check, not a release limit or the smallest acceptable learning loss.
Try less severe training reductions during calibration to measure smaller losses.

Fault reports retain the normal requested plan and the actual reduced plan, selected plan, and plan hashes.
The comparison reads the saved plans and permits only the declared budget changes.
Other input, execution, prediction, reload, and independent-label checks remain strict.
A plan mismatch alone does not demonstrate quality detection. Compare the actual scores.
Calibration requires both reporting mode and a reference wheel. It never approves release.
The release guard rejects fault-marked results even if an outer summary claims success.

Each additional task reuses its fitted model for full and partial prediction batches, then reloads for another prediction.
The test requires matching IDs, complete finite scores, and agreement within `abs(full-other) <= 1e-6 + 1e-6 * abs(other)`.
Forecasting generates prediction tables with 12 observed historical values and predicts 12 future steps.
Training, validation, and test target windows cannot cross stage boundaries.
Temporal sampling uses each row's timestamp as its cutoff.
Future evaluation rows and reserved static links stay outside the observed graph.
This suite covers generated forecast tables.
The `1.0.0` wheel fails when custom forecast inputs contain Float32 sequences. That path lacks passing coverage.

Add `--report-learning` to report the same decision without blocking on its average drop.
Reporting returns zero with `status: MEASURED` and `comparison.release_approved: false` when execution and evidence checks pass.
Installation, training, prediction, reload, and evidence failures return nonzero in both modes.
Missing tasks, seeds, controls, or unfinished results cannot pass.
Data, plan, dependency, source, and seed mismatches fail.
Additional task scores must match the saved predictions and answers.
Shuffled-label tests reject suspicious associations in the independent cases.
These checks remain strict in reporting mode.
Controls use 999 shuffles and seed `0`, shared across matching wheel runs.
The binary control fails when `p <= 0.01`. Each additional task fails when `p <= 0.01 / 7`.
Static link controls shuffle only among eligible items, excluding known links.
This policy is separate from measured repeat limits.
Evidence includes saved scores, answers, inputs, splits, plans, dependency versions, and source fingerprints.

Every manual or release build runs the strict comparison, including all seven additional tasks.
Release attachment and publication require the binary gate and seven validated quality gates, plus complete passing execution and controls.
The pending seven-task profile currently prevents release approval.
The manual `Kumo Engine - Wheel smoke` workflow exposes the same coverage through `learning_comparison` in reporting mode.
PR runs test the basic harness against a released wheel. They do not build the PR's package or run the full learning comparison.
No nightly or weekly learning schedule is added.
The binary limit covers five fixtures of one generated graph pattern. It does not establish broader quality or a false-alarm rate.
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

Run the learning and runner tests in the installed wheel environment with pytest:

```bash
python3.12 -m pytest --noconftest test/scripts/test_engine_wheel_*.py
```

### Measure GPU learning

Run the small workflow first on Linux x64 with Python 3.12 and one visible CUDA GPU:

```bash
python3.12 packages/kumo-engine/scripts/smoke_wheel.py /path/to/selected.whl \
  --device cuda --output-dir /path/to/new-gpu-workflow
```

After the workflow completes, compare the candidate and reference:

```bash
python3.12 packages/kumo-engine/scripts/smoke_wheel.py /path/to/candidate.whl \
  --compare-with /path/to/reference.whl --device cuda --report-learning \
  --output-dir /path/to/new-gpu-comparison
```

Both wheels install separately with the approved dependency versions.
The comparison reuses the existing eight task recipes without changing frozen CPU source or evidence.
It performs 160 fresh task fits: eight tasks, five seeds, two cases, and two wheels.
Its two basic workflow executions are additional.
The GPU entry point records actual training devices for every fit.
The seven additional tasks also require observed GPU prediction and prediction after reload.
The basic workflow covers those prediction stages for binary classification.
Missing GPU evidence or CPU fallback fails execution.

GPU results return `MEASURED` and cannot approve release.
The binary report retains the `0.001` mean-drop diagnostic.
CPU qualification does not qualify GPU quality limits.
GPU checks require complete IDs, matching shapes, finite values, and successful prediction and reload.
Record partial-batch and reload differences without applying CPU agreement limits.
The report includes maximum differences and counts that exceed the historical CPU tolerance.
These measurements do not approve a GPU consistency limit.
Existing CPU commands and release checks keep their current behavior.

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
