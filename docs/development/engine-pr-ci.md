# Engine PR check

The `Kumo Engine PR` workflow tests the current candidate wheel on the existing
T4 GPU runner. Vedaant Jain requested this smaller check. One job builds the
wheel, installs it in a fresh environment, and runs the shared smoke action.
The release-wheel workflow in PR40 uses the same action with its finished wheel.
No runner or cloud instance is created.
The action uses a cached parallel dependency installer. Pip then reinstalls
only the candidate wheel offline and verifies its archive hash. The environment
stays fresh, and dependency downloads use public PyPI.

## Eight task checks

Each normal run fits eight models with seed zero. The tasks are binary,
multiclass, multilabel, multilabel ranking, regression, forecasting, static
link prediction, and temporal link prediction. Each task fits, predicts,
saves its model, loads that model, and predicts again.

The runner requires the expected installed wheel version and actual CUDA
training and prediction batches. It rejects empty work, missing output rows,
duplicate coordinates, and nonfinite values. Forecasts require all twelve
horizons. Class and link outputs require every requested class or catalog item.
The restored model weights must exactly match the saved checkpoint.
Chosen labels and published ranks must also match exactly after repeat or reload.
Training metrics report only; they do not set a learning-quality gate.

The runner measures repeated and reloaded prediction differences. GPU results
can contain small numeric differences despite identical saved weights.
The manual qualification fits eight models and repeats prediction and reload
three times on each retained model. It adds no training block for those repeats.
Qualification returns `MEASURED`, not a normal CI pass. A normal run has eight
fits, one prediction/reload cycle, and enforced numeric comparison limits.
The [T4 qualification](https://github.com/NVIDIA/kumo-engine/actions/runs/40975805986)
completed all eight fits and 48 repeat/reload comparisons. Its largest absolute
difference was 0.00000190735. We use an absolute limit of 0.00001 with zero
relative tolerance, about 5.2 times that measured maximum. Absolute comparison
also handles values close to zero. Weights, chosen labels, and ranks stay exact.
This empirical limit covers the fixed PR172 inputs on Tesla T4 with
PyTorch 2.15.0 and CUDA 13.2. It does not qualify another device or a
learning-quality limit. Fresh normal and finished-wheel verification remain pending.
This smoke check sets no learning-quality score or baseline requirement.

## Shared inputs

The shared action downloads one versioned S3 archive and checks its SHA-256
before extraction. The archive contains the fixed synthetic PR172 tables and
public graph metadata. The workflow uses the repository's existing AWS role.
It needs no token for a private GitHub repository. No generated data archive or
1,000-line graph definition is committed here. See the
[fixture record](../../packages/kumo-engine/test/pr172/README.md).

## Agent-local feedback

Vedaant requested local feedback in strictly less than five minutes. Prepare
Python 3.12 dependencies once from the repository root. Setup is outside the
measured command:

```sh
python3.12 -m venv .venv-agent-local
.venv-agent-local/bin/python -m pip install -r test/scripts/agent-local-requirements.txt
```

Run the marked pure output checks:

```sh
.venv-agent-local/bin/python -I scripts/agent_local_tests.py
```

The command selects only `test/scripts/test_engine_gpu_smoke.py`. It disables
inherited pytest options and plugin loading. Its isolated configuration stops
parent fixture discovery. A parent process stops the suite at 300 seconds.
Failures, skips, empty or incomplete selection, timeouts, and elapsed time of
300 seconds or more return failure. The printed directory retains the pytest
log, counts, and elapsed time. The local suite requires no GPU or cloud access.
The complete command passed all nineteen selected cases with no skips in
0.70 seconds on macOS ARM64 with Python 3.12.13. Setup was excluded.

## Scope and timing

Source and package tests keep their existing workflows and failure policies.
This job neither waits for them nor replaces them. The previous observer,
reporting framework, and 39 additional installation/behavior fits are removed.
The existing package suite retains its ordinary behavior tests.

Twenty minutes remains the preferred full-CI target. The accepted budget is
about 22 minutes, with a strict under-25-minute guardrail. Those budgets include
queues, build, installation, input download, execution, and result upload.
The earlier 21m17s result belongs to the larger 47-fit implementation. It does
not establish the timing of this smaller job. GitHub job times and the short
GPU summary provide the new evidence. Nightly, weekly, and release policies
remain separate.

The three-cycle qualification took 37m55s overall. Its cold build took 14m16s,
dependency installation took 11m51s, and the GPU worker took 9m29s. This is not
a normal-run budget pass. The normal run still needs measurement with the
verified native cache and the cached parallel installer.
