# Engine nightly on the fixed AWS VM

The manual workflow builds the selected wheel on a CPU GitHub runner.
A separate CPU job stages it in S3 and controls the fixed AWS VM through
Systems Manager Run Command (SSM). Scheduling remains disabled.
The workflow publishes no release, wheel, or reference.

## Execution and cancellation

The build job has no AWS role or OIDC permission. It uses the existing builder
and records the selected revision, tree, version, and wheel checksum.
The controller uses the protected test environment and renews expired temporary
AWS credentials through [GitHub OIDC](https://docs.github.com/en/actions/reference/security/oidc).
It does not change IAM or forward its credentials to the VM.

The request binds the run ID, attempt, nonce, source, wheel, code, and inputs.
The VM checks their hashes and creates a fresh environment and result directory.
Immutable input files can be reused after their checksums pass.
Models and indexes stay remote. Reports and logs return through S3 and GitHub
artifacts on success or failure.

The selected VM is `i-06dd0e3c558780731` in `us-east-2`.
Its `g7e.12xlarge` configuration has 48 CPUs, 512 GiB nominal RAM, local NVMe,
and two RTX PRO 6000 GPUs. Model execution uses GPU0 only.
The VM has no GitHub runner. The workflow does not stop, resize, or provision it.

GitHub concurrency is scoped per ref. A separate exclusive host lock protects
the shared GPU machine. A busy host fails admission without stopping its job.
The unprivileged model process receives an isolated home and no personal AWS
configuration. Its subprocesses inherit the lock until their whole process
group stops. A systemd deadline bounds execution independently of GitHub.

The CI role lacks `CancelCommand`. Cancellation uploads a marker bound to the
exact request. The host polls it during installation and execution.
A separate cleanup job repeats cancellation after controller failure and
confirms termination. Unconfirmed termination remains an error.

The build allows 60 minutes. The host allows 180 minutes, the service 185,
the controller 195, and cleanup eight. These are provisional bounds, not a
measured full runtime. Keep the schedule off until a full run is measured.

## Sixteen cases and measurements

Run eight tasks with seed zero, each with normal labels and scrambled controls.
Each case has one primary fit. Use the full frozen approximately 100M-row
adtech, delivery, and commerce inputs. Do not add the eight small PR fits.
The shared PR action and helper remain unchanged from PR135.

Require actual CUDA execution, finite complete predictions, valid IDs,
unchanged labels and ranks, and restored checkpoint state after reload.
Keep product OOM recovery. Record batch counts and actual early stopping.
A recovered reduced-batch operation cannot supply a valid timed case.
Missing work or resource measurements invalidate the result.

Use PyTorch's strict deterministic algorithms and set
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization. Each Lightning
trainer also requests deterministic execution. Unsupported operations fail
instead of silently using a nondeterministic kernel. Record these runtime
settings; different or missing settings exclude performance percentages.
The wheel, model recipe, inputs, seed and batch setting remain unchanged.

The same sixteen fits supply preparation, training, prediction, repeat,
partial-prediction, reload, and complete-case times. Synchronize CUDA at
operation boundaries. Record process-tree PSS/RSS, device memory used,
and Torch allocated/reserved memory with the sampling interval and errors.
PSS apportions shared pages; RSS counts them in each process.
Sampled peaks do not guarantee the instantaneous maximum.

Record total controller time separately, including staging, waiting, collection,
and validation. Host time covers setup through process cleanup and excludes
final packaging and upload. GitHub records build and workflow duration.

Nightly uses the frozen training recipe, including its existing early stopping.
It does not require Group 5's twenty warmup and two hundred measured batches.
Whole-fit times are not steady-state training windows. Group 5 calibration and
repeated same-wheel pairs are separate work. They add no routine nightly fits.

## Reporting and references

Normal-model quality reports a 5% relative degradation margin.
Time, CPU memory, and GPU memory report 20% margins. Crossing these margins
does not fail CI or publication. Normal gain, scrambled controls, and numerical
score drift remain diagnostic. Labels and ranks remain strict.

A successful nightly requires sixteen completed cases and valid current
measurements. Historical metadata or score replay cannot replace that proof.
Comparison also checks the installed wheel, source, input hashes, and artifacts.

Performance percentages require matching inputs, configuration, dependencies,
hardware, driver, cache conditions, measurement scope, and actual work.
Missing or incompatible references leave percentages unavailable. Reports keep
raw values and explain the differences. A routine driver change alone does not
fail the functional nightly. This does not qualify incompatible performance
comparisons. Both actual dependency fingerprints remain visible.

The [descriptor](../../packages/kumo-engine/test/nightly/inputs.json) pins the
unchanged archive, manifest, and predecessor reference. Its historical T4 label
does not select hardware. The approved wheel is `kumo-engine==1.0.0` at its
recorded checksum. Install the checksum-bound dependency profile for each role,
then the exact wheel without dependency resolution. Run `pip check` and verify
installed files before fitting. Candidate pins include its Lightning requirement.
Do not add optional test packages to model environments.

The reference declares the Lightning 2.6.5 to 2.6.6 pair. Other dependency
changes remain incompatible for percentages. Manual `reference_only: true`
can prepare a proposed reference from a complete valid run. It never publishes
that reference. Admission requires review and preservation of its predecessor.

## Reuse and current gaps

Audit each retained case before another fit. Reuse compatible predictions for
scores and saved models for missing inference checks. Keep exact checkpoint
identities and known configuration differences. Do not default to another full
reference and candidate training run. This audit does not change the routine
sixteen-case nightly roster.

Twelve retained cases support score replay. The saved static-link normal model
passed repeat and partial checks, but reload reordered two positions for one
of 176,856 entities. The strict order assertion remains FAIL.
An isolated GPU replay reproduced the same swap without reloading the model.
With public deterministic PyTorch algorithms, the saved model then passed
full repeat, partial and reload rank checks on the original inputs, with zero
new fits. Reload verified 303 restored tensors. The three checks took 345.30
seconds and processed 88,428,000, 44,214,000 and 88,428,000 output rows.
Repeat and reload scores matched exactly. Partial scores differed by at most
`0.000322878360748291`, above the compact PR check's `1e-4` bound.
Keep that difference visible under nightly's existing reporting policy.
These checks compare the same new runtime protocol. They preserve the earlier
historical-protocol FAIL and do not establish a full sixteen-case workflow PASS.
Static-link control and both temporal-link cases lack located and verified
historical models and raw predictions. Preserve these gaps and every failure.

The separate [wheel-delivery run](https://github.com/NVIDIA/kumo-engine/actions/runs/41146970366)
passed build, transfer, installation, and import with zero fits.
It does not prove this full model-suite route. Keep the PR draft and scheduling
disabled until the remaining GPU checks and review are complete.
