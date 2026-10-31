# Engine nightly on the fixed AWS VM

The manual workflow builds the selected wheel on a CPU GitHub runner.
A separate CPU job stages it and the trusted harness in S3, then controls
`i-06dd0e3c558780731` in `us-east-2` through Systems Manager Run Command (SSM).
The VM has no GitHub runner. Scheduling remains disabled.
The workflow publishes no release, wheel, or reference.

## Execution and cancellation

The build job has no AWS role or OIDC permission. It uses the existing wheel
builder and records the exact revision, tree, version, and wheel hash.
The controller job uses the protected test environment. Its temporary AWS
credentials stay in the controller step. It renews expired credentials through
[GitHub OIDC](https://docs.github.com/en/actions/reference/security/oidc)
without changing IAM or forwarding controller credentials to the VM.

The request binds the run ID, attempt, nonce, source commit, wheel, three code
files, and frozen inputs. The VM checks the request and payload hashes.
Each run gets a fresh environment and result directory. Reuse immutable input
files only after their identities pass. Keep models and indexes remote.
JSON reports and logs return through S3 and GitHub artifacts on success or failure.

The selected `g7e.12xlarge` has 48 CPUs, 512 GiB nominal RAM, local NVMe, and
two RTX PRO 6000 GPUs. Only GPU0 is visible to model execution.
The host records actual capacity and rejects a busy host without stopping its
existing workload. It holds one lock across installation and model execution.
The unprivileged subprocesses inherit that lock. Cancellation stops their whole
process group before the host releases it. A systemd deadline bounds the group
independently of CI. The model receives a clean environment, separate home,
and short private temporary directory, with no personal AWS configuration.

The CI role lacks `CancelCommand`. The controller instead uploads a cancellation
marker bound to the exact request and waits for the same invocation to end.
The host polls the marker during setup and model execution.
A separate cleanup job repeats cancellation and confirms termination when the
controller fails or GitHub cancels its step. It can also use a bound terminal VM
receipt if dispatch returned no command ID. The cleanup job allows eight minutes.
Unconfirmed termination remains an error. The host lock and deadline stay in force.

The host allows 180 minutes for setup and execution. The service deadline is
185 minutes. SSM allows 11,200 seconds, and the controller waits 11,300 seconds
before requesting cancellation. The controller job allows 195 minutes.
The separate CPU build allows 60 minutes. These are provisional bounds based
on the existing three-hour job allowance, not a measured complete runtime.
The workflow does not stop, resize, or provision the VM.

## Workload and required proof

Run sixteen sequential primary fits: eight tasks with seed zero, normal labels,
and independently scrambled controls. Keep the full approximately 100M-row
adtech, delivery, and commerce inputs. Do not run the eight small PR fits first.
The shared PR action and helper remain unchanged from PR135.
Installation receipts, resource sampling, saved-state proof, source mappings,
and numerical reporting belong to nightly-owned code.

Require actual CUDA work, finite complete outputs, valid IDs, exact restored
checkpoint state, and unchanged decisions and ranks after repeat, partial
prediction, and reload. Keep product OOM recovery and record actual batches
and early stopping. Failed assertions and missing measurements invalidate results.
Historical metadata and score replay cannot satisfy a new sixteen-case run.

CUDA synchronization surrounds timed operations. Record preparation, training,
prediction, repeat, partial, reload, and complete-case scope separately.
Record controller CLI time in `controller-job-time.json`, including staging,
waiting, collection, and validation. Host elapsed time covers setup through
process cleanup and excludes final packaging and upload. GitHub records the
whole workflow and separate build duration. Full-fit time stays separate from
the completed-batch training window.
Resource records contain sampled process-tree PSS/RSS, device memory used,
Torch allocated/reserved memory, and sampling interval and errors.
PSS apportions shared pages. RSS counts shared pages in each process.
These sampled values do not guarantee instantaneous peaks.
Missing CPU or GPU measurements invalidate a primary measurement.

The nightly callback reuses Group 5's clock for twenty completed warmup batches
followed by two hundred measured batches. It synchronizes CUDA at both boundaries
and records examples per second, optimizer updates, process-tree memory, and
Torch allocation peaks. Batch counts continue across epoch boundaries.
Two optimizers still count as one completed training batch.
The interval includes validation and callback work between those boundaries,
including the existing per-batch CUDA proof. Later quality work cannot extend it.

The callback does not change epochs, early stopping, batch size, or sampler settings.
The frozen nightly recipe permits at most 200 batches and can stop earlier.
Before graph creation or fitting, the runner records each plan's maximum batch
budget. Unknown budgets or fewer than 220 possible batches stop the run.
The current recipe is short by at least twenty batches. It cannot start another
nightly fit until a compatible workload receives review.
Retained short fits record `INCOMPLETE` and no window throughput. A reduced batch,
retry, skipped optimizer update, or missing resource measurement invalidates
the window. These checks also inspect work after batch 220.
The comparison reports `steady_state_training_status: INVALID` when current
valid measurements are unavailable. It keeps `functional_status: PASS` for
completed functional checks. Missing measurements or incompatible candidate
references produce overall `status: INVALID` and a failing command after the
diagnostic report is saved. The CPU controller also requires valid windows in
all sixteen case records before it can print `Nightly PASS`.
Reference-only work can prepare a proposed reference from valid current
measurements without a matching older performance reference. Invalid current
measurements cannot produce a proposed reference.
A compatible 220-batch workload and actual GPU qualification remain open.

## Reporting and references

Normal-model quality reports a 5% relative degradation margin.
Training/prediction time, CPU memory, and GPU memory report 20% margins.
Crossing a margin does not fail CI, merge, or publication.
Normal gain, scrambled controls, and numerical score differences remain diagnostic.
There is no absolute learning-quality gate. The PR smoke keeps its own enforced
prediction tolerance.

Percentages require matching inputs, model configuration, dependencies,
hardware, driver, cache conditions, scope, and actual training work.
Batch reduction, missing work records, or different epoch counts cannot qualify
a timing comparison. The recorded work also identifies the completed-batch
clock protocol. Earlier full-fit measurements without that callback cannot
qualify percentages against runs with its additional sampling work.
Missing, zero, invalid, or incompatible references retain
raw values and absolute differences. Their percentages remain unavailable.

The [descriptor](../../packages/kumo-engine/test/nightly/inputs.json) pins the
unchanged archive, manifest, and predecessor reference by SHA-256.
Its historical `T4` label does not select hardware.
The approved wheel is `kumo-engine==1.0.0` at its recorded hash.
Reference installation uses frozen dependencies, including Lightning 2.6.5.
Candidate installation resolves the selected wheel's actual requirements,
including its Lightning requirement. Run `pip check` and check installed files.
Record all resolved versions. Two historical loader workers do not qualify
performance comparisons with twelve current workers.

Manual `reference_only: true` uses the approved wheel. After complete valid
execution, comparison prepares `summary/reference.json` without publishing it.
Reference admission requires review. Preserve every predecessor and descriptor.

## Reuse audit and qualification gaps

Audit each retained case before another fit. Match inputs, wheel, dependencies,
model settings, batch configuration, checkpoint, actual work, and predictions.
Reuse matching predictions for scores and compatible saved models for missing
prediction checks. Locate old artifacts before deciding on training.
Preserve every failed and interrupted result. This review procedure does not
change the routine sixteen-case nightly roster.

Twelve retained cases support score replay without another fit.
The saved static-link normal model passed repeat and partial checks on GPU.
Reload swapped positions four and five for one of 176,856 entities.
The same five items, recall@5, and checkpoint identity remain unchanged.
The strict order check remains FAIL. Its cause is not established.
Static-link control and both temporal-link cases still lack located and
verified historical models and raw predictions.

The separate [wheel-delivery run](https://github.com/NVIDIA/kumo-engine/actions/runs/41146970366)
passed CPU build, transfer, isolated installation, and imports with zero fits.
It does not verify this full SSM model-suite route or its cancellation path.
Local CPU checks do not establish a full current GPU PASS.
Keep the PR draft and scheduling disabled while these gaps remain.
