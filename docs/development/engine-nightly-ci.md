# Engine nightly learning checks

The suite checks eight tasks on frozen synthetic 100M inputs from S3.
Each task has a normal case and a case with scrambled answers.
Each case builds a fresh snapshot and calls the public trainer once.
That call can try several candidate models through the task's suggested FAST
plan. A candidate is a model tried during training. Each case compares full,
repeated, partial, and reloaded predictions. Cases share only the built wheel
and frozen inputs. Each command selects one case and uses fresh scratch and
evidence paths.

## Execution and wheel identity

This PR supplies a command-line caller and CPU checks. It adds no nightly GPU
workflow or schedule. Pool scheduling and automation on the owner's AWS VM are
separate work. The
[permission review](https://github.com/NVIDIA/kumo-engine/pull/137#issuecomment-6585440181)
defines the deferred trust requirements. They include main-only credentials,
restricted VM access, an unprivileged candidate, protected caches, and bounded
instance lifetimes. This PR changes no AWS permissions or infrastructure.

From a clean repository checkout, build the selected product once. Put the
output outside that checkout. `build.json` binds the source revision, tree,
version, and wheel checksum. The caller reuses the existing Python wheel builder.
The unused composite action is removed. No `wheel.json` receipt is required.

```bash
python packages/kumo-engine/scripts/nightly_job.py build \
  --source "$PRODUCT_SOURCE" --output "$WHEEL_OUTPUT"
```

Install that wheel with its own requirements in a fresh Python 3.12 environment.
Run `pip check`. The harness compares the installed product files with the wheel
archive before fitting. The frozen dependency profile applies only to ordinary
reference execution. Resource measurement records the installed dependencies
for a new reference proposal. It does not require the old smoke-run profile.
Candidate dependency changes do not require a new input archive.
The runtime requires Linux x86_64 and an RTX PRO 6000. The reference campaign
uses one AWS g7e.8xlarge per case, with one Blackwell GPU, 32 CPUs, and 256 GiB
of RAM. Candidate-wheel qualification remains separate work.

## Frozen inputs and reference review

The [input descriptor](../../packages/kumo-engine/test/nightly/inputs.json) pins
the shared S3 archive in `kumo-unit-test`, in `us-west-2`, with a SHA-256 checksum.
Its checked-in reference and policy also have pinned checksums. The archive
contains data and metadata. It covers all eight tasks, both answer arms, full
prediction populations, and three graph schemas. Static-link uses its separate
static commerce view. The reader rejects changed files, missing rows, invalid
keys, and incomplete forecast coordinates. No DuckDB step generates the inputs.

The [reference proposal](../../packages/kumo-engine/test/nightly/reference.json)
contains sixteen complete executions of the approved Engine 1.0.0 wheel.
Training uses each task's public suggested FAST plan. The Engine chooses
candidates and checkpoints through validation. The harness does not tune the
plan or select a model from its test score.

The [campaign evidence](../../packages/kumo-engine/test/nightly/evidence/reference-campaign.json)
records quality, candidate counts, timing, calibration, and resource file hashes.
The policy and resource proposals use the original sixteen completed reference cases.
The [baseline provenance](../../packages/kumo-engine/test/nightly/evidence/fast-baseline-provenance.json)
records the training statistics and frozen input identities used for baselines.
The earlier [baseline audit](../../packages/kumo-engine/test/nightly/evidence/baseline-audit.json)
remains a historical record.
Evidence paths are relative to each case's saved artifact directory.
The archive checksum identifies the retained evidence. Private storage locations
stay outside the public descriptors.

Static-link normal has recall@5 `0.118145`. The unchanged popularity
baseline scores `0.009904`. Recall@5 measures the share of each
customer's true merchants recovered by five recommendations, averaged across
customers. The older limited-training score, `0.007665`, remains a failure
counterexample in the CPU checks. The random baseline does not replace the
popularity requirement.

Mean absolute error is the average absolute difference between prediction and
answer. Forecasting normal has error `0.334672`, below the stronger
always-zero baseline of `0.382274`. The normal baseline stays unchanged.
Normal error must improve on this baseline by at least ten percent.
The proposed `min_normal_gain` is `0.03822739835549215`. The saved normal model passes.
Default FAST trains with Huber loss, which penalizes errors differently from
mean absolute error. The scrambled model predicts nearly constant values.

The proposed forecast control baseline minimizes Huber loss on raw training answers,
independently for each of twelve horizons. It has test error
`0.544186`. The control model has error `0.512046`.
The scrambled-control closeness check remains required. Its absolute error
difference from this baseline must stay within ten percent of that baseline.
The proposed `max_control_drift` is `0.05441856445482517`.
The observed difference, `0.032140`, passes this proposed rule.
These two ten-percent limits are policy choices, not statistical guarantees.
The [forecasting policy proposal](../../packages/kumo-engine/test/nightly/evidence/forecasting-policy-proposal.json)
records the formulas, saved scores, and previous limits.

The saved model applies `log(1 + answer)` before Huber loss.
The proposed raw-answer baseline therefore uses a different objective.
The constant for the encoded objective has test error `0.525629` and remains diagnostic evidence.
Reviewers must assess the proposed raw-answer comparator and both limits before reference approval.
The initial zero-baseline failure remains in the campaign evidence.
Other task limits and prediction tolerances stay unchanged.

The [policy proposal](../../packages/kumo-engine/test/nightly/policy.json) also
calibrates numerical comparisons from observed repeated, partial, and reloaded
predictions. Each absolute limit covers twice the largest observed score or
decision-gap difference, rounded upward to two significant digits. Existing
larger limits stay unchanged. Relative tolerance stays zero. These values are
observed headroom, not confidence intervals or guarantees across future runs.

All sixteen cases pass the proposed numerical checks. Normal models
meet their positive baseline margins and beat their paired scrambled controls.
The controls meet their proposed baseline limits. Original failed attempts and
comparisons remain in the retained evidence. A completed measurement alone does
not qualify a reference.

A reviewer must approve the baseline methods, bounds, complete reference evidence,
and measured resource declarations before the reference is approved. Change the
pinned checksums through review. A successful run does not publish or approve a
reference. The current files explicitly remain proposals.

## Independent cases and resource declarations

The [resource proposal index](../../packages/kumo-engine/test/nightly/resource-proposals.json)
links each of sixteen measured cases to its proposed manifest and SHA-256.
These declarations remain proposals until review.

Each qualification case requires a reviewed resource manifest and its checksum.
The manifest contains `minimum`, `measurement`, and `headroom`. It declares CPU
cores, RAM bytes, scratch bytes, GPU class and count, GPU memory bytes, and seconds.
The harness derives minima from measured peaks with 25 percent headroom. It
rejects incomplete evidence, changed minima, and evidence from another case or
input manifest. Startup compares available capacity with each minimum before
building the snapshot. A timer enforces the case time budget.

Use the installed candidate environment for the following command. Set
`INPUT_ROOT` to the extracted frozen archive. Set `CASE_NAME` to a task and answer
arm, such as `binary-normal`. Set `CASE_OUTPUT` and `CASE_SCRATCH` to fresh paths.
Set `CASE_REQUIREMENTS` and `REQUIREMENTS_SHA256` to the reviewed manifest and hash.
Keep scratch outside both inputs and evidence.

```bash
python packages/kumo-engine/scripts/nightly_job.py run \
  --build-receipt "$WHEEL_OUTPUT/build.json" --wheel "$CANDIDATE_WHEEL" \
  --input-root "$INPUT_ROOT" --case "$CASE_NAME" \
  --output-dir "$CASE_OUTPUT" --scratch "$CASE_SCRATCH" \
  --requirements "$CASE_REQUIREMENTS" \
  --requirements-sha256 "$REQUIREMENTS_SHA256"
```

The caller derives version and data checksums from the build receipt and input
descriptor. Case flags cannot override those bindings. The artifact helper's
`create` and `prepare` commands can stage and extract the archive. A caller supplies
`RUNNER_TEMP`, `NIGHTLY_CONFIGURATION`, and `GITHUB_ENV` for those commands. They do
not create a VM, submit a job, or supply credentials.

If no time override is supplied, a qualification case uses its declared time
limit. A caller with a shared deadline must pass the remaining time through
`--time-budget-seconds`.

When no measured manifest exists, use `--measure-resources` instead of the two
requirements flags. Also supply an explicit `--time-budget-seconds`. This mode
still performs real case work and needs separate execution approval. It emits
`requirements-proposal.json` and status `MEASURED`. It never reports qualification
as `PASS`, and it never approves its own declaration.

The resource observer samples process-tree RAM, CPU use, owned scratch and output
files, and device memory once per second. It includes snapshot construction and
records duration, wheel identity, harness identity, and input identity. Failed or
invalid observations remain `INCOMPLETE`. Measurements are sampled peaks, not
instantaneous maxima. RAM is the sum of process memory reports. Shared pages
can count more than once, so the proposed RAM requirement is conservative.
The declarations add 25 percent to measured peaks and duration. GPU class and
count record the tested hardware.

The campaign uses prepared VMs, installed environments, and frozen inputs.
It skips a cold wheel build and shared environment setup. Its overall elapsed
time includes retries and evidence collection. These measurements do not prove
that a cold nightly build, setup, and all cases fit the 180-minute budget.

## Learning and prediction gates

Normal gain must meet the task's positive `min_normal_gain`. Gain means score
minus baseline for higher-is-better metrics. For regression and forecasting,
it means baseline error minus model error. The scrambled score's absolute distance
from its own baseline must stay within `max_control_drift`. Missing or invalid
scores fail. A model with no gain or a control that learns too much fails.
Normal quality must also exceed the paired control by `min_normal_gain`.
This prevents a control that matches normal learning from qualifying inside
the separate baseline limits.
An individual case result does not qualify the whole suite. Combine all sixteen
results and apply the paired learning checks before declaring suite success.

Every prediction must return all expected entities with finite coordinates.
Numeric differences must meet the task's absolute `atol` bound by class ID.
Relative tolerance is zero. Non-link tasks use the shared Engine PR comparison.
Binary and multilabel decisions change only near their threshold. Multiclass
choices and reversed ranking pairs must meet `atol` in both outputs. Changed
top-K membership needs retained cutoff coordinates on both sides. The reader
retains up to 2K eligible scores to cover multiple swaps. A comparison fails
when those scores cannot prove every changed membership.
These gates apply to repeated, partial, and reloaded predictions.
Resource measurement mode retains every numerical comparison and its failure
status. It still rejects invalid output coordinates. A `MEASURED` result
records a complete observation and does not approve the reference or its limits.

The harness uses public training results, artifact paths, saved bundles, and
Lightning callbacks. It compares reloaded weights with the selected checkpoint.
It does not search private fit directories or replace Engine fit methods.
Training uses seed zero and each task's public suggested FAST plan. The Engine
selects candidates and checkpoints through its default validation process.
The harness records actual candidate batches, validation histories, stopping
callbacks, selected checkpoint bytes, and requested plan hashes.
The frozen input signature retains generator metadata from the archive.
`requested_plan_sha256` identifies the actual FAST execution plan separately.
The harness supports multiple candidates, variable batch sizes, and early stopping.
The selection metric can differ from the reported nightly quality metric.
Classification calibration can rewrite a candidate's checkpoint after fitting.
The observer records its final hash after the candidate's validation step.
Prediction keeps default product kernels and OOM recovery. A successful
out-of-memory recovery stays valid.
`result.json` records allocator OOM counter increases under `work.oom_recoveries`.
The harness does not match log strings to detect recovery. A failure before output
creates a failed case result. Other cases retain their results and continue.

## Reader evidence and cleanup

PyArrow reads Parquet in batches. Temporary sorting merges at most 32 inputs at
a time. The
[retained 88M-row measurement](../../packages/kumo-engine/test/nightly/evidence/reader-88m.json)
reports 586 seconds, about 9.8 minutes, for one synthetic prediction stream.
Peak process RSS was 3,184,672,768 bytes. Peak owned disk was 5,031,972,864 bytes.
Ordering, row count, checksum, and temporary-file cleanup passed. The streaming
sorter remains unchanged. This is CPU reader evidence, not complete-case or
GPU qualification. The evidence does not establish a complete 180-minute nightly budget.

The standalone
[reader benchmark](../../packages/kumo-engine/benchmarks/nightly_reader.py)
accepts `--rows` and `--output`. The default CPU suite uses a 512-row correctness
test across 64 files. The eight-million-row stress workload is outside that suite.
[Issue 162](https://github.com/NVIDIA/kumo-engine/issues/162) tracks the retained
full-versus-partial score difference. Its diagnostic used deterministic kernels,
so it does not qualify execution with current product defaults.

The artifact helper checks free space before work creation and archive extraction.
Its 256-GiB allowance is a proposal, separate from measured case requirements.
The harness retains model bundles, predictions, and case results. Preserve the
process log with them. Cleanup removes only marked, owned work. It removes
evidence only after the caller confirms successful upload. This PR supplies no
automatic uploader or retention schedule. Original S3 objects and earlier evidence
remain intact.
