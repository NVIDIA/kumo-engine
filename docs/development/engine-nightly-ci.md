# Engine nightly learning checks

The suite checks eight tasks on frozen synthetic 100M inputs from S3.
Each task has a normal case and a case with scrambled answers.
These sixteen cases each build a snapshot, fit once, and compare full, repeated,
partial, and reloaded predictions. Cases share only the built wheel and frozen
inputs. Each command selects one case and uses fresh scratch and evidence paths.

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
archive before fitting. The frozen dependency profile applies only to reference
execution. Candidate dependency changes do not require a new input archive.
The runtime requires Linux x86_64 and an RTX PRO 6000. AWS target qualification
remains open.

## Frozen inputs and reference review

The [input descriptor](../../packages/kumo-engine/test/nightly/inputs.json) pins
the shared S3 archive in `kumo-unit-test`, in `us-west-2`, with a SHA-256 checksum.
Its checked-in reference and policy also have pinned checksums. The archive
contains data and metadata. It covers all eight tasks, both answer arms, full
prediction populations, and three graph schemas. Static-link uses its separate
static commerce view. The reader rejects changed files, missing rows, invalid
keys, and incomplete forecast coordinates. No DuckDB step generates the inputs.

The [reference proposal](../../packages/kumo-engine/test/nightly/reference.json)
retains the historical 1.0.0 scores, fit identities, and checkpoint hashes.
It records original baselines beside the proposed baselines. The
[baseline audit](../../packages/kumo-engine/test/nightly/evidence/baseline-audit.json)
uses existing frozen data and results. It starts no new fits or model predictions.

Forecasting previously compared against repeating the last observed value.
That predictor has error `0.659099` on scrambled answers. An always-zero count
predictor has lower error, `0.382274`, on both answer arms. The scrambled model's
error is `0.385228`, only `0.002954` away. The proposal uses this stronger fixed
baseline and keeps the existing forecasting bounds. The normal model also passes.

The [policy proposal](../../packages/kumo-engine/test/nightly/policy.json) sets
static-link control drift to `0.00021` and temporal-link control drift to `0.0046`.
They cover retained differences of `0.000163982` and `0.003657920` with headroom.
These are calibration proposals from one historical run, not confidence intervals
across seeds. Other normal learning margins remain positive.

Static-link normal still fails. Its score, `0.007665`, is below the original
popularity baseline, `0.009904`. The weaker random baseline appears only as a
diagnostic. It does not replace the popularity gate. This remains a merge blocker.

A reviewer must approve the baseline methods, bounds, complete candidate evidence,
and measured resource declarations before the reference is approved. Change the
pinned checksums through review. A successful run does not publish or approve a
reference. The current files explicitly remain proposals.

## Independent cases and resource declarations

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

When no measured manifest exists, use `--measure-resources` instead of the two
requirements flags. Also supply an explicit `--time-budget-seconds`. This mode
still performs real case work and needs separate execution approval. It emits
`requirements-proposal.json` and status `MEASURED`. It never reports qualification
as `PASS`, and it never approves its own declaration.

The resource observer samples process-tree RAM, CPU use, owned scratch and output
files, and device memory once per second. It includes snapshot construction and
records duration, wheel identity, harness identity, and input identity. Failed or
invalid observations remain `INCOMPLETE`. Measurements are sampled peaks, not
instantaneous maxima. Historical records lack complete independent-case CPU and
scratch peaks. Reviewed declarations for all sixteen cases remain open. This PR
does not invent those values or claim a complete runtime budget.

## Learning and prediction gates

Normal gain must meet the task's positive `min_normal_gain`. Gain means score
minus baseline for higher-is-better metrics. For regression and forecasting,
it means baseline error minus model error. The scrambled score's absolute distance
from its own baseline must stay within `max_control_drift`. Missing or invalid
scores fail. A model with no gain or a control that learns too much fails.

Every prediction must return all expected entities with finite coordinates.
Numeric differences must meet the task's absolute `atol` bound by class ID.
Relative tolerance is zero. Non-link tasks use the shared Engine PR comparison.
Binary and multilabel decisions change only near their threshold. Multiclass
choices and reversed ranking pairs must meet `atol` in both outputs. Changed
top-K membership needs retained K+1 cutoff coordinates on both sides.
These gates apply to repeated, partial, and reloaded predictions.

The harness uses public training results, artifact paths, saved bundles, and
Lightning callbacks. It compares reloaded weights with the selected checkpoint.
It does not search private fit directories or replace Engine fit methods.
Training uses seed zero and the frozen recipe. Prediction keeps default product
kernels and OOM recovery. A successful out-of-memory recovery stays valid.
`result.json` records allocator OOM counter increases under `work.oom_recoveries`.
The harness does not match log strings to detect recovery. A failure before output
creates a failed case result. Other cases retain their results and continue.

## Reader evidence and cleanup

PyArrow reads Parquet in batches. Temporary sorting merges at most 32 inputs at
a time. The
[retained 88M-row measurement](../../packages/kumo-engine/test/nightly/evidence/reader-88m.json)
reports 586 seconds, about 9.8 minutes, for one synthetic prediction stream.
Peak process RSS was 3,184,672,768 bytes. Peak owned disk was 5,031,972,864 bytes.
Ordering, row count, checksum, and temporary-file cleanup passed. The reader
function remains unchanged. This is CPU reader evidence, not complete-case or
GPU qualification. The evidence does not establish a six-hour suite budget.

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
