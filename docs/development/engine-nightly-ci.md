# Engine nightly learning checks

The nightly suite checks eight tasks on the frozen synthetic 100M inputs.
Each task has one normal fit and one fit with scrambled answers, for sixteen
fits total. The same models supply full, repeated, partial, and reloaded
predictions. The suite requires real CUDA work and useful learning.

## Execution and permissions

The workflow accepts pushes to `main` and NVIDIA's approved
`pull-request/<N>` branches. The copied commit must still match the PR head.
It has no cron or arbitrary-branch manual trigger. Main GPU execution requires
`KUMO_ENGINE_NIGHTLY_ENABLED=true`; leave that variable unset until qualification.
The copied-PR path supplies the full qualification run after approval.

Public GitHub-hosted jobs select the commit and build its wheel without cloud
credentials. The GPU job uses NVIDIA's `nv-gpu-amd64-rtxpro6000-1gpu` group and
`gha-kumo-workflow-tests` environment, as in the Engine PR check. Only the data
download step receives AWS credentials. It reads the finished wheel from the
build artifact and the immutable data objects from shared S3 storage.

The job installs the candidate wheel with its own requirements in a fresh
environment. It runs `pip check` and compares installed wheel files with the
finished archive once before fitting. The frozen dependency profile applies
only to reference execution. Candidate dependency additions and upgrades do not
require a new input archive.

## Inputs, reference, and policy

The [input descriptor](../../packages/kumo-engine/test/nightly/inputs.json)
selects shared objects under `diskgraph-profiles/engine-nightly/100m/v1/` in
`kumo-unit-test`, in `us-west-2`. Each object has a SHA-256 checksum. The descriptor
also binds the input manifest and required free disk space.
`KUMO_ENGINE_NIGHTLY_CONFIG` can override the descriptor with the same schema.

The input archive contains data and metadata, not executable workers. It covers
all eight tasks, both answer arms, their full prediction populations, and the
three source graph schemas. Static-link uses its separate static commerce view.
The runner rejects changed files, missing rows, invalid keys, and incomplete
forecast coordinates.

The reference records baseline scores for every task and answer arm. The
[policy proposal](../../packages/kumo-engine/test/nightly/policy.json) supplies
separate learning and prediction bounds for every task. Review those bounds,
the complete run, and the reference before changing the pinned checksums.
A run does not approve or publish a new reference.

The learning proposal uses ten percent of each baseline's distance to the nearer
score boundary. For an error metric, it uses ten percent of the baseline error.
The normal margin and control allowance use their respective baselines.
These are explicit proposed margins, not measured confidence intervals.
Prediction bounds retain the PR check's `0.0001` absolute tolerance except for
static-link. Its proposed `0.0004` bound covers the retained partial-prediction
difference of `0.000322878360748291` with about 24 percent headroom.

The current historical reference does not satisfy all proposed learning bounds.
Its static-link normal score is below its baseline. Its forecasting control
also differs substantially from its baseline. The temporal-link control also
exceeds the proposed drift bound. These remain failures. The
proposal does not weaken the bounds to accept them, and no full run on NVIDIA
runners qualifies the proposal yet.

## Required behavior

For each task, normal gain must meet `min_normal_gain`. Gain means score minus
baseline for higher-is-better metrics. For regression and forecasting, it means
baseline error minus model error. The scrambled control's absolute difference
from its own baseline must stay within `max_control_drift`. Missing or invalid
scores fail. A model with no gain or a control that learns too much fails.

Every prediction call must complete the expected entities and finite coordinates.
Numeric differences must meet the task's absolute `atol` bound by class ID.
Relative tolerance is zero. Non-link tasks use the shared Engine PR comparison.
Binary and multilabel decisions can change only near their threshold.
Multiclass decisions and reversed ranking pairs must meet `atol` in
both outputs. A top-K membership change also requires retained K+1 cutoff
coordinates on both sides. Missing cutoff evidence or a wider change fails.
These checks apply to repeated, partial, and reloaded predictions.

The runner uses public training results, artifact paths, saved bundles, and
Lightning callbacks. It verifies that the reloaded weights match the selected
checkpoint. It does not search private fit directories or replace Engine fit
methods. Training uses seed zero and the frozen model recipe. Prediction uses
the product's default kernels and recovery behavior.

Successful out-of-memory recovery does not fail the suite. Normal product logs
retain recovery warnings. The harness does not parse those strings to decide
success. Complete CUDA work, valid outputs, and the learning gates remain
required. Later cases retain their results when an earlier case fails.

## Capacity, evidence, and cleanup

Before installation or extraction, the job checks free disk space. The proposed
256-GiB allowance is operational headroom, not a measured requirement. The job
also records CPU count and system memory. If the workload does not fit the
managed runner, raise its measured capacity requirements on PR42. Do not add a
separate VM transport.

The CPU build has a 60-minute limit. The GPU job has a 180-minute limit and
reserves time for artifact upload and cleanup. Setup, graph preparation, all
sixteen cases, and evidence handling need one complete measured run before the
suite can be scheduled. Per-case samples cannot prove that deadline.

The result artifact contains logs, model bundles, canonical predictions, and
raw outputs. It is retained for fourteen days. Each job creates its own marked
temporary directory. After upload, cleanup removes that job's environments,
installer cache, generated indexes, scratch, and local evidence. An upload
failure preserves the evidence directory while temporary execution work is
removed. Shared inputs, original references, and earlier evidence stay intact.

Elapsed times remain observations. The suite has no slowdown gate, nested
process-memory samplers, or historical performance percentages. It keeps
functional and learning failures separate from unmeasured performance claims.
