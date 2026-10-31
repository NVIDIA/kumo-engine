# Engine nightly learning check

The workflow runs manually on one authorized RTX PRO 6000 runner and GPU0.
It has no schedule or push trigger. Each run has a three-hour timeout.
Concurrency is separate for each workflow ref.
Keep scheduling disabled until a complete reference and candidate run passes.

The workload has sixteen sequential fits. Eight tasks each use normal labels
and independently scrambled labels with seed zero. It uses the existing full
PR172 approximately 100M-row inputs and the shared model recipe.
This change does not reduce the data or create a machine.

## Required results

Require actual CUDA training and prediction, complete finite outputs, exact
restored checkpoint state, and unchanged labels and ranks after repeat,
partial-input prediction, and reload. Keep the product's prediction and
evaluation recovery when GPU memory runs out.
Record partial training batches and actual early stopping without requiring
an exact batch size or patience value.

Each normal model must beat its frozen baseline. Each scrambled model must
remain near its own frozen baseline. The baseline is the existing simple
predictor recorded in the reference, not the reference model's learned score.
AUROC, accuracy, average precision, and recall range from zero to one.
For these metrics, use 10% of the smaller of the baseline and its remaining
distance to one. Require at least that normal gain and at most that control
difference. This keeps a reachable margin for high baselines and small recall.
For mean absolute error, require at least 10% lower normal error and control
error within 10% of its baseline. Mean absolute error measures prediction error.
These initial thresholds still need validation on a complete GPU run.
A failed learning condition writes a failed comparison and fails the job.

Numeric score drift and elapsed times remain reports without regression limits.
The shared PR smoke keeps its enforced prediction bound. Nightly selects numeric
reporting explicitly. Public Lightning callbacks retain CUDA and checkpoint proof.
Driver, Torch/CUDA, GPU memory, and resolved model configuration are recorded.
They do not need exact equality with the reference. GPU class remains required.
Package profiles have no fixed package count. The candidate installer resolves
the selected wheel's actual requirements. It runs `pip check` after installation.
Candidate dependency changes from the historical profile are recorded, not rejected.

## Inputs and reference

The [descriptor](../../packages/kumo-engine/test/nightly/inputs.json) pins the
existing S3 archive, manifest, and predecessor reference by SHA-256.
The historical `T4` label in input metadata does not select the runtime GPU.
Input bytes stay unchanged. No new data archive is needed for driver updates.
The approved reference wheel is `kumo-engine==1.0.0` with its recorded hash.
Use each wheel's supported default device selection and require observed CUDA.
The reference profile uses Lightning 2.6.5. The historical candidate profile
uses 2.6.6. Reference preparation keeps its profile pins.

Use manual input `reference_only: true` to prepare a reference on the selected
runner. After all sixteen cases and learning conditions pass, retain
`nightly-summary/reference.json` from the uploaded artifact.
Store that file at an immutable S3 key through the normal approved publishing
process. Update `reference_key` and `reference_sha256` in the descriptor.
The workflow itself does not publish or replace the reference.
An ordinary candidate run requires that reviewed reference and matching inputs.

## Scope and current evidence

Reuse the existing wheel builder, installer, CUDA observer, and output checks.
Keep the weighted-table and isolated model-reuse tests in a separate local
change. The CUDA reuse test skips on CPU in that separate change.
The nightly does not rerun those package tests or add their model fits.
It uploads results directly and excludes graph work directories.

The complete managed-runner workload, available capacity, and elapsed time
remain unverified. Local CPU tests do not qualify the GPU workflow.
Keep this change local and keep scheduling disabled until that evidence exists.
