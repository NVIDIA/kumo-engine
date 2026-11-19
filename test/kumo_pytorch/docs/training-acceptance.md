# CPU training acceptance

The shared runner applies the same training criteria to every enabled binding.
HI Small retains all five original model/workload combinations:
This extends criterion 8's original Fusion/a001 and PRAGMA/a017 requirement at
the user's request. Every combination has the same acceptance rules.

| Workload | Consumer | Anchors / split |
| --- | --- | --- |
| Account involvement | GraphSAGE | v2 / forecast_split |
| Account involvement | nuFormer | v1 / forecast_split |
| Account involvement | Fusion/combined (a001) | v2 / forecast_split |
| Masked-value pretraining | PRAGMA (a017) | v1 / split |
| Next-token pretraining | nuFormer (a018) | v1 / split |

All preparation uses DiskGraph. Graph cases retain the shifted v2 cutoff;
sequence-only classification retains its external involvement labels. Each task
uses its own independent binary, masked-value or next-token loss check.
Acceptance does not freeze or exclude parameters to make checks pass. Existing
controlled fixture learning and task-correctness tests keep their current owners.

## Run and selection

From the repository root with the test environment and verified downloads:

```sh
python -m test.kumo_pytorch.nightly run --binding E02 --mode training \
  --data-dir test/kumo_pytorch/.data \
  --output test/kumo_pytorch/.runs/training-acceptance \
  --anchors-per-split 32 --batch-size 4
```

Use a new output directory. The downloader's pinned identities and checksums
are verified before preparation. Selection takes the first physical anchors in
each original split, without balancing labels or filtering difficult examples.
Only the selected training split is fitted or trained; validation/test anchors
are not used for updates or loss acceptance. Selected roots retain complete
histories. This is bounded real-data training, not full-population training;
full delivery is checked separately by the population runner.

Each model uses the existing small dimension-16 configuration and AdamW with
learning rate 0.001 and zero weight decay. Both repeats initialize from seed 17,
use float32 CPU execution with one thread and deterministic algorithms, and
cycle shuffled training epochs until exactly 100 optimizer updates. Batches
without eligible targets remain in the delivery accounting and are reported
separately; they perform no optimizer update. An entirely ineligible epoch fails.
The loss probe is the first eligible training batch. There is no option to
shorten the 100-update requirement. The second repeat resets
model, optimizer and CPU RNG and recreates the batch iterator on the same fitted
task; this is a reproducibility check, not a fresh-process lifecycle test.

## Required assertions

- Finite loss, logits, gradients and updated parameter values throughout training.
- Every trainable parameter tensor receives a nonzero gradient at least once
  and differs from its initial value after 100 updates. This checks each tensor,
  not every scalar entry of an embedding. Zero weight decay prevents passive
  decay from satisfying the update check. No name/module exclusion is allowed.
- Final loss is below initial loss on the same fixed training probe batch, with
  identical targets/masks and eligible count. Initial/final probe measurements
  use evaluation mode; the curve records each optimizer step's training loss.
- Exact same-seed agreement for initialization, all batch/target hashes, losses,
  every training-step model hash, per-parameter results, final optimizer and RNG
  hashes. Two incomplete executions cannot establish reproducibility.
- Inputs/targets and fitted state remain unchanged by training.

An explicit declared preflight refusal passes under the
[capability policy](required-coverage.md#capability-refusals), labelled `refused`
with no training measurements. Once preparation succeeds, missing support,
unused parameters, nonfinite values, insufficient loss decrease and
nondeterminism are ordinary failures. A completed but failing first repeat
does not suppress the second repeat or the other workloads. The shared results gate rejects
missing reports, substituted binding identities and incomplete training evidence.

## CI and reports

The **Training** matrix in **PyTorch — Full Suite** uses the catalog's execution
settings and the shared `nightly run` command. All models train on CPU, including
bindings assigned to RTX runners for storage. HI Small's five combinations each
perform two 100-update repeats (1,000 updates total). Fixture learning remains
separate from real-data acceptance.

Each `kumo-pytorch-dataset-training-<binding>` artifact contains reports, contracts,
per-repeat parameter evidence and `loss-curve.csv`. The combined
`kumo-pytorch-campaign-results` report validates required binding outcomes.
Reports retain revision, dirty-checkout status, source/anchor identities,
versions, seeds, losses and per-parameter gradient/update results. Partial
progress remains running/failed. Source payloads and prepared indexes are excluded.

## Model scope and validation

Test-owned GraphSAGE/Fusion models omit final non-root branches that cannot
contribute to root prediction. Exact-output regression checks compare root
embeddings/predictions with the original full graph module. This is architecture
cleanup, not a parameter exemption; every remaining trainable tensor must update.
Older checkpoints containing removed branches must be regenerated.

Fault-injection fixtures require failures for unused parameters, missing updates,
nonfinite computation and inadequate loss reduction. Real-data acceptance still
requires passing reports from the intended package revision. Historical local
counts remain in Git history rather than being treated as current results.
