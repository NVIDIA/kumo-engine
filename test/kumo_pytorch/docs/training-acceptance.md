# Real-AML CPU training acceptance

Criterion 8 uses the existing AML HI-small v8 bindings: a001 with FusionModel
(the combined consumer, v2 forecast anchors), and a017 with NativePragmaModel
(PRAGMA masked-value pretraining, v1 split anchors). No model parameters are
removed or frozen to make acceptance pass. Existing controlled fixture learning
tests and task-correctness tests keep their current owners.

## Run and selection

From the repository root with the test environment and verified downloads:

```sh
python -m test.kumo_pytorch.training_acceptance \
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

Missing support, unused parameters, nonfinite values, insufficient loss decrease
and nondeterminism are ordinary failures. A completed but failing first repeat
does not suppress the second repeat or the other workload.

## CI and reports

`PyTorch nightly real AML CPU training acceptance` runs on `ubuntu-24.04` using
the existing CPU setup, DiskGraph wheel and Engine AWS test role. It runs on
schedule/manual dispatch independently of the GPU benchmark queue. Its two
workload runs each perform two 100-update repeats. Fixture coverage also runs
in the existing PR/nightly AML model-check job.

The job summary records the tested revision. Download
`kumo-pytorch-aml-training-acceptance` for the aggregate and per-case JSON,
contracts, per-repeat parameter reports and `loss-curve.csv` series. Each repeat
records parameter shapes, gradient/nonzero-gradient counts, update status and
maximum absolute change. Reports include source identities, selected anchors,
configuration, dependency versions, test/model source hashes and dirty-checkout status. Partial progress
is saved every 10 updates and remains running/failed, never accepted. Source
payloads, prepared views and indexes are excluded from the artifact.

## Local evidence

After integrating the upstream decimal fix and adapting Fusion to use an
encoder-only GraphSAGE, all 30 focused training, controlled-learning, finetuning
and checkpoint-resume checks pass with no skips. Both generated and raw-AML-format
training fixtures retain the original 100-update, every-trainable-parameter,
loss-decrease and repeatability requirements. Fault injection still rejects
unused parameters and missing optimizer updates.

Fusion omits the unused standalone graph prediction head and final non-root
updates. Earlier graph layers remain active. A regression verifies exactly equal
graph embeddings and predictions against the full graph module with matching
weights. Old Fusion checkpoints contain the removed parameters and must be
regenerated. These are local fixture results; real HI-small training must be
rerun on CI before claiming acceptance of the changed model.
