# Population acceptance

This runner implements the real-data criterion from the Kumo PyTorch 1.2 testing
plan. It prepares inputs and verifies complete batch delivery; it does not train
models or measure loader performance. Those checks retain their existing owners.

## Workloads and historical IDs

The [historical acceptance catalog](https://github.com/kumo-ai/kumo-engine-arena/blob/ebf3fd4ac4d4ee8a6bb549f4b2c0025174f78986/rdl_specs/verification-accepted.md)
identifies the PDF's three AML workloads:

| PDF ID | Task | Consumer/input family | Current anchors |
| --- | --- | --- | --- |
| [a001](https://github.com/kumo-ai/kumo-engine-arena/blob/ebf3fd4ac4d4ee8a6bb549f4b2c0025174f78986/rdl_specs/verification-accepted.md#a001) | Account involvement next 24h | combined / FusionModel | v2, `forecast_split` |
| [a017](https://github.com/kumo-ai/kumo-engine-arena/blob/ebf3fd4ac4d4ee8a6bb549f4b2c0025174f78986/rdl_specs/verification-accepted.md#a017) | Masked-value pretraining | PRAGMA | v1, `split` |
| [a018](https://github.com/kumo-ai/kumo-engine-arena/blob/ebf3fd4ac4d4ee8a6bb549f4b2c0025174f78986/rdl_specs/verification-accepted.md#a018) | Next-token pretraining | nuFormer | v1, `split` |

These original cases use HI-small v8. The shared runner also includes GraphSAGE and nuFormer
account-involvement cases, using [the shared declarations](../workloads/bindings.json).
Historical a001 used v1; our graph cases deliberately use the user-approved v2
cutoff correction. Historical acceptance is not evidence for these current
contracts or the current Engine commit. Arena is a documentation reference only,
with no runtime dependency. Synthetic a037 and cross-engine batch parity remain
explicitly deferred by the user; Polars/GPU reference comparisons have been removed.

## Execute

From the Engine repository root, after [downloading the pinned inputs](../README.md#data):

```sh
python -m test.kumo_pytorch.nightly run --binding E02 --mode smoke \
  --data-dir test/kumo_pytorch/.data \
  --output test/kumo_pytorch/.runs/population-smoke \
  --anchors-per-split 4 --batch-size 4 --threads 4

python -m test.kumo_pytorch.nightly run --binding E02 --mode full \
  --data-dir test/kumo_pytorch/.data \
  --output test/kumo_pytorch/.runs/population-full \
  --batch-size 4 --threads 4
```

Output directories must be new. Both modes verify the pinned downloads and
prepare complete source histories. Smoke bounds the selected anchors per split;
full retains every eligible anchor and the complete source tables.
Other bindings use the same command with their catalog ID. Full mode
rejects `--anchors-per-split`. Null targets do not remove anchors. No history cap,
step limit, dropped tail, sampling fallback or expected-failure exception is
introduced to make a run pass.

The anchor projection is written in Arrow batches, then passed as a native
`AnchorTable`. It checks unique sample IDs, original entity mappings and order,
v2 prediction/execution clocks, labels and split counts. An identity set is
retained for duplicate detection; package fitting also has its own memory costs.
Streaming the projection does not establish bounded fitting memory.

Each case fits on its complete selected training split. Batch checks compare
all delivered identities, original row positions, splits, execution clocks and
stored labels against the projected anchors. They check emitted finite values,
native target masks, graph cutoffs, final partial batches, ordered sample-ID
hashes and fitted-state stability. The runner does not independently reconstruct
every raw-source history or original pretraining target; the existing task and
history checks retain that responsibility.

## Reports and failures

The full-suite population matrix executes each binding independently, with the
runner/resources declared in its catalog entry. Smoke failure does not suppress
full execution. Source/preparation failures, termination and timeouts remain
failed/incomplete, never acceptance. A declared preflight capability refusal is
the only [approved exception](required-coverage.md#capability-refusals): it passes
as `refused`, with no delivery and `full_population_verified=false`.

The `kumo-pytorch-campaign-results` artifact contains the combined summary;
`kumo-pytorch-dataset-<mode>-<binding>` retains each worker's reports/contracts.
Source payloads, projected anchors and indexes are not uploaded. Reports include
input identities, Engine revision, dirty-checkout status, dependency versions,
expected/delivered counts, eligible target counts, fitted-state fingerprints,
current stage and errors. Per-case JSON is updated every 100 verified batches;
an interrupted report remains `running` or `pending`, not passed.

`full_population_verified` is true only after full delivery succeeds. Release
acceptance additionally requires a clean run at the chosen freeze commit;
smoke results and tiny fixture runs do not establish that evidence.

## Validation boundaries

Fixture tests cover projection beyond an Arrow read batch, partial tails, null
targets and corrupted identities/clocks/labels/masks. They require failure for
missing tails, duplicate/reordered IDs and unexpected padding. Current full-data
acceptance must come from the intended revision's reports; historical fixture
counts and superseded decimal failures remain in Git history.
