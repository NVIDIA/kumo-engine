# AML population acceptance

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

All use HI-small v8. The runner also includes the existing GraphSAGE and nuFormer
account-involvement cases, using [the shared declarations](../workloads/aml.json).
Historical a001 used v1; our graph cases deliberately use the user-approved v2
cutoff correction. Historical acceptance is not evidence for these current
contracts or the current Engine commit. Arena is a documentation reference only,
with no runtime dependency. Synthetic a037 and cross-engine batch parity remain
explicitly deferred by the user; Polars/GPU comparisons remain timing-only.

## Execute

From the Engine repository root, after [downloading the pinned inputs](../README.md#data):

```sh
python -m test.kumo_pytorch.population --mode smoke \
  --data-dir test/kumo_pytorch/.data \
  --output test/kumo_pytorch/.runs/population-smoke \
  --smoke-anchors 4 --batch-size 4 --threads 4

python -m test.kumo_pytorch.population --mode full \
  --data-dir test/kumo_pytorch/.data \
  --output test/kumo_pytorch/.runs/population-full \
  --batch-size 4 --threads 4
```

Output directories must be new. Both modes verify the pinned downloads and
prepare complete source histories. Smoke bounds the selected anchors per split;
full retains every anchor with an explicit train/valid/test split. Full mode
rejects `--smoke-anchors`. Null targets do not remove anchors. No history cap,
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

The nightly executes each binding and mode on a separate CPU runner using the
same verified AML inputs. Full runs even if smoke fails; matrix jobs disable
fail-fast so other cases still execute. Source/preparation failures, process
termination and timeouts remain failed/incomplete execution, never acceptance.
The only exception is a declared preflight capability refusal under the
[capability policy](required-coverage.md#capability-refusals): it passes as
`refused`, with no delivered samples and `full_population_verified=false`.

The job summary includes the population table. Download
`kumo-pytorch-aml-benchmarks` for aggregate and per-case reports and contracts
under `smoke/<case>` and `full/<case>`. Per-job evidence is also available in
`kumo-pytorch-aml-parallel-<mode>-<case>` artifacts.
Source payloads, projected anchors and indexes are not uploaded. Reports include
input identities, Engine revision, dirty-checkout status, dependency versions,
expected/delivered counts, eligible target counts, fitted-state fingerprints,
current stage and errors. Per-case JSON is updated every 100 verified batches;
an interrupted report remains `running` or `pending`, not passed.

`full_population_verified` is true only after full delivery succeeds. Release
acceptance additionally requires a clean run at the chosen freeze commit;
smoke results and tiny fixture runs do not establish that evidence.

## Current validation

The new local fixture checks passed 19 cases and failed four, with zero skips.
The failures are GraphSAGE and combined in both smoke and full modes, exposing
the branch's known decimal-decoding defect. The tests require success and stay
red until the defect is fixed. [PR #348](https://github.com/NVIDIA/kumo-engine/pull/348)
is the upstream fix; it has not been incorporated in this increment.

The three sequence combinations completed both modes on raw AML fixtures,
including partial final batches. A 65,540-row projection test verifies selection
past the first Arrow read batch. Fault injection verifies that omitted tails,
duplicated/reordered IDs, changed clocks/entities/labels/masks and sample padding
fail acceptance. Full real HI-small execution and freeze-commit CI evidence
remain outstanding.
