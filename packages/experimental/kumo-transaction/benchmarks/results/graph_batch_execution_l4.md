# Graph batch execution costs on L4

Compare clean `e27be76a556a8c9714ddb5c76a1082c3d1ecbf28` with clean `b4195a59d2a7c52d140eceb5b3d216e891d0ed25`. Architecture, recipe,
hop trimming and weights are unchanged. Only CPU validation, cached FK facts,
mutation guards and device transfer execution change. Full raw conditions,
commands, per-step values and parity-driver source are in the
[JSON record](graph_batch_execution_l4.json).

## Conditions

- One NVIDIA L4 (23,034 MiB), driver 595.71.05. Runs are sequential with no competing CUDA work.
- Python 3.14.6, Torch 2.13.0+cu130, PyG 2.8.1, Kumo Engine 1.0.1; TF32 disabled. Optional torch-scatter absent.
- Generated graph, seed 0: 64 users, 12 events and 4 transfers per user, batch 32, width 512, 6 hops, fanout 10.
- FP32, squared scalar loss, clipping 1.0, AdamW LR 0.003 and weight decay 0.002; same initial model hash and sampled node/edge counts across revisions.
- Three independent processes per revision/condition. Tables show the median of per-run medians. Verified transfer-cycle pairs alternate old/new revisions.

## Repeated CPU batch transfer plus training

The companion [execution driver](../ablations/graph_backends/execution.py) uses
the same fixed-batch GNN workload but transfers the CPU batch every step, as the
recipe trainer does. Five warm-up and 20 measured steps per arm. Sampling and
initial model transfer are excluded; blocking batch transfer, forward/backward,
gradient clipping and optimizer update are included. Each transfer is synchronized
to measure its cost separately. This is a fixed-batch execution measurement,
not full-job throughput.

| Training graphs | Batch transfer before / after (ms) | Transfer + train before / after (ms) | Cycle speedup |
|---|---:|---:|---:|
| shared | 33.13 / 0.78 | 70.47 / 36.92 | 1.91x |
| disjoint | 41.77 / 1.12 | 91.39 / 50.92 | 1.79x |

## Candidate resident measurements; baseline comparison withdrawn

The three archived `resident_before` records and their logs identify candidate
`b4195a5`, not baseline `e27be76`, and carry no baseline contract hash. The available
evidence cannot establish a resident baseline. The previously published resident
before/after table and the claim of unchanged warmed resident compute are
withdrawn. No revision was relabeled by inference.

Candidate `resident_after` values remain archived unchanged: median warmed step
latencies are 36.18 ms (Engine shared), 50.24 ms (Engine disjoint) and 107.29 ms
(historical custom disjoint). These are candidate-only values, not a comparison.
The transfer-cycle table above remains supported by explicit, different source
contract hashes in its baseline/candidate raw records.

Original raw data is preserved in the private checksummed artifact referenced by
the compact JSON manifest, including the excluded records for audit. Those
excluded records must not be used as a baseline comparison.

## Correctness and limits

- 646 unit/contract/recipe tests passed, 25 skipped; Ruff, Pyright and strict docs passed.
- Invalid bounds, future clocks, duplicate/cross-graph edges, root ordering, hop layout and multi-parent Engine gathers remain rejected.
- Tracked mutations to graph tensors, labels, targets and history invalidate transfer/model use; mapping writes are refused. CPU serialization rebuilds validation records.
- CUDA profiling confirms no tensor validation/scalar reads in the copy path. CPU and CUDA output/gradient tests cover shared/disjoint graphs with trimming on/off.
- A separate deterministic source-to-source CUDA check (width 32, 3 hops, batch 8) matches sampled inputs and initial weights, then bitwise matches outputs, every parameter gradient and BatchNorm buffers for both graph policies. Performance runs retain normal nondeterministic CUDA reductions.
- Both supervised and FutureSubgraph hybrid CUDA smokes passed from clean b4195a5.
- Construct batches on CPU and keep their tensors read-only. External/.data writes cannot be detected by version counters and are forbidden. See the [batch contract](../../docs/concepts/data-contract.md).
- No full-book, native-loader, distributed or quality campaign was run. Graph overlap, CPU/kernel environment, clocks, graph sizes and data-loading costs affect the resulting speedup.
