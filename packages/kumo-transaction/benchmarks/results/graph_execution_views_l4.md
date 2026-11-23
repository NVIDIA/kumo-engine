# Graph execution views: synthetic L4 measurements

This compares clean baseline `85f47bb0c666360cd870aadd00f9113943f98aba`
with clean candidate `1e73e84` (implementation `93e720d`). The identical driver
at `1e73e84` was used for both sources. The current
[driver](../ablations/graph_backends/views.py) additionally emits effective
settings; it has not been used to replace these measurements.
[Raw measurements](graph_execution_views_l4.json) include the full revisions,
driver/input/model hashes, configuration, every cycle, loss histories and final
embeddings. Earlier execution result files describe their own source revisions.

## Workload and method

Generated synthetic data: 64 owners, 100 events per owner, four-hop/fanout-10
Kumo Engine GNN with width 32, and width-32 SDPA sequence stack. Configured
training batch size is 32; the measured GNN batch has **29 roots**, hybrid
training has **32**, and validation has **12**. Training uses shared neighborhoods;
validation uses disjoint graphs. The raw record includes measured root counts in
its summary, node/edge counts, and the data/model setup config. GNN training
uses FutureSubgraph; hybrid sequence training uses the frozen GNN and binary
labels. The architecture, inputs and effective workload settings match across
sources. This does not compare shared and disjoint training architectures.

The embedded recipe config configures data/model preparation. The timing driver
overrides its training loop: both training workloads use AdamW at learning rate
0.003, weight decay 0.01, betas (0.9, 0.999), epsilon 1e-8, AMSGrad disabled,
and gradient clipping at maximum norm 1.0 (L2). Each executes 25 optimizer steps
(5 warmup + 20 measured), independently of the recipe's `max_steps: 3`. There is
no learning-rate schedule. In particular, the embedded sequence learning rate
0.001, weight decay 0.0 and absent clipping are not the measured settings.

| Workload | Measured roots | Effective optimizer | L2 clipping | Optimizer steps |
|---|---:|---|---:|---:|
| GNN training | 29 | AdamW, lr 0.003, weight decay 0.01 | 1.0 | 25 |
| Hybrid sequence training | 32 | AdamW, lr 0.003, weight decay 0.01 | 1.0 | 25 |
| Hybrid validation | 12 | None; eval forward only | None | 0 |

Effective settings were annotated after review from the recorded driver and
embedding row counts, beside the existing sample counts in the JSON summary.
The original six run records, hashes, timing samples and summaries are unchanged;
no performance measurements were rerun for this reporting correction.

Each cycle reconstructs and validates the CPU batch from identical pre-sampled
tensors, transfers it, then executes the model. Training includes backward,
gradient clipping and AdamW. Validation is forward-only. Sampling and initial
model transfer are excluded. Baseline selects full `.to()`; candidate selects
`.for_execution()`. Both copies are blocking and synchronized for timing.

Three sequential runs per source, five warmup and twenty measured cycles per
workload; order: before, after, after, before, before, after. Values below are
the median of the three run medians, not confidence intervals. No other heavy
workloads ran concurrently. Deterministic algorithms and a fixed Python hash
seed were used so training trajectories could also be checked.

Environment: NVIDIA L4, Python 3.14.6, Torch 2.13.0+cu130, PyG 2.8.1,
Kumo Engine 1.0.1, FP32, TF32 disabled. Optional torch-scatter was absent.

```bash
PYTHONPATH=<source>/src:<driver-checkout> OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  <prepared-environment>/bin/python \
  <driver-checkout>/benchmarks/ablations/graph_backends/views.py \
  --deterministic --warmup 5 --steps 20 --output /tmp/graph-views.json
```

## Results

| Workload | Transfer before → after, ms | Transfer speedup | Complete cycle before → after, ms |
|---|---:|---:|---:|
| GNN training | 1.428 → 0.693 | 2.06× | 33.766 → 32.315 |
| Hybrid sequence training | 1.755 → 1.108 | 1.58× | 28.166 → 27.666 |
| Hybrid validation | 1.348 → 0.997 | 1.35× | 19.803 → 19.715 |

Full-cycle median reductions were 4.3%, 1.8% and 0.4%. Validation cycle ranges
overlap (before 19.669–20.211 ms, after 19.038–20.334 ms); this does not establish
a meaningful validation-cycle gain. GNN ranges overlap too (before
33.651–35.560 ms, after 31.686–34.383 ms). Sequence ranges were 28.092–28.292 ms
and 26.519–27.759 ms. This small sequential microbenchmark is not a statistical
performance guarantee.

Compilation shifts some work to construction. Construction medians were
3.588 → 3.361 ms, 3.372 → 3.541 ms and 2.745 → 2.960 ms respectively. Prepared
CPU batches amortize that work across transfers; these complete-cycle results
conservatively include reconstruction every time.

All six deterministic runs had identical input/model hashes, exact loss
histories and exact final embeddings for all three workloads. Separate contract
checks covered exact deterministic CPU/CUDA outputs, losses, gradients and
BatchNorm buffers; tracked source/device mutations and invalid temporal,
duplicate, FK and hop data remain rejected. No model or recipe changes were
needed, and graph-v4 bundles remain compatible.

These results establish lower transfer overhead for this synthetic workload.
They do not reproduce Distress/H100 measurements or establish whole-book job
speed, native-loader parity, convergence or task quality. Normal nondeterministic
CUDA reduction timings and trajectories can differ from this deterministic run.


## Resource supplement and raw artifacts

The checked-in JSON is now a compact manifest. Full historical raw runs are in
the private project artifact referenced by `raw_artifact.url`, with its original
SHA256 and byte count. Historical latency values are unchanged. Throughput is
derived from actual roots divided by the historical complete-cycle median and
has explicit units/scope in the manifest.

A separate [resource supplement](graph_resource_supplement_l4.md) measured
post-warmup CUDA allocator peaks on the exact historical sources, three new runs
per source. Those peaks are linked beside each historical workload/source but
are explicitly **not measurements from the original latency runs**. The supplement
also records its own throughput and timings, driver hashes and source identity.
Do not combine the two campaigns as if they were simultaneous measurements.
