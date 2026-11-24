# Shared edge checks and mutation-record compilation on L4

This synthetic ablation compares three clean sources with the identical
[conversion driver](../ablations/graph_backends/conversion.py) and underlying
[execution driver](../ablations/graph_backends/views.py):

| Stage | Revision | Change |
|---|---|---|
| Baseline | `0602c5c` | Existing MR8, including event-first hybrid scheduling |
| Edges | `dafe31f` | Receiver-first sorting shared by duplicate and parent checks |
| Both | `05ba7d7` | Shared edge checks plus root mutation-record accumulators |

[Raw timings and fingerprints](graph_constructor_checks_l4.json) record full
source revisions, code/driver hashes, effective settings, individual samples,
run medians and ranges. Previous benchmark artifacts retain their own revisions.

## Correctness before timing

A separate deterministic FP32 preflight ran one cycle per workload/source,
with no warmup, before the timing campaign: one optimizer update for training,
and a forward-only cycle for validation. Inputs, initial state, final model
state (including buffers), gradients, loss histories and embedding values matched
exactly across all sources. All nine timed runs also matched these quantities
exactly within that campaign. Training final states differ between preflight and timing because they execute
different numbers of updates; validation performs no updates in either group.

Unit/contract/recipe-smoke validation finished with 793 passed, 25 skipped.
Twelve separate strict deterministic CPU/CUDA FP32/BF16 cases passed, including
full-copy/execution-view and event-first parity. New classification tests compare
both sorting paths with the previous duplicate/receiver-unique checks and
independent oracles. Empty/singleton/strided and maximal-int64 cases are covered.
Mutation tests retain field/container replacement, version/gradient checks,
metadata ownership, tensor lifetime and nested-record reuse. No architecture,
label, temporal, hop or graph-v4 semantics changed.

## Effective workload and measurement

Synthetic data: 64 owners, 100 events per owner; four-hop, fanout-10, width-32
Kumo Engine GNN and width-32 SDPA sequence model. Training graphs share
neighborhoods; validation graphs are disjoint. GNN training uses FutureSubgraph;
hybrid sequence training uses binary labels with a frozen GNN.

| Workload | Roots | Nodes | Directed edges | Effective optimizer | L2 clipping | Updates |
|---|---:|---:|---:|---|---:|---:|
| GNN training | 29 | 832 | 1964 | AdamW, lr 0.003, decay 0.01 | 1.0 | 25 |
| Hybrid training | 32 | 1163 | 3208 | AdamW, lr 0.003, decay 0.01 | 1.0 | 25 |
| Hybrid validation | 12 | 852 | 2232 | None; eval forward only | None | 0 |

AdamW uses betas (0.9, 0.999), epsilon 1e-8 and no AMSGrad or schedule. These
are the actual benchmark loop settings, rather than the embedded recipe's
learning-rate/decay/max-step fields. Each timed run has five warmup and twenty
measured cycles per workload. Nine sequential runs use balanced stage order:
baseline, edges, both, both, edges, baseline, edges, baseline, both. No other
GPU process was active at the initial check; no other local heavy checks ran
during timing.

Each cycle reconstructs and validates CPU batches from fixed pre-sampled tensors,
transfers them synchronously and runs the model. Training includes backward,
clipping and AdamW. Sampling, initial model transfer and raw/native-loader
adaptation are excluded. Thin identical timing wrappers measure `_seal()` within
reconstruction: **compilation is a subset of constructor time**, not an additional
stage. Final-state and gradient hashes are computed outside cycle timing.

Environment: NVIDIA L4, Python 3.14.6, Torch 2.13.0+cu130, PyG 2.8.1,
Kumo Engine 1.0.1, FP32, deterministic algorithms, TF32 disabled, Python hash seed
0, one Torch/OMP/MKL thread; optional torch-scatter absent. CPU model and exact
source/settings are in the JSON record.

```bash
PYTHONPATH=<source>/src:<driver-checkout> OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  <prepared-environment>/bin/python \
  <driver-checkout>/benchmarks/ablations/graph_backends/conversion.py \
  --deterministic --warmup 5 --steps 20 --output /tmp/construction.json
```

Use `--warmup 0 --steps 1` first for the separate correctness preflight. The raw
record replaces repeated final embedding arrays with their shapes and FP32 byte
hashes after checking exact equality; the driver emits full arrays for reproduction.

## Results

Values are medians of three run medians per source, in milliseconds. They are
small sequential measurements, not confidence intervals or a performance guarantee.

| Workload | Constructor baseline | Edges | Both | Compilation baseline | Edges | Both |
|---|---:|---:|---:|---:|---:|---:|
| GNN training | 3.198 | 3.094 | 3.041 | 0.643 | 0.639 | 0.604 |
| Hybrid training | 3.360 | 3.239 | 3.106 | 0.788 | 0.788 | 0.703 |
| Hybrid validation | 2.855 | 2.758 | 2.653 | 0.593 | 0.596 | 0.531 |

Sharing the edge sort reduces constructor medians by 0.10–0.12 ms. Accumulation
reduces compilation medians by a further 0.04–0.09 ms relative to edges-only.
Combined constructor reductions are 0.16–0.25 ms (about 4.9–7.6%). This workload
does not reproduce the reported roughly 2 ms receiver-sort cost on real batches.

Constructor run-median ranges, baseline → both: GNN 3.174–3.258 → 2.995–3.080 ms;
hybrid 3.351–3.371 → 3.092–3.115 ms; validation 2.809–2.882 → 2.650–2.664 ms.
Raw data includes ranges for every stage and metric.

| Workload | Complete cycle baseline | Edges | Both |
|---|---:|---:|---:|
| GNN training | 30.360 | 30.736 | 30.103 |
| Hybrid training | 25.851 | 25.743 | 25.460 |
| Hybrid validation | 18.629 | 18.514 | 18.324 |

Complete-cycle changes are small and include model-runtime variability: the
edge-only GNN median is slower even though its constructor is faster. Baseline
and both-stage GNN cycle ranges overlap (30.218–30.549 versus 29.984–30.300 ms).
No whole-job, convergence or quality gain is established by this microbenchmark.

These measurements isolate constructor/record changes on already sampled
synthetic CPU tensors. They do not measure a native loader, reproduce the
reported 12.1/25.6 ms real-loader conversion profile, or establish that the full
6–8 ms constructor is record compilation. The reported roughly 0.6 ms compilation
cost remains distinct from whole-constructor time. Larger real-loader costs and
full-book scaling remain separate validation work.


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
