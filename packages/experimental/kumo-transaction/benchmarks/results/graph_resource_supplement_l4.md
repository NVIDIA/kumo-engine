# Scoped throughput and GPU memory supplement

This is a **new measurement campaign**, separate from the historical latency
campaigns in the [execution-view](graph_execution_views_l4.md) and
[constructor](graph_constructor_checks_l4.md) reports. Historical latency samples
were not changed and these peaks are not retroactive measurements of those runs.

The [compact supplement manifest](graph_resource_supplement_l4.json) holds
conditions, summaries and a private raw-artifact URL/SHA256. Individual new runs,
latencies, throughputs, post-warmup peaks, losses and model/output fingerprints
are in that external artifact. The private registry also retains the exact
original raw payloads for audit; see [artifact retrieval](README.md#graph-benchmark-raw-artifacts).

## Method

Use driver `828d785` with each exact historical source: `0602c5c`, `dafe31f`,
`05ba7d7`, `85f47bb` and `1e73e84`. Three new runs per source, five warmup and
twenty measured cycles per workload, balanced sequential source order. Device
was idle at the initial check and no local heavy tests ran during measurement.

All runs used one NVIDIA L4, Python 3.14.6, Torch 2.13.0+cu130, Kumo Engine 1.0.1,
FP32 deterministic algorithms, TF32 off, Python hash seed 0 and one Torch/OMP/MKL
thread. The original fixed synthetic 64-owner, 100-event, four-hop/fanout-10,
width-32 GNN/SDPA configurations and actual roots (29 GNN / 32 hybrid / 12
validation) were used. Training uses AdamW lr 0.003, decay 0.01, betas
(0.9, 0.999), eps 1e-8, no AMSGrad/schedule, L2 clipping 1.0 and 25 updates.
Validation is forward-only. Full setup and effective settings are in each raw run.

Throughput is actual roots divided by the median complete-cycle seconds, in
**roots/s**. Training cycles include CPU construction, blocking transfer,
forward/backward, clipping and AdamW; validation includes construction, transfer
and forward only. Sampling and raw/native-loader adaptation are excluded.

GPU peaks are `torch.cuda.max_memory_allocated/reserved`, in **MiB**, reset after
warmup outside the timed region and sampled after all measured cycles. They
include live model/optimizer/batch tensors and allocator pools. They are not host
RSS or total device memory. Compilation timing remains a subset of constructor
time; hashing final state and gradients occurs outside cycle timing.

```bash
PYTHONPATH=<historical-source>/src:<driver-checkout> OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  <environment>/bin/python \
  <driver-checkout>/benchmarks/ablations/graph_backends/conversion.py \
  --deterministic --warmup 5 --steps 20 --output <run>.json
```

Inputs, initial weights, loss histories and embeddings matched historical values
exactly for every source/workload. Constructor sources also matched the original
final-state/buffer and gradient byte hashes. Repetitions within each source
matched exactly. This confirms workload identity; it does not make memory and
latency measurements from different campaigns simultaneous.

## Results

Values are medians of three new run values. Allocated/reserved peaks are shown
as `allocated / reserved MiB`; ranges and every individual run are in the manifest
and referenced raw artifact.

| Source | Workload | Roots/s | GPU peak allocated / reserved MiB |
|---|---|---:|---:|
| `0602c5c` baseline | GNN training | 969.0 | 66.523 / 68 |
| `dafe31f` edges | GNN training | 966.5 | 66.523 / 68 |
| `05ba7d7` both | GNN training | 963.2 | 66.523 / 68 |
| `0602c5c` baseline | Hybrid training | 1244.8 | 84.102 / 92 |
| `dafe31f` edges | Hybrid training | 1274.5 | 84.102 / 92 |
| `05ba7d7` both | Hybrid training | 1272.1 | 84.102 / 92 |
| `0602c5c` baseline | Validation | 661.7 | 65.965 / 68 |
| `dafe31f` edges | Validation | 654.5 | 65.965 / 68 |
| `05ba7d7` both | Validation | 662.8 | 65.965 / 68 |
| `85f47bb` full transfer | GNN training | 924.8 | 66.561 / 68 |
| `1e73e84` execution view | GNN training | 968.3 | 66.523 / 68 |
| `85f47bb` full transfer | Hybrid training | 1205.9 | 84.167 / 92 |
| `1e73e84` execution view | Hybrid training | 1253.3 | 84.113 / 92 |
| `85f47bb` full transfer | Validation | 644.7 | 66.008 / 68 |
| `1e73e84` execution view | Validation | 652.3 | 65.968 / 68 |

Constructor-source GPU peaks are unchanged at the shown precision. The record
accumulator changes CPU work; this campaign does not measure CPU peak memory.
Small throughput differences include ordinary timing variability and do not
establish whole-job performance. No real-loader, full-book quality, native-loader
parity or production-memory claim follows from this small synthetic workload.
