# Six-hop graph architecture and sharing costs on an L4

Historical generated-data systems check, not a downstream quality experiment.
This source predates hop trimming and input ReLU removal. These measurements
do not describe current trimmed-model throughput.
Raw measurements and all conditions: [JSON](graph_backends_l4_cost.json).
Implementation and interpretation: [benchmark guide](../ablations/graph_backends/README.md).

## Conditions

- Code: `479ee85b4c932c5b5e1293f94b01fbac3b1573e7`, clean worktree (`source_dirty: false`).
- Measured: `2026-11-18T22:49:09.097083+00:00` through `2026-11-18T22:49:16.283246+00:00`; no concurrent CUDA work.
- Hardware: one NVIDIA L4, driver 595.71.05, 23,034 MiB device memory; PyTorch allocator peaks below.
- Runtime: Python 3.14.6, PyTorch 2.13.0+cu130, PyG 2.8.1,
  CUDA runtime 13.0, TF32 disabled.
- Actual dependency: Kumo Engine 1.0.1,
  `kumo_core.modeling.gnn.hetero.HeteroGNN`. The GNN source SHA256 is in the JSON.
  Optional `torch-scatter` was **not installed**; PyG used its Torch max kernel.
- Generated graph: seed 0, 64 users, 12 events and 4 transfers per user;
  32 fixed prediction roots, width 512, 6 message-passing hops, fanout 10.
  Sampling is seeded/canonical; all observations precede the common UTC cutoff.
- All arms: FP32, the same scalar targets, AdamW LR 0.003 / weight decay 0.002,
  clipping at 1.0, 2 warm-up steps, 5 measured steps. No full training was launched.

The two Engine arms have identical full initial state, including BatchNorm
buffers, verified by equal `initial_state_sha256` values in the JSON. The
historical encoder intentionally changes architecture and parameter count;
its input graph is the same disjoint graph supplied to Engine disjoint.

## Measurements

| Arm | Nodes | Edges | Parameters | Median step (ms) | Derived roots/s | Peak allocated (MiB) | Peak reserved (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `engine_shared` | 960 | 2,304 | 23,642,625 | 35.44 | 903.05 | 447.78 | 512 |
| `engine_disjoint` | 22,449 | 54,780 | 23,642,625 | 122.80 | 260.59 | 1946.09 | 2188 |
| `legacy_disjoint` | 22,449 | 54,780 | 14,187,009 | 108.19 | 295.78 | 2361.26 | 2612 |

Throughput is derived from the 32 actual roots divided by each original median
step duration. It excludes initial batch/model transfer, construction and sampling.
No original latency or memory measurement changed, and this is not a new run.

On this generated batch, separate Engine graphs cost
**3.47x the step time** and
**4.35x peak allocator memory**
of shared Engine graphs. The shared graph has 960 nodes versus
22,449 duplicated nodes. Fixed model/optimizer memory prevents
memory ratios from simply equaling the node-count ratio.

The historical mean/LayerNorm/GELU encoder took
108.19 ms on the disjoint graph, compared with Engine's
122.80 ms, and used more allocated memory. This is a
controlled architecture comparison, not a claim that one encoder is faster or
better across workloads.

Sampling took 279.57 ms for the shared batch and
450.72 ms for the disjoint batch. Model/batch transfer
and every measured step/loss are retained separately in JSON. Step timing
includes synchronized forward, scalar MSE, backward, clipping, and optimizer
update. Peak memory uses PyTorch allocation counters after warm-up and excludes
CUDA driver/context allocations and external processes.

## Reproduce

From the repository root at the recorded source revision, with Engine installed:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --extra kumo python -m benchmarks.ablations.graph_backends.benchmark \
  --seed 0 --entities 64 --events-per-entity 12 --transfers-per-entity 4 \
  --batch-size 32 --width 512 --depth 6 --fanout 10 --warmup 2 --steps 5 \
  --device cuda --output outputs/graph_backend_cost.json
```

The recorded local invocation used the same arguments with the worktree's
existing Python environment: `PYTHONPATH=src:. .venv/bin/python -m ...`.
The committed `Settings` in the benchmark define these values explicitly,
including seed 0; the JSON records the resolved settings. Run serially on a
quiet GPU and record kernel dependencies when comparing another environment.

## Limits and decision

This is a single short run on a small graph with high neighborhood overlap.
It does not establish native-loader parity, full training throughput, loss or
quality equivalence, or full-book memory requirements. Layer trimming, native
loading, sequence stages, historical feature fitting, target preparation,
validation, and checkpointing are excluded. No downstream metric is reported.
Graph sharing changes BatchNorm multiplicities and potentially context; losses
need not match from identical initial weights.

Use the actual Engine GNN in both shipped graph architectures. Keep shared,
same-cutoff training as the default, with disjoint graphs as an explicit control.
Validation/test and embedding extraction stay disjoint. Preserve the original
custom encoder only as this benchmark control. Full runs remain a separate
experiment requiring a dataset/configuration and resource decision.
