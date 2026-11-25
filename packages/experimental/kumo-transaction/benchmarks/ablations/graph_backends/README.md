# Engine architecture and graph-sharing cost controls

This short synthetic benchmark compares three graph training arms:

| Arm | Encoder | Graph policy |
|---|---|---|
| `engine_shared` | Installed Kumo Engine `HeteroGNN` | Union the same-cutoff per-root neighborhoods |
| `engine_disjoint` | The same Engine class and identical initial full state | Separate copy of each root's sampled graph |
| `legacy_disjoint` | The initial MR !8 mean/LayerNorm/GELU encoder from `9a7e9b1` | The exact same separate input graph as Engine disjoint |

The historical encoder is a benchmark-only control. Recipes always use the
actual Engine implementation and fail explicitly if the `kumo` extra is absent.
No architecture is substituted to make an Engine test pass.

## Run the short check

From the repository root:

```bash
uv sync --extra dev --extra kumo
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m benchmarks.ablations.graph_backends.benchmark \
  --seed 0 --entities 64 --events-per-entity 12 --transfers-per-entity 4 \
  --batch-size 32 --width 512 --depth 6 --fanout 10 --warmup 2 --steps 5 \
  --device cuda --output outputs/graph_backend_cost.json
```

Use a quiet GPU and run all arms serially. Do not run another CUDA test while
collecting timing. A small CPU harness check needs no GPU:

```bash
uv run python -m benchmarks.ablations.graph_backends.benchmark \
  --entities 8 --events-per-entity 4 --transfers-per-entity 2 \
  --batch-size 4 --width 8 --depth 2 --warmup 0 --steps 1 \
  --device cpu --output outputs/graph_backend_cost_cpu.json
```

The dataset is generated in memory with explicit seed, node features, FK roles,
UTC clocks, and root identities. It adds no dataset role or private data source.
Sampling is deterministic and happens before timing model steps. The same
per-root sampled neighbors create the shared union and disjoint graph. The
Engine arms load the same initial weights and BatchNorm buffers, verified by a
state hash. All arms use the same scalar targets, FP32, AdamW (LR 0.003, weight
decay 0.002), clipping at 1.0, warm-up count, and measured step count.

## What is measured

The JSON records source revision/dirty state, Engine version and GNN module
hash, PyTorch/PyG/CUDA versions, optional torch-scatter availability, GPU name,
TF32 setting, parameters, node/edge/sample counts, initial-state hashes, every
step latency and loss, median step latency, and peak allocated/reserved CUDA
memory. Sampling and model/batch transfer time are separate. CUDA synchronization
bounds forward, scalar MSE, backward, clipping, and optimizer update. Peak memory
uses PyTorch allocator counters after warm-up; it excludes driver/context and
other processes' allocations. CPU runs report no CUDA memory value.

An out-of-memory arm is reported explicitly; there is no smaller-batch or CPU
fallback. This check is a prerequisite to deciding whether a longer experiment
fits, not authorization or evidence for a full quality campaign.

## Interpretation limits

Shared and separate training graphs weight unique/repeated nodes differently
under BatchNorm and may expose different context through unioned neighborhoods.
Their training losses need not match even from identical initial weights.
Validation/test and recipe embedding extraction always use separate graphs.
The Engine-versus-legacy disjoint comparison keeps those input graphs fixed but
changes the architecture and parameter count intentionally.

This is a fixed-batch GNN cost control, not a reproduction of an external full
model/trainer. Engine now uses validated hop trimming and linear inputs without
an extra ReLU; the historical custom encoder consumes the same input graph
without layer trimming. It excludes native loading, historical feature fitting,
target preparation, sequence stages, validation,
checkpointing, and full-dataset convergence. It cannot establish native-loader
parity, end-to-end throughput, downstream quality, or the best batching policy.
Record installed kernel dependencies because PyG's max aggregation may use a
slower Torch kernel when optional `torch-scatter` is absent. Do not extrapolate
one small generated graph's sharing ratio to another graph's overlap.

The recorded short check predates hop trimming and input ReLU removal; it is
historical evidence for its recorded source only, not current throughput.
Recorded short check: [six-hop L4 costs](../../results/graph_backends_l4_cost.md).

## CPU validation and repeated transfer

The execution optimization keeps architecture, trimming, objective, optimizer,
and sampling settings unchanged. CPU constructors validate temporal/index/hop
contracts and cache receiver cardinality. Device copies preserve validation;
tracked tensor changes invalidate it. See the batch contract for read-only
ownership and forbidden external writes.

The original three-arm harness above still measures GPU-resident optimizer
steps and reports one-time model/batch transfer separately. It does not include
the per-step transfer used by the recipe trainer. The companion driver measures
that execution path using the same six-hop, width-512, batch-32, fanout-10 graph,
FP32 Engine, loss, clipping and AdamW settings:

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync python benchmarks/ablations/graph_backends/execution.py \
  --warmup 5 --steps 20 --output /tmp/graph-execution.json
```

For a before/after comparison, run the **identical driver file** with each
checkout's `src` and repository root first on `PYTHONPATH`. It records revision
and dirty state from the imported graph module's checkout, plus the driver
hash, initial model hash, all measured transfer/cycle times and losses, final
predictions, allocator peaks, environment and settings. Alternate revisions,
run sequentially without other CUDA/CPU-heavy work, and compare repeated runs.
Sampling and initial model transfer are excluded from cycle timing; blocking
batch transfer is included. This isolates execution costs on a fixed batch,
not data loading, a full training job, or model quality.

Recorded before/after results: [CPU-validation execution costs on L4](../../results/graph_batch_execution_l4.md).


## Execution views

`views.py` reconstructs/validates fixed synthetic batches, transfers them, then
runs four-hop, fanout-10, width-32 FP32 GNN training, hybrid sequence training,
and hybrid validation. It selects `.for_execution()` when available and `.to()`
otherwise. Sampling is excluded; construction, blocking transfer and model work
are timed separately and together. It records input/model/driver hashes and
raw timings. This complements the earlier GPU-resident and transfer-only checks.

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  uv run --no-sync python benchmarks/ablations/graph_backends/views.py \
  --deterministic --warmup 5 --steps 20 --output /tmp/graph-views.json
```

Run this identical driver against each source checkout, alternating revisions
and avoiding other heavy work. This synthetic microbenchmark does not establish
Distress/native-loader throughput, convergence or quality.

The recorded comparison enables deterministic algorithms and fixes Python's
hash seed so training trajectories can also be compared. Normal CUDA scatter
reductions can vary between repeated runs of the same source; raw nondeterministic
training embeddings after many updates are not an exact parity check.

Recorded comparison: [execution views on L4](../../results/graph_execution_views_l4.md).


## Constructor and compilation ablations

`conversion.py` instruments the same fixed workload as `views.py`. Thin identical
wrappers measure `_seal()` compilation within each CPU reconstruction. Compilation
is a subset of constructor time. Final model-state (including buffers) and gradient
hashes are captured outside cycle timing, alongside existing input/initial-state
hashes, loss histories and final embeddings. Actual roots and effective optimizer
settings are recorded for every workload.

Use the identical driver and `views.py` files for baseline, combined-edge-only
and combined-edge-plus-accumulator sources. First run `--warmup 0 --steps 1` and
require matching inputs, initial/final state, gradients, losses and embeddings
across all sources. Then run repeated sequential timings in balanced source order:

```bash
PYTHONPATH=<source>/src:<driver-checkout> OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  <prepared-environment>/bin/python \
  <driver-checkout>/benchmarks/ablations/graph_backends/conversion.py \
  --deterministic --warmup 5 --steps 20 --output /tmp/graph-construction.json
```

Sampling, raw/native-loader adaptation and initial model transfer are excluded.
CPU reconstruction/validation, blocking transfer and model work are included.
This synthetic L4 check does not reproduce real-loader conversion costs or imply
full-book throughput or model-quality improvements.

Recorded ablation: [constructor and compilation costs on L4](../../results/graph_constructor_checks_l4.md).


All cost drivers use source-checkout provenance independent of cwd and inherited
Git overrides. Missing Git records null revision/dirty state while preserving
results; untracked source files count as dirty even when local Git config hides
them. `views.py`/`conversion.py` now record throughput from actual roots and the
complete-cycle median, plus PyTorch CUDA allocated/reserved peaks reset after
warmup. Peak counters are sampled outside timed work; they do not measure host
RSS or total device memory. Historical missing peaks are supplied only through
separate, explicitly identified resource runs, never filled into old raw payloads.
Full raw artifacts are referenced from compact manifests under `benchmarks/results`.
