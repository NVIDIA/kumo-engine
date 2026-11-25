# Hardware and runtimes

What runs where. Every unit, contract, recipe-smoke, and notebook check runs
on CPU; GPUs are needed only for the scaled configs.

## Training backends

`training.backend` selects how a model trains, not what it computes. Each
backend declares its capabilities (`BackendCapabilities`, backend contracts), and the
runtime refuses a config that needs something the backend does not declare,
before any data is prepared.

| Backend | Tier | Where | Backbones | Layouts | Precision | Notes |
|---|---|---|---|---|---|---|
| `eager` | `reference` | CPU or one GPU | `eager`, `huggingface`, `gru` | all three | `fp32`, `bf16` | The correctness baseline every other backend must match |
| `ddp` | `experimental` | several processes under `torchrun`, CPU (Gloo) or GPU (NCCL) | `eager`, `huggingface`, `gru` | all three | `fp32`, `bf16` | Full replica per rank; `training.batch_size` is per rank; rank 0 writes artifacts |
| `fsdp2` | `experimental` extra | GPUs under `torchrun` | `huggingface` | `flat_fields` | `bf16` | Shards parameters; writes a consolidated HuggingFace checkpoint |

`bf16` autocasts the forward pass over fp32 master weights. Attention
kernels are separate from backends: `attention: sdpa` lets PyTorch dispatch
to fused kernels on CUDA, and `eager` attention stays the reference.

## Recipe configs by hardware

| Config | Parameters | Hardware | Recorded result |
|---|---|---|---|
| `recipes/*/config.smoke.yaml` | tiny | CPU, seconds | Smoke only |
| `benchmarks/ablations/blueprint_v2_lab.yaml` | 8.4M | one GPU, about 55 min, 7.3 GiB | [`blueprint_v2_hybrid_lab_kvt_seeds`](../../benchmarks/results/blueprint_v2_hybrid_lab_kvt_seeds.md) |
| `blueprint_v2/config.yaml` | 8.4M + embedding table | one GPU, token cache; about 24 min to train | [`blueprint_v2_hybrid_gb10_tuning`](../../benchmarks/results/blueprint_v2_hybrid_gb10_tuning.md) |
| `benchmarks/ablations/blueprint_v2_final_d.yaml` | 8.4M + embedding table | one GPU, token cache; about 42 min to train | [`blueprint_v2_final_sweep`](../../benchmarks/results/blueprint_v2_final_sweep.md) (candidate D) |
| `blueprint_v2/config.scale.yaml` | about 100M | one or more GPUs (`ddp`) | none yet |
| `pragma_like/config.pragma_s/m/l.yaml` | PRAGMA-S/M/L | GPU; L needs `bf16` and two GPUs | none yet |
| `blueprint_v1/config.yaml` | 25.9M | one GPU with 128 GB memory; about 80 min to train, 2 h to evaluate every test row, about 63 GB host memory | [`blueprint_v1_seeds`](../../benchmarks/results/blueprint_v1_seeds.md) (three seeds) |
| `blueprint_v1/config.cached.yaml` | 25.9M | one or more GPUs (`ddp`); training reads a token cache, about 3 GB host memory per rank | none yet |
| `benchmarks/ablations/blueprint_v2_lab_v1_flat.yaml` | 6.6M | one GPU, about 58 min, 7.6 GiB | [`blueprint_v2_hybrid_lab_v1_flat_splice_seeds`](../../benchmarks/results/blueprint_v2_hybrid_lab_v1_flat_splice_seeds.md) |

Each recipe README lists the measured memory and step time for its configs.

## Moving a run to a GPU

```bash
uv run python recipes/<recipe>/train.py --config <config> --device cuda \
  --run-dir outputs/<recipe>/<run-id>
uv run torchrun --nproc-per-node N recipes/<recipe>/report.py \
  --config <config> --device cuda --run-dir outputs/<recipe>/<run-id>
```

`--device auto` picks CUDA when it is available. Pass the same `--run-dir` to
every rank. Add `--reuse-checkpoint` to re-evaluate a trained run without
retraining.

### Python headers for Triton

With torch 2.13 on CUDA, the HuggingFace Llama backbone's first forward pass
launches a Triton kernel, and Triton compiles a small C stub against
`Python.h`. Without the Python development headers, that fails with
`fatal error: Python.h: No such file or directory`. Install them with
`sudo apt install python3.12-dev`. Without root, point the compiler at a
uv-managed interpreter's headers, which are compatible within a minor
version:

```bash
uv python install 3.12
export C_INCLUDE_PATH=$(uv python find 3.12 --managed-python | \
  xargs -I{} sh -c '{} -c "import sysconfig; print(sysconfig.get_paths()[\"include\"])"')
```

The eager backbone and the CPU paths need neither.

## Optional extras

GPU-specific and third-party integrations are optional extras behind
capability checks (optional dependency guidance): `huggingface` (Llama backbone), `fsdp2`, `rapids`
(cuDF tokenizer), `xgboost` (probe head), `umap`. A missing extra raises a
clear error when a config asks for it; nothing imports it by default.

## Deferred

A container image, an NGC image, and a hosted launch environment are planned
but not part of this release.
