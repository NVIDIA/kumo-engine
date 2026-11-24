# Recipes

A recipe is one reference architecture: a directory of configs and a README.
Every stage runs through the CLI, resolved from [`catalog.yaml`](catalog.yaml):

```bash
uv run tfm smoke  --recipe blueprint_v1              # config.smoke.yaml, CPU
uv run tfm report --recipe blueprint_v1              # config.yaml
uv run tfm report --recipe blueprint_v1 --config small
uv run tfm report --recipe blueprint_v1 --config path/to/edited.yaml
```

`train`, `infer`, `evaluate`, and `report` each run the stages before them, so
`report` alone trains, embeds, evaluates, and writes a model card. Pass
`--run-dir` to choose the output directory and `--reuse-checkpoint` to
re-evaluate a trained run.

| Recipe | Architecture | Data | Notebooks |
|---|---|---|---|
| [`blueprint_v1`](blueprint_v1) | NVIDIA TFM Blueprint v1: flat field tokens, causal Llama decoder, XGBoost on raw features plus the embedding | TabFormer (smoke: synthetic) | [`blueprint_v1`](../notebooks/blueprint_v1) |
| [`pragma_like`](pragma_like) | Revolut PRAGMA: bidirectional profile, event, and history encoders, masked modelling | pragmatiq | [`pragmatiq`](../notebooks/pragmatiq) |
| [`blueprint_v2`](blueprint_v2) | NVIDIA TFM Blueprint v2: PRAGMA's profile and event encoders under a causal history decoder | pragmatiq | [`pragmatiq`](../notebooks/pragmatiq) |

| [`gnn_encoder`](gnn_encoder) | Heterogeneous GNN, binary supervision or future-subgraph pretraining | generated relational fixture; optional local pragmatiq | — |
| [`gnn_sequence`](gnn_sequence) | Frozen GNN conditions Blueprint v2 through its profile encoder | generated relational fixture; optional local pragmatiq | — |

Start with **blueprint_v1** for the end-to-end flow on public data. Use
**pragma_like** for a PRAGMA-style encoder, and **blueprint_v2** for the
hybrid, which also generates future events. `pragma_like/config.yaml` and
`blueprint_v2/config.yaml` share their rows, split, tokens, and probes, so the
two architectures are compared on the same users.

Each recipe has `config.smoke.yaml` (CPU, generated data, run by
`tests/recipe_smoke`) and `config.yaml` (its reference size); other variants
are listed in the catalog. Slot-by-slot ablations live in
[`benchmarks/ablations`](../benchmarks/ablations), not here.

Support status lives in
[docs/reference/compatibility-matrix.md](../docs/reference/compatibility-matrix.md).
What each dataset is for, and why results from different datasets are never
combined, is in [docs/reference/datasets.md](../docs/reference/datasets.md).
To add a recipe, follow
[docs/recipes/recipe-template.md](../docs/recipes/recipe-template.md).


Graph recipes default to direct binary supervision (`training.objective: binary`),
including supervised GNN warm-up for the hybrid. Both offer `--config pretrain`
and `--config pretrain_smoke` for optional FutureSubgraph training. Their Engine
GNN uses hop trimming and their sequence stack uses SDPA by default.


The graph recipe track includes `pretrain_only_smoke` and `downstream_smoke`
variants for separating label-free encoder training from supervised use. The
recipe READMEs describe the stage commands and artifact paths.
