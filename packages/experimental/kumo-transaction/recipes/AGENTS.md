# recipes/AGENTS.md

A recipe is one reference architecture. Every recipe must include:

- `README.md` (goal, architecture, task, data assumptions, commands, artifacts, limitations, support tier)
- `config.smoke.yaml`: CPU-safe, generated data, explicit seed; run by a test in `tests/recipe_smoke`
- `config.yaml`: the reference size, explicit seed
- Task-appropriate baselines

There are no per-recipe scripts. `tfm <stage> --recipe <slug> [--config <variant>]`
runs every stage; `--config small` resolves to `config.small.yaml`.

Support tiers: `reference`, `experimental`, `deprecated`.

Every `config*.yaml` in a recipe is listed under its `configs:` in
[`catalog.yaml`](catalog.yaml), and `tests/unit/test_recipe_catalog.py` checks
it. Keep [`README.md`](README.md) and `catalog.yaml` in the same change when
adding a recipe. Notebooks are one series per dataset under `notebooks/`;
`01`–`05` files inside a series are sequential.

Variants that swap one slot to measure it belong in `benchmarks/ablations`,
not in a recipe.

Do not require a GPU, RAPIDS, or private data for the smoke path.
Representation recipes must report a non-TFM baseline, an embedding-only model,
and a combined model when the task allows.
