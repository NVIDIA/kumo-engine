# Recipe template

A recipe is one reference architecture. Copy the closest of
`recipes/blueprint_v1/`, `pragma_like/`, or `blueprint_v2/` and change the
architecture and task sections.

Required files:

- `README.md`: goal, architecture, task, data assumptions, commands, expected
  outputs, limitations, and support tier.
- `config.smoke.yaml`: CPU, generated data, explicit seed, a few training
  steps. Add a test under `tests/recipe_smoke/` that runs it with
  `tfm.runtime.run_smoke`.
- `config.yaml`: the reference size, explicit seed.

There are no per-recipe scripts: `tfm <stage> --recipe <slug>` runs every
stage. Add a row to `recipes/catalog.yaml` listing every `config*.yaml` under
`configs:`, an entry in `recipes/README.md`, and a row in the compatibility
matrix, in the same change. A variant that swaps one slot to measure it is an
ablation under `benchmarks/ablations/`, not a recipe config.
