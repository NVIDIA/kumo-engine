# AGENTS.md

## Mission

Build a transaction foundation model reference-architecture kit.

This repository is for reproducible educational and solution-architecture examples.
It is not a production transaction-processing, decisioning, or serving system.

Primary goals:

- Demonstrate reusable transaction-model building blocks and architecture patterns.
- Support fair experiments across fraud, recommendations, and representation-learning tasks.
- Make extension points explicit for solution architects and data scientists.
- Prefer clear, runnable reference implementations over maximum feature breadth.

## Read first

Before changing code, read the documents relevant to the task:

- `README.md` for setup and the synthetic quickstart. Recipe tracks:
  `recipes/README.md`. Capability status:
  `docs/reference/compatibility-matrix.md`.
- `CONTRIBUTING.md` for branching, commits, merge requests, releases, and
  review policy. For vulnerabilities, follow `SECURITY.md`.
- `docs/concepts/building-blocks.md` for the five slots every component
 belongs to, and the only rules that connect them (building-block contracts).
- `docs/concepts/data-contract.md` for transaction/profile semantics.
- `docs/reference/datasets.md` before adding a dataset, a task, or a result:
 each dataset has one role, and results from different datasets are never
 combined (dataset strategy).
- `docs/concepts/leakage-and-temporal-splits.md` before changing data, labels, or evaluation.
- `docs/concepts/building-blocks.md` (reference architectures) before changing architecture modules.
- `docs/concepts/evaluation-protocol.md` before changing metrics or baselines.
- `docs/reference/compatibility-matrix.md` before declaring a capability supported.

When modifying a recipe, also read that recipe's `README.md` and configuration.

## Documentation sources of truth

Do not copy status tables into `README.md` or recipe READMEs.

- Recipe inventory: `recipes/catalog.yaml` and `recipes/README.md`.
- Support status: `docs/reference/compatibility-matrix.md`.
- Dataset roles, sources, splits, and tasks: `docs/reference/datasets.md`.
- Reproduction milestone status: `docs/reference/roadmap.md`.
- How to run a recipe: that recipe's `README.md`.
- User-visible changes: `CHANGELOG.md`.
- Public contracts and compatibility: `docs/concepts/`, `docs/reference/`, and `docs/migration/`.

Adding a recipe updates the catalog, `recipes/README.md`, and the compatibility
matrix in the same change. Recipe directories are named slugs (not a numbered
course). Notebook `01`–`05` files inside a series are sequential.

## Repository boundaries

- `src/tfm/contracts/` holds stable, typed, backend-neutral contracts.
- `src/tfm/data/` owns data validation, labels, sequence construction, and splits.
- `src/tfm/features/` owns tokenization and feature representation.
- `src/tfm/api/` owns the public `TrxFM(...)` composition API.
- `src/tfm/evaluation/` owns task and representation evaluation.
- `src/tfm/runtime/` owns recipe orchestration only; nothing else imports it.
- `recipes/` are canonical executable examples.
- `notebooks/` are explanatory only; do not put reusable logic there.

## Non-negotiable domain rules

- Never commit real customer, account, card, merchant, transaction, or identity data.
- Never add secrets, credentials, access tokens, private checkpoints, or proprietary dataset URLs.
- Use only synthetic, public, or explicitly approved de-identified fixture data.
- Preserve temporal ordering unless a recipe explicitly documents another split protocol.
- Do not use future events, post-outcome fields, chargeback-resolution fields, or label-derived fields as model inputs.
- Keep event-level, entity-level, and sequence-level labels distinct in types and APIs.
- Record timezone, observation window, prediction horizon, label delay, and cutoff time where relevant.
- Preserve padding-mask, attention-mask, causal-mask, and sequence-boundary semantics.
- Do not make performance, explainability, fairness, fraud, risk, or production-readiness claims without reproducible evidence.

If a change might introduce temporal leakage or alter label semantics, stop and propose the required contract and evaluation changes before implementing model code.

## Public composition API

- Keep `TrxFM(...)` declarative and readable.
- Treat exported constructor signatures as public API.
- Validate incompatible component combinations before training.
- Separate model semantics from attention-kernel implementation.
- Every exported component requires a minimal example and contract test.
- New objectives must declare label horizon, cutoff semantics, required outputs, gradient behavior, and leakage checks.
- Do not add a public option solely because one experimental recipe requires it.
- The core export surface is exactly: `TrxFM`, `Flat`, `Decoder`, `NoPE`, `EagerAttention`, `LastTokenPool`, `NextEvent`, `Binary`.

## Architecture rules

- Keep public contracts small, typed, stable, and independent of a specific backend.
- Implement new behavior through contracts such as event encoder, history model, pooling strategy, objective, and task head.
- Prefer composition and configuration/registry selection over architecture-specific conditionals.
- Do not put dataframe transformation or label logic in architecture modules.
- New architectures require a documented use case, synthetic smoke test, contract tests, and support tier.
- Mark a new architecture `experimental` until its quality, stability, and runtime behavior are documented.
- Do not silently change public batch, model-output, checkpoint, or configuration contracts.

For a breaking contract change:

1. Update the applicable public contract documentation.
2. Update affected recipes and the compatibility matrix.
3. Add migration notes.
4. Update contract and regression tests.

## Recipe rules

Every recipe must contain:

- A `README.md` with goal, architecture, task, data assumptions, commands, expected outputs, and limitations.
- A committed configuration with explicit random seed.
- Train, evaluate, and inference entry points, or a documented equivalent.
- A CPU-safe synthetic-data smoke test.
- Appropriate task baselines.
- A declared support tier: `reference`, `experimental`, or `deprecated`.
- A matching `recipes/catalog.yaml` row and `recipes/README.md` track entry.

## Testing and validation

Run the narrowest relevant checks while developing. Before finalizing, run:

```bash
uv run ruff check src tests recipes
uv run ruff format --check src tests recipes
uv run pyright src/tfm
uv run pytest tests/unit -q
uv run pytest tests/contract -q
```

After changing a recipe, data pipeline, architecture, objective, inference path, evaluator, or configuration, also run:

```bash
uv run pytest tests/recipe_smoke -q
```

Do not report a test as passing unless it was executed. If GPU validation cannot run, state that explicitly and run the relevant CPU-safe tests.

## Coding standards

- Target the Python version defined in `pyproject.toml`.
- Type all public functions, protocol methods, configurations, and output objects.
- Use structured configuration objects for non-trivial settings.
- Set seeds in tests and recipes where randomness affects behavior.
- Use small deterministic synthetic fixtures in tests.
- Test expected tensor shapes, mask behavior, sequence ordering, and label alignment.
- Raise explicit errors for invalid schema, incompatible components, missing labels, and unsupported combinations.
- Avoid silent fallbacks and broad exception handling.
- Do not add dependencies without documenting their purpose and updating lockfiles.

## Performance and hardware

- Keep a correct eager baseline for every optimized inference path.
- Do not require a GPU for unit, contract, or recipe-smoke tests.
- Treat GPU-specific kernels and external backends as optional integrations with explicit capability checks.

## Change workflow

1. Inspect nearby implementation, tests, contracts, and recipe documentation.
2. Make the smallest coherent change that satisfies the task.
3. Add or update tests in the same change.
4. Update documentation when behavior, support status, configuration, or outputs change.
5. Run required validation.
6. Report changed files, commands run, commands not run, and remaining risks.

Do not refactor unrelated code as part of a focused task.
Do not remove a working baseline merely because a newer architecture performs better.
