# src/tfm/AGENTS.md

## Public API

Export only documented names from `tfm`. The core export surface:

`TrxFM`, `Flat`, `Decoder`, `NoPE`, `EagerAttention`, `LastTokenPool`, `NextEvent`, `Binary`.

Treat those constructors as stable. Do not add a public option solely for one recipe.

Experimental PRAGMA-style names live in `tfm.experimental` (experimental composition guide) and are
not added to `tfm.__all__`.

## Runtime layer

`tfm/runtime/` is recipe orchestration, not library architecture. It may depend
on any other package here; nothing else may depend on it.

- `runtime/common.py` holds the lifecycle shared by every recipe. Put a new
  manifest writer or config helper here, not in a per-stack module.
- `runtime/decoder.py` and `runtime/kvt.py` implement only what genuinely
  differs between stacks.
- Both stacks train through `tfm.training.build_trainer`. Do not add an inline
  optimizer loop to a runtime module.
- Recipes have no scripts: `tfm <stage> --recipe <slug>` resolves the config
  from `recipes/catalog.yaml` (`cli/recipe.py`) and calls `runtime.pipeline`.

## Typing and tests

- Type public functions, protocols, configs, and output objects.
- Unit tests live in `tests/unit/`; contract tests in `tests/contract/`.
- Optional extras (`xgboost`, `huggingface`, `rapids`, `fsdp2`) live in
  `src/tfm/extras/` and must be isolated so CPU CI does not import them.

## Dependencies

Keep the required dependency set small. RAPIDS and Transformers are optional.
Do not import optional backends at module top-level in required packages.
