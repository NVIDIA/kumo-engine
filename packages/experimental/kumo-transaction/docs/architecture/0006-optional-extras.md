# ADR 0006 — Optional extras and local dataset roots

## Status

Accepted

*Update, 2026-10-03:* `training.backend: nemo` is removed in 2.0.0a2; `fsdp2` trains the same HuggingFace Llama. `decoder_fraud_baseline` is now `blueprint_v1/config.smoke.yaml`, the synthetic reference.

## Decision

- HuggingFace Llama, RAPIDS tokenization, and NeMo training are **experimental
  extras** under `src/tfm/extras/`. They are not part of the core export surface.
- CPU CI must not import `transformers`, `cudf`, or `nemo_automodel`. Extras
  raise `ExtraUnavailableError` with an install hint when missing.
- Dataset roots are configured (`data.path` or `TFM_TABFORMER_ROOT` /
  `TFM_PRAGMATIQ_ROOT`). Do not commit symlinks or TabFormer/pragmatiq files.
- Adapters subsample entities (`max_entities` / `n_entities`) so large books
  are not fully materialized. When fraud labels exist, the subsample prefers
  users with at least one positive.
- Llama is a `HistoryModel` backbone; FSDP2 is a `Trainer` backend (ADR 0011).
  Eager training can use the HF backbone without either.

## Consequences

Recipe `decoder_fraud_baseline` stays the synthetic reference. TabFormer +
HF lives in experimental recipe `blueprint_v1`.

## Native loader source dependency

After 2.0.0a4, the experimental `kumo-pytorch`/DiskGraph loader is an optional
source integration selected by the `native` extra and required by the default
recipe loader. The Python oracle remains
explicitly selectable; the Engine GNN stays optional through `--extra kumo`,
which installs matching `kumo-core` rather than the older umbrella wheel. CI
builds one verified native wheel set for the pinned main revision and tests the
optional GNN separately. See the [migration guide](../migration/native-loader-default.md).
