# ADR 0002 — Config composition

## Status

Accepted

*Update, 2026-10-03:* `training.backend: nemo` is removed in 2.0.0a2; `fsdp2` trains the same HuggingFace Llama.

## Decision

- Python 3.12 and `uv` are the standard installer.
- Configuration is Pydantic v2 models loaded from YAML. YAML is a serialization
  of the `TrxFM(...)` composition API, not a second model language.
- Training uses a small eager PyTorch loop. NeMo AutoModel is an optional extra
  and is not required for CI.
- Fraud smoke classifier is sklearn logistic regression; XGBoost is optional.
- Decoder implementation is a tiny PyTorch module (HuggingFace Llama remains an
  optional later backend).

Do not reopen these defaults without a superseding ADR.
