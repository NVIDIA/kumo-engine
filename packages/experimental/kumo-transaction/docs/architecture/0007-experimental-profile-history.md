# ADR 0007 — Experimental ProfileHistory composition

## Status

Accepted

## Context

The core export surface is only `TrxFM(Flat(Decoder), NoPE, EagerAttention, NextEvent, LastTokenPool, Binary)`.
Recipe `pragma_like` implements a PRAGMA-style profile / event / history encoder
with TimeRoPE and token/event/key MLM, but historically constructed `ProfileHistoryModel`
outside composition validation.

The kit needs a documented experimental composition path for the PRAGMA teaching recipe
without expanding the core public `tfm.__all__` surface.

## Decision

- `ProfileHistory` + `TimeRoPE` + `EagerAttention` + `TokenEventKeyMLM` + `UsrTokenPool`
  is a **valid experimental composition**. `validate_composition` accepts it and still
  rejects mixed stacks (e.g. `Flat` + `TimeRoPE`, `ProfileHistory` + `NextEvent`).
  `SdpaAttention` is interchangeable with `EagerAttention` here; eager stays the
  correctness baseline.
- Callers import from `tfm.experimental` (`compose_profile_history`, `ProfileHistory`,
  `TimeRoPE`, `TokenEventKeyMLM`, `UsrTokenPool`). These names are **not** added to
  `tfm.__all__`.
- Support tier remains `experimental` until tokenizer fidelity is documented and a
  scaled run is recorded. Row-level varlen (`granularity.varlen`) and LoRA
  (`fraud_task.finetune`, `method: lora`) are implemented and `experimental`; HuggingFace BPE
  tokenization and the Muon optimizer stay `planned`.
- Recipe `pragma_like` is the canonical executable path; notebooks under
  `notebooks/pragmatiq/` are explanatory only.

## Consequences

- Architects can compose the PRAGMA-style stack through the same validation gate as
  the core surface without implying a stable public API.
- Promoting ProfileHistory into `tfm.__all__` requires this ADR to be superseded,
  plus compatibility-matrix and migration notes.
