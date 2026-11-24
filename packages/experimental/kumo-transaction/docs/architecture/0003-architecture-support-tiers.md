# ADR 0003 — Architecture support tiers

## Status

Accepted

## Decision

Every recipe and public component declares `reference`, `experimental`, or
`deprecated` (or `planned` before implementation). The core export surface
implements only the flat causal decoder path. Hierarchical,
TwoBranch, FlashAttention, Mamba, LoRA, ranking, and explainability are not
stubbed as public API.

The PRAGMA-style `ProfileHistory` stack is **experimental**: validated by
`validate_composition` and imported from `tfm.experimental` (ADR 0007). It is
not part of the core export surface (`tfm.__all__`).
