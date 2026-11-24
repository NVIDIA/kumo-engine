# ADR 0001 — Domain contracts

## Status

Accepted

## Decision

Keep backend-neutral typed contracts (`TransactionEvent`, `TransactionSequence`,
`TransactionBatch`, `ModelOutput`) in `src/tfm/contracts/`. Model backends and
recipes depend on these contracts; contracts do not depend on NeMo, Hugging
Face, or RAPIDS.

## Consequences

Architecture modules receive validated batches only. Dataframe and label logic
stay in `src/tfm/data/`. Breaking contract changes require ADR updates, recipe
updates, and contract tests in the same merge request.
