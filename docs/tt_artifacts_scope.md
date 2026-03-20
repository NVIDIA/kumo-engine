# Diskgraph TT Artifact Scope

Branch: current working branch (base: `origin/main`)

This branch focuses on the producer-owned train-table artifact contract and avoids carrying legacy split-reconstruction pathways.

## Producer target
Producer owns and emits authoritative artifacts:
1. final train-table parquet with stable `arange`,
2. split assignment artifact (row-level authoritative mapping),
3. stats artifact,
4. manifest with checksums.

## Design constraints
- no JSON-heavy mask payloads; use compact/binary artifacts,
- manifest-last atomic publish,
- idempotent retries (no partial mixed outputs),
- deterministic row ordering contract.

## Carry-over policy
Only bring over reusable implementation details unrelated to legacy reconstruction paths.
