# ADR 0005 — Artifact policy

## Status

Accepted

## Decision

- `outputs/` is gitignored run scratch space.
- `artifacts/` stores manifests and model-card templates, not checkpoints or datasets.
- v1 Git LFS weights remain on tag `v1.0.0` / branch `v1`.
- Never commit real financial records, credentials, or proprietary URLs.
