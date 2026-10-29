# DiskGraph import provenance

This repository is the sole source of truth for DiskGraph. Make engine changes
and add tests directly in `packages/diskgraph`; no upstream synchronization or
mirror verification is required.

`source_revision.txt` and `generated/source_manifest.json` record the historical
import from `kumo-ai/kumo-diskgraph`. They are retained for provenance and do
not describe the current tree or need updating when DiskGraph changes. The
imported ancestry remains reachable in Git history.

DiskGraph retains its package-local formatting and lint policy. Repository-wide
Python format, lint, and type checks continue to exclude `packages/diskgraph`;
Kumo integration code remains covered by those checks. Follow the package's
build and test instructions and regenerate affected
[third-party notices](../../../3rdparty/README.md) after dependency changes.
