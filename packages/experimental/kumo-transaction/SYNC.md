# Source synchronization

This directory imports the tracked source tree identified in [UPSTREAM.json](UPSTREAM.json).
The snapshot is based on `6331312c3e85d6ce1fc868c99b1ffffc03bc50c5`,
with the native-loader delta through `c0205d98dec9d6fd918afc575dcf7a88af87e4f9`.
The imported distribution remains `2.0.0a4`; the loader changes are unreleased.
The source revision and hashes are recorded in `UPSTREAM.json`.

This sync ports native sequence-loader defaults and graph/hybrid recipes into
Engine. It preserves the standalone package's pinned Kumo PyTorch dependency
revision rather than changing the experimental runtime during migration.

## Local adaptations

- `README.md`: retain the Kumo Transaction entry point and explain package-local commands.
- `pyproject.toml`: use public metadata, SPDX `Apache-2.0`, the package-owned
  `LICENSE`, a setuptools>=77 build backend, and an independent uv workspace.
- `mkdocs.yml`: use public repository metadata and remove the private Pages URL.
- `benchmarks/results/pragmatiq_fraud_separability.json`: replace the original
  developer's absolute dataset path with its dataset name; preserve all measurements.

- `docs/_site/api/*.md`: use relative links that resolve within this source
  snapshot. The documentation generator adapts them in its virtual API-page
  copies to preserve links on the rendered site.
- `examples/kumo-engine/` uses Engine's managed DiskGraph build and snapshot
  APIs and installs the matching settings package from Engine source.
- `CHANGELOG.md`: record the managed DiskGraph example migration.
- `benchmarks/results/README.md`: repair the migration link and explain that
  upstream raw graph artifacts are not available from this public snapshot.
- `benchmarks/results/graph_*.json`: replace private artifact URLs with JSON
  `null` while preserving source revisions, hashes, sizes and measurements.
  This import-time transformation is applied to the pinned upstream manifests;
  reapply it if those manifests are regenerated upstream.
- `Makefile` and `scripts/run_checks.sh`: select the package-owned Pyright
  configuration explicitly rather than inheriting Engine-wide diagnostics.
- `scripts/setup_dev_env.sh` and `scripts/run_checks.sh`: install the native
  loader extra required by the new recipe default.
- `scripts/gen_doc_pages.py`: include sync/provenance files in the site and
  adapt source-relative API links for its rendered layout.

Engine's `docs/maintenance/agent-contract.json` registers the imported
instruction scopes.

`UPSTREAM.json` also lists 54 files under `header_added_files`. These authored
configuration/build/example files had no file-level attribution; their pinned
source tree declares NVIDIA copyright and Apache-2.0 in its own `LICENSE`.
Engine adds canonical SPDX headers without changing executable content. No
provenance exclusions are needed. Existing Blueprint two-line SPDX headers
stay byte-for-byte intact; Engine recognizes that exact form only here.

The always-reporting `transaction-blueprint` CI check detects package and policy
changes before running frozen CPU checks and wheel-license verification. Root
Flake8/isort defer to the package's Ruff policy. Making this status required
needs repository branch-protection configuration after the workflow is available.

All other imported files match the pinned source byte for byte. The source
hash inventory covers every imported file, including adapted files before
adaptation. `UPSTREAM.json`, this guide and the managed DiskGraph example smoke
test are local additions.
Nested GitLab CI, ownership, and governance files are source context only;
they do not configure Kumo Engine CI, ownership, releases, or contributions.

## Future sync PRs

1. Create a new branch from current Engine `main` for each sync. Fetch the
   source project and check out an explicit full commit SHA.
2. Compare that source tree with the previous `source_commit`, then apply the
   upstream delta here. Preserve Engine changes, review deletions, and reapply
   the documented local adaptations. Do not replace this directory blindly.
3. Review imported data, notebooks, URLs, and artifacts for public suitability.
   Preserve licenses and attribution. Update `UPSTREAM.json` with the new
   revision, source version, adapted-file list, and SHA-256 of every tracked
   upstream file before adaptations. Document any exclusions explicitly.
4. From this directory, run the package's lint, typing, unit, contract, and
   recipe-smoke checks (see `Makefile`), plus strict docs and notebook checks
   when their dependencies are available. Run Engine's publication checks.
5. Open a PR against Engine `main` recording the old and new source commits,
   local adaptations, validation results, and any skipped checks.

A source sync does not publish a release or change upstream support tiers.
The a4 graph recipes use the optional Kumo Engine backend described upstream.

## Native loader integration

The authoritative GitHub `transaction-blueprint` job now builds or verifies the
six matching native dependency wheels, installs them with the package's locked
CPU test environment, and exercises the real Engine graph models. Its cache
validates source/build identity and wheel hashes. A cold native build needs a
longer job budget; cache hits avoid rebuilding Rust. The nested GitLab workflow
is retained only as imported source context.

The package lockfile remains independent of the Engine workspace. Use the
package-local setup commands and `native` / `kumo` extras. The model, cutoff,
label and target contracts are unchanged; native inputs remain experimental,
and FutureSubgraph targets retain their separate Python reference pass.

The advanced example startup overlaps the narrower work proposed in #524.
This broader sync also updates loader defaults and canonical graph/hybrid
runners; it does not include #524's additional snapshot-reuse interface.
