# Source synchronization

This directory imports the tracked source tree identified in [UPSTREAM.json](UPSTREAM.json).
The branch name `transaction_foundation_2_2` identifies this sync; the imported
Python distribution's version is `2.0.0a4`.

PR #507 consolidates the initial import and the later sync previously proposed
in #508. It advances from `8a67f98d5a7cfb4c1feeee06c0ce6383f5520671` to
`6331312c3e85d6ce1fc868c99b1ffffc03bc50c5` while retaining the experimental
location and Engine integration checks. There is no separate prerequisite PR.

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
