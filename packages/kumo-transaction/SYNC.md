# Source synchronization

This directory imports the tracked source tree identified in [UPSTREAM.json](UPSTREAM.json).
The branch name `transaction_foundation_2_4` identifies this sync; the imported
Python distribution's version remains `2.0.0a4`.

This sync advances the source from `8a67f98d5a7cfb4c1feeee06c0ce6383f5520671`
to `6331312c3e85d6ce1fc868c99b1ffffc03bc50c5` and depends on
[PR #507](https://github.com/NVIDIA/kumo-engine/pull/507) merging into Engine
`main` first. Keep the child PR draft until that merge. After the parent merges,
rebase only this sync's commit(s) onto current `main` (especially after a squash
merge), verify the diff contains only the incremental sync, and rerun checks
before marking the child ready. Do not merge the child into the parent branch.

## Local adaptations

- `README.md`: retain the Kumo Transaction entry point and explain package-local commands.
- `pyproject.toml`: link to this public package and declare an independent uv
  workspace so package-local commands use this kit's lockfile and dependencies.
- `mkdocs.yml`: use public repository metadata and remove the private Pages URL.
- `benchmarks/results/pragmatiq_fraud_separability.json`: replace the original
  developer's absolute dataset path with its dataset name; preserve all measurements.

- `docs/_site/api/*.md`: use public source links for concept pages so the
  API pages resolve in both the source tree and the rendered site.
- `benchmarks/results/README.md`: repair the stale config-migration link and
  explain that raw graph artifacts are unavailable from this public snapshot.
- `benchmarks/results/graph_*.json`: replace private raw/resource artifact URLs
  with JSON `null`. Preserve measurements, source revisions, hashes, and sizes.
  These are import-time adaptations to the pinned generated manifests; rerun
  the same URL-only substitution after regenerating them upstream.
- `scripts/gen_doc_pages.py`: include the sync guide and provenance inventory
  in the generated documentation site.

Engine's `docs/maintenance/agent-contract.json` registers the imported
instruction scopes.

All other imported files match the pinned source byte for byte. The source
hash inventory covers every imported file, including adapted files before
adaptation. `UPSTREAM.json` and this guide are local additions.
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

A source sync does not publish a release or change the upstream support tier.
