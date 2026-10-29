# Third-party source and Rust license policy

`source-provenance.json` records third-party source copied or forked into this workspace. A workspace member is not automatically first-party: `diskgraph-ptrhash` remains MIT-licensed, retains its upstream notice, and must be excluded from any NVIDIA Apache header backfill.

`deny.toml` is the deny-by-default policy graph. It checks the full Cargo graph, including build dependencies. Licenses with continuing obligations, including MPL-2.0 and CDLA-Permissive-2.0, require a package-and-version exception with an OSRB approval reference instead of a global allow. The exact `option-ext` 0.2.0 exception records the MPL-2.0 retention and source-availability obligations from OSRB-6649311; that optional TUI dependency is policy-checked but absent from the shipped artifact manifests.

`scripts/generate_artifact_manifest.py` resolves the narrower notice graph: x86_64 and aarch64 Linux release targets, no development or build dependencies. It resolves `diskgraph-pybind` and `diskgraph-ingest` separately with their shipped feature sets; the artifact-and-target graphs must not be replaced with a `Cargo.lock` union.

`artifact-manifest.json` is the committed review boundary for those two graphs. Regenerate it with `python3 scripts/generate_artifact_manifest.py --output third_party/artifact-manifest.json`; CI runs the same command in `--check` mode so a dependency, target, feature, or lockfile change requires an explicit manifest delta review.
