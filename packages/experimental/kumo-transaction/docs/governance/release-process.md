# Release process

This process applies to maintainers releasing the v2 reference-architecture
kit. It does not authorize publishing packages, container images, models, or
datasets to an external registry.

## Release policy

- Release from a reviewed commit on the `v2` branch after the required GitLab
  pipeline and approvals have completed.
- Use the package version from `pyproject.toml` and create an annotated Git tag
  named `v<version>`, for example `v2.0.0a2`.
- Version changes communicate compatibility: increment the major version for a
  breaking public contract, the minor version for backward-compatible
  capability additions, and the patch version for backward-compatible fixes.
  Pre-releases use the package-version suffixes already supported by Python
  packaging, such as `aN`, `bN`, and `rcN`.
- Do not move or reuse a published release tag. Publish a corrective release
  with a new version instead.

## Maintainer checklist

1. Confirm the release commit is on `v2` and that its merge request satisfied
   the required CI, CODEOWNERS, and review policy in
   [CONTRIBUTING.md](../../CONTRIBUTING.md).
2. Update `pyproject.toml`, `CHANGELOG.md`, and documentation that records
   user-visible compatibility or support status together. Start a new
   `Unreleased` section in the changelog after documenting the release.
3. When contracts, configurations, checkpoints, recipes, metrics, or
   compatibility guarantees change, update the compatibility matrix and add or
   update migration notes under `docs/migration/`.
4. Run `make check` and record any intentionally skipped optional validation.
   Do not cut a release when required CPU CI validation fails.
5. Create and push an annotated `v<version>` tag pointing to the approved
   `v2` commit.
6. Verify that the tag, package version, and changelog version agree.

## Artifacts and historical releases

The default v2 workflows use synthetic data and do not require model artifacts.
Follow the repository artifact and data-governance policies before adding or
referencing any artifacts. The historical v1 release remains on the `v1`
branch and `v1.0.0` tag; it is not a compatibility baseline for v2.

Any package or artifact publication requires a separately approved publishing
process with its destination, access controls, provenance, and rollback
procedure documented first.
