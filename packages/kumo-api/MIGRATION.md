# Kumo API monorepo migration

The public `kumoapi` package now lives in `packages/kumo-api`. The standalone
repository's complete `master` ancestry was merged into Kumo without squashing,
so the original commits remain reachable from this repository.

The standalone repository is authoritative through commit
`9d63fc34a6829abe54e80e8f49a6c532ad5d1132`. Do not land new API changes there
after this migration. Archive it only after the migrated checks and release
permissions have been validated.

## Development and CI

- Install with `uv pip install -e packages/kumo-api` from the repository root.
- `kumo-api` is a uv workspace member, not an external Git pin.
- Package automation is in `.github/workflows/kumo-api-*`. Pull-request and
  push workflows filter on `packages/kumo-api/**` and their workflow files.
- Release branches are `kumo-api-vX.Y`; release tags are
  `kumo-api-vX.Y.Z` or `kumo-api-vX.Y.ZbN`. The namespace prevents collisions
  with Kumo platform release refs.

## Publication handoff

The release workflow defaults to a dry run. Before enabling a live release,
configure the release-bot app credentials, PyPI publication credentials, and
the `gha-kumo-api-schema-publish` AWS role for `kumo-ai/kumo`. Update PyPI
trusted-publisher and AWS OIDC trust records to name this repository and the
root `kumo-api-*` workflow files before retiring their standalone equivalents.
