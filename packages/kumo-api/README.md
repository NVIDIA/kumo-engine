# kumo-api

The `kumo-api` library defines a RESTful API resource schema and specification
for HTTP-compatible clients to interact with Kumo AI's cloud services. Resource
schemas are defined as Pydantic dataclasses.

While it is possible to interface with Kumo directly using these datamodels,
most users will want a higher-level Python library built on them, such as
`kumo-engine` (`pip install kumo-engine`).

## Development

This package is maintained in the Kumo monorepo at `packages/kumo-api`.
From the repository root:

```bash
uv pip install -e 'packages/kumo-api[test,release-check]'
uv run --no-project pytest packages/kumo-api/test
```

Package-specific automation lives in `.github/workflows/kumo-api-*` at the
repository root. See `MIGRATION.md` for history and release-ref details.
