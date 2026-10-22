# kumo-api

Shared Pydantic schemas for Kumo's service and clients.

The package is bundled into the `kumo-engine` release wheel.

## Development

From the repository root:

```bash
uv pip install -e 'packages/kumo-api[test]'
uv run --no-project pytest packages/kumo-api/test
```
