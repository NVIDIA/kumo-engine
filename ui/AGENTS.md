# UI agent guidance

Read [UI development](../docs/development/ui.md) before changing `ui/`.
For shared components also read [catalog guidance](../packages/kumo-ui-catalog/CLAUDE.md).
Both are in this repository; coordinate component and caller changes together.

- Trace API changes through shared schemas and [generated clients](../docs/development/generated-artifacts.md).
- `src/gen/`, `src/utils/PQLParsing/grammar/`, and `pqDocs.json` are generated.
  Editor help is updated by reviewed synchronization PRs from `kumo-ai/docs`.
  Other files under `src/utils/PQLParsing/` include authored code.
- Use existing query keys, invalidation helpers, route loaders, mocks, and tests.
- From the repository root use `npm --prefix ui run type-check`,
  `npm --prefix ui run lint-check`, and focused `npm --prefix ui run test -- --run` checks.
- Browser tools are optional. Use available local browser/test tooling and
  report missing visual evidence; do not require a particular MCP service.
