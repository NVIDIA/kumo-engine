# ML development and review

Read the [architecture](../../ARCHITECTURE.md) and trace changes across service
adapters, `packages/kumo-core`, PQL, and DiskGraph.

## Review priorities

- Preserve time boundaries and prevent leakage. Check empty-input behavior
  and numerical results when changing aggregations, sampling, or training.
- Check consumers of changed output schemas, model artifacts, module exports,
  defaults, and feature flags. Consider persisted data and cached statistics.
- Reuse existing helpers and keep validation with its owning component.
- Add focused regression coverage for behavior changes; explain removed tests
  and restrictions. Keep unrelated changes separate.

Use [ML review patterns](ml-review-patterns.md) for concrete examples,
[query orchestration guidance](../../kumo/pquery/AGENTS.md) for service rules,
and [PQL instructions](../../packages/kumo-pql/AGENTS.md) for parser changes.
Follow the [DiskGraph source boundary](generated-artifacts.md#diskgraph-source-mirror)
for native-engine changes.
