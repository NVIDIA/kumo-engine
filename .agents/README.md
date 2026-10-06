# Cross-Agent Configuration

This directory follows the [agentskills.io](https://agentskills.io) specification for skills, with a shared specialist source and Claude Code adapter.

## Structure

```text
.agents/
├── skills/                      # agentskills.io format
│   ├── build-ui/
│   │   ├── SKILL.md
│   │   ├── references/
│   │   │   ├── completion-criteria.md
│   │   │   ├── decision-tables.md
│   │   │   ├── handoff-examples.md
│   │   │   ├── headless-mode.md
│   │   │   ├── pr-workflow.md
│   │   │   └── workflow-test-scenarios.md
│   │   └── scripts/
│   │       ├── summarize-issue.sh
│   │       └── validate-skill-structure.sh
│   ├── code-ui/
│   │   ├── SKILL.md
│   │   ├── references/
│   │   │   └── catalog-components.md
│   │   └── scripts/
│   │       └── check-docs-freshness.sh
│   ├── review-ui/
│   │   ├── SKILL.md
│   │   └── scripts/
│   │       ├── diff-scan.sh
│   │       └── tests/
│   │           └── run-tests.sh
│   ├── debug-issue/
│   │   └── SKILL.md
│   ├── preflight/
│   │   ├── SKILL.md
│   │   └── references/
│   │       └── codex-mcp.md
│   ├── check-logs/
│   │   └── SKILL.md
│   ├── investigate-training-stage/
│   │   └── SKILL.md
│   └── review-ml/
│       └── SKILL.md
├── specialists/                 # Claude Code-specific
│   └── kumo-gitops-engineer.md
├── hooks/                       # Auto-triggered scripts
├── scripts/
│   └── sync-mcp.py              # Optional manual Codex MCP sync
└── mcp-servers.json             # Optional Codex MCP source
```

Shared engineering guidance is in [docs](../docs/README.md). Catalog workflows
remain in [their package](../packages/kumo-ui-catalog/.agents/README.md).

## Skills (agentskills.io spec)

| Skill | Purpose |
|-------|---------|
| `build-ui` | Orchestrate UI ticket work from GitHub issue through implementation, validation, and PR creation |
| `code-ui` | Scaffold and implement UI features following Kumo conventions (ui/src/) |
| `review-ui` | Review code against UI standards (ui/src/) |
| `debug-issue` | Investigate GitHub issues with source, logs, analysis, and optional consultation |
| `preflight` | Validate local prerequisites for the requested workflow |
| `check-logs` | Read supplied logs or collect logs from a selected Kubernetes workload |
| `investigate-training-stage` | Investigate stuck or slow Temporal training and prediction stages |
| `review-ml` | Review service ML adapters and the shared runtime in packages/kumo-core against ML standards |

## Additional catalog skills

The public [kumo-skills-catalog](https://github.com/kumo-ai/kumo-skills-catalog)
provides additional optional skills. In Claude, run
[`/init-skills`](../.claude/commands/init-skills.md) to clone or update it in
`.agents/skills-catalog/` and list available skills. Other agents can follow
the same command file. The catalog's `sync-skills-catalog.py` supports
`--list`, `--add <name> ...`, `--remove <name> ...`, and `--reset`.
Repository-local skills remain available without initializing the catalog.

## Specialist and hooks

The [GitOps specialist](specialists/kumo-gitops-engineer.md) covers manifest
rendering, environment branches, and deployment synchronization. Claude loads
its adapter in `.claude/agents/`; other agents can read the source when useful.
Its operational tools and access depend on the selected deployment.

See [hooks](hooks/README.md) for the local Python lint and documentation hooks
and their Claude adapters. For optional Codex MCP consultation, the
[Codex-only source](mcp-servers.json) and [manual sync script](scripts/sync-mcp.py)
remain available; running the script writes Claude permissions and a local
`.codex/config.toml`. Neither is needed for instruction or skill discovery.

## Compatibility

- **Claude Code**: Skill adapters in `.claude/commands/` reference the authored skills; `/init-skills` initializes the optional catalog.
- **Codex**: Discovers repository skills under `.agents/skills/`; no private MCP configuration is required.

See [agent setup](../docs/development/agents.md) for instruction compatibility
and optional tooling prerequisites. Migrated packages retain their
existing instructions and skill locations.
