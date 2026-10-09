# Coding agents

You can start at the repository root or in a component directory. Read the
applicable [AGENTS.md](../../AGENTS.md) files and follow their links when a
change spans components. Component instructions add local constraints; they
do not require separate sessions or independent checkouts.

Shared engineering guidance lives in ordinary docs. Root `CLAUDE.md` links to
`AGENTS.md`; component adapters import their adjacent instructions. Preserve
symlinks when checking out the repository.

Find repository skills and optional catalog setup in
[.agents/README.md](../../.agents/README.md). The
[package index](../README.md#package-documentation) links to component guidance.

## Verify discovery

Check the supported CLI versions from the repository root. Package owners
verify their existing component discovery separately. In Claude, inspect `/context` or `/memory` and the available skills;
in Codex, inspect the loaded instructions and skill list. Confirm the root
instructions, component instructions, and skills for that scope resolve to
their authored files. User/global configuration can add other rules; do not
commit employee settings to compensate for it.

Run the [static consistency checks](../maintenance/README.md) after changing
instruction imports, skill adapters, or document locations.

## Optional tooling

Repository workflows can use public CLIs, configured integrations, and a
user-selected Kubernetes deployment. These are task-specific prerequisites:
[check-logs](../../.agents/skills/check-logs/SKILL.md) needs cluster access only
for live collection; supplied logs and source investigation work without it.
[Codex consultation](../../.agents/skills/preflight/references/codex-mcp.md) is
optional and uses an available session or configured integration. The retained
MCP source and manual sync script are one opt-in configuration. These tools
and credentials are not prerequisites for repository instruction discovery.

The [Python lint and documentation hooks](../../.agents/hooks/README.md) run
locally through the committed Claude adapters. The session-end documentation
hook can sync an existing root “Available Skills” table and report undocumented
MCP permissions. The current root guide has no such table. The read-only
repository checker separately validates links, instructions, and skill adapters.

Use the target and authorization supplied for the task, and report missing
prerequisites. Initialize or install catalog skills only when requested. Do not
probe unspecified clusters, install global packages, or modify shell configuration.
