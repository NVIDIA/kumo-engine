# Optional Codex consultation

Use a second opinion when the user selects consultation in `debug-issue` or
requests it as part of the task. Codex access is optional and independent of
repository skill discovery.

## Available integration

Use the user's available Codex session or an explicitly configured integration.
If neither is available, provide the handoff prompt below; continue diagnosis or
an already requested fix without requiring new tooling. Installation, login,
and model availability follow the user's chosen Codex setup.

The repository retains an optional [Codex MCP source](../../../mcp-servers.json)
and [manual sync script](../../../scripts/sync-mcp.py) for the existing
`npx -y codex-mcp-server` integration. This external wrapper, its installation,
and authentication are not verified by this PR. Running the script changes
`.claude/settings.json` permissions and generates `.codex/config.toml`; it is
not required for consultation through an existing session or for skill discovery.

The former `codex mcp-server` command and standalone server binary have been
removed. See [OpenAI's compatibility notice](https://learn.chatgpt.com/docs/mcp-server).
The app server uses its own protocol and cannot replace an MCP server command
without an integration change.

## Handoff prompt

```text
Debugging Kumo issue {issue or report}:
- Reported behavior: {symptom}
- Relevant environment: {supplied deployment details, if needed}

My analysis: {root_cause}
Evidence: {relevant excerpts and source references}
Proposed fix: {solution}

Questions:
1. Does this analysis seem correct?
2. Other potential causes?
3. Risks with the proposed fix?

Review the analysis; do not modify code or contact services.
```

Use only evidence appropriate for the selected consultation. Compare the response
with the source and original evidence, then return to the workflow checkpoint.
