---
name: debug-issue
description: Use when the user asks to debug, investigate, or fix a GitHub issue or supplies a repository bug report.
---

# Debug Issue Workflow

## When to Use

Use this skill when the user:
- Asks to "debug", "investigate", or "fix" a GitHub issue
- Provides a GitHub issue number or URL
- Needs help diagnosing a repository bug

## Arguments

- `issue`: GitHub issue number, URL, or supplied report

---

## Phase 1: Pre-flight Checks

### 1.1 Verify gh CLI

```bash
which gh || echo "ERROR: gh CLI not installed. Install from https://cli.github.com/"
gh auth status
```

Skip GitHub checks when the user supplies the issue text. Remote issue access
requires a configured GitHub login; do not treat it as a prerequisite for local diagnosis.

### 1.2 Fetch and Validate Issue (when using GitHub)

```bash
gh issue view {issue} --json title,body,labels,assignees
```

Record the reported behavior, expected behavior, and reproduction steps. Record
any deployment target supplied for this investigation. Do not infer a namespace,
cluster context, or cloud account from a tenant name.

### 1.3 Optional Operational Prerequisites

Use [preflight](../preflight/SKILL.md) only for tools needed by this task.
Supplied logs need no cluster access. Live logs require the selected deployment
and authorization for this investigation; reuse those already supplied.
Codex consultation is checked only when selected, and does not block diagnosis.

### 1.4 Pre-flight Summary

```text
Issue:           {issue or supplied report}
Reported behavior: {symptom}
Evidence source: {local logs / selected context, namespace, workload}
Missing context: {details needed for the next investigation step, if any}
```

---

## Phase 2: Context Gathering

### 2.1 Check Debug Artifacts

```bash
ls -la .claude_debug/{issue}/ 2>/dev/null || echo "No debug artifacts found"
```

Read any screenshots (*.png) for visual context.

### 2.2 Read or Query Logs

**Option A**: Read local reproduction output or a log file supplied for this task.

**Option B**: Use [check-logs](../check-logs/SKILL.md) for the deployment the user
has selected and authorized for this investigation:

```text
/check-logs context={context} namespace={namespace} selector={selector} since=30m pattern="error|exception|failed"
```

If the target or access is missing, continue with available evidence and report
what is needed. For a stuck training or prediction stage, use
[investigate-training-stage](../investigate-training-stage/SKILL.md).
Keep customer data and credentials out of public reports.

### 2.3 Identify Affected Component

| Error Pattern | Component | Directory |
|---------------|-----------|-----------|
| REST/API errors | REST Service | `rest/` |
| Workflow/Activity failures | Temporal and pool orchestration | `temporalio_commons/`, `rest/backends/`, `pool/` |
| Data processing | Data Backend | `kumo/data/` |
| Training/model errors | ML Training | `packages/kumo-core/src/kumo_core/`, `kumo/train/`, `kumo/task/` |
| Connector/ingestion | Connectors | `kumo/connector/` |
| Graph/feature errors | Graph Engine | `packages/kumo-core/src/kumo_core/diskgraph/`, `kumo/diskgraph/` |

### 2.4 Search Codebase

Search for error messages and find related code with file:line references.

---

## Phase 3: Analysis

### 3.1 Document Findings

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                 ANALYSIS REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Issue Summary
{What the user reported}

## Root Cause
{Technical explanation with file:line references}

## Evidence
- Log excerpt: {relevant lines}
- Code location: {file:line}

## Suggested Fix
{Proposed solution}

## Risk Assessment
- Severity: {Low/Medium/High/Critical}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 3.2 Checkpoint: Next Steps

Use the user's existing instruction to select the next step:

1. **Consult Codex** — Get an optional second opinion when requested.
2. **Implement fix** — Proceed with code changes when the user has requested a fix.
3. **Done** — Report findings for an investigation-only request.

When the user has not chosen a next action, present the findings and these
options. Do not ask again for permission to implement an already requested fix.
Run focused checks from `docs/development/testing.md`. Post findings to GitHub
only when requested.

---

## Phase 3b: Codex Consultation (if selected)

Use the [consultation handoff prompt](../preflight/references/codex-mcp.md)
with the issue, evidence, analysis, and proposed fix. If the optional Codex MCP
integration is configured, call `mcp__codex__codex` with that completed prompt:

```text
Tool: mcp__codex__codex
prompt: {completed consultation handoff prompt}
```

Invoke it only after the user selects consultation; do not make a model call as
a preflight probe. If the tool is unavailable, use another available Codex
session or provide the prompt for manual consultation and report that limitation.
The MCP integration is not a prerequisite for diagnosis.

Present the response with any source-backed corrections to the analysis. Return
to the checkpoint, continuing an already requested fix or reporting findings
according to the user's chosen scope.

---

## Workflow Summary

```text
Pre-flight → Context Gathering → Analysis → [Checkpoint]
                                                │
                                    ├── Codex Consultation → Checkpoint
                                    ├── Implement Fix
                                    └── Done
```
