---
name: preflight
description: Use before a debugging workflow, when asked to check the environment, or when the local CLI tools needed for the requested task are not working.
---

# Preflight Checks

## When to Use

Use this skill when the user:
- Asks to "check environment", "preflight", or "validate setup"
- Before starting a debugging workflow
- When local CLI tools aren't working

---

Run only checks needed for the requested task. Do not install system packages,
change global configuration, or contact private services automatically.

## Step 1: GitHub CLI (skippable)

**Note**: User can skip this step if they don't need GitHub issue integration or will provide issue details manually.

```bash
echo "=== GitHub CLI ==="
if command -v gh &> /dev/null; then
    echo "✓ Installed: $(gh --version | head -1)"
    if gh auth status 2>&1 | grep -q "Logged in"; then
        echo "✓ Authenticated"
    else
        echo "✗ Not authenticated - run: gh auth login"
    fi
else
    echo "✗ Not installed - https://cli.github.com/"
fi
```

---

## Step 2: Optional Codex Consultation

Check this only when the user wants a second opinion. Identify the available
Codex session or configured integration; do not start a model call as a generic
preflight probe. See [the consultation reference](references/codex-mcp.md) for
the handoff prompt and optional Codex MCP configuration. Missing integration
does not block debugging or an already requested fix.

---

## Step 3: Kubernetes (when collecting live logs)

Check local prerequisites only after the user selects Kubernetes logs for this
task. Record the supplied context, namespace, and workload; ask only for missing
target details. These checks do not contact the cluster:

```bash
command -v kubectl
kubectl version --client
kubectl config get-contexts "{context}"
```

Use [check-logs](../check-logs/SKILL.md) for authorized log collection. A supplied
log file does not require `kubectl` or a Kubernetes login.

---

## Step 4: Docker (when needed)

```bash
echo "=== Docker ==="
if command -v docker &> /dev/null; then
    echo "✓ Installed: $(docker --version)"
    if docker info &> /dev/null 2>&1; then
        echo "✓ Daemon running"
    else
        echo "✗ Daemon not accessible"
    fi
else
    echo "✗ Not installed"
fi
```

---

## Summary Format

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
           PRE-FLIGHT CHECK SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Required (skippable if not using GitHub issues):
  gh CLI:              {✓/✗}
  gh authenticated:    {✓/✗}

Task-dependent:
  Docker:              {✓/✗/not needed}
  kubectl / context:   {✓/✗/not needed}

Optional:
  Codex consultation:  {available/unavailable/not requested}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

If any required checks fail, provide specific remediation steps.
