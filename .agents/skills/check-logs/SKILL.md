---
name: check-logs
description: Use when the user asks to check, view, or search Kumo application logs from a supplied file or a selected Kubernetes deployment.
---

# Check Logs

## When to Use

Use this skill when the user:
- Asks to "check logs", "view logs", or "get logs"
- Needs to investigate errors for a specific tenant
- Wants to search for patterns in application logs

## Arguments

- `logs_file`: Supplied local log file, when available
- `context`, `namespace`: Explicit Kubernetes target when collecting logs
- `selector` or `pod`: Workload label selector or pod name
- `pattern` (optional): Case-insensitive regular expression to filter logs
- `since` (optional, default: "30m"): Time range (e.g., 5m, 30m, 1h)

---

## Step 1: Select the Log Source

Use the supplied log file, or the deployment the user has authorized for this
investigation. For Kubernetes collection, require `kubectl`, configured cluster
access, and the context, namespace, and workload. Reuse target details and
permission already provided for this task; ask only for missing information.
Do not derive a namespace from a tenant name or use an implicit current context.

The application chart uses labels such as `app=mlworkflow`, `app=databackend`,
and `app=kumo-rest`. Select the workload relevant to the investigation; the
namespace is chosen at installation. See [the Helm guide](../../../helm/README.md).

---

## Step 2: Read or Collect Logs

For a supplied file, read it directly. Apply a filter when requested:

```bash
rg -i -- "{pattern}" "{logs_file}"
```

For a selected Kubernetes workload:

```bash
kubectl --context="{context}" logs -n "{namespace}" \
  -l "{selector}" --all-containers=true \
  --since="{since}" --tail=500 --timestamps=true --prefix=true
```

For a specific pod, replace `-l "{selector}"` with `"pod/{pod}"`.
Filter the collected output only when a pattern was supplied. Keep retrieval
errors visible: an access error, an empty workload selection, and no matching
log entries are different outcomes. These commands collect bounded recent logs;
report that limit when drawing conclusions.

---

## Step 3: Analyze Results

Look for:
- **Error patterns**: Stack traces, exceptions
- **Timing**: When errors started
- **Frequency**: One-time vs recurring
- **Correlation**: Related events at same timestamp

Common patterns:
- `Exception`, `Error`, `Failed`
- `timeout`, `deadline exceeded`
- `connection refused`, `unavailable`
- `permission denied`, `unauthorized`
- `OOM`, `out of memory`

---

## Step 4: Report

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
              LOG ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source: {logs_file or context / namespace / workload}
Time range: Last {since}
Pattern: {pattern}

Findings:
{summary}

Key Log Entries:
{excerpts with timestamps}

Recommendations:
{next steps}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
