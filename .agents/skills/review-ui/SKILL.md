---
name: review-ui
description: Review PR code changes against UI conventions. Use when reviewing PRs that touch frontend code (ui/src/). Applies lessons learned from past UI reviews and enforces coding guidelines.
---

# UI Code Review

You are a terse UI code reviewer. Be brief. No filler.

## Severity Levels

- **[HIGH]** — Always flag. Bugs, security issues, or hard conventions with zero exceptions.
- **[MEDIUM]** — Flag with context. Strong conventions that may have rare justified exceptions.

Only comment on violations you actually see. Do not repeat rules that aren't violated.

## Invocation

```
/review-ui <pr_number>              # Posts review to GitHub
/review-ui <pr_number> --dry-run    # Prints findings to stdout only (no GitHub post)
/review-ui <pr_number> --force      # Posts even if a previous review exists on this PR
```

When `--dry-run` is passed, execute Steps 0–3 normally but skip Step 4 entirely. Instead, print findings to stdout in the format:

```
[SEVERITY] **Rule Name** (file:line) explanation
```

## Step 0: Setup

### Resolve repository

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Use `$REPO` wherever the GitHub API needs `repos/{owner}/{repo}`.

### Check for draft PRs

```bash
IS_DRAFT=$(gh pr view <number> --json isDraft -q .isDraft)
```

If `$IS_DRAFT` is `true`:
- Skip [MEDIUM] rules entirely — only flag [HIGH] violations.
- Prefix the review body with: **"Draft PR — only [HIGH] violations flagged."**
- This avoids noisy feedback on work-in-progress code.

### Check for duplicate reviews

Before posting, check if an automated review already exists on this PR:

```bash
EXISTING=$(gh api "repos/${REPO}/pulls/<number>/reviews" \
  --jq '[.[] | select(.body | startswith("**UI Review**"))] | length')
```

If `$EXISTING > 0` and `--force` was NOT passed, reply with:

> UI Review already posted on this PR. Run with `--force` to post again.

Stop. If `--force` was passed, proceed normally (a new review will be posted).

## Step 1: Get the diff and PR context

### Fetch PR metadata

```bash
gh pr view <number> --json title,body,isDraft
```

Surface the PR title and description to inform the LLM review. A PR titled "Temporary debug logging for incident #1234" may justify `console.log` — the LLM should note the context but still flag [HIGH] violations with a softer tone (e.g., "If this is temporary, add a TODO comment with the issue link").

### Fetch the diff

Run `gh pr diff <number>` to get the code changes. Filter to only these paths:
- `ui/src/components/`, `ui/src/pages/`
- `ui/src/hooks/`, `ui/src/stores/`
- `ui/src/api/`, `ui/src/css/`
- `ui/src/services/`, `ui/src/utils/` (exclude emitted files; see below)
- `ui/src/types/`
- `ui/src/routes/`, `ui/src/main.css`
- `ui/src/tests/`

Inspect authored parsing/autocomplete changes in `ui/src/utils/PQLParsing/`
manually; the scanner skips that directory.

Exclude generated code:
- Generated files listed in `docs/development/generated-artifacts.md` — review their source; do not propose manual edits to emitted files.

If NO files match those paths, reply with exactly:

> No UI files changed. `/review-ui` not applicable.

Nothing else. Stop.

**Trivial PR shortcut:** If the diff touches <=3 lines total and is clearly a typo fix, copy change, or comment edit, only check the Scope and Generated Code rules. Skip the full rule scan.

**Large PR handling:**
- **>20 UI files changed:** Increase comment budget from 10 to 15. Focus LLM review on the files with the most additions first.
- **>40 UI files changed:** Post a note suggesting the author split the PR for more effective review, then review the 20 most-changed files only.

## Step 2: Pre-scan (automated)

Run the helper script to surface zero-judgment violations with correct source file:line mapping:

```bash
bash .agents/skills/review-ui/scripts/diff-scan.sh <number>
```

The script outputs TSV to stdout: `rule_id	file_path	line_number	matched_text`
A summary is printed to stderr: `# diff-scan: N findings in M files`

Check each hit against its surrounding code before including it as a review comment. Pre-scan hits do **not** count against the comment budget since they're machine-verified.

Most pre-scan rules are [HIGH]. Two are [MEDIUM] migration rules: `design-token` and `arbitrary-tailwind`. On **draft PRs**, discard pre-scan hits for these two (per the Step 0 draft policy of flagging [HIGH] only).

The script automatically:
- Excludes known exception files (e.g., `utils/storage.ts` for the direct-storage rule)
- Respects inline `review-ui-ignore:` suppression comments
- Skips commented-out code (e.g., `// console.log(...)`)

If the script is unavailable, fall back to manual grep. See `scripts/diff-scan.sh` header comments for the pattern list.

To validate the script after changes: `bash .agents/skills/review-ui/scripts/tests/run-tests.sh`

## Step 3: Review (LLM)

For each matching file, check the diff against the linked review conventions. **Skip rules already covered by pre-scan hits** — don't double-flag the same line.

**Respect inline suppression:** If a line has a `// review-ui-ignore: <rule-id>` comment, skip that specific rule for that line. The developer is asserting a justified exception.

### Load reference patterns based on what the diff touches

- **Always load:** `docs/development/ui-review-kumo.md`
- **If `tests/` or `__tests__/` files changed:** also load `docs/development/ui-review-testing.md`
- **If component, hook, page, or store files changed:** also load `docs/development/ui-review-react.md`

### Comment budget

| PR size (UI files) | Budget (beyond pre-scan) |
|---------------------|--------------------------|
| 1–20 files          | 10 comments              |
| 21–40 files         | 15 comments              |
| 40+ files           | 15 comments (top 20 files only) |

**Draft PRs:** Only [HIGH] violations count. Budget is unchanged.

**Prioritize:** [HIGH] > [MEDIUM], then bugs > style, then file order in the diff. If tied within the same severity, prefer issues that affect more code paths.

### Cross-file structural checks

After reviewing individual files, check for these structural issues across the whole PR:

- **New store without test reset:** If a new Zustand store is added, verify `tests/setup.ts` resets it in `beforeEach`.
- **New page route without test:** If a new route is added to `Routes.tsx`, check for a corresponding test file.
- **New hook without test coverage:** If a new custom hook is added in `hooks/`, check for a corresponding test.
- **New component without error boundary:** If a new component is exported from `components/`, verify it's wrapped with `WithErrorBoundary`.
- **New barrel export missing:** If a file is added to a directory that has an `index.ts`, verify the new export is re-exported.

These don't have pre-scan patterns — the LLM must check them manually. They count against the comment budget.

## Step 4: Post comments

Post only for an explicitly requested posted review (including the documented
posting invocation above). For a read-only review request or `--dry-run`,
return findings in the conversation and skip this step.

Combine pre-scan hits and LLM findings into a single GitHub PR review. Include summary stats in the body:

```bash
gh api "repos/${REPO}/pulls/<number>/reviews" \
  --method POST \
  --input - <<'EOF'
{
  "event": "COMMENT",
  "body": "**UI Review** — N [HIGH], M [MEDIUM] across F files (P auto-detected, L manual)",
  "comments": [
    {
      "path": "ui/src/components/Example.tsx",
      "line": 42,
      "side": "RIGHT",
      "body": "[HIGH] **Console Statement**: Remove `console.log`. Use `logStackTrace()` from `utils/errorHandler`.\n\n```suggestion\nimport { logStackTrace } from \"utils/errorHandler\";\nlogStackTrace(\"debug context\", data);\n```"
    }
  ]
}
EOF
```

Replace N, M, F, P, L with actual counts before posting.

### Comment format

Each comment body starts with `[HIGH]` or `[MEDIUM]`, then the **bold rule name**, then a one-sentence explanation. Keep each comment to 1–3 sentences max.

Format: `[SEVERITY] **Rule Name**: explanation`

### Suggestion blocks for auto-fixable rules

For rules marked as auto-fixable in the linked review conventions, include a GitHub suggestion block when the fix is mechanical and unambiguous. This lets the PR author click "Apply suggestion" to fix inline:

````markdown
```suggestion
corrected code here
```
````

**Auto-fixable rules and their typical suggestions:**

| Rule ID | Typical suggestion |
|---------|-------------------|
| `console-log` | Replace with `logStackTrace("context", error)` |
| `deep-import` | Rewrite as absolute import from `baseUrl: "src"` |
| `direct-storage` | Replace with `storeToLocalStorage(storageKeys.X, val)` or `getFromLocalStorage(storageKeys.X)` |
| `process-env` | Replace `process.env.REACT_APP_X` with `window._env_.X` |
| `hardcoded-route` | Replace string with `navRoutes.RouteName` |
| `raw-queryclient` | Replace `new QueryClient()` with `createQueryClient()` |
| `react-keys` | Replace `key={index}` with `key={item.id}` |
| `bundle-size` | Replace full import with specific import (`lodash/sortBy`) |
| `design-token` | Replace neutral palette class with the mapped Kumo token (see rule) |

Only include a suggestion when you can produce the correct replacement. If you're unsure of the exact fix (e.g., which `navRoutes` member to use), omit the suggestion block and just explain the rule.

### Line numbers and `side` field

- The `line` field is a line number in the **new file** (RIGHT side of the diff).
- Use `"side": "RIGHT"` for added or modified lines (the common case).
- Use `"side": "LEFT"` only when commenting on a deleted line (e.g., "don't remove this error boundary"). This is rare.
- To find the correct line: locate the `@@ -old,count +new,count @@` hunk header, then count forward from `+new` for added/unchanged lines.

### Fallback

If the review API call fails (e.g., line number out of range, stale diff), fall back to a single summary comment. Compute the timestamp before the heredoc to avoid expansion issues:

```bash
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
gh pr comment <number> --body "**UI Review** (${TIMESTAMP})

<list up to 10 findings, one per line, format: \`file:line\` — [SEVERITY] **Rule**: explanation>

---
*Automated review against UI conventions. Inline comments failed — posted as summary.*"
```

Never post more than 15 lines total in fallback mode.

### Diagnostic logging

Do not persist review statistics by default. If the user explicitly asks to save them, append the UTC timestamp, PR number, pre-scan finding count, LLM finding count, and triggered rule IDs as a TSV row to `${TMPDIR:-/tmp}/review-ui/review-log.tsv`, then report the path.

---

## Review conventions

Read [UI review conventions](../../../docs/development/ui-review.md) for
the rule definitions, severities, and inline suppression syntax.

## Reference

Pattern examples (loaded conditionally — see Step 3):
- `docs/development/ui-review-kumo.md` — Kumo conventions, runtime config, navigation, state
- `docs/development/ui-review-testing.md` — Testing infrastructure
- `docs/development/ui-review-react.md` — General React patterns

Pre-scan script: `.agents/skills/review-ui/scripts/diff-scan.sh`
Script tests: `.agents/skills/review-ui/scripts/tests/run-tests.sh`
Coding guidelines: `docs/development/ui.md`
