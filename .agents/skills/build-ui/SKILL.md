---
name: build-ui
description: Orchestrate Kumo UI ticket work from GitHub issue analysis through implementation, validation, and optional PR creation.
---

# Build UI

Use this skill when the user wants to implement a UI GitHub issue end-to-end, continue work from an existing UI PR, or run a ticket-to-PR workflow for changes in `ui/src/`.

This skill is the orchestrator.

- `/code-ui` owns implementation conventions and code edits.
- `/review-ui` owns diff review standards.
- `build-ui` owns scope, sequencing, approvals, validation flow, and PR readiness.

Planning remains read-only until the user approves implementation.

## Ownership Boundaries

`build-ui` owns:

- issue or PR intake
- scope definition and acceptance criteria
- checkpoint and approval flow
- branch and phase sequencing
- validation sequencing and result reporting
- PR readiness and PR creation flow

`build-ui` does not own:

- code-shape rules for React, routing, stores, or tests
- detailed implementation checklists
- targeted test authoring mechanics
- diff review standards

## Arguments

- `target` (required): GitHub issue number/URL or PR number/URL
- `--dry-run`: Run planning only, print the plan, stop before edits
- `--skip-figma`: Skip Figma design extraction
- `--skip-screenshots`: Skip screenshot capture and upload
- `--skip-browser`: Skip browser-based visual verification
- `--no-review`: Skip self-review
- `--validate-only`: Complete implementation and verification but stop before push/PR creation
- `--base-branch` (default `master`)
- `--resume <stage>`
- `--mode light|standard|full` (default `standard` for interactive/local use; the GitHub headless workflow defaults to `light` unless the trigger passes `--mode standard` or `--mode full`)

Mode behavior:

| Phase              | light                        | standard                     | full                                         |
| ------------------ | ---------------------------- | ---------------------------- | -------------------------------------------- |
| 0 Pre-flight       | yes                          | yes                          | yes                                          |
| P PR Inspection    | yes                          | yes                          | yes                                          |
| 1 Ticket Analysis  | yes                          | yes                          | yes                                          |
| 2 Recon            | minimal                      | yes                          | yes                                          |
| 3 Design Reference | skip                         | if Figma link exists         | yes, attempt even without explicit link      |
| Checkpoint 1       | yes                          | yes                          | yes                                          |
| 0b Branch Setup    | yes                          | yes                          | yes                                          |
| 4 Implementation   | yes                          | yes                          | yes                                          |
| 5 Static Analysis  | type-check + lint-check only | all four commands            | all four commands                            |
| 6 Self-Review      | skip                         | yes                          | yes                                          |
| 7 Unit Testing     | targeted tests only          | targeted + related           | targeted + related + coverage                |
| 8 Visual Testing   | skip                         | if prerequisites available   | yes, attempt even with partial prerequisites |
| Checkpoint 2       | yes                          | yes                          | yes                                          |
| 9 PR Creation      | yes unless `--validate-only` | yes unless `--validate-only` | yes unless `--validate-only`                 |

`light` maps closely to the Minimum Path. Use `light` for trivial fixes; use `full` when the ticket involves design-heavy or cross-cutting changes.

## Required Outputs

- planning summary or PR inspection summary
- Checkpoint 1
- `/code-ui` handoff contract
- Verification Status
- Checkpoint 2 when PR creation is in scope
- Handoff

## Path Resolution

Resolve all repo-relative paths from the monorepo root, not the current shell working directory.

Required lookups:

- `docs/development/ui.md`
- `.agents/skills/code-ui/SKILL.md`
- `.agents/skills/code-ui/references/catalog-components.md`
- `.agents/skills/code-ui/references/design-tokens.md`
- `docs/development/ui-review-kumo.md`

Conditional lookups:

- `docs/development/ui-review-react.md` if components, hooks, pages, or stores change
- `docs/development/ui-review-testing.md` if tests change
- `.github/PULL_REQUEST_TEMPLATE.md` if PR creation is requested

Supporting references:

- `references/decision-tables.md`
- `references/completion-criteria.md`
- `references/headless-mode.md`
- `references/handoff-examples.md`
- `references/output-contracts.md`
- `references/screenshot-operator-guide.md`
- `ui/src/screenshot/catalog.json`
- `references/pr-workflow.md`
- `references/run-state-template.json`
- `references/workflow-test-scenarios.md`

Screenshot quality bar:

- screenshot success means the changed UI was visually verified, not merely that a route loaded
- for known surfaces, prefer `ui/src/screenshot/catalog.json` recipes over freeform plans
- every user-visible requirement that should be proven in the screenshot should have a matching assertion
- catalog recipes should declare structured `visualCoverage` items with stable IDs, and setup/actions/assertions should use `covers` with those IDs so coverage intent is tied to concrete selectors
- reusable screenshot fixtures should be used for common deterministic states instead of repeating brittle live-data interactions
- a screenshot file is not sufficient: required coverage IDs must pass before the workflow can treat visual verification as captured
- in headless mode, PR creation is workflow-owned and must happen after required screenshot capture/coverage succeeds
- assertions must target the exact changed UI element, not a broad container such as a page shell, tab list, or generic heading
- if a stable exact selector does not exist, add a `data-testid` as part of the implementation rather than relying on brittle container text

Cross-skill reference rule:

- `build-ui` may reference its own files as `references/...`
- any file owned by another skill must use the full path, for example `.agents/skills/code-ui/references/design-tokens.md`

## Source Of Truth Priority

When sources conflict, use this priority order:

1. explicit user instruction
2. approved acceptance criteria or approved PR continuation scope
3. actual code behavior and repo constraints
4. issue or PR body/comments
5. Figma design reference
6. adjacent cleanup opportunities

Adjacent cleanup policy:

- if the extra change is required for correctness of the requested scope, allow it and record it as an `adjacent-fix`
- if the extra change is beneficial but not required for correctness, defer it unless the user explicitly approves it
- if the extra change expands scope across a separate feature area, reject it from the current implementation and defer it

## Mode Detection

Rules:

- If the input matches `.../issues/{n}`, `issue/{n}`, `issues/{n}`, `#n`, or a bare integer `{n}`, use `issue` mode.
- If the input matches `.../pull/{n}`, `.../pulls/{n}`, `pr/{n}`, `pull/{n}`, `pulls/{n}`, `PR {n}`, or `pr:{n}`, use `pr` mode.
- A bare integer is never inferred as `pr` mode. To inspect or continue a PR, the user must provide a PR URL or one of the explicit PR-prefixed forms above.
- If issue-mode intake later discovers an open PR linked to the issue, handle it through the existing PR check and ask for `inspect-only`, `continue-existing-pr`, or `supersede-with-new-work`.
- In `issue` mode, follow the standard ticket-to-implementation flow.
- In `pr` mode, inspect the PR first, infer the linked issue if present, and decide whether to inspect only, continue existing work, or supersede it.

Variable glossary:

- `{target}`: the original user input
- `{issue}`: resolved issue number when available
- `{pr}`: resolved PR number when available
- `{issue-or-pr}`: `{issue}` if present, otherwise `{pr}`
- `{headRefName}`: PR head branch in `pr` mode
- `{base-branch}`: target base branch
- `{slug}`: a short kebab-case summary derived from the issue title — strip prefixes like `[UI]` or `UI:`, lowercase, replace spaces and special chars with hyphens, truncate to ~40 chars at a word boundary (e.g., `"UI: Add cancel confirmation dialogs"` → `cancel-confirmation-dialogs`)
- `{work-branch}`: the branch this workflow should edit

## Workflow Summary

```text
Mode detection
    ├── issue mode
    └── pr mode

Phase 0:  Pre-flight
Phase P:  PR Inspection (pr mode only)
Phase 1:  Ticket Analysis
Phase 2:  Recon
Phase 3:  Design Reference (optional)
Checkpoint 1: Approve implementation
Phase 0b: Branch Setup
Phase 4:  Implementation via /code-ui
Phase 5:  Static Analysis
Phase 6:  Self-Review (optional)
Phase 7:  Unit Testing
Phase 8:  Visual Testing (optional)
Checkpoint 2: Approve PR
Phase 9:  PR Creation (optional)
```

For stop, skip, and fallback behavior, use `references/decision-tables.md`.

If the workflow is running from GitHub Actions or another non-interactive executor, also apply `references/headless-mode.md`. Headless-mode rules override human-checkpoint behavior where that file says they do.

## Phase-State Matrix

| Phase              | Issue Mode | PR Mode     | Mutates Repo | Requires Approval       |
| ------------------ | ---------- | ----------- | ------------ | ----------------------- |
| 0 Pre-flight       | yes        | yes         | no           | no                      |
| P PR Inspection    | no         | yes         | no           | no                      |
| 1 Ticket Analysis  | yes        | conditional | no           | no                      |
| 2 Recon            | yes        | yes         | no           | no                      |
| 3 Design Reference | yes        | yes         | no           | no                      |
| Checkpoint 1       | yes        | yes         | no           | yes                     |
| 0b Branch Setup    | yes        | yes         | yes          | covered by Checkpoint 1 |
| 4 Implementation   | yes        | yes         | yes          | covered by Checkpoint 1 |
| 5 Static Analysis  | yes        | yes         | conditional  | no                      |
| 6 Self-Review      | yes        | yes         | no           | no                      |
| 7 Unit Testing     | yes        | yes         | conditional  | no                      |
| 8 Visual Testing   | yes        | yes         | no           | no                      |
| Checkpoint 2       | yes        | yes         | no           | yes                     |
| 9 PR Creation      | yes        | yes         | yes          | yes                     |

## Phase Entry And Exit Criteria

Use these criteria to decide whether a phase can start, whether it is complete, and whether `--resume` is valid.

| Phase              | Entry Criteria                                     | Exit Criteria                                                                   |
| ------------------ | -------------------------------------------------- | ------------------------------------------------------------------------------- |
| 0 Pre-flight       | target resolved, repo readable                     | git and GitHub state recorded, environment availability recorded                |
| P PR Inspection    | `pr` mode and PR readable                          | recommendation emitted with evidence summary                                    |
| 1 Ticket Analysis  | issue readable when required                       | ACs, scope, design refs, and API dependencies extracted                         |
| 2 Recon            | relevant code readable                             | exemplars, impacted files, API status, catalog status, and test plan identified |
| 3 Design Reference | Figma link/tooling available or explicit skip      | token mapping or explicit skipped note recorded                                 |
| Checkpoint 1       | planning outputs complete                          | explicit approval, adjustment, or stop decision                                 |
| 0b Branch Setup    | Checkpoint 1 approved                              | intended work branch checked out and recorded                                   |
| 4 Implementation   | branch ready and `/code-ui` handoff complete       | accepted ACs implemented or blocker explicitly recorded                         |
| 5 Static Analysis  | implementation changes present                     | all required static commands classified and status emitted                      |
| 6 Self-Review      | diff available and review enabled                  | high-severity findings fixed or blocker recorded                                |
| 7 Unit Testing     | test targets identified                            | targeted or broader tests classified and status emitted                         |
| 8 Visual Testing   | route known and tooling/path chosen                | visual result recorded with any reduced-scope caveats                           |
| Checkpoint 2       | implementation and verification summaries complete | explicit PR decision or stop decision                                           |
| 9 PR Creation      | Checkpoint 2 approved                              | PR created or exact blocker reported                                            |

## Minimum Path

For a trivial bug fix or small scoped UI adjustment, prefer this reduced flow:

1. Phase 0: Pre-flight
2. Phase 1: Ticket Analysis
3. Phase 2: Minimal recon
4. Checkpoint 1
5. Phase 0b: Branch Setup
6. Phase 4: Implementation via `/code-ui`
7. Phase 5: Static Analysis
8. Phase 7: Targeted Unit Testing
9. Final Handoff

Use the full flow only when the task needs broader design, browser, or PR preparation work.

## Phase 0 — Pre-flight

Goals:

- confirm GitHub readiness if GitHub operations are needed
- inspect issue-linked open PRs when `{issue}` is known
- inspect worktree safety
- record optional environment availability:
  - dev server
  - backend
  - browser tooling
  - runtime config
  - catalog package

Required checks:

```bash
gh auth status
git status --porcelain
bash .agents/skills/build-ui/scripts/check-preflight.sh
```

Operational probes to record explicitly:

- dev server: whether the expected local UI server is already reachable
- backend: whether the target API/backend needed for the affected route is reachable
- browser tooling: whether interactive browser tooling is available
- runtime config: whether local runtime auth and required feature flags allow the route to load
- catalog package: whether the catalog dependency state is healthy enough for implementation

Do not leave these as implied observations. Record each as one of:

- `available`
- `unavailable`
- `unknown`
- `not-needed`

Use `bash .agents/skills/build-ui/scripts/check-preflight.sh` to gather these statuses in a consistent format before Checkpoint 1.

If a probe is intentionally out of scope for the run, set the matching environment variable before running pre-flight so it records `not-needed` instead of a misleading failure:

- `BUILD_UI_DEV_SERVER_NOT_NEEDED`
- `BUILD_UI_BACKEND_NOT_NEEDED`
- `BUILD_UI_BROWSER_NOT_NEEDED`
- `BUILD_UI_RUNTIME_CONFIG_NOT_NEEDED`
- `BUILD_UI_CATALOG_NOT_NEEDED`

If screenshots are not skipped and PR creation may publish visual evidence, check GitHub CLI write access early so screenshot publication problems are discovered before Phase 9.

### Existing PR check (issue mode)

When `{issue}` is known, check for open PRs that reference it:

```bash
gh pr list --search "#{issue}" --state open --json number,title,url
```

If a related PR exists, stop and require one of:

- `inspect-only`: do not create a branch or mutate code
- `continue-existing-pr`: checkout the PR branch and continue on it
- `supersede-with-new-work`: continue normal issue flow, recording that this intentionally overlaps

### Dirty worktree reporting

If `git status --porcelain` shows uncommitted changes, report:

- files clearly unrelated to this task
- files potentially overlapping the target work
- whether the workflow can proceed safely without touching them

Do not auto-stash or revert user changes. If overlap exists on target files, stop and ask before editing.

Do not mutate the repo in this phase.

## Phase P — PR Inspection

Run this phase only in `pr` mode.

Fetch PR metadata:

```bash
gh pr view {pr} --json number,title,body,url,headRefName,baseRefName,author,isDraft,files,commits
```

Inspect:

- title, body, branch, and base branch
- linked issue from `Fixes #...` or similar text if present
- changed files
- commit order and recent fix commits
- whether screenshots are attached, branch-hosted, or absent
- whether claimed checks align with actual evidence

Output exactly one:

- audit summary
- continuation plan
- supersede recommendation

Use this output shape:

```text
PR Inspection Summary
- target: {pr}
- linked issue: {issue-or-none}
- branch: {headRefName}
- draft: yes|no
- claimed in PR:
  - screenshots: yes|no
  - tests: ...
  - static checks: ...
- verified from evidence:
  - screenshots: verified|unverified|contradicted
  - tests: verified|unverified|contradicted
  - static checks: verified|unverified|contradicted
- changed files summary:
  - ...
- recommendation: inspect-only|continue-existing-pr|supersede-with-new-work
```

Continuation edges:

- `inspect-only`: stop after reporting gaps and risks.
- `continue-existing-pr`: checkout the PR branch and proceed into Phase 1 when a linked issue exists; if no linked issue exists, proceed into Phase 2 recon from the PR context.
- `supersede-with-new-work`: proceed through the normal issue flow and record that the new work intentionally supersedes or overlaps the existing PR.

## Phase 1 — Ticket Analysis

Run this phase in:

- `issue` mode always
- `pr` mode only if a linked `{issue}` was discovered

When `{issue}` exists, fetch metadata and comments with:

```bash
bash .agents/skills/build-ui/scripts/summarize-issue.sh {issue}
```

Never skip comments. They often carry critical scope changes, Figma links, and clarifications.

Extract:

- acceptance criteria
- Figma URLs
- API routes or endpoints
- affected page or component names
- related PRs or issues
- screenshots or image links
- feature flags or deployment-specific constraints

If requirements are too vague to synthesize concrete acceptance criteria, stop and ask for clarification.

Output a structured issue summary with:

- type
- scope
- size estimate
- AC list
- design references
- API dependencies

## Phase 2 — Recon

Read the code needed to answer:

- what existing feature is the best exemplar
- which files are likely impacted
- which tests and mocks already cover related behavior
- whether generated hooks exist in `src/gen/v1/` or `src/gen/v0/`
- whether routing, sidebar, query-key, or invalidation wiring applies
- whether feature flags or project authorization constraints apply

Check these integration points only when relevant:

- `src/routes/Routes.tsx`
- `src/routes/Loaders.ts`
- `src/utils/Constants/routes.ts`
- `src/hooks/sidebar/useSidebarData.ts`
- `src/utils/reactQueryKeys.ts`
- `src/utils/queryInvalidation.ts`
- `src/tests/mockAxios.ts`
- `src/tests/setup.ts`

During recon:

- classify API availability explicitly
- classify catalog gaps explicitly
- prefer the minimal-change path for small tickets

Do not silently approximate a missing catalog variant with ad hoc styling.

## Phase 3 — Design Reference

Skip if there is no Figma link, `--skip-figma` is set, or the selected mode omits design work.

If configured Figma tooling is available, or headless mode provides `figma-context.json` / `figma-summary.md` from workflow-owned Figma REST extraction:

- parse file and node IDs
- fetch node data
- inspect rendered node images when available
- map relevant styles to Kumo design tokens
- map frames and components to catalog primitives
- identify UI states to verify later

Figma is advisory, not a hard dependency. If no Figma tooling is configured, or if access fails, skip this phase and note that design verification was skipped.

## Dry-Run Exit Point

If `--dry-run` is set:

- print the plan summary
- stop before branch creation or dependency mutation

Use this output shape:

```text
Dry-Run Summary
- mode: issue|pr
- target: {issue-or-pr}
- linked issue: {issue-or-none}
- requested scope: ...
- adjacent cleanup discovered: ...
- deferred/out of scope: ...
- acceptance criteria:
  - ...
- file plan:
  - modify: ...
  - create: ...
- blockers:
  - ...
- verification planned:
  - static checks: yes|no
  - tests: yes|no
  - browser verification: yes|no
  - screenshot upload: yes|no
```

## Checkpoint 1 — Approve Implementation

Present:

- requested scope
- adjacent cleanup discovered
- deferred or out of scope
- catalog gap check
- acceptance criteria
- files to create
- files to modify
- exemplar references
- API hook plan
- UI states to cover
- blockers and assumptions
- size estimate

AC-to-file mapping is mandatory:

- `AC 1 -> planned file(s)`
- `AC 2 -> planned file(s)`
- `AC N -> planned test file(s)`

Rules:

- If an AC cannot be mapped to concrete files and tests, stop before implementation.
- Phase 4 may only implement requested scope by default.
- Anything in adjacent cleanup requires explicit approval.
- Adjacent cleanup that is required for correctness may proceed as an `adjacent-fix`, but it must still be recorded explicitly in Checkpoint 1 and the final handoff.
- If a required catalog variant is missing, stop unless the user explicitly approves the catalog-first path or a temporary local exception.
- If the file plan exceeds about 20 modified files or 3 feature areas, recommend splitting the work.

Do not mutate the repo before this checkpoint.

Use this output shape:

```text
Checkpoint 1
- mode: issue|pr
- target: {issue-or-pr}
- branch plan: create-new|reuse-existing|continue-pr-branch
- requested scope:
  - ...
- adjacent cleanup discovered:
  - ...
- deferred/out of scope:
  - ...
- catalog gap check:
  - none|details
- AC to file mapping:
  - AC 1 -> ...
  - AC 2 -> ...
- blockers:
  - ...
- blocker type: none|requirements|api|catalog|worktree-overlap|environmental|resume-mismatch|adjacent-fix|other
- scope change: none|adjacent-fix|approved-expansion|split-recommended
- decision needed: approve|adjust|stop
```

## Phase 0b — Branch Setup

Only after Checkpoint 1 approval:

```bash
git fetch origin
git branch --list "ui/{issue}-*"
git checkout -b ui/{issue}-{slug} origin/{base-branch}
```

Rules:

- Reuse a matching branch when `--resume` is set.
- Ask before reusing a matching branch when `--resume` is not set.
- In `pr` mode with continuation, prefer `{headRefName}`.
- Prepare dependencies only if actually needed.

Use `references/decision-tables.md` for dependency setup choices.

## Phase 4 — Implementation

Construct a `/code-ui` task using this handoff contract:

```text
/code-ui
Task: {short summary}
Accepted AC:
- ...
Impacted Files:
- ...
Exemplars:
- ...
API Hooks:
- ...
Design Notes:
- ...
UI States:
- ...
Limitations:
- ...
Flags And Constraints:
- ...
Validation Scope:
- static checks: ...
- targeted tests: ...
- related tests: ...
```

All fields are required. `Design Notes` carries Figma token mappings and spacing values from Phase 3, or `none` if Phase 3 was skipped. Use `.agents/skills/code-ui/references/design-tokens.md` class names, not raw Tailwind colors.

For complete examples, see `references/handoff-examples.md`.

After `/code-ui` completes, verify manually:

- each AC maps to real code
- no fabricated hooks, routes, or props
- no placeholder TODOs in critical paths
- the change does not silently exceed requested scope

Spot-check convention compliance — `/code-ui` should have followed `.agents/skills/code-ui/references/integration-checklist.md`, but verify:

- error boundaries on all exported components
- design tokens used instead of raw Tailwind colors
- Orval hooks used instead of direct axios
- absolute imports from `src/`, no `../../../`

`/code-ui` owns:

- implementation conventions and code-shape checks
- shared primitive compatibility and caller updates
- route, mock, and wiring checks
- UI state coverage
- copy consistency checks and incidental in-scope fixes
- targeted test design and initial failure diagnosis

### Commit strategy

- Use descriptive commit messages that reference the issue: `feat(ui): {short description} (#{issue})`
- If implementation spans multiple logical units (e.g., shared primitive + all caller updates + new page), prefer one commit per logical unit
- Phase 5 fixes and Phase 7 test additions should be separate commits from Phase 4 implementation
- Avoid generic messages like `"changes"`

### PR title strategy

- Use the PR title format: `UI: {issue title}`
- Strip redundant prefixes if the issue title already starts with `UI:` or `[UI]`
- Prefer the user-facing issue title over a conventional-commit style PR title

## Phase 5 — Static Analysis

Run:

```bash
npm run type-check
npm run lint-check
npm run format-check
npm run build
```

Rules:

- Never edit generated code under `src/gen/` or `src/utils/PQLParsing/grammar/`.
- Fix failing lint, type, or build issues directly and rerun until clean or clearly blocked.
- If `type-check` OOMs, classify it as `environmental`.
- `build-ui` uses `lint-check` and `format-check` (dry-run, no auto-fix) for validation. `/code-ui` may use `lint` and `format` (auto-fix) during implementation — that is intentional. Both are correct for their context.

Retry budget:

- static-analysis fix loops: maximum 2 full reruns of the failing command after making a fix
- if the same command still fails after 2 fix loops, stop and classify the failure

Emit:

```text
Verification Status
- type-check: passed|failed|skipped|not-run
- lint-check: passed|failed|skipped|not-run
- format-check: passed|failed|skipped|not-run
- build: passed|failed|skipped|not-run
- targeted-tests: passed|failed|skipped|not-run
- related-tests: passed|failed|skipped|not-run
- coverage-run: passed|failed|skipped|not-run
- browser-verification: passed|failed|skipped|not-run
- figma-verification: passed|failed|skipped|not-run
- screenshot-render-check: passed|failed|skipped|not-run
Failure Classification
- <command>: introduced-by-change|pre-existing|environmental|flaky-or-unclear|not-applicable
Evidence
- <command>: exact evidence for pre-existing/environmental/flaky classification, or `none`
```

Validation ownership:

- `build-ui` owns validation sequencing, status reporting, and stop or continue decisions.
- `/code-ui` owns implementation-time test design, mock completeness, and the first pass on failures introduced by the change.

Partial implementation recovery:

- do not rollback automatically if Phase 4 completed partially and Phase 5 fails
- preserve the branch state and fix forward on the same branch when practical
- if the failure cannot be resolved within the retry budget, stop and report the current branch state, failing command, and classified blocker in the handoff

## Phase 6 — Self-Review

Skip if `--no-review` is set.

Review the diff against:

- `docs/development/ui-review-kumo.md`
- `docs/development/ui-review-react.md` when relevant
- `docs/development/ui-review-testing.md` when relevant

High-severity means any issue that would be flagged `[HIGH]` by `/review-ui`:

- missing error boundaries on exported components
- raw `useNavigate()` instead of `useCustomNavigation()`
- direct axios usage instead of Orval hooks
- hardcoded routes, colors, or runtime config
- `console.log` instead of `logStackTrace()`
- missing loading, error, or empty states on data-backed surfaces

Fix high-severity issues before moving on. Medium-severity issues (naming, comment quality, minor style) can be noted but should not block.

## Phase 7 — Unit Testing

Derive tests from the accepted AC.

`build-ui` owns:

- choosing targeted vs related vs broader test scope
- deciding whether to stop, continue, or disclose skipped coverage
- aggregating classified failures into Verification Status and Handoff

Use `/code-ui` and `.agents/skills/code-ui/references/testing-playbook.md` for detailed test design, helper reuse, mock requirements, and the retry budget.

Phase-level retry budget:

- targeted test fix loops: maximum 3
- related suite retries: maximum 2
- coverage-mode retries: maximum 1
- if the retry budget is exhausted, stop and report the classified failure instead of continuing to broader phases

Classify every failing validation command using `references/decision-tables.md`.

A failure may only be marked `pre-existing` if there is concrete evidence from one of:

- the same failure on the base branch
- a clearly unrelated touched area already broken before the change
- prior CI or local evidence attached in the current run notes

If that evidence does not exist, do not classify it as `pre-existing`.

## Phase 8 — Visual Testing

Skip if `--skip-browser` is set, the selected mode omits browser work, or the dev server is down. If browser tooling is unavailable but `CI_SCREENSHOT_SCRIPT` is set, use the Playwright fallback below instead of skipping.

Use `references/decision-tables.md` section "Auth And Runtime Config For Visual Testing" before deciding whether visual verification can run, needs a local auth/runtime adjustment, or must be skipped.

Choose the capture path in this priority order:

1. **Local Playwright capture (default for known surfaces).** If a dev server is reachable and `ui/src/screenshot/catalog.json` has a recipe whose `patterns` match a changed file, capture with `ui/scripts/ci-screenshot.mjs` yourself — see "Local Playwright capture (interactive runs)" below. This is deterministic, reproducible, and produces the same evidence CI does. This applies in every interactive mode, not just headless CI.
2. **Browser MCP.** Use `mcp__chrome-devtools__*` tools only for freeform exploration of a surface with no matching recipe, or to investigate behavior a static screenshot cannot prove.
3. **Headless CI Playwright.** In headless/CI runs the workflow owns capture — see "CI Playwright Fallback".

Before visual testing:

- inspect runtime auth state
- inspect feature flag state
- understand whether backend data is available

Checks:

- final route loads
- console errors
- failed requests attributable to the change
- main AC path
- relevant UI states that are realistically reachable
- keyboard interaction and focus handling for newly introduced dialogs, menus, or forms
- visible labeling for interactive controls changed by the ticket

If screenshots are captured:

- save under `/tmp/build-ui-screenshots/{issue-or-pr}/`
- name them descriptively
- do not commit them to the branch

### Local Playwright capture (interactive runs)

In a local/interactive run (not headless CI), prefer this over browser MCP whenever a dev server is up and a catalog recipe matches the changed surface:

1. Find the matching recipe in `ui/src/screenshot/catalog.json` (match its `patterns` against the changed files; reuse its `canonicalRoutes`, `readySelector`, `setup`, and `visualCoverage`).
2. Write a plan to `/tmp/build-ui-screenshots/{issue-or-pr}/plan.json` — a `scenarios` array, each with `route`, deterministic `setup` steps, a `readySelector`, and `assertions` whose selectors target the exact changed element (not a page shell, tab list, or generic heading).
3. Capture against the local dev server:

   ```bash
   node ui/scripts/ci-screenshot.mjs \
     --plan /tmp/build-ui-screenshots/{issue-or-pr}/plan.json \
     --output /tmp/build-ui-screenshots/{issue-or-pr} \
     --base-url http://localhost:3000 \
     --auth-disabled
   ```

   `--auth-disabled` injects the local sentinel token; the local dev backend accepts it and supplies the canonical project/group authorization context used by every deployment.
4. Inspect `results.json` and the saved PNG. If capture fails, debug the artifacts (`screenshot-debug.md`, the failed `*.html`/`*.png` under `failed/`) before changing the plan. Common gotchas: a first-run "Welcome to Kumo" modal intercepts clicks (dismiss via `[data-testid="cancel-icon"]`), and list pages may not auto-select a row (click `[data-testid="entity-name-0"]` to open the detail panel).
5. Publish to the PR with `ui/scripts/build-ui-publish-screenshots.mjs` in Phase 9 — same `images` branch and `## Screenshots` section as headless CI.

The "agent must not shell out to `ci-screenshot.mjs`" rule below applies only to headless CI mode, where the workflow runs it for you. In local runs, running it yourself is the intended path.

### CI Playwright Fallback

When `mcp__chrome-devtools__*` tools are unavailable but the environment variable `CI_SCREENSHOT_SCRIPT` is set (headless CI mode), the workflow runs Playwright on the agent's behalf. In that headless mode the agent must not shell out to `ui/scripts/ci-screenshot.mjs`.

Instead, write a screenshot plan to `$BUILD_UI_ARTIFACT_DIR/screenshot-plan.json` with either:

- a simple route plan using routes and optional ready selectors, or
- a scenario plan using routes plus deterministic interaction steps and assertions when the changed UI is not visible on initial load.

See `references/headless-mode.md` for the plan schema, required-mode rules, and associated blocker codes, including `BLOCKER_SCREENSHOT_PLAN_MISSING`, `BLOCKER_SCREENSHOT_PLAN_INVALID`, `BLOCKER_SCREENSHOT_CAPTURE_FAILED`, `BLOCKER_SCREENSHOT_COVERAGE_MISSING`, `BLOCKER_SCREENSHOT_STATUS_EVALUATION_FAILED`, and `BLOCKER_SCREENSHOT_UPLOAD_FAILED`.

Route selection:

- always include the primary route affected by the implementation
- prefer the exact edited page over a nearby list, landing, or overview page
- include at most one secondary route, and only when the diff changes a shared component or cross-page UI surface
- if the change is isolated to one page, include only that page
- default fallback routes: `/connectors/,/tables/,/graphs/`

Selector quality:

- prefer route-specific readiness selectors that prove the changed surface rendered
- avoid weak selectors like `h1` when a page-specific test id or stable container exists
- use `[data-testid="main-layout"]` only as a last-resort fallback

Interaction-driven verification:

- if the changed UI requires typing, selection, or other user interaction before it becomes visible, write a scenario plan instead of switching to an easier static route
- include only deterministic setup actions the workflow can replay safely
- add an assertion that proves the intended visual state before the screenshot is taken; regex-style `matches` assertions are allowed for text that varies by task type or tenant data

In `standard`/`full` modes, the plan file is required even for non-visual changes; use the default fallback routes and note the rationale in the handoff.

The workflow handles auth bootstrap, page navigation, route validation, screenshot capture, manifest generation, artifact upload, and posting back to the PR or issue. In current CI, screenshot auth may be satisfied by browser-based login using dedicated screenshot credentials rather than a preissued token. The agent should treat screenshot auth as workflow-owned and must not make assumptions about the exact auth mechanism.

When screenshot capture fails in CI, prefer artifact-first debugging:

- inspect `screenshot-plan.json`
- inspect `ci-screenshot.log`
- inspect the screenshot manifest and failure HTML
- distinguish auth bootstrap failure from route/selector failure before changing the plan

When `CI_SCREENSHOT_SCRIPT` is not set and `mcp__chrome-devtools__*` is unavailable (a typical local run), do not skip by default: use **Local Playwright capture (interactive runs)** above when a dev server is reachable and a catalog recipe matches the changed surface. Only skip visual testing when no dev server is available or no recipe/plan can prove the changed surface.

## Checkpoint 2 — Approve PR

Present:

- implementation summary
- changed files summary
- static check results
- test results
- screenshots if available
- Figma or a11y notes if run
- caveats and skipped verification

Only proceed to PR creation if the user approves it or headless mode explicitly allows it. If `--validate-only` is set, stop after verification and handoff even if everything passed.

Use this output shape:

```text
Checkpoint 2
- mode: issue|pr
- target: {issue-or-pr}
- branch: {branch-name}
- changed files summary:
  - ...
- verification status:
  - type-check: ...
  - lint-check: ...
  - format-check: ...
  - build: ...
  - targeted-tests: ...
  - related-tests: ...
  - coverage-run: ...
  - browser-verification: ...
  - figma-verification: ...
  - screenshot-render-check: ...
- caveats:
  - ...
- blocker type: none|introduced-by-change|pre-existing|environmental|flaky-or-unclear|other
- decision needed: create-pr|adjust|stop
```

## Phase 9 — PR Creation

If approved:

1. fetch latest base branch
2. optionally rebase if the user wants that behavior
3. push the feature branch
4. determine the release-notes label
5. detect breaking changes
6. create the PR from `.github/PULL_REQUEST_TEMPLATE.md`
7. if screenshots were captured, publish them with `ui/scripts/build-ui-publish-screenshots.mjs` after PR creation so local PRs use the same `images` branch and `## Screenshots` section as headless CI

Before creating the PR, run a final diff sanity pass:

- confirm changed files still match the approved scope
- confirm no generated files were edited unless explicitly intended
- confirm lockfile deltas are intentional
- scan for `TODO`, `FIXME`, debug logging, and temporary comments in touched files
- confirm screenshot files are not accidentally staged unless the user explicitly wanted them committed

Use `references/pr-workflow.md` for detailed PR body, screenshot upload, and breaking-change rules.
Use `references/screenshot-operator-guide.md` when local or headless screenshot capture does not prove the ticket requirement.

## Resume Capability

Supported named stages:

- `preflight`
- `pr-inspection`
- `ticket-analysis`
- `recon`
- `design`
- `branch-setup`
- `implementation`
- `validation`
- `review`
- `tests`
- `visual`
- `pr`

Resume stage mapping:

| Resume stage      | Phase                     |
| ----------------- | ------------------------- |
| `preflight`       | Phase 0: Pre-flight       |
| `pr-inspection`   | Phase P: PR Inspection    |
| `ticket-analysis` | Phase 1: Ticket Analysis  |
| `recon`           | Phase 2: Recon            |
| `design`          | Phase 3: Design Reference |
| `branch-setup`    | Phase 0b: Branch Setup    |
| `implementation`  | Phase 4: Implementation   |
| `validation`      | Phase 5: Static Analysis  |
| `review`          | Phase 6: Self-Review      |
| `tests`           | Phase 7: Unit Testing     |
| `visual`          | Phase 8: Visual Testing   |
| `pr`              | Phase 9: PR Creation      |

Use the exact phase label from this table in `lastCompletedPhase`. Do not collapse validation, review, and tests into one run-state value.

Rules:

- reuse the existing work branch
- in `issue` mode, that is usually `ui/{issue}-*`
- in `pr` mode with continuation, that is usually `{headRefName}`
- verify current branch state before skipping earlier stages
- resume only when the branch state clearly corresponds to the requested stage
- if a previous run stopped after partial implementation, resume on the same branch and fix forward; do not discard or rollback partial changes automatically

### Run State Artifact

To make resume behavior reliable, each run should maintain a lightweight state artifact in the working notes for the session. The artifact should capture:

- mode and target
- linked issue or PR
- selected flags and mode
- chosen work branch
- last completed phase
- current blockers
- latest Verification Status

Use `references/run-state-template.json` as the canonical shape for this artifact.

Before honoring `--resume`, compare the requested stage against this artifact and the actual branch state. If they disagree, trust the branch state and report the mismatch explicitly before continuing.

Resume mismatch should be reported using:

```text
Resume Mismatch
- requested stage: ...
- recorded last completed phase: ...
- observed branch state: ...
- safe action: continue-from-actual-state|restart-planning|stop
```

## Command Installation Notes

The repository slash command entry is `.claude/commands/build-ui.md`, which
references this skill.

If `/build-ui` is not recognized, verify command entry placement before assuming the skill is unavailable.

## Workflow Validation

Use both validators after changing this skill:

- `bash .agents/skills/build-ui/scripts/validate-skill-structure.sh`
- `bash .agents/skills/build-ui/scripts/validate-workflow-behavior.sh`
- `bash .agents/skills/build-ui/scripts/validate-output-schemas.sh`

The structure validator checks required files and sections. The behavior validator checks that representative workflow scenarios still map to the required decisions, evidence, and fallback paths across the skill documents.
The output-schema validator checks that the documented output contracts and run-state artifact remain concrete and internally consistent.

## Final Handoff

At the end of any non-aborted run, emit:

```text
Handoff
- mode: issue|pr
- target: {issue-or-pr}
- linked issue: {issue-or-none}
- current branch: {work-branch}
- current PR: {pr-or-none}
- resume source: none|fresh-run|resumed-from-{stage}
- completed:
  - ...
- pending:
  - ...
- blocked:
  - ...
- blocker type: none|requirements|api|catalog|worktree-overlap|introduced-by-change|pre-existing|environmental|flaky-or-unclear|resume-mismatch|adjacent-fix|other
- next recommended action:
  - exact command, exact phase, or exact user action
```
