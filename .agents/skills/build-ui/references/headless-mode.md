# Build UI Headless Mode

When `build-ui` runs from a GitHub Actions workflow (triggered by a `/build-ui` comment), it operates in headless mode. No human is in the loop during execution.

## Default Flags

The default mode is `light`. The comment author can override with `--mode standard` or `--mode full`.

Do not use interactive Figma MCP in CI. Headless mode uses workflow-owned Figma REST extraction when issue context contains Figma links and `FIGMA_ACCESS_TOKEN` is configured. The comment author can still pass `--skip-figma` to skip Figma REST extraction for that run.

Do not force `--skip-browser` in CI. Headless mode still skips interactive browser MCP usage, but Phase 8 is workflow-owned: the agent proposes a capture plan and the workflow runs Playwright.

Screenshot capture uses Playwright via `ui/scripts/ci-screenshot.mjs` instead of Chrome DevTools MCP. The workflow starts a dev server proxying to the staging backend and owns screenshot execution directly. Screenshot auth is also workflow-owned and uses Direct Connect password-mode credentials or the auth-disabled sentinel, depending on the backend runtime config. The agent must not depend on any auth mechanism beyond the workflow contract.

The headless agent must not execute the screenshot script. It proposes a capture plan by writing `screenshot-plan.json` into the headless artifact directory, and the workflow validates that plan and runs Playwright itself. This keeps screenshot execution deterministic and makes missing screenshots a workflow-enforced outcome rather than an agent-side omission.

In `standard` and `full` modes, `screenshot-plan.json` is required regardless of whether the change is visual. If the implementation is non-visual (for example, a utility helper or a pure logic fix), the agent must still write a plan using the default app-shell routes (`/connectors/`, `/tables/`, `/graphs/`) and note the rationale in the handoff. Skipping the file in `standard`/`full` triggers a hard blocker and fails the job.

The workflow builds the in-tree catalog package with `npm run install:catalog`.
This command does not rewrite `ui/package.json` or `ui/package-lock.json`.

If the implementation later requires a real new npm dependency, install it during the implementation phase and keep the resulting `ui/package.json` / `ui/package-lock.json` changes in the PR. Do not include unrelated dependency-install changes.

## Checkpoint Behavior

### Checkpoint 1 — Approve Implementation

Auto-approve and proceed unless a hard blocker is detected.

Hard blockers that stop execution:

- acceptance criteria too vague to synthesize concrete ACs
- required API hook missing from both `gen/v1/` and `gen/v0/`
- catalog gap that cannot be worked around
- file plan exceeds 20 modified files or 3 feature areas

Use these stable blocker codes in CI artifacts and GitHub comments:

| Blocker                                                 | Code                                          |
| ------------------------------------------------------- | --------------------------------------------- |
| acceptance criteria too vague                           | `BLOCKER_REQUIREMENTS_VAGUE`                  |
| required API hook missing                               | `BLOCKER_API_MISSING`                         |
| catalog gap cannot be worked around                     | `BLOCKER_CATALOG_GAP`                         |
| file plan too large                                     | `BLOCKER_SCOPE_TOO_LARGE`                     |
| resume request does not match actual branch state       | `BLOCKER_RESUME_MISMATCH`                     |
| workflow failed before a structured blocker was emitted | `BLOCKER_WORKFLOW_FAILURE`                    |
| screenshot plan missing in `standard`/`full` mode       | `BLOCKER_SCREENSHOT_PLAN_MISSING`             |
| screenshot plan JSON invalid or malformed               | `BLOCKER_SCREENSHOT_PLAN_INVALID`             |
| plan present but Playwright capture failed              | `BLOCKER_SCREENSHOT_CAPTURE_FAILED`           |
| screenshot file exists but required coverage is missing | `BLOCKER_SCREENSHOT_COVERAGE_MISSING`         |
| screenshot status evaluation failed                     | `BLOCKER_SCREENSHOT_STATUS_EVALUATION_FAILED` |
| screenshot capture succeeded but publication failed     | `BLOCKER_SCREENSHOT_UPLOAD_FAILED`            |

When a hard blocker is detected, post the blocker details as a GitHub comment on the issue and stop. Do not proceed with implementation.

When no hard blockers exist, approve and continue. The PR review is the human approval gate.

### Checkpoint 2 — Approve PR

Auto-approve and create the PR unless:

- static checks (type-check, lint-check, format-check, build) are failing with `introduced-by-change` classification
- targeted tests are failing with `introduced-by-change` classification

When checks fail, attempt to fix within the retry budget. If the retry budget is exhausted and failures persist, post the Verification Status and failure details as a GitHub comment and stop without creating a PR.

When all checks pass (or failures are classified as `pre-existing`, `environmental`, or `flaky-or-unclear`), create the PR and note any caveats.

If the run was triggered in validate-only mode, auto-approve Checkpoint 2 for validation/reporting purposes but do not push branches or create a PR.

## Phase Behavior in Headless Mode

| Phase              | Behavior                                                                                                                                                                              |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0 Pre-flight       | Runs. GitHub auth provided by workflow. `check-preflight.sh`, `gh auth status`, and `git status --porcelain` output must be saved as artifacts and included in the job summary.       |
| P PR Inspection    | Runs if target is a PR.                                                                                                                                                               |
| 1 Ticket Analysis  | Runs.                                                                                                                                                                                 |
| 2 Recon            | Minimal in `light` mode, full in `standard`/`full`.                                                                                                                                   |
| 3 Design Reference | Uses workflow-owned `figma-context.json` / `figma-summary.md` when Figma REST extraction succeeds. Otherwise skipped with the recorded Figma status/reason. No interactive Figma MCP. |
| Checkpoint 1       | Auto-approved unless hard blocker.                                                                                                                                                    |
| 0b Branch Setup    | Runs.                                                                                                                                                                                 |
| 4 Implementation   | Runs via `/code-ui`.                                                                                                                                                                  |
| 5 Static Analysis  | Runs. `type-check` + `lint-check` only in `light`; all four in `standard`/`full`.                                                                                                     |
| 6 Self-Review      | Skipped in `light`; runs in `standard`/`full`.                                                                                                                                        |
| 7 Unit Testing     | Targeted tests only in `light`; targeted + related in `standard`/`full`.                                                                                                              |
| 8 Visual Testing   | Agent writes `screenshot-plan.json`; workflow runs Playwright screenshot capture when `CI_SCREENSHOT_SCRIPT` is set. No interactive browser testing.                                  |
| Checkpoint 2       | Auto-approved unless introduced-by-change failures remain.                                                                                                                            |
| 9 PR Creation      | Runs unless validate-only mode is active. In validate-only runs, stop after verification artifacts and handoff.                                                                       |

## Required CI Artifacts

Headless mode should persist these machine-readable artifacts for every run:

- `preflight.json`
- `run-state.json`
- `checkpoint-1.json` when Checkpoint 1 is reached
- `verification-status.json` after validation begins
- `checkpoint-2.json` when Checkpoint 2 is reached
- `handoff.json` on completion
- `blocker.json` when execution stops on a hard blocker
- `artifact-manifest.json` listing the artifact bundle contents for the run
- `figma-context.json` when Figma extraction ran or was explicitly skipped
- `figma-summary.md` when Figma extraction ran or was explicitly skipped

When visual verification is needed, the agent should also write:

- `screenshot-plan.json` with the routes and optional ready selectors the workflow should capture

Use `.agents/skills/build-ui/scripts/write-headless-artifact.sh` to emit stable JSON skeletons for these files. Attach them as workflow artifacts even when the run fails.
Use `.agents/skills/build-ui/scripts/update-headless-artifact.sh` to merge real CI metadata into the generated artifacts as the run progresses.

The JSON envelope schema (`artifactType`, `generatedAt`, `headless`, `target`, and `payload`) is owned by `references/output-contracts.md`. This file owns when artifacts are required and which workflow behavior should populate them.

## Figma REST Contract

Headless Figma extraction is workflow-owned. The agent must not call Figma MCP or fetch Figma directly in CI.

The workflow scans the issue body and comments for `figma.com/file/...`, `figma.com/design/...`, and `figma.com/proto/...` links. When `FIGMA_ACCESS_TOKEN` is configured, it fetches Figma REST metadata and rendered node images into the headless artifact directory.

Artifacts:

- `figma-context.json`: machine-readable extraction result
- `figma-summary.md`: human-readable extraction summary
- `figma-nodes/`: Figma file/node JSON payloads when fetched
- `figma-images/`: rendered PNG node images when node IDs are present

`figma-context.json` statuses:

- `extracted`: links found and REST artifacts fetched successfully
- `partial`: links found, but one or more Figma REST requests failed
- `not-found`: no Figma links found in issue context
- `skipped`: skipped by `--skip-figma` or missing `FIGMA_ACCESS_TOKEN`
- `failed`: extraction failed before producing normal context

If status is `extracted` or `partial`, Phase 3 should read `figma-summary.md` and inspect relevant node JSON/images before implementation. Figma is advisory; implementation still follows issue requirements, Kumo design tokens, and existing UI patterns. If status is `not-found`, `skipped`, or `failed`, continue and record the status in handoff/verification.

For `standard` and `full` mode runs, missing `screenshot-plan.json` or a failed workflow-owned capture should be treated as a structured blocker rather than a soft warning.

## Screenshot Plan Contract

When the workflow requires screenshots, the agent must write `screenshot-plan.json` with one of these shapes.

Use a route plan when the changed UI is visible on initial page load:

```json
{
  "routes": ["/primary-route/"],
  "readySelectors": {
    "*": "[data-testid=\"main-layout\"]",
    "/primary-route/": "[data-testid=\"route-ready-selector\"]"
  }
}
```

Use a scenario plan when the changed UI only appears after interaction, async state changes, or form input:

```json
{
  "_recipeKey": "primary_changed_surface",
  "_visualCoverage": [
    {
      "id": "primary_changed_surface_text",
      "label": "Changed surface text matches the accepted requirement"
    }
  ],
  "fixtures": {
    "seed_query_editor": {
      "setup": [
        {
          "action": "fill",
          "selector": "[data-testid=\"query-editor\"]",
          "value": "SELECT ..."
        }
      ],
      "readySelector": "[data-testid=\"query-editor\"]"
    }
  },
  "scenarios": [
    {
      "name": "primary-changed-state",
      "route": "/primary-route/",
      "fixtures": ["seed_query_editor"],
      "setup": [
        {
          "action": "click",
          "selector": "[data-testid=\"graph-selector\"]"
        }
      ],
      "readySelector": "[data-testid=\"changed-surface\"]",
      "assertions": [
        {
          "type": "text",
          "selector": "[data-testid=\"changed-surface\"]",
          "contains": "Expected text",
          "covers": ["primary_changed_surface_text"]
        }
      ]
    }
  ]
}
```

Known page-specific fallbacks are maintained in `ui/src/screenshot/catalog.json`. Route/workflow ownership is maintained in `ui/src/screenshot/route-map.json` so broad changed files can still resolve to catalog-backed recipes. When a page needs deterministic visual state that is hard to derive from live tenant data, prefer seeding state through `historyState`, `localStorage`, or `sessionStorage` in the scenario rather than inventing brittle tenant-specific interactions.

For catalog gaps, use `references/decision-tables.md`. When the decision-table row for a missing required catalog variant applies, an unmerged catalog PR can be an acceptable documented workaround if the UI PR body names the catalog branch and the temporary dependency/exception.

Scenario plans may define reusable `fixtures` for common states such as seeded query editor content, selected tabs, selected graph/table state, an opened drawer/modal, list/detail views, or empty states. Fixtures can include `historyState`, `localStorage`, `sessionStorage`, `setup`, `assertions`, and `readySelector(s)`. A scenario opts in with `fixtures: ["fixture_name"]`; scenario-specific state and setup are merged after fixture state.

`ui/scripts/ci-screenshot.mjs` also provides built-in fixtures for common states: `seeded_query_editor`, `selected_tab_model_planner`, `open_graph_table_view`, `opened_dialog`, `list_page_loaded`, `empty_state_loaded`, `selected_connector_state`, `selected_graph_state`, and `selected_table_state`. Plan-local fixtures with the same name override built-ins.

Catalog recipes must include `visualCoverage` as `{ "id": "...", "label": "..." }` objects and map each coverage ID to concrete setup actions or assertions with `covers`. This keeps recipes reviewable: readers can see both the visual intent and the selector or action that proves it.

The workflow writes `screenshot-plan-resolution.json` next to `screenshot-plan.json` after catalog resolution. It records changed files, matched recipes, selected recipe(s), screenshot override metadata, action (`wrote`, `rewrote-*`, `kept`, `skipped`, `wrote-override`, `failed-override`, `merged-primary`, or `rewrote-multi`), recipe coverage metadata, and a verification verdict. Use it to debug why a screenshot targeted a given page.

The workflow may also write `screenshot-overrides.json` when the trigger includes explicit screenshot intent flags: `--screenshot-route`, `--screenshot-recipe`, `--screenshot-coverage`, or `--screenshot-fixture`. Overrides must resolve to a catalog-backed recipe; they should not be used to invent arbitrary selectors or bypass coverage.

Screenshot planning priority:

1. exact recipe from `ui/src/screenshot/catalog.json`
2. deterministic seeded-state scenario for the changed surface
3. interaction-driven scenario for the changed surface
4. simple route plan only when the changed UI is already visible on initial load

Do not prefer an agent-authored freeform route or scenario over a matching catalog recipe for the same surface.

Rules:

- the first route must be the primary page changed by the fix
- prefer the exact edited page over adjacent list, overview, or landing pages
- include at most one secondary route, and only when shared components or cross-page UI behavior in the diff need validation
- if the change is isolated to one page, include only that page
- omit `readySelectors` entirely when route-specific readiness is not needed
- use `*` only for a fallback selector shared across all routes
- prefer route-specific readiness selectors that prove the changed surface rendered
- avoid weak selectors such as `h1` when a stable page-specific selector exists
- if the changed surface requires interaction, express that interaction in `scenarios[].setup` rather than choosing an easier but less relevant route
- when the changed surface can be rendered deterministically from seeded route or storage state, prefer that over tenant-specific live data setup
- keep scenario actions bounded to deterministic UI steps such as `click`, `fill`, `paste`, `type`, `press`, `select`, and `waitFor`
- scenario state seeding may use `historyState`, `localStorage`, and `sessionStorage`
- reusable scenario fixtures may provide common seeded state, setup steps, assertions, and readiness selectors
- prefer stable `data-testid` selectors for page headings, list pages, empty states, forms, graph/table pickers, dialogs, and exact changed controls
- prefer `data-screenshot-ready` markers for route/page readiness after auth, route load, and initial UI hydration
- use `assertions` to prove the fix is present before the screenshot is taken; `text` assertions may use `contains`, `equals`, or `matches`
- every user-visible acceptance criterion that the screenshot is meant to verify should be covered by at least one assertion
- catalog-backed setup actions and assertions should use `covers` with stable `visualCoverage[].id` values to identify which requirement they prove
- every declared `visualCoverage[].id` must be referenced by at least one setup action or assertion `covers` entry
- uncataloged scenario plans must include `covers` on assertions so review can tell which requirement each screenshot is proving
- assertions should point at the exact changed element; do not assert text on broad containers such as `main-layout`, tab lists, or generic headings
- if no stable exact selector exists for the changed element, add one in the implementation instead of relying on container text aggregation
- do not write an empty route list

The workflow owns execution after this file exists. It runs `ui/scripts/ci-screenshot.mjs`, uploads the resulting screenshot artifacts, and decides whether missing or failed capture should block completion.

Before running Playwright, the workflow must validate the `screenshot-plan.json` schema. Invalid JSON, empty routes, malformed selectors, malformed scenario actions/assertions, or selector keys that do not match the planned routes should stop the run with a structured blocker rather than being treated as a generic screenshot failure.

When the workflow emits `blocker.json` for screenshot problems, it should include the concrete failure reason and relevant artifact paths such as the screenshot plan, manifest, capture log, or screenshot debug summary. Failed screenshot runs should include `screenshot-debug.md` in the screenshot artifact bundle with route, final URL, DOM/auth snapshot, missing coverage, and recent browser/API events.

Recommended screenshot failure split:

- auth bootstrap failed: token/claims/login/session did not produce an authenticated app state
- route mismatch: page landed on the wrong route or wrong origin
- selector/readiness failure: route loaded but the expected UI never rendered
- app runtime failure: route loaded but console/API failures prevented readiness
- coverage missing: screenshot file exists but at least one declared `visualCoverage` item was not proven before capture
- screenshot upload failed: screenshot capture completed but publishing images/comments/PR body updates failed

## Pre-flight Publication

Headless mode must publish the output of `check-preflight.sh` in two places:

- workflow artifact bundle
- GitHub job summary or issue/PR comment

Headless Phase 0 must also persist `gh auth status` and `git status --porcelain` output in the artifact bundle and job summary.

If a pre-flight dependency is intentionally not required for a run, set the matching variable so `check-preflight.sh` emits `not-needed`:

- `BUILD_UI_DEV_SERVER_NOT_NEEDED`
- `BUILD_UI_BACKEND_NOT_NEEDED`
- `BUILD_UI_BROWSER_NOT_NEEDED`
- `BUILD_UI_RUNTIME_CONFIG_NOT_NEEDED`
- `BUILD_UI_CATALOG_NOT_NEEDED`

This is required for `environmental` classifications. Do not mark a failure as `environmental` in CI without preserving the supporting pre-flight output.

## Artifact-First Debugging

When a headless run fails, upload and retain:

- pre-flight output
- run-state JSON
- verification-status JSON
- screenshot manifest and failure artifacts if visual testing ran
- failing command summary
- changed-file summary
- artifact-manifest JSON

Prefer attaching structured artifacts over relying on prose-only failure comments.

For screenshot failures specifically, retain when available:

- `screenshot-plan.json`
- `ci-screenshot.log`
- screenshot manifest
- failed HTML and PNG captures
- auth/token diagnostic lines

## Stale-Branch Protection

Before creating or updating a PR in headless mode:

- check whether the base branch changed materially since the run started
- check for a newly opened or updated related PR that now overlaps the issue

If stale-branch risk is detected, stop and emit a blocker or caveat instead of blindly creating or updating the PR.

## Bounded Auto-Fix Behavior

Headless mode may apply only narrow, low-risk fix-forward actions automatically:

- formatting already touched files
- obvious lint autofixes on already touched files
- deterministic import-order cleanup on already touched files

Do not auto-apply broader speculative fixes in CI. If a failure needs non-trivial reasoning or wider edits, stop and emit structured evidence instead.

## Reporting

The workflow posts GitHub comments at these points:

- **Start**: acknowledges the command with a link to the workflow run
- **Blocker** (if any): checkpoint details and the specific blocker
- **Failure** (if workflow errors): link to workflow run for debugging
- **Completion**: the skill's Final Handoff output, plus a link to the created PR

In validate-only mode, the completion comment should explicitly say that implementation and verification finished without PR creation by design.

Use stable comment sections in headless mode:

- `Headless Run`
- `Blocker`
- `Verification Status`
- `Artifacts`
- `Next Action`

Include blocker codes and artifact names when applicable so humans and tools can parse the comments consistently.

The `claude-code-action` may also post intermediate comments as part of its normal operation.

## Triggering

From any GitHub issue comment:

```
/build-ui 12345
/build-ui 12345 --mode standard
/build-ui 12345 --dry-run
```

Or via manual workflow dispatch with `issue_number` and optional `flags` inputs.

Only users with write access to the repository can trigger the workflow.
