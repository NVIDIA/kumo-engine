# Build UI Output Contracts

Use this file as the canonical schema reference for workflow outputs. These are human-readable contracts, but they should stay stable enough for tooling and review automation.

## Checkpoint 1

Required keys:

- `mode`
- `target`
- `branch plan`
- `requested scope`
- `adjacent cleanup discovered`
- `deferred/out of scope`
- `catalog gap check`
- `AC to file mapping`
- `blockers`
- `blocker type`
- `scope change`
- `decision needed`

## Verification Status

Required keys:

- `type-check`
- `lint-check`
- `format-check`
- `build`
- `targeted-tests`
- `related-tests`
- `coverage-run`
- `browser-verification`
- `figma-verification`
- `screenshot-render-check`

Required subsections:

- `Failure Classification`
- `Evidence`

## Checkpoint 2

Required keys:

- `mode`
- `target`
- `branch`
- `changed files summary`
- `verification status`
- `caveats`
- `blocker type`
- `decision needed`

## Resume Mismatch

Required keys:

- `requested stage`
- `recorded last completed phase`
- `observed branch state`
- `safe action`

## Handoff

Required keys:

- `mode`
- `target`
- `linked issue`
- `current branch`
- `current PR`
- `resume source`
- `completed`
- `pending`
- `blocked`
- `blocker type`
- `next recommended action`

## Headless CI Artifacts

Required top-level keys for each JSON artifact:

- `artifactType`
- `generatedAt`
- `headless`
- `target`
- `payload`

Recommended artifact types:

- `preflight`
- `run-state`
- `checkpoint-1`
- `verification-status`
- `checkpoint-2`
- `handoff`
- `blocker`
- `artifact-manifest`

Auxiliary headless files that are not emitted by `write-headless-artifact.sh` but are part of the workflow contract:

- `screenshot-plan.json`
- `screenshot-plan-resolution.json`
- `screenshot-overrides.json`
- `figma-context.json`
- `figma-summary.md`

Recommended `run-state` payload fields in CI:

- `runId`
- `runUrl`
- `trigger`
- `actor`
- `flags`
- `baseBranch`
- `baseShaAtStart`
- `baseShaAtEnd`
- `staleBaseBranch`
- `jobStatus`
- `artifactDir`
- `figmaContext`

Rules:

- `flags` should include structured booleans for `dryRun`, `validateOnly`, `skipFigma`, `skipScreenshots`, `skipBrowser`, and `noReview`, plus `mode`, `baseBranch`, and `resumeStage`.

Recommended `artifact-manifest` payload fields:

- `artifacts`
- `count`
- `missingExpected`

Recommended `verification-status` payload fields in workflow-owned screenshot mode:

- `screenshotAttemptStatus`
- `requestedMode`
- `screenshotPlanPresent`
- `screenshotWorkflowOutcome`
- `screenshotFailureReason`

Recommended `blocker` payload fields for screenshot-related blockers:

- `code`
- `reason`
- `requestedMode`
- `screenshotPlanPath`
- `screenshotManifestPath`
- `screenshotLogPath`
- `screenshotWorkflowOutcome`

## Screenshot Plan

Required keys:

- exactly one of `routes` or `scenarios`

Optional keys:

- `readySelectors`
- `fixtures`
- `_recipeKey`
- `_recipeKeys`
- `_rationale`
- `_requiredFeatureFlags`
- `_screenshotOverride`
- `_visualCoverage`

Rules:

- `routes` must be a non-empty array when a simple route plan is used
- `scenarios` must be a non-empty array when an interaction-driven plan is used
- scenario state seeding may use `historyState`, `localStorage`, and `sessionStorage`
- seeded state fields must be JSON objects with non-empty string keys
- `fixtures` must be an object keyed by fixture name; fixture entries may include `historyState`, `localStorage`, `sessionStorage`, `setup`, `assertions`, and `readySelector(s)`
- scenarios may reference fixtures with `fixtures` as a string or string array
- built-in fixture names include `seeded_query_editor`, `selected_tab_model_planner`, `open_graph_table_view`, `opened_dialog`, `list_page_loaded`, `empty_state_loaded`, `selected_connector_state`, `selected_graph_state`, and `selected_table_state`
- `readySelectors` must be an object when present
- `readySelectors["*"]` may be used as a global fallback selector
- route-specific selector keys should match entries from the planned routes
- each scenario must include a `route`
- supported scenario setup actions are `click`, `fill`, `paste`, `type`, `press`, `select`, and `waitFor`
- supported scenario assertion types are `visible` and `text`
- `text` assertions may use `contains`, `equals`, or `matches`
- `_recipeKey` is a stable screenshot catalog key when a catalog recipe was selected
- `_recipeKeys` lists all selected screenshot catalog keys when multiple catalog-backed surfaces are selected
- `_rationale` explains why the helper preserved, merged, or rewrote the plan
- `_requiredFeatureFlags` is an object of feature-flag expectations applied to catalog-backed scenarios
- `_screenshotOverride` records explicit screenshot override intent derived from trigger flags
- `_visualCoverage` is an array of `{ "id": "...", "label": "..." }` objects describing the intended visual requirements
- scenario `visualCoverage` narrows or overrides plan-level `_visualCoverage` for that scenario
- scenario `requiredFeatureFlags` narrows or overrides plan-level `_requiredFeatureFlags` for that scenario
- scenario `presentation` is `primary`, `regression`, or absent; `primary` screenshots are intended for PR-body emphasis
- setup actions and assertions may include `covers` as a non-empty string or string array of `_visualCoverage[].id` values
- every `_visualCoverage[].id` must be referenced by at least one setup action or assertion `covers` entry
- uncataloged scenario plans without `_recipeKey` or `_visualCoverage` must include `covers` on assertions so the screenshot intent is explicit
- screenshot assertions should cover the user-visible requirements being verified, not only route readiness
- `text` assertions must target the specific changed element rather than a broad container such as `main-layout`, a tab list, or a generic heading
- interaction-driven fixes should prefer `scenarios` over unrelated fallback routes
- invalid screenshot-plan schema should produce a structured blocker before Playwright runs

## Screenshot Plan Resolution

Required keys:

- `generatedAt`
- `planPath`
- `changedFiles`
- `existingPlanPresent`
- `matchedRecipes`
- `selectedRecipe`
- `selectedRecipes`
- `screenshotOverride`
- `action`
- `reason`
- `verification`

Rules:

- `matchedRecipes` includes each recipe key, matched files, canonical routes, legacy routes, and structured visual coverage.
- `selectedRecipe` is set when the helper writes or rewrites the plan from the catalog.
- `selectedRecipes` lists every selected catalog recipe when the helper resolves a multi-surface plan.
- `screenshotOverride` records explicit screenshot override intent when trigger flags selected a route, recipe, coverage IDs, or fixtures.
- `action` is one of `wrote`, `rewrote-legacy`, `rewrote-mismatch`, `rewrote-recipe`, `kept`, `skipped`, `wrote-override`, `failed-override`, `merged-primary`, or `rewrote-multi`.
- `skipped` means no catalog recipe matched; a freeform plan must carry explicit assertion coverage.
- `wrote-override` means explicit screenshot override flags produced a catalog-backed plan.
- `failed-override` means explicit screenshot override flags were present but could not resolve to a valid catalog-backed plan.
- `merged-primary` means agent-authored primary evidence was preserved and catalog regression scenarios were appended.
- `rewrote-multi` means multiple matched catalog surfaces were resolved into one multi-scenario plan.
- `verification.status` is `passed`, `failed`, or `skipped` and records route/coverage consistency checks after recipe resolution.

## Screenshot Results

Recommended keys in `results.json`:

- `coverageFailed`
- `missingCoverage`
- `failureTypes`

Rules:

- `captured` status requires zero failed routes and zero missing coverage.
- missing coverage produces `BLOCKER_SCREENSHOT_COVERAGE_MISSING`, not a generic capture failure.
- failed route entries should include `failureType` with one of `auth`, `route-mismatch`, `selector-readiness`, `app-runtime`, `coverage-missing`, or `unknown`.
- screenshot status evaluation failures produce `BLOCKER_SCREENSHOT_STATUS_EVALUATION_FAILED`.
- screenshot publishing/upload failures produce `BLOCKER_SCREENSHOT_UPLOAD_FAILED` with `screenshotFailureType: screenshot-upload`.

## Headless GitHub Comment

Required sections:

- `Headless Run`
- `Blocker` when applicable
- `Verification Status` when validation was reached
- `Artifacts`
- `Next Action`
