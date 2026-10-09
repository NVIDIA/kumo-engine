# Build UI Decision Tables

Use this file for stop/skip decisions, dependency handling, catalog gaps, browser scope, and validation failure classification.

## Preconditions And Fallbacks

| Phase | Preconditions                                                                          | If unmet                             | Result                                  |
| ----- | -------------------------------------------------------------------------------------- | ------------------------------------ | --------------------------------------- |
| 0     | Working git repo, readable worktree                                                    | Stop                                 | Cannot plan safely                      |
| 0     | `gh` available/authenticated for issue or PR work                                      | Stop or continue local-only planning | No GitHub-backed issue/PR work          |
| 0b    | User approved implementation                                                           | Stop                                 | No mutations before approval            |
| 0b    | Base branch exists on origin                                                           | Stop                                 | Branch setup blocked                    |
| P     | PR exists and is readable                                                              | Stop                                 | Cannot inspect existing work            |
| 1     | Issue exists and is readable                                                           | Stop                                 | Cannot derive requirements              |
| 1     | Enough issue detail to synthesize AC                                                   | Ask user to clarify                  | No implementation without usable AC     |
| 2     | Relevant code is readable                                                              | Stop                                 | Recon blocked                           |
| 2     | Required API path exists or is understood                                              | Stop or proceed mock-only/UI-only    | Must be called out explicitly           |
| 3     | Figma link exists and configured tooling can access it                                 | Skip                                 | Continue without design verification    |
| 5     | Dependencies installed and repo builds locally                                         | Stop after retry budget is exhausted | Not CI-ready                            |
| 7     | Test runner available                                                                  | Stop after retry budget is exhausted | Validation incomplete                   |
| 8     | Dev server reachable and browser tooling available                                     | Skip                                 | Visual validation incomplete            |
| 9     | User approved PR creation and push access exists, and validate-only mode is not active | Stop                                 | No PR creation                          |
| 9     | GitHub CLI can write to the repo when screenshots need publication                     | Skip screenshot upload               | PR created without embedded screenshots |

Fallback policy:

- Required precondition fails: stop and report the exact blocker.
- Optional precondition fails: skip the phase and record that it was skipped.
- Never silently replace verified results with assumptions.

## Catalog And Dependency Setup

| Situation                                      | Action                                                                                                                                                                                                                                                         |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Catalog present and installs healthy           | Do nothing                                                                                                                                                                                                                                                     |
| Catalog source missing | Report an incomplete checkout; the source belongs at `packages/kumo-ui-catalog`                                                                                                                                                                                                                          |
| Catalog dependencies or build missing | From `ui/`, run `npm run install:catalog && npm install` when dependency setup is authorized; the catalog is an in-tree file dependency |
| Lockfile reinstall explicitly intended         | Run `npm ci`                                                                                                                                                                                                                                                   |
| Dependency path unclear                        | Stop and ask before mutating npm config                                                                                                                                                                                                                        |

## Catalog Gap Resolution

| Situation                                                 | Action                                                                                                       |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Required variant/component missing from `kumo-ui-catalog` | Add it in `packages/kumo-ui-catalog` and update UI callers in the same repository         |
| Catalog change exists in another PR but is not merged         | Proceed with UI implementation assuming the variant will exist and add a blocking note in the PR description |
| User approves a temporary local workaround                | Implement the workaround with a `TODO` tied to the catalog change and track removal in deferred scope       |
| User wants to wait for the catalog change                    | Pause implementation and record the blocker in the handoff                                                   |

Rules:

- If the catalog change is in another PR, document that dependency in the UI PR body, including the catalog branch name and what it adds.
- Do not silently approximate a missing variant with ad hoc inline styles.

## API Availability

| Situation                                 | Action                                                      |
| ----------------------------------------- | ----------------------------------------------------------- |
| Hook exists in `gen/v1/`                  | Use it                                                      |
| Missing in `gen/v1/`, exists in `gen/v0/` | Use `gen/v0/` and note it                                   |
| Missing in `gen/`, present in spec        | Codegen or explicit user decision required                  |
| Missing in spec and backend unavailable   | Stop, or proceed only with explicit mock-only/UI-only scope |

## Auth And Runtime Config For Visual Testing

| Situation                                                                                        | Action                                                                                                                                                     |
| ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `AUTH_DISABLED` already true at runtime                                                          | Proceed (auth-disabled sentinel; no credentials needed)                                                                                                    |
| `AUTH_DISABLED` false and `SCREENSHOT_AUTH_EMAIL`/`SCREENSHOT_AUTH_PASSWORD` in the env | Run the catalog-recipe capture (`npm run screenshots:build-ui`) — Direct Connect credentials take precedence and the browser password-login path is used, no `env.js` change |
| `AUTH_DISABLED` false and no credentials present                                                 | Ask the user to either provide credentials (env vars below), approve a local `env.js` bypass, or skip — do not silently skip                               |
| `AUTH_DISABLED` false and user approves local bypass change                                      | Edit `ui/env.js`, then regenerate runtime config                                                                                                           |
| `AUTH_DISABLED` false and user declines                                                          | Skip visual testing                                                                                                                                        |
| Runtime config still mismatched after update                                                     | Stop and report the exact mismatch                                                                                                                         |

Credentials are never stored in the repo. The skill sources them from the environment it runs in:

- `SCREENSHOT_AUTH_EMAIL` / `SCREENSHOT_AUTH_PASSWORD` — a Direct Connect screenshot/test account; the capture script performs the browser password login itself.

Export them before launching, or keep them in an untracked local env file the agent reads — never commit them. If neither is set and the backend enforces auth, the capture fails fast with a message asking for credentials rather than capturing a login screen.

## Browser Verification Scope

| Situation                                                                     | Action                                                                                                                                                                                                                             |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dev server up, backend up, route reachable                                    | Full browser verification                                                                                                                                                                                                          |
| Dev server up, backend down                                                   | Structure, layout, and navigation only                                                                                                                                                                                             |
| Dev server up, backend up, `CI_SCREENSHOT_SCRIPT` set, no interactive browser | Playwright screenshot capture via `ci-screenshot.mjs`. Auth is workflow-owned and may use browser login or token bootstrap. Requested routes must load successfully; artifact logs replace interactive console/network inspection. |
| Browser tooling unavailable                                                   | Skip browser phase                                                                                                                                                                                                                 |
| Feature flag hides route/feature                                              | Record flag state, adjust the target route, or skip                                                                                                                                                                                |

Browser fallback states to record explicitly:

- `browser-tooling-unavailable`
- `browser-available-github-auth-missing`
- `browser-available-upload-control-not-found`
- `browser-available-upload-succeeded-render-failed`
- `ci-playwright-screenshots-captured`
- `ci-playwright-screenshots-failed`

Screenshot auth and planning decision points:

| Situation                                                                   | Action                                                                          |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Workflow-issued token missing required customer permissions                 | Stop screenshot execution early and classify as auth bootstrap failure          |
| Browser login succeeds but authenticated API calls still fail                | Stop and report wrong screenshot account, disabled Direct Connect user, or missing customer access |
| Screenshot plan route is not the edited page                                | Replace with the primary changed page unless a concrete shared-UI reason exists |
| Secondary screenshot route has no diff-backed rationale                     | Drop it from the plan                                                           |
| Route loads but selector is too generic to prove the changed UI rendered    | Replace with a page-specific selector                                           |
| Route loads, selector never appears, and logs show API/console failures     | Classify as app runtime failure before changing route selection                 |

## PR Creation

| Situation                                                     | Action                                                                                                                                                                    |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| User approved PR creation and push works                      | Create PR                                                                                                                                                                 |
| Validate-only mode is active                                  | Do not create PR; stop after verification and handoff                                                                                                                     |
| Rebase conflicts occur                                        | Stop and present conflicts                                                                                                                                                |
| Release-notes label is obvious                                | Recommend it, confirm if needed                                                                                                                                           |
| Label unclear                                                 | Ask explicitly before PR creation                                                                                                                                         |
| Screenshots available and GitHub CLI can write to the repo    | After PR creation, run `ui/scripts/build-ui-publish-screenshots.mjs` to publish primary images to the `images` branch, patch the PR body, and post the screenshot comment |
| Screenshots available but GitHub CLI cannot write to the repo | Skip upload and note in the PR that screenshots were verified locally                                                                                                     |
| Screenshots available but browser tooling unavailable         | Use the shared image-branch publisher; browser tooling is not required for screenshot upload                                                                              |
| Screenshots captured via CI Playwright script                 | Screenshots are posted back to the PR or issue comment and uploaded as workflow artifacts. No browser-based GitHub upload needed.                                         |
| No screenshots                                                | Create PR without them                                                                                                                                                    |

## Validation Failure Classification

| Situation                                      | Action                                        |
| ---------------------------------------------- | --------------------------------------------- |
| Failure clearly caused by current change       | Mark `introduced-by-change` and fix           |
| Failure existed before current change          | Mark `pre-existing` and report                |
| Failure caused by missing services/tooling/env | Mark `environmental`                          |
| Cause uncertain or flaky                       | Mark `flaky-or-unclear` and state uncertainty |

Evidence rules:

- `introduced-by-change`: point to the touched file, changed behavior, or failing assertion linked to the current diff
- `pre-existing`: include concrete evidence from base branch behavior, prior CI, or a clearly unrelated broken area
- `environmental`: include the missing dependency, service, auth, tooling, or resource limit that caused it
- `flaky-or-unclear`: explain why the cause is uncertain and why stronger classification was not possible

Screenshot-specific classification hints:

- auth bootstrap failure: token claims missing, wrong org/customer, or browser login did not establish an authenticated session
- route/selector failure: authenticated route loaded but the selected UI marker was wrong or never appeared
- app runtime failure: authenticated route loaded but API `4xx/5xx`, console errors, or loading state prevented render

Do not use `pre-existing` without evidence.

## Shared Component Blast Radius

When a change touches shared UI under `src/components/@ui/`, `src/hooks/`, `src/stores/`, or other widely reused modules:

| Check                                            | Action                                                                  |
| ------------------------------------------------ | ----------------------------------------------------------------------- |
| Exported component or hook signature changes     | enumerate all callers before editing                                    |
| New prop is added                                | make it optional unless a coordinated caller migration is intentional   |
| Styling or behavior changes across many surfaces | list representative surfaces and verify each affected state             |
| Caller count is large or unclear                 | recommend split work or an explicit blast-radius review at Checkpoint 1 |
| Tests depend on shared behavior                  | update ripple tests and record them in the validation scope             |

Rules:

- do not modify a shared primitive first and discover callers later
- use grep or repo search to enumerate callers before implementation
- if all callers cannot be updated safely within scope, stop and recommend a split plan

## Final Diff Sanity

Before PR creation or final handoff, check:

| Check                                                     | Action                                     |
| --------------------------------------------------------- | ------------------------------------------ |
| Scope drift detected                                      | stop and report the extra changes          |
| Generated files changed unexpectedly                      | revert or stop, unless explicitly intended |
| Lockfile changed without real dependency intent           | stop and resolve before PR creation        |
| `TODO` or `FIXME` left in critical path                   | fix or report as blocker                   |
| Debug logging or temporary comments left in touched files | remove before PR creation                  |

Before final handoff in local mode, prefer this visual-debug order when screenshots are involved:

1. confirm route/customer/org/auth state
2. confirm the primary changed page and route-specific selector
3. capture the single most relevant screenshot first
4. add a secondary route only if the diff justifies it
