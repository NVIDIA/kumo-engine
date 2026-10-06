# Build UI Screenshot Operator Guide

Use this guide when build-ui screenshots are missing, wrong, or do not prove the ticket requirement.

## First Artifacts To Inspect

1. `screenshot-plan-resolution.json`: explains which catalog recipes matched, which recipe was selected, and whether the existing plan was kept or rewritten.
2. `results.json`: shows route success, final URL, matched selector, missing coverage, skipped scenarios, and failure type.
3. `screenshot-debug.md`: summarizes auth state, route state, DOM snippets, browser/API events, and missing coverage.
4. `ci-screenshot.log`: use this for Playwright timing, selector, route mismatch, and auth-bootstrap details.
5. `failed/*.html` and `failed/*.png`: use these when the page did not reach the expected route or ready selector.

## Failure Triage

| Symptom                              | Likely cause                                       | Action                                                                                |
| ------------------------------------ | -------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Login/auth screen                    | auth bootstrap failure or wrong screenshot account | Check auth diagnostics and screenshot tenant access before changing routes            |
| Correct route, wrong page state      | missing fixture/setup action                       | Add or select a scenario that seeds/opens the required state                          |
| Correct page, changed element absent | weak or broad assertion                            | Assert on the exact changed selector and add a stable `data-testid` if needed         |
| List page captured for detail change | route-map/catalog mismatch                         | Fix the route-map/catalog recipe, do not accept adjacent list evidence                |
| Screenshot exists but workflow fails | missing `visualCoverage` coverage                  | Add `covers` to the setup/action/assertion that proves the requirement                |
| Upload failed after capture          | GitHub write/API issue                             | Keep artifacts, retry publication with the local helper after `gh auth status` passes |

## Override Flags

Use explicit overrides when the automatic planner picks the wrong surface:

```bash
/build-ui 12345 --mode standard --screenshot-recipe new_prediction
/build-ui 12345 --mode standard --screenshot-route /newjob/prediction/
/build-ui 12345 --mode standard --screenshot-coverage prediction_settings_visible
```

Override rules:

- Prefer `--screenshot-recipe` when a known catalog recipe exists.
- Use `--screenshot-coverage` when a broad recipe has multiple visual requirements and the ticket needs one specific state.
- Use `--screenshot-route` only to disambiguate between valid catalog-backed routes.
- Do not use overrides to bypass missing selectors or missing assertions; fix the recipe or source selectors.

## Local Reproduction

When the UI is running locally and Direct Connect password-mode credentials are available through `SCREENSHOT_AUTH_EMAIL` / `SCREENSHOT_AUTH_PASSWORD`, use the shared local helper from the repository root:

```bash
npm --prefix ui run screenshots:build-ui -- \
  --issue 12345 \
  --plan /tmp/build-ui-artifacts/12345/screenshot-plan.json \
  --base-url http://localhost:3000
```

Prefer the environment variables: a flag value is visible in this wrapper's own argv (e.g. via `ps`) while it runs. Credentials may also be supplied as explicit flags — `--auth-email <email> --auth-password <pw>` — when argv exposure on your machine is not a concern:

```bash
npm --prefix ui run screenshots:build-ui -- \
  --issue 12345 \
  --plan /tmp/build-ui-artifacts/12345/screenshot-plan.json \
  --base-url http://localhost:3000 \
  --auth-email "$SCREENSHOT_AUTH_EMAIL" \
  --auth-password "$SCREENSHOT_AUTH_PASSWORD"
```

Flag credentials are forwarded to `ci-screenshot.mjs` through the environment (`SCREENSHOT_AUTH_EMAIL` / `SCREENSHOT_AUTH_PASSWORD`), so they never reach the child capture process's command line. They remain visible in the wrapper's own argv (e.g. via `ps`) for its lifetime, however — export the env vars instead of passing `--auth-email` / `--auth-password` to avoid argv exposure entirely. Providing credentials also suppresses the localhost auth-disabled default so the Direct Connect password login path is used.

### Auth-disabled local backends

A local dev server normally runs with auth disabled (the auth client config endpoint reports `auth_disabled: true`), so no Direct Connect credentials are required. The helper auto-enables auth-disabled mode for `localhost` / `127.0.0.1` base URLs, and you can force it for any host with `--auth-disabled`:

```bash
npm --prefix ui run screenshots:build-ui -- \
  --issue 12345 \
  --plan /tmp/build-ui-artifacts/12345/screenshot-plan.json \
  --base-url http://localhost:3000 \
  --auth-disabled
```

In this mode the capture script injects the same `AUTH_DISABLED` sentinel session the app itself uses. The catalog recipe, plan resolution, scenarios, and assertions are identical to CI — only the auth bootstrap differs, so local and CI prove the same surfaces. `SCREENSHOT_AUTH_DISABLED=1` is an equivalent env toggle for `ci-screenshot.mjs`.

Real credentials always take precedence: if Direct Connect screenshot credentials are present, the browser password-login path is used even on localhost. And if auth-disabled mode is requested (including the localhost default) but the backend actually enforces auth (`auth_disabled: false`), the run fails fast with a message asking for credentials rather than capturing a login/empty screen.

Publication writes images to a remote branch and updates the PR/comment; `--pr-number` also enables it. Follow the authorization and visibility rules in [the PR workflow](pr-workflow.md#shared-screenshot-publishing-flow). Set `OWNER/REPO` to the intended destination.

To publish the same evidence to an existing PR:

```bash
npm --prefix ui run screenshots:build-ui -- \
  --issue 12345 \
  --plan /tmp/build-ui-artifacts/12345/screenshot-plan.json \
  --repo OWNER/REPO \
  --pr-number 67890 \
  --publish
```

If capture already ran and `/tmp/build-ui-screenshots/{issue}/results.json` exists:

```bash
npm --prefix ui run screenshots:build-ui -- \
  --issue 12345 \
  --skip-capture \
  --repo OWNER/REPO \
  --pr-number 67890 \
  --publish
```

## Catalog Maintenance Rules

- Add recipes for reusable workflows and common app states, not one-off ticket screenshots.
- Prefer deterministic seeded state over tenant-specific live-data interactions.
- Keep recipe assertions tied to exact changed surfaces, not shell containers or generic headings.
- Every `visualCoverage` ID must be referenced by a concrete setup action or assertion with `covers`.
- When routes or selectors move, update `ui/src/screenshot/catalog.json`, `ui/src/screenshot/route-map.json`, and the corresponding source selector in the same PR.
