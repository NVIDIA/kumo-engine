# Kumo UI/UX (ReactJS)

## Quick Start

If you want to run the default configuration using docker (recommended), run
the following from the root of the kumo repository.

```bash
./scripts/run_ui_devserver.sh <ui_port_number> <api_port_number> <label>
```

Args:

-   `ui_port_number`: port where the UI will be served
-   `api_port_number`: port where the internal REST API should already be running
-   `label`: label of the docker image, can be anything

The UI port is also the browser-facing internal API origin. The launcher
proxies `/v0/*` and `/v1/*` to `api_port_number`; browser code must not call the
API port directly. Use `localhost` in the browser, especially for OIDC.

## Advanced Installation Guide

1. Install [NodeJs](https://nodejs.org/en/):

    ```bash
    curl -sL https://deb.nodesource.com/setup_22.x -o /tmp/nodesource_setup.sh
    sudo bash /tmp/nodesource_setup.sh
    sudo apt install nodejs
    ```

2. From the monorepo's `ui/` directory, build the in-tree catalog and install
   the UI dependencies:

    ```bash
    npm run install:catalog
    npm ci
    ```

    `@kumo-ai/kumo-ui-catalog` is a local file dependency on
    `packages/kumo-ui-catalog`. No separate checkout or GitHub credential is
    needed. Rebuild the catalog after changing it, or use hot reload below.

## Configuring the API origin

Local development uses the relative API base `/` and the Vite proxy. To run the
UI directly, point the server-side proxy at an already-running private REST
listener:

```bash
API_PROXY_TARGET=http://127.0.0.1:12001 \
REACT_APP_PORT=12000 \
npm run local
```

If no target is supplied, `npm run local` defaults to the standard private REST
listener at `http://127.0.0.1:10001`. The browser still opens
`http://localhost:3000` and calls relative `/v0` and `/v1` paths.

### Local UI with a remote devserver

After forwarding the remote devserver's private REST port to local port `10001`,
run:

```bash
npm run local:remote
```

This serves the UI at `http://localhost:3001` and proxies its API requests to
`http://127.0.0.1:10001`. No generated files or shell aliases need to be
modified. To use another reachable REST address or UI port, override either
default:

```bash
API_PROXY_TARGET=http://remote-dev-host:10001 \
REACT_APP_PORT=3002 \
npm run local:remote
```

Deployed builds omit `UI_LOCAL`; their gateway and `API_BASE_URL` deployment
configuration remain responsible for routing. Run the following command after
changing the runtime environment template:

```bash
npm run update-env
```

## Run the app in the development mode.

```bash
npm run local
```

After successful build, it invokes the browser and opens the app here [http://localhost:3000](http://localhost:3000)

The page will reload when any change is made.

## Developing with Local UI Catalog

The catalog lives in `packages/kumo-ui-catalog`. After installing dependencies
as above, hot reload can consume its source directly instead of its built output.

### Running with Local Catalog

The examples below opt out of cgroup fencing for one devserver. See
[devserver resource settings](../docs/devserver.md#quick-start) for fenced runs.

Using Docker:
```bash
# From kumo root
UI_CATALOG_DEV=1 UNFENCED=1 NUMS=1 make devserver-local

```

Using local npm:
```bash
# From ui folder
npm run local:catalog
```

The tsconfig is automatically configured:
- `npm run local:catalog` / `start:catalog` → sets up local catalog paths
- `npm run local` / `start` → resets to use npm package

After switching between modes, restart your TypeScript server for editor intellisense:
- Open any `.ts/.tsx` file
- Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
- Select "TypeScript: Restart TS Server"

### Additional Options

**Custom catalog path:**

```bash
# Docker
UI_CATALOG_DEV=1 UI_CATALOG_PATH=/path/to/catalog UNFENCED=1 NUMS=1 make devserver-local

# Local npm
UI_CATALOG_PATH=/path/to/catalog npm run local:catalog
```

**Using Docker script directly:**

```bash
./scripts/run_ui_devserver.sh <ui_port> <api_port> <label> [ui_catalog_path]

# Example:
./scripts/run_ui_devserver.sh 10000 10001 devserver packages/kumo-ui-catalog
```

## Build the app for production to the `build` folder.

```bash
npm run build
```

Above command bundles React in production mode and optimizes the build for the best performance.

The build also runs `scripts/check-built-index-csp.mjs` against the output.
It fails if `build/index.html` contains any inline `<script>`, or if the Trusted-Types `sanitizeScriptURL` bootstrap (`src/utils/sanitizeScriptUrlPolicy.ts`, the first import of `src/entry.ts`) does not evaluate before the app code in the entry chunk.
The deployment CSP `script-src` carries no inline hash, so the shipped `index.html` must stay free of inline scripts (kumo-ai/kumo#30121).

## Enable ESLint linting on VSCode editor

1. Install [ESLint](https://marketplace.visualstudio.com/items?itemName=dbaeumer.vscode-eslint)
   extension from VSCode Extensions

2. In VSCode settings.json, paste this

```
"editor.codeActionsOnSave": {
  "source.fixAll.eslint": "always"
},
"eslint.validate": ["javascript", "typescript", "typescriptreact"],
"eslint.useFlatConfig": true,
"eslint.workingDirectories": [{ "mode": "auto" }],
```

3. Close VSCode and reopen VSCode

4. In output terminal of VSCode, make sure ESLint is running by selecting the ESLint
   from dropdown and see same message as below

```
[Info  - 3:16:48 AM] ESLint server is starting.
[Info  - 3:16:48 AM] ESLint server running in node v20.16.0
[Info  - 3:16:48 AM] ESLint server is running.
[Info  - 3:16:50 AM] ESLint library loaded from:
/home/user/workspace/kumo/ui/node_modules/eslint/lib/api.js
```

5. Test the ESLint linting on VSCode editor by writing a unused variable in some tsx file
   and see if lint error shows up on that line and ESLint suggest to fix it.

## Enable ESLint linting and Prettier formatting on every commit to branch

1. Use the Node version declared in the repository `.mise.toml`.

2. Run these commands to install neccessary packages and pre-commit

```
git config --unset-all core.hooksPath
pip install pre-commit
pre-commit install
```

3. On commit to branch, check if below line in present in pre-commit output

```
ui-lint-staged.......................Passed
```

## UI Testing

Run `npm run test -- --run` for unit tests. See [UI commands](../docs/development/ui.md#commands) for the other focused checks.

## Coding Guidelines

Kumo UI coding guidelines : [Readme](docs/Coding_Guidelines.md)

Shared conventions and recipes: [UI development](../docs/development/ui.md).

## Agent Skills

Coding agents (Claude Code, Codex) have specialized skills for working with the UI codebase. These skills enforce Kumo conventions automatically.

### `/build-ui` — Ticket-to-PR Automation

Orchestrates the full UI development lifecycle from a GitHub issue to a pull request.

```bash
/build-ui 27326
/build-ui https://github.com/kumo-ai/kumo/issues/27326
```

**Workflow phases:**

```
Phase 0:  Pre-flight        — gh auth, worktree, dev server, browser checks
Phase 1:  Ticket Analysis    — parse issue + comments for AC, Figma URLs, API deps
Phase 2:  Codebase Recon     — find exemplars, check gen/ hooks, wiring points
Phase 3:  Design Reference   — Figma extraction + token mapping (optional)
  → Checkpoint 1: User approves implementation plan
Phase 0b: Branch Setup       — create branch, install in-tree catalog and UI dependencies
Phase 4:  Implementation     — /code-ui with full context, verify wiring
Phase 5:  Static Analysis    — type-check, lint-check, format-check, build
Phase 6:  Self-Review        — review-ui patterns against diff (optional)
Phase 7:  Unit Testing       — derive tests from AC, run targeted + broad
Phase 8:  Visual Testing     — interactive browser verification locally, or Playwright screenshot capture in headless CI when available (optional)
  → Checkpoint 2: User approves PR
Phase 9:  PR Creation        — rebase, push, release-notes label, gh pr create
```

**Options:**

| Flag | Description |
|------|-------------|
| `--mode light\|standard\|full` | Workflow depth (default: `standard`). `light` for trivial fixes, `full` for design-heavy work. |
| `--dry-run` | Planning phases only, no code changes |
| `--skip-figma` | Skip Figma design extraction |
| `--skip-browser` | Skip browser-based visual verification |
| `--no-review` | Skip self-review phase |
| `--base-branch <branch>` | Base branch (default: `master`) |
| `--resume <stage>` | Resume from a named stage (e.g. `implementation`, `validation`, `tests`, `pr`) |

**Examples:**

```bash
/build-ui 27326                                    # Full workflow
/build-ui 27326 --dry-run                          # Plan only
/build-ui 27326 --skip-figma --skip-browser        # No browser or Figma work
/build-ui 27326 --resume tests                         # Resume from testing
```

**Prerequisites (local):**
- `gh` CLI authenticated
- Dev server at `localhost:3000` (for visual testing)
- Chrome DevTools MCP configured in `ui/.mcp.json` (optional)
- Figma MCP (optional)

#### GitHub Comment Trigger (Headless)

`/build-ui` can also be triggered from a GitHub issue comment. The workflow runs on a GitHub Actions runner with no human in the loop.

```
/build-ui 12345                        # Headless run (light mode)
/build-ui 12345 --mode standard        # Override mode
/build-ui 12345 --dry-run              # Plan only, no code changes
```

In headless mode:
- Figma extraction uses the workflow-owned REST path when configured; `--skip-figma` disables it
- Default mode is `light` (override with `--mode standard` or `--mode full`)
- `light` mode does not require screenshots
- `standard` and `full` require workflow-owned screenshot capture in CI
- Visual testing runs through the Playwright CI screenshot fallback when `CI_SCREENSHOT_SCRIPT` is configured
- The agent writes `screenshot-plan.json`; the GitHub Actions workflow runs `ui/scripts/ci-screenshot.mjs`
- The catalog is built from `packages/kumo-ui-catalog` in the same checkout
- Checkpoints are auto-approved unless hard blockers are detected
- The PR is the human approval gate
- Only users with write access can trigger
- The workflow uploads machine-readable headless artifacts such as `preflight.json`, `run-state.json`, `verification-status.json`, `handoff.json`, and `artifact-manifest.json`
- The completion/failure comments include `Screenshot attempt status`

**Screenshot expectations in headless mode:**

- `/build-ui 12345` may finish without screenshots on small fixes because the default mode is `light`
- `/build-ui 12345 --mode standard` must produce a screenshot plan and the workflow will run screenshots
- `/build-ui 12345 --mode full` must also produce a screenshot plan and the workflow will run screenshots

When screenshots succeed, expect:

- a `## Screenshots` comment on the issue or PR
- an uploaded artifact bundle named `build-ui-screenshots-<issue-number>`
- screenshot files and `results.json` under `/tmp/build-ui-screenshots/<issue-number>/` in the workflow run

When screenshots do not succeed, inspect:

- `screenshot-plan.json` in `build-ui-headless-artifacts-<issue-number>`
- `verification-status.json` in `build-ui-headless-artifacts-<issue-number>`
- `blocker.json` when present
- workflow job summary entries for `Build UI Artifacts` and `Screenshot attempt status`

In `standard` and `full`, a missing screenshot plan or a failed Playwright capture is treated as a workflow blocker rather than a best-effort warning.

See `.agents/skills/build-ui/references/headless-mode.md` for full details. The workflow is defined in `.github/workflows/build-ui-agent.yml`.

### `/code-ui` — Scaffold & Implement UI Features

Use this skill when creating new components, pages, hooks, stores, or tests.

```bash
/code-ui add a new page for model comparison
/code-ui add store for notification preferences
/code-ui add test for useConnectorData hook
```

**What it does:**
- Loads compressed conventions from `docs/development/ui.md` automatically
- Checks `gen/v1/` for existing Orval types/hooks before writing manual API calls
- Checks `@kumo-ai/kumo-ui-catalog` for existing primitives before creating new ones
- Follows recipes for each artifact type (page, component, hook, store, test)
- Wires up integration points (routes, barrel exports, query keys, mock dispatch)
- Runs `tsc --noEmit`, lint, and tests to validate

**Key references loaded:**
- `docs/development/ui.md` — conventions, file index, how-to recipes (always loaded)
- `.agents/skills/code-ui/references/catalog-components.md` — full catalog component list with props
- `docs/development/ui-review-*.md` — loaded conditionally based on what's being built

### `/review-ui` — Review PRs Against UI Standards

Use this skill to review PRs that touch `ui/src/` files.

```bash
/review-ui 12345              # Posts review comments to the PR on GitHub
/review-ui 12345 --dry-run    # Prints findings to stdout only
/review-ui 12345 --force      # Posts even if a previous review exists
```

**What it does:**
- Fetches the PR diff and filters to UI-relevant paths
- Runs automated pre-scan (`diff-scan.sh`) for mechanical violations
- LLM reviews remaining diff against 30+ rules (HIGH and MEDIUM severity)
- Posts inline review comments with fix suggestions on GitHub
- Respects `// review-ui-ignore: <rule-id>` inline suppression comments
- Skips MEDIUM rules on draft PRs

**Rule categories:** Generated code, catalog components, query keys, storage access, error boundaries, route lazy loading, feature flags, console statements, navigation paths, runtime config, Zustand stores, TypeScript, testing infrastructure, React patterns, accessibility, and more.

Full rule list: `docs/development/ui-review.md`

### Agent Documentation

| File | Purpose |
|------|---------|
| `ui/CLAUDE.md` | Agent entry point; shared conventions live in `docs/development/ui.md` |
| `.agents/skills/build-ui/SKILL.md` | `/build-ui` skill definition (ticket-to-PR workflow) |
| `.agents/skills/build-ui/references/headless-mode.md` | Headless mode behavior for GitHub comment trigger |
| `.agents/skills/code-ui/SKILL.md` | `/code-ui` skill definition |
| `.agents/skills/review-ui/SKILL.md` | `/review-ui` skill definition (30+ rules) |
| `.agents/skills/code-ui/references/catalog-components.md` | Auto-generated catalog component reference |
| `.github/workflows/build-ui-agent.yml` | GitHub Actions workflow for `/build-ui` comment trigger |
| `packages/kumo-ui-catalog/CLAUDE.md` | Agent guide for the catalog package (patterns, stories, visual tests) |
| `ui/docs/Coding_Guidelines.md` | Human-readable coding guidelines |

### Keeping Docs Fresh

Run the freshness checker to detect when agent docs have drifted from source files:

```bash
bash .agents/skills/code-ui/scripts/check-docs-freshness.sh
```

This compares modification times of docs vs their source-of-truth files (routes, utils, test infra, catalog package) and flags any that may need updating.

Regenerate the catalog component reference after the catalog package is updated:

```bash
bash .agents/skills/code-ui/scripts/generate-catalog-docs.sh
```

## Auto types/schemas/hooks code generating using Orval

Docs - [orval](https://orval.dev/overview)

Remove directory `v1` and `v0` from `ui/src/gen/` before running following command

```
 sudo npx orval
```

## Debugging UI state

Install the [Redux Devtools Chrome extension](https://chromewebstore.google.com/detail/redux-devtools/lmhkpmbekcpmknklioeibfkpmmfibljd) to debug zustand stores.
