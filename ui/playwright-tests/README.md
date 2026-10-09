# Playwright - UI Testing Framework for Kumo

## Installation

Run the following commands from the `kumo/ui/` directory.

1. Install required npm packages:
    ```bash
    npm run install:catalog
    npm ci
    ```

2. Install Playwright browsers:
    ```bash
    npx playwright install
    ```
    Edge is optional for local runs. If Microsoft Edge is installed, or if you
    explicitly enable it with `PLAYWRIGHT_ENABLE_EDGE=1`, also run:
    ```bash
    npx playwright install msedge
    ```

## Prerequisites

The main suite runs against a disposable local Kumo compose stack with auth disabled. The normal, headed, and start/seed targets remove the selected stack's Docker volumes before starting it. `--reuse-stack` avoids that reset, but seeding and tests still modify its data. From the repo root:

```bash
make playwright-local-stack
```

The current CI build still uses private source/package and cloud artifact access. A public-only setup remains pending.

## Running Tests

Run `make` and `scripts/...` commands from the repository root. Run all
`npx playwright` commands below from `ui/`.

**Start/seed only:**
```bash
make playwright-local-stack-seed
```

**Seed an existing stack:**
```bash
make playwright-local-stack-seed-existing
```

**Headless (command line):**
```bash
BASE_URL=http://127.0.0.1:18888 AUTH_DISABLED=true npx playwright test
```

**Headed (visible browser):**
```bash
make playwright-local-stack-headed
```

**Interactive UI mode:**
```bash
BASE_URL=http://127.0.0.1:18888 AUTH_DISABLED=true npx playwright test --ui
```

**Specific browser only:**
```bash
scripts/run_playwright_local_stack.sh --reuse-stack --skip-build -- --project chromium
scripts/run_playwright_local_stack.sh --reuse-stack --skip-build -- --project firefox
scripts/run_playwright_local_stack.sh --reuse-stack --skip-build -- --project webkit
PLAYWRIGHT_ENABLE_EDGE=1 scripts/run_playwright_local_stack.sh --reuse-stack --skip-build -- --project edge
```

**Single test file:**
```bash
BASE_URL=http://127.0.0.1:18888 AUTH_DISABLED=true npx playwright test routes.spec.ts
```

## Recording Tests

Use Playwright codegen to record new tests by interacting with the browser:
```bash
npx playwright codegen http://127.0.0.1:18888
```

## View Test Report

```bash
npx playwright show-report
```

## Reference

- [Playwright Docs](https://playwright.dev/docs/intro)
- [CLI Reference](https://playwright.dev/docs/test-cli)
