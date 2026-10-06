# Code UI Testing Playbook

Use this file when `/code-ui` needs to design, run, and interpret tests for a UI change.

## Testing Checklist

- targeted tests derived from the accepted AC or explicit task description
- centralized mocks updated when new API requests were introduced
- existing test helpers reused before writing custom query or interaction code
- failure classification recorded when validation fails

## Expectations

- bug fix: add a regression test where practical
- new component, hook, or page: cover relevant render and interaction paths
- new API usage: update centralized mocks

## Preferred Execution Order

1. changed test-name filters
2. changed test files
3. related feature-area tests
4. broader suite only if it adds signal
5. coverage-mode run only when the environment supports it and the extra confidence matters

## Useful Commands

```bash
npx vitest run src/path/to/test.tsx -t "changed behavior name"
npx vitest run src/path/to/__tests__/Specific.test.tsx
npx vitest run src/path/to/feature/
npx vitest run --coverage --reporter=verbose --silent
```

## Failure Classification

- `introduced-by-change`
- `pre-existing`
- `environmental`
- `flaky-or-unclear`

## Common Diagnosis

- `"UnSupported request: {method} {url}"`: missing `mockAxios` dispatch entry
- `"Cannot find module"` or `"is not exported from"`: wrong generated hook path or export
- `"Unable to find role"` or `"Unable to find text"`: assertion no longer matches the rendered UI

## Retry Budget

- targeted test fix loops: max 3
- related suite retries: max 2
- coverage-mode retries: max 1
