# Build UI Completion Criteria

Use this file when deciding whether planning, implementation, validation, or PR preparation is actually complete.

## Planning Complete

Planning is complete only if:

- acceptance criteria are explicit or clearly synthesized
- impacted files and exemplars are identified
- API and catalog dependencies are called out
- blockers and assumptions are listed
- a size estimate is provided

## Implementation Complete

Implementation is complete only if:

- each accepted AC maps to code changes
- no critical placeholder logic remains
- relevant wiring checks are complete
- required UI states are implemented where applicable

## Validation Complete

Validation is complete only if:

- `npm run type-check` passes
- `npm run lint-check` passes
- `npm run format-check` passes
- `npm run build` passes
- targeted tests for the change pass

If broader or coverage-mode tests were skipped, say so explicitly.

## CI-Ready

The change may be described as CI-ready only if:

- static checks pass
- relevant tests pass
- skipped verification is disclosed

## Visual-Verified

The change may be described as visually verified only if:

- the intended route or surface was opened
- console and network checks were reviewed (interactive browser) or Playwright screenshots were captured (CI mode)
- the main acceptance path was exercised
- any reduced-scope conditions were disclosed

In CI mode, Playwright-based screenshots satisfy visual verification when:

- the target route loaded successfully without redirecting to `/login`
- the screenshot reflects the expected route rather than a generic shell or error state
- if route-specific ready selectors were needed, they were matched before capture
- screenshots were uploaded as workflow artifacts
- the scope limitation (no interactive console/network inspection) is disclosed

## PR-Ready

The change is PR-ready only if:

- branch state is intentional
- release-notes label choice is resolved
- PR summary is prepared against the repo template
- notable caveats and skipped checks are included when relevant
- screenshot rendering was verified if screenshots were attached
- any catalog gaps were either resolved or explicitly accepted as exceptions
