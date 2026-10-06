# Code UI Integration Checklist

Use this file when `/code-ui` is implementing or updating real UI behavior. Apply items only when relevant to the task.

## Implementation Checks

- shared callers checked before changing a shared primitive or exported type
- route wiring checked if the change adds or alters a page
- loader or prefetch wiring checked if the route is data-backed
- sidebar or breadcrumb wiring checked if the feature is navigable
- query keys and invalidation checked for new data fetching or mutation flows
- mocks updated if new API requests were introduced
- store reset added in `tests/setup.ts` if a new store was created
- loading, empty, error, and success states covered if the surface is data-backed
- copy consistency checked if repeated labels or dialog text changed

## Shared Primitive Modifications

When changing a shared `@ui/` component or a widely used hook or type:

1. Enumerate callers first with `rg`.
2. Keep the API backward-compatible where possible.
3. Update all required callers consistently.
4. Plan ripple test updates for affected callers.
5. Grep for the old pattern after edits.

## Route And Integration Checks

When relevant, verify:

- route + lazy module wiring
- `navRoutes` entry
- loader or prefetch wiring
- sidebar wiring
- breadcrumbs
- `ReactQueryKeys`
- invalidation helpers in `queryInvalidation.ts`
- barrel exports
- store reset in `tests/setup.ts`
- shared type updates when component APIs change

When adding a new route, check for path collisions in `Routes.tsx`, including overlapping prefixes, dynamic segments, and optional params.

## UI State Coverage

Where relevant, implement and verify:

- loading state
- empty state
- error state
- success or default state

## Copy Consistency And Adjacent Fixes

When changing repeated labels, dialog copy, or button variants across multiple callers:

- grep for the old copy or pattern after edits
- verify consistent formatting across all instances
- keep incidental fixes small and clearly record them so the orchestrator can surface them as adjacent fixes

## Mock Dispatch Verification

If a new API endpoint is called:

- add a matching entry to `tests/mockAxios.ts`
- make sure the URL matcher matches the generated hook's actual URL shape
- add or reuse a mock response factory

Common failure:

- `"UnSupported request: {method} {url}"` usually means the mock dispatch entry is missing or too narrow
