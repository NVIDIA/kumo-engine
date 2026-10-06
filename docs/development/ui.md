# UI Coding Guide

Conventions and recipes for `ui/`. Paths in examples are relative to `ui/src/`
unless stated otherwise. Commands run from `ui/`. The component library lives
in `packages/kumo-ui-catalog` in this same repository.

## Project Structure

| Directory               | Purpose                                                  |
| ----------------------- | -------------------------------------------------------- |
| `src/components/`       | Feature components, organized by domain                  |
| `src/components/@ui/`   | Application wrappers and shared UI helpers              |
| `src/components/@hoc/`  | HOCs: `WithErrorBoundary`, `WithErrorBoundaryForwardRef` |
| `src/hooks/`            | Custom hooks (data fetching, state, table logic)         |
| `src/stores/`           | Zustand stores (debugStore, userStore, cacheStore)       |
| `src/pages/`            | Page-level components                                    |
| `src/routes/`           | `Routes.tsx` (lazy loading), `Loaders.ts` (prefetch)     |
| `src/services/`         | Axios clients, token service                             |
| `src/utils/`            | Utilities, constants, error handling                     |
| `src/gen/`              | **GENERATED — never edit** (Orval codegen)               |
| `src/utils/PQLParsing/` | Authored parsing code; `grammar/` and synced `pqDocs.json` are generated                      |
| `src/tests/`            | Test infra: mockAxios, testUtil, mockRouter, setup       |
| `src/__tests__/`        | Top-level test files                                     |
| `src/types/`            | TypeScript type definitions                              |
| `src/main.css`          | Tailwind source directives and application styles                   |

## Key File Index

```
entry.ts                     — index.html entry; utils/sanitizeScriptUrlPolicy must stay its FIRST import, then ./main (order enforced by scripts/check-built-index-csp.mjs during npm run build)
utils/sanitizeScriptUrlPolicy.ts — Trusted-Types "sanitizeScriptURL" policy bootstrap (DOMPurify-backed); replaced the inline index.html script so the deploy CSP needs no inline hash (kumo#30121)
utils/reactQueryKeys.ts      — ReactQueryKeys enum (all query cache keys)
utils/queryInvalidation.ts   — Named invalidation helpers per resource
utils/storage.ts             — Typed application storage and session-storage helpers
utils/Constants/enums.ts     — storageKeys enum, inputState enum
utils/Constants/routes.ts    — navRoutes enum (global routes) and getProjectPath (project routes)
utils/errorHandler.ts        — handleErrorFunctional, adjustError, logStackTrace
utils/navigator.ts           — useCustomNavigation hook (ctrl+click support)
utils/timeConverter.ts       — Timestamp formatters (viewer-local for resource metadata, wire/UTC for data-domain) + parseDurationToSeconds (ISO-8601 durations)
utils/jobProgress.ts         — estimateJobProgressPercent (job progress % from ISO-duration elapsed_time over the start/estimated-end window)
hooks/utils/featureFlags.ts  — useFeatureFlags hook (IS_DATABRICKS_DEPLOY, etc.)
services/token_service.ts    — tokenService singleton (auth token management)
services/axios_client.ts     — Base axios instance (interceptors, headers)
services/orval_axios_client.ts — Orval axios with Zod validation
components/@hoc/WithErrorBoundary.tsx   — Error boundary HOC (default export) + WithErrorBoundaryForwardRef named export
routes/Routes.tsx             — createLazyModule, route definitions
routes/Loaders.ts             — Route loader / prefetch functions
tests/testUtil.ts             — createQueryClient, createWrapper, createMockResponse
tests/mockAxios.ts            — Centralized HTTP mock dispatch table
tests/mockRouter.ts           — mockNavigateCall, mockLocationStateValue
tests/testHelpers.ts          — DOM helpers: findAndClickButton, verifyTextExists, etc.
tests/setup.ts                — Global test setup (ResizeObserver, virtualization, etc.)
```

## Convention Cheat Sheet

### Imports & Paths

- **Absolute imports** from `baseUrl: "src"` — never use `../../../`
- **Catalog components** from `@kumo-ai/kumo-ui-catalog` — see the [catalog index](../../.agents/skills/code-ui/references/catalog-components.md)
- **Generated types/hooks** from `gen/v1/` (or `gen/v0/` fallback) — never define manually
- **Tree-shake**: `import sortBy from "lodash/sortBy"` not `import { sortBy } from "lodash"`

### API & Data Fetching

- Use **Orval hooks** from `gen/v1/` first, fall back to `gen/v0/` — not direct axios (exception: `services/`, file uploads)
- Query keys: `ReactQueryKeys.XxxList` enum — never hardcode strings
- Cache invalidation: use named helpers from `utils/queryInvalidation.ts`
- Route loaders: prefetch with `prefetchGetXxx(queryClient, ...)` in `Loaders.ts`
- The training-job warnings route (`POST /v0/training_jobs/{id}/warnings`) resolves only training-job ids (`trainingjob-<hex>`); `useGetQueryWarnings` in `hooks/jobs/training/query.ts` treats any other job-id type (e.g. `gen-traintable-job-*`) as "no id" and skips the request (kumo#30119)
- Train-table detail views pass the parent training-job id (resolved from the train-table job's `parent_job` tag) into `useGetQueryWarnings`, never the page's own gen-traintable id

### Components

- Wrap exports with `WithErrorBoundary(MyComponent)` (or `WithErrorBoundaryForwardRef`)
- Lazy-load pages: `const MyPage = createLazyModule(() => import("pages/MyPage/MyPage"))`
- Use catalog primitives (`Button`, `Dialog`, `Card`, etc.) — don't reimplement
- No `dangerouslySetInnerHTML` — zero instances in codebase

### State Management

- **Zustand** with `devtools(immer(...))` middleware (simple config stores may skip)
- Create **selector hooks**: `const useData = () => useStore((s) => s.data)`
- Separate `StoreState` and `StoreActions` interfaces
- **Discriminated unions** for async state: `{ status: 'idle' | 'loading' | 'success' | 'error', data?, error? }`
- Use `Promise.allSettled()` for parallel fetches in stores

### Navigation & Config

- Routes: `navRoutes.MyRoute` — never hardcode path strings
- Navigation: `useCustomNavigation()` — not raw `useNavigate()`
- Runtime config: `window._env_.MY_VAR` — not `process.env.REACT_APP_*`
- Feature flags: `useFeatureFlags()` — not hardcoded deployment checks

### Error Handling & Logging

- Catch blocks: `handleErrorFunctional(error, showBanner, alertError)` from `utils/errorHandler.ts`
- User-facing errors: `adjustError(description)` for empty descriptions
- Logging: `logStackTrace("context", error)` — never `console.log`

### Storage & Auth

- Local storage: import `storeToLocalStorage` / `getFromLocalStorage` from `@kumo-ai/kumo-ui-catalog` and use keys from `storageKeys`
- Auth: `tokenService` singleton — never read/write tokens directly

### Dates & Timestamps

- Resource metadata (`created_at` / `updated_at`): render in the viewer's local timezone with the zone labeled, via `formatToShortLocalDateString` (app default), `formatToLongLocalDateString`, or `formatToNumericLocalDateString` from `utils/timeConverter.ts`
- Data-domain values that describe the dataset's clock (split boundaries, anchor times, fact time ranges): keep the wire (UTC) rendering via `parseDateTime` / `parseDate`
- Sort date columns on the raw ISO `timestamps` values, never on formatted display strings
- The local formatters' `{ timeZone }` option is test-only; product code omits it so the viewer's own zone is used
- Wire-format timedeltas (e.g. `progress.elapsed_time`) are ISO-8601 duration strings like `"PT27M45.5S"`, not numbers; `Number()` on them yields `NaN` (kumo#29935)
- Job progress bars: use `estimateJobProgressPercent` from `utils/jobProgress.ts`, which parses the ISO-duration `elapsed_time` (via `parseDurationToSeconds`) over the `start_time` / `estimated_end_time` window

### Styling

- **Tailwind utilities** or CSS custom properties — no new classes in `main.css`
- No inline `style={{}}` when Tailwind works
- No hardcoded color/spacing values

### Forms

- Controlled components + `inputState` enum + API-side validation
- No third-party form libs (`react-hook-form`, `formik`, `yup`)
- Debounce API calls triggered by input
- Gate submit/Next buttons on server validation state, not just non-empty input: lift the field's async error into the parent (e.g. `GraphNameInput`'s `setError` prop) and include it in the button's `disabled` gate; the field clears it on every keystroke, so the button re-enables once an edited value validates clean (see `pages/Graphs/NewGraph/NewGraph.tsx`)
- Disable submit buttons while their mutation is in flight (`disabled={... || loading}`); a loading overlay only blocks pointer events, so a still-focused button re-fires on keyboard Enter
- Also guard the submit handler with a `useRef` in-flight flag reset in `finally`, since a queued activation can fire before the disabled re-render commits (see `pages/Graphs/NewGraph/NewGraph.tsx`)
- Dirty wizard forms guard every exit path, not just the Cancel button: block router navigations with `useBlocker` (reusing the same cancel confirmation dialog) and arm an `isDirty`-gated `beforeunload` handler for refresh/tab-close; skip the blocker for navigations the wizard issues on purpose (confirmed discard, post-create redirect) via a ref. Reference: `pages/NewJob/Training/Training.tsx` (kumo#30141)

### Naming

- Components: `PascalCase` (`UserProfileCard.tsx`)
- Hooks: `use*` prefix (`useTableData`)
- Event handlers: `on<Action><Event>` (`onSubmitClick`, `onNameChange`)
- Booleans: `is*`, `has*`, `should*` prefix
- Files: PascalCase for components, `lower_with_underscore` for utils/services

## Exemplar Files

Before writing new code, study these real files that follow all conventions:

| Type            | Exemplar File                                            | Why                                                                  |
| --------------- | -------------------------------------------------------- | -------------------------------------------------------------------- |
| Page            | `pages/Connectors/ConnectorsList/ConnectorsList.tsx`     | Clean page with hooks, error handling, has loader + lazy module      |
| Component       | `components/Connectors/Overview/Overview.tsx`            | Catalog imports (GridRow, PropertyCard), WithErrorBoundary, Tailwind |
| Hook (Orval)    | `hooks/connector/connector.ts`                           | Multiple Orval hooks, ReactQueryKeys, error callbacks, stale time    |
| Store (Zustand) | `stores/cacheStore.ts`                                   | Separate state/actions, selector hooks, localStorage persistence     |
| Component test  | `components/Debug/__tests__/CacheConfiguration.test.tsx` | render(), store mocks, accessible queries, userEvent                 |
| Hook test       | `hooks/snapshot/__tests__/tables.test.ts`                | renderHook + createWrapper, waitFor, success/error paths             |
| Mock endpoint   | `tests/mockAxios.ts` (`get:` handler, lines ~392-430)    | URL-based dispatch for GET endpoints                                 |

## How-To Recipes

### Add a New Page

1. Create `src/pages/MyPage/MyPage.tsx`:

```tsx
import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

const MyPage = () => {
  return <div>...</div>;
};

export default WithErrorBoundary(MyPage);
```

2. Add lazy module in `src/routes/Routes.tsx`:

```tsx
export const MyPage = createLazyModule(() => import("pages/MyPage/MyPage"));
```

3. Add route in the router config:

```tsx
{ path: navRoutes.MyPage, lazy: MyPage.lazy }
```

4. Add `navRoutes.MyPage` to `utils/Constants/routes.ts`

5. Add loader in `src/routes/Loaders.ts` (if data needed):

```tsx
export const myPageLoader = () => {
  prefetchGetMyData(queryClient, undefined, {
    ...getLoaderQueryOptions([ReactQueryKeys.MyData])
  });
  return null;
};
```

6. Add route to loader in router config: `loader: myPageLoader`

### Add a New Component

1. Create `src/components/MyFeature/MyComponent.tsx`:

```tsx
import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import { Button, Card } from "@kumo-ai/kumo-ui-catalog";

interface MyComponentProps {
  title: string;
}

const MyComponent = ({ title }: MyComponentProps) => {
  return (
    <Card>
      <Button>{title}</Button>
    </Card>
  );
};

export default WithErrorBoundary(MyComponent);
```

2. Add to barrel export if directory has `index.ts`

3. Create test: `src/components/MyFeature/__tests__/MyComponent.test.tsx`

### Add a New Hook

1. Create `src/hooks/myFeature/useMyHook.ts`:

```tsx
import { useGetV1MyData } from "gen/v1/api/myFeature/myFeature";
import { ReactQueryKeys } from "utils/reactQueryKeys";

export const useMyHook = (enabled = true) => {
  const { data, isLoading, error } = useGetV1MyData({
    query: {
      queryKey: [ReactQueryKeys.MyData],
      enabled,
    },
  });

  return { data: data?.data, isLoading, error };
};
```

2. Create test: `src/hooks/myFeature/__tests__/useMyHook.test.tsx`:

```tsx
import { renderHook, waitFor } from "@testing-library/react";
import type { QueryClient } from "@tanstack/react-query";
import { createQueryClient, createWrapper } from "tests/testUtil";
import "tests/mockAxios";

describe("useMyHook", () => {
  let queryClient: QueryClient;
  beforeEach(() => { queryClient = createQueryClient(); });
  afterEach(() => { queryClient.clear(); });

  it("fetches data", async () => {
    const { result } = renderHook(() => useMyHook(), {
      wrapper: createWrapper(queryClient),
    });
    await waitFor(() => expect(result.current.data).toBeDefined());
  });
});
```

3. Add mock response in `tests/mockAxios.ts` for the endpoint

### Add a New Zustand Store

1. Create `src/stores/myStore.ts`:

```tsx
import { create } from "zustand";
import { devtools } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";

interface MyStoreState {
  items: Item[];
  isLoading: boolean;
}

interface MyStoreActions {
  fetchItems: () => Promise<void>;
  reset: () => void;
}

const initialState: MyStoreState = { items: [], isLoading: false };

export const useMyStore = create<MyStoreState & MyStoreActions>()(
  devtools(
    immer((set) => ({
      ...initialState,
      fetchItems: async () => {
        set((state) => { state.isLoading = true; });
        const results = await Promise.allSettled([fetchA(), fetchB()]);
        set((state) => {
          state.isLoading = false;
          if (results[0].status === "fulfilled") state.items = results[0].value;
        });
      },
      reset: () => set(initialState),
    })),
    { name: "MyStore" }
  )
);

// Selector hooks
export const useMyItems = () => useMyStore((s) => s.items);
```

2. Add store reset in `tests/setup.ts` `beforeEach`

### Add a Test for an Existing Component

```tsx
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { QueryClient } from "@tanstack/react-query";
import { createQueryClient, createWrapper } from "tests/testUtil";
import { findAndClickButton, verifyTextExists } from "tests/testHelpers";
import "tests/mockAxios";

describe("MyComponent", () => {
  let queryClient: QueryClient;
  beforeEach(() => { queryClient = createQueryClient(); });
  afterEach(() => { queryClient.clear(); });

  const renderComponent = () =>
    render(<MyComponent />, { wrapper: createWrapper(queryClient) });

  it("renders title", async () => {
    renderComponent();
    await verifyTextExists("Expected Title");
  });

  it("handles click", async () => {
    renderComponent();
    await findAndClickButton("Submit");
    await waitFor(() => verifyTextExists("Success"));
  });
});
```

Pages that register a navigation blocker (`useBlocker`) need a data router in tests: build one with `createMemoryRouter` (seed a previous history entry to exercise browser Back) and render `<RouterProvider router={router} />` instead of wrapping in `MemoryRouter`. See `pages/NewJob/Training/__tests__/Training.test.tsx`.

### Add a Mock Endpoint to mockAxios.ts

The centralized mock dispatch table in `tests/mockAxios.ts` routes requests by URL. To add a new endpoint:

1. Find the correct HTTP method handler (`get`, `post`, `put`, `delete`) in `mockAxiosInstance`
2. Add a URL match condition in the dispatch chain:

```tsx
// Simple GET
if (url === "v1/myfeature/") {
  return Promise.resolve(createMockResponse(mockMyFeatureList));
}

// Parameterized GET (extract ID from URL)
if (/^v1\/myfeature\/[^/]+$/.test(url)) {
  const id = url.split("/")[2];
  return Promise.resolve(createMockResponse(mockMyFeatureById(id)));
}

// POST with payload inspection
if (url === "v1/myfeature/") {
  return Promise.resolve(createMockResponse({ id: "new-123", ...payload }));
}
```

3. Place new handlers **before** the catch-all `Promise.reject` at the end of each method

### General Rule for Other Typos

If you discover a spelling issue in any candidate file or folder, fix it in the same commit as its catalog promotion or token sweep. Do not introduce rename-only commits.

## UI Catalog Quick Reference

Component index: [catalog components](../../.agents/skills/code-ui/references/catalog-components.md). Verify exports against `packages/kumo-ui-catalog/src/index.ts`.

**Most-used components:** `Button`, `Card`, `CardContent`, `CardHeader`, `Dialog`, `DialogContent`, `DialogHeader`, `DialogTitle`, `DialogFooter`, `Alert`, `Badge`, `Input`, `TextArea`, `Select`, `Checkbox`, `Switch`, `Tabs`, `TabsList`, `TabsTrigger`, `TabsContent`, `Tooltip`, `TruncatedTooltip`, `DropdownMenu`, `Skeleton`, `DataTable`, `EmptyEntity`, `Spinner`, `toast`

**Charts:** `AreaChart`, `BarChart`, `HistogramChart`, `HeatMapChart`, `ScatterPointLineChart`

**Icons:** Material-style icons — import from `@kumo-ai/kumo-ui-catalog`

**Utility:** `cn()` for class merging

All imports from single entry point:

```tsx
import { Button, Card, Dialog, cn } from "@kumo-ai/kumo-ui-catalog";
```

## Commands

From the repository root, with the UI/catalog dependencies installed:

```bash
npm --prefix ui run type-check
npm --prefix ui run lint-check
npm --prefix ui run test -- --run
```

`npm run lint` and `npm run format` write changes; their check variants are
for verification. `npm run test` uses watch mode unless `--run` is passed.
For catalog work read its existing [README](../../packages/kumo-ui-catalog/README.md)
and [component guide](../../packages/kumo-ui-catalog/CLAUDE.md).
For API and grammar regeneration use the [generator map](generated-artifacts.md).

## Review references

Use the [review conventions](ui-review.md) and their linked code patterns for
UI reviews. Shared catalog changes also follow the
[catalog review procedure](../../packages/kumo-ui-catalog/.agents/skills/review-ui-catalog/SKILL.md).
