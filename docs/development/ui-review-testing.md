# Testing Infrastructure Patterns

> Loaded when the PR diff touches `ui/src/tests/` or `__tests__/` files. These patterns enforce the centralized testing conventions.

---

### Pattern 25: Centralized Axios Mocking

**What triggers it:**
Inline axios mocking per test file instead of using the shared dispatch table.

**Example:**
```typescript
// Flagged — custom per-file mock:
vi.mock("services/axios_client", () => ({
  default: { get: vi.fn().mockResolvedValue({ data: [] }) }
}));

// Should add to the centralized mock:
// In tests/mockAxios.ts, add a branch to the URL dispatch:
if (url === "v1/my-new-endpoint") {
  return Promise.resolve({ data: mockMyEndpointResponse });
}

// Tests then just import and use:
import "tests/mockAxios"; // auto-mocks all axios calls
```

**Why:** Centralized mocking ensures consistent behavior across tests, makes it easy to find all mocked endpoints, and prevents tests from having stale or inconsistent mock responses.

**Reference:** `ui/src/tests/mockAxios.ts` — URL dispatch table for all HTTP mocks.

---

### Pattern 26: QueryClient in Tests

**What triggers it:**
Creating raw `QueryClient` instances in tests.

**Example:**
```typescript
// Flagged — raw QueryClient retries on failure (flaky):
const queryClient = new QueryClient();

// Should use helper (retry disabled):
import { createQueryClient } from "tests/testUtil";
const queryClient = createQueryClient();
```

**Reference:** `ui/src/tests/testUtil.ts` — `createQueryClient()` sets `retry: false` and appropriate defaults.

---

### Pattern 27: Hook Test Wrapper

**What triggers it:**
`renderHook` without a QueryClient wrapper for hooks that use React Query.

**Example:**
```typescript
// Flagged — missing wrapper:
const { result } = renderHook(() => useMyDataHook());

// Should wrap with QueryClientProvider:
import { createQueryClient, createWrapper } from "tests/testUtil";
const queryClient = createQueryClient();
const { result } = renderHook(() => useMyDataHook(), {
  wrapper: createWrapper(queryClient),
});
```

**Reference:** `ui/src/tests/testUtil.ts` — `createWrapper()` provides `QueryClientProvider`.

---

### Pattern 28: Zustand State in Tests

**What triggers it:**
Mocking Zustand stores with `vi.mock()` instead of injecting state directly.

**Example:**
```typescript
// Flagged — vi.mock prevents real store logic from running:
vi.mock("stores/userStore", () => ({
  useUserStore: () => ({ isLoggedIn: true, name: "Test" }),
}));

// Should inject state directly:
import { useUserStore } from "stores/userStore";

beforeEach(() => {
  useUserStore.setState({
    userDetails: { ...useUserStore.getInitialState().userDetails, isLoggedIn: true, name: "Test" },
  });
});
```

**Why:** Direct state injection exercises the real store selectors and middleware. `vi.mock` bypasses the entire store implementation and can mask bugs.

**Reference:** `ui/src/tests/setup.ts` — sets `useDebugStore` state in `beforeEach`.

---

### Pattern 29: Router Mocking

**What triggers it:**
Inline `vi.mock("react-router", ...)` per test file.

**Example:**
```typescript
// Flagged — inline mock:
vi.mock("react-router", () => ({
  useNavigate: () => vi.fn(),
  useLocation: () => ({ pathname: "/test", state: null }),
}));

// Should use shared helpers:
import { mockNavigateCall, mockLocationStateValue, mockUseNavigate } from "tests/mockRouter";

// Set up navigation before rendering, then assert after the interaction:
mockNavigateCall();
// ...render the component and trigger its navigation...
expect(mockUseNavigate).toHaveBeenCalledWith(navRoutes.Home);

// To mock location state:
mockLocationStateValue({ fromPage: "dashboard" });
```

**Reference:** `ui/src/tests/mockRouter.ts` — provides `mockNavigateCall()` and `mockLocationStateValue()`.

---

### Pattern 30: Reuse Test Helpers

**What triggers it:**
Writing new test utility functions that duplicate existing shared helpers.

**Available helpers in `tests/testHelpers.ts`:**
```typescript
// Don't rewrite these — import them:
import {
  verifyTextExists,       // Assert text is in the document
  findAndClickButton,     // Find button by text and click
  findAndClickButtonByRole, // Find button by role+name and click
  verifySkeletonExists,   // Assert loading skeletons render
  sharedVerifyJobsApiCall,// Assert jobs API was called correctly
  getAlert,               // Get alert dialog element
  getDialog,              // Get dialog element
} from "tests/testHelpers";
```

**Rule:** Before writing a new test helper, check `testHelpers.ts`. If a similar helper exists, use it or extend it rather than duplicating.

**Reference:** `ui/src/tests/testHelpers.ts`, `ui/src/tests/testUtil.ts`.

---

### Pattern 31: Global Setup Mocks

**What triggers it:**
Per-test overrides of mocks that `setup.ts` already provides globally.

**What `tests/setup.ts` mocks globally:**
- `@tanstack/react-virtual` — renders all items (no virtualization in tests)
- `ResizeObserver` — no-op implementation
- `getBoundingClientRect` — returns zero rect
- `scrollTo`, `scrollIntoView` — no-ops
- `useDebugStore` state — reset in `beforeEach`
- `cleanup()` + `vi.clearAllMocks()` — in `afterEach`

```typescript
// Flagged — overriding global mock without justification:
vi.mock("@tanstack/react-virtual", () => ({
  useVirtualizer: () => ({ /* custom impl */ }),
}));

// Only override with a comment explaining why:
// Need custom virtualizer behavior to test scroll-based pagination
vi.mock("@tanstack/react-virtual", () => ({ ... }));
```

**Reference:** `ui/src/tests/setup.ts`.

---

### Pattern 32: data-testid on @ui Primitives

**What triggers it:**
Removing `data-testid` attributes from shared `@ui` components, or new `@ui` components missing them.

**Example:**
```typescript
// Flagged — removing existing data-testid:
-  <div data-testid="alert-dialog" className="alert">
+  <div className="alert">

// New @ui components should include data-testid:
const ValidationCard = ({ children }: Props) => (
  <Card data-testid="validation-card">
    {children}
  </Card>
);
```

**Why:** `data-testid` on `@ui` primitives is an intentional convention for testable UI atoms. Tests across the codebase depend on these selectors for shared primitives.

**Reference:** `packages/kumo-ui-catalog/src/components/ui/custom-alert.tsx`, `packages/kumo-ui-catalog/src/components/ui/confirmation-dialog.tsx`, `packages/kumo-ui-catalog/src/components/blocks/card/validation-card.tsx`.

---

## Source Material

| Source | Area |
|--------|------|
| `ui/src/tests/mockAxios.ts` | Centralized HTTP mock dispatch table |
| `ui/src/tests/mockRouter.ts` | Router mock helpers |
| `ui/src/tests/testUtil.ts` | createQueryClient, createWrapper, createMockResponse |
| `ui/src/tests/testHelpers.ts` | Shared test helpers (findAndClickButton, etc.) |
| `ui/src/tests/setup.ts` | Global test setup, virtualization mocks |
