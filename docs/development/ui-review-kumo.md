# Kumo-Specific Patterns

> Loaded for all UI reviews. These patterns cover Kumo platform conventions, runtime config, navigation, state management, and code quality.

---

### Pattern 1: Use Generated Code (Orval)

**What triggers it:**
Manually defining types, interfaces, or API call wrappers that Orval already generates in `ui/src/gen/`.

**Example:**
```typescript
// Flagged — manually redefining a generated type:
interface PQueryResponse {
  name: string | null;
  query_string: string;
  graph: KumoapiGraphGraphDefinition;
  desc?: string;
}

// Should use generated type:
import { PQueryResponse } from 'gen/v1/types';
```

**Resolution:** Run `generate-ui-client-code` or comment `/trigger-ui-codegen` on the PR. Generated code lives in `ui/src/gen/v0/` (legacy) and `ui/src/gen/v1/` (current). Orval config: `ui/orval.config.ts`.

---

### Pattern 2: Use Catalog Components

**What triggers it:**
Custom-building UI primitives that already exist in `@kumo-ai/kumo-ui-catalog`.

**Example:**
```typescript
// Flagged — custom dialog:
const CustomModal = ({ open, onClose, children }) => (
  <div className={open ? 'modal-overlay' : 'hidden'}>
    <div className="modal-content">{children}</div>
  </div>
);

// Should use catalog:
import { Dialog, DialogContent, DialogHeader } from "@kumo-ai/kumo-ui-catalog";
```

**Available catalog components** (imported in 198+ files across the codebase):
Button, Card, CardHeader, CardContent, Dialog, DialogContent, Alert, AlertDescription, Tooltip, Select, Skeleton, Tabs, HoverCard, Symbol, GridRow, GridCol, and more.

**Reference:** See usage in `ui/src/components/Validation/Validation.tsx` and `ui/src/components/Validation/ValidationStatus.tsx`.

---

### Pattern 3: ReactQueryKeys Enum

**What triggers it:**
Hardcoded query key strings instead of using the centralized enum.

**Example:**
```typescript
// Flagged — hardcoded key:
const { data } = useQuery({ queryKey: ['admin-info'], queryFn: fetchAdmin });

// Should use enum:
import { ReactQueryKeys } from "utils/reactQueryKeys";
const { data } = useGetAdminInfo(
  getQueryOptions({ queryKey: [ReactQueryKeys.AdminInfo] })
);
```

**Reference:** `ui/src/utils/reactQueryKeys.ts` defines 84 query keys. `ui/src/hooks/misc/queryOptions.ts` provides centralized query option configuration with per-key staleTime/cacheTime settings.

---

### Pattern 4: Query Invalidation Helpers

**What triggers it:**
Manual `queryClient.invalidateQueries()` calls instead of using centralized helpers.

**Example:**
```typescript
// Flagged — manual invalidation:
queryClient.invalidateQueries({ queryKey: ['jobs'] });

// Should use helper:
import { invalidateJobsList } from "utils/queryInvalidation";
invalidateJobsList(queryClient);
```

**Reference:** `ui/src/utils/queryInvalidation.ts` provides typed invalidation functions that ensure consistent cache management.

---

### Pattern 5: Storage Helpers and Keys

**What triggers it:**
Direct `localStorage` or `sessionStorage` calls bypassing the abstraction layer.

**Example:**
```typescript
// Flagged — direct access:
localStorage.setItem('userDetails', JSON.stringify(user));
const token = localStorage.getItem('accessToken');

// Should use helpers + enum:
import { storeToLocalStorage, getFromLocalStorage } from "@kumo-ai/kumo-ui-catalog";
import { storageKeys } from "utils/Constants/enums";

storeToLocalStorage(storageKeys.userDetails, user);
const projectId = getFromLocalStorage<string>(storageKeys.selectedProject);
```

**Why:** The storage helpers handle JSON serialization, optional encoding for sensitive data, and `clearLocalStorage()` preserves `navPath` and `DEBUG`.

**Reference:** `packages/kumo-ui-catalog/src/utils/storage.ts`, `ui/src/utils/storage.ts`, `ui/src/utils/Constants/enums.ts` (`storageKeys` enum).

---

### Pattern 6: Use Orval Hooks, Not Direct Axios

**What triggers it:**
Importing axios directly for API calls that have generated hooks.

**Example:**
```typescript
// Flagged — bypassing Orval + Zod validation:
import axios from 'axios';
const { data } = await axios.get(`/v0/projects/${projectId}/connectors`);

// Should use generated hook:
import { useGetProjectsConnectorsByProjectId } from "gen/v0/api/connectors/connectors";
const { data, isLoading, error } = useGetProjectsConnectorsByProjectId(projectId);
```

**Exceptions:** Direct axios is acceptable for file uploads with progress tracking (`hooks/tables/sourceTables.ts`, `hooks/tables/tables.ts`) where AbortController and `onUploadProgress` are needed.

**Why:** Generated hooks route through `services/orval_axios_client.ts` which provides Zod schema validation (when `ENABLE_UI_SCHEMA_VALIDATION` flag is on), consistent error handling, and toast on schema mismatches.

---

### Pattern 7: Error Boundary Wrapping

**What triggers it:**
Exported components without error boundary wrapping.

**Example:**
```typescript
// Flagged — no error boundary:
const MyComponent = (props: Props) => { ... };
export default MyComponent;

// Should wrap:
import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
const MyComponent = (props: Props) => { ... };
export default WithErrorBoundary(MyComponent);

// For forwardRef components, use the named export from the same module:
import { WithErrorBoundaryForwardRef } from "components/@hoc/WithErrorBoundary";
const MyDialog = forwardRef<DialogRef, Props>((props, ref) => {
  useImperativeHandle(ref, () => ({ toggleDialog: ... }), []);
  return <Dialog>...</Dialog>;
});
export default WithErrorBoundaryForwardRef(MyDialog);
```

**Exports:** `WithErrorBoundary` is the default export and `WithErrorBoundaryForwardRef` is a named export of `components/@hoc/WithErrorBoundary.tsx`.

**Reference:** `ui/src/components/@hoc/WithErrorBoundary.tsx` (both wrappers).

---

### Pattern 8: Lazy Loading Pages

**What triggers it:**
Page-level components imported eagerly instead of using the lazy + preload pattern.

**Example:**
```typescript
// Flagged — eager import:
import Admin from "pages/Admin/Admin";
{ path: "/admin", element: <Admin /> }

// Should use createLazyModule:
const Admin = createLazyModule(() => import("pages/Admin/Admin"));
{ path: navRoutes.Admin, lazy: Admin.lazy }
```

**Why:** The `createLazyModule` pattern caches the import promise (deduplicates) and exposes `preload()` for route loaders to prefetch on hover/navigation.

**Reference:** `ui/src/routes/Routes.tsx` — all page routes follow this pattern.

---

### Pattern 9: Feature Flags

**What triggers it:**
Hardcoded deployment-mode or feature checks instead of using the flag system.

**Example:**
```typescript
// Flagged — hardcoded:
if (window.ENV === 'databricks') { ... }
const isDatabricks = process.env.REACT_APP_DEPLOY_MODE === 'databricks';

// Should use hook:
import useFeatureFlags from "hooks/utils/featureFlags";
const flags = useFeatureFlags();
if (flags.IS_DATABRICKS_DEPLOY) { ... }
```

**Available flags:** `ENABLE_FILE_UPLOAD`, `ENABLE_COLUMN_ANALYSIS`, `MAX_TABLES_IN_A_GRAPH`, `IS_DATABRICKS_DEPLOY`, `ENABLE_UI_SCHEMA_VALIDATION`.

**Reference:** `ui/src/hooks/utils/featureFlags.ts`, `ui/src/stores/debugStore.ts`. Flags are fetched from `/constants/flags` endpoint.

---

### Pattern 10: Token Service

**What triggers it:**
Direct token manipulation outside the singleton service.

**Example:**
```typescript
// Flagged — direct access:
const token = localStorage.getItem('accessToken');
axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;

// Should use service:
import tokenService from "services/token_service";
const token = tokenService.getAccessToken();
// Note: axios interceptor in services/axios_client.ts adds the header automatically
```

**Why:** The `tokenService` handles token lifecycle, refresh, and cleanup on logout. The axios interceptor automatically injects the `Authorization` header.

**Reference:** `ui/src/services/token_service.ts`, `ui/src/services/axios_client.ts` (interceptors).

---

### Pattern 11: Error Handling Utility

**What triggers it:**
Custom error display logic instead of using the centralized handler.

**Example:**
```typescript
// Flagged — custom error handling:
try {
  await saveConnector(data);
} catch (err) {
  alert(err.message); // or setError(err.toString())
}

// Should use utility:
import { handleErrorFunctional, adjustError } from "utils/errorHandler";

try {
  await saveConnector(data);
} catch (err) {
  const error = err instanceof Error ? err : new Error(String(err));
  handleErrorFunctional(error, showBanner, alertError, "Failed to save connector");
}

// For errors that may have empty descriptions:
const userMessage = adjustError("");
// Returns a user-friendly "Contact support" message when description is empty
```

**What it does:**
- HTTP 500 → shows banner ("Internal error")
- HTTP 422 → shows validation details from response
- Others → shows alert dialog
- Calls `logStackTrace()` internally (backend in prod, console in dev, silent in test)
- `adjustError()` handles empty error descriptions with a user-friendly fallback

**Reference:** `ui/src/utils/errorHandler.ts` — `handleErrorFunctional()`, `adjustError()`, `parseErrorMessage()`, `logStackTrace()`.

---

### Pattern 12: Zustand Store Conventions

**What triggers it:**
New stores missing required middleware, or components accessing store state inline without selectors.

**Example:**
```typescript
// Flagged — missing middleware (for stores with async actions or mutable updates):
const useMyStore = create<State & Actions>((set) => ({ ... }));

// Should use devtools + immer:
const useMyStore = create<State & Actions>()(
  devtools(
    immer((set, get) => ({ ... })),
    { name: "MyStore" }
  )
);

// Exception — simple config stores (like cacheStore) can skip middleware:
const useCacheStore = create<CacheState & CacheActions>((set) => ({
  staleTime: 300000,
  setStaleTime: (time) => set({ staleTime: time }),
}));

// Flagged — inline selector in component:
const Component = () => {
  const isLoggedIn = useUserStore(state => state.userDetails.isLoggedIn);
};

// Should create reusable selector hook:
export const useIsLoggedIn = () => useUserStore((state) => state.userDetails.isLoggedIn);

const Component = () => {
  const isLoggedIn = useIsLoggedIn();
};
```

**Convention:** Separate `State` and `Actions` interfaces. Use `store.getInitialState()` for reset logic.

**Reference:** `ui/src/stores/debugStore.ts`, `ui/src/stores/userStore.ts`, `ui/src/stores/cacheStore.ts`.

---

### Pattern 13: Import Path Aliases

**What triggers it:**
Deep relative imports instead of using the `baseUrl: "src"` alias.

**Example:**
```typescript
// Flagged:
import { storageKeys } from "../../../utils/Constants/enums";
import { useConnectors } from "../../hooks/connector/connector";

// Should use alias:
import { storageKeys } from "utils/Constants/enums";
import { useConnectors } from "hooks/connector/connector";
```

**Why:** `tsconfig.json` sets `baseUrl: "src"`. All imports should be absolute from `src/`. The codebase has zero `../../../` imports.

---

### Pattern 14: TypeScript Strictness

**What triggers it:**
New `any` types or missing return types on exports.

**Example:**
```typescript
// Flagged — any without justification:
const processData = (input: any) => { ... };

// Should use proper type or unknown:
const processData = (input: unknown) => {
  if (isValidInput(input)) { ... }
};

// Flagged — missing return type on export:
export const useMyHook = () => { ... };

// Should be explicit:
export const useMyHook = (): { data: Data; isLoading: boolean } => { ... };
```

**Current state:** Only ~25 `any` instances across 5 files (mostly `queryOptions.ts` and test setup). `strict: true` and `noImplicitAny: true` are enforced in `tsconfig.json`.

---

### Pattern 15: No console.log in Production Code

**What triggers it:**
`console.log` statements in source files (not tests).

**Example:**
```typescript
// Flagged:
console.log('connector data:', data);
console.log('debug:', response);

// For errors, use the utility:
import { logStackTrace } from "utils/errorHandler";
logStackTrace("Connector save failed", error);
// → Production: logs to backend via createLogLogUiError()
// → Development: console.trace()
// → Test: silent

// For error catch blocks, console.error is acceptable:
try { ... } catch (err) {
  console.error("Unable to save to session storage", err);
}
```

**Reference:** `ui/src/utils/errorHandler.ts` — `logStackTrace()` routes logs based on `NODE_ENV`.

---

### Pattern 16: Navigation Paths and Hooks

**What triggers it:**
Hardcoded route strings or raw `useNavigate()` usage.

**Example:**
```typescript
// Flagged — hardcoded route:
navigate("/home");
<Link to="/admin">Admin</Link>

// Should use navRoutes enum:
import { navRoutes } from "utils/Constants/routes";
navigate(navRoutes.Home);
<Link to={navRoutes.Admin}>Admin</Link>

// Flagged — raw useNavigate (no ctrl+click support):
import { useNavigate } from "react-router";
const navigate = useNavigate();

// Should use custom hook:
import { useCustomNavigation } from "utils/navigator";
const navigate = useCustomNavigation();
// Supports ctrl+click / cmd+click to open in new tab
```

**Reference:** `ui/src/utils/Constants/routes.ts` (`navRoutes` enum), `ui/src/utils/navigator.ts` (`useCustomNavigation` hook).

---

### Pattern 17: Route Loaders for Prefetching

**What triggers it:**
New page routes without a `loader:` function.

**Example:**
```typescript
import { connectorsLoader } from "routes/Loaders";

const ConnectorsList = createLazyModule(() => import("pages/Connectors/ConnectorsList/ConnectorsList"));
// Both examples are children of navRoutes.ProjectBase.
// Missing loader — data fetches only after component mounts:
{ path: "connectors/:id?", lazy: ConnectorsList.lazy }

// The loader reads params.projectId and prefetches that project's connectors:
{
  path: "connectors/:id?",
  lazy: ConnectorsList.lazy,
  loader: connectorsLoader,
}
```

**Why:** Route loaders fire on navigation (or even on hover/preload), so data fetching starts before the component JS is even loaded. This eliminates the request waterfall of lazy-load-JS → mount → fetch-data.

**Reference:** `ui/src/routes/Routes.tsx` (loader pattern), `ui/src/routes/Loaders.ts`.

---

### Pattern 18: Runtime Config via window._env_

**What triggers it:**
Reading `process.env.REACT_APP_*` in production code paths.

**Example:**
```typescript
// Flagged — process.env is build-time only:
const apiUrl = process.env.REACT_APP_BASE_URL_PROD;

// Should use runtime config:
const apiUrl = window._env_.REACT_APP_BASE_URL_PROD;
```

**Why:** `env.js` injects `window._env_` at container startup, allowing runtime configuration without rebuilds. `process.env` values are baked in at build time. The only acceptable `process.env` usage is in `getBaseURL()` (to check `NODE_ENV`) and test guards.

**Reference:** `ui/env.js`, `ui/src/services/axios_client.ts` (`getBaseURL()`).

---

### Pattern 19: Request Cancellation

**What triggers it:**
Hooks that fetch on mount or poll without cleanup, causing leaked requests.

**Example:**
```typescript
// Flagged — no cancellation on unmount:
const useFetchData = (id: string) => {
  useEffect(() => {
    fetchData(id).then(setData);
  }, [id]);
};

// Should use AbortController:
const useFetchData = (id: string) => {
  const controllerRef = useRef<AbortController>();

  useEffect(() => {
    controllerRef.current?.abort();
    controllerRef.current = new AbortController();

    fetchData(id, { signal: controllerRef.current.signal })
      .then(setData)
      .catch(err => {
        if (!err.name?.includes('Abort')) throw err;
      });

    return () => controllerRef.current?.abort();
  }, [id]);
};

// For polling — use isCancelled ref:
const usePolling = () => {
  const isCancelled = useRef(false);

  useEffect(() => {
    isCancelled.current = false;
    const poll = async () => {
      while (!isCancelled.current) {
        await fetchStatus();
        await delay(5000);
      }
    };
    poll();
    return () => { isCancelled.current = true; };
  }, []);
};
```

**Reference:** `ui/src/hooks/tables/sourceTables.ts` (AbortController), `ui/src/hooks/connector/connector.ts` (isCancelled polling).

---

### Pattern 20: No Third-Party Form Libraries

**What triggers it:**
PRs introducing `react-hook-form`, `formik`, `yup`, or other form libraries.

**Example:**
```typescript
// Flagged — third-party form lib:
import { useForm } from "react-hook-form";
import * as yup from "yup";

// Should use controlled components + inputState enum:
import { inputState } from "utils/Constants/enums";

const [name, setName] = useState("");
const [nameState, setNameState] = useState(inputState.None);

const handleBlur = async () => {
  setNameState(inputState.Validating);
  const isValid = await validateName(name); // API validation
  setNameState(isValid ? inputState.Valid : inputState.Invalid);
};

return <Input value={name} onChange={(event) => setName(event.target.value)} onBlur={handleBlur} aria-invalid={nameState === inputState.Invalid} />;
```

**Why:** The codebase has a consistent validation pattern using `inputState` enum and API-side validation endpoints. Introducing a form library would create two competing patterns.

**Reference:** `ui/src/utils/Constants/enums.ts` (`inputState` enum), `ui/src/utils/validator.ts`.

---

### Pattern 21: Parallel Store Fetches with Promise.allSettled

**What triggers it:**
Using `Promise.all` in store actions that fetch from multiple endpoints.

**Example:**
```typescript
// Flagged — one failure blocks everything:
const fetchData = async () => {
  const [flags, config, user] = await Promise.all([
    getFlags(), getConfig(), getUser()
  ]);
  set({ flags, config, user });
};

// Should use Promise.allSettled:
const fetchData = async () => {
  const [flagsResult, configResult, userResult] = await Promise.allSettled([
    getFlags(), getConfig(), getUser()
  ]);

  set({
    flags: flagsResult.status === "fulfilled" ? flagsResult.value : defaults.flags,
    config: configResult.status === "fulfilled" ? configResult.value : defaults.config,
    user: userResult.status === "fulfilled" ? userResult.value : defaults.user,
  });
};
```

**Why:** Store initialization fetches should be resilient. If the flags endpoint is down, the user can still see the app with defaults. `Promise.all` would show a blank screen.

**Reference:** `ui/src/stores/debugStore.ts` — `fetchData()` uses this pattern.

---

### Pattern 22: HTML Error Sanitization

**What triggers it:**
Components that detect or parse HTML in error responses.

**Example:**
```typescript
// Flagged — duplicating interceptor logic:
catch (err) {
  if (typeof err.response?.data === 'string' && err.response.data.includes('<!DOCTYPE')) {
    setError("Service unavailable");
  }
}

// Not needed — the axios interceptor already handles this:
// services/axios_client.ts detects HTML_PATTERN in responses and converts
// to SERVICE_UNAVAILABLE_MESSAGE automatically. Just use handleErrorFunctional().
```

**Reference:** `ui/src/services/axios_client.ts` — response interceptor with `HTML_PATTERN` detection.

---

### Pattern 23: Discriminated Union State

**What triggers it:**
Multiple boolean flags modeling async state that can represent impossible combinations.

**Example:**
```typescript
// Flagged — allows impossible states (isLoading=true AND error=something):
type State = {
  isLoading: boolean;
  error: Error | null;
  data: Data | null;
};

// Preferred — discriminated union:
type State =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'success'; data: Data }
  | { status: 'error'; error: Error };

// Usage — TypeScript narrows the type:
switch (state.status) {
  case 'loading': return <Spinner />;
  case 'error': return <ErrorMessage error={state.error} />;
  case 'success': return <DataView data={state.data} />;
  default: return null;
}
```

**Why:** Discriminated unions make impossible states unrepresentable and give TypeScript exhaustive checking in switch statements.

**Reference:** `ui/docs/Coding_Guidelines.md` recommends this pattern.

---

### Pattern 24: Comment Quality

**What triggers it:**
Comments that restate what the code does instead of explaining why.

**Example:**
```typescript
// Flagged — restates the code:
// Set loading to true
setLoading(true);
// Fetch the connectors
const connectors = await fetchConnectors();

// Preferred — explains why:
// Fetch before component mount to avoid waterfall with lazy-loaded JS
const connectors = await fetchConnectors();

// Also preferred — explains non-obvious constraint:
// Direct Connect password mode needs runtime config before the login page
// decides whether to render the local password form or redirect to OIDC.
await loadAuthClientConfig();
```

---

### Pattern 25: Design Tokens, Not Default Tailwind Palette

**What triggers it:**
Using default Tailwind neutral palette classes (`slate`/`gray`/`zinc`/`neutral`/`stone`) for color, surface, border, or icon styling where a Kumo design token exists. Part of the UI Catalog + Presentation Migration — use tokens for semantic color, Tailwind only for layout/spacing/typography.

**Token-First mapping:**

| Need | Use (token) | Avoid (default palette) |
|------|-------------|-------------------------|
| Primary text / headings | `text-title-primary` | `text-slate-800`, `text-gray-900` |
| Secondary text / labels | `text-title-secondary` | `text-slate-500`, `text-gray-500` |
| Primary border | `border-border-primary` | `border-slate-500` |
| Secondary border | `border-border-secondary` | `border-slate-200`, `border-gray-300` |
| Neutral surface | `bg-surface-grey` | `bg-slate-100`, `bg-gray-100` |
| Primary icon | `text-icon-primary` | `text-gray-700` |
| Secondary icon | `text-icon-secondary` | `text-gray-400`, `text-gray-500` |

**Example:**
```tsx
// Flagged (design-token):
<div className="text-slate-800 border-gray-300 bg-gray-100" />

// Preferred:
<div className="text-title-primary border-border-secondary bg-surface-grey" />
```

**Exceptions (suppress with a reason):** tooltip-dark backgrounds (`style={{ backgroundColor: "var(--tooltip-dark)" }}` until a class exists) and semantic error/warning red/amber where no token exists yet. A missing token should become a catalog token request, not app-local hardcoding.

### Pattern 26: No Arbitrary Tailwind Pixels or Hex

**What triggers it:**
Arbitrary-value Tailwind classes with hardcoded pixels or hex colors — they bypass both the Tailwind scale and the token system.

**Example:**
```tsx
// Flagged (arbitrary-tailwind):
<div className="w-[12px] h-[44px] text-[#1a1a1a]" />

// Preferred — standard scale / token:
<div className="w-3 h-11 text-title-primary" />
```

**Allowed (not flagged):**
- Dynamic values via CSS vars: `w-[var(--sidebar-width)]` (runtime geometry, Radix, etc.)
- Viewport units where no token equivalent exists: `h-[50vh]`, `max-w-[90vw]`

Suppress documented layout exceptions with `// review-ui-ignore: arbitrary-tailwind — <reason>`.

---

## Source Material

| Source | Area |
|--------|------|
| `ui/src/utils/reactQueryKeys.ts` | 84 centralized query keys |
| `ui/src/utils/queryInvalidation.ts` | Cache invalidation helpers |
| `ui/src/utils/storage.ts` | Application storage and session-storage helpers |
| `packages/kumo-ui-catalog/src/utils/storage.ts` | Local-storage helpers |
| `ui/src/services/orval_axios_client.ts` | Zod-validated API client |
| `ui/src/services/axios_client.ts` | Interceptors, HTML sanitization, getBaseURL |
| `ui/src/services/token_service.ts` | Token lifecycle management |
| `ui/src/stores/debugStore.ts` | Feature flags, Zustand conventions, Promise.allSettled |
| `ui/src/stores/userStore.ts` | Zustand store with selectors |
| `ui/src/stores/cacheStore.ts` | Simple store without middleware |
| `ui/src/routes/Routes.tsx` | Lazy loading, route loaders, navRoutes |
| `ui/src/utils/Constants/routes.ts` | navRoutes enum |
| `ui/src/utils/navigator.ts` | useCustomNavigation hook |
| `ui/src/components/@hoc/WithErrorBoundary.tsx` | Regular and forwardRef error boundary HOCs |
| `ui/src/utils/errorHandler.ts` | handleErrorFunctional, adjustError, logStackTrace |
| `ui/src/utils/Constants/enums.ts` | inputState, storageKeys enums |
| `ui/src/utils/validator.ts` | Form validation utilities |
| `ui/env.js` | Runtime config via window._env_ |
| `ui/tsconfig.json` | strict: true, noImplicitAny, baseUrl: "src" |
| `ui/docs/Coding_Guidelines.md` | Component architecture, state, testing, naming |
