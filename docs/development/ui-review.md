# UI review conventions

Use these conventions with [UI development](ui.md). Check findings against the
changed source and its callers; a textual scan is a hint, not proof of a defect.

## Inline Suppression

Developers can suppress a specific rule on a specific line by adding a comment:

```typescript
localStorage.getItem('key'); // review-ui-ignore: direct-storage
```

Multiple rules can be suppressed on one line:

```typescript
console.log(localStorage.getItem('x')); // review-ui-ignore: console-log, direct-storage
```

The pre-scan script and LLM review both respect these comments. Suppression is per-line only — it does not affect other lines in the file.

**When reviewing:** If a suppression comment lacks justification, flag it as a [MEDIUM] violation:

> [MEDIUM] **Suppression Justification**: `review-ui-ignore` without explanation. Add a brief reason (e.g., `// review-ui-ignore: direct-storage — migration helper, tracked in #1234`).

---

## Rules

Each rule has a name, severity, and a **rule ID** in parentheses. The rule ID is used in:
- Pre-scan script output
- `review-ui-ignore:` suppression comments

Rule IDs are **not** shown in review comments — keep comments clean with just `[SEVERITY] **Rule Name**: explanation`.

### Kumo-Specific

**[HIGH] Generated Code** (`generated-code`): Flag manually defining types, interfaces, or API hooks that already exist in `ui/src/gen/`. Use Orval codegen (`generate-ui-client-code` or `/trigger-ui-codegen`) to regenerate instead.

**[HIGH] Catalog Components** (`catalog-component`): Flag reimplementing UI primitives (Button, Dialog, Card, Alert, Tooltip, Select, etc.) that exist in `@kumo-ai/kumo-ui-catalog`. Check catalog imports across 198+ files for precedent.

**[HIGH] React Query Keys** (`hardcoded-querykey`): Flag hardcoded query key strings. Must use `ReactQueryKeys` enum from `utils/reactQueryKeys.ts`. Flag manual `queryClient.invalidateQueries()` — use helpers from `utils/queryInvalidation.ts`.

**[HIGH] Storage Access** (`direct-storage`): Flag direct `localStorage.getItem/setItem` or `sessionStorage` calls. Use local-storage helpers from `@kumo-ai/kumo-ui-catalog` or application/session helpers from `utils/storage.ts`, with the `storageKeys` enum from `utils/Constants/enums.ts`. *Exception: `utils/storage.ts` itself.*

**[HIGH] Axios Bypass** (`axios-bypass`): Flag direct `axios.get/post/put/delete` imports for API calls that should use Orval-generated hooks. *Exceptions: `services/` directory, file uploads with progress tracking in `hooks/tables/`.*

**[HIGH] Error Boundaries** (`error-boundary`): Flag exported components not wrapped with `WithErrorBoundary` (from `components/@hoc/WithErrorBoundary.tsx`) or `WithErrorBoundaryForwardRef` (from `components/@hoc/WithErrorBoundary.tsx`) for components using `forwardRef` + `useImperativeHandle`.

**[HIGH] Route Lazy Loading** (`route-lazy-load`): Flag page-level routes not using the `createLazyModule()` pattern from `routes/Routes.tsx`. All page imports must be lazy with preload support.

**[HIGH] Feature Flags** (`feature-flags`): Flag hardcoded deployment-mode or feature checks. Must use `useFeatureFlags()` hook from `hooks/utils/featureFlags.ts`. Flags: `IS_DATABRICKS_DEPLOY`, `ENABLE_FILE_UPLOAD`, `ENABLE_UI_SCHEMA_VALIDATION`, etc.

**[HIGH] Auth / Tokens** (`auth-tokens`): Flag direct token manipulation (reading/writing access tokens to localStorage). Must use `tokenService` singleton from `services/token_service.ts`. Auth state comes from `useUserStore` and `useAuthentication`.

**[HIGH] XSS** (`xss`): Flag any `dangerouslySetInnerHTML` usage — the codebase has zero instances and should stay that way.

**[HIGH] Error Handling** (`error-handling`): Flag custom error display logic. Must use `handleErrorFunctional()` from `utils/errorHandler.ts` which handles HTTP status differentiation (500 → banner, 422 → validation details, others → alert). Use `adjustError()` for empty error descriptions. Flag raw error objects shown to users.

**[HIGH] Import Paths** (`deep-import`): Flag deep relative imports (`../../../`). Must use absolute paths from `baseUrl: "src"`. *Auto-fixable.*

### Runtime Config & Navigation

**[HIGH] Console Statements** (`console-log`): Flag `console.log` in production code. Use `logStackTrace()` from `utils/errorHandler.ts` for error logging. *Exception: `utils/errorHandler.ts` itself. Auto-fixable.*

**[HIGH] Navigation Paths** (`hardcoded-route`): Flag hardcoded route strings in `navigate()`, template literals in `navigate()`, or `<Link to=...>`. Must use `navRoutes` enum from `utils/Constants/routes.ts`. Flag raw `useNavigate()` — use `useCustomNavigation()` from `utils/navigator.ts`. *Auto-fixable.*

**[HIGH] Runtime Config** (`process-env`): Flag code reading `process.env.REACT_APP_*` in production paths. Runtime config must come from `window._env_`. *Exception: `services/axios_client.ts`. Auto-fixable.*

**[HIGH] No Third-Party Form Libraries** (`form-library`): Flag PRs introducing `react-hook-form`, `formik`, or `yup`. The convention is controlled components + `inputState` enum + API-side validation.

### Stores & State

**[MEDIUM] Zustand Stores** (`zustand-middleware`): Flag new stores with async actions or mutable state updates missing `devtools(immer(...))` middleware. Simple config stores may skip. Flag inline selectors — create selector hooks. Flag stores without separate state/actions interfaces.

**[MEDIUM] Zustand Parallel Fetches** (`promise-allsettled`): Flag `Promise.all` in store actions that fetch from multiple endpoints. Use `Promise.allSettled` and check each result individually.

**[MEDIUM] Discriminated Union State** (`discriminated-union`): Flag `{ isLoading, error, data }` patterns for complex async state. Prefer discriminated unions with `status` field.

### TypeScript

**[MEDIUM] TypeScript** (`typescript-any`): Flag new `any` types without eslint-disable justification. Flag missing explicit return types on exported functions/hooks.

### Routes & Data Fetching

**[MEDIUM] Route Loaders** (`route-loader`): Flag new page routes without a `loader:` function in `Routes.tsx`. List/detail routes should call `prefetchGetXxx(queryClient, ...)`.

**[MEDIUM] Request Cancellation** (`request-cancellation`): Flag hooks that fetch on mount or poll without cleanup. Use `AbortController` via `useRef` or `isCancelled` ref pattern.

### Styling & HTML

**[MEDIUM] Design Tokens** (`design-token`): Flag default Tailwind neutral palette classes (`text-slate-*`, `text-gray-*`, `bg-gray-*`, `border-gray-*`, and the `zinc`/`neutral`/`stone` families) where a Kumo token exists. Mapping: `text-slate-800`/`text-gray-900`→`text-title-primary`; `text-slate-500`/`text-gray-500`→`text-title-secondary`; `border-slate-500`→`border-border-primary`; `border-slate-200`/`border-slate-300`/`border-gray-300`→`border-border-secondary`; `bg-slate-100`/`bg-gray-100`→`bg-surface-grey`; `text-gray-700`→`text-icon-primary`; `text-gray-400`/`text-gray-500`→`text-icon-secondary`. *Exceptions: tooltip-dark backgrounds (`var(--tooltip-dark)`) and semantic error/warning red/amber where no token exists — suppress with `review-ui-ignore: design-token` and a reason. Auto-fixable when a mapping exists.*

**[MEDIUM] Arbitrary Tailwind Values** (`arbitrary-tailwind`): Flag arbitrary-value classes with hardcoded pixels or hex colors (`w-[12px]`, `h-[44px]`, `text-[#1a1a1a]`). Use the standard Tailwind scale or a Kumo token instead. *Allowed (not flagged): dynamic `var(--*)` values and viewport units (`vh`/`vw`/`vmin`/`vmax`) where no token equivalent exists. Suppress documented layout exceptions with `review-ui-ignore: arbitrary-tailwind`.*

**[MEDIUM] No New Global CSS** (`global-css`): Flag new class definitions in `main.css`. Use Tailwind utilities or CSS custom properties.

**[MEDIUM] HTML Error Sanitization** (`html-sanitization`): Flag components that detect or parse HTML in error responses. The axios interceptor handles this.

### Comments

**[MEDIUM] Comment Quality** (`comment-quality`): Flag comments that restate what the code does. Comments should explain "why".

### Testing Infrastructure

**[HIGH] Mock Architecture** (`mock-axios`): Flag inline `vi.mock("axios")` or custom axios mocking per test file. Must use `tests/mockAxios.ts` dispatch table.

**[HIGH] QueryClient in Tests** (`raw-queryclient`): Flag `new QueryClient()` in test files. Must use `createQueryClient()` from `tests/testUtil.ts`. *Auto-fixable.*

**[HIGH] Hook Test Wrapper** (`hook-test-wrapper`): Flag `renderHook(...)` without `{ wrapper: createWrapper(queryClient) }` for React Query hooks.

**[HIGH] Zustand State in Tests** (`mock-store`): Flag `vi.mock()` on Zustand stores. Inject state directly with `useXxxStore.setState(...)`.

**[HIGH] Router Mocking** (`mock-router`): Flag inline `vi.mock("react-router", ...)`. Use `mockNavigateCall()` and `mockLocationStateValue()` from `tests/mockRouter.ts`.

**[MEDIUM] Reuse Test Helpers** (`test-helpers`): Flag new test utility functions that duplicate existing helpers in `tests/testHelpers.ts`.

**[MEDIUM] Global Setup Mocks** (`global-setup-mocks`): Flag per-test overrides of mocks that `tests/setup.ts` already provides. Only override with a justifying comment.

**[MEDIUM] data-testid on @ui Primitives** (`data-testid`): Flag removal of `data-testid` attributes from shared `@ui` components. New `@ui` components should include them.

### General React

**[MEDIUM] Component Architecture** (`component-architecture`): Flag business logic mixed into presentation. Extract to custom hooks. Flag deeply nested ternary rendering — use early returns.

**[MEDIUM] State Management** (`state-management`): Flag `setState` passed as prop. Flag state synchronization (local state mirroring props via `useEffect`). Flag 3+ related `useState` that should be `useReducer`.

**[MEDIUM] Performance** (`performance`): Flag inline object/array literals in JSX props of memoized components. Flag missing hook dependencies.

**[MEDIUM] Test Quality** (`test-quality`): Flag arbitrary `setTimeout`/`waitUntilSec` in tests — use `waitFor` or `vi.useFakeTimers()`. Flag brittle selectors when accessible queries exist.

**[MEDIUM] Naming** (`naming`): Flag non-PascalCase component files. Flag hooks not starting with `use`. Flag event handlers not following `on<Action><Event>`.

**[MEDIUM] Accessibility** (`accessibility`): Flag `<div onClick>` that should be `<button>`. Flag inputs without `<label>`. Flag interactive elements missing ARIA labels.

**[HIGH] Scope** (`scope`): Flag unrelated formatting or refactoring mixed in. Flag edits to generated code (`ui/src/gen/`, `ui/src/utils/PQLParsing/grammar/`, `ui/src/utils/PQLParsing/pqDocs.json`).

**[MEDIUM] Styling** (`styling`): Flag inline `style={{}}` when Tailwind works. Flag hardcoded color/spacing values.

**[MEDIUM] React Keys** (`react-keys`): Flag array-index keys (`key={index}`) or missing `key` in `.map()`. Use stable identifiers. *Auto-fixable.*

**[MEDIUM] Event Listener Cleanup** (`event-cleanup`): Flag `addEventListener`, `IntersectionObserver`, `MutationObserver`, or `WebSocket` in `useEffect` without cleanup.

**[MEDIUM] Bundle Size** (`bundle-size`): Flag full-library imports (`lodash`, `moment`) when tree-shakeable or native alternatives exist. *Auto-fixable.*

---

## Reference

Pattern examples (loaded conditionally — see Step 3):
- [UI kumo patterns](ui-review-kumo.md) — Kumo conventions, runtime config, navigation, state
- [UI testing patterns](ui-review-testing.md) — Testing infrastructure
- [UI react patterns](ui-review-react.md) — General React patterns

Pre-scan script: `.agents/skills/review-ui/scripts/diff-scan.sh`
Script tests: `.agents/skills/review-ui/scripts/tests/run-tests.sh`
Coding guidelines: `ui/docs/Coding_Guidelines.md`
