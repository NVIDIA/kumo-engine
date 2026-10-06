---
name: code-ui
description: Scaffold and implement UI features following Kumo conventions. Use when creating new components, pages, hooks, stores, or tests in ui/src/.
---

# Code UI

You are a UI coding agent. You write production-quality React/TypeScript code following Kumo conventions.

## Ownership Boundaries

`/code-ui` owns:

- implementation planning from the provided handoff
- code edits and integration wiring
- implementation-specific checks
- targeted test design
- initial diagnosis of failures introduced by the change

`/code-ui` does not own:

- deciding whether a ticket should proceed
- approval checkpoints
- PR creation decisions
- overall workflow stop or continue policy
- diff review standards

## Invocation

```
/code-ui <task-description>
/code-ui add a new page for model comparison
/code-ui add store for notification preferences
/code-ui add test for useConnectorData hook
```

## Step 0: Load Context

**Always load:**
- `docs/development/ui.md` — compressed coding conventions and recipes
- `.agents/skills/code-ui/references/catalog-components.md` — available UI catalog components
- `.agents/skills/code-ui/references/design-tokens.md` — color tokens, typography, Figma mapping
- `.agents/skills/code-ui/references/integration-checklist.md` — integration and implementation checks

**Load conditionally:**
- If creating/modifying tests: `docs/development/ui-review-testing.md`
- If creating/modifying tests: `.agents/skills/code-ui/references/testing-playbook.md`
- If creating/modifying components or hooks: `docs/development/ui-review-react.md`
- If touching Kumo-specific patterns (stores, API, navigation): `docs/development/ui-review-kumo.md`

## Step 1: Understand the Task

Required outputs:

- implementation plan
- code changes aligned to the task
- relevant integration updates
- targeted validation results
- explicit notes for any blocked or skipped verification

1. Parse the task description to determine what type of artifact is needed:
   - **page** — new route with lazy loading, loader, error boundary
   - **component** — new UI component with error boundary
   - **hook** — custom hook (data fetching, state logic, utility)
   - **store** — Zustand store with devtools + immer
   - **test** — test file for existing component/hook/store
   - **feature** — combination of the above

2. Identify which existing files need to be modified (routes, barrel exports, mock dispatch, test setup)

3. Check for existing patterns — search for similar components/hooks in the codebase to match local conventions

## Step 2: Check Prerequisites

Before writing code:

1. **Check generated types**: Search `gen/v1/` first for relevant API types and hooks. If the endpoint isn't in `gen/v1/`, fall back to `gen/v0/` which has broader coverage. Use the generated hook — don't write manual axios calls.

2. **Check catalog**: Search catalog-components.md for UI primitives you need. Don't reimplement what exists.

3. **Check existing utils**: Search `utils/`, `hooks/`, `stores/` for helpers that already exist. Don't duplicate.

4. **Check query keys**: If adding data fetching, check `utils/reactQueryKeys.ts` for existing keys. Add new keys if needed.

5. **Check design tokens**: Load `references/design-tokens.md`. Never use generic Tailwind colors (`slate-*`, `gray-*`) — always map to Kumo tokens (`title-primary`, `border-secondary`, etc.).

6. **Check compiled CSS**: Before using a Tailwind class, verify it exists in the compiled bundle. Arbitrary values (`text-[26px]`, `gap-[36px]`) often don't compile in Tailwind v4. Use standard classes instead.

## Step 3: Implement

Follow the recipes in `docs/development/ui.md` for each artifact type. Key rules:

### All Files
- Absolute imports from `baseUrl: "src"` — never `../../../`
- TypeScript strict mode — no `any` without justification
- PascalCase component files, `use*` hooks, `on<Action><Event>` handlers

### Styling & Design Tokens
- **Always use Kumo design tokens** — see `references/design-tokens.md` for the full mapping
- `text-title-primary` / `text-title-secondary` — not `text-slate-800` / `text-slate-500`
- `border-border-secondary` — not `border-slate-200` / `border-gray-300`
- `bg-surface-grey` — not `bg-slate-100`
- `var(--tooltip-dark)` for popover backgrounds — not `bg-gray-800`
- `text-icon-secondary` for info icons — not `text-gray-400`
- **Inline styles for values that Tailwind v4 won't compile** (arbitrary values, CSS variable references)
- **Scoped `<style>` with `#id` selectors** to override shared component internals — don't modify shared components, don't rely on `[&_]` syntax
- **Verify in browser** after styling changes — computed values may differ from class names due to Tailwind v4 JIT behavior

### Components
- Wrap with `WithErrorBoundary` (or `WithErrorBoundaryForwardRef` for forwardRef)
- Use catalog primitives from `@kumo-ai/kumo-ui-catalog`
- Use `cn()` for conditional classes
- Use Tailwind for layout — inline styles only when Tailwind classes don't compile
- Extract business logic into hooks — components only render

### Pages
- Use `createLazyModule()` pattern in `Routes.tsx`
- Add `navRoutes` enum entry in `utils/Constants/routes.ts`
- Add route loader in `Loaders.ts` with `prefetchGetXxx`
- Use `useCustomNavigation()` — not raw `useNavigate()`
- Guard dirty wizard forms on every exit path (`useBlocker` + `beforeunload`, reusing the cancel confirmation dialog); see `pages/NewJob/Training/Training.tsx` (kumo#30141)

### Hooks
- Use Orval-generated hooks from `gen/v1/` for API calls
- Use `ReactQueryKeys` enum for query keys
- Use `handleErrorFunctional()` for error handling
- Use `AbortController` or `isCancelled` ref for cleanup

### Stores
- Middleware: `devtools(immer(...))` with `{ name: "StoreName" }`
- Separate `StoreState` and `StoreActions` interfaces
- Create selector hooks: `const useXxx = () => useStore((s) => s.xxx)`
- Use `Promise.allSettled()` for parallel fetches
- Use `logStackTrace()` — not `console.log`

### Tests
- Import `"tests/mockAxios"` before test code
- Use `createQueryClient()` and `createWrapper()` from `tests/testUtil`
- Use `store.setState()` for Zustand — not `vi.mock()`
- Use `mockNavigateCall()` and `mockLocationStateValue()` from `tests/mockRouter`
- Use accessible queries (`getByRole`, `getByText`) — not `getByTestId`
- Use `waitFor()` — not `setTimeout`
- Render pages that call `useBlocker` inside a data router (`createMemoryRouter` + `RouterProvider`); plain `MemoryRouter` cannot host a blocker
- Reuse helpers from `tests/testHelpers.ts`

### API Mocks
- Add new endpoint mocks to `tests/mockAxios.ts` dispatch table
- Follow existing URL-matching pattern

## Step 3a: Apply Integration Checks

Use `references/integration-checklist.md`.

This is where `/code-ui` owns:

- shared primitive compatibility and caller updates
- route, query, invalidation, and mock wiring
- UI state coverage
- copy consistency checks
- incidental in-scope fixes while editing related files

## Step 4: Wire Up

After implementing, don't forget integration points:

- [ ] Add to barrel export (`index.ts`) if directory has one
- [ ] Add route + lazy module + loader (for pages)
- [ ] Add `navRoutes` enum entry (for pages)
- [ ] Add `ReactQueryKeys` entry (for new data fetching)
- [ ] Add invalidation helper in `queryInvalidation.ts` (for new mutations)
- [ ] Add store reset in `tests/setup.ts` `beforeEach` (for new stores)
- [ ] Add mock endpoints in `tests/mockAxios.ts` (for new API calls)

## Step 5: Validate

Run validation after implementation:

```bash
# Type check
cd ui && npx tsc --noEmit

# Lint
cd ui && npm run lint

# Run tests (specific file or all)
cd ui && npx vitest run src/path/to/__tests__/MyFile.test.tsx
cd ui && npm test
```

Fix any errors before considering the task complete.

## Step 6: Test Strategy

Use `references/testing-playbook.md`.

`/code-ui` owns:

- targeted test design
- test helper reuse
- mock completeness for changed behavior
- initial failure diagnosis during implementation

## Working with Figma Designs

When a Figma URL or node ID is provided:

### Step 1: Fetch design specs via API
```bash
curl -s -H "X-Figma-Token: $TOKEN" \
  "https://api.figma.com/v1/files/$FILE_ID/nodes?ids=$NODE_ID&depth=10"
```

### Step 2: Extract design tokens
Parse the JSON for: `fills` (colors), `strokes` (borders), `style.fontSize`, `style.fontWeight`, `cornerRadius`, `itemSpacing` (gap), `paddingTop/Left/Right/Bottom`, `layoutMode`.

### Step 3: Map to Kumo tokens (CRITICAL)
**Always use design tokens from `references/design-tokens.md`** — never hardcode Tailwind generic colors.

| Figma Property | Map To |
|---|---|
| `fills.color rgb(41,45,54)` | `text-title-primary` |
| `fills.color rgb(95,113,140)` | `text-title-secondary` |
| `strokes.color rgb(210,220,228)` | `border-border-secondary` |
| `fills.color rgb(244,246,248)` | `bg-surface-grey` |
| `fills.color rgb(58,58,65)` | `var(--tooltip-dark)` |
| `cornerRadius: 4` | `rounded` |
| `cornerRadius: 16` | `rounded-2xl` |
| `fontSize: 26` | `text-3xl` (NOT `text-[26px]`) |
| `fontSize: 16` | `text-base` |
| `fontSize: 14` | `text-sm` |
| `fontWeight: 700` | `font-bold` |
| `fontWeight: 500` | `font-medium` |
| `itemSpacing: 4` | `gap-1` |
| `itemSpacing: 8` | `gap-2` |
| `itemSpacing: 16` | `gap-4` |
| `itemSpacing: 24` | `gap-6` |
| `itemSpacing: 32` | `gap-8` |
| `padding: 16` | `p-4` |
| `padding: 24` | `p-6` |

### Step 4: Verify in browser
After implementing, use Chrome DevTools or MCP browser tools to verify:
- Computed font sizes and colors match Figma
- Gaps and padding match pixel values
- Alignment between adjacent panels
- Design token CSS variables resolve correctly

### Step 5: Verification checklist
After implementing, verify each of these in the browser (Chrome DevTools or MCP tools):
- [ ] All text uses `text-title-primary` / `text-title-secondary` — no hardcoded slate/gray
- [ ] All borders use `border-border-secondary` — no hardcoded slate/gray
- [ ] All tooltips/popovers use `var(--tooltip-dark)` bg, `text-sky-300` links, `size-3.5 text-icon-secondary` icons
- [ ] Font sizes match Figma (use `getComputedStyle` to verify px values)
- [ ] Gaps between elements match Figma `itemSpacing` values
- [ ] Padding matches Figma `paddingTop/Left/Right/Bottom`
- [ ] Adjacent panels align horizontally (titles at same Y, tab rows at same Y, underlines at same Y)
- [ ] Active tab underlines sit ON TOP of separator lines (not above with a gap)
- [ ] Shared components (TabCard) render without extra card chrome (border, shadow, padding)

### Tailwind v4 Gotchas
- **Arbitrary values (`text-[26px]`, `gap-[36px]`) often don't compile.** Use standard Tailwind classes instead.
- **`[&_selector]` descendant overrides may not work.** Use scoped `<style>` with `#id` selectors.
- **New classes may need a page reload** to appear in the JIT bundle.
- **When overriding shared components (TabCard, etc.)**, use scoped CSS with a wrapper `<div id="...">` + `<style>` tag rather than modifying the shared component.
- **`border-r-0`, `rounded-l` may not apply via HMR.** Use inline styles for border-radius and selective borders when Tailwind classes don't take effect.

## Working with UI Catalog

### When to modify the catalog

Modify the catalog when you need to:
- Add a new reusable primitive or block component
- Fix a bug in an existing catalog component
- Add variants/props to an existing component

Do NOT modify the catalog for:
- One-off components specific to a single feature (put these in `ui/src/components/`)
- Quick layout tweaks (use Tailwind overrides via `className` prop)

### In-tree catalog workflow

The application consumes `packages/kumo-ui-catalog` through a local file
reference. Read `packages/kumo-ui-catalog/CLAUDE.md` and its README,
then change shared components and UI callers in this same repository.
Validate the affected catalog stories/tests and UI integration. A separate
repository, registry publication, or two-PR sequence is not a prerequisite.

From the repository root, `npm --prefix ui run local:catalog` enables catalog
source development with the configured local API. Continue importing catalog
components through `@kumo-ai/kumo-ui-catalog`, not package-internal paths.

## Common Patterns Reference

### Error handling in a hook
```tsx
import { handleErrorFunctional } from "utils/errorHandler";
import { logStackTrace } from "utils/errorHandler";
import type { AlertData, BannerData } from "types/validation";

const useMyHook = (
  showBanner: (data: BannerData) => void,
  alertError: (data: AlertData) => void
) => {

  const doAction = async () => {
    try {
      await mutation.mutateAsync(params);
    } catch (e) {
      const error = e instanceof Error ? e : new Error(String(e));
      handleErrorFunctional(error, showBanner, alertError);
      logStackTrace("doAction failed", error);
    }
  };
};
```

### Feature flag conditional
```tsx
import useFeatureFlags from "hooks/utils/featureFlags";

const MyComponent = () => {
  const { IS_DATABRICKS_DEPLOY } = useFeatureFlags();
  if (IS_DATABRICKS_DEPLOY) return <DatabricksView />;
  return <DefaultView />;
};
```

### Storage access
```tsx
import { storeToLocalStorage, getFromLocalStorage } from "@kumo-ai/kumo-ui-catalog";
import { storageKeys } from "utils/Constants/enums";

storeToLocalStorage(storageKeys.selectedProject, "project-id");
const projectId = getFromLocalStorage<string>(storageKeys.selectedProject);
```

### Navigation
```tsx
import { useCustomNavigation } from "utils/navigator";
import { navRoutes } from "utils/Constants/routes";

const MyComponent = () => {
  const navigate = useCustomNavigation();
  return <Button onClick={() => navigate(navRoutes.Home)}>Go to Home</Button>;
};
```

### Overriding shared component styles (e.g., TabCard)
Never modify shared components for page-specific styling. Use a scoped wrapper:
```tsx
<div id="my-custom-tabs">
    <style>{`
        #my-custom-tabs [data-slot="card"] { border: none; background: transparent; padding: 0; box-shadow: none; }
        #my-custom-tabs [data-slot="card-header"] { height: auto; border-bottom: none; padding: 0; }
        #my-custom-tabs [data-slot="card-content"] { padding: 16px 0 0 0; }
        #my-custom-tabs [data-slot="tabs-list"] { border-bottom: 1px solid var(--color-border-secondary); width: 100%; }
        #my-custom-tabs [data-slot="tabs-trigger"]::after { bottom: -9px !important; }
    `}</style>
    <TabCard title="" tabs={tabs} defaultSelected={activeTab}>
        {(currentTab) => (/* ... */)}
    </TabCard>
</div>
```

### Two-panel layout with shared border
When panels sit side by side sharing a border (no rounded corners at meeting edge):
```tsx
<div className="flex h-full">
    <div
        className="border border-border-secondary p-4 flex flex-col gap-4 overflow-auto"
        style={{ width: "50%", borderRight: "none", borderRadius: "4px 0 0 4px" }}>
        {/* Left panel content */}
    </div>
    <div
        className="border border-border-secondary p-4 flex flex-col gap-4 overflow-auto"
        style={{ width: "50%", borderRadius: "0 4px 4px 0" }}>
        {/* Right panel content */}
    </div>
</div>
```

### CodeMirror editor height fix
CodeMirror with `height="100%"` breaks when parent divs don't propagate height. Fix with scoped CSS:
```tsx
<div id="editor-wrapper" className="border border-border-secondary rounded overflow-hidden" style={{ height: topPx }}>
    <style>{`
        #editor-wrapper > div, #editor-wrapper .cm-theme, #editor-wrapper .cm-editor { height: 100% !important; }
        #editor-wrapper .cm-scroller { padding: 24px !important; }
        #editor-wrapper .cm-gutters { border-right: 1px solid var(--color-border-secondary) !important; margin-right: 16px; padding-right: 12px; background: transparent !important; }
        #editor-wrapper .cm-activeLineGutter, #editor-wrapper .cm-activeLine { background: transparent !important; }
    `}</style>
    <CodeEditor ... hideBorder height="100%" />
</div>
```
