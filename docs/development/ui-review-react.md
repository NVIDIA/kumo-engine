# General React Patterns

> Loaded when the PR diff touches component, hook, page, or store files. These are framework-level best practices enforced in the Kumo UI codebase.

---

### Pattern 33: Separate Business Logic from Presentation

**What triggers it:**
Components that mix API calls, data transformations, and rendering.

**Example of well-refactored code** (from JobsList refactoring):
```typescript
// Hooks handle logic:
const useJobsUrlState = () => { /* URL state sync */ };
const useJobsTableData = () => { /* table config, sorting, columns */ };
const useMergedActiveJobs = () => { /* merge polling + cached jobs */ };

// Component just renders:
const JobsList = () => {
  const { jobs, isLoading, error } = useMergedActiveJobs();
  const { columns, sorting } = useJobsTableData();
  if (isLoading) return <Spinner />;
  if (error) return <ErrorMessage error={error} />;
  return <Table data={jobs} columns={columns} />;
};
```

**Reference:** The JobsList refactoring (hooks: `useJobsUrlState`, `useJobsTableData`, `useMergedActiveJobs` in `ui/src/hooks/jobs/`) demonstrates this pattern well.

---

### Pattern 34: No setState as Props

**What triggers it:**
Passing `setState` directly to child components.

**Example:**
```typescript
// Flagged:
<ChildComponent setParentState={setFormData} />

// Preferred — callback prop:
<ChildComponent onUpdate={(value) => setFormData(prev => ({ ...prev, field: value }))} />
```

---

### Pattern 35: State Synchronization Anti-Pattern

**What triggers it:**
Local state mirroring props via `useEffect`.

**Example:**
```typescript
// Flagged:
const Child = ({ value }) => {
  const [localValue, setLocalValue] = useState(value);
  useEffect(() => { setLocalValue(value); }, [value]);
};

// Preferred — controlled component:
const Child = ({ value, onChange }) => {
  return <input value={value} onChange={e => onChange(e.target.value)} />;
};
```

---

### Pattern 36: Test Timeouts

**What triggers it:**
Arbitrary `setTimeout` or `waitUntilSec` in tests.

**Example:**
```typescript
// Flagged:
await new Promise(resolve => setTimeout(resolve, 2000));

// Preferred:
await waitFor(() => {
  expect(screen.getByText('Updated')).toBeInTheDocument();
});

// For debounced inputs:
vi.useFakeTimers();
await userEvent.type(screen.getByRole('textbox'), 'query');
vi.advanceTimersByTime(300);
vi.useRealTimers();
```

**Reference:** `ui/src/tests/setup.ts` configures Vitest globals. `ui/docs/Coding_Guidelines.md` documents timeout anti-patterns.

---

### Pattern 37: Accessible Test Queries

**What triggers it:**
Brittle selectors instead of accessible queries.

**Example:**
```typescript
// Flagged — brittle:
const checkboxes = screen.getAllByRole('checkbox');
const first = checkboxes[0]; // Array indexing
screen.getByTestId('submit-btn'); // data-testid when better query exists

// Preferred:
screen.getByRole('checkbox', { name: 'Select all columns' });
screen.getByRole('button', { name: 'Submit' });
```

**Priority:** `getByRole` > `getByLabelText` > `getByText` > `getByTestId`

---

### Pattern 38: Nested Ternary Rendering

**What triggers it:**
Deeply nested conditional rendering in JSX.

**Example:**
```typescript
// Flagged:
return (
  <div>
    {isLoading ? <Spinner /> : error ? <Error /> : data ? <List /> : null}
  </div>
);

// Preferred — early returns:
if (isLoading) return <Spinner />;
if (error) return <Error />;
if (!data) return <EmptyState />;
return <List items={data} />;
```

---

### Pattern 39: XSS Prevention

**What triggers it:**
Any `dangerouslySetInnerHTML` usage. The codebase has zero instances.

**Rule:** Never use `dangerouslySetInnerHTML`. If you must render dynamic content, use Markdoc (`@markdoc/markdoc` is already a dependency) or escape the content.

**Reference:** `ui/src/services/axios_client.ts` sanitizes HTML error responses from proxy with `HTML_PATTERN` detection.

---

### Pattern 40: useEffect Dependency Issues

**What triggers it:**
Missing or incorrect dependencies causing stale closures or infinite loops.

**Example:**
```typescript
// Flagged — missing dep:
useEffect(() => { fetchResults(searchTerm); }, []); // searchTerm missing

// Flagged — object dep causing infinite loop:
useEffect(() => { doSomething(options); }, [options]); // if options is new every render

// Preferred:
const stableOptions = useMemo(() => ({ page: 1, limit: 10 }), []);
useEffect(() => { doSomething(stableOptions); }, [stableOptions]);
```

---

### Pattern 41: No New Global CSS Classes

**What triggers it:**
Adding new class definitions to `main.css` instead of using Tailwind utilities.

**Example:**
```typescript
// Flagged — new class in main.css:
.my-custom-card {
  padding: 16px;
  border-radius: 8px;
  background: white;
}

// Should use Tailwind:
<div className="p-4 rounded-lg bg-white">

// Or CSS custom properties for theme values:
<div style={{ color: 'var(--teal)' }}>
// Better: use Tailwind with extended theme
```

**Exception:** Legacy chart/XAI components (`.xai-table-item`, `.histogram-table`, etc.) still use global CSS classes. Don't migrate these without testing those specific components.

**Reference:** `packages/kumo-ui-catalog/src/kumo-ui-catalog.css` (custom properties), `ui/src/main.css` (legacy globals).

---

### Pattern 42: React Keys in Lists

**What triggers it:**
Using array indices as `key` props, or omitting `key` entirely in `.map()` rendering.

**Example:**
```typescript
// Flagged — index key (breaks on reorder, insert, delete):
{items.map((item, index) => <Card key={index} data={item} />)}

// Flagged — missing key:
{items.map(item => <Card data={item} />)}

// Should use stable, unique identifier from the data:
{items.map(item => <Card key={item.id} data={item} />)}

// If no natural ID exists, combine stable fields:
{items.map(item => <Row key={`${item.tableName}-${item.columnName}`} data={item} />)}
```

**Why:** Index keys cause React to reuse DOM nodes incorrectly when items are reordered, inserted, or deleted. This leads to stale state in child components (especially inputs), incorrect animations, and subtle bugs that only appear with dynamic lists.

---

### Pattern 43: Event Listener and Subscription Cleanup

**What triggers it:**
`addEventListener`, `IntersectionObserver`, `MutationObserver`, `WebSocket`, or similar subscriptions in `useEffect` without a corresponding cleanup return.

**Example:**
```typescript
// Flagged — no cleanup:
useEffect(() => {
  window.addEventListener('resize', handleResize);
}, []);

// Should clean up:
useEffect(() => {
  window.addEventListener('resize', handleResize);
  return () => window.removeEventListener('resize', handleResize);
}, []);

// Same pattern for observers:
useEffect(() => {
  const observer = new IntersectionObserver(handleIntersect, options);
  observer.observe(elementRef.current);
  return () => observer.disconnect();
}, []);

// Same for WebSocket:
useEffect(() => {
  const ws = new WebSocket(url);
  ws.onmessage = handleMessage;
  return () => ws.close();
}, [url]);
```

**Why:** Missing cleanup causes memory leaks and stale event handlers that fire after the component unmounts, leading to "setState on unmounted component" warnings and unpredictable behavior.

---

### Pattern 44: Bundle Size Awareness

**What triggers it:**
Importing heavy libraries without tree-shaking, or adding large new dependencies without justification.

**Example:**
```typescript
// Flagged — imports entire lodash (70KB+ gzipped):
import _ from 'lodash';
const sorted = _.sortBy(items, 'name');

// Should use specific import (tree-shakeable):
import sortBy from 'lodash/sortBy';

// Or better — use native when equivalent:
const sorted = [...items].sort((a, b) => a.name.localeCompare(b.name));

// Flagged — heavy date library:
import moment from 'moment'; // ~300KB

// Should use lighter alternative or native:
import { format } from 'date-fns'; // tree-shakeable
// Or native Intl:
new Intl.DateTimeFormat('en-US').format(date);
```

**Why:** Every KB added to the bundle increases load time for all users. Prefer native APIs, tree-shakeable imports, or lighter alternatives. When a large dependency is genuinely needed, justify it in the PR description.

---

## Source Material

| Source | Area |
|--------|------|
| `ui/src/components/@hoc/WithErrorBoundary.tsx` | Regular and forwardRef error boundary HOCs |
| `ui/src/components/Validation/Validation.tsx` | Catalog dialog and alert usage |
| `ui/src/components/Validation/ValidationStatus.tsx` | Catalog validation card usage |
| `packages/kumo-ui-catalog/src/kumo-ui-catalog.css` | CSS custom properties |
| `ui/src/main.css` | Legacy global CSS |
| `ui/docs/Coding_Guidelines.md` | Component architecture, state, testing, naming |
