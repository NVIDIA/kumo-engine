# Coding Guidelines

## Table of Contents
1. [Component Architecture](#component-architecture)
2. [State Management](#state-management)
3. [Performance Optimization](#performance-optimization)
4. [Code Organization](#code-organization)
5. [Naming Conventions](#naming-conventions)
6. [Form Handling & User Input](#form-handling--user-input)
7. [Async Operations](#async-operations)
8. [Accessibility & Semantic HTML](#accessibility--semantic-html)
9. [Testing Best Practices](#testing-best-practices)

---

## Component Architecture

### Separate Business Logic from Presentation
Keep components focused on rendering UI. Extract business logic, API calls, and data transformations into custom hooks or separate service layers.

**Instead of:**
```jsx
// Component with mixed concerns
const Component = () => {
  const [data, setData] = useState();
  
  const fetchData = async () => {
    const response = await api.get('/endpoint');
    const transformed = response.data.map(item => ({...}));
    setData(transformed);
  };
  
  return <div>{/* UI */}</div>;
};
```

**Prefer:**
```jsx
// Custom hook for business logic
const useData = () => {
  const [data, setData] = useState();
  // Business logic here
  return { data };
};

// Clean component
const Component = () => {
  const { data } = useData();
  return <div>{/* UI */}</div>;
};
```

### Avoid Deeply Nested Conditional Rendering
Flatten complex conditional rendering logic using early returns or extracted components.

**Instead of:**
```jsx
return (
  <div>
    {isLoading ? (
      <Spinner />
    ) : error ? (
      <Error />
    ) : data ? (
      data.items ? (
        <ItemList items={data.items} />
      ) : (
        <EmptyState />
      )
    ) : null}
  </div>
);
```

**Prefer:**
```jsx
if (isLoading) return <Spinner />;
if (error) return <Error />;
if (!data || !data.items) return <EmptyState />;
return <ItemList items={data.items} />;
```

### Create Reusable Custom Hooks
Extract common patterns into custom hooks.

```jsx
// hooks/useDebounce.js
export const useDebounce = (value, delay) => {
  const [debouncedValue, setDebouncedValue] = useState(value);
  
  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);
    
    return () => clearTimeout(handler);
  }, [value, delay]);
  
  return debouncedValue;
};
```

### Extract Complex Validation Logic
Move validation rules out of components into pure functions or validation schemas.

```javascript
// validators/predictionJob.js
export const validatePredictionJob = (job) => {
  const errors = [];
  
  if (!job.workers || job.workers < 1) {
    errors.push({
      field: 'workers',
      message: 'Workers must be greater than 0'
    });
  }
  
  // More validation rules...
  return errors;
};

// In component
const errors = validatePredictionJob(formData);
```

---

## State Management

### Use useReducer for Complex State
When components need to manage multiple related state variables or complex state transitions, use `useReducer` instead of multiple `useState` calls.

```jsx
// Instead of multiple useState calls
const Component = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);
  const [filter, setFilter] = useState('');
  // Complex state updates scattered throughout
};

// Use useReducer for cohesive state management
const initialState = { loading: false, error: null, data: null, filter: '' };

function reducer(state, action) {
  switch (action.type) {
    case 'FETCH_START':
      return { ...state, loading: true, error: null };
    case 'FETCH_SUCCESS':
      return { ...state, loading: false, data: action.payload };
    case 'FETCH_ERROR':
      return { ...state, loading: false, error: action.payload };
    case 'SET_FILTER':
      return { ...state, filter: action.payload };
    default:
      return state;
  }
}

const Component = () => {
  const [state, dispatch] = useReducer(reducer, initialState);
  // Centralized, predictable state updates
};
```

### Use Discriminated Unions for State Modeling
Model component state more precisely using discriminated unions to make impossible states unrepresentable.

```typescript
// Instead of multiple booleans that can conflict
interface State {
  isLoading: boolean;
  error: Error | null;
  data: Data | null;
}

// Use discriminated unions
type State = 
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'success'; data: Data }
  | { status: 'error'; error: Error };
```

### Use Callback Patterns Instead of Passing setState
Never pass `setState` directly to child components. Use callback functions that encapsulate the state update logic.

**Instead of:**
```jsx
<ChildComponent setParentState={setState} />
```

**Prefer:**
```jsx
<ChildComponent onUpdate={(value) => setState(prevState => ({...prevState, field: value}))} />
```

### Avoid State Synchronization
Don't sync state between parent and child components. Let one component be the source of truth.

**Instead of:**
```jsx
// Child component
const [localValue, setLocalValue] = useState(props.value);
useEffect(() => {
  setLocalValue(props.value);
}, [props.value]);
```

**Prefer:**
```jsx
// Use props directly or lift state up
const handleChange = (newValue) => {
  props.onChange(newValue);
};
```

### Use Refs for Non-Rendering Data
Store values that don't affect rendering in refs instead of state to avoid unnecessary re-renders.

```jsx
const Component = () => {
  const [renderingData, setRenderingData] = useState();
  const intervalId = useRef();
  const previousValue = useRef();
  
  // Use refs for values that don't need to trigger re-renders
  useEffect(() => {
    intervalId.current = setInterval(() => {
      // ...
    }, 1000);
    
    return () => clearInterval(intervalId.current);
  }, []);
};
```

### Use Context over Prop Drilling
Leverage React Context or component composition instead of passing props through multiple levels. If the state should reside at a global level, prefer creating a Zustand store. See debugStore.ts for reference.

```jsx
// Use Context for deeply nested data
const ThemeContext = React.createContext();

// Or use component composition
<Parent>
  <Child data={data} />
</Parent>
```

---

## Performance Optimization

### Memoize Expensive Computations
Use `useMemo` for expensive calculations and `useCallback` for stable function references when performance matters.

```jsx
const expensiveValue = useMemo(() => 
  computeExpensiveValue(data), [data]);

const stableCallback = useCallback((value) => {
  doSomething(value);
}, [dependency]);
```

### Structure useEffect for Clarity
Break complex effects into multiple single-purpose effects. Each effect should handle one concern.

```jsx
// Instead of one effect doing everything
useEffect(() => {
  fetchData();
  subscribeToEvents();
  updateLocalStorage();
  // cleanup...
}, [multiple, dependencies]);

// Prefer focused effects with clear purposes
useEffect(() => {
  fetchData();
}, [dataId]); // Effect 1: Fetch data when ID changes

useEffect(() => {
  const unsubscribe = subscribeToEvents();
  return unsubscribe;
}, []); // Effect 2: Set up event subscription

useEffect(() => {
  localStorage.setItem('key', value);
}, [value]); // Effect 3: Sync with localStorage
```

### Batch State Updates and API Calls
Group related state updates and API calls to reduce renders and network requests.

```jsx
// Instead of multiple sequential calls
const handleSubmit = async () => {
  await updateUser(userData);
  await updatePreferences(preferences);
  await logActivity('user_update');
};

// Batch operations when possible
const handleSubmit = async () => {
  await Promise.all([
    updateUser(userData),
    updatePreferences(preferences),
    logActivity('user_update')
  ]);
};
```

---

## Code Organization

### Use Intention-Revealing, Consistent Names
Names should make code self-explanatory. A reader should understand the purpose of a variable, function, or component without opening its implementation.

**Principles:**
- Prefer **descriptive over short** names (`usePaginatedUsers` > `useUsers`).
- Reflect **behavior and intent**, not type (`getUserData` > `fetch`).
- Keep **consistent patterns** across the codebase (`Form`, `Modal`, `List`, `Card`, `Hook`).
- Use **PascalCase** for components, **camelCase** for variables/functions, **SCREAMING_SNAKE_CASE** for constants.

**Examples:**
```jsx
// ✅ Good
const UserInviteModal = () => { ... }
const usePaginatedUsers = () => { ... }
const isFormDirty = values !== initialValues;

// 🚫 Bad
const Modal1 = () => { ... }
const getData = () => { ... }
const check = () => { ... }
```

### Comments Explain Why, Not What

Write comments when:
- Explaining non-obvious design decisions
- Performance or browser quirks
- Business rules or side effects
- Critical assumptions

Avoid:
- Restating obvious code
- Compensating for unclear naming
- Transient debug info

Below, the first comment explains why a seemingly inefficient re-render exists — it captures context that would otherwise be lost in history or tribal knowledge:

✅ Good:
```javascript
// We intentionally re-render the chart on every data refresh instead of diffing updates.
// This avoids inconsistent tooltip behavior caused by D3's internal state.
useEffect(() => {
  renderChart(data);
}, [data]);
```

🚫 Bad:
```javascript
// Re-render the chart when data changes
useEffect(() => {
  renderChart(data);
}, [data]);
```

### Extract Constants and Magic Values
Move all hardcoded values to constants files or configuration objects.

```javascript
// constants.js
export const PAGINATION = {
  DEFAULT_PAGE_SIZE: 20,
  MAX_PAGE_SIZE: 100
};

export const API_ENDPOINTS = {
  USERS: '/api/users',
  PREDICTIONS: '/api/predictions'
};
```

### Document Complex Business Logic
Add JSDoc comments for complex functions and business rules.

```javascript
/**
 * Validates prediction job parameters
 * @param {PredictionJob} job - The job to validate
 * @returns {ValidationResult} Validation errors and warnings
 * @throws {ValidationError} If critical validation fails
 */
function validatePredictionJob(job) {
  // implementation
}
```

---

## Naming Conventions

### Components and Files
- **UI Components**: Use **PascalCase** for both component and file names (e.g., `UserAvatar.tsx`, `DashboardPanel.jsx`)
  - Component name must match file name exactly
  - Use descriptive suffixes: `Form`, `Modal`, `List`, `Card`, `Button`
- **Non-UI files**: Use **lower_with_underscore** (e.g., `connector_service.js`, `validation_utils.js`)

```
/components/
  UserProfileCard.tsx    // Component: UserProfileCard
  UserProfileList.tsx    // Component: UserProfileList
/utils/
  api_client.js
  date_helpers.js
```

### Variables and Functions
- **camelCase** for variables, functions, and methods
  ```javascript
  const requestBody = "";
  function getUserData() {}
  ```
- **Boolean prefixes**: `is`, `has`, `should` (e.g., `isVisible`, `hasError`, `shouldFetch`)
- **Arrays**: Use plural names (`users`, `selectedIds`, `errorMessages`)
- **Pattern**: Nouns for data, verbs for functions (`userData` vs `loadData`)

### Hooks
- Must start with `use` and clearly express purpose
- Examples: `useAuthToken`, `useDebouncedSearch`, `usePaginatedData`

### Event Handlers
Use consistent patterns for event handler naming:

| Element Type | Pattern | Example |
|-------------|---------|---------|
| Text Input | `on<Name>Change` | `onSourceNameChange` |
| Dropdown | `on<Name>Change` | `onConnectorChange` |
| Button | `on<Action>Click` | `onDeleteClick`, `onRefreshClick` |
| Dialog | `on<Action>Confirm/Cancel` | `onDeleteConfirm`, `onDeleteCancel` |
| Alert | `onAlertClose` | `onAlertClose` |

```javascript
// Examples
this.onSourceNameChange = this.onSourceNameChange.bind(this);
this.onDeleteClick = this.onDeleteClick.bind(this);
this.onDeleteConfirm = this.onDeleteConfirm.bind(this);
```

---

## Form Handling & User Input

### Prefer Controlled Components
Keep form inputs and interactive elements as controlled components with single sources of truth.

```jsx
// Controlled component with clear data flow
const Form = () => {
  const [formData, setFormData] = useState({ name: '', email: '' });
  
  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };
  
  return (
    <input 
      value={formData.name}
      onChange={(e) => handleChange('name', e.target.value)}
    />
  );
};
```

### Implement Debouncing for User Input
Debounce API calls and expensive operations triggered by user input.

```jsx
const debouncedSearch = useMemo(
  () => debounce(searchFunction, 300),
  [searchFunction]
);

// Or use a custom hook
const debouncedValue = useDebounce(searchTerm, 300);
```

### Gate Submission on Server Validation State
When a field runs async server-side validation, lift its error into the parent and include it in the submit/Next button's `disabled` state.
An inline field error alone is not a gate; a still-enabled button lets the user advance with a value the server already rejected, only to fail at the final create.

```jsx
// Parent owns the server-reported error; the input sets it when
// validation fails and clears it on every keystroke.
const [nameServerError, setNameServerError] = useState("");

<GraphNameInput name={name} setName={setName} setError={setNameServerError} />

// The button must reflect the same validity the field displays
<Button disabled={!isValid(name) || nameServerError !== ""}>Next</Button>
```

---

## Async Operations

### Implement Proper Loading and Error States
Every async operation should have corresponding loading, error, and success states with appropriate UI feedback.

```jsx
const Component = () => {
  const { data, isLoading, error } = useQuery();
  
  if (isLoading) return <Spinner />;
  if (error) return <ErrorMessage error={error} />;
  return <DataDisplay data={data} />;
};
```

### Handle Async Operations Properly
Avoid arbitrary timeouts and use proper async patterns.

**Never do this:**
```jsx
await waitUntilSec(2000); // Arbitrary wait
setTimeout(() => setShowModal(false), 500); // Code smell
```

**Instead:**
```jsx
// Use proper async patterns
const closeModal = async () => {
  await animationComplete();
  setShowModal(false);
};

// Or use transition events
const handleTransitionEnd = () => {
  setShowModal(false);
};
```

### Guard Mutations Against Duplicate Submission
Disable the triggering button while its request is in flight, and guard the handler itself with an in-flight ref.
A loading overlay only blocks pointer events, so the still-focused button re-fires on keyboard Enter, and a queued activation can arrive before the disabled re-render commits.

```jsx
const submitInFlightRef = useRef(false);

const onSaveClick = async () => {
  if (submitInFlightRef.current) return; // Ignore re-entrant activations
  submitInFlightRef.current = true;
  setLoading(true);
  try {
    await saveResource();
  } finally {
    submitInFlightRef.current = false;
  }
};

// Disable the button too; the overlay alone is not a guard
<Button onClick={onSaveClick} disabled={isInvalid || loading}>Save</Button>
```

---

## Accessibility & Semantic HTML

### Use Semantic HTML and Accessibility Attributes
Always include proper ARIA labels, roles, and semantic HTML elements.

```jsx
// Good accessibility practices
<nav aria-label="Main navigation">
  <ul>
    <li><a href="/home">Home</a></li>
    <li><a href="/about">About</a></li>
  </ul>
</nav>

<button 
  aria-label="Close dialog"
  aria-pressed={isPressed}
  onClick={handleClose}
>
  <CloseIcon aria-hidden="true" />
</button>

<form>
  <label htmlFor="email">Email Address</label>
  <input 
    id="email" 
    type="email" 
    required
    aria-describedby="email-error"
  />
  <span id="email-error" role="alert">
    Please enter a valid email
  </span>
</form>
```

---

## Testing Best Practices

### Write Focused, Single-Purpose Tests
Each test should verify one specific behavior.

**Instead of:**
```jsx
it("Should render Create New Table with unnamed S3 Connector for Duplicate entity", async () => {
  // Setup mocks
  // Render component
  // Test page load
  // Test API calls
  // Test form interactions
  // Test navigation
  // 100+ lines of test code
});
```

**Prefer:**
```jsx
it("should call validate API when Add Table is clicked", async () => {
  const { getByRole } = renderWithProviders(<ConnectTables />);
  const addButton = getByRole("button", { name: "Add table" });
  
  await userEvent.click(addButton);
  
  expect(mockAxiosInstance.post).toHaveBeenCalledWith(
    "v1/tables/validate",
    expectedArgs
  );
});
```

### Structure Tests with AAA Pattern
Organize tests with clear Arrange, Act, Assert sections.

```jsx
describe("ConnectTables", () => {
  it("should display error when table creation fails", async () => {
    // Arrange
    const errorMessage = "Table name already exists";
    mockAxiosInstance.post.mockRejectedValue({ 
      response: { data: errorMessage }
    });
    const { getByRole, getByText } = renderComponent();
    
    // Act
    const addButton = getByRole("button", { name: "Add table" });
    await userEvent.click(addButton);
    
    // Assert
    await waitFor(() => {
      expect(getByText(errorMessage)).toBeInTheDocument();
    });
  });
});
```

### Extract Test Setup and Utilities
Create reusable test utilities instead of duplicating setup code.

```jsx
// testUtils.js
export const setupTableTest = () => {
  const mocks = {
    navigate: vi.fn(),
    axiosGet: vi.spyOn(axios, 'get'),
    axiosPost: vi.spyOn(axios, 'post')
  };
  
  const renderComponent = (props = {}) => {
    return render(
      <TestProviders>
        <ConnectTables {...props} />
      </TestProviders>
    );
  };
  
  return { mocks, renderComponent };
};

// In test
const { mocks, renderComponent } = setupTableTest();
```

### Use Testing Library Best Practices
Follow Testing Library's guiding principles - test user behavior, not implementation details.

**Avoid:**
```jsx
// Testing implementation details
const checkBoxes = screen.getAllByRole("checkbox");
const checkBox = checkBoxes[0]; // Brittle array indexing
```

**Prefer:**
```jsx
// Test by accessible name or label
const selectAllCheckbox = screen.getByRole("checkbox", { 
  name: "Select all columns" 
});
```

### Eliminate Arbitrary Timeouts
Fix the root cause instead of adding delays.

**Common timeout anti-patterns and fixes:**

| Anti-pattern | Solution |
|-------------|----------|
| Debounced inputs | Mock the debounce in tests or use `vi.useFakeTimers()` |
| Animations | Disable or mock animations in test environment |
| Sequential API calls | Use `waitFor` with proper assertions for each call |
| State updates | Trust React's batching and Testing Library's async utilities |

```jsx
// For animations, mock or disable them in tests
beforeEach(() => {
  vi.stubGlobal('requestAnimationFrame', (cb) => cb());
});

// For debounced inputs
beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.runOnlyPendingTimers();
  vi.useRealTimers();
});
```

### Red Flags in Tests
Watch out for these indicators of test/code problems:
- Any use of `setTimeout` in tests
- Arbitrary `wait` or `sleep` functions
- Tests with timeouts > 5000ms (except for legitimate integration tests)
- Multiple `waitFor` calls for the same action
- Using `waitFor` without specific assertions inside
