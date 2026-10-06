# Fsutil Guidance

Applies to `io_utils/fsutil/**/*.py`, with naming guidance specifically for `io_utils/fsutil/impl_*.py`.

## Fsutil File Roles and Refactor Direction

- `api.py`: Public fsutil surface area and dispatch entrypoints. Keep this focused on typed APIs, normalization, and shared routing/guardrails.
- `impl_*.py`: Backend-specific behavior registered to the API dispatch functions.
- `uri.py`: URI parsing and backend URI type definitions used by dispatch and implementation modules.
- `common.py`: Shared fsutil helpers and cross-backend utilities that are intentionally backend-agnostic.
- `compat_fsspec.py`: Compatibility adapters for fsspec integration points.
- `logging_policy.py`: fsutil logging controls/policy helpers.
- `__init__.py`: Package exports and module initialization wiring.
- `benchmark/*.py`: Performance and stress tooling only; not part of the runtime fsutil API contract.

### Legacy Modules

- `generic.py` and `s3_filesystem.py` are legacy modules.
- For new feature development, you should not add new logic to these files wherever possible.
- During refactors, you should migrate behavior out of these legacy modules into `api.py` + `impl_*.py` (and related typed/shared modules) and remove legacy code as it is replaced.
- If a narrow compatibility bridge must temporarily remain in a legacy file, you should keep it minimal and document why it cannot yet be moved.

## Type Safety from `api.py`

- Public fsutil entrypoints should expose precise type signatures and typed return values.
- APIs with mode-dependent return types should use `@overload` with `Literal[...]` selectors rather than loose unions at call sites.
- Shared behavior contracts should use typed abstractions (`Protocol`, bounded `TypeVar`, `Generic`, `ABC`) instead of `Any`/`object` placeholders.
- In `api.py`, public signatures should avoid `None | str | ConcreteUri`-style unions; prefer `str` for external inputs, or `AnyURI` when URI objects must be accepted, and avoid concrete backend URI types in API-level annotations.
- Type escapes should be rare and narrowly scoped; if unavoidable, they should include a brief why-comment and the narrowest possible ignore.
- Avoid isinstance checks, type-safe behaviors should be handled by ParseURI and type hints

## Singledispatch Handling from `api.py`

- Public API functions should accept `str` paths and should immediately normalize via `ParseURI(...)` before backend-specific behavior.
- `@singledispatch` base implementations should define only generic routing/validation logic; backend semantics should live in registered `impl_*.py` handlers.
- In `api.py`, acceptable first-argument forms for top-level `@singledispatch` functions are: `str`, `None`-inclusive forms (for example `Optional[str]`), or no annotation.
- Unsupported URI/backends should be handled explicitly in base dispatch with clear failures (typically `NotImplementedError`), not silent fallbacks.
- Backend-specific behavior should remain explicit per registered URI type so differences are deliberate and reviewable.
- Wrapper APIs should preserve sync/async safety boundaries (for example via `@blocking_func` / `@blocking_method`) rather than hiding blocking behavior.

## Interface Pattern: Bare Functions + `impl_*` Registrations

- Most fsutil interfaces should be plain module-level functions in `api.py`, not class-heavy service layers.
- These interface functions should act as stable API entrypoints and should delegate backend-specific work through `@singledispatch`.
- Each backend implementation file (`impl_local.py`, `impl_s3.py`, `impl_gcs.py`, etc.) should register handlers for the relevant API functions.
- New backend behavior should usually be added by registering in the corresponding `impl_*.py` file, not by branching on backend types inside `api.py`.
- This split should keep `api.py` focused on typing, normalization, and shared guardrails, while `impl_*` files hold concrete backend semantics.

## Explicit Dispatch Per Backend

- You should not rely on broad `singledispatch` fallback behavior to define backend semantics.
- You should define backend behavior explicitly in each relevant `impl_*.py` registration so outcomes are deliberate and visible.
- You should keep behavior local to backend implementation modules, similar to other explicit fsutil registrations.
- If a backend does not support an API contract, you should codify that explicitly in that backend registration.

```python
# Avoid: behavior implied by broad fallback
@singledispatch
def some_api(path):
    raise NotImplementedError
```

```python
# Prefer: explicit backend behavior in each impl_*.py
@api.some_api.register
def some_api_backend_a(uri: BackendAURI):
    return perform_backend_a_action(uri)

@api.some_api.register
def some_api_backend_b(uri: BackendBURI):
    return None
```

## Import Placement

- You should not add lazy function-local imports in `io_utils/fsutil` by default.
- You should prefer top-level imports so dependencies are explicit, code paths are clearer, and import behavior is consistent.
- If an import must remain local due to unavoidable environment/bootstrap constraints, you should keep it narrowly scoped and add a brief why-comment.
- You should not use local imports as a style preference or to hide optional behavior.

```python
# Avoid
def run_operation(...):
    import dependency_a
    import dependency_b
    from package import dependency_c
    ...
```

```python
# Prefer
import dependency_a
import dependency_b
from package import dependency_c

def run_operation(...):
    ...
```

## Impl Module Naming

- In `io_utils/fsutil/impl_*.py`, you should avoid underscore-prefixed module-level implementation function names when there is no special reason.
- You should keep names simple and readable because module scoping already namespaces these functions.
- You should prefer plain descriptive names and let registration decorators establish API mapping.
- You should use a leading underscore only when the symbol is truly private helper logic and not a primary implementation entrypoint.

```python
# Avoid
@api.some_interface.register
def _handle_backend_a(uri: BackendAURI):
    return BackendAResult(uri)
```

```python
# Prefer
@api.some_interface.register
def handle_backend_a(uri: BackendAURI):
    return BackendAResult(uri)
```
