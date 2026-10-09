# TESTING.md — kumo-pql

Run these commands from `packages/kumo-pql` after installing the in-tree API and
PQL package: `pip install -e ../kumo-api -e '.[test]'`.

## Framework

Tests use **pytest** and live in the `test/` directory, mirroring the source structure.

## Running Tests

### All Tests

```bash
pytest test/ --capture=no --color=yes -vv
```

### By Module

```bash
# Parser tests (parsing, visitor, error translation)
pytest test/parser/ -vv

# Validator tests (all 9 validation stages)
pytest test/validator/ -vv

# Rewriting tests (filter splitting, transformations)
pytest test/rewriting/ -vv
```

### Single Test

```bash
pytest test/validator/test_type_validator.py::test_name -vv
```

### With Specific Markers or Keywords

```bash
# Run tests matching a keyword
pytest test/ -vv -k 'rank'

# Run only failing tests from last run
pytest test/ -vv --lf
```

## Test Structure

```
test/
├── conftest.py                          # Shared fixtures
├── parser/
│   ├── test_parser.py                   # PQLParser end-to-end tests
│   ├── test_visitor.py                  # PQLVisitor AST construction tests
│   └── test_error_translator.py         # Error message translation tests
├── validator/
│   ├── test_validator.py                # Full validation chain integration tests
│   ├── test_type_validator.py           # Type inference and validation tests
│   ├── test_problem_type_validator.py   # RANK/CLASSIFY/FORECAST tests
│   ├── test_time_range_validator.py     # Time window validation tests
│   ├── test_join_validator.py           # FK inferral and join tests
└── rewriting/
    ├── test_filter_transformer.py       # Filter splitting tests
    ├── test_bp_filter_transformer.py    # Boolean-predicate filter tests
    └── test_static_filter_transformer.py # Static filter splitting tests
```

## Test Fixtures

### `conftest.py`

The shared `conftest.py` provides the `mock_graph()` fixture, which builds a mock graph with 5 tables:

| Table       | Role                         |
| ----------- | ---------------------------- |
| `USERS`     | Entity table (user profiles) |
| `TRANS`     | Transaction events           |
| `STORES`    | Store dimension table        |
| `RETURNS`   | Return events                |
| `LOCATIONS` | Geographic location data     |

This fixture is the standard mock graph used across all test modules. Use it rather than creating ad-hoc graphs unless testing a specific schema scenario.

## Linting and Static Analysis

### Pre-commit Hooks

From the monorepo root, run hooks on the changed PQL paths, or on staged files:

```bash
pre-commit run --files packages/kumo-pql/kumopql/parser/visitor.py  # example changed file
pre-commit run  # staged files
```

Avoid `--all-files` for a routine PQL change; it scans the whole monorepo.

Pre-commit runs the following checks:

| Tool   | Purpose                                  |
| ------ | ---------------------------------------- |
| yapf   | Code formatting                          |
| isort  | Import sorting                           |
| flake8 | Style and error checking                 |
| ruff   | Fast linting (80-char lines, pydocstyle) |
| ty     | Type checking                            |

### Type Checking with mypy

```bash
mypy kumopql/
```

mypy is configured in strict mode with `disallow_untyped_defs=true`. The `grammar/` directory is excluded from type checking since it contains auto-generated code.

All functions must have complete type annotations. Missing annotations will cause mypy failures.

## When to Write Tests

### Always Write Tests For

- **Grammar changes**: Any modification to `PQLGrammar.g4` must include parser tests verifying the new syntax is correctly lexed, parsed, and visited
- **New validator rules**: Each new validation rule needs positive tests (valid queries pass) and negative tests (invalid queries produce correct errors)
- **Modified AST construction**: Changes to `PQLVisitor` must include tests verifying the output AST structure
- **New rewriting rules**: Each rewriting transformation needs before/after tests
- **Bug fixes**: Every bug fix must include a regression test reproducing the original issue
- **Error message changes**: Verify error messages are user-friendly and include actionable context

### Test Patterns

When writing new tests, follow existing conventions:

1. Use the `mock_graph()` fixture for standard scenarios
2. Test both valid and invalid inputs
3. For validators, test boundary conditions around key constants:
   - `MAX_TIMEFRAMES` (10000)
   - `MAX_TOP_K` (20)
4. For error messages, assert on the specific error text to catch regressions

## Monorepo Validation

Run the package tests, changed-file formatting/lint checks, and applicable
consumer tests. Root CI owns the required checks and verifies generated grammar
output. Run pre-commit on changed paths rather than the entire monorepo.

The upstream strict mypy configuration is retained as an additional audit.
It currently reports existing errors against the in-tree API's optional AST
annotations; a successful package test run does not imply a clean mypy run.
