# AGENTS.md — kumo-pql

Prescriptive guide for changes to the in-tree `packages/kumo-pql` package.
Paths and commands below are relative to that package unless stated otherwise.

## Scope

This package owns:

- **PQL grammar** (`kumopql/grammar/PQLGrammar.g4`)
- **Parser** (`kumopql/parser/`) — lexing, parsing, AST construction, error translation
- **Validators** (`kumopql/validator/`) — the 9-step validation chain
- **Rewriting** (`kumopql/rewriting/`) — filter splitting and transformation

Other in-tree components own:

- **AST node definitions** — `packages/kumo-api` (`kumoapi.pquery.AST`)
- **PQL execution/runtime** — `kumo/pquery/` and `packages/kumo-core/src/kumo_core/pquery.py`
- **SDK-level PQL interfaces** — `packages/kumo-sdk` (`kumoai_studio.pquery`)
  and `packages/kumo-engine` (`kumo.engine.pquery`)

## Ownership

Use the current monorepo review ownership for affected paths. Grammar or AST
changes also need review from the in-tree consumers they affect.

## Interfaces

### Upstream Dependencies

| Dependency               | What It Provides                                    | Version Constraint |
| ------------------------ | --------------------------------------------------- | ------------------ |
| `kumoapi`                | AST node types (`kumoapi.pquery.AST`), shared types | See `pyproject.toml`     |
| `antlr4-python3-runtime` | ANTLR4 parser runtime                               | Pinned to `4.9.3`  |
| `pandas`                 | DataFrame operations in validators                  | See `pyproject.toml`     |

### Downstream Consumers

| Consumer  | How It Uses kumo-pql                                               |
| --------- | ------------------------------------------------------------------ |
| `kumo`    | Imports `kumopql` to parse and validate PQL in the `pquery` module |
| `kumo-core` | Parses and validates PQL for local execution (`kumo_core.pquery`) |

## Change Risk Assessment

### Safe Changes

- Adding a new validation rule within an existing validator
- Fixing or improving error messages in `error_translator.py`
- Adding new test cases
- Fixing bugs in existing validators (with tests)
- Updating docstrings or comments
- Adding new rewriting rules (with tests)

### Risky Changes (Require Careful Review)

- **Modifying the grammar** (`PQLGrammar.g4`) — affects all downstream parsing.
- **Removing validation rules** — Enabling previously blocked queries should only be done if they are supported by execution in `kumo` and `kumo-core`.
- **Changing validator ordering** — validators depend on outputs from prior steps; reordering can cause subtle breakage
- **Modifying `PQLVisitor` AST construction** — changes the shape of AST output consumed by validators and downstream systems
- **Changing type inference logic** in `type_validator.py` — affects all subsequent validators

### Needs Approval (Cross-Team Coordination)

- **Adding new PQL keywords or syntax** — requires grammar change + downstream consumer updates in `kumo` and `kumo-core`
- **Breaking the AST output format** — requires coordinated changes in `packages/kumo-api` (AST definitions), `kumo`, and `packages/kumo-core`
- **Bumping `antlr4-python3-runtime` version** — requires regeneration and compatibility testing

## Key Rules

### Grammar and Generated Code

- **NEVER edit generated files**: `PQLGrammarLexer.py`, `PQLGrammarParser.py`, `PQLGrammarVisitor.py`, `*.tokens`, `*.interp`
- **Only edit** `PQLGrammar.g4` to change the grammar
- **Regenerate** after any `.g4` change:
  ```bash
  python scripts/generate_grammar.py
  ```
- Requires ANTLR 4.9.3 (must match the pinned runtime version)
- After regeneration, update `PQLVisitor` in `parser/visitor.py` if new grammar rules were added
- Any changes to the grammar need to be accompanied with updated autocomplete in the UI. See `kumopql/grammar/README.md` here and
  `../../ui/src/utils/PQLParsing/README.md` in this same checkout.

### Validator Ordering

Validators run in this **FIXED ORDER**. Do not reorder without understanding cascading effects:

01. Column validation
02. Wildcard validation
03. Type validator (infer_dtypes, validate_dtypes, infer_stypes, validate_stypes)
04. Problem type validator (RANK/CLASSIFY/FORECAST)
05. Time range validator
06. Target filter validation
07. Join validator (FK inferral)
08. List distinct validator
09. Target/entity key validation

Each step may enrich the AST with inferred data that subsequent steps depend on. For example, the type validator (step 3) infers column types that the problem type validator (step 4) uses to check compatibility.

### Code Style

- **Line length**: 80 characters maximum (enforced by ruff)
- **Type hints**: All functions must have type annotations (`disallow_untyped_defs=true` in mypy)
- **Docstrings**: Google style (ruff pydocstyle convention)
- **Quotes**: Single quotes preferred
- **Imports**: Sorted by isort

### AST Nodes

AST node types (`Column`, `Filter`, `Condition`, `LogicalOperation`, `Aggregation`, `Constant`, `Join`, `ArrayDtype`, `DateOffsetRange`, `LocationInterval`, `ASTNode`) are defined in `kumoapi.pquery.AST`. To modify AST node definitions, coordinate changes in `packages/kumo-api` with PQL and its callers.

## Development Workflow

### Setup

```bash
pip install -e ../kumo-api -e '.[test]'
```

### Running Tests

```bash
pytest test/ --capture=no --color=yes -vv
```

### Linting and Type Checking

```bash
pre-commit run --files $(git ls-files "*.py")
mypy kumopql/                 # Strict type checking (excludes grammar/)
```

### Full Validation Before Submitting

```bash
pip install -e ../kumo-api -e '.[test]'
# Run pre-commit on changed paths as described in TESTING.md.
mypy kumopql/
pytest test/ --capture=no --color=yes -vv
```

### Pull Request Hygiene

Every pull request should address a well defined issue/problem/feature. If a project requires
several steps, they should be submitted as separate follow-up pull requests.
Pull requests should have less than around 300 lines of logic (tests and
comments excluded) unless this is a refactoring pull request that cannot be
reasonably broken down.

Each pull request should contain a brief description of the implemented
features, description of tests conducted to verify it (unit tests or e2e)
and the list of crucial design choices that the reviewers should pay
attention to.

## Change Checklist

Before submitting a PR, verify:

- [ ] All existing tests pass: `pytest test/ -vv`
- [ ] New tests added for any new or modified behavior
- [ ] Review the optional strict mypy audit; see `TESTING.md` for its current limitation.
- [ ] Pre-commit checks run on changed package paths (see `TESTING.md`)
- [ ] If grammar was modified: regenerated files are included in the commit
- [ ] If grammar was modified: `PQLVisitor` updated for new rules
- [ ] If grammar was modified: `ErrorTranslator` heuristics are updated to still correctly translate the auto-generated errors into more user-friendly ones.
- [ ] If validator behavior changed: verified no ordering dependencies broken
- [ ] If AST output shape changed: coordinated with the API, Kumo runtime, and core owners
- [ ] 80-character line limit respected
- [ ] All functions have type annotations
- [ ] Google-style docstrings on public functions

## Adjacent References

| In-tree path | Relevance |
| --- | --- |
| `packages/kumo-api` | AST node definitions (`kumoapi.pquery.AST`) |
| `kumo/pquery` and `rest/driver` | Service parser and validator consumers |
| `packages/kumo-core` | Local parsing, validation, and execution plan |
| `packages/kumo-sdk` | Service SDK PQL interface (`kumoai_studio.PredictiveQuery`) |
| `packages/kumo-engine` | Local PQL interface (`kumo.engine.PredictiveQuery`) |
