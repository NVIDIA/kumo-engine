# kumo-pql

> **Related docs:** [AGENTS.md](AGENTS.md) (contribution rules) | [TESTING.md](TESTING.md) (testing guide)

Parser and validator for Kumo's Predictive Query Language (PQL), a SQL-like DSL for declaratively specifying ML tasks on relational data.

## Overview

The `kumopql` package transforms PQL query strings into validated, structured representations (`ValidatedPredictiveQuery`) that in-tree service and local consumers (`kumo/pquery` and `kumo-core`) use to configure training and prediction jobs. The pipeline is:

1. **Lex** the query string into tokens (ANTLR4)
2. **Parse** tokens into a concrete syntax tree (ANTLR4)
3. **Visit** the parse tree to construct an AST (visitor pattern)
4. **Validate** the AST through a fixed-order chain of 9 validation stages
5. **Emit** a `ValidatedPredictiveQuery` ready for consumption

RFM syntax and validation were removed during inlining because the in-tree API no longer supports them.

## Directory Structure

```
kumopql/
├── __init__.py                           # Package entry point
├── grammar/
│   ├── PQLGrammar.g4                     # ANTLR4 grammar definition (source of truth)
│   ├── PQLGrammarLexer.py                # [GENERATED] Do not edit
│   ├── PQLGrammarParser.py               # [GENERATED] Do not edit
│   ├── PQLGrammarVisitor.py              # [GENERATED] Do not edit
│   ├── PQLGrammar.tokens                 # [GENERATED] Token definitions
│   ├── PQLGrammar.interp                 # [GENERATED] Interpreter data
│   ├── PQLGrammarLexer.interp            # [GENERATED] Lexer interpreter data
│   ├── PQLGrammarLexer.tokens            # [GENERATED] Lexer token definitions
│   ├── README.md                         # Grammar regeneration instructions
│   └── __init__.py
├── parser/
│   ├── __init__.py
│   ├── parser.py                         # PQLParser: main entry point class
│   ├── visitor.py                        # PQLVisitor: parse tree -> AST construction
│   └── error_translator.py               # ANTLR errors -> user-friendly messages
├── validator/
│   ├── __init__.py
│   ├── predictive_query_validator.py     # Orchestrates the 9-step validation chain
│   ├── type_validator.py                 # Type inference and validation (dtypes, stypes)
│   ├── problem_type_validator.py         # RANK/CLASSIFY/FORECAST validation
│   ├── time_range_validator.py           # Time window validation
│   ├── join_validator.py                 # Foreign key inferral and join validation
│   └── utils.py                          # Shared validator utilities
└── rewriting/
    ├── __init__.py
    ├── filter_splitter.py                # Split compound filters for optimization
    ├── bp_filter_transformer.py          # Batch prediction filter transforms
    └── static_filter_splitter.py         # Split static vs. temporal filters
test/
├── conftest.py                           # Fixtures: mock_graph() (5 tables)
├── parser/
│   ├── test_parser.py                    # PQLParser end-to-end tests
│   ├── test_visitor.py                   # PQLVisitor AST construction tests
│   └── test_error_translator.py          # Error message tests
├── validator/
│   ├── test_validator.py                 # Integration tests for full validation chain
│   ├── test_type_validator.py            # Type inference/validation tests
│   ├── test_problem_type_validator.py    # Problem type validation tests
│   ├── test_time_range_validator.py      # Time range validation tests
│   ├── test_join_validator.py            # Join/FK validation tests
└── rewriting/
    ├── test_filter_transformer.py        # Filter splitting tests
    ├── test_bp_filter_transformer.py     # BP filter transform tests
    └── test_static_filter_transformer.py # Static filter splitting tests
```

## ANTLR4 Grammar

The grammar is defined in `kumopql/grammar/PQLGrammar.g4` (~202 lines). It specifies PQL syntax as a context-free grammar parsed by ANTLR4.

### PQL Syntax

```sql
PREDICT <target_column>.<aggregation>
FOR EACH <entity_table>.<entity_key>
[WHERE <conditions>]
[OVER <time_window>]
```

### Key Terminals

| Terminal       | Examples                                              | Purpose                          |
| -------------- | ----------------------------------------------------- | -------------------------------- |
| Column ref     | `TRANS.amount`, `USERS.age`                           | Table-qualified column reference |
| Aggregation    | `SUM`, `COUNT`, `AVG`, `MIN`, `MAX`, `COUNT DISTINCT` | Target aggregation function      |
| Problem type   | `RANK`, `CLASSIFY`, `FORECAST`                        | ML task type modifier            |
| Time window    | `30 DAYS`, `2 WEEKS`, `1 YEAR`                        | Prediction horizon               |
| Logical ops    | `AND`, `OR`, `NOT`                                    | Condition combinators            |
| Comparison ops | `=`, `!=`, `>`, `<`, `>=`, `<=`, `IN`, `LIKE`         | Filter operators                 |
| Constants      | `42`, `3.14`, `'text'`, `TRUE`, `NULL`                | Literal values                   |

### Regenerating Generated Code

```bash
cd kumopql/grammar/
antlr4 -Dlanguage=Python3 PQLGrammar.g4 -visitor -no-listener
```

Requires ANTLR 4.9.3. Never hand-edit generated files (`PQLGrammarLexer.py`, `PQLGrammarParser.py`, `PQLGrammarVisitor.py`).

## Parser Pipeline

```
                    ┌─────────────┐
  Query string ───> │ PQLGrammar  │ ──> Token stream
                    │   Lexer     │
                    └─────────────┘
                          │
                          v
                    ┌─────────────┐
                    │ PQLGrammar  │ ──> Parse tree (CST)
                    │   Parser    │
                    └─────────────┘
                          │
                          v
                    ┌─────────────┐
                    │ PQLVisitor  │ ──> AST dict (unvalidated)
                    │  (visitor)  │
                    └─────────────┘
                          │
                          v
                    ┌─────────────┐
                    │ Validators  │ ──> ValidatedPredictiveQuery
                    │ (9 stages)  │
                    └─────────────┘
```

### PQLParser Class (`parser/parser.py`)

The `PQLParser` class is the main entry point. Key methods:

| Method                         | Purpose                                        |
| ------------------------------ | ---------------------------------------------- |
| `parse_tree()`                 | Lex + parse query string into ANTLR parse tree |
| `to_ast()`                     | Visit parse tree to produce AST dict           |
| `validate()`                   | Check syntax; semantic validation uses `PredictiveQueryValidator`               |
| `to_parsed_predictive_query()` | Full pipeline: parse, visit, validate, return  |

### PQLVisitor (`parser/visitor.py`)

Subclasses the generated `PQLGrammarVisitor` to walk the parse tree and construct AST nodes. Each `visit*` method corresponds to a grammar rule and returns an AST node from `kumoapi.pquery.AST`.

### Error Handling (`parser/error_translator.py`)

Translates raw ANTLR4 error messages into user-friendly PQL error messages with context about what went wrong and how to fix it.

## Validator Ordering

**CRITICAL: Validators run in a fixed order. Each validator depends on outputs from prior validators. Do not reorder.**

| Step | Validator                    | Description                                                                                                   |
| ---- | ---------------------------- | ------------------------------------------------------------------------------------------------------------- |
| 1    | Column validation            | Verifies all referenced columns exist in the graph schema                                                     |
| 2    | Wildcard validation          | Expands wildcard column references (e.g., `*`) into explicit column lists                                     |
| 3    | Type validator               | Infers and validates dtypes and stypes (`infer_dtypes`, `validate_dtypes`, `infer_stypes`, `validate_stypes`) |
| 4    | Problem type validator       | Validates problem type (RANK/CLASSIFY/FORECAST) compatibility with target                                     |
| 5    | Time range validator         | Validates time window specifications and temporal constraints                                                 |
| 6    | Target filter validation     | Validates filters applied to the prediction target                                                            |
| 7    | Join validator               | Infers foreign key paths and validates join relationships between tables                                      |
| 8    | List distinct validator      | Validates LIST DISTINCT operations and entity enumeration                                                     |
| 9    | Target/entity key validation | Validates target column and entity key compatibility                                                          |

The type validator (step 3) uses an abstract `NodeTypeInferrer` hierarchy to handle type inference across different AST node types.

## AST Node Hierarchy

AST nodes are defined in `kumoapi.pquery.AST` (owned by `packages/kumo-api`):

| Node Type          | Purpose                                           |
| ------------------ | ------------------------------------------------- |
| `ASTNode`          | Base class for all AST nodes                      |
| `Column`           | Table-qualified column reference (`table.column`) |
| `Filter`           | WHERE clause filter expression                    |
| `Condition`        | Single comparison condition (`col op value`)      |
| `LogicalOperation` | AND/OR/NOT compound condition                     |
| `Aggregation`      | Target aggregation (SUM, COUNT, etc.)             |
| `Constant`         | Literal value (number, string, boolean, null)     |
| `Join`             | Explicit join specification                       |
| `ArrayDtype`       | Array data type specification                     |
| `DateOffsetRange`  | Temporal offset range for time windows            |
| `LocationInterval` | Geographic interval specification                 |

## Rewriting Module

The rewriting module (`kumopql/rewriting/`) transforms validated AST structures for downstream optimization:

| Module                      | Purpose                                                        |
| --------------------------- | -------------------------------------------------------------- |
| `filter_splitter.py`        | Splits compound filter expressions into components             |
| `bp_filter_transformer.py`  | Transforms batch prediction filters for RHS table filtering    |
| `static_filter_splitter.py` | Separates static filters (time-independent) from temporal ones |

These rewrites occur after validation and prepare the query for efficient execution by the service and local runtime.

## Integration Points

| Dependency               | Direction  | Purpose                                                        |
| ------------------------ | ---------- | -------------------------------------------------------------- |
| `kumoapi`                | Upstream   | AST node types (`kumoapi.pquery.AST`), shared type definitions |
| `antlr4-python3-runtime` | Upstream   | Parser runtime (pinned to 4.9.3)                               |
| `pandas`                 | Upstream   | Data manipulation in validators                                |
| `kumo`                   | Downstream | Consumes `ValidatedPredictiveQuery` via `pquery` module        |
| `kumo-core`              | Downstream | Parses and validates PQL for local execution                   |

## Key Patterns

- **Visitor pattern**: `PQLVisitor` subclasses the ANTLR-generated `PQLGrammarVisitor` to decouple AST construction from grammar rules. Each grammar production maps to a `visit*` method.
- **Abstract NodeTypeInferrer hierarchy**: The type validator uses an abstract base class with concrete implementations for each AST node type, enabling extensible type inference without modifying the validator core.
- **Fixed validation chain**: Validators are deliberately ordered and run sequentially. Each validator may enrich the AST with inferred information (e.g., types, join paths) that subsequent validators depend on.
- **Separation of concerns**: Grammar definition (`.g4`) is separate from generated code, which is separate from the visitor, which is separate from validation. Each layer has a single responsibility.
- **Error translation**: ANTLR errors are intercepted and translated into domain-specific messages before reaching the user, making PQL syntax errors actionable.

## Key Constants

| Constant                 | Value | Purpose                                             |
| ------------------------ | ----- | --------------------------------------------------- |
| `MAX_TIMEFRAMES`         | 10000 | Maximum number of timeframes in a forecasting query |
| `MAX_TOP_K`              | 20    | Maximum top-K value for RANK queries                |

## Entry Points

| Task                  | Start Here                                                           |
| --------------------- | -------------------------------------------------------------------- |
| Parse a PQL query     | `PQLParser().to_parsed_predictive_query(query_string)`               |
| Get parse tree only   | `PQLParser().parse_tree(query_string)`                               |
| Get unvalidated AST   | `PQLParser().to_ast(query_string)`                                   |
| Validate an AST       | `PredictiveQueryValidator(graph).validate_predictive_query(parsed_query)`           |
| Rewrite filters       | `filter_splitter`, `bp_filter_transformer`, `static_filter_splitter` |
| Modify PQL syntax     | Edit `kumopql/grammar/PQLGrammar.g4`, then regenerate                |
| Add a validation rule | Add to appropriate validator in `kumopql/validator/`                 |
| Run tests             | `pytest test/ -vv`                                                   |
