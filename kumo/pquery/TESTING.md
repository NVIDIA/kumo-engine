# TESTING.md — kumo/pquery

Testing guide for the `kumo/pquery` module.

## Framework

Tests use **pytest** and live in `test/pquery/` (not under `kumo/pquery/`). DiskGraph table-generation adapter tests live in `test/data/generic/`.

## Running Tests

### All pquery tests

```bash
.venv/bin/pytest test/pquery
```

### DiskGraph table-generation tests

```bash
.venv/bin/pytest test/data/generic
```

### Single test file

```bash
.venv/bin/pytest test/pquery/test_weight_spec.py
```

### Single test by name

```bash
.venv/bin/pytest test/pquery/test_weight_spec.py -k 'test_parse_weight_spec_column'
```

### With verbose output

```bash
.venv/bin/pytest test/pquery --capture=no --color=yes -vv
```

## Test Structure

```
test/pquery/
├── conftest.py                          # Shared fixtures
├── ast/                                 # AST node tests
├── filter_optimization/                 # Static filter transformer tests
├── test_predictive_query_v2.py          # PredictiveQueryV2 end-to-end
├── test_training_table_validator.py     # TrainingTableValidator checks
├── test_batch_pred_table_validation.py  # PredictionTableValidator checks
├── test_pre_training_table_validation.py # PreTrainTableValidator checks
├── test_pre_prediction_table_validation.py # PrePredTableValidator checks
├── test_pql_batch_pred_generator.py     # PredictionTableGeneratorV2
├── test_weight_spec.py                  # WeightSpec parsing and validation
├── test_pquery_utils.py                 # Utility function tests
└── train_table_generator/               # TrainingTableGeneratorV2 tests
```

## Test Fixtures

### `conftest.py`

The shared `conftest.py` provides query and graph fixtures for pquery tests.

For new table-generation behavior, prefer DiskGraph-backed tests in `test/data/generic/` or focused generator tests under `test/pquery/train_table_generator/`. Do not add tests that depend on retired Python table-generation backends.

## Linting and Type Checking

```bash
.venv/bin/flake8 kumo/pquery && .venv/bin/isort --check kumo/pquery
.venv/bin/pyright -p pyrightconfig.json
```

## When to Write Tests

### Always Write Tests For

- **DiskGraph adapter input changes**: add focused adapter coverage in `test/data/generic/`
- **New generation validators**: add positive (valid input passes) and negative (invalid input raises) tests
- **New `StatsManager` statistics**: add tests verifying the computed value
- **Bug fixes**: every fix must include a regression test reproducing the original issue
- **Feature flag changes**: add tests for both the on and off states

### Test Patterns

1. For compiler behavior, add tests in [the DiskGraph package](../../packages/diskgraph/README.md#python-wheel-builds) and compatible adapter coverage here
2. For generated parquet behavior, add a DiskGraph adapter or end-to-end test
3. For validation changes, test both the happy path and the failure mode
4. For schema changes, validate generated columns, row count, split behavior, and public error messages

## What Must Pass Before Merging

| Check | Command | Required |
|-------|---------|----------|
| All pquery tests pass | `.venv/bin/pytest test/pquery` | Yes |
| DiskGraph table-generation tests pass | `.venv/bin/pytest test/data/generic` | When table-generation behavior changes |
| Lint passes | `flake8` + `isort --check` | Yes |
| Type check passes | `pyright -p pyrightconfig.json` | Yes |
