<!--
ARCHITECTURE_META
last_updated: 2026-09-29T00:00:00Z
-->

# kumo/pquery

> **Related docs:** [DEVELOPMENT.md](DEVELOPMENT.md) (contribution rules) | [TESTING.md](TESTING.md) (testing guide)

Predictive Query Language orchestration within the Kumo monorepo. The module accepts validated PQL queries, prepares generation options such as temporal anchors and splits, and delegates compilation and physical table generation to DiskGraph.

## Overview

Parsing and semantic validation are owned by the `kumopql` package. The `kumo/pquery` module picks up after validation and handles:

1. **Table-generation orchestration** — route training and prediction table generation through DiskGraph adapters
2. **Generation inputs** — derive temporal anchors, split configuration, task type, target shape, and other compiler options
3. **Generation-time validation** — validate inputs before generation and the resulting table statistics afterward
4. **Statistics** — consume DiskGraph-produced statistics and expose them to the training pipeline

DiskGraph is the sole PQL compiler and physical table generator. Its `DiskGraphPlanCompiler` consumes `ValidatedPredictiveQuery` directly; Kumo does not construct an intermediate Python plan tree.

## Directory Structure

```
kumo/pquery/
├── predictive_query.py           # PredictiveQuery, PredictiveQueryV2 — ML pipeline entry point
├── parser.py                     # Thin wrapper: adds QueryDialect; parse errors raise ValidationException
├── predictive_query_validator.py # Injects connector credentials into validated queries
├── train_table_generator.py      # TrainingTableGeneratorV2: training table orchestration
├── batch_pred_table_generator.py # PredictionTableGeneratorV2: prediction table orchestration
├── stats_manager.py              # StatsManager: column stats, table quality validation
├── generator_utils.py            # Query classification and default time-range helpers
├── formatting.py                 # PQL query formatting / pretty-printing
├── utils.py                      # General utilities (col_name, table_name, etc.)
├── weight_spec.py                # WeightSpec: parsed weight column specification
├── AST/
│   └── ast_node.py               # Feature flag helpers (ALLOW_ARRAY_TARGETS, allow_individual_split_tables)
└── pquery_validators/            # Generation-time validators
    ├── generation_validator.py   # GenerationValidator base class
    ├── pre_train_table_validator.py # Pre-checks before training table generation
    ├── train_table_validator.py  # Post-checks on generated training table
    ├── pre_pred_table_validator.py # Pre-checks before prediction table generation
    └── pred_table_validator.py   # Post-checks on generated prediction table
```

Tests live in `test/pquery/`. DiskGraph table-generation adapter tests live in `test/data/generic/`.

## Pipeline

```
PQL query string
      │
      ▼  [kumopql.PQLParser + PredictiveQueryValidator]
ValidatedPredictiveQuery
      │
      ▼  [TrainingTableGeneratorV2 / PredictionTableGeneratorV2]
Generation inputs: timestamps, task type, split, target shape
      │
      ▼  [diskgraph.DiskGraphPlanCompiler]
DiskGraph-native plan and generated parquet
      │
      ▼  [pquery_validators + StatsManager]
Validated tables + statistics  → consumed by kumo/train and REST ML workflows
```

## Compiler Ownership

`TrainingTableGeneratorV2` and `PredictionTableGeneratorV2` require a `diskgraph_index_path` and delegate to `kumo.data.generic.diskgraph_train_table` and `kumo.data.generic.diskgraph_batch_pred`. The production training adapter delegates to `kumo_core.training.table.generate_train_table()` in `packages/kumo-core/src/kumo_core/training/table.py`, which instantiates `DiskGraphPlanCompiler`. The prediction adapter instantiates the compiler directly. Both pass the validated query to the compiler from the `diskgraph` wheel.

The retired Python planner and `DataProcessingBackend` execution model are not part of production paths. Compiler semantics and plan-node changes belong with [DiskGraph in this repository](../../docs/development/generated-artifacts.md#diskgraph-source), with compatible adapter changes in this repository when the wheel interface changes.

## Key Components

| Component | File | Responsibility |
|-----------|------|----------------|
| `PredictiveQueryV2` | `predictive_query.py` | Top-level ML pipeline entry point: config, train/pred table orchestration, persistence |
| `DiskGraphPlanCompiler` | `diskgraph` wheel | Compiles validated PQL and produces DiskGraph execution plans |
| `TrainingTableGeneratorV2` | `train_table_generator.py` | Prepares training-table generation inputs, invokes the DiskGraph adapter, and consumes native statistics |
| `PredictionTableGeneratorV2` | `batch_pred_table_generator.py` | Prepares prediction-table generation inputs and invokes the DiskGraph adapter |
| `StatsManager` | `stats_manager.py` | Consumes column and table statistics and checks data quality |
| `GenerationValidator` | `pquery_validators/generation_validator.py` | Base validator for generation-time checks |
| `TrainingTableValidator` | `pquery_validators/train_table_validator.py` | Validates generated training-table statistics |

## Integration Points

| Dependency | Direction | Purpose |
|------------|-----------|---------|
| `kumopql` | Upstream | PQL parsing and semantic validation (`PQLParser`, `PredictiveQueryValidator`) |
| `kumoapi` | Upstream | AST node types, `ValidatedPredictiveQuery`, `QueryType`, shared enums |
| `diskgraph` | Downstream compiler/executor | PQL compilation and physical table generation |
| `kumo.data.generic` | Downstream adapter | DiskGraph train/pred table wrappers |
| `kumo_core.training.table` | Shared runtime | Native training-table generation and compiler invocation |
| `kumo.graph` | Upstream | `Graph`, `Column`, `ForeignKey`, `TimeColumn` — graph schema |
| `kumo.connector` | Upstream | `Connector`, `ParquetConnector` — data source access |
| `kumo.task` | Upstream | `Split` hierarchy — temporal train/test split specifications |
| `kumo.train` | Downstream | Consumes generated training tables |
| `rest.backends.ml_backend` | Downstream | Orchestrates table generation as a Temporal workflow stage |

## Entry Points

| Task | Start Here |
|------|-----------|
| Generate training table | `PredictiveQueryV2.generate_train_table()` in `predictive_query.py` |
| Generate prediction table | `PredictiveQueryV2.generate_current_pred_table()` in `predictive_query.py` |
| Parse a PQL string | `parse_query_string_v2()` in `parser.py` |
| Modify compiler semantics | `DiskGraphPlanCompiler`; edit `packages/diskgraph` directly |
| Modify generated parquet behavior | `kumo.data.generic.diskgraph_*` and the `diskgraph` wheel |
| Add a generation-time check | `pquery_validators/` |
| Run tests | `.venv/bin/pytest test/pquery` and `.venv/bin/pytest test/data/generic` |
