Read [architecture](../../ARCHITECTURE.md), [testing](../../docs/development/testing.md),
and [ML development](../../docs/development/ml.md).

# Developing query orchestration

Contributor guidance for the service query-orchestration module. Source paths below are relative to the repository root unless noted otherwise.

## Scope

This module owns:

- **Training table orchestration** — `TrainingTableGeneratorV2`, temporal splits, entity sampling, and DiskGraph compiler inputs
- **Prediction table orchestration** — `PredictionTableGeneratorV2` and DiskGraph compiler inputs
- **Generation-time validation** — `pquery_validators/` (cardinality, NA fraction, empty table checks)
- **Statistics** — `StatsManager` (column stats, table quality metrics)
- **Top-level ML pipeline entry point** — `PredictiveQuery`, `PredictiveQueryV2`

This module does **NOT** own:

- **PQL parsing and semantic validation** — owned by `kumopql` in `packages/kumo-pql`
- **AST node definitions** — owned by `kumoapi` (`kumoapi.pquery.AST`)
- **PQL compilation and physical plan execution** — owned by the in-tree `packages/diskgraph` compiler (`DiskGraphPlanCompiler`)
- **DiskGraph ingest, index, and sampling** — engine source in `packages/diskgraph/`, service integration in `kumo/diskgraph/`
- **ML training** — service adapters in `kumo/train/`, shared execution in `packages/kumo-core`

## Ownership

Use the repository CODEOWNERS for domain review.

## Interfaces

### Upstream Dependencies

| Dependency | What It Provides |
|------------|-----------------|
| `kumopql` | `PQLParser`, `PredictiveQueryValidator` — parsing and semantic validation |
| `kumoapi` | `ValidatedPredictiveQuery`, AST node types, `QueryType`, shared enums |
| `diskgraph` | `DiskGraphPlanCompiler` — compiles validated PQL and generates physical tables |
| `kumo.data.generic` | DiskGraph train/pred table adapters |
| `kumo.graph` | `Graph`, `Column`, `ForeignKey`, `TimeColumn` — relational schema |
| `kumo.task` | `Split` hierarchy — temporal train/test split specifications |
| `kumo.connector` | `Connector` — data source access and credential management |

### Downstream Consumers

| Consumer | How It Uses This Module |
|----------|------------------------|
| `kumo/train/` | Consumes training tables produced by `TrainingTableGeneratorV2` |
| `rest/backends/ml_backend/` | Orchestrates table generation as a Temporal workflow stage |

## Change Risk Classification

### Safe Changes

- Adding a generation-time validation rule in `pquery_validators/` (with tests)
- Fixing error messages or validation thresholds in `StatsManager`
- Adding utility functions to `utils.py` or `generator_utils.py`
- Fixing bugs in existing validators (with regression tests)
- Documentation or comment updates

### Risky Changes — Require Careful Review

- **Changing DiskGraph adapter inputs** — execution changes must remain compatible with the `diskgraph` APIs
- **Changing `TrainingTableGeneratorV2` or `PredictionTableGeneratorV2`** — these are in the critical path for every training and prediction job
- **Modifying `StatsManager`** — statistics are consumed downstream for model configuration and AutoML; changing them may silently break downstream consumers
- **Changing `PredictiveQueryV2` config or persistence format** — affects serialized task configs stored in S3/GCS

### Changes requiring domain-owner review

- **Changing the training/prediction table schema** (column names, types, presence) — consumed by `kumo/train/` and `packages/kumo-core`
- **Modifying `QueryDialect` enum** in `parser.py` or credential injection logic in `predictive_query_validator.py`
- **Removing or renaming public API** on `PredictiveQuery`, `PredictiveQueryV2`
- **Changes to weight column handling** — affects sample weighting in downstream training

## Key Rules

### DiskGraph Compatibility

Live table generation requires a `diskgraph_index_path` and delegates PQL compilation and physical execution to the `diskgraph` APIs. If a change is backend-specific, isolate it in the DiskGraph adapter layer rather than reintroducing retired Python planners or execution backends.

### Compiler Ownership

`DiskGraphPlanCompiler` consumes `ValidatedPredictiveQuery` directly. Compiler semantics and plan-node changes belong in `packages/diskgraph`; this module should only prepare and pass supported compiler inputs. This repository is the [source of truth](../../packages/diskgraph/README.md#python-wheel-builds) for DiskGraph.

### Generation validation

`pquery_validators/` contains both pre-generation checks and post-generation table/statistics checks. Keep AST-level syntax and semantic checks in `packages/kumo-pql`; keep data-dependent generation checks with the appropriate pre/post validator.

### Validate at the earliest applicable stage

To guarantee a good experience and fast iterations, validations should always happen as soon as possible.
Validations in `pquery_validators/` require access to the data (e.g. timestamps or generated training table). If a validation can be performed without data access, it should be done in `kumopql` (if it's a query validation) or model plan validation (if it's validation of arguments).

### Statistics Are Cached

`StatsManager` caches computed stats. If you change what statistics are computed or their key names, existing cached stats may be stale. Consider backward compatibility or explicit cache invalidation.

### Feature Flags in `AST/ast_node.py`

`AST/ast_node.py` parses `ALLOW_ARRAY_TARGETS` and `ENABLE_INDIVIDUAL_SPLIT_TABLES`, both defaulting to false. `allow_individual_split_tables()` exposes the latter. Document names/defaults and inspect consumers when changing these flags.

### Code Style

- **Line length**: 80 characters (enforced by pre-commit)
- **Type hints**: all functions must have type annotations
- **Imports**: sorted by isort; use `# isort: off / on` fencing only when circular imports require it (see existing examples)

## Development Workflow

```bash
# Run pquery unit tests
.venv/bin/pytest test/pquery

# Run DiskGraph table-generation adapter tests
.venv/bin/pytest test/data/generic

# Lint
.venv/bin/flake8 kumo/pquery
.venv/bin/isort --check kumo/pquery

# Type check
.venv/bin/pyright -p pyrightconfig.json
```

### Pull Request Hygiene

Every PR should contain a single change. If a project requires several steps, submit them as separate follow-up PRs. PRs should not exceed 300 lines of logic (tests and comments excluded) unless it is a refactoring PR that cannot be reasonably broken down.

Each PR should contain a brief description of the implemented feature or fix, a description of how it was tested (unit tests or e2e), and the list of key design choices reviewers should pay attention to.

## Change Checklist

Before submitting a PR, verify:

- [ ] `.venv/bin/pytest test/pquery` passes
- [ ] `flake8` and `isort --check` pass on `kumo/pquery`
- [ ] `pyright -p pyrightconfig.json` passes
- [ ] New tests added for any new or modified behavior
- [ ] If modifying DiskGraph adapter inputs: verified compatibility with the `diskgraph` APIs
- [ ] If changing training/prediction table schema: coordinated with `kumo/train/` and `packages/kumo-core` teams
- [ ] If adding a feature flag in `AST/ast_node.py`: env var name and default documented
- [ ] 80-character line limit respected
- [ ] Any added validations and error reports happen at the earliest stage possible.

## Adjacent References

| In-tree module | Relevance |
|---------------|-----------|
| `kumo/data/generic/` | Owns DiskGraph train/pred table adapters |
| `kumopql` | Owns PQL parsing and semantic validation; AST-level checks go there |
| `kumoapi` | Owns AST node definitions; coordinate type changes in `packages/kumo-api` with affected consumers in the same repository |
| `kumo/train/` | Primary consumer of training tables produced here |
