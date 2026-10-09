# Supplied Snowflake population contract

The constructors take a `SourceTable`, not a pandas `data_df` and not a table name string.
Use a Snowflake connector scoped to the supplied table, which may differ from the graph's
source schema. The query `pq` and `snapshot` describe the already confirmed graph/task.

## Columns and task shapes

| Field | Contract |
| --- | --- |
| `ENTITY1` | Non-null entity key, with numeric/string type compatible with the graph entity PK |
| `TARGET1` | Required for training; scalar for binary, regression and multiclass. Multilabel uses a scalar string encoded with the target's multicategorical delimiter |
| `TIMESTAMP` | Required temporal anchor; also required for a static task, training or prediction, when the entity table has a time column. Only a static task on an entity without a time column omits it |
| Named weight | Optional training-only finite nonnegative scalar for scalar targets; at least one positive weight is required |
| Named split | Optional training-only values 0=train, 1=validation, 2=test (numeric values or digit strings) |

Training link targets require a typed list of RHS IDs, not a scalar string or arbitrary JSON.
Link weights must be lists matching target lengths; temporal link tasks permit signed weights,
with at least one positive weight. Forecast rows require a scalar `TARGET1` plus precomputed `future_labels` for training,
`autoregression_labels` when lags are enabled, and `year_over_year_labels` when enabled.
The constructors preserve these sequences; they do not derive them from scalar anchors.

Snowflake ARRAY/VARIANT source columns are exposed as strings by this connector. Do not
promise that an ordinary ARRAY label or forecast-sequence column supplies the typed lists
these paths require. Use generated link/forecast populations for those shapes in this
Snowflake workflow, or report the supplied-shape limitation. Do not silently replace an
explicit exact-population request with generated rows. Scalar supplied populations and
entity/anchor prediction populations remain supported for tasks without sequence requirements.
Exact forecast prediction still needs enabled lag and year-over-year sequences.

A prediction population must not contain numbered target columns (`TARGET1`, `TARGET2`, etc.).
Keep evaluation labels separately rather than leaking them into the scoring population.
No public column-renaming map exists on these constructors. Fields need their contract names.

## Train with supplied labels, weights and manual splits

The following consumes an existing scalar-label table `CURATED_LABELS` containing `ENTITY1`,
`TARGET1`, the applicable `TIMESTAMP`, `WEIGHT` and `SPLIT`. Select the source only after its
semantics and split design are confirmed.

```python
import kumo.engine as kumo
from kumoapi.train import TrainingTableSpec
from kumoapi.model_plan import RunMode

spec = TrainingTableSpec(weight_col="WEIGHT", split_col="SPLIT")
training_table = kumo.TrainingTable.from_source_table(
    population_connector["CURATED_LABELS"], spec, pq=pq, snapshot=snapshot,
)
model_plan = pq.suggest_model_plan(snapshot, run_mode=RunMode.FAST, train_table_spec=spec)
trainer = kumo.Trainer(model_plan)
training_result = trainer.fit(training_table, snapshot=snapshot)
```

Use `TrainingTableSpec()` when no weights or manual splits are supplied. Without `split_col`,
normal generation-plan split inference applies; a public `TrainingTableGenerationPlan` may
be passed as `plan=`. Do not assume every task uses the same random split. A supplied split
column overrides split assignment with 0/1/2. Inspect required nonempty splits for the
requested evaluation. Generated training tables are a distinct path and do not accept a
caller-supplied weight column by passing `TrainingTableSpec` to generation.

## Score exact rows

`SCORING_POPULATION` contains entity IDs and row-specific timestamps for a temporal task,
without target columns. It can be used with either a fitted or loaded trainer:

```python
population = kumo.PredictionTable.from_source_table(
    population_connector["SCORING_POPULATION"], pq=pq, snapshot=snapshot,
    plan=trainer.training_table_generation_plan,
)
# Binary example; other task arguments follow kumoai-reference/references/task_journeys.md.
scores = trainer.predict(
    population, snapshot=snapshot, binary_classification_threshold=THRESHOLD,
)
print(population.row_count, scores.num_rows)
```

Do not assume output rows equal population rows: class, ranking and forecast outputs expand
rows. Preserve every requested anchor; do not replace row anchors with one global time.

## Actionable validation

Empty sources, missing required columns, null entities/anchors, incompatible key types,
list/scalar target mismatch, nonfinite or disallowed negative weights and split values outside 0/1/2 are
errors. Report the offending field and required representation before proposing a correction.
Verify that non-null IDs actually exist in the intended graph when exact coverage matters;
schema validation alone does not guarantee membership or prediction coverage.

Population construction rejects a snapshot from another graph. Later fit/predict receives a
snapshot separately, so retain explicit lineage rather than assuming the population object
pins all graph data. A loaded trainer checks the target/entity query contract; do not bypass a
mismatch by rewriting the target. Inspect output key/anchor coverage as well as total rows.
