# Explanation artifact workflow

## Find the correct files

In the existing Snowflake notebook, use the fresh fit's result, or replace `run_dir` with an
explicit retained run directory. Do not change `kumo.configure` while inspecting an existing run.

```python
from pathlib import Path
import json
import pandas as pd
import kumo.engine as kumo

print("Kumo runtime:", kumo.__version__)

run_dir = Path(training_result.bundle_dir).parent.resolve()  # fresh fit only
if not run_dir.is_dir():
    raise FileNotFoundError(f"Training run is unavailable: {run_dir}")

gradient_files = sorted(run_dir.rglob("explain_test/scores.csv"))
column_files = sorted(run_dir.rglob("column_analysis/test/column_analysis.json"))
for kind, paths in (("gradient", gradient_files), ("column analysis", column_files)):
    print(kind, len(paths))
    for path in paths:
        print(path.relative_to(run_dir))
for path in sorted(run_dir.rglob("best_trainer.json")):
    print("selection metadata:", path.relative_to(run_dir))
    print(json.loads(path.read_text()))
```

An empty list is “no matching artifact found in this run”, not “no important features”.
Use `best_trainer.json`, the run configuration, and logs to associate files with the selected
trial. Test explanations use the train+validation refit when available, otherwise the original
fit; do not label them as full-data-refit explanations. Keep trainer and refit paths visible.
Selection metadata may contain paths from the original host after a copy. Do not follow those
paths into another run; resolve against the retained run or report the missing mapping.

Choose an exact file from the inventory, preserving any refit subdirectory. The examples below
use `gradient_path` and `column_path` assigned to those selected `Path` objects; they deliberately
do not default to the first match.

## Read gradient scores

The reviewed CSV schema is `table`, `column`, `score`. Extra columns may exist, for example a
serialized dataframe index; do not use those as features.

```python
def read_gradient_scores(path):
    frame = pd.read_csv(path)
    required = ["table", "column", "score"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Unexpected gradient schema in {path}: missing {sorted(missing)}")
    frame = frame[required].copy()
    frame["score"] = pd.to_numeric(frame["score"], errors="raise")
    if frame["score"].isna().any() or not frame["score"].map(
        lambda value: float("-inf") < value < float("inf")
    ).all():
        raise ValueError(f"Non-finite gradient scores in {path}")
    if (frame["score"] < 0).any():
        raise ValueError(f"Unexpected negative scores in {path}; check producer semantics")
    return frame.sort_values("score", ascending=False).reset_index(drop=True)

scores = read_gradient_scores(gradient_path)
if scores.empty:
    print("No gradient rows were recorded.")
else:
    print("Recorded score sum:", scores["score"].sum())
    display(scores.head(20))
```

The producer aggregates gradients within examples using gradient-based weights, averages across
examples, then normalizes across all table-column scores when the total is positive. A positive
normalized total is approximately one; CSV rounding can change that sum, and an all-zero output
remains zero. Do not renormalize a
top-20 subset and present it as a full-graph percentage. Bar charts should label values “gradient
score”, include table and column, and identify the run/stage.

## Read column cohorts

The JSON contains `task_type` and a `cohorts` list. Each entry identifies `table_name`, `hop`,
`col_name`, and `stype`; its inner `cohorts` list contains bucket labels. `populations` is a list
of relative shares, not integer row counts. `predictions` and optional `targets` are either
lists or mappings from class labels to lists. The latter must be interpreted one class at a time.
Nonempty cohort labels require prediction data; an empty prediction mapping is incomplete,
not an empty cohort result. Strings are not valid series, even when their lengths match.

```python
def read_column_cohorts(path):
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("cohorts"), list):
        raise ValueError(f"Unexpected column-analysis schema in {path}")
    rows = []
    for entry in payload["cohorts"]:
        if not isinstance(entry, dict):
            raise ValueError(f"Unexpected cohort entry in {path}")
        labels = entry.get("cohorts")
        shares = entry.get("populations")
        if not isinstance(labels, list) or not isinstance(shares, list):
            raise ValueError(f"Cohort labels and populations must be lists in {path}")
        predictions = entry.get("predictions")
        targets = entry.get("targets")
        if not isinstance(predictions, (list, dict)):
            raise ValueError(f"Predictions must be a list or class mapping in {path}")
        if targets is not None and not isinstance(targets, (list, dict)):
            raise ValueError(f"Targets must be a list or class mapping in {path}")
        predictions = predictions if isinstance(predictions, dict) else {None: predictions}
        targets = targets if isinstance(targets, dict) else {None: targets}
        if labels and not predictions:
            raise ValueError(f"Missing predictions for nonempty cohorts in {path}")
        if len(shares) != len(labels):
            raise ValueError(f"Cohort/population length mismatch in {path}")
        for name, series in (("predictions", predictions), ("targets", targets)):
            for values in series.values():
                if name == "targets" and values is None:
                    continue
                if not isinstance(values, list):
                    raise ValueError(f"Each {name} series must be a list in {path}")
                if len(values) != len(labels):
                    raise ValueError(f"Cohort/{name} length mismatch in {path}")
        for class_label, values in predictions.items():
            observed = targets.get(class_label)
            for index, label in enumerate(labels):
                rows.append({
                    "table": entry["table_name"], "hop": entry["hop"],
                    "column": entry["col_name"], "class": class_label,
                    "cohort": label, "population_share": shares[index],
                    "mean_prediction": values[index],
                    "mean_target": None if observed is None else observed[index],
                    "prediction_score": entry["prediction_score"],
                    "target_score": entry.get("target_score"),
                })
    return payload.get("task_type"), pd.DataFrame(rows)

task_type, cohorts = read_column_cohorts(column_path)
print("Recorded task:", task_type)
if cohorts.empty:
    print("No column cohorts were recorded.")
else:
    display(cohorts.head(30))
```

Select one table/column/hop/class before plotting means. Preserve cohort ordering from the
producer. `COUNT(*)` is a relationship-count cohort; `OTHER(*)` groups remaining categories;
`N/A` describes missing values. None is a literal user feature to remove automatically.
Do not derive exact counts from population shares without the correct denominator. Scores are
weighted standard deviations before normalization, not measures of predictive accuracy.

## If artifacts are missing

Check, in order: the selected run/path and retained files; runtime generation support;
`disable_explain` in the actual run plan; task/module support; and post-training warning/error
logs. A successful fit alone does not prove explanations succeeded. An older run cannot acquire
files merely by changing its plan now. Report malformed or empty artifacts explicitly instead
of interpreting them as zero importance.

For an already requested **new fit**, the public path is:

```python
# Existing model_plan, train_table and snapshot belong to the requested workflow.
model_plan.training_job.disable_explain = False
trainer = kumo.Trainer(model_plan)
training_result = trainer.fit(train_table, snapshot=snapshot)
```

This runs training; it is not a read-only regeneration command. There is no public standalone
explanation regeneration API here. `Trainer.load` does not restore missing explanation files.
Do not invoke internal `best_trainer.explain` as customer API code.

## Evidence and compatibility

Source: Kumo modules, as shipped in the runtime:

- `kumo.engine.trainer.trainer.Trainer.fit`: post-training dispatch, disable flag, error
  handling, local result fields and run directory.
- `kumo_core.train.trainer.Trainer.set_explainers`, `kumo_core.train.trainer.Trainer.explain`,
  and the explanation directory methods: task/module gates, test-stage refit selection and file
  roots.
- `kumoapi.explain.gradient`: CSV schema and normalization.
- `kumoapi.explain.column_analysis`: cohort schema and separate score normalization.

Source: Kumo ML producer modules:

- `kumoml.enterprise.explain.gradient`: `scores.csv` writer and aggregation.
- `kumoml.common.explain.column_analysis`: `column_analysis/test/column_analysis.json`
  writer, cohort semantics and supported task list.

These are source contracts, not a Snowflake runtime certification. The ML producer revision
must be checked against the executing build before expanding support claims. Local scratch
artifacts can disappear with the notebook runtime and are not in the model publication
allowlist. See [retention](../../kumo-model-persistence/references/retention.md) to
preserve requested files.
