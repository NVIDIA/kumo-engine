# Prediction evidence bundles

Prediction evidence is opt-in and retained separately from model bundles and ordinary prediction exports.
A saved checkpoint or exported Parquet file does not imply that an original scoring graph or explanation was retained.
The new evidence APIs are experimental and require the installed-wheel acceptance gate before deployment qualification.

## Capture and retain

```python
from kumo.engine import EvidenceConfig, PredictionEvidenceBundle

result = trainer.predict(
    prediction_table,
    snapshot=snapshot,
    num_workers=1,
    batch_size=1,
    evidence=EvidenceConfig(
        scoring_graph=True,
        numerical_sensitivity="signed_and_absolute",
        replay=True,
    ),
)
bundle = result.evidence.export("/new/path/evidence")
reopened = PredictionEvidenceBundle.load("/copied/path/evidence")
print(reopened.validate())
for row in reopened.iter_predictions():
    print(row["PREDICTION_ID"], row["TARGET_PRED"])
```

Export before cleaning the prediction scratch directory.
A completed local directory can be copied and reopened without the original model or scratch files.
Loading and validation do not rerun prediction or deserialize a checkpoint.
The destination must not already exist.
A failed publication leaves a separately named partial directory for inspection.
Hashes detect corruption, not an attacker who can replace both data and hashes.

`iter_predictions()` and `iter_numerical_evidence()` read bounded batches or individual records.
`numerical_evidence(prediction_id)`, `scoring_graph(prediction_id)` and `availability(prediction_id)` inspect one row.
Row IDs bind the scoring run, materialized task-row occurrence, entity, cutoff and output coordinate, rather than the position in a scoring batch.
A new scoring run receives new IDs; copying a bundle preserves IDs.

The bundle contains original predictions, an inventory with file sizes and SHA256 hashes, native evidence and provenance, and independent family status counts.
Original graph records preserve sampled occurrences and their source table/PK identities, directed relations, sampled-count metadata, timestamps and supported feature tensors.
Raw upstream source values are not automatically recovered from encoded tensors.
Without a provider, source-value status is `not_provided`; the numerical `raw_value` compatibility field is the tensor-mapper coordinate.
Pass `source_records=...` to `result.evidence.export()` to retain original numerical source values and lineage.
Each mapping supplies `table`, `pkey`, `column`, `source_value`, and a `source` object containing logical `file`, SHA256 `sha256`, physical one-based `row` and source `column`.
The exporter performs a disk-backed join against captured cells and requires exact equality after the declared Float32/Float64 cast, including separate missingness checks.
`iter_source_records()` returns the retained mappings.
The `provided_mapper_verified` status confirms mapping-to-tensor consistency; it does not independently verify external source bytes that were not embedded.

`generation_closed` means the producer has finalized its bookkeeping.
It does not imply all evidence succeeded.
`evidence_complete` and row/family statuses distinguish failed, unsupported, disabled, zero-signal and available results.
Explanation-only failures do not invalidate completed predictions.
A failure of the prediction storage itself remains a prediction failure.

## Numerical interpretation

The signed value is the derivative of the decoded scalar forecast with respect to a transformed numerical input.
The magnitude is its absolute value.
Neither value is an additive contribution, a percentage or a causal effect.
Other inputs, sampled graph structure and missingness are held fixed.

Cells record mapper and transformed values, missingness, dtype, scaling, optional logarithm, imputation and clipping metadata.
A genuine zero input and a missing value imputed to zero are different states.
A successfully computed zero derivative is available evidence; disconnected or nonfinite gradients have explicit unavailable status.
A positive transformed-coordinate derivative at an upper-clipped value does not establish that increasing its raw source value would increase the forecast.
Raw-source derivatives are unsupported.

Target decoding participates in differentiation, including the target encoder's scaling and exponential transformation.
These values explain the raw decoded model output.
Application rounding, clipping negative forecasts to zero or other decision policies have different semantics and are not implicitly explained.

## Supported combinations

| Capability | Initial support | Explicit exclusions |
| --- | --- | --- |
| Original graph and numerical derivatives | CPU scalar MSE regression, one scorer, explicit batch size one, supported tabular tensors | GPU, multiple scorers/seeds, other heads and unsupported tensor encoders |
| Static and temporal capture | Both; actual timestamp and sampling configuration retained | Strict historical availability cannot be inferred from event time alone |
| Reloaded models | Same capture interface after model reload | Model-only reload does not recover prior evidence |
| Graph-only capture | `numerical_sensitivity=None` | Does not fabricate numerical gradients |
| Replay | Existing cohort/sample producer support for the task/module | Replay never satisfies original-pass coverage |
| Controlled removals | Explicit non-seed leaf or membership records | Parent cascades, arbitrary directed-edge removal, resampling and causal claims |
| Intervals | Frozen scalar point predictor plus compatible calibration artifact | Uncalibrated models and unsupported output coordinates |
| Additive attribution | Deferred | Signed derivatives must not be relabeled as contributions |

`on_unsupported="raise"` is the default preflight policy.
`on_unsupported="record"` requests ordinary predictions with an explicit unsupported reason when the requested capture combination is unsupported.
Runtime producer failures remain separate from capability rejection.
The legacy `explanations=True` option requests replay, and `local_sensitivity=True` requests original numerical capture with its legacy one-seed behavior.
Do not combine legacy flags with `EvidenceConfig`.

The native temporal-most-recent sampler admits equality at the cutoff.
Capture preserves the actual graph; it does not remove boundary records after scoring and pretend the modified graph emitted the prediction.
Strict-before applications must construct appropriate historical inputs and test the boundary explicitly.
Each row retains `temporal_validation`, also exposed by `bundle.availability(id)`.
The collector checks retained native source timestamps against the prediction cutoff in Unix nanoseconds; equality or future records produce `violated`, missing required timestamps produce `unavailable`, and static tasks report `not_applicable`.
Violations and unavailable temporal checks prevent complete evidence, while preserving successful predictions and the actual scoring graph.
Optional source-record mappings verify numerical mapper coordinates only; they do not establish event availability, validate source timestamp columns, or replace this native temporal check.

## Controlled record removals

```python
from kumo.engine.evidence import RecordRemovalPlan

plan = RecordRemovalPlan(
    groups=({"table": "events", "pkey": 17},),
    removable_tables=("events",),
    foreign_keys=(("events", "product_id", "products"),),
)
child = bundle.perturb(
    model=loaded_trainer,
    plan=plan,
    prediction_ids=[prediction_id],
    output="/new/path/removal-evidence",
)
```

Supply the complete graph FK schema, not only the relation being removed.
The selected source record's sampled occurrences and actual incident edges are removed together.
All surviving values, node order, temporal anchors and relations are preserved; no replacement neighbors are sampled.
Seed/task removal and tables with inbound schema dependencies are rejected.
Forward and reverse relation types are matched exactly.

The adapter verifies the loaded model against retained role-file hashes, reconstructs the retained tensors, and requires the unmodified graph to reproduce the retained forecast before proceeding.
Each result records baseline, perturbed prediction, signed and absolute response changes, original and perturbed fingerprints, and origin `perturbation`.
Changes need not sum to the prediction because interactions are possible.
The default maximum is 32 selected groups per prediction.
Model staging currently occurs per selected prediction, which adds overhead but bounds model/batch lifetime.

## Calibrate intervals separately

```python
from kumo.engine import CalibrationSpec, PredictionIntervalCalibrator

spec = CalibrationSpec(
    model_sha256=model_sha256,
    scoring_sha256=scoring_sha256,
    coverage=0.90,
    unit="row",
    nonnegative=True,
)
calibrator = PredictionIntervalCalibrator.fit(
    calibration_predictions,
    calibration_outcomes,
    spec=spec,
    population_id="new-calibration-population",
    separation_declaration="Separate from fitting, selection and evaluation",
    excluded_population_ids=previously_used_populations,
    excluded_prediction_ids=previously_used_predictions,
)
calibrator.save("/new/path/calibration")
child = bundle.with_intervals(calibrator, output="/new/path/interval-evidence")
```

The two compatibility hashes must come from the frozen model and scoring policy, not be chosen to match an arbitrary calibration artifact.
The scoring policy includes task, horizon, sampler, preprocessing, batch configuration and decoded-output semantics.
Calibration is a separate artifact; fitting it does not retrain the predictor.
Known population/row overlap is rejected, while unknown historical separation remains a recorded operator declaration.

The method uses absolute calibration residuals and the finite-sample order statistic at `ceil((n + 1) * coverage)`.
An insufficient population reports unavailable intervals rather than choosing an invalid lower quantile.
For related records, select `unit="block"`, provide `block_id` and `member_id` in calibration rows and declare `expected_block_members`.
Each calibration block contributes its maximum residual.
When attaching block intervals, provide `block_column` and `member_column` to `with_intervals()`; both refer to retained prediction columns.

A block interval targets simultaneous coverage of that declared population under exchangeability of blocks.
Chronological splitting alone does not establish exchangeability.
The method does not establish entity-conditional coverage or guarantee coverage under drift.
If `fitted_through_ns` is supplied, calibration rows require cutoff and label-availability timestamps, and predictions must follow calibration label availability.
Use `evaluate()` on a separate forward population to record coverage, widths, interval score and block coverage.

For known nonnegative targets, intervals intersect the nonnegative support.
Empty intersections remain empty, endpoints remain unrounded, and original real-valued endpoints are retained.
These are intervals for target outcomes around the raw predictor, not bounds obtained from historical variation or another entity's history.
A previously inspected holdout must not be reused as fresh calibration or selection data.

## Compatibility and resources

`export_legacy(new_directory)` writes the retained `xai` tree for consumers using the existing local artifact layout.
Application schemas that disallow unknown properties need an explicit adapter for new methods; they must not receive new fields silently.
Ordinary prediction export and model saving still do not include evidence.

Build the merged test wheel using [the supported builder](../packages/kumo-engine/scripts/build_wheel.sh), then run [the installed-wheel smoke harness](../packages/kumo-engine/test/wheel_xai_smoke.py) outside the source checkout.
Source-overlay unit tests are not installed-wheel acceptance evidence.
The [local wheel runbook](evidence-validation.md) gives exact commands, acceptance stages, resource estimates and rollback instructions.

Methodology references: [conformal tutorial](https://arxiv.org/abs/2107.07511), [exchangeability and drift](https://arxiv.org/abs/2202.13415), [conformalized quantile regression](https://arxiv.org/abs/1905.03222), [GNNExplainer](https://arxiv.org/abs/1903.03894), and [Integrated Gradients](https://proceedings.mlr.press/v70/sundararajan17a.html).
The last two are methodological alternatives, not claims of implemented causal or additive attribution.
