# Public model plans

Generate a plan against the confirmed query and snapshot. This inspects metadata and does not
launch training. For supplied weights, pass the same `TrainingTableSpec` as `train_table_spec=`.

```python
from kumoapi.model_plan import RunMode

plan = pq.suggest_model_plan(snapshot, run_mode=RunMode.FAST)
print(plan)
```

Run modes include FAST, NORMAL, BEST and DEBUG. They change suggested search settings; names
are not quality guarantees or permission for larger workloads. Keep the requested mode. DEBUG is
for diagnostic runs, not for judging model quality.

The suggestion fills unspecified fields from the task, graph and run mode, and some values (an
`"inferred"` fanout, for example) are only resolved during training. Before training, present the
suggested plan as suggested; after training, read the resolved plan
(`version_*/model_plan.json`, see [training diagnostics](../../kumo-training-diagnostics/SKILL.md))
before stating what a run actually used.

`ModelPlan` has five public sections:

| Section | Configuration examples |
| --- | --- |
| `training_job` | `num_experiments`, `metrics`, `tune_metric`, `pruning`, `refit_trainval`, `refit_full`, `disable_explain`, `manual_seed` |
| `column_processing` | `encoder_overrides` by `table.column`; `na_strategy` by semantic type |
| `neighbor_sampling` | `num_neighbors`, `adaptive_sampling`, static-task `sample_from_entity_table` |
| `optimization` | `max_epochs`, step limits, `base_lr`, `batch_size`, `loss`, `optimizer`, `early_stopping`, `lr_scheduler`, weighting |
| `model_architecture` | Task-compatible `model` candidates such as GNNModelPlan and GraphTransformerModelPlan, plus shared head, forecasting and link-prediction settings |

Some tunable fields take lists of candidates, not scalar values: `base_lr=[0.001, 0.003]` offers
two candidates, `base_lr=[0.001]` fixes one, and `num_neighbors=[[16, 8]]` is one two-hop
candidate. Scalars (`max_epochs`, `num_experiments`, `manual_seed`) take one value. Public
dataclasses validate assignments, and task validation further restricts options. Do not copy
arbitrary fields from an internal plan. Generation settings, split plans, lag settings and
`forecast_length` belong to `TrainingTableGenerationPlan`, not `ModelPlan`; changing a model plan
does not regenerate an existing training table.

## Where to look first

This section is tuning guidance, not a package fact: present it as typical effects. When the user asks why a suggested plan underperforms, check labels, evaluation and graph coverage
in [training diagnostics](../../kumo-training-diagnostics/SKILL.md) before proposing settings: a
missing relationship or a poor encoder usually costs more than any field below. Then, in order of
typical effect:

1. `neighbor_sampling.num_neighbors`: hops and fanouts decide which rows the model can see. For
   the GNN, hop count is also the number of message-passing layers.
2. `num_experiments`, `max_epochs`: too small a budget undertrains or leaves the search unexplored.
3. `base_lr`, model `channels`, `batch_size`: stability and capacity. Learning rate is usually the most
   sensitive optimization setting; search it on a multiplicative scale.
4. `loss`, `tune_metric`, class balancing: what is optimized and how a trial is selected.
5. Model `dropout`, `weight_decay`, GNN `aggregation`, normalization, MLP layers: usually smaller
   effects.

These are hypotheses to test on validation under the same protocol, not promised improvements.
Read learning curves to separate overfitting from undertraining before raising capacity or
sampling.

## Smoke tests and requested changes

For a user-requested one-experiment, three-epoch smoke test without refits:

```python
plan.training_job.num_experiments = 1
plan.training_job.refit_trainval = False
plan.training_job.refit_full = False
plan.optimization.max_epochs = 3
```

This limits functional validation, not production modeling quality. For requested candidate
optimization and two-hop sampling, assignment shapes are:

```python
plan.optimization.base_lr = [0.001]
plan.optimization.batch_size = [256]
plan.neighbor_sampling.num_neighbors = [[16, 8]]
```

For a requested GNN architecture candidate, use the public nested model shape:

```python
from kumoapi.model_plan import GNNModelPlan

plan.model_architecture.model = [GNNModelPlan(channels=[128])]
```

Apply these values only when chosen for the task. Candidate count is not experiment count;
`num_experiments` caps trials and the search may not try every combination, and refits add work.
Explain all three before execution.

For an explicitly requested numerical encoder on an existing numerical feature:

```python
from kumoapi.encoder import Numerical

plan.column_processing.encoder_overrides = {
    "customers.account_balance": Numerical(),
}
```

Merge into existing overrides when present instead of erasing unrelated choices. The override
key must match the graph's actual table/column and semantic type. Do not use encoder choices
to hide source type errors or remove task-required label columns.

## Per-relationship fanouts

To give one relationship more or less context than the hop default, use `NumNeighborsConfig`
with one `HopConfig` per hop. Keys are `"<table>.<column>-><table>.<column>"` with the graph's
table names and actual key columns; each direction is a separate key, and unlisted relationships
use the hop's `default`. Assigning `num_neighbors` replaces the suggested value, including its
data-derived `"inferred"` hop-1 entry; say so, or keep that entry in the new `hop1`.

```python
from kumoapi.model_plan import HopConfig, NumNeighborsConfig

hop1 = HopConfig(default=20)
hop1["customers.customer_id->orders.customer_id"] = 40
plan.neighbor_sampling.num_neighbors = [
    NumNeighborsConfig(hop1=hop1, hop2=HopConfig(default=10)),
]
```

Hops are consecutive from `hop1`, at most six. `0` samples none and `-1` samples all, which can
exhaust GPU memory on high-degree relationships. `"inferred"` is valid only on the
target-to-entity connection in `hop1`, where suggested plans put it. With `adaptive_sampling`
off, more than three hops raises a GPU-memory warning.

## Reproducibility and pruning

`training_job.manual_seed` fixes random initialization, sampling and trial selection. GPU kernels
are not forced to be deterministic, so identical runs can still differ slightly, and more so
across devices or library versions. To estimate run-to-run noise, fix every candidate list to one value and repeat with
different seeds: differences smaller than that spread are ties. The search selects one model; it
does not ensemble trials.

`training_job.pruning` (a `PruningConfig` with `min_epochs`, `k`, `min_delta`, `patience`, or
`None`) stops a trial that trails the best `k` earlier trials. That is different from
`optimization.early_stopping` (`EarlyStoppingConfig(min_delta=..., patience=...)` candidates, or
`[None]`), which watches one trial's own validation curve. Tight pruning can stop slow starters
such as low-learning-rate trials.

## Field limits and compatibility

Valid values, not recommended capacity:

| Field | Valid values |
| --- | --- |
| `optimization.batch_size` | candidates 1–2048 |
| `optimization.base_lr` | candidates > 0 |
| `optimization.weight_decay` | candidates ≥ 0 |
| `optimization.min_steps_per_epoch` | 1–4000 |
| `optimization.target_fraction`, `overlap_ratio` | candidates in [0, 1]; static link prediction |
| `HopConfig.default` / a named relationship | −1–128 / −1–512, or `"inferred"` as above |
| `GNNModelPlan` / `GraphTransformerModelPlan` `channels` | even candidates up to 512 |
| `model_architecture.num_pre_message_passing_layers` | 0–4 |
| `model_architecture.num_post_message_passing_layers` | 1–8 |
| `model_architecture.output_embedding_dim` | 1–256; link prediction only |
| `GraphTransformerModelPlan.num_layers` | 1–12 |
| `GNNModelPlan` / `GraphTransformerModelPlan` `dropout` | in [0, 1) |

Set `channels` and `aggregation` on the model candidates in `model_architecture.model`, not on
`model_architecture` itself: the top-level fields are deprecated, accept assignment, and silently
override every candidate's value.

`optimization.loss` takes a list of candidates: loss strings, or config objects whose `name`
argument is required.

```python
from kumoapi.model_plan import FocalLossConfig

plan.optimization.loss = [FocalLossConfig(name="focal", alpha=0.25)]
```

Losses by task (strings, or the named config class):

| Task | `optimization.loss` |
| --- | --- |
| Binary, multilabel classification | `binary_cross_entropy`, `focal` (`FocalLossConfig`) |
| Multiclass classification, multilabel ranking | `cross_entropy`, `ordinal_log` (`OrdinalLogLossConfig`) |
| Link prediction | `cross_entropy`, `binary_cross_entropy` |
| Regression, forecasting | `mae`, `mse`, `huber` (`HuberLossConfig`), `wape`, `normal_distribution`, `negative_binomial_distribution`, `log_normal_distribution`, `multi_quantile`, `QuantileLossConfig(name="quantile", q=...)` (there is no `"quantile"` string) |

Pick the loss that matches the metric the user reports: a mismatched loss can lower that metric
even when training converges.

Validation rejects combinations including:

- `GraphTransformerModelPlan` on any link-prediction task.
- Graph Transformer `channels` not divisible by every `num_heads` candidate.
- `optimization.weight_mode` other than `[None]` unless the task is binary classification or the
  training table has a weight column; never for forecasting; weighted link prediction allows only
  `"weighted_loss"` or `None`.
- `majority_sampling_ratio` on any task but binary classification, or a numeric candidate while
  any `weight_mode` candidate is `None`.
- The `context` or `pairwise` link-prediction `module` on static link prediction.
- `adaptive_sampling=True` with a `-1` fanout.
- `handle_new_target_entities=True` with a `"lookup"` or `"fusion"` `target_embedding_mode` on the
  `shallow_rhs` or `context` link-prediction modules. Suggested link-prediction plans use
  `["lookup"]`, so set `target_embedding_mode = ["feature"]` (or `["shallow_feature"]`) together
  with it.

## Metrics and refits

Metric names belong to `plan.training_job.metrics` and `tune_metric`; include `tune_metric` in
`metrics`. Choose names supported by the actual task and keep tuning/evaluation direction
explicit: AUROC/AP and ranking precision are maximized; RMSE/MAE are minimized. Preserve
averaging, label, K and horizon. No universal metric set or arbitrary Python metric callable is
supported here.

`refit_trainval=True` fits the selected configuration on train plus validation; `refit_full=True`
includes test data too. Full refit metrics are not an independent test estimate. Explanation
artifacts may use the selected train/validation refit. Record the selected trial and refit
when comparing results. Preserve the resolved `plan.as_json()` with run metadata, without
credential material. Execute exactly the requested experiments, then compare using diagnostics.

Source: shipped package (`kumoapi.model_plan`, `kumo_core.model_plan`,
`kumo_core.evaluation.suite`)

## User documentation

The Kumo Engine **Model Plan Guide** is
<https://docs.nvidia.com/sdgm/kumo-engine/user-code-guide/train-a-model/model-plan-guide>; link
it when the user asks where model plans are documented. You may fetch it, or the Kumo Engine user
guide named in `kumoai-reference/SKILL.md`, to answer a user's
conceptual question (what a field does, how settings trade off); fetch nothing else. Never fetch
them or use their content while building or editing a plan or choosing SDK names, signatures or
values: this file and the shipped package are the only source for those. When you answer from
the guide, say so; if it disagrees with this file or the shipped package, say that too and
follow the package.
