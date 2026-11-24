# `tfm.runtime`

One pipeline for every recipe: `setup_run`, then train, infer, evaluate, and
report. Recipes and notebooks call it; no other `tfm` package imports it.

::: tfm.runtime

## Evaluation

::: tfm.runtime.evaluate
    options:
      members:
        - evaluate_run
        - fraud_entity_split
        - fraud_task
        - forecast_task
        - history_window_tasks

## Slot-swap comparison

::: tfm.runtime.compare
