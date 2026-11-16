# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Enterprise benchmark: run BenchmarkingDatasets via trainer (pquery+fit).

Run from repo root:
python -m examples.paper.benchmark_enterprise --dataset ...
"""

import argparse
import inspect
import os
from dataclasses import replace
from pathlib import Path

import pandas as pd
from kumoapi.model_plan import LossType
from kumoapi.task import TaskType

import kumoai_studio as kumo
from examples.paper.datasets import DATASETS, get_dataset
from examples.paper.datasets.benchmarking_dataset import (
    BenchmarkingDataset,
    limit_entities,
)
from examples.paper.forecasting_runners import save_forecasting_benchmark
from examples.paper.metrics import (
    METRIC_AP,
    METRIC_AUROC,
    METRIC_F1,
    METRIC_MAE,
)


def _apply_metrics_for_task_type(model_plan, task_type: TaskType) -> None:
    """Set metrics and tune_metric on model_plan from dataset task type."""
    if not hasattr(model_plan, "training_job"):
        return
    job = model_plan.training_job
    if not hasattr(job, "metrics"):
        return
    if task_type == TaskType.BINARY_CLASSIFICATION:
        job.metrics = ["acc", METRIC_AUROC, METRIC_AP, METRIC_F1]
        if hasattr(job, "tune_metric"):
            job.tune_metric = METRIC_AUROC
    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
        job.metrics = ["acc", METRIC_F1]
    elif task_type == TaskType.REGRESSION:
        job.metrics = [METRIC_MAE]


def _format_metric_list(metrics_list) -> str:
    """Format a list of Metric(name=..., value=...) as 'name=value, ...'."""
    if not metrics_list:
        return ""
    parts = []
    for m in metrics_list:
        name = getattr(m, "name", str(m))
        value = getattr(m, "value", None)
        if value is not None:
            parts.append(f"{name}={value:.4f}")
        else:
            parts.append(str(m))
    return ", ".join(parts)


def _print_metrics(metrics) -> None:
    """Print train, val, test metrics (test last) in a readable way."""
    print("\nMetrics:")
    for label, attr in [
        ("Train", "training_metrics"),
        ("Val", "validation_metrics"),
        ("Test", "test_metrics"),
    ]:
        lst = getattr(metrics, attr, None)
        if lst is not None:
            print(f"  {label}: {_format_metric_list(lst)}")
    print()


def _save_enterprise_holdout(
    save_dir: Path,
    *,
    dataset_name: str,
    dataset: BenchmarkingDataset,
    holdout: pd.DataFrame,
    max_entities: int | None,
    output_selection: str | None = None,
) -> None:
    tables = limit_entities(dataset.get_forecasting_tables(), max_entities)
    entities = set(tables.test[tables.entity_col].astype(str).unique())
    holdout = holdout[holdout["ENTITY"].astype(str).isin(entities)].copy()
    predictions = dataset.enterprise_holdout_predictions(
        holdout, tables, output_selection=output_selection)
    keys = [tables.entity_col, tables.time_col]
    matched_test = tables.test.merge(predictions[keys].drop_duplicates(),
                                     on=keys, how="inner")
    n_total = len(tables.test)
    n_matched = len(matched_test)
    if n_matched == 0:
        raise ValueError("Enterprise holdout has no matching test labels.")
    print(f"Enterprise holdout: {n_matched}/{n_total} test rows matched.")
    eval_tables = replace(tables, test=matched_test.reset_index(drop=True))
    metrics = dataset.evaluate_forecasting_predictions(predictions,
                                                       eval_tables)
    save_forecasting_benchmark(
        save_dir,
        dataset=dataset_name,
        model="enterprise",
        tables=eval_tables,
        predictions=predictions,
        metrics=metrics,
    )


def _print_plan_summary(train_table_plan, model_plan) -> None:
    """Print a short summary of the training table plan and model plan."""
    lines = ["Plan summary:"]
    split = getattr(train_table_plan, "split", None)
    if split is not None:
        lines.append(f"  split: {split}")
    timeframe_step = getattr(train_table_plan, "timeframe_step", None)
    if timeframe_step is not None:
        lines.append(f"  timeframe_step: {timeframe_step}")
    lag_timesteps = getattr(train_table_plan, "lag_timesteps", None)
    if lag_timesteps is not None:
        lines.append(f"  lag_timesteps: {lag_timesteps}")
    if hasattr(model_plan, "training_job"):
        tj = model_plan.training_job
        if hasattr(tj, "metrics") and tj.metrics is not None:
            lines.append(f"  metrics: {tj.metrics}")
        if hasattr(tj, "tune_metric") and getattr(tj, "tune_metric", None):
            lines.append(f"  tune_metric: {tj.tune_metric}")
    if hasattr(model_plan, "neighbor_sampling"):
        ns = model_plan.neighbor_sampling
        if hasattr(ns, "num_neighbors") and ns.num_neighbors is not None:
            lines.append(f"  num_neighbors: {ns.num_neighbors}")
    if hasattr(model_plan, "optimization"):
        opt = model_plan.optimization
        if hasattr(opt, "max_epochs") and opt.max_epochs is not None:
            lines.append(f"  max_epochs: {opt.max_epochs}")
        if hasattr(opt, "batch_size") and opt.batch_size is not None:
            lines.append(f"  batch_size: {opt.batch_size}")
    print("\n".join(lines))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Enterprise benchmark (trainer.fit via pquery)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ibm_aml",
        help=f"Dataset name or 'all'. Available: {list(DATASETS)}",
    )
    parser.add_argument("--data-root", type=str, default=None,
                        help="Data root (parquet dir or S3 URI)")
    parser.add_argument(
        "--num-neighbors",
        type=str,
        default=None,
        metavar="CSV",
        help="Per-hop neighbor counts as CSV (e.g. --num-neighbors 15,10,5)",
    )
    parser.add_argument(
        "--history-only",
        action="store_true",
        help="Use a dataset's history-only Enterprise graph if supported",
    )
    parser.add_argument(
        "--non-blocking",
        action="store_true",
        help=("Return after launching the training job. Training-table "
              "generation still runs synchronously."),
    )
    parser.add_argument(
        "--max-entities",
        type=int,
        default=None,
        help="Report/save eval metrics on the first N holdout entities.",
    )
    parser.add_argument(
        "--lag-timesteps",
        type=int,
        default=None,
        help="Override forecasting autoregressive lag timesteps.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help=(
            "Write metrics/<dataset>/enterprise.csv and "
            "predictions/<dataset>/enterprise.parquet under this directory."),
    )
    parser.add_argument(
        "--output-selection",
        type=str,
        choices=("mean", "median"),
        default=None,
        help=("Point forecast from the multi-quantile model: 'mean' "
              "(decile-integrated) or 'median'. Default keeps the median."),
    )
    args = parser.parse_args()
    if args.max_entities is not None and args.max_entities <= 0:
        raise ValueError("--max-entities must be positive.")
    if args.lag_timesteps is not None and args.lag_timesteps < 0:
        raise ValueError("--lag-timesteps must be non-negative.")
    if args.save is not None and args.non_blocking:
        raise ValueError("--save requires a blocking Enterprise run.")

    api_url = os.environ.get("KUMO_API_URL", "http://localhost:10002")
    api_key = os.environ.get("KUMO_API_KEY", "test:DISABLED")
    kumo.init(api_key=api_key, url=api_url)

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]

    for name in names:
        dataset = get_dataset(name, data_root=args.data_root)
        print(f"\n--- Enterprise benchmark: {name} ---")
        graph_kwargs = {}
        if args.history_only:
            sig = inspect.signature(dataset.get_enterprise_graph)
            if "history_only" not in sig.parameters:
                raise ValueError(
                    f"Dataset {name!r} does not support --history-only")
            graph_kwargs["history_only"] = True
        graph = dataset.get_enterprise_graph(**graph_kwargs)

        query = dataset.get_pquery()
        pquery = kumo.PredictiveQuery(graph=graph, query=query)
        train_table_plan = pquery.suggest_training_table_plan()
        if (split := dataset.get_enterprise_split()) is not None:
            train_table_plan.split = split
        dataset.configure_train_table_plan(train_table_plan)
        lag_timesteps = args.lag_timesteps
        if lag_timesteps is None and dataset.supports_forecast_holdout():
            lag_timesteps = dataset.get_forecasting_tables().lag_timesteps
        if lag_timesteps is not None:
            train_table_plan.lag_timesteps = lag_timesteps
        train_table = pquery.generate_training_table(plan=train_table_plan)
        model_plan = pquery.suggest_model_plan()
        dataset.configure_model_plan(model_plan)
        task_type = dataset.get_entity_tables().task_type
        _apply_metrics_for_task_type(model_plan, task_type)
        if task_type == TaskType.REGRESSION and hasattr(
                model_plan, "optimization"):
            model_plan.optimization.loss = [LossType.MULTI_QUANTILE]
        if (args.num_neighbors is not None
                and hasattr(model_plan, "neighbor_sampling")
                and hasattr(model_plan.neighbor_sampling, "num_neighbors")):
            per_hop = [int(x.strip()) for x in args.num_neighbors.split(",")]
            model_plan.neighbor_sampling.num_neighbors = [per_hop]

        _print_plan_summary(train_table_plan, model_plan)
        trainer = kumo.Trainer(model_plan)

        result = trainer.fit(
            graph=graph,
            train_table=train_table,
            non_blocking=args.non_blocking,
        )
        if not args.non_blocking:
            _print_metrics(result.metrics())
            if args.save is not None:
                if task_type != TaskType.REGRESSION:
                    raise ValueError("--save currently supports regression.")
                _save_enterprise_holdout(
                    args.save,
                    dataset_name=name,
                    dataset=dataset,
                    holdout=result.holdout_df(),
                    max_entities=args.max_entities,
                    output_selection=args.output_selection,
                )
        else:
            print(f"Training job launched: {result}")


if __name__ == "__main__":
    main()
