# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Forecasting benchmark entrypoint.

Run from repo root:
python -m examples.paper.benchmark_forecasting --dataset ...
"""

import argparse
from pathlib import Path

from examples.paper.datasets import DATASETS, get_dataset
from examples.paper.datasets.benchmarking_dataset import limit_entities
from examples.paper.forecasting_runners import (
    run_forecasting_model,
    save_forecasting_benchmark,
)


def _add_arima_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--arima-order",
        type=str,
        default="auto",
        help=("ARIMA order as p,d,q (e.g. 2,0,1), or 'auto' to select per "
              "entity by AIC over a predefined candidate set. Only used with "
              "--model arima."),
    )
    parser.add_argument(
        "--no-arima-exog",
        action="store_true",
        default=False,
        help="Disable circular temporal exogenous features for ARIMA.",
    )


def _add_tabpfn_ts_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tabpfn-ts-context-length",
        type=int,
        default=4096,
        help="Maximum TabPFN-TS context length per entity.",
    )


def _add_chronos2_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--chronos2-batch-size",
        type=int,
        default=100,
        help="Chronos-2 inference batch size.",
    )
    parser.add_argument(
        "--chronos2-context-length",
        type=int,
        default=None,
        help="Chronos-2 max context length.",
    )


def _add_timesfm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--timesfm-batch-size",
        type=int,
        default=32,
        help="TimesFM 2.5 per-core batch size.",
    )
    parser.add_argument(
        "--timesfm-context-length",
        type=int,
        default=8064,
        help="TimesFM 2.5 max context length.",
    )


def _add_toto2_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--toto2-model",
        type=str,
        default="Datadog/Toto-2.0-313m",
        help="Toto 2.0 checkpoint (HF repo id), e.g. Datadog/Toto-2.0-2.5B.",
    )
    parser.add_argument(
        "--toto2-context-length",
        type=int,
        default=4096,
        help="Toto 2.0 context length (history steps fed per entity).",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run forecasting benchmarks.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="metr_la",
        help=(f"Dataset name. Available: {list(DATASETS)}. The dataset must "
              "implement get_forecasting_tables()."),
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Data root for the dataset, if supported.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=("arima", "tabpfn_ts", "chronos2", "timesfm2_5", "toto2",
                 "seasonal_naive"),
        default="arima",
        help="Forecasting model to evaluate.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Cap training rows, distributed across entities.",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=None,
        help="Cap test rows, distributed across entities.",
    )
    parser.add_argument(
        "--max-entities",
        type=int,
        default=None,
        help="Cap entities for quick forecasting smoke tests.",
    )
    parser.add_argument(
        "--output-selection",
        type=str,
        choices=("mean", "median"),
        default=None,
        help="Point-forecast type for distributional models. If unset, "
        "each model keeps its own default.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers. Currently used by ARIMA.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help=("Write metrics/<dataset>/<model>.csv and "
              "predictions/<dataset>/*.parquet under this directory."),
    )
    _add_arima_args(parser)
    _add_tabpfn_ts_args(parser)
    _add_chronos2_args(parser)
    _add_timesfm_args(parser)
    _add_toto2_args(parser)
    args = parser.parse_args()
    if args.arima_order == "auto":
        arima_order = "auto"
    else:
        arima_order = tuple(
            int(part.strip()) for part in args.arima_order.split(","))
        if len(arima_order) != 3:
            raise ValueError(
                "--arima-order must have exactly three values, e.g. 2,0,1")
    print(f"Getting dataset: {args.dataset} (data_root={args.data_root})")

    dataset = get_dataset(args.dataset, data_root=args.data_root)
    try:
        tables = dataset.get_forecasting_tables()
    except NotImplementedError as exc:
        raise ValueError(
            f"Dataset '{args.dataset}' does not support forecasting "
            "benchmarks. Implement get_forecasting_tables() to use "
            "benchmark_forecasting.py.") from exc
    tables = limit_entities(tables, args.max_entities)

    # Print statistics about the dataset.
    train_n = min(len(tables.train), args.max_train_rows or float("inf"))
    context_n = (0 if tables.context is None else len(tables.context))
    test_n = min(len(tables.test), args.max_test_rows or float("inf"))
    tc = tables.target_col

    print(f"Dataset: {args.dataset}, model: {args.model}")
    print(
        f"Fit rows: {train_n}, context rows: {context_n}, test rows: {test_n}")
    print(f"Entity column: {tables.entity_col}, time column: "
          f"{tables.time_col}, target column: {tc}, freq: {tables.freq}")
    train_vals = tables.train[tc]
    test_vals = tables.test[tc]
    print(f"  Train target: mean={train_vals.mean():.4f}, "
          f"std={train_vals.std():.4f}, "
          f"min={train_vals.min():.4f}, max={train_vals.max():.4f}")
    print(f"  Test  target: mean={test_vals.mean():.4f}, "
          f"std={test_vals.std():.4f}, "
          f"min={test_vals.min():.4f}, max={test_vals.max():.4f}")

    model_kwargs: dict = {}
    feature_preset = tables.feature_preset
    if args.model == "arima":
        model_kwargs["order"] = arima_order
        model_kwargs["use_exog"] = not args.no_arima_exog
    elif args.model == "tabpfn_ts":
        model_kwargs["max_context_length"] = args.tabpfn_ts_context_length
        model_kwargs["feature_preset"] = feature_preset
    elif args.model == "chronos2":
        model_kwargs["batch_size"] = args.chronos2_batch_size
        model_kwargs["context_length"] = args.chronos2_context_length
        model_kwargs["feature_preset"] = feature_preset
    elif args.model == "timesfm2_5":
        model_kwargs["batch_size"] = args.timesfm_batch_size
        model_kwargs["context_length"] = args.timesfm_context_length
        model_kwargs["feature_preset"] = feature_preset
    elif args.model == "toto2":
        model_kwargs["model_name_or_path"] = args.toto2_model
        model_kwargs["context_length"] = args.toto2_context_length

    if args.output_selection is not None and args.model in (
            "tabpfn_ts", "chronos2", "timesfm2_5", "toto2"):
        model_kwargs["output_selection"] = args.output_selection

    if args.model == "seasonal_naive":
        model_kwargs["seasonal_period"] = tables.seasonal_period
    predictions = run_forecasting_model(
        args.model,
        tables,
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
        n_jobs=args.n_jobs,
        model_kwargs=model_kwargs,
    )
    test_metrics = dataset.evaluate_forecasting_predictions(
        predictions,
        tables,
    )
    print("Test metrics:", test_metrics)
    if args.save is not None:
        save_forecasting_benchmark(
            args.save,
            dataset=args.dataset,
            model=args.model,
            tables=tables,
            predictions=predictions,
            metrics=test_metrics,
        )


if __name__ == "__main__":
    main()
