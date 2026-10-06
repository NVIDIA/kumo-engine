# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-table benchmark entrypoint.

Run from repo root:
python -m examples.paper.benchmark_single_table --dataset ...

Requires the chosen model's deps:
lightgbm, xgboost, tabicl, tabpfn, or autogluon.
"""

import argparse

from examples.paper.datasets import DATASETS, get_dataset
from examples.paper.single_table_runners import run_model, run_model_per_entity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run single-table benchmarks.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ibm_aml",
        help=f"Dataset name. Available: {list(DATASETS)}",
    )
    parser.add_argument(
        "--table-type",
        type=str,
        choices=("entity", "joined"),
        default="entity",
        help="Use get_entity_tables or get_joined_tables.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=("lightgbm", "xgboost", "tabicl", "tabpfn", "autogluon"),
        default="lightgbm",
        help="Model to train and evaluate.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Fraction of train split to use as validation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/val split and subsampling.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Subsample train set to this many rows (optional).",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=None,
        help="Subsample test set to this many rows (optional).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Data root for the dataset (if supported).",
    )
    parser.add_argument(
        "--only-entity-history",
        action="store_true",
        default=False,
        help="Train and predict per entity using only its own history.",
    )
    args = parser.parse_args()
    print(f"Getting dataset: {args.dataset} (data_root={args.data_root})")

    dataset = get_dataset(args.dataset, data_root=args.data_root)
    if args.table_type == "entity":
        sup_tables = dataset.get_entity_tables()
    elif args.table_type == "joined":
        sup_tables = dataset.get_joined_tables()
    else:
        raise ValueError(f"Invalid table type: {args.table_type}")

    # Print statistics about the dataset.
    train_n = min(len(sup_tables.train), args.max_train_rows or float("inf"))
    test_n = min(len(sup_tables.test), args.max_test_rows or float("inf"))
    tc = sup_tables.target_col

    print(f"Dataset: {args.dataset}, table: {args.table_type}, "
          f"model: {args.model}")
    print(f"Train rows: {train_n}, test rows: {test_n}")
    if sup_tables.task_type.is_classification:
        train_dist = sup_tables.train[tc].value_counts().sort_index()
        test_dist = sup_tables.test[tc].value_counts().sort_index()
        train_total = train_dist.sum()
        test_total = test_dist.sum()
        print("  Train: " +
              " | ".join(f"{cls}: {n:,} ({100 * n / train_total:.1f}%)"
                         for cls, n in train_dist.items()))
        print("  Test:  " +
              " | ".join(f"{cls}: {n:,} ({100 * n / test_total:.1f}%)"
                         for cls, n in test_dist.items()))
    else:
        train_vals = sup_tables.train[tc]
        test_vals = sup_tables.test[tc]
        print(f"  Train target: mean={train_vals.mean():.4f}, "
              f"std={train_vals.std():.4f}, "
              f"min={train_vals.min():.4f}, max={train_vals.max():.4f}")
        print(f"  Test  target: mean={test_vals.mean():.4f}, "
              f"std={test_vals.std():.4f}, "
              f"min={test_vals.min():.4f}, max={test_vals.max():.4f}")

    # Run the model.
    if args.only_entity_history:
        if sup_tables.entity_col is None:
            raise ValueError(
                f"Dataset '{args.dataset}' does not define entity_col; "
                "--only-entity-history is not supported for this dataset.")
        val_metrics, test_metrics = run_model_per_entity(
            args.model,
            sup_tables,
            val_size=args.val_size,
            random_state=args.random_state,
            max_train_rows=args.max_train_rows,
            max_test_rows=args.max_test_rows,
        )
    else:
        val_metrics, test_metrics = run_model(
            args.model,
            sup_tables,
            val_size=args.val_size,
            random_state=args.random_state,
            max_train_rows=args.max_train_rows,
            max_test_rows=args.max_test_rows,
        )
    print("Val metrics:", val_metrics)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
