# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from autogluon.tabular import TabularPredictor
from kumoapi.task import TaskType
from tabicl import TabICLClassifier
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tqdm import tqdm

from examples.paper.datasets import SupervisedTables
from examples.paper.metrics import compute_metrics
from examples.paper.paper_utils import (
    prepare_splits,
    prepare_splits_dataframes,
)


def run_gbdt(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    task_type: TaskType,
) -> tuple[dict[str, float], dict[str, float]]:
    is_binary = task_type == TaskType.BINARY_CLASSIFICATION
    is_multiclass = task_type == TaskType.MULTICLASS_CLASSIFICATION
    num_class = None
    if is_binary:
        obj_lgb, metric_lgb = "binary", "auc"
        obj_xgb, eval_metric = "binary:logistic", "auc"
    elif is_multiclass:
        obj_lgb, metric_lgb = "multiclass", "multi_logloss"
        obj_xgb, eval_metric = "multi:softprob", "mlogloss"
        all_labels = np.concatenate([y_train, y_val, y_test])
        num_class = int(all_labels.max()) + 1
        y_train = y_train.astype(int)
        y_val = y_val.astype(int)
        y_test = y_test.astype(int)
    else:
        obj_lgb, metric_lgb = "regression", "mae"
        obj_xgb, eval_metric = "reg:squarederror", "mae"

    if model_name == "lightgbm":
        params = {
            "objective": obj_lgb,
            "metric": metric_lgb,
            "verbosity": -1,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "random_state": 42,
        }
        if num_class is not None:
            params["num_class"] = num_class
        train_set = lgb.Dataset(x_train, label=y_train)
        val_set = lgb.Dataset(x_val, label=y_val, reference=train_set)
        model = lgb.train(
            params,
            train_set,
            num_boost_round=500,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )
        val_out = model.predict(x_val)
        test_out = model.predict(x_test)
    else:
        params = {
            "objective": obj_xgb,
            "eval_metric": eval_metric,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "random_state": 42,
        }
        if num_class is not None:
            params["num_class"] = num_class
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dval = xgb.DMatrix(x_val, label=y_val)
        dtest = xgb.DMatrix(x_test)
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=20,
            verbose_eval=False,
        )
        val_out = model.predict(dval)
        test_out = model.predict(dtest)

    if is_binary:
        val_proba = np.column_stack([1 - val_out, val_out])
        test_proba = np.column_stack([1 - test_out, test_out])
        val_metrics = compute_metrics(y_val, val_proba, None, task_type)
        test_metrics = compute_metrics(y_test, test_proba, None, task_type)
    elif is_multiclass:
        val_proba = val_out
        test_proba = test_out
        val_metrics = compute_metrics(y_val, val_proba, None, task_type)
        test_metrics = compute_metrics(y_test, test_proba, None, task_type)
    else:
        val_metrics = compute_metrics(y_val, None, val_out, task_type)
        test_metrics = compute_metrics(y_test, None, test_out, task_type)
    return val_metrics, test_metrics


def run_tabular_fm_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int | None = None,
    max_train_rows: int = 10_000,
) -> tuple[dict[str, float], dict[str, float]]:
    """Run TabPFNRegressor. TabICL does not support regression."""
    if len(x_train) > max_train_rows:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x_train), max_train_rows, replace=False)
        x_train = x_train[idx]
        y_train = y_train[idx]
        print(f"TabPFN: subsampled train to {max_train_rows} rows")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = TabPFNRegressor(
        random_state=42,
        device=device,
        ignore_pretraining_limits=True,
    )
    model.fit(pd.DataFrame(x_train), y_train)

    if batch_size is not None:

        def predict_batched(X: np.ndarray) -> np.ndarray:
            if len(X) <= batch_size:
                return model.predict(pd.DataFrame(X))
            out = []
            for start in tqdm(range(0, len(X), batch_size), desc="Predicting"):
                end = min(start + batch_size, len(X))
                out.append(model.predict(pd.DataFrame(X[start:end])))
            return np.concatenate(out)

        test_pred = predict_batched(x_test)
    else:
        test_pred = model.predict(pd.DataFrame(x_test))

    test_metrics = compute_metrics(y_test, None, test_pred,
                                   TaskType.REGRESSION)
    return {}, test_metrics


def run_tabular_fm_classification(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    task_type: TaskType,
    batch_size: int | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Run TabICLClassifier or TabPFNClassifier."""
    # TabICL / TabPFN expect integer class labels.
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if model_name == "tabicl":
        model = TabICLClassifier(random_state=42, device=device)
    else:
        model = TabPFNClassifier(
            random_state=42,
            device=device,
            ignore_pretraining_limits=True,
        )

    model.fit(pd.DataFrame(x_train), y_train)

    if batch_size is not None:

        def predict_proba_batched(X: np.ndarray) -> np.ndarray:
            if len(X) <= batch_size:
                return model.predict_proba(pd.DataFrame(X))
            out = []
            for start in tqdm(range(0, len(X), batch_size), desc="Predicting"):
                end = min(start + batch_size, len(X))
                out.append(model.predict_proba(pd.DataFrame(X[start:end])))
            return np.vstack(out)

        test_proba = predict_proba_batched(x_test)
    else:
        test_proba = model.predict_proba(pd.DataFrame(x_test))

    test_metrics = compute_metrics(y_test, test_proba, None, task_type)
    return {}, test_metrics


def run_autogluon(
    sup_tables: SupervisedTables,
    val_size: float,
    random_state: int,
    max_train_rows: int | None,
    max_test_rows: int | None,
    presets: str = "medium_quality",
) -> tuple[dict[str, float], dict[str, float]]:
    (
        train_df,
        val_df,
        test_df,
        target_col,
        task_type,
    ) = prepare_splits_dataframes(
        sup_tables,
        val_size=val_size,
        random_state=random_state,
        max_train_rows=max_train_rows,
        max_test_rows=max_test_rows,
    )

    # Replace datetime features with simple calendar-derived numeric
    # features so we preserve temporal signal without triggering
    # AutoGluon's datetime parsing bug.
    datetime_cols: list[str] = []
    for col in list(train_df.columns):
        if col == target_col:
            continue
        if np.issubdtype(train_df[col].dtype, np.datetime64):
            datetime_cols.append(col)
    for col in datetime_cols:
        for df in (train_df, val_df, test_df):
            if col not in df.columns:
                continue
            dt = df[col]
            df[f"{col}_year"] = dt.dt.year
            df[f"{col}_month"] = dt.dt.month
            df[f"{col}_day"] = dt.dt.day
            df[f"{col}_dow"] = dt.dt.dayofweek
        train_df = train_df.drop(columns=[col])
        val_df = val_df.drop(columns=[col])
        test_df = test_df.drop(columns=[col])

    # Ensure labels are int for classification so AutoGluon infers correctly.
    if task_type in (
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
    ):
        for df in (train_df, val_df, test_df):
            df[target_col] = df[target_col].astype(int)

    if task_type == TaskType.BINARY_CLASSIFICATION:
        problem_type = "binary"
    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
        problem_type = "multiclass"
    else:
        problem_type = "regression"
    predictor = TabularPredictor(
        label=target_col,
        problem_type=problem_type,
    ).fit(
        train_df,
        tuning_data=val_df,
        presets=presets,
    )

    val_features = val_df.drop(columns=[target_col])
    test_features = test_df.drop(columns=[target_col])
    y_val = val_df[target_col].to_numpy()
    y_test = test_df[target_col].to_numpy()

    if task_type == TaskType.BINARY_CLASSIFICATION:
        val_proba = predictor.predict_proba(val_features)
        test_proba = predictor.predict_proba(test_features)
        # Ensure columns ordered [0, 1] so [:, 1] is P(positive class).
        order = sorted(val_proba.columns)
        val_proba_np = val_proba[order].to_numpy()
        test_proba_np = test_proba[order].to_numpy()
        val_metrics = compute_metrics(y_val, val_proba_np, None, task_type)
        test_metrics = compute_metrics(y_test, test_proba_np, None, task_type)
    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
        val_proba = predictor.predict_proba(val_features).to_numpy()
        test_proba = predictor.predict_proba(test_features).to_numpy()
        val_metrics = compute_metrics(y_val, val_proba, None, task_type)
        test_metrics = compute_metrics(y_test, test_proba, None, task_type)
    else:
        val_pred = predictor.predict(val_features).to_numpy()
        test_pred = predictor.predict(test_features).to_numpy()
        val_metrics = compute_metrics(y_val, None, val_pred, task_type)
        test_metrics = compute_metrics(y_test, None, test_pred, task_type)

    return val_metrics, test_metrics


def run_model(
    model_name: str,
    sup_tables: SupervisedTables,
    val_size: float,
    random_state: int,
    max_train_rows: int | None,
    max_test_rows: int | None,
) -> tuple[dict[str, float], dict[str, float]]:
    if model_name == "autogluon":
        return run_autogluon(
            sup_tables,
            val_size=val_size,
            random_state=random_state,
            max_train_rows=max_train_rows,
            max_test_rows=max_test_rows,
        )

    if model_name not in ("lightgbm", "xgboost"):
        # Foundation models don't require a validation set
        val_size = 0

    print(f"Preparing splits: val_size={val_size}, "
          f"max_train_rows={max_train_rows}, "
          f"max_test_rows={max_test_rows}")
    prepared = prepare_splits(
        sup_tables,
        val_size=val_size,
        random_state=random_state,
        max_train_rows=max_train_rows,
        max_test_rows=max_test_rows,
    )

    print(f"Running model: {model_name} with:\n"
          f"\tx_train shape {prepared.x_train.shape}, \n"
          f"\tx_val shape {prepared.x_val.shape}, \n"
          f"\tx_test shape {prepared.x_test.shape}")
    if model_name in ("lightgbm", "xgboost"):
        return run_gbdt(
            model_name,
            prepared.x_train,
            prepared.y_train,
            prepared.x_val,
            prepared.y_val,
            prepared.x_test,
            prepared.y_test,
            prepared.task_type,
        )

    if prepared.task_type == TaskType.REGRESSION:
        if model_name == "tabicl":
            raise ValueError("TabICL does not support regression tasks")
        return run_tabular_fm_regression(
            prepared.x_train,
            prepared.y_train,
            prepared.x_test,
            prepared.y_test,
            batch_size=1000,
        )

    return run_tabular_fm_classification(
        model_name,
        prepared.x_train,
        prepared.y_train,
        prepared.x_test,
        prepared.y_test,
        prepared.task_type,
        batch_size=1000,
    )


def run_model_per_entity(
    model_name: str,
    sup_tables: SupervisedTables,
    val_size: float,
    random_state: int,
    max_train_rows: int | None,
    max_test_rows: int | None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Run model independently per entity, combining predictions for metrics.

    For each unique value of sup_tables.entity_col in the test set, trains on
    that entity's train rows only and predicts on its test rows.  Per-entity
    metrics are combined via a test-size weighted average, which equals the
    metric computed on all concatenated predictions for MAE and MSE.  RMSE
    is recomputed as sqrt(weighted_MSE) rather than averaged directly.
    """
    entity_col = sup_tables.entity_col
    entities = sup_tables.test[entity_col].unique()
    all_metrics: list[dict[str, float]] = []
    all_sizes: list[int] = []

    for entity in tqdm(entities, desc="Entities"):
        train_e = sup_tables.train[sup_tables.train[entity_col] ==
                                   entity].reset_index(drop=True)
        test_e = sup_tables.test[sup_tables.test[entity_col] ==
                                 entity].reset_index(drop=True)
        if len(train_e) == 0 or len(test_e) == 0:
            continue
        sup_e = SupervisedTables(
            train=train_e,
            test=test_e,
            target_col=sup_tables.target_col,
            task_type=sup_tables.task_type,
        )
        _, test_metrics = run_model(
            model_name,
            sup_e,
            val_size=val_size,
            random_state=random_state,
            max_train_rows=max_train_rows,
            max_test_rows=max_test_rows,
        )
        all_metrics.append(test_metrics)
        all_sizes.append(len(test_e))

    if not all_metrics:
        raise ValueError(f"No entities with both train and test rows in "
                         f"column '{entity_col}'")

    combined_size = sum(all_sizes)
    combined: dict[str, float] = {}
    for key in all_metrics[0]:
        combined[key] = float(
            sum(m[key] * n
                for m, n in zip(all_metrics, all_sizes)) / combined_size)
    if "mse" in combined:
        combined["rmse"] = float(np.sqrt(combined["mse"]))

    return {}, combined
