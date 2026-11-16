# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for paper benchmarks: dataset registry, preprocessing,
encoding, metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from kumoapi.task import TaskType
from sklearn.preprocessing import OrdinalEncoder

from examples.paper.datasets import SupervisedTables


@dataclass(frozen=True)
class PreparedSplits:
    """Train/val/test feature arrays and labels, plus task type."""
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    task_type: TaskType


def numeric_and_categorical_columns(
        df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = []
    categorical = []
    for col in df.columns:
        dtype = df[col].dtype
        if np.issubdtype(dtype, np.number):
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def sanitize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast extension dtypes and replace pd.NA so models don't choke."""
    out = df.copy()
    for col in out.columns:
        dtype = out[col].dtype
        if hasattr(dtype, "kind"):
            if dtype.kind in "iu" or (dtype.kind == "b"
                                      and dtype.name != "bool"):
                out[col] = out[col].astype(np.float64)
            elif dtype.name == "string" or dtype.name == "object":
                out[col] = out[col].astype(object)
        out[col] = out[col].replace({pd.NA: np.nan})
    return out


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replace datetime columns with circular temporal features."""
    df = df.copy()
    for col in list(df.columns):
        if not np.issubdtype(df[col].dtype, np.datetime64):
            continue
        ts = df[col]
        for values, period, name in [
            (ts.dt.hour, 24, "hour"),
            (ts.dt.minute, 60, "minute"),
            (ts.dt.dayofweek, 7, "dow"),
        ]:
            angle = values.to_numpy() * (2 * np.pi / period)
            df[f"{col}_sin_{name}"] = np.sin(angle)
            df[f"{col}_cos_{name}"] = np.cos(angle)
        df = df.drop(columns=[col])
    return df


def prepare_splits_dataframes(
    sup_tables: SupervisedTables,
    val_size: float,
    random_state: int,
    max_train_rows: int | None,
    max_test_rows: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, TaskType]:
    """Subsample, sanitize, train/val split. Returns DataFrames with target."""
    train_df = sup_tables.train.copy()
    test_df = sup_tables.test.copy()
    target_col = sup_tables.target_col
    task_type = sup_tables.task_type

    if max_train_rows is not None and len(train_df) > max_train_rows:
        train_df = train_df.sample(n=max_train_rows, random_state=random_state)
    if max_test_rows is not None and len(test_df) > max_test_rows:
        test_df = test_df.sample(n=max_test_rows, random_state=random_state)
    train_df = sanitize_dtypes(train_df)
    test_df = sanitize_dtypes(test_df)

    if val_size > 0:
        n_rows = len(train_df)
        idx_val = int(n_rows * (1.0 - val_size))
        idx_val = max(1, min(n_rows - 1, idx_val))
        val_df = train_df.iloc[idx_val:].copy()
        train_df = train_df.iloc[:idx_val].copy()
    else:
        val_df = train_df.iloc[:0]

    return train_df, val_df, test_df, target_col, task_type


def prepare_splits(
    sup_tables: SupervisedTables,
    val_size: float,
    random_state: int,
    max_train_rows: int | None,
    max_test_rows: int | None,
) -> PreparedSplits:
    train_df, val_df, test, target_col, task_type = prepare_splits_dataframes(
        sup_tables,
        val_size=val_size,
        random_state=random_state,
        max_train_rows=max_train_rows,
        max_test_rows=max_test_rows,
    )

    y_train = train_df[target_col].to_numpy()
    y_val = val_df[target_col].to_numpy()
    y_test = test[target_col].to_numpy()

    if task_type == TaskType.BINARY_CLASSIFICATION:
        y_train = y_train.astype(np.int64)
        y_val = y_val.astype(np.int64)
        y_test = y_test.astype(np.int64)

    train_df = train_df.drop(columns=[target_col])
    val_df = val_df.drop(columns=[target_col])
    test_df = test.drop(columns=[target_col])

    if train_df.shape[1] == 0:
        raise ValueError(
            "No non-target features remain after preprocessing; this is "
            "likely due to leakage-avoidance column exclusions. This "
            "dataset cannot be used with single-table models.", )

    _numeric_cols, cat_cols = numeric_and_categorical_columns(train_df)
    x_train, x_val, x_test = encode_ordinal(train_df, val_df, test_df,
                                            cat_cols)

    return PreparedSplits(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        task_type=task_type,
    )


def encode_ordinal(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ordinal-encode categoricals; fill nan with a sentinel;
    return float arrays.
    """
    fill = "__nan__"

    def to_str(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        return df[cols].apply(lambda c: c.astype(object).replace(
            [np.nan, pd.NA], fill).astype(str))

    cols = [
        c for c in cat_cols
        if c in train.columns and c in val.columns and c in test.columns
    ]
    if not cols:
        return (
            train.to_numpy().astype(np.float64),
            val.to_numpy().astype(np.float64),
            test.to_numpy().astype(np.float64),
        )
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_enc = enc.fit_transform(to_str(train, cols)).astype(np.float64)
    if len(val) == 0:
        val_enc = np.empty((0, train_enc.shape[1]), dtype=np.float64)
    else:
        val_enc = enc.transform(to_str(val, cols)).astype(np.float64)
    test_enc = enc.transform(to_str(test, cols)).astype(np.float64)

    num_cols = [c for c in train.columns if c not in cols]
    if not num_cols:
        return train_enc, val_enc, test_enc

    train_num = train[num_cols].to_numpy(dtype=np.float64, copy=True)
    if len(val) == 0:
        val_num = np.empty((0, len(num_cols)), dtype=np.float64)
    else:
        val_num = val[num_cols].to_numpy(dtype=np.float64, copy=True)
    test_num = test[num_cols].to_numpy(dtype=np.float64, copy=True)
    np.nan_to_num(train_num, copy=False, nan=0.0)
    np.nan_to_num(val_num, copy=False, nan=0.0)
    np.nan_to_num(test_num, copy=False, nan=0.0)
    return (
        np.hstack([train_num, train_enc]),
        np.hstack([val_num, val_enc]),
        np.hstack([test_num, test_enc]),
    )
