# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for formatting prediction output columns.

This module provides lightweight, reusable functions for prettifying
prediction output DataFrames with minimal dependencies (only pandas).

Re-exported from ``kumo.utils.output`` for backwards compatibility.
"""
import pandas as pd


def prettify_prediction_columns(
    df: pd.DataFrame,
    entity_col_name: str | None = None,
    target_col_name: str | None = None,
    round_decimals: int = 4,
    convert_to_string: bool = True,
) -> pd.DataFrame:
    r"""Prettify prediction output columns with consistent formatting.

    This is a shared utility for formatting prediction outputs.

    Transformations applied:
    - Remove ANCHOR_TIMESTAMP column if present
    - Rename ENTITY column to entity_col_name if provided
    - Rename TARGET_PRED column to target_col_name if provided
    - Rename *_PROB columns to "* Probability" (e.g., True_PROB -> True Probability)
    - Round float columns to specified decimal places
    - Optionally convert rounded values to strings

    Args:
        df: The prediction output DataFrame.
        entity_col_name: Name to rename ENTITY column to (e.g., "users.user_id").
            If None, ENTITY column is left unchanged.
        target_col_name: Name to rename TARGET_PRED column to
            (e.g., "SUM(orders.price)"). If None, TARGET_PRED is left unchanged.
        round_decimals: Number of decimal places to round float values.
        convert_to_string: Whether to convert rounded values to strings.

    Returns:
        DataFrame with prettified column names and formatted values.
    """
    df = df.copy()

    # Remove ANCHOR_TIMESTAMP if present
    if "ANCHOR_TIMESTAMP" in df.columns:
        df = df.drop(columns=["ANCHOR_TIMESTAMP"])

    # Rename ENTITY column
    if entity_col_name and "ENTITY" in df.columns:
        df = df.rename(columns={"ENTITY": entity_col_name})

    # Rename and round TARGET_PRED column
    if target_col_name and "TARGET_PRED" in df.columns:
        df = df.rename(columns={"TARGET_PRED": target_col_name})
        if df[target_col_name].dtype in ['float64', 'float32']:
            rounded = df[target_col_name].round(round_decimals)
            df[target_col_name] = rounded.astype(str) if convert_to_string else rounded
    elif "TARGET_PRED" in df.columns:
        # Round TARGET_PRED even if not renamed
        if df["TARGET_PRED"].dtype in ['float64', 'float32']:
            rounded = df["TARGET_PRED"].round(round_decimals)
            df["TARGET_PRED"] = rounded.astype(str) if convert_to_string else rounded

    # Round SCORE column if present (for link prediction / multiclass)
    if "SCORE" in df.columns and df["SCORE"].dtype in ['float64', 'float32']:
        rounded = df["SCORE"].round(round_decimals)
        df["SCORE"] = rounded.astype(str) if convert_to_string else rounded

    # Prettify probability columns: *_PROB -> * Probability
    prob_cols = [col for col in df.columns if "_PROB" in col]
    for col in prob_cols:
        label = col.split("_PROB")[0]
        new_name = f"{label} Probability"
        if df[col].dtype in ['float64', 'float32']:
            rounded = df[col].round(round_decimals)
            df[new_name] = rounded.astype(str) if convert_to_string else rounded
        else:
            df[new_name] = df[col]
        df = df.drop(columns=[col])

    return df
