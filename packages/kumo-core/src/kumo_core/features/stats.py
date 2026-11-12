# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import builtins
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from kumo_core.encoders.tensor_mapper import SequenceTensorMapper


def mean(ser: pd.Series) -> float:
    return float(np.mean(ser)) if len(ser) > 0 else 0.0


def std(ser: pd.Series) -> float:
    return float(np.std(ser)) if len(ser) > 1 else 1.0


def min(ser: pd.Series) -> float:
    return float(np.min(ser)) if len(ser) > 0 else 0.0


def max(ser: pd.Series) -> float:
    return float(np.max(ser)) if len(ser) > 0 else 0.0


def quantiles(ser: pd.Series) -> tuple[float, float, float]:
    if len(ser) == 0:
        return (0.0, 0.0, 0.0)
    return tuple(np.quantile(ser, [0.25, 0.5, 0.75]).tolist())


def category_count(ser: pd.Series) -> list[tuple[Any, int]]:
    counts = ser.value_counts()
    counts = counts.rename_axis("category").reset_index(name="count")
    counts = counts.sort_values(
        by=["count", "category"],
        ascending=[False, True],
        kind="mergesort",
    )
    return counts.to_numpy().tolist()


def sep_multi(ser: pd.Series) -> str | None:
    index = ser.first_valid_index()
    if index is None or pd.api.types.is_list_like(ser[index]):
        return None

    text = "\n".join(ser.iloc[:1000].fillna(""))
    white_list = {",", ";", ":", "|", "\t"}
    candidates: dict[str, int] = defaultdict(int)
    for char in text:
        if char in white_list:
            candidates[char] += 1

    if len(candidates) > 0:
        return builtins.max(candidates, key=candidates.get)  # type: ignore

    return None


def sequence_min_length(ser: pd.Series) -> int:
    ser = SequenceTensorMapper.to_list_like(ser)
    return int(np.min(ser.apply(len))) if len(ser) > 0 else 0


def sequence_max_length(ser: pd.Series) -> int:
    ser = SequenceTensorMapper.to_list_like(ser)
    return int(np.max(ser.apply(len))) if len(ser) > 0 else 0


def sequence_mean(ser: pd.Series) -> float:
    ser = SequenceTensorMapper.to_list_like(ser)
    return float(np.mean(np.concatenate(ser.values))) if len(ser) > 0 else 0.0


def sequence_std(ser: pd.Series) -> float:
    ser = SequenceTensorMapper.to_list_like(ser)
    return float(np.std(np.concatenate(ser.values))) if len(ser) > 0 else 1.0


def notna(ser: pd.Series) -> np.ndarray:
    if pd.api.types.is_float_dtype(ser):
        return np.isfinite(ser.to_numpy())

    notna_mask = ser.notna()

    if pd.api.types.infer_dtype(ser) == "string":
        notna_mask &= ser != ""

    return notna_mask.to_numpy()
