# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared grouping, histogram limits and finite-value helpers for holdouts."""

import math
from typing import Optional

import numpy as np

TENTHS = 10
# Trailing histogram bins holding this many rows or fewer fold into one
# final bin, the boundary of the Kumo UI's Hide Outliers control.
FOLD_AT_MOST = 10


def equal_count_groups(values: np.ndarray, entities: np.ndarray,
                       groups: int = TENTHS) -> np.ndarray:
    """Zero-based group of each row, ascending by value then entity.

    Row ``i`` in sorted order goes to group ``floor(groups * i / n)``, so
    group sizes differ by at most one and the last group holds the highest.
    """
    n = len(values)
    order = np.lexsort((entities, values))
    out = np.empty(n, dtype=int)
    out[order] = np.arange(n) * groups // max(n, 1)
    return out


def finite(value: float) -> Optional[float]:
    return value if math.isfinite(value) else None
