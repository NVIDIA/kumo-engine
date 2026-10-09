# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from decimal import Decimal
from typing import Any, Dict

from kumo_core.types import ColStatType

from kumo.stats.stats import Stats


def compare_stat_values(stat_value1: Any, stat_value2: Any):
    r"""
    Compare the raw values of two stats and assert that they are equal.
    For floating point values, equality is decided using `math.isclose()`
    """

    # ensure the types are the same.
    if isinstance(stat_value1, Decimal):
        stat_value1 = float(stat_value1)
    if isinstance(stat_value2, Decimal):
        stat_value2 = float(stat_value2)

    assert type(stat_value1) == type(stat_value2)  # noqa: E721

    # compare the values based on the types
    if isinstance(stat_value1, int):
        assert stat_value1 == stat_value2
    elif isinstance(stat_value1, float):
        assert math.isclose(stat_value1, stat_value2, rel_tol=0.1, abs_tol=1)
    elif isinstance(stat_value1, list):
        assert len(stat_value1) == len(stat_value2)
        for i in range(len(stat_value1)):
            compare_stat_values(stat_value1[i], stat_value2[i])
    elif isinstance(stat_value1, dict):
        assert len(stat_value1) == len(stat_value2)
        assert set(stat_value1.keys()) == set(stat_value2.keys())
        for key in stat_value1.keys():
            compare_stat_values(stat_value1[key], stat_value2[key])
    else:
        # otherwise just compare the values
        # TODO add more types as needed
        assert stat_value1 == stat_value2


def compare_column_stats(stat1: Dict[str, Dict[ColStatType, Stats]],
                         stat2: Dict[str, Dict[ColStatType, Stats]]):
    r"""
    Utility to compare two sets of column stats and assert that they are equal.
    """
    # both should be None or not.
    if not stat1 or not stat2:
        assert stat1 == stat2

    assert set(stat1.keys()) == set(stat2.keys())
    for column in stat1.keys():
        assert set(stat1[column].keys()) == (set(stat2[column].keys()))
        for stat_type in stat1[column].keys():
            try:
                compare_stat_values(stat1[column][stat_type].value,
                                    stat2[column][stat_type].value)
            except AssertionError as e:
                raise AssertionError(
                    f"Stats comparsion failed for {column=} {stat_type=}. "
                    f"Values are {stat1[column][stat_type].value=} "
                    f"{stat2[column][stat_type].value=}") from e
