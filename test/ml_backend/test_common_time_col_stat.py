# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
from datetime import datetime

from kumo_core.types import ColStatType

from kumo.graph.column import Stats

common = importlib.import_module('rest.backends.ml_backend.activities.common')


def _stats(value):
    return {ColStatType.MIN: Stats(value=value, approx=False)}


def test_get_time_col_stat_normalizes_int_epoch_millis():
    # TT-NG predict col_stats carry the time column as an epoch-millis int
    # (whatif anchor). Before the fix this raised "must be datetime but got
    # int"; it must now be converted to a datetime.
    result = common._get_time_col_stat(ColStatType.MIN, _stats(1659304801388))
    assert isinstance(result, datetime)
    assert result.year == 2022


def test_get_time_col_stat_passes_through_datetime():
    # Legacy path already provides a datetime; normalization is a no-op.
    dt = datetime(2022, 7, 31)
    result = common._get_time_col_stat(ColStatType.MIN, _stats(dt))
    assert result == dt


def test_get_time_col_stat_returns_none_for_missing():
    assert common._get_time_col_stat(ColStatType.MIN, _stats(None)) is None
