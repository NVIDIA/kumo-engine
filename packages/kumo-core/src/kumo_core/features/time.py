# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import pandas as pd
import torch
from pandas.tseries.holiday import USFederalHolidayCalendar
from torch import Tensor
from torch_geometric.utils import cumsum

from kumo_core.types.features import NANOSECONDS_IN_DAY, TimeUnit
from kumo_core.encoders.index_mapping import Mapper


def to_datetime(arg: Any, format: str | None = None) -> pd.Series:
    r"""Converts the input to ``datetime64[ns]`` format.

    Args:
        arg: The data to convert.
        format: The ``strftime`` to parse time in case input values are given
            as string. If :obj:`None`, will try to infer the correct format.
    """
    if not pd.api.types.is_datetime64_any_dtype(arg):
        arg = pd.to_datetime(arg, format=format, errors='coerce')

    if not isinstance(arg, pd.Series):
        arg = pd.Series(arg)

    if arg.dtype != 'datetime64[ns]':
        arg = arg.astype('datetime64[ns]')

    return arg


class TimestampHelper:
    r"""Timestamp information to be shared across different
    :class:`TimestampTensorMapper` instantiations.
    """
    _instance: 'TimestampHelper | None' = None

    def __init__(self) -> None:
        holidays = USFederalHolidayCalendar().holidays(
            start=pd.Timestamp.min + pd.Timedelta(500, unit='days'),
            end=pd.Timestamp.max - pd.Timedelta(500, unit='days'),
            return_name=True,
        )

        self.holidays = torch.from_numpy(
            holidays.index.astype('int64').to_numpy())
        self.holidays = self.holidays // NANOSECONDS_IN_DAY
        self.holiday_names = Mapper(sorted(holidays.unique())).map(holidays)

        years = []
        for i in range(pd.Timestamp.min.year + 1, pd.Timestamp.max.year + 1):
            years.append(pd.Timestamp(f'{i}-01-01').value // 10**9)
        self.year_offset = torch.tensor(years)

        self.common_days_per_month = torch.tensor(
            [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        self.cumsum_common_days_per_month = cumsum(self.common_days_per_month)
        self.leap_days_per_month = torch.tensor(
            [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        self.cumsum_leap_days_per_month = cumsum(self.leap_days_per_month)

    @classmethod
    def instance(cls) -> 'TimestampHelper':
        if cls._instance is None:
            cls._instance = cls()
        assert cls._instance is not None
        return cls._instance


def get_rel_time(
    time: Tensor,
    seed_time: Tensor,
    time_unit: TimeUnit = 'days',
) -> Tensor:
    r"""Computes relative time according to a seed time with respect to a
    specified time unit.

    Args:
        time: Time tensor in seconds.
        seed_time: Seed time tensor in seconds.
        time_unit: The time unit.
    """
    assert time.dim() == 2

    if time_unit == 'seconds':
        denominator = 1
    elif time_unit == 'minutes':
        denominator = 60
    elif time_unit == 'hours':
        denominator = 60 * 60
    elif time_unit == 'days':
        denominator = 60 * 60 * 24
    elif time_unit == 'weeks':
        denominator = 60 * 60 * 24 * 7
    elif time_unit == 'months':
        denominator = 60 * 60 * 24 * 30
    else:
        raise ValueError(f"Unsupported time unit '{time_unit}'")

    out = (seed_time.view(-1, 1) - time) / denominator
    out[time == -1] = float('NaN')
    return out
