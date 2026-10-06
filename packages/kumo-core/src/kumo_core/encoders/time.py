# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math

import torch
from torch import Tensor
from torch_geometric.utils import one_hot

from kumo_core.encoders.tensor_mapper import TimestampTensorMapper
from kumo_core.types.features import NANOSECONDS_IN_DAY
from kumo_core.features.time import to_datetime


class TimeEncoder(torch.nn.Module):
    r"""A non-parametrized time utility module.

    Args:
        unix_step_size: The UNIX step size in case of forecasting.
    """
    # Hard-coded reference to the channel size of seed time encodings.
    out_channels: int = 12 + TimestampTensorMapper.num_holiday_names

    def __init__(
        self,
        unix_step_size: int | None = None,
    ) -> None:
        super().__init__()

        self.unix_step_size = unix_step_size
        self.mapper = TimestampTensorMapper()

    def is_holiday(
        self,
        time: Tensor,
        length: int = 1,
        padding_size: int = 0,
    ) -> Tensor:
        r"""Returns whether time dates fall on a holiday.

        Args:
            time: The time.
            length: The forecast length.
            padding_size: The padding applied at the start and end of the time
                series.
        """
        time = self._apply_offset(time, length, padding_size)
        day = time.view(-1) // NANOSECONDS_IN_DAY
        is_holiday = self.mapper.get_closest_holiday(day).days_to_holiday == 0
        return is_holiday.view(time.size())

    def forward(
        self,
        time: Tensor,
        length: int = 1,
        padding_size: int = 0,
    ) -> Tensor:
        r"""Returns fixed time encodings.

        Args:
            time: The time.
            length: The forecast length.
            padding_size: The padding applied at the start and end of the time
                series.
        """
        time = self._apply_offset(time, length, padding_size)
        window_size = time.size(1)

        # NOTE This currently forces a CUDA device synchronization.
        ser = to_datetime(time.view(-1).cpu().numpy())

        time = self.mapper(ser, device=time.device)

        tensors: list[Tensor] = []
        index = TimestampTensorMapper.index

        # Scale values to [0, 1) and perform cyclical encoding on top:
        minute = time[:, index['minute']] / 60
        tensors.append(minute)

        hour = time[:, index['hour']] / 24
        hour += (1 / 24) * minute
        tensors.append(hour)

        day_of_week = time[:, index['day_of_week']] / 7
        day_of_week += (1 / 7) * hour
        tensors.append(day_of_week)

        days_per_month = time[:, index['days_per_month']]
        day_of_month = time[:, index['day_of_month']] / days_per_month
        day_of_month += (1 / days_per_month) * hour
        tensors.append(day_of_month)

        days_per_year = time[:, index['days_per_year']]
        day_of_year = time[:, index['day_of_year']] / days_per_year
        day_of_year += (1 / days_per_year) * hour
        tensors.append(day_of_year)

        # Cylic features get encoded via sinus/cosinus:
        time_enc = torch.stack(tensors, dim=-1)
        time_enc = (2 * math.pi * time_enc) - math.pi
        time_enc = torch.cat([time_enc.sin(), time_enc.cos()], dim=-1)

        # Holiday gets one-hot encoded:
        days_to_holiday = time[:, index['days_to_holiday']]
        is_not_holiday = days_to_holiday != 0
        holiday_name = time[:, index['to_holiday_name']]
        holiday_name = one_hot(
            holiday_name,
            num_classes=TimestampTensorMapper.num_holiday_names,
        )
        holiday_name[is_not_holiday] = 0.0
        time_enc = torch.cat([
            time_enc,
            1.0 - is_not_holiday.float().view(-1, 1),
            is_not_holiday.float().view(-1, 1),
            holiday_name,
        ], dim=-1)

        assert time_enc.size(-1) == self.out_channels
        return time_enc.view(-1, window_size, self.out_channels)

    def _apply_offset(
        self,
        time: Tensor,
        length: int,
        padding_size: int,
    ) -> Tensor:
        time = time.view(-1, 1)

        if length != 1 or padding_size != 0:
            if self.unix_step_size is None:
                raise ValueError("'unix_step_size' needs to be defined in "
                                 "order to apply offsets")
            step = torch.arange(
                start=-padding_size * self.unix_step_size,
                end=(length + padding_size) * self.unix_step_size,
                step=self.unix_step_size,
                dtype=time.dtype,
                device=time.device,
            )
            time = time + step.view(1, -1)

        return time

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
