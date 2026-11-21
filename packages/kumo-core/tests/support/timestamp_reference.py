# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent timestamp feature expectations using pandas calendar accessors."""

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from kumo_core.encoders.tensor_mapper import TimestampTensorMapper
from kumo_core.features.time import to_datetime
from torch import Tensor


def timestamp_reference(
    mapper: TimestampTensorMapper,
    array: pa.Array,
    device: torch.device,
) -> Tensor:
    """Compute all features independently of the lookup table.

    Args:
        mapper: Supplies the format and existing holiday definitions.
        array: Timestamp values, including nulls and extreme dates.
        device: Destination for the expected int64 tensor.

    Returns:
        The sixteen legacy features with null rows filled with -1.
    """
    series = to_datetime(array.to_pandas(), format=mapper.format)
    nanoseconds = series.astype("int64").to_numpy(copy=True)
    missing = nanoseconds == np.iinfo(np.int64).min
    nanoseconds[missing] = 0
    nanoseconds = np.maximum(nanoseconds, pd.Timestamp("1678-01-01").value)
    dates = pd.to_datetime(nanoseconds)
    unix = nanoseconds // 10**9
    days = unix // 86400
    holidays = mapper.helper.holidays.cpu().numpy()
    names = mapper.helper.holiday_names.cpu().numpy()
    following = np.searchsorted(holidays, days, side="left").clip(max=len(holidays) - 1)
    previous = (np.searchsorted(holidays, days, side="right") - 1).clip(min=0)
    days_per_year = np.where(dates.is_leap_year, 366, 365)
    features = np.column_stack(
        [
            dates.minute,
            dates.hour,
            dates.year,
            dates.dayofweek,
            dates.day - 1,
            dates.dayofyear - 1,
            dates.days_in_month,
            days_per_year,
            (days_per_year + 6) // 7,
            dates.month - 1,
            (dates.dayofyear - 1) // 7,
            holidays[following] - days,
            days - holidays[previous],
            names[following],
            names[previous],
            unix,
        ]
    ).astype(np.int64)
    features[missing] = -1
    return torch.from_numpy(features).to(device)
