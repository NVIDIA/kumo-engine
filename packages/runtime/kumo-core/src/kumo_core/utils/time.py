# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

from kumoapi.typing import TimeUnit
from torch import Tensor

PSEUDO_TIME_COL_NAME = 'PSEUDO_TIMESTAMP'
# 100 trillion. A large number that can be handled by int64.
MAX_PSEUDO_TIME = 100 * (10**12)


def now_utc() -> datetime:
    """ Produce a time-zone aware datetime in the UTC timezone.
    """
    return datetime.now(timezone.utc)


def compute_unix_step(
    timestep: Tensor,
    time_unit: TimeUnit = TimeUnit.DAYS,
) -> Tensor:
    r"""Computes a timestep in unix time with respect to a specified time unit.

    Args:
        timestep (Tensor): Time tensor.
        time_unit (TimeUnit): The time unit. (default: :obj:`"days"`)
    """
    time = timestep
    if time_unit == TimeUnit.SECONDS:
        time = time
    elif time_unit == TimeUnit.MINUTES:
        time = time * 60
    elif time_unit == TimeUnit.HOURS:
        time = time * (60 * 60)
    elif time_unit == TimeUnit.DAYS:
        time = time * (60 * 60 * 24)
    elif time_unit == TimeUnit.WEEKS:
        time = time * (60 * 60 * 24 * 7)
    elif time_unit == TimeUnit.MONTHS:
        time = time * (60 * 60 * 24 * 30)
    else:
        raise ValueError(f"Unsupported time unit '{time_unit.value}'")

    time = time * (1000 * 1000 * 1000)  # Convert days to nanoseconds.

    return time
