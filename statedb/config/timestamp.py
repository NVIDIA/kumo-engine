# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import copy
from datetime import datetime, timedelta, timezone

from temporalio import workflow

from io_utils import kumo_dataclass


def now_utc() -> datetime:
    """
    Produce a time-zone aware datetime in the UTC timezone.
    If called within Temporal workflow execution, return a deterministic,
    current time from the workflow perspective, i.e., workflow.now()
    """
    try:
        return workflow.now()
    except Exception:
        # If not running within workflow event loop, either RuntimeError or
        # TemporalError would be raised.
        return datetime.now(timezone.utc)


def now_plus_seconds_utc(n_seconds: int):
    """
    Produce a time-zone aware datetime in the UTC timezone `n_seconds`
    seconds from now
    """
    return now_utc() + timedelta(seconds=n_seconds)


@kumo_dataclass(order=True)
class ResourceTimestamps:
    created_at: datetime
    updated_at: datetime


def get_current_timestamps() -> ResourceTimestamps:
    now = now_utc()
    return ResourceTimestamps(now, now)


def update_timestamps(timestamps: ResourceTimestamps) -> ResourceTimestamps:
    updated_timestamps = copy(timestamps)
    updated_timestamps.updated_at = now_utc()
    return updated_timestamps
