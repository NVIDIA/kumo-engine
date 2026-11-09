# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation and UTC normalization for temporal label-window bounds."""

from pandas import Timestamp, isna


def parse_time_bound(value: int | str | None,
                     name: str) -> int | Timestamp | None:
    """Keep relative offsets numerical and normalize absolute times to UTC."""
    if value is None or (name == 'start_time' and value == '-INF'):
        return None
    if isinstance(value, int):
        if value > 0:
            raise ValueError(f"'{name}' must be a non-positive relative offset")
        return value
    try:
        timestamp = Timestamp(value)
        if isna(timestamp):
            raise ValueError('timestamp is NaT')
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert('UTC').tz_localize(None)
        # Generation stores timestamps as signed nanoseconds.
        timestamp = Timestamp(timestamp.value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"'{name}' must be a valid timestamp: {value!r}") from error
    return timestamp


def validate_time_bounds(
    start: int | str | None,
    end: int | str | None,
    *,
    train_start_offset: int | None = None,
    train_end_offset: int | None = None,
    split_family: str | None = None,
) -> tuple[int | Timestamp | None, int | Timestamp | None]:
    """Validate explicit parameters without changing their serialized form."""
    if ((start is not None or end is not None)
            and (train_start_offset is not None
                 or train_end_offset is not None)):
        raise ValueError('Cannot specify both start_time/end_time and '
                         'train_start_offset/train_end_offset.')
    parsed_start = parse_time_bound(start, 'start_time')
    parsed_end = parse_time_bound(end, 'end_time')
    if parsed_start is not None and parsed_end is not None:
        if isinstance(parsed_start, int) != isinstance(parsed_end, int):
            raise ValueError('start_time and end_time must both be offsets '
                             'or both be timestamps.')
        if ((isinstance(parsed_start, int) and isinstance(parsed_end, int)
             and parsed_start >= parsed_end)
                or (isinstance(parsed_start, Timestamp)
                    and isinstance(parsed_end, Timestamp)
                    and parsed_start >= parsed_end)):
            raise ValueError('start_time must be smaller than end_time.')
    if (isinstance(parsed_start, Timestamp)
            or isinstance(parsed_end, Timestamp)) and split_family in {
                'BacktestingSplit', 'DateOffsetSplit', 'TimeRangeSplit',
                'TimeSplit',
            }:
        raise ValueError('Absolute start_time/end_time cannot be combined '
                         f'with {split_family}; use a ratio split or inferred.')
    return parsed_start, parsed_end
