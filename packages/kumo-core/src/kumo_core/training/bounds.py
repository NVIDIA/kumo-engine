# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation and UTC normalization for temporal label-window bounds."""

from pandas import DateOffset, Timestamp


def parse_time_bound(value: int | str | DateOffset | Timestamp | None,
                     name: str) -> DateOffset | Timestamp | None:
    """Normalize absolute times to UTC and retain parsed internal offsets."""
    if value is None or (name == 'start_time' and value == '-INF'):
        return None
    if isinstance(value, int):
        raise ValueError(
            f"Integer '{name}' bounds are not supported. Use timestamp strings "
            "or 'inferred'; use train_start_offset/train_end_offset for "
            "relative training bounds.")
    if isinstance(value, DateOffset):
        if Timestamp('1970-01-01') + value > Timestamp('1970-01-01'):
            raise ValueError(f"'{name}' must be a non-positive relative offset")
        return value
    try:
        timestamp = Timestamp(value)
        if not isinstance(timestamp, Timestamp):
            raise ValueError('timestamp is NaT')
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert('UTC').tz_localize(None)
        # Generation stores timestamps as signed nanoseconds.
        _ = timestamp.value
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"'{name}' must be a valid timestamp: {value!r}") from error
    return timestamp


def validate_time_bounds(
    start: int | str | DateOffset | Timestamp | None,
    end: int | str | DateOffset | Timestamp | None,
    *,
    train_start_offset: int | DateOffset | None = None,
    train_end_offset: int | DateOffset | None = None,
    split_family: str | None = None,
) -> tuple[DateOffset | Timestamp | None, DateOffset | Timestamp | None]:
    """Parse start/end bounds and reject invalid combinations."""
    if ((start is not None or end is not None)
            and (train_start_offset is not None
                 or train_end_offset is not None)):
        raise ValueError('Cannot specify both start_time/end_time and '
                         'train_start_offset/train_end_offset.')
    parsed_start = parse_time_bound(start, 'start_time')
    parsed_end = parse_time_bound(end, 'end_time')
    if parsed_start is not None and parsed_end is not None:
        if (isinstance(parsed_start, Timestamp)
                != isinstance(parsed_end, Timestamp)):
            raise ValueError('start_time and end_time must both be offsets '
                             'or both be timestamps.')
        if ((isinstance(parsed_start, Timestamp)
             and isinstance(parsed_end, Timestamp)
             and parsed_start >= parsed_end)
                or (isinstance(parsed_start, DateOffset)
                    and isinstance(parsed_end, DateOffset)
                    and Timestamp('1970-01-01') + parsed_start
                    >= Timestamp('1970-01-01') + parsed_end)):
            raise ValueError('start_time must be smaller than end_time.')
    if (isinstance(parsed_start, Timestamp)
            or isinstance(parsed_end, Timestamp)) and split_family in {
                'BacktestingSplit', 'DateOffsetSplit', 'TimeRangeSplit',
                'TimeSplit',
            }:
        raise ValueError('Absolute start_time/end_time cannot be combined '
                         f'with {split_family}; use a ratio split or inferred.')
    return parsed_start, parsed_end
