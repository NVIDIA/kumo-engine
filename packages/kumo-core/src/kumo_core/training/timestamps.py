# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared temporal anchor enumeration for hosted and portable generation."""

from typing import Any, Callable, Mapping, Sequence

from kumoapi.pquery.AST import DateOffsetRange
from pandas import DateOffset, Timedelta, Timestamp


def end_anchored_timestamps(
    start: Timestamp,
    end: Timestamp,
    *,
    timeframe: DateOffset | Timedelta,
    start_offset: DateOffset | Timedelta,
    step: DateOffset | Timedelta,
) -> list[Timestamp]:
    """Include the newest complete target window and step toward ``start``.

    Keep the two window offsets separate: hosted callers can use calendar
    offsets, whose subtraction need not be interchangeable with fixed units.
    Split-range selection and earliest-window filtering remain caller-owned.
    """
    timestamps = []
    current = end
    while current - timeframe - start_offset >= start:
        timestamps.append(current - timeframe - start_offset)
        previous = current - step
        if previous >= current:
            raise ValueError("predictive-query timeframe step must be positive")
        current = previous
    return timestamps


def maybe_drop_earliest_timestamp(
    timestamps: list[Timestamp],
    *,
    enabled: bool,
    original_start: Timestamp,
    effective_end: Timestamp,
    step: DateOffset | Timedelta,
    filter_buffer: Timedelta | None = None,
    train_start_offset: DateOffset | Timedelta | None = None,
    start_time: DateOffset | Timedelta | Timestamp | None = None,
) -> list[Timestamp]:
    """Drop an earliest anchor that has no preceding complete step."""
    if (enabled and len(timestamps) > 1 and filter_buffer is None
            and (train_start_offset is None or original_start >
                 effective_end - train_start_offset)
            and (start_time is None or
                 (isinstance(start_time, (DateOffset, Timedelta))
                  and original_start > effective_end + start_time) or
                 (isinstance(start_time, Timestamp)
                  and original_start > start_time))
            and timestamps[-1] < original_start + step):
        return timestamps[:-1]
    return timestamps


def default_target_time_range(
    bounds: Sequence[tuple[Timestamp, Timestamp]],
    *,
    target_range: DateOffsetRange,
    entity_filter_range: DateOffsetRange | None,
    target_filter_range: DateOffsetRange | None,
) -> tuple[Timestamp, Timestamp, Timedelta | None]:
    """Combine target-table bounds and reserve history needed by filters."""
    range_start, range_end, min_date = None, None, None
    for _min, _max in bounds:
        if range_start is None or range_start < _min:
            range_start = _min
        if min_date is None or min_date > _min:
            min_date = _min
        if range_end is None or range_end < _max:
            range_end = _max
    assert range_start is not None
    assert range_end is not None
    assert min_date is not None
    assert target_filter_range is not None
    if entity_filter_range is None:
        filtered_range = target_filter_range
    else:
        filtered_range = DateOffsetRange.merge_ranges(
            entity_filter_range, target_filter_range)
    f_buffer = None
    assert filtered_range is not None
    if filtered_range.start_date_offset is not None:
        filter_history = DateOffsetRange.date_offset_subtract(
            target_range.start_date_offset, filtered_range.start_date_offset)
        if min_date + filter_history > range_start:
            f_buffer = min_date + filter_history - range_start
            range_start = min_date + filter_history
    return range_start, range_end, f_buffer


def timedelta_to_dateoffset(timedelta: Timedelta) -> DateOffset:
    date_offset = DateOffset(days=timedelta.days, seconds=timedelta.seconds)
    return date_offset


def generate_legacy_timestamps(
    *,
    original_start: Timestamp,
    original_end: Timestamp,
    filter_buffer: Timedelta | None,
    target_timeframe: DateOffset | Timedelta,
    target_start_offset: DateOffset | Timedelta,
    target_end_offset: DateOffset | Timedelta,
    step: DateOffset | Timedelta,
    split_family: str | None,
    split: Mapping[str, Any] | None,
    train_start_offset: DateOffset | Timedelta | None,
    train_end_offset: DateOffset | Timedelta | None,
    start_time: DateOffset | Timestamp | None,
    end_time: DateOffset | Timestamp | None,
    drop_earliest: bool,
    on_fixed_boundaries: Callable[[], None] | None = None,
) -> tuple[list[Timestamp], bool, bool]:
    """The hosted legacy timestamp algorithm over backend-free inputs."""
    orig_min_date = original_start
    orig_max_date = original_end
    f_buffer = filter_buffer
    min_date = orig_min_date
    max_date = orig_max_date

    if train_start_offset is not None or train_end_offset is not None:
        if train_start_offset is not None:
            if orig_max_date - train_start_offset > orig_min_date:
                min_date = orig_max_date - train_start_offset
                f_buffer = None
        if train_end_offset is not None:
            max_date = orig_max_date - train_end_offset
    else:
        if start_time is not None:
            if isinstance(start_time, DateOffset):
                if orig_max_date + start_time > orig_min_date:
                    min_date = orig_max_date + start_time
                    f_buffer = None
            else:
                min_date = start_time
                f_buffer = None
        if end_time is not None:
            if isinstance(end_time, DateOffset):
                max_date = orig_max_date + end_time
            else:
                max_date = end_time
    assert isinstance(min_date, Timestamp)
    assert isinstance(max_date, Timestamp)

    target_time_range = timedelta_to_dateoffset(max_date - min_date)
    if DateOffsetRange.date_offset_gt(target_timeframe, target_time_range):
        return [], False, True

    split = split or {}
    ranges_of_interest = []
    fixed_boundaries = False
    if split_family == "DateOffsetSplit":
        boundaries = [min_date]
        for offset in split["offsets"]:
            boundaries.append(max_date - target_end_offset + DateOffset(
                **{split["unit"]: offset}))
        boundaries.append(max_date)
        if boundaries[0] > boundaries[1]:
            boundaries[0] = boundaries[1]
        ranges_of_interest = [
            x for x in zip(boundaries[:-1], boundaries[1:])
        ]
        fixed_boundaries = True
        if on_fixed_boundaries is not None:
            on_fixed_boundaries()
    elif split_family == "TimeRangeSplit":
        for start_timestamp, end_timestamp in split["ranges"]:
            start_timestamp = max(start_timestamp, min_date)
            end_timestamp = max(end_timestamp, min_date)
            ranges_of_interest.append((start_timestamp, end_timestamp))
        fixed_boundaries = True
        if on_fixed_boundaries is not None:
            on_fixed_boundaries()
    elif split_family == "BacktestingSplit":
        ranges_of_interest = [(split["timestamp"],
                               split["timestamp"] + target_end_offset)]
    else:
        ranges_of_interest = [(min_date, max_date)]

    ranges_of_interest.sort(reverse=True)
    for i in range(len(ranges_of_interest) - 1):
        assert ranges_of_interest[i][0] >= ranges_of_interest[i + 1][1]
        assert ranges_of_interest[i][0] <= ranges_of_interest[i][1]

    timestamps = []
    for start_timestamp, end_timestamp in ranges_of_interest:
        timestamps.extend(
            end_anchored_timestamps(
                start_timestamp,
                end_timestamp,
                timeframe=target_timeframe,
                start_offset=target_start_offset,
                step=step,
            ))
    timestamps = maybe_drop_earliest_timestamp(
        timestamps,
        enabled=drop_earliest,
        original_start=orig_min_date,
        effective_end=max_date,
        step=step,
        filter_buffer=f_buffer,
        train_start_offset=train_start_offset,
        start_time=start_time,
    )
    return timestamps, fixed_boundaries, False


def generate_decoupled_timestamps(
    *,
    original_start: Timestamp,
    original_end: Timestamp,
    filter_buffer: Timedelta | None,
    target_timeframe: DateOffset | Timedelta,
    target_start_offset: DateOffset | Timedelta,
    step: DateOffset | Timedelta,
    start_time: DateOffset | Timestamp | None,
    end_time: DateOffset | Timestamp | None,
    drop_earliest: bool,
) -> list[Timestamp]:
    """The hosted V2 timestamp algorithm over backend-free inputs."""
    orig_min_date = original_start
    orig_max_date = original_end
    f_buffer = filter_buffer
    min_date = orig_min_date
    max_date = orig_max_date

    if start_time is not None:
        if isinstance(start_time, DateOffset):
            if orig_max_date + start_time > orig_min_date:
                min_date = orig_max_date + start_time
                f_buffer = None
        else:
            min_date = start_time
            f_buffer = None
    if end_time is not None:
        if isinstance(end_time, DateOffset):
            max_date = orig_max_date + end_time
        else:
            max_date = end_time
    assert isinstance(min_date, Timestamp)
    assert isinstance(max_date, Timestamp)

    timestamps = end_anchored_timestamps(
        min_date,
        max_date,
        timeframe=target_timeframe,
        start_offset=target_start_offset,
        step=step,
    )
    return maybe_drop_earliest_timestamp(
        timestamps,
        enabled=drop_earliest,
        original_start=orig_min_date,
        effective_end=max_date,
        step=step,
        filter_buffer=f_buffer,
        start_time=start_time,
    )
