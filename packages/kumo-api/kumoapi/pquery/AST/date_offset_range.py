# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT

import logging
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic.dataclasses import dataclass

from kumoapi.pquery.utils import validate_int
from kumoapi.typing import _FORBID_EXTRA_CONFIG, TimeUnit

logger = logging.getLogger(__name__)


@dataclass(config=_FORBID_EXTRA_CONFIG)
class DateOffsetRange:
    r"""An internal class for keeping track of time ranges in the query.

    Args:
        start: Start of the interval. If :obj:`None`, it is
            treated as -inf.
        end: End of the interval.
        unit: Time unit of the time interval. See
            :class:`kumo.typing.TimeUnit` for valid values.
            (default: 'days')
    """
    start: Optional[int]
    end: int
    unit: Union[TimeUnit, str] = 'days'

    def __post_init__(self) -> None:
        if self.start is not None and self.start >= self.end:
            raise ValueError(
                f"The start date offset '{self.start}' needs to be smaller "
                f"than the end date offset '{self.end}' to specify a valid "
                f"time range.")
        self.unit = TimeUnit(
            self.unit.lower() if isinstance(self.unit, str) else self.unit)
        if not self.is_open:
            assert self.start is not None
            self._to_dateOffset(self.unit, self.start)
        self._to_dateOffset(self.unit, self.end)

    @property
    def start_date_offset(self) -> Optional[pd.DateOffset]:
        if self.is_open:
            return None
        assert self.start is not None
        assert isinstance(self.unit, TimeUnit)
        return self._to_dateOffset(self.unit, self.start)

    @property
    def end_date_offset(self) -> pd.DateOffset:
        assert isinstance(self.unit, TimeUnit)
        return self._to_dateOffset(self.unit, self.end)

    @staticmethod
    def _to_dateOffset(unit: TimeUnit, value: int) -> pd.DateOffset:
        min_int = np.iinfo(np.int32).min
        max_int = np.iinfo(np.int32).max
        response = validate_int(value, min_int=min_int, max_int=max_int)
        if not response.ok:
            raise ValueError(response.message())
        return pd.DateOffset(**{unit.value: value})  # type: ignore

    @staticmethod
    def merge_ranges(offset1: 'DateOffsetRange',
                     offset2: 'DateOffsetRange') -> 'DateOffsetRange':
        r"""Merges two :class:`DateOffsetRange` objects by taking the earlier
        of the two starts and the later of the two ends."""
        offset1, offset2 = DateOffsetRange.move_to_same_unit(offset1, offset2)
        _start = None
        if offset1.start is not None and offset2.start is not None:
            _start = min(offset1.start, offset2.start)
        _end = max(offset1.end, offset2.end)
        return DateOffsetRange(_start, _end, offset1.unit)

    @staticmethod
    def move_to_same_unit(
        offset1: 'DateOffsetRange', offset2: 'DateOffsetRange'
    ) -> Tuple['DateOffsetRange', 'DateOffsetRange']:
        r"""Returns a pair of `DateOffsetRange`s where start and end use the
        same time unit. Note that this may result in inconsistencies if
        combining months to days."""
        if offset1.unit == offset2.unit:
            return (offset1, offset2)

        def months_to_days(unit: TimeUnit, value: int) -> Tuple[TimeUnit, int]:
            if unit != TimeUnit.MONTHS:
                return unit, value
            logger.warning(
                'Turning a month time unit to days. This may result in '
                'inconsistent results for months that are not 30 days long. '
                'To avoid this, do not combine different time units in the '
                'same query.')
            return TimeUnit.DAYS, 30 * value

        def days_to_hours(unit: TimeUnit, value: int) -> Tuple[TimeUnit, int]:
            if unit != TimeUnit.DAYS:
                return unit, value
            return TimeUnit.HOURS, 24 * value

        def hours_to_minutes(unit: TimeUnit,
                             value: int) -> Tuple[TimeUnit, int]:
            if unit != TimeUnit.HOURS:
                return unit, value
            return TimeUnit.MINUTES, 60 * value

        for offset in [offset1, offset2]:
            assert isinstance(offset.unit, TimeUnit)
            if offset.start is not None:
                _, offset.start = months_to_days(offset.unit, offset.start)
            offset.unit, offset.end = months_to_days(offset.unit, offset.end)

        if offset1.unit == offset2.unit:
            return (offset1, offset2)
        for offset in [offset1, offset2]:
            assert isinstance(offset.unit, TimeUnit)
            if offset.start is not None:
                _, offset.start = days_to_hours(offset.unit, offset.start)
            offset.unit, offset.end = days_to_hours(offset.unit, offset.end)

        if offset1.unit == offset2.unit:
            return (offset1, offset2)
        for offset in [offset1, offset2]:
            assert isinstance(offset.unit, TimeUnit)
            if offset.start is not None:
                _, offset.start = hours_to_minutes(offset.unit, offset.start)
            offset.unit, offset.end = hours_to_minutes(offset.unit, offset.end)

        return offset1, offset2

    @staticmethod
    def date_offset_repr(date_offset: pd.DateOffset) -> str:
        r"""String representation of a `pd.DateOffset` object.
        e.g. <DateOffset: months=8> -> 8 months.
        Args:
            date_offset (pd.DateOffset): A `pd.DateOffset` object.
        Returns:
            str: The string representation of the `pd.DateOffset` object.
        """
        unit, value = list(date_offset.kwds.items())[0]
        return f"{value} {unit}"

    @staticmethod
    def timedelta_repr(td: pd.Timedelta, unit: TimeUnit) -> str:
        r"""String representation of a `pd.Timedelta` object with
        specified time unit.
        e.g. Timedelta('30 days 18:22:19') with hours ->
            `'738 hours and 22 minutes'`,
        Timedelta('30 days 18:22:19') with days -> `'30 days and 18 hours'`,
        Timedelta('30 days 18:22:19') with months -> `'1 month and 18 hours'`.
        Args:
            td (pd.Timedelta): A `pd.Timedelta` object.
            unit (TimeUnit): Time unit for the string representation.
        Returns:
            str: The string representation of the `pd.Timedelta` object.
        """
        def seconds_to_minutes(value: int) -> Tuple[int, int]:
            return divmod(value, 60)

        def minutes_to_hours(value: int) -> Tuple[int, int]:
            return divmod(value, 60)

        def hours_to_days(value: int) -> Tuple[int, int]:
            return divmod(value, 24)

        def days_to_months(value: int) -> Tuple[int, int]:
            return divmod(value, 30)

        seconds = int(td.total_seconds())
        minutes, _ = seconds_to_minutes(seconds)
        # Number of hours and remaining minutes
        hours, r_minutes = minutes_to_hours(minutes)
        # Number of days and remain hours
        days, r_hours = hours_to_days(hours)
        # Number of months and remain days
        months, r_days = days_to_months(days)

        minutes_repr = (f'{minutes} minutes'
                        if minutes > 1 else f'{minutes} minute')
        r_minutes_repr = (f'{r_minutes} minutes'
                          if r_minutes > 1 else f'{r_minutes} minute')
        hours_repr = f'{hours} hours' if hours > 1 else f'{hours} hour'
        r_hours_repr = f'{r_hours} hours' if r_hours > 1 else f'{r_hours} hour'
        days_repr = f'{days} days' if days > 1 else f'{days} day'
        r_days_repr = f'{r_days} days' if r_days > 1 else f'{r_days} day'
        months_repr = f'{months} months' if months > 1 else f'{months} month'

        if unit == TimeUnit.MINUTES:
            return minutes_repr
        if unit == TimeUnit.HOURS:
            if r_minutes == 0:
                return hours_repr
            return f'{hours_repr} and {r_minutes_repr}'
        elif unit == TimeUnit.DAYS:
            # We don't return minutes granularity for DAYS or longer because
            # who cares
            if r_hours == 0:
                return days_repr
            return f'{days_repr} and {r_hours_repr}'
        else:
            if r_days == 0 and r_hours == 0:
                return months_repr
            elif r_days == 0:
                return f'{months_repr} and {r_hours_repr}'
            elif r_hours == 0:
                return f'{months_repr} and {r_days_repr}'
            else:
                return (f'{months_repr}, {r_days_repr} and {r_hours_repr}')

    @staticmethod
    def date_offset_multiply(
        date_offset: pd.DateOffset,
        multiplier: int,
    ) -> pd.DateOffset:
        r"""Obtain the date offset after applying a multiplier.
        e.g. date_offset_multiply(pd.DateOffset(days=3), 4) =
        pd.DateOffset(days=12).
        Args:
            date_offset (pd.DateOffset): date offset to be multiplied.
            multiplier (int): An integer multiplier.
            pd.DateOffset: date offset after multiplication.
        Returns:
            pd.DateOffset: date offset representing the product of
            `date_offset` and `multiplier`.
        """
        date_offset_product = pd.DateOffset(
            **{k: v * multiplier
               for (k, v) in list(date_offset.kwds.items())})
        return date_offset_product

    @staticmethod
    def date_offset_add(
        date_offset_lhs_addend: pd.DateOffset,
        date_offset_rhs_addend: pd.DateOffset,
    ) -> pd.DateOffset:
        r"""Calculate the sum between `date_offset_lhs_addend` and
        `date_offset_rhs_addend`.
        e.g. date_offset_add(pd.DateOffset(months=3),
        pd.DateOffset(months=1, days=1)) = pd.DateOffset(months=4, days=1)
        Args:
            date_offset_lhs_addend (pd.DateOffset): left hand side date
                offset addend.
            date_offset_rhs_addend (pd.DateOffset): right hand side date
                offset addend.
        Returns:
            pd.DateOffset: date offset representing the sum between
                `date_offset_lhs_addend` and `date_offset_rhs_addend`.
        """
        date_offset_lhs_addend_dict = date_offset_lhs_addend.kwds
        date_offset_rhs_addend_dict = date_offset_rhs_addend.kwds
        timeframe_dict = date_offset_lhs_addend_dict
        for key, value in date_offset_rhs_addend_dict.items():
            if key in timeframe_dict:
                timeframe_dict[key] += value
            else:
                timeframe_dict[key] = value
        timeframe = pd.DateOffset(**timeframe_dict)
        return timeframe

    @staticmethod
    def date_offset_subtract(
        date_offset_minuend: pd.DateOffset,
        date_offset_subtrahend: pd.DateOffset,
    ) -> pd.DateOffset:
        r"""Calculate the difference between `date_offset_minuend` and
        `date_offset_subtrahend`.
        e.g. date_offset_subtract(pd.DateOffset(months=3),
        pd.DateOffset(months=1, days=1)) = pd.DateOffset(months=2, days=-1)
        Args:
            date_offset_minuend (pd.DateOffset): date offset to be subtracted.
            date_offset_subtrahend (pd.DateOffset): date offset to subtract.
        Returns:
            pd.DateOffset: date offset representing the difference between
                `date_offset_minuend` and `date_offset_subtrahend`.
        """
        date_offset_minuend_dict = date_offset_minuend.kwds
        date_offset_subtrahend_dict = date_offset_subtrahend.kwds
        timeframe_dict = date_offset_minuend_dict
        for key in date_offset_subtrahend_dict:
            if key in timeframe_dict:
                timeframe_dict[key] -= date_offset_subtrahend_dict[key]
            else:
                timeframe_dict[key] = -date_offset_subtrahend_dict[key]
        timeframe = pd.DateOffset(**timeframe_dict)
        return timeframe

    @staticmethod
    def date_offset_gt(
        date_offset_left: pd.DateOffset,
        date_offset_right: pd.DateOffset,
    ) -> bool:
        """Compare between `date_offset_left` and `date_offset_right`.
        Not guaranteed to be consistent on undefined comparisons, e.g.
        30 days > 1 month.

        Args:
            date_offset_left (pd.DateOffset): The left hand side
                pd.DateOffset object.
            date_offset_right (pd.DateOffset): The right hand side
                pd.DateOffset object.

        Returns:
            (bool): Whether `date_offset_left` is larger than
                `date_offset_right`.
        """
        abs_date = pd.Timestamp("1970-01-01")
        assert isinstance(abs_date, pd.Timestamp)
        return (abs_date + date_offset_left) > (abs_date + date_offset_right)

    @property
    def is_open(self) -> bool:
        return self.start is None

    @property
    def timeframe(self) -> Optional[pd.Timedelta]:
        # We cannot subtract two date offsets. As such, we first convert each
        # offset into a timestamp, and subtract them afterwards.
        if self.is_open:
            return None
        abs_date = pd.Timestamp("1970-01-01")
        assert isinstance(abs_date, pd.Timestamp)
        # Additional type checks to not confuse type checkers
        start_date_offset = self.start_date_offset
        assert isinstance(start_date_offset, pd.DateOffset)
        return (abs_date + self.end_date_offset) - (abs_date +
                                                    start_date_offset)
