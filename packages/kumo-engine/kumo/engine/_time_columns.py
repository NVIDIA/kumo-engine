# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Time columns are settled from every staged value before DiskGraph parses
them. A column of whole numbers is converted only when one reading fits
every value: a `YYYYMMDD` or `YYYYMMDDHHMMSS` date shape, or epoch seconds,
milliseconds, microseconds or nanoseconds told apart by digit count. A
column of slash dates takes the one day and month order the data allows.
Either decision is declared as the column's `timestamp_format`, so DiskGraph
applies it to every value instead of voting on a sample. A value that no
calendar holds (`2024-02-30`, hour 25, `0000-00-00`) or that falls outside
the nanosecond timestamp range (1677-09-21 to 2262-04-11) is made missing in
the staged data and counted, never moved to a neighbouring date. Bare
timestamps are read as UTC and zoned ones converted to UTC; a column mixing
the two is reported and given a format that reads both.
"""

import os
import warnings
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Whole epoch seconds a nanosecond timestamp can hold.
_MIN_SECS = -9_223_372_036
_MAX_SECS = 9_223_372_036
_RANGE = '1677-09-21 to 2262-04-11'
_EXAMPLES = 3

_READINGS = {
    'ymd': ('%Y%m%d', 'YYYYMMDD dates', 'a YYYYMMDD date'),
    'ymdhms':
    ('%Y%m%d%H%M%S', 'YYYYMMDDHHMMSS date-times', 'a YYYYMMDDHHMMSS date-time'),
    's': ('s', 'epoch seconds', 'epoch seconds'),
    'ms': ('ms', 'epoch milliseconds', 'epoch milliseconds'),
    'us': ('us', 'epoch microseconds', 'epoch microseconds'),
    'ns': ('ns', 'epoch nanoseconds', 'epoch nanoseconds'),
    'none': (None, None, 'too small or too large to be a date'),
}
_ISO = (r'^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})'
        r'(?:(?P<sep>[T ])(?P<H>\d{2}):(?P<M>\d{2})'
        r'(?::(?P<S>\d{2})(?:\.\d+)?)?'
        r'(?: ?(?P<z>Z|[+-]\d{2}:?\d{2}))?)?$')
_YEAR_SLASH = (r'^(?P<y>\d{4})/(?P<m>\d{1,2})/(?P<d>\d{1,2})'
               r'(?: (?P<H>\d{1,2}):(?P<M>\d{2})(?::(?P<S>\d{2}))?)?$')
_SLASH = (r'^(?P<a>\d{1,2})/(?P<b>\d{1,2})/(?P<y>\d{4})'
          r'(?: (?P<H>\d{1,2}):(?P<M>\d{2})(?::(?P<S>\d{2}))?)?$')
_DIGITS = r'^\d+$'
_INT64_MAX = '9223372036854775807'
_SLASH_SHAPES = {0: '', 1: ' %H:%M', 2: ' %H:%M:%S'}


@dataclass
class _Tally:
    count: int = 0
    examples: List[Any] = field(default_factory=list)

    def add(self, mask: pa.Array, values: pa.Array) -> None:
        hits = pc.sum(mask).as_py() or 0
        if hits:
            self.count += hits
            for value in pc.filter(values, mask)[:_EXAMPLES].to_pylist():
                if len(self.examples) < _EXAMPLES:
                    self.examples.append(value)


@dataclass
class _Scan:
    r"""What one pass over a column found, by the decision it may lead to.
    Invalid values are tallied under `fixed` when the reading does not
    depend on a decision, and under `md`/`dm` for slash dates, which are
    valid under one order or the other.
    """
    values: int = 0
    families: Counter = field(default_factory=Counter)
    readings: Counter = field(default_factory=Counter)
    reading_examples: Dict[str, Any] = field(default_factory=dict)
    zoned: int = 0
    bare: int = 0
    zoned_example: Any = None
    bare_example: Any = None
    separators: Counter = field(default_factory=Counter)
    slash_dm: _Tally = field(default_factory=_Tally)
    slash_md: _Tally = field(default_factory=_Tally)
    slash_both: _Tally = field(default_factory=_Tally)
    shapes: Counter = field(default_factory=Counter)
    impossible: Dict[str, _Tally] = field(
        default_factory=lambda: {k: _Tally()
                                 for k in ('fixed', 'md', 'dm')})
    far: Dict[str, _Tally] = field(
        default_factory=lambda: {k: _Tally()
                                 for k in ('fixed', 'md', 'dm')})
    invalid_parts: Dict[str, set] = field(
        default_factory=lambda: {k: set()
                                 for k in ('fixed', 'md', 'dm')})


@dataclass(frozen=True)
class _Decision:
    timestamp_format: Optional[str] = None
    order: str = 'fixed'
    reading: Optional[str] = None


def _number(parts: pa.Array, name: str) -> pa.Array:
    text = pc.struct_field(parts, name)
    return pc.cast(pc.if_else(pc.equal(text, ''), None, text), pa.int64())


def _fill(array: pa.Array, value: int) -> pa.Array:
    return pc.fill_null(array, pa.scalar(value, pa.int64()))


def _valid_date(y: pa.Array, m: pa.Array, d: pa.Array) -> pa.Array:
    leap = pc.and_(
        pc.equal(pc.bit_wise_and(y, 3), 0),
        pc.or_(
            pc.not_equal(pc.subtract(y, pc.multiply(pc.divide(y, 100), 100)),
                         0),
            pc.equal(pc.subtract(y, pc.multiply(pc.divide(y, 400), 400)), 0)))
    days = pc.if_else(
        pc.is_in(m, value_set=pa.array([4, 6, 9, 11], pa.int64())), 30,
        pc.if_else(pc.equal(m, 2), pc.add(28, pc.cast(leap, pa.int64())), 31))
    return pc.and_(pc.and_(pc.greater_equal(m, 1), pc.less_equal(m, 12)),
                   pc.and_(pc.greater_equal(d, 1), pc.less_equal(d, days)))


def _valid_time(hour: pa.Array, minute: pa.Array, second: pa.Array) -> pa.Array:
    return pc.and_(
        pc.less_equal(hour, 23),
        pc.and_(pc.less_equal(minute, 59), pc.less_equal(second, 59)))


def _epoch_seconds(y: pa.Array, m: pa.Array, d: pa.Array, hour: pa.Array,
                   minute: pa.Array, second: pa.Array) -> pa.Array:
    r"""Days from civil (Howard Hinnant's algorithm) in whole seconds. The
    year is offset by one 400-year era so every division is of a positive
    number, which Arrow truncates.
    """
    y = pc.add(pc.subtract(y, pc.cast(pc.less_equal(m, 2), pa.int64())), 400)
    era = pc.divide(y, 400)
    yoe = pc.subtract(y, pc.multiply(era, 400))
    mp = pc.if_else(pc.greater(m, 2), pc.subtract(m, 3), pc.add(m, 9))
    doy = pc.add(pc.divide(pc.add(pc.multiply(mp, 153), 2), 5),
                 pc.subtract(d, 1))
    doe = pc.add(
        pc.subtract(pc.add(pc.multiply(yoe, 365), pc.divide(yoe, 4)),
                    pc.divide(yoe, 100)), doy)
    days = pc.subtract(pc.add(pc.multiply(era, 146097), doe), 719468 + 146097)
    return pc.add(
        pc.multiply(days, 86400),
        pc.add(pc.multiply(hour, 3600), pc.add(pc.multiply(minute, 60),
                                               second)))


def _in_range(secs: pa.Array) -> pa.Array:
    return pc.and_(pc.greater_equal(secs, _MIN_SECS),
                   pc.less_equal(secs, _MAX_SECS))


def _offset_seconds(zone: pa.Array) -> pa.Array:
    utc = pc.or_(pc.equal(zone, ''), pc.equal(zone, 'Z'))
    digits = pc.replace_substring(pc.if_else(utc, '+0000', zone), ':', '')
    offset = pc.add(
        pc.multiply(pc.cast(pc.utf8_slice_codeunits(digits, 1, 3), pa.int64()),
                    3600),
        pc.multiply(pc.cast(pc.utf8_slice_codeunits(digits, 3, 5), pa.int64()),
                    60))
    return _fill(
        pc.if_else(pc.starts_with(digits, '-'), pc.negate(offset), offset), 0)


def _digits_between(values: pa.Array, low: int, high: int) -> pa.Array:
    return pc.and_(pc.greater_equal(values, low), pc.less(values, high))


def _int_readings(values: pa.Array) -> Tuple[pa.Array, pa.Array]:
    r"""Each whole number's one reading and the epoch seconds under it.
    Shapes come first: eight digits that are a calendar date, fourteen that
    are a date and a time. Then size: nine or ten digits are seconds, twelve
    or thirteen milliseconds, fifteen or sixteen microseconds, eighteen or
    more nanoseconds. Anything else, including a negative number, has no
    reading.
    """
    ymd = pc.and_(
        _digits_between(values, 10**7, 10**8),
        _valid_date(
            pc.divide(values, 10**4),
            pc.subtract(pc.divide(values, 100),
                        pc.multiply(pc.divide(values, 10**4), 100)),
            pc.subtract(values, pc.multiply(pc.divide(values, 100), 100))))
    long = pc.divide(values, 10**6)
    ymdhms = pc.and_(
        _digits_between(values, 10**13, 10**14),
        pc.and_(
            _valid_date(
                pc.divide(long, 10**4),
                pc.subtract(pc.divide(long, 100),
                            pc.multiply(pc.divide(long, 10**4), 100)),
                pc.subtract(long, pc.multiply(pc.divide(long, 100), 100))),
            _valid_time(
                pc.subtract(pc.divide(values, 10**4),
                            pc.multiply(pc.divide(values, 10**6), 100)),
                pc.subtract(pc.divide(values, 100),
                            pc.multiply(pc.divide(values, 10**4), 100)),
                pc.subtract(values, pc.multiply(pc.divide(values, 100), 100)))))
    seconds = _digits_between(values, 10**8, 10**10)
    millis = _digits_between(values, 10**11, 10**13)
    micros = _digits_between(values, 10**14, 10**16)
    nanos = pc.greater_equal(values, 10**17)
    reading = pc.case_when(
        pc.make_struct(ymd, ymdhms, seconds, millis, micros, nanos), 'ymd',
        'ymdhms', 's', 'ms', 'us', 'ns', 'none')
    date = pc.if_else(ymd, values, long)
    secs = pc.case_when(
        pc.make_struct(pc.or_(ymd, ymdhms), seconds, millis, micros, nanos),
        _epoch_seconds(
            pc.divide(date, 10**4),
            pc.subtract(pc.divide(date, 100),
                        pc.multiply(pc.divide(date, 10**4), 100)),
            pc.subtract(date, pc.multiply(pc.divide(date, 100), 100)),
            pc.if_else(
                ymd, 0,
                pc.subtract(pc.divide(values, 10**4),
                            pc.multiply(pc.divide(values, 10**6), 100))),
            pc.if_else(
                ymd, 0,
                pc.subtract(pc.divide(values, 100),
                            pc.multiply(pc.divide(values, 10**4), 100))),
            pc.if_else(
                ymd, 0,
                pc.subtract(values, pc.multiply(pc.divide(values, 100), 100)))),
        values, pc.divide(values, 10**3), pc.divide(values, 10**6),
        pc.divide(values, 10**9), pa.scalar(None, pa.int64()))
    return pc.if_else(pc.is_null(values), None, reading), secs


def _as_int64(values: pa.Array) -> Optional[pa.Array]:
    r"""Whole numbers as `int64`, or `None` for a type DiskGraph reads its
    own way. A value beyond `int64` has no reading and becomes null here,
    which the caller counts as such.
    """
    kind = values.type
    if pa.types.is_dictionary(kind):
        values, kind = values.cast(kind.value_type), kind.value_type
    if pa.types.is_integer(kind):
        if kind == pa.uint64():
            values = pc.if_else(pc.greater(values, pa.scalar(2**63 - 1, kind)),
                                None, values)
        return values.cast(pa.int64())
    if pa.types.is_decimal(kind) and kind.scale == 0:
        if kind.precision >= len(_INT64_MAX):
            fits = pc.and_(
                pc.greater_equal(values, pa.scalar(Decimal(-2**63), kind)),
                pc.less_equal(values, pa.scalar(Decimal(2**63 - 1), kind)))
            values = pc.if_else(fits, values, None)
        return values.cast(pa.int64())
    return None


def _native_seconds(values: pa.Array) -> Optional[pa.Array]:
    kind = values.type
    if pa.types.is_date32(kind):
        return pc.multiply(values.cast(pa.int32()).cast(pa.int64()), 86400)
    if pa.types.is_date64(kind):
        return pc.divide(values.cast(pa.int64()), 1000)
    if pa.types.is_timestamp(kind):
        ticks = {'s': 1, 'ms': 10**3, 'us': 10**6, 'ns': 10**9}[kind.unit]
        return pc.divide(values.cast(pa.int64()), ticks)
    return None


def _text_numbers(text: pa.Array, digits: pa.Array) -> pa.Array:
    r"""The digit-only strings as `int64`, null where too wide for it."""
    length = pc.utf8_length(text)
    fits = pc.and_(
        digits,
        pc.or_(
            pc.less(length, len(_INT64_MAX)),
            pc.and_(pc.equal(length, len(_INT64_MAX)),
                    pc.less_equal(text, _INT64_MAX))))
    return pc.cast(pc.if_else(fits, text, None), pa.int64())


def _strings(values: pa.Array) -> Optional[pa.Array]:
    kind = values.type
    if pa.types.is_dictionary(kind):
        values, kind = values.cast(kind.value_type), kind.value_type
    if pa.types.is_string(kind) or pa.types.is_large_string(kind):
        return values.cast(pa.string())
    return None


class _Column:
    r"""One time column's scan over the staged parts and the decision it
    leads to; `rewrite` applies the decision's missing values in place.
    """
    def __init__(self, alias: str, name: str, role: str):
        self.alias = alias
        self.name = name
        self.role = role
        self.scan = _Scan()
        self.decision = _Decision()

    @property
    def label(self) -> str:
        return f"{self.role} column {self.alias}.{self.name}"

    def observe(self, part: str, values: pa.Array) -> None:
        present = values.drop_null()
        self.scan.values += len(present)
        if not len(present):
            return
        secs = _native_seconds(values)
        if secs is not None:
            self.scan.families['native'] += len(present)
            self._tally('fixed', part, values, pc.invert(_in_range(secs)), None)
            return
        numbers = _as_int64(values)
        if numbers is not None:
            self.scan.families['digits'] += len(present)
            self._observe_numbers(part, values, pc.is_valid(values), numbers)
            return
        text = _strings(values)
        if text is None:
            self.scan.families['other'] += len(present)
            return
        self._observe_text(part, text)

    def _observe_numbers(self, part: str, values: pa.Array, present: pa.Array,
                         numbers: pa.Array) -> None:
        r"""Tallies the reading of each whole number; one too wide for
        `int64` arrives as null and has none.
        """
        reading, secs = _int_readings(numbers)
        reading = pc.if_else(pc.and_(present, pc.is_null(numbers)), 'none',
                             reading)
        for code, count in zip(*pc.value_counts(reading).flatten()):
            code = code.as_py()
            if code is None:
                continue
            self.scan.readings[code] += count.as_py()
            if code not in self.scan.reading_examples:
                self.scan.reading_examples[code] = pc.filter(
                    values, pc.equal(reading, code))[0].as_py()
        self._tally('fixed', part, values, pc.invert(_in_range(secs)), None)

    def _observe_text(self, part: str, text: pa.Array) -> None:
        iso = pc.extract_regex(text, _ISO)
        year_slash = pc.extract_regex(text, _YEAR_SLASH)
        slash = pc.extract_regex(text, _SLASH)
        families = {
            'iso': pc.is_valid(iso),
            'year_slash': pc.is_valid(year_slash),
            'slash': pc.is_valid(slash),
            'digits': pc.fill_null(pc.match_substring_regex(text, _DIGITS),
                                   False),
        }
        for family, mask in families.items():
            self.scan.families[family] += pc.sum(mask).as_py() or 0
        recognized = pc.or_(pc.or_(families['iso'], families['year_slash']),
                            pc.or_(families['slash'], families['digits']))
        other = pc.and_(pc.is_valid(text), pc.invert(recognized))
        self.scan.families['other'] += pc.sum(other).as_py() or 0

        if pc.sum(families['digits']).as_py():
            self._observe_numbers(part, text, families['digits'],
                                  _text_numbers(text, families['digits']))

        for parts, kind in ((iso, 'iso'), (year_slash, 'year_slash')):
            matched = pc.fill_null(pc.is_valid(parts), False)
            if not pc.sum(matched).as_py():
                continue
            hour = _fill(_number(parts, 'H'), 0)
            minute = _fill(_number(parts, 'M'), 0)
            second = _fill(_number(parts, 'S'), 0)
            y, m, d = (_number(parts, k) for k in 'ymd')
            valid = pc.and_(_valid_date(y, m, d),
                            _valid_time(hour, minute, second))
            secs = _epoch_seconds(y, m, d, hour, minute, second)
            if kind == 'iso':
                zone = pc.fill_null(pc.struct_field(parts, 'z'), '')
                secs = pc.subtract(secs, _offset_seconds(zone))
                zoned = pc.and_(matched, pc.not_equal(zone, ''))
                bare = pc.and_(matched, pc.equal(zone, ''))
                self._note_zones(text, zoned, bare)
                for separator, count in zip(*pc.value_counts(
                        pc.filter(pc.struct_field(parts, 'sep'),
                                  matched)).flatten()):
                    if separator.as_py():
                        self.scan.separators[separator.as_py()] += (
                            count.as_py())
            self._tally('fixed', part, text, pc.and_(matched, pc.invert(valid)),
                        'impossible')
            self._tally(
                'fixed', part, text,
                pc.and_(pc.and_(matched, valid), pc.invert(_in_range(secs))),
                'far')

        matched = pc.fill_null(pc.is_valid(slash), False)
        if pc.sum(matched).as_py():
            self._observe_slash(part, text, slash, matched)

    def _note_zones(self, text: pa.Array, zoned: pa.Array,
                    bare: pa.Array) -> None:
        for attribute, mask in (('zoned', zoned), ('bare', bare)):
            count = pc.sum(mask).as_py() or 0
            if count:
                setattr(self.scan, attribute,
                        getattr(self.scan, attribute) + count)
                if getattr(self.scan, f'{attribute}_example') is None:
                    setattr(self.scan, f'{attribute}_example',
                            pc.filter(text, mask)[0].as_py())

    def _observe_slash(self, part: str, text: pa.Array, parts: pa.Array,
                       matched: pa.Array) -> None:
        a, b, y = (_number(parts, k) for k in 'aby')
        hour = _fill(_number(parts, 'H'), 0)
        minute = _fill(_number(parts, 'M'), 0)
        second = _fill(_number(parts, 'S'), 0)
        time_ok = _valid_time(hour, minute, second)
        shape = pc.add(pc.cast(pc.is_valid(_number(parts, 'H')), pa.int64()),
                       pc.cast(pc.is_valid(_number(parts, 'S')), pa.int64()))
        for value, count in zip(
                *pc.value_counts(pc.filter(shape, matched)).flatten()):
            self.scan.shapes[value.as_py()] += count.as_py()
        orders = {
            'md': (_valid_date(y, a,
                               b), _epoch_seconds(y, a, b, hour, minute,
                                                  second)),
            'dm': (_valid_date(y, b,
                               a), _epoch_seconds(y, b, a, hour, minute,
                                                  second)),
        }
        valid_md = pc.and_(matched, pc.and_(orders['md'][0], time_ok))
        valid_dm = pc.and_(matched, pc.and_(orders['dm'][0], time_ok))
        self.scan.slash_md.add(pc.and_(valid_md, pc.invert(valid_dm)), text)
        self.scan.slash_dm.add(pc.and_(valid_dm, pc.invert(valid_md)), text)
        self.scan.slash_both.add(pc.and_(valid_md, valid_dm), text)
        for order, (valid_date, secs) in orders.items():
            valid = pc.and_(valid_date, time_ok)
            self._tally(order, part, text, pc.and_(matched, pc.invert(valid)),
                        'impossible')
            self._tally(
                order, part, text,
                pc.and_(pc.and_(matched, valid), pc.invert(_in_range(secs))),
                'far')

    def _tally(self, order: str, part: str, values: pa.Array, mask: pa.Array,
               kind: Optional[str]) -> None:
        mask = pc.fill_null(mask, False)
        if not (pc.sum(mask).as_py() or 0):
            return
        self.scan.invalid_parts[order].add(part)
        if kind is None:
            self.scan.far[order].add(mask, values)
        else:
            getattr(self.scan, kind)[order].add(mask, values)

    def decide(self) -> None:
        scan = self.scan
        if not scan.values:
            return
        known = {k: v for k, v in scan.families.items() if v and k != 'other'}
        if set(known) == {'digits'}:
            self._decide_numbers()
        elif set(known) == {'slash'}:
            self._decide_slash()
        elif set(known) == {'iso'} and scan.zoned and scan.bare:
            separator = (scan.separators
                         or Counter({' ': 1})).most_common(1)[0][0]
            self.decision = _Decision(f'%Y-%m-%d{separator}%H:%M:%S%.f%#z')
            warnings.warn(
                f"Time column {self.alias}.{self.name} mixes {scan.zoned} "
                f"value{'s' if scan.zoned != 1 else ''} with a timezone, "
                f"converted to UTC, and {scan.bare} without one, read as "
                f"UTC (e.g. {scan.zoned_example} and {scan.bare_example}). "
                "Declare one timezone at the source so the events line up.",
                UserWarning,
                stacklevel=5,
            )
        self._report('impossible', 'are not dates on the calendar',
                     'is not a date on the calendar',
                     'Correct them at the source')
        self._report('far', f'fall outside {_RANGE}', f'falls outside {_RANGE}',
                     'Replace placeholder dates with nulls at the source')

    def _decide_numbers(self) -> None:
        scan = self.scan
        readings = [code for code in _READINGS if scan.readings.get(code)]
        if readings == ['none'] or len(readings) != 1:
            parts = ', '.join(
                f"{scan.readings[code]} "
                f"{'is' if scan.readings[code] == 1 else 'are'} "
                f"{_READINGS[code][2]} (e.g. {scan.reading_examples[code]})"
                for code in readings)
            raise ValueError(
                f"Time column {self.alias}.{self.name} holds whole numbers "
                f"with no single reading: {parts}. Set timestamp_format on "
                f"the column to 's', 'ms', 'us', 'ns', '%Y%m%d' or "
                f"'%Y%m%d%H%M%S', or convert it to a timestamp at the "
                "source.")
        code = readings[0]
        hint, description, _ = _READINGS[code]
        example = scan.reading_examples[code]
        rendered = _render(
            _int_readings(pa.array([int(example)], pa.int64()))[1][0].as_py())
        self.decision = _Decision(hint, reading=code)
        warnings.warn(
            f"Time column {self.alias}.{self.name} holds whole numbers; its "
            f"{scan.values} values are read as {description} ({example} is "
            f"{rendered}). Set timestamp_format on the column to read them "
            "differently.",
            UserWarning,
            stacklevel=5,
        )

    def _decide_slash(self) -> None:
        scan = self.scan
        dm, md = scan.slash_dm, scan.slash_md
        if dm.count and md.count:
            raise ValueError(
                f"Time column {self.alias}.{self.name} holds slash dates in "
                f"both orders: {dm.count} value{'s' if dm.count != 1 else ''}"
                f" (e.g. {dm.examples[0]}) read{'s' if dm.count == 1 else ''}"
                f" only day first and {md.count} (e.g. {md.examples[0]}) "
                "only month first. Fix the column at the source, or set "
                "timestamp_format on the column to the order you want and "
                "the other values become missing.")
        if not dm.count and not md.count:
            example = (scan.slash_both.examples
                       or scan.impossible['md'].examples)[0]
            raise ValueError(
                f"Time column {self.alias}.{self.name} holds slash dates "
                "whose day and month order the data does not settle: every "
                f"one of its {scan.values} values (e.g. {example}) reads "
                "both ways. Set timestamp_format on the column to "
                "'%d/%m/%Y' or '%m/%d/%Y'.")
        order, settled = ('dm', dm) if dm.count else ('md', md)
        shape = _SLASH_SHAPES[scan.shapes.most_common(1)[0][0]]
        fmt = ('%d/%m/%Y' if order == 'dm' else '%m/%d/%Y') + shape
        self.decision = _Decision(fmt, order=order)
        warnings.warn(
            f"Time column {self.alias}.{self.name} holds slash dates read "
            f"{'day' if order == 'dm' else 'month'} first ({fmt}) because "
            f"{settled.count} of {scan.values} values (e.g. "
            f"{settled.examples[0]}) read only that way. Set "
            "timestamp_format on the column to override.",
            UserWarning,
            stacklevel=5,
        )

    def _report(self, kind: str, plural: str, singular: str,
                action: str) -> None:
        tallies = getattr(self.scan, kind)
        tally = _Tally()
        for order in dict.fromkeys(('fixed', self.decision.order)):
            if tallies[order].count:
                tally.count += tallies[order].count
                tally.examples.extend(tallies[order].examples)
        if not tally.count:
            return
        examples = ', '.join(str(v) for v in tally.examples[:_EXAMPLES])
        if self.role == 'time':
            outcome = (
                f"{'it is' if tally.count == 1 else 'they are'} "
                "treated as missing and "
                f"{'its row is' if tally.count == 1 else 'their rows are'}"
                " dropped like any row without a time")
            keep = f"keep the row{'s' if tally.count != 1 else ''}"
        else:
            outcome = (f"{'it is' if tally.count == 1 else 'they are'} "
                       "treated as missing")
            keep = f"keep the value{'s' if tally.count != 1 else ''}"
        warnings.warn(
            f"{tally.count} value{'s' if tally.count != 1 else ''} in "
            f"{self.label} {plural if tally.count != 1 else singular} (e.g. "
            f"{examples}); {outcome}. {action} to {keep}.",
            UserWarning,
            stacklevel=5,
        )

    def invalid_parts(self) -> set:
        parts = set(self.scan.invalid_parts['fixed'])
        if self.decision.order != 'fixed':
            parts |= self.scan.invalid_parts[self.decision.order]
        return parts

    def invalid(self, values: pa.Array) -> pa.Array:
        r"""The values the decision makes missing: none when the column
        is left to DiskGraph's own reading.
        """
        secs = _native_seconds(values)
        if secs is not None:
            return pc.fill_null(pc.invert(_in_range(secs)), False)
        numbers = _as_int64(values)
        if numbers is not None:
            return pc.fill_null(pc.invert(_in_range(_int_readings(numbers)[1])),
                                False)
        text = _strings(values)
        if text is None:
            return pa.array([False] * len(values), pa.bool_())
        invalid = pa.array([False] * len(values), pa.bool_())
        if self.decision.reading is not None:
            numbers = _text_numbers(
                text,
                pc.fill_null(pc.match_substring_regex(text, _DIGITS), False))
            invalid = pc.or_(
                invalid,
                pc.fill_null(pc.invert(_in_range(_int_readings(numbers)[1])),
                             False))
        for pattern, kind in ((_ISO, 'iso'), (_YEAR_SLASH, 'year_slash'),
                              (_SLASH, 'slash')):
            parts = pc.extract_regex(text, pattern)
            matched = pc.fill_null(pc.is_valid(parts), False)
            if not pc.sum(matched).as_py():
                continue
            hour = _fill(_number(parts, 'H'), 0)
            minute = _fill(_number(parts, 'M'), 0)
            second = _fill(_number(parts, 'S'), 0)
            if kind == 'slash':
                if self.decision.order == 'fixed':
                    continue
                a, b, y = (_number(parts, k) for k in 'aby')
                m, d = (a, b) if self.decision.order == 'md' else (b, a)
            else:
                y, m, d = (_number(parts, k) for k in 'ymd')
            valid = pc.and_(_valid_date(y, m, d),
                            _valid_time(hour, minute, second))
            secs = _epoch_seconds(y, m, d, hour, minute, second)
            if kind == 'iso':
                secs = pc.subtract(
                    secs,
                    _offset_seconds(
                        pc.fill_null(pc.struct_field(parts, 'z'), '')))
            bad = pc.or_(pc.invert(valid), pc.invert(_in_range(secs)))
            invalid = pc.or_(invalid, pc.and_(matched, pc.fill_null(bad,
                                                                    False)))
        return pc.fill_null(invalid, False)


def _render(secs: Optional[int]) -> str:
    if secs is None or not _MIN_SECS <= secs <= _MAX_SECS:
        return f'outside {_RANGE}'
    return datetime.fromtimestamp(secs,
                                  timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


def _parquet_files(paths: Sequence[str]) -> List[Path]:
    return [
        file for path in paths
        for file in (sorted(Path(path).rglob('*.parquet')) if Path(path).is_dir(
        ) else [Path(path)])
    ]


def _rewrite(file: Path, columns: Dict[str, Callable[[pa.Array],
                                                     pa.Array]]) -> None:
    r"""Rewrites one staged part with the given columns' invalid values set
    to null, keeping every other byte of the schema and data. The part is
    replaced atomically so a failure leaves the original in place.
    """
    parquet = pq.ParquetFile(file)
    schema = parquet.schema_arrow
    for name in columns:
        index = schema.get_field_index(name)
        field_ = schema.field(index).with_nullable(True)
        if pa.types.is_dictionary(field_.type):
            field_ = field_.with_type(field_.type.value_type)
        schema = schema.set(index, field_)
    temporary = file.with_name(f'.{file.name}.rewrite')
    with pq.ParquetWriter(temporary, schema) as writer:
        for batch in parquet.iter_batches():
            arrays = []
            for field_ in schema:
                array = batch.column(field_.name)
                if field_.name in columns:
                    if pa.types.is_dictionary(array.type):
                        array = array.cast(field_.type)
                    array = pc.if_else(columns[field_.name](array), None, array)
                arrays.append(array)
            writer.write_batch(pa.RecordBatch.from_arrays(
                arrays, schema=schema))
    os.replace(temporary, file)


def resolve_time_columns(alias: str, definition: Any,
                         paths: Sequence[str]) -> Dict[str, str]:
    r"""Settles the time and end-time columns of one staged table from every
    value and returns the `timestamp_format` each one needs, by column
    name; a column DiskGraph already reads unambiguously, or whose format
    the table declares, gets none. Values no calendar or clock holds are
    nulled in the staged parts. Raises when a column of whole numbers has
    no single reading or a column of slash dates no single order.
    """
    declared = {
        column.name: column.timestamp_format
        for column in definition.cols
    }
    columns = [
        _Column(alias, name, role)
        for name, role in ((definition.time_col, 'time'),
                           (definition.end_time_col, 'end time'))
        if name and declared.get(name) is None
    ]
    if not columns:
        return {}
    files = _parquet_files(paths)
    for file in files:
        parquet = pq.ParquetFile(file)
        names = [
            column.name for column in columns
            if column.name in parquet.schema_arrow.names
        ]
        if not names:
            continue
        for batch in parquet.iter_batches(columns=names):
            for column in columns:
                if column.name in names:
                    column.observe(str(file), batch.column(column.name))
    for column in columns:
        column.decide()
    for file in files:
        invalid = {
            column.name: column.invalid
            for column in columns if str(file) in column.invalid_parts()
        }
        if invalid:
            _rewrite(file, invalid)
    return {
        column.name: column.decision.timestamp_format
        for column in columns if column.decision.timestamp_format is not None
    }
