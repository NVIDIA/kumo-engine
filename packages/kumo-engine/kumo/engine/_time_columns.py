# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Time columns are settled from every staged value before DiskGraph parses
them. A column of whole numbers is converted only when one reading fits
every value that has one: a `YYYYMMDD` or `YYYYMMDDHHMMSS` date shape, or
epoch seconds, milliseconds, microseconds or nanoseconds told apart by
digit count; a value with no reading (a placeholder `0`, a number too small
or too wide to be a date) is made missing and counted. A column of slash
dates takes the one day and month order the data allows and is staged in
the fullest time shape any value has.
Either decision is declared as the column's `timestamp_format`, so DiskGraph
applies it to every value instead of voting on a sample. A value that no
calendar holds (`2024-02-30`, hour 25, `0000-00-00`) or that falls outside
the nanosecond timestamp range (1677-09-21 to 2262-04-11) is made missing in
the staged data and counted, never moved to a neighbouring date. Bare
timestamps are read as UTC and zoned ones converted to UTC; a column mixing
the two is reported, staged in one shape (a space before a full time part)
and given a format that reads both. A column of ISO or year-first values in
which the scan made values missing is handed to DiskGraph with its format
declared, since DiskGraph otherwise decides the format from a part's first
rows and refuses a column whose sample the scan has made all null. A column
that also holds a value in a shape the scan does not read is left to
DiskGraph's own reading, with the value named, wherever the scan would
otherwise have declared a format.
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
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# The first and last instant a nanosecond timestamp can hold, as epoch
# seconds and the nanoseconds past them.
_MIN = (-9_223_372_037, 145_224_193)
_MAX = (9_223_372_036, 854_775_807)
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
        r'(?::(?P<S>\d{2})(?:\.(?P<f>\d+))?)?'
        r'(?: ?(?P<z>[Zz]|[+-]\d{2}:?\d{2}))?)?$')
_YEAR_SLASH = (r'^(?P<y>\d{4})/(?P<m>\d{1,2})/(?P<d>\d{1,2})'
               r'(?: (?P<H>\d{1,2}):(?P<M>\d{2})(?::(?P<S>\d{2}))?)?$')
_SLASH = (r'^(?P<a>\d{1,2})/(?P<b>\d{1,2})/(?P<y>\d{4})'
          r'(?: (?P<H>\d{1,2}):(?P<M>\d{2})(?::(?P<S>\d{2}))?)?$')
_DIGITS = r'^\s*(?P<n>\d+)(?:\.0+)?\s*$'
_ONE_SHAPE = (
    (r'^(\d{4}-\d{2}-\d{2})T', r'\1 '),
    (r'^(\d{4}-\d{2}-\d{2})$', r'\1 00:00:00'),
    (r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2})( ?(?:[Zz]|[+-]\d{2}:?\d{2}))?$',
     r'\1:00\2'),
    (r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?) ?z$', r'\1Z'),
)
_ZONE_AFTER_SPACE = r' (?:[Zz]|[+-]\d{2}:?\d{2})$'
_TEXT_SHAPES = {
    'iso': ('%Y-%m-%d', {
        1: '%H:%M%#z',
        2: '%H:%M:%S%.f%#z'
    }),
    'year_slash': ('%Y/%m/%d', {
        1: '%H:%M',
        2: '%H:%M:%S'
    }),
}
_INT64_MAX = '9223372036854775807'
_SLASH_SHAPES = {0: '', 1: ' %H:%M', 2: ' %H:%M:%S'}
_SLASH_DATE = r'^(\d{1,2}/\d{1,2}/\d{4})$'
_SLASH_ONE_SHAPE = {
    1: ((_SLASH_DATE, r'\1 00:00'), ),
    2: ((_SLASH_DATE, r'\1 00:00:00'),
        (r'^(\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2})$', r'\1:00')),
}
_YEAR_SLASH_DATE = r'^(\d{4}/\d{1,2}/\d{1,2})$'
_YEAR_SLASH_ONE_SHAPE = {
    1: ((_YEAR_SLASH_DATE, r'\1 00:00'), ),
    2: ((_YEAR_SLASH_DATE, r'\1 00:00:00'),
        (r'^(\d{4}/\d{1,2}/\d{1,2} \d{1,2}:\d{2})$', r'\1:00')),
}


@dataclass
class _Tally:
    count: int = 0
    examples: List[Any] = field(default_factory=list)

    def add(self, mask: pa.Array, values: pa.Array) -> None:
        hits = pc.sum(mask).as_py() or 0
        if hits:
            self.count += hits
            for value in _examples(pc.filter(values, mask)[:_EXAMPLES]):
                if len(self.examples) < _EXAMPLES:
                    self.examples.append(value)


def _examples(values: pa.Array) -> List[Any]:
    r"""The values as Python objects, or written out from their epoch
    seconds when Python's datetime cannot hold them (an Arrow instant past
    year 9999).
    """
    try:
        return values.to_pylist()
    except (OverflowError, ValueError):
        return [
            _civil(secs) for secs in _native_instants(values)[0].to_pylist()
        ]


def _civil(secs: int) -> str:
    r"""`secs` since the epoch as `YYYY-MM-DD HH:MM:SS` for any year
    (Howard Hinnant's civil-from-days)."""
    days, rest = divmod(secs, 86400)
    days += 719468
    era = days // 146097
    doe = days - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    day = doy - (153 * mp + 2) // 5 + 1
    month = mp + 3 if mp < 10 else mp - 9
    year = yoe + era * 400 + (month <= 2)
    return (f'{year:04d}-{month:02d}-{day:02d} {rest // 3600:02d}:'
            f'{rest % 3600 // 60:02d}:{rest % 60:02d}')


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
    other_example: Any = None
    zoned: int = 0
    bare: int = 0
    zoned_example: Any = None
    bare_example: Any = None
    reshape_parts: set = field(default_factory=set)
    unread: _Tally = field(default_factory=_Tally)
    unread_parts: set = field(default_factory=set)
    padded_parts: set = field(default_factory=set)
    text_shapes: Counter = field(default_factory=Counter)
    awkward: int = 0
    year_parts: Dict[int, set] = field(
        default_factory=lambda: {k: set()
                                 for k in _SLASH_SHAPES})
    slash_dm: _Tally = field(default_factory=_Tally)
    slash_md: _Tally = field(default_factory=_Tally)
    slash_both: _Tally = field(default_factory=_Tally)
    shapes: Dict[str, Counter] = field(
        default_factory=lambda: {k: Counter()
                                 for k in ('md', 'dm')})
    slash_parts: Dict[int, set] = field(
        default_factory=lambda: {k: set()
                                 for k in _SLASH_SHAPES})
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
    one_shape: bool = False
    slash_shape: int = 0
    year_shape: int = 0


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


def _in_range(secs: pa.Array, nanos: Any = 0) -> pa.Array:
    def past(bound: Tuple[int, int], before: bool) -> pa.Array:
        compare = pc.less if before else pc.greater
        return pc.or_(
            compare(secs, bound[0]),
            pc.and_(pc.equal(secs, bound[0]), compare(nanos, bound[1])))

    return pc.invert(pc.or_(past(_MIN, True), past(_MAX, False)))


def _split(values: pa.Array, ticks: int) -> Tuple[pa.Array, pa.Array]:
    r"""Epoch seconds and the nanoseconds past them from a count of
    `ticks` per second, flooring so an instant before 1970 keeps a
    positive fraction.
    """
    secs = pc.divide(values, ticks)
    rest = pc.subtract(values, pc.multiply(secs, ticks))
    borrow = pc.cast(pc.less(rest, 0), pa.int64())
    secs = pc.subtract(secs, borrow)
    rest = pc.add(rest, pc.multiply(borrow, ticks))
    return secs, pc.multiply(rest, 10**9 // ticks)


def _fraction_nanos(parts: pa.Array) -> pa.Array:
    digits = pc.utf8_slice_codeunits(
        pc.utf8_rpad(pc.fill_null(pc.struct_field(parts, 'f'), ''), 9, '0'), 0,
        9)
    return pc.cast(digits, pa.int64())


def _offset_seconds(zone: pa.Array) -> pa.Array:
    utc = pc.or_(pc.equal(zone, ''), pc.equal(pc.utf8_upper(zone), 'Z'))
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


def _int_readings(values: pa.Array) -> Tuple[pa.Array, pa.Array, pa.Array]:
    r"""Each whole number's one reading, with the epoch seconds and the
    nanoseconds past them under it.
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
    nanos = pc.case_when(pc.make_struct(millis, micros, nanos),
                         _split(values, 10**3)[1],
                         _split(values, 10**6)[1],
                         _split(values, 10**9)[1], pa.scalar(0, pa.int64()))
    return pc.if_else(pc.is_null(values), None, reading), secs, nanos


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


def _native_instants(values: pa.Array) -> Optional[Tuple[pa.Array, pa.Array]]:
    kind = values.type
    if pa.types.is_date32(kind):
        days = values.cast(pa.int32()).cast(pa.int64())
        return pc.multiply(days, 86400), pc.multiply(days, 0)
    if pa.types.is_date64(kind):
        return _split(values.cast(pa.int64()), 10**3)
    if pa.types.is_timestamp(kind):
        ticks = {'s': 1, 'ms': 10**3, 'us': 10**6, 'ns': 10**9}[kind.unit]
        return _split(values.cast(pa.int64()), ticks)
    return None


def _text_numbers(text: pa.Array, digits: pa.Array) -> pa.Array:
    r"""The whole numbers spelled in text, as `int64`: the spellings
    DiskGraph's integer parser accepts (padding, leading zeros, a zero
    fraction), null where the significant digits are too wide for `int64`.
    """
    plain = pc.utf8_ltrim(pc.struct_field(pc.extract_regex(text, _DIGITS), 'n'),
                          '0')
    plain = pc.if_else(pc.equal(plain, ''), '0', plain)
    length = pc.utf8_length(plain)
    fits = pc.and_(
        digits,
        pc.or_(
            pc.less(length, len(_INT64_MAX)),
            pc.and_(pc.equal(length, len(_INT64_MAX)),
                    pc.less_equal(plain, _INT64_MAX))))
    return pc.cast(pc.if_else(fits, plain, None), pa.int64())


def _spelled(numbers: pa.Array, kind: pa.DataType) -> pa.Array:
    r"""The whole numbers written out in plain digits, the one spelling
    DiskGraph's format parser reads; null where there is no number.
    """
    return pc.cast(numbers, kind)


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
        instants = _native_instants(values)
        if instants is not None:
            self.scan.families['native'] += len(present)
            self._tally('fixed', part, values, pc.invert(_in_range(*instants)),
                        None)
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
        reading, secs, nanos = _int_readings(numbers)
        reading = pc.if_else(pc.and_(present, pc.is_null(numbers)), 'none',
                             reading)
        for code, count in zip(*pc.value_counts(reading).flatten()):
            code = code.as_py()
            if code is None:
                continue
            self.scan.readings[code] += count.as_py()
            if code not in self.scan.reading_examples:
                matches = pc.equal(reading, code)
                self.scan.reading_examples[code] = (pc.filter(
                    values, matches)[0].as_py(), pc.filter(numbers,
                                                           matches)[0].as_py())
        self._tally('fixed', part, values, pc.invert(_in_range(secs, nanos)),
                    None)
        unread = pc.fill_null(pc.equal(reading, 'none'), False)
        if pc.sum(unread).as_py():
            self.scan.unread_parts.add(part)
            self.scan.unread.add(unread, values)

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
        if self.scan.other_example is None and pc.sum(other).as_py():
            self.scan.other_example = pc.filter(text, other)[0].as_py()

        if pc.sum(families['digits']).as_py():
            numbers = _text_numbers(text, families['digits'])
            self._observe_numbers(part, text, families['digits'], numbers)
            if pc.any(pc.not_equal(_spelled(numbers, text.type), text)).as_py():
                self.scan.padded_parts.add(part)

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
            nanos = pa.scalar(0, pa.int64())
            depth = pc.add(
                pc.cast(pc.is_valid(_number(parts, 'H')), pa.int64()),
                pc.cast(pc.is_valid(_number(parts, 'S')), pa.int64()))
            sep = pc.if_else(pc.greater(depth, 0), ' ', '')
            if kind == 'iso':
                sep = pc.fill_null(pc.struct_field(parts, 'sep'), '')
                zone = pc.fill_null(pc.struct_field(parts, 'z'), '')
                secs = pc.subtract(secs, _offset_seconds(zone))
                nanos = _fraction_nanos(parts)
                dated = pc.and_(matched, valid)
                self._note_zones(text, pc.and_(dated, pc.not_equal(zone, '')),
                                 pc.and_(dated, pc.equal(zone, '')))
                awkward = pc.or_(
                    pc.equal(zone, 'z'),
                    pc.fill_null(
                        pc.match_substring_regex(text, _ZONE_AFTER_SPACE),
                        False))
                self.scan.awkward += pc.sum(pc.and_(matched,
                                                    awkward)).as_py() or 0
                if pc.any(
                        pc.and_(
                            matched,
                            pc.or_(pc.equal(sep, 'T'),
                                   pc.is_null(_number(parts, 'S'))))).as_py():
                    self.scan.reshape_parts.add(part)
            shape = pc.binary_join_element_wise(sep,
                                                pc.cast(depth, pa.string()),
                                                '|')
            for value, count in zip(
                    *pc.value_counts(pc.filter(shape, matched)).flatten()):
                self.scan.text_shapes[value.as_py()] += count.as_py()
            if kind == 'year_slash':
                for value in pc.unique(pc.filter(depth, matched)).to_pylist():
                    self.scan.year_parts[value].add(part)
            self._tally('fixed', part, text, pc.and_(matched, pc.invert(valid)),
                        'impossible')
            self._tally(
                'fixed', part, text,
                pc.and_(pc.and_(matched, valid),
                        pc.invert(_in_range(secs, nanos))), 'far')

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
        for value in pc.unique(pc.filter(shape, matched)).to_pylist():
            self.scan.slash_parts[value].add(part)
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
        for order, valid in (('md', valid_md), ('dm', valid_dm)):
            for value, count in zip(
                    *pc.value_counts(pc.filter(shape, valid)).flatten()):
                self.scan.shapes[order][value.as_py()] += count.as_py()
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
        other = scan.families.get('other', 0)
        decide = None
        if set(known) == {'digits'}:
            decide = self._decide_numbers
        elif set(known) == {'slash'}:
            decide = self._decide_slash
        elif set(known) == {'iso'} and scan.zoned and scan.bare:
            decide = self._decide_zones
        elif set(known) in ({'iso'}, {'year_slash'}) and (
                scan.impossible['fixed'].count or scan.far['fixed'].count):
            decide = self._decide_shape
        if decide is not None and other:
            warnings.warn(
                f"Time column {self.alias}.{self.name} holds {other} "
                f"value{'s' if other != 1 else ''} in a shape the scan does "
                f"not read (e.g. {scan.other_example}), so its format is "
                "left to DiskGraph's own reading. Set timestamp_format on "
                "the column to decide it.",
                UserWarning,
                stacklevel=5,
            )
        elif decide is not None:
            decide()
        self._report(self._settled('impossible'),
                     'are not dates on the calendar',
                     'is not a date on the calendar',
                     'Correct them at the source to {keep}')
        self._report(
            self._settled('far'), f'fall outside {_RANGE}',
            f'falls outside {_RANGE}',
            'Replace placeholder dates with nulls at the source to '
            '{keep}')

    def _decide_shape(self) -> None:
        r"""A column of ISO or year-first values the scan made values missing
        in is handed to DiskGraph with its format declared, since DiskGraph
        otherwise decides the format from a part's first rows and refuses a
        column whose sample the scan has made all null: the exact format of
        the one shape every value has, or the one-shape staging when shapes
        differ (ISO values in a space-separated full time part, year-first
        slashes in the fullest time part any value has).
        """
        scan = self.scan
        family = 'iso' if scan.families.get('iso') else 'year_slash'
        base, times = _TEXT_SHAPES[family]
        if len(scan.text_shapes) == 1 and not scan.awkward:
            sep, depth = next(iter(scan.text_shapes)).split('|')
            time = times[int(depth)] if int(depth) else ''
            self.decision = _Decision(base + (sep + time if time else ''))
        elif family == 'iso':
            self.decision = _Decision('%Y-%m-%d %H:%M:%S%.f%#z', one_shape=True)
        else:
            depth = max(int(key.split('|')[1]) for key in scan.text_shapes)
            self.decision = _Decision(base + ' ' + times[depth],
                                      year_shape=depth)

    def _decide_zones(self) -> None:
        scan = self.scan
        self.decision = _Decision('%Y-%m-%d %H:%M:%S%.f%#z', one_shape=True)
        warnings.warn(
            f"Time column {self.alias}.{self.name} mixes {scan.zoned} "
            f"value{'s' if scan.zoned != 1 else ''} with a timezone, "
            f"converted to UTC, and {scan.bare} without one, read as "
            f"UTC (e.g. {scan.zoned_example} and {scan.bare_example}). "
            "Declare one timezone at the source so the events line up.",
            UserWarning,
            stacklevel=5,
        )

    def _decide_numbers(self) -> None:
        scan = self.scan
        readings = [code for code in _READINGS if scan.readings.get(code)]
        settled = [code for code in readings if code != 'none']
        if len(settled) != 1:
            parts = ', '.join(f"{scan.readings[code]} "
                              f"{'is' if scan.readings[code] == 1 else 'are'} "
                              f"{_READINGS[code][2]} (e.g. "
                              f"{scan.reading_examples[code][0]})"
                              for code in readings)
            raise ValueError(
                f"Time column {self.alias}.{self.name} holds whole numbers "
                f"with no single reading: {parts}. Set timestamp_format on "
                f"the column to 's', 'ms', 'us', 'ns', '%Y%m%d' or "
                f"'%Y%m%d%H%M%S', or convert it to a timestamp at the "
                "source.")
        code = settled[0]
        hint, description, singular = _READINGS[code]
        example, number = scan.reading_examples[code]
        rendered = _render(
            _int_readings(pa.array([number], pa.int64()))[1][0].as_py())
        self.decision = _Decision(hint, reading=code)
        read = scan.readings[code]
        warnings.warn(
            f"Time column {self.alias}.{self.name} holds whole numbers; its "
            f"{read} value{'s' if read != 1 else ''} "
            f"{'are' if read != 1 else 'is'} read as {description} ({example} "
            f"is {rendered}). Set timestamp_format on the column to read them "
            "differently.",
            UserWarning,
            stacklevel=5,
        )
        if code in ('ymd', 'ymdhms'):
            action = (f"Correct {'them' if scan.unread.count != 1 else 'it'} "
                      "at the source to {keep}")
        else:
            action = ("Replace placeholders with nulls at the source to "
                      "{keep}, or set timestamp_format on the column to "
                      f"'{hint}' to read every value as {description}")
        self._report(scan.unread, f'are not read as {description}',
                     f'is not read as {singular}', action)

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
            if not scan.slash_both.count:
                self.decision = _Decision(order='md')
                return
            example = scan.slash_both.examples[0]
            raise ValueError(
                f"Time column {self.alias}.{self.name} holds slash dates "
                "whose day and month order the data does not settle: every "
                f"one of its {scan.values} values (e.g. {example}) reads "
                "both ways. Set timestamp_format on the column to "
                "'%d/%m/%Y' or '%m/%d/%Y'.")
        order, settled = ('dm', dm) if dm.count else ('md', md)
        shape = max(scan.shapes[order])
        base = '%d/%m/%Y' if order == 'dm' else '%m/%d/%Y'
        fmt = base + _SLASH_SHAPES[shape]
        self.decision = _Decision(fmt, order=order, slash_shape=shape)
        warnings.warn(
            f"Time column {self.alias}.{self.name} holds slash dates read "
            f"{'day' if order == 'dm' else 'month'} first ({fmt}) because "
            f"{settled.count} of {scan.values} values (e.g. "
            f"{settled.examples[0]}) read only that way. Set "
            "timestamp_format on the column to override.",
            UserWarning,
            stacklevel=5,
        )

    def _settled(self, kind: str) -> _Tally:
        tallies = getattr(self.scan, kind)
        tally = _Tally()
        for order in dict.fromkeys(('fixed', self.decision.order)):
            if tallies[order].count:
                tally.count += tallies[order].count
                tally.examples.extend(tallies[order].examples)
        return tally

    def _report(self, tally: _Tally, plural: str, singular: str,
                action: str) -> None:
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
            f"{examples}); {outcome}. {action.format(keep=keep)}.",
            UserWarning,
            stacklevel=5,
        )

    def parts_to_rewrite(self) -> set:
        parts = set(self.scan.invalid_parts['fixed'])
        if self.decision.order != 'fixed':
            parts |= self.scan.invalid_parts[self.decision.order]
        if self.decision.one_shape:
            parts |= self.scan.reshape_parts
        if self.decision.reading:
            parts |= self.scan.unread_parts | self.scan.padded_parts
        for shape, held in self.scan.slash_parts.items():
            if shape < self.decision.slash_shape:
                parts |= held
        for shape, held in self.scan.year_parts.items():
            if shape < self.decision.year_shape:
                parts |= held
        return parts

    def rewritten(self, values: pa.Array) -> pa.Array:
        r"""The column as the decision stages it: invalid values null; ISO
        values in one shape, a space before a full time part, when the
        column mixes bare and zoned ones, so one chrono format reads them
        all (a date alone is its midnight); slash dates in the fullest time
        shape any value has, for the same reason; and digit strings as the
        plain whole number they spell when the column is read as numbers,
        since the declared format reads no padding or zero fraction.
        """
        invalid = self.invalid(values)
        reshapes = list(_ONE_SHAPE) if self.decision.one_shape else []
        reshapes += _SLASH_ONE_SHAPE.get(self.decision.slash_shape, ())
        reshapes += _YEAR_SLASH_ONE_SHAPE.get(self.decision.year_shape, ())
        for pattern, replacement in reshapes:
            values = pc.replace_substring_regex(values, pattern, replacement)
        if self.decision.reading and _strings(values) is not None:
            digits = pc.fill_null(pc.match_substring_regex(values, _DIGITS),
                                  False)
            spelled = _spelled(_text_numbers(values, digits), values.type)
            values = pc.if_else(pc.is_valid(spelled), spelled, values)
        return pc.if_else(invalid, None, values)

    def invalid(self, values: pa.Array) -> pa.Array:
        r"""The values made missing. Native instants and ISO or year-first
        values no calendar holds or outside the range are missing in every
        column, as are whole numbers outside the range under their own
        digit-count reading, even when the column is left to DiskGraph's
        own reading because it also holds a shape the scan does not read.
        Whole numbers with no reading are missing only once the column
        settled on one, and slash dates only under a settled order.
        """
        instants = _native_instants(values)
        if instants is not None:
            return pc.fill_null(pc.invert(_in_range(*instants)), False)
        numbers = _as_int64(values)
        if numbers is not None:
            return self._invalid_numbers(pc.is_valid(values), numbers)
        text = _strings(values)
        if text is None:
            return pa.array([False] * len(values), pa.bool_())
        digits = pc.fill_null(pc.match_substring_regex(text, _DIGITS), False)
        invalid = self._invalid_numbers(digits, _text_numbers(text, digits))
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
            nanos = pa.scalar(0, pa.int64())
            if kind == 'iso':
                secs = pc.subtract(
                    secs,
                    _offset_seconds(
                        pc.fill_null(pc.struct_field(parts, 'z'), '')))
                nanos = _fraction_nanos(parts)
            bad = pc.or_(pc.invert(valid), pc.invert(_in_range(secs, nanos)))
            invalid = pc.or_(invalid, pc.and_(matched, pc.fill_null(bad,
                                                                    False)))
        return pc.fill_null(invalid, False)

    def _invalid_numbers(self, present: pa.Array,
                         numbers: pa.Array) -> pa.Array:
        r"""Whole numbers outside the range under their own reading and,
        once the column settled on one reading, those with none.
        """
        secs, nanos = _int_readings(numbers)[1:]
        invalid = pc.fill_null(pc.invert(_in_range(secs, nanos)), False)
        if self.decision.reading:
            invalid = pc.or_(invalid, pc.and_(present, pc.is_null(secs)))
        return invalid


def _render(secs: Optional[int]) -> str:
    if secs is None or not _MIN[0] < secs <= _MAX[0]:
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
    r"""Rewrites one staged part with the given columns replaced by what
    their callables return, keeping every other byte of the schema and
    data. The part is read as staging reads one, in batches bounded by
    decoded bytes, so a wide table is never decoded whole. The part is
    replaced atomically so a failure leaves the original in place.
    """
    from kumo_core.sources.object_store import _parquet_batches

    parquet = pq.ParquetFile(file)
    schema = parquet.schema_arrow
    fragment = next(
        iter(ds.dataset(str(file), format='parquet').get_fragments()))
    batches = _parquet_batches(fragment, parquet.schema_arrow)
    for name in columns:
        index = schema.get_field_index(name)
        field_ = schema.field(index).with_nullable(True)
        if pa.types.is_dictionary(field_.type):
            field_ = field_.with_type(field_.type.value_type)
        schema = schema.set(index, field_)
    temporary = file.with_name(f'.{file.name}.rewrite')
    with pq.ParquetWriter(temporary, schema) as writer:
        for batch in batches:
            arrays = []
            for field_ in schema:
                array = batch.column(field_.name)
                if field_.name in columns:
                    if pa.types.is_dictionary(array.type):
                        array = array.cast(field_.type)
                    array = columns[field_.name](array)
                arrays.append(array)
            writer.write_batch(pa.RecordBatch.from_arrays(
                arrays, schema=schema))
    os.replace(temporary, file)


def resolve_time_columns(
        alias: str, definition: Any,
        paths: Sequence[str]) -> Tuple[Dict[str, str], List[str]]:
    r"""Settles the time and end-time columns of one staged table from every
    value and returns the `timestamp_format` each one needs, by column
    name, with the sorted names of the columns whose staged parts were
    rewritten; a column DiskGraph already reads unambiguously, or whose
    format the table declares, gets none. Values no calendar or clock
    holds, or with no reading once a column of whole numbers has one, are
    nulled in the staged parts; a column mixing bare and zoned ISO values,
    slash dates in more than one time shape and digit strings with padding
    or a zero fraction are staged in the one shape the declared format
    reads. Raises when a column of whole numbers has no single reading or
    a column of slash dates no single order.
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
        return {}, []
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
    rewritten = set()
    parts = {column.name: column.parts_to_rewrite() for column in columns}
    for file in files:
        touched = {
            column.name: column.rewritten
            for column in columns if str(file) in parts[column.name]
        }
        if touched:
            _rewrite(file, touched)
            rewritten.update(touched)
    return {
        column.name: column.decision.timestamp_format
        for column in columns if column.decision.timestamp_format is not None
    }, sorted(rewritten)
