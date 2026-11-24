# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import random
import struct
from datetime import date, datetime

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from kumo.fm.native_json import dumps, encode_rows


def reference(table):
    def cell(value):
        if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 2**53 - 1:
            return str(value)
        if isinstance(value, datetime):
            return (f'{value.year:04d}-{value.month:02d}-{value.day:02d}T'
                    f'{value.hour:02d}:{value.minute:02d}:{value.second:02d}.{value.microsecond:06d}Z')
        if isinstance(value, date):
            return f'{value.year:04d}-{value.month:02d}-{value.day:02d}'
        return value
    rows = [[cell(value) for value in row] for row in zip(*(c.to_pylist() for c in table.columns))]
    return json.dumps(rows, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')


def test_scalar_bytes_chunks_slices_and_budget():
    table = pa.table({
        'int': pa.array([-(2**63), -(2**53), -(2**53 - 1), 2**53 - 1, 2**53, 2**63 - 1, None]),
        'float': [0.0, -0.0, 1e-5, 1e-4, 1e15, 1e16, None],
        'bool': [True, False, None, True, False, True, None],
        'string': ['"\\\b\f\n\r\t', '\x00\x1f\x7f', 'é中文😀', '', 'ascii', None, '𐀀'],
        'date': pa.array([-719162, -1, 0, 1, 2932896, None, 20000], pa.date32()),
        'timestamp': pa.array([-62135596800000000, -1000001, -1, 0, 253402300799999999, None, 1],
                              pa.timestamp('us', tz='UTC')),
    })
    for candidate in (table, table.slice(1, 5), pa.concat_tables([table.slice(0, 2), table.slice(2)])):
        expected = reference(candidate)
        assert encode_rows(candidate, len(expected)).data == expected
        with pytest.raises(ValueError, match='byte budget'):
            encode_rows(candidate, len(expected) - 1)
        payload = {'é': encode_rows(candidate, len(expected))}
        encoded = dumps(payload, len(expected) + 20)
        assert json.loads(encoded)['é'] == json.loads(expected)
        with pytest.raises(ValueError, match='byte budget'):
            dumps(payload, len(encoded) - 1)


def test_float_spelling_matches_python():
    rng = random.Random(391)
    values = [struct.unpack('d', rng.randbytes(8))[0] for _ in range(10000)]
    values += [5e-324, -5e-324, 1.7976931348623157e308, 2.2250738585072014e-308,
               -1420520845158808.2, 1580236626536537.2, 241023029448431.62]
    table = pa.table({'value': values})
    table = table.filter(pc.is_finite(table['value']))
    expected = reference(table)
    assert encode_rows(table, len(expected)).data == expected


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_is_rejected(value):
    with pytest.raises(ValueError, match='not JSON compliant'):
        encode_rows(pa.table({'value': [value]}), 100)


def test_empty_and_invalid_inputs():
    assert encode_rows(pa.table({}), 2).data == b'[]'
    assert encode_rows(pa.table({'value': pa.array([], pa.int64())}), 2).data == b'[]'
    assert encode_rows(pa.table({'value': pa.array([None, None], pa.int64())}), 15).data == b'[[null],[null]]'
    with pytest.raises(ValueError, match='Negative payload'):
        encode_rows(pa.table({}), -1)
    with pytest.raises(ValueError, match='Unsupported native arrays'):
        encode_rows(pa.table({'value': pa.array([1], pa.uint64())}), 100)


def test_out_of_python_calendar_range_keeps_arrow_formatting():
    table = pa.table({
        'date': pa.array([-719163, 2932897], pa.date32()),
        'timestamp': pa.array([-62135596800000001, 253402300800000000], pa.timestamp('us', tz='UTC')),
    })
    dates = pc.strftime(table['date'], format='%Y-%m-%d').to_pylist()
    timestamps = pc.strftime(table['timestamp'], format='%Y-%m-%dT%H:%M:%S').to_pylist()
    expected = [[d, t + 'Z'] for d, t in zip(dates, timestamps)]
    assert json.loads(encode_rows(table, 1000).data) == expected
