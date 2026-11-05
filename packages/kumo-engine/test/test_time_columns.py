# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from datetime import date, datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumoapi.table import Column
from kumoapi.typing import Dtype, Stype

from kumo.engine._time_columns import resolve_time_columns


def _definition(dtype=Dtype.string, *, end=False, declared=None):
    return SimpleNamespace(
        cols=[
            Column(name='id', dtype=Dtype.int64, stype=Stype.ID),
            Column(name='ts', dtype=dtype, stype=Stype.timestamp,
                   timestamp_format=declared),
        ],
        time_col=None if end else 'ts',
        end_time_col='ts' if end else None,
    )


def _part(tmp_path, values, name='part-0.parquet'):
    path = tmp_path / name
    pq.write_table(
        pa.table({
            'id': pa.array(range(len(values)), pa.int64()),
            'ts': values
        }), path)
    return path


def _ts(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


@pytest.mark.parametrize('values', [
    pa.array([20240115, 20240116], pa.int32()),
    pa.array([20240115, 20240116], pa.uint32()),
    pa.array([20240115, 20240116], pa.int64()),
    pa.array([Decimal(20240115), Decimal(20240116)], pa.decimal128(18, 0)),
    pa.array(['20240115', '20240116'], pa.string()),
])
def test_every_integer_shape_shares_the_reading(tmp_path, values):
    path = _part(tmp_path, values)
    before = path.read_bytes()
    with pytest.warns(UserWarning, match='read as YYYYMMDD dates'):
        formats = resolve_time_columns('events', _definition(Dtype.int64),
                                       [str(path)])
    assert formats == {'ts': '%Y%m%d'}
    assert path.read_bytes() == before


@pytest.mark.parametrize('values, fmt, missing', [
    (pa.array([1705276800, 9999999999], pa.int64()), 's', [1]),
    (pa.array([20240115, 99991231], pa.int64()), '%Y%m%d', [1]),
    (pa.array([date(2024, 1, 15), date(9999, 12, 31)], pa.date32()), None, [1]),
    (pa.array([datetime(2024, 1, 15),
               datetime(9999, 12, 31)], pa.timestamp('us')), None, [1]),
    (pa.array(
        [datetime(2024, 1, 15), datetime(2262, 4, 12)],
        pa.timestamp('ms', tz='America/New_York')), None, [1]),
    (pa.array([
        '1677-09-21 00:12:44', '2262-04-11 23:47:16', '1677-09-21 00:12:43',
        '2262-04-11 23:47:17'
    ]), None, [2, 3]),
])
def test_values_outside_the_nanosecond_range_are_made_missing(
        tmp_path, values, fmt, missing):
    path = _part(tmp_path, values)
    with pytest.warns(UserWarning) as caught:
        formats = resolve_time_columns('events', _definition(), [str(path)])
    assert formats == ({'ts': fmt} if fmt else {})
    message = [
        str(w.message) for w in caught
        if 'falls outside' in str(w.message) or 'fall outside' in str(w.message)
    ]
    assert len(message) == 1
    assert '1677-09-21 to 2262-04-11' in message[0]
    stored = pq.read_table(path).column('ts')
    assert [i for i, v in enumerate(stored.to_pylist()) if v is None] == missing
    assert stored.type == values.type


@pytest.mark.parametrize('values, missing', [
    (pa.array([
        '2024-02-29', '2023-02-29', '2024-04-31', '2024-00-10', '2024-13-01',
        '2024-01-00', '2024-06-30', '2024-06-31'
    ]), [1, 2, 3, 4, 5, 7]),
    (pa.array([
        '2024-01-15 23:59:59.999999', '2024-01-15 24:00:00',
        '2024-01-15 23:60:00', '2024-01-15 23:59:60', '2024-01-15T23:59:59Z'
    ]), [1, 2, 3]),
    (pa.array(['1990/02/28', '1990/02/30', '1990/13/01']), [1, 2]),
    (pa.array(['28/02/1990', '30/02/1990', '13/01/1990']), [1]),
])
def test_dates_not_on_the_calendar_are_made_missing(tmp_path, values, missing):
    path = _part(tmp_path, values)
    with pytest.warns(
            UserWarning, match=(
                f'{len(missing)} value(s)? in time column events.ts (is|are) '
                'not (a date|dates) on the calendar')):
        resolve_time_columns('events', _definition(), [str(path)])
    stored = pq.read_table(path).column('ts').to_pylist()
    assert [i for i, v in enumerate(stored) if v is None] == missing
    assert [v for v in stored if v is not None] == [
        v for i, v in enumerate(values.to_pylist()) if i not in missing
    ]


def test_an_end_time_column_is_made_missing_without_dropping_rows(tmp_path):
    path = _part(tmp_path, pa.array(['2024-01-15', '9999-12-31']))
    with pytest.warns(UserWarning) as caught:
        resolve_time_columns('events', _definition(end=True), [str(path)])
    assert [str(w.message) for w in caught] == [
        '1 value in end time column events.ts falls outside 1677-09-21 to '
        '2262-04-11 (e.g. 9999-12-31); it is treated as missing. Replace '
        'placeholder dates with nulls at the source to keep the value.'
    ]
    assert pq.read_table(path).column('ts').to_pylist() == ['2024-01-15', None]


@pytest.mark.parametrize('values, example', [
    (pa.array([2**63, 20240115], pa.uint64()), 9223372036854775808),
    (pa.array([Decimal(10**19), Decimal(20240115)], pa.decimal128(
        38, 0)), Decimal(10**19)),
    (pa.array(['99999999999999999999', '20240115']), '99999999999999999999'),
    (pa.array(['9223372036854775808', '1705276800']), '9223372036854775808'),
])
def test_numbers_too_wide_for_int64_have_no_reading(tmp_path, values, example):
    path = _part(tmp_path, values)
    with pytest.raises(
            ValueError, match=re.escape(
                f'1 is too small or too large to be a date (e.g. {example})')):
        resolve_time_columns('events', _definition(), [str(path)])


def test_a_declared_format_is_left_alone(tmp_path):
    path = _part(tmp_path, pa.array(['05/03/1990', '30/02/1990']))
    before = path.read_bytes()
    formats = resolve_time_columns('events', _definition(declared='%d/%m/%Y'),
                                   [str(path)])
    assert formats == {}
    assert path.read_bytes() == before


def test_only_parts_holding_an_invalid_value_are_rewritten(tmp_path):
    clean = _part(tmp_path, pa.array(['2024-01-15', '2024-01-16']),
                  'part-0.parquet')
    dirty = _part(tmp_path, pa.array(['2024-02-30', '2024-01-17']),
                  'part-1.parquet')
    before = clean.read_bytes()
    with pytest.warns(UserWarning, match='1 value in time column'):
        resolve_time_columns('events', _definition(), [str(clean), str(dirty)])
    assert clean.read_bytes() == before
    assert pq.read_table(dirty).to_pydict() == {
        'id': [0, 1],
        'ts': [None, '2024-01-17']
    }
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        'part-0.parquet', 'part-1.parquet'
    ]


@pytest.mark.parametrize('values', [
    pa.array(['15 Jan 2024', '16 Jan 2024']),
    pa.array([1705276800.0, 1705363200.0], pa.float64()),
    pa.array(['2024-01-15', '15/01/2024']),
    pa.array([None, None], pa.string()),
])
def test_other_shapes_are_left_to_diskgraph(tmp_path, values):
    path = _part(tmp_path, values)
    before = path.read_bytes()
    formats = resolve_time_columns('events', _definition(), [str(path)])
    assert formats == {}
    assert path.read_bytes() == before
