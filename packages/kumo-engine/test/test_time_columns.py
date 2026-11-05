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
    (pa.array([
        '1677-09-21 00:12:43.145224193', '2262-04-11 23:47:16.854775807',
        '1677-09-21 00:12:43.145224192', '2262-04-11 23:47:16.854775808'
    ]), None, [2, 3]),
    (pa.array([
        datetime(2262, 4, 11, 23, 47, 16, 854775),
        datetime(2262, 4, 11, 23, 47, 16, 854776)
    ], pa.timestamp('us')), None, [1]),
    (pa.array([1705276800854, 9223372036854, 9223372036855],
              pa.int64()), 'ms', [2]),
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
        '2024-01-15 23:60:00', '2024-01-15 23:59:60', '2024-01-15 23:59:59Z'
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


def test_slash_shapes_are_voted_among_the_values_that_are_dates(tmp_path):
    path = _part(tmp_path,
                 pa.array(['13/01/2024 10:30:00', '30/02/2024', '31/02/2024']))
    with pytest.warns(UserWarning) as caught:
        formats = resolve_time_columns('events', _definition(), [str(path)])
    assert formats == {'ts': '%d/%m/%Y %H:%M:%S'}
    assert [str(w.message)[:60] for w in caught] == [
        'Time column events.ts holds slash dates read day first (%d/%',
        '2 values in time column events.ts are not dates on the calen',
    ]
    assert pq.read_table(path).column('ts').to_pylist() == [
        '13/01/2024 10:30:00', None, None
    ]


def test_mixed_timezones_are_staged_with_one_separator(tmp_path):
    path = _part(
        tmp_path,
        pa.array([
            '2024-01-16T10:00:00', '2024-01-15 10:00:00+02:00',
            '2024-01-17T10:00:00.250Z', 'not a time', None
        ]))
    with pytest.warns(UserWarning, match='mixes 2 values with a timezone'):
        formats = resolve_time_columns('events', _definition(), [str(path)])
    assert formats == {'ts': '%Y-%m-%d %H:%M:%S%.f%#z'}
    assert pq.read_table(path).column('ts').to_pylist() == [
        '2024-01-16 10:00:00', '2024-01-15 10:00:00+02:00',
        '2024-01-17 10:00:00.250Z', 'not a time', None
    ]


def test_mixed_timezones_are_staged_with_a_full_time_part(tmp_path):
    path = _part(
        tmp_path,
        pa.array([
            '2024-01-15', '2024-01-16T10:00', '2024-01-17 10:00+02:00',
            '2024-01-18 10:00:00.5Z'
        ]))
    with pytest.warns(UserWarning, match='mixes 2 values with a timezone'):
        formats = resolve_time_columns('events', _definition(), [str(path)])
    assert formats == {'ts': '%Y-%m-%d %H:%M:%S%.f%#z'}
    assert pq.read_table(path).column('ts').to_pylist() == [
        '2024-01-15 00:00:00', '2024-01-16 10:00:00',
        '2024-01-17 10:00:00+02:00', '2024-01-18 10:00:00.5Z'
    ]


def test_far_numbers_among_other_shapes_are_still_made_missing(tmp_path):
    path = _part(tmp_path, pa.array(['99991231', '2024-01-16']))
    with pytest.warns(UserWarning, match='1 value in time column events.ts '
                      'falls outside'):
        formats = resolve_time_columns('events', _definition(), [str(path)])
    assert formats == {}
    assert pq.read_table(path).column('ts').to_pylist() == [None, '2024-01-16']


def test_a_native_instant_past_year_9999_is_made_missing(tmp_path):
    path = _part(tmp_path,
                 pa.array([1705276800, 253402300800], pa.timestamp('s')))
    with pytest.warns(
            UserWarning, match=re.escape(
                '1 value in time column events.ts falls outside 1677-09-21 to '
                '2262-04-11 (e.g. 10000-01-01 00:00:00)')):
        resolve_time_columns('events', _definition(), [str(path)])
    assert pq.read_table(path).column('ts').to_pylist() == [
        datetime(2024, 1, 15), None
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


def test_a_non_nullable_staged_column_still_takes_the_nulls(tmp_path):
    path = tmp_path / 'part-0.parquet'
    schema = pa.schema([
        pa.field('id', pa.int64(), nullable=False),
        pa.field('ts', pa.string(), nullable=False),
    ])
    pq.write_table(
        pa.table({
            'id': [0, 1],
            'ts': ['2024-02-30', '2024-01-17']
        }, schema=schema), path)
    with pytest.warns(UserWarning, match='1 value in time column'):
        resolve_time_columns('events', _definition(), [str(path)])
    stored = pq.read_table(path)
    assert stored.column('ts').to_pylist() == [None, '2024-01-17']
    assert not stored.schema.field('id').nullable


@pytest.mark.parametrize('values, outcome', [
    (pa.array([' 1705276800 ', '1705363200']), {
        'ts': 's'
    }),
    (pa.array([' 1705276800 ', '1705276800000']), ValueError),
    (pa.array([' 20240115', '20240116 ']), {
        'ts': '%Y%m%d'
    }),
])
def test_padded_digit_strings_count_as_numbers(tmp_path, values, outcome):
    path = _part(tmp_path, values)
    if outcome is ValueError:
        with pytest.raises(ValueError, match='no single reading'):
            resolve_time_columns('events', _definition(), [str(path)])
        return
    with pytest.warns(UserWarning, match='holds whole numbers'):
        assert resolve_time_columns('events', _definition(),
                                    [str(path)]) == outcome


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
