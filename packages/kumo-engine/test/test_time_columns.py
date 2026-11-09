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
        formats, rewritten = resolve_time_columns('events',
                                                  _definition(Dtype.int64),
                                                  [str(path)])
    assert (formats, rewritten) == ({'ts': '%Y%m%d'}, [])
    assert path.read_bytes() == before


@pytest.mark.parametrize('values, fmt, missing', [
    (pa.array([1705276800, 9999999999], pa.int64()), 's', [1]),
    (pa.array([20240115, 99991231], pa.int64()), '%Y%m%d', [1]),
    (pa.array([date(2024, 1, 15), date(9999, 12, 31)],
              pa.date32()), '%Y-%m-%d', [1]),
    (pa.array([datetime(2024, 1, 15),
               datetime(9999, 12, 31)], pa.timestamp('us')), 'us', [1]),
    (pa.array(
        [datetime(2024, 1, 15), datetime(2262, 4, 12)],
        pa.timestamp('ms', tz='America/New_York')), 'ms', [1]),
    (pa.array([
        '1677-09-21 00:12:44', '2262-04-11 23:47:16', '1677-09-21 00:12:43',
        '2262-04-11 23:47:17'
    ]), '%Y-%m-%d %H:%M:%S%.f%#z', [2, 3]),
    (pa.array([
        '1677-09-21 00:12:43.145224193', '2262-04-11 23:47:16.854775807',
        '1677-09-21 00:12:43.145224192', '2262-04-11 23:47:16.854775808'
    ]), '%Y-%m-%d %H:%M:%S%.f%#z', [2, 3]),
    (pa.array([
        datetime(2262, 4, 11, 23, 47, 16, 854775),
        datetime(2262, 4, 11, 23, 47, 16, 854776)
    ], pa.timestamp('us')), 'us', [1]),
    (pa.array([1705276800854, 9223372036854, 9223372036855],
              pa.int64()), 'ms', [2]),
])
def test_values_outside_the_nanosecond_range_are_made_missing(
        tmp_path, values, fmt, missing):
    path = _part(tmp_path, values)
    with pytest.warns(UserWarning) as caught:
        formats, rewritten = resolve_time_columns('events', _definition(),
                                                  [str(path)])
    assert (formats, rewritten) == (({'ts': fmt} if fmt else {}), ['ts'])
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


def test_slash_shapes_are_taken_from_the_values_that_are_dates(tmp_path):
    path = _part(tmp_path,
                 pa.array(['13/01/2024 10:30:00', '30/02/2024', '31/02/2024']))
    with pytest.warns(UserWarning) as caught:
        formats, rewritten = resolve_time_columns('events', _definition(),
                                                  [str(path)])
    assert (formats, rewritten) == ({'ts': '%d/%m/%Y %H:%M:%S'}, ['ts'])
    assert [str(w.message)[:60] for w in caught] == [
        'Time column events.ts holds slash dates read day first (%d/%',
        '2 values in time column events.ts are not dates on the calen',
    ]
    assert pq.read_table(path).column('ts').to_pylist() == [
        '13/01/2024 10:30:00', None, None
    ]


@pytest.mark.parametrize('parts, fmt, stored', [
    ([['1/13/2024', '1/14/2024', '1/15/2024 10:00']], '%m/%d/%Y %H:%M',
     [['1/13/2024 00:00', '1/14/2024 00:00', '1/15/2024 10:00']]),
    ([['13/01/2024 10:00:00'], ['14/01/2024', '15/01/2024 10:00', None]],
     '%d/%m/%Y %H:%M:%S',
     [None, ['14/01/2024 00:00:00', '15/01/2024 10:00:00', None]]),
])
def test_slash_dates_are_staged_in_the_fullest_time_shape(
        tmp_path, parts, fmt, stored):
    paths = [
        _part(tmp_path, pa.array(values, pa.string()), f'part-{i}.parquet')
        for i, values in enumerate(parts)
    ]
    before = [path.read_bytes() for path in paths]
    with pytest.warns(UserWarning) as caught:
        formats, rewritten = resolve_time_columns('events', _definition(),
                                                  [str(p) for p in paths])
    assert (formats, rewritten) == ({'ts': fmt}, ['ts'])
    assert len(caught) == 1
    for path, untouched, expected in zip(paths, before, stored):
        if expected is None:
            assert path.read_bytes() == untouched
        else:
            assert pq.read_table(path).column('ts').to_pylist() == expected


def test_mixed_timezones_are_staged_with_one_separator(tmp_path):
    path = _part(
        tmp_path,
        pa.array([
            '2024-01-16T10:00:00', '2024-01-15 10:00:00+02:00',
            '2024-01-17T10:00:00.250z', None
        ]))
    with pytest.warns(UserWarning, match='mixes 2 values with a timezone'):
        formats, rewritten = resolve_time_columns('events', _definition(),
                                                  [str(path)])
    assert (formats, rewritten) == ({'ts': '%Y-%m-%d %H:%M:%S%.f%#z'}, ['ts'])
    assert pq.read_table(path).column('ts').to_pylist() == [
        '2024-01-16 10:00:00', '2024-01-15 10:00:00+02:00',
        '2024-01-17 10:00:00.250Z', None
    ]


def test_mixed_timezones_are_staged_with_a_full_time_part(tmp_path):
    path = _part(
        tmp_path,
        pa.array([
            '2024-01-15', '2024-01-16T10:00', '2024-01-17 10:00+02:00',
            '2024-01-18 10:00:00.5Z'
        ]))
    with pytest.warns(UserWarning, match='mixes 2 values with a timezone'):
        formats, rewritten = resolve_time_columns('events', _definition(),
                                                  [str(path)])
    assert (formats, rewritten) == ({'ts': '%Y-%m-%d %H:%M:%S%.f%#z'}, ['ts'])
    assert pq.read_table(path).column('ts').to_pylist() == [
        '2024-01-15 00:00:00', '2024-01-16 10:00:00',
        '2024-01-17 10:00:00+02:00', '2024-01-18 10:00:00.5Z'
    ]


_LEFT = (", so its format is left to DiskGraph's own reading; its {found} "
         "{verb} read as DiskGraph reads {them}. Set timestamp_format on the "
         "column to decide it.")


@pytest.mark.parametrize('values, message', [
    (['99991231', '2024-01-16'],
     'Time column events.ts holds whole numbers beside ISO dates' +
     _LEFT.format(found='1 value outside 1677-09-21 to 2262-04-11 (e.g. '
                  '99991231)', verb='is', them='it')),
    (['2024-02-30', '2024-01-15', '15 Jan 2024'],
     'Time column events.ts holds 1 value in a shape the scan does not read '
     '(e.g. 15 Jan 2024)' +
     _LEFT.format(found='1 value no calendar holds (e.g. 2024-02-30)',
                  verb='is', them='it')),
    (['2024-02-30', '9999-12-31', '2024-02-31', '20240116'],
     'Time column events.ts holds whole numbers beside ISO dates' +
     _LEFT.format(found='2 values no calendar holds (e.g. 2024-02-30, '
                  '2024-02-31) and 1 value outside 1677-09-21 to 2262-04-11 '
                  '(e.g. 9999-12-31)', verb='are', them='them')),
])
def test_a_column_left_to_diskgraph_keeps_its_invalid_values(
        tmp_path, values, message):
    path = _part(tmp_path, pa.array(values, pa.string()))
    before = path.read_bytes()
    with pytest.warns(UserWarning) as caught:
        formats, rewritten = resolve_time_columns('events', _definition(),
                                                  [str(path)])
    assert (formats, rewritten) == ({}, [])
    assert [str(w.message) for w in caught] == [message]
    assert path.read_bytes() == before


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


_HATCH = (" Replace placeholders with nulls at the source to keep the "
          "{rows}, or set timestamp_format on the column to '{unit}' to read "
          "every value as epoch {unit_name}.")


@pytest.mark.parametrize('values, fmt, missing, message', [
    (pa.array([1700000000, 1700000100, 0], pa.int64()), 's', [2],
     '1 value in time column events.ts is not read as epoch seconds (e.g. 0);'
     ' it is treated as missing and its row is dropped like any row without a'
     ' time.' + _HATCH.format(rows='row', unit='s', unit_name='seconds')),
    (pa.array([1700000000, -86400, 50000000], pa.int64()), 's', [1, 2],
     '2 values in time column events.ts are not read as epoch seconds (e.g. '
     '-86400, 50000000); they are treated as missing and their rows are '
     'dropped like any row without a time.' +
     _HATCH.format(rows='rows', unit='s', unit_name='seconds')),
    (pa.array([1700000000000, 1700000100000, 0], pa.int64()), 'ms', [2],
     '1 value in time column events.ts is not read as epoch milliseconds '
     '(e.g. 0); it is treated as missing and its row is dropped like any row '
     'without a time.' +
     _HATCH.format(rows='row', unit='ms', unit_name='milliseconds')),
    (pa.array(['1705276800', '0', None]), 's', [1, 2],
     '1 value in time column events.ts is not read as epoch seconds (e.g. 0);'
     ' it is treated as missing and its row is dropped like any row without a'
     ' time.' + _HATCH.format(rows='row', unit='s', unit_name='seconds')),
    (pa.array([20240115, 0, 20241301], pa.int64()), '%Y%m%d', [1, 2],
     '2 values in time column events.ts are not read as YYYYMMDD dates (e.g. '
     '0, 20241301); they are treated as missing and their rows are dropped '
     'like any row without a time. Correct them at the source to keep the '
     'rows.'),
    (pa.array([2**63, 20240115], pa.uint64()), '%Y%m%d', [0],
     '1 value in time column events.ts is not read as a YYYYMMDD date (e.g. '
     '9223372036854775808); it is treated as missing and its row is dropped '
     'like any row without a time. Correct it at the source to keep the row.'),
    (pa.array([Decimal(10**19), Decimal(20240115)], pa.decimal128(
        38, 0)), '%Y%m%d', [0],
     '1 value in time column events.ts is not read as a YYYYMMDD date (e.g. '
     '10000000000000000000); it is treated as missing and its row is dropped '
     'like any row without a time. Correct it at the source to keep the row.'),
    (pa.array(['99999999999999999999', '20240115']), '%Y%m%d', [0],
     '1 value in time column events.ts is not read as a YYYYMMDD date (e.g. '
     '99999999999999999999); it is treated as missing and its row is dropped '
     'like any row without a time. Correct it at the source to keep the row.'),
    (pa.array(['9223372036854775808', '1705276800']), 's', [0],
     '1 value in time column events.ts is not read as epoch seconds (e.g. '
     '9223372036854775808); it is treated as missing and its row is dropped '
     'like any row without a time.' +
     _HATCH.format(rows='row', unit='s', unit_name='seconds')),
])
def test_numbers_without_the_settled_reading_are_made_missing(
        tmp_path, values, fmt, missing, message):
    path = _part(tmp_path, values)
    with pytest.warns(UserWarning) as caught:
        formats, rewritten = resolve_time_columns('events', _definition(),
                                                  [str(path)])
    assert (formats, rewritten) == ({'ts': fmt}, ['ts'])
    read = len(values) - len(missing)
    messages = [str(w.message) for w in caught]
    assert messages[0].startswith(
        f'Time column events.ts holds whole numbers; its {read} value'
        f"{'s' if read != 1 else ''} {'are' if read != 1 else 'is'} read as")
    assert messages[1:] == [message]
    stored = pq.read_table(path).column('ts')
    assert [i for i, v in enumerate(stored.to_pylist()) if v is None] == missing
    assert stored.type == values.type


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


@pytest.mark.parametrize('values, outcome, stored', [
    (pa.array([' 1705276800 ', '1705363200']), {
        'ts': 's'
    }, ['1705276800', '1705363200']),
    (pa.array([' 1705276800 ', '1705276800000']), ValueError, None),
    (pa.array([' 20240115', '20240116 ']), {
        'ts': '%Y%m%d'
    }, ['20240115', '20240116']),
    (pa.array(['1705276800.0', '1705363200.0']), {
        'ts': 's'
    }, ['1705276800', '1705363200']),
    (pa.array(['20240115.0', ' 20240116', '20240117', None]), {
        'ts': '%Y%m%d'
    }, ['20240115', '20240116', '20240117', None]),
    (pa.array(['020240115', '20240116'], pa.large_string()), {
        'ts': '%Y%m%d'
    }, ['20240115', '20240116']),
    (pa.array(['20240115.0', '20240116']).dictionary_encode(), {
        'ts': '%Y%m%d'
    }, ['20240115', '20240116']),
    (pa.array(['1705276800.0', '0']), {
        'ts': 's'
    }, ['1705276800', None]),
    (pa.array(['00000000001705276800', '1705363200']), {
        'ts': 's'
    }, ['1705276800', '1705363200']),
    (pa.array(['00000000001705276800', '00000000001705363200']), {
        'ts': 's'
    }, ['1705276800', '1705363200']),
    (pa.array(['00000000000000000000', '1705363200']), {
        'ts': 's'
    }, [None, '1705363200']),
])
def test_padded_digit_strings_are_staged_as_the_numbers_they_spell(
        tmp_path, values, outcome, stored):
    path = _part(tmp_path, values)
    if outcome is ValueError:
        with pytest.raises(ValueError, match='no single reading'):
            resolve_time_columns('events', _definition(), [str(path)])
        return
    with pytest.warns(UserWarning, match='holds whole numbers'):
        assert resolve_time_columns('events', _definition(),
                                    [str(path)]) == (outcome, ['ts'])
    column = pq.read_table(path).column('ts')
    assert column.to_pylist() == stored
    kind = values.type
    assert column.type == (kind.value_type
                           if pa.types.is_dictionary(kind) else kind)


def test_a_declared_format_is_left_alone(tmp_path):
    path = _part(tmp_path, pa.array(['05/03/1990', '30/02/1990']))
    before = path.read_bytes()
    formats, rewritten = resolve_time_columns('events',
                                              _definition(declared='%d/%m/%Y'),
                                              [str(path)])
    assert (formats, rewritten) == ({}, [])
    assert path.read_bytes() == before


@pytest.mark.parametrize('values, fmt, stored', [
    (['2024-02-30', '2024-01-15'], '%Y-%m-%d', [None, '2024-01-15']),
    (['2024-02-30T10:00:00', '2024-01-15T10:00:00.250Z'],
     '%Y-%m-%dT%H:%M:%S%.f%#z', [None, '2024-01-15T10:00:00.250Z']),
    (['2024-02-30 10:00', '2024-01-15 10:00+02:00'], '%Y-%m-%d %H:%M%#z',
     [None, '2024-01-15 10:00+02:00']),
    (['2024/02/30', '2024/1/15'], '%Y/%m/%d', [None, '2024/1/15']),
    (['2024/02/30 10:00:00', '2024/01/15 10:00:00'], '%Y/%m/%d %H:%M:%S',
     [None, '2024/01/15 10:00:00']),
    (['2024-02-30', '2024-01-15T10:00:00'], '%Y-%m-%d %H:%M:%S%.f%#z',
     [None, '2024-01-15 10:00:00']),
    (['2024-02-30T10:00:00z', '2024-01-15T10:00:00z'],
     '%Y-%m-%d %H:%M:%S%.f%#z', [None, '2024-01-15 10:00:00Z']),
    (['2024/02/30', '2024/01/15', '2024/01/16 10:00'], '%Y/%m/%d %H:%M',
     [None, '2024/01/15 00:00', '2024/01/16 10:00']),
    (['2024/02/30 10:00:00', '2024/1/15', '2024/01/16 10:00'],
     '%Y/%m/%d %H:%M:%S',
     [None, '2024/1/15 00:00:00', '2024/01/16 10:00:00']),
])
def test_a_column_with_values_made_missing_declares_its_format(
        tmp_path, values, fmt, stored):
    path = _part(tmp_path, pa.array(values, pa.string()))
    with pytest.warns(UserWarning) as caught:
        formats, rewritten = resolve_time_columns('events', _definition(),
                                                  [str(path)])
    assert (formats, rewritten) == ({'ts': fmt}, ['ts'])
    assert len(caught) == 1
    assert pq.read_table(path).column('ts').to_pylist() == stored


def test_the_parts_to_rewrite_are_settled_once_per_column(
        tmp_path, monkeypatch):
    from kumo.engine import _time_columns

    calls = []
    settle = _time_columns._Column.parts_to_rewrite

    def counted(column):
        calls.append(column.name)
        return settle(column)

    monkeypatch.setattr(_time_columns._Column, 'parts_to_rewrite', counted)
    paths = [
        _part(tmp_path, pa.array(['2024-02-30', f'2024-01-1{i}']),
              f'part-{i}.parquet') for i in range(3)
    ]
    with pytest.warns(UserWarning, match='3 values in time column'):
        resolve_time_columns('events', _definition(), [str(p) for p in paths])
    assert calls == ['ts']
    assert all(
        pq.read_table(path).column('ts').to_pylist()[0] is None
        for path in paths)


def test_a_part_is_rewritten_in_batches_bounded_by_decoded_bytes(
        tmp_path, monkeypatch):
    rows = 3000
    path = tmp_path / 'part-0.parquet'
    pq.write_table(
        pa.table({
            'id': pa.array(range(rows), pa.int64()),
            'ts': ['2024-02-30'] + ['2024-01-15'] * (rows - 1),
            'payload': ['x' * 16384] * rows,
        }), path)
    sizes = []
    write = pq.ParquetWriter.write_batch

    def recorded(writer, batch, *args, **kwargs):
        sizes.append(batch.nbytes)
        return write(writer, batch, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetWriter, 'write_batch', recorded)
    with pytest.warns(UserWarning, match='1 value in time column'):
        resolve_time_columns('events', _definition(), [str(path)])
    assert len(sizes) >= 3
    assert max(sizes) <= 17 * 2**20
    stored = pq.read_table(path)
    assert stored.num_rows == rows
    assert stored.column('ts').to_pylist()[:2] == [None, '2024-01-15']
    assert stored.column('payload')[rows - 1].as_py() == 'x' * 16384


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
    pa.array(['13/01/2024', '2024-01-16T10:00:00z', '2024-01-17T10:00:00z']),
    pa.array([None, None], pa.string()),
])
def test_other_shapes_are_left_to_diskgraph(tmp_path, values):
    path = _part(tmp_path, values)
    before = path.read_bytes()
    formats, rewritten = resolve_time_columns('events', _definition(),
                                              [str(path)])
    assert (formats, rewritten) == ({}, [])
    assert path.read_bytes() == before


@pytest.mark.parametrize('values, example', [
    (pa.array(['13/01/2024', '14/01/2024', 'unknown']), 'unknown'),
    (pa.array(['1705276800', '2023', 'unknown']), 'unknown'),
    (pa.array([
        '2024-01-15T08:00:00Z', '2024-1-16T09:00:00', '2024-01-17T10:00:00'
    ]), '2024-1-16T09:00:00'),
    (pa.array([
        '2024-01-16T10:00:00', '2024-01-15 10:00:00+02:00', 'not a time'
    ]), 'not a time'),
])
def test_a_shape_the_scan_does_not_read_leaves_the_column_to_diskgraph(
        tmp_path, values, example):
    path = _part(tmp_path, values)
    before = path.read_bytes()
    with pytest.warns(
            UserWarning, match=re.escape(
                f'Time column events.ts holds 1 value in a shape the scan '
                f'does not read (e.g. {example}), so its format is left to '
                'DiskGraph')):
        formats, rewritten = resolve_time_columns('events', _definition(),
                                                  [str(path)])
    assert (formats, rewritten) == ({}, [])
    assert path.read_bytes() == before


def test_slash_dates_invalid_under_both_orders_are_made_missing(tmp_path):
    path = _part(tmp_path, pa.array(['00/00/0000', '30/02/2024']))
    with pytest.warns(UserWarning) as caught:
        formats, rewritten = resolve_time_columns('events',
                                                  _definition(end=True),
                                                  [str(path)])
    assert (formats, rewritten) == ({}, ['ts'])
    assert [str(w.message) for w in caught] == [
        '2 values in end time column events.ts are not dates on the calendar '
        '(e.g. 00/00/0000, 30/02/2024); they are treated as missing. Correct '
        'them at the source to keep the values.'
    ]
    assert pq.read_table(path).column('ts').to_pylist() == [None, None]
