# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import warnings
from types import SimpleNamespace

import pytest

from kumo.engine.graph import _warn_ingest_data_loss


def test_snapshot_reports_one_parse_warning_per_column(tmp_path):
    directory = tmp_path / 'col_stats_min'
    directory.mkdir()
    (directory / 'events.json').write_text(
        json.dumps({
            'amount__kumo.numerical__NUM_PARSE_FAILURES__False': {
                'value': 4
            },
            'date__kumo.timestamp__NUM_PARSE_FAILURES__False': {
                'value': 6
            },
            'clean__kumo.numerical__NUM_PARSE_FAILURES__False': {
                'value': 0
            },
            'malformed__kumo.numerical__NUM_PARSE_FAILURES__False': {},
            'nullable__kumo.numerical__NUM_NA__False': {
                'value': 10
            },
        }))
    result = SimpleNamespace(kumo_stats_dir=str(tmp_path))
    with pytest.warns(UserWarning) as caught:
        _warn_ingest_data_loss(result, {'tables': [{'table_name': 'events'}]})
    assert [str(w.message) for w in caught] == [
        '4 invalid values found in events.amount. Invalid values are parsed '
        'as null. Please correct and retrain if needed.',
        '6 invalid values found in events.date. Invalid values are parsed '
        'as null. Please correct and retrain if needed.',
    ]


@pytest.mark.parametrize('column_stats', [
    None,
    [],
    {
        'amount__kumo.numerical__NUM_PARSE_FAILURES__False': None
    },
    {
        'amount__kumo.numerical__NUM_PARSE_FAILURES__False': {}
    },
    {
        'amount__kumo.numerical__NUM_PARSE_FAILURES__False': {
            'value': 'bad'
        }
    },
    {
        'amount__kumo.numerical__NUM_PARSE_FAILURES__False': {
            'value': -1
        }
    },
    {
        'amount__kumo.numerical__NUM_PARSE_FAILURES__False': {
            'value': True
        }
    },
])
def test_optional_parse_stats_do_not_block_existing_warnings(
        tmp_path, column_stats):
    directory = tmp_path / 'col_stats_min'
    directory.mkdir()
    (directory / 'events.json').write_text(json.dumps(column_stats))
    ingest_stats = tmp_path / 'ingest_stats.json'
    ingest_stats.write_text(
        json.dumps({
            'stats': {
                'pk_duplicate_counts': {
                    'events:id': 2
                }
            },
        }))
    result = SimpleNamespace(kumo_stats_dir=str(tmp_path),
                             ingest_stats_path=str(ingest_stats))
    with pytest.warns(UserWarning, match='2 duplicate primary-key') as caught:
        _warn_ingest_data_loss(
            result,
            {'tables': [
                {
                    'table_name': 'events',
                    'pkey': {
                        'name': 'id'
                    }
                },
            ]})
    assert len(caught) == 1


def _min_stats(tmp_path, table, columns):
    directory = tmp_path / 'col_stats_min'
    directory.mkdir(exist_ok=True)
    (directory / f'{table}.json').write_text(
        json.dumps({
            f'{column}__kumo.stype__{stat}__False': {
                'value': value
            }
            for column, stats in columns.items()
            for stat, value in stats.items()
        }))
    return SimpleNamespace(
        kumo_stats_dir=str(tmp_path),
        ingest_stats_path=str(tmp_path / 'ingest_stats.json'))


def test_snapshot_reports_non_finite_values_per_column(tmp_path):
    result = _min_stats(
        tmp_path, 'events', {
            'amount': {
                'COUNT': 12,
                'INVALID_FRACTION': 0.25,
                'NUM_PARSE_FAILURES': 0
            },
            'ratio': {
                'COUNT': 12,
                'INVALID_FRACTION': 1 / 6
            },
            'clean': {
                'COUNT': 12,
                'INVALID_FRACTION': 0.0
            },
            'odd': {
                'COUNT': 12,
                'INVALID_FRACTION': 'bad'
            },
            'uncounted': {
                'INVALID_FRACTION': 0.5
            },
            'flag': {
                'COUNT': 3
            },
        })
    with pytest.warns(UserWarning) as caught:
        _warn_ingest_data_loss(result, {'tables': [{'table_name': 'events'}]})
    assert [str(w.message) for w in caught] == [
        '3 non-finite value(s) (inf, -inf or NaN) found in events.amount. '
        'They are stored as is and left out of the column statistics. '
        'Infinity usually comes from a division by zero upstream; replace '
        'non-finite values at the source.',
        '2 non-finite value(s) (inf, -inf or NaN) found in events.ratio. '
        'They are stored as is and left out of the column statistics. '
        'Infinity usually comes from a division by zero upstream; replace '
        'non-finite values at the source.',
    ]


@pytest.mark.parametrize('column, counts, expected', [
    ({
        'name': 'country',
        'dtype': 'string',
        'stype': 'categorical'
    }, [['US', 5], ['NA', 2], ['N/A', 1], ['None', 1], ['', 3]
        ], "events.country holds 7 value(s) that look like missing ('', 'N/A', "
     "'NA', 'None'); they are kept as text. If they mean missing, replace "
     "them with nulls at the source; a number or date column reads them as "
     "missing."),
    ({
        'name': 'country',
        'dtype': 'string',
        'stype': 'categorical'
    }, [['US', 5], ['DE', 2]], None),
    ({
        'name': 'code',
        'dtype': 'string',
        'stype': 'ID'
    }, [['NA', 5]], None),
    ({
        'name': 'notes',
        'dtype': 'string',
        'stype': 'text'
    }, [['NA', 5]], None),
    ({
        'name': 'tags',
        'dtype': 'string',
        'stype': 'multicategorical'
    }, [['NA', 5]], None),
    ({
        'name': 'age',
        'dtype': 'int64',
        'stype': 'categorical'
    }, [[1, 5], [None, 2]], None),
    ({
        'name': 'flag',
        'dtype': 'bool',
        'stype': 'categorical'
    }, [[False, 5], [True, 2]], None),
    ({
        'name': 'country',
        'dtype': 'string',
        'stype': 'categorical'
    }, 'bad', None),
    ({
        'name': 'country',
        'dtype': 'string',
        'stype': 'categorical'
    }, [['NA'], [], ['N/A', 'x'], ['null', 0]], None),
])
def test_snapshot_reports_text_that_looks_like_missing(tmp_path, column, counts,
                                                       expected):
    result = _min_stats(tmp_path, 'events',
                        {column['name']: {
                             'CATEGORY_COUNTS': counts
                         }})
    config = {'tables': [{'table_name': 'events', 'cols': [column]}]}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _warn_ingest_data_loss(result, config)
    messages = [str(w.message) for w in caught]
    assert messages == ([expected] if expected else [])


def test_missing_like_count_is_a_lower_bound_past_the_kept_categories(tmp_path):
    result = _min_stats(
        tmp_path, 'events', {
            'country': {
                'COUNT': 100,
                'NUM_NA': 5,
                'CATEGORY_COUNTS': [['US', 50], ['NA', 10]]
            }
        })
    config = {
        'tables': [{
            'table_name':
            'events',
            'cols': [{
                'name': 'country',
                'dtype': 'string',
                'stype': 'categorical'
            }]
        }]
    }
    with pytest.warns(UserWarning, match='holds at least 10 value'):
        _warn_ingest_data_loss(result, config)


@pytest.mark.parametrize('narrowed, stats, expected', [
    ({
        ('events', 'amount'): ('double', 0)
    }, {
        'MIN': 0.0,
        'MAX': 16777216.0
    }, "events.amount is stored as float32 but comes from double and reaches "
     "16,777,216; from 16,777,216 on float32 no longer resolves every whole "
     "number, so values are rounded. Declare dtype=Dtype.float64 for the "
     "column to keep full precision."),
    ({
        ('events', 'amount'): ('double', 0)
    }, {
        'MIN': 0.0,
        'MAX': 16777215.0
    }, None),
    ({
        ('events', 'amount'): ('decimal128(10, 2)', 2)
    }, {
        'MIN': 0.25,
        'MAX': 131072.015625
    }, "events.amount is stored as float32 but comes from decimal128(10, 2) "
     "and reaches 131,072.02; from 131,072 on float32 no longer resolves "
     "steps of 0.01, so values are rounded. Declare dtype=Dtype.float64 for "
     "the column to keep full precision."),
    ({
        ('events', 'amount'): ('decimal128(10, 2)', 2)
    }, {
        'MIN': 0.25,
        'MAX': 131071.98
    }, None),
    ({
        ('events', 'amount'): ('int64', 0)
    }, {
        'MIN': -16777300.0,
        'MAX': 5.0
    }, "events.amount is stored as float32 but comes from int64 and reaches "
     "16,777,300; from 16,777,216 on float32 no longer resolves every whole "
     "number, so values are rounded. Declare dtype=Dtype.float64 for the "
     "column to keep full precision."),
    ({}, {
        'MIN': 0.0,
        'MAX': 1e12
    }, None),
    ({
        ('events', 'other'): ('double', 0)
    }, {
        'MIN': 0.0,
        'MAX': 1e12
    }, None),
    ({
        ('events', 'amount'): ('decimal128(10, 2)', 2)
    }, {
        'CATEGORY_COUNTS': [[0.25, 2], [131072.015625, 2], ['inf', 1]]
    }, "events.amount is stored as float32 but comes from decimal128(10, 2) "
     "and reaches 131,072.02; from 131,072 on float32 no longer resolves "
     "steps of 0.01, so values are rounded. Declare dtype=Dtype.float64 for "
     "the column to keep full precision."),
    ({
        ('events', 'amount'): ('double', 0)
    }, {
        'CATEGORY_COUNTS': [['a', 2], [], None, [True, 1]],
        'MIN': 'x'
    }, None),
])
def test_snapshot_reports_values_beyond_float32_precision(
        tmp_path, narrowed, stats, expected):
    result = _min_stats(tmp_path, 'events', {'amount': stats})
    config = {
        'tables': [{
            'table_name':
            'events',
            'cols': [{
                'name': 'amount',
                'dtype': 'float32',
                'stype': 'numerical'
            }]
        }]
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _warn_ingest_data_loss(result, config, narrowed)
    messages = [str(w.message) for w in caught]
    assert messages == ([expected] if expected else [])


def test_snapshot_exposes_invalid_value_counts_per_column(tmp_path):
    from kumo.engine.graph import GraphSnapshot

    index = tmp_path / 'index'
    index.mkdir()
    tables = [{'table_name': name} for name in ('events', 'users', 'absent')]
    snapshot = GraphSnapshot('graph@data', index, {'tables': tables})
    assert snapshot.invalid_value_counts == {}
    stats = index / 'kumo_col_stats'
    stats.mkdir()
    _min_stats(
        stats, 'events', {
            'amount': {
                'NUM_PARSE_FAILURES': 4,
                'INVALID_FRACTION': 1 / 6,
                'COUNT': 12
            },
            'clean': {
                'NUM_PARSE_FAILURES': 0,
                'INVALID_FRACTION': 0.0,
                'COUNT': 12
            },
            'flag': {
                'NUM_PARSE_FAILURES': 1
            },
            'bad': {
                'NUM_PARSE_FAILURES': 'x'
            },
        })
    _min_stats(stats, 'users', {'age': {'NUM_PARSE_FAILURES': 0}})
    (stats / 'col_stats_min' / 'broken.json').write_text('[]')
    assert snapshot.invalid_value_counts == {'events': {'amount': 6, 'flag': 1}}
