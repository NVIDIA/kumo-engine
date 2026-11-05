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


def _col_stats(tmp_path, stats):
    path = tmp_path / 'col_stats.json'
    path.write_text(json.dumps(stats))
    return SimpleNamespace(
        kumo_stats_dir=str(tmp_path), col_stats_path=str(path),
        ingest_stats_path=str(tmp_path / 'ingest_stats.json'))


def test_snapshot_reports_non_finite_values_per_column(tmp_path):
    result = _col_stats(
        tmp_path, {
            'events': {
                'amount': {
                    'count': 12,
                    'invalid_count': 3,
                    'parse_fail_count': 0
                },
                'clean': {
                    'count': 12,
                    'invalid_count': 0
                },
                'odd': {
                    'invalid_count': 'bad'
                },
                'flag': {
                    'count': 3
                },
            }
        })
    with pytest.warns(UserWarning) as caught:
        _warn_ingest_data_loss(result, {'tables': [{'table_name': 'events'}]})
    assert [str(w.message) for w in caught] == [
        '3 non-finite value(s) (inf, -inf or NaN) found in events.amount. '
        'They are left out of the column statistics and treated as missing. '
        'Infinity usually comes from a division by zero upstream; replace '
        'non-finite values at the source.',
    ]


@pytest.mark.parametrize('column, stat, expected', [
    ({
        'name': 'country',
        'dtype': 'string',
        'stype': 'categorical'
    }, {
        'category_counts': [['US', 5], ['NA', 2], ['N/A', 1], ['None', 1],
                            ['', 3]],
        'category_counts_capped': False
    }, "events.country holds 7 value(s) that look like missing ('', 'N/A', "
     "'NA', 'None'); they are kept as text. If they mean missing, replace "
     "them with nulls at the source; a number or date column reads them as "
     "missing."),
    ({
        'name': 'country',
        'dtype': 'string',
        'stype': 'categorical'
    }, {
        'category_counts': [['null', 4]],
        'category_counts_capped': True
    }, "events.country holds at least 4 value(s) that look like missing "
     "('null'); they are kept as text. If they mean missing, replace them "
     "with nulls at the source; a number or date column reads them as "
     "missing."),
    ({
        'name': 'country',
        'dtype': 'string',
        'stype': 'categorical'
    }, {
        'category_counts': [['US', 5], ['DE', 2]]
    }, None),
    ({
        'name': 'code',
        'dtype': 'string',
        'stype': 'ID'
    }, {
        'category_counts': [['NA', 5]]
    }, None),
    ({
        'name': 'tags',
        'dtype': 'string',
        'stype': 'multicategorical'
    }, {
        'category_counts': [['NA', 5]]
    }, None),
    ({
        'name': 'age',
        'dtype': 'int64',
        'stype': 'categorical'
    }, {
        'category_counts': [[1, 5], [None, 2]]
    }, None),
    ({
        'name': 'country',
        'dtype': 'string',
        'stype': 'categorical'
    }, {
        'category_counts': 'bad'
    }, None),
])
def test_snapshot_reports_text_that_looks_like_missing(tmp_path, column, stat,
                                                       expected):
    result = _col_stats(tmp_path, {'events': {column['name']: stat}})
    config = {'tables': [{'table_name': 'events', 'cols': [column]}]}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _warn_ingest_data_loss(result, config)
    messages = [str(w.message) for w in caught]
    assert messages == ([expected] if expected else [])


@pytest.mark.parametrize('narrowed, low, high, expected', [
    ({
        ('events', 'amount'): ('double', 0)
    }, 0.0, 16777216.0, "events.amount is stored as float32 but comes from "
     "double and reaches 16,777,216; from 16,777,216 on float32 no longer "
     "resolves every whole number, so values are rounded. Declare "
     "dtype=Dtype.float64 for the column to keep full precision."),
    ({
        ('events', 'amount'): ('double', 0)
    }, 0.0, 16777215.0, None),
    ({
        ('events', 'amount'): ('decimal128(10, 2)', 2)
    }, 0.25, 131072.015625, "events.amount is stored as float32 but comes "
     "from decimal128(10, 2) and reaches 131,072.02; from 131,072 on float32 "
     "no longer resolves steps of 0.01, so values are rounded. Declare "
     "dtype=Dtype.float64 for the column to keep full precision."),
    ({
        ('events', 'amount'): ('decimal128(10, 2)', 2)
    }, 0.25, 131071.98, None),
    ({
        ('events', 'amount'): ('int64', 0)
    }, -16777300.0, 5.0, "events.amount is stored as float32 but comes from "
     "int64 and reaches 16,777,300; from 16,777,216 on float32 no longer "
     "resolves every whole number, so values are rounded. Declare "
     "dtype=Dtype.float64 for the column to keep full precision."),
    ({}, 0.0, 1e12, None),
    ({
        ('events', 'other'): ('double', 0)
    }, 0.0, 1e12, None),
    ({
        ('events', 'amount'): ('decimal128(10, 2)', 2)
    }, [['0.25', 2], ['131072.015625', 2]], None,
     "events.amount is stored as float32 but comes from decimal128(10, 2) "
     "and reaches 131,072.02; from 131,072 on float32 no longer resolves "
     "steps of 0.01, so values are rounded. Declare dtype=Dtype.float64 for "
     "the column to keep full precision."),
    ({
        ('events', 'amount'): ('double', 0)
    }, [['a', 2], [], None], None, None),
])
def test_snapshot_reports_values_beyond_float32_precision(
        tmp_path, narrowed, low, high, expected):
    stat = ({
        'category_counts': low
    } if isinstance(low, list) else {
        'min': low,
        'max': high
    })
    result = _col_stats(tmp_path, {'events': {'amount': stat}})
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
    snapshot = GraphSnapshot('graph@data', index, {'tables': []})
    assert snapshot.invalid_value_counts == {}
    (index / 'col_stats.json').write_text(
        json.dumps({
            'events': {
                'amount': {
                    'parse_fail_count': 4,
                    'invalid_count': 2
                },
                'clean': {
                    'parse_fail_count': 0,
                    'invalid_count': 0
                },
                'flag': {
                    'parse_fail_count': 1
                },
                'bad': {
                    'parse_fail_count': 'x'
                },
                'odd': None,
            },
            'users': {
                'age': {
                    'parse_fail_count': 0
                }
            },
            'broken': [],
        }))
    assert snapshot.invalid_value_counts == {'events': {'amount': 6, 'flag': 1}}
