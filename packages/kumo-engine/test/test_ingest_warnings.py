# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
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
