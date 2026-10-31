# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from dataclasses import replace

import pytest

import kumo.engine as kumo
from kumo.engine.graph import schema_changes
from kumo.engine.pquery.predictive_query import PredictiveQuery


class _Stop(Exception):
    pass


def _other(snapshot, edit):
    config = copy.deepcopy(snapshot.graph_config)
    edit({t['table_name']: t for t in config['tables']})
    return replace(snapshot, id='f' * 64 + '@' + snapshot.id.partition('@')[2],
                   graph_config=config)


def _moved(tables):
    for table in tables.values():
        table['connector_id'] = 'prod'
        table['source_table_name'] = 'prod.' + table['source_table_name']


def _predict(trainer, snapshot, monkeypatch):
    def stop(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(PredictiveQuery, '_from_snapshot', stop)
    with pytest.raises(_Stop):
        trainer.predict(snapshot, binary_classification_threshold=.5)


def test_predict_says_what_differs_from_the_training_graph(
        fitted, monkeypatch, tmp_path):
    trainer, _, snapshot = fitted
    assert kumo.Trainer.load(trainer.result.bundle_dir)._graph_config == (
        snapshot.graph_config)

    with pytest.warns(UserWarning, match='The schema is the same; only '
                      'table connectors or locations differ'):
        _predict(trainer, _other(snapshot, _moved), monkeypatch)

    def dropped(tables):
        _moved(tables)
        tables['events']['fkeys'] = []

    with pytest.warns(UserWarning, match=r'trained on a snapshot of graph '
                      r'\w+\. Schema changes: events edges\.$'):
        _predict(trainer, _other(snapshot, dropped), monkeypatch)

    monkeypatch.setattr(trainer, '_graph_config', None)
    with pytest.warns(UserWarning, match='does not record its training graph'):
        _predict(trainer, _other(snapshot, _moved), monkeypatch)


def test_schema_changes_ignore_connectors_and_locations():
    def config(location, columns, tables=('users', )):
        return {
            'name': location,
            'tables': [{
                'table_name': name,
                'source_table_name': f'{location}.{name}',
                'connector_id': location,
                'pkey': {'name': 'id', 'dtype': 'int64', 'stype': 'ID'},
                'cols': columns,
                'fkeys': [],
            } for name in tables],
        }

    age = {'name': 'age', 'dtype': 'int64', 'stype': 'numerical'}
    label = {'name': 'label', 'dtype': 'bool', 'stype': 'categorical'}
    assert schema_changes(config('dev', [age, label]),
                          config('prod', [label, age])) == []
    assert schema_changes(
        config('dev', [age]),
        config('dev', [{**age, 'dtype': 'float64'}], tables=('users', 'x'))
    ) == ['table x added', 'users columns']
