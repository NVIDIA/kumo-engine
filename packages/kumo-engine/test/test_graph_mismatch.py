# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import warnings
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


def _predict(trainer, snapshot, monkeypatch, **kwargs):
    def stop(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(PredictiveQuery, '_from_snapshot', stop)
    with pytest.raises(_Stop):
        trainer.predict(snapshot, binary_classification_threshold=.5,
                        **kwargs)


def test_predict_says_what_differs_from_the_training_graph(
        fitted, monkeypatch, tmp_path):
    trainer, table, snapshot = fitted
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

    strict = kumo.MLOps(str(tmp_path / 'root'), on_error='raise')
    with pytest.raises(ValueError, match=r"Schema changes: events edges; "
                       r"declare the same"):
        trainer.predict(_other(snapshot, dropped), mlops=strict,
                        binary_classification_threshold=.5)
    with pytest.warns(UserWarning, match='The schema is the same'):
        _predict(trainer, _other(snapshot, _moved), monkeypatch, mlops=strict)

    def left_out(snapshot):
        config = copy.deepcopy(snapshot.graph_config)
        config['tables'] = [
            table for table in config['tables']
            if table['table_name'] != 'events'
        ]
        return replace(snapshot, graph_config=config)

    with pytest.warns(
            UserWarning, match=r'differs from the snapshot this '
            r'model was trained on\. Schema changes: table events '
            r'removed\.$'):
        _predict(trainer, left_out(snapshot), monkeypatch)
    with pytest.raises(
            ValueError, match=r'Schema changes: table events '
            r'removed; declare the same'):
        trainer.predict(left_out(snapshot), mlops=strict,
                        binary_classification_threshold=.5)
    with pytest.raises(
            ValueError, match=r'Schema changes: table events '
            r'removed; declare the same'):
        trainer.predict(left_out(snapshot), prediction_table=table,
                        mlops=strict, binary_classification_threshold=.5)

    def stop(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(trainer, '_predict', stop)
    with pytest.warns(
            UserWarning, match='differs from the snapshot this '
            'model was trained on'):
        with pytest.raises(_Stop):
            trainer.predict(left_out(snapshot), prediction_table=table,
                            binary_classification_threshold=.5)

    def derived(snapshot):
        config = copy.deepcopy(snapshot.graph_config)
        for table in config['tables']:
            if 'time_col' in table:
                table['time_col']['timestamp_format'] = 'YYYY-MM-DD'
            for column in [*table['cols'], *table['fkeys'], table['pkey']]:
                if column['dtype'] == 'int64':
                    column['dtype'] = 'int32'
        return replace(snapshot, graph_config=config)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        _predict(trainer, derived(snapshot), monkeypatch)
        _predict(trainer, derived(snapshot), monkeypatch, mlops=strict)
    assert not [
        str(record.message) for record in records
        if 'trained on' in str(record.message)
    ]

    monkeypatch.setattr(trainer, '_graph_config', None)
    with pytest.warns(UserWarning, match='does not record its training graph'):
        _predict(trainer, _other(snapshot, _moved), monkeypatch)
    _predict(trainer, left_out(snapshot), monkeypatch)


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
    ) == ['table x added', 'users columns (age)']
