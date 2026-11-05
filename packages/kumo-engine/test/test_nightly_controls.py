# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controls preserve observed rows and move complete, valid answer tuples."""

import importlib.util
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPTS = Path(__file__).parents[1] / 'scripts'


def test_control_generator_produces_current_rows(tmp_path, helpers):
    controls = load('nightly_controls')
    files = controls.generate(write_rows(tmp_path / 'source', scalar_rows()),
                              'regression', tmp_path / 'output',
                              dict(method='stage-permutation-v1',
                                   seed=0), helpers=helpers)
    assert len(read_rows(files)) == 24


def load(name):
    path = SCRIPTS / (name + '.py')
    assert path.is_file(), 'The bounded control generator is missing'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def controls():
    return load('nightly_controls')


@pytest.fixture
def helpers():
    return SimpleNamespace(
        ordered_batches=load('nightly_predictions').ordered_batches)


def write_rows(root, rows, width=7):
    root.mkdir()
    files = []
    schema = pa.Table.from_pylist(rows).schema
    for offset in range(0, len(rows), width):
        path = root / f'{offset}.parquet'
        pq.write_table(
            pa.Table.from_pylist(rows[offset:offset + width], schema=schema),
            path)
        files.append(str(path))
    return files


def read_rows(files):
    return [
        row for path in files for batch in pq.ParquetFile(path).iter_batches()
        for row in batch.to_pylist()
    ]


def scalar_rows():
    return [
        dict(ENTITY1=entity, SPLIT=stage,
             TIMESTAMP=datetime(2024, 1, 1) + timedelta(days=stage * 10),
             TARGET1=stage * 100 + entity, observed=f'entity-{entity}')
        for entity in range(8) for stage in range(3)
    ]


def canonical(rows):
    return sorted(
        rows, key=lambda row:
        (row['ENTITY1'], row.get('TIMESTAMP'), row['SPLIT']))


def test_stage_controls_ignore_parquet_order_and_keep_observed_rows(
        controls, helpers, tmp_path):
    rows = scalar_rows()
    first = controls.generate(write_rows(tmp_path / 'first',
                                         rows), 'regression', tmp_path / 'a',
                              dict(method='stage-permutation-v1', seed=0),
                              helpers=helpers)
    second = controls.generate(
        list(reversed(write_rows(tmp_path / 'second', list(reversed(rows)),
                                 5))), 'regression', tmp_path / 'b',
        dict(method='stage-permutation-v1', seed=0), helpers=helpers)
    actual = read_rows(first)
    assert actual == read_rows(second)
    assert [{
        k: v
        for k, v in row.items() if k != 'TARGET1'
    } for row in actual] == [{
        k: v
        for k, v in row.items() if k != 'TARGET1'
    } for row in canonical(rows)]
    assert any(a['TARGET1'] != b['TARGET1']
               for a, b in zip(actual, canonical(rows)))
    for stage in range(3):
        assert Counter(row['TARGET1'] for row in actual
                       if row['SPLIT'] == stage) == Counter(
                           row['TARGET1'] for row in rows
                           if row['SPLIT'] == stage)


@pytest.mark.parametrize(
    'task', ['regression', 'forecasting', 'static_link', 'temporal_link'])
def test_ordered_controls_match_canonical_generation_without_sorting(
        controls, helpers, tmp_path, monkeypatch, task):
    rows = scalar_rows()
    for row in rows:
        if task == 'forecasting':
            row['future_labels'] = [float(row['TARGET1'])] * 12
            row['autoregression_labels'] = [float(row['ENTITY1'])] * 12
        elif task.endswith('link'):
            row['TARGET1'] = [row['ENTITY1'] * 10 + row['SPLIT']]
            if task == 'static_link':
                row.pop('TIMESTAMP')
    recipe = dict(
        method='entity-trajectory-permutation-v1'
        if task == 'static_link' else 'stage-permutation-v1', seed=0)
    expected = read_rows(
        controls.generate(
            write_rows(tmp_path / 'shuffled', list(reversed(rows))), task,
            tmp_path / 'expected', recipe, helpers=helpers))

    def reject_sort(*args, **kwargs):
        raise AssertionError('Ordered controls repeated the full sort')

    monkeypatch.setattr(controls, 'BATCH_ROWS', 3)
    source = write_rows(tmp_path / 'ordered', canonical(rows), width=5)
    actual = controls.generate(
        source, task, tmp_path / 'actual', recipe,
        helpers=SimpleNamespace(ordered_batches=reject_sort), ordered=True)
    assert read_rows(actual) == expected
    assert all(
        pq.read_schema(path) == pq.read_schema(source[0]) for path in actual)
    assert max(pq.ParquetFile(path).metadata.num_rows for path in actual) <= 3


def test_stage_donor_queries_do_not_need_a_temporary_sort(
        controls, helpers, tmp_path, monkeypatch):
    generate_donors = controls._donors
    plans = []

    def inspect_donors(connection, task, work, seed):
        generate_donors(connection, task, work, seed)
        plans.extend(
            connection.execute(
                'EXPLAIN QUERY PLAN SELECT position FROM rows '
                'WHERE stage=? ORDER BY position', (0, )).fetchall())

    monkeypatch.setattr(controls, '_donors', inspect_donors)
    controls.generate(write_rows(tmp_path / 'source', scalar_rows()),
                      'regression', tmp_path / 'output',
                      dict(method='stage-permutation-v1',
                           seed=0), helpers=helpers)
    assert plans
    assert not any('TEMP B-TREE' in detail.upper() for *_, detail in plans)


def test_ordered_controls_reject_changed_parquet_schema(controls, helpers,
                                                        tmp_path):
    source = write_rows(tmp_path / 'source', scalar_rows())
    changed = pa.Table.from_pylist(scalar_rows()[7:14]).drop(['observed'])
    pq.write_table(changed, source[1])
    with pytest.raises(ValueError, match='schemas differ'):
        controls.generate(source, 'regression', tmp_path / 'output',
                          dict(method='stage-permutation-v1', seed=0),
                          helpers=helpers, ordered=True)


@pytest.mark.parametrize(
    'task', ['binary', 'multiclass', 'multilabel', 'multilabel_ranking'])
def test_classification_controls_keep_labels_and_frequencies_together(
        controls, helpers, tmp_path, task):
    rows = scalar_rows()
    for row in rows:
        answer = row['TARGET1']
        row['TARGET1'] = bool(answer % 2) if task == 'binary' else (
            f'class-{answer}'
            if task == 'multiclass' else [f'class-{answer}', 'shared'])
        if task == 'multilabel_ranking':
            row['__kumo_internal_frequency_column__'] = [answer + 1, 1]
    actual = read_rows(
        controls.generate(write_rows(tmp_path / 'source',
                                     rows), task, tmp_path / 'output',
                          dict(method='stage-permutation-v1', seed=0),
                          helpers=helpers))
    for row in actual:
        assert row['observed'] == f'entity-{row["ENTITY1"]}'
        if task == 'multilabel_ranking':
            assert row['__kumo_internal_frequency_column__'] == [
                int(row['TARGET1'][0][6:]) + 1, 1
            ]


def test_forecast_moves_all_future_values_with_scalar_and_keeps_history(
        controls, helpers, tmp_path):
    rows = scalar_rows()
    for row in rows:
        row['future_labels'] = [
            float(row['TARGET1'] + offset) for offset in range(12)
        ]
        row['autoregression_labels'] = [float(row['ENTITY1'])] * 12
    actual = read_rows(
        controls.generate(write_rows(tmp_path / 'source',
                                     rows), 'forecasting', tmp_path / 'output',
                          dict(method='stage-permutation-v1', seed=0),
                          helpers=helpers))
    for row in actual:
        assert row['future_labels'] == [
            float(row['TARGET1'] + offset) for offset in range(12)
        ]
        assert row['autoregression_labels'] == [float(row['ENTITY1'])] * 12


def test_temporal_controls_never_borrow_from_another_anchor(
        controls, helpers, tmp_path):
    rows = []
    for row in scalar_rows():
        for anchor in (0, 1):
            item = dict(row,
                        TIMESTAMP=row['TIMESTAMP'] + timedelta(days=anchor),
                        TARGET1=[row['TARGET1'] + anchor * 1000],
                        __kumo_internal_frequency_column__=[row['ENTITY1'] + 1])
            rows.append(item)
    actual = read_rows(
        controls.generate(write_rows(tmp_path / 'source', rows),
                          'temporal_link', tmp_path / 'output',
                          dict(method='stage-permutation-v1',
                               seed=0), helpers=helpers))
    for row in actual:
        anchor = row['TIMESTAMP'].day % 10 - 1
        target = row['TARGET1'][0]
        assert target // 1000 == anchor
        assert row['__kumo_internal_frequency_column__'] == [target % 100 + 1]


def test_static_controls_move_whole_histories_with_equal_key_membership(
        controls, helpers, tmp_path):
    rows = scalar_rows()
    for row in rows:
        row['TARGET1'] = [row['ENTITY1'] * 10 + row['SPLIT']]
        row.pop('TIMESTAMP')
    rows = [row for row in rows if row['ENTITY1'] != 7 or row['SPLIT'] != 1]
    first = controls.generate(
        write_rows(tmp_path / 'source',
                   rows), 'static_link', tmp_path / 'output',
        dict(method='entity-trajectory-permutation-v1', seed=0),
        helpers=helpers)
    actual = read_rows(first)
    second = controls.generate(
        write_rows(tmp_path / 'reordered', list(reversed(rows)),
                   4), 'static_link', tmp_path / 'again',
        dict(method='entity-trajectory-permutation-v1', seed=0),
        helpers=helpers)
    assert actual == read_rows(second)
    assert [(row['ENTITY1'], row['SPLIT'], row['observed']) for row in actual
            ] == [(row['ENTITY1'], row['SPLIT'], row['observed'])
                  for row in canonical(rows)]
    for entity in range(8):
        targets = [
            row['TARGET1'][0] for row in actual if row['ENTITY1'] == entity
        ]
        assert len({target // 10 for target in targets}) == 1
        assert len(targets) == len(set(targets))
    assert [row['TARGET1'] for row in actual if row['ENTITY1'] == 7] == [[70],
                                                                         [72]]


def test_static_rejects_a_target_in_multiple_stages(controls, helpers,
                                                    tmp_path):
    rows = [dict(row, TARGET1=[row['ENTITY1']]) for row in scalar_rows()]
    with pytest.raises(ValueError, match='overlap'):
        controls.generate(
            write_rows(tmp_path / 'source',
                       rows), 'static_link', tmp_path / 'output',
            dict(method='entity-trajectory-permutation-v1', seed=0),
            helpers=helpers)


@pytest.mark.parametrize('recipe', [
    None, {}, {
        'method': 'stage-permutation-v1'
    }, {
        'method': 'stage-permutation-v1',
        'seed': True
    }, {
        'method': 'stage-permutation-v1',
        'seed': 1
    }, {
        'method': 'old-payload',
        'seed': 0
    }, {
        'method': 'stage-permutation-v1',
        'seed': 0,
        'other': 1
    }, {
        'method': 'entity-trajectory-permutation-v1',
        'seed': 0
    }
])
def test_rejects_unpinned_or_wrong_recipes(controls, helpers, tmp_path, recipe):
    files = write_rows(tmp_path / 'source', scalar_rows())
    with pytest.raises(ValueError, match='recipe'):
        controls.generate(files, 'regression', tmp_path / 'output', recipe,
                          helpers=helpers)


def test_recipe_validation_fails_before_any_rows_are_needed(controls):
    with pytest.raises(ValueError, match='recipe'):
        controls.validate_recipe(
            'regression', dict(method='entity-trajectory-permutation-v1',
                               seed=0))
    controls.validate_recipe('regression',
                             dict(method='stage-permutation-v1', seed=0))
    controls.validate_recipe(
        'static_link', dict(method='entity-trajectory-permutation-v1', seed=0))


@pytest.mark.parametrize(
    'broken', ['duplicate', 'empty-stage', 'null-anchor', 'short-forecast'])
@pytest.mark.parametrize('ordered', [False, True])
def test_rejects_invalid_current_rows(controls, helpers, tmp_path, broken,
                                      ordered):
    rows = scalar_rows()
    task = 'regression'
    if broken == 'duplicate':
        rows.append(dict(rows[0]))
    elif broken == 'empty-stage':
        rows = [row for row in rows if row['SPLIT'] != 1]
    elif broken == 'null-anchor':
        task = 'temporal_link'
        rows[0]['TIMESTAMP'] = None
    else:
        task = 'forecasting'
        for row in rows:
            row['future_labels'] = [1.] * 11
    with pytest.raises(ValueError):
        controls.generate(write_rows(tmp_path / 'source',
                                     rows), task, tmp_path / 'output',
                          dict(method='stage-permutation-v1', seed=0),
                          helpers=helpers, ordered=ordered)


def test_streams_small_batches_without_reading_whole_answer_tables(
        controls, tmp_path, monkeypatch):
    sorter = load('nightly_predictions')
    monkeypatch.setattr(sorter, '_BATCH_ROWS', 3)
    monkeypatch.setattr(sorter, '_SORT_ROWS', 6)
    monkeypatch.setattr(controls, 'BATCH_ROWS', 3)
    files = write_rows(tmp_path / 'source', scalar_rows(), 4)

    def reject_whole_table(*args, **kwargs):
        raise AssertionError('Control generation read a whole Parquet table')

    monkeypatch.setattr(pq, 'read_table', reject_whole_table)
    actual = controls.generate(
        files, 'regression', tmp_path / 'output',
        dict(method='stage-permutation-v1', seed=0),
        helpers=SimpleNamespace(ordered_batches=sorter.ordered_batches))
    assert len(read_rows(actual)) == 24
    assert max(pq.ParquetFile(path).metadata.num_rows for path in actual) <= 3
