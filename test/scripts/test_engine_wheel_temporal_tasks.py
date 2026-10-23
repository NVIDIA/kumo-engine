# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import runpy
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

WORKER = (Path(__file__).resolve().parents[2] /
          'packages/kumo-engine/test/wheel_temporal_tasks.py')


def worker():
    assert WORKER.is_file(), 'The temporal learning worker is missing'
    return runpy.run_path(str(WORKER))


@pytest.mark.parametrize('task',
                         ['forecasting', 'static_link', 'temporal_link'])
def test_reserved_answers_are_absent_from_observed_graph(task):
    build = worker()['make_fixture']
    for independent in (False, True):
        fixture = build(task, independent)
        assert fixture == build(task, independent)
        entity_count = 60 if task == 'forecasting' else 100
        assert fixture['entity_ids'] == list(range(entity_count))
        if task == 'static_link':
            observed = {(row['ENTITY_ID'], row['ITEM_ID'])
                        for row in fixture['events']}
            assert all((entity, item) not in observed
                       for entity, answers in enumerate(fixture['targets'])
                       for item, present in enumerate(answers) if present)
        else:
            assert max(row['DAY']
                       for row in fixture['events']) < fixture['anchor']
        assert np.asarray(
            fixture['targets']).shape == (entity_count,
                                          12 if task == 'forecasting' else 100)


def test_static_control_breaks_group_signal_without_changing_task_shape():
    build = worker()['make_fixture']
    learned, control = [
        build('static_link', independent) for independent in (False, True)
    ]
    assert learned['entity_ids'] == control['entity_ids']
    assert len(learned['events']) == len(control['events'])
    for entity, targets in enumerate(learned['targets']):
        assert sum(targets) == 3
        assert all(item // 10 == entity % 10
                   for item, present in enumerate(targets) if present)
    assert any(item // 10 != entity % 10
               for entity, targets in enumerate(control['targets'])
               for item, present in enumerate(targets) if present)


@pytest.mark.parametrize('seed', range(5))
def test_static_link_cohorts_cover_items_evenly(seed):
    fixture = worker()['make_fixture']('static_link', False, seed)
    observed = np.bincount(
        [row['ITEM_ID'] for row in fixture['events']],
        minlength=len(fixture['catalog']))
    reserved = np.asarray(fixture['targets']).sum(axis=0)
    assert observed.min() == observed.max() > 0
    assert reserved.min() == reserved.max() > 0


def test_temporal_recall_ignores_arbitrary_order_inside_a_learned_group():
    score = runpy.run_path(str(
        WORKER.with_name('wheel_task_checks.py')))['score']
    for seed in range(5):
        fixture = worker()['make_fixture']('temporal_link', False, seed)
        for group in range(10):
            counts = np.asarray(fixture['targets'])[group::10].sum(axis=0)
            assert counts.tolist() == [
                3 if item // 10 == group else 0 for item in range(100)
            ]
        for reverse in (False, True):
            predictions = np.zeros((len(fixture['entity_ids']), 100))
            for entity in fixture['entity_ids']:
                items = list(range(entity % 10 * 10, (entity % 10 + 1) * 10))
                if reverse:
                    items.reverse()
                predictions[entity, items] = range(10, 0, -1)
            assert score('recall@5', fixture['targets'],
                         predictions) == pytest.approx(.5)
            # One wrong recommendation in one complete interest cohort loses
            # three hits among 100 entities with three hidden answers each.
            predictions[::10, 10] = 20
            assert score('recall@5', fixture['targets'],
                         predictions) == pytest.approx(.49)


@pytest.mark.parametrize('task',
                         ['forecasting', 'static_link', 'temporal_link'])
def test_seeds_select_independent_matching_fixtures(task):
    build = worker()['make_fixture']
    one = build(task, True, seed=0)
    two = build(task, True, seed=1)
    assert one != two
    assert two == build(task, True, seed=1)
    if task == 'static_link':
        eligible = np.asarray(two['control_eligible'])
        assert eligible.dtype == np.bool_
        assert eligible.sum(axis=1).tolist() == [94] * len(two['entity_ids'])
        assert not (np.asarray(two['targets'], dtype=bool) & ~eligible).any()


def test_chronology_rejects_a_training_horizon_that_reaches_validation():
    validate = worker()['validate_chronology']
    stages = {
        'train': [dict(ENTITY1=0, DAY=12)],
        'val': [dict(ENTITY1=0, DAY=24)],
        'test': [dict(ENTITY1=0, DAY=36)]
    }
    assert validate(stages, horizon=12, anchor=48) == 'PASS'
    invalid = copy.deepcopy(stages)
    invalid['train'][0]['DAY'] = 13
    with pytest.raises(ValueError, match='overlap'):
        validate(invalid, horizon=12, anchor=48)
    invalid = copy.deepcopy(stages)
    invalid['test'][0]['DAY'] = 37
    with pytest.raises(ValueError, match='reserved'):
        validate(invalid, horizon=12, anchor=48)


def test_forecast_alignment_rejects_missing_and_duplicate_horizons():
    align = worker()['align_predictions']
    fixture = dict(entity_ids=[2], horizons=['2024-01-01', '2024-01-02'])
    frame = pd.DataFrame({
        'ENTITY': [2, 2],
        'TIMESTAMP': pd.to_datetime(fixture['horizons']),
        'TARGET_PRED': [3., 4.]
    })
    np.testing.assert_array_equal(align('forecasting', frame, fixture),
                                  [[3, 4]])
    for broken in (frame.iloc[:1], pd.concat([frame, frame.iloc[:1]])):
        with pytest.raises(ValueError, match='population|duplicate'):
            align('forecasting', broken, fixture)


def test_link_alignment_rejects_truncated_candidates_and_masks_known_links():
    align = worker()['align_predictions']
    fixture = dict(entity_ids=[2], catalog=[0, 1, 2], known={'2': [1]})
    frame = pd.DataFrame({
        'ENTITY': [2, 2, 2],
        'CLASS': [2, 1, 0],
        'SCORE': [3., 10., 2.]
    })
    values = align('static_link', frame, fixture)
    assert values.shape == (1, 3)
    assert values[0, 1] < min(values[0, 0], values[0, 2])
    with pytest.raises(ValueError, match='population'):
        align('static_link', frame.iloc[:2], fixture)


def test_nonfinite_prediction_is_an_execution_failure():
    align = worker()['align_predictions']
    fixture = dict(entity_ids=[2], catalog=[0, 1], known={})
    frame = pd.DataFrame({
        'ENTITY': [2, 2],
        'CLASS': [0, 1],
        'SCORE': [1., float('nan')]
    })
    with pytest.raises(ValueError, match='finite'):
        align('temporal_link', frame, fixture)


@pytest.mark.parametrize('task',
                         ['forecasting', 'static_link', 'temporal_link'])
def test_task_plan_excludes_binary_options_and_samples_all_neighbors(
        task, monkeypatch):
    monkeypatch.syspath_prepend(str(WORKER.parents[2] / 'kumo-api'))
    from kumoapi.model_plan import MissingType

    plan = worker()['make_plan'](task, 0)
    assert plan.optimization.majority_sampling_ratio == MissingType.VALUE
    assert plan.optimization.weight_mode == MissingType.VALUE
    assert plan.neighbor_sampling.num_neighbors[0].hop1.default == -1
    assert plan.neighbor_sampling.num_neighbors[0].hop2.default == -1


@pytest.mark.parametrize('task',
                         ['forecasting', 'static_link', 'temporal_link'])
def test_every_explicit_plan_field_supports_the_actual_task(task, monkeypatch):
    monkeypatch.syspath_prepend(str(WORKER.parents[2] / 'kumo-api'))
    from kumoapi.model_plan import MissingType
    from kumoapi.pquery import QueryType
    from kumoapi.task import TaskType

    module = worker()
    plan = module['make_plan'](task, 0)
    actual_task = TaskType(module['TASK_TYPES'][task])
    query_type = QueryType.STATIC if task == 'static_link' else QueryType.TEMPORAL
    unsupported = []
    for field in fields(plan):
        section_name = field.name
        section = getattr(plan, section_name)
        if not hasattr(section, 'items'):
            continue
        defaults = type(section)()
        for name, value, metadata in section.items():
            if value == MissingType.VALUE or value == getattr(defaults, name):
                continue
            if not section.is_valid_option(name, metadata, actual_task,
                                           query_type):
                unsupported.append(f'{section_name}.{name}')
    assert not unsupported, unsupported


def test_temporal_sampling_requires_seed_times_and_timestamp_cutoff():
    validate = worker()['validate_temporal_sampling']
    runtime = {
        'task': {
            'task_type': 'forecasting',
            'is_temporal': True
        },
        'table': {
            'columns': {
                'timestamp': 'TIMESTAMP'
            }
        }
    }
    trainer = {'sampler_kwargs': {'temporal_strategy': 'last'}}
    schema = {'node_types': {'events': {'time_column': 'TS'}}}
    assert validate('forecasting', runtime, trainer,
                    schema)['use_seed_timestamps'] is True
    for broken in (dict(runtime, task=dict(runtime['task'], is_temporal=False)),
                   dict(runtime, table={'columns': {
                       'timestamp': None
                   }})):
        with pytest.raises(ValueError, match='temporal'):
            validate('forecasting', broken, trainer, schema)
    with pytest.raises(ValueError, match='cutoff'):
        validate('forecasting', runtime,
                 {'sampler_kwargs': {
                     'temporal_strategy': 'uniform'
                 }}, schema)


def test_forecast_fixture_has_bounded_trends_and_a_constant_median_diagnostic():
    module = worker()
    fixture = module['make_fixture']('forecasting', False)
    first = [row['VALUE'] for row in fixture['events'] if row['ENTITY_ID'] == 0]
    assert first[:4] == [0., .25, .5, .75]
    assert first[46:51] == [11.5, 11.75, 12., 11.75, 11.5]
    assert fixture['targets'][0] == [step / 4 for step in range(12)]
    assert min(first) == 0 and max(first) == 12
    result = module['constant_median_diagnostic']([1, 2, 3], [[3, 5]])
    assert result == {'training_median': 2., 'training_median_mae': 2.}


def test_stage_identity_ignores_row_order_but_preserves_forecast_targets():
    canonical = worker()['canonical_stage']
    rows = [
        dict(ENTITY1=2, TIMESTAMP=pd.Timestamp('2024-01-01'),
             future_labels=[1., 2.]),
        dict(ENTITY1=1, TIMESTAMP=pd.Timestamp('2024-01-02'),
             future_labels=[3., 4.])
    ]
    assert canonical('forecasting', rows) == canonical('forecasting',
                                                       rows[::-1])
    for changed in (dict(rows[0], TIMESTAMP=pd.Timestamp('2024-01-02')),
                    dict(rows[0], future_labels=[2., 1.])):
        assert canonical('forecasting',
                         rows) != canonical('forecasting', [changed, rows[1]])


def test_link_stage_identity_normalizes_sets_without_losing_frequencies():
    canonical = worker()['canonical_stage']
    first = dict(ENTITY1=0, TARGET1=[2, 1], exclude_target_keys=[4, 3],
                 __kumo_internal_frequency_column__=[.5, .8])
    reordered = dict(ENTITY1=0, TARGET1=[1, 2], exclude_target_keys=[3, 4],
                     __kumo_internal_frequency_column__=[.8, .5])
    assert canonical('temporal_link',
                     [first]) == canonical('temporal_link', [reordered])
    changed = dict(reordered, __kumo_internal_frequency_column__=[.5, .8])
    assert canonical('temporal_link',
                     [first]) != canonical('temporal_link', [changed])


def test_stage_evidence_saves_actual_arrow_inputs_and_detects_changed_targets(
        tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    save = worker()['save_stages']
    staging = tmp_path / 'generated'
    staging.mkdir()
    actual = pa.table(
        dict(ENTITY1=[1, 0], TARGET1=[[3, 2], [4, 1]],
             exclude_target_keys=[[7, 6], [5, 8]]))
    for stage in ('train', 'val', 'test'):
        pq.write_table(actual, staging / f'{stage}_seeds')
    rows, hashes = save('static_link', staging, tmp_path / 'evidence')
    assert set(hashes) == {'train', 'val', 'test'}
    assert rows['train'] == actual.to_pylist()
    for stage in hashes:
        saved = pq.read_table(tmp_path / 'evidence' / 'stage-seeds' /
                              f'{stage}.parquet')
        assert saved.equals(actual)
    reordered = actual.take(pa.array([1, 0]))
    pq.write_table(reordered, staging / 'train_seeds')
    _, reordered_hashes = save('static_link', staging, tmp_path / 'reordered')
    assert hashes == reordered_hashes
    changed = actual.set_column(1, 'TARGET1', pa.array([[3, 9], [4, 1]]))
    pq.write_table(changed, staging / 'train_seeds')
    _, changed_hashes = save('static_link', staging, tmp_path / 'changed')
    assert hashes['train'] != changed_hashes['train']
    assert hashes['val'] == changed_hashes['val']


def test_forecast_history_uses_twelve_observed_days_in_requested_entity_order():
    history = worker()['forecast_history']
    fixture = dict(
        anchor=20, entity_ids=[1, 2], events=[
            dict(ENTITY_ID=entity, DAY=day, VALUE=100 * entity + day)
            for entity in (1, 2) for day in range(20)
        ])
    assert history(fixture,
                   [2, 1]) == [list(range(208, 220)),
                               list(range(108, 120))]
    for broken in (dict(fixture, events=fixture['events'][1:19] +
                        fixture['events'][20:]),
                   dict(fixture,
                        events=fixture['events'] + [fixture['events'][-1]]),
                   dict(
                       fixture, events=fixture['events'] +
                       [dict(ENTITY_ID=1, DAY=20, VALUE=999)])):
        with pytest.raises(ValueError, match='history'):
            history(broken, [1, 2])


def test_forecast_prediction_inputs_reject_missing_history_and_hidden_answers():
    module = worker()
    fixture = module['make_fixture']('forecasting', False)
    histories = module['forecast_history'](fixture, [0])
    row = dict(ENTITY1=0, TIMESTAMP=pd.Timestamp('2024-04-06'),
               autoregression_labels=histories[0])
    validate = module['validate_forecast_inputs']
    assert validate(fixture, [0], [row]) == 'PASS'
    for broken in (dict(row, autoregression_labels=histories[0][1:]),
                   dict(row, autoregression_labels=histories[0][::-1]),
                   dict(row, TIMESTAMP=pd.Timestamp('2024-04-07')),
                   dict(row, future_labels=fixture['targets'][0]),
                   dict(row, TARGET1=fixture['targets'][0][0])):
        with pytest.raises(ValueError, match='prediction inputs'):
            validate(fixture, [0], [broken])
    with pytest.raises(ValueError, match='prediction inputs'):
        validate(fixture, [0], [row, row])
