# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from kumo_core.io.json import from_json_dict, to_json_dict
from kumo_core.model_plan import TrainingTableGenerationPlan
from kumo_core.training.bounds import validate_time_bounds
from kumo_core.types.enums import TaskType
from kumoapi.model_plan import InferredType, RunMode
from kumoapi.model_plan import \
    TrainingTableGenerationPlan as PublicGenerationPlan
from kumoapi.pquery import QueryType
from pandas import DateOffset, Timestamp


@pytest.mark.parametrize('start,end,message', [
    ('invalid-date', None, 'valid timestamp'),
    (None, 'NaT', 'valid timestamp'),
    ('2024-01-02', '2024-01-02', 'smaller than'),
    ('2024-01-03', '2024-01-02', 'smaller than'),
    (-2, '2024-01-02', 'Integer'),
    (1, None, 'Integer'),
    (None, 1, 'Integer'),
    (-1, -2, 'Integer'),
    (None, '3000-01-01', 'valid timestamp'),
])
def test_invalid_bounds(start, end, message):
    with pytest.raises(ValueError, match=message):
        validate_time_bounds(start, end)


def test_bounds_normalize_utc():
    assert validate_time_bounds(
        '2024-01-01T00:00:00-06:00',
        '2024-01-01T08:00:00+01:00',
    ) == (Timestamp('2024-01-01T06:00:00'),
          Timestamp('2024-01-01T07:00:00'))
    assert validate_time_bounds('-INF', '2024-01-02') == (
        None, Timestamp('2024-01-02'))
    offset = DateOffset(days=-1)
    assert validate_time_bounds(None, offset) == (None, offset)


@pytest.mark.parametrize('split',
                         ['BacktestingSplit', 'DateOffsetSplit',
                          'TimeRangeSplit'])
def test_absolute_split_conflict(split):
    with pytest.raises(ValueError, match='cannot be combined'):
        validate_time_bounds(None, '2024-01-02', split_family=split)


@pytest.mark.parametrize('split',
                         ['ApproxDateOffsetSplit', 'RandomSplit',
                          'TemporalSplit'])
def test_absolute_ratio_split(split):
    assert validate_time_bounds('2024-01-01', None, split_family=split) == (
        Timestamp('2024-01-01'), None)


@pytest.mark.parametrize('flag', ['false', 'true'])
@pytest.mark.parametrize('task', [
    TaskType.BINARY_CLASSIFICATION, TaskType.TEMPORAL_LINK_PREDICTION,
])
def test_integer_bounds_rejected(flag, task, monkeypatch):
    monkeypatch.setenv('USE_DECOUPLED_SPLITS', flag)
    errors, _ = TrainingTableGenerationPlan(start_time=-10, end_time=0).validate(
        task_type=task,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    assert any("Integer 'start_time' bounds are not supported" in str(error)
               for error in errors)


@pytest.mark.parametrize('task_type', TaskType.get_node_pred_tasks())
@pytest.mark.parametrize('flag', [None, 'false', 'true'])
def test_node_bound_validation(task_type, flag, monkeypatch):
    if flag is None:
        monkeypatch.delenv('USE_DECOUPLED_SPLITS', raising=False)
    else:
        monkeypatch.setenv('USE_DECOUPLED_SPLITS', flag)
    plan = TrainingTableGenerationPlan(
        start_time='2024-01-01T00:00:00-06:00',
        end_time='2024-01-02T00:00:00-06:00',
    )
    errors, _ = plan.validate(
        task_type=task_type,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    assert not errors
    assert plan.start_time == '2024-01-01T00:00:00-06:00'


@pytest.mark.parametrize('flag', [None, 'false', 'true'])
@pytest.mark.parametrize('inferred', ['inferred', InferredType.VALUE])
@pytest.mark.parametrize('bound', ['start_time', 'end_time'])
def test_public_inferred_bound(flag, inferred, bound, monkeypatch):
    if flag is None:
        monkeypatch.delenv('USE_DECOUPLED_SPLITS', raising=False)
    else:
        monkeypatch.setenv('USE_DECOUPLED_SPLITS', flag)
    public = PublicGenerationPlan(**{
        'start_time': '2024-01-01',
        'end_time': '2024-02-01',
        bound: inferred,
    })
    # Match generate_training_table's public-plan preflight conversion.
    plan = from_json_dict(to_json_dict(public), TrainingTableGenerationPlan)
    errors, _ = plan.validate(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    assert not errors


def test_link_bounds_keep_gate(monkeypatch):
    monkeypatch.delenv('USE_DECOUPLED_SPLITS', raising=False)
    errors, _ = TrainingTableGenerationPlan(start_time='2024-01-01').validate(
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    assert any("'start_time' field is not supported" in str(error)
               for error in errors)


@pytest.mark.parametrize('options,message', [
    ({'start_time': 'NaT'}, 'valid timestamp'),
    ({'start_time': '2024-01-01', 'train_start_offset': 2},
     'Cannot specify both'),
    ({'start_time': '2024-01-01',
      'split': "TimeRangeSplit([('2024-01-01', '2024-01-02')])"},
     'cannot be combined'),
])
def test_plan_rejects_invalid_bounds(options, message, monkeypatch):
    monkeypatch.setenv('USE_DECOUPLED_SPLITS', 'true')
    errors, _ = TrainingTableGenerationPlan(**options).validate(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    assert any(message in str(error) for error in errors)
