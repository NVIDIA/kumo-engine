# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pandas as pd
import pytest
from kumo_core.types import TaskType
from kumoapi.typing import TimeUnit

from kumo.pquery import PredictiveQueryV2
from kumo.pquery.pquery_validators import PreTrainTableValidator
from kumo.task.split import (
    BacktestingSplit,
    DateOffsetSplit,
    RandomSplit,
    TimeRangeSplit,
)
from kumo.task.splits.time_split import TimeSplit
from kumo.testing.pquery import get_validated_query


def test_forecasting_task_type_requires_valid_lag_timesteps(
        create_node_pred_graph, tempdir):
    query = 'PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 20) FOR EACH USERS.USER_ID'
    graph = create_node_pred_graph('test_graph', tempdir.name)
    validated_query = get_validated_query(graph, query)
    validator = PreTrainTableValidator(
        RandomSplit([0.8, 0.1, 0.1]),
        validated_query,
        graph,
        task_type=TaskType.FORECASTING,
    )

    response = validator.validate_forecasting_task_type(lag_timesteps=0,
                                                        forecast_length=1)

    assert not response.ok
    assert len(response.errors) == 1
    assert response.errors[0].title == 'Invalid lag timesteps'
    assert ('Lag timesteps are required for forecasting tasks.'
            in response.errors[0].message)


def test_validate_temporal_splits_for_a_range_smaller_than_aggr(
        create_node_pred_graph, tempdir):
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 20) FOR EACH USERS.USER_ID')
    graph = create_node_pred_graph('test_graph', tempdir.name)
    validated_query = get_validated_query(graph, query)

    validator = PreTrainTableValidator(
        TimeRangeSplit(ranges=[
            ('01-01-2024', '03-31-2024'),
            ('04-01-2024', '04-30-2024'),
            ('05-01-2024', '05-15-2024'),
        ]),
        validated_query,
        graph,
        timeframe_step=pd.DateOffset(days=2),
        train_start_offset=None,
        train_end_offset=None,
    )

    response = validator.validate_temporal_splits()

    assert response is not None
    assert response.errors is not None
    assert len(response.errors) == 1
    error = response.errors[0]
    assert error.title.startswith('Incorrect TimeRangeSplit')
    assert error.message.startswith('The range')
    assert 'is smaller than' in error.message
    assert '2024-05-01' in error.message
    assert '2024-05-15' in error.message


@pytest.mark.parametrize(
    "unit,window,start,end",
    [
        (TimeUnit.MINUTES, 30, "2024-01-01 00:00:00", "2024-01-01 00:45:00"),
        (TimeUnit.HOURS, 2, "2024-01-01 00:00:00", "2024-01-01 03:00:00"),
    ],
)
def test_validate_temporal_splits_uses_complete_subday_duration(
    create_node_pred_graph,
    tempdir,
    unit,
    window,
    start,
    end,
):
    graph = create_node_pred_graph('test_graph', tempdir.name)
    validated_query = get_validated_query(
        graph,
        'PREDICT SUM(TRANSACTIONS.AMOUNT, 0, '
        f'{window}, {unit.value}) FOR EACH USERS.USER_ID',
    )
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    duration = end_timestamp - start_timestamp
    validator = PreTrainTableValidator(
        TimeRangeSplit(
            ranges=[
                (start_timestamp, end_timestamp),
                (end_timestamp, end_timestamp + duration),
            ],
            unit=unit,
        ),
        validated_query,
        graph,
        timeframe_step=pd.DateOffset(minutes=1),
        train_start_offset=None,
        train_end_offset=None,
    )

    response = validator.validate_temporal_splits()

    assert response.ok, response.errors


@pytest.mark.parametrize("query,split_unit,offsets,expected_timeframe", [
    ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 20) FOR EACH USERS.USER_ID',
     TimeUnit.DAYS, [-10, 0],
     '10 days is smaller than the target timeframe: 20 days'),
    ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 20, hours) FOR EACH USERS.USER_ID',
     TimeUnit.HOURS, [-10, 0],
     '10 hours is smaller than the target timeframe: 20 hours'),
    ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 20, minutes) FOR EACH USERS.USER_ID',
     TimeUnit.MINUTES, [-10, 0],
     '10 minutes is smaller than the target timeframe: 20 minutes'),
    ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 20) FOR EACH USERS.USER_ID',
     TimeUnit.HOURS, [-(10 * 24), 0],
     '240 hours is smaller than the target timeframe: 20 days'),
    ('PREDICT SUM(TRANSACTIONS.AMOUNT, 30, 60) FOR EACH USERS.USER_ID',
     TimeUnit.DAYS, [-90, -60],
     '30 days is smaller than the target timeframe: 60 days'),
])
def test_validate_date_offset_val_split_too_small(
    create_node_pred_graph,
    tempdir,
    query: str,
    split_unit: TimeUnit,
    offsets: List[int],
    expected_timeframe: str,
):
    graph = create_node_pred_graph('test_graph', tempdir.name)
    validated_query = get_validated_query(graph, query)

    validator = PreTrainTableValidator(
        DateOffsetSplit(offsets, offset_unit=split_unit),
        validated_query,
        graph,
    )

    response = validator.validate_temporal_splits()

    assert response is not None
    assert response.errors is not None
    assert len(response.errors) == 1
    error = response.errors[0]
    assert error.title.startswith('Split size is too small')
    assert error.message.startswith(
        'The size of the validation split is too small:')
    assert expected_timeframe in error.message


def test_validate_temporal_splits_for_a_range_smaller_than_aggr_months(
        create_node_pred_graph, tempdir):
    query = (
        'PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 2, months) FOR EACH USERS.USER_ID')
    graph = create_node_pred_graph('test_graph', tempdir.name)
    validated_query = get_validated_query(graph, query)

    validator = PreTrainTableValidator(
        TimeRangeSplit(
            ranges=[
                ('01-2024', '04-2024'),
                ('05-2024', '07-2024'),
                ('08-2024', '09-2024'),
            ],
            unit=TimeUnit.MONTHS,
        ),
        validated_query,
        graph,
        timeframe_step=pd.DateOffset(days=2),
        train_start_offset=None,
        train_end_offset=None,
    )

    response = validator.validate_temporal_splits()

    assert response is not None
    assert response.errors is not None
    assert len(response.errors) == 1
    error = response.errors[0]
    assert error.title.startswith('Incorrect TimeRangeSplit')
    assert error.message.startswith('The range')
    assert 'is smaller than' in error.message
    assert '2024-08' in error.message
    assert '2024-09' in error.message


def test_validate_temporal_splits_for_different_unit_in_time_range_split(
        create_node_pred_graph, tempdir):
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 20, HOURS) '
             'FOR EACH USERS.USER_ID')
    graph = create_node_pred_graph('test_graph', tempdir.name)
    validated_query = get_validated_query(graph, query)

    validator = PreTrainTableValidator(
        TimeRangeSplit(ranges=[
            ('01-01-2024', '03-31-2024'),
            ('04-01-2024', '04-30-2024'),
            ('05-01-2024', '05-15-2024'),
        ]),
        validated_query,
        graph,
        timeframe_step=pd.DateOffset(days=2),
        train_start_offset=None,
        train_end_offset=None,
    )

    response = validator.validate_temporal_splits()

    assert response is not None
    assert response.errors is not None
    assert len(response.errors) == 1
    error = response.errors[0]
    assert error.title.startswith('Time unit mismatch error')
    assert error.message.startswith('Smallest time')
    assert 'hours' in error.message

    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 1320, MINUTES) '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)

    validator = PreTrainTableValidator(
        TimeRangeSplit(ranges=[
            ('01-01-2024', '03-31-2024'),
            ('04-01-2024', '04-30-2024'),
            ('05-01-2024', '05-15-2024'),
        ]),
        validated_query,
        graph,
        timeframe_step=pd.DateOffset(days=2),
        train_start_offset=None,
        train_end_offset=None,
    )

    response = validator.validate_temporal_splits()

    assert response is not None
    assert response.errors is not None
    assert len(response.errors) == 1
    error = response.errors[0]
    assert error.title.startswith('Time unit mismatch error')
    assert error.message.startswith('Smallest time')
    assert 'minutes' in error.message


def test_validate_timestamps_one(create_node_pred_graph, tempdir):
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 2) FOR EACH USERS.USER_ID '
             'WHERE COUNT(TRANSACTIONS.*, -10, 0) > 0')
    graph = create_node_pred_graph('test_graph', tempdir.name)
    validated_query = get_validated_query(graph, query)
    split = DateOffsetSplit([-2, 0], unit=TimeUnit.DAYS, timeframe=2)
    validator = PreTrainTableValidator(
        split, validated_query, graph, timeframe_step=pd.DateOffset(days=2),
        train_start_offset=pd.DateOffset(days=-30),
        train_end_offset=pd.DateOffset(days=-20))
    timestamps = [pd.Timestamp('2020-01-05')]

    response = validator.validate_timestamps(
        timestamps,  # type: ignore
        pd.Timestamp('2020-01-01'),
        pd.Timestamp('2020-02-02'))
    message = response.message()
    assert "target table TRANSACTIONS contains 32 days" in message
    assert "Try increasing 'train_start_offset'" in message
    assert "Try decreasing 'train_end_offset'" in message
    assert "Decrease 'timeframe_step'" in message
    assert "Use '-INF' as the start time" in message
    assert "Only one timeframe remains" in message


def test_validate_timestamps_zero(create_node_pred_graph, tempdir):
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 2) FOR EACH USERS.USER_ID '
             'WHERE COUNT(TRANSACTIONS.*, -10, 0) > 0')
    graph = create_node_pred_graph('test_graph', tempdir.name)
    validated_query = get_validated_query(graph, query)
    split = DateOffsetSplit([-2, 0], unit=TimeUnit.DAYS, timeframe=2)
    validator = PreTrainTableValidator(split, validated_query, graph)
    timestamps = []
    response = validator.validate_timestamps(timestamps, None, None)
    assert response.errors[0].title == 'Empty training table'


def test_validate_timestamps_too_many(create_node_pred_graph, tempdir):
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 2) FOR EACH USERS.USER_ID '
             'WHERE COUNT(TRANSACTIONS.*, -10, 0) > 0')
    graph = create_node_pred_graph('test_graph', tempdir.name)
    validated_query = get_validated_query(graph, query)
    split = DateOffsetSplit([-2, 0], unit=TimeUnit.DAYS, timeframe=2)
    validator = PreTrainTableValidator(
        split, validated_query, graph, timeframe_step=pd.DateOffset(days=2),
        train_start_offset=pd.DateOffset(days=-30),
        train_end_offset=pd.DateOffset(days=-20))
    timestamps = [pd.Timestamp('2020-01-05')] * 201

    response = validator.validate_timestamps(
        timestamps,  # type: ignore
        pd.Timestamp('2020-01-01'),
        pd.Timestamp('2020-02-02'))
    message = response.message()
    assert "target table TRANSACTIONS contains 32 days" not in message
    assert "Try increasing 'train_start_offset'" not in message
    assert "Try decreasing 'train_end_offset'" not in message
    assert "Decrease 'timeframe_step'" not in message
    assert "Use '-INF' as the start time" not in message
    assert "Large number of time intervals " in message


def test_validate_timestamps_too_many_error(create_node_pred_graph, tempdir):
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 2) FOR EACH USERS.USER_ID '
             'WHERE COUNT(TRANSACTIONS.*, -10, 0) > 0')
    graph = create_node_pred_graph('test_graph', tempdir.name)
    validated_query = get_validated_query(graph, query)
    split = DateOffsetSplit([-2, 0], unit=TimeUnit.DAYS, timeframe=2)
    validator = PreTrainTableValidator(
        split, validated_query, graph, timeframe_step=pd.DateOffset(days=2),
        train_start_offset=pd.DateOffset(days=-30),
        train_end_offset=pd.DateOffset(days=-20))
    timestamps = [pd.Timestamp('2020-01-05')] * 10000

    response = validator.validate_timestamps(
        timestamps,  # type: ignore
        pd.Timestamp('2020-01-01'),
        pd.Timestamp('2020-02-02'))
    message = response.message()
    assert not response.ok
    assert "Large number of time intervals" in message


def test_validate_timestamps_backtesting_large_date(create_node_pred_graph,
                                                    tempdir):
    query = 'PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 2) FOR EACH USERS.USER_ID'
    graph = create_node_pred_graph('test_graph', tempdir.name)
    validated_query = get_validated_query(graph, query)
    split = BacktestingSplit('2020-03-03')
    validator = PreTrainTableValidator(
        split, validated_query, graph, timeframe_step=pd.DateOffset(days=2),
        train_start_offset=pd.DateOffset(days=-30),
        train_end_offset=pd.DateOffset(days=-20))
    timestamps = [pd.Timestamp('2020-03-03')]

    response = validator.validate_timestamps(
        timestamps,  # type: ignore
        pd.Timestamp('2020-01-01'),
        pd.Timestamp('2020-02-02'))
    assert not response.ok
    assert len(response.errors) == 1
    assert 'timestamp later than the data' in response.errors[0].title


def test_validate_timestamps_backtesting_small_date(create_node_pred_graph,
                                                    tempdir):
    query = 'PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 2) FOR EACH USERS.USER_ID'
    graph = create_node_pred_graph('test_graph', tempdir.name)
    validated_query = get_validated_query(graph, query)
    split = BacktestingSplit('2019-03-03')
    validator = PreTrainTableValidator(
        split, validated_query, graph, timeframe_step=pd.DateOffset(days=2),
        train_start_offset=pd.DateOffset(days=-30),
        train_end_offset=pd.DateOffset(days=-20))
    timestamps = [pd.Timestamp('2019-03-03')]

    response = validator.validate_timestamps(
        timestamps,  # type: ignore
        pd.Timestamp('2020-01-01'),
        pd.Timestamp('2020-02-02'))
    message = response.message()
    assert "is earlier than the earliest timestamp" in message


@pytest.mark.parametrize('split', [
    DateOffsetSplit([0], unit='days'),
    TimeRangeSplit([
        ('1989-12-29', '1990-01-01'),
        ('1990-01-01', '1990-01-03'),
        ('1990-01-03', '1990-01-05'),
    ]),
])
def test_validate_num_split_timeframe(
    create_node_pred_graph,
    tempdir,
    split,
):
    graph = create_node_pred_graph('node_pred_graph', tempdir.name)
    query = 'PREDICT SUM(TRANSACTIONS.AMOUNT, 1, 2) FOR EACH USERS.USER_ID'
    validated_query = get_validated_query(graph, query)
    validator = PreTrainTableValidator(
        split, validated_query, graph, timeframe_step=pd.DateOffset(days=2),
        train_start_offset=pd.DateOffset(days=-30),
        train_end_offset=pd.DateOffset(days=-20))
    response = validator.validate_timestamps(
        timestamps=[pd.Timestamp('1990-01-03 00:00:00')],
        min_date=pd.Timestamp('1990-01-04 00:00:00'),
        max_date=pd.Timestamp('1990-01-05 00:00:00'),
    )
    assert ("The number of generated timeframes is too small"
            in response.errors[0].title)


pquery_targets = [
    'SUM(actions.amount, 0, 2)',
    'AVG(actions.amount, 0, 5)',
    'COUNT(actions.*, 2, 5)',
    'SUM(actions.amount, 0, 3)',
    'SUM(actions.amount, 0, 1)',
    'SUM(actions.amount, 0, 8, hours)',
    'SUM(actions.amount, 0, 560, minutes)',
]
pquery_entities = ['entity.entity_id']


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(lag_timesteps=None, pquery_target=pquery_targets[1],
             pquery_entity=pquery_entities[0],
             error_message='Lag timesteps are required for forecasting tasks.'),
        dict(lag_timesteps=4, pquery_target=pquery_targets[1],
             pquery_entity=pquery_entities[0],
             error_message="Invalid aggregation type for forecasting task"),
        dict(lag_timesteps=4, pquery_target=pquery_targets[2],
             pquery_entity=pquery_entities[0],
             error_message="Start offset for aggregationshould be 0"),
        dict(lag_timesteps=1, pquery_target=pquery_targets[0],
             pquery_entity=pquery_entities[0], forecast_length=2,
             error_message=None),
        # Test entities with end time column.
        dict(lag_timesteps=1, pquery_target=pquery_targets[0],
             pquery_entity=pquery_entities[0], forecast_length=2,
             add_entity_end_time=True, error_message=None),
        # Test default split for forecasting.
        dict(lag_timesteps=1, pquery_target=pquery_targets[0],
             pquery_entity=pquery_entities[0], forecast_length=2,
             error_message=None, split=None),
        # Test dateoffset split with hour level aggregation.
        dict(lag_timesteps=1, pquery_target=pquery_targets[5],
             pquery_entity=pquery_entities[0], forecast_length=2,
             error_message=None),
        # Test dateoffset split with minute level aggregation.
        dict(lag_timesteps=1, pquery_target=pquery_targets[6],
             pquery_entity=pquery_entities[0], forecast_length=2,
             error_message=None),
        # Test dateoffset split with hour level aggregation.
        # But the split has too few timeframes.
        dict(lag_timesteps=1, pquery_target=pquery_targets[5],
             pquery_entity=pquery_entities[0], forecast_length=2,
             split=DateOffsetSplit([-16, -8], unit='hours'),
             error_message="has too few timeframes"),
        # Test dateoffset split with minute level aggregation.
        # But the split has too few timeframes.
        dict(lag_timesteps=1, pquery_target=pquery_targets[6],
             pquery_entity=pquery_entities[0], forecast_length=2,
             split=DateOffsetSplit([-1120, -560], unit='minutes'),
             error_message="has too few timeframes"),
        # Test timerange split for forecasting.
        dict(
            lag_timesteps=1,
            pquery_target=pquery_targets[0],
            pquery_entity=pquery_entities[0],
            forecast_length=2,
            split=TimeRangeSplit([
                ('2022-01-15', '2022-01-29'),
                ('2022-01-29', '2022-02-03'),
                ('2022-02-03', '2022-02-08'),
            ]),
            error_message="Timestamps are not uniformly spaced",
        ),
        # Test timerange split where train set is too small.
        dict(
            lag_timesteps=1,
            pquery_target=pquery_targets[0],
            pquery_entity=pquery_entities[0],
            forecast_length=2,
            error_message=("not enough timestamps to generate forecasting "
                           "data in train split"),
            split=TimeRangeSplit([
                ('2022-01-15', '2022-01-20'),
                ('2022-01-20', '2022-02-09'),
            ]),
        ),
        # Test disallowed split for forecasting.
        dict(
            lag_timesteps=1,
            pquery_target=pquery_targets[0],
            pquery_entity=pquery_entities[0],
            forecast_length=2,
            error_message=(
                "only supported with DateOffsetSplit and TimeRangeSplit"),
            split=RandomSplit([0.5, 0.25, 0.25]),
        ),
        dict(
            lag_timesteps=1, pquery_target=pquery_targets[3],
            pquery_entity=pquery_entities[0], forecast_length=2,
            error_message=("At least 7 timestamps "
                           "are required for forecasting (got 5)")),
        dict(
            lag_timesteps=4,
            pquery_target=pquery_targets[0],
            pquery_entity=pquery_entities[0],
            error_message=("timeframe_step(<DateOffset: days=1>) "
                           "should be equal to"),
            timeframe_step=1,
        ),
    ])
def test_validations_for_forecasting(create_tiny_node_pred_graph, kwargs,
                                     tempdir):
    query = (f"PREDICT {kwargs['pquery_target']} "
             f"FOR EACH {kwargs['pquery_entity']} ")
    graph = create_tiny_node_pred_graph(
        'test_graph', tempdir.name, kwargs.get('add_entity_end_time', False))
    validated_query = get_validated_query(graph, query)

    table_generator = PredictiveQueryV2(
        name='temporal_link_pred_empty_table_v2',
        graph=graph,
        validated_predictive_query=validated_query,
        forecast_length=kwargs.get('forecast_length', 2),
        split=kwargs.get('split', None),
        lag_timesteps=kwargs['lag_timesteps'],
        timeframe_step=kwargs.get('timeframe_step', None),
    ).table_generator

    response = table_generator.validate_pre_train_table_generate(
        skip_timestamp_validation=False)

    if kwargs.get('error_message') is None:
        assert response.ok
    else:
        assert not response.ok
        assert any(kwargs['error_message'] in error.message
                   or kwargs['error_message'] in error.title
                   for error in response.errors)


@pytest.mark.parametrize(
    "kwargs",
    [
        # forecast_length > 1 is treated as forecasting and requires
        # forecasting-specific validation.
        dict(task_type=None, lag_timesteps=None, forecast_length=3,
             pquery_target=pquery_targets[0], pquery_entity=pquery_entities[0],
             error_message='Lag timesteps are required for forecasting tasks'),
        # forecast_length > 1 with valid lag_timesteps should pass.
        dict(task_type=None, lag_timesteps=1, forecast_length=2,
             pquery_target=pquery_targets[0], pquery_entity=pquery_entities[0],
             error_message=None),
        # Normal regression (forecast_length=1) with lag_timesteps=0
        # should pass (no forecasting validation).
        dict(task_type=None, lag_timesteps=0, forecast_length=1,
             pquery_target=pquery_targets[0], pquery_entity=pquery_entities[0],
             error_message=None),
        # Normal regression (forecast_length=None) with lag_timesteps > 0
        # should error (lag_timesteps not supported for regression).
        dict(
            task_type=None, lag_timesteps=5, forecast_length=None,
            pquery_target=pquery_targets[0], pquery_entity=pquery_entities[0],
            error_message="Lag timesteps are only supported for "
            "forecasting tasks"),
    ])
def test_forecast_length_affects_lag_timesteps_validation(
        create_tiny_node_pred_graph, kwargs, tempdir):
    """Test that forecast_length parameter controls whether lag_timesteps
    validation rejects forecast_length > 1 before lag-specific checks."""
    query = (f"PREDICT {kwargs['pquery_target']} "
             f"FOR EACH {kwargs['pquery_entity']} ")
    graph = create_tiny_node_pred_graph('test_graph', tempdir.name)
    validated_query = get_validated_query(graph, query)

    table_generator = PredictiveQueryV2(
        name='forecast_length_validation_test',
        graph=graph,
        validated_predictive_query=validated_query,
        forecast_length=kwargs.get('forecast_length', 2),
        split=kwargs.get('split', None),
        lag_timesteps=kwargs['lag_timesteps'],
    ).table_generator

    response = table_generator.validate_pre_train_table_generate(
        skip_timestamp_validation=False)
    if kwargs.get('error_message') is None:
        assert response.ok
    else:
        assert any(kwargs['error_message'] in error.message
                   for error in response.errors)


def test_validate_timestamps_for_forecasting_v2(create_node_pred_graph,
                                                tempdir):
    """Test that validate_timestamps_for_forecasting_v2 correctly validates
    timestamps for forecasting tasks with various scenarios.
    """
    query = (
        'PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 2, hours) FOR EACH USERS.USER_ID')
    graph = create_node_pred_graph('test_graph', tempdir.name)
    validated_query = get_validated_query(graph, query)

    # Use TimeSplit with hourly unit for testing with finer granularity
    split = TimeSplit(
        # offset_boundaries=[(-48, -24), (-24, -12), (-12, 0)],
        timestamp_boundaries=[
            ('2020-01-01 00:00', '2020-01-01 10:00'),
            ('2020-01-01 10:00', '2020-01-01 16:00'),
            ('2020-01-01 16:00', '2020-01-01 22:00'),
        ],
        unit=TimeUnit.HOURS,
        timeframe_size=2,
        anchor_start_offset=0,
        strict_split_boundaries=False,
    )

    validator = PreTrainTableValidator(
        split,
        validated_query,
        graph,
        timeframe_step=pd.DateOffset(hours=4),
        task_type=TaskType.FORECASTING,
    )

    # Test case 1: Not enough timestamps (less than minimum required)
    timestamps = [
        pd.Timestamp('2020-01-01 00:00'),
        pd.Timestamp('2020-01-01 04:00'),
        pd.Timestamp('2020-01-01 08:00'),  # Only 3 timestamps
    ]

    # Set forecast_length and lag_timesteps as the 2nd and 3rd parameters
    response = validator.validate_timestamps_for_forecasting_v2(
        timestamps, forecast_length=3, lag_timesteps=2)  # type: ignore

    assert not response.ok
    assert any("Not enough timestamps for forecasting" in error.title
               for error in response.errors)

    # Test case 2: Valid timestamps configuration
    timestamps = [
        pd.Timestamp('2020-01-01 00:00') +
        pd.DateOffset(hours=i * 2)  # type: ignore
        for i in range(11)
    ]  # 10 evenly spaced timestamps

    response = validator.validate_timestamps_for_forecasting_v2(
        timestamps, forecast_length=3, lag_timesteps=2)  # type: ignore

    assert response.ok

    # Test case 3: Enough timestamps in the dataset but not enough in a
    # specific split
    timestamps = [
        pd.Timestamp('2020-01-01 00:00') +
        pd.DateOffset(hours=i * 4)  # type: ignore
        for i in range(11)
    ]  # 10 evenly spaced timestamps

    response = validator.validate_timestamps_for_forecasting_v2(
        timestamps, forecast_length=3, lag_timesteps=2)  # type: ignore

    assert not response.ok
    assert any("The split train has too few timeframes for forecasting." in
               error.message for error in response.errors)


def test_large_date_offset_handling(create_node_pred_graph, tempdir):
    """Test that large date offsets (200+ years) don't cause OutOfBoundsDatetime.

    Regression test: validates the fix uses epoch (1970-01-01) instead of
    pd.Timestamp.now() as reference, preventing overflow with large offsets.
    """
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 20) FOR EACH USERS.USER_ID')
    graph = create_node_pred_graph('test_graph', tempdir.name)
    validated_query = get_validated_query(graph, query)

    # Create split with large offsets that would overflow with now()
    validator = PreTrainTableValidator(
        DateOffsetSplit(
            [-73000, -72980],  # ~200 year offset
            offset_unit=TimeUnit.DAYS,
        ),
        validated_query,
        graph,
    )

    # Should not raise OutOfBoundsDatetime
    response = validator.validate_temporal_splits()
    assert response is not None
