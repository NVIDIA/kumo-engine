# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
from test.fixtures.dg_indexed_graph import (
    _ttng_stage_masks_from_bundle,
    materialize_task_dg,
)
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.types import ColStatType, TaskType
from kumoapi.pquery import QueryType
from kumoapi.train import TrainingStage
from kumoapi.typing import TimeUnit

from kumo.pquery.predictive_query import PQuerySettings, PredictiveQueryV2
from kumo.pquery.stats_manager import StatsManager
from kumo.pquery.train_table_generator import TrainingTableGeneratorV2
from kumo.pquery.utils import get_max_train_rows
from kumo.stats.stats import Stats
from kumo.stats.train_table_stats import STATS_PER_TIMEFRAME
from kumo.task.split import (
    BacktestingSplit,
    DateOffsetSplit,
    RandomSplit,
    TimeRangeSplit,
)
from kumo.task.splits.time_split import TimeSplit
from kumo.testing.pquery import get_validated_query
from kumo.validation import ValidationException

_SKIP_REASON_FINANCIAL_FULL_SCALE_LITERALS = (
    "blocked-fixture: the ops are supported on DG, but these rows assert "
    "hard-coded full-scale literals from the full 1993 FINANCIAL dataset that "
    "the unit-scale dg_indexed_* fixtures cannot reproduce; op-level coverage "
    "is retained by the migrated test_temporal_node_regression_stats sibling.")

NANOSECONDS_PER_SECOND = 1_000_000_000


def test_generate_time_intervals_single_aggr(create_node_pred_graph, tempdir):
    graph = create_node_pred_graph('test_graph', tempdir.name)
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 2) '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)

    split = DateOffsetSplit([-2, 0], unit=TimeUnit.DAYS, timeframe=2)
    generator = TrainingTableGeneratorV2(graph, validated_query, split)
    task_type = generator.task_type

    timestamps = generator.generate_timestamps()
    # TRANSACTIONS.DATE contains dates between 1990-01-01 and 1990-01-05
    # This gives two valid timestamps, 1990-01-01 and 1990-01-03, the earlier
    # dropped to avoid an empty input
    assert timestamps == [pd.Timestamp('1990-01-03 00:00:00')]
    assert task_type == TaskType.REGRESSION


def test_generate_time_intervals_nonzero_offset(create_node_pred_graph,
                                                tempdir):
    graph = create_node_pred_graph('test_graph', tempdir.name)
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 12, 24, hours) '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)

    split = DateOffsetSplit([-24, 0], unit=TimeUnit.HOURS, timeframe=12)
    generator = TrainingTableGeneratorV2(graph, validated_query, split)
    task_type = generator.task_type

    timestamps = generator.generate_timestamps()
    # TRANSACTIONS.DATE contains dates between 1990-01-01 and 1990-01-05
    # This gives five valid timestamps:
    # 1990-01-01:00, 1990-01-01:12, 1990-01-02:00, (all train)
    # 1990-01-03:00, (val) and 1990-01-04:00 (test)
    # the earliest dropped to avoid an empty input
    assert timestamps == [
        pd.Timestamp('1990-01-04 00:00:00'),
        pd.Timestamp('1990-01-03 00:00:00'),
        pd.Timestamp('1990-01-02 00:00:00'),
        pd.Timestamp('1990-01-01 12:00:00')
    ]
    assert task_type == TaskType.REGRESSION


def test_generate_time_intervals_multi_aggr(create_node_pred_graph, tempdir):
    graph = create_node_pred_graph('test_graph', tempdir.name)
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 2) > 30 '
             'OR COUNT(ORDERS.*, 0, 1) > 0'
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)

    split = DateOffsetSplit([-2, 0], unit=TimeUnit.DAYS, timeframe=2)
    generator = TrainingTableGeneratorV2(graph, validated_query, split)
    task_type = generator.task_type

    timestamps = generator.generate_timestamps()
    # TRANSACTIONS.DATE contains dates between 1990-01-01 and 1990-01-05
    # ORDERS.DATE contains dates between 1990-01-02 and 1990-01-04
    # The only valid date generated is 1990-01-03, not dropped
    assert timestamps == [pd.Timestamp('1990-01-03 00:00:00')]
    assert task_type == TaskType.BINARY_CLASSIFICATION


def test_generate_time_intervals_entity_filter(create_node_pred_graph, tempdir):
    graph = create_node_pred_graph('test_graph', tempdir.name)
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 2) > 30 '
             'OR COUNT(ORDERS.*, 0, 1) > 0'
             'FOR EACH USERS.USER_ID WHERE COUNT(TRANSACTIONS.*, -2, 0) > 0')
    validated_query = get_validated_query(graph, query)

    split = DateOffsetSplit([-2, 0], unit=TimeUnit.DAYS, timeframe=2)
    generator = TrainingTableGeneratorV2(graph, validated_query, split)
    task_type = generator.task_type

    timestamps = generator.generate_timestamps()
    # TRANSACTIONS.DATE contains dates between 1990-01-01 and 1990-01-05
    # ORDERS.DATE contains dates between 1990-01-02 and 1990-01-04
    # The only valid date generated is 1990-01-03;
    # In an attempt to also have a val/train set, the system will also
    # generate 1990-01-01 and 1989-12-30, dropping the earliest one.
    assert timestamps == [
        pd.Timestamp('1990-01-03 00:00:00'),
        pd.Timestamp('1990-01-01 00:00:00')
    ]
    assert task_type == TaskType.BINARY_CLASSIFICATION


def test_default_target_time_range_converts_int_time_stats(
        create_node_pred_graph, tempdir):
    graph = create_node_pred_graph('test_graph', tempdir.name)
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 2) '
             'FOR EACH USERS.USER_ID WHERE COUNT(TRANSACTIONS.*, -2, 0) > 0')
    validated_query = get_validated_query(graph, query)

    table = graph.table('TRANSACTIONS')
    time_col = table.time_col
    min_date = pd.Timestamp('1990-01-01')
    max_date = pd.Timestamp('1990-01-05')

    with patch.object(
            table,
            'compute_col_stats',
            return_value={
                time_col.name: {
                    ColStatType.MIN: Stats(value=int(min_date.value),
                                           approx=True),
                    ColStatType.MAX: Stats(value=int(max_date.value),
                                           approx=True),
                }
            },
    ):
        generator = TrainingTableGeneratorV2(
            graph,
            validated_query,
            DateOffsetSplit([-2, 0], unit=TimeUnit.DAYS, timeframe=2),
        )
        range_start, range_end, filter_buffer = (
            generator.get_default_target_time_range())

    assert range_start == pd.Timestamp('1990-01-03')
    assert range_end == max_date
    assert filter_buffer == pd.Timedelta(days=2)


def test_generate_time_intervals_entity_filter_inf(create_node_pred_graph,
                                                   tempdir):
    graph = create_node_pred_graph('test_graph', tempdir.name)
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 1) > 30 '
             'FOR EACH USERS.USER_ID WHERE COUNT(TRANSACTIONS.*, -INF, 0) > 0')
    validated_query = get_validated_query(graph, query)

    split = DateOffsetSplit([-1, 0], unit=TimeUnit.DAYS, timeframe=1)
    generator = TrainingTableGeneratorV2(graph, validated_query, split)
    task_type = generator.task_type

    timestamps = generator.generate_timestamps()
    # TRANSACTIONS.DATE contains dates between 1990-01-01 and 1990-01-05
    # No dates should be dropped by the entity filter
    assert timestamps == [
        pd.Timestamp('1990-01-04 00:00:00'),
        pd.Timestamp('1990-01-03 00:00:00'),
        pd.Timestamp('1990-01-02 00:00:00')
    ]
    assert task_type == TaskType.BINARY_CLASSIFICATION


def test_generate_time_intervals_entity_filter_inf_and_non_inf(
        create_node_pred_graph, tempdir):
    graph = create_node_pred_graph('test_graph', tempdir.name)
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 1) > 30 '
             'OR COUNT(ORDERS.*, 0, 1) > 0'
             'FOR EACH USERS.USER_ID WHERE COUNT(TRANSACTIONS.*, -2, 0) > 0 '
             'AND COUNT(TRANSACTIONS.*, -INF, 0) > 1')
    validated_query = get_validated_query(graph, query)

    split = DateOffsetSplit([-1, 0], unit=TimeUnit.DAYS, timeframe=1)
    generator = TrainingTableGeneratorV2(graph, validated_query, split)
    task_type = generator.task_type

    timestamps = generator.generate_timestamps()
    # TRANSACTIONS.DATE contains dates between 1990-01-01 and 1990-01-05
    # ORDERS.DATE contains dates between 1990-01-02 and 1990-01-04
    # The only valid dates generated are 1990-01-03, and 1990-01-04
    # None dropped because buffer was used.
    assert timestamps == [
        pd.Timestamp('1990-01-04 00:00:00'),
        pd.Timestamp('1990-01-03 00:00:00')
    ]
    assert task_type == TaskType.BINARY_CLASSIFICATION


def test_generate_time_intervals_train_start_offset(create_node_pred_graph,
                                                    tempdir):
    graph = create_node_pred_graph('test_graph', tempdir.name)
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 1) > 30 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)

    split = DateOffsetSplit([-1, 0], unit=TimeUnit.DAYS, timeframe=1)
    generator = TrainingTableGeneratorV2(graph, validated_query, split,
                                         train_start_offset=2)
    task_type = generator.task_type

    timestamps = generator.generate_timestamps()
    assert timestamps == [
        pd.Timestamp('1990-01-04 00:00:00'),
        pd.Timestamp('1990-01-03 00:00:00')
    ]
    assert task_type == TaskType.BINARY_CLASSIFICATION


def test_generate_time_intervals_train_end_offset(create_node_pred_graph,
                                                  tempdir):
    graph = create_node_pred_graph('test_graph', tempdir.name)
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 1) > 30 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)

    split = DateOffsetSplit([-1, 0], unit=TimeUnit.DAYS, timeframe=1)
    generator = TrainingTableGeneratorV2(graph, validated_query, split,
                                         train_end_offset=1)
    task_type = generator.task_type

    timestamps = generator.generate_timestamps()
    assert timestamps == [
        pd.Timestamp('1990-01-03 00:00:00'),
        pd.Timestamp('1990-01-02 00:00:00')
    ]
    assert task_type == TaskType.BINARY_CLASSIFICATION


def test_generate_time_intervals_timeframe_step(create_node_pred_graph,
                                                tempdir):
    graph = create_node_pred_graph('test_graph', tempdir.name)
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 1440, minutes) > 30 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)

    split = DateOffsetSplit([-1440, 0], unit=TimeUnit.MINUTES, timeframe=1440)
    generator = TrainingTableGeneratorV2(graph, validated_query, split,
                                         timeframe_step=720)
    task_type = generator.task_type

    timestamps = generator.generate_timestamps()
    assert timestamps == [
        pd.Timestamp('1990-01-04 00:00:00'),
        pd.Timestamp('1990-01-03 00:00:00'),
        pd.Timestamp('1990-01-02 00:00:00'),
        pd.Timestamp('1990-01-01 12:00:00')
    ]
    assert task_type == TaskType.BINARY_CLASSIFICATION


def test_generate_time_intervals_time_range_split(create_node_pred_graph,
                                                  tempdir):
    graph = create_node_pred_graph('test_graph', tempdir.name)
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 6, 18, hours) '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)

    # TimeRangeSplit needs to be robust against strings
    split = TimeRangeSplit([(pd.Timestamp('1990-01-01 01:00:00'),
                             pd.Timestamp('1990-01-02 12:00:00')),
                            (pd.Timestamp('1990-01-02 14:00:00'),
                             pd.Timestamp('1990-01-03 23:30:00')),
                            ('1990-01-04 00:00:00', '1990-01-04 19:00:00')],
                           unit=TimeUnit.HOURS, timeframe=12)
    generator = TrainingTableGeneratorV2(graph, validated_query, split)
    timestamps = generator.generate_timestamps()
    # TRANSACTIONS.DATE contains dates between 1990-01-01 and 1990-01-05
    # The split gives five valid timestamps:
    # 1990-01-01 06:00, 1990-01-01 18:00, (train)
    # 1990-01-03 05:30, 1990-01-02 17:30 (val)
    # 1990-01-04 01:00 (test)
    # the earliest dropped to avoid an empty input
    assert timestamps == [
        pd.Timestamp('1990-01-04 01:00:00'),
        pd.Timestamp('1990-01-03 05:30:00'),
        pd.Timestamp('1990-01-02 17:30:00'),
        pd.Timestamp('1990-01-01 18:00:00')
    ]


def test_generate_time_intervals_backtesting_split(create_node_pred_graph,
                                                   tempdir):
    graph = create_node_pred_graph('test_graph', tempdir.name)
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 1, 2) '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)

    split = BacktestingSplit('1990-01-04 12:00:00')
    generator = TrainingTableGeneratorV2(graph, validated_query, split)
    timestamps = generator.generate_timestamps()
    assert timestamps == [
        pd.Timestamp('1990-01-04 12:00:00'),
    ]


@pytest.mark.parametrize(
    'kwargs',
    [
        # Earliest timeframe not dropped as there is
        # one timframe before it
        dict(
            train_start_offset=20,
            train_end_offset=10,
            num_timeframes=5,
            first_timeframe=pd.Timestamp('2018-01-10'),
            last_timeframe=pd.Timestamp('2018-01-18'),
        ),
        # Earliest timeframe dropped as there is no
        # timeframe before it
        dict(
            train_start_offset=None,
            train_end_offset=None,
            num_timeframes=13,
            first_timeframe=pd.Timestamp('2018-01-04'),
            last_timeframe=pd.Timestamp('2018-01-28'),
        ),
        dict(
            train_start_offset=28,
            train_end_offset=10,
            num_timeframes=8,
            first_timeframe=pd.Timestamp('2018-01-04'),
            last_timeframe=pd.Timestamp('2018-01-18'),
        ),
    ])
def test_dropping_last_timeframe(
    kwargs,
    graph_with_entity_dates,
    tempdir,
):
    """This test checks that the earliest timeframe is dropped
    only when there is no timeframe before it"""
    query = ('PREDICT SUM(TRANS.AMOUNT, 0, 2) '
             'FOR EACH USERS.USER_ID')
    graph = graph_with_entity_dates(True, True)
    validated_query = get_validated_query(graph, query)
    split = DateOffsetSplit([-2, 0], unit=TimeUnit.DAYS, timeframe=2)
    generator = TrainingTableGeneratorV2(
        graph, validated_query, split,
        train_start_offset=kwargs['train_start_offset'],
        train_end_offset=kwargs['train_end_offset'])
    timestamps = generator.generate_timestamps()
    assert len(timestamps) == kwargs['num_timeframes']
    assert timestamps[-1] == kwargs['first_timeframe']
    assert timestamps[0] == kwargs['last_timeframe']


@pytest.mark.skip(reason=_SKIP_REASON_FINANCIAL_FULL_SCALE_LITERALS)
def test_whatif(create_graph):
    graph = create_graph('FINANCIAL', infer=True)
    graph['TRANS'].set_time_col('DATE')

    query = ('PREDICT SUM(TRANS.AMOUNT WHERE TRANS.AMOUNT > 10 '
             'AND TRANS.DATE > 1993-03-30, 15, 30) '
             'FOR EACH ACCOUNT.ACCOUNT_ID '
             'WHERE ACCOUNT.ACCOUNT_ID < 100 '
             'ASSUMING SUM(TRANS.AMOUNT, 0, 15) > 0')
    validated_query = get_validated_query(graph, query)
    split = DateOffsetSplit([-30, 0], unit=TimeUnit.DAYS, timeframe=15)
    generator = TrainingTableGeneratorV2(graph, validated_query, split,
                                         train_start_offset=90)
    train_df = generator.generate_table()
    assert len(train_df) == 21


@pytest.mark.skip(reason=_SKIP_REASON_FINANCIAL_FULL_SCALE_LITERALS)
def test_complex_filters_with_train_start_offset(create_graph):
    graph = create_graph('FINANCIAL', infer=True)
    graph['TRANS'].set_time_col('DATE')

    cond_1 = 'SUM(TRANS.AMOUNT, -60, 0) > 0'
    cond_2 = 'AVG(TRANS.AMOUNT, -60, 0) > 2000'
    cond_3 = 'MIN(TRANS.AMOUNT, -60, 0) > 500'
    cond_4 = 'MAX(TRANS.AMOUNT, -60, 0) < 5000'
    cond_5 = 'LIST_DISTINCT(TRANS.OPERATION, -30, 0) CONTAINS "VKLAD"'
    cond_6 = 'LIST_DISTINCT(TRANS.TYPE, -30, 0) NOT CONTAINS "VYBER"'
    t_e_filter = (f'((({cond_1}) AND ({cond_2})) '
                  f'OR (({cond_3}) AND ({cond_4}))) '
                  f'AND (({cond_5}) AND ({cond_6}))')
    query = (f'PREDICT SUM(TRANS.AMOUNT WHERE TRANS.AMOUNT > 10 '
             f'AND TRANS.DATE > 1993-03-30, 15, 30) '
             f'FOR EACH ACCOUNT.ACCOUNT_ID WHERE {t_e_filter} '
             f'AND ACCOUNT.ACCOUNT_ID < 100 '
             f'ASSUMING SUM(TRANS.AMOUNT, 0, 15) > 0')
    validated_query = get_validated_query(graph, query)
    split = DateOffsetSplit([-30, 0], unit=TimeUnit.DAYS, timeframe=15)
    generator = TrainingTableGeneratorV2(graph, validated_query, split,
                                         train_start_offset=90)
    train_df = generator.generate_table()

    # `target_filter` requires train date from '1993-03-31'
    # `train_start_offset` requires train date from '1993-04-01'
    # ('1993-06-30' - 90 = '1993-04-01')
    min_date = min(train_df['TIMESTAMP']).strftime('%Y-%m-%d')
    assert min_date == '1993-04-01'
    assert train_df.shape == (7, 3)

    # Check temporal entity filter:
    df = graph['TRANS'].pandas()
    entity_col = 'ACCOUNT_ID'
    time_col = 'DATE'
    # We'll check correctness for entities in the first timeframe
    e_ids = train_df[train_df['TIMESTAMP'] == min(train_df['TIMESTAMP'])][
        generator.ENTITY].unique()
    # Select the min date to test
    date = train_df[generator.TIMESTAMP].min()
    # Select TRANS df which has the training table entities
    df_entity = df[df[entity_col].isin(e_ids)]
    date_60 = date - pd.DateOffset(days=60)
    # Select df which has entities and time between date and 60 days before
    df_60 = df_entity[(df_entity[time_col] > date_60)
                      & (df_entity[time_col] <= date)]
    df_agg = df_60.groupby(entity_col)['AMOUNT'].agg(sum='sum', avg='mean',
                                                     min='min', max='max')
    cond_1_2 = (df_agg['sum'] > 0) & (df_agg['avg'] > 2000)
    cond_3_4 = (df_agg['min'] > 500) & (df_agg['max'] < 5000)
    date_30 = date - pd.DateOffset(days=30)
    # Select df which has entities and time between date and 30 days before
    df_30 = df_entity[(df_entity[time_col] > date_30)
                      & (df_entity[time_col] <= date)]
    op_ser = df_30.groupby(entity_col)['OPERATION'].unique()
    type_ser = df_30.groupby(entity_col)['TYPE'].unique()
    cond_5_ser = pd.Series(
        ['VKLAD' in row.dropna().to_numpy().tolist() for row in op_ser])
    cond_6_ser = pd.Series(
        ['VYBER' not in row.dropna().to_numpy().tolist() for row in type_ser])
    cond_5_6 = (cond_5_ser & cond_6_ser)
    # Make sure train entities fulfill temporal entity filter conditions
    assert ((cond_1_2 | cond_3_4).reset_index(drop=True) & cond_5_6).all()

    # Invert the temporal entity filter
    t_e_filter = f'NOT ({t_e_filter})'
    query = (f'PREDICT SUM(TRANS.AMOUNT WHERE TRANS.AMOUNT > 10, 15, 30) '
             f'FOR EACH ACCOUNT.ACCOUNT_ID WHERE {t_e_filter} '
             f'ASSUMING SUM(TRANS.AMOUNT, 0, 15) > 0')
    validated_query = get_validated_query(graph, query)
    generator = TrainingTableGeneratorV2(graph, validated_query, split,
                                         train_start_offset=90)
    train_df = generator.generate_table()
    # Select the same train table date
    df = train_df[train_df[generator.TIMESTAMP] == min_date]
    # Entities exist in the train table before inverting the
    # temporal entity filter should not appear here
    assert len(set(df[generator.ENTITY]).intersection(set(e_ids))) == 0


def test_temporal_node_regression_stats(dg_indexed_temporal_financial_graph,
                                        tmp_path):
    # Exercises get_train_table_stats on the temporal FINANCIAL fixture, which
    # re-derives ACCOUNT/TRANS from the CSV (parsing TRANS.DATE, shifting the
    # origin to 2023 to dodge the pre-2001 ns-DATE split-stats corruption). The
    # shipped FINANCIAL parquet TRANS.DATE is degenerate (nunique=1) so it
    # cannot back a temporal window; the CSV is the unlock.
    graph, index_path = dg_indexed_temporal_financial_graph
    graph.infer_metadata()

    query = 'PREDICT SUM(TRANS.AMOUNT, 15, 30) FOR EACH ACCOUNT.ACCOUNT_ID'
    validated_query = get_validated_query(graph, query)
    pquery = PredictiveQueryV2(
        name='financial_temporal_regression',
        graph=graph,
        validated_predictive_query=validated_query,
        split=DateOffsetSplit([-30, 0], unit=TimeUnit.DAYS, timeframe=30),
    )
    task = materialize_task_dg(pquery, index_path, tmp_path)

    # The six regression target-distribution keys must be present in the
    # DG-produced per-stage stats. After-split stats come from the native
    # split-stats manifest; the per-stage
    # TemporalNodeTrainTableStats.target_distribution carries the distribution.
    split_stats = task.train_table.get_train_table_stats(after_split=True)
    assert split_stats is not None
    distribution_keys = [
        'target_min', 'target_percentile_25', 'target_median',
        'target_percentile_75', 'target_max', 'target_mean'
    ]
    stages = [
        split_stats.train_table_stats,
        split_stats.val_table_stats,
        split_stats.test_table_stats,
    ]
    populated = [s for s in stages if s is not None]
    assert populated, "expected at least one non-empty split-stats stage"
    for stage in populated:
        for key in distribution_keys:
            assert key in stage.target_distribution


def test_get_table_stats(dg_indexed_static_node_pred_graph, tmp_path):
    # Exercises get_train_table_stats on the static node-pred fixture: one
    # USERS table, USER_ID 0..3, STATUS ['A','B',NA,'B'], no time_col. The
    # NA-STATUS entity (id 2) is dropped, so the producer keeps 3 entities /
    # 3 train rows over 2 distinct targets ('A','B').
    graph, index_path = dg_indexed_static_node_pred_graph
    graph.infer_metadata()

    query = 'PREDICT USERS.STATUS FOR EACH USERS.USER_ID'
    validated_query = get_validated_query(graph, query)
    pquery = PredictiveQueryV2(
        name='static_node_status',
        graph=graph,
        validated_predictive_query=validated_query,
        split=RandomSplit([0.8, 0.1, 0.1]),
    )
    task = materialize_task_dg(pquery, index_path, tmp_path)

    # After-split stats come from the native DG split-stats manifest. Aggregate
    # the per-stage counts back to whole-train-table totals: 3 surviving
    # entities (NA-STATUS id 2 dropped) -> 3 train rows across all split stages.
    split_stats = task.train_table.get_train_table_stats(after_split=True)
    assert split_stats is not None
    stages = [
        split_stats.train_table_stats,
        split_stats.val_table_stats,
        split_stats.test_table_stats,
    ]
    populated = [s for s in stages if s is not None]
    assert populated, "expected at least one non-empty split-stats stage"

    # The four-row fixture (one NA-STATUS row dropped) leaves 3 train rows over
    # 3 entities and 2 distinct categorical targets ('A','B'); the small random
    # split lands every surviving row in a single stage, so the per-stage totals
    # equal the whole-train-table counts (num_entities=3, num_train_rows=3,
    # num_targets=2).
    total_train_rows = sum(s.num_train_rows for s in populated)
    total_entities = sum(s.num_entities for s in populated)
    assert total_train_rows == 3
    assert total_entities == 3
    assert max(s.num_targets for s in populated) == 2

    # The native categorical target distribution must carry both observed
    # STATUS values with their real counts (A:1, B:2 -> the NA row excluded).
    counts = {
        label:
        sum((s.target_distribution or {}).get('count', {}).get(label, 0)
            for s in populated)
        for label in ('A', 'B')
    }
    assert counts == {'A': 1, 'B': 2}


def test_static_stage_masks_follow_seed_parts(tmp_path):
    # tt.parquet row order differs from run to run; masks must follow the
    # stage seed parts, not row position.
    def write(path, entities):
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table({
                'ENTITY1': pa.array(entities, pa.int64()),
                'TIMESTAMP': pa.array([0] * len(entities),
                                      pa.timestamp('ns')),
            }), path)

    write(tmp_path / 'tt.parquet', [3, 1, 0])
    write(tmp_path / 'train_seeds' / 'part-00000.parquet', [1, 0])
    write(tmp_path / 'val_seeds' / 'part-00000.parquet', [3])
    (tmp_path / 'split_stats.json').write_text(
        '{"train_table_stats": {"num_train_rows": 2}, '
        '"val_table_stats": {"num_train_rows": 1}}')

    masks = _ttng_stage_masks_from_bundle(str(tmp_path), 3)
    assert masks[TrainingStage.TRAIN].tolist() == [False, True, True]
    assert masks[TrainingStage.VAL].tolist() == [True, False, False]
    assert not masks[TrainingStage.TEST].any()


def test_update_min_max_anchor_date_cache():
    # ``update_min_max_anchor_date_cache`` only writes to the stats cache, so
    # the behavior asserted here is unchanged on DiskGraph.
    stats_manager = StatsManager(TaskType.REGRESSION, QueryType.TEMPORAL,
                                 MagicMock())
    cache_mock = MagicMock()

    min_date = pd.Timestamp('2020-01-01')
    max_date = pd.Timestamp('2021-02-02')
    anchor_date = pd.Timestamp('2021-01-01')
    job_id = 'test_job_id'
    # `StatsManager.write_to_cache` resolves `get_active_cache` from the
    # `kumo.pquery.stats_manager` module namespace (it is imported there at
    # module load), so patch it there rather than at `kumo.data` -- patching the
    # source module has no effect on the already-bound name.
    with patch('kumo.pquery.stats_manager.get_active_cache',
               new=lambda: cache_mock):
        stats_manager.update_min_max_anchor_date_cache(job_id, min_date,
                                                       max_date, anchor_date)

    cache_mock.pquery_table_stats_cache.update.assert_called_once_with(
        job_id, {
            'min_date': Stats(min_date, approx=False),
            'max_date': Stats(max_date, approx=False),
            'anchor_date': Stats(anchor_date, approx=False)
        })


def test_native_v2_stats_seed_omits_per_timeframe_diagnostics():
    query = MagicMock(query_type=QueryType.TEMPORAL)
    manager = StatsManager(TaskType.REGRESSION, query, MagicMock())
    cached = {}
    manager.write_to_cache = lambda _job_id, values: cached.update(values)
    manager.load_from_cache = lambda _job_id, keys=None: {
        key: value
        for key, value in cached.items() if keys is None or key in keys
    }
    stats = {
        'version':
        2,
        'kind':
        'tt_stats',
        'num_total_rows':
        100,
        'num_train_rows':
        100,
        'num_entities':
        20,
        'num_targets':
        2,
        'num_target_rows':
        100,
        'num_timeframes':
        3,
        'num_timeframes_post_filter':
        3,
        'avg_entities_per_timeframe':
        33,
        'min_date':
        '2026-01-01T00:00:00Z',
        'max_date':
        '2026-01-03T00:00:00Z',
        'anchor_date':
        '2026-01-03T00:00:00Z',
        'node_degree_stats':
        None,
        'scoring_table_column_stats':
        None,
        'target_distribution': {},
    }

    assert manager.seed_train_stats_cache_from_native_artifact('job',
                                                               stats) == []
    assert cached[STATS_PER_TIMEFRAME].value is None


def test_validate_offset_validation(create_node_pred_graph, tempdir):
    graph = create_node_pred_graph('test_graph', tempdir.name)
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 2) > 30 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)
    pquery = PredictiveQueryV2(name='name', graph=graph,
                               validated_predictive_query=validated_query,
                               train_start_offset=2, train_end_offset=3)
    response = pquery.get_validate_response()
    assert response.errors[0].title == 'Train start/end offsets error'


@pytest.mark.parametrize('query, expected_path', [
    ('PREDICT USERS.AGE FOR EACH USERS.USER_ID', []),
    ('PREDICT LOCATIONS.NAME FOR EACH USERS.USER_ID', [[
        ('parquet.USERS', 'LOCATION_ID', 'parquet.LOCATIONS')
    ]]),
    ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 5) > 4 OR '
     'COUNT(SESSIONS.*, 0, 2) > 0 FOR EACH USERS.USER_ID', [[
         ('parquet.SESSIONS', 'USER_ID', 'parquet.USERS')
     ]]),
    ('PREDICT LIST_DISTINCT(TRANSACTIONS.BANK_ID, 0, 5) RANK TOP 10 '
     'FOR EACH USERS.USER_ID', [[
         ('parquet.TRANSACTIONS', 'USER_ID', 'parquet.USERS')
     ], [('parquet.TRANSACTIONS', 'BANK_ID', 'parquet.BANKS')]]),
])
def test_target_to_entity_paths(query, expected_path, create_node_pred_graph,
                                tempdir):
    graph = create_node_pred_graph('test_graph', tempdir.name)
    validated_query = get_validated_query(graph, query)
    pquery = PredictiveQueryV2(
        name='test_query',
        graph=graph,
        validated_predictive_query=validated_query,
    )
    assert pquery.target_to_entity_paths == expected_path


@pytest.mark.parametrize('disable', [True, False])
def test_settings_temporal_link_prediction(disable, create_node_pred_graph,
                                           tempdir):
    graph = create_node_pred_graph('test_graph', tempdir.name)

    query = ('PREDICT LIST_DISTINCT(TRANSACTIONS.BANK_ID, 0, 2) '
             'RANK TOP 10 FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)
    settings = PQuerySettings()
    settings.disable_temporal_link_prediction = disable
    if disable:
        with pytest.raises(
                ValidationException,
                match="`LIST_DISTINCT` operator is currently not supported"):
            PredictiveQueryV2(
                name='pquery_name',
                graph=graph,
                validated_predictive_query=validated_query,
                pquery_settings=settings,
            )
    else:
        PredictiveQueryV2(
            name='pquery_name',
            graph=graph,
            validated_predictive_query=validated_query,
            pquery_settings=settings,
        )


@pytest.mark.parametrize('disable', [True, False])
def test_settings_static_link_prediction(disable, create_lp_graph, tempdir):
    graph = create_lp_graph(root_dir=tempdir.name, num_entities=3,
                            key_name_match=False, set_trans_time_col=False,
                            set_entity_time_col=False, missing_link=True,
                            dangling_entity=True, use_parquet=True)
    graph["PRODUCTS"].remove_time_col()
    query = ('PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID_FK) RANK TOP 2 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)
    settings = PQuerySettings()
    settings.disable_static_link_prediction = disable
    if disable:
        with pytest.raises(
                ValidationException,
                match="`LIST_DISTINCT` operator is currently not supported"):
            PredictiveQueryV2(
                name='pquery_name',
                graph=graph,
                validated_predictive_query=validated_query,
                pquery_settings=settings,
            )
    else:
        PredictiveQueryV2(
            name='pquery_name',
            graph=graph,
            validated_predictive_query=validated_query,
            pquery_settings=settings,
        )


@pytest.mark.parametrize('disable', [True, False])
def test_settings_regression(disable, create_node_pred_graph, tempdir):
    graph = create_node_pred_graph('test_graph', tempdir.name)

    query = 'PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 2) FOR EACH USERS.USER_ID'
    validated_query = get_validated_query(graph, query)
    settings = PQuerySettings()
    settings.disable_temporal_node_regression = disable
    if disable:
        with pytest.raises(
                ValidationException, match="Regression and Forecasting are "
                "currently not supported"):
            PredictiveQueryV2(
                name='pquery_name',
                graph=graph,
                validated_predictive_query=validated_query,
                pquery_settings=settings,
            )
    else:
        PredictiveQueryV2(
            name='pquery_name',
            graph=graph,
            validated_predictive_query=validated_query,
            pquery_settings=settings,
        )


@pytest.mark.parametrize('disable', [True, False])
def test_settings_multilabel(disable, create_node_pred_graph, tempdir):
    graph = create_node_pred_graph('test_graph', tempdir.name)

    query = ('PREDICT LIST_DISTINCT(TRANSACTIONS.TYPE, 0, 2) '
             'CLASSIFY FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)
    settings = PQuerySettings()
    settings.disable_multilabel_classification = disable
    if disable:
        with pytest.raises(
                ValidationException,
                match="Multilabel classification is currently not supported"):
            PredictiveQueryV2(
                name='pquery_name',
                graph=graph,
                validated_predictive_query=validated_query,
                pquery_settings=settings,
            )
    else:
        PredictiveQueryV2(
            name='pquery_name',
            graph=graph,
            validated_predictive_query=validated_query,
            pquery_settings=settings,
        )


@pytest.mark.parametrize('disable', [True, False])
def test_settings_static(disable, create_node_pred_graph, tempdir):
    graph = create_node_pred_graph('test_graph', tempdir.name)

    query = 'PREDICT LOCATIONS.NAME FOR EACH USERS.USER_ID'
    validated_query = get_validated_query(graph, query)
    settings = PQuerySettings()
    settings.disable_static_node_prediction = disable
    if disable:
        with pytest.raises(
                ValidationException,
                match="Static prediction problems are not supported"):
            PredictiveQueryV2(
                name='pquery_name',
                graph=graph,
                validated_predictive_query=validated_query,
                pquery_settings=settings,
            )
    else:
        PredictiveQueryV2(
            name='pquery_name',
            graph=graph,
            validated_predictive_query=validated_query,
            pquery_settings=settings,
        )


def test_get_max_train_rows(tempdir):
    exp_max_rows = 200_000_000
    assert get_max_train_rows() == exp_max_rows


def test_generate_time_intervals_start_end_time_absolute(
        create_node_pred_graph, tempdir):
    """Test using absolute timestamps for start_time and end_time"""
    graph = create_node_pred_graph('test_graph', tempdir.name)
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 1) > 30 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)

    split = DateOffsetSplit([0], unit=TimeUnit.DAYS, timeframe=1)
    generator = TrainingTableGeneratorV2(
        graph,
        validated_query,
        split,
        start_time='1990-01-01',  # Absolute timestamp
        end_time='1990-01-03'  # Absolute timestamp
    )
    task_type = generator.task_type

    timestamps = generator.generate_timestamps()
    # Should only include timestamps between start_time and end_time
    assert timestamps == [
        pd.Timestamp('1990-01-02 00:00:00'),
        pd.Timestamp('1990-01-01 00:00:00')
    ]
    assert task_type == TaskType.BINARY_CLASSIFICATION


def test_generate_time_intervals_start_end_time_relative(
        create_node_pred_graph, tempdir):
    """Test using relative offsets for start_time and end_time"""
    graph = create_node_pred_graph('test_graph', tempdir.name)
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 1) > 30 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)

    split = DateOffsetSplit([-1, 0], unit=TimeUnit.DAYS, timeframe=1)
    generator = TrainingTableGeneratorV2(
        graph,
        validated_query,
        split,
        start_time=-4,  # 4 days before latest date
        end_time=-1  # 1 day before latest date
    )
    task_type = generator.task_type

    timestamps = generator.generate_timestamps()
    # Should only include timestamps between max_date-4 and max_date-1
    assert timestamps == [
        pd.Timestamp('1990-01-03 00:00:00'),
        pd.Timestamp('1990-01-02 00:00:00')
    ]
    assert task_type == TaskType.BINARY_CLASSIFICATION


def test_generate_time_intervals_start_time_inf(create_node_pred_graph,
                                                tempdir):
    """Test using -INF for start_time"""
    graph = create_node_pred_graph('test_graph', tempdir.name)
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 1) > 30 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)

    split = DateOffsetSplit([0], unit=TimeUnit.DAYS, timeframe=1)
    generator = TrainingTableGeneratorV2(
        graph,
        validated_query,
        split,
        start_time='-INF',  # From beginning of time
        end_time='1990-01-04'  # Up to specific date
    )
    task_type = generator.task_type

    timestamps = generator.generate_timestamps()
    # Should include all valid timestamps up to end_time
    assert timestamps == [
        pd.Timestamp('1990-01-03 00:00:00'),
        pd.Timestamp('1990-01-02 00:00:00')
    ]
    assert task_type == TaskType.BINARY_CLASSIFICATION


def test_generate_time_intervals_invalid_start_end_time(create_node_pred_graph,
                                                        tempdir):
    """Test validation errors when start_time >= end_time"""
    graph = create_node_pred_graph('test_graph', tempdir.name)
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 1) > 30 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)

    split = DateOffsetSplit([-1, 0], unit=TimeUnit.DAYS, timeframe=1)

    # Test with absolute timestamps
    generator = TrainingTableGeneratorV2(graph, validated_query, split,
                                         start_time='1990-01-04',
                                         end_time='1990-01-02')
    validation_response = generator.validate_pre_train_table_generate()
    assert not validation_response.ok
    assert ("start_time (1990-01-04 00:00:00) must be smaller than "
            "end_time (1990-01-02 00:00:00)"
            in validation_response.error_message())

    # Test with relative offsets
    generator = TrainingTableGeneratorV2(
        graph,
        validated_query,
        split,
        start_time=-1,  # 1 day before latest
        end_time=-3  # 3 days before latest
    )
    validation_response = generator.validate_pre_train_table_generate()
    assert not validation_response.ok
    assert ("start_time must be more negative than end_time"
            in validation_response.error_message())


def test_generate_time_intervals_mixed_types_error(create_node_pred_graph,
                                                   tempdir):
    """Test error when mixing absolute and relative timestamps"""
    graph = create_node_pred_graph('test_graph', tempdir.name)
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 1) > 30 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)

    split = DateOffsetSplit([-1, 0], unit=TimeUnit.DAYS, timeframe=1)
    generator = TrainingTableGeneratorV2(
        graph,
        validated_query,
        split,
        start_time=-2,  # Relative offset
        end_time='1990-01-04'  # Absolute timestamp
    )
    validation_response = generator.validate_pre_train_table_generate()
    assert not validation_response.ok
    assert ("start_time and end_time must both be offsets or both be timestamps"
            in validation_response.error_message())


def test_generate_time_intervals_with_legacy_params_error(
        create_node_pred_graph, tempdir):
    r"""Test error when mixing new time parameters with legacy offset
    parameters
    """
    graph = create_node_pred_graph('test_graph', tempdir.name)
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 1) > 30 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)

    split = DateOffsetSplit([-1, 0], unit=TimeUnit.DAYS, timeframe=1)
    generator = TrainingTableGeneratorV2(
        graph,
        validated_query,
        split,
        start_time='1990-01-02',
        end_time='1990-01-04',
        train_start_offset=2  # Should not be used with start_time/end_time
    )
    validation_response = generator.validate_pre_train_table_generate()
    assert not validation_response.ok
    assert ("Cannot specify both start_time/end_time and "
            "train_start_offset/train_end_offset"
            in validation_response.error_message())


def test_generate_timestamps_v2_with_timeframe_step(create_node_pred_graph,
                                                    tempdir):
    """Test generate_timestamps_v2 with TimeSplit and timeframe_step"""
    graph = create_node_pred_graph('test_graph', tempdir.name)
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 24, hours) > 30 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)

    # Use TimeSplit with fractions for train/val/test
    split = TimeSplit(
        fractions=[0.6, 0.2, 0.2],  # 60% train, 20% val, 20% test
        unit=TimeUnit.HOURS,
        anchor_start_offset=0,
        timeframe_size=24,
    )
    generator = TrainingTableGeneratorV2(
        graph,
        validated_query,
        split,
        timeframe_step=12  # 12 hour steps
    )
    task_type = generator.task_type

    timestamps = generator.generate_timestamps_v2()
    # TRANSACTIONS.DATE contains dates between 1990-01-01 and 1990-01-05
    # With 12 hour steps, should generate timestamps every 12 hours
    # Total range is ~4 days, so with 60/20/20 split:
    # Test: 1990-01-04 (last ~0.8 days)
    # Val: 1990-01-03 12:00 to 1990-01-04 (next ~0.8 days)
    # Train: Rest of the dates
    assert timestamps == [
        pd.Timestamp('1990-01-04 00:00:00'),  # Test
        pd.Timestamp('1990-01-03 12:00:00'),  # Val
        pd.Timestamp('1990-01-03 00:00:00'),  # Train
        pd.Timestamp('1990-01-02 12:00:00'),  # Train
        pd.Timestamp('1990-01-02 00:00:00'),  # Train
        pd.Timestamp('1990-01-01 12:00:00')  # Train
    ]
    assert task_type == TaskType.BINARY_CLASSIFICATION


def test_generate_timestamps_v2_with_days_unit(create_node_pred_graph, tempdir):
    """Test generate_timestamps_v2 with days as unit"""
    graph = create_node_pred_graph('test_graph', tempdir.name)
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 2, days) > 30 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)

    # Use TimeSplit with fractions for train/val/test
    split = TimeSplit(
        fractions=[0.5, 0.5],  # 50% train, 50% val
        unit=TimeUnit.DAYS,
        anchor_start_offset=0,
        timeframe_size=2,
    )
    generator = TrainingTableGeneratorV2(
        graph,
        validated_query,
        split,
        timeframe_step=1  # 1 day steps
    )
    task_type = generator.task_type
    timestamps = generator.generate_timestamps_v2()
    # TRANSACTIONS.DATE contains dates between 1990-01-01 and 1990-01-05
    # With 1 day steps, should generate timestamps every day
    # Total range is ~4 days, so with 50/50 split:
    # Val: 1990-01-03 - 1990-01-04
    # Train: 1990-01-02 - 1990-01-03
    assert timestamps == [
        pd.Timestamp('1990-01-03 00:00:00'),  # Val
        pd.Timestamp('1990-01-02 00:00:00'),  # Train
    ]
    assert task_type == TaskType.BINARY_CLASSIFICATION


def test_generate_timestamps_v2_with_start_end_time(create_node_pred_graph,
                                                    tempdir):
    """Test generate_timestamps_v2 with explicit start_time and end_time"""
    graph = create_node_pred_graph('test_graph', tempdir.name)
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 8, hours) > 30 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)

    split = TimeSplit(
        fractions=[0.6, 0.2, 0.2],
        unit=TimeUnit.HOURS,
        anchor_start_offset=0,
        timeframe_size=8,
    )
    generator = TrainingTableGeneratorV2(
        graph,
        validated_query,
        split,
        start_time='1990-01-02',  # Explicit start time
        end_time='1990-01-03',  # Explicit end time
    )
    task_type = generator.task_type

    timestamps = generator.generate_timestamps_v2()
    # Should only include timestamps between start_time and end_time
    assert timestamps == [
        pd.Timestamp('1990-01-02 16:00:00'),  # Test
        pd.Timestamp('1990-01-02 08:00:00'),  # Val
        pd.Timestamp('1990-01-02 00:00:00')  # Train
    ]
    assert task_type == TaskType.BINARY_CLASSIFICATION


def test_generate_timestamps_v2_with_relative_start_end_time(
        create_node_pred_graph, tempdir):
    """Test generate_timestamps_v2 with relative start_time and end_time"""
    graph = create_node_pred_graph('test_graph', tempdir.name)
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 8, hours) > 30 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)

    split = TimeSplit(
        fractions=[0.6, 0.2, 0.2],
        unit=TimeUnit.HOURS,
        anchor_start_offset=0,
        timeframe_size=8,
    )
    generator = TrainingTableGeneratorV2(
        graph,
        validated_query,
        split,
        start_time=-48,  # 2 days before latest date
        end_time=-24,  # 1 day before latest date
        timeframe_step=8,  # 24 hour steps
    )
    task_type = generator.task_type

    timestamps = generator.generate_timestamps_v2()
    # Should only include timestamps between max_date-3 and max_date-1
    assert timestamps == [
        pd.Timestamp('1990-01-03 16:00:00'),  # Test
        pd.Timestamp('1990-01-03 08:00:00'),  # Val
        pd.Timestamp('1990-01-03 00:00:00'),  # Train
    ]
    assert task_type == TaskType.BINARY_CLASSIFICATION


def test_generate_timestamps_v2_legacy_params_error(create_node_pred_graph,
                                                    tempdir):
    """Test generate_timestamps_v2 raises error when using legacy offset
    parameters
    """
    graph = create_node_pred_graph('test_graph', tempdir.name)
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 24, hours) > 30 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)

    split = TimeSplit(
        fractions=[0.6, 0.2, 0.2],
        unit=TimeUnit.HOURS,
        anchor_start_offset=0,
        timeframe_size=24,
    )

    # Test with train_start_offset
    generator = TrainingTableGeneratorV2(graph, validated_query, split,
                                         train_start_offset=2,
                                         timeframe_step=24)

    with pytest.raises(
            ValidationException,
            match="train_start_offset and train_end_offset are deprecated",
    ):
        generator.generate_timestamps_v2()

    # Test with train_end_offset
    generator = TrainingTableGeneratorV2(graph, validated_query, split,
                                         train_end_offset=1, timeframe_step=24)

    with pytest.raises(
            ValidationException,
            match="train_start_offset and train_end_offset are deprecated",
    ):
        generator.generate_timestamps_v2()

    # Test with both legacy parameters
    generator = TrainingTableGeneratorV2(graph, validated_query, split,
                                         train_start_offset=2,
                                         train_end_offset=1, timeframe_step=24)

    with pytest.raises(
            ValidationException,
            match="train_start_offset and train_end_offset are deprecated",
    ):
        generator.generate_timestamps_v2()


@pytest.mark.parametrize("method",
                         ["generate_timestamps", "generate_timestamps_v2"])
def test_non_aligned_time_bounds_keep_newest_window(method):
    from types import MethodType, SimpleNamespace

    generator = SimpleNamespace(
        get_default_target_time_range=lambda:
        (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-10"), None),
        split=None,
        train_start_offset=None,
        train_end_offset=None,
        start_time=pd.Timestamp("2024-01-11"),
        end_time=pd.Timestamp("2024-01-22"),
        timeframe_step=pd.DateOffset(days=3),
        drop_earliest_timeframe=False,
        query=SimpleNamespace(
            get_combined_date_offset_range=lambda: SimpleNamespace(
                start_date_offset=pd.DateOffset(days=0), end_date_offset=pd.
                DateOffset(days=1))),
    )
    generator._timestamp_inputs = MethodType(
        TrainingTableGeneratorV2._timestamp_inputs, generator)

    actual = getattr(TrainingTableGeneratorV2, method)(generator)

    assert actual == [
        pd.Timestamp("2024-01-21"),
        pd.Timestamp("2024-01-18"),
        pd.Timestamp("2024-01-15"),
        pd.Timestamp("2024-01-12"),
    ]


def _pre_move_time_col(task_type, query_type, is_array, entity_has_time):
    from kumo_core.utils.time import PSEUDO_TIME_COL_NAME
    from kumoapi.typing import Dtype

    from kumo.graph import TimeColumn
    if (task_type == TaskType.STATIC_LINK_PREDICTION and not is_array):
        return TimeColumn(PSEUDO_TIME_COL_NAME, dtype=Dtype.int64,
                          is_pseudo=True)
    if query_type == QueryType.TEMPORAL or entity_has_time:
        return TimeColumn(name=TrainingTableGeneratorV2.TIMESTAMP)
    return None


@pytest.mark.parametrize(
    'task_type, query_type, is_array, entity_has_time',
    itertools.product(TaskType, QueryType, (None, False, True),
                      (False, True)),
)
def test_get_time_col_is_unchanged_by_the_shared_rule(task_type, query_type,
                                                      is_array,
                                                      entity_has_time):
    generator = SimpleNamespace(
        task_type=task_type,
        query_type=query_type,
        list_distinct_target_is_array=is_array,
        query=SimpleNamespace(entity_column='USERS.USER_ID'),
        graph=SimpleNamespace(table=lambda name: SimpleNamespace(
            has_time_col=lambda: entity_has_time)),
        TIMESTAMP=TrainingTableGeneratorV2.TIMESTAMP,
    )

    column = TrainingTableGeneratorV2.get_time_col(generator)
    expected = _pre_move_time_col(task_type, query_type, is_array,
                                  entity_has_time)

    def shape(col):
        return None if col is None else (col.name, col.dtype, col.is_pseudo)

    assert shape(column) == shape(expected)
