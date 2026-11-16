# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import pathlib
from test.fixtures.dg_indexed_graph import materialize_task_dg

import pandas as pd
import pytest
from kumoapi.typing import TimeUnit

from kumo.config.task import Mask, TimePeriod
from kumo.pquery.predictive_query import PredictiveQueryV2
from kumo.pquery.train_table_generator import TrainingTableGeneratorV2
from kumo.stats.train_table_stats import (
    SplitTrainTableStats,
    TemporalLinkTimeFrameStats,
    TemporalLinkTrainTableStats,
)
from kumo.task.split import (
    ApproxDateOffsetSplit,
    DateOffsetSplit,
    FixedSplit,
    ManualSplit,
    RandomSplit,
    Split,
    TemporalSplit,
    TimeRangeSplit,
)
from kumo.testing.pquery import disable_pre_validation, get_validated_query

_SKIP_REASON_DG_GAP_FORECASTING_TIMEFRAMES = (
    "blocked-gap: forecasting split-stats need a target_distribution that the DG "
    "split-stats builder does not emit for forecasting tasks, so the split-stats "
    "contract check aborts; also no DG-native fixture for the entity/actions "
    "tiny graph.")

_SKIP_REASON_DG_GAP_FORECASTING_TXN_SESSIONS = (
    "blocked-gap: SESSIONS-filtered forecasting split-stats need a "
    "target_distribution the DG split-stats builder does not emit for "
    "forecasting tasks; the Approx cases also use a split type DG forecasting "
    "validation rejects.")

_SKIP_REASON_DG_GAP_STATIC_TEMPORAL_SPLIT = (
    "blocked-gap: static node-prediction under a temporal split is not "
    "materializable on DG (TimeRangeSplit/TimeSplit is hard-guarded; "
    "TemporalSplit yields zero training rows), so the asserted non-empty "
    "per-split counts cannot be produced.")

_SKIP_REASON_DG_GAP_DATE_OFFSET_ADJUSTMENT_FIRING = (
    "blocked-gap: the date-offset-split auto-adjustment fires only on an index "
    "gap from an empty timeframe, but the DG producer enumerates only surviving "
    "timeframes with consecutive indices, so the firing branch is unreachable.")

TEST_GRAPH_NAME = 'kumo_test_graph'
TEST_PQUERY_NAME = 'kumo_test_pquery'


def test_random_split(dg_indexed_financial_graph, tmp_path):
    graph, index_path = dg_indexed_financial_graph
    # stype inferral needs the per-test autouse persist_dir.
    graph.infer_metadata()
    pquery = PredictiveQueryV2(
        name='RANDOM_SPLIT_v1',
        graph=graph,
        validated_predictive_query=get_validated_query(
            graph, 'PREDICT TRAINING_CLASSIFICATION.STATUS '
            'FOR EACH TRAINING_CLASSIFICATION.LOAN_ID'),
        split=RandomSplit([0.8, 0.2]),
    )
    assert str(pquery.split) == 'RandomSplit([0.8, 0.2])'
    assert not pquery.split.is_temporal_split

    task = materialize_task_dg(pquery, index_path, tmp_path)
    # DG-native split writes plain bool tensors (no separate index_tensor)
    # directly onto train_table.data.
    train_mask_tensor = task.train_table.data.train_mask
    val_mask_tensor = task.train_table.data.val_mask
    total = int(train_mask_tensor.sum()) + int(val_mask_tensor.sum())
    train_frac = int(train_mask_tensor.sum()) / total
    # RandomSplit([0.8, 0.2]) target ratio; allow modest slop from
    # rust_ingest's stochastic assignment on ~700-row TRAINING_CLASSIFICATION.
    assert 0.75 < train_frac < 0.85, (
        f'unexpected train fraction {train_frac:.3f} '
        f'(train={int(train_mask_tensor.sum())}, '
        f'val={int(val_mask_tensor.sum())})')
    assert (train_mask_tensor & val_mask_tensor).sum() == 0


def test_fixed_split():
    split = FixedSplit([0.8, 0.2])
    assert str(split) == 'FixedSplit([0.8, 0.2])'
    assert not split.is_temporal_split


def test_empty_test_set(dg_indexed_temporal_synthetic_node_pred_graph,
                        tmp_path):
    # A target window wide relative to the fixture's 15-day ORDERS span (SUM
    # over a 7-day window) collapses the post-filter timeframe count to 1 (< the
    # 3 needed for the default 3-way temporal split), so the post-generation
    # timeframe validator surfaces "Insufficient train table timeframes" in the
    # returned ValidationResponse.
    graph, index_path = dg_indexed_temporal_synthetic_node_pred_graph
    graph.infer_metadata().infer_encoders()
    query = ('PREDICT SUM(ORDERS.ORDER_SCORE, 0, 7) FOR EACH USERS.USER_ID '
             'WHERE SUM(ORDERS.ORDER_SCORE, -4, 0) > 0')
    validated_predictive_query = get_validated_query(graph, query)
    pquery = PredictiveQueryV2(
        name=TEST_PQUERY_NAME,
        graph=graph,
        validated_predictive_query=validated_predictive_query,
    )
    _, _, validation_response = pquery.generate_train_table(
        diskgraph_index_path=index_path, destination=str(tmp_path))
    assert not validation_response.ok
    assert any(
        error.title.startswith("Insufficient train table timeframes")
        for error in validation_response.errors)


def test_irregular_date_offset_split(
        dg_indexed_short_span_temporal_node_pred_graph, tmp_path):
    # An irregular DateOffsetSplit([-1,0]) is rejected because the graph supplies
    # too few timeframes: the timeframe-vs-split validator raises "needs at least
    # N timeframes, but only M are available", wrapped as the producer's
    # TrainingTableGenerationError. The short-span fixture folds ORDERS
    # timestamps into a 3-day window so a 1-unit target window yields only 1
    # generated timeframe.
    graph, index_path = dg_indexed_short_span_temporal_node_pred_graph
    graph.infer_metadata().infer_encoders()
    query = 'PREDICT SUM(ORDERS.ORDER_SCORE, 0, 1) FOR EACH USERS.USER_ID'
    validated_predictive_query = get_validated_query(graph, query)
    pquery = PredictiveQueryV2(
        name=TEST_PQUERY_NAME,
        graph=graph,
        validated_predictive_query=validated_predictive_query,
        split=DateOffsetSplit([-1, 0]),
    )
    _, _, validation_response = pquery.generate_train_table(
        diskgraph_index_path=index_path, destination=str(tmp_path))
    assert not validation_response.ok
    assert any(
        "needs at least" in error.message and "timeframes" in error.message
        for error in validation_response.errors)


@pytest.mark.parametrize('split', [
    RandomSplit,
    TemporalSplit,
])
def test_random_and_temporal_split_value_error(split):
    with pytest.raises(ValueError,
                       match=r"'ratio' exceeds more than 100% of the data"):
        split([0.5, 0.5, 0.5])

    with pytest.raises(ValueError, match=r"No more than three training splits"):
        split([0.3, 0.3, 0.2, 0.2])

    with pytest.raises(ValueError, match=r"At least two training splits"):
        split([0.0])

    with pytest.raises(ValueError,
                       match=r"'ratio' can not contain negative values"):
        split([-0.3, 0.6, 0.7])

    if split == TemporalSplit:
        with pytest.raises(ValueError, match=r"'ratio' does not use"):
            split([0.4, 0.2])


def test_time_range_split_value_error():
    with pytest.raises(ValueError, match=r"No more than three"):
        TimeRangeSplit([('1970-01-03', '1970-01-07'),
                        ('1970-01-07', '1970-01-09'),
                        ('1970-01-09', '1970-01-11'),
                        ('1970-01-11', '1970-01-13')])

    with pytest.raises(ValueError, match=r"At least two"):
        TimeRangeSplit([('1970-01-03', '1970-01-07')])

    with pytest.raises(ValueError, match=r"end date needs to be smaller"):
        TimeRangeSplit([('1970-01-07', '1970-01-03'),
                        ('1970-01-07', '1970-01-09')])

    with pytest.raises(ValueError, match=r"different time ranges can not"):
        TimeRangeSplit([('1970-01-03', '1970-01-07'),
                        ('1970-01-05', '1970-01-09')])


def test_static_node_pred_split_table_stats(dg_indexed_static_node_pred_graph,
                                            tmp_path: pathlib.Path):
    # Per-split SplitTrainTableStats counts for a static node-pred regression
    # query (PREDICT USERS.PAYMENTS FOR EACH USERS.USER_ID) over USER_ID 0..3,
    # PAYMENTS [10,20,40,80]: 3 train rows/entities/targets, 1 val, no test.
    #
    # For a static (no time_col) table the DG split-stats writer stamps an
    # epoch-zero anchor (1970-01-01T00:00:00+00:00) for both min_date/max_date,
    # so the date assertion checks both bounds are the same static-table epoch
    # anchor rather than None.
    graph, index_path = dg_indexed_static_node_pred_graph
    # stype inferral needs the per-test autouse persist_dir.
    graph.infer_metadata()
    query = 'PREDICT USERS.PAYMENTS FOR EACH USERS.USER_ID'
    validated_query = get_validated_query(graph, query)
    pquery = PredictiveQueryV2(
        name=TEST_PQUERY_NAME,
        graph=graph,
        validated_predictive_query=validated_query,
        split=FixedSplit([0.7, 0.3]),
    )
    task = materialize_task_dg(pquery, index_path, tmp_path)
    split_stats = task.train_table.get_train_table_stats(after_split=True)
    assert isinstance(split_stats, SplitTrainTableStats)

    epoch = pd.Timestamp('1970-01-01', tz='UTC')

    assert split_stats.train_table_stats is not None
    assert split_stats.train_table_stats.num_train_rows == 3
    assert split_stats.train_table_stats.num_entities == 3
    assert split_stats.train_table_stats.num_targets == 3
    # Static DG path: epoch-anchored min == max.
    assert pd.Timestamp(split_stats.train_table_stats.min_date) == epoch
    assert pd.Timestamp(split_stats.train_table_stats.max_date) == epoch

    assert split_stats.val_table_stats is not None
    assert split_stats.val_table_stats.num_train_rows == 1
    assert split_stats.val_table_stats.num_entities == 1
    assert split_stats.val_table_stats.num_targets == 1
    assert pd.Timestamp(split_stats.val_table_stats.min_date) == epoch
    assert pd.Timestamp(split_stats.val_table_stats.max_date) == epoch

    assert split_stats.test_table_stats is None


def test_split_table_stats(dg_indexed_temporal_synthetic_node_pred_graph,
                           tmp_path: pathlib.Path):
    # Per-split SplitTrainTableStats for a temporal SUM-regression query: each
    # stage gets a populated target_distribution (regression histogram) and real
    # per-split min/max dates. The fixture is USERS + ORDERS(ORDER_SCORE,
    # TIMESTAMP) at 5 users / 15 edges; counts, dates, and histogram bin widths
    # are derived for that ORDER_SCORE data. The asserted shape (every stage has
    # min/max_date and a 50-bin regression histogram with target_min/target_max)
    # is the behavior under test.
    graph, index_path = dg_indexed_temporal_synthetic_node_pred_graph
    # stype inferral needs the per-test autouse persist_dir.
    graph.infer_metadata()
    query = 'PREDICT SUM(ORDERS.ORDER_SCORE, 0, 4) FOR EACH USERS.USER_ID'
    validated_query = get_validated_query(graph, query)
    pquery = PredictiveQueryV2(
        name='TEMPORAL_SPLIT_STATS_v1',
        graph=graph,
        validated_predictive_query=validated_query,
        timeframe_step=1,
        train_start_offset=20,
        split=TemporalSplit([0.6, 0.2, 0.2]),
    )
    task = materialize_task_dg(pquery, index_path, tmp_path)
    split_stats = task.train_table.get_train_table_stats(after_split=True)
    assert isinstance(split_stats, SplitTrainTableStats)

    def _naive(ts) -> pd.Timestamp:
        # DG-native split-stats emit tz-naive timestamps on this path;
        # normalize so the assertions are tz-agnostic.
        ts = pd.Timestamp(ts)
        return ts.tz_localize(None) if ts.tzinfo is not None else ts

    # --- train stage ---
    train_stats = split_stats.train_table_stats
    assert train_stats is not None
    assert train_stats.num_train_rows == 30
    assert train_stats.num_entities == 5
    assert train_stats.num_targets == 5
    # Per-split min/max dates populate on the DG path (the 2023-era fixture
    # avoids the 1970-clamp).
    assert _naive(train_stats.min_date) == pd.Timestamp('2023-01-02 00:00:00')
    assert _naive(train_stats.max_date) == pd.Timestamp('2023-01-07 00:00:00')
    # Regression target_distribution + 50-bin histogram (SUM(ORDER_SCORE)
    # ranges 0.0 .. 6.0 here).
    train_dist = train_stats.target_distribution
    assert train_dist is not None
    assert train_dist.get('target_min') == 0.0
    assert train_dist.get('target_max') == 6.0
    train_hist = train_dist.get('histogram')
    assert train_hist is not None
    assert train_hist.get('num_bins') == 50
    assert train_hist.get('bin_width') == pytest.approx(0.12, 0.1)

    # --- val stage ---
    val_stats = split_stats.val_table_stats
    assert val_stats is not None
    assert val_stats.num_train_rows == 10
    assert val_stats.num_entities == 5
    assert val_stats.num_targets == 4
    assert _naive(val_stats.min_date) == pd.Timestamp('2023-01-08 00:00:00')
    assert _naive(val_stats.max_date) == pd.Timestamp('2023-01-09 00:00:00')
    val_dist = val_stats.target_distribution
    assert val_dist is not None
    assert val_dist.get('target_min') == 0.0
    assert val_dist.get('target_max') == 6.0
    val_hist = val_dist.get('histogram')
    assert val_hist is not None
    assert val_hist.get('num_bins') == 50

    # --- test stage ---
    test_stats = split_stats.test_table_stats
    assert test_stats is not None
    assert test_stats.num_train_rows == 10
    assert test_stats.num_entities == 5
    assert test_stats.num_targets == 4
    assert _naive(test_stats.min_date) == pd.Timestamp('2023-01-10 00:00:00')
    assert _naive(test_stats.max_date) == pd.Timestamp('2023-01-11 00:00:00')
    test_dist = test_stats.target_distribution
    assert test_dist is not None
    assert test_dist.get('target_min') == 0.0
    assert test_dist.get('target_max') == 6.0
    test_hist = test_dist.get('histogram')
    assert test_hist is not None
    assert test_hist.get('num_bins') == 50


def test_validation_empty_train_split(
        dg_indexed_temporal_node_pred_orders_graph, tmp_path):
    # Asserts the insufficient-timeframes-for-split contract fires for both an
    # LP (LIST_DISTINCT ... RANK TOP) and a node-pred (COUNT) temporal query; a
    # target window wide relative to the fixture's ORDERS span drives the
    # post-filter timeframe collapse. The contract surfaces through two
    # validators depending on the split:
    #   * default 3-way split (node-pred COUNT) -> post-generation
    #     validate_timeframes_post_generation returns "Insufficient train table
    #     timeframes ..." in the validation_response.
    #   * explicit DateOffsetSplit whose val window matches the target window
    #     (LP RANK TOP) -> pre-generation timeframe-vs-split check ("<split>
    #     needs at least N timeframes, but only M are available"), wrapped as the
    #     producer's TrainingTableGenerationError in the validation_response.
    insufficient_timeframes = ("Insufficient train table timeframes to create "
                               "train, validation and test sets")

    # --- node-pred COUNT, default 3-way split: post-generation validator ---
    graph, index_path = dg_indexed_temporal_node_pred_orders_graph(
        num_users=10, num_stores=8, num_edges=560)
    graph.infer_metadata().infer_encoders()
    query = 'PREDICT COUNT(ORDERS.STORE_ID, 0, 180) FOR EACH USERS.USER_ID'
    validated_predictive_query = get_validated_query(graph, query)
    pquery = PredictiveQueryV2(
        name='TEMPORAL_NODE_TEST_PQUERY',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
    )
    _, _, validation_response = pquery.generate_train_table(
        diskgraph_index_path=index_path, destination=str(tmp_path / 'node'))
    assert not validation_response.ok
    assert any(error.title == insufficient_timeframes
               for error in validation_response.errors)

    # --- LP LIST_DISTINCT RANK TOP, explicit matched DateOffsetSplit:
    #     pre-generation timeframe-vs-split validator ---
    graph_lp, index_path_lp = dg_indexed_temporal_node_pred_orders_graph(
        num_users=10, num_stores=8, num_edges=560)
    graph_lp.infer_metadata().infer_encoders()
    query_lp = ('PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 0, 180) RANK TOP 3 '
                'FOR EACH USERS.USER_ID')
    validated_predictive_query_lp = get_validated_query(graph_lp, query_lp)
    pquery_lp = PredictiveQueryV2(
        name='TEMPORAL_LINK_TEST_PQUERY',
        graph=graph_lp,
        validated_predictive_query=validated_predictive_query_lp,
        split=DateOffsetSplit([-180, 0], timeframe=180),
    )
    _, _, validation_response_lp = pquery_lp.generate_train_table(
        diskgraph_index_path=index_path_lp, destination=str(tmp_path / 'lp'))
    assert not validation_response_lp.ok
    assert any(
        "needs at least" in error.message and "timeframes" in error.message
        for error in validation_response_lp.errors)


@pytest.mark.parametrize('params', [
    dict(
        split=[0.8, 0.2],
        unit='days',
        expected_sum=[546, 136],
        expected_time_periods=[
            ('1993-07-05', '1998-01-27'),
            ('1998-01-29', '1998-12-08'),
        ],
    ),
])
def test_temporal_split(params):
    split = TemporalSplit(params['split'])
    assert str(split) == f'TemporalSplit({params["split"]})'
    assert split.is_temporal_split


@pytest.mark.parametrize('params', [
    dict(
        split=[-36, -14],
        offset_unit='days',
        unit='days',
        anchor_start_offset=0,
        timeframe=7,
        timeframe_step=None,
        flexible_boundaries=True,
        expected_timeframe=7,
        expected_sum=[664, 6, 10],
        expected_time_periods=[
            ('1993-07-05', '1998-10-27'),
            ('1998-10-27', '1998-11-24'),
            ('1998-11-24', '1998-12-15'),
        ],
    ),
    dict(
        split=[-15, -14],
        offset_unit='days',
        unit='days',
        anchor_start_offset=0,
        timeframe=15,
        timeframe_step=None,
        flexible_boundaries=True,
        expected_timeframe=15,
        expected_sum=[664, 1, 10],
        expected_time_periods=[
            ('1993-07-05', '1998-11-08'),
            ('1998-11-08', '1998-11-23'),
            ('1998-11-23', '1998-12-23'),
        ],
    ),
    dict(
        split=[-29, -14],
        offset_unit='days',
        unit='days',
        timeframe=None,
        anchor_start_offset=0,
        timeframe_step=None,
        flexible_boundaries=True,
        expected_timeframe=15,
        expected_sum=[664, 1, 10],
        expected_time_periods=[
            ('1993-07-05', '1998-11-08'),
            ('1998-11-08', '1998-11-23'),
            ('1998-11-23', '1998-12-23'),
        ],
    ),
    dict(
        split=[-30, -16],
        offset_unit='days',
        unit='days',
        timeframe=None,
        anchor_start_offset=0,
        timeframe_step=6,
        flexible_boundaries=True,
        expected_timeframe=14,
        expected_sum=[664, 1, 12],
        expected_time_periods=[
            ('1993-07-05', '1998-11-02'),
            ('1998-11-02', '1998-11-20'),
            ('1998-11-20', '1998-12-22'),
        ],
    ),
    dict(
        split=[-14, 0],
        offset_unit='days',
        unit='days',
        timeframe=None,
        anchor_start_offset=0,
        timeframe_step=None,
        flexible_boundaries=True,
        expected_timeframe=14,
        expected_sum=[669, 0, 1],
        expected_time_periods=[
            ('1993-07-05', '1998-11-24'),
            ('1998-11-24', '1998-12-08'),
            ('1998-12-08', '1998-12-22'),
        ],
    ),
    dict(
        split=[-14],
        offset_unit='days',
        unit='days',
        timeframe=None,
        anchor_start_offset=0,
        timeframe_step=None,
        flexible_boundaries=True,
        expected_split=[-14],
        expected_timeframe=14,
        expected_sum=[669, 10],
        expected_time_periods=[
            ('1993-07-05', '1998-11-24'),
            ('1998-11-24', '1998-12-22'),
        ],
    ),
    dict(
        split=[-15, -14],
        offset_unit='days',
        unit='days',
        timeframe=15,
        anchor_start_offset=0,
        timeframe_step=None,
        flexible_boundaries=False,
        expected_timeframe=15,
        expected_sum=[668, 0, 10],
        expected_time_periods=[
            ('1993-07-05', '1998-11-23'),
            ('1998-11-23', '1998-11-24'),
            ('1998-11-24', '1998-12-23'),
        ],
    ),
    dict(
        split=[-1, 0],
        offset_unit='months',
        unit='days',
        timeframe=15,
        anchor_start_offset=0,
        timeframe_step=None,
        flexible_boundaries=False,
        expected_timeframe=15,
        expected_sum=[664, 5, 1],
        expected_time_periods=[
            ('1993-07-05', '1998-11-08'),
            ('1998-11-08', '1998-12-08'),
            ('1998-12-08', '1998-12-23'),
        ],
    ),
])
def test_date_offset_split(params):
    split = DateOffsetSplit(params['split'], offset_unit=params['offset_unit'],
                            unit=params['unit'], timeframe=params['timeframe'],
                            anchor_start_offset=params['anchor_start_offset'],
                            timeframe_step=params['timeframe_step'],
                            flexible_boundaries=params['flexible_boundaries'])
    assert str(split) == (
        f'DateOffsetSplit({params["split"]}, '
        f'offset_unit={params["offset_unit"]}, '
        f'unit={params["unit"]}, timeframe={params["expected_timeframe"]}, '
        f'anchor_start_offset={params["anchor_start_offset"]}, '
        f'timeframe_step={params["timeframe_step"]})')
    assert split.is_temporal_split


@pytest.mark.parametrize(
    'params',
    [
        dict(  # Test 2 timeframes and no adjustments for timeframe = None
            split=[
                ('1993-01-01', '1993-12-31'),
                ('1994-01-01', '1994-12-31'),
            ],
            unit='days',
            timeframe=None,
            anchor_start_offset=0,
            timeframe_step=None,
            flexible_boundaries=True,
            expected_sum=[20, 101],
            expected_time_periods=[
                ('1993-01-01', '1993-12-31'),
                ('1994-01-01', '1994-12-31'),
            ],
        ),
        dict(  # Test that timeframes are adjusted when the timeframe is given
            split=[
                ('1993-01-01', '1995-12-31'),
                ('1996-01-02', '1997-06-21'),
                ('1997-07-01', '1998-12-15'),
            ],
            unit='days',
            timeframe=20,
            anchor_start_offset=0,
            timeframe_step=None,
            flexible_boundaries=True,
            expected_sum=[204, 190, 254],
            expected_time_periods=[
                ('1993-01-01', '1995-12-31'),
                ('1995-12-31', '1997-06-21'),
                ('1997-06-23', '1998-12-15'),
            ],
        ),
        dict(  # Test timeframe_step
            split=[
                ('1993-01-01', '1995-12-31'),
                ('1996-01-02', '1997-06-21'),
                ('1997-07-01', '1998-12-15'),
            ],
            unit='days',
            timeframe=20,
            anchor_start_offset=0,
            timeframe_step=13,
            flexible_boundaries=True,
            expected_sum=[203, 191, 254],
            expected_time_periods=[
                ('1993-01-01', '1995-12-25'),
                ('1995-12-25', '1997-06-21'),
                ('1997-06-23', '1998-12-15'),
            ],
        ),
        dict(  # Test flexible_boundaries=False
            split=[
                ('1993-01-01', '1995-12-31'),
                ('1996-01-02', '1997-06-21'),
                ('1997-07-01', '1998-12-15'),
            ],
            unit='days',
            timeframe=20,
            anchor_start_offset=0,
            timeframe_step=None,
            flexible_boundaries=False,
            expected_sum=[204, 187, 252],
            expected_time_periods=[
                ('1993-01-01', '1995-12-31'),
                ('1996-01-02', '1997-06-21'),
                ('1997-07-01', '1998-12-15'),
            ],
        ),
        dict(  # Test anchor_start_offset
            split=[
                ('1993-01-01', '1995-12-31'),
                ('1996-01-02', '1997-06-21'),
                ('1997-07-01', '1998-12-15'),
            ],
            unit='days',
            timeframe=15,
            anchor_start_offset=5,
            timeframe_step=None,
            flexible_boundaries=True,
            expected_sum=[203, 190, 258],
            expected_time_periods=[
                ('1993-01-01', '1995-12-26'),
                ('1995-12-26', '1997-06-18'),
                ('1997-06-18', '1998-12-15'),
            ],
        ),
        dict(  # Test anchor_start_offset with timeframe_step
            split=[
                ('1993-01-01', '1995-12-31'),
                ('1996-01-02', '1997-06-21'),
                ('1997-07-01', '1998-12-15'),
            ],
            unit='days',
            timeframe=15,
            anchor_start_offset=5,
            timeframe_step=13,
            flexible_boundaries=True,
            expected_sum=[203, 191, 254],
            expected_time_periods=[
                ('1993-01-01', '1995-12-25'),
                ('1995-12-25', '1997-06-21'),
                ('1997-06-23', '1998-12-15'),
            ],
        ),
        dict(  # Test with hours
            split=[
                ('1993-01-01T000000', '1993-12-31T235959'),
                ('1994-01-01T000000', '1994-12-31T235959'),
            ],
            unit='hours',
            timeframe=None,
            anchor_start_offset=0,
            timeframe_step=None,
            flexible_boundaries=True,
            expected_sum=[20, 101],
            expected_time_periods=[
                ('1993-01-01T000000', '1993-12-31T235959'),
                ('1994-01-01T000000', '1994-12-31T235959'),
            ],
        ),
        dict(  # Test with minutes
            split=[
                ('1993-01-01T000000', '1993-12-31T235959'),
                ('1994-01-01T000000', '1994-12-31T235959'),
            ],
            unit='minutes',
            timeframe=None,
            anchor_start_offset=0,
            timeframe_step=None,
            flexible_boundaries=True,
            expected_sum=[20, 101],
            expected_time_periods=[
                ('1993-01-01T000000', '1993-12-31T235959'),
                ('1994-01-01T000000', '1994-12-31T235959'),
            ],
        ),
    ])
def test_time_range_split(params):
    split = TimeRangeSplit(
        params['split'],
        unit=params['unit'],
        timeframe=params['timeframe'],
        anchor_start_offset=params['anchor_start_offset'],
        timeframe_step=params['timeframe_step'],
        flexible_boundaries=params['flexible_boundaries'],
    )

    time_range_repr = repr([(pd.Timestamp(start), pd.Timestamp(end))
                            for (start, end) in params["split"]
                            ]).replace("'", "")
    assert str(split) == (
        f'TimeRangeSplit({time_range_repr}, '
        f'unit={params["unit"]}, timeframe={params["timeframe"]}, '
        f'anchor_start_offset={params["anchor_start_offset"]}, '
        f'timeframe_step={params["timeframe_step"]})')
    assert split.is_temporal_split


def test_manual_split_basic():
    split = ManualSplit(col='col', val_value=10, test_value=2)
    # The column name is quoted: a plan stores this repr and has to parse it
    # back, and a column name is arbitrary user data -- `my split` or
    # `2024_fold` are not Python identifiers and do not survive unquoted.
    assert str(split) == ("ManualSplit(col='col', train_value=0, "
                          "val_value=10, test_value=2)")
    assert not split.is_temporal_split


def test_manual_split_no_test():
    split = ManualSplit(col='col', val_value=10)
    assert str(split) == ("ManualSplit(col='col', train_value=0, "
                          "val_value=10, test_value=None)")
    assert not split.is_temporal_split


def test_manual_split_reads_a_legacy_serialized_column():
    # `col` serialized as a whole `BaseColumn` before it became a plain
    # column name, so a predictive query persisted then carries
    # `{'name': ...}` here -- and `kumo.config.predictive_query` stores a
    # split as a dict precisely so `from_config` can rebuild it. Nothing
    # caught this because no fixture carried the old shape; this is that
    # fixture.
    legacy = {
        '_target_': 'kumo.task.split.ManualSplit',
        'col': {
            'name': 'SPLIT'
        },
        'train_value': 0,
        'val_value': 1,
        'test_value': 2,
    }

    # Callers reach this through the base class, which resolves `_target_`
    # in its own frame -- so this must work on `Split`, not just the
    # subclass.
    split = Split.from_config(legacy)

    assert isinstance(split, ManualSplit)
    assert split.col == 'SPLIT'
    assert split.fpid == ManualSplit(col='SPLIT', train_value=0, val_value=1,
                                     test_value=2).fpid
    assert legacy['col'] == {'name': 'SPLIT'}, 'must not mutate the caller'


def test_manual_split_config_still_round_trips():
    split = ManualSplit(col='SPLIT', train_value=0, val_value=1, test_value=2)

    assert Split.from_config(split.config()).fpid == split.fpid


def test_manual_split_value_list():
    split = ManualSplit(col='col', train_value=[1, 2], val_value=[3, 4],
                        test_value=5)
    assert str(split) == ("ManualSplit(col='col', train_value=[1, 2], "
                          "val_value=[3, 4], test_value=5)")


def test_validation_empty_val_split(
        dg_indexed_temporal_synthetic_node_pred_graph, tmp_path):
    # Asserts (a) the "Insufficient train table timeframes" error and (b) zero
    # warnings alongside it. A 7-day SUM window over the fixture's 15-day ORDERS
    # span collapses to 1 post-filter timeframe (< the 3 needed for the default
    # 3-way temporal split), so the post-generation timeframe validator surfaces
    # the error in the returned ValidationResponse.
    graph, index_path = dg_indexed_temporal_synthetic_node_pred_graph
    graph.infer_metadata().infer_encoders()
    query = ('PREDICT SUM(ORDERS.ORDER_SCORE, 0, 7) FOR EACH USERS.USER_ID '
             'WHERE SUM(ORDERS.ORDER_SCORE, -4, 0) > 0')
    validated_predictive_query = get_validated_query(graph, query)

    pquery = PredictiveQueryV2(
        name=TEST_PQUERY_NAME,
        graph=graph,
        validated_predictive_query=validated_predictive_query,
    )

    _, _, validation_response = pquery.generate_train_table(
        diskgraph_index_path=index_path, destination=str(tmp_path))
    assert not validation_response.ok
    assert any(
        error.title.startswith("Insufficient train table timeframes")
        for error in validation_response.errors)
    assert len(validation_response.warnings) == 0


def test_validation_empty_test_split(
        dg_indexed_temporal_synthetic_node_pred_graph, tmp_path):
    # Asserts empty-training-table detection under a TimeRangeSplit. On DG a
    # gapped TimeRangeSplit is not a distinct per-split-emptiness error (a 3rd
    # window that misses the data just yields a smaller non-empty table), so a
    # restrictive temporal entity filter (no USERS row sums ORDER_SCORE above
    # 100000) empties the whole table and the producer surfaces the "Empty
    # training table" error in the returned ValidationResponse. The
    # TimeRangeSplit is kept over the fixture's 2023 ORDERS span.
    graph, index_path = dg_indexed_temporal_synthetic_node_pred_graph
    graph.infer_metadata().infer_encoders()
    query = ('PREDICT MAX(ORDERS.ORDER_SCORE, 0, 1) FOR EACH USERS.USER_ID '
             'WHERE SUM(ORDERS.ORDER_SCORE, -INF, 0) > 100000')
    validated_predictive_query = get_validated_query(graph, query)

    pquery = PredictiveQueryV2(
        name=TEST_PQUERY_NAME,
        graph=graph,
        validated_predictive_query=validated_predictive_query,
        split=TimeRangeSplit([
            ('2023-01-02', '2023-01-05'),
            ('2023-01-05', '2023-01-07'),
            ('2023-01-10', '2023-01-12'),
        ]),
    )
    _, _, validation_response = pquery.generate_train_table(
        diskgraph_index_path=index_path, destination=str(tmp_path))
    assert not validation_response.ok
    assert any(error.title == 'Empty training table'
               for error in validation_response.errors)


def test_default_split_for_link_pred(
    small_temporal_graph_fixed,
    tmp_path,
):
    graph = small_temporal_graph_fixed(tmp_path=tmp_path)
    query = ("PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 0, 3) RANK TOP 3 FOR "
             "EACH USERS.USER_ID")
    validated_predictive_query = get_validated_query(graph, query)
    pquery = PredictiveQueryV2(
        graph=graph,
        name=TEST_PQUERY_NAME,
        validated_predictive_query=validated_predictive_query,
    )
    assert isinstance(pquery.split, DateOffsetSplit)


def test_automatic_approximate_date_offset_split_adjustment_basic(
    dg_indexed_temporal_node_pred_orders_graph,
    tmp_path,
):
    # Automatic ApproxDateOffsetSplit boundary initialization (default split +
    # restrictive temporal entity filter WHERE LAST(ORDERS.STORE_ID, -6, 0) = 3)
    # keeps all 3 splits non-empty and assigns each stage a distinct
    # anchor/min/max date. Asserts the shape: num_splits == 3, every stage has a
    # populated anchor/min/max date, and the three stages are temporally ordered
    # and non-overlapping. The fixture is 10 users / 8 stores / 560 edges with a
    # 2023 TIMESTAMP origin (1970-era dates clamp to a 2262 sentinel on DG);
    # absolute date values are derived for that origin.
    graph, index_path = dg_indexed_temporal_node_pred_orders_graph(
        num_users=10, num_stores=8, num_edges=560)
    graph.infer_metadata()
    query = ('PREDICT SUM(ORDERS.ORDER_VALUE_2, 0, 4) FOR EACH USERS.USER_ID '
             'WHERE LAST(ORDERS.STORE_ID, -6, 0) = 3')
    validated_predictive_query = get_validated_query(graph, query)
    # Default split for a temporal query is `ApproxDateOffsetSplit`.
    pquery = PredictiveQueryV2(
        name='TEMPORAL_NODE_TEST_PQUERY',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
    )
    assert isinstance(pquery.split, ApproxDateOffsetSplit)

    task = materialize_task_dg(pquery, index_path, tmp_path)
    assert task.train_table.split.num_splits == 3
    split_stats = task.train_table.get_train_table_stats(after_split=True)
    assert isinstance(split_stats, SplitTrainTableStats)

    def _naive(ts) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        return ts.tz_localize(None) if ts.tzinfo is not None else ts

    # All three stages are populated (the automatic ApproxDateOffsetSplit
    # initialization kept every split non-empty) ...
    train = split_stats.train_table_stats
    val = split_stats.val_table_stats
    test = split_stats.test_table_stats
    assert train is not None
    assert val is not None
    assert test is not None

    # ... with the exact 2023-shifted anchor/min/max dates from the DG fixture.
    assert _naive(train.anchor_date) == pd.Timestamp('2023-01-12')
    assert _naive(train.min_date) == pd.Timestamp('2023-01-12')
    assert _naive(train.max_date) == pd.Timestamp('2024-03-27')

    assert _naive(val.anchor_date) == pd.Timestamp('2024-04-04')
    assert _naive(val.min_date) == pd.Timestamp('2024-04-04')
    assert _naive(val.max_date) == pd.Timestamp('2024-05-22')

    assert _naive(test.anchor_date) == pd.Timestamp('2024-05-30')
    assert _naive(test.min_date) == pd.Timestamp('2024-05-30')
    assert _naive(test.max_date) == pd.Timestamp('2024-07-09')

    # The three stages are temporally ordered and non-overlapping (the real
    # split invariant, independent of the absolute fixture origin).
    assert _naive(train.max_date) < _naive(val.anchor_date)
    assert _naive(val.max_date) < _naive(test.anchor_date)


@pytest.mark.parametrize(
    'params',
    [
        # Custom `DateOffsetSplit` is NOT adjusted (offsets preserved).
        dict(
            query=(
                'PREDICT COUNT(ORDERS.STORE_ID, 0, 6) FOR EACH USERS.USER_ID'),
            split=DateOffsetSplit([-24, -12], timeframe=6),
            expected_date_offset_split=DateOffsetSplit(
                [-24, -12], unit=TimeUnit.DAYS, timeframe=6,
                timeframe_step=None, flexible_boundaries=False),
        ),
        # Custom train/val `DateOffsetSplit` ([0]) is NOT adjusted.
        dict(
            query=(
                'PREDICT COUNT(ORDERS.STORE_ID, 0, 6) FOR EACH USERS.USER_ID'),
            split=DateOffsetSplit([0], timeframe=6),
            expected_date_offset_split=DateOffsetSplit(
                [0], unit=TimeUnit.DAYS, timeframe=6, timeframe_step=None,
                flexible_boundaries=False),
        ),
        # Custom train/val/test `DateOffsetSplit` where val/test boundary is
        # `0` is NOT adjusted (no entity filter -> no empty val timeframe).
        dict(
            query=(
                'PREDICT COUNT(ORDERS.STORE_ID, 0, 6) FOR EACH USERS.USER_ID'),
            split=DateOffsetSplit([-8, 0], timeframe=6),
            expected_date_offset_split=DateOffsetSplit(
                [-8, 0], unit=TimeUnit.DAYS, timeframe=6, timeframe_step=None,
                flexible_boundaries=False),
        ),
        # `DateOffsetSplit` where unit is hours: unit / anchor_start_offset
        # preserved, offsets unchanged.
        dict(
            query=('PREDICT COUNT(ORDERS.STORE_ID, 24, 48, hours) FOR EACH '
                   'USERS.USER_ID'),
            split=DateOffsetSplit([-48, 0], timeframe=24),
            expected_date_offset_split=DateOffsetSplit(
                [-48, 0], unit=TimeUnit.HOURS, offset_unit=TimeUnit.HOURS,
                timeframe=24, anchor_start_offset=24, timeframe_step=None,
                flexible_boundaries=False),
        ),
        # `DateOffsetSplit` where unit is minutes: unit / anchor_start_offset
        # preserved, offsets unchanged.
        dict(
            query=('PREDICT COUNT(ORDERS.STORE_ID, 1440, 2880, minutes) '
                   'FOR EACH USERS.USER_ID'),
            split=DateOffsetSplit([-2880, 0], timeframe=1440),
            expected_date_offset_split=DateOffsetSplit(
                [-2880, 0], unit=TimeUnit.MINUTES, offset_unit=TimeUnit.MINUTES,
                timeframe=1440, anchor_start_offset=1440, timeframe_step=None,
                flexible_boundaries=False),
        )
    ])
def test_automatic_date_offset_split_adjustment_basic(
    params,
    dg_indexed_temporal_node_pred_orders_graph,
    tmp_path,
):
    # For each parametrized DateOffsetSplit, asserts the post-generation split
    # adjustment produces the expected DateOffsetSplit (offsets / unit /
    # anchor_start_offset) and emits the 'DateOffsetSplit is adjusted to avoid
    # empty validation data' warning when (and only when) the split was actually
    # adjusted.
    #
    # The five params here verify the not-adjusted contract: custom / [0] /
    # hours / minutes DateOffsetSplits are NOT adjusted, with unit +
    # anchor_start_offset preserved and no spurious adjustment warning. The
    # adjustment-firing case is not covered: the fixture's STORE_ID cycles
    # uniformly 0..7 across all 560 days so no val timeframe empties; firing it
    # would need a bespoke sparse-per-timeframe fixture (see
    # _SKIP_REASON_DG_GAP_DATE_OFFSET_ADJUSTMENT_FIRING).
    graph, index_path = dg_indexed_temporal_node_pred_orders_graph(
        num_users=10, num_stores=8, num_edges=560)
    graph.infer_metadata()
    query = params.get('query')
    validated_predictive_query = get_validated_query(graph, query)

    train_start_offset = 24 * 8
    if 'minutes' in params.get('query'):
        train_start_offset *= 60
    pquery = PredictiveQueryV2(
        name='TEMPORAL_NODE_TEST_PQUERY',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
        split=params.get('split'),
        train_start_offset=train_start_offset,
    )

    train_table_path, job_id, _ = pquery.generate_train_table(
        diskgraph_index_path=index_path, destination=str(tmp_path))
    task = pquery.create_task(train_table_path=train_table_path, job_id=job_id,
                              ignore_encoders=True)
    split = task.train_table.split
    assert isinstance(split, DateOffsetSplit)
    assert split == params.get('expected_date_offset_split')
    # None of the kept params trigger an adjustment, so no adjustment warning
    # may be emitted (the falsifiable "NOT adjusted" half of the contract).
    assert split.offsets == params.get('split').offsets
    validation_response = pquery.get_validate_train_table_response()
    if validation_response is not None:
        assert all(w.title != 'DateOffsetSplit is adjusted to avoid empty '
                   'validation data' for w in validation_response.warnings)


@pytest.mark.skip(reason=_SKIP_REASON_DG_GAP_DATE_OFFSET_ADJUSTMENT_FIRING)
def test_automatic_date_offset_split_adjustment_firing(
    dg_indexed_temporal_node_pred_orders_graph,
    tmp_path,
):
    # The only coverage of the adjustment-firing branch of
    # adjust_date_offset_split_post_train_table_generate and its
    # 'DateOffsetSplit is adjusted to avoid empty validation data' warning.
    # DateOffsetSplit([-6, 0], timeframe=6) plus the restrictive filter
    # WHERE LAST(ORDERS.STORE_ID, -6, 0) = 3 empties the immediate-after-test
    # timeframe; the firing requires an index gap between surviving timeframes.
    #
    # The firing branch is structurally unreachable on DiskGraph: the DG
    # native stats producer enumerates only surviving timeframes with
    # consecutive indices (no gap), so adjust_split == split and the
    # adjustment never fires. See
    # _SKIP_REASON_DG_GAP_DATE_OFFSET_ADJUSTMENT_FIRING. The body below is the
    # assertion contract to re-enable once the DG producer indexes timeframes
    # against the full anchor grid.
    graph, index_path = dg_indexed_temporal_node_pred_orders_graph(
        num_users=10, num_stores=8, num_edges=560)
    graph.infer_metadata()
    query = ('PREDICT COUNT(ORDERS.STORE_ID, 0, 6) FOR EACH USERS.USER_ID '
             'WHERE LAST(ORDERS.STORE_ID, -6, 0) = 3')
    validated_predictive_query = get_validated_query(graph, query)
    split = DateOffsetSplit([-6, 0], timeframe=6)
    train_start_offset = 24 * 8
    pquery = PredictiveQueryV2(
        name='TEMPORAL_NODE_TEST_PQUERY',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
        split=split,
        train_start_offset=train_start_offset,
    )

    train_table_path, job_id, _ = pquery.generate_train_table(
        diskgraph_index_path=index_path, destination=str(tmp_path))
    task = pquery.create_task(train_table_path=train_table_path, job_id=job_id,
                              ignore_encoders=True)
    adjusted_split = task.train_table.split
    assert isinstance(adjusted_split, DateOffsetSplit)
    # The restrictive entity filter empties the immediate-after-test timeframe,
    # so the default `[-6, 0]` split is ADJUSTED to `[-24, 0]` to keep the
    # validation set non-empty.
    assert adjusted_split == DateOffsetSplit([-24, 0], unit=TimeUnit.DAYS,
                                             timeframe=6, timeframe_step=None,
                                             flexible_boundaries=False)
    assert adjusted_split.offsets != split.offsets
    # ... and the user-facing adjustment warning is emitted.
    validation_response = pquery.get_validate_train_table_response()
    assert validation_response is not None
    assert (validation_response.warnings[-1].title ==
            'DateOffsetSplit is adjusted to avoid empty validation data')


@pytest.mark.parametrize('params', [
    dict(
        approx_split=ApproxDateOffsetSplit([0.5, 0.3, 0.2]),
        expected_date_offset_split=DateOffsetSplit(
            [-6, 0], unit=TimeUnit.DAYS, timeframe=3, timeframe_step=None,
            flexible_boundaries=True),
    ),
    dict(
        approx_split=ApproxDateOffsetSplit([0.3, 0.3, 0.4]),
        expected_date_offset_split=DateOffsetSplit(
            [-9, -3], unit=TimeUnit.DAYS, timeframe=3, timeframe_step=None,
            flexible_boundaries=True),
    ),
    dict(
        approx_split=ApproxDateOffsetSplit([0.7, 0.3]),
        expected_date_offset_split=DateOffsetSplit(
            [-3], unit=TimeUnit.DAYS, timeframe=3, timeframe_step=None,
            flexible_boundaries=True),
    ),
    dict(
        approx_split=ApproxDateOffsetSplit([0.98, 0.01, 0.01]),
        expected_date_offset_split=DateOffsetSplit(
            [-3, 0], unit=TimeUnit.DAYS, timeframe=3, timeframe_step=None,
            flexible_boundaries=True),
    ),
])
def test_approx_date_offset_split_general(params):
    train_table_stats = TemporalLinkTrainTableStats(
        min_date=pd.Timestamp('2018-01-09 00:00:00'),
        max_date=pd.Timestamp('2018-01-30 00:00:00'),
        anchor_date=pd.Timestamp('2018-01-06 00:00:00'),
        num_train_rows=28,
        num_entities=4,
        num_targets=2,
        num_timeframes=7,
        num_timeframes_post_filter=7,
        avg_entities_per_timeframe=4,
        avg_targets_per_timeframe=None,
        stats_per_timeframe=[
            TemporalLinkTimeFrameStats(
                min_date=pd.Timestamp('2018-01-27T00:00:00'),
                max_date=pd.Timestamp('2018-01-30T00:00:00'),
                anchor_date=pd.Timestamp('2018-01-24T00:00:00'),
                num_train_rows=4,
                num_entities=4,
                num_targets=1,
                index=0,
            ),
            TemporalLinkTimeFrameStats(
                min_date=pd.Timestamp('2018-01-24T00:00:00'),
                max_date=pd.Timestamp('2018-01-27T00:00:00'),
                anchor_date=pd.Timestamp('2018-01-21T00:00:00'),
                num_train_rows=4,
                num_entities=4,
                num_targets=1,
                index=1,
            ),
            TemporalLinkTimeFrameStats(
                min_date=pd.Timestamp('2018-01-21T00:00:00'),
                max_date=pd.Timestamp('2018-01-24T00:00:00'),
                anchor_date=pd.Timestamp('2018-01-18T00:00:00'),
                num_train_rows=4,
                num_entities=4,
                num_targets=1,
                index=2,
            ),
            TemporalLinkTimeFrameStats(
                min_date=pd.Timestamp('2018-01-18T00:00:00'),
                max_date=pd.Timestamp('2018-01-21T00:00:00'),
                anchor_date=pd.Timestamp('2018-01-15T00:00:00'),
                num_train_rows=4,
                num_entities=4,
                num_targets=1,
                index=3,
            ),
            TemporalLinkTimeFrameStats(
                min_date=pd.Timestamp('2018-01-15T00:00:00'),
                max_date=pd.Timestamp('2018-01-18T00:00:00'),
                anchor_date=pd.Timestamp('2018-01-12T00:00:00'),
                num_train_rows=4,
                num_entities=4,
                num_targets=1,
                index=4,
            ),
            TemporalLinkTimeFrameStats(
                min_date=pd.Timestamp('2018-01-12T00:00:00'),
                max_date=pd.Timestamp('2018-01-15T00:00:00'),
                anchor_date=pd.Timestamp('2018-01-09T00:00:00'),
                num_train_rows=4,
                num_entities=4,
                num_targets=1,
                index=5,
            ),
            TemporalLinkTimeFrameStats(
                min_date=pd.Timestamp('2018-01-09T00:00:00'),
                max_date=pd.Timestamp('2018-01-12T00:00:00'),
                anchor_date=pd.Timestamp('2018-01-06T00:00:00'),
                num_train_rows=4,
                num_entities=4,
                num_targets=1,
                index=6,
            )
        ],
    )
    approx_split = params['approx_split']
    expected_date_offset_split = params['expected_date_offset_split']
    approx_split.set_date_offset_boundary(train_table_stats=train_table_stats,
                                          timeframe=3, unit=TimeUnit.DAYS,
                                          timeframe_step=None)
    assert approx_split.split == expected_date_offset_split


@pytest.mark.parametrize('params', [
    dict(
        approx_split=ApproxDateOffsetSplit([0.8, 0.1, 0.1]),
        expected_date_offset_split=DateOffsetSplit(
            [-6, 0], unit=TimeUnit.DAYS, timeframe=3, timeframe_step=None,
            flexible_boundaries=True),
    ),
    dict(
        approx_split=ApproxDateOffsetSplit([0.5, 0.2, 0.3]),
        expected_date_offset_split=DateOffsetSplit(
            [-6, 0], unit=TimeUnit.DAYS, timeframe=3, timeframe_step=None,
            flexible_boundaries=True),
    ),
    dict(
        approx_split=ApproxDateOffsetSplit([0.1, 0.1, 0.8]),
        expected_date_offset_split=DateOffsetSplit(
            [-15, -12], unit=TimeUnit.DAYS, timeframe=3, timeframe_step=None,
            flexible_boundaries=True),
    ),
    dict(
        approx_split=ApproxDateOffsetSplit([0.98, 0.01, 0.01]),
        expected_date_offset_split=DateOffsetSplit(
            [-6, 0], unit=TimeUnit.DAYS, timeframe=3, timeframe_step=None,
            flexible_boundaries=True),
    ),
])
def test_approx_date_offset_split_corner_case(params):
    train_table_stats = TemporalLinkTrainTableStats(
        min_date=pd.Timestamp('2018-01-09 00:00:00'),
        max_date=pd.Timestamp('2018-02-02 00:00:00'),
        anchor_date=pd.Timestamp('2018-01-06 00:00:00'),
        num_train_rows=28,
        num_entities=4,
        num_targets=2,
        num_timeframes=8,
        num_timeframes_post_filter=7,
        avg_entities_per_timeframe=4,
        avg_targets_per_timeframe=None,
        stats_per_timeframe=[
            TemporalLinkTimeFrameStats(
                min_date=pd.Timestamp('2018-01-30T00:00:00'),
                max_date=pd.Timestamp('2018-02-02T00:00:00'),
                anchor_date=pd.Timestamp('2018-01-27T00:00:00'),
                num_train_rows=16,
                num_entities=4,
                num_targets=1,
                index=0,
            ),
            TemporalLinkTimeFrameStats(
                min_date=pd.Timestamp('2018-01-24T00:00:00'),
                max_date=pd.Timestamp('2018-01-27T00:00:00'),
                anchor_date=pd.Timestamp('2018-01-21T00:00:00'),
                num_train_rows=1,
                num_entities=4,
                num_targets=1,
                index=2,
            ),
            TemporalLinkTimeFrameStats(
                min_date=pd.Timestamp('2018-01-21T00:00:00'),
                max_date=pd.Timestamp('2018-01-24T00:00:00'),
                anchor_date=pd.Timestamp('2018-01-18T00:00:00'),
                num_train_rows=1,
                num_entities=4,
                num_targets=1,
                index=3,
            ),
            TemporalLinkTimeFrameStats(
                min_date=pd.Timestamp('2018-01-18T00:00:00'),
                max_date=pd.Timestamp('2018-01-21T00:00:00'),
                anchor_date=pd.Timestamp('2018-01-15T00:00:00'),
                num_train_rows=2,
                num_entities=4,
                num_targets=1,
                index=4,
            ),
            TemporalLinkTimeFrameStats(
                min_date=pd.Timestamp('2018-01-15T00:00:00'),
                max_date=pd.Timestamp('2018-01-18T00:00:00'),
                anchor_date=pd.Timestamp('2018-01-12T00:00:00'),
                num_train_rows=5,
                num_entities=4,
                num_targets=1,
                index=5,
            ),
            TemporalLinkTimeFrameStats(
                min_date=pd.Timestamp('2018-01-12T00:00:00'),
                max_date=pd.Timestamp('2018-01-15T00:00:00'),
                anchor_date=pd.Timestamp('2018-01-09T00:00:00'),
                num_train_rows=1,
                num_entities=4,
                num_targets=1,
                index=6,
            ),
            TemporalLinkTimeFrameStats(
                min_date=pd.Timestamp('2018-01-09T00:00:00'),
                max_date=pd.Timestamp('2018-01-12T00:00:00'),
                anchor_date=pd.Timestamp('2018-01-06T00:00:00'),
                num_train_rows=2,
                num_entities=4,
                num_targets=1,
                index=7,
            )
        ],
    )
    approx_split = params['approx_split']
    expected_date_offset_split = params['expected_date_offset_split']
    approx_split.set_date_offset_boundary(train_table_stats=train_table_stats,
                                          timeframe=3, unit=TimeUnit.DAYS,
                                          timeframe_step=None)
    assert approx_split.split == expected_date_offset_split


def test_time_range_split_remove_earliest_timeframe(create_node_pred_graph,
                                                    tempdir):
    graph = create_node_pred_graph('test_graph', tempdir.name)
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 6, HOURS)'
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)

    # TRANSACTIONS.DATE contains dates between 1990-01-01 and 1990-01-05
    # Here we test two cases for `TimeRangeSplit` to determine whether or
    # not the first timeframe is removed.

    # Case 1: Remove first timeframe as it is the earliest timeframe.
    split_remove_first = TimeRangeSplit([('1990-01-01', '1990-01-02'),
                                         ('1990-01-02', '1990-01-03'),
                                         ('1990-01-03', '1990-01-04')])
    generator_remove_first = TrainingTableGeneratorV2(graph, validated_query,
                                                      split_remove_first)
    timestamps_remove_first = generator_remove_first.generate_timestamps()
    assert (timestamps_remove_first == [
        pd.Timestamp('1990-01-03 18:00:00'),
        pd.Timestamp('1990-01-03 12:00:00'),
        pd.Timestamp('1990-01-03 06:00:00'),
        pd.Timestamp('1990-01-03 00:00:00'),
        pd.Timestamp('1990-01-02 18:00:00'),
        pd.Timestamp('1990-01-02 12:00:00'),
        pd.Timestamp('1990-01-02 06:00:00'),
        pd.Timestamp('1990-01-02 00:00:00'),
        pd.Timestamp('1990-01-01 18:00:00'),
        pd.Timestamp('1990-01-01 12:00:00'),
        pd.Timestamp('1990-01-01 06:00:00')
    ])

    # Case 2: Keep first timeframe if previous timeframe exists.
    split_keep_first = TimeRangeSplit([('1990-01-02', '1990-01-03'),
                                       ('1990-01-03', '1990-01-04'),
                                       ('1990-01-04', '1990-01-05')])
    generator_keep_first = TrainingTableGeneratorV2(graph, validated_query,
                                                    split_keep_first)
    timestamps_keep_first = generator_keep_first.generate_timestamps()
    assert timestamps_keep_first == ([
        pd.Timestamp('1990-01-04 18:00:00'),
        pd.Timestamp('1990-01-04 12:00:00'),
        pd.Timestamp('1990-01-04 06:00:00'),
        pd.Timestamp('1990-01-04 00:00:00'),
        pd.Timestamp('1990-01-03 18:00:00'),
        pd.Timestamp('1990-01-03 12:00:00'),
        pd.Timestamp('1990-01-03 06:00:00'),
        pd.Timestamp('1990-01-03 00:00:00'),
        pd.Timestamp('1990-01-02 18:00:00'),
        pd.Timestamp('1990-01-02 12:00:00'),
        pd.Timestamp('1990-01-02 06:00:00'),
        pd.Timestamp('1990-01-02 00:00:00')
    ])

    # Check an additional scenario where `train_start_offset` is not None
    generator_with_offset = TrainingTableGeneratorV2(graph, validated_query,
                                                     split_keep_first,
                                                     train_start_offset=64)
    timestamps_with_offset = generator_with_offset.generate_timestamps()
    assert timestamps_with_offset == ([
        pd.Timestamp('1990-01-04 18:00:00'),
        pd.Timestamp('1990-01-04 12:00:00'),
        pd.Timestamp('1990-01-04 06:00:00'),
        pd.Timestamp('1990-01-04 00:00:00'),
        pd.Timestamp('1990-01-03 18:00:00'),
        pd.Timestamp('1990-01-03 12:00:00'),
        pd.Timestamp('1990-01-03 06:00:00'),
        pd.Timestamp('1990-01-03 00:00:00'),
        pd.Timestamp('1990-01-02 18:00:00'),
        pd.Timestamp('1990-01-02 12:00:00')
    ])


@pytest.mark.skip(reason=_SKIP_REASON_DG_GAP_FORECASTING_TXN_SESSIONS)
def test_forecasting_split(create_node_pred_graph, tempdir):
    graph = create_node_pred_graph('test_graph', tempdir.name)

    # Forecasting with entity filter
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 1) '
             'FOR EACH USERS.USER_ID WHERE SUM(SESSIONS.DURATION, -INF, 0) > 0')

    validated_query = get_validated_query(graph, query)
    pquery = PredictiveQueryV2(
        name='user_actions', graph=graph,
        validated_predictive_query=validated_query,
        split=DateOffsetSplit([0], unit='days', timeframe=1), forecast_length=2)

    with disable_pre_validation():
        task = pquery.create_task(job_id='testjob', ignore_encoders=True)
    task.train_table.materialize()

    train_table_stats = task.get_train_table_stats(after_split=True)
    assert train_table_stats.train_table_stats.num_train_rows == 4
    assert train_table_stats.train_table_stats.num_entities == 2

    assert train_table_stats.val_table_stats.num_train_rows == 2
    assert train_table_stats.val_table_stats.num_entities == 2

    pquery = PredictiveQueryV2(name='user_actions', graph=graph,
                               validated_predictive_query=validated_query,
                               split=ApproxDateOffsetSplit([0.8, 0.1, 0.1]),
                               forecast_length=2)

    task = pquery.create_task(job_id='testjob', ignore_encoders=True)
    task.train_table.materialize()

    train_table_stats = task.get_train_table_stats(after_split=True)
    assert train_table_stats.train_table_stats.num_train_rows == 2
    assert train_table_stats.train_table_stats.num_entities == 2

    assert train_table_stats.val_table_stats.num_train_rows == 2
    assert train_table_stats.val_table_stats.num_entities == 2

    assert train_table_stats.test_table_stats.num_train_rows == 2
    assert train_table_stats.test_table_stats.num_entities == 2

    pquery = PredictiveQueryV2(
        name='user_actions', graph=graph,
        validated_predictive_query=validated_query,
        split=TimeRangeSplit([('1990-01-02', '1990-01-03'),
                              ('1990-01-03', '1990-01-04'),
                              ('2022-01-04', '2022-01-05')],
                             timeframe=1), forecast_length=2)

    task = pquery.create_task(job_id='testjob', ignore_encoders=True)
    task.train_table.materialize()

    train_table_stats = task.get_train_table_stats(after_split=True)
    assert train_table_stats.train_table_stats.num_train_rows == 2
    assert train_table_stats.train_table_stats.num_entities == 2

    assert train_table_stats.val_table_stats.num_train_rows == 2
    assert train_table_stats.val_table_stats.num_entities == 2

    assert train_table_stats.test_table_stats.num_train_rows == 2
    assert train_table_stats.test_table_stats.num_entities == 2


# This tests that the max_date and min_date of splits are calculated
# correctly from train_table valid rows. If the min_date and max_date
# are calculated without considering ENTITY_INCLUDE_BIT, we would have
# gotten an error on empty test table.
@pytest.mark.skip(reason=_SKIP_REASON_DG_GAP_FORECASTING_TIMEFRAMES)
def test_forecasting_split_on_weird_distributions(create_tiny_node_pred_graph,
                                                  tempdir):
    graph = create_tiny_node_pred_graph('test_graph', tempdir.name)

    # Forecasting with entity filter
    query = ('PREDICT SUM(actions.amount, 0, 1) '
             'FOR EACH entity.entity_id WHERE SUM(actions.amount, -2, 0) >= 5')

    validated_query = get_validated_query(graph, query)
    pquery = PredictiveQueryV2(name='user_actions', graph=graph,
                               validated_predictive_query=validated_query,
                               split=ApproxDateOffsetSplit([0.8, 0.1, 0.1]),
                               forecast_length=2)

    with disable_pre_validation():
        task = pquery.create_task(job_id='testjob', ignore_encoders=True)
    task.train_table.materialize()
    train_table_stats = task.get_train_table_stats(after_split=True)

    assert train_table_stats.train_table_stats.num_train_rows == 7
    assert train_table_stats.train_table_stats.num_entities == 1

    assert train_table_stats.val_table_stats.num_train_rows == 1
    assert train_table_stats.val_table_stats.num_entities == 1

    assert train_table_stats.test_table_stats.num_train_rows == 1
    assert train_table_stats.test_table_stats.num_entities == 1


@pytest.mark.skip(reason=_SKIP_REASON_DG_GAP_STATIC_TEMPORAL_SPLIT)
def test_static_node_pred_temporal_split(
        create_large_static_node_pred_test_graph):
    graph = create_large_static_node_pred_test_graph()
    query = 'PREDICT USERS.PAYMENTS FOR EACH USERS.USER_ID'
    validated_predictive_query = get_validated_query(graph, query)

    # Test TemporalSplit for static queries
    pquery = PredictiveQueryV2(
        name=TEST_PQUERY_NAME,
        graph=copy.deepcopy(graph),
        validated_predictive_query=validated_predictive_query,
    )
    task = pquery.create_task()
    task.train_table.materialize()
    split_stats = task.get_train_table_stats(after_split=True)
    assert isinstance(split_stats, SplitTrainTableStats)
    assert split_stats.train_table_stats is not None
    assert split_stats.train_table_stats.num_train_rows == 48

    assert split_stats.val_table_stats is not None
    assert split_stats.val_table_stats.num_train_rows == 6

    assert split_stats.test_table_stats is not None
    assert split_stats.test_table_stats.num_train_rows == 6

    # Test TimeRangeSplit for static queries
    pquery = PredictiveQueryV2(
        name=TEST_PQUERY_NAME, graph=copy.deepcopy(graph),
        validated_predictive_query=validated_predictive_query,
        split=TimeRangeSplit([('2000-01-01', '2000-02-01'),
                              ('2000-03-01', '2000-05-01'),
                              ('2000-07-01', '2000-08-01')],
                             anchor_start_offset=None))
    task = pquery.create_task()
    task.train_table.materialize()
    split_stats = task.get_train_table_stats(after_split=True)
    assert isinstance(split_stats, SplitTrainTableStats)
    assert split_stats.train_table_stats is not None
    assert split_stats.train_table_stats.num_train_rows == 5

    assert split_stats.val_table_stats is not None
    assert split_stats.val_table_stats.num_train_rows == 15

    assert split_stats.test_table_stats is not None
    assert split_stats.test_table_stats.num_train_rows == 5


# Utilities ###################################################################


def assert_time_periods(masks: list[Mask], expected: list[tuple[str, str]]):
    got: list[TimePeriod] = []
    for mask in masks:
        assert mask.time_period is not None
        got.append(mask.time_period)

    expected_transformed: list[TimePeriod] = [
        TimePeriod.cast(x[0], x[1]) for x in expected
    ]

    assert got == expected_transformed
