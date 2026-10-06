# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pandas as pd
from kumo_core.types import TaskType
from kumoapi.pquery import ValidatedPredictiveQuery
from kumoapi.typing import TimeUnit
from kumopql.parser import PQLParser
from kumopql.validator import PredictiveQueryValidator

from kumo.graph.graph import Graph
from kumo.pquery.pquery_validators import TrainTableValidator
from kumo.pquery.predictive_query import PredictiveQueryV2
from kumo.pquery.utils import validator_kwargs
from kumo.stats.train_table_stats import TemporalNodeTimeFrameStats
from kumo.task.split import (
    ApproxDateOffsetSplit,
    BacktestingSplit,
    DateOffsetSplit,
    RandomSplit,
)
from kumo.testing.pquery import get_validated_query

_SKIP_REASON_LEGACY_TRAIN_TABLE_MATERIALIZE = (
    "blocked-gap: drives train-table generation without a DiskGraph index, so "
    "it raises UnsupportedInDiskgraphError; needs a rebuild on a DG-indexed "
    "graph fixture that threads a real diskgraph_index_path.")

# On TT-NG the train-table gen (``execute_tt_gen``) reports an empty multipart
# dataset when an entity filter selects zero rows. The
# kumo-side empty-table guard (``train_table_generator.py``
# ``_diskgraph_empty_train_table_error_message`` / ``_DISKGRAPH_ZERO_ROWS_MARKERS``)
# recognises that deterministic empty signal and reclassifies it into a graceful
# "Empty training table" ``TrainingTableGenerationError``, which
# ``generate_train_table`` surfaces as a non-ok ``ValidationResponse`` (never
# the raw DG error). The empty-entity tests below assert that clean outcome.


def get_mock_stats_manager(graph: Graph, query: ValidatedPredictiveQuery,
                           task_type=TaskType.MULTICLASS_CLASSIFICATION,
                           **kwargs) -> MagicMock:
    mock_stats = MagicMock(num_entities=0, num_target_rows=0, min_date=0,
                           max_date=0, num_timeframes=0,
                           num_timeframes_post_filter=0,
                           avg_entities_per_timeframe=0, num_train_rows=0,
                           num_targets=0)
    for key, val in kwargs.items():
        setattr(mock_stats, key, val)
    stats_manager = MagicMock(graph=graph, query=query, task_type=task_type,
                              get_train_table_stats=lambda: mock_stats)
    return stats_manager


def test_train_table_generation_validation_many_categories(
        create_node_pred_graph, tempdir):
    query = 'PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 2) > 0 FOR EACH USERS.USER_ID'
    graph = create_node_pred_graph('test_graph', tempdir.name)
    validated_query = get_validated_query(graph, query)
    stats_manager = get_mock_stats_manager(
        graph, validated_query,
        target_distribution={x: 1 / 10001
                             for x in range(10001)}, num_targets=10001)
    split = DateOffsetSplit([-2, 0], unit=TimeUnit.DAYS, timeframe=2)
    validator = TrainTableValidator(stats_manager, split)

    response = validator.validate_post_train_table_generate()

    # test for 1. entity table is too small 2. train table is too small
    # 3. num_classes is too large
    validate_train_table_errors = response.errors
    validate_train_table_warnings = response.warnings
    assert len(validate_train_table_errors) == 2

    insufficient_timeframes = ('Insufficient train table timeframes to '
                               'create train, validation and test sets')
    too_many_classes = 'Too many classes to predict'

    assert any(error.title == insufficient_timeframes
               for error in validate_train_table_errors)
    assert any(error.title == too_many_classes
               for error in validate_train_table_errors)
    assert len(validate_train_table_warnings) == 0


def test_train_table_generation_validation_single_target(
        create_node_pred_graph, tempdir):
    query = 'PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 2) > 0 FOR EACH USERS.USER_ID'
    graph = create_node_pred_graph('test_graph', tempdir.name)
    validated_query = get_validated_query(graph, query)
    stats_manager = get_mock_stats_manager(
        graph, validated_query, target_distribution={
            'count': {
                'True': 1
            },
            'percentage': {
                'True': 100.0
            }
        }, num_targets=1, num_train_rows=1)
    split = DateOffsetSplit([-2, 0], unit=TimeUnit.DAYS, timeframe=2)
    validator = TrainTableValidator(stats_manager, split)

    response = validator.validate_post_train_table_generate()

    # Test training table only has one type of target value in
    # classification problems
    insufficient_timeframes = (
        "Insufficient train table timeframes to create train, "
        "validation and test sets")
    too_few_classes = "Too few classes to predict"
    assert any(error.title == insufficient_timeframes
               for error in response.errors)
    assert any(error.title == too_few_classes for error in response.errors)


def test_train_table_generation_validation_no_positive_samples(
        create_node_pred_graph, tempdir):
    query = 'PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 2) > 0 FOR EACH USERS.USER_ID'
    graph = create_node_pred_graph('test_graph', tempdir.name)
    validated_query = get_validated_query(graph, query)
    stats_manager = get_mock_stats_manager(
        graph, validated_query, target_distribution={
            'count': {},
            'percentage': {}
        }, num_targets=1, num_train_rows=1)
    split = DateOffsetSplit([-2, 0], unit=TimeUnit.DAYS, timeframe=2)
    validator = TrainTableValidator(stats_manager, split)

    response = validator.validate_post_train_table_generate()

    insufficient_timeframes = (
        "Insufficient train table timeframes to create train, "
        "validation and test sets")
    no_positive_samples = "No positive samples"
    assert any(error.title == insufficient_timeframes
               for error in response.errors)
    assert any(error.title == no_positive_samples for error in response.errors)


def test_train_table_generation_validation_large_RHS(create_node_pred_graph,
                                                     tempdir):
    query = ('PREDICT LIST_DISTINCT(TRANSACTIONS.BANK_ID, 0, 2) '
             'RANK TOP 10 FOR EACH USERS.USER_ID')
    graph = create_node_pred_graph('test_graph', tempdir.name)
    validated_query = get_validated_query(graph, query)
    stats_manager = get_mock_stats_manager(
        graph,
        validated_query,
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        num_train_rows=10,
        # Array-target LP (the only LP regime under TT-NG) sizes the table by
        # ``num_links`` rather than ``num_train_rows`` in validate_table_size;
        # give it a real int so the size guard runs instead of comparing a
        # MagicMock against MIN_TRAIN_ROWS.
        num_links=10,
        num_targets=10**8 + 5,
        num_timeframes=10,
        num_timeframes_post_filter=3,
    )
    split = DateOffsetSplit([-2, 0], unit=TimeUnit.DAYS, timeframe=2)
    validator = TrainTableValidator(stats_manager, split)

    response = validator.validate_post_train_table_generate()

    # Test for there are train table timeframes being completely removed due
    # to restrictive temporal entity filter
    assert any(error.title == 'Target column contains too many unique values'
               and "The maximum supported number of unique values in the "
               "link prediction target column is 10000000, while parquet.BANKS "
               "contains 100000005" in error.message
               for error in response.errors)


def test_topk_larger_than_num_targets(create_node_pred_graph, tempdir):
    query = ('PREDICT LIST_DISTINCT(TRANSACTIONS.CATEGORY, 0, 2) RANK TOP 10'
             'FOR EACH USERS.USER_ID ')
    # Test that the number of targets is less than the topk
    graph = create_node_pred_graph('test_graph', tempdir.name)
    validated_query = get_validated_query(graph, query)
    stats_manager = get_mock_stats_manager(graph, validated_query,
                                           num_targets=2)
    split = DateOffsetSplit([-2, 0], unit=TimeUnit.DAYS, timeframe=2)
    validator = TrainTableValidator(stats_manager, split)
    response = validator.validate_post_train_table_generate()
    # The first error is insufficient data for the split
    assert len(response.errors) == 2
    assert any(error.title == 'Top k is too large' for error in response.errors)


def test_static_node_empty_entity_table(dg_indexed_synthetic_node_pred_graph,
                                        tmp_path):
    # Drive the production DG producer directly with the real index. A
    # restrictive entity filter that selects no rows materializes zero training
    # rows. On TT-NG the empty case surfaces as a 0-byte ``tt.parquet`` that the
    # DG demux rejects ("Parquet file too small. Size is 0"); the kumo-side
    # empty-table guard reclassifies that deterministic empty signal into a
    # graceful "Empty training table" validation error (see
    # ``_diskgraph_empty_train_table_error_message`` +
    # ``_DISKGRAPH_ZERO_ROWS_MARKERS``), so ``generate_train_table`` returns a
    # non-ok ``ValidationResponse`` rather than raising the raw DG error.
    graph, index_path = dg_indexed_synthetic_node_pred_graph
    graph.infer_metadata().infer_encoders()

    # static training table is empty due to a restrictive entity filter that
    # excludes every USERS row (USER_ID range is 0..3).
    query = ('PREDICT USERS_TARGET.STATUS FOR EACH USERS.USER_ID '
             'WHERE USERS.USER_ID > 3')
    validated_predictive_query = get_validated_query(graph, query)
    pquery = PredictiveQueryV2(
        name='static_node_v1',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
    )
    _, _, response = pquery.generate_train_table(
        diskgraph_index_path=index_path, destination=str(tmp_path / 'v1'))
    assert not response.ok
    assert any(error.title == 'Empty training table'
               for error in response.errors)

    # second restrictive entity filter (PAYMENTS lives on USERS; no row has
    # PAYMENTS > 1000) -> empty training table again.
    query = ('PREDICT USERS_TARGET.STATUS FOR EACH USERS.USER_ID WHERE '
             'USERS.PAYMENTS > 1000')
    validated_predictive_query = get_validated_query(graph, query)
    pquery = PredictiveQueryV2(
        name='static_node_v2',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
    )
    _, _, response = pquery.generate_train_table(
        diskgraph_index_path=index_path, destination=str(tmp_path / 'v2'))
    assert not response.ok
    assert any(error.title == 'Empty training table'
               for error in response.errors)


def test_temporal_node_empty_entity_table(
        dg_indexed_temporal_synthetic_node_pred_graph, tmp_path):
    # Drive the production DG producer directly with the real index. A strict
    # temporal entity filter that selects no rows materializes zero training
    # rows. On TT-NG the empty case surfaces as a 0-byte ``tt.parquet`` that the
    # DG demux rejects ("Parquet file too small. Size is 0"); the kumo-side
    # empty-table guard reclassifies that deterministic empty signal into a
    # graceful "Empty training table" validation error, so
    # ``generate_train_table`` returns a non-ok ``ValidationResponse`` rather
    # than raising the raw DG error.
    graph, index_path = dg_indexed_temporal_synthetic_node_pred_graph
    graph.infer_metadata().infer_encoders()

    # Strict temporal entity filter: no USERS row has accumulated an
    # ORDER_SCORE sum above 1000 (ORDER_SCORE values are 1..3), so the entity
    # set is empty after filtering -> empty training table.
    query = ('PREDICT SUM(ORDERS.ORDER_SCORE, 0, 2) FOR EACH USERS.USER_ID '
             'WHERE SUM(ORDERS.ORDER_SCORE, -INF, 0) > 1000')
    validated_query = get_validated_query(graph, query)
    pquery = PredictiveQueryV2(
        name='empty_entity_table',
        graph=graph,
        validated_predictive_query=validated_query,
    )
    _, _, response = pquery.generate_train_table(
        diskgraph_index_path=index_path, destination=str(tmp_path / 'empty'))
    assert not response.ok
    assert any(error.title == 'Empty training table'
               for error in response.errors)


def test_static_link_time_col_raise_error(create_lp_graph, tempdir):
    graph = create_lp_graph(root_dir=tempdir.name, set_trans_time_col=True)
    query = ('predict LIST_DISTINCT(TRANS.PRODUCT_ID) RANK TOP 1 for each '
             'USERS.USER_ID')
    parser = PQLParser()
    validator = PredictiveQueryValidator(**validator_kwargs(graph))
    parsed_query = parser.to_parsed_predictive_query(query)
    _, response = validator.validate_predictive_query(parsed_query)
    assert not response.ok
    assert any(error.title == 'Time range required for LIST_DISTINCT with '
               'time column' for error in response.errors)

    graph = create_lp_graph(root_dir=tempdir.name, set_trans_time_col=False)
    query = ('predict LIST_DISTINCT(TRANS.PRODUCT_ID) RANK TOP 1 for each '
             'USERS.USER_ID')
    parser = PQLParser()
    validator = PredictiveQueryValidator(**validator_kwargs(graph))
    parsed_query = parser.to_parsed_predictive_query(query)
    _, response = validator.validate_predictive_query(parsed_query)
    assert not response.ok
    assert any(error.title == 'Unexpected time columns in tables.'
               for error in response.errors)


def test_temporal_link_pred_validate_generate_train_table(
        dg_indexed_temporal_link_pred_graph, tmp_path):
    # Drive the production DG producer directly with the real index. Each
    # sub-block keeps a restrictive filter that materializes zero training rows;
    # under TT-NG the way that surfaces depends on WHICH side is emptied.
    graph, index_path = dg_indexed_temporal_link_pred_graph(
        num_users=5, num_stores=3, num_edges=15)
    graph.infer_metadata().infer_encoders()

    # RHS target filter excludes every candidate (STORE_ID range is 0..2). The
    # LHS entity pool is still non-empty, so gen succeeds, but the RANK TOP 10
    # request now exceeds the (zero) available target count -> the validator
    # returns "Top k is too large" rather than the empty-table title (an empty
    # RHS is a top-k mismatch on TT-NG, not an empty training table).
    query = ('PREDICT LIST_DISTINCT(ORDERS.STORE_ID WHERE '
             'ORDERS.STORE_ID > 100, 0, 2) RANK TOP 10 FOR EACH '
             'USERS.USER_ID')
    validated_predictive_query = get_validated_query(graph, query)
    pquery = PredictiveQueryV2(
        name='link_prediction_v1',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
    )
    _, _, validation_response = pquery.generate_train_table(
        diskgraph_index_path=index_path, destination=str(tmp_path / 'lp1'))
    assert not validation_response.ok
    assert validation_response.errors[0].title == 'Top k is too large'

    # Entity filter excludes every USERS row (USER_ID range is 0..4) -> empty
    # LHS seed table. On TT-NG the empty case surfaces as a 0-byte ``tt.parquet``
    # that the DG demux rejects ("Parquet file too small. Size is 0"); the
    # kumo-side empty-table guard reclassifies that deterministic empty signal
    # into a graceful "Empty training table" validation error, so
    # ``generate_train_table`` returns a non-ok ``ValidationResponse``.
    query = ('PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 0, 2) RANK TOP 10 '
             'FOR EACH USERS.USER_ID WHERE USERS.USER_ID > 100')
    validated_predictive_query = get_validated_query(graph, query)
    pquery = PredictiveQueryV2(
        name='link_prediction_v2',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
    )
    _, _, validation_response = pquery.generate_train_table(
        diskgraph_index_path=index_path, destination=str(tmp_path / 'lp2'))
    assert not validation_response.ok
    assert any(error.title == 'Empty training table'
               for error in validation_response.errors)

    # Empty train table due to a large target-aggregation window that runs past
    # the data, combined with a train_start_offset that leaves no valid
    # timestamps -- the pre-generate validate_pre_train_table_generate surface.
    query = ('PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 0, 45) RANK TOP 10 '
             'FOR EACH USERS.USER_ID')
    validated_predictive_query = get_validated_query(graph, query)
    pquery = PredictiveQueryV2(
        name='temporal_link_pred_empty_table_v3',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
        train_start_offset=10,
    )
    table_generator = pquery.table_generator
    response = table_generator.validate_pre_train_table_generate(
        skip_timestamp_validation=False)
    assert response.errors[0].title == 'Empty training table'


def test_validate_temporal_regression_target_values(create_node_pred_graph,
                                                    tempdir):
    query = ("PREDICT SUM(TRANSACTIONS.AMOUNT WHERE "
             "TRANSACTIONS.AMOUNT <= 10, 0, 1) FOR EACH USERS.USER_ID")
    graph = create_node_pred_graph('test_graph', tempdir.name)
    validated_query = get_validated_query(graph, query)
    stats_manager = get_mock_stats_manager(
        graph,
        validated_query,
        num_targets=1,
        target_distribution={"target_mode": 0},
        num_train_rows=10,
        num_timeframes=3,
        num_timeframes_post_filter=3,
        task_type=TaskType.REGRESSION,
    )
    split = ApproxDateOffsetSplit([0.8, 0.1, 0.1])
    validator = TrainTableValidator(stats_manager, split)
    response = validator.validate_post_train_table_generate()
    assert response.errors[0].title == "Too few unique target values to predict"


def test_validate_ttng_forecasting_skips_scalar_target_values(
        create_node_pred_graph, tempdir, monkeypatch):
    query = ("PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 2) "
             "FORECAST 3 TIMEFRAMES FOR EACH USERS.USER_ID")
    graph = create_node_pred_graph('test_graph', tempdir.name)
    validated_query = get_validated_query(graph, query)
    stats_manager = get_mock_stats_manager(
        graph,
        validated_query,
        task_type=TaskType.FORECASTING,
        num_targets=0,
        num_train_rows=2000,
        num_entities=200,
        num_timeframes=3,
        num_timeframes_post_filter=3,
    )
    split = DateOffsetSplit([-2, 0], unit=TimeUnit.DAYS, timeframe=2)
    validator = TrainTableValidator(stats_manager, split)

    # Forecasting now UNCONDITIONALLY skips the scalar unique-target-values
    # check (validate_post_train_table_generate:
    # ``skip_scalar_target_check = task_type == FORECASTING``), so a
    # forecasting stats payload with num_targets=0 no longer trips the
    # ``validate_num_unique_target_values`` assertion regardless of the
    # KUMO_DISKGRAPH_TT_NG_ENABLED env var (TT-NG is the only path). The
    # env-gated fall-through that used to raise AssertionError is gone.
    monkeypatch.setenv('KUMO_DISKGRAPH_TT_NG_ENABLED', 'true')
    response = validator.validate_post_train_table_generate()
    assert not response.errors

    monkeypatch.delenv('KUMO_DISKGRAPH_TT_NG_ENABLED')
    response = validator.validate_post_train_table_generate()
    assert not response.errors


def test_validate_static_regression_target_values(
        create_static_node_pred_graph):
    graph = create_static_node_pred_graph()
    query = ("PREDICT USERS.PAYMENTS FOR EACH USERS.USER_ID WHERE "
             "USERS.PAYMENTS <= 10")
    validated_query = get_validated_query(graph, query)
    stats_manager = get_mock_stats_manager(
        graph,
        validated_query,
        num_targets=1,
        target_distribution={"target_mode": 0},
        num_train_rows=3,
        num_total_rows=3,
        task_type=TaskType.REGRESSION,
    )
    split = RandomSplit([0.8, 0.1, 0.1])
    validator = TrainTableValidator(stats_manager, split)
    response = validator.validate_post_train_table_generate()
    assert response.errors[0].title == "Too few unique target values to predict"


def test_validate_task_type_for_mlc_with_binary_column(
    create_node_pred_graph,
    tempdir,
):
    query = ('PREDICT LIST_DISTINCT(TRANSACTIONS.TYPE, 0, 30) RANK TOP 20'
             'FOR EACH USERS.USER_ID '
             'WHERE COUNT(TRANSACTIONS.*, -10, 0) > 0')
    graph = create_node_pred_graph('test_graph', tempdir.name)
    validated_query = get_validated_query(graph, query)
    stats_manager = get_mock_stats_manager(
        graph,
        validated_query,
        min_date=pd.Timestamp('2020-01-01'),
        max_date=pd.Timestamp('2020-02-02'),
    )
    split = BacktestingSplit('2019-03-03')
    validator = TrainTableValidator(
        stats_manager,
        split,
        timeframe_step=pd.DateOffset(days=2),
    )
    validation_response = validator.validate_task_type(
        TemporalNodeTimeFrameStats(
            min_date=pd.Timestamp('2024-01-01'),
            max_date=pd.Timestamp('2024-01-01'),
            anchor_date=pd.Timestamp('2024-01-01'),
            target_distribution={
                'a': 0.7,
                'b': 0.3
            },
            num_targets=2,
        ))
    assert validation_response is not None
    assert validation_response.warnings is not None
    assert len(validation_response.warnings) == 1
    warning = validation_response.warnings[0]
    assert warning.message.startswith("Query is multiclass")
    assert warning.title.startswith("Multiclass classification")
