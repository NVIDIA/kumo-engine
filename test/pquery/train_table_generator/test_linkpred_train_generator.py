# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import pathlib
from test.fixtures.dg_indexed_graph import (
    generate_train_table_dg,
    materialize_selected_task_dg,
    materialize_task_dg,
)
from typing import List
from unittest.mock import patch

import pandas as pd
import pytest
from kumo_core.types import TaskType
from kumoapi.pquery import QueryType
from kumoapi.typing import Dtype, TimeUnit
from kumopql.parser import PQLParser
from kumopql.validator import PredictiveQueryValidator

from kumo.connector.parquet_connector import ParquetConnector
from kumo.graph.graph import Graph
from kumo.pquery.predictive_query import PredictiveQueryV2
from kumo.pquery.train_table_generator import (
    FREQUENCY_COL_NAME,
    TrainingTableGeneratorV2,
)
from kumo.pquery.utils import validator_kwargs
from kumo.task.split import DateOffsetSplit, RandomSplit, TemporalSplit
from kumo.testing.pquery import get_validated_query

NANOSECONDS_PER_SECOND = 1_000_000_000

_SKIP_REASON_DG_WEIGHT_COL = (
    "blocked-gap: weighted training tables raise UnsupportedInDiskgraphError "
    "because DiskGraphPlanCompiler has no weight parameters; needs DG Rust "
    "weight plumbing.")


@pytest.mark.parametrize('aggr', ['LIST_DISTINCT'])  # TODO Support LAST/FIRST
@pytest.mark.parametrize('static', [False])  # TODO Add True once supported
def test_task_type_temporal_lp(
    aggr,
    static,
    create_lp_graph,
    tempdir,
):
    time_range = '' if static else ', 0, 30'
    query = (f'PREDICT {aggr}(TRANS.PRODUCT_ID_FK '
             f'WHERE PRODUCTS.BRAND = "PATAGONIA"{time_range}) '
             f'RANK TOP 10 FOR EACH USERS.USER_ID WHERE USERS.AGE > 21')
    graph = create_lp_graph(root_dir=tempdir.name, num_entities=3,
                            key_name_match=False)
    validated_query = get_validated_query(graph, query)
    split = DateOffsetSplit([-30, 0], unit=TimeUnit.DAYS, timeframe=30)
    generator = TrainingTableGeneratorV2(graph, validated_query, split)
    task_type = generator.task_type
    assert task_type == TaskType.TEMPORAL_LINK_PREDICTION
    if static:
        assert generator.query_type == QueryType.STATIC
    else:
        assert generator.query_type == QueryType.TEMPORAL


@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=False)
def test_small_temporal_link_pred_train_table_stats(
        dg_indexed_temporal_link_pred_graph, tmp_path: pathlib.Path):
    # Drives a RANK-TOP-3 LP query over the DG-indexed
    # `dg_indexed_temporal_link_pred_graph` (5 users / 3 stores / 15 edges,
    # 2023 origin) through the DG-native manifest path (`materialize_task_dg`).
    #
    # COUNT parity (origin-invariant): 12 train rows, 4 entities, 3 targets,
    # 6 timeframes. DATE asserts are computed for the 2023 origin. A 30-day
    # default timeframe over-windows this 14-day span (ZERO anchor windows ->
    # "split stats artifact must include non-empty train_table_stats"), so the
    # split uses a 2-day cadence, producing the 12-row / 6-timeframe table.
    #
    # `node_degree_stats` is not surfaced on the DG-native pre-split aggregate
    # path, so the node-degree block is asserted-None. Bounded DG v2 omits the
    # former per-timeframe diagnostic expansion.
    graph, index_path = dg_indexed_temporal_link_pred_graph()
    # stype inferral needs the per-test autouse persist_dir.
    graph.infer_metadata()
    query = ('PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 0, 2) RANK TOP 3 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)
    split = DateOffsetSplit([-2, 0], unit=TimeUnit.DAYS, timeframe=2)
    pquery = PredictiveQueryV2(
        name='SMALL_TEMPORAL_LP_STATS_v1',
        graph=graph,
        validated_predictive_query=validated_query,
        split=split,
    )
    materialize_task_dg(pquery, index_path, tmp_path)
    table_stats = pquery.get_train_table_stats()
    assert table_stats.min_date == pd.Timestamp('2023-01-03 00:00:00', tz='UTC')
    assert table_stats.max_date == pd.Timestamp('2023-01-13 00:00:00', tz='UTC')
    # Dense (entity x timeframe) grid: 5 entities x 6 timeframes = 30 rows.
    assert table_stats.num_train_rows == 30
    # num_entities / num_targets asserted in the skipped
    # ..._entity_target_counts companion below.
    assert table_stats.num_timeframes == 6
    assert table_stats.num_timeframes_post_filter == 6
    # node_degree_stats is not populated on the DG-native aggregate path.
    assert table_stats.node_degree_stats is None
    # Bounded DG v2 stats intentionally omit the diagnostic per-timeframe
    # expansion. Aggregate timeframe counts and split-stage bounds above remain
    # authoritative for training; the omitted series only powered UI charts.
    assert table_stats.stats_per_timeframe is None


@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=False)
def test_small_temporal_link_pred_train_table_stats_entity_target_counts(  # noqa: E501
        dg_indexed_temporal_link_pred_graph, tmp_path: pathlib.Path):
    # Companion to test_small_temporal_link_pred_train_table_stats pinning the
    # CORRECT intended whole-table counts. True values: 5 distinct entities and
    graph, index_path = dg_indexed_temporal_link_pred_graph()
    graph.infer_metadata()
    query = ('PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 0, 2) RANK TOP 3 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)
    split = DateOffsetSplit([-2, 0], unit=TimeUnit.DAYS, timeframe=2)
    pquery = PredictiveQueryV2(
        name='SMALL_TEMPORAL_LP_STATS_v1',
        graph=graph,
        validated_predictive_query=validated_query,
        split=split,
    )
    materialize_task_dg(pquery, index_path, tmp_path)
    table_stats = pquery.get_train_table_stats()
    assert table_stats.num_entities == 5
    assert table_stats.num_targets == 3


@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=True)
def test_small_temporal_link_pred_train_table_stats_array(
        dg_indexed_temporal_link_pred_graph, tmp_path: pathlib.Path):
    # Array-target temporal link-pred train-table-stats (see the non-array
    # sibling for the timeframe-2 split + 2023 date recompute).
    #
    # COUNT NOTE: in array mode the DG-native train table materializes the DENSE
    # (entity x timeframe) grid -- one row per active entity per timeframe,
    # INCLUDING empty-target rows -- giving 30 train rows (5 entities x
    # 6 timeframes), 30 links, 5 entities. The values are read straight off the
    # DG manifest, so the underlying LINK STRUCTURE (8 non-empty rows /
    # 12 real links) is re-asserted via the train-table readback below.
    graph, index_path = dg_indexed_temporal_link_pred_graph()
    # stype inferral needs the per-test autouse persist_dir.
    graph.infer_metadata()
    query = ('PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 0, 2) RANK TOP 3 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)
    split = DateOffsetSplit([-2, 0], unit=TimeUnit.DAYS, timeframe=2)
    pquery = PredictiveQueryV2(
        name='SMALL_TEMPORAL_LP_STATS_ARRAY_v1',
        graph=graph,
        validated_predictive_query=validated_query,
        split=split,
    )
    train_table_path, job_id, _ = generate_train_table_dg(
        pquery, diskgraph_index_path=index_path, destination=str(tmp_path))
    # Read back the array train table to re-assert the real link structure
    # (the dense grid has 8 non-empty target rows carrying 12 total links).
    train_df = pd.read_parquet(train_table_path)
    num_non_empty = int(sum(len(t) > 0 for t in train_df['TARGET1']))
    num_real_links = int(sum(len(t) for t in train_df['TARGET1']))
    assert num_non_empty == 8
    assert num_real_links == 12

    task = pquery.create_task(train_table_path=train_table_path,
                              train_table_name='train_table', job_id=job_id,
                              ignore_encoders=True)
    task.train_table.materialize(execute_split=False,
                                 calculate_split_stats=False,
                                 skip_compute_stats=True)
    from test.fixtures.dg_indexed_graph import (
        _apply_ttng_native_split_from_bundle,
    )
    staging_dir = str(pathlib.Path(train_table_path).resolve().parent)
    _apply_ttng_native_split_from_bundle(task.train_table, staging_dir,
                                         task.task_type)

    table_stats = pquery.get_train_table_stats()
    assert table_stats.min_date == pd.Timestamp('2023-01-03 00:00:00', tz='UTC')
    assert table_stats.max_date == pd.Timestamp('2023-01-13 00:00:00', tz='UTC')
    # DG-native dense-grid aggregate counts (see COUNT NOTE above).
    assert table_stats.num_train_rows == 30
    assert table_stats.num_links == 12
    # num_entities / num_targets asserted in the skipped
    # ..._entity_target_counts companion below.
    assert table_stats.num_timeframes == 6
    assert table_stats.num_timeframes_post_filter == 6
    # node_degree_stats is not populated on the DG-native aggregate path.
    assert table_stats.node_degree_stats is None
    # Bounded DG v2 stats omit the diagnostic UI timeframe series.
    assert table_stats.stats_per_timeframe is None


@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=True)
def test_small_temporal_link_pred_train_table_stats_array_entity_target_counts(  # noqa: E501
        dg_indexed_temporal_link_pred_graph, tmp_path: pathlib.Path):
    # Companion to test_small_temporal_link_pred_train_table_stats_array pinning
    # the CORRECT intended whole-table counts: 5 distinct entities, 3 distinct
    graph, index_path = dg_indexed_temporal_link_pred_graph()
    graph.infer_metadata()
    query = ('PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 0, 2) RANK TOP 3 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)
    split = DateOffsetSplit([-2, 0], unit=TimeUnit.DAYS, timeframe=2)
    pquery = PredictiveQueryV2(
        name='SMALL_TEMPORAL_LP_STATS_ARRAY_v1',
        graph=graph,
        validated_predictive_query=validated_query,
        split=split,
    )
    materialize_task_dg(pquery, index_path, tmp_path)
    table_stats = pquery.get_train_table_stats()
    assert table_stats.num_entities == 5
    assert table_stats.num_targets == 3


@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=False)
def test_small_temporal_link_pred_train_table_stats_filtered(
        dg_indexed_temporal_link_pred_graph, tmp_path: pathlib.Path):
    # Filtered temporal link-pred train-table-stats (see the non-filtered
    # sibling for the timeframe-2 split + 2023 date origin).
    #
    # The `WHERE ORDERS.STORE_ID < 2` filter restricts the realized links to
    # stores {0, 1}. The materialized train table only contains targets {0, 1}
    # (the readback below proves it), so num_targets is 2 -- the distinct
    # targets actually present after the filter (read straight off the DG
    # manifest); num_train_rows (8) and num_entities (4) hold.
    graph, index_path = dg_indexed_temporal_link_pred_graph()
    # stype inferral needs the per-test autouse persist_dir.
    graph.infer_metadata()
    # The filter filters the ORDERS table, not the STORE table -- it drops
    # training rows for store 2 but keeps stores {0, 1} as targets.
    query = ('PREDICT LIST_DISTINCT(ORDERS.STORE_ID '
             'WHERE ORDERS.STORE_ID < 2, 0, 2) RANK TOP 3 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)
    split = DateOffsetSplit([-2, 0], unit=TimeUnit.DAYS, timeframe=2)
    pquery = PredictiveQueryV2(
        name='SMALL_TEMPORAL_LP_STATS_FILTERED_v1',
        graph=graph,
        validated_predictive_query=validated_query,
        split=split,
    )
    train_table_path, job_id, _ = generate_train_table_dg(
        pquery, diskgraph_index_path=index_path, destination=str(tmp_path))
    # The filter keeps only stores {0, 1} as realized targets. The
    # DG-native train table is the dense grid with an array TARGET1 column.
    train_df = pd.read_parquet(train_table_path)
    realized_targets = {int(v) for t in train_df['TARGET1'] for v in t}
    assert realized_targets == {0, 1}

    task = pquery.create_task(train_table_path=train_table_path,
                              train_table_name='train_table', job_id=job_id,
                              ignore_encoders=True)
    task.train_table.materialize(execute_split=False,
                                 calculate_split_stats=False,
                                 skip_compute_stats=True)
    from test.fixtures.dg_indexed_graph import (
        _apply_ttng_native_split_from_bundle,
    )
    staging_dir = str(pathlib.Path(train_table_path).resolve().parent)
    _apply_ttng_native_split_from_bundle(task.train_table, staging_dir,
                                         task.task_type)

    table_stats = pquery.get_train_table_stats()
    assert table_stats.min_date == pd.Timestamp('2023-01-03 00:00:00', tz='UTC')
    assert table_stats.max_date == pd.Timestamp('2023-01-13 00:00:00', tz='UTC')
    # Dense (entity x timeframe) grid: 30 rows; 8 realized links post-filter.
    assert table_stats.num_train_rows == 30
    assert table_stats.num_links == 8
    # num_entities / num_targets asserted in the skipped
    # ..._entity_target_counts companion below.
    assert table_stats.num_timeframes == 6
    assert table_stats.num_timeframes_post_filter == 6
    # node_degree_stats is not populated on the DG-native aggregate path.
    assert table_stats.node_degree_stats is None
    # Bounded DG v2 stats omit the diagnostic UI timeframe series.
    assert table_stats.stats_per_timeframe is None


@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=False)
def test_small_temporal_link_pred_train_table_stats_filtered_entity_target_counts(  # noqa: E501
        dg_indexed_temporal_link_pred_graph, tmp_path: pathlib.Path):
    # Companion pinning the CORRECT intended post-filter counts: 5 distinct
    graph, index_path = dg_indexed_temporal_link_pred_graph()
    graph.infer_metadata()
    query = ('PREDICT LIST_DISTINCT(ORDERS.STORE_ID '
             'WHERE ORDERS.STORE_ID < 2, 0, 2) RANK TOP 3 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)
    split = DateOffsetSplit([-2, 0], unit=TimeUnit.DAYS, timeframe=2)
    pquery = PredictiveQueryV2(
        name='SMALL_TEMPORAL_LP_STATS_FILTERED_v1',
        graph=graph,
        validated_predictive_query=validated_query,
        split=split,
    )
    materialize_task_dg(pquery, index_path, tmp_path)
    table_stats = pquery.get_train_table_stats()
    assert table_stats.num_entities == 5
    assert table_stats.num_targets == 2


@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=True)
def test_small_temporal_link_pred_train_table_stats_filtered_array(
        dg_indexed_temporal_link_pred_graph, tmp_path: pathlib.Path):
    # Filtered + array-target temporal link-pred train-table-stats. Combines
    # the two DG-native semantics the sibling rows document:
    #   * array mode -> the DG-native train table materializes the dense
    #     (entity x timeframe) grid (30 rows incl. empty-target rows), so the
    #     aggregate counts are 30 train rows / 30 links / 5 entities; and
    #   * `WHERE ORDERS.STORE_ID < 2` -> realized targets are {0, 1}, so
    #     num_targets is 2.
    # The underlying link structure (6 non-empty rows / 8 real links over
    # targets {0, 1}) is re-asserted via the readback below.
    graph, index_path = dg_indexed_temporal_link_pred_graph()
    # stype inferral needs the per-test autouse persist_dir.
    graph.infer_metadata()
    # The filter filters the ORDERS table, not the STORE table -- it drops
    # training rows for store 2 but keeps stores {0, 1} as targets.
    query = ('PREDICT LIST_DISTINCT(ORDERS.STORE_ID '
             'WHERE ORDERS.STORE_ID < 2, 0, 2) RANK TOP 3 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)
    split = DateOffsetSplit([-2, 0], unit=TimeUnit.DAYS, timeframe=2)
    pquery = PredictiveQueryV2(
        name='SMALL_TEMPORAL_LP_STATS_FILTERED_ARRAY_v1',
        graph=graph,
        validated_predictive_query=validated_query,
        split=split,
    )
    train_table_path, job_id, _ = generate_train_table_dg(
        pquery, diskgraph_index_path=index_path, destination=str(tmp_path))
    # Real link structure: 6 non-empty target rows / 8 links over {0, 1}.
    train_df = pd.read_parquet(train_table_path)
    num_non_empty = int(sum(len(t) > 0 for t in train_df['TARGET1']))
    num_real_links = int(sum(len(t) for t in train_df['TARGET1']))
    realized_targets = {int(v) for t in train_df['TARGET1'] for v in t}
    assert num_non_empty == 6
    assert num_real_links == 8
    assert realized_targets == {0, 1}

    task = pquery.create_task(train_table_path=train_table_path,
                              train_table_name='train_table', job_id=job_id,
                              ignore_encoders=True)
    task.train_table.materialize(execute_split=False,
                                 calculate_split_stats=False,
                                 skip_compute_stats=True)
    from test.fixtures.dg_indexed_graph import (
        _apply_ttng_native_split_from_bundle,
    )
    staging_dir = str(pathlib.Path(train_table_path).resolve().parent)
    _apply_ttng_native_split_from_bundle(task.train_table, staging_dir,
                                         task.task_type)

    table_stats = pquery.get_train_table_stats()
    assert table_stats.min_date == pd.Timestamp('2023-01-03 00:00:00', tz='UTC')
    assert table_stats.max_date == pd.Timestamp('2023-01-13 00:00:00', tz='UTC')
    # DG-native dense-grid + realized-target aggregate counts (see above).
    assert table_stats.num_train_rows == 30
    assert table_stats.num_links == 8
    # num_entities / num_targets asserted in the skipped
    # ..._entity_target_counts companion below.
    assert table_stats.num_timeframes == 6
    assert table_stats.num_timeframes_post_filter == 6
    # node_degree_stats is not populated on the DG-native aggregate path.
    assert table_stats.node_degree_stats is None
    # Bounded DG v2 stats omit the diagnostic UI timeframe series.
    assert table_stats.stats_per_timeframe is None


@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=True)
def test_small_temporal_link_pred_train_table_stats_filtered_array_entity_target_counts(  # noqa: E501
        dg_indexed_temporal_link_pred_graph, tmp_path: pathlib.Path):
    # Companion pinning the CORRECT intended post-filter counts: 5 distinct
    graph, index_path = dg_indexed_temporal_link_pred_graph()
    graph.infer_metadata()
    query = ('PREDICT LIST_DISTINCT(ORDERS.STORE_ID '
             'WHERE ORDERS.STORE_ID < 2, 0, 2) RANK TOP 3 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)
    split = DateOffsetSplit([-2, 0], unit=TimeUnit.DAYS, timeframe=2)
    pquery = PredictiveQueryV2(
        name='SMALL_TEMPORAL_LP_STATS_FILTERED_ARRAY_v1',
        graph=graph,
        validated_predictive_query=validated_query,
        split=split,
    )
    materialize_task_dg(pquery, index_path, tmp_path)
    table_stats = pquery.get_train_table_stats()
    assert table_stats.num_entities == 5
    assert table_stats.num_targets == 2


@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=False)
@pytest.mark.parametrize('na_values', [True, False])
def test_small_temporal_link_pred_train_table_fkeys(small_temporal_graph_fixed,
                                                    tmp_path: pathlib.Path,
                                                    na_values: bool):
    graph = small_temporal_graph_fixed(tmp_path=tmp_path,
                                       add_na_pkeys=na_values,
                                       add_na_timestamps=na_values)
    parser = PQLParser()
    validator = PredictiveQueryValidator(**validator_kwargs(graph))

    query = ('PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 0, 2) RANK TOP 2 '
             'FOR EACH USERS.USER_ID')
    parsed_query = parser.to_parsed_predictive_query(query)

    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert response.ok
    assert validated_query is not None
    pquery = PredictiveQueryV2(name='temporal_link', graph=graph,
                               validated_predictive_query=validated_query)
    generator = pquery.get_table_generator()

    entity_name, target_name = generator.ENTITY, generator.TARGET
    fkeys = generator.get_fkeys()
    lhs_entity_fkey = fkeys[0]
    rhs_entity_fkey = fkeys[1]

    assert lhs_entity_fkey.name == entity_name
    assert lhs_entity_fkey.dst_table == 'USERS'
    assert lhs_entity_fkey.dst_key == 'USER_ID'
    assert lhs_entity_fkey.dtype == Dtype.int

    assert rhs_entity_fkey.name == target_name
    assert rhs_entity_fkey.dst_table == 'parquet.STORES'
    assert rhs_entity_fkey.dst_key == 'STORE_ID'
    assert rhs_entity_fkey.dtype == Dtype.int


@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=True)
@pytest.mark.parametrize('na_values', [True, False])
def test_small_temporal_link_pred_train_table_fkeys_arrays(
        small_temporal_graph_fixed, tmp_path: pathlib.Path, na_values: bool,
        tempdir):
    graph = small_temporal_graph_fixed(tmp_path=tmp_path,
                                       add_na_pkeys=na_values,
                                       add_na_timestamps=na_values)
    parser = PQLParser()
    validator = PredictiveQueryValidator(**validator_kwargs(graph))

    query = ('PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 0, 2) RANK TOP 2 '
             'FOR EACH USERS.USER_ID')
    parsed_query = parser.to_parsed_predictive_query(query)

    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert response.ok
    assert validated_query is not None
    pquery = PredictiveQueryV2(name='temporal_link', graph=graph,
                               validated_predictive_query=validated_query)
    generator = pquery.get_table_generator()

    entity_name, target_name = generator.ENTITY, generator.TARGET
    fkeys = generator.get_fkeys()
    lhs_entity_fkey = fkeys[0]
    rhs_entity_fkey = fkeys[1]

    assert lhs_entity_fkey.name == entity_name
    assert lhs_entity_fkey.dst_table == 'USERS'
    assert lhs_entity_fkey.dst_key == 'USER_ID'
    assert lhs_entity_fkey.dtype == Dtype.int

    assert rhs_entity_fkey.name == target_name
    assert rhs_entity_fkey.dst_table == 'parquet.STORES'
    assert rhs_entity_fkey.dst_key == 'STORE_ID'
    assert rhs_entity_fkey.dtype == Dtype.int


@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=False)
def test_small_temporal_link_pred_train_table_generation(
        dg_indexed_temporal_link_pred_graph, tmp_path: pathlib.Path):
    # Temporal link-pred train-table GENERATION over the DG-indexed fixture via
    # the production `generate_train_table_dg(pquery, diskgraph_index_path=...)`
    # path; reads the written parquet back to assert the (entity, target,
    # timestamp, frequency) label-row set.
    #
    # The `str(validated_query)` assertion (query canonicalization) is kept; the
    # expected label rows are computed for this fixture's shape (5u/3s/15e) with
    # a 2023 date origin. The DG fixture has no NA-pkey/NA-timestamp variant, so
    # only the non-NA generation behavior is covered here.
    graph, index_path = dg_indexed_temporal_link_pred_graph()
    # stype inferral needs the per-test autouse persist_dir.
    graph.infer_metadata()
    query = ('PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 0, 2) RANK TOP 2 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)
    assert str(validated_query) == (
        "ValidatedPredictiveQuery(entity_ast=USERS.USER_ID, "
        "target_ast=LIST_DISTINCT(ORDERS.STORE_ID, 0, 2, days), "
        "whatif_ast=None, top_k=2, problem_type=RANK, for_each='FOR EACH', "
        "num_forecasts=1, validation_response="
        "ValidationResponse(warnings=[], errors=[], info_items=[]))")

    split = DateOffsetSplit([-2, 0], unit=TimeUnit.DAYS, timeframe=2)
    pquery = PredictiveQueryV2(
        name='SMALL_TEMPORAL_LP_GENERATION_v1',
        graph=graph,
        validated_predictive_query=validated_query,
        split=split,
    )
    train_table_path, _, _ = generate_train_table_dg(
        pquery, diskgraph_index_path=index_path, destination=str(tmp_path))
    train_df = pd.read_parquet(train_table_path)

    # DG-native train tables use the canonical multi-entity column names. The
    # RANK LP train table is always the dense (entity x timeframe) grid with an
    # ARRAY TARGET1 column (the same shape as the ALLOW_ARRAY_TARGETS sibling);
    # the non-empty labelled rows carry the (entity, sorted-targets, timestamp,
    # frequency) tuples asserted below.
    entity_name = 'ENTITY1'
    target_name = 'TARGET1'
    time_col_name = TrainingTableGeneratorV2.TIMESTAMP

    # 8 non-empty labelled rows recomputed for the 5u/3s/15e fixture, dates
    # translated 1970->2023 (day-spacing identical).
    expected_train_df = pd.DataFrame({
        entity_name: [1, 1, 2, 2, 3, 3, 4, 4],
        time_col_name:
        pd.to_datetime([
            '2023-01-03', '2023-01-05', '2023-01-05', '2023-01-07',
            '2023-01-09', '2023-01-11', '2023-01-11', '2023-01-13'
        ]),
        target_name: [[0, 1], [2], [0], [1, 2], [0, 1], [2], [0], [1, 2]],
        FREQUENCY_COL_NAME: [[1, 1], [1], [1], [1, 1], [1, 1], [1], [1], [1,
                                                                          1]],
    })

    expected_train_df_set = sorted([
        x for x in zip(expected_train_df[entity_name],
                       [sorted(t) for t in expected_train_df[target_name]],
                       expected_train_df[time_col_name],
                       expected_train_df[FREQUENCY_COL_NAME])
    ])
    # Filter the dense grid to non-empty target rows.
    non_empty = train_df[train_df[target_name].map(len) > 0]
    train_df_set = sorted([
        x for x in zip(non_empty[entity_name], [
            sorted(json.loads(y)) if isinstance(y, str) else sorted(list(y))
            for y in non_empty[target_name]
        ], non_empty[time_col_name], [
            json.loads(y) if isinstance(y, str) else list(y)
            for y in non_empty[FREQUENCY_COL_NAME]
        ])
    ])
    assert train_df_set == expected_train_df_set


@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=True)
def test_small_temporal_link_pred_train_table_generation_array0(
        dg_indexed_temporal_link_pred_graph, tmp_path: pathlib.Path):
    # Array-target temporal link-pred train-table GENERATION (see the non-array
    # sibling for the 2023 date origin).
    #
    # NOTE: the DG-native array train table materializes the dense
    # (entity x timeframe) grid, so it carries empty-target rows in addition to
    # the real labelled rows. The readback below filters to non-empty TARGET
    # arrays (8 rows) before the (entity, sorted-targets, timestamp, frequency)
    # set-equality check.
    graph, index_path = dg_indexed_temporal_link_pred_graph()
    # stype inferral needs the per-test autouse persist_dir.
    graph.infer_metadata()
    query = ('PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 0, 2) RANK TOP 2 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)
    assert str(validated_query) == (
        "ValidatedPredictiveQuery(entity_ast=USERS.USER_ID, "
        "target_ast=LIST_DISTINCT(ORDERS.STORE_ID, 0, 2, days), "
        "whatif_ast=None, top_k=2, problem_type=RANK, for_each='FOR EACH', "
        "num_forecasts=1, validation_response="
        "ValidationResponse(warnings=[], errors=[], info_items=[]))")

    split = DateOffsetSplit([-2, 0], unit=TimeUnit.DAYS, timeframe=2)
    pquery = PredictiveQueryV2(
        name='SMALL_TEMPORAL_LP_GENERATION_ARRAY0_v1',
        graph=graph,
        validated_predictive_query=validated_query,
        split=split,
    )
    train_table_path, _, _ = generate_train_table_dg(
        pquery, diskgraph_index_path=index_path, destination=str(tmp_path))
    train_df = pd.read_parquet(train_table_path)

    # DG-native train tables use the canonical multi-entity column names.
    entity_name = 'ENTITY1'
    target_name = 'TARGET1'
    time_col_name = TrainingTableGeneratorV2.TIMESTAMP

    # 8 non-empty labelled rows recomputed for the 5u/3s/15e fixture, dates
    # translated 1970->2023 (day-spacing identical).
    expected_train_df = pd.DataFrame({
        entity_name: [1, 1, 2, 2, 3, 3, 4, 4],
        time_col_name:
        pd.to_datetime([
            '2023-01-03', '2023-01-05', '2023-01-05', '2023-01-07',
            '2023-01-09', '2023-01-11', '2023-01-11', '2023-01-13'
        ]),
        target_name: [[0, 1], [2], [0], [1, 2], [0, 1], [2], [0], [1, 2]],
        FREQUENCY_COL_NAME: [[1, 1], [1], [1], [1, 1], [1, 1], [1], [1], [1,
                                                                          1]],
    })

    expected_train_df_set = sorted([
        x for x in zip(expected_train_df[entity_name],
                       [sorted(t) for t in expected_train_df[target_name]],
                       expected_train_df[time_col_name],
                       expected_train_df[FREQUENCY_COL_NAME])
    ])
    # Filter the dense grid to non-empty target rows.
    non_empty = train_df[train_df[target_name].map(len) > 0]
    train_df_set = sorted([
        x for x in zip(non_empty[entity_name], [
            sorted(json.loads(y)) if isinstance(y, str) else sorted(list(y))
            for y in non_empty[target_name]
        ], non_empty[time_col_name], [
            json.loads(y) if isinstance(y, str) else list(y)
            for y in non_empty[FREQUENCY_COL_NAME]
        ])
    ])
    assert train_df_set == expected_train_df_set


@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=True)
@pytest.mark.parametrize(
    'col_name, value, target',
    [
        # LITERAL TYPING: the DG CONTAINS-on-array path matches the literal
        # against the column's stored numeric type EXACTLY. AMOUNT is a float
        # column so the literal must be `30.0` (`'30'` matches nothing ->
        # all-False); CATEGORY is
        # an int column so the literal must be `1` (`'1.0'` matches nothing).
        ('AMOUNT', '30.0', [True, False, False, True]),
        ('CATEGORY', '1', [True, True, False, True]),
    ])
def test_small_temporal_link_pred_train_table_generation_array1(
        dg_indexed_node_pred_txn_sessions_graph, tmp_path: pathlib.Path,
        col_name: str, value: str, target: List[bool]):
    # CONTAINS-array temporal NODE-PRED test (despite the LP filename) for
    # `PREDICT LIST_DISTINCT(TRANSACTIONS.{AMOUNT|CATEGORY}, 0, 2) CONTAINS
    # {value} FOR EACH USERS.USER_ID` under `DateOffsetSplit([-2,0], DAYS,
    # timeframe=2)`.
    #
    # Runs on the WIDE-SPAN variant of the DG-native node-pred
    # TRANSACTIONS/SESSIONS fixture (`wide_span=True`, a 12-day span). A narrow
    # 4-day span resolves to <3 timeframes under this timeframe-2 split (validator
    # "needs at least 3 timeframes" / empty DG `train_table_stats`); the 3x-wide
    # span resolves to 5 timeframes whose forward `(0, 2)` aggregation windows
    # still capture the AMOUNT=30 / CATEGORY=1 transactions.
    #
    # The DG array producer materializes one boolean row per (entity,
    # timeframe-anchor) (20 rows = 4 entities x 5 anchors). The deterministic
    # invariant -- "does this entity EVER have a qualifying transaction in some
    # forward window" -- is asserted as the per-entity OR over the dense grid.
    graph, index_path = dg_indexed_node_pred_txn_sessions_graph(
        with_category=True, wide_span=True)
    # stype inferral needs the per-test autouse persist_dir.
    graph.infer_metadata()
    parser = PQLParser()
    validator = PredictiveQueryValidator(**validator_kwargs(graph))

    query = (f'PREDICT LIST_DISTINCT(TRANSACTIONS.{col_name}, 0, 2) CONTAINS '
             f'{value} FOR EACH USERS.USER_ID')
    parsed_query = parser.to_parsed_predictive_query(query)

    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert response.ok
    assert validated_query is not None

    split = DateOffsetSplit([-2, 0], unit=TimeUnit.DAYS, timeframe=2)
    pquery = PredictiveQueryV2(
        name=f'SMALL_TEMPORAL_NP_CONTAINS_ARRAY1_{col_name}_v1',
        graph=graph,
        validated_predictive_query=validated_query,
        split=split,
    )
    train_table_path, _, _ = generate_train_table_dg(
        pquery, diskgraph_index_path=index_path, destination=str(tmp_path))
    train_df = pd.read_parquet(train_table_path)

    # DG-native train tables use the canonical multi-entity column names.
    entity_name = 'ENTITY1'
    target_name = 'TARGET1'

    # All 4 entities appear in the dense grid.
    assert set(train_df[entity_name].tolist()) == {0, 1, 2, 3}
    # Per-entity ever-contains (OR over the dense (entity x anchor) grid).
    expected = dict(zip([0, 1, 2, 3], target))
    for ent in [0, 1, 2, 3]:
        ever = bool(train_df[train_df[entity_name] == ent][target_name].any())
        assert ever == expected[ent], (
            f'entity {ent}: ever-contains {ever} != legacy {expected[ent]}')


@pytest.mark.skip(reason=_SKIP_REASON_DG_WEIGHT_COL)
@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=False)
def test_temporal_link_pred_weight_col_respects_time_bucket(
        tmp_path: pathlib.Path):
    users_df = pd.DataFrame({
        'USER_ID': [0],
        'AGE': [25],
    })
    stores_df = pd.DataFrame({
        'STORE_ID': [0, 1],
    })
    orders_df = pd.DataFrame({
        'ORDER_ID':
        list(range(8)),
        'USER_ID': [0] * 8,
        'STORE_ID': [0, 1, 0, 1, 0, 1, 0, 1],
        # Distinct values let us verify bucket-local aggregation.
        'ORDER_VALUE_3': [10, 11, 20, 21, 30, 31, 40, 41],
        'TIMESTAMP':
        pd.to_datetime(list(range(8)), unit='d'),
    })

    root_dir = tmp_path.resolve().as_posix()
    users_df.to_parquet(f'{root_dir}/USERS.parquet')
    stores_df.to_parquet(f'{root_dir}/STORES.parquet')
    orders_df.to_parquet(f'{root_dir}/ORDERS.parquet')

    graph = Graph(name='temporal_weight_bucket_graph')
    graph.connect(ParquetConnector(name='parquet', root_dir=root_dir))
    graph.add(name='USERS', pkey='USER_ID', cols=['AGE'])
    graph.add(name='STORES', pkey='STORE_ID')
    graph.add(
        name='ORDERS',
        pkey='ORDER_ID',
        fkeys=['USER_ID', 'STORE_ID'],
        cols=['ORDER_VALUE_3', 'TIMESTAMP'],
        time_col='TIMESTAMP',
    )
    graph['ORDERS'].time_col.timestamp_unit = 'ns'
    graph.infer_links()
    graph.fully_qualify_all_fkeys()
    graph.infer_metadata()
    graph.infer_encoders()

    query = ('PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 0, 1) RANK TOP 2 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)
    split = DateOffsetSplit([-6, 0], unit=TimeUnit.DAYS, timeframe=1)
    generator = TrainingTableGeneratorV2(graph, validated_query, split,
                                         weight_col='ORDERS.ORDER_VALUE_3')
    train_df = generator.generate_table()

    assert generator.WEIGHT in train_df.columns
    assert (train_df.groupby([generator.ENTITY, generator.TARGET]).size()
            > 1).any()

    source_orders = orders_df.copy()
    source_orders['TIMESTAMP'] = pd.to_datetime(source_orders['TIMESTAMP'])

    for _, row in train_df.iterrows():
        row_ts = pd.Timestamp(row[generator.TIMESTAMP])
        bucket_orders = source_orders[
            (source_orders['USER_ID'] == row[generator.ENTITY])
            & (source_orders['STORE_ID'] == row[generator.TARGET])
            & (source_orders['TIMESTAMP'] >= row_ts)
            & (source_orders['TIMESTAMP'] <= row_ts + pd.Timedelta(days=1))]
        assert len(bucket_orders) == 1
        expected_weight = float(bucket_orders['ORDER_VALUE_3'].iloc[0])
        assert float(row[generator.WEIGHT]) == expected_weight


@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=False)
def test_static_link_pred_table_stats(dg_indexed_static_link_pred_graph,
                                      tmp_path: pathlib.Path):
    # Static-LP stats on the DG-indexed static-LP fixture (USERS x PRODUCTS x
    # TRANS, `missing_link=True, dangling_entity=True`) for the
    # `PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID_FK) RANK TOP 2 FOR EACH
    # USERS.USER_ID` query, materialized via `materialize_task_dg`.
    #
    # NO time columns (set_entity/trans_time_col=False): the static-LP PQL has
    # no time range, so any time col would fail validation ("Time range
    # required for LIST_DISTINCT with time column"). The pseudo time col is
    # synthesized by the producer for the static split.
    #
    # COUNTS are read straight off the DG manifest for this fixture's shape:
    # 3 entities, 2 distinct product targets, 4 realized train rows.
    graph, index_path = dg_indexed_static_link_pred_graph(
        num_entities=3, missing_link=True, dangling_entity=True,
        set_entity_time_col=False, set_trans_time_col=False)
    graph.infer_metadata()
    query = ('PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID_FK) RANK TOP 2 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)

    pquery = PredictiveQueryV2(name='static_link_pred', graph=graph,
                               validated_predictive_query=validated_query,
                               split=TemporalSplit([0.5, 0.25, 0.25]))

    task = materialize_task_dg(pquery, index_path, tmp_path)
    table_stats = task.get_train_table_stats(after_split=False)

    # Dense grid over the static fixture: 8 entity rows. num_entities (8) is
    # correct here (matches the 8 distinct ENTITY1 ids). num_targets is
    # asserted in the companion below (2 realized product labels {0, 1}).
    assert table_stats.num_train_rows == 8
    assert table_stats.num_entities == 8

    # The DG-native static link-prediction task carries no time column (the
    # static split maps to a RandomLinkSplit); there is no real temporal
    # dimension, so time_col is None.
    assert task.train_table.time_col is None


@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=False)
def test_static_link_pred_table_stats_entity_target_counts(
        dg_indexed_static_link_pred_graph, tmp_path: pathlib.Path):
    # Companion pinning the CORRECT intended num_targets for the static-LP
    graph, index_path = dg_indexed_static_link_pred_graph(
        num_entities=3, missing_link=True, dangling_entity=True,
        set_entity_time_col=False, set_trans_time_col=False)
    graph.infer_metadata()
    query = ('PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID_FK) RANK TOP 2 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)
    pquery = PredictiveQueryV2(name='static_link_pred', graph=graph,
                               validated_predictive_query=validated_query,
                               split=TemporalSplit([0.5, 0.25, 0.25]))
    task = materialize_task_dg(pquery, index_path, tmp_path)
    table_stats = task.get_train_table_stats(after_split=False)
    assert table_stats.num_targets == 2


def test_static_link_pred_with_dangling_rhs(dg_indexed_static_link_pred_graph,
                                            tmp_path: pathlib.Path):
    # Dangling-RHS static-LP test on the DG-indexed static-LP fixture
    # (`missing_link=True, dangling_entity=True`: some users have no targets,
    # some entities have zero transactions) for the
    # `PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID_FK) RANK TOP 2 FOR EACH
    # USERS.USER_ID` query through `materialize_task_dg`.
    #
    # COUNT: the dangling/zero-transaction users contribute no train rows; the
    # 4 realized (entity, target) rows come from the active users' distinct
    # purchases. Asserted both off the DG manifest stats (num_train_rows) and a
    # direct train-table parquet readback.
    graph, index_path = dg_indexed_static_link_pred_graph(
        num_entities=3, missing_link=True, dangling_entity=True,
        set_entity_time_col=False, set_trans_time_col=False)
    graph.infer_metadata()
    query = ('PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID_FK) RANK TOP 2 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)

    pquery = PredictiveQueryV2(name='static_link_pred', graph=graph,
                               validated_predictive_query=validated_query,
                               split=TemporalSplit([0.5, 0.25, 0.25]))

    train_table_path, job_id, _ = generate_train_table_dg(
        pquery, diskgraph_index_path=index_path, destination=str(tmp_path))
    train_df = pd.read_parquet(train_table_path)
    # The DG-native RANK LP train table is the dense (entity x timeframe) grid,
    # so all 8 entities get a row (dangling/zero-transaction users carry an
    # empty TARGET1 array) -> 8 train rows.
    assert train_df.shape[0] == 8

    task = materialize_selected_task_dg(pquery, train_table_path, job_id)
    assert task.train_table is not None
    assert task.get_train_table_stats(after_split=False).num_train_rows == 8


@pytest.mark.skip(reason=_SKIP_REASON_DG_WEIGHT_COL)
@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=False)
def test_static_link_pred_weight_col_default_max(create_lp_graph, tempdir):
    graph = create_lp_graph(root_dir=tempdir.name, num_entities=2,
                            key_name_match=False, set_entity_time_col=False,
                            set_trans_time_col=False, use_parquet=True)
    graph['PRODUCTS'].remove_time_col()
    query = ('PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID_FK) RANK TOP 2 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)
    split = RandomSplit([0.8, 0.1, 0.1])
    generator = TrainingTableGeneratorV2(graph, validated_query, split,
                                         weight_col='TRANS.AMOUNT')
    train_df = generator.generate_table()
    assert generator.WEIGHT in train_df.columns
    assert set(train_df[generator.WEIGHT].tolist()) == {3.0}


@pytest.mark.skip(reason=_SKIP_REASON_DG_WEIGHT_COL)
@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=False)
def test_static_link_pred_weight_col_with_target_filter_join(
        create_lp_graph, tempdir):
    graph = create_lp_graph(root_dir=tempdir.name, num_entities=2,
                            key_name_match=False, set_entity_time_col=False,
                            set_trans_time_col=False, use_parquet=True)
    graph['PRODUCTS'].remove_time_col()
    query = ('PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID_FK '
             'WHERE PRODUCTS.BRAND = "PATAGONIA") '
             'RANK TOP 2 FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)
    split = RandomSplit([0.8, 0.1, 0.1])
    generator = TrainingTableGeneratorV2(
        graph,
        validated_query,
        split,
        weight_col='TRANS.AMOUNT',
    )
    train_df = generator.generate_table()
    assert generator.WEIGHT in train_df.columns
    assert set(train_df[generator.WEIGHT].tolist()) == {3.0}


@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=True)
def test_static_link_pred_table_stats_array(dg_indexed_static_link_pred_graph,
                                            tmp_path: pathlib.Path):
    # Array-target static-LP stats. Same graph + query as the non-array
    # `test_static_link_pred_table_stats` but with ALLOW_ARRAY_TARGETS=True;
    # driven through `materialize_task_dg` over the DG-indexed static-LP fixture
    # (missing_link+dangling, no time cols).
    #
    # COUNTS off the DG manifest for this fixture's shape.
    #   STABLE invariants (asserted exactly): num_entities == 3, num_targets
    #   == 2. These hold regardless of how the array producer groups targets.
    #
    #   num_train_rows / num_links are deliberately NOT pinned to an exact value:
    #   with ALLOW_ARRAY_TARGETS=True the DG-native producer non-deterministically
    #   groups the one multi-target entity (entity 0, which buys both products)
    #   either as a SINGLE array row TARGET=[0, 1] (-> 3 train rows / 3 links) or
    #   as TWO single-element array rows TARGET=[0], TARGET=[1] (-> 4 train rows /
    #   4 links). Both encodings carry identical link information; only the row
    #   packing differs. num_train_rows and num_links always move together and
    #   land in {3, 4}. Assert the falsifiable invariant the array path
    #   guarantees: train_rows == links (no dangling/zero rows) and both in
    #   {3, 4}.
    graph, index_path = dg_indexed_static_link_pred_graph(
        num_entities=3, missing_link=True, dangling_entity=True,
        set_entity_time_col=False, set_trans_time_col=False)
    graph.infer_metadata()
    query = ('PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID_FK) RANK TOP 2 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)

    pquery = PredictiveQueryV2(name='static_link_pred', graph=graph,
                               validated_predictive_query=validated_query,
                               split=TemporalSplit([0.5, 0.25, 0.25]))

    task = materialize_task_dg(pquery, index_path, tmp_path)
    table_stats = task.get_train_table_stats(after_split=False)
    # Dense grid over the static fixture: 8 entity rows carrying 4 realized
    # links. num_entities (8) is correct (matches 8 distinct ENTITY1). The
    # num_targets assert lives in the companion below.
    assert table_stats.num_train_rows == 8
    assert table_stats.num_entities == 8
    assert table_stats.num_links == 4


@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=True)
def test_static_link_pred_table_stats_array_entity_target_counts(  # noqa: E501
        dg_indexed_static_link_pred_graph, tmp_path: pathlib.Path):
    # Companion pinning the CORRECT intended num_targets for the array static-LP
    graph, index_path = dg_indexed_static_link_pred_graph(
        num_entities=3, missing_link=True, dangling_entity=True,
        set_entity_time_col=False, set_trans_time_col=False)
    graph.infer_metadata()
    query = ('PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID_FK) RANK TOP 2 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)
    pquery = PredictiveQueryV2(name='static_link_pred', graph=graph,
                               validated_predictive_query=validated_query,
                               split=TemporalSplit([0.5, 0.25, 0.25]))
    task = materialize_task_dg(pquery, index_path, tmp_path)
    table_stats = task.get_train_table_stats(after_split=False)
    assert table_stats.num_targets == 2


@pytest.mark.skip(reason=_SKIP_REASON_DG_WEIGHT_COL)
@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=True)
def test_static_link_pred_weight_col_default_max_array(create_lp_graph,
                                                       tempdir):
    graph = create_lp_graph(root_dir=tempdir.name, num_entities=2,
                            key_name_match=False, set_entity_time_col=False,
                            set_trans_time_col=False, use_parquet=True)
    graph['PRODUCTS'].remove_time_col()
    query = ('PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID_FK) RANK TOP 2 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)
    split = RandomSplit([0.8, 0.1, 0.1])
    generator = TrainingTableGeneratorV2(graph, validated_query, split,
                                         weight_col='TRANS.AMOUNT')
    train_df = generator.generate_table()
    assert generator.WEIGHT in train_df.columns
    for target, weight in zip(train_df[generator.TARGET],
                              train_df[generator.WEIGHT]):
        target_list = json.loads(target) if isinstance(target,
                                                       str) else list(target)
        weight_list = json.loads(weight) if isinstance(weight,
                                                       str) else list(weight)
        assert len(weight_list) == len(target_list)
        assert set(weight_list) == {3.0}


@pytest.mark.skip(reason=_SKIP_REASON_DG_WEIGHT_COL)
@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=True)
def test_static_link_pred_weight_col_with_target_filter_join_array(
        create_lp_graph, tempdir):
    graph = create_lp_graph(root_dir=tempdir.name, num_entities=2,
                            key_name_match=False, set_entity_time_col=False,
                            set_trans_time_col=False, use_parquet=True)
    graph['PRODUCTS'].remove_time_col()
    query = ('PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID_FK '
             'WHERE PRODUCTS.BRAND = "PATAGONIA") '
             'RANK TOP 2 FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)
    split = RandomSplit([0.8, 0.1, 0.1])
    generator = TrainingTableGeneratorV2(
        graph,
        validated_query,
        split,
        weight_col='TRANS.AMOUNT',
    )
    train_df = generator.generate_table()
    assert generator.WEIGHT in train_df.columns
    for target, weight in zip(train_df[generator.TARGET],
                              train_df[generator.WEIGHT]):
        target_list = json.loads(target) if isinstance(target,
                                                       str) else list(target)
        weight_list = json.loads(weight) if isinstance(weight,
                                                       str) else list(weight)
        assert len(weight_list) == len(target_list)
        assert set(weight_list) == {3.0}


@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=False)
def test_static_link_pred_table_generation(dg_indexed_static_link_pred_graph,
                                           tmp_path: pathlib.Path):
    # Static-LP train-table GENERATION on the DG-indexed static-LP fixture
    # (USERS x PRODUCTS x TRANS) for the canonical query
    # `LIST_DISTINCT(TRANS.PRODUCT_ID_FK) RANK TOP 2 FOR EACH USERS.USER_ID`,
    # reading the train table back via `generate_train_table` + parquet.
    #
    # str(validated_query) is computed for this query (TRANS.PRODUCT_ID_FK,
    # top_k=2).
    #
    # REALIZED ROWS (this fixture's dense default shape -- 3 users each buying
    # both products 15x): the canonical 6-row dense grid, one row per
    # (user, distinct product) with frequency 15:
    #   (0,0,15) (0,1,15) (1,0,15) (1,1,15) (2,0,15) (2,1,15)
    # i.e. entities {0,1,2}, targets {0,1}, every frequency == 15.
    #
    # PSEUDO-TIME NOTE: the DG static producer does NOT enumerate the pseudo
    # column 0..n-1; it fills the whole column with the single PSEUDO sentinel
    # (Timestamp.max). Assert that DG behaviour (all rows carry the same pseudo
    # sentinel).
    graph, index_path = dg_indexed_static_link_pred_graph(
        num_entities=3, set_entity_time_col=False, set_trans_time_col=False)
    graph.infer_metadata()
    query = ('PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID_FK) RANK TOP 2 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)
    assert str(validated_query) == (
        "ValidatedPredictiveQuery(entity_ast=USERS.USER_ID, "
        "target_ast=LIST_DISTINCT(TRANS.PRODUCT_ID_FK), whatif_ast=None, "
        "top_k=2, problem_type=RANK, for_each='FOR EACH', num_forecasts=1, "
        "validation_response=ValidationResponse(warnings=[], errors=[], "
        "info_items=[]))")

    pquery = PredictiveQueryV2(name='static_link_pred_gen', graph=graph,
                               validated_predictive_query=validated_query,
                               split=RandomSplit([0.8, 0.1, 0.1]))

    train_table_path, _job_id, _ = generate_train_table_dg(
        pquery, diskgraph_index_path=index_path, destination=str(tmp_path))
    train_df = pd.read_parquet(train_table_path)

    # DG-native train tables use the canonical multi-entity column names. The
    # RANK LP train table is always the dense ARRAY-target grid (one row per
    # entity, TARGET1/frequency as arrays), so aggregate the arrays per entity
    # into a {target: frequency} map and assert the deterministic realized link
    # structure: each of the 3 users bought both products 15x.
    entity_name = 'ENTITY1'
    target_name = 'TARGET1'
    assert {entity_name, target_name,
            FREQUENCY_COL_NAME} <= set(train_df.columns)
    per_entity: dict = {}
    for entity, targets, freqs in zip(
            train_df[entity_name],
            (json.loads(x) if isinstance(x, str) else list(x)
             for x in train_df[target_name]),
            (json.loads(x) if isinstance(x, str) else list(x)
             for x in train_df[FREQUENCY_COL_NAME])):
        assert len(targets) == len(freqs)
        per_entity.setdefault(entity, {})
        for tgt, frq in zip(targets, freqs):
            per_entity[entity][tgt] = frq
    assert per_entity == {
        0: {
            0: 15,
            1: 15
        },
        1: {
            0: 15,
            1: 15
        },
        2: {
            0: 15,
            1: 15
        }
    }


@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=True)
def test_static_link_pred_table_generation_array(
        dg_indexed_static_link_pred_graph, tmp_path: pathlib.Path):
    # Array-target static-LP GENERATION on the DG-indexed static-LP fixture for
    # the canonical `LIST_DISTINCT(TRANS.PRODUCT_ID_FK) RANK TOP 2` query in
    # array mode (ALLOW_ARRAY_TARGETS=True), reading the train table back via
    # `generate_train_table` + parquet.
    #
    # READBACK SHAPE: in array mode the DG train table
    # shards each entity's array row by the RandomSplit assignment, so a single
    # entity can appear across MULTIPLE rows (e.g. entity 1 split into a `[1]`
    # row and a `[0]` row) and the per-row arrays are NOT a stable per-run
    # value. So we do NOT assert exact per-row arrays (that would be flaky).
    # Instead we AGGREGATE the array targets/frequencies per entity across all
    # rows and assert the realized link structure, which IS deterministic:
    # each of the 3 users bought both products 15x, so every entity aggregates
    # to targets {0, 1} each with frequency 15 -- 3 entities, 6 distinct links.
    graph, index_path = dg_indexed_static_link_pred_graph(
        num_entities=3, set_entity_time_col=False, set_trans_time_col=False)
    graph.infer_metadata()
    query = ('PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID_FK) RANK TOP 2 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)
    assert str(validated_query) == (
        "ValidatedPredictiveQuery(entity_ast=USERS.USER_ID, "
        "target_ast=LIST_DISTINCT(TRANS.PRODUCT_ID_FK), whatif_ast=None, "
        "top_k=2, problem_type=RANK, for_each='FOR EACH', num_forecasts=1, "
        "validation_response=ValidationResponse(warnings=[], errors=[], "
        "info_items=[]))")

    pquery = PredictiveQueryV2(name='static_link_pred_gen_array', graph=graph,
                               validated_predictive_query=validated_query,
                               split=RandomSplit([0.8, 0.1, 0.1]))

    train_table_path, _job_id, _ = generate_train_table_dg(
        pquery, diskgraph_index_path=index_path, destination=str(tmp_path))
    train_df = pd.read_parquet(train_table_path)

    # DG-native train tables use the canonical multi-entity column names.
    entity_name = 'ENTITY1'
    target_name = 'TARGET1'
    assert {entity_name, target_name,
            FREQUENCY_COL_NAME} <= set(train_df.columns)

    # Aggregate the (possibly split-sharded) array rows per entity into a
    # {target: frequency} map; the realized link structure is deterministic.
    per_entity: dict = {}
    for entity, targets, freqs in zip(
            train_df[entity_name],
            (json.loads(x) if isinstance(x, str) else list(x)
             for x in train_df[target_name]),
            (json.loads(x) if isinstance(x, str) else list(x)
             for x in train_df[FREQUENCY_COL_NAME])):
        assert len(targets) == len(freqs)
        per_entity.setdefault(entity, {})
        for tgt, frq in zip(targets, freqs):
            per_entity[entity][tgt] = frq

    expected_per_entity = {
        0: {
            0: 15,
            1: 15
        },
        1: {
            0: 15,
            1: 15
        },
        2: {
            0: 15,
            1: 15
        }
    }
    assert per_entity == expected_per_entity


@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=False)
def test_temporal_link_pred_target_nonzero_offset_table_stats(
        dg_indexed_temporal_link_pred_graph, tmp_path: pathlib.Path):
    # NONZERO-OFFSET link-pred stats: the aggregation window is the NONZERO
    # offset `(3, 6)` -- targets links 3-6 days AFTER each anchor (the
    # load-bearing "nonzero offset" behaviour this test covers).
    #
    # Drives the nonzero-offset `(3, 6)` RANK-TOP-2 query over a DG-indexed
    # link-pred graph through `materialize_task_dg`. SIZING: the forward `(3, 6)`
    # window needs anchors with 3-6 days of FUTURE edges, so the default
    # 5u/3s/15e fixture (14-day span) produces an empty train table (the window
    # falls past the last edge). The wider `dg_indexed_temporal_link_pred_graph(
    # num_users=10, num_stores=6, num_edges=100)` (~100-day span) is the
    # smallest sizing on which `(3, 6)` resolves to a non-empty table.
    #
    # COUNTS are read straight off the DG manifest for this fixture's shape:
    # 87 train rows over 10 entities x 6 store targets across 29 day-stepped
    # anchor windows; aggregate span 2023-01-07..2023-04-10, anchor 2023-01-04.
    # The nonzero-offset semantics are exercised by the `(3, 6)` window plus the
    # chronological train<val<test split-stat ordering below. `node_degree_stats`
    # is not surfaced on the DG-native aggregate path (None).
    graph, index_path = dg_indexed_temporal_link_pred_graph(
        num_users=10, num_stores=6, num_edges=100)
    # stype inferral needs the per-test autouse persist_dir.
    graph.infer_metadata()
    query = ('PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 3, 6) '
             'RANK TOP 2 FOR EACH USERS.USER_ID')
    validated_predictive_query = get_validated_query(graph, query)
    split = DateOffsetSplit([-30, 0], unit=TimeUnit.DAYS, timeframe=30)
    pquery = PredictiveQueryV2(
        name='LP_NONZERO_OFFSET_STATS_v1',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
        split=split,
    )

    task = materialize_task_dg(pquery, index_path, tmp_path)
    table_stats = pquery.get_train_table_stats()
    assert table_stats.min_date == pd.Timestamp('2023-01-04 00:00:00', tz='UTC')
    assert table_stats.max_date == pd.Timestamp('2023-04-04 00:00:00', tz='UTC')
    assert table_stats.anchor_date == pd.Timestamp('2023-04-04 00:00:00',
                                                   tz='UTC')
    # Dense (entity x timeframe) grid: 10 entities x 29 timeframes = 290 rows,
    # 87 realized links. num_entities / num_targets asserted in the skipped
    # companion below.
    assert table_stats.num_train_rows == 290
    assert table_stats.num_links == 87
    assert table_stats.num_timeframes == 29
    # node_degree_stats is not populated on the DG-native aggregate path.
    assert table_stats.node_degree_stats is None

    # Per-timeframe diagnostics are omitted from bounded DG v2 stats. The
    # split-stage date bounds below retain the chronology required by training.
    assert table_stats.stats_per_timeframe is None

    # Native split stats report the exact materialized seed ranges. Dates are
    # tz-aware UTC; the per-stage anchor_date is the earliest anchor in the
    # stage (== min_date), matching Kumo's split-stats convention.
    split_stats = task.get_train_table_stats(after_split=True)
    assert split_stats.train_table_stats is not None
    assert (split_stats.train_table_stats.anchor_date == pd.Timestamp(
        '2023-01-04T00:00:00', tz='UTC'))
    assert (split_stats.train_table_stats.min_date == pd.Timestamp(
        '2023-01-04T00:00:00', tz='UTC'))
    assert (split_stats.train_table_stats.max_date == pd.Timestamp(
        '2023-01-10T00:00:00', tz='UTC'))

    assert split_stats.val_table_stats is not None
    assert (split_stats.val_table_stats.anchor_date == pd.Timestamp(
        '2023-01-13T00:00:00', tz='UTC'))
    assert (split_stats.val_table_stats.min_date == pd.Timestamp(
        '2023-01-13T00:00:00', tz='UTC'))
    assert (split_stats.val_table_stats.max_date == pd.Timestamp(
        '2023-01-16T00:00:00', tz='UTC'))

    assert split_stats.test_table_stats is not None
    assert (split_stats.test_table_stats.anchor_date == pd.Timestamp(
        '2023-01-19T00:00:00', tz='UTC'))
    assert (split_stats.test_table_stats.min_date == pd.Timestamp(
        '2023-01-19T00:00:00', tz='UTC'))
    assert (split_stats.test_table_stats.max_date == pd.Timestamp(
        '2023-01-19T00:00:00', tz='UTC'))


@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=False)
def test_temporal_link_pred_target_nonzero_offset_table_stats_entity_target_counts(  # noqa: E501
        dg_indexed_temporal_link_pred_graph, tmp_path: pathlib.Path):
    # Companion pinning the CORRECT intended counts for the nonzero-offset
    # fixture: 10 distinct entities, 6 distinct RHS store labels {0..5}. DG HLL
    graph, index_path = dg_indexed_temporal_link_pred_graph(
        num_users=10, num_stores=6, num_edges=100)
    graph.infer_metadata()
    query = ('PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 3, 6) '
             'RANK TOP 2 FOR EACH USERS.USER_ID')
    validated_predictive_query = get_validated_query(graph, query)
    split = DateOffsetSplit([-30, 0], unit=TimeUnit.DAYS, timeframe=30)
    pquery = PredictiveQueryV2(
        name='LP_NONZERO_OFFSET_STATS_v1',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
        split=split,
    )
    materialize_task_dg(pquery, index_path, tmp_path)
    table_stats = pquery.get_train_table_stats()
    assert table_stats.num_entities == 10
    assert table_stats.num_targets == 6


@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=True)
def test_temporal_link_pred_target_nonzero_offset_table_stats_array(
        dg_indexed_temporal_link_pred_graph, tmp_path: pathlib.Path):
    # Nonzero-offset + array-target link-pred stats; see the non-array sibling
    # for the 10u/6s/100e sizing.
    #
    # In array mode the DG train table materializes the dense (entity x
    # timeframe) grid including empty-target rows: 290 train rows (10 entities x
    # 29 timeframes) / 290 links / 10 entities / 6 targets. The underlying link
    # structure (35 non-empty rows carrying 87 real links) is re-asserted via
    # the train-table readback below.
    graph, index_path = dg_indexed_temporal_link_pred_graph(
        num_users=10, num_stores=6, num_edges=100)
    # stype inferral needs the per-test autouse persist_dir.
    graph.infer_metadata()
    query = ('PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 3, 6) '
             'RANK TOP 2 FOR EACH USERS.USER_ID')
    validated_predictive_query = get_validated_query(graph, query)
    split = DateOffsetSplit([-30, 0], unit=TimeUnit.DAYS, timeframe=30)
    pquery = PredictiveQueryV2(
        name='LP_NONZERO_OFFSET_STATS_ARRAY_v1',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
        split=split,
    )
    train_table_path, job_id, _ = generate_train_table_dg(
        pquery, diskgraph_index_path=index_path, destination=str(tmp_path))
    # Real link structure: 35 non-empty target rows carrying 87 real links
    # (matches the non-array sibling's 87 realized links).
    train_df = pd.read_parquet(train_table_path)
    target_name = 'TARGET1'
    num_non_empty = int(sum(len(t) > 0 for t in train_df[target_name]))
    num_real_links = int(sum(len(t) for t in train_df[target_name]))
    assert num_non_empty == 35
    assert num_real_links == 87

    task = pquery.create_task(train_table_path=train_table_path,
                              train_table_name='train_table', job_id=job_id,
                              ignore_encoders=True)
    task.train_table.materialize(execute_split=False,
                                 calculate_split_stats=False,
                                 skip_compute_stats=True)
    from test.fixtures.dg_indexed_graph import (
        _apply_ttng_native_split_from_bundle,
    )
    staging_dir = str(pathlib.Path(train_table_path).resolve().parent)
    _apply_ttng_native_split_from_bundle(task.train_table, staging_dir,
                                         task.task_type)

    table_stats = pquery.get_train_table_stats()
    assert table_stats.min_date == pd.Timestamp('2023-01-04 00:00:00', tz='UTC')
    assert table_stats.max_date == pd.Timestamp('2023-04-04 00:00:00', tz='UTC')
    assert table_stats.anchor_date == pd.Timestamp('2023-04-04 00:00:00',
                                                   tz='UTC')
    # DG-native dense-grid aggregate counts (see COUNT NOTE above). num_entities
    # / num_targets asserted in the companion below.
    assert table_stats.num_train_rows == 290
    assert table_stats.num_links == 87
    assert table_stats.num_timeframes == 29
    # node_degree_stats is not populated on the DG-native aggregate path.
    assert table_stats.node_degree_stats is None

    # Per-timeframe diagnostics are omitted from bounded DG v2 stats. The
    # split-stage date bounds below retain the chronology required by training.
    assert table_stats.stats_per_timeframe is None

    # Native split stats report the exact materialized seed ranges. Dates are
    # tz-aware UTC; the per-stage anchor is the earliest anchor in the stage.
    split_stats = task.get_train_table_stats(after_split=True)
    assert split_stats.train_table_stats is not None
    assert (split_stats.train_table_stats.anchor_date == pd.Timestamp(
        '2023-01-04T00:00:00', tz='UTC'))
    assert (split_stats.train_table_stats.min_date == pd.Timestamp(
        '2023-01-04T00:00:00', tz='UTC'))
    assert (split_stats.train_table_stats.max_date == pd.Timestamp(
        '2023-01-10T00:00:00', tz='UTC'))

    assert split_stats.val_table_stats is not None
    assert (split_stats.val_table_stats.anchor_date == pd.Timestamp(
        '2023-01-13T00:00:00', tz='UTC'))
    assert (split_stats.val_table_stats.min_date == pd.Timestamp(
        '2023-01-13T00:00:00', tz='UTC'))
    assert (split_stats.val_table_stats.max_date == pd.Timestamp(
        '2023-01-16T00:00:00', tz='UTC'))

    assert split_stats.test_table_stats is not None
    assert (split_stats.test_table_stats.anchor_date == pd.Timestamp(
        '2023-01-19T00:00:00', tz='UTC'))
    assert (split_stats.test_table_stats.min_date == pd.Timestamp(
        '2023-01-19T00:00:00', tz='UTC'))
    assert (split_stats.test_table_stats.max_date == pd.Timestamp(
        '2023-01-19T00:00:00', tz='UTC'))


@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=True)
def test_temporal_link_pred_target_nonzero_offset_table_stats_array_entity_target_counts(  # noqa: E501
        dg_indexed_temporal_link_pred_graph, tmp_path: pathlib.Path):
    # Companion pinning the CORRECT intended counts for the array nonzero-offset
    # fixture: 10 distinct entities, 6 distinct RHS store labels {0..5}. DG HLL
    graph, index_path = dg_indexed_temporal_link_pred_graph(
        num_users=10, num_stores=6, num_edges=100)
    graph.infer_metadata()
    query = ('PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 3, 6) '
             'RANK TOP 2 FOR EACH USERS.USER_ID')
    validated_predictive_query = get_validated_query(graph, query)
    split = DateOffsetSplit([-30, 0], unit=TimeUnit.DAYS, timeframe=30)
    pquery = PredictiveQueryV2(
        name='LP_NONZERO_OFFSET_STATS_ARRAY_v1',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
        split=split,
    )
    materialize_task_dg(pquery, index_path, tmp_path)
    table_stats = pquery.get_train_table_stats()
    assert table_stats.num_entities == 10
    assert table_stats.num_targets == 6


@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=False)
def test_invalid_targets(dg_indexed_static_link_pred_graph,
                         tmp_path: pathlib.Path):
    # Invalid-target validation: dangling fkeys (NULL or non-existent id 100,
    # no matching PRODUCTS pkey) must be DROPPED, leaving only the valid {0, 1}
    # link targets.
    #
    # Runs the STATIC invalid-target case on the DG-indexed static-LP fixture's
    # `invalid_targets=True` variant (per-user
    # `[0]*10 + [1]*10 + [NULL]*5 + [100]*5` PRODUCT_ID_FK pattern) and the
    # canonical `LIST_DISTINCT(TRANS.PRODUCT_ID_FK) RANK TOP 2 FOR EACH
    # USERS.USER_ID` query, reading the DG train table back via
    # `generate_train_table` + parquet.
    #
    # Only the static case is covered: the temporal `(,0,3)` window adds no
    # invalid-target coverage and resolves to an EMPTY train table on this
    # fixture's cadence (DiskgraphConfigError "split stats artifact must include
    # non-empty train_table_stats").
    #
    # DG OUTPUT (verified empirically, stable across 10 runs): the 100 / NULL
    # links are dropped; the train table has exactly 6 rows -- one per (user,
    # valid target) for 3 users x {0, 1} -- with TARGET set {0, 1} and no nulls.
    graph, index_path = dg_indexed_static_link_pred_graph(
        num_entities=3, invalid_targets=True, set_entity_time_col=False,
        set_trans_time_col=False)
    graph.infer_metadata()
    query = ('PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID_FK) RANK TOP 2 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)
    pquery = PredictiveQueryV2(name='link_prediction', graph=graph,
                               validated_predictive_query=validated_query,
                               split=RandomSplit([0.8, 0.1, 0.1]))

    train_table_path, _job_id, _ = generate_train_table_dg(
        pquery, diskgraph_index_path=index_path, destination=str(tmp_path))
    train_df = pd.read_parquet(train_table_path)

    # DG-native RANK LP train tables carry an ARRAY TARGET1 column (one dense
    # row per entity). The invalid 100 / NULL targets are DROPPED: every
    # per-row array is non-empty and a subset of the valid {0, 1} target set.
    target = 'TARGET1'
    # 3 users, each a single dense [0, 1] array row.
    assert train_df.shape[0] == 3
    for row in train_df[target]:
        labels = set(int(x) for x in row)
        assert len(labels) >= 1
        assert labels.issubset({0, 1})
        assert 100 not in labels


@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=True)
def test_invalid_targets_array(dg_indexed_static_link_pred_graph,
                               tmp_path: pathlib.Path):
    # DG-native migration of the array-target invalid-target test (see the
    # non-array sibling `test_invalid_targets` for the full migration
    # rationale). Same `invalid_targets=True` static-LP fixture + canonical
    # query, but in array mode (ALLOW_ARRAY_TARGETS=True), reading the DG train
    # table back via `generate_train_table` + parquet.
    #
    # ARRAY-MODE INVARIANTS (non-determinism-robust, matching the known DG
    # array-packing behaviour): the DG array producer packs entity 0/1/2's
    # LIST_DISTINCT array non-deterministically run-to-run (a multi-target
    # entity may be a single [0, 1] row or split into [0] / [1] rows), so the
    # row COUNT lands in {3, 4} and is NOT pinned. What IS guaranteed -- and is
    # the whole point of this test -- is that the invalid 100 / NULL targets are
    # DROPPED: every per-row array is a subset of the valid {0, 1} target set.
    graph, index_path = dg_indexed_static_link_pred_graph(
        num_entities=3, invalid_targets=True, set_entity_time_col=False,
        set_trans_time_col=False)
    graph.infer_metadata()
    query = ('PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID_FK) RANK TOP 2 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)
    pquery = PredictiveQueryV2(name='link_prediction', graph=graph,
                               validated_predictive_query=validated_query,
                               list_distinct_target_is_array=True,
                               split=RandomSplit([0.8, 0.1, 0.1]))

    train_table_path, _job_id, _ = generate_train_table_dg(
        pquery, diskgraph_index_path=index_path, destination=str(tmp_path))
    train_df = pd.read_parquet(train_table_path)

    # DG-native train tables use the canonical multi-entity TARGET1 name.
    target = 'TARGET1'
    # Array-mode row count is non-deterministic in {3, 4}; never pin it.
    assert train_df.shape[0] in (3, 4)
    # The invalid 100 / NULL targets are dropped: every array is a subset of the
    # valid {0, 1} target set, and no array is empty (no dangling-only rows).
    for row in train_df[target]:
        assert len(row) >= 1
        assert set(int(x) for x in row).issubset({0, 1})


@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=False)
def test_link_pred_unix_step_size_and_offset(
        dg_indexed_temporal_link_pred_graph, tmp_path: pathlib.Path):
    # LP unix-offset test: drives two RANK-TOP-3 LP queries (windows (6, 7) and
    # (0, 3)) over a DG-indexed link-pred graph
    # (`dg_indexed_temporal_link_pred_graph(num_users=10, num_stores=6,
    # num_edges=100)`, a ~100-day span so the (6, 7) / (0, 3) aggregation
    # windows resolve to non-empty anchor windows) through the DG-native
    # `materialize_task_dg` path.
    #
    # The asserted unix offsets are ORIGIN-INVARIANT: `compute_unix_step`
    # computes them purely from the PQL aggregation-window offsets (start / end)
    # and the timeframe step, NOT from any absolute epoch date -- a window of N
    # days is always
    # N * 86400 * 1e9 ns regardless of the 1970->2023 origin shift. So:
    #   * window (6, 7): start_offset = 6 days, end_offset = 7 days, and with no
    #     explicit timeframe_step the step = end - start = 1 day; and
    #   * window (0, 3): start_offset = 0, end_offset = 3 days, step = 3 days.
    # The wider 10u/6s/100e sizing (vs. the default 5u/3s/15e) is needed so the
    # forward-looking aggregation windows ((6, 7) targets data 6-7 days AFTER
    # each anchor, (0, 3) targets 0-3 days after) have room on the span; the
    # offsets themselves do not depend on the sizing. The split here is the
    # 30-day cadence `DateOffsetSplit([-30, 0], DAYS, timeframe=30)`: on this
    # ~100-day span a 30-day timeframe is NOT over-wide (unlike the default
    # 14-day fixture where 30 days yields zero windows). The 2-day cadence used
    # by the small-fixture stat tests packs anchors against the data edge, so a
    # forward window of 6-7 days falls past the last edge and the train table is
    # empty (no manifest is written) -- the 30-day cadence leaves anchor room
    # so the windows resolve to non-empty tables and `create_task` can read the
    # offsets.
    graph, index_path = dg_indexed_temporal_link_pred_graph(
        num_users=10, num_stores=6, num_edges=100)
    # stype inferral needs the per-test autouse persist_dir.
    graph.infer_metadata()
    split = DateOffsetSplit([-30, 0], unit=TimeUnit.DAYS, timeframe=30)

    query = ('PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 6, 7) RANK TOP 3 FOR EACH '
             'USERS.USER_ID')
    validated_predictive_query = get_validated_query(graph, query)
    pquery = PredictiveQueryV2(
        name='TEMPORAL_LINK_TEST_PQUERY_67_v1',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
        split=split,
    )
    task = materialize_task_dg(pquery, index_path, tmp_path)
    assert task.unix_start_offset == 24 * 60 * 60 * NANOSECONDS_PER_SECOND * 6
    assert task.unix_end_offset == 24 * 60 * 60 * NANOSECONDS_PER_SECOND * 7
    assert task.unix_step_size == 24 * 60 * 60 * NANOSECONDS_PER_SECOND * 1

    query = ('PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 0, 3) RANK TOP 3 FOR EACH '
             'USERS.USER_ID')
    validated_predictive_query = get_validated_query(graph, query)
    pquery = PredictiveQueryV2(
        name='TEMPORAL_LINK_TEST_PQUERY_03_v1',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
        split=split,
    )
    task = materialize_task_dg(pquery, index_path, tmp_path)
    assert task.unix_start_offset == 0
    assert task.unix_end_offset == 24 * 60 * 60 * NANOSECONDS_PER_SECOND * 3
    assert task.unix_step_size == 24 * 60 * 60 * NANOSECONDS_PER_SECOND * 3
