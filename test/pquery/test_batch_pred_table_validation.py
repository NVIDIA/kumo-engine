# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from test.fixtures.dg_indexed_graph import materialize_bp_dg

import pandas as pd
import pytest

from kumo.pquery.predictive_query import PredictiveQueryV2
from kumo.pquery.utils import validate_forecasting_parameters_consistency
from kumo.testing.pquery import get_validated_query
from kumo.validation.train_table_validation import TrainingTableGenerationError


def test_temporal_link_batch_pred_empty_lhs_table(
    dg_indexed_static_link_pred_graph,
    tmp_path,
):
    # DG-native empty-LHS link-pred batch-pred case. On the static fixture
    # (USERS.USER_ID 0..2) an entity-side filter excludes every USERS row,
    # producing an empty entity pool, driving the current-pred-table
    # validation flow with the real DG index threaded in.
    #
    # The empty-LHS case raises a TrainingTableGenerationError (empty
    # predict-seed guard) that generate_current_pred_table_generic converts
    # into a ValidationResponse with ok=False; we assert that error. On TT-NG
    # LP BP gens only the LHS predict seeds through the shared one-gen path, so
    # an empty entity pool now surfaces the shared ``_raise_empty_prediction_table``
    # message ("No rows to predict") rather than the retired HStack-specific
    # "No entities to predict" string.
    graph, index_path = dg_indexed_static_link_pred_graph()
    graph.infer_metadata().infer_encoders()
    # USERS.USER_ID range is 0..2; ``> 100`` excludes every entity.
    query = ('PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID_FK WHERE PRODUCTS.BRAND '
             '= "PATAGONIA", 0, 3) RANK TOP 10 FOR EACH USERS.USER_ID '
             'WHERE USERS.USER_ID > 100')
    validated_predictive_query = get_validated_query(graph, query)

    pquery = PredictiveQueryV2(
        name='temporal_link_pred_empty_lhs_v2',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
    )
    _, validation = pquery.generate_current_pred_table(
        pred_time=pd.Timestamp('2023-12-31'),
        diskgraph_index_path=index_path, destination=str(tmp_path))

    assert not validation.ok
    assert "No rows to predict" in validation.error_message()


def test_temporal_link_batch_pred_empty_rhs_table(
    dg_indexed_static_link_pred_graph,
    tmp_path,
):
    # DG-native empty-RHS link-pred batch-pred case. On the static fixture a
    # target-side filter matches no PRODUCTS, so the RANK candidate pool is
    # empty. Under TT-NG the RHS candidate set is seeded at INGEST -- it is NOT
    # materialized at gen time -- so LP BP gens only the LHS predict seeds. An
    # empty RHS filter therefore no longer raises at gen: the LHS entity pool is
    # still non-empty, so the predict-seed gen succeeds and validation is ok.
    # (The retired legacy HStack path drained the RHS iterator at gen time and
    # raised "No candidates to rank"; that gen-time RHS guard is gone on TT-NG.)
    graph, index_path = dg_indexed_static_link_pred_graph()
    graph.infer_metadata().infer_encoders()
    # No PRODUCTS BRAND matches ``"NONEXISTENT"``; the RANK candidate pool is
    # therefore empty -- but the RHS pool is an ingest concern on TT-NG.
    query = ('PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID_FK WHERE PRODUCTS.BRAND '
             '= "NONEXISTENT", 0, 3) RANK TOP 10 FOR EACH USERS.USER_ID')
    validated_predictive_query = get_validated_query(graph, query)

    pquery = PredictiveQueryV2(
        name='temporal_link_pred_empty_rhs_v2',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
    )
    _, validation = pquery.generate_current_pred_table(
        pred_time=pd.Timestamp('2023-12-31'),
        diskgraph_index_path=index_path, destination=str(tmp_path))

    # LHS predict seeds gen successfully; RHS emptiness is not a gen-time error.
    assert validation.ok


def test_temporal_link_batch_pred_empty_target_table(
    dg_indexed_static_link_pred_graph,
    tmp_path,
):
    # DG-native success case: the current-pred-table validation flow over the
    # DG-indexed static link-pred fixture (USERS entity + PRODUCTS RHS +
    # PRODUCT_BRANDS feature + TRANS two-fkey edge, key_name_match=False) with
    # the real DG index threaded in. A 2023-origin fixture (pre-2001 ns dates
    # clamp to the DG 2262 sentinel) with a pred_time after the data anchors a
    # valid, non-empty pred table; we assert ok=True.
    graph, index_path = dg_indexed_static_link_pred_graph()
    graph.infer_metadata().infer_encoders()
    query = ('PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID_FK WHERE PRODUCTS.BRAND '
             '= "PATAGONIA", 0, 3) RANK TOP 10 FOR EACH USERS.USER_ID')
    validated_predictive_query = get_validated_query(graph, query)

    pquery = PredictiveQueryV2(
        name='temporal_link_pred_empty_target_v2',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
    )
    _, validation = pquery.generate_current_pred_table(
        pred_time=pd.Timestamp('2023-12-31'),
        diskgraph_index_path=index_path, destination=str(tmp_path))

    assert validation.ok


def test_temporal_node_empty_pred_table(
        dg_indexed_temporal_synthetic_node_pred_graph, tmp_path):
    # Drive the production DG bp producer with the real index via the
    # `materialize_bp_dg` read-back helper. A restrictive entity filter that
    # excludes every row makes the DG empty-bp guard fire, raising
    # TrainingTableGenerationError with the DG message "No rows to predict".
    graph, index_path = dg_indexed_temporal_synthetic_node_pred_graph
    graph.infer_metadata().infer_encoders()
    # empty pred table due to an entity filter that excludes every USERS row
    # (USER_ID range is 0..4).
    query = ('PREDICT SUM(ORDERS.ORDER_SCORE, 0, 2) FOR EACH '
             'USERS.USER_ID WHERE USERS.USER_ID > 100')
    validated_predictive_query = get_validated_query(graph, query)
    with pytest.raises(TrainingTableGenerationError,
                       match=r"No rows to predict"):
        materialize_bp_dg(graph, validated_predictive_query, index_path,
                          tmp_path)


def test_forecasting_parameters_consistency():

    # Test failure case (forecast_length does not match)
    result = validate_forecasting_parameters_consistency(
        training_forecast_length=2,
        prediction_forecast_length=1,
    )
    assert not result.ok
    assert ("forecast length in the training job "
            "(2) does not match the forecast length in the "
            "prediction job (1)") in result.error_message()

    # Test success case (forecast_length defaults to 1)
    result = validate_forecasting_parameters_consistency(
        training_forecast_length=None,
        prediction_forecast_length=1,
    )
    assert result.ok

    # Test failure case (lag_timesteps does not match)
    result = validate_forecasting_parameters_consistency(
        training_lag_timesteps=None,
        prediction_lag_timesteps=1,
    )
    assert not result.ok
    assert ("lag timesteps in the training job "
            "(0) does not match the lag timesteps in the "
            "prediction job (1)") in result.error_message()

    # Test success case (lag_timesteps defaults to 0)
    result = validate_forecasting_parameters_consistency(
        training_lag_timesteps=None,
        prediction_lag_timesteps=0,
    )
    assert result.ok
