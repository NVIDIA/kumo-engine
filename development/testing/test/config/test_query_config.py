# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from kumoapi.encoder import Null

from io_utils.json import load_json, save_json
from kumo.config.graph import GraphConfig
from kumo.config.predictive_query import (
    PredictiveQueryConfigBase,
    PredictiveQueryV2Config,
)
from kumo.config.task import SplitConfig
from kumo.graph.graph import Graph
from kumo.task.split import RandomSplit, Split, TemporalSplit


def test_bp_entity_col_name(tmp_path, create_graph):
    # Create the graph and "infer" encoders
    graph: Graph = create_graph(name='FINANCIAL')
    graph.infer_metadata()
    for table in graph:
        for col in table:
            col.encoder = Null()

    # Create the config
    split_config = RandomSplit([0.8, 0.1, 0.1]).config()
    test_config = PredictiveQueryConfigBase(
        graph=graph.config(),
        name='testquery',
        entity='ORDERS.ORDERS_ID',
        target='ORDERS.K_SYMBOL',
        query_string='PREDICT ORDERS.K_SYMBOL\nFOR EACH ORDERS.ORDERS_ID',
        split=split_config,
        is_pquery_v2=True,
    ).to_predictive_query_v2_config()

    # Check BP Entity col name
    assert test_config.get_bp_entity_col_name() == 'ENTITY'


def test_pquery_split_config_deserialization(tmp_path, create_graph):
    # Create the graph and "infer" encoders
    graph: Graph = create_graph(name='FINANCIAL')
    graph.infer_metadata()
    for table in graph:
        for col in table:
            col.encoder = Null()

    # Create the config
    split_config = RandomSplit([0.8, 0.1, 0.1]).config()
    test_config = PredictiveQueryConfigBase(
        graph=graph.config(),
        name='testquery',
        entity='ORDERS.ORDERS_ID',
        target='ORDERS.K_SYMBOL',
        query_string='PREDICT ORDERS.K_SYMBOL\nFOR EACH ORDERS.ORDERS_ID',
        split=split_config,
    ).to_predictive_query_v2_config()

    # Serialize and deserialize the config
    cfg_path = os.path.join(tmp_path, 'config.json')
    save_json(test_config, cfg_path)
    deserialized_test_config = load_json(cfg_path, PredictiveQueryV2Config)

    # Don't test graph serialization, leave to different unit test
    test_config.graph = GraphConfig()
    deserialized_test_config.graph = GraphConfig()

    # Verify the split first (dicts should be the same)
    test_config_split_dict = test_config.split.__dict__
    test_config_split_dict.pop('__pydantic_initialised__', None)
    assert test_config_split_dict == deserialized_test_config.split
    assert Split.from_config(test_config.split) == Split.from_config(
        deserialized_test_config.split)
    test_config.split = SplitConfig()
    deserialized_test_config.split = SplitConfig()

    # Make sure everything else is the same
    assert test_config == deserialized_test_config


def test_pqlv2_forecast_length_deserialization(tmp_path,
                                               create_node_pred_graph):
    # Create the graph and "infer" encoders
    graph: Graph = create_node_pred_graph('test_temporal_graph', tmp_path.name)
    graph.infer_metadata()
    for table in graph:
        for col in table:
            col.encoder = Null()

    # Create the config
    split_config = TemporalSplit([0.8, 0.1, 0.1]).config()
    test_config = PredictiveQueryConfigBase(
        graph=graph.config(),
        name='testquery',
        entity='USERS.USER_ID',
        target='COUNT(TRANSACTIONS.*, 0, 10)',
        split=split_config,
        forecast_length=1,
        query_string=('PREDICT COUNT(TRANSACTIONS.*, 0, 10) '
                      'FOR EACH USERS.USER_ID'),
    ).to_predictive_query_v2_config()
    assert test_config.forecast_length == 1
