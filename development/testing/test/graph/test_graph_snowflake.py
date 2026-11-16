# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp

import pytest
from kumoapi.encoder import Datetime, Index, Numerical

from kumo.connector.snowflake_connector import SnowflakeConnector
from kumo.graph.graph import Graph
from kumo.testing import onlyFullTest, withSnowflakeCredentials
from kumo.testing.data import financial_csv_dir


@onlyFullTest
@pytest.mark.snowflake
@withSnowflakeCredentials
def test_graph_infer_encoders_from_snowflake():
    root_dir = financial_csv_dir()
    path = osp.join(root_dir, 'metadata.json')

    connector = SnowflakeConnector(
        'snowflake',
        account='xva19026',
        warehouse='WH_XS',
        database='KUMO',
        schema_name='FINANCIAL',
    )

    graph = Graph()
    graph.connect(connector)
    graph.add_from_config(path, data_source='snowflake')
    graph.infer_metadata()
    graph.infer_encoders()

    assert isinstance(graph['LOAN', 'DATE'].encoder, Datetime)
    assert isinstance(graph['LOAN', 'AMOUNT'].encoder, Numerical)
    assert isinstance(graph['LOAN', 'DURATION'].encoder, Index)
    assert isinstance(graph['LOAN', 'PAYMENTS'].encoder, Numerical)
    assert isinstance(graph['LOAN', 'STATUS'].encoder, Index)
