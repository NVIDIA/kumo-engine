# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pandas as pd
import pytest
from kumoapi.typing import Dtype

from kumo.config import BaseColumnConfig
from kumo.connector.bigquery_connector import BigQueryConnector
from kumo.connector.snowflake_connector import SnowflakeConnector


@pytest.fixture(params=['bigquery', 'snowflake'], ids=['bq', 'sf'])
def connector(request):
    if request.param == 'bigquery':
        return BigQueryConnector('sample', project_id='project',
                                 dataset_id='ds')
    return SnowflakeConnector('sample', 'account', 'warehouse', 'db', 'schema')


def test_table_sample_preserves_projection_index_and_limit(connector):
    connector._get_df_result = Mock(return_value=pd.DataFrame({
        'value': [1.25],
        'id': [7],
    }))
    result = connector._read_table_sample(
        None, 'events', 5, None, pkey=BaseColumnConfig(name='id',
                                                       dtype=Dtype.int64),
        cols=[BaseColumnConfig(name='value', dtype=Dtype.float64)])

    assert result.index.tolist() == [7]
    assert result.value.tolist() == [1.25]
    query = connector._get_df_result.call_args.args[1]
    projection = ', '.join(
        connector.sql_column_name(n) for n in ['value', 'id'])
    assert query == (f'SELECT {projection} FROM '
                     f'{connector.sql_table_name("events")} LIMIT 5')


@pytest.mark.parametrize('limit', [None, -1])
def test_source_sample_requires_a_bounded_query(connector, limit):
    with pytest.raises(ValueError):
        connector.get_query('events', limit=limit, sample=True)
