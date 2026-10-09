# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp

import pandas as pd
import pytest
from kumoapi.typing import Dtype, Stype

from kumo.connector.csv_connector import CSVConnector
from kumo.connector.databricks_connector import DatabricksConnector
from kumo.connector.snowflake_connector import SnowflakeConnector
from kumo.testing import withDatabricksCredentials, withSnowflakeCredentials
from service_framework.deploy_env import DATABRICKS_HOST


@pytest.mark.snowflake
@withSnowflakeCredentials
def test_csv_and_snowflake_connector():
    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'csv_data', 'FINANCIAL')
    connector1 = CSVConnector('csv', root_dir)

    connector2 = SnowflakeConnector('snowflake', account='xva19026',
                                    warehouse='WH_XS', database='KUMO',
                                    schema_name='FINANCIAL')

    pkey = 'LOAN_ID'
    cols = [
        ('ACCOUNT_ID', Stype.ID, Dtype.int),
        ('DATE', Stype.timestamp, Dtype.date),
        ('AMOUNT', Stype.numerical, Dtype.float),
        ('PAYMENTS', Stype.numerical, Dtype.float),
        ('STATUS', Stype.categorical, Dtype.string),
    ]

    df1 = connector1.read_table('LOAN', pkey, cols).sort_index()
    df2 = connector2.read_table('LOAN', pkey, cols).sort_index()
    # pandas-2 infers a narrower int dtype for the pkey index from one source
    # (int16) than the other (int64). check_dtype=False only relaxes column
    # dtypes, so also relax the index dtype to compare values, not dtypes.
    pd.testing.assert_frame_equal(df1, df2, check_dtype=False,
                                  check_index_type=False)


@withDatabricksCredentials
def test_databricks_connector():
    connector = DatabricksConnector(
        'databricks',
        host=DATABRICKS_HOST.parse_value(),
        cluster_id='0626-191119-75431625',
        warehouse_id='82d859e4a6b283a0',
        catalog='hive_mapped_uc',
        schema_name='hm',
    )

    pkey = 'customer_id'

    df = connector.read_table('hm.customers', pkey=pkey, num_rows=100)
    assert len(df) == 100
    for col in [
            'FN', 'Active', 'club_member_status', 'fashion_news_frequency',
            'age', 'postal_code'
    ]:
        assert col in df
    assert df.index.name == 'customer_id'
