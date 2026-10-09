# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime

import pandas as pd
from kumo_core.types.conversions import dtype_to_pandas
from kumoapi.typing import Dtype, Stype

from kumo.connector.parquet_connector import ParquetConnector


def test_parquet_connector_load_full_s3_dataset(
        tempdir, require_kumo_public_datasets_s3):
    connector = ParquetConnector(
        'parquet', root_dir="s3://kumo-public-datasets/financial/parquet/")

    pkey = 'TRANS_ID'
    cols = [
        ('ACCOUNT_ID', Stype.ID, Dtype.int),
        ('TYPE', Stype.categorical, Dtype.string),
        ('OPERATION', Stype.categorical, Dtype.string),
        ('AMOUNT', Stype.numerical, Dtype.float),
    ]

    table_name = 'TRANS'
    # TRANS table has 10 parquet files, each with equal number of rows
    exact_num_rows = connector._get_table_num_rows(table_name)
    assert exact_num_rows == 1056320

    df = connector.read_table(table_name, pkey, cols)
    assert len(df.columns) == len(cols)
    assert df.columns.values.tolist() == [col[0] for col in cols]
    assert df.dtypes.values.tolist() == [
        dtype_to_pandas(col[2]) for col in cols
    ]
    assert df.shape[0] == 1056320


def test_parquet_connector_common_error_year_timestamp(
        tempdir, require_kumo_public_datasets_s3):
    connector = ParquetConnector(
        'parquet', root_dir="s3://kumo-public-datasets/common_error/")

    cols = [
        ('timestamp', Stype.timestamp, Dtype.time),
        ('value', Stype.categorical, Dtype.int),
    ]

    table_name = 'out_of_range_year'

    df = connector.read_table(table_name, cols=cols)
    assert len(df.columns) == len(cols)
    assert df.columns.values.tolist() == [col[0] for col in cols]
    assert df.dtypes.values.tolist() == [
        dtype_to_pandas(col[2]) for col in cols
    ]
    assert df.shape[0] == 3

    assert pd.isna(df['timestamp'][0])
    assert pd.isna(df['timestamp'][1])
    assert df['timestamp'][2] == datetime.datetime(2000, 3, 1)
