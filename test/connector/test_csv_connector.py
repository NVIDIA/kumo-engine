# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo_core.types.conversions import dtype_to_pandas
from kumoapi.typing import Dtype, Stype

from kumo.connector.csv_connector import CSVConnector


def test_csv_connector_load_full_s3_dataset(tempdir,
                                            require_kumo_public_datasets_s3):
    connector = CSVConnector(
        'csv', root_dir="s3://kumo-public-datasets/financial/csv/")

    pkey = 'TRANS_ID'
    cols = [
        ('ACCOUNT_ID', Stype.ID, Dtype.int),
        ('TYPE', Stype.categorical, Dtype.string),
        ('OPERATION', Stype.categorical, Dtype.string),
        ('AMOUNT', Stype.numerical, Dtype.float),
    ]
    df = connector.read_table('TRANS', pkey, cols)
    assert len(df.columns) == len(cols)
    assert df.columns.values.tolist() == [col[0] for col in cols]
    assert df.dtypes.values.tolist() == [
        dtype_to_pandas(col[2]) for col in cols
    ]
    assert df.shape[0] == 1056320


def test_csv_connector_load_string_ids(tempdir,
                                       require_kumo_public_datasets_s3):
    connector = CSVConnector('csv',
                             root_dir="s3://kumo-public-datasets/test_csv/")

    df = connector.read_table('test_csv')
    # Note that if it becomes 3, the test should fail since id `01` and `1`
    # are different ids.
    assert len(df['id'].unique()) == 6

    pkey = 'id'
    cols = [
        ('id', Stype.ID, Dtype.string),
        ('value', Stype.numerical, Dtype.float),
    ]
    df = connector.read_table('test_csv', pkey, cols)
    assert len(df.index.unique()) == 6
