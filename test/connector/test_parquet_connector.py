# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import decimal
import logging
import os
import os.path as osp
import shutil
import tempfile
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest
from kumo_core.sources.object_store import column_dtypes
from kumo_core.types import ColStatType
from kumo_core.types.conversions import dtype_to_pandas
from kumoapi.typing import Dtype, Stype

from io_utils import fsutil
from kumo.config import BaseColumnConfig, TableMetadataConfig
from kumo.connector.connector import Connector, ConnectorError
from kumo.connector.parquet_connector import ParquetConnector
from kumo.connector.parquet_utils import (
    add_metadata_cols_from_parquet_dataset,
    clean_parquet_table,
    to_pandas_ns_safe,
    validate_parquet_schema,
)
from kumo.data import get_active_cache
from kumo.graph.graph import Graph
from kumo.home import get_home_dir
from kumo.pquery.AST import ast_node
from kumo.stats import compute_col_stats_python
from kumo.validation.validation import ValidationException
from statedb.config.typing import to_pandas_type

TEST_TABLE_PARQUET_FILE = "test_table.parquet"


def get_cur_dir_path(file_name="test_table.parquet"):
    root_dir = osp.dirname(osp.realpath(__file__))
    file_path = osp.join(root_dir, file_name)
    return file_path, root_dir


def teardown_function():
    file_path, _ = get_cur_dir_path(TEST_TABLE_PARQUET_FILE)
    if os.path.exists(file_path):
        os.remove(file_path)


def test_parquet_connector_config():
    connector = ParquetConnector('parquet', root_dir='my_test_dir')

    assert asdict(connector.config()) == {
        '_target_': 'kumo.connector.parquet_connector.ParquetConnector',
        'name': 'parquet',
        'root_dir': 'my_test_dir',
        'suffix': 'parquet',
    }
    assert connector == Connector.from_config(connector.config())


def test_parquet_connector_load():
    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'parquet_data', 'FINANCIAL')
    connector = ParquetConnector('parquet', root_dir=root_dir)

    pkey = 'TRANS_ID'
    cols = [
        ('ACCOUNT_ID', Stype.ID, Dtype.int),
        ('DATE', Stype.timestamp, Dtype.date),
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

    df = connector.read_table('TRANS', pkey, cols, num_rows=1234)
    assert df.shape[0] == 1234

    df = connector.read_table_sample('TRANS', pkey, cols, num_rows=12)
    assert df.shape[0] == 12


def test_bounded_read_spans_row_groups_and_dataset_files(tmp_path):
    root = tmp_path / 'parts'
    root.mkdir()
    for part, start in enumerate((0, 4)):
        table = pa.table(
            {'id': pa.array(range(start, start + 4), type=pa.int64())})
        pq.write_table(table, root / f'part-{part}.parquet', row_group_size=2)

    connector = ParquetConnector('parquet', root_dir=str(tmp_path))
    sample = connector.load_dataset(str(root), 'parquet', cols=['id'],
                                    num_rows=6)

    assert sample.columns.tolist() == ['id']
    assert sample['id'].tolist() == list(range(6))
    assert connector.load_dataset(str(root), 'parquet', cols=[],
                                  num_rows=6).shape == (6, 0)


def test_parquet_connector_load_ordered(mocker):
    root_dir = osp.dirname(osp.realpath(__file__))
    # Use abspath to remove the .. in the path
    root_dir = os.path.abspath(osp.join(root_dir, '..', 'parquet_data'))
    connector = ParquetConnector('parquet', root_dir=root_dir)
    spy_load_file = mocker.spy(connector, 'load_file')
    # Call read_table_sample to load the table
    df = connector.read_table_sample('COUPON_PURCHASE_PREDICTION', num_rows=12,
                                     skip_cache=True)
    assert df.shape[0] == 12
    # Check that the smallest file was loaded.
    spy_load_file.assert_called_once_with(
        osp.join(root_dir, 'COUPON_PURCHASE_PREDICTION', 'USER_LIST.parquet'),
        cols=None, num_rows=12, col_types={})


def test_error_on_large_partition(monkeypatch: pytest.MonkeyPatch, caplog):
    root_dir = osp.dirname(osp.realpath(__file__))
    # Use abspath to remove the .. in the path
    root_dir = os.path.abspath(osp.join(root_dir, '..', 'parquet_data'))
    connector = ParquetConnector('parquet', root_dir=root_dir)

    monkeypatch.setenv('PARQUET_PARTITION_SIZE', '1')
    with caplog.at_level(logging.ERROR):
        connector.read_table_sample('COUPON_PURCHASE_PREDICTION', num_rows=12,
                                    skip_cache=True)
    assert 'Bad parquet partitioning' in caplog.text


def test_object_store_hive_dataset_preserves_partitions(monkeypatch, tmp_path):
    root = tmp_path / 'events'
    files = []
    for country, year, row_id in [('US', 2025, 1), ('CA', 2026, 2)]:
        file_path = (root / f'country={country}' / f'year={year}' /
                     'day=2026-07-11' / 'part-000.parquet')
        file_path.parent.mkdir(parents=True)
        pd.DataFrame({
            'id': [row_id, row_id + 10]
        }).to_parquet(file_path, index=False)
        files.append(file_path)

    physical = tmp_path / 'physical'
    monkeypatch.setattr('kumo.data.utils.get_materialize_path',
                        lambda *args: str(physical))
    ParquetConnector('parquet',
                     str(tmp_path)).export_table('events',
                                                 SimpleNamespace(cols={}),
                                                 'version')
    physical_table = pd.read_parquet(physical)
    assert physical_table.columns.tolist() == ['id', 'country', 'year', 'day']
    assert sorted(physical_table['country']) == ['CA', 'CA', 'US', 'US']
    assert sorted(physical_table['year']) == [2025, 2025, 2026, 2026]

    monkeypatch.setattr('kumo.connector.object_store_access.source_access',
                        lambda path: None)
    uri = f"gs://{root.as_posix().lstrip('/')}"
    monkeypatch.setattr(fsutil, 'is_gcs', lambda path: path == uri)
    monkeypatch.setattr(fsutil, 'is_s3', lambda path: False)
    monkeypatch.setattr(fsutil, 'isdir', lambda path: path == uri)
    monkeypatch.setattr(
        'kumo.connector.parquet_connector._list_dataset_files',
        lambda path, suffixes:
        [f"gs://{file.as_posix().lstrip('/')}" for file in files],
    )
    monkeypatch.setattr(
        fsutil,
        'pyarrow_filesystem',
        lambda path: pafs.SubTreeFileSystem('/', pafs.LocalFileSystem()),
    )

    dataset = ParquetConnector('parquet', uri)._parquet_dataset(uri)
    assert dataset.schema.names == ['id', 'country', 'year', 'day']
    assert sorted(
        dataset.to_table().to_pandas()['country']) == ['CA', 'CA', 'US', 'US']
    sample = ParquetConnector('parquet',
                              uri).load_dataset(uri, 'parquet', num_rows=4)
    assert sorted(sample['country']) == ['CA', 'CA', 'US', 'US']
    assert isinstance(sample['country'].dtype, pd.CategoricalDtype)
    assert sorted(sample['year']) == [2025, 2025, 2026, 2026]
    partition_only = ParquetConnector('parquet',
                                      uri).load_dataset(uri, 'parquet',
                                                        cols=['country'],
                                                        num_rows=1)
    assert partition_only.columns.tolist() == ['country']
    assert len(partition_only) == 1
    assert partition_only['country'].iloc[0] in {'CA', 'US'}


def test_object_store_flat_dataset_uses_physical_schema(monkeypatch, tmp_path):
    root = tmp_path / 'events'
    root.mkdir()
    files = [root / 'part-1.parquet', root / 'part-2.parquet']
    for row_id, file_path in enumerate(files, start=1):
        pd.DataFrame({'id': [row_id]}).to_parquet(file_path, index=False)

    monkeypatch.setattr('kumo.connector.object_store_access.source_access',
                        lambda path: None)
    uri = f"gs://{root.as_posix().lstrip('/')}"
    monkeypatch.setattr(fsutil, 'is_gcs', lambda path: path == uri)
    monkeypatch.setattr(fsutil, 'is_s3', lambda path: False)
    monkeypatch.setattr(fsutil, 'isdir', lambda path: path == uri)
    monkeypatch.setattr(
        'kumo.connector.parquet_connector._list_dataset_files',
        lambda path, suffixes:
        [f"gs://{file.as_posix().lstrip('/')}" for file in files],
    )
    monkeypatch.setattr(
        fsutil,
        'pyarrow_filesystem',
        lambda path: pafs.SubTreeFileSystem('/', pafs.LocalFileSystem()),
    )

    dataset = ParquetConnector('parquet', uri)._parquet_dataset(uri)
    assert dataset.schema.names == ['id']
    assert sorted(dataset.to_table().to_pandas()['id']) == [1, 2]


# test that we properly ignore __index_level_x__ columns which are
# parquet read/write artifacts
def test_parquet_connector_index_level_col():
    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'parquet_data', 'NETFLIX')
    connector = ParquetConnector('parquet', root_dir=root_dir)
    # read_table_sample adds it to the backend cache too
    table_sample = connector.read_table_sample(table_name='Movies', num_rows=10)
    assert set(table_sample.columns) == {'MovieID', 'YearOfRelease', 'Title'}

    # save and reload the cache sample to check that the parquet index col
    # is not added
    table_version = connector.get_data_version('Movies')
    # Create a TableFingerprint for the table
    from kumo_core.types.schema import IngestedTableFingerprint
    table_fp = IngestedTableFingerprint(connector.name, 'Movies', cols={},
                                        pkey=None, time_col=None,
                                        end_time_col=None)
    get_active_cache().save_col_stats_cache(table_fp, table_version)
    get_active_cache().load_col_stats_cache(table_fp, table_version)
    cache_sample = get_active_cache().table_samples_cache.get_v1(
        connector.name, 'Movies', table_version)
    assert set(cache_sample.columns) == {'MovieID', 'YearOfRelease', 'Title'}


def test_parquet_connector_infer_timestamp_dtypes():
    # Generate a test DataFrame
    file_path, root_dir = get_cur_dir_path(TEST_TABLE_PARQUET_FILE)
    df = pd.DataFrame({
        "0": [0, 1],
        "1": ["random", "string"],
        "2": [pd.Timestamp('2017-01-01T12'),
              pd.Timestamp('2018-01-01T12')],
        "some_valid_DATE": ["05/11/2011", "03/11/2011"],
        "another_TIME": ["23:58:27", "22:53:21"],
        "bad_DATE": ["wait! these", "aren't datas"],
        "OOB_DATE": ["05/11/1011", "01/05/2994"],
        "not_DATE": [123123.01, 420e-13]
    })
    # Write it to Parquet file
    df.to_parquet(file_path)

    # Read if from ParquetConnector
    connector = ParquetConnector('parquet', root_dir=root_dir)
    metadata = connector.get_table_metadata(["test_table"])[0]
    assert metadata.cols[0].dtype == Dtype.int64
    assert metadata.cols[1].dtype == Dtype.string
    assert metadata.cols[2].dtype == Dtype.date
    assert metadata.cols[3].dtype == Dtype.string
    assert metadata.cols[4].dtype == Dtype.string
    assert metadata.cols[5].dtype == Dtype.string
    assert metadata.cols[6].dtype == Dtype.string
    assert metadata.cols[7].dtype == Dtype.float64


def test_parquet_connector_read_boolean_dtypes():
    # Generate a test DataFrame
    file_path, root_dir = get_cur_dir_path(TEST_TABLE_PARQUET_FILE)
    int_ser = pd.Series([0, 1, 2], dtype="int64")
    bool_ser = pd.Series([True, False, True], dtype="bool")
    df = pd.DataFrame({
        "0": int_ser,
        "1": bool_ser,
    })
    # Write it to Parquet file
    df.to_parquet(file_path)

    # Read if from ParquetConnector
    connector = ParquetConnector('parquet', root_dir=root_dir)
    metadata = connector.get_table_metadata(["test_table"])[0]
    assert metadata.cols[0].dtype == Dtype.int64
    assert metadata.cols[1].dtype == Dtype.bool

    # Reading the data should have Pd.BooleanDtype for the boolean column
    df = connector.read_table('test_table', pkey='0',
                              cols=[('1', Stype.categorical, Dtype.bool)])
    assert df.dtypes.values.tolist() == [pd.BooleanDtype()]


def test_parquet_connector_handle_unsupported_types():
    import pandas.api.types as ptypes
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Generate a test DataFrame
    file_path, root_dir = get_cur_dir_path(TEST_TABLE_PARQUET_FILE)
    df = pd.DataFrame({
        "int_col": [-1, 0, 1],
        "list_col": [[1.0, 2.0], [3.1, 4.1], [5.1, 6.1]],
    })

    table = pa.Table.from_pandas(df)
    # Write it to Parquet file
    pq.write_table(table, file_path)

    # Read if from ParquetConnector
    connector = ParquetConnector('parquet', root_dir=root_dir)
    metadata = connector.get_table_metadata(["test_table"])[0]
    assert metadata.cols[0].dtype == Dtype.int64
    assert metadata.cols[1].dtype == Dtype.floatlist

    # Read TEST dataset
    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'parquet_data', 'TEST')
    connector = ParquetConnector('parquet', root_dir=root_dir)
    metadata = connector.get_table_metadata(["TRANS4"])[0]
    assert metadata.cols[0].dtype == Dtype.int64
    assert metadata.cols[1].dtype == Dtype.unsupported  # null type
    assert metadata.cols[2].dtype == Dtype.int64
    assert metadata.cols[3].dtype == Dtype.intlist

    pkey = None
    cols = [
        BaseColumnConfig('ACCOUNT_ID', None, Dtype.int64),
        BaseColumnConfig('AMOUNT', None, Dtype.unsupported),
        BaseColumnConfig('TRANS_ID', None, Dtype.int64),
        BaseColumnConfig('UNSUPPORTED', None, Dtype.unsupported),
    ]
    df = connector.read_table('TRANS4', pkey, cols)
    assert len(df.columns) == 4  # can't read unsupported columns
    assert df.shape[0] == 5000
    assert ptypes.is_string_dtype(df['AMOUNT'])
    assert ptypes.is_string_dtype(df['UNSUPPORTED'])

    df = connector.read_table_sample('TRANS4', pkey, cols, num_rows=100)
    assert len(df.columns) == 4  # can't read unsupported columns
    assert df.shape[0] == 100
    assert ptypes.is_string_dtype(df['AMOUNT'])
    assert ptypes.is_string_dtype(df['UNSUPPORTED'])


def test_parquet_connector_handle_categorical_types():
    # Generate a test DataFrame
    file_path, root_dir = get_cur_dir_path(TEST_TABLE_PARQUET_FILE)
    cat_ser = pd.Series(
        ['Awesome', 'Amazing', 'Awesome'],
        dtype=(pd.CategoricalDtype(categories=['Awesome', 'Amazing'])))
    df = pd.DataFrame({
        "0": cat_ser,
    })
    # Write it to Parquet file
    df.to_parquet(file_path)

    # Read if from ParquetConnector
    connector = ParquetConnector('parquet', root_dir=root_dir)
    metadata = connector.get_table_metadata(["test_table"])[0]
    col = connector.read_table('test_table', cols=metadata.cols)['0'].tolist()
    assert col == ['Awesome', 'Amazing', 'Awesome']
    assert metadata.cols[0].dtype == Dtype.string


def test_parquet_connector_list_types(monkeypatch):
    monkeypatch.setattr(
        ast_node,
        'ALLOW_ARRAY_TARGETS',
        True,
    )
    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'parquet_data', 'movielens_lists')
    connector = ParquetConnector('parquet', root_dir=root_dir)
    metadata = connector.get_table_metadata(["movie_stringlist"])[0]

    genres_col = [col for col in metadata.cols if col.name == 'genres'][0]
    assert genres_col.dtype == Dtype.stringlist

    metadata = connector.get_table_metadata(["movie_intlist"])[0]
    genres_col = [col for col in metadata.cols if col.name == 'genres'][0]
    assert genres_col.dtype == Dtype.intlist


def test_parquet_connector_load_dataset():
    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'parquet_data')
    connector = ParquetConnector('parquet', root_dir=root_dir)
    table_name = 'trans_dataset'

    # trans_dataset table has 3 parquet files, each with equal number of rows
    exact_num_rows = connector._get_table_num_rows(table_name)
    assert exact_num_rows == 3072

    pkey = 'TRANS_ID'
    cols = [
        ('ACCOUNT_ID', Stype.ID, Dtype.int),
        ('TYPE', Stype.categorical, Dtype.string),
        ('OPERATION', Stype.categorical, Dtype.string),
        ('AMOUNT', Stype.numerical, Dtype.float),
    ]
    df = connector.read_table(table_name, pkey, cols)
    assert len(df.columns) == len(cols)
    assert df.columns.values.tolist() == [col[0] for col in cols]
    assert df.dtypes.values.tolist() == [
        dtype_to_pandas(col[2]) for col in cols
    ]
    assert df.shape[0] == 3072

    df = connector.read_table(table_name, pkey, cols, num_rows=1234)
    assert df.shape[0] == 1234

    df = connector.read_table_sample(table_name, pkey, cols, num_rows=12)
    assert df.shape[0] == 12

    frac = 0.1
    df = connector.read_table_sample(table_name, pkey, cols, frac=frac)
    assert df.shape[0] == 307


def test_parquet_connector_decimal():
    import pyarrow as pa

    data = [{
        "name": "ted",
        "age": 34,
        "value": 0.10
    }, {
        "name": "barney",
        "age": 35,
        "value": 1.2
    }]
    df = pd.DataFrame.from_dict(data, )
    df['age'] = df['age'].astype(int)
    df['age_decimal'] = df['age'].astype(str).map(decimal.Decimal)
    df['value'] = df['value'].astype(float, 2)
    df['value_decimal'] = df['value'].astype(str).map(decimal.Decimal)
    table = pa.Table.from_pandas(df)

    table = clean_parquet_table(table)
    df = table.to_pandas(types_mapper=to_pandas_type, timestamp_as_object=True)
    assert df.dtypes.values.tolist() == [
        pd.StringDtype(storage='pyarrow'),
        np.dtype('int64'),
        np.dtype('float64'),
        np.dtype('int64'),
        np.dtype('float64')
    ]


def test_parquet_connector_decimal_to_str():
    import pyarrow as pa

    data = [{
        "value": decimal.Decimal(12345678901234567),
    }, {
        "value": decimal.Decimal(12345678901234568),
    }]
    df = pd.DataFrame.from_dict(data)
    col_types = {
        "value": np.dtype('int64'),
    }
    table = clean_parquet_table(pa.Table.from_pandas(df), col_types)
    df = table.to_pandas(types_mapper=to_pandas_type, timestamp_as_object=True)
    assert df.dtypes.values.tolist() == [np.dtype('int64')]


def test_parquet_connector_validate():
    import pyarrow as pa

    data = [{
        "value": decimal.Decimal(12345678901234567),
        "sv": "abc",
    }, {
        "value": decimal.Decimal(12345678901234568),
        "sv": "abc",
    }]
    df = pd.DataFrame.from_dict(data)
    validate_parquet_schema(pa.Table.from_pandas(df).schema)

    # Dictionary types are not supported in Kumo, should fail:
    with pytest.raises(ValidationException):
        data = [{
            'col1': 1,
            'col2': 'a',
            'col3': {
                'a': 'f'
            }
        }, {
            'col1': 2,
            'col2': 'b',
            'col3': {
                'b': 'e'
            }
        }, {
            'col1': 3,
            'col2': 'c',
            'col3': {
                'c': 'd'
            }
        }]
        df = pd.DataFrame.from_records(data)
        validate_parquet_schema(pa.Table.from_pandas(df).schema)


def test_parquet_connector_list_tables():
    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'parquet_data', 'FINANCIAL')
    connector = ParquetConnector('parquet', root_dir=root_dir)
    # TODO(Dulloor): source_name is not populated by get_table_metadata
    table_metadata_configs = connector.get_table_metadata(ignore_errors=True)
    expect_names = [
        'ACCOUNT',
        'ACCOUNT_NA',
        'CARD',
        'CLIENT',
        'CLIENT_INDEXED',
        'DISP',
        'DISTRICT',
        'LOAN',
        'LOAN_TRAIN_SCHEMA',
        'ORDERS',
        'PREDICTION',
        'TRAINING_BINARY',
        'TRAINING_CLASSIFICATION',
        'TRAINING_REGRESSION',
        'TRANS',
        'TRANS_SMALL',
    ]
    assert len(table_metadata_configs) == len(expect_names)
    names = sorted(table.table_name for table in table_metadata_configs)
    assert names == expect_names

    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'parquet_data', 'TEST')
    connector = ParquetConnector('parquet', root_dir=root_dir)
    table_metadata_configs = connector.get_table_metadata(ignore_errors=True)
    assert len(table_metadata_configs) == 4
    assert 'TRANS' in table_metadata_configs[0].table_name
    assert 'TRANS' in table_metadata_configs[1].table_name
    assert 'TRANS' in table_metadata_configs[2].table_name
    assert 'TRANS' in table_metadata_configs[3].table_name

    with pytest.raises(ConnectorError):
        table_metadata_configs = connector.get_table_metadata()


def test_parquet_connector_iterator():
    root_dir = osp.dirname(osp.realpath(__file__))
    input_dir = osp.join(root_dir, '..', 'parquet_data')
    connector = ParquetConnector('parquet', root_dir=input_dir)

    pkey = 'TRANS_ID'
    cols = [
        ('ACCOUNT_ID', Stype.ID, Dtype.int),
        ('TYPE', Stype.categorical, Dtype.string),
        ('OPERATION', Stype.categorical, Dtype.string),
        ('AMOUNT', Stype.numerical, Dtype.float),
    ]
    df = connector.read_table('trans_dataset', pkey, cols)

    test_dir = osp.join(root_dir, '..', 'test_parquet_data')
    try:
        fsutil.makedirs(osp.join(test_dir, 'TRANS_TEST'), exist_ok=True)
        for i in range(int(len(df) / 1000) + 1):
            seg_df = df[i * 1000:i * 1000 + 1000]
            seg_df.to_parquet(
                osp.join(test_dir, 'TRANS_TEST', 'seg_%d.parquet' % i))

        connector = ParquetConnector('parquet', root_dir=test_dir)
        test_df = pd.concat([
            d
            for d in connector.get_iterator('TRANS_TEST', pkey=pkey, cols=cols)
        ])

        assert test_df.equals(df)
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


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


def test_write_parquet_table():
    data = [{
        "name": "ted",
        "age": 34,
        "value": 0.10
    }, {
        "name": "barney",
        "age": 35,
        "value": 1.2
    }]
    df = pd.DataFrame.from_dict(data, )

    temp_dir = osp.join(get_home_dir(), 'test_csv')
    fsutil.makedirs(temp_dir, exist_ok=True)
    connector = ParquetConnector('csv', root_dir=temp_dir)
    connector.write_pandas('test', df)
    assert fsutil.exists(osp.join(temp_dir, 'test.parquet'))
    fsutil.rm(temp_dir)


def test_parquet_col_types_infer_and_cast():
    # Kumo-owned col-type infer/cast assertions over LOCAL parquet. The
    # removed remote-catalog parametrization and the trailing
    # ``cast_spark_df`` / ``_get_spark_df`` block
    # (the removed legacy Backend-ABC factory path that this row was parked
    # on) were dropped. The surviving load-bearing surface is the local
    # dtype + stype inference and the ``compute_col_stats_python`` NUM_NA
    # checks over a read-table-sample frame.
    root = osp.join(osp.dirname(osp.realpath(__file__)), '../parquet_data',
                    'SYNTHETIC')
    connector = ParquetConnector('parquet', root_dir=root)

    table_metadata_configs = connector.get_table_metadata()

    graph = Graph(name='test_graph')
    graph.connect(connector)

    table_name_1 = 'synthetic_types_variety_1'
    table_name_2 = 'synthetic_types_variety_2'
    table_name_3 = 'synthetic_types_id_float'

    graph.add(table_name_1)
    graph.add(table_name_2)
    graph.add(table_name_3)

    expected_dtypes = {
        f'parquet.{table_name_2}': {
            'ID': Dtype.int32,
            'integer': Dtype.int64,
            'uinteger': Dtype.int64,
            'integer8': Dtype.int32,
            'duration': Dtype.timedelta,  # parquet type = duration[ns]
            'seconds': Dtype.string,
            'date': Dtype.string,
            'lists': Dtype.stringlist,
            'multicat1': Dtype.string,
            'multicat2': Dtype.string
        },
        # DAT_1 and TIME_1 have string types from pyarrow since they are not
        # time since epoch formatted - date formats like DAT_1 can be manually
        # changed to a a timestamp stype but are not automatically assumed
        f'parquet.{table_name_1}': {
            'NUM_1': Dtype.int64,
            'CHAR_1': Dtype.string,
            'CHAR_2': Dtype.string,
            'CHAR_3': Dtype.string,
            'BIN_1': Dtype.string,
            'BOOL': Dtype.bool,
            'DAT_1': Dtype.string,
            'DAT_2': Dtype.date,
            'TIME_1': Dtype.string,
            'TIME_2': Dtype.date,
            'TIME_3': Dtype.date,
            'TIME_4': Dtype.date,
            'TIME_5': Dtype.date,
        },
        f'parquet.{table_name_3}': {
            'ID': Dtype.float64
        }
    }
    expected_stypes = {
        f'parquet.{table_name_2}': {
            'ID': Stype.ID,
            'integer': Stype.numerical,
            'uinteger': Stype.numerical,
            'integer8': Stype.categorical,
            'duration': Stype.numerical,
            # Duration-like strings do not prove a supported multicategorical
            # delimiter encoding.
            'seconds': Stype.categorical,
            'date': Stype.categorical,
            # Current Kumo schema inference returns ``multicategorical`` for
            # a stringlist column (``Table._infer_stype_from_dtype``); the
            # legacy backend inferred ``sequence`` here. Reconciliation of the
            # stringlist stype decision is tracked as the Item-12 dg-bug (see
            # the sibling ``test_composite_file_connector.test_parquet_col_
            # types_infer_and_cast`` xfail).
            'lists': Stype.multicategorical,
            'multicat1': Stype.multicategorical,
            'multicat2': Stype.multicategorical
        },
        f'parquet.{table_name_1}': {
            'NUM_1': Stype.numerical,
            'CHAR_1': Stype.categorical,
            'CHAR_2': Stype.categorical,
            'CHAR_3': Stype.categorical,
            'BIN_1': Stype.categorical,
            'BOOL': Stype.categorical,
            'DAT_1': Stype.categorical,
            'DAT_2': Stype.timestamp,
            # Colon-formatted time text is not delimiter-encoded
            # multicategorical data.
            'TIME_1': Stype.categorical,
            'TIME_2': Stype.timestamp,
            'TIME_3': Stype.timestamp,
            'TIME_4': Stype.timestamp,
            'TIME_5': Stype.timestamp,
        },
        f'parquet.{table_name_3}': {
            'ID': Stype.numerical
        }
    }

    table_dict = {table.name: table for table in table_metadata_configs}

    for table_name in [table_name_1, table_name_2, table_name_3]:
        table = table_dict[f'parquet.{table_name}']

        for col in table.cols:
            if col.dtype != Dtype.unsupported:
                graph[table.name].add_col(col.name, dtype=col.dtype)
            assert col.dtype == expected_dtypes[table.name][col.name]

        graph[table.name].infer_stypes()

        for col in graph[table.name].cols:
            assert col.stype == expected_stypes[table.name][col.name]

    stats_dict_1 = {
        col: [ColStatType.NUM_NA]
        for col in graph[table_name_1].metadata_config().all_cols()
    }

    stats_dict_2 = {
        col: [ColStatType.NUM_NA]
        for col in graph[table_name_2].metadata_config().all_cols()
    }

    # read table sample checks if the data can be cast to the assigned types
    # without failures
    df = connector.read_table_sample(
        table_name_1,
        cols=[col.base_config() for col in graph[table_name_1].cols])
    assert df.shape[0] == 100
    assert df.shape[1] == 13

    for col, stats in stats_dict_1.items():
        stats_result = compute_col_stats_python(df[col.name], stats)
        assert stats_result[ColStatType.NUM_NA].value == 0

    df = connector.read_table_sample(
        table_name_2,
        cols=[col.base_config() for col in graph[table_name_2].cols])
    assert df.shape[0] == 10000
    assert df.shape[1] == 10

    for col, stats in stats_dict_2.items():
        stats_result = compute_col_stats_python(df[col.name], stats)
        assert stats_result[ColStatType.NUM_NA].value == 0


def test_duration_cast_to_timestamp():
    root = osp.join(osp.dirname(osp.realpath(__file__)), '../parquet_data',
                    'SYNTHETIC')
    connector = ParquetConnector('parquet', root_dir=root)
    table_metadata_configs = connector.get_table_metadata()

    graph = Graph(name='test_graph')
    graph.connect(connector)

    table_name_2 = 'synthetic_types_variety_2'

    graph.add(table_name_2)

    table_dict = {table.name: table for table in table_metadata_configs}
    table = table_dict[f'parquet.{table_name_2}']

    for col in table.cols:
        if col.dtype != Dtype.unsupported:
            graph[table.name].add_col(col.name, dtype=col.dtype)

    graph[table.name].infer_stypes()

    graph[table.name]['duration'].stype = Stype.timestamp

    # drop a column that we allow in source but not in actual table
    del graph[table.name]['lists']

    graph[table.name].infer_dtypes_from_stypes()

    connector.read_table_sample(
        table_name_2,
        cols=[col.base_config() for col in graph[table.name].cols])


def test_numeric_no_cast_to_timestamp():
    root = osp.join(osp.dirname(osp.realpath(__file__)), '../parquet_data',
                    'SYNTHETIC')
    connector = ParquetConnector('parquet', root_dir=root)
    table_metadata_configs = connector.get_table_metadata()

    graph = Graph(name='test_graph')
    graph.connect(connector)

    table_name_2 = 'synthetic_types_timetest'

    graph.add(table_name_2)

    table_dict = {table.name: table for table in table_metadata_configs}
    table = table_dict[f'parquet.{table_name_2}']

    for col in table.cols:
        if col.dtype != Dtype.unsupported:
            graph[table.name].add_col(col.name, dtype=col.dtype)

    graph[table.name].infer_stypes()

    graph[table.name].infer_dtypes_from_stypes()

    connector.read_table_sample(
        table_name_2,
        cols=[col.base_config() for col in graph[table.name].cols])


def test_tz_cast():
    """Exercise the DG-native TZ-preservation path.

    The legacy assertions used ``cast_spark_df`` to verify that a mixed-TZ
    parquet table produces a non-null TIMESTAMP column and that row-count
    after dropping NA is 1000. The DG-native equivalent reads via
    ``read_table_sample`` and inspects the resulting pandas frame; the Rust
    ingest path preserves TZ-bearing rows, so all 1000 survive.
    """
    root = osp.join(osp.dirname(osp.realpath(__file__)), '../parquet_data',
                    'SYNTHETIC')
    connector = ParquetConnector('parquet', root_dir=root)

    graph = Graph(name='test_graph')
    graph.connect(connector)

    table_name = 'synthetic_mixed_tz'
    graph.add(table_name)

    graph.infer_metadata()

    df = connector.read_table_sample(
        table_name,
        cols=[col.base_config() for col in graph[table_name].cols],
        num_rows=1000,
    )
    assert df['TIMESTAMP'].isna().sum() == 0
    assert len(df.dropna(subset=['TIMESTAMP'])) == 1000


def test_to_pandas_ns_safe_clamps_out_of_range_timestamp(caplog):
    """An out-of-nanosecond-range timestamp must be visibly clamped (with a
    warning), never silently integer-wrapped.

    With ``safe=False`` + ``coerce_temporal_nanoseconds=True``, pyarrow used
    to wrap a year-3000 ``us`` timestamp to ~1830 with no error. The
    nanosecond-safe conversion must instead clamp it to ``pd.Timestamp.max``
    and log which column was affected.
    """
    import pyarrow as pa

    in_range = datetime.datetime(2000, 3, 1, 12, 30)
    future = datetime.datetime(3000, 1, 1)  # well past pd.Timestamp.max (2262)
    past = datetime.datetime(1011, 5, 11)  # well before pd.Timestamp.min (1677)

    table = pa.table({
        'ts':
        pa.array([in_range, future, past, None], type=pa.timestamp('us'))
    })

    with caplog.at_level(logging.WARNING):
        df = to_pandas_ns_safe(table, types_mapper=to_pandas_type)

    col = df['ts']
    # Kumo's datetime64[ns] contract is preserved for the common case.
    assert col.dtype == np.dtype('datetime64[ns]')

    # In-range value round-trips exactly.
    assert col[0] == pd.Timestamp('2000-03-01 12:30:00')

    # Out-of-range values are clamped to the boundaries, NOT wrapped.
    assert col[1] == pd.Timestamp.max.floor('us')
    assert col[2] == pd.Timestamp.min.ceil('us')
    # Specifically, the year-3000 value must not have wrapped to ~1830.
    assert col[1].year == 2262
    assert col[1] > pd.Timestamp('2200-01-01')

    # NULLs stay NULL.
    assert pd.isna(col[3])

    # The clamp is visible: a warning naming the column was emitted.
    assert "'ts'" in caplog.text
    assert 'clamped' in caplog.text


def test_to_pandas_ns_safe_in_range_roundtrip(caplog):
    """In-range temporal values must round-trip unchanged at ns resolution,
    with no warning emitted."""
    import pyarrow as pa

    values = [
        datetime.datetime(1970, 1, 1),
        datetime.datetime(2000, 3, 1, 12, 30, 45),
        datetime.datetime(2018, 1, 1, 12),
    ]
    for unit in ('s', 'ms', 'us'):
        table = pa.table({'ts': pa.array(values, type=pa.timestamp(unit))})
        with caplog.at_level(logging.WARNING):
            caplog.clear()
            df = to_pandas_ns_safe(table, types_mapper=to_pandas_type)
        assert df['ts'].dtype == np.dtype('datetime64[ns]')
        assert df['ts'].tolist() == [pd.Timestamp(v) for v in values]
        # No clamping should have happened for in-range data.
        assert 'clamped' not in caplog.text


def test_to_pandas_ns_safe_clamps_out_of_range_duration(caplog):
    """Out-of-range duration columns wrap silently under the old conversion;
    they must be clamped instead."""
    import pyarrow as pa

    # ~1.16e13 days far exceeds the ns-representable timedelta range.
    table = pa.table(
        {'dur': pa.array([1, 10**18, -10**18, None], type=pa.duration('s'))})
    with caplog.at_level(logging.WARNING):
        df = to_pandas_ns_safe(table, types_mapper=to_pandas_type)

    col = df['dur']
    assert col.dtype == np.dtype('timedelta64[ns]')
    assert col[0] == pd.Timedelta(seconds=1)
    # Clamped to in-range boundaries (not wrapped to a small/negative value).
    assert col[1] == pd.Timedelta(pd.Timedelta.max.value // 1_000_000_000 *
                                  1_000_000_000)
    assert col[1] > pd.Timedelta(days=100_000)
    assert col[2] < pd.Timedelta(days=-100_000)
    assert pd.isna(col[3])
    assert "'dur'" in caplog.text


def test_parquet_connector_out_of_range_timestamp_not_wrapped():
    """End-to-end: a parquet file with a far-future timestamp column read via
    ParquetConnector must clamp (not wrap) the out-of-range value."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    file_path, root_dir = get_cur_dir_path(TEST_TABLE_PARQUET_FILE)
    table = pa.table({
        'id':
        pa.array([0, 1, 2], type=pa.int64()),
        'ts':
        pa.array(
            [
                datetime.datetime(2000, 3, 1),
                datetime.datetime(3000, 1, 1),  # out of ns range
                datetime.datetime(2018, 6, 15),
            ],
            type=pa.timestamp('us'),
        ),
    })
    pq.write_table(table, file_path)

    connector = ParquetConnector('parquet', root_dir=root_dir)
    df = connector.read_table('test_table', pkey='id',
                              cols=[('ts', Stype.timestamp, Dtype.date)])

    ts = df['ts']
    assert ts.dtype == np.dtype('datetime64[ns]')
    assert ts.iloc[0] == pd.Timestamp('2000-03-01')
    # The year-3000 value is clamped to the max representable ns timestamp,
    # NOT silently wrapped to ~1830.
    assert ts.iloc[1].year == 2262
    assert ts.iloc[1] > pd.Timestamp('2200-01-01')
    assert ts.iloc[2] == pd.Timestamp('2018-06-15')


def test_parquet_connector_dont_infer_temporal_types():
    """
    Test to check that Parquet connector preserves original types for
    DATE/TIME columns of type int/string rather than inferring them as since we
    skip read_table_sample().
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        data = {
            'DATE_1': [
                '2024-01-01 00:00:00', '2024-02-15 00:00:00',
                '2024-03-30 00:00:00', '2024-04-12 00:00:00',
                '2024-05-25 00:00:00'
            ],
            'TIME_1':
            [1704060000, 1707984000, 1711800000, 1712908800, 1716609600]
        }
        df = pd.DataFrame(data)

        table_name = 'preserve_types_test'
        table_path = os.path.join(tmp_dir, f'{table_name}.parquet')
        df.to_parquet(table_path)

        connector = ParquetConnector('parquet', root_dir=tmp_dir)
        table_metadata_configs = connector.get_table_metadata([table_name])
        expected_dtypes = {
            table_name: {
                'DATE_1': Dtype.string,
                'TIME_1': Dtype.int64,
            }
        }

        for col in table_metadata_configs[0].cols:
            assert col.dtype == expected_dtypes[table_name][col.name]


def test_hosted_and_local_parquet_schema_inference_share_types():
    schema = pa.schema([
        ('small_integer', pa.decimal128(18, 0)),
        ('wide_integer', pa.decimal128(38, 0)),
        ('fraction', pa.decimal128(38, 2)),
        ('dictionary', pa.dictionary(pa.int32(), pa.string())),
        ('nested', pa.list_(pa.struct([('value', pa.int64())]))),
        ('__index_level_0__', pa.int64()),
        ('__index_level_12__suffix', pa.int64()),
        ('__index_level_text__', pa.string()),
    ])
    expected = {
        'small_integer': Dtype.int64,
        'wide_integer': Dtype.int64,
        'fraction': Dtype.float32,
        'dictionary': Dtype.string,
        'nested': Dtype.unsupported,
        '__index_level_text__': Dtype.string,
    }
    existing = BaseColumnConfig('existing', dtype=Dtype.bool)
    metadata = TableMetadataConfig('table', 'table', 'source', cols=[existing])
    result = add_metadata_cols_from_parquet_dataset(
        metadata, SimpleNamespace(schema=schema))
    assert result is metadata
    assert result.cols[0] is existing
    assert [(column.name, column.dtype)
            for column in result.cols[1:]] == list(expected.items())
    assert column_dtypes(schema) == expected


def test_parquet_metadata_keeps_duplicate_field_occurrences():
    schema = pa.schema([('same', pa.int64()), ('same', pa.string())])
    metadata = TableMetadataConfig('table', 'table', 'source')
    add_metadata_cols_from_parquet_dataset(metadata,
                                           SimpleNamespace(schema=schema))
    assert [(column.name, column.dtype)
            for column in metadata.cols] == [('same', Dtype.int64),
                                             ('same', Dtype.string)]
    assert column_dtypes(schema) == {'same': Dtype.string}
