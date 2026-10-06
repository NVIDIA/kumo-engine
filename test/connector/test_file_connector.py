# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
import time

import pandas as pd
import pytest

from io_utils import fsutil
from kumo.connector.connector import SourceTableNotFoundError
from kumo.connector.csv_connector import CSVConnector
from kumo.connector.file_connector import DEFAULT_DATA_VERSION
from kumo.connector.parquet_connector import ParquetConnector
from kumo.graph.table import Table
from kumo.testing.s3 import withS3Credentials

LOCAL_PATH = osp.join(osp.dirname(osp.realpath(__file__)), 'test')
S3_PATH = 's3://kumo-unit-test/file_connector/'


def create_file(root_dir: str, file_name: str, df: pd.DataFrame):
    fsutil.makedirs(root_dir, exist_ok=True)
    df.to_parquet(osp.join(root_dir, file_name))


def teardown_function():
    if fsutil.exists(LOCAL_PATH):
        fsutil.rm(LOCAL_PATH)


def get_default_df() -> pd.DataFrame:
    return pd.DataFrame({'colA': [1, 2], 'colB': [3, 4]})


def test_get_data_version_local():
    file_connector = ParquetConnector('parquet_connector', LOCAL_PATH,
                                      'parquet')
    current_time = time.time()
    create_file(LOCAL_PATH, 'table.parquet', get_default_df())

    version = file_connector.get_data_version('table')

    # Ensuring that the version id was created within 10 seconds of the noted
    # time. Using this method since simply relying on current_time < version
    # could be flaky due to IO delays
    assert int(current_time) - 10 <= int(version)
    assert int(current_time) + 10 >= int(version)

    # Create a table object with pre-assigned data version. Verify that the
    # connector retrieves the pre-assigned data version for this table.
    table = Table(table_name='table', data_source='parquet',
                  source_name='table', data_version='test_version')
    table.connector = file_connector
    assert file_connector.get_data_version('table') == 'test_version'


@pytest.mark.parametrize(('connector_type', 'suffix'), [
    (CSVConnector, 'csv'),
    (ParquetConnector, 'parquet'),
])
def test_missing_direct_file_has_typed_error_and_can_be_ignored(
        tmp_path, connector_type, suffix):
    connector = connector_type('file_connector', str(tmp_path), suffix)
    table_name = f'missing.{suffix}'

    with pytest.raises(SourceTableNotFoundError, match='does not exist'):
        connector.get_data_version(table_name)

    assert connector.get_table_metadata([table_name], ignore_errors=True) == []


def test_nested_dataset_can_use_default_data_version(tmp_path, monkeypatch):
    connector = ParquetConnector('file_connector', str(tmp_path), 'parquet')
    table_path = tmp_path / 'table' / 'partition'
    table_path.mkdir(parents=True)
    (table_path / 'part.parquet').touch()
    monkeypatch.setattr(fsutil, 'get_last_modified_time', lambda _path: -1.0)

    assert connector.get_data_version('table') == DEFAULT_DATA_VERSION


def test_get_data_version_local_multiple_files():
    file_connector = ParquetConnector('parquet_connector', LOCAL_PATH,
                                      'parquet')

    table_path = osp.join(LOCAL_PATH, 'table')
    fsutil.makedirs(table_path)
    create_file(table_path, 'file1.parquet', get_default_df())
    current_time = time.time()
    time.sleep(1)
    create_file(table_path, 'file2.parquet', get_default_df())

    version = file_connector.get_data_version('table')
    assert int(version) > int(current_time)


@withS3Credentials
def test_get_data_version_s3_multiple_files():
    file_connector = ParquetConnector('parquet_connector', S3_PATH, 'parquet')
    version = file_connector.get_data_version('s3_table')

    # Epoch timestamp corresponding to original creation time of the file in s3
    assert version == '1663191047'

    # Create a table object with pre-assigned data version. Verify that the
    # connector retrieves the pre-assigned data version for this table.
    table = Table(table_name='s3_table', data_source='parquet',
                  source_name='s3_table', data_version='test_version')
    table.connector = file_connector
    assert file_connector.get_data_version('s3_table') == 'test_version'


@withS3Credentials
def test_get_data_version_s3_single_file():
    file_connector = ParquetConnector('parquet_connector', S3_PATH, 'parquet')

    version = file_connector.get_data_version('table')

    # Epoch timestamp corresponding to original creation time of the file in s3
    assert version == '1663190838'


def test_hosted_export_rejects_declared_column_missing_in_later_shard(tmp_path):
    from kumo_core.sources.errors import PermanentSourceError
    from kumoapi.typing import Dtype

    create_file(str(tmp_path / "events"), "part-0.parquet",
                pd.DataFrame({
                    "id": [1],
                    "value": [2.0]
                }))
    create_file(str(tmp_path / "events"), "part-1.parquet",
                pd.DataFrame({"id": [2]}))
    connector = ParquetConnector("files", str(tmp_path), "parquet")
    with pytest.raises(PermanentSourceError,
                       match="missing declared columns: value"):
        with connector._local_source(
                "events", columns={
                    "id": Dtype.int64,
                    "value": Dtype.float64
                }):
            pytest.fail("invalid source must fail before export")
