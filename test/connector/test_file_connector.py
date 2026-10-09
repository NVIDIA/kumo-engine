# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

from kumo.connector.parquet_connector import ParquetConnector
from kumo.graph.table import Table
from kumo.testing.s3 import withS3Credentials

S3_PATH = 's3://kumo-unit-test/file_connector/'


@withS3Credentials
def test_get_data_version_s3_multiple_files():
    file_connector = ParquetConnector('parquet_connector', S3_PATH, 'parquet')
    version = file_connector.get_data_version('s3_table')

    # Directories fingerprint every source part's metadata.
    assert re.fullmatch(r'[0-9a-f]{64}', version)
    assert file_connector.get_data_version(
        's3_table', skip_cache=True, save_to_cache=False) == version

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
