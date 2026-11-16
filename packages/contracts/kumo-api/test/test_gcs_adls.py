# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from kumoapi.adls import normalize_adls_source_uri
from kumoapi.gcs import normalize_gcs_source_uri

_UNSAFE_PATH_SUFFIXES = (
    '',
    '//table.csv',
    '/./table.csv',
    '/../table.csv',
    '/a//table.csv',
    '/a%2Ftable.csv',
    '/a%5Ctable.csv',
    '/%252e%252e/table.csv',
    '/table.csv?version=1',
    '/table.csv#fragment',
)


@pytest.mark.parametrize(
    'value,expected',
    [
        ('GS://company-data/deep/orders%20archive.parquet',
         'gs://company-data/deep/orders%20archive.parquet'),
        ('gs://company-data/deep/dataset.parquet/',
         'gs://company-data/deep/dataset.parquet/'),
        ('gs://company-data/year=2026/month=07/orders.csv',
         'gs://company-data/year=2026/month=07/orders.csv'),
    ],
)
def test_gcs_source_uri_canonicalization(value, expected):
    assert normalize_gcs_source_uri(value) == expected


@pytest.mark.parametrize(
    'value,expected',
    [
        (
            'ABFSS://curated@companylake.dfs.core.windows.net/'
            'deep/orders%20archive.parquet',
            'abfss://curated@companylake.dfs.core.windows.net/'
            'deep/orders%20archive.parquet',
        ),
        (
            'abfss://curated@companylake.dfs.core.usgovcloudapi.net/'
            'deep/dataset.parquet/',
            'abfss://curated@companylake.dfs.core.usgovcloudapi.net/'
            'deep/dataset.parquet/',
        ),
        (
            'abfss://curated@companylake.dfs.core.chinacloudapi.cn/'
            'year=2026/orders.csv',
            'abfss://curated@companylake.dfs.core.chinacloudapi.cn/'
            'year=2026/orders.csv',
        ),
    ],
)
def test_adls_source_uri_canonicalization_and_sovereign_hosts(value, expected):
    assert normalize_adls_source_uri(value) == expected


@pytest.mark.parametrize(('normalizer', 'base_uri', 'provider_values'), [
    (
        normalize_gcs_source_uri,
        'gs://company-data',
        (
            '',
            's3://company-data/table.csv',
            'gs://user@company-data/table.csv',
            'gs://192.168.0.1/table.csv',
        ),
    ),
    (
        normalize_adls_source_uri,
        'abfss://curated@companylake.dfs.core.windows.net',
        (
            '',
            'abfs://curated@companylake.dfs.core.windows.net/table.csv',
            'abfss://companylake.dfs.core.windows.net/table.csv',
            'abfss://curated@companylake.blob.core.windows.net/'
            'table.csv',
            'abfss://bad--name@companylake.dfs.core.windows.net/table.csv',
            'abfss://curated@CompanyLake.dfs.core.windows.net/table.csv',
        ),
    ),
])
def test_object_store_source_uri_rejects_unsafe_coordinates(
        normalizer, base_uri, provider_values):
    values = (*provider_values, *(f'{base_uri}{suffix}'
                                  for suffix in _UNSAFE_PATH_SUFFIXES))
    for value in values:
        with pytest.raises((ValueError, UnicodeError)):
            normalizer(value)
