# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT

import pytest

from kumoapi.s3 import (
    normalize_s3_bucket,
    normalize_s3_destination_relative_path,
    normalize_s3_destination_uri,
    normalize_s3_root_prefix,
    normalize_s3_source_relative_path,
    normalize_s3_source_uri,
)


@pytest.mark.parametrize(
    'bucket',
    [
        'ABCD',
        'ab',
        'a' * 64,
        'bucket..name',
        '192.168.5.4',
        'bucket%2ename',
        'xn--bucket',
        'sthree-bucket',
        'amzn-s3-demo-bucket',
        'bucket-s3alias',
        'bucket--ol-s3',
        'bucket.mrap',
        'bucket--x-s3',
        'bucket--table-s3',
        ' bucket',
    ],
)
def test_s3_bucket_rejects_noncanonical_or_reserved_names(bucket):
    with pytest.raises(ValueError):
        normalize_s3_bucket(bucket)


def test_s3_uri_canonicalization_is_idempotent():
    raw = 'S3://company-data/deep/table%20name.parquet/'
    canonical = 's3://company-data/deep/table%20name.parquet/'
    assert normalize_s3_source_uri(raw) == canonical
    assert normalize_s3_source_uri(canonical) == canonical

    destination = 's3://company-data/deep/output/'
    assert normalize_s3_destination_uri(destination) == destination
    assert normalize_s3_destination_uri(
        normalize_s3_destination_uri(destination)) == destination

    escaped_delimiters = (
        's3://company-data/deep/question%3Fhash%23percent%25.parquet')
    assert normalize_s3_source_uri(escaped_delimiters) == escaped_delimiters
    assert normalize_s3_source_uri(
        normalize_s3_source_uri(escaped_delimiters)) == escaped_delimiters


@pytest.mark.parametrize(
    'value',
    [
        's3://company-data',
        's3a://company-data/table.csv',
        'https://company-data/table.csv',
        's3://user@company-data/table.csv',
        's3://company-data:443/table.csv',
        's3://company-data//table.csv',
        's3://company-data///table.csv',
        's3://company-data/a//table.csv',
        's3://company-data/a/./table.csv',
        's3://company-data/a/../table.csv',
        's3://company-data/a/%2f/table.csv',
        's3://company-data/a/%5C/table.csv',
        's3://company-data/a/%00/table.csv',
        's3://company-data/a/%252f/table.csv',
        's3://company-data/a/%252e%252e/table.csv',
        's3://company-data/a/%2500/table.csv',
        's3://company-data/table.csv?X-Amz-Signature=secret',
        's3://company-data/table.csv#fragment',
    ],
)
def test_s3_uri_rejects_ambiguous_or_unsafe_coordinates(value):
    with pytest.raises(ValueError):
        normalize_s3_source_uri(value)


@pytest.mark.parametrize(
    'value',
    [
        '',
        '/absolute/table.csv',
        's3://company-data/table.csv',
        'one//table.csv',
        'one/./table.csv',
        'one/../table.csv',
        'one\\table.csv',
        'one/%2f/table.csv',
        'one/%5c/table.csv',
        'one/%00/table.csv',
        'one/%252f/table.csv',
        'one/\x00/table.csv',
        'one/\x1f/table.csv',
        'one/\x7f/table.csv',
        'table.csv?version=1',
        'table.csv#fragment',
    ],
)
def test_s3_relative_path_rejects_uri_absolute_or_unsafe_values(value):
    with pytest.raises(ValueError):
        normalize_s3_source_relative_path(value)


def test_s3_relative_source_and_destination_preserve_directory_intent():
    assert normalize_s3_source_relative_path('partitioned/table/') == (
        'partitioned/table/')
    assert normalize_s3_source_relative_path(
        normalize_s3_source_relative_path('partitioned/table/')) == (
            'partitioned/table/')
    assert normalize_s3_destination_relative_path('partitioned/output/') == (
        'partitioned/output/')
    assert normalize_s3_destination_relative_path(
        normalize_s3_destination_relative_path('partitioned/output/')) == (
            'partitioned/output/')


def test_s3_source_uri_preserves_directory_intent_idempotently():
    directory = 's3://company-data/deep/dataset.parquet/'
    assert normalize_s3_source_uri(directory) == directory
    assert normalize_s3_source_uri(
        normalize_s3_source_uri(directory)) == (directory)


@pytest.mark.parametrize(
    'prefix',
    [
        '/leading',
        'trailing/',
        'a//b',
        'a/./b',
        'a/../b',
        'a\\b',
        's3://company-data/root',
        'a/query?value',
        'a/fragment#value',
        'a/%2f/b',
        'a/%2e%2e/b',
        'a/%252f/b',
        ' a/b',
        'a/b ',
        'a/\x00/b',
    ],
)
def test_s3_root_prefix_rejects_noncanonical_or_ambiguous_values(prefix):
    with pytest.raises(ValueError):
        normalize_s3_root_prefix(prefix)


def test_empty_s3_root_prefix_is_explicit_and_stable():
    assert normalize_s3_root_prefix('') == ''
    prefix = 'tenant/deep/data'
    assert normalize_s3_root_prefix(prefix) == prefix
    assert normalize_s3_root_prefix(normalize_s3_root_prefix(prefix)) == prefix
