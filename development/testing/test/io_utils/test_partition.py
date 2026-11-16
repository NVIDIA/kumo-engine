# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test partition info for different storage systems."""

import random
import string
from copy import deepcopy

import pytest

from io_utils import fsutil
from io_utils.fsutil.uri import S3URI, UCURI
from io_utils.partition import get_pred_table_partition_info
from kumo.testing import withDatabricksCredentials
from kumo.testing.s3 import withMockedS3Filesystem


def random_numbers(length=8):
    return ''.join(random.choices(string.digits, k=length))


@withMockedS3Filesystem
def test_s3_partition_info(s3):
    """Test get_pred_table_partition_info with various scenarios.

    Test cases:
    1. Single partition: Verifies correct handling of a directory with one
       partition file.
    2. Multiple partitions: Verifies handling of a directory with two partition
       files.
    3. Empty directory: Tests behavior with an empty directory.
    4. File path error: Verifies that passing a file path instead of a
       directory raises a ValueError.
    5. Path reconstruction: Tests get_prefix_path method for:
       - Valid partition numbers
       - Invalid partition numbers (negative and out of range)
       - Empty directory case
    """
    s3.create_bucket(Bucket='test-bucket')

    # Generate a random job ID
    job_id = random_numbers(8)

    # Test single partition
    uri = S3URI(
        bucket='test-bucket',
        path=f'test/prod/jobs/bp-job-{job_id}/pred_table/test_model/pred_table'
    )
    s3.put_object(
        Bucket='test-bucket',
        Key=f'test/prod/jobs/bp-job-{job_id}/pred_table/test_model/pred_table/'
        'part-00000-123456789-a97b-4e51-8463-d715e38f583b-c000.snappy.parquet',
        Body=b'')

    result = get_pred_table_partition_info(uri.as_s3())
    assert result.uri_type == S3URI
    assert result.num_partitions == 1
    assert result.path_prefix == (
        f"s3://test-bucket/test/prod/jobs/bp-job-{job_id}/"
        "pred_table/test_model/pred_table/part-")

    # Test multiple partitions
    s3.put_object(
        Bucket='test-bucket',
        Key=f'test/prod/jobs/bp-job-{job_id}/pred_table/test_model/pred_table/'
        'part-00001-123456789-a97b-4e51-8463-d715e38f583b-c000.snappy.parquet',
        Body=b'')
    # including a multi-file partition
    s3.put_object(
        Bucket='test-bucket',
        Key=f'test/prod/jobs/bp-job-{job_id}/pred_table/test_model/pred_table/'
        'part-00001-9876544321-a97b-4e51-8463-d715e38f583b-c000.snappy.parquet',
        Body=b'')

    result = get_pred_table_partition_info(uri.as_s3())
    assert result.uri_type == S3URI
    assert result.num_partitions == 2
    assert result.path_prefix == (
        f"s3://test-bucket/test/prod/jobs/bp-job-{job_id}/"
        "pred_table/test_model/pred_table/part-")

    # Test empty directory
    empty_uri = S3URI(
        bucket='test-bucket',
        path=(f'test/prod/jobs/bp-job-{job_id}/pred_table/'
              'empty_model/pred_table'))
    s3.put_object(
        Bucket='test-bucket',
        Key=(f'test/prod/jobs/bp-job-{job_id}/pred_table/'
             'empty_model/pred_table/'), Body=b'')
    result = get_pred_table_partition_info(empty_uri.as_s3())
    assert result.uri_type == S3URI
    assert result.num_partitions == 0
    assert result.path_prefix == ""

    # Test file path error
    file_uri = S3URI(
        bucket='test-bucket',
        path=(f'test/prod/jobs/bp-job-{job_id}/pred_table/test_model/'
              'pred_table/part-00000-123456789-a97b-4e51-8463-d715e38f583b-'
              'c000.snappy.parquet'))
    with pytest.raises(ValueError, match="Expected a directory, got"):
        get_pred_table_partition_info(file_uri.as_s3())

    # Test path reconstruction
    result = get_pred_table_partition_info(uri.as_s3())
    expected_path = (f"s3://test-bucket/test/prod/jobs/bp-job-{job_id}/"
                     "pred_table/test_model/pred_table/part-00000*")
    assert result.get_prefix_path(0) == expected_path

    expected_path = (f"s3://test-bucket/test/prod/jobs/bp-job-{job_id}/"
                     "pred_table/test_model/pred_table/part-00001*")
    assert result.get_prefix_path(1) == expected_path

    # Test error cases
    with pytest.raises(ValueError, match="Partition number out of range"):
        result.get_prefix_path(-1)
    with pytest.raises(ValueError, match="Partition number out of range"):
        result.get_prefix_path(2)

    # Test error case for non-consecutive partitions
    s3.put_object(
        Bucket='test-bucket',
        Key=f'test/prod/jobs/bp-job-{job_id}/pred_table/test_model/pred_table/'
        'part-00003-123456789-a97b-4e51-8463-d715e38f583b-c000.snappy.parquet',
        Body=b'')
    with pytest.raises(ValueError,
                       match="Partitions are not consecutively numbered at"):
        get_pred_table_partition_info(uri.as_s3())

    # Test empty directory case
    empty_result = get_pred_table_partition_info(empty_uri.as_s3())
    with pytest.raises(ValueError, match="Partition number out of range"):
        empty_result.get_prefix_path(0)


@withDatabricksCredentials
def test_ucv_partition_info(temp_ucv):
    """Test get_pred_table_partition_info for UC Volume files.

    Test cases:
    1. Single partition: Verifies correct handling of a directory with one
       partition file.
    2. Multiple partitions: Verifies handling of a directory with two partition
       files.
    3. Empty directory: Tests behavior with an empty directory.
    4. File path error: Verifies that passing a file path instead of a
       directory raises a ValueError.
    5. Path reconstruction: Tests get_prefix_path method for:
       - Valid partition numbers (0 and 1)
       - Invalid partition numbers (negative and out of range)
       - Empty directory case
    """
    # Generate a random job ID
    job_id = random_numbers(8)

    # Test single partition
    uri = UCURI(
        catalog=temp_ucv.catalog, uc_schema=temp_ucv.uc_schema,
        volume=temp_ucv.volume,
        path=f'test/prod/jobs/bp-job-{job_id}/pred_table/test_model/pred_table'
    )

    # Create the directory structure
    fsutil.makedirs(uri)

    # Create a partition file
    partition_file = deepcopy(uri)
    partition_file.path += (
        '/part-00000-tid-8398707246426104439-2264b669-5641-4216-a42a-'
        '096603307e83-9738-1.c000.snappy.parquet')
    with fsutil.get_binary_handle(partition_file, 'w') as f:
        f.write(b'')

    result = get_pred_table_partition_info(uri.as_url())
    assert result.uri_type == UCURI
    assert result.num_partitions == 1
    assert result.path_prefix == (
        f'dbfs:/Volumes/{temp_ucv.catalog}/{temp_ucv.uc_schema}/'
        f'{temp_ucv.volume}/test/prod/jobs/bp-job-{job_id}/'
        'pred_table/test_model/pred_table/part-')

    # Test multiple partitions
    partition_file2 = deepcopy(uri)
    partition_file2.path += (
        '/part-00001-tid-8398707246426104439-2264b669-5641-4216-a42a-'
        '096603307e83-9745-1.c000.snappy.parquet')
    with fsutil.get_binary_handle(partition_file2, 'w') as f:
        f.write(b'')

    result = get_pred_table_partition_info(uri.as_url())
    assert result.uri_type == UCURI
    assert result.num_partitions == 2
    assert result.path_prefix == (
        f'dbfs:/Volumes/{temp_ucv.catalog}/{temp_ucv.uc_schema}/'
        f'{temp_ucv.volume}/test/prod/jobs/bp-job-{job_id}/'
        'pred_table/test_model/pred_table/part-')

    # Test empty directory
    empty_uri = UCURI(
        catalog=temp_ucv.catalog, uc_schema=temp_ucv.uc_schema,
        volume=temp_ucv.volume, path=(f'test/prod/jobs/bp-job-{job_id}/'
                                      'pred_table/empty_model/pred_table'))
    fsutil.makedirs(empty_uri)

    result = get_pred_table_partition_info(empty_uri.as_url())
    assert result.uri_type == UCURI
    assert result.num_partitions == 0
    assert result.path_prefix == ""

    # Test file path error
    file_uri = UCURI(
        catalog=temp_ucv.catalog, uc_schema=temp_ucv.uc_schema,
        volume=temp_ucv.volume, path=(
            f'test/prod/jobs/bp-job-{job_id}/pred_table/test_model/pred_table/'
            'part-00001-tid-8398707246426104439-2264b669-5641-4216-a42a-'
            '096603307e83-9738-1.c000.snappy.parquet'))
    with pytest.raises(ValueError, match="Expected a directory, got"):
        get_pred_table_partition_info(file_uri.as_url())

    # Test path reconstruction
    result = get_pred_table_partition_info(uri.as_url())
    expected_path = (f'dbfs:/Volumes/{temp_ucv.catalog}/{temp_ucv.uc_schema}/'
                     f'{temp_ucv.volume}/test/prod/jobs/bp-job-{job_id}/'
                     'pred_table/test_model/pred_table/part-00000*')
    assert result.get_prefix_path(0) == expected_path

    expected_path = (f'dbfs:/Volumes/{temp_ucv.catalog}/{temp_ucv.uc_schema}/'
                     f'{temp_ucv.volume}/test/prod/jobs/bp-job-{job_id}/'
                     'pred_table/test_model/pred_table/part-00001*')
    assert result.get_prefix_path(1) == expected_path

    # Test error cases
    with pytest.raises(ValueError, match="Partition number out of range"):
        result.get_prefix_path(-1)
    with pytest.raises(ValueError, match="Partition number out of range"):
        result.get_prefix_path(2)

    # Test error case for non-consecutive partitions
    partition_file3 = deepcopy(uri)
    partition_file3.path += (
        '/part-00003-tid-8398707246426104439-2264b669-5641-4216-a42a-'
        '096603307e83-9745-1.c000.snappy.parquet')
    with fsutil.get_binary_handle(partition_file3, 'w') as f:
        f.write(b'')

    with pytest.raises(ValueError,
                       match="Partitions are not consecutively numbered at"):
        get_pred_table_partition_info(uri.as_url())

    # Test empty directory case
    empty_result = get_pred_table_partition_info(empty_uri.as_url())
    with pytest.raises(ValueError, match="Partition number out of range"):
        empty_result.get_prefix_path(0)
