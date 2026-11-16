# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor, as_completed
from os import path as osp
from typing import Generator, List

import pytest

from io_utils import fsutil
from io_utils.fsutil import s3_filesystem
from io_utils.fsutil.common import fsutil_ctx
from io_utils.fsutil.uri import S3URI
from kumo.testing.s3 import withS3Credentials

from .common import no_retry, random_name


@pytest.fixture
def temp_prefix(k: int = 16) -> Generator[S3URI, None, None]:
    uri = S3URI(
        bucket='kumo-test-dataplane-bucket',
        path=random_name(k),
    )

    fsutil.makedirs(uri)
    yield uri
    fsutil.rm(uri, ignore_errors=True)


@withS3Credentials
def test_s3_rmdir(temp_prefix):
    bucket = "kumo-test-dataplane-bucket"
    dir_path = temp_prefix.as_s3()

    import io_utils.fsutil.s3_filesystem as util

    def touch_all(paths):
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(fsutil.touch, p) for p in paths]
            for f in as_completed(futures):
                f.result()

    touch_all([f"{dir_path}/{i}/{j}" for i in range(5) for j in range(5)])

    assert len(util.listdir_recursive(dir_path)) == 25
    util.rmdir(dir_path)
    assert len(util.listdir_recursive(dir_path)) == 0

    # Invalid bucket
    with pytest.raises(Exception):
        util.rmdir("s3://invalid-bucket/path")

    with pytest.raises(Exception):
        util._delete_batch("invalid-bucket", ["path"])

    # Should succeed if path is already deleted
    util._delete_batch(bucket, ["non-existant-path"])
    util.rmdir(dir_path)


@withS3Credentials
def test_s3_copydir(temp_prefix):
    bucket = temp_prefix.bucket
    prefix = temp_prefix.as_s3()

    src_relative = "tests/src_dir"
    src_key = osp.join(temp_prefix.path, src_relative)
    src_path = f"{prefix}/{src_relative}"
    dst_path = f"{prefix}/tests/dst_dir"
    dst_path2 = f"{prefix}/tests/dst_dir2"

    import io_utils.fsutil.s3_filesystem as util

    def touch_all(paths):
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(fsutil.touch, p) for p in paths]
            for f in as_completed(futures):
                f.result()

    touch_all([f"{src_path}/{i}/{j}" for i in range(3) for j in range(3)])

    assert len(util.listdir_recursive(src_path)) == 9
    util.copydir(src_path, dst_path2)
    assert len(util.listdir_recursive(dst_path2)) == 9

    # Set multiprocessing_threshold = 5 to force many processes to be launched
    util.s3_ctx._copy_multiprocessing_threshold = 5

    util.copydir(src_path, dst_path)
    assert len(util.listdir_recursive(dst_path)) == 9

    # Should overwrite existing file
    util.copydir(src_path, dst_path)
    assert len(util.listdir_recursive(dst_path)) == 9

    # Invalid src bucket
    with no_retry():
        with pytest.raises(Exception, match="Access Denied"):
            util.copydir("s3://invalid-bucket/path", dst_path)

    with no_retry():
        with pytest.raises(Exception, match="Forbidden"):
            util._copy_batch("invalid-bucket", "path", ["key"], bucket, "path",
                             util.s3_ctx._retry_delay_sec,
                             fsutil_ctx.thread_pool, util.s3_ctx._s3_client)

    # Invalid dst bucket
    with no_retry():
        with pytest.raises(Exception, match="Access Denied"):
            util.copydir(src_path, "s3://invalid-bucket")

    with pytest.raises(Exception, match="Access Denied"):
        util._copy_batch(
            bucket,
            f"{src_key}/0/",
            [f"{src_key}/0/0"],
            "invalid-bucket",
            "path",
            util.s3_ctx._retry_delay_sec,
            fsutil_ctx.thread_pool,
            util.s3_ctx._s3_client,
        )

    # Empty src path
    with pytest.raises(ValueError, match="source directory is empty"):
        util.copydir(f"s3://{bucket}/empty_path/", dst_path)

    with pytest.raises(Exception, match="Not Found"):
        util._copy_batch(
            bucket,
            "empty_dir/",
            ["empty_dir/0"],
            bucket,
            "path",
            util.s3_ctx._retry_delay_sec,
            fsutil_ctx.thread_pool,
            util.s3_ctx._s3_client,
        )


@withS3Credentials
def test_filesize():
    assert fsutil.filesize("s3://kumo-test-dataplane-bucket") is None
    assert fsutil.filesize("s3://kumo-test-dataplane-bucket/") is None
    assert fsutil.filesize(
        "s3://kumo-test-dataplane-bucket/test_fixtures") is None
    assert fsutil.filesize(
        "s3://kumo-test-dataplane-bucket/test_fixtures/") is None
    assert fsutil.filesize(
        "s3://kumo-test-dataplane-bucket/test_fixtures/hnm_10k_churn.json"
    ) == 7037
    assert fsutil.filesize(
        "s3://kumo-test-dataplane-bucket/test_fixtures/hnm_10k_churn.json/"
    ) is None


@withS3Credentials
def test_s3_listdir(temp_prefix):
    prefix = temp_prefix.as_s3()

    import io_utils.fsutil.s3_filesystem as util

    for i in range(5):
        for j in range(5):
            fsutil.touch(f"{prefix}/{i}/{j}")

    def listdir(path) -> List[str]:
        return util.listdir_recursive(path)

    # Base case
    assert len(listdir(f"{prefix}/")) == 25
    assert len(listdir(f"{prefix}")) == 25

    # Empty Dir
    assert len(listdir(f"{prefix}/empty_path")) == 0

    # Invalid bucket
    with no_retry():
        with pytest.raises(Exception, match="Access Denied"):
            listdir("s3://invalid-bucket/path")

    # Path with //
    fsutil.touch(f"{prefix}///slashpath//file")
    fsutil.touch(f"{prefix}///slashpath/file")
    assert len(listdir(f"{prefix}///slashpath")) == 2
    assert len(listdir(f"{prefix}///slashpath/")) == 2
    assert len(listdir(f"{prefix}///slashpath//")) == 1
    assert len(listdir(f"{prefix}///slashpath//file")) == 0
    assert len(listdir(f"{prefix}///slashpath//file/")) == 0


@withS3Credentials
def test_listdir(temp_prefix):
    prefix = temp_prefix.as_s3()

    def touch_all(paths):
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(fsutil.touch, p) for p in paths]
            for f in as_completed(futures):
                f.result()

    # Ensure all previous state is deleted.
    fsutil.rm(prefix)

    touch_all([f"{prefix}/{i}.txt" for i in range(10)] +
              [f"{prefix}/subdir/{i}.txt" for i in range(10)])

    assert len(fsutil.listdir(prefix)) == 11
    assert len(fsutil.listdir(prefix, max_items=5)) == 5
    assert len(fsutil.listdir(prefix, max_items=11)) == 11

    # Test max_items being out of range.
    with pytest.raises(ValueError):
        fsutil.listdir(prefix, max_items=-1)
    with pytest.raises(ValueError):
        fsutil.listdir(prefix, max_items=0)
    with pytest.raises(ValueError):
        fsutil.listdir(prefix, max_items=1005)

    fsutil.rm(prefix)

    touch_all([f"{prefix}/{i}.txt" for i in range(1005)])

    # Test that paginator returns all files.
    assert len(fsutil.listdir(prefix)) == 1005
    assert len(fsutil.listdir(prefix, max_items=1000)) == 1000
    with pytest.raises(ValueError):
        fsutil.listdir(prefix, max_items=1001)

    fsutil.rm(prefix)


@withS3Credentials
def test_isdir_bucket_root():
    bucket_path = "s3://kumo-test-dataplane-bucket"
    assert fsutil.isdir(bucket_path)
    assert fsutil.isdir(f"{bucket_path}/")


@withS3Credentials
def test_s3_generate_download_url():
    """S3PresignedHelper.generate_download_url produces a presigned HTTPS URL."""
    helper = fsutil.get_presigned_helper(
        's3://kumo-public-datasets/test_fsutil/existing_file')
    assert helper is not None
    url = helper.generate_download_url(
        's3://kumo-public-datasets/test_fsutil/existing_file')
    assert url.startswith('https://')
    assert 'kumo-public-datasets' in url


@withS3Credentials
def test_get_bucket_aws_region():
    assert s3_filesystem.get_bucket_aws_region(
        'kumo-public-datasets') == 'us-west-2'

    assert s3_filesystem.get_bucket_aws_region(
        'kumo-public-datasets-ue1') == 'us-east-1'

    # no permissions
    with pytest.raises(ValueError):
        s3_filesystem.get_bucket_aws_region(
            'kumo-inaccessible-bucket-for-automated-tests')

    # nonexistent
    with pytest.raises(ValueError):
        s3_filesystem.get_bucket_aws_region('kumo-fake-bucket')
