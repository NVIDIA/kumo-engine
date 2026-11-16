# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Generator

import pytest

from io_utils import fsutil
from io_utils.fsutil.uri import GCSURI
from kumo.testing.gcs import withGCSCredentials

from .common import random_name


@pytest.fixture
def temp_prefix(k: int = 16) -> Generator[GCSURI, None, None]:
    uri = GCSURI(
        bucket='kumo-test-bucket',
        path=random_name(k),
    )

    fsutil.makedirs(uri)
    yield uri
    fsutil.rm(uri, ignore_errors=True)


@withGCSCredentials
def test_isdir_bucket_root(temp_prefix):
    bucket_path = f"gs://{temp_prefix.bucket}"
    assert fsutil.isdir(bucket_path)
    assert fsutil.isdir(f"{bucket_path}/")


@withGCSCredentials
def test_zip_directory_gcs(temp_prefix):
    """zip_directory downloads GCS files, creates a zip, and uploads it."""
    test_dir = temp_prefix.as_gs()
    fsutil.write_text(f"{test_dir}/part-0.parquet", "data0")
    fsutil.write_text(f"{test_dir}/part-1.parquet", "data1")

    result = fsutil.zip_directory(test_dir)

    assert result == test_dir.rstrip('/') + '.zip'
    assert fsutil.exists(result)
    # Clean up the zip
    fsutil.rm(result, ignore_errors=True)


@withGCSCredentials
def test_zip_directory_gcs_idempotent(temp_prefix):
    """zip_directory skips if zip already exists."""
    test_dir = temp_prefix.as_gs()
    fsutil.write_text(f"{test_dir}/part-0.parquet", "data0")
    # Pre-create the zip
    zip_path = test_dir.rstrip('/') + '.zip'
    fsutil.write_text(zip_path, "existing_zip")

    result = fsutil.zip_directory(test_dir)

    assert result == zip_path
    # The existing zip should not be overwritten
    assert fsutil.read_text(zip_path) == "existing_zip"
    fsutil.rm(zip_path, ignore_errors=True)


@withGCSCredentials
def test_zip_directory_gcs_empty_raises(temp_prefix):
    """zip_directory raises ValueError when no files exist under prefix."""
    # temp_prefix is an empty directory
    empty_dir = f"{temp_prefix.as_gs()}/empty_subdir"

    with pytest.raises(ValueError, match="No files found"):
        fsutil.zip_directory(empty_dir)


@withGCSCredentials
def test_gcs_generate_download_url(temp_prefix):
    """GCSPresignedHelper.generate_download_url produces a signed HTTPS URL."""
    test_dir = temp_prefix.as_gs()
    file_path = f"{test_dir}/presign_test.txt"
    fsutil.write_text(file_path, "hello")

    helper = fsutil.get_presigned_helper(file_path)
    assert helper is not None
    url = helper.generate_download_url(file_path)
    assert url.startswith('https://')
    assert temp_prefix.bucket in url
