# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor, as_completed
from os import path as osp
from typing import Generator

import pytest

from io_utils import fsutil
from io_utils.fsutil.uri import ADLSURI, ParseURI
from kumo.testing.adls import adls_test_base_uri, withADLSCredentials

from .common import random_name


@pytest.fixture
def temp_prefix(k: int = 16) -> Generator[ADLSURI, None, None]:
    base_uri = ParseURI(adls_test_base_uri())
    assert isinstance(base_uri, ADLSURI)
    uri = ADLSURI(storage_account=base_uri.storage_account,
                  container=base_uri.container, path=random_name(k))

    fsutil.makedirs(uri)
    yield uri
    fsutil.rm(uri, ignore_errors=True)


@withADLSCredentials
def test_contains_subdir_container_root():
    # Regression test: contains_subdir on a bare container URI (empty path)
    # previously crashed with ValueError because get_directory_client('')
    # is rejected by the Azure SDK. The fix uses fs_client.get_paths() directly.
    # The test filesystem has known fixture directories, so this returns True.
    assert fsutil.contains_subdir(adls_test_base_uri())


@withADLSCredentials
def test_filesize():
    base_uri = adls_test_base_uri()

    assert fsutil.filesize(base_uri) is None
    assert fsutil.filesize(f"{base_uri}/") is None
    assert fsutil.filesize(f"{base_uri}/test_fixtures") is None
    assert fsutil.filesize(f"{base_uri}/test_fixtures/") is None
    assert fsutil.filesize(
        f"{base_uri}/test_fixtures/hnm_10k_churn.json") == 7037
    assert fsutil.filesize(
        f"{base_uri}/test_fixtures/hnm_10k_churn.json/") is None


@withADLSCredentials
def test_listdir(temp_prefix):
    prefix = temp_prefix.as_adls()

    def touch_all(paths):
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(fsutil.touch, p) for p in paths]
            for f in as_completed(futures):
                f.result()

    # Ensure all previous state is deleted.
    fsutil.rm(prefix)

    touch_all([f"{prefix}/{i}.txt" for i in range(10)] +
              [f"{prefix}/subdir/{i}.txt" for i in range(10)])

    entries = fsutil.listdir(prefix)
    assert len(entries) == 11
    # Verify directories and files both appear as immediate children — subdir
    # itself is one entry, its contents must not leak into the listing.
    assert sum(1 for e in entries if e.endswith('subdir')) == 1
    assert sum(1 for e in entries if e.endswith('.txt')) == 10

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


@withADLSCredentials
def test_adls_generate_download_url(temp_prefix):
    """ADLSPresignedHelper.generate_download_url produces a signed HTTPS URL."""
    file_path = osp.join(temp_prefix.as_adls(), 'presign_test.txt')
    fsutil.write_text(file_path, 'hello')

    helper = fsutil.get_presigned_helper(file_path)
    assert helper is not None
    url = helper.generate_download_url(file_path)
    assert url.startswith('https://')
    assert temp_prefix.storage_account in url


@withADLSCredentials
def test_isdir_container_root():
    base_uri = adls_test_base_uri()
    assert fsutil.isdir(base_uri)
    assert fsutil.isdir(f"{base_uri}/")
