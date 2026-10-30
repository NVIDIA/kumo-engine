# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import os.path as osp
import tempfile
from test.io_utils.fsutil.common import random_name
from test.io_utils.fsutil.contract import FilesystemContract
from typing import Generator

import pytest

from io_utils import fsutil
from kumo.testing.adls import adls_test_base_uri, withADLSCredentials
from kumo.testing.gcs import withGCSCredentials
from kumo.testing.s3 import withS3Credentials


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as td:
        yield td


@pytest.fixture
@withS3Credentials
def root_s3() -> Generator[str, None, None]:
    yield "s3://kumo-test-dataplane-bucket/fsutil-test/"


@pytest.fixture
@withGCSCredentials
def root_gcs() -> Generator[str, None, None]:
    from kumo.testing.gcs import ENV_GCS_TEST_BUCKET
    bucket = os.getenv(ENV_GCS_TEST_BUCKET, 'kumo-test-bucket')
    yield f"gs://{bucket}/fsutil-test/"


@pytest.fixture
@withADLSCredentials
def root_adls() -> Generator[str, None, None]:
    yield f"{adls_test_base_uri()}/fsutil-test/"


@pytest.fixture
def root_ucv(temp_ucv) -> Generator[str, None, None]:
    yield temp_ucv.as_url()


@pytest.fixture(
    params=[
        "root_s3",
        pytest.param("root_gcs", marks=pytest.mark.requires_credentials('gcp')),
        "root_ucv",
        "root_adls",
    ],
    ids=["s3", "gcs", "ucv", "adls"],
)
def root(request) -> Generator[str, None, None]:
    base: str = request.getfixturevalue(request.param)
    path = osp.join(base, random_name())
    fsutil.makedirs(path)
    yield path
    fsutil.rm(path)


class TestProviderFilesystem(FilesystemContract):
    pass
