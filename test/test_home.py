# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access

import shutil
from pathlib import Path

import pytest

import kumo
from io_utils import fsutil
from kumo.home import get_home_dir, get_local_home_dir, set_home_dir
from service_framework.deploy_env import ENV_KUMO_HOME, get_default_home_dir

TEST_PATH = "/tmp/test/path"
EXISTING_S3_TEST_PATH = "s3://kumo-public-datasets/financial/"


def teardown_function():
    shutil.rmtree(TEST_PATH, ignore_errors=True)


def mock_s3_home_path(monkeypatch, existing_s3_path=EXISTING_S3_TEST_PATH):
    original_exists = kumo.home.fsutil.exists
    original_makedirs = kumo.home.fsutil.makedirs

    def exists(path):
        if isinstance(path, str) and path.startswith("s3://"):
            return path == existing_s3_path
        return original_exists(path)

    def makedirs(path, *args, **kwargs):
        if isinstance(path, str) and path.startswith("s3://"):
            return None
        return original_makedirs(path, *args, **kwargs)

    monkeypatch.setattr(kumo.home.fsutil, "exists", exists)
    monkeypatch.setattr(kumo.home.fsutil, "makedirs", makedirs)


def test_default_home_dir_is_repo_local(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_KUMO_HOME, raising=False)
    monkeypatch.chdir(tmp_path)

    repo_root = Path(__file__).resolve().parents[1]
    assert get_default_home_dir() == str(repo_root / ".cache" / "kumo")


def test_relative_kumo_home_override_is_workdir_local(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_KUMO_HOME, "custom-cache")
    monkeypatch.chdir(tmp_path)

    assert get_default_home_dir() == str(tmp_path / "custom-cache")


def test_remote_kumo_home_override_is_not_absolutized(monkeypatch):
    home = "dbfs:/Volumes/catalog/schema/volume/cache"
    monkeypatch.setenv(ENV_KUMO_HOME, home)

    assert get_default_home_dir() == home


# NOTE: These tests use a global variable - make sure to reset home_dir
# at the end of every test.
def test_local_home_dir():
    # Ensure the directory didn't exist already
    assert not fsutil.exists(TEST_PATH)
    set_home_dir(TEST_PATH)
    # This directory should be created.
    assert fsutil.exists(TEST_PATH)
    assert get_home_dir() == get_local_home_dir()

    # Reset the home_dir
    set_home_dir(kumo.home.get_default_home_dir())


def test_s3_path_home_dir(monkeypatch):
    mock_s3_home_path(monkeypatch)

    # Using a non-existent path should raise an error
    with pytest.raises(ValueError, match="doesn't exist"):
        set_home_dir("s3://kumo-public-datasets/random/path/does/not/exist")

    # Use an existing path for testing
    s3_path = EXISTING_S3_TEST_PATH
    set_home_dir(s3_path)

    assert get_home_dir() == s3_path
    # Local home_dir should be the default local cache root.
    assert get_local_home_dir() == kumo.home.get_default_home_dir()

    # Reset the home_dir
    set_home_dir(kumo.home.get_default_home_dir())


def test_first_task_name_handles_canonicalized_adls_scheme(monkeypatch):
    manager = kumo.home.HomeDirManager(
        'abfs://jobs@account.dfs.core.windows.net/training/job-1')
    monkeypatch.setattr(kumo.home.fsutil, 'exists', lambda _path: True)
    monkeypatch.setattr(
        kumo.home.fsutil, 'listdir',
        lambda *_args, **_kwargs: [
            'abfss://jobs@account.dfs.core.windows.net/training/job-1/models/NA/'
        ])

    assert manager.first_task_name() == 'NA'
