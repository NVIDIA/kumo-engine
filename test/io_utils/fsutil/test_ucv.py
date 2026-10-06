# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gzip
import os.path as osp
import tempfile
import time
from copy import deepcopy
from typing import Optional
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest
from databricks.sdk.errors.platform import NotFound

from io_utils import fsutil
from io_utils.databricks import DatabricksConnectionInfo, validate_host
from io_utils.fsutil.api import PathDetail
from io_utils.fsutil.common import DEFAULT_DATA_VERSION
from io_utils.fsutil.uri import UCURI, LocalURI, ParseURI
from kumo.connector.parquet_connector import ParquetConnector
from kumo.testing.databricks import withDatabricksCredentials
from service_framework.deploy_env import (
    DATABRICKS_HOST,
    DATABRICKS_WAREHOUSE_ID,
)

from .common import no_retry, random_name

TEST_CATALOG: str = 'ucv_test'


def test_db_connection_info_host():
    info = DatabricksConnectionInfo(host='host',
                                    cluster_id='cluster',
                                    warehouse_id='warehouse',
                                    catalog='catalog', credentials={})
    assert info.host == 'https://host'
    info = DatabricksConnectionInfo(host='https://host/',
                                    cluster_id='cluster',
                                    warehouse_id='warehouse',
                                    catalog='catalog', credentials={})
    assert info.host == 'https://host'
    info = DatabricksConnectionInfo(
                                    host='https://host/login.html',
                                    cluster_id='cluster',
                                    warehouse_id='warehouse',
                                    catalog='catalog', credentials={})
    assert info.host == 'https://host'
    # Test http scheme is converted to https
    info = DatabricksConnectionInfo(host='http://host',
                                    cluster_id='cluster',
                                    warehouse_id='warehouse',
                                    catalog='catalog', credentials={})
    assert info.host == 'https://host'


def test_temp_ucv(temp_ucv):
    temp_file = deepcopy(temp_ucv)
    temp_file.path = f"file-{random_name()}"
    content = b"some text data"
    with fsutil.get_binary_handle(temp_file.as_url(), 'w') as f:
        f.write(content)
    with fsutil.get_binary_handle(temp_file.as_url(), 'r') as f:
        assert f.read() == content
    fsutil.rm(temp_file)


def test_filesize(temp_ucv):
    temp_file = deepcopy(temp_ucv)
    temp_file.path = f"file-{random_name()}"
    assert fsutil.filesize(temp_file) is None
    data = b"some text data"
    with fsutil.get_binary_handle(temp_file.as_url(), 'w') as f:
        f.write(data)
    assert fsutil.filesize(temp_file) == len(data)
    fsutil.rm(temp_file)
    assert fsutil.filesize(temp_file) is None


def test_dirsize(temp_ucv):
    assert fsutil.dirsize(temp_ucv) == 0
    temp_file1 = deepcopy(temp_ucv)
    temp_file1.path = f"file-{random_name()}"
    data = b"some text data"
    with fsutil.get_binary_handle(temp_file1.as_url(), 'w') as f:
        f.write(data)
    assert fsutil.dirsize(temp_ucv) == len(data)
    temp_file2 = deepcopy(temp_ucv)
    temp_file2.path = f"subdir/file-{random_name()}"
    assert fsutil.dirsize(temp_ucv) == len(data)
    with fsutil.get_binary_handle(temp_file2.as_url(), 'w') as f:
        f.write(data)
    assert fsutil.dirsize(temp_ucv) == 2 * len(data)
    fsutil.rm(temp_file1)
    assert fsutil.dirsize(temp_ucv) == len(data)
    fsutil.rm(temp_file2)
    assert fsutil.dirsize(temp_ucv) == 0


def test_get_handle(ucws, temp_ucv):
    temp_file = deepcopy(temp_ucv)
    temp_file.path = f"file-{random_name()}"

    with pytest.raises(NotFound):
        ucws.files.download(temp_file.as_path()).contents
    with no_retry():
        with pytest.raises(FileNotFoundError):
            with fsutil.get_handle(temp_file, mode='r'):
                pass

    # uncompressed text
    with fsutil.get_handle(temp_file, mode='w') as f:
        f.write("some text data1")
    with ucws.files.download(temp_file.as_path()).contents as f:
        assert f.read() == b"some text data1"
    with fsutil.get_handle(temp_file, mode='r') as f:
        assert f.read() == "some text data1"
    with fsutil.get_binary_handle(temp_file, mode='r') as f:
        assert f.read() == b"some text data1"

    # uncompressed binary
    with fsutil.get_binary_handle(temp_file, mode='w') as f:
        f.write(b"some text data2")
    with ucws.files.download(temp_file.as_path()).contents as f:
        assert f.read() == b"some text data2"
    with fsutil.get_handle(temp_file, mode='r') as f:
        assert f.read() == "some text data2"
    with fsutil.get_binary_handle(temp_file, mode='r') as f:
        assert f.read() == b"some text data2"

    # compressed text
    with fsutil.get_handle(temp_file, mode='w', compression=True) as f:
        f.write("some text data3")
    with ucws.files.download(temp_file.as_path()).contents as f:
        assert gzip.decompress(f.read()) == b"some text data3"
    with fsutil.get_handle(temp_file, mode='r', compression=True) as f:
        assert f.read() == "some text data3"
    with fsutil.get_binary_handle(temp_file, mode='r', compression=True) as f:
        assert f.read() == b"some text data3"

    # compressed binary
    with fsutil.get_binary_handle(temp_file, mode='w', compression=True) as f:
        f.write(b"some text data4")
    with ucws.files.download(temp_file.as_path()).contents as f:
        assert gzip.decompress(f.read()) == b"some text data4"
    with fsutil.get_handle(temp_file, mode='r', compression=True) as f:
        assert f.read() == "some text data4"
    with fsutil.get_binary_handle(temp_file, mode='r', compression=True) as f:
        assert f.read() == b"some text data4"

    ucws.files.delete(temp_file.as_path())
    with pytest.raises(NotFound):
        ucws.files.download(temp_file.as_path()).contents
    with no_retry():
        with pytest.raises(FileNotFoundError):
            with fsutil.get_handle(temp_file, mode='r'):
                pass


def test_large_text_handle(ucws, temp_ucv):
    temp_file = deepcopy(temp_ucv)
    temp_file.path = f"file-{random_name()}"

    def _make_data(n_bytes: int) -> tuple[str, bytes]:
        s = (str(n_bytes) + "data" * (n_bytes // 4))[:n_bytes]
        b = s.encode("utf-8")
        assert len(b) == n_bytes
        return (s, b)

    def _run_test(n_bytes: int):
        s, b = _make_data(n_bytes)
        with fsutil.get_handle(temp_file, mode='w') as f:
            f.write(s)
        with ucws.files.download(temp_file.as_path()).contents as f:
            assert f.read() == b

    _run_test(0)
    _run_test(100)
    _run_test(1000)
    _run_test(8000)
    _run_test(8191)
    _run_test(8192)
    _run_test(8193)
    _run_test(16383)
    _run_test(16384)
    _run_test(16385)


def test_exists(ucws, temp_ucv):
    assert fsutil.exists(temp_ucv)

    # file
    temp_file = deepcopy(temp_ucv)
    temp_file.path = f"file-{random_name()}"
    assert not fsutil.exists(temp_file)
    content = b"some text data"
    with fsutil.get_binary_handle(temp_file.as_url(), 'w') as f:
        f.write(content)
    assert fsutil.exists(temp_file)
    fsutil.rm(temp_file)
    assert not fsutil.exists(temp_file)

    # dir
    temp_dir = deepcopy(temp_ucv)
    temp_dir.path = f"dir-{random_name()}"
    assert not fsutil.exists(temp_dir)
    ucws.dbfs.mkdirs(temp_dir.as_path())
    assert fsutil.exists(temp_dir)
    ucws.dbfs.delete(temp_dir.as_path(), recursive=True)
    assert not fsutil.exists(temp_dir)


def test_isdir(ucws, temp_ucv):
    assert fsutil.isdir(temp_ucv)
    temp_dir = deepcopy(temp_ucv)
    temp_dir.path = f"dir-{random_name()}"
    assert not fsutil.isdir(temp_dir)
    ucws.dbfs.mkdirs(temp_dir.as_path())
    assert fsutil.isdir(temp_dir)

    temp_file1 = deepcopy(temp_ucv)
    temp_file1.path = f"file-{random_name()}"
    assert not fsutil.isdir(temp_file1)
    data = b"some text data"
    with fsutil.get_binary_handle(temp_file1.as_url(), 'w') as f:
        f.write(data)
    assert not fsutil.isdir(temp_file1)
    ucws.files.delete(temp_file1.as_path())
    assert not fsutil.isdir(temp_file1)

    temp_file2 = deepcopy(temp_dir)
    temp_file2.path += f"/file-{random_name()}"
    assert not fsutil.isdir(temp_file2)
    with fsutil.get_binary_handle(temp_file2.as_url(), 'w') as f:
        f.write(data)
    assert not fsutil.isdir(temp_file2)
    ucws.files.delete(temp_file2.as_path())
    assert not fsutil.isdir(temp_file2)

    ucws.dbfs.delete(temp_dir.as_path(), recursive=True)
    assert not fsutil.isdir(temp_dir)


def test_get_many_if_present(temp_ucv):
    files: list[tuple[UCURI, Optional[bytes]]] = []
    for i in range(10):
        f = random_file(temp_ucv)
        if i % 3 != 0:
            c = f"some text data {i}".encode()
            with fsutil.get_binary_handle(f.as_url(), 'w') as f_handle:
                f_handle.write(c)
            files.append((f, c))
        else:
            # file missing
            files.append((f, None))

    paths = [f.as_url() for f, _ in files]
    content1 = fsutil.get_many_if_present(paths, parallel=False)
    assert content1 == [c for _, c in files]

    content2 = fsutil.get_many_if_present(paths, parallel=True)
    assert content2 == [c for _, c in files]

    content4 = fsutil.get_many_if_present([])
    assert content4 == []


def random_file(prefix: UCURI, k=16) -> UCURI:
    u = deepcopy(prefix)
    f = random_name(k=k)
    u.path = osp.join(u.path, f) if u.path else f
    return u


def test_touch(temp_ucv):
    temp_file1 = deepcopy(temp_ucv)
    temp_file1.path = f"file-{random_name()}"
    assert not fsutil.exists(temp_file1)
    fsutil.touch(temp_file1)
    assert fsutil.exists(temp_file1)
    # repeated touch is safe
    fsutil.touch(temp_file1)
    assert fsutil.exists(temp_file1)

    temp_file2 = deepcopy(temp_ucv)
    temp_file2.path = f"file-{random_name()}"
    data = b"some text data"
    with fsutil.get_binary_handle(temp_file2, 'w') as f:
        f.write(data)
    # touch does not change existing contents
    fsutil.touch(temp_file2)
    with fsutil.get_binary_handle(temp_file2, 'r') as f:
        assert f.read() == data


def test_has_write_permission(temp_ucv):
    assert fsutil.has_write_permission(temp_ucv)
    temp_file1 = deepcopy(temp_ucv)
    temp_file1.path = f"file-{random_name()}"
    assert fsutil.has_write_permission(temp_file1)

    # if the volume doesn't exist, can't write files to it
    bad_volume = UCURI(catalog=TEST_CATALOG, uc_schema=temp_ucv.uc_schema,
                       volume="does_not_exist", path="")
    assert not fsutil.exists(bad_volume)
    assert not fsutil.has_write_permission(bad_volume)

    # a volume that exists, but account has no permissions
    denied_volume = UCURI(catalog=TEST_CATALOG,
                          uc_schema="access_denied_schema",
                          volume="access_denied_volume", path="")
    # N.B. `exists` cannot succeed without read permissions, so returns False
    # ucv_test/access_denied_schema/access_denied_volume exists in UI
    assert not fsutil.exists(denied_volume)
    assert not fsutil.has_write_permission(denied_volume)

    # a volume that exists, but account has read-only permissions
    read_only_volume = UCURI(catalog=TEST_CATALOG,
                             uc_schema="access_denied_schema",
                             volume="read_only_volume", path="")
    assert fsutil.exists(read_only_volume)
    assert not fsutil.has_write_permission(read_only_volume)

    # an unowned volume where account has all privileges
    all_privileges_volume = UCURI(catalog=TEST_CATALOG,
                                  uc_schema="access_denied_schema",
                                  volume="all_privileges_volume", path="")
    assert fsutil.exists(all_privileges_volume)
    assert fsutil.has_write_permission(all_privileges_volume)

    # an unowned volume where account has read-write permissions
    read_write_volume = UCURI(catalog=TEST_CATALOG,
                              uc_schema="access_denied_schema",
                              volume="read_write_volume", path="")
    assert fsutil.exists(read_write_volume)
    assert fsutil.has_write_permission(read_write_volume)

    # edgecase, an unowned volume where account has write-only permissions
    writable_volume = UCURI(catalog=TEST_CATALOG,
                            uc_schema="access_denied_schema",
                            volume="writable_volume", path="")
    # N.B. `exists` cannot succeed without read permissions, so returns False
    # ucv_test/access_denied_schema/writable_volume exists in UI
    assert not fsutil.exists(writable_volume)
    assert fsutil.has_write_permission(writable_volume)


def test_copy_file(ucws, temp_ucv):
    data = b"some text data"
    data2 = b"some other text data"
    # local to remote
    with tempfile.NamedTemporaryFile() as temp_file1:
        temp_file1.write(data)
        temp_file1.flush()
        temp_file2 = deepcopy(temp_ucv)
        temp_file2.path = osp.basename(temp_file1.name)
        assert not fsutil.exists(temp_file2)
        fsutil.copy_file(temp_file1.name, temp_file2.as_url())
        assert fsutil.exists(temp_file2)
        # changing underlying data
        temp_file1.seek(0)
        temp_file1.truncate()
        temp_file1.write(data2)
        temp_file1.flush()
        with no_retry():
            with pytest.raises(FileExistsError):
                fsutil.copy_file(temp_file1.name, temp_file2.as_url())
        fsutil.copy_file(temp_file1.name, temp_file2.as_url(),
                         raise_on_exists=False)
        # failed copies do not change data
        with ucws.files.download(temp_file2.as_path()).contents as f:
            assert f.read() == data
        fsutil.copy_file(temp_file1.name, temp_file2.as_url(), overwrite=True)
        # overwrite does change data
        with ucws.files.download(temp_file2.as_path()).contents as f:
            assert f.read() == data2
        temp_file1.seek(0)
        temp_file1.truncate()
        temp_file1.write(data)
        temp_file1.flush()
        fsutil.copy_file(temp_file1.name, temp_file2.as_url(), overwrite=True)

    with ucws.files.download(temp_file2.as_path()).contents as f:
        assert f.read() == data

    # remote to remote
    temp_file3 = deepcopy(temp_ucv)
    temp_file3.path = f"file-{random_name()}"
    assert not fsutil.exists(temp_file3)
    fsutil.copy_file(temp_file2.as_url(), temp_file3.as_url())

    with ucws.files.download(temp_file3.as_path()).contents as f:
        assert f.read() == data

    with no_retry():
        with pytest.raises(FileExistsError):
            fsutil.copy_file(temp_file2.as_url(), temp_file3.as_url())
    fsutil.copy_file(temp_file2.as_url(), temp_file3.as_url(),
                     raise_on_exists=False)
    fsutil.copy_file(temp_file2.as_url(), temp_file3.as_url(), overwrite=True)
    ucws.files.delete(temp_file2.as_path())
    with no_retry():
        with pytest.raises(FileExistsError):
            fsutil.copy_file(temp_file2.as_url(), temp_file3.as_url())
    with no_retry():
        with pytest.raises(FileNotFoundError):
            fsutil.copy_file(temp_file2.as_url(), temp_file3.as_url(),
                             overwrite=True)

    # remote to local
    with tempfile.NamedTemporaryFile() as temp_file4:
        temp_file4.close()
        fsutil.copy_file(temp_file3.as_url(), temp_file4.name)
        with open(temp_file4.name, 'rb') as f:
            assert f.read() == data


def fill_dir(uri: UCURI, n_files: int, n_dirs: int, depth: int):
    content = b"some text data"
    fsutil.makedirs(uri, exist_ok=True)
    for i in range(n_files):
        file_uri = deepcopy(uri)
        file_uri.path = osp.join(file_uri.path, f"file{i}-{random_name()}")
        with fsutil.get_binary_handle(file_uri.as_url(), 'w') as f:
            f.write(content)
    if depth < 1:
        return
    for i in range(n_dirs):
        dir_uri = deepcopy(uri)
        dir_uri.path = osp.join(dir_uri.path, f"dir{i}-{random_name()}")
        fill_dir(uri=dir_uri, n_files=n_files, n_dirs=n_dirs, depth=depth - 1)


def test_listdir(ucws, temp_ucv):
    temp_dir = deepcopy(temp_ucv)
    temp_dir.path = f"dir-{random_name()}"
    n_files = 5
    n_dirs = 1
    fill_dir(temp_dir, n_files, n_dirs, 1)
    l0: list[PathDetail] = fsutil.listdir(temp_dir, detail=True)
    assert len(l0) == n_files + n_dirs
    assert sum([sub.is_directory for sub in l0]) == 1

    for sub in l0:
        assert sub.path.startswith(temp_dir.as_url())
        if fsutil.isdir(sub.path):
            l1: list[PathDetail] = fsutil.listdir(sub.path, detail=True)
            assert len(l1) == n_files
            assert sum([subsub.is_directory for subsub in l1]) == 0


def test_makedirs(temp_ucv):
    # one-level
    temp_dir1 = deepcopy(temp_ucv)
    temp_dir1.path = f"dir-{random_name()}"
    fsutil.makedirs(temp_dir1)
    assert fsutil.isdir(temp_dir1)

    # two-level
    temp_dir2 = deepcopy(temp_ucv)
    temp_dir2.path = f"dir-{random_name()}/dir-{random_name()}"
    fsutil.makedirs(temp_dir2)
    assert fsutil.isdir(temp_dir2)

    with no_retry():
        with pytest.raises(FileExistsError):
            fsutil.makedirs(temp_dir2)
    fsutil.makedirs(temp_dir2, exist_ok=True)


def test_rm(ucws, temp_ucv):
    # file
    temp_file = deepcopy(temp_ucv)
    temp_file.path = f"file-{random_name()}"
    content = b"some text data"
    with fsutil.get_binary_handle(temp_file.as_url(), 'w') as f:
        f.write(content)
    assert fsutil.exists(temp_file)
    fsutil.rm(temp_file)
    assert not fsutil.exists(temp_file)
    # nonexistent
    fsutil.rm(temp_file)

    # dir
    temp_dir = deepcopy(temp_ucv)
    temp_dir.path = f"dir-{random_name()}"
    ucws.dbfs.mkdirs(temp_dir.as_path())
    assert fsutil.exists(temp_dir)
    assert fsutil.isdir(temp_dir)
    fsutil.rm(temp_dir)
    assert not fsutil.exists(temp_dir)

    # dir containing files
    fill_dir(temp_dir, 5, 0, 1)
    assert fsutil.exists(temp_dir)
    assert fsutil.isdir(temp_dir)
    fsutil.rm(temp_dir)
    assert not fsutil.exists(temp_dir)

    # dir containing dirs and files
    fill_dir(temp_dir, 2, 2, 2)
    assert fsutil.exists(temp_dir)
    assert fsutil.isdir(temp_dir)
    fsutil.rm(temp_dir)
    assert not fsutil.exists(temp_dir)


def test_modified_time(temp_ucv):
    assert fsutil.get_last_modified_time(temp_ucv) == DEFAULT_DATA_VERSION
    success_file = deepcopy(temp_ucv)
    success_file.path = "_SUCCESS"
    permitted_skew = 10  # seconds
    t1 = time.time()
    fsutil.touch(success_file)
    t2 = time.time()
    ts = fsutil.get_last_modified_time(success_file)
    assert t1 - permitted_skew < ts < t2 + permitted_skew
    td = fsutil.get_last_modified_time(temp_ucv)
    assert td == ts


def test_parquet_on_ucv(temp_ucv, tempdir):
    # Write out a parquet file to UCV by copying from local.
    data = np.random.rand(5, 3)
    df = pd.DataFrame(data, columns=['A', 'B', 'C'])
    filename = 'test.parquet'
    fsutil.makedirs(tempdir.name)
    temp_file = osp.join(tempdir.name, filename)
    df.to_parquet(temp_file)
    fsutil.copy(temp_file, osp.join(temp_ucv.as_url(), filename))
    # read it back using the parquet connector
    connector = ParquetConnector('ucv_parquet_connector', temp_ucv.as_url())
    df2 = connector.read_table('test')
    assert df.equals(df2)


def test_contains_subdir_on_ucv(temp_ucv, ucws):
    temp_dir = deepcopy(temp_ucv)
    temp_dir.path = f"dir-{random_name()}"
    fsutil.makedirs(temp_dir.as_url())
    assert not fsutil.contains_subdir(temp_dir)
    n_files = 5
    n_dirs = 2
    fill_dir(temp_dir, n_files, n_dirs, 1)
    assert fsutil.contains_subdir(temp_dir)


@withDatabricksCredentials
def test_auth():
    old_client = fsutil.impl_ucv.uc_ctx._local.client
    fsutil.impl_ucv.uc_ctx._local.client = None
    # .client property does on-demand login
    # auth defaults to aws secrets, can access catalogs, but not TEST_CATALOG
    fsutil.impl_ucv.uc_ctx.client.catalogs.list()
    missing_creds = DatabricksConnectionInfo.from_environment()
    missing_creds.credentials = {}
    with pytest.raises(RuntimeError):
        fsutil.impl_ucv.uc_ctx._connect(missing_creds)
    fsutil.impl_ucv.uc_ctx._local.client = old_client


@withDatabricksCredentials
def test_connection_info_load():
    info = DatabricksConnectionInfo.from_environment()
    assert info.host == validate_host(DATABRICKS_HOST.parse_value())
    assert info.credentials.get('pat') is not None
    assert info.warehouse_id == DATABRICKS_WAREHOUSE_ID.parse_value()
    assert not info.cluster_id
    assert not info.catalog


# Tests for auto-recovery on Databricks credential rotation.

def test_is_auth_error():
    """_is_auth_error matches invalid_client errors and nothing else."""
    from io_utils.fsutil.impl_ucv import _is_auth_error
    assert _is_auth_error(ValueError("invalid_client: Client authentication failed"))
    assert _is_auth_error(Exception("invalid_client"))
    assert not _is_auth_error(ValueError("some other error"))
    assert not _is_auth_error(RuntimeError("invalid_request"))


def test_uc_context_reconnect_clears_cached_state():
    """reconnect() clears both the cached client and connection_info."""
    from io_utils.fsutil.impl_ucv import UCContext
    ctx = UCContext()
    ctx._local.client = object()
    ctx._connection_info = DatabricksConnectionInfo(
        host='https://h', cluster_id='c', warehouse_id='w',
        catalog='', credentials={})
    ctx.reconnect()
    assert ctx._local.client is None
    assert ctx._connection_info is None


def test_retry_policy_reconnects_on_auth_failure(caplog):
    """retry_policy invokes uc_ctx.reconnect() on invalid_client errors,
    then retries the operation."""
    from io_utils.fsutil import impl_ucv

    call_count = 0
    canary = 'invalid_client: token=ucv-auth-secret-canary'

    @impl_ucv.retry_policy()
    def operation():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError(canary)
        return "success"

    with patch.object(impl_ucv.uc_ctx, 'reconnect') as mock_reconnect:
        result = operation()

    assert result == "success"
    assert call_count == 2
    mock_reconnect.assert_called_once()
    assert canary not in caplog.text
    assert 'ucv-auth-secret-canary' not in caplog.text


def test_upload_retry_log_never_exposes_provider_error(monkeypatch, caplog):
    from io_utils.fsutil import impl_ucv

    canary = 'databricks-provider-body token=ucv-upload-secret-canary'
    files = MagicMock()
    files.upload_from.side_effect = [RuntimeError(canary), None]
    client = MagicMock(files=files)
    monkeypatch.setattr(impl_ucv, '_check_size_match', lambda **_kwargs: None)
    monkeypatch.setattr('io_utils.retry.time.sleep', lambda _delay: None)
    local = LocalURI(path='/tmp/local-artifact')
    remote = ParseURI('dbfs:/Volumes/catalog/schema/volume/artifact')

    with patch.object(type(impl_ucv.uc_ctx), 'client', new_callable=PropertyMock,
                      return_value=client):
        impl_ucv._upload(local=local, remote=remote)

    assert files.upload_from.call_count == 2
    assert canary not in caplog.text
    assert 'ucv-upload-secret-canary' not in caplog.text


def test_retry_policy_does_not_reconnect_on_other_errors():
    """retry_policy does NOT call reconnect for non-auth errors (e.g.
    transient RuntimeError). It still retries via the base retry."""
    from io_utils.fsutil import impl_ucv

    call_count = 0

    @impl_ucv.retry_policy()
    def operation():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("transient failure")
        return "success"

    with patch.object(impl_ucv.uc_ctx, 'reconnect') as mock_reconnect:
        result = operation()

    assert result == "success"
    assert call_count == 2
    mock_reconnect.assert_not_called()


def test_retry_policy_gives_up_after_max_tries():
    """If auth failure persists across all retries, retry_policy still
    raises after exhausting attempts."""
    from io_utils.fsutil import impl_ucv

    call_count = 0

    @impl_ucv.retry_policy()
    def operation():
        nonlocal call_count
        call_count += 1
        raise ValueError("invalid_client: Client authentication failed")

    with patch.object(impl_ucv.uc_ctx, 'reconnect'):
        with pytest.raises(ValueError, match="invalid_client"):
            operation()

    assert call_count == 3
