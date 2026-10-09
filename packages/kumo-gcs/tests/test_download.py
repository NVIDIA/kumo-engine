# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import datetime
import errno
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pyarrow as pa
import pytest
from google.api_core import exceptions as gcs_exceptions
from google.auth.credentials import AnonymousCredentials
from kumo_gcs import download
from kumo_gcs.download import GCSDownloader


@pytest.fixture
def downloader(monkeypatch):
    monkeypatch.delenv('STORAGE_EMULATOR_HOST', raising=False)
    client = GCSDownloader()
    blob = mock.Mock(size=7, content_encoding=None, generation=111)
    monkeypatch.setattr(client, '_blob', lambda bucket, key: blob)
    monkeypatch.setattr(client, 'filesystem', lambda: 'native-fs')
    monkeypatch.setattr(download, 'gcs_io_threads', lambda: 1)
    return client, blob


def test_download_chunk_size_is_applied_to_the_client_library():
    from google.cloud.storage._media.requests import _request_helpers
    assert _request_helpers._SINGLE_GET_CHUNK_SIZE == 1024 * 1024


@pytest.mark.parametrize('name', [
    'plain.bin',
    'my file.bin',
    'dt=2026-01-01 00:00:00/part.bin',
    '100%.bin',
])
def test_native_download_preserves_object_names(downloader, monkeypatch,
                                                tmp_path, name):
    client, _ = downloader

    def copy(source, target, **kwargs):
        assert source == f'bucket/{name}'
        Path(target).write_bytes(b'payload')

    monkeypatch.setattr(download.pafs, 'copy_files', copy)
    target = tmp_path / 'output'
    assert client.download('bucket', name, str(target)) is None
    assert target.read_bytes() == b'payload'
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize('generation', [111, 222, None])
def test_native_download_checks_fresh_generation(downloader, monkeypatch,
                                                 tmp_path, generation):
    client, pinned = downloader
    current = mock.Mock(generation=generation)
    if generation is None:
        current.reload.side_effect = gcs_exceptions.NotFound('removed')
    blobs = iter([pinned, current])
    monkeypatch.setattr(client, '_blob', lambda bucket, key: next(blobs))
    monkeypatch.setattr(
        download.pafs, 'copy_files',
        lambda source, target, **kwargs: Path(target).write_bytes(b'payload'))
    target = tmp_path / 'output'
    target.write_bytes(b'previous')
    if generation == 111:
        client.download('bucket', 'object', str(target))
        assert target.read_bytes() == b'payload'
    else:
        with pytest.raises(OSError, match='was replaced'):
            client.download('bucket', 'object', str(target))
        assert target.read_bytes() == b'previous'
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize('error,expected', [
    (None, OSError),
    (gcs_exceptions.NotFound('removed'), FileNotFoundError),
    (OSError('UNAVAILABLE'), ConnectionError),
    (pa.ArrowCapacityError('RESOURCE_EXHAUSTED'), ConnectionError),
    (OSError(errno.ENOSPC, 'full'), OSError),
    (PermissionError('denied'), PermissionError),
])
def test_failed_native_download_preserves_destination(downloader, monkeypatch,
                                                      tmp_path, error,
                                                      expected):
    client, _ = downloader

    def copy(source, target, **kwargs):
        Path(target).write_bytes(b'short')
        if error is not None:
            raise error

    monkeypatch.setattr(download.pafs, 'copy_files', copy)
    target = tmp_path / 'output'
    target.write_bytes(b'previous')
    with pytest.raises(expected) as raised:
        client.download('bucket', 'object', str(target))
    if isinstance(error, OSError) and error.errno is not None:
        assert raised.value is error
    assert target.read_bytes() == b'previous'
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize('reason', ['gzip', 'emulator', 'requester', 'quota'])
@pytest.mark.parametrize('fail', [False, True])
def test_sdk_routing_and_atomic_download(downloader, monkeypatch, tmp_path,
                                         reason, fail):
    client, blob = downloader
    if reason == 'gzip':
        blob.size, blob.content_encoding = 30, 'gzip'
    elif reason == 'emulator':
        monkeypatch.setenv('STORAGE_EMULATOR_HOST', 'http://localhost:4443')
    elif reason == 'requester':
        client.requester_pays_project = 'billing-project'
    else:
        client.credentials = SimpleNamespace(quota_project_id='billing-project')

    def sdk(output):
        output.write(b'payload')
        blob.content_encoding = None
        if fail:
            raise gcs_exceptions.NotFound('removed during transfer')

    blob.download_to_file.side_effect = sdk
    monkeypatch.setattr(download.pafs, 'copy_files',
                        lambda *a, **kw: pytest.fail('SDK routing required'))
    target = tmp_path / 'output'
    target.write_bytes(b'previous')
    if fail:
        with pytest.raises(FileNotFoundError):
            client.download('bucket', 'object', str(target))
        assert target.read_bytes() == b'previous'
    else:
        size = client.download('bucket', 'object', str(target))
        assert size == (7 if reason == 'gzip' else None)
        assert target.read_bytes() == b'payload'
    assert list(tmp_path.iterdir()) == [target]


def test_ambient_filesystem_is_shared_and_bounds_retry(monkeypatch):
    constructed = []
    start = threading.Barrier(4, timeout=5)

    def filesystem(**kwargs):
        constructed.append(kwargs)
        return object()

    monkeypatch.setattr(download.pafs, 'GcsFileSystem', filesystem)
    client = GCSDownloader()

    def get(_):
        start.wait()
        return client.filesystem()

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(get, range(4)))
    assert all(result is results[0] for result in results)
    assert constructed == [{
        'anonymous': False,
        'retry_time_limit': datetime.timedelta(seconds=120),
    }]


def test_anonymous_access_never_discovers_ambient_credentials(monkeypatch):
    monkeypatch.setattr('google.auth.default',
                        lambda *a, **kw: pytest.fail('ambient discovery'))
    client = GCSDownloader(AnonymousCredentials())
    assert client._blob('bucket',
                        'object').client._credentials is client.credentials
    assert client.filesystem() is not None


@pytest.mark.parametrize('budget,override,expected', [
    (2, None, 32),
    (12, None, 48),
    (96, None, 64),
    (96, '64', 64),
    (96, '1000000', 128),
    (96, '0', 64),
])
def test_native_io_budget(monkeypatch, budget, override, expected):
    monkeypatch.delenv('FSUTIL_GCS_IO_THREADS', raising=False)
    if override is not None:
        monkeypatch.setenv('FSUTIL_GCS_IO_THREADS', override)
    monkeypatch.setattr(download, 'cpu_budget', lambda: budget)
    assert download.gcs_io_threads() == expected


@pytest.mark.parametrize('existing,changes', [(8, [64]), (96, [])])
def test_native_download_never_lowers_shared_io_pool(downloader, monkeypatch,
                                                     existing, changes):
    client, _ = downloader
    observed = []
    monkeypatch.setattr(download, 'gcs_io_threads', lambda: 64)
    monkeypatch.setattr(download.pa, 'io_thread_count', lambda: existing)
    monkeypatch.setattr(download.pa, 'set_io_thread_count', observed.append)
    monkeypatch.setattr(download.pafs, 'copy_files', lambda *a, **kw: None)
    client._download_native('bucket', 'object', '/unused')
    assert observed == changes


def test_object_identity_does_not_reuse_a_pinned_blob(monkeypatch):
    from types import SimpleNamespace

    from google.auth.credentials import AnonymousCredentials

    downloader = download.GCSDownloader(credentials=AnonymousCredentials())
    created = []

    def blob(bucket, key):
        assert (bucket, key) == ('bucket', 'key')
        generation = len(created) + 1
        probe = SimpleNamespace(size=100, generation=None)
        probe.reload = lambda: setattr(probe, 'generation', generation)
        created.append(probe)
        return probe

    monkeypatch.setattr(downloader, '_blob', blob)
    assert downloader.object_identity('bucket', 'key') == ('1', '100')
    assert downloader.object_identity('bucket', 'key') == ('2', '100')
    assert created[0] is not created[1]
