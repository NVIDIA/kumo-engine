# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import datetime
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.auth.credentials import Credentials
from kumo_gcs import download

from io_utils.fsutil import api, impl_gcs
from io_utils.fsutil.credentials import credential_scope
from io_utils.fsutil.uri import GCSURI

UTC = datetime.timezone.utc
SOURCE = GCSURI(bucket='bucket', path='object')


class _Credential(Credentials):
    def __init__(self, clock, token='old', lifetime=60):
        super().__init__()
        self.clock = clock
        self.token = token
        self.expiry = clock.now + datetime.timedelta(minutes=lifetime)
        self.refreshes = 0
        self.lifetime = 60

    def refresh(self, request):
        self.refreshes += 1
        self.token = f'renewed-{self.refreshes}'
        self.expiry = self.clock.now + datetime.timedelta(minutes=self.lifetime)


@pytest.fixture
def native(monkeypatch):
    clock = SimpleNamespace(now=datetime.datetime(2026, 1, 1, tzinfo=UTC))

    class Clock(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return clock.now if tz else clock.now.replace(tzinfo=None)

    monkeypatch.setattr(
        download, 'datetime',
        SimpleNamespace(datetime=Clock, timedelta=datetime.timedelta,
                        timezone=datetime.timezone))
    made, copied = [], []

    def filesystem(**kwargs):
        result = SimpleNamespace(**kwargs)
        made.append(result)
        return result

    monkeypatch.setattr(download.pafs, 'GcsFileSystem', filesystem)
    monkeypatch.setattr(
        download.pafs, 'copy_files',
        lambda *args, **kwargs: copied.append(kwargs['source_filesystem']))
    monkeypatch.setattr(download, 'gcs_io_threads', lambda: 32)
    monkeypatch.setattr(download.pa, 'io_thread_count', lambda: 32)
    return SimpleNamespace(clock=clock, made=made, copied=copied)


def _scope(credential):
    return credential_scope({
        'provider': 'gcs',
        'credential': credential
    }, require_explicit=True)


def _download():
    impl_gcs._download_client()._download_native('bucket', 'object',
                                                 '/unused/native-test')


def test_distinct_scopes_on_same_worker_do_not_share_tokens(native):
    for token in ('customer-a', 'customer-b'):
        with _scope(_Credential(native.clock, token=token)):
            client = impl_gcs._download_client()
            assert impl_gcs._download_client() is client
            _download()
    assert [fs.access_token
            for fs in native.copied] == ['customer-a', 'customer-b']
    assert native.copied[0] is not native.copied[1]


def test_missing_scoped_credential_never_uses_ambient(native, monkeypatch):
    def ambient(*args, **kwargs):
        pytest.fail('explicit scope must not use ambient credentials')

    monkeypatch.setattr(impl_gcs, 'GCSDownloader', ambient)
    with _scope(None), pytest.raises(RuntimeError, match='Missing federated'):
        _download()
    assert not native.copied


def test_snapshot_restore_uses_native_and_publishes_success_last(
        native, monkeypatch, tmp_path):
    from kumo.diskgraph.service import _copy_dataplane_mirror

    source = 'gs://bucket/snapshot'
    contents = {
        'data': b'new data',
        'nested/part': b'part',
        '_SUCCESS': b'ready'
    }
    (tmp_path / 'data').write_bytes(b'old data')
    (tmp_path / '_SUCCESS').write_bytes(b'stale marker')
    (tmp_path / 'unrelated').write_bytes(b'keep me')
    original_isdir = api.isdir
    monkeypatch.setattr(
        api, 'isdir', lambda path:
        (path in (source, source + '/nested')
         if str(path).startswith('gs://') else original_isdir(path)))
    listed = []

    def listdir(path, *, detail):
        assert detail
        listed.append(path)
        names = (('data', False), ('nested', True), ('_SUCCESS', False))
        if path.endswith('/nested'):
            names = (('part', False), )
        return [
            api.PathDetail(f'{path}/{name}', directory)
            for name, directory in names
        ]

    def blob(self, bucket, key):
        name = key.removeprefix('snapshot/')
        return SimpleNamespace(reload=lambda: None, size=len(contents[name]),
                               generation=1, content_encoding=None)

    transferred = []

    def copy(source_path, destination, *, source_filesystem, **kwargs):
        name = source_path.removeprefix('bucket/snapshot/')
        assert not (tmp_path / '_SUCCESS').exists()
        if name == '_SUCCESS':
            for data_file in ('data', 'nested/part'):
                assert (tmp_path /
                        data_file).read_bytes() == contents[data_file]
        Path(destination).write_bytes(contents[name])
        transferred.append(name)
        assert source_filesystem.access_token == 'restore-identity'

    monkeypatch.setattr(api, 'listdir', listdir)
    monkeypatch.setattr(download.GCSDownloader, '_blob', blob)
    monkeypatch.setattr(download.pafs, 'copy_files', copy)
    with _scope(_Credential(native.clock, token='restore-identity')):
        _copy_dataplane_mirror(source, str(tmp_path))
    assert set(listed) == {source, source + '/nested'}
    assert set(transferred[:-1]) == {'data', 'nested/part'}
    assert transferred[-1] == '_SUCCESS'
    assert (tmp_path / '_SUCCESS').read_bytes() == b'ready'
    assert (tmp_path / 'unrelated').read_bytes() == b'keep me'
    assert not list(tmp_path.rglob('*.gcsdl-*'))


def test_importing_arrow_first_registers_gcs_handlers():
    code = ("import io_utils.pyarrow; "
            "from io_utils.fsutil import api; "
            "from io_utils.fsutil.uri import GCSURI; "
            "assert api.isdir.dispatch(GCSURI).__module__ == "
            "'io_utils.fsutil.impl_gcs'")
    subprocess.run([sys.executable, '-c', code], check=True, timeout=30,
                   capture_output=True, text=True)
