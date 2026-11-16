# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Token renewal at file boundaries, without serializing native transfers."""
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from google.auth.credentials import Credentials
from google.cloud import storage
from google.oauth2 import service_account
from kumo_gcs import download as impl_gcs
from kumo_gcs.download import GCSDownloader

UTC = datetime.timezone.utc


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
        impl_gcs, 'datetime',
        SimpleNamespace(datetime=Clock, timedelta=datetime.timedelta,
                        timezone=datetime.timezone))
    made, copied = [], []

    def filesystem(**kwargs):
        result = SimpleNamespace(**kwargs)
        made.append(result)
        return result

    monkeypatch.setattr(impl_gcs.pafs, 'GcsFileSystem', filesystem)
    monkeypatch.setattr(
        impl_gcs.pafs, 'copy_files',
        lambda *args, **kwargs: copied.append(kwargs['source_filesystem']))
    monkeypatch.setattr(impl_gcs, 'gcs_io_threads', lambda: 32)
    monkeypatch.setattr(impl_gcs.pa, 'io_thread_count', lambda: 32)
    return SimpleNamespace(clock=clock, made=made, copied=copied)


def test_cached_expiry_does_not_follow_sdk_token_renewal(native):
    credential = _Credential(native.clock)
    downloader = GCSDownloader(credential)
    downloader._download_native('bucket', 'object', '/unused/native-test')
    old = native.copied[-1]
    # An SDK metadata request renews the shared credential independently.
    credential.token = 'sdk-renewed'
    credential.expiry += datetime.timedelta(hours=1)
    downloader._download_native('bucket', 'object', '/unused/native-test')
    assert native.copied[-1] is old
    native.clock.now += datetime.timedelta(minutes=51)
    downloader._download_native('bucket', 'object', '/unused/native-test')
    assert native.copied[-1] is not old
    assert old.access_token == 'old'
    assert native.copied[-1].access_token == 'sdk-renewed'
    assert credential.refreshes == 0


def test_short_lived_tokens_refresh_once_and_are_reused(native):
    credential = _Credential(native.clock, lifetime=-1)
    credential.lifetime = 5
    downloader = GCSDownloader(credential)
    downloader._download_native('bucket', 'object', '/unused/native-test')
    native.clock.now += datetime.timedelta(minutes=1)
    downloader._download_native('bucket', 'object', '/unused/native-test')
    assert credential.refreshes == 1
    assert native.copied[0] is native.copied[1]
    native.clock.now += datetime.timedelta(minutes=2)
    downloader._download_native('bucket', 'object', '/unused/native-test')
    assert credential.refreshes == 2
    assert native.copied[-1].access_token == 'renewed-2'


def test_existing_token_keeps_the_full_ten_minute_margin(native):
    credential = _Credential(native.clock, lifetime=11)
    downloader = GCSDownloader(credential)
    downloader._download_native('bucket', 'object', '/unused/native-test')
    assert credential.refreshes == 0
    native.clock.now += datetime.timedelta(minutes=1)
    downloader._download_native('bucket', 'object', '/unused/native-test')
    assert credential.refreshes == 1
    assert native.copied[-1].access_token == 'renewed-1'


def test_concurrent_files_share_refresh_and_transfer_in_parallel(
        native, monkeypatch):
    credential = _Credential(native.clock, lifetime=-1)
    downloads = threading.Barrier(2, timeout=5)

    def copy(*args, source_filesystem, **kwargs):
        native.copied.append(source_filesystem)
        downloads.wait()

    monkeypatch.setattr(impl_gcs.pafs, 'copy_files', copy)
    downloader = GCSDownloader(credential)
    with ThreadPoolExecutor(2) as pool:
        futures = [
            pool.submit(downloader._download_native, 'bucket', 'object',
                        '/unused/native-test') for _ in range(2)
        ]
        for future in futures:
            future.result(timeout=10)
    assert credential.refreshes == 1
    assert len(native.made) == 1
    assert native.copied[0] is native.copied[1]


def test_inflight_transfer_keeps_old_filesystem_during_refresh(
        native, monkeypatch):
    credential = _Credential(native.clock)
    first_started = threading.Event()
    transfers_overlap = threading.Barrier(2, timeout=5)

    def copy(*args, source_filesystem, **kwargs):
        native.copied.append(source_filesystem)
        if source_filesystem.access_token == 'old':
            first_started.set()
        transfers_overlap.wait()

    monkeypatch.setattr(impl_gcs.pafs, 'copy_files', copy)
    downloader = GCSDownloader(credential)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(downloader._download_native, 'bucket', 'object',
                            '/unused/native-test')
        assert first_started.wait(timeout=5)
        native.clock.now += datetime.timedelta(minutes=51)
        second = pool.submit(downloader._download_native, 'bucket', 'object',
                             '/unused/native-test')
        first.result(timeout=10)
        second.result(timeout=10)
    assert [fs.access_token for fs in native.copied] == ['old', 'renewed-1']
    assert native.copied[0] is not native.copied[1]


@pytest.mark.parametrize('invalid', ['token', 'expiry', 'expired'])
def test_invalid_refreshed_credentials_never_start_transfer(native, invalid):
    credential = _Credential(native.clock, lifetime=-1)

    def refresh(request):
        credential.token = None if invalid == 'token' else 'new'
        credential.expiry = (
            None if invalid == 'expiry' else native.clock.now +
            datetime.timedelta(minutes=-1 if invalid == 'expired' else 60))

    credential.refresh = refresh
    downloader = GCSDownloader(credential)
    with pytest.raises(ValueError):
        downloader._download_native('bucket', 'object', '/unused/native-test')
    assert not native.made
    assert not native.copied


def test_refresh_error_never_reuses_expired_filesystem(native):
    credential = _Credential(native.clock)
    downloader = GCSDownloader(credential)
    downloader._download_native('bucket', 'object', '/unused/native-test')
    native.clock.now += datetime.timedelta(hours=2)

    def fail(request):
        raise RuntimeError('token endpoint unavailable')

    credential.refresh = fail
    with pytest.raises(RuntimeError, match='token endpoint unavailable'):
        downloader._download_native('bucket', 'object', '/unused/native-test')
    assert len(native.copied) == 1


@pytest.mark.parametrize('offset', [None, datetime.timedelta(hours=5)])
def test_token_expiry_preserves_absolute_time(native, offset):
    credential = _Credential(native.clock)
    expected = credential.expiry
    credential.expiry = (expected.replace(tzinfo=None) if offset is None else
                         expected.astimezone(datetime.timezone(offset)))
    downloader = GCSDownloader(credential)
    downloader._download_native('bucket', 'object', '/unused/native-test')
    assert native.copied[-1].credential_token_expiration == expected


@pytest.mark.parametrize('scopes', [None, ('scope:existing', )])
def test_native_scopes_service_account_once_without_changing_existing_scopes(
        native, monkeypatch, scopes):
    credential = service_account.Credentials(
        signer=SimpleNamespace(key_id='test'),
        service_account_email='test@test',
        token_uri='https://oauth2.googleapis.com/token', scopes=scopes)
    refreshed = []

    def refresh(scoped, request):
        assert not scoped.requires_scopes
        assert scoped.scopes == (scopes or storage.Client.SCOPE)
        assert scoped.service_account_email == credential.service_account_email
        refreshed.append(scoped)
        scoped.token = f'scoped-token-{len(refreshed)}'
        scoped.expiry = native.clock.now + datetime.timedelta(hours=1)

    monkeypatch.setattr(service_account.Credentials, 'refresh', refresh)
    downloader = GCSDownloader(credential)
    downloader._download_native('bucket', 'object', '/unused/native-test')
    downloader._download_native('bucket', 'object', '/unused/native-test')
    assert len(refreshed) == 1
    assert native.copied[0] is native.copied[1]
    native.clock.now += datetime.timedelta(minutes=51)
    downloader._download_native('bucket', 'object', '/unused/native-test')
    assert native.copied[-1].access_token == 'scoped-token-2'
    assert len(refreshed) == 2
    assert refreshed[0] is refreshed[1]
    if scopes is None:
        assert refreshed[0] is not credential
        assert credential.requires_scopes
        assert credential.token is None
    else:
        assert refreshed[0] is credential
