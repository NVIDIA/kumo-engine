# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import errno
import io
from types import SimpleNamespace
from unittest import mock

import pytest
from google.api_core import exceptions as gcs_exceptions
from kumo_gcs import download

from io_utils import fsutil
from io_utils.fsutil import impl_gcs
from io_utils.fsutil.credentials import credential_scope
from io_utils.fsutil.uri import GCSURI, ParseURI
from kumo.testing.gcs import withMockedGCSFilesystem


def test_get_signing_credentials_uses_wif_service_account(monkeypatch):
    class FederatedCredentials:
        service_account_email = (
            'gha-kumo-gcp-tests@example-project.iam.gserviceaccount.com')

        def refresh(self, request):
            self.request = request

    credentials = FederatedCredentials()
    signing_credentials = mock.Mock(name='signing_credentials')
    create_signer = mock.Mock(return_value=signing_credentials)
    load_credentials = mock.Mock(return_value=(credentials, 'example-project'))
    monkeypatch.setattr(impl_gcs, 'auth_default', load_credentials)
    monkeypatch.setattr(impl_gcs.impersonated_credentials, 'Credentials',
                        create_signer)

    result = impl_gcs._get_signing_credentials()

    assert result is signing_credentials
    load_credentials.assert_called_once_with(
        scopes=['https://www.googleapis.com/auth/devstorage.read_only'])
    create_signer.assert_called_once_with(
        source_credentials=credentials,
        target_principal=credentials.service_account_email,
        target_scopes=['https://www.googleapis.com/auth/devstorage.read_only'],
    )


def test_object_identity_uses_generation_and_etag(monkeypatch):
    blob = mock.Mock()
    blob.size = 456
    blob.etag = 'etag-value'
    blob.generation = 789
    monkeypatch.setattr(impl_gcs, '_get_blob', lambda _uri: blob)

    identity = fsutil.object_identity('gs://bucket/path/file.parquet')

    blob.reload.assert_called_once_with()
    assert identity == fsutil.ObjectIdentity(
        size=456,
        etag='etag-value',
        version='789',
    )


def test_is_gcs():
    """Test GCS path detection."""
    assert fsutil.is_gcs('gs://bucket')
    assert fsutil.is_gcs('gs://bucket/prefix')
    assert fsutil.is_gcs('gs://bucket/path/to/file.txt')
    assert not fsutil.is_gcs('s3://bucket')
    assert not fsutil.is_gcs('/home/tester')
    assert not fsutil.is_gcs('../tester/path')


def test_parse_gcs_uri():
    """Test GCS URI parsing."""
    uri = ParseURI('gs://my-bucket/path/to/file.txt')
    assert isinstance(uri, GCSURI)
    assert uri.bucket == 'my-bucket'
    assert uri.path == 'path/to/file.txt'
    assert uri.as_gs() == 'gs://my-bucket/path/to/file.txt'

    # Test bucket-only path
    uri = ParseURI('gs://my-bucket')
    assert isinstance(uri, GCSURI)
    assert uri.bucket == 'my-bucket'
    assert uri.path == ''


@pytest.mark.parametrize(('error', 'expected'), [
    (gcs_exceptions.TooManyRequests('quota'), True),
    (gcs_exceptions.ServiceUnavailable('down'), True),
    (gcs_exceptions.GatewayTimeout('timeout'), True),
    (TimeoutError('transport'), True),
    (gcs_exceptions.Unauthorized('auth'), False),
    (gcs_exceptions.BadRequest('bad request'), False),
    (gcs_exceptions.NotFound('missing'), False),
])
def test_gcs_retry_classification(error, expected):
    assert impl_gcs.is_retryable_gcs_error(error) is expected


def test_gcs_retry_policy_is_bounded(monkeypatch):
    attempts = 0

    @impl_gcs.gcs_retry_policy()
    def transient_read():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise gcs_exceptions.ServiceUnavailable('transient')
        return b'ok'

    sleep = mock.Mock()
    monkeypatch.setattr('io_utils.retry.time.sleep', sleep)
    assert transient_read() == b'ok'
    assert attempts == 3
    assert [entry.args[0] for entry in sleep.call_args_list] == [0.5, 1.0]


def test_gcs_retry_policy_does_not_retry_non_transient(monkeypatch):
    attempts = 0

    @impl_gcs.gcs_retry_policy()
    def malformed_read():
        nonlocal attempts
        attempts += 1
        raise gcs_exceptions.BadRequest('invalid')

    sleep = mock.Mock()
    monkeypatch.setattr('io_utils.retry.time.sleep', sleep)
    with pytest.raises(gcs_exceptions.BadRequest):
        malformed_read()
    assert attempts == 1
    sleep.assert_not_called()


def test_gcs_safe_read_retries_but_non_idempotent_write_does_not(monkeypatch):
    uri = GCSURI(bucket='customer-data', path='table.parquet')
    blob = mock.Mock(size=2)
    blob.reload.side_effect = [
        gcs_exceptions.ServiceUnavailable('transient'),
        gcs_exceptions.ServiceUnavailable('transient'),
        None,
    ]
    blob.download_as_bytes.return_value = b'ok'
    monkeypatch.setattr(impl_gcs, '_get_blob', lambda _uri: blob)
    sleep = mock.Mock()
    monkeypatch.setattr('io_utils.retry.time.sleep', sleep)

    assert impl_gcs.read_binary(uri, compression=False) == b'ok'
    assert blob.reload.call_count == 3
    assert [entry.args[0] for entry in sleep.call_args_list] == [0.5, 1.0]

    sleep.reset_mock()
    blob.upload_from_file.side_effect = gcs_exceptions.ServiceUnavailable(
        'transient')
    with pytest.raises(gcs_exceptions.ServiceUnavailable):
        impl_gcs.write_binary(uri, io.BytesIO(b'payload'), compression=False)
    blob.upload_from_file.assert_called_once()
    sleep.assert_not_called()


def test_gcs_copy_file_upload_streams_from_local(monkeypatch, tmp_path):
    payload = b'y' * 4096
    src = tmp_path / 'in.bin'
    src.write_bytes(payload)

    blob = mock.Mock()
    captured = {}

    def fake_upload_from_file(f, **_kwargs):
        captured['handle'] = f
        captured['data'] = f.read()

    blob.upload_from_file.side_effect = fake_upload_from_file
    monkeypatch.setattr(impl_gcs, '_get_blob', lambda _uri: blob)

    impl_gcs.copy_file(str(src), 'gs://bucket/dst.bin')

    assert captured['data'] == payload
    assert not isinstance(captured['handle'], (bytes, bytearray))
    assert hasattr(captured['handle'], 'read')


def test_gcs_copy_file_upload_retries_transient_error(monkeypatch, tmp_path):
    """`_upload_from_local` opens `local_path` itself (unlike `write_binary`,
    which receives a caller-owned handle), so a retry re-enters the function
    and reopens from byte 0 -- re-uploading the same complete object is safe.
    """
    payload = b'z' * 4096
    src = tmp_path / 'in.bin'
    src.write_bytes(payload)

    blob = mock.Mock()
    captured = []

    def fake_upload_from_file(f, **_kwargs):
        captured.append(f.read())
        if len(captured) < 3:
            raise gcs_exceptions.ServiceUnavailable('transient')

    blob.upload_from_file.side_effect = fake_upload_from_file
    monkeypatch.setattr(impl_gcs, '_get_blob', lambda _uri: blob)
    sleep = mock.Mock()
    monkeypatch.setattr('io_utils.retry.time.sleep', sleep)

    impl_gcs.copy_file(str(src), 'gs://bucket/dst.bin')

    assert captured == [payload, payload, payload]


@pytest.mark.parametrize(('binary', 'mode'), [(False, 'r'), (True, 'rb')])
def test_gcs_handles_bill_requester_project_on_bucket_request(
        monkeypatch, binary, mode):
    raw_client = mock.Mock()
    raw_client._credentials = object()
    bucket = mock.Mock()
    raw_client.bucket.return_value = bucket

    @contextlib.contextmanager
    def open_request(_uri, _mode, **kwargs):
        scoped_client = kwargs['transport_params']['client']
        assert scoped_client.bucket('requester-pays-bucket') is bucket
        yield io.BytesIO() if binary else io.StringIO()

    monkeypatch.setattr(impl_gcs.threadsafe_gcs, 'client', lambda: raw_client)
    monkeypatch.setattr(impl_gcs, 'smart_open', open_request)
    uri = GCSURI(bucket='requester-pays-bucket', path='table.csv')
    with credential_scope(
        {
            'provider': 'gcs',
            'credential': object(),
            'requester_pays_project': 'billing-project',
        }, require_explicit=True):
        handle = (fsutil.get_binary_handle(uri.as_gs(), mode)
                  if binary else fsutil.get_handle(uri.as_gs(), mode))
        with handle:
            pass

    raw_client.bucket.assert_called_once_with('requester-pays-bucket',
                                              user_project='billing-project')


@withMockedGCSFilesystem
def test_gcs_copydir(gcs_client):
    """Server-side ``copydir`` fast path: triggered by overwrite=True +
    raise_on_exists=False between two GCS paths.
    """
    bucket = gcs_client.create_bucket('copydir-bucket')
    bucket.blob('src_dir/a.txt').upload_from_string('hello')
    bucket.blob('src_dir/sub/b.txt').upload_from_string('world')

    fsutil.copy(
        'gs://copydir-bucket/src_dir',
        'gs://copydir-bucket/dst_dir',
        overwrite=True,
        raise_on_exists=False,
    )

    assert fsutil.exists('gs://copydir-bucket/dst_dir/a.txt')
    assert fsutil.exists('gs://copydir-bucket/dst_dir/sub/b.txt')
    assert fsutil.read_binary('gs://copydir-bucket/dst_dir/a.txt') == b'hello'
    assert fsutil.read_binary(
        'gs://copydir-bucket/dst_dir/sub/b.txt') == b'world'

    # Source must remain (copy, not move).
    assert fsutil.exists('gs://copydir-bucket/src_dir/a.txt')
    assert fsutil.exists('gs://copydir-bucket/src_dir/sub/b.txt')


@withMockedGCSFilesystem
def test_gcs_copy_file_server_side(gcs_client):
    """Single-file GCS->GCS copy should succeed via the server-side path."""
    bucket = gcs_client.create_bucket('single-copy-bucket')
    bucket.blob('a.txt').upload_from_string('payload')

    fsutil.copy_file(
        'gs://single-copy-bucket/a.txt',
        'gs://single-copy-bucket/b.txt',
    )
    assert fsutil.exists('gs://single-copy-bucket/b.txt')
    assert fsutil.read_binary('gs://single-copy-bucket/b.txt') == b'payload'


@withMockedGCSFilesystem
def test_gcs_contains_subdir_bucket_root(gcs_client):
    bucket = gcs_client.create_bucket('test_bucket')
    bucket.blob("root_dir/file.txt").upload_from_string("data")
    bucket.blob("root_file.txt").upload_from_string("data")

    assert fsutil.contains_subdir("gs://test_bucket")
    assert fsutil.contains_subdir("gs://test_bucket/")


@withMockedGCSFilesystem
def test_gcs_listdir_includes_prefixes(gcs_client):
    bucket = gcs_client.create_bucket('test_bucket')
    bucket.blob("dir_a/file.txt").upload_from_string("data")
    bucket.blob("dir_b/file.txt").upload_from_string("data")
    bucket.blob("root.txt").upload_from_string("data")

    entries = fsutil.listdir("gs://test_bucket")
    assert "gs://test_bucket/dir_a" in entries
    assert "gs://test_bucket/dir_b" in entries
    assert "gs://test_bucket/root.txt" in entries


class _ListingPage(list):
    def __init__(self, blobs, prefixes=()):
        super().__init__(blobs)
        self.prefixes = set(prefixes)


def test_gcs_listdir_detail_paginates_and_preserves_provider_types(monkeypatch):
    bucket = mock.Mock()
    bucket.list_blobs.return_value = SimpleNamespace(pages=[
        _ListingPage(
            [SimpleNamespace(name='events/part.parquet')],
            ['events/year=2026/'],
        ),
    ])
    monkeypatch.setattr(impl_gcs, '_get_bucket', lambda _uri: bucket)

    details = impl_gcs._listdir_details(
        GCSURI(bucket='customer-data', path='events'))

    assert [(entry.path, entry.is_directory) for entry in details] == [
        ('gs://customer-data/events/part.parquet', False),
        ('gs://customer-data/events/year=2026', True),
    ]
    bucket.list_blobs.assert_called_once_with(
        prefix='events/',
        delimiter='/',
        max_results=None,
    )


@withMockedGCSFilesystem
def test_gcs_zip_directory_supports_lifecycle_archive_path(gcs_client):
    bucket = gcs_client.create_bucket('archive-bucket')
    bucket.blob('outputs/part-0.csv').upload_from_string(b'zero')
    bucket.blob('outputs/part-1.csv').upload_from_string(b'one')
    archive = 'gs://archive-bucket/tmp_archives/job.zip'

    assert fsutil.zip_directory('gs://archive-bucket/outputs',
                                output_path=archive) == archive
    assert fsutil.exists(archive)
    assert not fsutil.exists('gs://archive-bucket/outputs.zip')


def test_copy_still_exposes_its_async_form():
    """Four production sites call `await fsutil.copy.coroutine(...)`, and a
    helper inserted between copy's decorators and its def once removed it
    while every unit test still passed."""
    assert hasattr(fsutil.copy, 'coroutine')


@pytest.mark.parametrize('error,attempts', [
    (ConnectionError('transport unavailable'), 3),
    (FileNotFoundError('missing'), 1),
    (PermissionError('denied'), 1),
    (OSError(errno.ENOSPC, 'full'), 1),
])
def test_download_retains_outer_retry_budget(monkeypatch, tmp_path, error,
                                             attempts):
    client = mock.Mock()
    client.download.side_effect = error
    monkeypatch.setattr(impl_gcs, '_download_client', lambda: client)
    monkeypatch.setattr('io_utils.retry.time.sleep', lambda _: None)
    with pytest.raises(type(error)):
        impl_gcs.copy_file('gs://bucket/object', str(tmp_path / 'output'))
    assert client.download.call_count == attempts


def test_sdk_download_releases_permit_before_retry(monkeypatch, tmp_path):
    import threading

    permit = threading.BoundedSemaphore(1)
    monkeypatch.setattr(download, '_gcs_download_semaphore', permit)
    client = download.GCSDownloader()
    blob = mock.Mock(size=7, content_encoding='gzip')
    calls = []

    def transfer(output):
        calls.append(output.name)
        assert not permit.acquire(blocking=False)
        if len(calls) == 1:
            raise gcs_exceptions.ServiceUnavailable('retry')
        output.write(b'payload')

    def backoff(_):
        assert permit.acquire(blocking=False)
        permit.release()

    blob.download_to_file.side_effect = transfer
    monkeypatch.setattr(client, '_blob', lambda bucket, key: blob)
    monkeypatch.setattr(impl_gcs, '_download_client', lambda: client)
    monkeypatch.setattr('io_utils.retry.time.sleep', backoff)
    target = tmp_path / 'output'
    impl_gcs.copy_file('gs://bucket/object', str(target))
    assert len(calls) == 2
    assert target.read_bytes() == b'payload'
    assert list(tmp_path.iterdir()) == [target]
    assert permit.acquire(blocking=False)
