# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import kumo_s3
import pytest
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import NoCredentialsError
from kumo_core.sources.errors import ReconnectRequiredError
from kumo_s3.access import source_access
from pyarrow.fs import AwsDefaultS3RetryStrategy


@pytest.fixture
def filesystem(monkeypatch):
    result = SimpleNamespace(region='us-west-2')
    monkeypatch.setattr(kumo_s3, 'filesystem', lambda *args, **kwargs: result)
    return result


@pytest.mark.parametrize(
    'failure',
    [OSError('interrupted'), NoCredentialsError()])
def test_failed_transfer_preserves_target_without_unsigned_retry(
        filesystem, tmp_path, failure):
    def download(bucket, key, target, **options):
        Path(target).write_bytes(b'partial')
        raise failure

    session = Mock()
    session.client.return_value.download_file.side_effect = download
    access = source_access('bucket', session=session)
    target = tmp_path / 'output'
    target.write_bytes(b'previous')
    with pytest.raises(ReconnectRequiredError if isinstance(
            failure, NoCredentialsError) else OSError):
        access.fetch('bucket/key', str(target))
    assert target.read_bytes() == b'previous'
    assert list(tmp_path.iterdir()) == [target]
    session.client.assert_called_once()
    assert session.client.call_args.kwargs[
        'config'].signature_version != UNSIGNED


@pytest.mark.parametrize('config',
                         [None, Config(s3={'addressing_style': 'path'})])
def test_connection_options_preserve_transport_and_authorized_credentials(
        monkeypatch, tmp_path, config):
    from fakes import FakeSession

    built = {}
    clients = []

    def filesystem(**kwargs):
        built.update(kwargs)
        return SimpleNamespace(region=kwargs['region'])

    def client(service, **kwargs):
        clients.append(kwargs)
        return SimpleNamespace(download_file=lambda bucket, key, target, **
                               options: Path(target).write_bytes(b'data'))

    monkeypatch.setattr('kumo_s3.session.fs.S3FileSystem', filesystem)
    session = FakeSession(region_name='us-west-2')
    monkeypatch.setattr(session, 'client', client, raising=False)
    options = {
        'access_key': 'unrelated-key',
        'secret_key': 'unrelated-secret',
        'session_token': 'unrelated-token',
        'region': 'unrelated-region',
        'anonymous': True,
        'endpoint_override': 's3.private',
        'scheme': 'https',
        'tls_ca_file_path': '/trusted/ca.pem',
        'force_virtual_addressing': False,
    }
    access = source_access('bucket', session=session,
                           endpoint_url='https://s3.private', options=options,
                           verify='/trusted/ca.pem', client_config=config)
    access.fetch('bucket/key', str(tmp_path / 'output'))
    retry = built.pop('retry_strategy')
    assert isinstance(retry, AwsDefaultS3RetryStrategy)
    assert built == {
        'access_key': 'key',
        'secret_key': 'secret',
        'session_token': 'token',
        'region': 'us-west-2',
        'endpoint_override': 's3.private',
        'scheme': 'https',
        'tls_ca_file_path': '/trusted/ca.pem',
        'force_virtual_addressing': False,
    }
    assert clients[0]['endpoint_url'] == 'https://s3.private'
    assert clients[0]['region_name'] == 'us-west-2'
    assert clients[0]['verify'] == '/trusted/ca.pem'
    assert clients[0]['config'].signature_version != UNSIGNED
    assert clients[0]['config'].s3 == {'addressing_style': 'path'}
    if config is not None:
        assert clients[0]['config'] is config
    assert not {
        'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token'
    } & clients[0].keys()


@pytest.mark.parametrize('anonymous, options', [
    (False, {}),
    (True, {}),
    (False, {
        'tls_ca_file_path': '/trusted/ca.pem'
    }),
    (True, {
        'endpoint_override': 'https://override.private'
    }),
])
def test_metadata_and_download_share_effective_endpoint(monkeypatch, tmp_path,
                                                        anonymous, options):
    from fakes import FakeSession

    built = {}
    clients = []

    def filesystem(**kwargs):
        built.update(kwargs)
        return SimpleNamespace(region=kwargs['region'])

    def client(service, **kwargs):
        clients.append(kwargs)
        return SimpleNamespace(download_file=lambda bucket, key, target, **
                               options: Path(target).write_bytes(b'data'))

    monkeypatch.setattr('kumo_s3.session.fs.S3FileSystem', filesystem)
    monkeypatch.setattr(kumo_s3, 'bucket_region',
                        Mock(side_effect=AssertionError('AWS region lookup')))
    session = FakeSession(region_name='us-west-2',
                          credentials=None if anonymous else 'default')
    monkeypatch.setattr(session, 'client', client, raising=False)
    access = source_access('bucket', session=session,
                           endpoint_url='https://default.private',
                           options=options, allow_anonymous=anonymous)
    access.fetch('bucket/key', str(tmp_path / 'output'))
    expected = options.get('endpoint_override', 'https://default.private')
    assert built['endpoint_override'] == expected
    assert clients[0]['endpoint_url'] == expected
    assert built['force_virtual_addressing'] is True
    if 'tls_ca_file_path' in options:
        assert built['tls_ca_file_path'] == options['tls_ca_file_path']


def test_ambient_download_fallback_keeps_serial_transfer_config(
        filesystem, tmp_path):
    signed = Mock()
    signed.download_file.side_effect = NoCredentialsError()
    unsigned = Mock()
    unsigned.download_file.side_effect = lambda bucket, key, target, **kwargs: Path(
        target).write_bytes(b'public')
    session = Mock()
    session.client.side_effect = [signed, unsigned]
    access = source_access('bucket', session=session, allow_anonymous=True)
    target = tmp_path / 'output'
    access.fetch('bucket/key', str(target))
    assert target.read_bytes() == b'public'
    assert list(tmp_path.iterdir()) == [target]
    transfer = signed.download_file.call_args.kwargs['Config']
    assert transfer.use_threads is False
    assert transfer.preferred_transfer_client == 'classic'
    assert unsigned.download_file.call_args.kwargs['Config'] is transfer
    assert session.client.call_args.kwargs[
        'config'].signature_version == UNSIGNED


@pytest.mark.parametrize('error_name', [
    'EndpointConnectionError',
    'ConnectTimeoutError',
    'ReadTimeoutError',
    'ConnectionClosedError',
])
def test_network_download_failure_remains_retryable(filesystem, tmp_path,
                                                    error_name):
    from botocore import exceptions
    from kumo_core.sources.errors import TransientSourceError

    session = Mock()
    session.client.return_value.download_file.side_effect = getattr(
        exceptions, error_name)(endpoint_url='https://s3.invalid')
    access = source_access('bucket', session=session)
    with pytest.raises(TransientSourceError):
        access.fetch('bucket/key', str(tmp_path / 'output'))
    assert list(tmp_path.iterdir()) == []
    session.client.assert_called_once()
