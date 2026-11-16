# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for io_utils.pyarrow credentialed cloud-filesystem resolution.

On the py3.14 / pyarrow-24 stack pyarrow's native S3 credential resolution
(``FileSystem.from_uri``) is not fork-safe for the STS web-identity (IRSA)
provider: in a ``multiprocessing`` fork child it resolves anonymously and
cross-account / credentialed reads 403. ``arrow_cloud_filesystem`` now resolves
credentials with boto3 (pure-Python, fork-safe) and injects them explicitly,
falling back to the native chain when no credentials resolve.
"""
import types
from unittest import mock

import kumo_s3
import pyarrow as pa
import pytest

import io_utils.pyarrow as pyu
from io_utils.fsutil import s3_connection
from io_utils.fsutil.s3_connection import S3Connection
from service_framework._raw_deploy_config import S3AddressingStyle


@pytest.fixture(autouse=True)
def _clear_fs_caches():
    """Clear both filesystem caches around every test.

    The filesystem cache (TTL) and the immutable per-bucket region cache
    would otherwise leak entries across tests and make results depend on the
    order tests happen to run in.
    """
    pyu._arrow_cloud_filesystem.cache_clear()
    kumo_s3.bucket_region.cache_clear()
    yield
    pyu._arrow_cloud_filesystem.cache_clear()
    kumo_s3.bucket_region.cache_clear()


def _fake_frozen(access_key='AKIA', secret_key='secret', token='tok'):
    return types.SimpleNamespace(access_key=access_key, secret_key=secret_key,
                                 token=token)


def _patch_boto(monkeypatch, *, credentials, region='us-west-2',
                bucket_region='us-west-2', endpoint_url=None):
    """Patch the boto3 session stack that _resolve_s3_filesystem uses."""
    session = mock.MagicMock()
    session.region_name = region
    session.get_credentials.return_value = credentials

    fake_tb = mock.MagicMock()
    fake_tb.session.return_value = session
    # Default to no custom endpoint (the SaaS / singlehost case). Stage
    # vending sets a real endpoint string; default it to a concrete value so it
    # is not an auto-truthy MagicMock that would leak into S3FileSystem.
    fake_tb._endpoint_url = endpoint_url
    monkeypatch.setattr('io_utils.fsutil.s3_filesystem.threadsafe_boto',
                        fake_tb)
    # Region now comes from pyarrow's own resolver (unsigned, 403/301-robust),
    # not a boto3 head_bucket.
    monkeypatch.setattr(pyu.fs, 'resolve_s3_region',
                        lambda bucket: bucket_region)
    return fake_tb


def test_resolve_s3_filesystem_uses_explicit_boto3_credentials(monkeypatch):
    # With boto3 credentials resolved, we build an explicit-credential
    # S3FileSystem (NOT the native default-chain one).
    creds = mock.MagicMock()
    creds.get_frozen_credentials.return_value = _fake_frozen()
    fake_tb = _patch_boto(monkeypatch, credentials=creds,
                          bucket_region='eu-west-1')

    captured = {}

    def fake_s3fs(**kwargs):
        captured.update(kwargs)
        return 'EXPLICIT_FS'

    monkeypatch.setattr(pyu.fs, 'S3FileSystem', fake_s3fs)

    out = pyu._resolve_s3_filesystem('s3://some-bucket')

    assert out == 'EXPLICIT_FS'
    # lazy_auth must run before credentials are read (mirrors boto3 read path).
    fake_tb.lazy_auth.assert_called_once()
    assert captured['access_key'] == 'AKIA'
    assert captured['secret_key'] == 'secret'
    # The session token MUST be threaded through -- IRSA creds are temporary.
    assert captured['session_token'] == 'tok'
    # Region preference: the bucket's actual region (resolve_s3_region) wins so
    # a cross-region bucket is not pinned wrong.
    assert captured['region'] == 'eu-west-1'
    # No custom endpoint in the default (SaaS/singlehost) case -> pyarrow uses
    # the default AWS endpoint with path/virtual autodetection, unchanged.
    assert 'endpoint_override' not in captured
    assert 'force_virtual_addressing' not in captured


def test_resolve_s3_filesystem_falls_back_to_session_region(monkeypatch):
    # When bucket-region resolution fails, fall back to the session region so we
    # still build a valid credentialed FS instead of dropping to from_uri.
    creds = mock.MagicMock()
    creds.get_frozen_credentials.return_value = _fake_frozen()
    _patch_boto(monkeypatch, credentials=creds, region='ap-south-1')
    monkeypatch.setattr(pyu.fs, 'resolve_s3_region',
                        mock.MagicMock(side_effect=ValueError('no region')))

    captured = {}
    monkeypatch.setattr(pyu.fs, 'S3FileSystem',
                        lambda **kw: captured.update(kw) or 'FS')

    assert pyu._resolve_s3_filesystem('s3://bucket') == 'FS'
    assert captured['region'] == 'ap-south-1'


def test_resolve_s3_filesystem_threads_stage_endpoint_override(monkeypatch):
    # Stage vending points boto3 at a non-AWS endpoint
    # (threadsafe_boto._endpoint_url). The credentialed pyarrow FS must target
    # the SAME host and force virtual addressing to match the boto3 client, or
    # the stage reads would break. The stage region comes from the session, not
    # from an AWS query.
    creds = mock.MagicMock()
    creds.get_frozen_credentials.return_value = _fake_frozen()
    _patch_boto(monkeypatch, credentials=creds, region='ca-central-1',
                endpoint_url='https://acct.snowflakecomputing.com')
    # resolve_s3_region queries AWS S3; it must NOT be consulted for a non-AWS
    # stage host. Use a plain spy and assert it stays untouched -- an
    # AssertionError side_effect would now be swallowed by the caller's except.
    resolver = mock.MagicMock()
    monkeypatch.setattr(pyu.fs, 'resolve_s3_region', resolver)

    captured = {}
    monkeypatch.setattr(pyu.fs, 'S3FileSystem',
                        lambda **kw: captured.update(kw) or 'FS')

    assert pyu._resolve_s3_filesystem('s3://stage-bucket') == 'FS'
    assert (
        captured['endpoint_override'] == 'https://acct.snowflakecomputing.com')
    assert captured['force_virtual_addressing'] is True
    # Region is the stage region from the session (used for SigV4 signing).
    assert captured['region'] == 'ca-central-1'
    # Credentials are still threaded through alongside the endpoint.
    assert captured['access_key'] == 'AKIA'
    assert captured['session_token'] == 'tok'
    # The stage path must never query AWS S3 for the bucket region.
    resolver.assert_not_called()


def test_resolve_s3_filesystem_threads_deployment_minio_settings(monkeypatch):
    connection = S3Connection(
        endpoint_url='https://minio.example.test:9000',
        region='us-east-1',
        addressing_style=S3AddressingStyle.PATH,
        credential_secret_name='MINIO_KUMO_CREDENTIALS',
        access_key='long-lived-access',
        secret_key='long-lived-secret',
        session_token=None,
        ca_bundle_path='/etc/kumo/minio-ca/ca.crt',
        send_bucket_owner_full_control=False,
    )
    monkeypatch.setattr(s3_connection, 'active_s3_connection',
                        lambda: connection)
    captured = {}
    monkeypatch.setattr(pyu.fs, 'S3FileSystem',
                        lambda **kw: captured.update(kw) or 'MINIO_FS')

    assert pyu._resolve_s3_filesystem('s3://example-dataplane') == 'MINIO_FS'
    expected = connection.pyarrow_options()
    captured.pop('retry_strategy')
    expected.pop('retry_strategy')
    assert captured == expected


def test_resolve_s3_filesystem_returns_none_without_credentials(monkeypatch):
    # No credentials -> None so the caller falls back to FileSystem.from_uri
    # (anonymous / public reads -- the common singlehost path is unchanged).
    _patch_boto(monkeypatch, credentials=None)
    monkeypatch.setattr(
        pyu.fs, 'S3FileSystem',
        mock.MagicMock(side_effect=AssertionError(
            'must not build an explicit FS without creds')))
    assert pyu._resolve_s3_filesystem('s3://b') is None


def test_resolve_s3_filesystem_returns_none_when_keys_blank(monkeypatch):
    # A resolvable but empty credential set (blank keys) must not build a
    # credentialed FS -- fall back to from_uri instead.
    creds = mock.MagicMock()
    creds.get_frozen_credentials.return_value = _fake_frozen(
        access_key='', secret_key='', token=None)
    _patch_boto(monkeypatch, credentials=creds)
    monkeypatch.setattr(
        pyu.fs, 'S3FileSystem',
        mock.MagicMock(side_effect=AssertionError(
            'must not build an explicit FS with blank keys')))
    assert pyu._resolve_s3_filesystem('s3://b') is None


class _FakeFs:
    """Stand-in for the immutable pyarrow.fs module (from_uri is a C method
    that cannot be monkeypatched in place), used to observe routing."""
    def __init__(self, from_uri_result=None, from_uri_exc=None):
        self.from_uri_calls = []
        self._result = from_uri_result
        self._exc = from_uri_exc

        class FileSystem:
            @staticmethod
            def from_uri(path):
                self.from_uri_calls.append(path)
                if self._exc is not None:
                    raise self._exc
                return self._result

        self.FileSystem = FileSystem


def test_arrow_cloud_filesystem_s3_prefers_credentialed(monkeypatch):
    monkeypatch.setattr(pyu, '_resolve_s3_filesystem',
                        lambda path: 'CREDENTIALED')
    fake_fs = _FakeFs(from_uri_result=('NATIVE', 'bucket'))
    monkeypatch.setattr(pyu, 'fs', fake_fs)
    assert pyu.arrow_cloud_filesystem('s3://bucket/key') == 'CREDENTIALED'
    # from_uri must NOT be consulted when the credentialed FS resolves.
    assert fake_fs.from_uri_calls == []


def test_arrow_cloud_filesystem_s3_falls_back_to_native(monkeypatch):
    # When no credentialed FS is available, fall back to the native chain so
    # public / singlehost reads are unaffected.
    monkeypatch.setattr(pyu, '_resolve_s3_filesystem', lambda path: None)
    sentinel = object()
    fake_fs = _FakeFs(from_uri_result=(sentinel, 'bucket'))
    monkeypatch.setattr(pyu, 'fs', fake_fs)
    assert pyu.arrow_cloud_filesystem('s3://bucket/key') is sentinel
    assert fake_fs.from_uri_calls  # native chain WAS consulted


def test_arrow_cloud_filesystem_s3_falls_back_when_resolver_raises(
        monkeypatch, caplog):
    # A resolver exception must not break reads: swallow and fall back to the
    # native chain.
    canary = 's3://restricted-bucket/password=credential-secret'
    monkeypatch.setattr(
        pyu, '_resolve_s3_filesystem',
        mock.MagicMock(side_effect=RuntimeError(canary)))
    sentinel = object()
    fake_fs = _FakeFs(from_uri_result=(sentinel, 'bucket'))
    monkeypatch.setattr(pyu, 'fs', fake_fs)
    with caplog.at_level('DEBUG', logger='io_utils.pyarrow'):
        assert pyu.arrow_cloud_filesystem(canary) is sentinel
    assert fake_fs.from_uri_calls
    warning_text = ' '.join(record.getMessage()
                            for record in caplog.records
                            if record.levelname == 'WARNING')
    assert canary not in warning_text
    assert canary in caplog.text


def test_arrow_cloud_filesystem_minio_never_falls_back_to_aws(monkeypatch):
    connection = mock.Mock(identity=('minio', ))
    monkeypatch.setattr(s3_connection, 'active_s3_connection',
                        lambda: connection)
    monkeypatch.setattr(
        pyu, '_resolve_s3_filesystem',
        mock.MagicMock(side_effect=RuntimeError('minio setup failed')))
    fake_fs = _FakeFs(from_uri_result=('AWS_NATIVE', 'bucket'))
    monkeypatch.setattr(pyu, 'fs', fake_fs)

    with pytest.raises(RuntimeError, match='minio setup failed'):
        pyu._arrow_cloud_filesystem('s3://bucket', connection.identity)

    assert fake_fs.from_uri_calls == []


def test_arrow_cloud_filesystem_gcs_uses_native(monkeypatch):
    # GCS paths must never touch the S3 credential resolver -- they keep using
    # from_uri unchanged.
    monkeypatch.setattr(
        pyu, '_resolve_s3_filesystem',
        mock.MagicMock(side_effect=AssertionError(
            'GCS must not resolve S3 credentials')))
    sentinel = object()
    fake_fs = _FakeFs(from_uri_result=(sentinel, 'bucket'))
    monkeypatch.setattr(pyu, 'fs', fake_fs)
    assert pyu.arrow_cloud_filesystem('gs://bucket/key') is sentinel
    assert fake_fs.from_uri_calls


def test_resolve_s3_filesystem_builds_real_pyarrow_s3fs(monkeypatch):
    # Sanity: the kwargs we pass are accepted by the real pyarrow S3FileSystem
    # constructor (guards against an API drift in the credential kwargs).
    # resolve_s3_region is patched so the constructor needs no network.
    creds = mock.MagicMock()
    creds.get_frozen_credentials.return_value = _fake_frozen(
        access_key='AKIAEXAMPLE', secret_key='shh', token='session-token')
    _patch_boto(monkeypatch, credentials=creds, region='us-west-2',
                bucket_region='us-west-2')
    out = pyu._resolve_s3_filesystem('s3://bucket')
    assert isinstance(out, pa.fs.S3FileSystem)
    assert out.region == 'us-west-2'
