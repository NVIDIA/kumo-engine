# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError
from google.api_core import exceptions as gcs_exceptions

from io_utils import fsutil
from io_utils.fsutil import impl_adls, impl_gcs, impl_local, impl_s3, impl_ucv
from io_utils.fsutil.uri import ADLSURI, GCSURI, S3URI, UCURI


def test_local_write_text_if_absent_never_replaces(tmp_path):
    path = str(tmp_path / 'claim.json')
    assert fsutil.write_text_if_absent(path, 'first')
    assert not fsutil.write_text_if_absent(path, 'second')
    assert fsutil.read_text(path) == 'first'


def test_local_write_text_if_absent_is_invisible_until_complete(
        monkeypatch, tmp_path):
    path = tmp_path / 'claim.json'
    payload = 'complete-publisher-identity' * 4096
    entered_link = threading.Event()
    release_link = threading.Event()
    real_link = impl_local.os.link
    result = []

    def blocking_link(source, destination):
        with open(source) as handle:
            assert handle.read() == payload
        entered_link.set()
        assert release_link.wait(timeout=10)
        real_link(source, destination)

    monkeypatch.setattr(impl_local.os, 'link', blocking_link)
    writer = threading.Thread(target=lambda: result.append(
        fsutil.write_text_if_absent(str(path), payload)))
    writer.start()
    assert entered_link.wait(timeout=10)

    assert not path.exists()
    assert list(tmp_path.glob('.claim.json.*.tmp'))

    release_link.set()
    writer.join(timeout=10)
    assert not writer.is_alive()
    assert result == [True]
    assert path.read_text() == payload
    assert not list(tmp_path.glob('.claim.json.*.tmp'))


def test_local_write_text_if_absent_link_failure_leaves_no_claim_or_temp(
        monkeypatch, tmp_path):
    path = tmp_path / 'claim.json'

    def fail_link(_source, _destination):
        raise OSError('injected publication failure')

    monkeypatch.setattr(impl_local.os, 'link', fail_link)
    with pytest.raises(OSError, match='injected publication failure'):
        fsutil.write_text_if_absent(str(path), 'complete')

    assert not path.exists()
    assert not list(tmp_path.glob('.claim.json.*.tmp'))


def test_s3_write_text_if_absent_uses_put_precondition(monkeypatch):
    client = Mock()
    monkeypatch.setattr(impl_s3.threadsafe_boto, 's3_client', lambda: client)
    monkeypatch.setattr(impl_s3, 'bucket_owner_write_args', lambda: {})
    uri = S3URI(bucket='bucket', path='claim.json')

    assert impl_s3.write_text_if_absent(uri, 'first')
    client.put_object.assert_called_once_with(
        Bucket='bucket',
        Key='claim.json',
        Body=b'first',
        IfNoneMatch='*',
    )

    client.put_object.side_effect = ClientError(
        {
            'Error': {
                'Code': 'PreconditionFailed'
            },
            'ResponseMetadata': {
                'HTTPStatusCode': 412
            },
        },
        'PutObject',
    )
    assert not impl_s3.write_text_if_absent(uri, 'second')


def test_gcs_write_text_if_absent_uses_generation_precondition(monkeypatch):
    blob = Mock()
    monkeypatch.setattr(impl_gcs, '_get_blob', lambda _uri: blob)
    uri = GCSURI(bucket='bucket', path='claim.json')

    assert impl_gcs.write_text_if_absent(uri, 'first')
    blob.upload_from_string.assert_called_once_with(b'first',
                                                    if_generation_match=0)

    blob.upload_from_string.side_effect = gcs_exceptions.PreconditionFailed(
        'already exists')
    assert not impl_gcs.write_text_if_absent(uri, 'second')


def _adls_uri() -> ADLSURI:
    return ADLSURI(
        storage_account='account',
        endpoint_suffix='dfs.core.windows.net',
        container='container',
        path='claim.json',
        scheme='abfss',
    )


def test_adls_write_text_if_absent_conditionally_promotes_complete_temp(
        monkeypatch):
    client = Mock()
    client.get_file_properties.return_value = SimpleNamespace(etag='temp-etag')
    monkeypatch.setattr(impl_adls.adls_ctx, 'get_file_client',
                        lambda _uri: client)
    operation = SimpleNamespace(hex='operation-id')
    monkeypatch.setattr(impl_adls.uuid, 'uuid4', lambda: operation)
    uri = _adls_uri()

    assert impl_adls.write_text_if_absent(uri, 'first')

    client.upload_data.assert_called_once_with(
        b'first',
        overwrite=True,
        metadata={impl_adls._UPLOAD_METADATA_KEY: operation.hex},
        match_condition=impl_adls.MatchConditions.IfMissing,
    )
    client.rename_file.assert_called_once_with(
        'container/claim.json',
        source_etag='temp-etag',
        source_match_condition=impl_adls.MatchConditions.IfNotModified,
        match_condition=impl_adls.MatchConditions.IfMissing,
    )
    client.delete_file.assert_called_once_with()


def test_adls_write_text_if_absent_preserves_existing_destination(
        monkeypatch):
    client = Mock()
    client.get_file_properties.return_value = SimpleNamespace(etag='temp-etag')
    client.rename_file.side_effect = impl_adls.ResourceExistsError('exists')
    monkeypatch.setattr(impl_adls.adls_ctx, 'get_file_client',
                        lambda _uri: client)
    operation = SimpleNamespace(hex='operation-id')
    monkeypatch.setattr(impl_adls.uuid, 'uuid4', lambda: operation)

    assert not impl_adls.write_text_if_absent(_adls_uri(), 'second')
    client.delete_file.assert_called_once_with()


def test_adls_write_text_if_absent_supports_empty_payload(monkeypatch):
    client = Mock()
    client.get_file_properties.return_value = SimpleNamespace(etag='temp-etag')
    monkeypatch.setattr(impl_adls.adls_ctx, 'get_file_client',
                        lambda _uri: client)
    operation = SimpleNamespace(hex='operation-id')
    monkeypatch.setattr(impl_adls.uuid, 'uuid4', lambda: operation)

    assert impl_adls.write_text_if_absent(_adls_uri(), '')
    client.upload_data.assert_not_called()
    client.create_file.assert_called_once_with(
        metadata={impl_adls._UPLOAD_METADATA_KEY: operation.hex},
        match_condition=impl_adls.MatchConditions.IfMissing,
    )


def test_ucv_write_text_if_absent_disables_overwrite(monkeypatch):
    client = Mock()
    monkeypatch.setattr(
        type(impl_ucv.uc_ctx),
        'client',
        property(lambda _self: client),
    )
    uri = UCURI(catalog='catalog', uc_schema='schema', volume='volume',
                path='claim.json')

    assert impl_ucv.write_text_if_absent(uri, 'first')
    args, kwargs = client.files.upload.call_args
    assert args[0].endswith('/claim.json')
    assert args[1].read() == b'first'
    assert kwargs == {'overwrite': False}

    from databricks.sdk.errors.platform import AlreadyExists
    client.files.upload.side_effect = AlreadyExists('exists')
    assert not impl_ucv.write_text_if_absent(uri, 'second')


def test_ucv_object_identity_uses_closed_object_metadata(monkeypatch):
    metadata = Mock(content_length=1234, last_modified='version-17')
    client = Mock()
    client.files.get_metadata.return_value = metadata
    monkeypatch.setattr(
        type(impl_ucv.uc_ctx),
        'client',
        property(lambda _self: client),
    )
    uri = UCURI(catalog='catalog', uc_schema='schema', volume='volume',
                path='part-00000.parquet')

    identity = impl_ucv.object_identity(uri)

    assert identity.size == 1234
    assert identity.etag is None
    assert identity.version == 'version-17'
    client.files.get_metadata.assert_called_once_with(uri.as_path())
