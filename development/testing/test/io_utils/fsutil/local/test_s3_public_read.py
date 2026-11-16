# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from botocore.exceptions import (
    ClientError,
    NoCredentialsError,
    PartialCredentialsError,
)

from io_utils import fsutil
from io_utils.fsutil import s3_filesystem


class _MissingCredentialsS3Client:
    def list_objects_v2(self, **kwargs):
        raise NoCredentialsError()

    def head_object(self, **kwargs):
        raise NoCredentialsError()

    def get_paginator(self, operation_name):
        raise NoCredentialsError()


class _PublicReadPaginator:
    def __init__(self, pages):
        self._pages = pages

    def paginate(self, **kwargs):
        yield from self._pages


class _PublicReadS3Client:
    def __init__(self):
        self.keys = [
            "card_fraud_sample/account_table.parquet",
            "card_fraud_sample/card_table.parquet",
            "card_fraud_sample/merch_table.parquet",
            "card_fraud_sample/transaction_table.parquet",
        ]

    def list_objects_v2(self, Bucket, Prefix, Delimiter=None, MaxKeys=None):
        matching_keys = [key for key in self.keys if key.startswith(Prefix)]
        if not matching_keys:
            return {"KeyCount": 0}

        contents = [{"Key": key} for key in matching_keys]
        if MaxKeys is not None:
            contents = contents[:MaxKeys]
        response = {"KeyCount": len(contents), "Contents": contents}
        if Delimiter == "/":
            response["CommonPrefixes"] = []
        return response

    def head_object(self, Bucket, Key):
        if Key not in self.keys:
            raise AssertionError(f"unexpected head_object key: {Key}")
        return {"ContentLength": 123}

    def get_paginator(self, operation_name):
        assert operation_name == "list_objects_v2"
        return _PublicReadPaginator([
            {
                "Contents": [{
                    "Key": key
                } for key in self.keys[:2]],
                "CommonPrefixes": [],
            },
            {
                "Contents": [{
                    "Key": key
                } for key in self.keys[2:]],
                "CommonPrefixes": [],
            },
        ])


class _FakeS3Context:
    _retry_delay_sec = 0

    def __init__(self):
        self._s3_client = _MissingCredentialsS3Client()


def _install_public_read_fallback(monkeypatch):
    from io_utils.fsutil import impl_s3

    fake_ctx = _FakeS3Context()
    unsigned_client = _PublicReadS3Client()
    monkeypatch.setattr(impl_s3, "s3_ctx", fake_ctx)
    monkeypatch.setattr(s3_filesystem, "s3_ctx", fake_ctx)
    monkeypatch.setattr(impl_s3, "_unsigned_s3_client", lambda: unsigned_client)
    monkeypatch.setattr(s3_filesystem, "_unsigned_s3_client",
                        lambda: unsigned_client)
    return unsigned_client


def test_s3_public_read_fallback_exists_and_limited_listdir(monkeypatch):
    _install_public_read_fallback(monkeypatch)
    root = "s3://public-bucket/card_fraud_sample/"

    assert fsutil.exists(root)
    assert fsutil.listdir(root, max_items=2) == [
        "s3://public-bucket/card_fraud_sample/account_table.parquet",
        "s3://public-bucket/card_fraud_sample/card_table.parquet",
    ]


def test_s3_public_read_fallback_file_head_and_paginated_listdir(monkeypatch):
    _install_public_read_fallback(monkeypatch)
    root = "s3://public-bucket/card_fraud_sample/"
    file_path = "s3://public-bucket/card_fraud_sample/account_table.parquet"

    assert fsutil.listdir(root) == [
        "s3://public-bucket/card_fraud_sample/account_table.parquet",
        "s3://public-bucket/card_fraud_sample/card_table.parquet",
        "s3://public-bucket/card_fraud_sample/merch_table.parquet",
        "s3://public-bucket/card_fraud_sample/transaction_table.parquet",
    ]
    assert fsutil.exists(file_path)
    assert fsutil.filesize(file_path) == 123


@pytest.mark.parametrize('scoped', [False, True])
@pytest.mark.parametrize('failure', [
    NoCredentialsError(),
    PartialCredentialsError(provider='test', cred_var='secret'),
    ClientError({'Error': {
        'Code': 'AccessDenied'
    }}, 'GetObject'),
])
def test_download_anonymous_fallback_preserves_credential_boundary(
        monkeypatch, tmp_path, scoped, failure):
    signed = Mock()
    signed.download_file.side_effect = failure
    unsigned = Mock()
    unsigned.download_file.side_effect = lambda bucket, key, target: Path(
        target).write_bytes(b'public')
    monkeypatch.setattr(s3_filesystem, 's3_ctx',
                        SimpleNamespace(_s3_client=signed))
    unsigned_factory = Mock(return_value=unsigned)
    monkeypatch.setattr(s3_filesystem, '_unsigned_s3_client', unsigned_factory)
    monkeypatch.setattr(s3_filesystem, 'credentials_for_provider',
                        lambda provider: object() if scoped else None)
    target = tmp_path / 'output'
    if scoped or isinstance(failure, ClientError):
        with pytest.raises(type(failure)) as caught:
            s3_filesystem.download_file('s3://bucket/key', str(target))
        assert caught.value is failure
        unsigned_factory.assert_not_called()
    else:
        s3_filesystem.download_file('s3://bucket/key', str(target))
        assert target.read_bytes() == b'public'
        unsigned.download_file.assert_called_once_with('bucket', 'key',
                                                       str(target))
    signed.download_file.assert_called_once_with('bucket', 'key', str(target))
