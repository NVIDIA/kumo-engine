# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError

from io_utils import fsutil
from io_utils.fsutil import impl_s3


@pytest.mark.parametrize(('code', 'operation'), [
    ('NoSuchKey', 'HeadObject'),
    ('404', 'HeadObject'),
    ('NoSuchBucket', 'ListObjectsV2'),
])
def test_exists_returns_false_only_for_missing_s3_resources(
        monkeypatch, code, operation):
    error = ClientError(
        {'Error': {
            'Code': code,
            'Message': 'not found'
        }},
        operation,
    )
    if operation == 'ListObjectsV2':
        monkeypatch.setattr(impl_s3, 'isdir', Mock(side_effect=error))
    else:
        monkeypatch.setattr(impl_s3, 'isdir', Mock(return_value=False))
        monkeypatch.setattr(
            impl_s3,
            '_read_with_unsigned_fallback',
            Mock(side_effect=error),
        )

    assert not fsutil.exists('s3://bucket/missing.json')


@pytest.mark.parametrize('code', ['AccessDenied', 'SlowDown'])
def test_exists_propagates_ambiguous_s3_client_errors(monkeypatch, code):
    error = ClientError(
        {'Error': {
            'Code': code,
            'Message': 'not proof of absence'
        }},
        'ListObjectsV2',
    )
    if code == 'AccessDenied':
        monkeypatch.setattr(impl_s3, 'isdir', Mock(return_value=False))
        monkeypatch.setattr(
            impl_s3,
            '_read_with_unsigned_fallback',
            Mock(side_effect=error),
        )
    else:
        monkeypatch.setattr(impl_s3, 'isdir', Mock(side_effect=error))

    with pytest.raises(ClientError) as exc_info:
        fsutil.exists('s3://bucket/db.json.gz')

    assert exc_info.value is error
