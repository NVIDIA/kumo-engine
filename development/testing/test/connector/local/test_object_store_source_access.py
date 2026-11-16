# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from io_utils.fsutil.credentials import (
    ExplicitFilesystemCredentialsRequired,
    credential_scope,
)
from kumo.connector.object_store_access import source_access


@pytest.mark.parametrize('path', ['s3://bucket/key', 'gs://bucket/key'])
def test_customer_source_requires_matching_explicit_identity(path):
    with credential_scope({'provider': 'adls'}, require_explicit=True):
        with pytest.raises(ExplicitFilesystemCredentialsRequired):
            source_access(path)


@pytest.mark.parametrize('path,provider', [('s3://bucket/key', 's3'),
                                           ('gs://bucket/key', 'gcs')])
def test_incomplete_customer_identity_never_uses_ambient(
        path, provider, monkeypatch):
    ambient = Mock(side_effect=AssertionError('ambient fallback'))
    monkeypatch.setattr('google.auth.default', ambient)
    monkeypatch.setattr('boto3.session.Session', ambient)
    with credential_scope({'provider': provider}, require_explicit=True):
        with pytest.raises(RuntimeError, match='Missing explicit'):
            source_access(path)
    ambient.assert_not_called()


def test_s3_attested_source_cannot_become_static_credentials(monkeypatch):
    factory = Mock()
    monkeypatch.setattr('kumo_s3.access.source_access', factory)
    session = Mock()
    session.get_credentials.return_value.get_frozen_credentials.return_value.token = None
    with credential_scope({
            'provider': 's3',
            'ambient_session': session
    }, require_explicit=True):
        with pytest.raises(RuntimeError, match='no temporary credentials'):
            source_access('s3://customer-bucket/table')
    factory.assert_not_called()
