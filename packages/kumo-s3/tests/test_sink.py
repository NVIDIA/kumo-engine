# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import kumo_s3
import pytest
from fakes import FakeSession
from kumo_s3 import backend, sink
from kumo_s3.sink import S3SinkBackend
from kumoapi.data_source import (
    S3AnyURIConnectorTarget,
    S3ConnectorResourceConfig,
)


@pytest.mark.parametrize('delayed_open', [True, False])
@pytest.mark.parametrize('credentials', ['default', None])
def test_parallel_uploads_reach_every_filesystem_construction(
        monkeypatch, credentials, delayed_open):
    known = {
        'access_key', 'secret_key', 'session_token', 'region', 'anonymous',
        'retry_strategy', 'background_writes', 'endpoint_override', 'scheme',
        'force_virtual_addressing'
    } | ({'allow_delayed_open'} if delayed_open else set())
    built = []

    class Record:
        def __init__(self, **kwargs):
            if set(kwargs) - known:
                raise TypeError(f'unexpected {set(kwargs) - known}')
            built.append(kwargs)

    monkeypatch.setattr('pyarrow.fs.S3FileSystem', Record)
    monkeypatch.setattr(backend, 'S3FileSystem', Record)
    monkeypatch.setattr(kumo_s3, 'ambient_session',
                        lambda: FakeSession(credentials=credentials))
    monkeypatch.setattr('kumo_s3.session.bucket_region', lambda bucket: 'x')
    monkeypatch.setattr(
        'pyarrow.fs.FileSystem',
        SimpleNamespace(from_uri=lambda uri:
                        (SimpleNamespace(region='us-west-2'), '')))
    sink._parallel_upload_options.cache_clear()
    connector = SimpleNamespace(
        id='connector', identity=None, identity_id=None,
        config=S3ConnectorResourceConfig(target=S3AnyURIConnectorTarget()))

    filesystem, path = S3SinkBackend().filesystem(
        connector, 's3://bucket/prefix/', parallel_uploads=True)
    sink._parallel_upload_options.cache_clear()

    assert isinstance(filesystem, Record)
    assert path == 'bucket/prefix'
    assert built[-1].get('allow_delayed_open') is (True if delayed_open else
                                                   None)
    assert built[-1]['background_writes'] is False
    assert 'retry_strategy' in built[-1]
