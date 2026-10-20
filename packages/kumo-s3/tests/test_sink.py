# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import kumo_s3
import pytest
from fakes import FakeSession
from kumo_s3 import backend
from kumo_s3.sink import S3SinkBackend
from kumoapi.data_source import (
    S3AnyURIConnectorTarget,
    S3ConnectorResourceConfig,
)


@pytest.mark.parametrize('credentials', ['default', None])
def test_parallel_uploads_reach_every_filesystem_construction(
        monkeypatch, credentials):
    built = []

    class Record:
        def __init__(self, **kwargs):
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
    connector = SimpleNamespace(
        id='connector', identity=None, identity_id=None,
        config=S3ConnectorResourceConfig(target=S3AnyURIConnectorTarget()))

    filesystem, path = S3SinkBackend().filesystem(
        connector, 's3://bucket/prefix/', parallel_uploads=True)

    assert isinstance(filesystem, Record)
    assert path == 'bucket/prefix'
    assert built[-1]['allow_delayed_open'] is True
    assert built[-1]['background_writes'] is False
    assert 'retry_strategy' in built[-1]
