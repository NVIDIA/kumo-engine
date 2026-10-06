# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from botocore import UNSIGNED
from kumoapi.data_source import S3ConnectionConfig
from pyarrow.fs import LocalFileSystem, SubTreeFileSystem

from kumo.engine.connector.identity import S3Identity
from kumo.engine.connector.s3 import S3Connector


@pytest.mark.parametrize('endpoint, configured_region, root_region', [
    (None, None, None),
    (None, 'eu-west-1', None),
    (None, None, 'eu-west-1'),
    ('https://s3.private', 'eu-west-1', None),
])
def test_identity_connection_reaches_discovery_acquisition_and_sink(
        monkeypatch, tmp_path, endpoint, configured_region, root_region):
    storage = tmp_path / 'storage'
    (storage / 'bucket').mkdir(parents=True)
    pq.write_table(pa.table({'id': [1, 2]}), storage / 'bucket/events.parquet')
    built = []
    clients = []
    region = configured_region or root_region or ('eu-west-3'
                                                  if not endpoint else None)

    class Filesystem(SubTreeFileSystem):
        def __init__(self, **kwargs):
            built.append(kwargs)
            super().__init__(str(storage), LocalFileSystem())

        @property
        def region(self):
            return region

    def client(service, **kwargs):
        assert service == 's3'
        clients.append(kwargs)
        return SimpleNamespace(
            download_file=lambda bucket, key, target, **kw: Path(
                target).write_bytes((storage / bucket / key).read_bytes()))

    session = Mock(region_name=None)
    session.get_credentials.return_value = None
    session.client.side_effect = client
    monkeypatch.setattr(boto3, 'Session', Mock(return_value=session))
    monkeypatch.setattr('kumo_s3.session.fs.S3FileSystem', Filesystem)
    monkeypatch.setattr('kumo_s3.backend.S3FileSystem', Filesystem)
    resolver = Mock(return_value='eu-west-3')
    monkeypatch.setattr('kumo_s3.bucket_region', resolver)
    connector = S3Connector(
        root_dir='s3://bucket', region=root_region,
        identity=S3Identity(connection=S3ConnectionConfig(
            endpoint_url=endpoint, region=configured_region,
            addressing_style='path')))
    assert connector.source_backend.list_tables(connector).table_names == [
        'events'
    ]
    acquired = tmp_path / 'acquired'
    connector.source_backend.prepare_ingest(connector, 'events', str(acquired))
    assert pq.read_table(acquired).to_pydict() == {'id': [1, 2]}
    connector.sink_backend.write_table(connector, 'scores',
                                       pd.DataFrame({'score': [1.0]}))
    assert pq.read_table(storage / 'bucket/scores.parquet').num_rows == 1
    assert bool(resolver.call_count) == (not endpoint and not configured_region
                                         and not root_region)
    assert built and clients
    for options in built:
        assert options['anonymous'] is True
        assert options['region'] == region
        assert options['force_virtual_addressing'] is False
        assert options.get('endpoint_override') == endpoint
    for options in clients:
        assert options['region_name'] == region
        assert options['endpoint_url'] == endpoint
        assert options['config'].signature_version == UNSIGNED
        assert options['config'].s3 == {'addressing_style': 'path'}
