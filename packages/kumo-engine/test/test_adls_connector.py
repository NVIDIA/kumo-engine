# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.sources.object_store_access import ObjectStoreAccess
from kumoapi.data_source import AzureCloud, DataSourceType
from kumoapi.source_table import ADLSRelativeSource, ADLSSourceTable
from pyarrow.fs import LocalFileSystem, SubTreeFileSystem

from kumo.engine import ADLSConnector, ADLSIdentity, Table
from kumo.engine._portable import _source_table_name, portable_graph_config

ROOT = 'abfss://curated@companylake.dfs.core.windows.net/orders/'


@pytest.fixture
def storage(monkeypatch, tmp_path):
    directory = tmp_path / 'storage'
    target = directory / 'curated@companylake.dfs.core.windows.net/orders'
    target.mkdir(parents=True)
    pq.write_table(pa.table({
        'id': [1, 2],
        'value': [10., 20.]
    }), target / 'events.parquet')
    filesystem = SubTreeFileSystem(str(directory), LocalFileSystem())

    def identity(path):
        with filesystem.open_input_file(path) as stream:
            return (hashlib.sha256(stream.read()).hexdigest(), )

    monkeypatch.setattr(
        'kumo_adls.backend.source_access',
        lambda *args: ObjectStoreAccess(filesystem, object_identity=identity))
    return directory


@pytest.mark.parametrize(
    'root, prefix', [(ROOT, 'orders'), (ROOT.rstrip('/'), 'orders'),
                     (ROOT.removesuffix('orders/'), ''),
                     (ROOT.removesuffix('/orders/'), ''),
                     (ROOT.replace('orders/', 'space%20name/'), 'space name')])
def test_root_parsing_without_connecting(root, prefix):
    connector = ADLSConnector(root_dir=root)
    assert connector.prefix == prefix
    assert connector.filesystem == 'curated'
    assert connector.storage_account == 'companylake'
    assert connector.source_type == DataSourceType.ADLS


@pytest.mark.parametrize('root', [
    'abfs://curated@companylake.dfs.core.windows.net/orders/',
    ROOT + '../outside', ROOT + '?token=secret',
    ROOT.replace('orders/', 'orders//')
])
def test_invalid_roots_fail_locally(root):
    with pytest.raises(ValueError):
        ADLSConnector(root_dir=root)


def test_workload_identity_is_registered_without_credentials():
    identity = ADLSIdentity(tenant_id='tenant', client_id='client',
                            cloud=AzureCloud.PUBLIC)
    connector = ADLSConnector(root_dir=ROOT, identity=identity)
    assert connector.identity is identity
    assert connector.identity_id == identity.slug
    assert identity._credentials is None
    assert 'tenant' not in repr(identity)
    assert connector.source_backend._identity is identity
    assert connector.sink_backend._identity is identity
    assert connector.source_backend is connector.source_backend
    assert connector.sink_backend is connector.sink_backend


def test_table_inference_portable_graph_and_ingest(storage, tmp_path):
    connector = ADLSConnector(root_dir=ROOT)
    assert connector.table_names() == ['events']
    source = connector.source_table_for('events')
    assert isinstance(source, ADLSSourceTable)
    assert isinstance(source.source, ADLSRelativeSource)
    assert _source_table_name(source).startswith('adls.')
    assert len(_source_table_name(source).split('.')) == 3
    table = Table.from_source_table(connector['events'],
                                    primary_key='id').infer_metadata()
    graph = portable_graph_config({'events': table}, [], name='test')
    assert graph['tables'][0]['source_table_name'].startswith('adls.')
    ingest = connector.source_backend.prepare_ingest(connector, 'events',
                                                     str(tmp_path / 'ingest'))
    assert pq.read_table(ingest.source['paths']).num_rows == 2
    connector.sink_backend.write_table(connector, 'scores',
                                       pd.DataFrame({'score': [0.5]}))
    assert pq.read_table(
        storage /
        'curated@companylake.dfs.core.windows.net/orders/scores.parquet'
    ).num_rows == 1


def test_rootless_graph_coordinate_contains_exact_source(storage):
    connector = ADLSConnector()
    source = connector.source_table_for(ROOT + 'events.parquet')
    coordinate = _source_table_name(source)
    assert 'companylake' in coordinate
    assert len(coordinate.split('.')) == 3
