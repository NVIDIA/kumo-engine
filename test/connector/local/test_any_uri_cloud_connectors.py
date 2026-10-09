# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import kumoapi.data_source as data_source_api
import kumoapi.source_table as source_api
import pandas as pd
import pytest

from io_utils import fsutil
from io_utils.fsutil.api import PathDetail
from io_utils.fsutil.credentials import credentials_for_provider
from kumo.connector.any_uri_adls_connector import AnyURIADLSConnector
from kumo.connector.any_uri_gcs_connector import AnyURIGCSConnector
from kumo.connector.any_uri_s3_connector import AnyURIS3Connector
from kumo.connector.composite_file_connector import CompositeFileConnector
from kumo.connector.connector import ConnectorError
from kumo.validation import ValidationException
from rest.config.connector import (
    ADLSConnectorResourceConfig,
    GCSConnectorResourceConfig,
    object_store_source_for_connector,
)
from rest.config.table_v2 import ADLSSourceTable, GCSSourceTable


def _detailed_listing(children, listed=None):
    def listdir(path, *, max_items=None, detail=False):
        if listed is not None:
            listed.append((path, max_items))
        entries = list(children.get(path.rstrip('/'), []))
        if max_items is not None:
            entries = entries[:max_items]
        if not detail:
            return entries
        return [
            PathDetail(
                path=entry,
                is_directory=entry.rstrip('/') in children,
            ) for entry in entries
        ]

    return listdir


@pytest.fixture(params=[
    (AnyURIS3Connector, 's3', 's3://company-data/events/table.parquet'),
    (AnyURIGCSConnector, 'gcs', 'gs://company-data/events/table.parquet'),
    (AnyURIADLSConnector, 'adls',
     ('abfss://events@company.dfs.core.windows.net/'
      'tables/table.parquet')),
])
def connector_case(request):
    connector_type, provider, uri = request.param
    connector = connector_type(name=f'rootless_{provider}')
    connector.set_credentials({'provider': provider, 'credential': object()})
    connector.require_explicit_credentials()
    return connector, provider, uri


def test_exact_resolution_runs_only_under_attested_identity(
        connector_case, monkeypatch):
    connector, provider, uri = connector_case

    def exists(path):
        assert path == uri
        scope = credentials_for_provider(provider)
        assert scope is not None
        assert scope.credentials is connector._credentials
        return True

    monkeypatch.setattr(fsutil, 'isdir', Mock(return_value=False))
    monkeypatch.setattr(fsutil, 'exists', exists)

    assert connector._get_path(uri) == uri
    assert connector.get_path_and_type(uri) == (uri, 'parquet')
    assert credentials_for_provider(provider) is None


def test_exact_missing_uri_never_probes_suffixed_neighbor(
        connector_case, monkeypatch):
    connector, _provider, uri = connector_case
    coordinate = uri.rsplit('.', 1)[0]
    isdir = Mock(return_value=False)
    exists = Mock(side_effect=lambda path: path.endswith('.parquet'))
    monkeypatch.setattr(fsutil, 'isdir', isdir)
    monkeypatch.setattr(fsutil, 'exists', exists)

    with pytest.raises(FileNotFoundError, match='source URI does not exist'):
        connector._get_path(coordinate)

    exists.assert_called_once_with(coordinate)
    assert all(not call.args[0].endswith(('.csv', '.parquet'))
               for call in exists.call_args_list)


@pytest.mark.parametrize(
    ('connector_type', 'provider', 'uri'),
    [
        (AnyURIGCSConnector, 'gcs',
         'gs://company-data/events/partitioned.parquet/'),
        (AnyURIADLSConnector, 'adls',
         ('abfss://events@company.dfs.core.windows.net/'
          'tables/partitioned.parquet/')),
    ],
    ids=['gcs', 'adls'],
)
def test_exact_partitioned_directory_preserves_one_trailing_slash(
        connector_type, provider, uri, monkeypatch):
    connector = connector_type(name=f'rootless_{provider}')
    connector.set_credentials({'provider': provider, 'credential': object()})
    connector.require_explicit_credentials()
    monkeypatch.setattr(fsutil, 'isdir', lambda path: path == uri)
    monkeypatch.setattr(
        fsutil,
        'listdir',
        _detailed_listing({
            uri.rstrip('/'): [f'{uri}part-0.parquet'],
        }),
    )

    assert connector._get_path(uri) == uri
    assert connector.get_path_and_type(uri) == (uri, 'parquet')


@pytest.mark.parametrize(
    ('connector_type', 'provider', 'root'),
    [
        (AnyURIS3Connector, 's3', 's3://company-data/events'),
        (AnyURIGCSConnector, 'gcs', 'gs://company-data/events'),
        (AnyURIADLSConnector, 'adls',
         'abfss://events@company.dfs.core.windows.net/tables'),
    ],
    ids=['s3', 'gcs', 'adls'],
)
@pytest.mark.parametrize('mode', ['rooted', 'any-uri'])
def test_partitioned_dataset_infers_nested_parquet_via_bounded_discovery(
        connector_type, provider, root, mode, monkeypatch):
    dataset = f'{root}/partitioned'
    partitions = [f'{dataset}/year={year}' for year in (2025, 2026)]
    children = {
        dataset: partitions,
        **{
            partition: [f'{partition}/part-000.parquet']
            for partition in partitions
        },
    }
    listed = []

    monkeypatch.setattr(fsutil, 'isdir',
                        lambda path: path.rstrip('/') in children)
    monkeypatch.setattr(fsutil, 'listdir', _detailed_listing(children, listed))
    if mode == 'rooted':
        connector = CompositeFileConnector('rooted', root_dir=root)
        table_name = 'partitioned'
    else:
        connector = connector_type(name=f'rootless_{provider}')
        connector.set_credentials({
            'provider': provider,
            'credential': object(),
        })
        connector.require_explicit_credentials()
        table_name = dataset

    assert connector.get_path_and_type(table_name) == (f'{dataset}/', 'parquet')
    assert listed
    assert listed[0][0].rstrip('/') == dataset

    load = Mock(return_value=pd.DataFrame({'id': [1]}))
    monkeypatch.setattr(connector, '_load', load)
    monkeypatch.setattr(
        fsutil, 'contains_subdir',
        Mock(side_effect=AssertionError('Parquet traversal is supported')))
    sample = connector._read_table_sample(None, table_name, num_rows=1,
                                          frac=None)
    assert sample['id'].tolist() == [1]
    load.assert_called_once()


def test_nested_csv_dataset_remains_unsupported(monkeypatch):
    root = 'gs://company-data/events'
    dataset = f'{root}/partitioned'
    partition = f'{dataset}/year=2026'
    children = {
        dataset: [partition],
        partition: [f'{partition}/part-000.csv'],
    }
    monkeypatch.setattr(fsutil, 'isdir',
                        lambda path: path.rstrip('/') in children)
    monkeypatch.setattr(fsutil, 'listdir', _detailed_listing(children))
    monkeypatch.setattr(fsutil, 'contains_subdir', lambda path: path == dataset)
    connector = CompositeFileConnector('rooted', root_dir=root)

    with pytest.raises(ValidationException, match='nested sub-directories'):
        connector._read_table_sample(None, 'partitioned', num_rows=1, frac=None)


def test_nested_dataset_rejects_mixed_csv_and_parquet(monkeypatch):
    root = 'gs://company-data/events'
    dataset = f'{root}/partitioned'
    partition = f'{dataset}/year=2026'
    children = {
        dataset: [partition],
        partition: [
            f'{partition}/part-000.csv',
            f'{partition}/part-001.parquet',
        ],
    }
    monkeypatch.setattr(fsutil, 'isdir',
                        lambda path: path.rstrip('/') in children)
    monkeypatch.setattr(fsutil, 'listdir', _detailed_listing(children))
    connector = CompositeFileConnector('rooted', root_dir=root)

    with pytest.raises(ValidationException, match='mix CSV and Parquet'):
        connector.get_path_and_type('partitioned')


@pytest.mark.parametrize('layout', ['single-file', 'flat', 'hive-nested'])
def test_adls_parquet_dataset_range_reads_instead_of_localizing(
        monkeypatch, layout, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyarrow.fs import LocalFileSystem, SubTreeFileSystem

    root = 'abfss://events@company.dfs.core.windows.net/tables/partitioned'
    relative = 'tables/partitioned'
    parts = ([f'{relative}/year={year}/part.parquet' for year in (2024, 2025)]
             if layout == 'hive-nested' else [f'{relative}/part.parquet'])
    for name in parts:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({'id': [1]}), path)
    uri = f'{root}/part.parquet' if layout == 'single-file' else root
    filesystem = SubTreeFileSystem(str(tmp_path), LocalFileSystem())
    monkeypatch.setattr(fsutil, 'pyarrow_filesystem', lambda path: filesystem)
    copy_file = Mock(side_effect=AssertionError('must not localize'))
    monkeypatch.setattr(fsutil, 'copy_file', copy_file)
    parquet = AnyURIADLSConnector(name='rootless_adls').connectors['parquet']
    table = parquet._parquet_dataset(uri).to_table()
    assert table.column('id').to_pylist() == [1] * len(parts)
    if layout == 'hive-nested':
        assert sorted(table.column('year').to_pylist()) == [2024, 2025]
    copy_file.assert_not_called()


def test_rootless_connector_never_attempts_discovery(connector_case,
                                                     monkeypatch):
    connector, _provider, _uri = connector_case
    listdir = Mock(side_effect=AssertionError('must not discover storage'))
    monkeypatch.setattr(fsutil, 'listdir', listdir)
    with pytest.raises(ConnectorError, match='require an exact URI'):
        connector.list_tables()
    listdir.assert_not_called()


def test_identity_scope_is_cleared_after_provider_exception(
        connector_case, monkeypatch):
    connector, provider, uri = connector_case

    def fail(_path):
        assert credentials_for_provider(provider) is not None
        raise RuntimeError('provider failed')

    monkeypatch.setattr(fsutil, 'isdir', fail)
    with pytest.raises(RuntimeError, match='provider failed'):
        connector._get_path(uri)
    assert credentials_for_provider(provider) is None


def test_rootless_connector_propagates_identity_to_file_delegates(
        connector_case):
    connector, _provider, _uri = connector_case
    for child in connector.connectors.values():
        assert child._credentials is connector._credentials
        assert child._require_explicit_credentials is True


@pytest.mark.parametrize('uri', [
    'abfss://events@company.dfs.core.windows.net/table.parquet',
    ('abfss://events@company.dfs.core.windows.net/path/'
     'dfs.core.usgovcloudapi.net/table.parquet'),
])
def test_adls_identity_cloud_mismatch_rejected_before_filesystem_io(
        monkeypatch, uri):
    connector = AnyURIADLSConnector(name='government_cloud')
    connector.set_credentials({
        'provider': 'adls',
        'credential': object(),
        'endpoint_suffix': 'dfs.core.usgovcloudapi.net',
    })
    provider_io = Mock(side_effect=AssertionError('must not reach fsutil'))
    monkeypatch.setattr(fsutil, 'isdir', provider_io)
    monkeypatch.setattr(fsutil, 'exists', provider_io)

    with pytest.raises(ValueError, match='identity cloud'):
        connector._get_path(uri)

    provider_io.assert_not_called()


@pytest.mark.parametrize(
    ('config', 'full_path', 'relative_path', 'table_type'),
    [
        (
            GCSConnectorResourceConfig(
                name='gcs',
                target=data_source_api.GCSRootedConnectorTarget(
                    bucket='customer-data', prefix='root'),
            ),
            'gs://customer-data/root/tables/orders.parquet',
            'tables/orders.parquet',
            GCSSourceTable,
        ),
        (
            ADLSConnectorResourceConfig(
                name='adls',
                target=data_source_api.ADLSRootedConnectorTarget(
                    storage_account='account',
                    filesystem='filesystem',
                    prefix='root',
                ),
            ),
            ('abfss://filesystem@account.dfs.core.windows.net/'
             'root/tables/orders.parquet'),
            'tables/orders.parquet',
            ADLSSourceTable,
        ),
    ],
    ids=['gcs', 'adls'],
)
def test_rooted_full_source_canonicalizes_to_relative_fingerprint(
        config, full_path, relative_path, table_type):
    full_source = object_store_source_for_connector(config, full_path)
    relative_source = object_store_source_for_connector(config, relative_path)

    assert full_source == relative_source
    assert table_type(
        connector_id=config.name,
        source=full_source,
        file_type=source_api.FileType.PARQUET,
    ).fpid == table_type(
        connector_id=config.name,
        source=relative_source,
        file_type=source_api.FileType.PARQUET,
    ).fpid


@pytest.mark.parametrize(
    ('table_type', 'first', 'second'),
    [
        (GCSSourceTable, 'gs://first/table.parquet',
         'gs://second/table.parquet'),
        (GCSSourceTable, 'gs://bucket/first.parquet',
         'gs://bucket/second.parquet'),
        (ADLSSourceTable,
         'abfss://data@first.dfs.core.windows.net/table.parquet',
         'abfss://data@second.dfs.core.windows.net/table.parquet'),
        (ADLSSourceTable,
         'abfss://first@account.dfs.core.windows.net/table.parquet',
         'abfss://second@account.dfs.core.windows.net/table.parquet'),
        (ADLSSourceTable,
         'abfss://data@account.dfs.core.windows.net/first.parquet',
         'abfss://data@account.dfs.core.windows.net/second.parquet'),
    ],
    ids=[
        'gcs-bucket', 'gcs-path', 'adls-account', 'adls-filesystem', 'adls-path'
    ],
)
def test_any_uri_fingerprint_distinguishes_storage_coordinates(
        table_type, first, second):
    def fingerprint(uri):
        return table_type(
            connector_id='connector',
            source={
                'mode': 'uri',
                'uri': uri
            },
            file_type=source_api.FileType.PARQUET,
        ).fpid

    assert fingerprint(first) != fingerprint(second)


@pytest.mark.parametrize(
    ('config', 'coordinate'),
    [
        (
            GCSConnectorResourceConfig(
                name='gcs', target=data_source_api.GCSAnyURIConnectorTarget()),
            'tables/orders.parquet',
        ),
        (
            ADLSConnectorResourceConfig(
                name='adls',
                target=data_source_api.ADLSAnyURIConnectorTarget()),
            'tables/orders.parquet',
        ),
    ],
    ids=['gcs', 'adls'],
)
def test_any_uri_source_requires_exact_full_coordinate(config, coordinate):
    with pytest.raises(ValueError, match='URI'):
        object_store_source_for_connector(config, coordinate)


@pytest.mark.parametrize(('connector_type', 'provider', 'uris'), [
    (AnyURIGCSConnector, 'gcs',
     ('gs://first/table.parquet', 'gs://second/table.parquet')),
    (AnyURIADLSConnector, 'adls',
     ('abfss://filesystem@first.dfs.core.windows.net/table.parquet',
      'abfss://filesystem@second.dfs.core.windows.net/table.parquet')),
])
def test_concurrent_exact_uri_operations_cannot_observe_other_identity(
        connector_type, provider, uris, monkeypatch):
    barrier = threading.Barrier(2)
    connectors = []
    expected = {}
    for index, uri in enumerate(uris):
        connector = connector_type(name=f'connector_{index}')
        marker = f'identity-{index}'
        connector.set_credentials({
            'provider': provider,
            'credential': object(),
            'marker': marker,
        })
        connector.require_explicit_credentials()
        connectors.append(connector)
        expected[uri] = marker

    def isdir(path):
        barrier.wait(timeout=5)
        scope = credentials_for_provider(provider)
        assert scope is not None
        assert scope.credentials['marker'] == expected[path]
        return False

    def exists(path):
        scope = credentials_for_provider(provider)
        assert scope is not None
        assert scope.credentials['marker'] == expected[path]
        return True

    monkeypatch.setattr(fsutil, 'isdir', isdir)
    monkeypatch.setattr(fsutil, 'exists', exists)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda pair: pair[0]._get_path(pair[1]),
                zip(connectors, uris),
            ))
    assert results == list(uris)
