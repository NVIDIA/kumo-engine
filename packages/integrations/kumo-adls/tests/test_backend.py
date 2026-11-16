# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_adls.backend import ADLSSourceBackend
from kumo_adls.sink import ADLSSinkBackend
from kumo_core.sources.errors import UnsupportedReadModeError
from kumo_core.sources.types import ReadMode
from kumoapi.source_table import ADLSRelativeSource, ADLSURISource

URI = 'abfss://curated@companylake.dfs.core.windows.net/orders/'


def test_discovery_schema_sample_and_ingestion(service, connector, tmp_path):
    source = ADLSSourceBackend()
    target = connector()
    assert source.list_tables(target).table_names == ['events']
    config = source.get_table_config(target, 'events').source_table
    assert isinstance(config.source, ADLSRelativeSource)
    assert config.source.path == 'events'
    assert [col.name for col in source.get_columns(target, 'events').cols
            ] == ['id', 'value']
    assert '10.0' in source.get_sample(target, 'events', 1).sample_rows
    version = source.get_data_version(target, 'events')
    output = source.prepare_ingest(target, 'events', str(tmp_path / 'ingest'),
                                   data_version=version)
    assert output.version == version
    assert pq.read_table(tmp_path / 'ingest').to_pydict() == {
        'id': [1, 2],
        'value': [10.0, 20.0]
    }
    assert source.get_download_urls(target,
                                    'events') == [URI + 'events.parquet']
    assert service.downloads


def test_rootless_exact_source_and_source_only_contract(service, connector):
    source = ADLSSourceBackend()
    target = connector(any_uri=True)
    uri = URI + 'events.parquet'
    config = source.get_table_config(target, uri).source_table
    assert isinstance(config.source, ADLSURISource)
    assert config.source.uri == uri
    with pytest.raises(NotImplementedError, match='cannot list'):
        source.list_tables(target)
    with pytest.raises(NotImplementedError, match='source-only'):
        ADLSSinkBackend().write_table(target, uri, pd.DataFrame({'id': [1]}))


def test_versions_separate_accounts_clouds_and_same_second_rewrites(
        service, connector):
    source = ADLSSourceBackend()
    first = source.get_data_version(connector(), 'events')
    assert first != source.get_data_version(connector(account='otherlake'),
                                            'events')
    assert first != source.get_data_version(
        connector(suffix='dfs.core.chinacloudapi.cn'), 'events')
    content = service.files['curated/orders/events.parquet']
    service.files['curated/orders/events.parquet'] = content[:-1] + b'x'
    assert first != source.get_data_version(connector(), 'events')


@pytest.mark.parametrize('path', [
    '../outside', 'foo/../outside', '%2e%2e/outside',
    URI.replace('orders/', 'elsewhere/') + 'events.parquet',
    URI.replace('companylake', 'otherlake') + 'events.parquet'
])
def test_read_and_write_fences_before_storage_access(service, connector, path):
    with pytest.raises(ValueError):
        ADLSSourceBackend().get_columns(connector(), path)
    with pytest.raises(ValueError):
        ADLSSinkBackend().write_table(connector(), path,
                                      pd.DataFrame({'id': [1]}))
    assert not service.created


@pytest.mark.parametrize('filename',
                         ['scores.parquet', 'scores.csv', 'scores/'])
def test_output_roundtrip(service, connector, filename):
    target = connector()
    uri = ADLSSinkBackend().write_table(target, filename,
                                        pd.DataFrame({'id': [3, 4]}))
    assert uri.startswith(URI + filename.rstrip('/'))
    sample = ADLSSourceBackend().get_sample(target, filename, 2).sample_rows
    assert '3' in sample and '4' in sample


def test_encoded_uri_paths_and_decoded_relative_paths(service, connector):
    service.files['curated/orders/space name.parquet'] = service.files[
        'curated/orders/events.parquet']
    source = ADLSSourceBackend()
    assert source.get_download_urls(
        connector(), 'space name.parquet') == [URI + 'space%20name.parquet']
    assert source.get_columns(connector(), URI + 'space%20name.parquet').cols
    assert source.get_columns(connector(any_uri=True),
                              URI + 'space%20name.parquet').cols
    uri = ADLSSinkBackend().write_table(connector(), 'new name.parquet',
                                        pd.DataFrame({'id': [1]}))
    assert uri == URI + 'new%20name.parquet'


def test_model_bundle_roundtrip(service, connector, tmp_path):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'model.bin').write_bytes(b'model-bytes')
    sink = ADLSSinkBackend()
    published = sink.publish_model(connector(), str(bundle),
                                   path='model bundle')
    assert published.uri == URI + 'model%20bundle'
    result = sink.fetch_model(connector(), str(tmp_path / 'fetched'),
                              path='model bundle')
    assert result.uri == published.uri
    assert (tmp_path / 'fetched/model.bin').read_bytes() == b'model-bytes'


def test_rejects_sql_and_staged_reads(service, connector, tmp_path):
    with pytest.raises(UnsupportedReadModeError):
        ADLSSourceBackend().get_columns(connector(), 'events', sql='SELECT 1')
    with pytest.raises(UnsupportedReadModeError):
        ADLSSourceBackend(mode=ReadMode.STAGED).prepare_ingest(
            connector(), 'events', str(tmp_path / 'ingest'))


def test_requested_version_is_stamped_without_refresh(service, connector,
                                                      tmp_path, monkeypatch):
    from unittest.mock import Mock

    from kumo_adls.backend import ADLSSourceReader

    source = ADLSSourceBackend()
    target = connector()
    version = source.get_data_version(target, 'events')
    original = service.files['curated/orders/events.parquet']
    # A new valid Parquet object with the same fixed last_modified timestamp.
    sink = pa.BufferOutputStream()
    pq.write_table(pa.table({'id': [3, 4], 'value': [30.0, 40.0]}), sink)
    service.files['curated/orders/events.parquet'] = sink.getvalue().to_pybytes(
    )
    assert original != service.files['curated/orders/events.parquet']
    refresh = Mock(side_effect=AssertionError('must not refresh the version'))
    monkeypatch.setattr(ADLSSourceReader, 'data_version', refresh)
    monkeypatch.setattr(ADLSSourceReader, '_data_version', refresh)
    output = source.prepare_ingest(target, 'events', str(tmp_path / 'ingest'),
                                   data_version=version)
    assert output.version == version
    assert pq.read_table(output.source['paths']).to_pydict() == {
        'id': [3, 4],
        'value': [30.0, 40.0]
    }
    refresh.assert_not_called()


def test_mutation_during_read_discards_staging(service, connector, tmp_path,
                                               monkeypatch):
    from kumo_adls.sdk_filesystem import ADLSFileSystem
    from kumo_core.sources.errors import TransientSourceError

    sink = pa.BufferOutputStream()
    pq.write_table(pa.table({'id': [3, 4], 'value': [30.0, 40.0]}), sink)
    changed = sink.getvalue().to_pybytes()
    original = ADLSFileSystem.fetch

    def fetch_then_mutate(self, path, destination):
        result = original(self, path, destination)
        service.files['curated/orders/events.parquet'] = changed
        return result

    monkeypatch.setattr(ADLSFileSystem, 'fetch', fetch_then_mutate)
    with pytest.raises(TransientSourceError, match='changed while'):
        ADLSSourceBackend().prepare_ingest(connector(), 'events',
                                           str(tmp_path / 'ingest'))
    assert not list((tmp_path / 'ingest').rglob('*.parquet'))


def test_stream_file_export_and_retry_marker(service, connector, tmp_path):
    from kumoapi.jobs import WriteMode

    source = tmp_path / 'predictions.parquet'
    pq.write_table(pa.table({'score': [0.5, 0.75]}), source)
    sink = ADLSSinkBackend()
    target = connector()
    first = sink.write_files(target, 'predictions/', [str(source)],
                             export_id='test-run')
    second = sink.write_files(target, 'predictions/', [str(source)],
                              export_id='test-run')
    assert first == second
    assert ADLSSourceBackend().get_sample(target, 'predictions', 2).sample_rows
    with pytest.raises(NotImplementedError):
        sink.write_table(target, 'predictions', pd.DataFrame({'score': [1.0]}),
                         write_mode=WriteMode.APPEND)


def test_declared_ingest_dtype_preserves_physical_source_type(
        service, connector, tmp_path):
    from kumoapi.typing import Dtype

    source = ADLSSourceBackend()
    target = connector()
    assert source.source_types(target, 'events') == {
        'id': 'int64',
        'value': 'double',
    }
    output = source.prepare_ingest(target, 'events', str(tmp_path / 'ingest'),
                                   dtypes={'id': Dtype.float64})
    staged = pq.read_table(output.source['paths'])
    assert staged.schema.field('id').type == pa.int64()
    assert staged.to_pydict() == {'id': [1, 2]}


@pytest.mark.parametrize('project', [False, True])
def test_csv_ingestion_downloads_raw_files_before_local_conversion(
        service, connector, tmp_path, project):
    from kumoapi.typing import Dtype

    name = 'curated/orders/csv_events.csv'
    service.files[name] = b'id,value\n1,10.5\n2,20.5\n'
    source = ADLSSourceBackend()
    target = connector()
    version = source.get_data_version(target, 'csv_events.csv')
    output = source.prepare_ingest(
        target, 'csv_events.csv', str(tmp_path / 'ingest'),
        dtypes={'id': Dtype.int64} if project else None, data_version=version)

    expected = {'id': [1, 2]}
    if not project:
        expected['value'] = [10.5, 20.5]
    assert pq.read_table(output.source['paths']).to_pydict() == expected
    assert output.version == version
    assert any(
        path == name and offset == 0 and length == len(service.files[name])
        for path, offset, length, _ in service.downloads)
    assert not list(tmp_path.rglob('kumo-raw-source-*'))


@pytest.mark.parametrize('project', [False, True])
def test_hive_parquet_ingestion_preserves_partition_values(
        service, connector, tmp_path, project):
    from kumoapi.typing import Dtype

    root = 'curated/orders/partitioned'
    service.directories.add(root)
    for region, row in [('north', 1), ('south', 2)]:
        directory = f'{root}/region={region}'
        service.directories.add(directory)
        buffer = pa.BufferOutputStream()
        pq.write_table(pa.table({'id': [row], 'value': [row * 10.0]}), buffer)
        service.files[f'{directory}/part.parquet'] = buffer.getvalue(
        ).to_pybytes()

    output = ADLSSourceBackend().prepare_ingest(
        connector(), 'partitioned', str(tmp_path / 'ingest'), dtypes={
            'id': Dtype.int64,
            'region': Dtype.string
        } if project else None)
    result = pq.read_table(output.source['paths']).to_pydict()
    assert sorted(zip(result['id'], result['region'])) == [(1, 'north'),
                                                           (2, 'south')]
    assert ('value' not in result) == project
    if not project:
        assert sorted(result['value']) == [10.0, 20.0]


@pytest.mark.parametrize('authority', [
    'companylake.dfs.core.windows.net',
    'curated@companylake',
    'curated@',
    '@companylake.dfs.core.windows.net',
    'curated@@companylake.dfs.core.windows.net',
    'curated@companylake.blob.core.windows.net',
    'curated@companylake.dfs.core.windows.net:443',
    'curated@companylake.dfs.core.windows.net:invalid',
    'curated:password@companylake.dfs.core.windows.net',
    'curated@.dfs.core.windows.net',
    'curated@bad.name.dfs.core.windows.net',
    'curated@[invalid',
])
def test_rootless_invalid_authority_fails_before_source_access(
        connector, monkeypatch, authority):
    from unittest.mock import Mock
    access = Mock(side_effect=AssertionError('must not access storage'))
    monkeypatch.setattr('kumo_adls.backend.source_access', access)
    with pytest.raises(ValueError):
        ADLSSourceBackend().get_columns(connector(any_uri=True),
                                        f'abfss://{authority}/events.parquet')
    access.assert_not_called()


@pytest.mark.parametrize('path', [
    URI.replace('orders/', 'orders-sibling/') + 'events.parquet',
    URI.replace('curated@', 'otherfs@') + 'events.parquet',
])
def test_full_uri_sibling_and_filesystem_escape_is_rejected(
        connector, monkeypatch, path):
    from unittest.mock import Mock
    access = Mock(side_effect=AssertionError('must not access storage'))
    monkeypatch.setattr('kumo_adls.backend.source_access', access)
    with pytest.raises(ValueError, match='source is outside'):
        ADLSSourceBackend().get_columns(connector(), path)
    with pytest.raises(ValueError, match='destination is outside'):
        ADLSSinkBackend().write_table(connector(), path,
                                      pd.DataFrame({'id': [1]}))
    access.assert_not_called()
