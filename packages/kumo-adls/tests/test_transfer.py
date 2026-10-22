# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from kumo_adls import transfer
from kumo_adls.sdk_filesystem import ADLSFileSystem
from pyarrow.fs import FileSelector, FSSpecHandler, PyFileSystem

AUTHORITY = 'curated@companylake.dfs.core.windows.net'


@pytest.mark.parametrize('adls_limit, copy_limit, expected', [
    (None, None, 128),
    (None, '4', 4),
    ('4', None, 4),
    ('16', '4', 4),
    ('4', '16', 4),
    ('1', '128', 1),
    ('256', None, 128),
])
def test_transfer_ceiling_is_provider_specific_and_respects_copy_limit(
        monkeypatch, adls_limit, copy_limit, expected):
    from kumo_core.sources import cpu

    monkeypatch.setattr(cpu, 'cpu_budget', lambda: 32)
    for name, value in [(transfer.TRANSFER_WORKERS_ENV, adls_limit),
                        (cpu.COPY_WORKERS_ENV, copy_limit)]:
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert transfer._transfer_workers() == expected
    # The ADLS ceiling does not change the shared pool used by other providers.
    assert cpu.copy_workers() == (int(copy_limit) if copy_limit else 128)


@pytest.mark.parametrize('value', ['', '0', '-1', '1.5', 'four'])
def test_invalid_transfer_ceiling_is_rejected(monkeypatch, value):
    monkeypatch.setenv(transfer.TRANSFER_WORKERS_ENV, value)
    with pytest.raises(ValueError, match='must be a positive integer'):
        transfer._transfer_workers()


def test_arrow_projection_does_not_read_unrequested_bytes(service):
    filesystem = ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')
    path = AUTHORITY + '/orders/events.parquet'
    with PyFileSystem(
            FSSpecHandler(filesystem)).open_input_file(path) as stream:
        stream.seek(4)
        assert len(stream.read(8)) == 8
    assert [(offset, length)
            for _, offset, length, _ in service.downloads] == [(4, 8)]


def test_recursive_arrow_listing_uses_one_sdk_traversal(service, monkeypatch):
    from fakes import FakeFilesystem

    for i in range(20):
        directory = f'curated/orders/year={i}'
        service.directories.add(directory)
        service.files[directory + '/part.parquet'] = b'PAR1'
    original = FakeFilesystem.get_paths
    calls = []

    def paths(self, **kwargs):
        calls.append(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(FakeFilesystem, 'get_paths', paths)
    filesystem = ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')
    arrow = PyFileSystem(FSSpecHandler(filesystem))
    entries = arrow.get_file_info(
        FileSelector(AUTHORITY + '/orders', recursive=True))
    assert len(entries) == 41
    assert calls == [{'path': 'orders', 'recursive': True}]
    assert len(filesystem.find(AUTHORITY + '/orders', maxdepth=1)) == 1
    assert len(filesystem.find(AUTHORITY + '/orders', withdirs=True)) == 42


def test_request_budget_is_shared_and_released_after_failure(monkeypatch):
    monkeypatch.setattr(transfer, '_SLOTS', threading.BoundedSemaphore(4))
    entered = threading.Event()
    release = threading.Event()

    def hold():
        with transfer.concurrency(100 << 20) as count:
            assert count == 4
            entered.set()
            assert release.wait(5)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(hold)
        assert entered.wait(5)
        assert not transfer._SLOTS.acquire(blocking=False)
        release.set()
        future.result()
    with pytest.raises(RuntimeError):
        with transfer.concurrency(100 << 20) as count:
            assert count == 4
            raise RuntimeError('transfer failed')
    with transfer.concurrency(1) as count:
        assert count == 1
    with transfer.concurrency(100 << 20) as count:
        assert count == 4


@pytest.mark.parametrize('bulk', [True, False])
def test_download_memory_failure_has_guidance_and_releases_resources(
        service, tmp_path, monkeypatch, bulk):
    from fakes import FakeFile

    def download(*args, **kwargs):
        raise MemoryError

    monkeypatch.setattr(FakeFile, 'download_file', download)
    monkeypatch.setattr(transfer, '_SLOTS', threading.BoundedSemaphore(4))
    filesystem = ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')
    destination = tmp_path / 'existing'
    destination.write_bytes(b'previous')
    path = AUTHORITY + '/orders/events.parquet'
    with pytest.raises(MemoryError) as result:
        if bulk:
            filesystem.fetch(path, str(destination))
        else:
            with filesystem.open(path, 'rb', cache_type='none') as stream:
                stream.read(4)
    assert 'KUMO_ADLS_TRANSFER_WORKERS' in '\n'.join(result.value.__notes__)
    assert destination.read_bytes() == b'previous'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['existing']
    with transfer.concurrency(100 << 20, upload=True) as workers:
        assert workers == 4


def test_failed_fetch_keeps_previous_destination(service, tmp_path,
                                                 monkeypatch):
    from fakes import FakeFile
    from kumo_core.sources.errors import TransientSourceError

    filesystem = ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')
    target = tmp_path / 'existing'
    target.write_bytes(b'previous')
    monkeypatch.setattr(
        FakeFile, 'download_file', lambda *a, **k: SimpleNamespace(
            readinto=lambda stream: stream.write(b'bad')))
    with pytest.raises(TransientSourceError, match='truncated'):
        filesystem.fetch(AUTHORITY + '/orders/events.parquet', str(target))
    assert target.read_bytes() == b'previous'
    assert not list(tmp_path.glob('.kumo-adls-*'))


@pytest.mark.parametrize('failure', [None, 'download', 'promotion'])
def test_fetch_closes_temporary_file_before_promotion_and_cleanup(
        service, tmp_path, monkeypatch, failure):
    import os
    import stat

    from fakes import FakeFile
    from kumo_adls import sdk_filesystem

    opened = []
    make_temp = sdk_filesystem.NamedTemporaryFile
    replace = os.replace
    unlink = os.unlink

    def temporary(*args, **kwargs):
        stream = make_temp(*args, **kwargs)
        opened.append(stream)
        return stream

    def promote(source, destination):
        assert opened[0].closed
        if failure == 'promotion':
            raise OSError('promotion failed')
        replace(source, destination)

    def remove(path, *args, **kwargs):
        if opened and path == opened[0].name:
            assert opened[0].closed
        return unlink(path, *args, **kwargs)

    def download(*args, **kwargs):
        raise OSError('download failed')

    monkeypatch.setattr(sdk_filesystem, 'NamedTemporaryFile', temporary)
    monkeypatch.setattr(os, 'replace', promote)
    monkeypatch.setattr(os, 'unlink', remove)
    if failure == 'download':
        monkeypatch.setattr(FakeFile, 'download_file', download)
    filesystem = ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')
    destination = tmp_path / 'existing'
    destination.write_bytes(b'previous')
    if failure:
        with pytest.raises(OSError, match=failure + ' failed'):
            filesystem.fetch(AUTHORITY + '/orders/events.parquet',
                             str(destination))
        assert destination.read_bytes() == b'previous'
    else:
        filesystem.fetch(AUTHORITY + '/orders/events.parquet', str(destination))
        expected = service.files['curated/orders/events.parquet']
        assert destination.read_bytes() == expected
        if os.name == 'posix':
            assert stat.S_IMODE(destination.stat().st_mode) & 0o077 == 0
    assert opened[0].closed
    assert not list(tmp_path.glob('.kumo-adls-*'))


def test_file_export_uses_native_copy_and_keeps_prepared_columns(
        service, connector, tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from fakes import FakeFile
    from kumo_adls.sink import ADLSSinkBackend

    def unexpected_upload(*args, **kwargs):
        pytest.fail('Prepared export must use native Arrow output')

    monkeypatch.setattr(FakeFile, 'upload_data', unexpected_upload)
    monkeypatch.setattr(ADLSFileSystem, 'put_file', unexpected_upload)
    source = tmp_path / 'source.parquet'
    pq.write_table(pa.table({'id': [1, 2]}), source)
    ADLSSinkBackend().write_files(connector(), 'predictions/', [str(source)],
                                  extra_columns={'tag':
                                                 'test'}, export_id='run')
    outputs = [
        name for name in service.files
        if '/predictions/' in name and name.endswith('.parquet')
    ]
    assert len(outputs) == 1
    table = pq.read_table(pa.BufferReader(service.files[outputs[0]]))
    assert table.to_pydict() == {'id': [1, 2], 'tag': ['test', 'test']}


def test_owned_http_pool_is_reused_and_closed(monkeypatch):
    from kumo_adls import session

    session.close()
    credential = Mock()
    monkeypatch.setattr(session, 'DefaultAzureCredential',
                        Mock(return_value=credential))
    transport_session = Mock()
    monkeypatch.setattr(session, 'Session',
                        Mock(return_value=transport_session))
    # Real SDK clients exercise ownership and child-client propagation.
    original_transport = session.RequestsTransport
    transports = []

    def transport(**kwargs):
        item = original_transport(**kwargs)
        transports.append(item)
        return item

    monkeypatch.setattr(session, 'RequestsTransport', transport)
    client = session.session_for(None, 'companylake', 'dfs.core.windows.net')
    assert session.session_for(None, 'companylake',
                               'dfs.core.windows.net') is client
    assert transports[
        0].connection_config.data_block_size == transfer.NETWORK_BLOCK_BYTES
    assert transport_session.mount.call_count == 2
    adapter = transport_session.mount.call_args.args[1]
    assert adapter._pool_maxsize == transfer.CONNECTIONS
    assert adapter._pool_block
    session.close()
    transport_session.close.assert_called_once()
    credential.close.assert_called_once()


def test_empty_local_file_creates_and_truncates_remote_file(service, tmp_path):
    source = tmp_path / 'empty'
    source.touch()
    filesystem = ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')
    for previous in (None, b'old data'):
        service.files.pop('curated/empty', None)
        if previous is not None:
            service.files['curated/empty'] = previous
        filesystem.put_file(str(source), AUTHORITY + '/empty')
        assert service.files['curated/empty'] == b''


def test_upload_parallelism_uses_all_chunks_without_download_initial_request(
        monkeypatch):
    monkeypatch.setattr(transfer, '_SLOTS', threading.BoundedSemaphore(4))
    with transfer.concurrency(12 << 20, upload=True) as workers:
        assert workers == 3


def test_failed_export_does_not_delete_another_writers_replacement(
        service, monkeypatch, tmp_path):
    from azure.core.exceptions import ResourceModifiedError
    from fakes import FakeFile
    from kumo_core.sources.errors import TransientSourceError

    def replaced(self, *args, **kwargs):
        self.service.files[self.name] = b'other writer completed'
        raise ResourceModifiedError('replaced during upload')

    monkeypatch.setattr(FakeFile, 'upload_data', replaced)
    source = tmp_path / 'source'
    source.write_bytes(b'new contents')
    filesystem = ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')
    with pytest.raises(TransientSourceError):
        filesystem.put_file(str(source), AUTHORITY + '/shared')
    assert service.files['curated/shared'] == b'other writer completed'


def test_upload_reserves_and_releases_memory_on_failure(service, tmp_path,
                                                        monkeypatch):
    from fakes import FakeFile
    from kumo_core.sources import cpu

    before = cpu._reserved

    def fail(self, *args, **kwargs):
        assert cpu._reserved >= before + cpu.COPY_WORKER_BYTES
        raise RuntimeError('upload failed')

    monkeypatch.setattr(FakeFile, 'upload_data', fail)
    source = tmp_path / 'source'
    source.write_bytes(b'content')
    filesystem = ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')
    with pytest.raises(RuntimeError, match='upload failed'):
        filesystem.put_file(str(source), AUTHORITY + '/failed')
    assert cpu._reserved == before
    assert 'curated/failed' not in service.files


@pytest.mark.parametrize('upload_fails', [False, True])
def test_put_file_conditions_bind_upload_and_cleanup_to_creation(
        tmp_path, upload_fails):
    from azure.core import MatchConditions
    from azure.core.exceptions import ResourceModifiedError
    from azure.storage.filedatalake import DataLakeFileClient
    from kumo_core.sources.errors import TransientSourceError

    source = tmp_path / 'source'
    source.write_bytes(b'prepared bytes')
    client = Mock(spec=DataLakeFileClient)
    client.create_file.return_value = {'etag': '"created"'}
    service = Mock()
    service.get_file_system_client.return_value.get_file_client.return_value = client

    def upload(stream, **kwargs):
        assert not stream.closed
        assert stream.read() == b'prepared bytes'
        assert kwargs == {
            'length': source.stat().st_size,
            'overwrite': False,
            'etag': '"created"',
            'match_condition': MatchConditions.IfNotModified,
            'chunk_size': transfer.CHUNK_BYTES,
            'max_concurrency': 1,
        }
        if upload_fails:
            raise ResourceModifiedError('object replaced')

    client.upload_data.side_effect = upload
    filesystem = ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')
    if upload_fails:
        with pytest.raises(TransientSourceError):
            filesystem.put_file(str(source), AUTHORITY + '/prepared')
        client.delete_file.assert_called_once_with(
            etag='"created"', match_condition=MatchConditions.IfNotModified)
    else:
        filesystem.put_file(str(source), AUTHORITY + '/prepared')
        client.delete_file.assert_not_called()
    client.create_file.assert_called_once_with()
    client.upload_data.assert_called_once()


def test_version_checks_overlap_and_remain_independent_of_listing_order():
    from kumo_adls.backend import ADLSSourceReader

    barrier = threading.Barrier(2, timeout=5)

    def identity(path):
        barrier.wait()
        return ('etag:' + path, )

    entries = [SimpleNamespace(path=f'object-{i}') for i in range(4)]
    reader = ADLSSourceReader(None, object_identity=identity)
    assert reader._data_version('table', entries) == reader._data_version(
        'table', entries[::-1])


def test_model_download_uses_conditional_bulk_fetch(service, connector,
                                                    tmp_path):
    from kumo_adls.sink import ADLSSinkBackend

    service.directories.add('curated/orders/bundle')
    service.files['curated/orders/bundle/model.bin'] = b'model contents'
    target = tmp_path / 'model'
    ADLSSinkBackend().fetch_model(connector(), str(target), path='bundle')
    assert (target / 'model.bin').read_bytes() == b'model contents'
    _, offset, length, options = service.downloads[-1]
    assert offset == 0 and length == len(b'model contents')
    assert options['max_concurrency'] == 1


def test_filesystem_client_cache_is_bounded_and_does_not_cache_metadata(
        service, monkeypatch):
    original = service.get_file_system_client
    factory = Mock(side_effect=original)
    monkeypatch.setattr(service, 'get_file_system_client', factory)
    filesystem = ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')
    path = AUTHORITY + '/orders/events.parquet'
    first = filesystem.object_identity(path)
    service.files['curated/orders/events.parquet'] = b'rewritten'
    assert filesystem.object_identity(path) != first
    factory.assert_called_once_with('curated')
    for i in range(40):
        filesystem._address(
            f'filesystem-{i}@companylake.dfs.core.windows.net/path')
    assert len(filesystem._filesystems) == 32


def test_large_fetch_uses_one_sdk_downloader_and_cleans_failure(
        service, tmp_path, monkeypatch):
    from azure.core import MatchConditions
    from fakes import FakeFile
    from kumo_core.sources.errors import TransientSourceError

    original = FakeFile.get_file_properties
    calls = []

    def properties(self):
        result = original(self)
        result.size = 100 << 30
        return result

    def download(self, **kwargs):
        calls.append(kwargs)
        assert kwargs == dict(offset=0, length=100 << 30,
                              etag=original(self).etag,
                              match_condition=MatchConditions.IfNotModified,
                              max_concurrency=1)
        raise TransientSourceError('later download failed')

    monkeypatch.setattr(FakeFile, 'get_file_properties', properties)
    monkeypatch.setattr(FakeFile, 'download_file', download)
    monkeypatch.setattr(transfer, '_SLOTS', threading.BoundedSemaphore(4))
    destination = tmp_path / 'existing'
    destination.write_bytes(b'previous')
    filesystem = ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')
    with pytest.raises(TransientSourceError, match='later download failed'):
        filesystem.fetch(AUTHORITY + '/orders/events.parquet', str(destination))
    assert len(calls) == 1
    assert destination.read_bytes() == b'previous'
    assert not list(tmp_path.glob('.kumo-adls-*'))
    with transfer.concurrency(100 << 20, upload=True) as workers:
        assert workers == 4


def test_empty_fetch_omits_range_arguments(service, tmp_path, monkeypatch):
    from azure.core import MatchConditions
    from fakes import FakeFile

    service.files['curated/empty'] = b''
    calls = []

    def download(self, **kwargs):
        calls.append(kwargs)
        assert 'offset' not in kwargs and 'length' not in kwargs
        assert kwargs['match_condition'] == MatchConditions.IfNotModified
        assert kwargs['max_concurrency'] == 1
        return SimpleNamespace(readinto=lambda target: 0)

    monkeypatch.setattr(FakeFile, 'download_file', download)
    destination = tmp_path / 'existing'
    destination.write_bytes(b'previous')
    ADLSFileSystem(service, 'companylake',
                   'dfs.core.windows.net').fetch(AUTHORITY + '/empty',
                                                 str(destination))
    assert destination.read_bytes() == b''
    assert len(calls) == 1


def test_serial_fetch_preserves_callers_context(service, tmp_path, monkeypatch):
    import contextvars

    from azure.core import MatchConditions
    from fakes import FakeFile

    tracing_context = contextvars.ContextVar('adls-test-span')
    marker = object()
    token = tracing_context.set(marker)
    contents = service.files['curated/orders/events.parquet']
    calls = []
    original_properties = FakeFile.get_file_properties

    def download(self, **kwargs):
        assert tracing_context.get() is marker
        assert kwargs['etag'] == original_properties(self).etag
        assert kwargs['match_condition'] == MatchConditions.IfNotModified
        assert kwargs['max_concurrency'] == 1
        calls.append(kwargs)

        def readinto(target):
            assert tracing_context.get() is marker
            return target.write(contents)

        return SimpleNamespace(readinto=readinto)

    monkeypatch.setattr(FakeFile, 'download_file', download)
    destination = tmp_path / 'download'
    try:
        ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net').fetch(
            AUTHORITY + '/orders/events.parquet', str(destination))
    finally:
        tracing_context.reset(token)
    assert destination.read_bytes() == contents
    assert len(calls) == 1


@pytest.mark.parametrize('size', [0, 1, 16 << 20, (16 << 20) + 1, 100 << 30])
def test_bulk_download_uses_one_slot_and_releases_on_failure(monkeypatch, size):
    monkeypatch.setattr(transfer, '_SLOTS', threading.BoundedSemaphore(4))
    with pytest.raises(RuntimeError, match='failed'):
        with transfer.concurrency(size, ranged=True) as workers:
            assert workers == 1
            raise RuntimeError('failed')
    with transfer.concurrency(100 << 20, upload=True) as workers:
        assert workers == 4


@pytest.mark.parametrize('size, expected', [(0, 1), (16 << 20, 1),
                                            ((16 << 20) + 1, 1), (32 << 20, 1),
                                            ((32 << 20) + 1, 2), (48 << 20, 2),
                                            ((48 << 20) + 1, 3), (80 << 20, 4),
                                            (64 << 20, 3), (100 << 30, 4)])
def test_projection_download_counts_remaining_sdk_chunks(
        monkeypatch, size, expected):
    monkeypatch.setattr(transfer, '_SLOTS', threading.BoundedSemaphore(4))
    with transfer.concurrency(size) as workers:
        assert workers == expected
    with transfer.concurrency(100 << 20, upload=True) as workers:
        assert workers == 4


def test_bulk_tuning_preserves_upload_chunking_and_parallelism(monkeypatch):
    monkeypatch.setattr(transfer, '_SLOTS', threading.BoundedSemaphore(4))
    assert transfer.INITIAL_GET_BYTES == transfer.BULK_DOWNLOAD_CHUNK_BYTES == 16 << 20
    assert transfer.NETWORK_BLOCK_BYTES == 16 << 20
    assert transfer.MAX_BULK_DOWNLOAD_CONCURRENCY == 1
    assert transfer.CHUNK_BYTES == 4 << 20
    with transfer.concurrency(12 << 20, upload=True) as workers:
        assert workers == 3


@pytest.mark.parametrize('short_write, wrong_report', [(True, False),
                                                       (False, True),
                                                       (True, True)])
def test_fetch_checks_actual_position_and_reported_length(
        service, tmp_path, monkeypatch, short_write, wrong_report):
    from fakes import FakeFile
    from kumo_core.sources.errors import TransientSourceError

    def download(self, **kwargs):
        def readinto(target):
            length = kwargs['length']
            target.write(b'x' * (length - 1 if short_write else length))
            return length - 1 if wrong_report else length

        return SimpleNamespace(readinto=readinto)

    monkeypatch.setattr(FakeFile, 'download_file', download)
    destination = tmp_path / 'existing'
    destination.write_bytes(b'previous')
    filesystem = ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')
    with pytest.raises(TransientSourceError, match='truncated'):
        filesystem.fetch(AUTHORITY + '/orders/events.parquet', str(destination))
    assert destination.read_bytes() == b'previous'
    assert not list(tmp_path.glob('.kumo-adls-*'))
