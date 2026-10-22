# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_adls.native_output import NativeOutputHandler
from kumo_adls.sdk_filesystem import ADLSFileSystem
from kumo_core.object_store_sink import ObjectStoreSinkBackend
from pyarrow.fs import (
    FileSelector,
    FileType,
    LocalFileSystem,
    PyFileSystem,
    SubTreeFileSystem,
)

AUTHORITY = 'curated@companylake.dfs.core.windows.net'


@pytest.fixture
def storage(service, tmp_path):
    native = SubTreeFileSystem(str(tmp_path), LocalFileSystem())
    native.create_dir('curated/orders')
    sdk = ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')
    handler = NativeOutputHandler(sdk, native)
    return PyFileSystem(handler), native, sdk


def test_metadata_and_reads_keep_sdk_authority(storage, service, monkeypatch):
    arrow, native, sdk = storage
    paths = []
    original = sdk.info

    def info(path, **kwargs):
        paths.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(sdk, 'info', info)
    path = AUTHORITY + '/orders/events.parquet'
    result = arrow.get_file_info(path)
    assert result.path == path
    assert result.type == FileType.File
    assert result.size == len(service.files['curated/orders/events.parquet'])
    assert path in paths
    entries = arrow.get_file_info(FileSelector(AUTHORITY + '/orders'))
    assert [entry.path for entry in entries] == [path]
    with arrow.open_input_file(path) as reader:
        assert reader.read() == service.files['curated/orders/events.parquet']
    assert native.get_file_info(
        'curated/orders/events.parquet').type == FileType.NotFound


@pytest.mark.parametrize('key',
                         ['orders/plain.bin', 'orders/a space %2F # ü.bin'])
def test_output_is_native_stream_and_preserves_literal_key(storage, key):
    arrow, native, _ = storage
    with arrow.open_output_stream(AUTHORITY + '/' + key) as writer:
        assert isinstance(writer, pa.NativeFile)
        writer.write(b'first')
        writer.write(b'\x00second')
        writer.flush()
    with native.open_input_file('curated/' + key) as reader:
        assert reader.read() == b'first\x00second'


def test_generated_parquet_writer_uses_native_output(storage):
    arrow, native, _ = storage
    table = pa.table({'id': [1, 2, 3], 'value': ['alpha', None, 'ü']})
    path = AUTHORITY + '/orders/generated.parquet'
    with arrow.open_output_stream(path) as writer:
        with pq.ParquetWriter(writer, table.schema) as parquet:
            parquet.write_table(table.slice(0, 1))
            parquet.write_table(table.slice(1))
    with native.open_input_file('curated/orders/generated.parquet') as reader:
        assert pq.read_table(reader).equals(table)


def test_shared_sink_copies_prepared_file_through_native_stream(
        storage, tmp_path):
    arrow, native, _ = storage
    # Cross the shared sink's 8 MiB copy boundary, including a final short write.
    payload = bytes(range(256)) * ((8 << 20) // 256) + b'last chunk'
    source = tmp_path / 'prepared.bin'
    source.write_bytes(payload)
    ObjectStoreSinkBackend._upload_file(None, arrow, str(source),
                                        AUTHORITY + '/orders/copied.bin')
    with native.open_input_file('curated/orders/copied.bin') as reader:
        assert reader.read() == payload


def test_wrong_account_is_rejected_before_native_output_opens(
        storage, monkeypatch):
    arrow, _, _ = storage
    opened = []

    class UnopenedOutput:
        def open_output_stream(self, *args, **kwargs):
            opened.append((args, kwargs))
            pytest.fail('Invalid authority reached native output')

    arrow.handler.output_filesystem = UnopenedOutput()
    with pytest.raises(ValueError, match='bound storage account'):
        arrow.open_output_stream(
            'curated@otherlake.dfs.core.windows.net/orders/x')
    assert opened == []


@pytest.mark.parametrize('extension', ['csv', 'parquet'])
def test_sink_generated_table_routes_native_and_reads_sdk(
        service, connector, monkeypatch, extension):
    import pandas as pd
    from fakes import FakeFile
    from kumo_adls.sink import ADLSSinkBackend
    from kumoapi.jobs import WriteMode

    def unexpected_upload(*args, **kwargs):
        pytest.fail('Native output must not call an SDK upload')

    monkeypatch.setattr(ADLSFileSystem, 'put_file', unexpected_upload)
    monkeypatch.setattr(FakeFile, 'upload_data', unexpected_upload)
    backend = ADLSSinkBackend()
    backend.write_table(connector(), 'result.' + extension,
                        pd.DataFrame({'id': [1, 2]}),
                        write_mode=WriteMode.OVERWRITE)
    path = 'curated/orders/result.' + extension
    payload = service.files[path]
    if extension == 'parquet':
        assert pq.read_table(pa.BufferReader(payload)).to_pydict() == {
            'id': [1, 2]
        }
    else:
        assert payload.decode().splitlines() == ['"id"', '1', '2']
    filesystem, destination = backend._destination(connector(),
                                                   'result.' + extension)
    assert filesystem.get_file_info(destination).size == len(payload)
    with filesystem.open_input_file(destination) as stream:
        assert stream.read() == payload


def test_sink_factory_is_lazy_for_reads_and_reused_for_concurrent_outputs(
        service, connector, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from conftest import FakeNativeFilesystem
    from kumo_adls.sink import ADLSSinkBackend

    calls = []
    barrier = threading.Barrier(4)

    def factory(*args):
        calls.append(args)
        return FakeNativeFilesystem(service)

    monkeypatch.setattr('kumo_adls.sink.filesystem_for', factory)
    filesystem, _ = ADLSSinkBackend()._destination(connector(), 'unused')
    with filesystem.open_input_file(AUTHORITY +
                                    '/orders/events.parquet') as stream:
        assert stream.read(4) == b'PAR1'
    assert calls == []

    def write(index):
        barrier.wait(timeout=5)
        with filesystem.open_output_stream(AUTHORITY +
                                           f'/orders/out-{index}') as stream:
            stream.write(str(index).encode())

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, range(4)))
    assert len(calls) == 1
    for index in range(4):
        assert service.files[f'curated/orders/out-{index}'] == str(
            index).encode()


def test_sink_unsupported_output_factory_does_not_block_model_reads(
        service, connector, monkeypatch, tmp_path):
    from kumo_adls.sink import ADLSSinkBackend

    def unsupported(*args):
        raise NotImplementedError('unsupported native output identity')

    monkeypatch.setattr('kumo_adls.sink.filesystem_for', unsupported)
    backend = ADLSSinkBackend()
    filesystem, path = backend._destination(connector(), 'events.parquet')
    output = tmp_path / 'model.parquet'
    backend._download_file(filesystem, path, str(output))
    assert output.read_bytes() == service.files['curated/orders/events.parquet']
    with pytest.raises(NotImplementedError, match='unsupported native output'):
        filesystem.open_output_stream(AUTHORITY + '/orders/new')


def test_sink_integrates_real_local_native_stream_and_shared_copy(
        service, connector, monkeypatch, tmp_path):
    from kumo_adls.sink import ADLSSinkBackend

    native = SubTreeFileSystem(str(tmp_path), LocalFileSystem())
    native.create_dir('curated/orders')
    monkeypatch.setattr('kumo_adls.sink.filesystem_for', lambda *args: native)
    backend = ADLSSinkBackend()
    filesystem, destination = backend._destination(connector(), 'native.bin')
    source = tmp_path / 'prepared.bin'
    source.write_bytes(b'native output')
    assert ADLSSinkBackend._upload_file is ObjectStoreSinkBackend._upload_file
    backend._upload_file(filesystem, str(source), destination)
    with native.open_input_file('curated/orders/native.bin') as stream:
        assert stream.read() == b'native output'
    assert 'curated/orders/native.bin' not in service.files


def test_export_metadata_fence_rejects_other_identity_before_native_open(
        service, connector, monkeypatch, tmp_path):
    from kumo_adls.sink import ADLSSinkBackend
    from kumo_core.artifact_export import (
        ArtifactDestinationConflict,
        export_marker,
    )

    source = tmp_path / 'source.parquet'
    pq.write_table(pa.table({'id': [1]}), source)
    backend = ADLSSinkBackend()
    backend.write_files(connector(), 'predictions/', [str(source)],
                        export_id='first')
    marker = 'curated/orders/predictions/_KUMO_EXPORT'
    assert service.files[marker].decode() == export_marker('first')
    committed = dict(service.files)

    def unexpected_factory(*args):
        pytest.fail('Conflicting export must fail before opening native output')

    monkeypatch.setattr('kumo_adls.sink.filesystem_for', unexpected_factory)
    with pytest.raises(ArtifactDestinationConflict, match='another export'):
        backend.write_files(connector(), 'predictions/', [str(source)],
                            export_id='second')
    assert service.files == committed


@pytest.mark.parametrize('operation',
                         ['write_table', 'write_files', 'publish_model'])
def test_unsupported_native_config_fails_before_sdk_mutation(
        service, connector, monkeypatch, tmp_path, operation):
    import pandas as pd
    from kumo_adls.sink import ADLSSinkBackend

    calls = []

    def unsupported(*args):
        calls.append(args)
        raise NotImplementedError('unsupported native output identity')

    monkeypatch.setattr('kumo_adls.sink.filesystem_for', unsupported)
    initial_directories = set(service.directories)
    initial_files = dict(service.files)
    initial_created = list(service.created)
    backend = ADLSSinkBackend()
    source = tmp_path / 'source.parquet'
    pq.write_table(pa.table({'id': [1]}), source)
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'model.bin').write_bytes(b'model')
    with pytest.raises(NotImplementedError, match='unsupported native output'):
        if operation == 'write_table':
            backend.write_table(connector(), 'new/output.parquet',
                                pd.DataFrame({'id': [1]}))
        elif operation == 'write_files':
            backend.write_files(connector(), 'new/predictions/', [str(source)],
                                export_id='test')
        else:
            backend.publish_model(connector(), str(bundle), path='new/model')
    assert len(calls) == 1
    assert service.directories == initial_directories
    assert service.files == initial_files
    assert service.created == initial_created
