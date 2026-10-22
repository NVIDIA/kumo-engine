# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import traceback
from unittest.mock import Mock

import pytest
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceModifiedError,
    ResourceNotFoundError,
)
from kumo_adls.errors import translated
from kumo_adls.sdk_filesystem import ADLSFileSystem
from kumo_core.sources.errors import (
    PermanentSourceError,
    ReconnectRequiredError,
    SourceAccessDeniedError,
    TransientSourceError,
)
from pyarrow.fs import FileType, FSSpecHandler, PyFileSystem

AUTHORITY = 'curated@companylake.dfs.core.windows.net'


def test_real_sdk_property_models_and_conditional_ranges(service):
    filesystem = ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')
    arrow = PyFileSystem(FSSpecHandler(filesystem))
    assert arrow.get_file_info(AUTHORITY + '/orders').type == FileType.Directory
    assert arrow.get_file_info(AUTHORITY +
                               '/orders/events.parquet').type == FileType.File
    with filesystem.open(AUTHORITY + '/orders/events.parquet', 'rb',
                         block_size=4, cache_type='none') as stream:
        assert stream.read(4) == b'PAR1'
        service.files['curated/orders/events.parquet'] = b'changed'
        with pytest.raises(TransientSourceError, match='changed'):
            stream.seek(5)
            stream.read(4)


def test_chunked_output_and_empty_file(service):
    filesystem = ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')
    for name, chunks in [('chunked', [b'abcd', b'efgh', b'ij']), ('empty', [])]:
        with filesystem.open(AUTHORITY + '/' + name, 'wb',
                             block_size=4) as stream:
            for chunk in chunks:
                stream.write(chunk)
        assert service.files['curated/' + name] == b''.join(chunks)


def test_arrow_wrapper_closes_partial_output_after_caller_error(service):
    # Arrow does not forward the body exception to the fsspec __exit__.
    # Keep this boundary explicit: direct-context abort tests do not cover it.
    filesystem = ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')
    arrow = PyFileSystem(FSSpecHandler(filesystem))
    with pytest.raises(RuntimeError, match='serialization failed'):
        with arrow.open_output_stream(AUTHORITY + '/partial') as writer:
            writer.write(b'partial serialized content')
            raise RuntimeError('serialization failed')
    assert service.files['curated/partial'] == b'partial serialized content'


@pytest.mark.parametrize('error, expected', [
    (ClientAuthenticationError('token=secret'), ReconnectRequiredError),
    (ResourceNotFoundError('token=secret'), FileNotFoundError),
    (ResourceModifiedError('token=secret'), TransientSourceError),
    (HttpResponseError(
        'token=secret', response=Mock(status_code=403, reason='denied',
                                      headers={})), SourceAccessDeniedError),
    (HttpResponseError(
        'token=secret', response=Mock(status_code=429, reason='throttled',
                                      headers={})), TransientSourceError),
    (HttpResponseError('token=secret', response=Mock(
        status_code=400, reason='bad', headers={})), PermanentSourceError),
])
def test_errors_are_classified_without_request_metadata(error, expected):
    with pytest.raises(expected) as result:
        with translated():
            raise error
    assert 'token=secret' not in str(result.value)
    assert result.value.__suppress_context__


def test_memory_failure_preserves_allocation_traceback_and_adds_advice():
    error = MemoryError('allocation failed')

    def allocate_buffer():
        raise error

    with pytest.raises(MemoryError) as result:
        with translated():
            allocate_buffer()
    assert result.value is error
    assert str(error) == 'allocation failed'
    assert not error.__suppress_context__
    assert 'allocate_buffer' in [
        frame.name for frame in traceback.extract_tb(error.__traceback__)
    ]
    message = '\n'.join(error.__notes__)
    assert 'If this occurred during a data transfer' in message
    assert 'may help' in message
    assert 'KUMO_ADLS_TRANSFER_WORKERS' in message
    assert 'KUMO_SOURCE_COPY_WORKERS' in message
    assert 'before importing' in message
    assert 'restart' in message


@pytest.mark.parametrize('metadata_key',
                         ['hdi_isfolder', 'Hdi_isfolder', 'HDI_ISFOLDER'])
def test_directory_metadata_is_case_insensitive(service, connector, monkeypatch,
                                                metadata_key):
    from fakes import FakeFile
    from kumo_adls.backend import ADLSSourceBackend

    original = FakeFile.get_file_properties

    def properties_with_header_spelling(self):
        properties = original(self)
        if properties.metadata.get('hdi_isfolder') == 'true':
            properties.metadata = {metadata_key: 'true'}
        return properties

    monkeypatch.setattr(FakeFile, 'get_file_properties',
                        properties_with_header_spelling)
    filesystem = ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')
    arrow = PyFileSystem(FSSpecHandler(filesystem))
    directory = AUTHORITY + '/orders'
    assert arrow.get_file_info(directory).type == FileType.Directory
    assert ADLSSourceBackend().list_tables(
        connector()).table_names == ['events']
    with pytest.raises(ValueError, match='versioned file'):
        filesystem.object_identity(directory)


@pytest.mark.parametrize('failure', ['append', 'flush', 'body'])
@pytest.mark.parametrize('cleanup_fails', [False, True])
def test_failed_write_aborts_without_retry_or_masking_original_error(
        service, monkeypatch, failure, cleanup_fails):
    from fakes import FakeFile

    attempts = []
    original_append = FakeFile.append_data
    original_delete = FakeFile.delete_file

    def append(self, *args, **kwargs):
        attempts.append('append')
        if failure == 'append':
            raise RuntimeError('original failure')
        return original_append(self, *args, **kwargs)

    def flush(self, offset):
        attempts.append('flush')
        raise RuntimeError('original failure')

    def delete(self):
        attempts.append('delete')
        if cleanup_fails:
            raise RuntimeError('cleanup failure')
        original_delete(self)

    monkeypatch.setattr(FakeFile, 'append_data', append)
    monkeypatch.setattr(FakeFile, 'delete_file', delete)
    if failure == 'flush':
        monkeypatch.setattr(FakeFile, 'flush_data', flush)
    filesystem = ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')
    with pytest.raises(RuntimeError, match='original failure'):
        with filesystem.open(AUTHORITY + '/failed', 'wb',
                             block_size=4) as stream:
            stream.write(b'abcd')
            if failure == 'body':
                raise RuntimeError('original failure')
    before = list(attempts)
    stream.close()
    assert attempts == before
    assert attempts.count('delete') == 1
    assert stream.closed
    if not cleanup_fails:
        assert 'curated/failed' not in service.files


def test_exception_before_first_upload_does_not_create_file(service):
    filesystem = ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')
    with pytest.raises(RuntimeError):
        with filesystem.open(AUTHORITY + '/never-created', 'wb') as stream:
            stream.write(b'small buffered write')
            raise RuntimeError('caller failed')
    assert not service.created


@pytest.mark.parametrize('started', [False, True])
def test_discard_only_deletes_started_upload_and_is_idempotent(
        service, started):
    filesystem = ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')
    service.files['curated/existing'] = b'original'
    stream = filesystem.open(AUTHORITY + '/existing', 'wb', block_size=4)
    stream.write(b'abcd' if started else b'ab')
    stream.discard()
    stream.discard()
    stream.close()
    if started:
        assert 'curated/existing' not in service.files
    else:
        assert service.files['curated/existing'] == b'original'
