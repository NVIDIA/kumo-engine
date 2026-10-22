# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real Azure SDK transfer behavior against mocked HTTP; no cloud benchmarks."""

import base64
import io
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
import requests
from azure.core.pipeline.transport import RequestsTransport
from azure.storage.filedatalake import DataLakeServiceClient
from kumo_adls import transfer
from kumo_adls.sdk_filesystem import ADLSFileSystem
from kumo_adls.transfer import (
    BULK_DOWNLOAD_CHUNK_BYTES,
    CHUNK_BYTES,
    INITIAL_GET_BYTES,
    NETWORK_BLOCK_BYTES,
)
from kumo_core.sources.errors import SourceError
from pyarrow.fs import FSSpecHandler, PyFileSystem
from requests.adapters import BaseAdapter
from requests.structures import CaseInsensitiveDict
from urllib3.response import HTTPResponse

AUTHORITY = 'curated@companylake.dfs.core.windows.net'
KEY = base64.b64encode(b'test-key').decode()
HEADERS = {
    'ETag': '"stable"',
    'Last-Modified': 'Wed, 21 Oct 2026 00:00:00 GMT',
    'x-ms-blob-type': 'BlockBlob',
    'x-ms-request-id': 'test'
}


class HTTPAdapter(BaseAdapter):
    """In-memory HTTP responses without serializing concurrent SDK sends."""
    def __init__(self):
        self.routes = []
        self.fragments = []

    def register_uri(self, method, url, *, headers=None, status_code=200,
                     content=b''):
        self.routes.append((method, url, headers or {}, status_code, content))

    def send(self, request, **kwargs):
        for method, url, headers, status, content in self.routes:
            if request.method != method or not url.match(request.url):
                continue
            context = SimpleNamespace(headers=dict(headers), status_code=status)
            body = content(request, context) if callable(content) else content
            response = requests.Response()
            response.status_code = context.status_code
            response.headers = CaseInsensitiveDict(context.headers)
            response.request = request
            response.url = request.url
            response.raw = HTTPResponse(body=io.BytesIO(body),
                                        headers=context.headers,
                                        request_method=request.method,
                                        preload_content=False)
            original_stream = response.raw.stream

            def stream(*args, **kwargs):
                try:
                    for fragment in original_stream(*args, **kwargs):
                        self.fragments.append(
                            (request.method, request.headers.get('x-ms-range'),
                             len(fragment)))
                        yield fragment
                finally:
                    closed = getattr(context, 'on_stream_closed', None)
                    if closed is not None:
                        closed()

            response.raw.stream = stream
            return response
        raise AssertionError(
            f'Unexpected SDK request: {request.method} {request.url}')

    def close(self):
        pass


@pytest.fixture
def mocked_http():
    return HTTPAdapter()


def client(adapter, *, network_block_bytes=NETWORK_BLOCK_BYTES,
           read_chunk_bytes=BULK_DOWNLOAD_CHUNK_BYTES,
           initial_get_bytes=INITIAL_GET_BYTES):
    http = requests.Session()
    http.mount('https://', adapter)
    return DataLakeServiceClient(
        'https://companylake.dfs.core.windows.net', credential=KEY,
        max_single_get_size=initial_get_bytes,
        max_chunk_get_size=read_chunk_bytes,
        connection_data_block_size=network_block_bytes, retry_total=0,
        transport=RequestsTransport(
            session=http, session_owner=True,
            connection_data_block_size=network_block_bytes))


@pytest.mark.parametrize(
    'initial_get_bytes,read_chunk_bytes,network_block_bytes',
    [(8 << 20, read, block) for read in (4 << 20, 8 << 20)
     for block in (1 << 20, 4 << 20, 8 << 20)] +
    [(16 << 20, 16 << 20, 16 << 20), (16 << 20, 4 << 20, 16 << 20)])
def test_sdk_large_arrow_range_preserves_geometry_and_transport_blocks(
        mocked_http, initial_get_bytes, read_chunk_bytes, network_block_bytes):
    offset = (2 << 20) + 17
    length = initial_get_bytes + read_chunk_bytes + 98_765
    size = offset + length + 123
    pattern = bytes(range(251))
    content = (pattern * ((size + len(pattern) - 1) // len(pattern)))[:size]
    mocked_http.register_uri('HEAD', re.compile(r'https://companylake\.'),
                             headers={
                                 **HEADERS, 'Content-Length': str(size)
                             })
    ranges = []

    def get(request, context):
        assert request.headers['If-Match'] == '"stable"'
        assert 'Range' not in request.headers
        start, end = map(int, request.headers['x-ms-range'][6:].split('-'))
        ranges.append((start, end))
        assert offset <= start <= end < offset + length
        context.status_code = 206
        context.headers.update(HEADERS)
        context.headers['Content-Range'] = f'bytes {start}-{end}/{size}'
        context.headers['Content-Length'] = str(end - start + 1)
        return content[start:end + 1]

    mocked_http.register_uri('GET', re.compile(r'https://companylake\.'),
                             content=get)
    with client(mocked_http, network_block_bytes=network_block_bytes,
                read_chunk_bytes=read_chunk_bytes,
                initial_get_bytes=initial_get_bytes) as service:
        filesystem = ADLSFileSystem(service, 'companylake',
                                    'dfs.core.windows.net')
        arrow = PyFileSystem(FSSpecHandler(filesystem))
        with arrow.open_input_file(AUTHORITY + '/data') as stream:
            assert stream.read_at(length,
                                  offset) == content[offset:offset + length]

    expected = [(offset, offset + initial_get_bytes - 1)
                ] + [(start, min(start + read_chunk_bytes, offset + length) - 1)
                     for start in range(offset + initial_get_bytes, offset +
                                        length, read_chunk_bytes)]
    assert sorted(ranges) == expected
    for start, end in expected:
        actual = [
            length for method, interval, length in mocked_http.fragments
            if method == 'GET' and interval == f'bytes={start}-{end}'
        ]
        full, remainder = divmod(end - start + 1, network_block_bytes)
        assert actual == ([network_block_bytes] * full +
                          ([remainder] if remainder else []))


@pytest.mark.parametrize('failure', ['changed', 'truncated_body'])
@pytest.mark.parametrize(
    'initial_get_bytes,read_chunk_bytes,network_block_bytes',
    [(8 << 20, 4 << 20, 1 << 20), (16 << 20, 16 << 20, 16 << 20),
     (16 << 20, 4 << 20, 16 << 20)])
def test_sdk_later_chunk_failure_preserves_destination_and_removes_temporary(
        mocked_http, tmp_path, monkeypatch, failure, initial_get_bytes,
        read_chunk_bytes, network_block_bytes):
    monkeypatch.setattr(transfer, '_SLOTS', threading.BoundedSemaphore(1))
    size = initial_get_bytes + read_chunk_bytes + 113
    mocked_http.register_uri('HEAD', re.compile(r'https://companylake\.'),
                             headers={
                                 **HEADERS, 'Content-Length': str(size)
                             })
    ranges = []

    def get(request, context):
        assert request.headers['If-Match'] == '"stable"'
        start, end = map(int, request.headers['x-ms-range'][6:].split('-'))
        ranges.append((start, end))
        context.headers.update(HEADERS)
        if start and failure == 'changed':
            body = b'<Error><Code>ConditionNotMet</Code><Message>changed</Message></Error>'
            context.status_code = 412
            context.headers.update({
                'Content-Length': str(len(body)),
                'Content-Type': 'application/xml',
                'x-ms-error-code': 'ConditionNotMet'
            })
            return body
        context.status_code = 206
        context.headers['Content-Range'] = f'bytes {start}-{end}/{size}'
        context.headers['Content-Length'] = str(end - start + 1)
        # A later body ends early despite its declared full Content-Length.
        return b'x' * (end - start + 1 - int(bool(start)))

    mocked_http.register_uri('GET', re.compile(r'https://companylake\.'),
                             content=get)
    destination = tmp_path / 'existing'
    destination.write_bytes(b'previous complete file')
    with client(mocked_http, initial_get_bytes=initial_get_bytes,
                read_chunk_bytes=read_chunk_bytes,
                network_block_bytes=network_block_bytes) as service:
        filesystem = ADLSFileSystem(service, 'companylake',
                                    'dfs.core.windows.net')
        with pytest.raises(SourceError):
            filesystem.fetch(AUTHORITY + '/data', str(destination))
    assert ranges[0] == (0, initial_get_bytes - 1)
    assert any(start == initial_get_bytes for start, _ in ranges)
    assert destination.read_bytes() == b'previous complete file'
    assert not list(tmp_path.glob('.kumo-adls-*'))
    assert transfer._SLOTS.acquire(blocking=False)
    transfer._SLOTS.release()


@pytest.mark.parametrize('ceiling', [1, 2, 4])
def test_sdk_download_obeys_transfer_ceiling_and_preserves_content(
        mocked_http, tmp_path, monkeypatch, ceiling):
    monkeypatch.setattr(transfer, '_SLOTS', threading.BoundedSemaphore(ceiling))
    size = INITIAL_GET_BYTES + BULK_DOWNLOAD_CHUNK_BYTES + 113
    names = [f'data-{i}' for i in range(4)]
    content = {
        name:
        b''.join(
            bytes([i + j]) * min(CHUNK_BYTES, size - offset)
            for j, offset in enumerate(range(0, size, CHUNK_BYTES)))
        for i, name in enumerate(names)
    }
    etags = {name: f'"{name}-stable"' for name in names}

    def filename(request):
        return urlsplit(request.url).path.rsplit('/', 1)[-1]

    def head(request, context):
        context.headers.update(HEADERS)
        context.headers['ETag'] = etags[filename(request)]
        context.headers['Content-Length'] = str(size)
        return b''

    mocked_http.register_uri('HEAD', re.compile(r'https://companylake\.'),
                             content=head)
    barrier = threading.Barrier(ceiling, timeout=10)
    lock = threading.Lock()
    ranges = {name: [] for name in names}
    active = {name: 0 for name in names}
    request_threads = {name: set() for name in names}
    fetch_threads = {}
    maximum = 0

    def get(request, context):
        nonlocal maximum
        name = filename(request)
        assert request.headers['If-Match'] == etags[name]
        start, end = map(int, request.headers['x-ms-range'][6:].split('-'))
        with lock:
            active[name] += 1
            assert active[name] == 1
            assert sum(active.values()) <= ceiling
            maximum = max(maximum, sum(active.values()))
            ranges[name].append((start, end))
            request_threads[name].add(threading.get_ident())

        def closed():
            with lock:
                active[name] -= 1

        context.on_stream_closed = closed
        if start == 0:
            # Prove cross-file overlap while every file remains serial.
            barrier.wait()
        context.status_code = 206
        context.headers.update(HEADERS)
        context.headers['ETag'] = etags[name]
        context.headers['Content-Range'] = f'bytes {start}-{end}/{size}'
        context.headers['Content-Length'] = str(end - start + 1)
        return content[name][start:end + 1]

    mocked_http.register_uri('GET', re.compile(r'https://companylake\.'),
                             content=get)
    with client(mocked_http) as service:
        filesystem = ADLSFileSystem(service, 'companylake',
                                    'dfs.core.windows.net')

        def fetch(name):
            fetch_threads[name] = threading.get_ident()
            filesystem.fetch(AUTHORITY + '/' + name, str(tmp_path / name))

        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            list(pool.map(fetch, names))
    expected = [(0, INITIAL_GET_BYTES - 1)] + [
        (start, min(start + BULK_DOWNLOAD_CHUNK_BYTES, size) - 1)
        for start in range(INITIAL_GET_BYTES, size, BULK_DOWNLOAD_CHUNK_BYTES)
    ]
    for name in names:
        assert (tmp_path / name).read_bytes() == content[name]
        assert ranges[name] == expected
        assert request_threads[name] == {fetch_threads[name]}
        assert active[name] == 0
    assert maximum == ceiling
    with transfer.concurrency(size, ranged=True) as workers:
        assert workers == 1


def test_sdk_huge_serial_download_fails_without_scheduling_chunk_futures(
        mocked_http, tmp_path, monkeypatch):
    import concurrent.futures

    size = 100 << 30
    mocked_http.register_uri('HEAD', re.compile(r'https://companylake\.'),
                             headers={
                                 **HEADERS, 'Content-Length': str(size)
                             })
    ranges = []

    def get(request, context):
        assert request.headers['If-Match'] == '"stable"'
        start, end = map(int, request.headers['x-ms-range'][6:].split('-'))
        ranges.append((start, end))
        context.headers.update(HEADERS)
        if start:
            body = b'<Error><Code>ConditionNotMet</Code><Message>changed</Message></Error>'
            context.status_code = 412
            context.headers.update({
                'Content-Length': str(len(body)),
                'Content-Type': 'application/xml',
                'x-ms-error-code': 'ConditionNotMet'
            })
            return body
        assert end == INITIAL_GET_BYTES - 1
        context.status_code = 206
        context.headers['Content-Range'] = f'bytes {start}-{end}/{size}'
        context.headers['Content-Length'] = str(end - start + 1)
        return b'x' * INITIAL_GET_BYTES

    def unexpected_pool(*args, **kwargs):
        pytest.fail('Serial SDK download must not create a chunk executor')

    monkeypatch.setattr(concurrent.futures, 'ThreadPoolExecutor',
                        unexpected_pool)
    mocked_http.register_uri('GET', re.compile(r'https://companylake\.'),
                             content=get)
    destination = tmp_path / 'existing'
    destination.write_bytes(b'previous complete file')
    with client(mocked_http) as service:
        filesystem = ADLSFileSystem(service, 'companylake',
                                    'dfs.core.windows.net')
        with pytest.raises(SourceError):
            filesystem.fetch(AUTHORITY + '/huge', str(destination))
    assert ranges == [(0, INITIAL_GET_BYTES - 1),
                      (INITIAL_GET_BYTES,
                       INITIAL_GET_BYTES + BULK_DOWNLOAD_CHUNK_BYTES - 1)]
    assert destination.read_bytes() == b'previous complete file'
    assert not list(tmp_path.glob('.kumo-adls-*'))


def test_sdk_streaming_cleanup_uses_creation_etag_not_unfenced_append_etag(
        mocked_http):
    path = re.compile(
        r'https://companylake\.dfs\.core\.windows\.net/curated/streamed(?:\?|$)'
    )
    deleted = []

    def reject_delete(request, context):
        deleted.append(request.headers['If-Match'])
        context.status_code = 412
        return b''

    mocked_http.register_uri('PUT', path, headers=HEADERS, status_code=201)
    mocked_http.register_uri('PATCH', path, headers={
        **HEADERS, 'ETag': '"replacement"'
    }, status_code=202)
    mocked_http.register_uri('DELETE', path, headers=HEADERS,
                             content=reject_delete)
    service = client(mocked_http)
    try:
        filesystem = ADLSFileSystem(service, 'companylake',
                                    'dfs.core.windows.net')
        with pytest.raises(RuntimeError, match='caller failed'):
            with filesystem.open(AUTHORITY + '/streamed', 'wb',
                                 block_size=4) as stream:
                stream.write(b'abcd')
                raise RuntimeError('caller failed')
        assert stream.closed
        assert deleted == ['"stable"']
    finally:
        service.close()


def test_sdk_seekable_upload_overlaps_chunks_and_flushes_after_all_appends(
        mocked_http, tmp_path):
    source = tmp_path / 'upload'
    source.write_bytes(b''.join(bytes([i]) * CHUNK_BYTES for i in range(4)))
    barrier = threading.Barrier(4, timeout=10)
    lock = threading.Lock()
    appended = {}
    flushed = []
    created = []

    def create(request, context):
        created.append(request)
        return b''

    mocked_http.register_uri('PUT', re.compile(r'https://companylake\.'),
                             status_code=201, headers=HEADERS, content=create)

    def patch(request, context):
        query = parse_qs(urlsplit(request.url).query)
        action = query['action'][0]
        assert 'If-None-Match' not in request.headers
        if action == 'append':
            offset = int(query['position'][0])
            data = request.body
            if hasattr(data, 'read'):
                data = data.read()
            barrier.wait()
            with lock:
                appended[offset] = data
        elif action == 'flush':
            assert len(appended) == 4
            assert request.headers['If-Match'] == HEADERS['ETag']
            flushed.append(int(query['position'][0]))
        else:
            raise AssertionError(action)
        context.status_code = 202 if action == 'append' else 200
        context.headers.update(HEADERS)
        return b''

    mocked_http.register_uri('PATCH', re.compile(r'https://companylake\.'),
                             content=patch)
    with client(mocked_http) as service:
        filesystem = ADLSFileSystem(service, 'companylake',
                                    'dfs.core.windows.net')
        filesystem.put_file(str(source), AUTHORITY + '/data')
    assert b''.join(appended[offset]
                    for offset in sorted(appended)) == source.read_bytes()
    assert flushed == [source.stat().st_size]
    # overwrite=False must not recreate the object made by put_file().
    assert len(created) == 1


def test_sdk_prepared_upload_conflict_keeps_cleanup_conditional(
        mocked_http, tmp_path):
    from kumo_core.sources.errors import TransientSourceError

    source = tmp_path / 'upload'
    source.write_bytes(b'prepared bytes')
    actions = []
    path = re.compile(r'https://companylake\.')
    mocked_http.register_uri('PUT', path, status_code=201, headers=HEADERS)

    def patch(request, context):
        action = parse_qs(urlsplit(request.url).query)['action'][0]
        actions.append(action)
        assert 'If-None-Match' not in request.headers
        context.headers.update(HEADERS)
        if action == 'append':
            context.status_code = 202
            context.headers['ETag'] = '"append-response"'
        else:
            assert action == 'flush'
            assert request.headers['If-Match'] == HEADERS['ETag']
            context.status_code = 412
        return b''

    def delete(request, context):
        actions.append('delete')
        assert request.headers['If-Match'] == HEADERS['ETag']
        # Another writer's replacement must not be removed by cleanup.
        context.status_code = 412
        return b''

    mocked_http.register_uri('PATCH', path, content=patch)
    mocked_http.register_uri('DELETE', path, content=delete, headers=HEADERS)
    with client(mocked_http) as service:
        filesystem = ADLSFileSystem(service, 'companylake',
                                    'dfs.core.windows.net')
        with pytest.raises(TransientSourceError):
            filesystem.put_file(str(source), AUTHORITY + '/data')
    assert actions == ['append', 'flush', 'delete']


def test_sdk_empty_download_handles_azure_range_416(mocked_http, tmp_path):
    mocked_http.register_uri('HEAD', re.compile(r'https://companylake\.'),
                             headers={
                                 **HEADERS, 'Content-Length': '0'
                             })
    attempts = []

    def get(request, context):
        attempts.append(request.headers.get('x-ms-range'))
        context.headers.update(HEADERS)
        context.headers['Content-Length'] = '0'
        assert request.headers['If-Match'] == '"stable"'
        if request.headers.get('x-ms-range'):
            context.status_code = 416
            context.headers['x-ms-error-code'] = 'InvalidRange'
            context.headers['Content-Range'] = 'bytes */0'
        else:
            context.status_code = 200
        return b''

    mocked_http.register_uri('GET', re.compile(r'https://companylake\.'),
                             content=get)
    with client(mocked_http) as service:
        filesystem = ADLSFileSystem(service, 'companylake',
                                    'dfs.core.windows.net')
        destination = tmp_path / 'empty'
        filesystem.fetch(AUTHORITY + '/empty', str(destination))
    assert destination.read_bytes() == b''
    assert attempts[-1] is None
