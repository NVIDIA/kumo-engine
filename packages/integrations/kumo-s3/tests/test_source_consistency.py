# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real Arrow HEAD/ranged GETs against a changing S3 endpoint."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.sources.errors import TransientSourceError
from kumo_core.sources.object_store import ObjectStoreSourceReader
from kumo_core.sources.types import ReadMode
from kumo_s3.access import source_access
from kumoapi.typing import Dtype

pytestmark = pytest.mark.allow_hosts(['127.0.0.1'])


@pytest.fixture
def endpoint():
    versions = []
    for offset in [0, 1]:
        sink = pa.BufferOutputStream()
        pq.write_table(
            pa.table({
                'id': range(offset, 100000 + offset),
                'unused': range(100000),
            }), sink, compression='NONE', use_dictionary=False,
            row_group_size=10000)
        versions.append(sink.getvalue().to_pybytes())
    assert len(versions[0]) == len(versions[1])
    state = SimpleNamespace(version=0, rewrite=False, gets=0, heads=0, bytes=0,
                            versions=versions, versioned=False, identity=True)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            pass

        def response_headers(self, length):
            self.send_header('Content-Length', str(length))
            self.send_header('Last-Modified', 'Wed, 07 Oct 2026 12:00:00 GMT')
            # A separate version-ID check must work even if ETag is equal.
            etag = 0 if state.versioned else state.version
            if state.identity:
                self.send_header('ETag', f'"etag-{etag}"')
            if state.versioned:
                self.send_header('x-amz-version-id', str(state.version))
            self.end_headers()

        def do_HEAD(self):
            state.heads += 1
            self.send_response(200)
            self.response_headers(len(versions[state.version]))

        def do_GET(self):
            state.gets += 1
            start, end = self.headers.get('Range').removeprefix('bytes=').split(
                '-')
            payload = versions[state.version][int(start):int(end) + 1]
            self.send_response(206)
            self.send_header('Content-Range',
                             f'bytes {start}-{end}/{len(versions[0])}')
            self.response_headers(len(payload))
            self.wfile.write(payload)
            state.bytes += len(payload)
            if state.rewrite:
                state.version = 1

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f'http://127.0.0.1:{server.server_port}'
    yield state
    server.shutdown()
    server.server_close()
    thread.join()


def access_for(endpoint):
    return source_access(
        'bucket', session=boto3.Session(aws_access_key_id='test',
                                        aws_secret_access_key='test',
                                        region_name='us-east-1'),
        endpoint_url=endpoint.url, options={'force_virtual_addressing': False})


@pytest.mark.parametrize('versioned', [False, True])
@pytest.mark.parametrize('copy', [False, True])
def test_same_size_same_timestamp_rewrite_is_rejected(endpoint, tmp_path,
                                                      versioned, copy):
    endpoint.versioned = versioned
    endpoint.rewrite = True
    access = access_for(endpoint)
    destination = tmp_path / 'out'
    reader = ObjectStoreSourceReader(access.filesystem,
                                     root='bucket/data.parquet',
                                     file_format='parquet', fetch=access.fetch,
                                     object_identity=access.object_identity)
    with pytest.raises(TransientSourceError):
        reader.prepare('', str(destination),
                       columns=None if copy else {'id': Dtype.int64},
                       mode=ReadMode.STREAMED)
    assert endpoint.gets >= 1
    assert not destination.exists()


def test_unchanged_native_projection_does_not_download_unused_columns(
        endpoint, tmp_path):
    access = access_for(endpoint)
    reader = ObjectStoreSourceReader(access.filesystem,
                                     root='bucket/data.parquet',
                                     file_format='parquet', fetch=access.fetch,
                                     object_identity=access.object_identity)
    result = reader.prepare('', str(tmp_path / 'out'),
                            columns={'id': Dtype.int64}, mode=ReadMode.STREAMED)
    assert pq.read_table(list(result.paths)).column('id').to_pylist() == list(
        range(100000))
    assert endpoint.bytes < len(endpoint.versions[0])


def test_missing_provider_identity_fails_closed(endpoint, tmp_path):
    endpoint.identity = False
    access = access_for(endpoint)
    target = tmp_path / 'copy'
    target.write_bytes(b'previous')
    with pytest.raises(TransientSourceError, match='No S3 object identity'):
        access.fetch('bucket/data.parquet', str(target))
    assert target.read_bytes() == b'previous'
    assert endpoint.gets == 0
