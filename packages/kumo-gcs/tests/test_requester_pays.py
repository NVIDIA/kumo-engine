# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from google.api_core.exceptions import Forbidden
from google.auth.credentials import AnonymousCredentials
from kumo_core.sources.errors import (
    SourceAccessDeniedError,
    SourceNotFoundError,
)
from kumo_core.sources.object_store import ObjectStoreSourceReader
from kumo_gcs.source_access import source_access


@pytest.fixture(params=[None, 'identity-project'])
def transport(monkeypatch, request):
    output = pa.BufferOutputStream()
    pq.write_table(pa.table({'id': [1, 2]}), output)
    contents = {
        'data/a.csv': b'id\n1\n2\n',
        'data/b.parquet': output.getvalue().to_pybytes()
    }
    requests = []
    failures = {}

    class Blob:
        def __init__(self, name):
            self.name = name
            self.size = len(contents[name])
            self.updated = datetime(2026, 1, 1, tzinfo=timezone.utc)
            self.generation = None
            self.content_encoding = None

        def reload(self):
            self.generation = '123'

        def download_to_file(self, output):
            output.write(contents[self.name])

        def download_as_bytes(self, *, start, end, **kwargs):
            assert self.generation == '123'
            if 'read' in failures:
                raise failures['read']
            requests.append(('range', self.name, start, end))
            return contents[self.name][start:end + 1]

    class Listing:
        prefixes = set()

        def __init__(self, names):
            self.names = names

        def __iter__(self):
            # A page boundary must not stop directory enumeration.
            for page in [self.names[:1], self.names[1:]]:
                for name in page:
                    yield Blob(name)

    class Bucket:
        name = 'bucket'

        def get_blob(self, key):
            if 'metadata' in failures:
                raise failures['metadata']
            return Blob(key) if key in contents else None

        def blob(self, key):
            return Blob(key)

        def list_blobs(self, *, prefix, max_results=None, delimiter=None):
            requests.append(('list', prefix, max_results, delimiter))
            names = [name for name in contents if name.startswith(prefix)]
            return Listing(names[:max_results] if max_results else names)

    class Client:
        SCOPE = ()

        def __init__(self, *, credentials, project):
            requests.append(('client', credentials, project))

        def bucket(self, name, *, user_project):
            assert name == 'bucket'
            assert user_project == 'billing-project'
            requests.append(('billing', user_project))
            return Bucket()

    monkeypatch.setattr('kumo_gcs.source_access.storage.Client', Client)
    credentials = AnonymousCredentials()
    access = source_access(credentials=credentials, project=request.param,
                           requester_pays_project='billing-project')
    return SimpleNamespace(access=access, requests=requests, failures=failures,
                           credentials=credentials, contents=contents,
                           project=request.param or 'billing-project')


def test_requester_pays_shared_reader_metadata_csv_and_parquet(
        transport, tmp_path):
    reader = ObjectStoreSourceReader(transport.access.filesystem,
                                     root='bucket/data')
    assert reader.tables() == ['a', 'b']
    assert reader.head('a', 1).to_pydict() == {'id': [1]}
    assert reader.head('b', 1).to_pydict() == {'id': [1]}
    assert reader.data_version('a')
    target = tmp_path / 'download.csv'
    transport.access.fetch('bucket/data/a.csv', str(target))
    assert target.read_bytes() == transport.contents['data/a.csv']
    assert transport.requests[0] == ('client', transport.credentials,
                                     transport.project)
    assert any(request[0] == 'range' for request in transport.requests)
    assert ('list', 'data/', 1, None) in transport.requests
    assert ('list', 'data/', None, '/') in transport.requests


@pytest.mark.parametrize('operation', ['metadata', 'read'])
def test_requester_pays_permission_errors_remain_typed(transport, operation):
    transport.failures[operation] = Forbidden('denied')
    reader = ObjectStoreSourceReader(transport.access.filesystem,
                                     root='bucket/data')
    with pytest.raises(SourceAccessDeniedError):
        reader.head('a', 1)


def test_requester_pays_missing_source_remains_typed(transport):
    reader = ObjectStoreSourceReader(transport.access.filesystem,
                                     root='bucket/data/missing.csv')
    with pytest.raises(SourceNotFoundError):
        reader.head('', 1)


def test_empty_bucket_is_a_directory(transport):
    from pyarrow.fs import FileType
    transport.contents.clear()
    assert transport.access.filesystem.get_file_info(
        'bucket').type == FileType.Directory


def test_identity_uses_fresh_generation_metadata_and_billing(transport):
    assert transport.access.object_identity('bucket/data/a.csv') == ('123', '7')
    assert transport.access.object_identity('bucket/data/a.csv') == ('123', '7')
    assert ('billing', 'billing-project') in transport.requests


def test_identity_preserves_access_errors(transport):
    # reload is the SDK metadata request in the real client.
    from unittest.mock import patch

    with patch('kumo_gcs.download.GCSDownloader.object_identity',
               side_effect=Forbidden('denied')):
        with pytest.raises(SourceAccessDeniedError):
            transport.access.object_identity('bucket/data/a.csv')
