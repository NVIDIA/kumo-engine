# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import re
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pyarrow.parquet as pq
import pytest
from azure.core.exceptions import HttpResponseError
from fakes import FakeFile, FakeFilesystem
from kumo_adls.backend import ADLSSourceReader
from kumo_adls.sdk_filesystem import ADLSFileSystem
from kumo_core.sources.errors import TransientSourceError
from kumo_core.sources.types import ReadMode
from pyarrow.fs import FSSpecHandler, PyFileSystem
from test_sdk_transfer import HEADERS, HTTPAdapter, client

AUTHORITY = 'curated@companylake.dfs.core.windows.net'
PATHS = [AUTHORITY + '/orders/a.parquet', AUTHORITY + '/orders/b.parquet']


def reader(filesystem):
    return ADLSSourceReader(PyFileSystem(FSSpecHandler(filesystem)),
                            root=AUTHORITY + '/orders',
                            object_identity=filesystem.object_identity,
                            fetch=filesystem.fetch)


@pytest.fixture
def listing_service(service, monkeypatch):
    service.files['curated/orders/a.parquet'] = b'aaaa'
    service.files['curated/orders/b.parquet'] = b'bbbb'
    original = FakeFilesystem.get_paths
    calls = []

    def listing(self, path=None, recursive=True):
        calls.append((self.filesystem, path, recursive))
        for item in original(self, path=path, recursive=recursive):
            if not item.is_directory:
                item.etag = hashlib.sha256(
                    self.service.files[self.filesystem + '/' +
                                       item.name]).hexdigest()
            yield item

    monkeypatch.setattr(FakeFilesystem, 'get_paths', listing)
    return service, calls


def test_fresh_listing_replaces_heads_and_keeps_order(listing_service,
                                                      monkeypatch):
    service, calls = listing_service
    filesystem = ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')
    identities = [filesystem.object_identity(path) for path in PATHS]
    heads = Mock(side_effect=AssertionError('listing should avoid HEAD'))
    monkeypatch.setattr(FakeFile, 'get_file_properties', heads)
    source = reader(filesystem)
    found = [SimpleNamespace(path=path) for path in reversed(PATHS)]
    assert source._identities(found, 1) == list(reversed(identities))
    service.files['curated/orders/a.parquet'] = b'AAAA'
    changed = source._identities(found, 1)
    assert changed[0] == identities[1]
    assert changed[1] != identities[0]
    assert calls == [('curated', 'orders', True)] * 2
    heads.assert_not_called()


@pytest.mark.parametrize(
    'kind',
    ['etag', 'size', 'missing', 'duplicate', 'directory', 'denied', 'bound'])
def test_incomplete_or_unusable_listing_falls_back(listing_service, monkeypatch,
                                                   kind):
    service, _ = listing_service
    filesystem = ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')
    expected = [filesystem.object_identity(path) for path in PATHS]
    original = FakeFilesystem.get_paths

    def broken(self, **kwargs):
        if kind == 'denied':
            raise HttpResponseError(
                'denied', response=Mock(status_code=403, reason='denied',
                                        headers={}))
        items = list(original(self, **kwargs))
        selected = next(item for item in items
                        if item.name == 'orders/a.parquet')
        if kind == 'etag':
            selected.etag = None
        elif kind == 'size':
            selected.content_length = None
        elif kind == 'directory':
            selected.is_directory = True
        elif kind == 'missing':
            items.remove(selected)
        elif kind == 'duplicate':
            items.append(selected)
        elif kind == 'bound':
            items = [
                SimpleNamespace(name='orders/unrelated', is_directory=False,
                                etag='x', content_length=1)
            ] * 1025 + items
        yield from items

    monkeypatch.setattr(FakeFilesystem, 'get_paths', broken)
    assert filesystem.object_identities(PATHS,
                                        root=AUTHORITY + '/orders') is None
    assert reader(filesystem)._identities(
        [SimpleNamespace(path=path) for path in PATHS], 1) == expected


def test_scopes_nested_paths_and_never_scans_filesystem_root(listing_service):
    service, calls = listing_service
    filesystem = ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')
    nested = [
        AUTHORITY + '/orders/year=1/same.parquet',
        AUTHORITY + '/orders/year=2/same.parquet'
    ]
    service.files['curated/orders/year=1/same.parquet'] = b'one'
    service.files['curated/orders/year=2/same.parquet'] = b'two'
    result = filesystem.object_identities(nested, root=AUTHORITY + '/orders')
    assert result[0] != result[1]
    assert calls[-1] == ('curated', 'orders', True)
    count = len(calls)
    assert filesystem.object_identities(
        [AUTHORITY + '/one', AUTHORITY + '/two']) is None
    assert filesystem.object_identities(PATHS,
                                        root=AUTHORITY + '/private') is None
    assert filesystem.object_identities(PATHS[:1]) is None
    assert len(calls) == count


def test_custom_identity_callback_is_never_bypassed(listing_service):
    service, calls = listing_service
    filesystem = ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')
    source = reader(filesystem)
    source.object_identity = Mock(return_value=('custom', '4'))
    assert source._identities([SimpleNamespace(path=path) for path in PATHS],
                              1) == [('custom', '4')] * 2
    assert not calls
    assert source.object_identity.call_count == 2


@pytest.mark.parametrize('listing_identities', [False, True])
def test_prepare_stamps_and_reads_one_discovery(listing_service, tmp_path,
                                                monkeypatch,
                                                listing_identities):
    service, _ = listing_service
    # Keep source valid and use two parts so identities use listing.
    original = service.files.pop('curated/orders/events.parquet')
    service.files['curated/orders/a.parquet'] = original
    service.files['curated/orders/b.parquet'] = original
    filesystem = ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')
    if not listing_identities:
        monkeypatch.setattr(filesystem, 'object_identities',
                            Mock(return_value=None))
    source = reader(filesystem)
    version = source.data_version('')
    objects = ADLSSourceReader._objects
    versions = ADLSSourceReader._data_version
    listings = []
    stamps = []

    def listed(self, *args, **kwargs):
        found = objects(self, *args, **kwargs)
        listings.append(found)
        return found

    def stamp(self, path, found):
        stamps.append(found)
        return versions(self, path, found)

    monkeypatch.setattr(ADLSSourceReader, '_objects', listed)
    monkeypatch.setattr(ADLSSourceReader, '_data_version', stamp)
    monkeypatch.setattr(
        ADLSSourceReader, 'data_version',
        Mock(side_effect=AssertionError(
            'must not relist the table for its version')))
    fetch = ADLSFileSystem.fetch

    def add_part(self, path, destination):
        result = fetch(self, path, destination)
        service.files['curated/orders/added.parquet'] = original
        return result

    monkeypatch.setattr(ADLSFileSystem, 'fetch', add_part)
    # The callback must be bound after patching the SDK fetch method.
    source.fetch = filesystem.fetch
    output = source.prepare('', str(tmp_path / 'out'), columns=None,
                            mode=ReadMode.STREAMED)
    assert output.version == version
    assert len(listings) == len(stamps) == 1
    assert stamps[0] is listings[0]
    assert len(listings[0]) == 2
    assert pq.read_table(list(output.paths)).num_rows == 4
    assert 'curated/orders/added.parquet' in service.files


def test_real_sdk_json_pagination_and_quoted_head_etags():
    http = HTTPAdapter()
    list_calls = []
    heads = []

    def listing(request, context):
        query = parse_qs(urlsplit(request.url).query)
        assert query['directory'] == ['orders']
        assert query['recursive'] == ['true']
        continuation = query.get('continuation', [''])[0]
        list_calls.append(continuation)
        if not continuation:
            context.headers['x-ms-continuation'] = 'next-page'
        name = 'a' if not continuation else 'b'
        return json.dumps({
            'paths': [{
                'name': f'orders/{name}.parquet',
                'etag': 'stable',
                'contentLength': '3',
                'lastModified': HEADERS['Last-Modified'],
            }]
        }).encode()

    def head(request, context):
        heads.append(request.url)
        return b''

    def download(request, context):
        assert request.headers['If-Match'] == '"stable"'
        return b'abc'

    http.register_uri('GET',
                      re.compile(r'https://companylake\.dfs\.[^/]+/curated\?'),
                      headers={'Content-Type':
                               'application/json'}, content=listing)
    http.register_uri('HEAD', re.compile(r'https://companylake\.'), headers={
        **HEADERS, 'Content-Length': '3'
    }, content=head)
    http.register_uri(
        'GET', re.compile(r'https://companylake\.[^/]+/curated/orders/'),
        headers={
            **HEADERS, 'Content-Length': '3',
            'Content-Range': 'bytes 0-2/3'
        }, content=download)
    with client(http) as service:
        filesystem = ADLSFileSystem(service, 'companylake',
                                    'dfs.core.windows.net')
        assert filesystem.object_identities(PATHS) == [('stable', '3')] * 2
        assert list_calls == ['', 'next-page']
        assert not heads
        assert filesystem.object_identity(PATHS[0]) == ('stable', '3')
        with filesystem.open(PATHS[0], 'rb') as stream:
            assert stream.read() == b'abc'


def test_multiple_filesystems_keep_exact_authorities(listing_service):
    service, calls = listing_service
    filesystem = ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')
    other = [path.replace('curated@', 'another@') for path in PATHS]
    service.files['another/orders/a.parquet'] = b'other-a'
    service.files['another/orders/b.parquet'] = b'other-b'
    selected = [PATHS[0], other[0], PATHS[1], other[1]]
    assert filesystem.object_identities(selected) == [
        filesystem.object_identity(path) for path in selected
    ]
    assert calls == [('curated', 'orders', True), ('another', 'orders', True)]


def test_missing_selected_object_does_not_reuse_old_identity(listing_service):
    service, _ = listing_service
    filesystem = ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')
    source = reader(filesystem)
    found = [SimpleNamespace(path=path) for path in PATHS]
    assert source._identities(found, 1)
    del service.files['curated/orders/b.parquet']
    with pytest.raises(FileNotFoundError):
        source._identities(found, 1)


def test_non_sdk_filesystem_keeps_custom_identity_path(tmp_path):
    from pyarrow.fs import LocalFileSystem

    identity = Mock(return_value=('version', '4'))
    source = ADLSSourceReader(LocalFileSystem(), object_identity=identity)
    assert source._identities([SimpleNamespace(path=str(tmp_path / 'one'))],
                              1) == [('version', '4')]
    identity.assert_called_once()


@pytest.mark.parametrize('status', [401, 429, 503])
def test_listing_auth_and_transient_errors_propagate(listing_service,
                                                     monkeypatch, status):
    from kumo_core.sources.errors import ReconnectRequiredError

    service, _ = listing_service
    filesystem = ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')

    def failed(*args, **kwargs):
        raise HttpResponseError(
            'failed', response=Mock(status_code=status, reason='failed',
                                    headers={}))

    monkeypatch.setattr(FakeFilesystem, 'get_paths', failed)
    expected = ReconnectRequiredError if status == 401 else TransientSourceError
    with pytest.raises(expected):
        filesystem.object_identities(PATHS)
