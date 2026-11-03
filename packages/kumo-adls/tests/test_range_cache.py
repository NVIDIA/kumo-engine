# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from azure.core.exceptions import ResourceModifiedError
from fakes import FakeFile
from kumo_adls.backend import ADLSSourceBackend
from kumo_adls.range_cache import MAX_BYTES, MAX_ENTRIES, RangeBytes
from kumo_adls.sdk_filesystem import ADLSFileSystem
from kumo_core.sources.errors import TransientSourceError

AUTHORITY = 'curated@companylake.dfs.core.windows.net'
PATH = AUTHORITY + '/orders/events.parquet'


@pytest.fixture
def cache(monkeypatch):
    value = RangeBytes()
    monkeypatch.setattr('kumo_adls.sdk_filesystem.RANGE_BYTES', value)
    return value


def filesystem(service):
    return ADLSFileSystem(service, 'companylake', 'dfs.core.windows.net')


def read_four(fs, path=PATH):
    with fs.open(path, 'rb', cache_type='none') as stream:
        return stream.read(4)


def test_reopened_file_checks_properties_and_changed_etag_misses(
        service, cache, monkeypatch):
    fs = filesystem(service)
    heads = []
    original = FakeFile.get_file_properties

    def head(self):
        heads.append(self.name)
        return original(self)

    monkeypatch.setattr(FakeFile, 'get_file_properties', head)
    first = read_four(fs)
    initial_heads = len(heads)
    assert read_four(fs) == first
    assert len(heads) > initial_heads  # Cached bytes never replace open's HEAD.
    assert len(service.downloads) == 1
    content = service.files['curated/orders/events.parquet']
    service.files['curated/orders/events.parquet'] = b'NEWW' + content[4:]
    assert read_four(fs) == b'NEWW'
    assert len(service.downloads) == 2


def test_open_snapshot_can_reuse_old_bytes_but_identity_guard_rejects_mutation(
        service, connector, cache, tmp_path):
    reader, name = ADLSSourceBackend()._reader(connector(), 'events')
    fs = reader.filesystem.handler.fs
    found = reader._objects(name, PATH)
    identities = reader._identities(found, 1)
    destination = tmp_path / 'staging'
    destination.mkdir()
    (destination / 'partial.parquet').write_bytes(b'partial')
    with fs.open(PATH, 'rb', cache_type='none') as stream:
        before = stream.read(4)
        content = service.files['curated/orders/events.parquet']
        service.files['curated/orders/events.parquet'] = b'NEWW' + content[4:]
        stream.seek(0)
        assert stream.read(4) == before
        assert len(service.downloads) == 1
        with pytest.raises(TransientSourceError, match='changed'):
            reader._unchanged(name, found, str(destination), 1, identities)
    assert not destination.exists()


@pytest.mark.parametrize('response', ['error', 'short', 'empty'])
def test_errors_and_incomplete_reads_never_enter_cache(service, cache,
                                                       monkeypatch, response):
    def broken(self, **kwargs):
        if response == 'error':
            raise ResourceModifiedError('changed')
        return SimpleNamespace(
            readall=lambda: b'ab' if response == 'short' else b'')

    monkeypatch.setattr(FakeFile, 'download_file', broken)
    fs = filesystem(service)
    for _ in range(2):
        with pytest.raises(TransientSourceError):
            read_four(fs)
    assert not cache._entries
    assert cache._bytes == 0


def test_zero_and_large_ranges_not_admitted(service, cache):
    service.files['curated/orders/events.parquet'] = b'a' * (128 * 1024 + 1)
    fs = filesystem(service)
    with fs.open(PATH, 'rb', cache_type='none') as stream:
        assert stream._fetch_range(0, 0) == b''
        assert not service.downloads
        assert len(stream.read()) == 128 * 1024 + 1
    assert not cache._entries


def test_raw_etags_and_full_authorities_are_distinct(service, cache,
                                                     monkeypatch):
    fs = filesystem(service)
    original = FakeFile.get_file_properties
    token = ['"opaque"']
    calls = []

    def properties(self):
        value = original(self)
        value.etag = token[0]
        return value

    def download(self, offset=None, length=None, **kwargs):
        calls.append(kwargs['etag'])
        return SimpleNamespace(readall=lambda: self.service.files[self.name][
            offset:offset + length])

    monkeypatch.setattr(FakeFile, 'get_file_properties', properties)
    monkeypatch.setattr(FakeFile, 'download_file', download)
    assert read_four(fs)
    token[0] = 'opaque'
    assert read_four(fs)
    service.files['another/orders/events.parquet'] = b'else'
    assert read_four(fs, PATH.replace('curated@', 'another@')) == b'else'
    assert calls == ['"opaque"', 'opaque', 'opaque']
    assert len(cache._entries) == 3


def test_filesystems_do_not_share_bytes_and_gc_releases_namespace(
        service, cache):
    one = filesystem(service)
    two = filesystem(service)
    assert read_four(one) == read_four(two)
    assert len(service.downloads) == 2
    namespace = one._range_namespace
    reference = weakref.ref(one)
    assert any(key[0] is namespace for key in cache._entries)
    del one
    gc.collect()
    assert reference() is None
    assert not any(key[0] is namespace for key in cache._entries)
    assert any(key[0] is two._range_namespace for key in cache._entries)


def test_process_byte_and_entry_caps_are_global_and_thread_safe():
    cache = RangeBytes(max_bytes=32, max_entries=3, max_entry_bytes=16)
    namespaces = [object() for _ in range(8)]

    def put(index):
        key = (namespaces[index % 8], f'file-{index}', '"etag"', 0, 12)
        cache.put(key, b'x' * 12)
        cache.get(key)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(put, range(64)))
    assert cache._bytes == sum(map(len, cache._entries.values()))
    assert cache._bytes <= 32
    assert len(cache._entries) <= 3
    assert MAX_BYTES == 64 << 20
    assert MAX_ENTRIES == 1024


def test_lru_updates_on_hits_replacements_and_namespace_cleanup():
    cache = RangeBytes(max_bytes=12, max_entries=2, max_entry_bytes=8)
    namespace = object()
    keys = [(namespace, str(index), '"etag"', 0, 4) for index in range(3)]
    cache.put(keys[0], b'aaaa')
    cache.put(keys[1], b'bbbb')
    assert cache.get(keys[0]) == b'aaaa'
    cache.put(keys[2], b'cccc')
    assert cache.get(keys[1]) is None
    cache.put(keys[0], b'AAAA')
    assert cache._bytes == 8
    cache.put((namespace, 'oversize', 'etag', 0, 9), b'x' * 9)
    cache.put((namespace, 'short', 'etag', 0, 4), b'x')
    assert cache._bytes == 8
    cache.clear_namespace(namespace)
    assert cache._bytes == 0
    assert not cache._entries


def test_reads_past_eof_use_pinned_size_and_do_not_cache_empty_ranges(
        service, cache):
    service.files['curated/orders/events.parquet'] = b'abc'
    fs = filesystem(service)
    with fs.open(PATH, 'rb', cache_type='none') as stream:
        assert stream._fetch_range(0, 50000) == b'abc'
        assert stream._fetch_range(0, 50000) == b'abc'
        assert stream._fetch_range(3, 50000) == b''
    assert len(service.downloads) == 1
    assert next(iter(cache._entries))[-2:] == (0, 3)
