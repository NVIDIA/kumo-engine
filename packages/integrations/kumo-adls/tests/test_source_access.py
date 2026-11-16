# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from azure.core.credentials import AccessToken
from fakes import FakeService
from kumo_adls import client as clients
from kumo_adls.backend import ADLSSourceBackend
from kumo_adls.reader import ADLSSourceReader, source_reader
from kumo_core.sources.types import ReadMode
from test_sdk_transfer import HEADERS, HTTPAdapter


@pytest.fixture
def no_auth_environment(monkeypatch):
    for key in list(os.environ):
        if key.startswith('AZURE_'):
            monkeypatch.delenv(key)


def parquet(value):
    stream = pa.BufferOutputStream()
    pq.write_table(pa.table({'id': [value]}), stream)
    return stream.getvalue().to_pybytes()


def reader_for(client):
    return source_reader(client, client.account_name, filesystem='curated',
                         prefix='orders')


@pytest.mark.usefixtures('no_auth_environment')
def test_shared_reader_matches_engine_rows_versions_and_borrowed_lifetime(
        monkeypatch, connector, tmp_path):
    credential = SimpleNamespace(closed=False, scopes=[])

    def get_token(*scopes, **kwargs):
        credential.scopes.append(scopes)
        return AccessToken('resolved-token', int(time.time()) + 3600)

    credential.get_token = get_token
    payload = parquet(7)
    adapter = HTTPAdapter()
    properties = dict(HEADERS, **{'Content-Length': str(len(payload))})
    adapter.register_uri('HEAD', re.compile(r'https://companylake\.'),
                         headers=properties)
    conditions = []

    def download(request, context):
        conditions.append(request.headers.get('If-Match'))
        requested = request.headers.get('x-ms-range', 'bytes=0-')
        start, end = requested.removeprefix('bytes=').split('-', 1)
        first, last = int(start), int(end) if end else len(payload) - 1
        context.headers.update({
            'Content-Length':
            str(last - first + 1),
            'Content-Range':
            f'bytes {first}-{last}/{len(payload)}'
        })
        context.status_code = 206
        return payload[first:last + 1]

    adapter.register_uri(
        'GET',
        re.compile(r'https://companylake\.[^/]+/curated/orders/events.parquet'),
        headers=properties, content=download)
    monkeypatch.setattr(clients, 'HTTPAdapter', lambda **kwargs: adapter)
    borrowed = clients.create_client(credential, 'companylake')
    close = Mock(wraps=borrowed.close)
    monkeypatch.setattr(borrowed, 'close', close)
    selected = Mock(return_value=borrowed)
    monkeypatch.setattr('kumo_adls.backend.session_for', selected)
    shared = reader_for(borrowed)
    engine, name = ADLSSourceBackend()._reader(connector(), 'events.parquet')
    selected.assert_called_once_with(None, 'companylake',
                                     'dfs.core.windows.net')
    assert type(shared) is type(engine) is ADLSSourceReader
    version = shared.data_version(name)
    assert version == engine.data_version(name)
    outputs = [
        reader.prepare(name, str(tmp_path / str(index)), columns=None,
                       mode=ReadMode.STREAMED, transform=False)
        for index, reader in enumerate((shared, engine))
    ]
    for output in outputs:
        assert output.version == version
        assert pq.read_table(list(output.paths)).to_pydict() == {'id': [7]}
    assert conditions and all(condition == '"stable"'
                              for condition in conditions)
    assert credential.scopes and all(
        scopes == ('https://storage.azure.com/.default', )
        for scopes in credential.scopes)
    del shared, engine
    gc.collect()
    close.assert_not_called()
    assert not credential.closed
    # The caller can still use and close its client after both readers disappear.
    assert reader_for(borrowed).data_version(name) == version
    borrowed.close()
    close.assert_called_once()
    assert not credential.closed


@pytest.mark.usefixtures('no_auth_environment')
def test_injected_clients_isolate_concurrent_identities_and_accounts(tmp_path):
    cases = []
    for account in ('companylake', 'otherlake'):
        for identity in (1, 2):
            value = len(cases) + 1
            client = FakeService(account)
            client.files['curated/orders/events.parquet'] = parquet(value)
            cases.append((account, identity, client, value))

    def read(case):
        account, identity, client, value = case
        reader = reader_for(client)
        version = reader.data_version('events.parquet')
        output = reader.prepare('events.parquet',
                                str(tmp_path / f'{account}-{identity}'),
                                columns=None, mode=ReadMode.STREAMED,
                                transform=False)
        assert pq.read_table(list(output.paths)).to_pydict() == {'id': [value]}
        assert output.version == version
        assert not client.closed
        return version

    with ThreadPoolExecutor(max_workers=4) as executor:
        versions = list(executor.map(read, cases))
    assert len(set(versions)) == 4
    for _, _, client, _ in cases:
        client.close()


@pytest.mark.usefixtures('no_auth_environment')
def test_engine_cleanup_does_not_close_injected_client(monkeypatch):
    from kumo_adls import session

    borrowed = FakeService()
    borrowed.files['curated/orders/events.parquet'] = parquet(1)
    reader = reader_for(borrowed)
    version = reader.data_version('events.parquet')
    owned = FakeService('ownedlake')
    credential = Mock()
    monkeypatch.setattr(session, 'credential_for', lambda *args:
                        (credential, 'https://storage.azure.com'))
    monkeypatch.setattr(session, 'create_client', lambda *args: owned)
    session.close()
    session.session_for(None, 'ownedlake', 'dfs.core.windows.net')
    session.close()
    assert owned.closed
    credential.close.assert_called_once()
    assert not borrowed.closed
    assert reader.data_version('events.parquet') == version


@pytest.mark.parametrize('prefix', ['', 'orders/team'])
def test_shared_reader_preserves_root_spelling(connector, monkeypatch, prefix):
    client = FakeService()
    monkeypatch.setattr('kumo_adls.backend.session_for', lambda *args: client)
    shared = source_reader(client, 'companylake', filesystem='curated',
                           prefix=prefix)
    engine = ADLSSourceBackend()._discovery(connector(prefix=prefix))
    assert shared.root == engine.root


@pytest.mark.parametrize('prefix', ['../outside', 'orders/', 'orders#fragment'])
def test_reader_rejects_noncanonical_prefix_before_client_use(prefix):
    client = Mock()
    with pytest.raises(ValueError):
        source_reader(client, 'companylake', filesystem='curated',
                      prefix=prefix)
    client.get_file_system_client.assert_not_called()


def test_shared_reader_imports_and_construction_do_not_load_identity_adapters():
    script = '''
import importlib.abc, os, sys
for key in list(os.environ):
    if key.startswith('AZURE_'): del os.environ[key]
forbidden=('azure.identity', 'kumo_adls.session', 'kumo_adls.backend',
           'kumo_adls.native_output', 'kumo_core.sources.principals', 'kumo.engine')
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname==name or fullname.startswith(name+'.') for name in forbidden):
            raise AssertionError('Unexpected identity import: '+fullname)
sys.meta_path.insert(0, Guard())
from kumo_adls import create_client, source_reader
from kumo_adls.source_access import source_access
from kumo_core.sources.types import ReadMode
class Credential:
    def get_token(self, *args, **kwargs):
        raise AssertionError('Unexpected authentication during construction')
client=create_client(Credential(), 'companylake')
reader=source_reader(client, 'companylake', filesystem='curated', prefix='orders')
assert reader.root=='curated@companylake.dfs.core.windows.net/orders'
assert source_access(client, 'companylake').filesystem is not None
assert not any(name in sys.modules for name in forbidden)
client.close()
'''
    completed = subprocess.run([sys.executable, '-c', script],
                               capture_output=True, text=True,
                               env=dict(os.environ,
                                        PYTHONPATH=os.pathsep.join(sys.path)))
    assert completed.returncode == 0, completed.stderr
