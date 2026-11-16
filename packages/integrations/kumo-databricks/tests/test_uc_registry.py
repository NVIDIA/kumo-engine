# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct UC requests and SDK transfers against a local Databricks stand-in."""

import builtins
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit

import pytest
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from databricks.sdk.credentials_provider import credentials_strategy
from kumo_databricks.sink import DatabricksSinkBackend
from kumoapi.data_source import DatabricksConnectorResourceConfig
from kumoapi.plugin import ModelRegistryLocation

NAME = 'main.analytics.churn'


@pytest.fixture
def server():
    objects, versions, records = {}, {}, []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, body, status=200):
            data = body if isinstance(body,
                                      bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def handle_request(self):
            path = unquote(urlsplit(self.path).path)
            token = self.headers.get('Authorization', '')
            identity = token.removeprefix('Bearer ').split('-')[0]
            if identity not in {'alice', 'bob'}:
                return self.send({'error_code': 'PERMISSION_DENIED'}, 403)
            with lock:
                records.append((identity, token, self.command, path))
            data = self.rfile.read(int(self.headers.get('Content-Length', '0')))
            if path.startswith('/api/2.0/fs/files/'):
                remote = path.removeprefix('/api/2.0/fs/files')
                if self.command == 'PUT':
                    objects[identity, remote] = data
                    return self.send({})
                return self.send(objects[identity, remote])
            if path.startswith('/api/2.0/fs/directories/'):
                remote = path.removeprefix('/api/2.0/fs/directories')
                children = {}
                for who, key in list(objects):
                    if who == identity and key.startswith(remote + '/'):
                        relative = key[len(remote) + 1:]
                        child = relative.split('/')[0]
                        children[child] = dict(path=remote + '/' + child,
                                               is_directory='/' in relative)
                return self.send(dict(contents=list(children.values())))
            payload = json.loads(data) if data else {}
            model = dict(full_name=NAME, name='churn', catalog_name='main',
                         schema_name='analytics')
            if path.endswith('/models'):
                if self.command == 'POST':
                    return self.send(model)
                return self.send(dict(registered_models=[model]))
            if path.endswith('/models/versions') and self.command == 'POST':
                assert payload['source'].startswith('file:')
                assert {'key': 'kumo_engine', 'value': '1'} in payload['tags']
                versions[identity] = dict(version=1,
                                          status='PENDING_REGISTRATION',
                                          tags=payload['tags'])
                return self.send(versions[identity])
            if path.endswith('/finalize'):
                assert any(who == identity for who, _ in objects)
                versions[identity]['status'] = 'READY'
                return self.send(versions[identity])
            if path.endswith('/versions'):
                return self.send(dict(model_versions=[dict(version=1)]))
            if path.endswith('/versions/1'):
                return self.send(versions[identity])
            return self.send({'error_code': 'NOT_FOUND', 'message': path}, 404)

        do_GET = handle_request
        do_POST = handle_request
        do_PUT = handle_request
        do_PATCH = handle_request

    http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    base = f'http://127.0.0.1:{http.server_port}'
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield base, objects, records
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


@pytest.mark.allow_hosts(['127.0.0.1'])
def test_concurrent_publish_fetch_and_list_preserve_identity_and_globals(
        server, tmp_path, monkeypatch):
    original_import = builtins.__import__

    def without_mlflow(name, *args, **kwargs):
        if name == 'mlflow' or name.startswith('mlflow.'):
            raise ImportError('MLflow is not installed')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_mlflow)
    monkeypatch.setenv('DATABRICKS_HOST', 'https://wrong.example')
    monkeypatch.setenv('DATABRICKS_TOKEN', 'wrong-token')
    before = dict(os.environ)
    base, objects, records = server

    def execute(identity):
        counter = 0

        @credentials_strategy('kumo-test', ['host'])
        def authentication(config):
            def headers():
                nonlocal counter
                counter += 1
                return {'Authorization': f'Bearer {identity}-{counter}'}

            return headers

        client = WorkspaceClient(
            config=Config(host=base, credentials_strategy=authentication,
                          disable_experimental_files_api_client=True,
                          retry_timeout_seconds=2, http_timeout_seconds=2))
        connector = SimpleNamespace(config=DatabricksConnectorResourceConfig(
            catalog='main', schema_name='analytics', warehouse_id='unused'))
        sink = DatabricksSinkBackend()
        sink._client = lambda _: client
        bundle = tmp_path / identity
        bundle.mkdir()
        metadata = dict(spec_version='1', predictive_query='x' * 300,
                        snapshot_id=identity)
        (bundle / 'training-spec.json').write_text(
            json.dumps({
                **metadata, 'spec_version': 1
            }))
        (bundle / 'metrics.json').write_text('{"auc":0.9}')
        (bundle / 'weights.bin').write_bytes(identity.encode())
        artifacts = [
            dict(
                role=role, relative_path=name,
                size_bytes=(bundle / name).stat().st_size,
                sha256=hashlib.sha256((bundle / name).read_bytes()).hexdigest())
            for role, name in [('training_spec', 'training-spec.json'
                                ), ('selected_checkpoint',
                                    'weights.bin'), ('metrics', 'metrics.json')]
        ]
        (bundle / 'publication-manifest.json').write_text(
            json.dumps(dict(schema_version=1, artifacts=artifacts)))
        published = sink.publish_model(connector, str(bundle),
                                       location=ModelRegistryLocation('churn'),
                                       metadata=metadata, metrics={'auc': 0.9})
        assert published.version == '1'
        assert published.metadata == metadata
        discovered, = sink.list_models(connector)
        assert discovered.version == published.version
        assert discovered.metadata == {}
        assert discovered.metrics == {}
        destination = tmp_path / f'{identity}-restored'
        fetched = sink.fetch_model(
            connector, str(destination),
            location=ModelRegistryLocation('churn', version=published.version))
        assert fetched == published
        assert {
            p.name: p.read_bytes()
            for p in destination.iterdir()
        } == {
            p.name: p.read_bytes()
            for p in bundle.iterdir()
        }
        assert (destination / 'weights.bin').read_bytes() == identity.encode()
        return counter

    with ThreadPoolExecutor(max_workers=2) as pool:
        counts = list(pool.map(execute, ['alice', 'bob']))
    assert all(count > 5 for count in counts)
    assert dict(os.environ) == before
    assert {identity for identity, _, _, _ in records} == {'alice', 'bob'}
    for identity in ('alice', 'bob'):
        tokens = [token for who, token, _, _ in records if who == identity]
        assert len(set(tokens)) == len(tokens)
        assert objects[(
            identity,
            '/Models/main/analytics/churn/1/weights.bin')] == identity.encode()
