# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry protocol and direct UC publication failure handling."""

import hashlib
import io
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from kumo_core.artifacts import verify_model_bundle
from kumo_databricks import registry
from kumo_databricks.sink import DatabricksSinkBackend
from kumoapi.data_source import DatabricksConnectorResourceConfig
from kumoapi.plugin import (
    ModelFetchingSink,
    ModelListingSink,
    ModelPublishingSink,
    ModelRegistryLocation,
    PublishedModel,
)


@pytest.fixture
def connector():
    return SimpleNamespace(
        config=DatabricksConnectorResourceConfig(catalog='main',
                                                 schema_name='analytics',
                                                 warehouse_id='warehouse'),
        id='dbx', identity='connector-identity')


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / 'bundle'
    root.mkdir()
    (root / 'training-spec.json').write_text(
        json.dumps(
            dict(spec_version=1, predictive_query='PREDICT COUNT(*)',
                 snapshot_id='snapshot')))
    (root / 'metrics.json').write_text('{"auc":0.9}')
    (root / 'weights.bin').write_bytes(b'model weights')
    artifacts = [
        dict(role=role, relative_path=name,
             size_bytes=(root / name).stat().st_size, sha256=hashlib.sha256(
                 (root / name).read_bytes()).hexdigest())
        for role, name in [(
            'training_spec',
            'training-spec.json'), ('selected_checkpoint',
                                    'weights.bin'), ('metrics', 'metrics.json')]
    ]
    (root / 'publication-manifest.json').write_text(
        json.dumps(dict(schema_version=1, artifacts=artifacts)))
    return root


def test_sink_implements_registry_protocols():
    sink = DatabricksSinkBackend()
    assert isinstance(sink, ModelPublishingSink)
    assert isinstance(sink, ModelFetchingSink)
    assert isinstance(sink, ModelListingSink)


def test_publish_uses_connector_client_and_reports_assigned_version(
        connector, bundle, monkeypatch):
    sink = DatabricksSinkBackend(identity='fallback')
    client = object()
    calls = []

    def resolve(target):
        assert target is connector
        return client

    def publish(resolved, name, bundle_dir, *, comment, metadata, metrics):
        assert resolved is client
        calls.append(
            dict(name=name, bundle_dir=bundle_dir, comment=comment,
                 metadata=metadata, metrics=metrics))
        return PublishedModel(name='churn', version='7',
                              uri='models:/main.analytics.churn/7',
                              metadata=metadata, metrics=metrics)

    monkeypatch.setattr(sink, '_client', resolve)
    monkeypatch.setattr(registry, 'publish_bundle', publish)
    published = sink.publish_model(
        connector, str(bundle),
        location=ModelRegistryLocation('churn', comment='GNN'),
        metadata={'task': 'churn'}, metrics={'auc': 0.9})
    assert published.version == '7'
    assert published.metrics == {'auc': 0.9}
    assert calls[0]['name'] == 'main.analytics.churn'
    assert calls[0]['bundle_dir'] == bundle
    assert calls[0]['comment'] == 'GNN'


@pytest.mark.parametrize(
    'name',
    ['other.analytics.churn', 'main.other.churn', 'other.churn', 'a.b.c.d', ''])
def test_cross_schema_or_invalid_names_are_refused(connector, name):
    with pytest.raises(ValueError):
        registry._address(connector, None, ModelRegistryLocation(name))


@pytest.mark.parametrize('kwargs', [
    dict(path='models/one'), {},
    dict(location=ModelRegistryLocation('churn', version='7'))
])
def test_invalid_publish_address_fails_before_authentication(
        connector, bundle, kwargs, monkeypatch):
    sink = DatabricksSinkBackend()
    monkeypatch.setattr(sink, '_client',
                        lambda _: pytest.fail('resolved identity'))
    with pytest.raises(ValueError):
        sink.publish_model(connector, str(bundle), **kwargs)


@pytest.mark.parametrize('marker',
                         ['training-spec.json', 'publication-manifest.json'])
def test_incomplete_bundles_are_refused(connector, bundle, marker, monkeypatch):
    (bundle / marker).unlink()
    sink = DatabricksSinkBackend()
    monkeypatch.setattr(sink, '_client',
                        lambda _: pytest.fail('resolved identity'))
    with pytest.raises(ValueError, match=marker):
        sink.publish_model(connector, str(bundle),
                           location=ModelRegistryLocation('churn'))


@pytest.mark.parametrize('number', ['0', '-1', 'latest', '1.5'])
def test_invalid_fetch_version_fails_before_authentication(
        connector, tmp_path, number, monkeypatch):
    sink = DatabricksSinkBackend()
    monkeypatch.setattr(sink, '_client',
                        lambda _: pytest.fail('resolved identity'))
    with pytest.raises(ValueError, match='positive integers'):
        sink.fetch_model(
            connector, str(tmp_path),
            location=ModelRegistryLocation('churn', version=number))


def version(number, kumo=True, status='READY'):
    tags = {registry._ENGINE_MARKER: '1'} if kumo else {}
    return dict(version=number, status=status,
                tags=[dict(key=k, value=v) for k, v in tags.items()])


class Client:
    def __init__(self):
        self.api_client = SimpleNamespace(do=self.do)
        self.registered_models = SimpleNamespace(create=self.create)
        self.model_versions = SimpleNamespace(delete=self.delete)
        self.files = SimpleNamespace(
            upload=self.upload, list_directory_contents=self.list_directory,
            download=self.download)
        self.objects = {}
        self.calls = []
        self.deleted = []
        self.versions = [
            version(2),
            version(10),
            version(11, kumo=False),
            version(12, status='PENDING_REGISTRATION')
        ]

    def create(self, **kwargs):
        from databricks.sdk.errors import ResourceAlreadyExists
        raise ResourceAlreadyExists('Already exists')

    def delete(self, name, number):
        self.deleted.append((name, number))

    def do(self, method, path, query=None, body=None):
        self.calls.append((method, path, query, body))
        if method == 'POST':
            assert all(len(tag['value']) <= 256 for tag in body['tags'])
            self.versions.append(
                dict(version=7, status='PENDING_REGISTRATION',
                     tags=body['tags']))
            return self.versions[-1]
        if method == 'PATCH':
            self.versions[-1]['status'] = 'READY'
            return self.versions[-1]
        if path.endswith('/models'):
            assert query['catalog_name'] == 'main'
            assert query['schema_name'] == 'analytics'
            return dict(registered_models=[
                dict(full_name='main.analytics.churn'),
                dict(full_name='main.other.foreign')
            ])
        if path.endswith('/versions'):
            if not query.get('page_token'):
                return dict(model_versions=[], next_page_token='second')
            return dict(model_versions=[
                dict(version=v['version']) for v in self.versions
            ])
        return next(v for v in self.versions
                    if str(v['version']) == path.rsplit('/', 1)[-1])

    def upload(self, path, stream, **kwargs):
        self.objects[path] = stream.read()

    def list_directory(self, path):
        children = {}
        for key in self.objects:
            if key.startswith(path + '/'):
                relative = key[len(path) + 1:]
                child = relative.split('/')[0]
                children[child] = SimpleNamespace(path=path + '/' + child,
                                                  is_directory='/' in relative)
        return children.values()

    def download(self, path):
        return SimpleNamespace(contents=io.BytesIO(self.objects[path]))


@pytest.fixture
def client():
    return Client()


def test_direct_publish_uploads_original_bundle_before_finalize(client, bundle):
    result = registry.publish_bundle(client, 'main.analytics.churn', bundle,
                                     comment='test', metadata={'task': 'churn'},
                                     metrics={'auc': 0.9})
    assert result.version == '7'
    assert result.metadata == {'task': 'churn'}
    assert result.metrics == {'auc': 0.9}
    assert client.objects[
        '/Models/main/analytics/churn/7/weights.bin'] == b'model weights'
    assert client.calls[0][:2] == ('POST',
                                   '/api/2.1/unity-catalog/models/versions')
    assert {
        path.rsplit('/', 1)[-1]: data
        for path, data in client.objects.items()
    } == {
        path.name: path.read_bytes()
        for path in bundle.iterdir()
    }
    assert client.calls[0][3]['comment'] == 'test'
    assert client.calls[0][3]['source'] == bundle.as_uri()
    assert {t['key'] for t in client.calls[0][3]['tags']} == {'kumo_engine'}
    assert client.calls[1][0] == 'PATCH'
    assert client.calls[1][3] == dict(full_name='main.analytics.churn',
                                      version=7)
    assert not client.deleted


@pytest.mark.parametrize('stage', ['upload', 'finalize', 'not_ready'])
def test_failed_publication_deletes_created_version(client, bundle, monkeypatch,
                                                    stage):
    def fail(*args, **kwargs):
        raise PermissionError('Registry access denied')

    if stage == 'upload':
        monkeypatch.setattr(client.files, 'upload', fail)
    else:
        original = client.api_client.do

        def do(method, *args, **kwargs):
            if method == 'PATCH':
                if stage == 'finalize':
                    fail()
                return dict(status='PENDING_REGISTRATION')
            return original(method, *args, **kwargs)

        monkeypatch.setattr(client.api_client, 'do', do)
    with pytest.raises((PermissionError, RuntimeError)):
        registry.publish_bundle(client, 'main.analytics.churn', bundle,
                                comment='test', metadata={'task': 'churn'},
                                metrics={'auc': 0.9})
    assert client.deleted == [('main.analytics.churn', 7)]


def test_cleanup_failure_preserves_original_error(client, bundle, monkeypatch):
    def upload(*args, **kwargs):
        raise PermissionError('upload failed')

    def delete(*args):
        raise OSError('cleanup failed')

    monkeypatch.setattr(client.files, 'upload', upload)
    monkeypatch.setattr(client.model_versions, 'delete', delete)
    with pytest.raises(PermissionError, match='upload failed'):
        registry.publish_bundle(client, 'main.analytics.churn', bundle,
                                comment='test', metadata={'task': 'churn'},
                                metrics={'auc': 0.9})


def test_publish_rejects_symlinks_before_creating_version(client, bundle):
    (bundle / 'link').symlink_to(bundle / 'weights.bin')
    with pytest.raises(ValueError, match='symbolic links'):
        registry.publish_bundle(client, 'main.analytics.churn', bundle,
                                comment='test', metadata={'task': 'churn'},
                                metrics={'auc': 0.9})
    assert not client.calls


def test_list_is_paginated_scoped_and_only_lists_ready_kumo_versions(client):
    found = registry.list_models(client, catalog='main', schema='analytics')
    assert [item.version for item in found] == ['2', '10']


def test_fetch_resolves_latest_numeric_version(client, bundle, tmp_path):
    prefix = '/Models/main/analytics/churn/10/artifacts/bundle/'
    client.objects = {prefix + p.name: p.read_bytes() for p in bundle.iterdir()}
    destination = tmp_path / 'restored'
    destination.mkdir()
    (destination / 'stale').write_text('old')
    result = registry.fetch_bundle(client, name='main.analytics.churn',
                                   local_dir=destination, version=None)
    assert result.version == '10'
    assert (destination / 'weights.bin').read_bytes() == b'model weights'
    assert not (destination / 'stale').exists()


@pytest.mark.parametrize('failure', ['download', 'traversal', 'missing_bundle'])
def test_failed_fetch_keeps_destination(client, tmp_path, monkeypatch, failure):
    destination = tmp_path / 'restored'
    destination.mkdir()
    (destination / 'weights.bin').write_bytes(b'previous')
    client.objects = {'/Models/main/analytics/churn/2/weights.bin': b'new'}
    if failure == 'download':

        def fail(*args):
            raise OSError('download failed')

        monkeypatch.setattr(client.files, 'download', fail)
    elif failure == 'traversal':
        monkeypatch.setattr(
            client.files, 'list_directory_contents', lambda _: [
                SimpleNamespace(path='/Models/main/analytics/churn/2/../escape',
                                is_directory=False)
            ])
    with pytest.raises((OSError, ValueError)):
        registry.fetch_bundle(client, name='main.analytics.churn',
                              local_dir=destination, version='2')
    assert (destination / 'weights.bin').read_bytes() == b'previous'


@pytest.mark.parametrize('failure', ['corruption', 'replace'])
def test_verified_fetch_failure_preserves_previous_bundle(
        client, bundle, tmp_path, monkeypatch, failure):
    prefix = '/Models/main/analytics/churn/2/'
    client.objects = {prefix + p.name: p.read_bytes() for p in bundle.iterdir()}
    destination = tmp_path / 'restored'
    destination.mkdir()
    (destination / 'weights.bin').write_bytes(b'previous')
    if failure == 'corruption':
        client.objects[prefix + 'weights.bin'] = b'broken model'
    else:
        original = registry.os.replace

        def replace(source, target):
            if str(source).endswith('/model'):
                raise OSError('replacement failed')
            return original(source, target)

        monkeypatch.setattr(registry.os, 'replace', replace)
    with pytest.raises((ValueError, OSError)):
        registry.fetch_bundle(client, name='main.analytics.churn',
                              local_dir=destination, version='2')
    assert (destination / 'weights.bin').read_bytes() == b'previous'


def test_corrupt_publication_fails_before_registry_calls(client, bundle):
    (bundle / 'weights.bin').write_bytes(b'broken model')
    with pytest.raises(ValueError, match='weights.bin'):
        registry.publish_bundle(client, 'main.analytics.churn', bundle,
                                comment='test', metadata={'task': 'churn'},
                                metrics={'auc': 0.9})
    assert not client.calls
    assert not client.objects


def test_full_catalog_data_roundtrips_without_oversized_tags(
        client, bundle, tmp_path):
    metadata = dict(spec_version='1', snapshot_id='snapshot-' + 'x' * 300,
                    predictive_query='PREDICT COUNT(*) ' + 'x' * 300)
    metrics = {f'test_metric_{i}': i / 100 for i in range(20)}
    (bundle / 'training-spec.json').write_text(
        json.dumps({
            **metadata, 'spec_version': 1
        }))
    (bundle / 'metrics.json').write_text(json.dumps(metrics))
    manifest_path = bundle / 'publication-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest['artifacts']:
        data = (bundle / entry['relative_path']).read_bytes()
        entry['size_bytes'] = len(data)
        entry['sha256'] = hashlib.sha256(data).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    original = {p.name: p.read_bytes() for p in bundle.iterdir()}
    supplied_metadata = {**metadata, 'custom_label': 'not persisted'}
    published = registry.publish_bundle(client, 'main.analytics.churn', bundle,
                                        metadata=supplied_metadata,
                                        metrics=metrics)
    tags = registry._tags(client.versions[-1])
    assert tags == {'kumo_engine': '1'}
    assert all(len(value) <= 256 for value in tags.values())
    assert {p.name: p.read_bytes() for p in bundle.iterdir()} == original
    found = registry.list_models(client, catalog='main', schema='analytics')
    discovered = next(item for item in found if item.version == '7')
    assert discovered.metadata == {}
    assert discovered.metrics == {}
    destination = tmp_path / 'restored'
    fetched = registry.fetch_bundle(client, name='main.analytics.churn',
                                    local_dir=destination, version='7')
    assert published.metadata == supplied_metadata
    assert fetched == replace(published, metadata=metadata)
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == original
    assert fetched.metadata == metadata
    assert fetched.metrics == metrics
    verify_model_bundle(destination)


@pytest.mark.parametrize('operation', ['list', 'latest', 'explicit'])
def test_disappearing_version_only_skipped_during_enumeration(
        client, bundle, tmp_path, monkeypatch, operation):
    from databricks.sdk.errors import NotFound
    original = client.api_client.do

    def do(method, path, **kwargs):
        if path.endswith('/versions/12'):
            raise NotFound('Version was deleted')
        return original(method, path, **kwargs)

    monkeypatch.setattr(client.api_client, 'do', do)
    if operation == 'list':
        found = registry.list_models(client, catalog='main', schema='analytics')
        assert [item.version for item in found] == ['2', '10']
    elif operation == 'latest':
        prefix = '/Models/main/analytics/churn/10/'
        client.objects = {
            prefix + p.name: p.read_bytes()
            for p in bundle.iterdir()
        }
        fetched = registry.fetch_bundle(client, name='main.analytics.churn',
                                        local_dir=tmp_path / 'restored',
                                        version=None)
        assert fetched.version == '10'
    else:
        with pytest.raises(NotFound):
            registry.fetch_bundle(client, name='main.analytics.churn',
                                  local_dir=tmp_path / 'restored', version='12')


@pytest.mark.parametrize('error_name', ['PermissionDenied', 'Unauthenticated'])
def test_enumeration_propagates_authorization_errors(client, monkeypatch,
                                                     error_name):
    from databricks.sdk import errors
    error = getattr(errors, error_name)
    original = client.api_client.do

    def do(method, path, **kwargs):
        if path.endswith('/versions/12'):
            raise error('Access denied')
        return original(method, path, **kwargs)

    monkeypatch.setattr(client.api_client, 'do', do)
    with pytest.raises(error):
        registry.list_models(client, catalog='main', schema='analytics')


@pytest.mark.parametrize('filename', ['training-spec.json', 'metrics.json'])
def test_invalid_bundle_metadata_keeps_destination(client, bundle, tmp_path,
                                                   filename):
    registry.publish_bundle(client, 'main.analytics.churn', bundle,
                            comment='test', metadata={'task': 'churn'},
                            metrics={'auc': 0.9})
    prefix = '/Models/main/analytics/churn/7/'
    client.objects[prefix + filename] = b'[]'
    manifest = json.loads(client.objects[prefix + 'publication-manifest.json'])
    entry = next(item for item in manifest['artifacts']
                 if item['relative_path'] == filename)
    entry['size_bytes'] = 2
    entry['sha256'] = hashlib.sha256(b'[]').hexdigest()
    client.objects[prefix +
                   'publication-manifest.json'] = json.dumps(manifest).encode()
    destination = tmp_path / 'restored'
    destination.mkdir()
    (destination / 'weights.bin').write_bytes(b'previous')
    with pytest.raises(ValueError):
        registry.fetch_bundle(client, name='main.analytics.churn',
                              local_dir=destination, version='7')
    assert (destination / 'weights.bin').read_bytes() == b'previous'
