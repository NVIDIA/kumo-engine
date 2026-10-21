# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unity Catalog model bundles, using the connector's own identity."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, cast
from urllib.parse import quote

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, ResourceAlreadyExists
from kumo_databricks.session import _DatabricksSessions, _target_config
from kumo_databricks.sql import normalize_table_coordinate
from kumoapi.plugin import (
    ModelRegistryLocation,
    PublishedModel,
    SourceConnector,
)

_ENGINE_MARKER = 'kumo_engine'
_MODELS = '/api/2.1/unity-catalog/models'
_LOG = logging.getLogger(__name__)


def _address(connector: SourceConnector, path: Optional[str],
             location: Optional[ModelRegistryLocation]) -> str:
    if path is not None:
        raise ValueError('Databricks models take a ModelRegistryLocation, '
                         'rather than a path.')
    if location is None:
        raise ValueError('Pass a ModelRegistryLocation naming the model.')
    config = _target_config(connector)
    schema, name = normalize_table_coordinate(location.model_name,
                                              catalog=config.catalog,
                                              schema=config.schema_name)
    return f'{config.catalog}.{schema}.{name}'


def _version_path(name: str, number: int | str) -> str:
    return f'{_MODELS}/{quote(name, safe="")}/versions/{int(number)}'


def _files_path(name: str, number: int | str) -> str:
    return f'/Models/{name.replace(".", "/")}/{int(number)}'


def _tags(version: Mapping[str, Any]) -> dict[str, str]:
    return {tag['key']: tag['value'] for tag in version.get('tags', [])}


def _published(
    name: str,
    version: Mapping[str, Any],
    *,
    metadata: Mapping[str, str] = {},
    metrics: Mapping[str, float] = {},
) -> PublishedModel:
    _validate_catalog(dict(metadata=dict(metadata), metrics=dict(metrics)))
    number = str(version['version'])
    return PublishedModel(name=name.rsplit('.', 1)[-1], version=number,
                          uri=f'models:/{name}/{number}',
                          metadata=dict(metadata), metrics=dict(metrics))


def _pages(client: WorkspaceClient, path: str, field: str,
           **query: str) -> Iterator[dict[str, Any]]:
    while True:
        page = cast(dict[str, Any],
                    client.api_client.do('GET', path, query=query))
        yield from page.get(field, [])
        token = page.get('next_page_token')
        if not token:
            return
        query['page_token'] = token


def _get_version(client: WorkspaceClient, name: str,
                 number: int | str) -> dict[str, Any]:
    return cast(dict[str, Any],
                client.api_client.do('GET', _version_path(name, number)))


def _versions(client: WorkspaceClient, name: str) -> Iterator[dict[str, Any]]:
    # List responses omit tags; get each version's enriched metadata.
    path = f'{_MODELS}/{quote(name, safe="")}/versions'
    for version in _pages(client, path, 'model_versions'):
        try:
            yield _get_version(client, name, version['version'])
        except NotFound:
            # A version may be deleted after enumeration, including cleanup.
            continue


def _validate_catalog(catalog: object) -> None:
    if not isinstance(catalog, dict) or set(catalog) != {'metadata', 'metrics'}:
        raise ValueError('Invalid Kumo registry metadata or metrics.')
    metadata, metrics = catalog['metadata'], catalog['metrics']
    if (not isinstance(metadata, dict) or not all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in metadata.items()) or not isinstance(metrics, dict)
            or not all(
                isinstance(k, str) and isinstance(v, (int, float))
                for k, v in metrics.items())):
        raise ValueError('Invalid Kumo registry metadata or metrics.')


def publish_bundle(
    client: WorkspaceClient,
    name: str,
    bundle_dir: Path,
    *,
    comment: Optional[str] = None,
    metadata: Mapping[str, str] = {},
    metrics: Mapping[str, float] = {},
) -> PublishedModel:
    from kumo_core.artifacts import verify_model_bundle

    root = bundle_dir
    files = []
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Model bundles must not contain symbolic links.')
        if path.is_file():
            files.append(path)
    verify_model_bundle(root)
    _validate_catalog(dict(metadata=dict(metadata), metrics=dict(metrics)))
    catalog_name, schema, model = name.split('.')
    tags = {_ENGINE_MARKER: '1'}
    try:
        client.registered_models.create(catalog_name=catalog_name,
                                        schema_name=schema, name=model)
    except ResourceAlreadyExists:
        # Publishing another version reuses the registered model.
        pass
    # UC requires source provenance even though we upload the files ourselves.
    body = dict(catalog_name=catalog_name, schema_name=schema, model_name=model,
                source=bundle_dir.resolve().as_uri(),
                tags=[dict(key=k, value=v) for k, v in tags.items()])
    if comment is not None:
        body['comment'] = comment
    version = cast(
        dict[str, Any],
        client.api_client.do('POST', f'{_MODELS}/versions', body=body))
    number = version['version']
    try:
        remote = _files_path(name, number)
        for path in files:
            with path.open('rb') as stream:
                client.files.upload(
                    f'{remote}/{path.relative_to(root).as_posix()}', stream,
                    overwrite=True)
        version = cast(
            dict[str, Any],
            client.api_client.do('PATCH',
                                 f'{_version_path(name, number)}/finalize',
                                 body=dict(full_name=name,
                                           version=int(number))))
        if version.get('status') != 'READY':
            raise RuntimeError(
                'Databricks model registration did not reach READY.')
    except Exception:
        try:
            client.model_versions.delete(name, int(number))
        except Exception:
            _LOG.warning('Could not clean up model version %s/%s', name, number,
                         exc_info=True)
        raise
    # Finalize need not return tags; we just wrote these on creation.
    version['tags'] = body['tags']
    return _published(name, version, metadata=metadata, metrics=metrics)


def _bundle_metadata(bundle: Path) -> tuple[dict[str, str], dict[str, float]]:
    spec = json.loads((bundle / 'training-spec.json').read_text())
    if (not isinstance(spec, dict)
            or not isinstance(spec.get('spec_version'), int)
            or not isinstance(spec.get('predictive_query'), str)
            or not isinstance(spec.get('snapshot_id'), str)):
        raise ValueError('Invalid training specification in model bundle.')
    metadata = {
        key: str(spec[key])
        for key in ('spec_version', 'predictive_query', 'snapshot_id')
    }
    metrics_path = bundle / 'metrics.json'
    metrics = json.loads(
        metrics_path.read_text()) if metrics_path.is_file() else {}
    catalog = dict(metadata=metadata, metrics=metrics)
    _validate_catalog(catalog)
    return metadata, metrics


def _download(client: WorkspaceClient, remote: str, destination: Path) -> None:
    for entry in client.files.list_directory_contents(remote):
        # A server-provided path must stay an immediate child of this directory.
        path = (entry.path or '').rstrip('/')
        prefix = remote + '/'
        child = path[len(prefix):] if path.startswith(prefix) else ''
        if not child or '/' in child or child in ('.', '..') or '\\' in child:
            raise ValueError(
                'Invalid model artifact path returned by Databricks.')
        local = destination / child
        if entry.is_directory:
            local.mkdir()
            _download(client, path, local)
        else:
            response = client.files.download(path)
            if response.contents is None:
                raise OSError(f'No content returned for model artifact {path}.')
            with response.contents as stream, local.open('wb') as output:
                shutil.copyfileobj(stream, output)


def fetch_bundle(client: WorkspaceClient, name: str, local_dir: Path, *,
                 version: Optional[str] = None) -> PublishedModel:
    from kumo_core.artifacts import verify_model_bundle

    if version is None:
        versions = [
            v for v in _versions(client, name) if v.get('status') == 'READY'
            and _tags(v).get(_ENGINE_MARKER) == '1'
        ]
        if not versions:
            raise ValueError('No ready Kumo model version in this registry.')
        model_version = max(versions, key=lambda v: int(v['version']))
    else:
        model_version = _get_version(client, name, version)
    if model_version.get('status') != 'READY' or _tags(model_version).get(
            _ENGINE_MARKER) != '1':
        raise ValueError('This version is not a ready Kumo model bundle.')
    destination = local_dir
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent,
                                     prefix='kumo-registry-download-') as tmp:
        root = Path(tmp) / 'model'
        root.mkdir()
        _download(client, _files_path(name, model_version['version']), root)
        bundles = [
            p.parent for p in root.rglob('training-spec.json')
            if (p.parent / 'publication-manifest.json').is_file()
        ]
        if len(bundles) != 1:
            raise ValueError(
                'The registered model must contain one Kumo bundle.')
        bundle = bundles[0]
        verify_model_bundle(bundle)
        published = _published(name, model_version)
        metadata, metrics = _bundle_metadata(bundle)
        published = replace(published, metadata=metadata, metrics=metrics)
        backup = Path(tmp) / 'previous'
        if destination.exists():
            os.replace(destination, backup)
        try:
            os.replace(bundle, destination)
        except Exception:
            if backup.exists():
                os.replace(backup, destination)
            raise
    return published


def list_models(client: WorkspaceClient, catalog: str,
                schema: str) -> list[PublishedModel]:
    found: list[PublishedModel] = []
    for model in _pages(client, _MODELS, 'registered_models',
                        catalog_name=catalog, schema_name=schema):
        name = model['full_name']
        prefix = f'{catalog}.{schema}.'
        if not name.startswith(prefix):
            continue
        for version in _versions(client, name):
            if version.get('status') == 'READY' and _tags(version).get(
                    _ENGINE_MARKER) == '1':
                found.append(_published(name, version))
    return sorted(found, key=lambda item: (item.name, int(item.version)))


class DatabricksModelRegistryMixin(_DatabricksSessions):
    """Publish, discover and reload bundles; prediction remains in Kumo."""
    def publish_model(
        self,
        connector: SourceConnector,
        bundle_dir: str,
        *,
        path: Optional[str] = None,
        location: Optional[ModelRegistryLocation] = None,
        metadata: Mapping[str, str] = {},
        metrics: Mapping[str, float] = {},
        project_id: Optional[str] = None,
    ) -> PublishedModel:
        """Register a bundle with an automatically assigned numeric version."""
        del project_id
        name = _address(connector, path, location)
        assert location is not None
        if location.version is not None:
            raise ValueError('Databricks assigns numeric model versions; '
                             'leave ModelRegistryLocation.version unset '
                             'when publishing.')
        bundle = Path(bundle_dir).resolve()
        for marker in ('training-spec.json', 'publication-manifest.json'):
            if not (bundle / marker).is_file():
                raise ValueError(f'{bundle} is not a published Kumo bundle: '
                                 f'missing {marker}.')
        return publish_bundle(self._client(connector), name, bundle,
                              comment=location.comment, metadata=metadata,
                              metrics=metrics)

    def fetch_model(
        self,
        connector: SourceConnector,
        local_dir: str,
        *,
        path: Optional[str] = None,
        location: Optional[ModelRegistryLocation] = None,
        project_id: Optional[str] = None,
    ) -> PublishedModel:
        """Download the named version, or the latest ready Kumo version."""
        del project_id
        name = _address(connector, path, location)
        assert location is not None
        if location.version is not None and (not location.version.isdecimal()
                                             or int(location.version) < 1):
            raise ValueError('Databricks model versions are positive integers.')
        return fetch_bundle(self._client(connector), name,
                            Path(local_dir).resolve(), version=location.version)

    def list_models(self, connector: SourceConnector, *,
                    project_id: Optional[str] = None) -> list[PublishedModel]:
        """List Kumo versions inside this connector's catalog and schema."""
        del project_id
        config = _target_config(connector)
        return list_models(self._client(connector), config.catalog,
                           config.schema_name)
