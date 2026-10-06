# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project-owned internal connector provisioning for local E2E fixtures."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import posixpath
from http import HTTPStatus
from pathlib import Path
from typing import Any

from kumoapi.source_table import S3RelativeSource

from io_utils import fsutil
from rest.config.api_command import FixtureDefinition
from rest.config.connector import FileConnectorResourceConfig
from rest.config.connector_db import InternalConnectorDBEntry
from rest.config.graph_v2 import GraphDefinition
from rest.config.table_v2 import S3SourceTable, TableDefinition
from statedb.config.common import ConnectorID, ProjectID
from statedb.config.timestamp import get_current_timestamps

_INTEG_DIR = Path(__file__).resolve().parent
_CONNECTOR_ARGS_PATH = (_INTEG_DIR /
                        'connector_args/test_e2e_connector_args.json')


def load_fixture_connectors(
    fixture: FixtureDefinition,
    project_id: ProjectID,
    fixture_root: Path | None = None,
) -> list[InternalConnectorDBEntry]:
    """Build test-only internal file connectors referenced by ``fixture``.

    Customer connectors cannot be credentialless file connectors in the final
    model. These local E2E datasets are Kumo-managed test inputs, so they use
    the structurally separate project-owned internal connector variant.
    """
    raw_args = json.loads(_CONNECTOR_ARGS_PATH.read_text(encoding='utf-8'))

    # The checked-in fixture uses descriptive aliases which are replaced by
    # ``randomize_fixture_ids_v2`` before runtime.  Keep these import-only
    # aliases as strings; constructing a ConnectorID here would incorrectly
    # treat legacy fixture data as a durable resource identifier.
    def fixture_source_root(raw_root: str) -> str:
        root = Path(raw_root)
        if fixture_root is not None and not root.is_absolute():
            try:
                root = fixture_root / root.relative_to('test/csv_data')
            except ValueError:
                pass
        return str(root)

    roots_by_original_id = {
        str(raw['config']['name']):
        fixture_source_root(raw['config']['root_dir'])
        for raw in raw_args
    }
    referenced_ids = {
        table.get_connector_id()
        for graph in fixture.graphs
        for table in graph.tables.values()
    }

    # Randomized fixture copies preserve table ordering but change connector
    # IDs. Match roots by source-table family when an original ID is absent.
    roots_by_source_name: dict[str, str] = {}
    for raw in raw_args:
        root = fixture_source_root(raw['config']['root_dir'])
        if 'FINANCIAL_E2E' in root:
            roots_by_source_name['LOAN'] = root
        elif 'COUPON_PURCHASE_PREDICTION' in root:
            roots_by_source_name['COUPON_'] = root
        else:
            roots_by_source_name['default'] = root

    root_by_connector: dict[ConnectorID, str] = {}
    for graph in fixture.graphs:
        for table in graph.tables.values():
            connector_id = table.get_connector_id()
            source_name = str(table.get_source_table_name())
            root = roots_by_original_id.get(str(connector_id))
            if root is None:
                if source_name == 'LOAN':
                    root = roots_by_source_name['LOAN']
                elif source_name.startswith(
                        'COUPON_') or source_name == 'USER_LIST':
                    root = roots_by_source_name['COUPON_']
                else:
                    root = roots_by_source_name['default']
            previous = root_by_connector.setdefault(connector_id, root)
            if previous != root:
                raise ValueError(
                    f'Connector {connector_id} maps to multiple local roots')

    if referenced_ids != set(root_by_connector):
        raise ValueError('Not every fixture connector has a local test root')

    return [
        InternalConnectorDBEntry(
            id=connector_id,
            project_id=project_id,
            name=str(connector_id),
            normalized_name=str(connector_id).casefold(),
            description='Project-owned local E2E fixture data',
            config=FileConnectorResourceConfig(
                name=str(connector_id),
                root_dir=root,
            ),
            timestamps=get_current_timestamps(),
        ) for connector_id, root in root_by_connector.items()
    ]


def _project_upload_fixture(
    fixture: FixtureDefinition,
    connector_id: ConnectorID,
    roots_by_connector: dict[ConnectorID, str],
) -> FixtureDefinition:
    """Move one local fixture onto the canonical project-upload connector.

    This helper is deliberately narrow. The generic integration fixture files
    model checked-in, connector-relative S3 tables backed by local directories.
    Provider and warehouse fixtures must use their real external connector
    provisioners instead of being disguised as project uploads.
    """

    def remap_table(table: TableDefinition) -> TableDefinition:
        source_table = table.source_table
        if not isinstance(source_table, S3SourceTable):
            raise ValueError(
                'Local fixture provisioning supports only S3 source tables; '
                'use the provider-specific external connector fixture')
        if not isinstance(source_table.source, S3RelativeSource):
            raise ValueError(
                'Local fixture provisioning supports only connector-relative '
                'S3 paths; use a supported external S3 connector for URIs')
        old_connector_id = ConnectorID(str(source_table.connector_id))
        if old_connector_id not in roots_by_connector:
            raise ValueError(
                f'Local fixture connector {old_connector_id} has no source '
                'directory')
        namespace = posixpath.join('fixture-data', str(old_connector_id))
        relative_path = posixpath.join(
            namespace, source_table.source.path.strip('/'))
        return dataclasses.replace(
            table,
            source_table=dataclasses.replace(
                source_table,
                connector_id=str(connector_id),
                source=dataclasses.replace(source_table.source,
                                           path=relative_path),
            ),
        )

    old_to_new_fpid = {}
    remapped_graphs = []
    for graph in fixture.graphs:
        remapped = GraphDefinition(
            **dataclasses.asdict(
                dataclasses.replace(
                    graph,
                    tables={
                        name: remap_table(table)
                        for name, table in graph.tables.items()
                    },
                )))
        old_to_new_fpid[graph.fpid] = remapped.fpid
        remapped_graphs.append(remapped)

    remapped_tables = [remap_table(table) for table in fixture.tables]
    remapped_queries = [
        dataclasses.replace(
            query,
            graph_fpid=old_to_new_fpid.get(query.graph_fpid,
                                           query.graph_fpid),
        ) for query in fixture.queries
    ]
    return dataclasses.replace(
        fixture,
        tables=remapped_tables,
        graphs=remapped_graphs,
        queries=remapped_queries,
    )


def state_services_for_test_client(client: Any):
    """Return the StateServices instance shared by an integration client."""

    state_services = getattr(client, 'state_services', None)
    if state_services is not None:
        return state_services
    app_builder = getattr(client, 'app_builder', None)
    state_services = (getattr(app_builder, '_state_services', None)
                      if app_builder is not None else None)
    if state_services is None:
        raise RuntimeError('Test app StateServices is not initialized')
    return state_services


async def ensure_test_project(
    client: Any,
    project_id: ProjectID | str | None = None,
) -> ProjectID:
    if project_id is not None:
        return ProjectID(str(project_id))
    state_services = state_services_for_test_client(client)
    _, project = await state_services.groups.ensure_default_resources()
    return project.id


def bind_test_client_project(client: Any, project_id: ProjectID | str) -> None:
    """Send the owning project on every later request from ``client``."""

    client.params = client.params.merge({'project_id': str(project_id)})


async def _canonical_project_upload_connector(
    client: Any,
    project_id: ProjectID,
    *,
    api_prefix: str,
) -> InternalConnectorDBEntry:
    """Resolve the one authorized project-upload connector for a test flow."""
    state_services = state_services_for_test_client(client)
    response = client.get(
        f'{api_prefix}/projects/{project_id}/connectors/uploads/connector')
    if response.status_code != HTTPStatus.OK:
        raise RuntimeError(
            'Failed to provision the project upload connector: '
            f'HTTP {response.status_code}')
    connector_id = ConnectorID(response.json()['connector_id'])
    return await state_services.connectors.get_internal(project_id,
                                                        connector_id)


async def provision_fixture_connectors(
    client: Any,
    fixture: FixtureDefinition,
    project_id: ProjectID | str | None = None,
    *,
    api_prefix: str = '/v0',
    fixture_root: Path | None = None,
) -> tuple[ProjectID, list[InternalConnectorDBEntry]]:
    """Upload checked-in local fixture data into the owning project.

    The supplied fixture is updated in place with the canonical upload
    connector ID and namespaced relative paths. This preserves the historical
    caller contract while ensuring child workflows never depend on arbitrary
    internal connectors.
    """

    typed_project_id = await ensure_test_project(client, project_id)
    source_connectors = load_fixture_connectors(
        fixture, typed_project_id, fixture_root)
    connector = await _canonical_project_upload_connector(
        client, typed_project_id, api_prefix=api_prefix)

    roots_by_connector = {
        source.id: source.config.root_dir for source in source_connectors
    }
    for old_connector_id, source_root in roots_by_connector.items():
        if not Path(source_root).is_dir():
            raise ValueError(
                f'Local fixture source directory does not exist: '
                f'{source_root}')
        namespace = posixpath.join('fixture-data', str(old_connector_id))
        destination = posixpath.join(connector.config.root_dir.rstrip('/'),
                                     namespace)
        await fsutil.copy.coroutine(
            source_root,
            destination,
            overwrite=True,
            raise_on_exists=False,
        )

    remapped = _project_upload_fixture(fixture, connector.id,
                                       roots_by_connector)
    fixture.tables[:] = remapped.tables
    fixture.graphs[:] = remapped.graphs
    fixture.queries[:] = remapped.queries
    fixture.predictions = remapped.predictions
    bind_test_client_project(client, typed_project_id)
    return typed_project_id, [connector]


def _duplicate_logical_source_path(source_path: str) -> str:
    """Return a distinct table path while preserving an explicit file suffix."""

    stem, suffix = posixpath.splitext(source_path)
    if suffix.casefold() in {'.csv', '.parquet'}:
        return f'{stem}-duplicate{suffix}'
    return f'{source_path}-duplicate'


async def _physical_fixture_source(
    file_connector: Any,
    source_table: S3SourceTable,
) -> tuple[str, str]:
    """Resolve a fixture's logical table path to its copied object path.

    Fixture tables normally use extensionless logical names such as
    ``ACCOUNT`` while the checked-in objects are ``ACCOUNT.csv`` and
    ``ACCOUNT.parquet``. Resolve through the canonical upload connector so
    duplication preserves its real lookup precedence (Parquet before CSV),
    rather than trusting the source-table declaration. The returned suffix is
    appended to the duplicate logical path.
    """

    source = source_table.source
    assert isinstance(source, S3RelativeSource)
    physical_path, resolved_file_type = (
        await file_connector.get_path_and_type.coroutine(source.path))
    if physical_path.endswith('/'):
        return physical_path, ''

    resolved_suffix = f'.{resolved_file_type.casefold()}'
    physical_suffix = ('' if source.path.casefold().endswith(resolved_suffix)
                       else resolved_suffix)
    return physical_path, physical_suffix


def _set_local_fixture_data_version(path: str, timestamp: int) -> None:
    """Set a local copied source's effective version without changing bytes."""

    path_obj = Path(path)
    targets = ([child for child in path_obj.rglob('*') if child.is_file()]
               if path_obj.is_dir() else [path_obj])
    for target in targets:
        target_stat = target.stat()
        os.utime(target, (target_stat.st_atime, timestamp))


async def _copy_fixture_source(
    physical_source: str,
    physical_destination: str,
) -> None:
    """Copy fixture bytes and require a distinct effective data version."""

    await fsutil.copy.coroutine(
        physical_source,
        physical_destination,
        overwrite=True,
        raise_on_exists=False,
    )
    source_modified, destination_modified = await asyncio.gather(
        asyncio.to_thread(fsutil.get_last_modified_time, physical_source),
        asyncio.to_thread(fsutil.get_last_modified_time,
                          physical_destination),
    )
    source_version = int(source_modified)
    destination_version = int(destination_modified)
    if source_version == destination_version:
        if not fsutil.is_local(physical_destination):
            raise RuntimeError(
                'Fixture copy did not create a distinct data version')
        await asyncio.to_thread(
            _set_local_fixture_data_version,
            physical_destination,
            source_version + 1,
        )
        destination_version = int(await asyncio.to_thread(
            fsutil.get_last_modified_time, physical_destination))
    if source_version == destination_version:
        raise RuntimeError(
            'Fixture copy did not create a distinct data version')


async def duplicate_fixture_sources(
    connector: InternalConnectorDBEntry,
    fixture: FixtureDefinition,
) -> dict[str, str]:
    """Duplicate every distinct file source used by public-API E2E graphs.

    The graph continues to use the canonical project-upload connector; only
    the relative source path changes. Copy physical CSV/Parquet objects while
    keeping their suffixes so connector lookup of the new logical path works.
    """

    duplicate_paths: dict[str, str] = {}
    copied_objects: set[tuple[str, str]] = set()
    # Internal project-upload connectors intentionally use a reserved name
    # which is not a customer connector identifier.  The resolver is only a
    # local helper object, so give it a valid disposable name while preserving
    # the configured root.
    file_connector = dataclasses.replace(
        connector.config, name='fixture_source_resolver').as_connector()
    for graph in fixture.graphs:
        for table in graph.tables.values():
            source_table = table.source_table
            if not isinstance(source_table, S3SourceTable):
                raise ValueError(
                    'Local fixture provisioning supports only S3 source '
                    'tables')
            source = source_table.source
            if not isinstance(source, S3RelativeSource):
                raise ValueError(
                    'Local fixture provisioning supports only '
                    'connector-relative S3 paths')

            old_path = source.path
            duplicate_path = duplicate_paths.setdefault(
                old_path, _duplicate_logical_source_path(old_path))
            physical_source, physical_suffix = await _physical_fixture_source(
                file_connector, source_table)
            physical_destination = (
                posixpath.join(connector.config.root_dir, duplicate_path)
                + physical_suffix)
            object_pair = (physical_source, physical_destination)
            if object_pair in copied_objects:
                continue
            await _copy_fixture_source(physical_source, physical_destination)
            copied_objects.add(object_pair)
    return duplicate_paths


async def provision_project_upload_connector(
    client: Any,
    *,
    source_root: str,
    api_prefix: str,
    project_id: ProjectID | str | None = None,
) -> tuple[ProjectID, InternalConnectorDBEntry]:
    """Seed fixture data behind the project's canonical upload connector.

    Public connector routes intentionally hide arbitrary internal connectors.
    Local integration data that must cross those routes belongs behind the
    one active project-upload connector, whose special visibility is derived
    from canonical StateDB metadata rather than a caller-controlled flag.
    """

    typed_project_id = await ensure_test_project(client, project_id)
    connector = await _canonical_project_upload_connector(
        client, typed_project_id, api_prefix=api_prefix)

    if await fsutil.exists_async(source_root):
        await fsutil.copy.coroutine(
            source_root,
            connector.config.root_dir,
            overwrite=True,
            raise_on_exists=False,
        )

    bind_test_client_project(client, typed_project_id)
    return typed_project_id, connector
