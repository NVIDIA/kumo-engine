# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

import kumoapi.data_source as api
import pytest

from rest.config.connector import FileConnectorResourceConfig
from statedb.config.common import ConnectorID, GroupID, IdentityID, ProjectID
from statedb.config.connector import (
    ADLSConnectorResourceConfig,
    GCSConnectorResourceConfig,
)
from statedb.config.timestamp import ResourceTimestamps
from statedb.db import StateDB
from statedb.db.connectors import (
    ConnectorAlreadyExistsError,
    StateDBConnectorRepository,
)
from statedb.db.repositories import JsonRepository
from statedb.rest.connector_db import (
    ExternalConnectorDBEntry,
    InternalConnectorDBEntry,
)
from statedb.rest.project import ProjectDBEntry
from statedb.utils.errors import ConnectorDoesNotExistError

PROJECT_A = ProjectID("project_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
PROJECT_B = ProjectID("project_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
CONNECTOR_A = ConnectorID("connector_0000000000000000000000000000000a")
CONNECTOR_B = ConnectorID("connector_0000000000000000000000000000000b")
CONNECTOR_C = ConnectorID("connector_0000000000000000000000000000000c")


def _timestamps() -> ResourceTimestamps:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return ResourceTimestamps(created_at=now, updated_at=now)


def _connector(
    connector_id: ConnectorID,
    config,
    project_id: ProjectID = PROJECT_A,
) -> InternalConnectorDBEntry:
    return InternalConnectorDBEntry(
        id=connector_id,
        project_id=project_id,
        name=config.name,
        normalized_name=config.name.casefold(),
        config=config,
        timestamps=_timestamps(),
    )


def _assert_connector_equal(actual, expected) -> None:
    """Compare persisted connector semantics, independent of UTC tzinfo type."""
    assert actual.id == expected.id
    assert actual.project_id == expected.project_id
    assert actual.name == expected.name
    assert actual.normalized_name == expected.normalized_name
    assert actual.description == expected.description
    assert actual.kind == expected.kind
    assert actual.config == expected.config
    assert actual.timestamps.created_at == expected.timestamps.created_at
    assert actual.timestamps.updated_at == expected.timestamps.updated_at


async def _seed_projects(db: StateDB) -> None:
    projects = JsonRepository('projects', ProjectDBEntry)
    async with db.write_tx() as tx:
        for project_id in (PROJECT_A, PROJECT_B):
            await projects.upsert(
                tx,
                str(project_id),
                ProjectDBEntry(
                    id=project_id,
                    group_id=GroupID('group_' + 'a' * 32),
                    slug=f"project-{str(project_id)[-8:]}",
                    display_name=str(project_id),
                    normalized_display_name=str(project_id),
                    timestamps=_timestamps(),
                ),
            )


def test_legacy_flat_cloud_targets_hydrate_as_rooted_targets():
    gcs = GCSConnectorResourceConfig(name='gcs', bucket='customer-data',
                                     prefix='tables')
    adls = ADLSConnectorResourceConfig(
        name='adls',
        storage_account='account',
        filesystem='filesystem',
        prefix='tables',
        endpoint_suffix='dfs.core.usgovcloudapi.net',
    )

    assert gcs.target == api.GCSRootedConnectorTarget(bucket='customer-data',
                                                      prefix='tables')
    assert adls.target == api.ADLSRootedConnectorTarget(
        storage_account='account',
        filesystem='filesystem',
        prefix='tables',
        endpoint_suffix='dfs.core.usgovcloudapi.net',
    )


@pytest.mark.asyncio
async def test_connector_repository_create_get_list_and_delete(tmp_path):
    db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    await _seed_projects(db)
    repo = StateDBConnectorRepository()
    connector = _connector(
        CONNECTOR_A,
        FileConnectorResourceConfig(name="uploads", root_dir="/tmp/csv"),
    )

    async with db.write_tx() as tx:
        await repo.create(tx, connector)

    async with db.read_tx() as tx:
        _assert_connector_equal(await repo.get_required(tx, CONNECTOR_A),
                                connector)
        project_connectors = await repo.list(tx, project_id=PROJECT_A)
        assert len(project_connectors) == 1
        _assert_connector_equal(project_connectors[0], connector)
        assert await repo.list(tx, project_id=PROJECT_B) == []

    async with db.write_tx() as tx:
        with pytest.raises(ConnectorAlreadyExistsError):
            await repo.create(tx, connector)
        await repo.delete(tx, CONNECTOR_A)

    async with db.read_tx() as tx:
        assert await repo.get(tx, CONNECTOR_A) is None
        with pytest.raises(ConnectorDoesNotExistError):
            await repo.get_required(tx, CONNECTOR_A)
    await db.close()


@pytest.mark.asyncio
async def test_connector_repository_filters_exact_project(tmp_path):
    db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    await _seed_projects(db)
    repo = StateDBConnectorRepository()
    entries = [
        _connector(
            CONNECTOR_A,
            FileConnectorResourceConfig(name="csv", root_dir="/tmp/csv"),
        ),
        _connector(
            CONNECTOR_B,
            FileConnectorResourceConfig(name="parquet",
                                        root_dir="/tmp/parquet"),
            PROJECT_B,
        ),
        _connector(
            CONNECTOR_C,
            FileConnectorResourceConfig(name="files",
                                        root_dir="s3://bucket/root"),
        ),
    ]
    async with db.write_tx() as tx:
        for entry in entries:
            await repo.create(tx, entry)

    async with db.read_tx() as tx:
        assert set(await repo.list_ids(tx, project_id=PROJECT_A)) == {
            CONNECTOR_A,
            CONNECTOR_C,
        }
        assert await repo.list_ids(tx, project_id=PROJECT_B) == [CONNECTOR_B]
        assert [
            entry.id
            for entry in await repo.list(tx, root_dir="s3://bucket/root")
        ] == [CONNECTOR_C]
    await db.close()


@pytest.mark.asyncio
async def test_connector_repository_upsert_preserves_immutable_target(tmp_path):
    db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    await _seed_projects(db)
    repo = StateDBConnectorRepository()
    old = _connector(
        CONNECTOR_A,
        FileConnectorResourceConfig(name="uploads", root_dir="/tmp/old"),
    )
    invalid_target_change = _connector(
        CONNECTOR_A,
        FileConnectorResourceConfig(name="uploads", root_dir="/tmp/new"),
    )
    async with db.write_tx() as tx:
        await repo.create(tx, old)
        with pytest.raises(ValueError, match='target configuration'):
            await repo.upsert(tx, invalid_target_change)
        old.description = 'updated metadata'
        await repo.upsert(tx, old)
    async with db.read_tx() as tx:
        _assert_connector_equal(await repo.get_required(tx, CONNECTOR_A), old)
    await db.close()


@pytest.mark.asyncio
async def test_connector_repository_rejects_missing_project(tmp_path):
    db = await StateDB.open_sqlite(str(tmp_path / 'state.db'))
    repo = StateDBConnectorRepository()
    connector = _connector(
        CONNECTOR_A,
        FileConnectorResourceConfig(name='uploads', root_dir='/tmp/csv'),
    )
    async with db.write_tx() as tx:
        with pytest.raises(ValueError, match='project.*not active'):
            await repo.create(tx, connector)
    await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'config',
    [
        GCSConnectorResourceConfig(
            name=str(CONNECTOR_A),
            target=api.GCSRootedConnectorTarget(bucket='customer-data',
                                                prefix='tables'),
        ),
        GCSConnectorResourceConfig(name=str(CONNECTOR_A),
                                   target=api.GCSAnyURIConnectorTarget()),
        ADLSConnectorResourceConfig(
            name=str(CONNECTOR_A),
            target=api.ADLSRootedConnectorTarget(
                storage_account='account',
                filesystem='filesystem',
                prefix='tables',
                endpoint_suffix='dfs.core.usgovcloudapi.net',
            ),
        ),
        ADLSConnectorResourceConfig(name=str(CONNECTOR_A),
                                    target=api.ADLSAnyURIConnectorTarget()),
    ],
    ids=['gcs-rooted', 'gcs-any-uri', 'adls-rooted', 'adls-any-uri'],
)
async def test_cloud_connector_target_persistence_round_trip(tmp_path, config):
    db = await StateDB.open_sqlite(str(tmp_path / 'state.db'))
    await _seed_projects(db)
    repo = StateDBConnectorRepository()
    connector = ExternalConnectorDBEntry(
        id=CONNECTOR_A,
        project_id=PROJECT_A,
        identity_id=IdentityID.generate(),
        name='cloud',
        normalized_name='cloud',
        created_by='user',
        config=config,
        timestamps=_timestamps(),
    )

    async with db.write_tx() as tx:
        await repo.create(tx, connector)
    async with db.read_tx() as tx:
        persisted = await repo.get_required(tx, CONNECTOR_A)
        assert persisted.id == connector.id
        assert persisted.project_id == connector.project_id
        assert persisted.config.name == config.name
        assert persisted.config.type == config.type
        assert persisted.config.target == config.target
    await db.close()
