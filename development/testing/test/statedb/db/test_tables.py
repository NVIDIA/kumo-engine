# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

import kumoapi.data_source as api
import pytest
from kumoapi.typing import Dtype, Stype

import statedb.db.connector_admission as connector_admission_module
from rest.config.connector import (
    FileConnectorResourceConfig,
    GCSConnectorResourceConfig,
)
from rest.config.table_v2 import (
    Column,
    DataSourceType,
    GCSSourceTable,
    S3SourceTable,
    TableDBEntryV2,
    TableDefinition,
)
from statedb.config.common import (
    ConnectorID,
    GroupID,
    IdentityGroupID,
    IdentityID,
    ProjectID,
)
from statedb.config.timestamp import ResourceTimestamps
from statedb.db import StateDB
from statedb.db.connectors import StateDBConnectorRepository
from statedb.db.repositories import JsonRepository
from statedb.db.tables import StateDBTableRepository
from statedb.rest.connector_db import (
    ExternalConnectorDBEntry,
    InternalConnectorDBEntry,
)
from statedb.rest.group import KumoGroupDBEntry
from statedb.rest.identity import IdentityDBEntry, IdentityGroupEntry
from statedb.rest.project import ProjectDBEntry

GROUP_ID = GroupID('group_' + 'a' * 32)
PROJECT_A = ProjectID('project_' + '1' * 32)
PROJECT_B = ProjectID('project_' + '2' * 32)
CONNECTOR_A = ConnectorID('connector_' + '1' * 32)
CONNECTOR_B = ConnectorID('connector_' + '2' * 32)
CONNECTOR_ANY_URI = ConnectorID('connector_' + '3' * 32)
IDENTITY_ANY_URI = IdentityID('identity_' + '3' * 32)


def _timestamps() -> ResourceTimestamps:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return ResourceTimestamps(created_at=now, updated_at=now)


def _entry(project_id: ProjectID, label: str, alias: str,
           ref_count: int = 1) -> TableDBEntryV2:
    connector_id = CONNECTOR_A if project_id == PROJECT_A else CONNECTOR_B
    definition = TableDefinition(
        cols=[Column(name='a', stype=Stype.ID, dtype=Dtype.int)],
        source_table=S3SourceTable(
            connector_id=connector_id,
            source={'mode': 'relative', 'path': f'{label}.parquet'},
            file_type='PARQUET',
            data_source_type=DataSourceType.S3,
        ),
    )
    return TableDBEntryV2(
        id=definition.fpid,
        project_id=project_id,
        config=definition,
        col_dtypes={},
        alias=alias,
        ref_count=ref_count,
        timestamps=_timestamps(),
    )


def test_table_entry_rejects_noncanonical_definition_fingerprint() -> None:
    canonical = _entry(PROJECT_A, 'canonical', 'canonical')
    with pytest.raises(RuntimeError, match='does not match'):
        TableDBEntryV2(
            id=canonical.id,
            project_id=PROJECT_A,
            config=_entry(PROJECT_A, 'different', 'different').config,
            col_dtypes={},
            timestamps=_timestamps(),
        )


async def _seed_projects_and_connectors(db: StateDB) -> None:
    projects = JsonRepository('projects', ProjectDBEntry)
    groups = JsonRepository('groups', KumoGroupDBEntry)
    connectors = StateDBConnectorRepository()
    async with db.write_tx() as tx:
        await groups.upsert(
            tx,
            str(GROUP_ID),
            KumoGroupDBEntry(
                id=GROUP_ID,
                slug='test-group',
                display_name='Test group',
                normalized_display_name='test group',
                timestamps=_timestamps(),
            ),
        )
        for project_id, connector_id in ((PROJECT_A, CONNECTOR_A),
                                         (PROJECT_B, CONNECTOR_B)):
            await projects.upsert(
                tx,
                str(project_id),
                ProjectDBEntry(
                    id=project_id,
                    group_id=GROUP_ID,
                    slug=f"project-{str(project_id)[-8:]}",
                    display_name=str(project_id),
                    normalized_display_name=str(project_id),
                    timestamps=_timestamps(),
                ),
            )
            await connectors.create(
                tx,
                InternalConnectorDBEntry(
                    id=connector_id,
                    project_id=project_id,
                    name='uploads',
                    normalized_name='uploads',
                    config=FileConnectorResourceConfig(
                        name='uploads', root_dir=f'/tmp/{project_id}'),
                    timestamps=_timestamps(),
                ),
            )


@pytest.mark.asyncio
async def test_table_repository_is_project_scoped(tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / 'state.db'))
    await _seed_projects_and_connectors(db)
    repo = StateDBTableRepository()
    first = _entry(PROJECT_A, 'same', 'Customers')
    second = _entry(PROJECT_B, 'same', 'customers')
    assert first.id != second.id

    async with db.write_tx() as tx:
        await repo.upsert(tx, first.id, first)
        await repo.upsert(tx, second.id, second)

    async with db.read_tx() as tx:
        assert await repo.get(tx, PROJECT_A, first.id) == first
        assert await repo.get(tx, PROJECT_B, second.id) == second
        assert await repo.get_by_alias(tx, PROJECT_A, 'CUSTOMERS') == first
        assert await repo.list(tx, PROJECT_A) == [first]

    await db.close()


@pytest.mark.asyncio
async def test_table_repository_round_trips_any_uri_source(tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / 'state.db'))
    await _seed_projects_and_connectors(db)
    connectors = StateDBConnectorRepository()
    tables = StateDBTableRepository()
    identities = JsonRepository('identities', IdentityDBEntry)
    bindings = JsonRepository('identity_groups', IdentityGroupEntry)
    definition = TableDefinition(
        cols=[Column(name='a', stype=Stype.ID, dtype=Dtype.int)],
        source_table=GCSSourceTable(
            connector_id=CONNECTOR_ANY_URI,
            source={
                'mode': 'uri',
                'uri': 'gs://customer-data/deep/orders.parquet',
            },
            file_type='PARQUET',
            data_source_type=DataSourceType.GCS,
        ),
    )
    entry = TableDBEntryV2(
        id=definition.fpid,
        project_id=PROJECT_A,
        config=definition,
        col_dtypes={},
        alias='orders',
        timestamps=_timestamps(),
    )

    async with db.write_tx() as tx:
        await identities.upsert(
            tx,
            str(IDENTITY_ANY_URI),
            IdentityDBEntry(
                id=IDENTITY_ANY_URI,
                slug='gcs-reader',
                display_name='GCS reader',
                normalized_display_name='gcs reader',
                description='',
                identity_key=('gcs:workload_identity:'
                              'reader@example-project.iam.gserviceaccount.com'),
                credentials_rotated_at=None,
                config=api.GCSIdentityConfig(
                    authentication=api.GoogleWorkloadIdentityAuthentication(
                        service_account_email=(
                            'reader@example-project.iam.gserviceaccount.com'))),
                timestamps=_timestamps(),
            ),
        )
        binding = IdentityGroupEntry(
            id=IdentityGroupID.generate(),
            identity_id=IDENTITY_ANY_URI,
            group_id=GROUP_ID,
            timestamps=_timestamps(),
        )
        await bindings.upsert(tx, str(binding.id), binding)
        await connectors.create(
            tx,
            ExternalConnectorDBEntry(
                id=CONNECTOR_ANY_URI,
                project_id=PROJECT_A,
                identity_id=IDENTITY_ANY_URI,
                name='gcs-any-uri',
                normalized_name='gcs-any-uri',
                created_by='user',
                config=GCSConnectorResourceConfig(
                    name=str(CONNECTOR_ANY_URI),
                    target=api.GCSAnyURIConnectorTarget(),
                ),
                timestamps=_timestamps(),
            ),
        )
        await tables.upsert(tx, entry.id, entry)

    async with db.read_tx() as tx:
        restored = await tables.get(tx, PROJECT_A, entry.id)
    assert restored is not None
    assert restored == entry
    assert restored.config.source_table.source.uri == (
        'gs://customer-data/deep/orders.parquet')
    await db.close()


@pytest.mark.asyncio
async def test_table_alias_uniqueness_is_per_project(tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / 'state.db'))
    await _seed_projects_and_connectors(db)
    repo = StateDBTableRepository()
    first = _entry(PROJECT_A, 'first', 'customers')
    collision = _entry(PROJECT_A, 'second', 'CUSTOMERS')
    other_project = _entry(PROJECT_B, 'second', 'CUSTOMERS')

    async with db.write_tx() as tx:
        await repo.upsert(tx, first.id, first)
        with pytest.raises(ValueError, match='already used'):
            await repo.upsert(tx, collision.id, collision)
        await repo.upsert(tx, other_project.id, other_project)

    await db.close()


@pytest.mark.asyncio
async def test_table_alias_uses_one_nfkc_key_for_lookup_batch_and_collision(
        tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / 'state.db'))
    await _seed_projects_and_connectors(db)
    repo = StateDBTableRepository()
    fullwidth = _entry(PROJECT_A, 'first', 'Ｃｕｓｔｏｍｅｒｓ')
    collision = _entry(PROJECT_A, 'second', ' customers ')

    async with db.write_tx() as tx:
        await repo.upsert(tx, fullwidth.id, fullwidth)
        assert await repo.get_by_alias(tx, PROJECT_A, 'CUSTOMERS') == fullwidth
        assert await repo.get_many_by_alias(tx, PROJECT_A, ['customers']) == {
            'customers': fullwidth
        }
        with pytest.raises(ValueError, match='already used'):
            await repo.upsert(tx, collision.id, collision)

    await db.close()


@pytest.mark.asyncio
async def test_table_delete_and_reference_count_cannot_cross_projects(
        tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / 'state.db'))
    await _seed_projects_and_connectors(db)
    repo = StateDBTableRepository()
    first = _entry(PROJECT_A, 'same', 'a', ref_count=2)
    second = _entry(PROJECT_B, 'same', 'b')
    async with db.write_tx() as tx:
        await repo.upsert(tx, first.id, first)
        await repo.upsert(tx, second.id, second)
        decremented = await repo.decrement_or_delete(tx, PROJECT_A, first.id)
        assert decremented
        deleted = await repo.delete(tx, PROJECT_A, first.id)
        assert deleted
    async with db.read_tx() as tx:
        assert await repo.get(tx, PROJECT_A, first.id) is None
    assert await repo.get(tx, PROJECT_B, second.id) == second
    await db.close()


@pytest.mark.asyncio
async def test_table_repository_rejects_cross_project_connector(
        tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / 'state.db'))
    await _seed_projects_and_connectors(db)
    repo = StateDBTableRepository()
    definition = TableDefinition(
        cols=[Column(name='a', stype=Stype.ID, dtype=Dtype.int)],
        source_table=S3SourceTable(
            connector_id=CONNECTOR_A,
            source={'mode': 'relative', 'path': 'cross-project.parquet'},
            file_type='PARQUET',
            data_source_type=DataSourceType.S3,
        ),
    )
    entry = TableDBEntryV2(
        id=definition.fpid,
        project_id=PROJECT_B,
        config=definition,
        col_dtypes={},
        timestamps=_timestamps(),
    )
    async with db.write_tx() as tx:
        with pytest.raises(ValueError, match='not active in project'):
            await repo.upsert(tx, entry.id, entry)
    await db.close()


@pytest.mark.asyncio
async def test_table_repository_locks_connector_before_validation_read(
        tmp_path, monkeypatch) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / 'state.db'))
    await _seed_projects_and_connectors(db)
    entry = _entry(PROJECT_A, 'race', 'race')
    events = []
    original_get = StateDBConnectorRepository.get

    async def record_lock(tx, project_id, connector_id) -> None:
        events.append(('lock', project_id, connector_id))

    async def record_get(repository, tx, connector_id):
        events.append(('get', PROJECT_A, connector_id))
        return await original_get(repository, tx, connector_id)

    monkeypatch.setattr(connector_admission_module,
                        'lock_connector_admission_and_delete',
                        record_lock)
    monkeypatch.setattr(StateDBConnectorRepository, 'get', record_get)

    async with db.write_tx() as tx:
        await StateDBTableRepository().upsert(tx, entry.id, entry)

    assert events == [
        ('lock', PROJECT_A, CONNECTOR_A),
        ('get', PROJECT_A, CONNECTOR_A),
    ]
    await db.close()
