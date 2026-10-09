# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from test.statedb.db.matrix_assertions import only_instance

import asyncpg
import kumoapi.data_source as api
import pytest
from kumo_core.model_plan import ModelPlan
from kumo_core.types import TaskType
from kumo_core.types.schema import FPID
from kumoapi.data_source import DEFAULT_GROUP_ID, DEFAULT_PROJECT_ID
from kumoapi.jobs import S3PredictionTableSource
from kumoapi.source_table import FileType, S3RelativeSource
from kumoapi.typing import Dtype, Stype

import statedb.db.connector_admission as connector_admission_module
import statedb.db.identities as identities_module
import statedb.db.services as services_module
from rest.config.batch_prediction_job import BatchPredictionJobConfig
from rest.config.connector import FileConnectorResourceConfig
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.graph_v2 import GraphDBEntryV2
from rest.config.table_v2 import (
    Column,
    DataSourceType,
    S3SourceTable,
    TableDBEntryV2,
    TableDefinition,
)
from rest.config.training_job import CustomTrainingTable
from rest.deps.authentication import UserAuthorization
from statedb.config.common import (
    ArtifactExportJobID,
    BatchPredictionJobID,
    ConnectorID,
    GraphIDv1,
    IdentityID,
    PredictiveQueryID,
    ProjectID,
    TableName,
    TrainingJobID,
    UserID,
    now_utc,
)
from statedb.config.connector import S3ConnectorResourceConfig
from statedb.config.timestamp import get_current_timestamps
from statedb.db import StateDB
from statedb.db.artifact_exports import ArtifactExportJobService
from statedb.db.connector_admission import lock_connector_admission_and_delete
from statedb.db.connector_authorizations import ConnectorAuthorizationService
from statedb.db.connectors import StateDBConnectorRepository
from statedb.db.graphs import StateDBGraphRepository
from statedb.db.groups import GroupService
from statedb.db.identities import ConnectorService, IdentityService
from statedb.db.ml_jobs import MLJobService
from statedb.db.services import ProjectService
from statedb.db.tables import StateDBTableRepository
from statedb.rest.connector_db import (
    ExternalConnectorDBEntry,
    InternalConnectorDBEntry,
)
from statedb.rest.group import (
    ExternalGroupMappingEntry,
    GroupMemberEntry,
    KumoGroupCreateSpec,
    PrincipalType,
)
from statedb.rest.project import ProjectDBEntry, ProjectStatus
from statedb.utils.errors import (
    ConnectorInUseError,
    GroupAlreadyExists,
    GroupNotFoundError,
    GroupRelationConflictError,
    IdentityDoesNotExistError,
    IdentityForbiddenError,
    ProjectNameConflictError,
    ProjectNotFoundError,
    ResourceSlugConflictError,
)


async def _reset_postgres_schema(connection) -> None:
    await connection.execute('DROP SCHEMA public CASCADE')
    await connection.execute('CREATE SCHEMA public')


def _new_identity_service(state_db: StateDB, **kwargs) -> IdentityService:
    return IdentityService(state_db, ConnectorAuthorizationService(state_db),
                           **kwargs)


def _new_connector_service(state_db: StateDB, **kwargs) -> ConnectorService:
    return ConnectorService(state_db, ConnectorAuthorizationService(state_db),
                            **kwargs)


@pytest.mark.asyncio
async def test_postgres_table_and_graph_alias_races_preserve_unique_nfkc_owner(
    postgres_dsn: str,
    postgres_connection,
) -> None:
    await _reset_postgres_schema(postgres_connection)
    state_db = await StateDB.open_postgres(postgres_dsn)
    admin = UserAuthorization.disabled()
    groups = GroupService(state_db)
    projects = ProjectService(state_db)
    connectors = StateDBConnectorRepository()
    tables = StateDBTableRepository()
    graphs = StateDBGraphRepository()
    try:
        group = (await groups.create_kumo_groups(
            [KumoGroupCreateSpec(display_name='Alias race group')], admin))[0]
        project = await projects.create_project(
            name='Alias race project',
            group_id=str(group.id),
            user_authorization=admin,
        )
        connector_id = ConnectorID.generate()
        async with state_db.write_tx() as tx:
            await connectors.create(
                tx,
                InternalConnectorDBEntry(
                    id=connector_id,
                    project_id=project.id,
                    name='uploads',
                    normalized_name='uploads',
                    config=FileConnectorResourceConfig(
                        name='uploads', root_dir='/tmp/alias-race'),
                    timestamps=get_current_timestamps(),
                ),
            )

        table_entries = []
        for source, alias in (('one', 'ＲＡＣＥ'), ('two', ' race ')):
            definition = TableDefinition(
                cols=[Column(name='id', stype=Stype.ID, dtype=Dtype.int)],
                source_table=S3SourceTable(
                    connector_id=connector_id,
                    source={
                        'mode': 'relative',
                        'path': f'{source}.parquet'
                    },
                    file_type='PARQUET',
                    data_source_type=DataSourceType.S3,
                ),
            )
            table_entries.append(
                TableDBEntryV2(
                    id=definition.fpid,
                    project_id=project.id,
                    config=definition,
                    col_dtypes={},
                    alias=alias,
                    timestamps=get_current_timestamps(),
                ))

        async def persist_table(entry) -> None:
            async with state_db.write_tx() as tx:
                await tables.upsert(tx, entry.id, entry)

        table_results = await asyncio.gather(
            *(persist_table(entry) for entry in table_entries),
            return_exceptions=True,
        )
        assert sum(result is None for result in table_results) == 1
        assert sum(
            isinstance(result, (ValueError, asyncpg.UniqueViolationError))
            for result in table_results) == 1

        winner = next(entry
                      for entry, result in zip(table_entries, table_results)
                      if result is None)
        graph_entries = [
            GraphDBEntryV2(
                id=FPID(character * 32),
                project_id=project.id,
                table_name_to_dbid={TableName('source'): winner.id},
                col_groups=[],
                alias=alias,
                timestamps=get_current_timestamps(),
            ) for character, alias in (('a', 'first'), ('b', 'second'))
        ]
        for entry in graph_entries:
            async with state_db.write_tx() as tx:
                await graphs.upsert(tx, entry.id, entry)
            entry.alias = 'ＧＲＡＰＨ'

        async def rename_graph(entry) -> None:
            async with state_db.write_tx() as tx:
                await graphs.upsert(tx, entry.id, entry, force_rename=True)

        graph_results = await asyncio.gather(
            *(rename_graph(entry) for entry in graph_entries),
            return_exceptions=True,
        )
        assert any(result is None for result in graph_results)
        assert all(
            result is None or isinstance(result, (ValueError,
                                                  asyncpg.UniqueViolationError))
            for result in graph_results)
        async with state_db.read_tx() as tx:
            named = await graphs.list_named(tx, project.id)
            assert len([
                entry for entry in named if entry.normalized_alias == 'graph'
            ]) == 1
            assert await graphs.get_by_alias(tx, project.id,
                                             ' graph ') is not None
    finally:
        await state_db.close()
        await _reset_postgres_schema(postgres_connection)


@pytest.mark.asyncio
async def test_postgres_artifact_request_insert_is_atomic_for_same_id(
    postgres_dsn: str,
    postgres_connection,
) -> None:
    await _reset_postgres_schema(postgres_connection)
    state_db = await StateDB.open_postgres(postgres_dsn)
    service = ArtifactExportJobService(state_db)
    job_id = ArtifactExportJobID.make_random_id()
    base = dict(
        id=job_id,
        project_id=ProjectID.generate(),
        created_by=UserID('postgres-race'),
        request_id='00000000-0000-4000-8000-000000000001',
    )
    try:
        same = await asyncio.gather(
            service.create(**base, request_digest='same'),
            service.create(**base, request_digest='same'),
        )
        assert {inserted for _, inserted in same} == {False, True}
        sealed = await asyncio.gather(
            service.seal_dispatch(job_id, project_id=base['project_id']),
            service.seal_dispatch(job_id, project_id=base['project_id']),
        )
        assert all(entry.dispatch_sealed for entry in sealed)
        assert not any(entry.cancel_requested for entry in sealed)

        cancel_id = ArtifactExportJobID.make_random_id()
        await service.create(**{
            **base, 'id': cancel_id
        }, request_digest='cancel-race')
        await asyncio.gather(
            service.seal_dispatch(cancel_id, project_id=base['project_id']),
            service.request_cancel(cancel_id, project_id=base['project_id']),
        )
        cancelled = await service.request_cancel(cancel_id,
                                                 project_id=base['project_id'])
        assert cancelled.dispatch_sealed
        assert cancelled.cancel_requested

        conflict_base = {**base, 'id': ArtifactExportJobID.make_random_id()}
        results = await asyncio.gather(
            service.create(**conflict_base, request_digest='first'),
            service.create(**conflict_base, request_digest='second'),
            return_exceptions=True,
        )
        only_instance(results, ValueError)
        only_instance(results, tuple)
    finally:
        await state_db.close()
        await _reset_postgres_schema(postgres_connection)


@pytest.mark.asyncio
async def test_postgres_default_bootstrap_and_global_name_races(
    postgres_dsn: str,
    postgres_connection,
) -> None:
    await _reset_postgres_schema(postgres_connection)
    state_db = await StateDB.open_postgres(postgres_dsn)
    admin = UserAuthorization.disabled()
    groups = GroupService(state_db)
    projects = ProjectService(state_db)
    try:
        bootstrapped = await asyncio.gather(
            groups.ensure_default_resources(),
            groups.ensure_default_resources(),
        )
        assert {str(result[0].id)
                for result in bootstrapped} == {DEFAULT_GROUP_ID}
        assert {str(result[1].id)
                for result in bootstrapped} == {DEFAULT_PROJECT_ID}
        async with state_db.read_tx() as tx:
            rules = await groups._rules.list_by_field(tx, 'group_id',
                                                      DEFAULT_GROUP_ID)
        assert len(rules) == 1

        group_a, group_b = await groups.create_kumo_groups([
            KumoGroupCreateSpec(display_name='Engineering'),
            KumoGroupCreateSpec(display_name='Finance'),
        ], admin)
        group_results = await asyncio.gather(
            groups.create_kumo_groups([
                KumoGroupCreateSpec(display_name='Platform',
                                    slug='platform-one')
            ], admin),
            groups.create_kumo_groups([
                KumoGroupCreateSpec(display_name=' ＰＬＡＴＦＯＲＭ ',
                                    slug='platform-two')
            ], admin),
            return_exceptions=True,
        )
        only_instance(group_results, list)
        only_instance(group_results, GroupAlreadyExists)

        member_results = await asyncio.gather(*(groups.add_member(
            group_a.id,
            principal_id,
            PrincipalType.USER,
            principal=admin,
        ) for principal_id in (' member ', 'member', ' member ')))
        assert all(
            isinstance(result, GroupMemberEntry) for result in member_results)
        assert len({result.id for result in member_results}) == 1
        assert member_results[0].principal_id == 'member'

        await asyncio.gather(*(groups.set_all_authenticated_users(
            group_a.id, True, principal=admin) for _ in range(4)))
        async with state_db.read_tx() as tx:
            rules = await groups._rules.list_by_field(tx, 'group_id',
                                                      str(group_a.id))
        assert len(rules) == 1

        mapping_results = await asyncio.gather(*(groups.create_external_mapping(
            group_a.id,
            provider_id,
            external_id,
            principal=admin,
        ) for provider_id, external_id in (
            (' DIRECT_CONNECT ', ' same-external '),
            ('direct_connect', 'same-external'),
            ('Direct_Connect', ' same-external'),
        )))
        assert all(
            isinstance(result, ExternalGroupMappingEntry)
            for result in mapping_results)
        assert len({result.id for result in mapping_results}) == 1

        mapping_conflict_results = await asyncio.gather(
            groups.create_external_mapping(
                group_a.id,
                'direct_connect',
                'contended-external',
                principal=admin,
            ),
            groups.create_external_mapping(
                group_b.id,
                'direct_connect',
                'contended-external',
                principal=admin,
            ),
            return_exceptions=True,
        )
        only_instance(mapping_conflict_results, ExternalGroupMappingEntry)
        only_instance(mapping_conflict_results, GroupRelationConflictError)

        race_group = (await groups.create_kumo_groups(
            [KumoGroupCreateSpec(display_name='Delete race')], admin))[0]
        project_delete_results = await asyncio.gather(
            projects.create_project(
                name='Race project',
                group_id=str(race_group.id),
                user_authorization=admin,
            ),
            groups.delete_kumo_group(str(race_group.id), admin),
            return_exceptions=True,
        )
        created_project, delete_result = project_delete_results
        if isinstance(created_project, ProjectDBEntry):
            assert isinstance(delete_result, ValueError)
            assert 'owns a project' in str(delete_result)
            assert await groups.get_kumo_group(str(race_group.id), admin)
        else:
            assert isinstance(created_project, GroupNotFoundError)
            assert delete_result is None
            with pytest.raises(GroupNotFoundError):
                await groups.get_kumo_group(str(race_group.id), admin)

        create_results = await asyncio.gather(
            projects.create_project(
                name='Analytics',
                slug='analytics-one',
                group_id=str(group_a.id),
                user_authorization=admin,
            ),
            projects.create_project(
                name=' ＡＮＡＬＹＴＩＣＳ ',
                slug='analytics-two',
                group_id=str(group_b.id),
                user_authorization=admin,
            ),
            return_exceptions=True,
        )
        only_instance(create_results, ProjectDBEntry)
        only_instance(create_results, ProjectNameConflictError)

        first = await projects.create_project(
            name='First',
            group_id=str(group_a.id),
            user_authorization=admin,
        )
        second = await projects.create_project(
            name='Second',
            group_id=str(group_b.id),
            user_authorization=admin,
        )
        update_results = await asyncio.gather(
            projects.update_project(first.id, name='Winner',
                                    user_authorization=admin),
            projects.update_project(second.id, name=' ＷＩＮＮＥＲ ',
                                    user_authorization=admin),
            return_exceptions=True,
        )
        only_instance(update_results, ProjectDBEntry)
        only_instance(update_results, ProjectNameConflictError)
    finally:
        await state_db.close()
        await _reset_postgres_schema(postgres_connection)


@pytest.mark.asyncio
async def test_postgres_global_slug_indexes_have_one_concurrent_winner(
    postgres_dsn: str,
    postgres_connection,
) -> None:
    await _reset_postgres_schema(postgres_connection)
    state_db = await StateDB.open_postgres(postgres_dsn)
    admin = UserAuthorization.disabled()
    groups = GroupService(state_db)
    projects = ProjectService(state_db)
    identities = _new_identity_service(state_db,
                                       identity_validator=lambda *_: None)
    try:
        group_results = await asyncio.gather(
            groups.create_kumo_groups([
                KumoGroupCreateSpec(display_name='Slug Group One',
                                    slug='postgres-group')
            ], admin),
            groups.create_kumo_groups([
                KumoGroupCreateSpec(display_name='Slug Group Two',
                                    slug='postgres-group')
            ], admin),
            return_exceptions=True,
        )
        owner_group = only_instance(group_results, list)[0]
        only_instance(group_results, ResourceSlugConflictError)

        project_results = await asyncio.gather(
            projects.create_project(
                name='Slug Project One',
                slug='postgres-project',
                group_id=str(owner_group.id),
                user_authorization=admin,
            ),
            projects.create_project(
                name='Slug Project Two',
                slug='postgres-project',
                group_id=str(owner_group.id),
                user_authorization=admin,
            ),
            return_exceptions=True,
        )
        only_instance(project_results, ProjectDBEntry)
        only_instance(project_results, ResourceSlugConflictError)

        def identity_args(name: str, role: str) -> api.CreateIdentityArgs:
            return api.CreateIdentityArgs(
                display_name=name,
                slug='postgres-identity',
                config=api.S3IdentityConfig(
                    authentication=api.S3AssumeRoleAuthentication(
                        partition='aws',
                        role_arn=(f'arn:aws:iam::123456789012:role/{role}'),
                    ), ),
            )

        identity_results = await asyncio.gather(
            identities.create(identity_args('Slug Identity One', 'slug-one'),
                              admin),
            identities.create(identity_args('Slug Identity Two', 'slug-two'),
                              admin),
            return_exceptions=True,
        )
        only_instance(identity_results, api.IdentityResponse)
        only_instance(identity_results, ResourceSlugConflictError)
    finally:
        await state_db.close()
        await _reset_postgres_schema(postgres_connection)


@pytest.mark.asyncio
async def test_postgres_connector_name_is_reusable_after_tombstone(
    postgres_dsn: str,
    postgres_connection,
) -> None:
    await _reset_postgres_schema(postgres_connection)
    state_db = await StateDB.open_postgres(postgres_dsn)
    admin = UserAuthorization.disabled()
    groups = GroupService(state_db)
    projects = ProjectService(state_db)
    connectors = StateDBConnectorRepository()
    try:
        group = (await groups.create_kumo_groups(
            [KumoGroupCreateSpec(display_name='Reusable connector group')],
            admin,
        ))[0]
        project = await projects.create_project(
            name='Reusable connector project',
            group_id=str(group.id),
            user_authorization=admin,
        )
        identity_id = IdentityID.generate()
        first_id = ConnectorID.generate()
        first = ExternalConnectorDBEntry(
            id=first_id,
            project_id=project.id,
            identity_id=identity_id,
            name='Data Lake',
            normalized_name='data lake',
            created_by='admin',
            config=S3ConnectorResourceConfig(
                name=str(first_id),
                target=api.S3RootedConnectorTarget(bucket='first-bucket'),
            ),
            timestamps=get_current_timestamps(),
        )
        async with state_db.write_tx() as tx:
            await connectors.create(tx, first)
        async with state_db.write_tx() as tx:
            persisted_first = await connectors.get_required(tx, first_id)
            persisted_first.tombstoned_at = now_utc()
            await connectors.upsert(tx, persisted_first)

        second_id = ConnectorID.generate()
        second = ExternalConnectorDBEntry(
            id=second_id,
            project_id=project.id,
            identity_id=identity_id,
            name=' DATA LAKE ',
            normalized_name='data lake',
            created_by='admin',
            config=S3ConnectorResourceConfig(
                name=str(second_id),
                target=api.S3RootedConnectorTarget(bucket='second-bucket'),
            ),
            timestamps=get_current_timestamps(),
        )
        async with state_db.write_tx() as tx:
            await connectors.create(tx, second)

        async with state_db.read_tx() as tx:
            live = await connectors.find_live_by_project_and_name(
                tx, project.id, 'data lake')
            history = await connectors.list(tx, project_id=project.id)
        assert live is not None
        assert live.id == second.id
        assert live.normalized_name == 'data lake'
        assert live.config.target.bucket == 'second-bucket'
        assert {connector.id for connector in history} == {first_id, second_id}
    finally:
        await state_db.close()
        await _reset_postgres_schema(postgres_connection)


@pytest.mark.asyncio
async def test_postgres_connector_mutations_serialize_with_binding_and_table(
    postgres_dsn: str,
    postgres_connection,
    monkeypatch,
) -> None:
    await _reset_postgres_schema(postgres_connection)
    state_db = await StateDB.open_postgres(postgres_dsn)
    admin = UserAuthorization.disabled()
    groups = GroupService(state_db)
    projects = ProjectService(state_db)
    identities = _new_identity_service(
        state_db,
        identity_validator=lambda *_: None,
    )
    connectors = _new_connector_service(
        state_db,
        connector_validator=lambda *_: None,
    )
    connector_repository = StateDBConnectorRepository()
    table_repository = StateDBTableRepository()
    try:
        group = (await groups.create_kumo_groups(
            [KumoGroupCreateSpec(display_name='Race group')], admin))[0]
        project = await projects.create_project(
            name='Race project',
            group_id=str(group.id),
            user_authorization=admin,
        )
        identity = await identities.create(
            api.CreateIdentityArgs(
                display_name='Race role',
                config=api.S3IdentityConfig(
                    authentication=api.S3AssumeRoleAuthentication(
                        partition='aws',
                        role_arn=('arn:aws:iam::123456789012:role/race-reader'),
                    ), ),
            ),
            admin,
        )
        await identities.replace_groups(
            identity.id,
            api.ReplaceIdentityGroupsArgs(group_ids=[str(group.id)]),
            admin,
        )

        connector_id = ConnectorID.generate()
        timestamps = get_current_timestamps()
        async with state_db.write_tx() as tx:
            await connector_repository.create(
                tx,
                ExternalConnectorDBEntry(
                    id=connector_id,
                    project_id=project.id,
                    identity_id=identity.id,
                    name='Race storage',
                    normalized_name='race storage',
                    created_by='admin',
                    config=S3ConnectorResourceConfig(
                        name=str(connector_id),
                        target=api.S3RootedConnectorTarget(
                            bucket='race-bucket'),
                    ),
                    timestamps=timestamps,
                ),
            )

        original_group_lock = identities_module.lock_group_relations
        unbind_has_lock = asyncio.Event()
        mutation_is_waiting = asyncio.Event()
        release_unbind = asyncio.Event()

        async def controlled_group_lock(tx, group_ids) -> None:
            task = asyncio.current_task()
            if task is not None and task.get_name() == 'unbind':
                await original_group_lock(tx, group_ids)
                unbind_has_lock.set()
                await asyncio.wait_for(release_unbind.wait(), timeout=5)
                return
            mutation_is_waiting.set()
            await original_group_lock(tx, group_ids)

        monkeypatch.setattr(
            identities_module,
            'lock_group_relations',
            controlled_group_lock,
        )
        unbind_task = asyncio.create_task(
            identities.replace_groups(
                identity.id,
                api.ReplaceIdentityGroupsArgs(group_ids=[]),
                admin,
            ),
            name='unbind',
        )
        await asyncio.wait_for(unbind_has_lock.wait(), timeout=5)
        mutation_task = asyncio.create_task(
            connectors.update_metadata(
                project.id,
                connector_id,
                api.UpdateConnectorMetadataArgs(
                    description='must not commit after unbind'),
                admin,
            ),
            name='mutation',
        )
        await asyncio.wait_for(mutation_is_waiting.wait(), timeout=5)
        release_unbind.set()

        await unbind_task
        mutation_result = await asyncio.gather(
            mutation_task,
            return_exceptions=True,
        )
        assert len(mutation_result) == 1
        assert isinstance(mutation_result[0], IdentityForbiddenError)

        async with state_db.read_tx() as tx:
            unchanged_connector = await connector_repository.get(
                tx, connector_id)
        assert unchanged_connector is not None
        assert unchanged_connector.description == ''
        assert unchanged_connector.tombstoned_at is None

        monkeypatch.setattr(
            identities_module,
            'lock_group_relations',
            original_group_lock,
        )
        await identities.replace_groups(
            identity.id,
            api.ReplaceIdentityGroupsArgs(group_ids=[str(group.id)]),
            admin,
        )

        for mutation_name in ('update', 'delete'):
            archive_has_lock = asyncio.Event()
            mutation_is_waiting = asyncio.Event()
            release_archive = asyncio.Event()

            async def controlled_project_group_lock(tx, group_ids) -> None:
                task = asyncio.current_task()
                if task is not None and task.get_name() == 'archive':
                    await original_group_lock(tx, group_ids)
                    archive_has_lock.set()
                    await asyncio.wait_for(release_archive.wait(), timeout=5)
                    return
                mutation_is_waiting.set()
                await original_group_lock(tx, group_ids)

            monkeypatch.setattr(
                identities_module,
                'lock_group_relations',
                controlled_project_group_lock,
            )
            monkeypatch.setattr(
                services_module,
                'lock_group_relations',
                controlled_project_group_lock,
            )
            archive_task = asyncio.create_task(
                projects.archive_project(project.id, admin),
                name='archive',
            )
            await asyncio.wait_for(archive_has_lock.wait(), timeout=5)
            if mutation_name == 'update':
                mutation = connectors.update_metadata(
                    project.id,
                    connector_id,
                    api.UpdateConnectorMetadataArgs(
                        description='must not commit after archive'),
                    admin,
                )
            else:
                mutation = connectors.delete(
                    project.id,
                    connector_id,
                    admin,
                )
            mutation_task = asyncio.create_task(mutation, name='mutation')
            await asyncio.wait_for(mutation_is_waiting.wait(), timeout=5)
            release_archive.set()

            archived_project = await archive_task
            assert archived_project.status == ProjectStatus.ARCHIVED
            mutation_result = await asyncio.gather(
                mutation_task,
                return_exceptions=True,
            )
            assert len(mutation_result) == 1
            assert isinstance(mutation_result[0], ProjectNotFoundError)

            async with state_db.write_tx() as tx:
                project_row = await projects._projects.get(tx, str(project.id))
                assert project_row is not None
                project_row.payload.status = ProjectStatus.ACTIVE
                await projects._projects.upsert(tx, str(project.id),
                                                project_row.payload)
                unchanged_connector = await connector_repository.get(
                    tx, connector_id)
            assert unchanged_connector is not None
            assert unchanged_connector.description == ''
            assert unchanged_connector.tombstoned_at is None

            monkeypatch.setattr(
                identities_module,
                'lock_group_relations',
                original_group_lock,
            )
            monkeypatch.setattr(
                services_module,
                'lock_group_relations',
                original_group_lock,
            )

        definition = TableDefinition(
            cols=[Column(name='id', stype=Stype.ID, dtype=Dtype.int)],
            source_table=S3SourceTable(
                connector_id=connector_id,
                source={
                    'mode': 'relative',
                    'path': 'race.parquet'
                },
                file_type='PARQUET',
                data_source_type=DataSourceType.S3,
            ),
        )
        table = TableDBEntryV2(
            id=definition.fpid,
            project_id=project.id,
            config=definition,
            col_dtypes={},
            alias='race',
            timestamps=timestamps,
        )

        both_waiting = asyncio.Event()
        waiter_count = 0
        waiter_guard = asyncio.Lock()

        async def synchronized_connector_lock(
            tx,
            project_id,
            locked_connector_id,
        ) -> None:
            nonlocal waiter_count
            async with waiter_guard:
                waiter_count += 1
                if waiter_count == 2:
                    both_waiting.set()
            await asyncio.wait_for(both_waiting.wait(), timeout=5)
            await lock_connector_admission_and_delete(tx, project_id,
                                                      locked_connector_id)

        monkeypatch.setattr(
            connector_admission_module,
            'lock_connector_admission_and_delete',
            synchronized_connector_lock,
        )
        monkeypatch.setattr(
            identities_module,
            'lock_connector_admission_and_delete',
            synchronized_connector_lock,
        )

        async def persist_table() -> None:
            async with state_db.write_tx() as tx:
                await table_repository.upsert(tx, table.id, table)

        table_result, delete_result = await asyncio.gather(
            persist_table(),
            connectors.delete(project.id, connector_id, admin),
            return_exceptions=True,
        )

        async with state_db.read_tx() as tx:
            persisted_table = await table_repository.get(
                tx, project.id, table.id)
            persisted_connector = await connector_repository.get(
                tx, connector_id)

        assert persisted_connector is not None
        if table_result is None:
            assert isinstance(delete_result, ConnectorInUseError)
            assert persisted_table == table
            assert persisted_connector.tombstoned_at is None
        else:
            assert isinstance(table_result, ValueError)
            assert 'not active in project' in str(table_result)
            assert not isinstance(delete_result, BaseException)
            assert persisted_table is None
            assert persisted_connector.tombstoned_at is not None
    finally:
        await state_db.close()
        await _reset_postgres_schema(postgres_connection)


@pytest.mark.parametrize('mutation_name', ('update', 'rotate', 'bind'))
@pytest.mark.asyncio
async def test_postgres_identity_delete_serializes_all_identity_mutations(
    postgres_dsn: str,
    postgres_connection,
    monkeypatch,
    mutation_name: str,
) -> None:
    await _reset_postgres_schema(postgres_connection)
    state_db = await StateDB.open_postgres(postgres_dsn)
    admin = UserAuthorization.disabled()
    groups = GroupService(state_db)
    identities = _new_identity_service(
        state_db,
        identity_validator=lambda *_: None,
    )
    try:
        group = (await groups.create_kumo_groups([
            KumoGroupCreateSpec(display_name=f'Identity race {mutation_name}')
        ], admin))[0]
        identity = await identities.create(
            api.CreateIdentityArgs(
                display_name=f'Identity {mutation_name}',
                config=api.S3IdentityConfig(
                    authentication=api.S3AssumeRoleAuthentication(
                        partition='aws',
                        role_arn=('arn:aws:iam::123456789012:role/'
                                  f'identity-{mutation_name}'),
                    ), ),
            ),
            admin,
        )

        original_identity_lock = identities_module.lock_identity_mutations
        delete_has_lock = asyncio.Event()
        mutation_is_waiting = asyncio.Event()
        release_delete = asyncio.Event()

        async def controlled_identity_lock(tx, identity_ids) -> None:
            task = asyncio.current_task()
            if task is not None and task.get_name() == 'identity-delete':
                await original_identity_lock(tx, identity_ids)
                delete_has_lock.set()
                await asyncio.wait_for(release_delete.wait(), timeout=5)
                return
            mutation_is_waiting.set()
            await original_identity_lock(tx, identity_ids)

        monkeypatch.setattr(
            identities_module,
            'lock_identity_mutations',
            controlled_identity_lock,
        )
        delete_task = asyncio.create_task(
            identities.delete(identity.id, admin),
            name='identity-delete',
        )
        await asyncio.wait_for(delete_has_lock.wait(), timeout=5)

        if mutation_name == 'update':
            mutation = identities.update_metadata(
                identity.id,
                api.UpdateIdentityArgs(description='must not resurrect'),
                admin,
            )
        elif mutation_name == 'rotate':
            mutation = identities.rotate_credentials(
                identity.id,
                api.RotateIdentityCredentialsArgs(credentials=None),
                admin,
            )
        else:
            mutation = identities.replace_groups(
                identity.id,
                api.ReplaceIdentityGroupsArgs(group_ids=[str(group.id)]),
                admin,
            )
        mutation_task = asyncio.create_task(mutation, name='identity-mutation')
        await asyncio.wait_for(mutation_is_waiting.wait(), timeout=5)
        release_delete.set()

        deleted = await delete_task
        mutation_result = await asyncio.gather(
            mutation_task,
            return_exceptions=True,
        )
        assert deleted.id == identity.id
        assert len(mutation_result) == 1
        assert isinstance(mutation_result[0], IdentityDoesNotExistError)

        async with state_db.read_tx() as tx:
            identity_row = await identities._identities.get(
                tx, str(identity.id))
            binding_rows = await identities._bindings.list_by_field(
                tx, 'identity_id', str(identity.id))
        assert identity_row is not None
        assert identity_row.payload.tombstoned_at is not None
        assert identity_row.payload.encrypted_credentials is None
        assert binding_rows == []
    finally:
        await state_db.close()
        await _reset_postgres_schema(postgres_connection)


@pytest.mark.parametrize('revocation', ('unbind', 'archive'))
@pytest.mark.parametrize(
    'resource_kind',
    ('table', 'custom-training', 'batch-prediction'),
)
@pytest.mark.asyncio
async def test_postgres_connector_admission_rechecks_after_revocation(
    postgres_dsn: str,
    postgres_connection,
    monkeypatch,
    revocation: str,
    resource_kind: str,
) -> None:
    await _reset_postgres_schema(postgres_connection)
    state_db = await StateDB.open_postgres(postgres_dsn)
    admin = UserAuthorization.disabled()
    groups = GroupService(state_db)
    projects = ProjectService(state_db)
    identities = _new_identity_service(
        state_db,
        identity_validator=lambda *_: None,
    )
    connector_repository = StateDBConnectorRepository()
    table_repository = StateDBTableRepository()
    jobs = MLJobService(state_db)
    try:
        suffix = f'{resource_kind}-{revocation}'
        group = (await groups.create_kumo_groups(
            [KumoGroupCreateSpec(display_name=f'Admission {suffix}')],
            admin))[0]
        project = await projects.create_project(
            name=f'Admission project {suffix}',
            group_id=str(group.id),
            user_authorization=admin,
        )
        identity = await identities.create(
            api.CreateIdentityArgs(
                display_name=f'Admission identity {suffix}',
                config=api.S3IdentityConfig(
                    authentication=api.S3AssumeRoleAuthentication(
                        partition='aws',
                        role_arn=('arn:aws:iam::123456789012:role/'
                                  f'admission-{suffix}'),
                    ), ),
            ),
            admin,
        )
        await identities.replace_groups(
            identity.id,
            api.ReplaceIdentityGroupsArgs(group_ids=[str(group.id)]),
            admin,
        )
        connector_id = ConnectorID.generate()
        timestamps = get_current_timestamps()
        async with state_db.write_tx() as tx:
            await connector_repository.create(
                tx,
                ExternalConnectorDBEntry(
                    id=connector_id,
                    project_id=project.id,
                    identity_id=identity.id,
                    name=f'Admission connector {suffix}',
                    normalized_name=f'admission connector {suffix}',
                    created_by='admin',
                    config=S3ConnectorResourceConfig(
                        name=str(connector_id),
                        target=api.S3RootedConnectorTarget(
                            bucket='admission-bucket'),
                    ),
                    timestamps=timestamps,
                ),
            )

        table = None
        training_id = None
        batch_id = None
        if resource_kind == 'table':
            definition = TableDefinition(
                cols=[Column(name='id', stype=Stype.ID, dtype=Dtype.int)],
                source_table=S3SourceTable(
                    connector_id=connector_id,
                    source={
                        'mode': 'relative',
                        'path': f'{suffix}.parquet'
                    },
                    file_type='PARQUET',
                    data_source_type=DataSourceType.S3,
                ),
            )
            table = TableDBEntryV2(
                id=definition.fpid,
                project_id=project.id,
                config=definition,
                col_dtypes={},
                alias=f'admission-{suffix}',
                timestamps=timestamps,
            )
        elif resource_kind == 'custom-training':
            training_id = TrainingJobID.make_random_id()
        elif resource_kind == 'batch-prediction':
            batch_id = BatchPredictionJobID.make_random_id()

        original_group_lock = connector_admission_module.lock_group_relations
        revocation_has_lock = asyncio.Event()
        admission_is_waiting = asyncio.Event()
        release_revocation = asyncio.Event()

        async def controlled_group_lock(tx, group_ids) -> None:
            task = asyncio.current_task()
            if task is not None and task.get_name() == 'revocation':
                await original_group_lock(tx, group_ids)
                revocation_has_lock.set()
                await asyncio.wait_for(release_revocation.wait(), timeout=5)
                return
            admission_is_waiting.set()
            await original_group_lock(tx, group_ids)

        monkeypatch.setattr(
            connector_admission_module,
            'lock_group_relations',
            controlled_group_lock,
        )
        if revocation == 'unbind':
            monkeypatch.setattr(
                identities_module,
                'lock_group_relations',
                controlled_group_lock,
            )
            revocation_call = identities.replace_groups(
                identity.id,
                api.ReplaceIdentityGroupsArgs(group_ids=[]),
                admin,
            )
        else:
            monkeypatch.setattr(
                services_module,
                'lock_group_relations',
                controlled_group_lock,
            )
            revocation_call = projects.archive_project(project.id, admin)

        revocation_task = asyncio.create_task(
            revocation_call,
            name='revocation',
        )
        await asyncio.wait_for(revocation_has_lock.wait(), timeout=5)

        if resource_kind == 'table':
            assert table is not None

            async def admit_resource() -> None:
                async with state_db.write_tx() as tx:
                    await table_repository.upsert(tx, table.id, table)
        elif resource_kind == 'custom-training':
            assert training_id is not None

            async def admit_resource() -> None:
                await jobs.create_training_job(
                    pquery_id=PredictiveQueryID('risk'),
                    pquery_def_fpid=None,
                    graph_id=GraphIDv1('graph'),
                    now=now_utc(),
                    advanced_options=ModelPlan(),
                    custom_train_table=CustomTrainingTable(
                        connector_spec=ConnectorExecutionRef(
                            project_id=project.id,
                            connector_id=connector_id,
                        ),
                        source_table=S3SourceTable(
                            connector_id=str(connector_id),
                            source=S3RelativeSource(path='training_data'),
                            file_type=FileType.PARQUET,
                        ),
                        weight_col_name='weight',
                    ),
                    reserved_job_id=training_id,
                    project_id=project.id,
                )
        elif resource_kind == 'batch-prediction':
            assert batch_id is not None

            async def admit_resource() -> None:
                await jobs.create_batch_prediction_job(
                    id=batch_id,
                    now=now_utc(),
                    config=BatchPredictionJobConfig(
                        predictive_query_id=PredictiveQueryID('risk'),
                        training_job_id=TrainingJobID.make_random_id(),
                        task_type=TaskType.BINARY_CLASSIFICATION,
                        custom_pred_table_source=S3PredictionTableSource(
                            connector_id=str(connector_id),
                            source=S3RelativeSource(path='prediction.parquet'),
                        ),
                        graph_id=GraphIDv1('graph'),
                    ),
                    project_id=project.id,
                )
        else:
            assert resource_kind == 'batch-prediction'

            async def admit_resource() -> None:
                raise AssertionError('unreachable')

        admission_task = asyncio.create_task(
            admit_resource(),
            name='admission',
        )
        await asyncio.wait_for(admission_is_waiting.wait(), timeout=5)
        release_revocation.set()

        await revocation_task
        admission_result = await asyncio.gather(
            admission_task,
            return_exceptions=True,
        )
        assert len(admission_result) == 1
        assert isinstance(admission_result[0], ValueError)

        if resource_kind == 'table':
            assert table is not None
            async with state_db.read_tx() as tx:
                assert await table_repository.get(tx, project.id,
                                                  table.id) is None
        elif resource_kind == 'custom-training':
            assert training_id is not None
            assert await jobs.get_training_job(training_id) is None
        elif resource_kind == 'batch-prediction':
            assert batch_id is not None
            assert await jobs.get_batch_prediction_jobs([batch_id],
                                                        project_ids=None) == []
    finally:
        await state_db.close()
        await _reset_postgres_schema(postgres_connection)
