# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from copy import deepcopy
from datetime import datetime, timezone
from test.statedb.db.connector_fixtures import seed_connector
from test.statedb.ml.data_model import (
    case_batch_prediction_job,
    case_generate_pred_table_job,
    case_generate_train_table_job,
    case_training_job,
)

import pytest
import pytest_asyncio
from kumo_core.types.schema import FPID
from kumoapi.source_table import FileType
from kumoapi.typing import Dtype, Stype
from temporalio.client import WorkflowExecutionStatus

import statedb.db.runtime as runtime_mod
from rest.config.batch_prediction_job import (
    BatchPredictionJobOutput,
    BatchPredictionJobResults,
)
from rest.config.connector import FileConnectorResourceConfig
from rest.config.encryption import hash_idp_group_id
from rest.config.graph_v2 import GraphDBEntryV2, GraphDefinition
from rest.config.table_v2 import (
    Column,
    DataSourceType,
    S3SourceTable,
    TableDBEntryV2,
    TableDefinition,
)
from rest.config.training_job import TrainingJobOutput, TrainingJobResults
from rest.deps.authentication import Permissions, UserAuthorization
from rest.public.routers.training import TrainingRouterV1
from statedb.config.common import (
    APIKeyID,
    BatchPredictionJobID,
    ConnectorID,
    GeneratePredictionTableJobID,
    GenerateTrainTableJobID,
    GraphIDv2,
    GroupID,
    JobEventLogEntry,
    JobFailureInfo,
    JobID,
    JobStatus,
    JobType,
    ProjectID,
    TableIDv2,
    TableName,
    TrainingJobID,
    UserID,
)
from statedb.config.job_tag import (
    TAG_KEY_PQUERY_NAME,
    JobTagKey,
    JobTags,
    JobTagValue,
)
from statedb.config.timestamp import ResourceTimestamps
from statedb.db import StateDB
from statedb.db.connectors import StateDBConnectorRepository
from statedb.db.graphs import StateDBGraphRepository
from statedb.db.job_tags import job_tag_row_id, job_tags_row_id
from statedb.db.ml_jobs import MLJobService
from statedb.db.repositories import JsonRepository
from statedb.db.services import (
    APIKeyNameAlreadyExistsError,
    APIKeyService,
    GroupService,
    ProjectService,
    StateServices,
    TableService,
)
from statedb.db.tables import StateDBTableRepository
from statedb.ml.jobs import BatchPredictionJobDBEntry
from statedb.rest.api_key import (
    APIKeyMetadataRecord,
    APIKeySecretRecord,
    APIKeyType,
)
from statedb.rest.connector_db import InternalConnectorDBEntry
from statedb.rest.group import (
    ExternalIdentityProviderID,
    KumoGroupCreateSpec,
    KumoGroupDBEntry,
)
from statedb.rest.project import ProjectDBEntry, ProjectStatus
from statedb.utils.errors import (
    ExternalMembershipUnavailableError,
    GroupNotFoundError,
    ProjectForbiddenError,
    ProjectNameConflictError,
    ProjectNotAvailableError,
    ProjectNotFoundError,
)


@pytest_asyncio.fixture
async def state_db(tmp_path):
    database = await StateDB.open_sqlite(str(tmp_path / 'state.db'))
    try:
        yield database
    finally:
        await database.close()


async def _insert_normalized_job_tag_row(
    tx,
    *,
    row_id: str,
    job_type: JobType,
    job_id: JobID,
    key: str,
    value: str,
) -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()
    await tx.execute(
        """
        INSERT INTO job_tags (id, payload, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        row_id,
        json.dumps(
            {
                "job_type": job_type.value,
                "job_id": str(job_id),
                "key": key,
                "value": value,
                "timestamps": {
                    "created_at": now,
                    "updated_at": now,
                },
            },
            sort_keys=True,
        ),
        now,
        now,
    )


async def _job_tag_payloads(service: MLJobService) -> dict[str, dict]:
    async with service._state_db.read_tx() as tx:
        rows = await tx.fetch("SELECT id, payload FROM job_tags ORDER BY id")
    return {str(row["id"]): json.loads(str(row["payload"])) for row in rows}


@pytest.mark.asyncio
async def test_ml_job_service_saves_override_model_path(state_db) -> None:
    service = MLJobService(state_db)
    batch_job = case_batch_prediction_job().value

    async with state_db.write_tx() as tx:
        await service._batch_prediction_jobs.upsert(
            tx,
            str(batch_job.id),
            batch_job,
        )

    await service.save_job_override_model_path(
        batch_job.id,
        "s3://pytest-model/override",
    )

    saved = await service.get_job_if_exists_generic(batch_job.id)
    assert saved.registry_metadata is not None
    assert saved.registry_metadata.override_model_path == (
        "s3://pytest-model/override")

    registry_entries = await service.select_jobs({"job_id": str(batch_job.id)})
    assert len(registry_entries) == 1
    assert registry_entries[0].override_model_path == (
        "s3://pytest-model/override")


def _metadata(
    api_key_id: str,
    *,
    name: str = "default",
    owner_user_id: str = "user_1",
    type: APIKeyType = APIKeyType.USER,
) -> APIKeyMetadataRecord:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return APIKeyMetadataRecord(
        timestamps=ResourceTimestamps(created_at=now, updated_at=now),
        id=APIKeyID(api_key_id),
        name=name,
        type=type,
        owner_user_id=owner_user_id,
        owner_user_hash="12345678",
        owner_display_name=owner_user_id,
        created_time=now,
        value_preview="pytest...secret",
    )


def _secret(api_key_id: str) -> APIKeySecretRecord:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return APIKeySecretRecord(
        timestamps=ResourceTimestamps(created_at=now, updated_at=now),
        id=APIKeyID(api_key_id),
        hash_version="v1",
        hash_algorithm="scrypt",
        hash_params={
            "n": 2,
            "r": 1,
            "p": 1,
            "dklen": 32
        },
        salt_b64="salt",
        hash_b64="hash",
    )


def _timestamps() -> ResourceTimestamps:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return ResourceTimestamps(created_at=now, updated_at=now)


async def _ensure_test_project_and_connectors(
    state_db: StateDB,
    project_id: ProjectID,
    connector_ids: list[ConnectorID],
) -> None:
    projects = JsonRepository('projects', ProjectDBEntry)
    groups = JsonRepository('groups', KumoGroupDBEntry)
    connectors = StateDBConnectorRepository()
    async with state_db.write_tx() as tx:
        group_id = GroupID('group_' + 'f' * 32)
        if await groups.get(tx, str(group_id)) is None:
            await groups.upsert(tx, str(group_id), _group(str(group_id)))
        if await projects.get(tx, str(project_id)) is None:
            await projects.upsert(
                tx,
                str(project_id),
                ProjectDBEntry(
                    id=project_id,
                    group_id=group_id,
                    slug=f"project-{str(project_id)[-8:]}",
                    display_name=str(project_id),
                    normalized_display_name=str(project_id),
                    timestamps=_timestamps(),
                ),
            )
        for connector_id in connector_ids:
            existing = await connectors.get(tx, connector_id)
            if existing is not None:
                if existing.project_id != project_id:
                    raise ValueError(
                        'Test connector belongs to another project')
                continue
            name = f'test-{str(connector_id)[-8:]}'
            await connectors.create(
                tx,
                InternalConnectorDBEntry(
                    id=connector_id,
                    project_id=project_id,
                    name=name,
                    normalized_name=name,
                    config=FileConnectorResourceConfig(
                        name=name, root_dir=f'/tmp/{project_id}/{name}'),
                    timestamps=_timestamps(),
                ),
            )


@pytest.mark.asyncio
async def test_table_service_reads_table_entries(state_db) -> None:
    repository = StateDBTableRepository()
    service = TableService(state_db)
    project_id = ProjectID("project_" + "1" * 32)
    s3_connector_id = ConnectorID("connector_" + "a" * 32)
    csv_connector_id = ConnectorID("connector_" + "b" * 32)
    await _ensure_test_project_and_connectors(
        state_db, project_id, [s3_connector_id, csv_connector_id])
    definition = TableDefinition(
        cols=[Column(name="customer_id", stype=Stype.ID, dtype=Dtype.int64)],
        source_table=S3SourceTable(
            connector_id=str(s3_connector_id),
            source={
                'mode': 'relative',
                'path': "customers.parquet"
            },
            file_type=FileType.PARQUET,
            data_source_type=DataSourceType.S3,
        ),
    )
    connected_definition = TableDefinition(
        cols=[Column(name="account_id", stype=Stype.ID, dtype=Dtype.int64)],
        source_table=S3SourceTable(
            data_source_type=DataSourceType.S3,
            connector_id=str(csv_connector_id),
            source={
                'mode': 'relative',
                'path': "accounts.parquet"
            },
            file_type=FileType.PARQUET,
        ),
    )
    entry = TableDBEntryV2(
        id=definition.fpid,
        project_id=project_id,
        config=definition,
        col_dtypes={},
        alias=TableName("customers"),
        ref_count=1,
        timestamps=_timestamps(),
    )
    connected_entry = TableDBEntryV2(
        id=connected_definition.fpid,
        project_id=project_id,
        config=connected_definition,
        col_dtypes={},
        alias=TableName("accounts"),
        ref_count=1,
        timestamps=_timestamps(),
    )

    async with state_db.write_tx() as tx:
        await repository.upsert(tx, definition.fpid, entry)
        await repository.upsert(tx, connected_definition.fpid, connected_entry)

    assert await service.get_table_entry(project_id, definition.fpid) == entry
    listed = await service.list_tables(project_id)
    assert {table.id
            for table in listed} == {
                entry.id,
                connected_entry.id,
            }
    assert await service.list_tables(
        project_id, connector_id=csv_connector_id) == [connected_entry]
    assert await service.list_tables(
        project_id, connector_id=ConnectorID("connector_" + "c" * 32)) == []


@pytest.mark.asyncio
async def test_table_service_connector_filter_does_not_list_all(
        state_db, monkeypatch) -> None:
    repository = StateDBTableRepository()
    service = TableService(state_db)
    project_id = ProjectID("project_" + "1" * 32)
    connector_id = ConnectorID("connector_" + "b" * 32)
    await _ensure_test_project_and_connectors(state_db, project_id,
                                              [connector_id])
    definition = TableDefinition(
        cols=[Column(name="account_id", stype=Stype.ID, dtype=Dtype.int64)],
        source_table=S3SourceTable(
            data_source_type=DataSourceType.S3,
            connector_id=str(connector_id),
            source={
                'mode': 'relative',
                'path': "accounts.parquet"
            },
            file_type=FileType.PARQUET,
        ),
    )
    entry = TableDBEntryV2(
        id=definition.fpid,
        project_id=project_id,
        config=definition,
        col_dtypes={},
        alias=TableName("accounts"),
        ref_count=1,
        timestamps=_timestamps(),
    )
    async with state_db.write_tx() as tx:
        await repository.upsert(tx, definition.fpid, entry)

    async def fail_list_all(*_args, **_kwargs):
        raise AssertionError("filtered table listing must not call list_all")

    monkeypatch.setattr(JsonRepository, "list_all", fail_list_all)

    assert await service.list_tables(project_id,
                                     connector_id=connector_id) == [entry]


@pytest.mark.asyncio
async def test_state_services_share_connector_authorization_service(
        state_db) -> None:
    services = StateServices(state_db)

    assert (services.connectors._connector_authorizations
            is services.connector_authorizations)
    assert (services.identities._connector_authorizations
            is services.connector_authorizations)


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 20])
async def test_graph_service_bulk_compute_inputs_use_fixed_query_count(
        state_db, monkeypatch, count) -> None:
    services = StateServices(state_db)
    project_id = ProjectID("project_" + "1" * 32)
    group_id = GroupID("group_" + "1" * 32)
    user = _user("alice", group_ids=[str(group_id)])
    await _put_group(state_db, _group(str(group_id)))
    await _put_project(
        state_db,
        _project(str(project_id), group_id=str(group_id), created_by="alice"),
    )
    connector = InternalConnectorDBEntry(
        id=ConnectorID("connector_" + "1" * 32),
        project_id=project_id,
        name="uploads",
        normalized_name="uploads",
        config=FileConnectorResourceConfig(name="uploads", root_dir="/tmp"),
        timestamps=_timestamps(),
    )
    await seed_connector(state_db, connector)
    graph_ids = []
    for index in range(count):
        table = TableDefinition(
            cols=[Column(name="id", stype=Stype.ID, dtype=Dtype.int64)],
            source_table=S3SourceTable(
                connector_id=str(connector.id),
                source={
                    'mode': 'relative',
                    'path': f"table_{index}.parquet"
                },
                file_type=FileType.PARQUET,
                data_source_type=DataSourceType.S3,
            ),
        )
        graph = GraphDefinition(tables={f"table_{index}": table}, col_groups=[])
        await services.graphs.save_graph_definition(project_id, graph)
        graph_ids.append(GraphIDv2.public(graph.fpid))

    fetch_count = 0
    real_fetch = runtime_mod._SqliteStateTx.fetch

    async def count_fetch(tx, sql, *args):
        nonlocal fetch_count
        fetch_count += 1
        return await real_fetch(tx, sql, *args)

    monkeypatch.setattr(runtime_mod._SqliteStateTx, "fetch", count_fetch)

    results = await services.graphs.get_many_graph_data_compute_inputs(
        graph_ids,
        project_id=project_id,
        user_authorization=user,
    )

    assert len(results) == count
    assert all(not isinstance(result, Exception) for result in results)
    # One bulk graph read, one bulk table read, and one fixed connector
    # authorization read. The count must remain independent of graph count.
    assert fetch_count == 3


@pytest.mark.asyncio
async def test_graph_service_clear_graph_alias_by_id(state_db) -> None:
    services = StateServices(state_db)
    project_id = ProjectID("project_" + "1" * 32)
    group_id = GroupID("group_" + "1" * 32)
    await _put_group(state_db, _group(str(group_id)))
    await _put_project(
        state_db,
        _project(str(project_id), group_id=str(group_id), created_by="alice"),
    )
    fpid = FPID("a" * 32)
    repository = StateDBGraphRepository()
    async with state_db.write_tx() as tx:
        await repository.upsert(
            tx,
            fpid,
            GraphDBEntryV2(
                id=fpid,
                project_id=project_id,
                table_name_to_dbid={},
                col_groups=[],
                alias="orders_graph",
                timestamps=_timestamps(),
            ),
        )

    entry = await services.graphs.clear_graph_alias_by_id(project_id, fpid)

    assert entry is not None
    assert entry.id == fpid
    assert entry.alias is None
    persisted = await services.graphs.get_graph_entry(project_id, fpid)
    assert persisted is not None
    assert persisted.alias is None

    # The alias is already cleared, so a second clear reports "not found",
    # as does a fingerprint that never existed.
    assert await services.graphs.clear_graph_alias_by_id(project_id,
                                                         fpid) is None
    assert await services.graphs.clear_graph_alias_by_id(
        project_id, FPID("b" * 32)) is None


@pytest.mark.asyncio
async def test_ml_job_service_lists_open_parent_jobs_without_list_all(
        state_db, monkeypatch) -> None:
    service = MLJobService(state_db)

    training_job = case_training_job().value
    training_job.id = TrainingJobID.make_random_id()
    closed_training_job = case_training_job().value
    closed_training_job.id = TrainingJobID.make_random_id()
    closed_training_job.execution.workflow_end_state = (
        WorkflowExecutionStatus.COMPLETED)

    batch_job = case_batch_prediction_job().value
    batch_job.id = BatchPredictionJobID.make_random_id()
    closed_batch_job = case_batch_prediction_job().value
    closed_batch_job.id = BatchPredictionJobID.make_random_id()
    closed_batch_job.execution.workflow_end_state = (
        WorkflowExecutionStatus.COMPLETED)

    training_child = case_generate_train_table_job().value
    training_child.id = GenerateTrainTableJobID.as_child_of(training_job.id)
    orphan_train_child = case_generate_train_table_job().value
    orphan_train_child.id = GenerateTrainTableJobID.make_random_id()
    closed_orphan_train_child = case_generate_train_table_job().value
    closed_orphan_train_child.id = GenerateTrainTableJobID.make_random_id()
    closed_orphan_train_child.execution.workflow_end_state = (
        WorkflowExecutionStatus.COMPLETED)

    batch_child = case_generate_pred_table_job().value
    batch_child.id = GeneratePredictionTableJobID.as_child_of(batch_job.id)
    orphan_pred_child = case_generate_pred_table_job().value
    orphan_pred_child.id = GeneratePredictionTableJobID.make_random_id()

    async with state_db.write_tx() as tx:
        await service._training_jobs.upsert(tx, str(training_job.id),
                                            training_job)
        await service._training_jobs.upsert(tx, str(closed_training_job.id),
                                            closed_training_job)
        await service._batch_prediction_jobs.upsert(
            tx,
            str(batch_job.id),
            batch_job,
        )
        await service._batch_prediction_jobs.upsert(
            tx,
            str(closed_batch_job.id),
            closed_batch_job,
        )
        await service._generate_train_table_jobs.upsert(tx,
                                                        str(training_child.id),
                                                        training_child)
        await service._generate_train_table_jobs.upsert(
            tx, str(orphan_train_child.id), orphan_train_child)
        await service._generate_train_table_jobs.upsert(
            tx, str(closed_orphan_train_child.id), closed_orphan_train_child)
        await service._generate_pred_table_jobs.upsert(tx, str(batch_child.id),
                                                       batch_child)
        await service._generate_pred_table_jobs.upsert(
            tx, str(orphan_pred_child.id), orphan_pred_child)

    async def fail_list_all(*_args, **_kwargs):
        raise AssertionError("open parent job listing must not call list_all")

    monkeypatch.setattr(JsonRepository, "list_all", fail_list_all)

    open_jobs = await service.list_open_parent_jobs()

    assert {job.id
            for job in open_jobs} == {
                training_job.id,
                batch_job.id,
                orphan_train_child.id,
                orphan_pred_child.id,
            }


async def _put_group(state_db: StateDB, group: KumoGroupDBEntry) -> None:
    repository = JsonRepository("groups", KumoGroupDBEntry)
    async with state_db.write_tx() as tx:
        await repository.upsert(tx, str(group.id), group)


async def _put_project(state_db: StateDB, project: ProjectDBEntry) -> None:
    repository = JsonRepository("projects", ProjectDBEntry)
    async with state_db.write_tx() as tx:
        await repository.upsert(tx, str(project.id), project)


async def _put_tables(
    state_db: StateDB,
    project_id: ProjectID | str,
    fpids: list[FPID],
) -> None:
    project_id = ProjectID(str(project_id))
    connector_id = ConnectorID('connector_' + 'f' * 32)
    await _ensure_test_project_and_connectors(state_db, project_id,
                                              [connector_id])
    repository = StateDBTableRepository()
    async with state_db.write_tx() as tx:
        for fpid in fpids:
            definition = TableDefinition(
                cols=[Column(name="id", stype=Stype.ID, dtype=Dtype.int64)],
                source_table=S3SourceTable(
                    connector_id=connector_id,
                    source={
                        'mode': 'relative',
                        'path': f"{fpid}.parquet"
                    },
                    file_type=FileType.PARQUET,
                    data_source_type=DataSourceType.S3,
                ),
            )
            await repository.upsert(
                tx,
                fpid,
                TableDBEntryV2(
                    id=fpid,
                    project_id=project_id,
                    config=definition,
                    col_dtypes={},
                    timestamps=_timestamps(),
                ),
            )


async def _put_graphs(
    state_db: StateDB,
    project_id: ProjectID | str,
    fpids: list[FPID],
) -> None:
    project_id = ProjectID(str(project_id))
    repository = StateDBGraphRepository()
    async with state_db.write_tx() as tx:
        for fpid in fpids:
            await repository.upsert(
                tx,
                fpid,
                GraphDBEntryV2(
                    id=fpid,
                    project_id=project_id,
                    table_name_to_dbid={},
                    col_groups=[],
                    timestamps=_timestamps(),
                ),
            )


def _group(group_id: str) -> KumoGroupDBEntry:
    return KumoGroupDBEntry(
        id=GroupID(group_id),
        slug=f"group-{group_id[-8:]}",
        display_name=group_id,
        normalized_display_name=group_id.casefold(),
        timestamps=_timestamps(),
    )


def _project(
    project_id: str,
    *,
    group_id: str,
    created_by: str,
    is_private: bool = False,
    status: ProjectStatus = ProjectStatus.ACTIVE,
    display_name: str | None = None,
) -> ProjectDBEntry:
    return ProjectDBEntry(
        id=ProjectID(project_id),
        group_id=GroupID(group_id),
        slug=f"project-{project_id[-8:]}",
        display_name=display_name or project_id,
        normalized_display_name=(display_name or project_id).casefold(),
        created_by=created_by,
        is_private=is_private,
        status=status,
        timestamps=_timestamps(),
    )


def _user(
    user_id: str,
    *,
    group_ids: list[str] | None = None,
    is_admin: bool = False,
    external_membership_unavailable: bool = False,
) -> UserAuthorization:
    permissions = [
        Permissions.READ.compose(),
        Permissions.WRITE.compose(),
    ]
    if is_admin:
        permissions.append(Permissions.ADMIN.compose())
    return UserAuthorization(
        user_id=UserID(user_id),
        access_token="token",
        permissions=permissions,
        kumo_group_ids=tuple(group_ids or []),
        external_membership_unavailable=external_membership_unavailable,
    )


@pytest.mark.asyncio
async def test_api_key_service_create_lists_and_rejects_duplicate_names(
        state_db) -> None:
    service = APIKeyService(state_db)

    await service.create_unique(
        metadata=_metadata("k-1"),
        secret=_secret("k-1"),
    )

    with pytest.raises(APIKeyNameAlreadyExistsError):
        await service.create_unique(
            metadata=_metadata("k-2"),
            secret=_secret("k-2"),
        )

    listed = await service.list_metadata(type=APIKeyType.USER)
    assert [record.id for record in listed] == [APIKeyID("k-1")]


@pytest.mark.asyncio
async def test_api_key_service_delete_if_and_updates_last_used_time(
        state_db) -> None:
    service = APIKeyService(state_db)
    key_id = APIKeyID("k-1")

    await service.create_unique(
        metadata=_metadata(str(key_id)),
        secret=_secret(str(key_id)),
    )

    assert await service.delete_if(key_id, lambda _record: False) is None
    await service.update_last_used_time(
        key_id,
        datetime(2026, 2, 1, tzinfo=timezone.utc),
    )
    candidate = await service.find_verification_candidate(
        owner_user_hash="12345678",
        name="default",
    )
    assert candidate is not None
    assert candidate.secret.last_used_time == datetime(2026, 2, 1,
                                                       tzinfo=timezone.utc)

    deleted = await service.delete_if(key_id, lambda _record: True)
    assert deleted is not None
    assert await service.get_metadata(key_id) is None


@pytest.mark.asyncio
async def test_group_service_maps_plain_and_hashed_idp_groups(state_db) -> None:
    service = GroupService(state_db)
    group_a = "group_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    group_b = "group_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    admin = _user("admin", is_admin=True)
    await _put_group(state_db, _group(group_a))
    await _put_group(state_db, _group(group_b))
    await service.create_external_mapping(GroupID(group_a), "direct_connect",
                                          "idp-a", principal=admin)
    await service.create_external_mapping(GroupID(group_b), "direct_connect",
                                          "idp-b", principal=admin)

    assert await service.map_idp_group_ids(["idp-b", "", "idp-a", "idp-b"]) == [
        group_a,
        group_b,
    ]
    assert await service.map_idp_group_hashes([
        hash_idp_group_id(ExternalIdentityProviderID.DIRECT_CONNECT, "idp-a"),
        "missing",
    ]) == [group_a]


@pytest.mark.asyncio
async def test_group_service_crud_and_user_membership_readback(
        state_db, monkeypatch) -> None:
    service = GroupService(state_db)
    admin = _user("admin", is_admin=True)

    created = await service.create_kumo_groups([
        KumoGroupCreateSpec(
            display_name="Engineering",
            description="Builds product",
        ),
        KumoGroupCreateSpec(display_name="Finance", ),
    ], principal=admin)

    assert [group.display_name for group in created] == [
        "Engineering",
        "Finance",
    ]
    engineering_id = created[0].id
    await service.create_external_mapping(engineering_id, "direct_connect",
                                          "idp-eng", principal=admin)
    await service.create_external_mapping(created[1].id, "direct_connect",
                                          "idp-finance", principal=admin)
    listed = await service.list_kumo_groups(admin)
    assert sorted(group.display_name for group in listed) == [
        "Engineering",
        "Finance",
    ]
    assert await service.map_idp_group_ids(["idp-eng"]) == [str(engineering_id)]

    async def fail_list_all(*_args, **_kwargs):
        raise AssertionError(
            "group display-name/deleted lookups must not call list_all")

    monkeypatch.setattr(JsonRepository, "list_all", fail_list_all)

    assert (await service.get_kumo_group_by_alias(
        "Engineering",
        _user("alice", group_ids=[str(engineering_id)]),
    )).id == engineering_id
    assert (await service.get_kumo_group(str(engineering_id),
                                         admin)).display_name == "Engineering"
    user_groups = await service.list_user_groups(
        _user("alice", group_ids=[str(engineering_id)]))
    assert [(group.id, group.display_name) for group in user_groups] == [
        (engineering_id, "Engineering"),
    ]

    updated = await service.update_kumo_group(
        engineering_id,
        display_name="Eng",
        description="Updated",
        principal=admin,
    )
    assert updated.display_name == "Eng"
    assert updated.description == "Updated"
    await service.delete_kumo_group(str(engineering_id), admin)
    assert [
        group.display_name for group in await service.list_kumo_groups(admin)
    ] == ["Finance"]


@pytest.mark.asyncio
async def test_project_service_lists_visible_project_ids(state_db,
                                                         monkeypatch) -> None:
    service = ProjectService(state_db)
    group_a = "group_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    group_b = "group_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    public = "project_dddddddddddddddddddddddddddddddd"
    private_owned = "project_cccccccccccccccccccccccccccccccc"
    private_other = "project_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    other_group = "project_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    await _put_project(
        state_db,
        _project(public, group_id=group_a, created_by="alice"),
    )
    await _put_project(
        state_db,
        _project(
            private_owned,
            group_id=group_a,
            created_by="bob",
            is_private=True,
        ),
    )
    await _put_project(
        state_db,
        _project(
            private_other,
            group_id=group_a,
            created_by="alice",
            is_private=True,
        ),
    )
    await _put_project(
        state_db,
        _project(other_group, group_id=group_b, created_by="alice"),
    )

    async def fail_list_all(*_args, **_kwargs):
        raise AssertionError("project visibility must not call list_all")

    monkeypatch.setattr(JsonRepository, "list_all", fail_list_all)

    assert await service.list_visible_project_ids(
        _user("bob", group_ids=[group_a])) == [
            private_owned,
            public,
        ]
    assert await service.list_visible_project_ids(_user("admin",
                                                        is_admin=True)) == [
                                                            other_group,
                                                            private_other,
                                                            private_owned,
                                                            public,
                                                        ]


@pytest.mark.asyncio
async def test_project_service_crud_preserves_access_rules(state_db) -> None:
    service = ProjectService(state_db)
    group_id = "group_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    other_group_id = "group_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    await _put_group(state_db, _group(group_id))
    await _put_group(state_db, _group(other_group_id))
    owner = _user("alice", group_ids=[group_id])
    group_member = _user("bob", group_ids=[group_id])
    outsider = _user("mallory", group_ids=[other_group_id])

    created = await service.create_project(
        name="Risk",
        group_id=group_id,
        description="Initial",
        is_private=True,
        user_authorization=owner,
    )

    assert created.display_name == "Risk"
    assert created.created_by == "alice"
    assert created.group_id == group_id
    assert created.status == ProjectStatus.ACTIVE
    assert await service.get_project(created.id, owner) == created
    assert await service.list_visible_projects(group_member) == []

    with pytest.raises(ProjectNotFoundError, match="not found"):
        await service.get_project(created.id, group_member)
    unavailable_owner = _user('alice', external_membership_unavailable=True)
    with pytest.raises(ExternalMembershipUnavailableError):
        await service.update_project(
            created.id,
            description='must fail closed',
            user_authorization=unavailable_owner,
        )
    with pytest.raises(ProjectForbiddenError,
                       match="Not a member of this group"):
        await service.create_project(
            name="Blocked",
            group_id=group_id,
            user_authorization=outsider,
        )
    with pytest.raises(GroupNotFoundError, match="Group not found"):
        await service.create_project(
            name="Missing",
            group_id="group_cccccccccccccccccccccccccccccccc",
            user_authorization=outsider,
        )

    updated = await service.update_project(
        created.id,
        name="Risk shared",
        description="Updated",
        is_private=False,
        user_authorization=owner,
    )
    assert updated.display_name == "Risk shared"
    assert updated.description == "Updated"
    assert updated.is_private is False
    assert await service.get_project(created.id, group_member) == updated

    member_updated = await service.update_project(
        created.id,
        description="Member-maintained resources",
        user_authorization=group_member,
    )
    assert member_updated.display_name == "Risk shared"
    assert member_updated.description == "Member-maintained resources"
    with pytest.raises(ProjectForbiddenError,
                       match="project owner or an administrator"):
        await service.update_project(
            created.id,
            name="Risk team",
            user_authorization=group_member,
        )
    with pytest.raises(ProjectForbiddenError,
                       match="project owner or an administrator"):
        await service.update_project(
            created.id,
            is_private=True,
            user_authorization=group_member,
        )
    with pytest.raises(ProjectForbiddenError,
                       match="project owner or an administrator"):
        await service.archive_project(created.id, group_member)

    admin_updated = await service.update_project(
        created.id,
        name="Risk team",
        is_private=False,
        user_authorization=_user("admin", is_admin=True),
    )
    assert admin_updated.display_name == "Risk team"

    archived = await service.archive_project(created.id, owner)
    assert archived.status == ProjectStatus.ARCHIVED
    assert await service.get_project(created.id, owner) == archived
    with pytest.raises(ProjectNotAvailableError, match='cannot be mutated'):
        await service.update_project(
            created.id,
            description='archived mutation',
            user_authorization=owner,
        )
    with pytest.raises(ProjectNotAvailableError, match='cannot be mutated'):
        await service.archive_project(created.id, owner)
    with pytest.raises(ProjectNotAvailableError,
                       match="cannot accept new resources"):
        await service.validate_project_for_creation(created.id, owner)


@pytest.mark.asyncio
async def test_project_names_are_global_normalized_and_archived_names_reserved(
        state_db) -> None:
    service = ProjectService(state_db)
    group_a = "group_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    group_b = "group_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    await _put_group(state_db, _group(group_a))
    await _put_group(state_db, _group(group_b))
    user = _user("alice", group_ids=[group_a, group_b])

    first = await service.create_project(
        name="  Analytics  ",
        group_id=group_a,
        user_authorization=user,
    )
    with pytest.raises(ProjectNameConflictError,
                       match="Project name is not available"):
        await service.create_project(
            name="ＡＮＡＬＹＴＩＣＳ",
            group_id=group_b,
            user_authorization=user,
        )

    await service.archive_project(first.id, user)
    with pytest.raises(ProjectNameConflictError,
                       match="Project name is not available"):
        await service.create_project(
            name="analytics",
            group_id=group_b,
            user_authorization=user,
        )


@pytest.mark.asyncio
async def test_project_update_enforces_global_normalized_name(state_db) -> None:
    service = ProjectService(state_db)
    group_a = "group_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    group_b = "group_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    await _put_group(state_db, _group(group_a))
    await _put_group(state_db, _group(group_b))
    user = _user("alice", group_ids=[group_a, group_b])
    await service.create_project(name="Risk", group_id=group_a,
                                 user_authorization=user)
    other = await service.create_project(name="Forecasting", group_id=group_b,
                                         user_authorization=user)

    with pytest.raises(ProjectNameConflictError,
                       match="Project name is not available"):
        await service.update_project(
            other.id,
            name="  ＲＩＳＫ  ",
            user_authorization=user,
        )


@pytest.mark.asyncio
async def test_concurrent_project_create_and_update_have_one_winner(
        state_db) -> None:
    service = ProjectService(state_db)
    group_a = "group_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    group_b = "group_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    await _put_group(state_db, _group(group_a))
    await _put_group(state_db, _group(group_b))
    user = _user("alice", group_ids=[group_a, group_b])

    created = await asyncio.gather(
        service.create_project(name="Concurrent", group_id=group_a,
                               user_authorization=user),
        service.create_project(
            name=" ＣＯＮＣＵＲＲＥＮＴ ",
            group_id=group_b,
            user_authorization=user,
        ),
        return_exceptions=True,
    )
    assert sum(isinstance(result, ProjectDBEntry) for result in created) == 1
    assert sum(
        isinstance(result, ProjectNameConflictError) for result in created) == 1

    first = await service.create_project(name="First", group_id=group_a,
                                         user_authorization=user)
    second = await service.create_project(name="Second", group_id=group_b,
                                          user_authorization=user)
    updated = await asyncio.gather(
        service.update_project(first.id, name="Winner",
                               user_authorization=user),
        service.update_project(second.id, name=" ＷＩＮＮＥＲ ",
                               user_authorization=user),
        return_exceptions=True,
    )
    assert sum(isinstance(result, ProjectDBEntry) for result in updated) == 1
    assert sum(
        isinstance(result, ProjectNameConflictError) for result in updated) == 1


@pytest.mark.asyncio
async def test_project_service_validates_project_name(state_db,
                                                      monkeypatch) -> None:
    service = ProjectService(state_db)
    group_id = "group_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    other_group_id = "group_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    project_id = "project_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    user = _user("alice", group_ids=[group_id])
    await _put_group(state_db, _group(group_id))
    await _put_group(state_db, _group(other_group_id))
    await _put_project(
        state_db,
        _project(
            project_id,
            group_id=group_id,
            created_by="alice",
            display_name="Existing",
        ),
    )
    await _put_project(
        state_db,
        _project(
            "project_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            group_id=other_group_id,
            created_by="mallory",
            display_name="Hidden",
        ),
    )

    async def fail_list_all(*_args, **_kwargs):
        raise AssertionError("project name validation must not call list_all")

    monkeypatch.setattr(JsonRepository, "list_all", fail_list_all)

    assert not (await service.validate_project_name(
        name="",
        user_authorization=user,
    )).is_valid
    duplicate = await service.validate_project_name(
        name="Existing",
        user_authorization=user,
    )
    assert duplicate.is_valid is False
    assert duplicate.error_msg == "Project name is not available."
    hidden_duplicate = await service.validate_project_name(
        name=" hidden ",
        user_authorization=user,
    )
    assert hidden_duplicate.is_valid is False
    assert hidden_duplicate.error_msg == "Project name is not available."
    assert (await service.validate_project_name(
        name="Existing",
        exclude_project_id=project_id,
        user_authorization=user,
    )).is_valid
    assert (await service.validate_project_name(
        name=' ' + 'a' * 64 + ' ',
        user_authorization=user,
    )).is_valid
    too_long = await service.validate_project_name(
        name=' ' + 'a' * 65 + ' ',
        user_authorization=user,
    )
    assert too_long.is_valid is False
    assert too_long.error_msg == 'Project name cannot exceed 64 characters.'


def _public_table_id(fpid_character: str) -> TableIDv2:
    return TableIDv2.public(
        table_name=TableName("NA"),
        fpid=FPID(fpid_character * 32),
        pquery_id=None,
    )


@pytest.mark.asyncio
async def test_ml_job_service_updates_training_graph_version(state_db) -> None:
    service = MLJobService(state_db)
    training_job = case_training_job().value
    training_job.graph_data_version = None
    repository = JsonRepository("training_jobs", type(training_job))

    async with state_db.write_tx() as tx:
        await repository.upsert(tx, str(training_job.id), training_job)

    await service.save_training_job_graph_version(
        training_job.id,
        "graph-version-1",
    )

    async with state_db.read_tx() as tx:
        row = await repository.get(tx, str(training_job.id))

    assert row is not None
    assert row.payload.graph_data_version == "graph-version-1"
    # A write that is not a QUEUED -> RUNNING transition preserves the
    # recorded admission time rather than re-pinning it to created_at.
    assert row.payload.execution.start_time == training_job.execution.start_time
    assert row.payload.execution.last_updated_at is not None


@pytest.mark.asyncio
async def test_ml_job_service_updates_job_outputs(state_db) -> None:
    service = MLJobService(state_db)
    training_job = case_training_job().value
    batch_prediction_job = case_batch_prediction_job().value
    batch_prediction_row = batch_prediction_job
    training_output = TrainingJobOutput(
        status=JobStatus.RUNNING,
        log=JobEventLogEntry.starting(),
    )
    batch_prediction_output = BatchPredictionJobOutput(
        status=JobStatus.RUNNING,
        log=JobEventLogEntry.starting(),
    )

    # Persist the training job as QUEUED so the first output write below is a
    # genuine QUEUED -> RUNNING admission and stamps execution.start_time.
    training_job.execution.status = JobStatus.QUEUED

    async with state_db.write_tx() as tx:
        await JsonRepository("training_jobs",
                             type(training_job)).upsert(tx,
                                                        str(training_job.id),
                                                        training_job)
        await JsonRepository("batch_prediction_jobs",
                             BatchPredictionJobDBEntry).upsert(
                                 tx, str(batch_prediction_row.id),
                                 batch_prediction_row)

    await service.save_training_job_output(training_job.id, training_output)
    await service.save_batch_prediction_job_output(batch_prediction_job.id,
                                                   batch_prediction_output)

    async with state_db.read_tx() as tx:
        training_row = await JsonRepository("training_jobs",
                                            type(training_job)).get(
                                                tx, str(training_job.id))
        batch_row = await JsonRepository("batch_prediction_jobs",
                                         BatchPredictionJobDBEntry).get(
                                             tx, str(batch_prediction_row.id))

    assert training_row is not None
    assert training_row.payload.execution.status == JobStatus.RUNNING
    assert (training_row.payload.execution.start_time
            > training_job.timestamps.created_at)
    assert batch_row is not None
    assert batch_row.payload.execution is not None
    assert batch_row.payload.execution.status == JobStatus.RUNNING
    assert batch_row.payload.status == JobStatus.RUNNING

    reserved_id = await service.reserve_training_job_id()
    assert isinstance(reserved_id, TrainingJobID)
    assert reserved_id != training_job.id


@pytest.mark.asyncio
async def test_batch_prediction_terminal_retry_keeps_original_end_time(
        state_db, monkeypatch) -> None:
    service = MLJobService(state_db)
    job = case_batch_prediction_job().value
    job.execution.status = JobStatus.RUNNING
    job.execution.workflow_end_state = None

    async with state_db.write_tx() as tx:
        await JsonRepository('batch_prediction_jobs',
                             BatchPredictionJobDBEntry).upsert(
                                 tx, str(job.id), job)

    first_end_time = datetime(2026, 8, 16, 8, 34, tzinfo=timezone.utc)
    later_retry_time = datetime(2026, 8, 16, 8, 40, tzinfo=timezone.utc)
    monkeypatch.setattr('statedb.db.ml_jobs.now_utc', lambda: first_end_time)
    terminal = BatchPredictionJobOutput(
        status=JobStatus.DONE,
        log=JobEventLogEntry.done(first_end_time),
        results=BatchPredictionJobResults(key_cols=['entity_id']),
        workflow_end_state=WorkflowExecutionStatus.COMPLETED,
    )
    await service.save_batch_prediction_job_output(job.id, terminal)

    monkeypatch.setattr('statedb.db.ml_jobs.now_utc', lambda: later_retry_time)
    await service.save_batch_prediction_job_output(job.id, deepcopy(terminal))

    # Auxiliary work such as detached BP statistics cannot rewrite the
    # prediction result after durable output has frozen the job DONE.
    await service.save_batch_prediction_job_output(
        job.id,
        BatchPredictionJobOutput.failed(
            err_msg='late BP statistics failure',
            workflow_end_state=WorkflowExecutionStatus.FAILED,
        ),
    )

    saved = await service.get_job_if_exists_generic(job.id)
    assert saved is not None
    assert saved.execution.status == JobStatus.DONE
    assert saved.execution.last_updated_at == first_end_time
    assert saved.execution.log.last_updated_at == first_end_time


@pytest.mark.asyncio
async def test_ml_job_service_cancels_parent_and_child_atomically(
        state_db) -> None:
    service = MLJobService(state_db)
    training_job = case_training_job().value
    training_job.id = TrainingJobID.make_random_id()
    training_job.execution.status = JobStatus.RUNNING
    training_job.execution.workflow_end_state = None
    child_job = case_generate_train_table_job().value
    child_job.id = GenerateTrainTableJobID.as_child_of(training_job.id)
    child_job.execution.status = JobStatus.RUNNING
    child_job.execution.workflow_end_state = None
    training_job.generate_train_table_job_id = child_job.id

    async with state_db.write_tx() as tx:
        await JsonRepository("training_jobs",
                             type(training_job)).upsert(tx,
                                                        str(training_job.id),
                                                        training_job)
        await JsonRepository("generate_train_table_jobs",
                             type(child_job)).upsert(tx, str(child_job.id),
                                                     child_job)

    assert await service.cancel_job_if_open_generic(training_job.id)

    saved_parent = await service.get_training_job(training_job.id)
    saved_child = await service.get_generate_train_table_job(
        child_job.id, project_ids=None)
    assert saved_parent is not None
    assert saved_child is not None
    assert saved_parent.execution.status == JobStatus.CANCELLED
    assert saved_parent.execution.workflow_end_state == (
        WorkflowExecutionStatus.CANCELED)
    assert saved_child.execution.status == JobStatus.CANCELLED
    assert saved_child.execution.workflow_end_state == (
        WorkflowExecutionStatus.CANCELED)
    router = TrainingRouterV1.__new__(TrainingRouterV1)
    router._customer_id = "test"
    public_parent = router._make_training_job_resource_v1(
        saved_parent,
        JobTags(timestamps=saved_parent.timestamps,
                tags={TAG_KEY_PQUERY_NAME: JobTagValue("p")}),
    )
    assert public_parent.job_status_report.event_log[-1].detail == (
        "TRAINING_JOB was cancelled by user")
    assert saved_child.execution.log.detail == (
        f"This job is cancelled by its parent job {training_job.id} "
        "which has reached CANCELLED state")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parent_case, child_case, child_id_of, save_output, parent_status, "
    "child_status, expected_status, expected_end_state",
    [
        (case_training_job, case_generate_train_table_job,
         GenerateTrainTableJobID.as_child_of, "save_training_job_output",
         JobStatus.FAILED, JobStatus.QUEUED, JobStatus.CANCELLED,
         WorkflowExecutionStatus.CANCELED),
        (case_training_job, case_generate_train_table_job,
         GenerateTrainTableJobID.as_child_of, "save_training_job_output",
         JobStatus.DONE, JobStatus.QUEUED, JobStatus.CANCELLED,
         WorkflowExecutionStatus.CANCELED),
        (case_training_job, case_generate_train_table_job,
         GenerateTrainTableJobID.as_child_of, "save_training_job_output",
         JobStatus.FAILED, JobStatus.DONE, JobStatus.DONE, None),
        (case_training_job, case_generate_train_table_job,
         lambda _: GenerateTrainTableJobID.make_random_id(),
         "save_training_job_output", JobStatus.FAILED, JobStatus.QUEUED,
         JobStatus.QUEUED, None),
        (case_batch_prediction_job, case_generate_pred_table_job,
         GeneratePredictionTableJobID.as_child_of,
         "save_batch_prediction_job_output", JobStatus.FAILED, JobStatus.QUEUED,
         JobStatus.CANCELLED, WorkflowExecutionStatus.CANCELED),
    ],
    ids=("training-failed-parent", "training-done-parent",
         "training-terminal-child", "training-decoupled-child",
         "batch-prediction-open-child"),
)
async def test_ml_job_service_terminal_output_sweeps_open_child(
        state_db, parent_case, child_case, child_id_of, save_output,
        parent_status, child_status, expected_status,
        expected_end_state) -> None:
    """A workflow's own terminal write must sweep its implicit child.

    ``finalize_job``'s safety-net sweep is skipped once the workflow has set
    ``workflow_end_state`` itself, which is the common failure path, so a
    child whose workflow never spawned was left QUEUED forever. A parent that
    succeeds sweeps the same way, because the child still never recorded a
    generated table. A child that reached its own terminal state, and a
    decoupled table-generation job that merely got referenced, are left alone.
    """
    service = MLJobService(state_db)
    parent = parent_case().value
    parent.execution.status = JobStatus.RUNNING
    parent.execution.workflow_end_state = None
    child = child_case().value
    child.id = child_id_of(parent.id)
    child.execution.status = child_status
    child.execution.workflow_end_state = None
    if parent_case is case_training_job:
        parent.generate_train_table_job_id = child.id
    elif parent_case is case_batch_prediction_job:
        # A batch-prediction job references its table-generation job; the two
        # sources are mutually exclusive (see ``create_batch_prediction_job``).
        parent.config.custom_pred_table_source = None
        parent.config.generate_pred_table_job_id = child.id

    repository_for = service._repository_for_job_type
    async with state_db.write_tx() as tx:
        await repository_for(JobType.from_id(parent.id)
                             ).upsert(tx, str(parent.id), parent)
        await repository_for(JobType.from_id(child.id)
                             ).upsert(tx, str(child.id), child)

    terminal = deepcopy(parent.execution)
    terminal.status = parent_status
    terminal.workflow_end_state = WorkflowExecutionStatus.COMPLETED
    if parent_status == JobStatus.FAILED:
        terminal.failure_info = JobFailureInfo(message="boom", stacktrace="")
        terminal.log = JobEventLogEntry.failed(message="boom")
    else:
        terminal.results = TrainingJobResults()
        terminal.log = JobEventLogEntry.done()
    await getattr(service, save_output)(parent.id, terminal)

    saved_parent = await service.get_job_if_exists_generic(parent.id)
    saved_child = await service.get_job_if_exists_generic(child.id)
    assert saved_parent is not None and saved_child is not None
    assert saved_parent.execution.status == parent_status
    assert saved_child.execution.status == expected_status
    assert saved_child.execution.workflow_end_state == expected_end_state
    if expected_status == JobStatus.CANCELLED:
        assert saved_child.execution.log.detail == (
            f"This job is cancelled by its parent job {parent.id} "
            f"which has reached {parent_status.value} state")


@pytest.mark.asyncio
async def test_ml_job_service_rejected_parent_write_leaves_child_alone(
        state_db) -> None:
    """A write the open-gate rejects must not sweep the child either.

    Parents stranded by this defect are exactly terminal-parent plus
    open-child, so a replayed workflow write must not relabel the child with an
    outcome the rejected write claimed.
    """
    service = MLJobService(state_db)
    parent = case_training_job().value
    parent.execution.status = JobStatus.FAILED
    parent.execution.failure_info = JobFailureInfo(message="real",
                                                   stacktrace="")
    parent.execution.log = JobEventLogEntry.failed(message="real")
    parent.execution.workflow_end_state = WorkflowExecutionStatus.COMPLETED
    child = case_generate_train_table_job().value
    child.id = GenerateTrainTableJobID.as_child_of(parent.id)
    child.execution.status = JobStatus.QUEUED
    child.execution.workflow_end_state = None

    async with state_db.write_tx() as tx:
        await service._training_jobs.upsert(tx, str(parent.id), parent)
        await service._generate_train_table_jobs.upsert(tx, str(child.id),
                                                        child)

    await service.save_training_job_output(
        parent.id,
        TrainingJobOutput(status=JobStatus.DONE, results=TrainingJobResults(),
                          log=JobEventLogEntry.done(),
                          workflow_end_state=WorkflowExecutionStatus.COMPLETED))

    saved_parent = await service.get_training_job(parent.id)
    saved_child = await service.get_generate_train_table_job(
        child.id, project_ids=None)
    assert saved_parent is not None and saved_child is not None
    assert saved_parent.execution.status == JobStatus.FAILED
    assert saved_child.execution.status == JobStatus.QUEUED


@pytest.mark.asyncio
async def test_ml_job_service_mark_workflow_not_found_messages_use_enum_values(
        state_db) -> None:
    service = MLJobService(state_db)
    training_job = case_training_job().value
    training_job.id = TrainingJobID.make_random_id()
    training_job.execution.status = JobStatus.RUNNING
    training_job.execution.workflow_end_state = None

    async with state_db.write_tx() as tx:
        await JsonRepository("training_jobs",
                             type(training_job)).upsert(tx,
                                                        str(training_job.id),
                                                        training_job)

    await service.mark_workflow_not_found(training_job.id,
                                          datetime.now(timezone.utc))

    saved = await service.get_training_job(training_job.id)
    assert saved is not None
    assert saved.execution.status == JobStatus.FAILED
    assert saved.execution.workflow_end_state == (
        WorkflowExecutionStatus.TIMED_OUT)
    assert saved.execution.failure_info is not None
    assert saved.execution.failure_info.message == (
        "This TRAINING_JOB has expired and its workflow is not found.")
    assert saved.execution.log.detail == (
        "TRAINING_JOB workflow has expired and not found.")


@pytest.mark.asyncio
async def test_ml_job_service_reads_job_results_from_repositories(
        state_db) -> None:
    service = MLJobService(state_db)

    train_table = case_generate_train_table_job().value
    train_table.id = GenerateTrainTableJobID.make_random_id()
    pred_table = case_generate_pred_table_job().value
    pred_table.id = GeneratePredictionTableJobID.make_random_id()
    training_job = case_training_job().value
    training_job.id = TrainingJobID.make_random_id()
    training_job.generate_train_table_job_id = str(train_table.id)
    pred_table_tags = {
        TAG_KEY_PQUERY_NAME: JobTagValue("riskpquery"),
    }
    child_train_table = case_generate_train_table_job().value
    child_train_table.id = GenerateTrainTableJobID.as_child_of(training_job.id)
    child_pred_table = case_generate_pred_table_job().value
    running_batch_prediction = case_batch_prediction_job().value
    running_batch_prediction.id = BatchPredictionJobID.make_random_id()
    child_pred_table.id = GeneratePredictionTableJobID.as_child_of(
        running_batch_prediction.id)
    stopped_batch_prediction = case_batch_prediction_job().value
    stopped_batch_prediction.id = BatchPredictionJobID.make_random_id()
    stopped_batch_prediction.execution.status = JobStatus.CANCELLED
    stopped_batch_prediction.execution.workflow_end_state = (
        WorkflowExecutionStatus.COMPLETED)
    stopped_batch_prediction._sync_index_fields()
    async with state_db.write_tx() as tx:
        await JsonRepository("generate_train_table_jobs",
                             type(train_table)).upsert(tx, str(train_table.id),
                                                       train_table)
        await JsonRepository("generate_train_table_jobs",
                             type(child_train_table)).upsert(
                                 tx, str(child_train_table.id),
                                 child_train_table)
        await JsonRepository("generate_pred_table_jobs",
                             type(pred_table)).upsert(tx, str(pred_table.id),
                                                      pred_table)
        await JsonRepository("generate_pred_table_jobs",
                             type(child_pred_table)).upsert(
                                 tx, str(child_pred_table.id), child_pred_table)
        await JsonRepository("training_jobs",
                             type(training_job)).upsert(tx,
                                                        str(training_job.id),
                                                        training_job)
        await _insert_normalized_job_tag_row(
            tx,
            row_id="pred-table-pquery",
            job_type=JobType.GENERATE_PREDICTION_TABLE_JOB,
            job_id=pred_table.id,
            key=str(TAG_KEY_PQUERY_NAME),
            value="riskpquery",
        )
        batch_repository = JsonRepository("batch_prediction_jobs",
                                          type(running_batch_prediction))
        await batch_repository.upsert(tx, str(running_batch_prediction.id),
                                      running_batch_prediction)
        await batch_repository.upsert(tx, str(stopped_batch_prediction.id),
                                      stopped_batch_prediction)

    assert await service.get_generate_train_table_job(
        train_table.id, project_ids=None) == train_table
    assert await service.get_generate_train_table_jobs([
        GenerateTrainTableJobID.make_random_id(),
        train_table.id,
    ], project_ids=None) == [train_table]
    assert await service.get_generate_pred_table_job(pred_table.id
                                                     ) == pred_table
    assert await service.get_generate_pred_table_jobs([
        GeneratePredictionTableJobID.make_random_id(),
        pred_table.id,
    ], project_ids=None) == [pred_table]
    assert await service.get_training_job(training_job.id) == training_job
    aggregate_jobs = await service.list_aggregate_job_entries(
        project_id=running_batch_prediction.project_id)
    assert {job.id
            for job in aggregate_jobs} == {
                training_job.id,
                running_batch_prediction.id,
                stopped_batch_prediction.id,
                train_table.id,
                child_train_table.id,
                pred_table.id,
                child_pred_table.id,
            }
    listed_pred_tags = await service.list_job_tags(
        JobType.GENERATE_PREDICTION_TABLE_JOB,
        {TAG_KEY_PQUERY_NAME: "riskpquery"},
    )
    assert listed_pred_tags[pred_table.id].tags == pred_table_tags
    assert await service.list_job_tags(
        JobType.GENERATE_PREDICTION_TABLE_JOB,
        {TAG_KEY_PQUERY_NAME: "missing"},
    ) == {}
    assert await service.get_training_jobs([
        TrainingJobID.make_random_id(),
        training_job.id,
    ]) == [training_job]
    assert await service.get_job_if_exists_generic(training_job.id
                                                   ) == training_job

    assert await service.get_batch_prediction_jobs([
        BatchPredictionJobID.make_random_id(),
        running_batch_prediction.id,
    ], project_ids=None) == [running_batch_prediction]
    assert await service.get_job_if_exists_generic(running_batch_prediction.id
                                                   ) == running_batch_prediction
    assert await service.get_job_if_exists_generic(train_table.id
                                                   ) == train_table
    assert await service.get_job_if_exists_generic(pred_table.id) == pred_table
    assert await service.is_child_job(training_job) is False
    assert await service.is_child_job(running_batch_prediction) is False
    assert await service.is_child_job(train_table) is False
    assert await service.is_child_job(pred_table) is False
    assert await service.is_child_job(child_train_table) is True
    assert await service.is_child_job(child_pred_table) is True
    assert await service.get_job_if_exists_generic(
        TrainingJobID.make_random_id()) is None
    assert await service.get_running_batch_prediction_job_ids(
        project_ids=None) == [running_batch_prediction.id]
    assert {job.id
            for job in await service.list_open_parent_jobs()} == {
                training_job.id,
                running_batch_prediction.id,
                train_table.id,
                pred_table.id,
            }


@pytest.mark.asyncio
async def test_ml_job_service_job_tags_mutations_write_normalized_rows(
        state_db) -> None:
    service = MLJobService(state_db)
    training_id = TrainingJobID.make_random_id()

    assert await service.apply_job_tags(
        JobType.TRAINING_JOB,
        training_id,
        {
            JobTagKey("owner"): JobTagValue("sdk"),
            JobTagKey("phase"): JobTagValue("one"),
        },
    )
    payloads = await _job_tag_payloads(service)
    assert set(payloads) == {
        job_tag_row_id(JobType.TRAINING_JOB, training_id, "owner"),
        job_tag_row_id(JobType.TRAINING_JOB, training_id, "phase"),
    }
    assert job_tags_row_id(JobType.TRAINING_JOB, training_id) not in payloads
    assert payloads[job_tag_row_id(JobType.TRAINING_JOB, training_id,
                                   "phase")]["value"] == "one"
    read_tags = await service.get_job_tags(JobType.TRAINING_JOB, training_id)
    assert read_tags is not None
    assert read_tags.tags == {
        JobTagKey("owner"): JobTagValue("sdk"),
        JobTagKey("phase"): JobTagValue("one"),
    }
    assert not await service.apply_job_tags(
        JobType.TRAINING_JOB,
        training_id,
        {
            JobTagKey("owner"): JobTagValue("sdk"),
            JobTagKey("phase"): JobTagValue("one"),
        },
    )
    assert await service.get_job_tags(JobType.TRAINING_JOB.value,
                                      training_id) == read_tags
    assert await service.apply_job_tags(
        JobType.TRAINING_JOB.value,
        training_id,
        {JobTagKey("temporary"): JobTagValue("sdk")},
    )
    assert await service.delete_job_tag(JobType.TRAINING_JOB.value, training_id,
                                        JobTagKey("temporary"))
    assert not await service.delete_job_tag(JobType.TRAINING_JOB, training_id,
                                            TAG_KEY_PQUERY_NAME)

    assert await service.apply_job_tags(
        JobType.TRAINING_JOB,
        training_id,
        {
            JobTagKey("owner"): JobTagValue("null"),
            JobTagKey("phase"): JobTagValue("two"),
        },
    )
    payloads = await _job_tag_payloads(service)
    assert set(payloads) == {
        job_tag_row_id(JobType.TRAINING_JOB, training_id, "phase"),
    }
    assert payloads[job_tag_row_id(JobType.TRAINING_JOB, training_id,
                                   "phase")]["value"] == "two"

    assert await service.delete_job_tag(
        JobType.TRAINING_JOB,
        training_id,
        JobTagKey("phase"),
    )
    read_tags = await service.get_job_tags(JobType.TRAINING_JOB, training_id)
    assert read_tags is not None
    assert read_tags.tags == {}
    assert not await service.delete_job_tag(JobType.TRAINING_JOB, training_id,
                                            JobTagKey("phase"))
    payloads = await _job_tag_payloads(service)
    assert len(payloads) == 1
    assert next(iter(payloads.values()))["key"] == ""

    assert await service.apply_job_tags(
        JobType.TRAINING_JOB,
        training_id,
        {JobTagKey("phase"): JobTagValue("three")},
    )
    assert await service.delete_job_tags(JobType.TRAINING_JOB, training_id)
    assert await service.get_job_tags(JobType.TRAINING_JOB, training_id) is None
    assert await _job_tag_payloads(service) == {}


@pytest.mark.asyncio
async def test_ml_job_service_list_job_tags_uses_sql_filters(
        state_db, monkeypatch) -> None:
    service = MLJobService(state_db)
    matching_training_id = TrainingJobID.make_random_id()
    wrong_key_training_id = TrainingJobID.make_random_id()
    wrong_type_batch_id = BatchPredictionJobID.make_random_id()

    assert await service.apply_job_tags(
        JobType.TRAINING_JOB,
        matching_training_id,
        {JobTagKey("phase"): JobTagValue("one")},
    )
    assert await service.apply_job_tags(
        JobType.TRAINING_JOB,
        wrong_key_training_id,
        {JobTagKey("owner"): JobTagValue("one")},
    )
    assert await service.apply_job_tags(
        JobType.BATCH_PREDICTION_JOB,
        wrong_type_batch_id,
        {JobTagKey("phase"): JobTagValue("one")},
    )

    async def fail_list_all(*args, **kwargs):
        raise AssertionError("job tag list filters must not call list_all")

    def fail_match(*args, **kwargs):
        raise AssertionError("job tag list filters must be SQL-backed")

    monkeypatch.setattr(JsonRepository, "list_all", fail_list_all)
    monkeypatch.setattr(JobTags, "match", fail_match)

    phase_matches = await service.list_job_tags(
        JobType.TRAINING_JOB,
        {"phase": "one"},
    )
    assert set(phase_matches) == {matching_training_id}
    assert phase_matches[matching_training_id].tags == {
        JobTagKey("phase"): JobTagValue("one")
    }

    missing_phase_matches = await service.list_job_tags(
        JobType.TRAINING_JOB,
        {"phase": "null"},
    )
    assert set(missing_phase_matches) == {wrong_key_training_id}
    assert missing_phase_matches[wrong_key_training_id].tags == {
        JobTagKey("owner"): JobTagValue("one")
    }


@pytest.mark.asyncio
async def test_ml_job_service_list_job_tags_reads_normalized_rows(
        state_db) -> None:
    service = MLJobService(state_db)
    matching_training_id = TrainingJobID.make_random_id()
    owner_only_training_id = TrainingJobID.make_random_id()
    wrong_type_batch_id = BatchPredictionJobID.make_random_id()

    async with service._state_db.write_tx() as tx:
        await _insert_normalized_job_tag_row(
            tx,
            row_id="training-phase",
            job_type=JobType.TRAINING_JOB,
            job_id=matching_training_id,
            key="phase",
            value="one",
        )
        await _insert_normalized_job_tag_row(
            tx,
            row_id="training-owner",
            job_type=JobType.TRAINING_JOB,
            job_id=matching_training_id,
            key="owner",
            value="sdk",
        )
        await _insert_normalized_job_tag_row(
            tx,
            row_id="training-owner-only",
            job_type=JobType.TRAINING_JOB,
            job_id=owner_only_training_id,
            key="owner",
            value="sdk",
        )
        await _insert_normalized_job_tag_row(
            tx,
            row_id="batch-phase",
            job_type=JobType.BATCH_PREDICTION_JOB,
            job_id=wrong_type_batch_id,
            key="phase",
            value="one",
        )

    phase_matches = await service.list_job_tags(
        JobType.TRAINING_JOB,
        {"phase": "one"},
    )
    assert set(phase_matches) == {matching_training_id}
    assert phase_matches[matching_training_id].tags == {
        JobTagKey("phase"): JobTagValue("one"),
        JobTagKey("owner"): JobTagValue("sdk"),
    }
    read_tags = await service.get_job_tags(JobType.TRAINING_JOB.value,
                                           matching_training_id)
    assert read_tags is not None
    expected_tags = {
        JobTagKey("phase"): JobTagValue("one"),
        JobTagKey("owner"): JobTagValue("sdk"),
    }
    assert read_tags.tags == expected_tags
    read_tags.tags[JobTagKey("phase")] = JobTagValue("mutated")
    reread_tags = await service.get_job_tags(JobType.TRAINING_JOB,
                                             matching_training_id)
    assert reread_tags is not None
    assert reread_tags.tags == expected_tags
    assert await service.list_job_tags(JobType.TRAINING_JOB.value,
                                       {"phase": "one"}) == phase_matches

    missing_phase_matches = await service.list_job_tags(
        JobType.TRAINING_JOB,
        {"phase": "null"},
    )
    assert set(missing_phase_matches) == {owner_only_training_id}
    assert missing_phase_matches[owner_only_training_id].tags == {
        JobTagKey("owner"): JobTagValue("sdk"),
    }
