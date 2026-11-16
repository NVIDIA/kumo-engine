# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest
from kumo_core.types import TaskType
from kumo_core.types.schema import FPID
from kumoapi.source_table import FileType
from kumoapi.typing import Dtype, Stype

from rest.config.batch_prediction_job import BatchPredictionJobConfig
from rest.config.connector import FileConnectorResourceConfig
from rest.config.connector_db import ConnectorDBEntry
from rest.config.graph_v2 import GraphDBEntryV2
from rest.config.table_v2 import (
    Column,
    DataSourceType,
    S3SourceTable,
    TableDBEntryV2,
    TableDefinition,
)
from statedb.config.common import (
    BatchPredictionJobID,
    BatchPredictionWorkflowID,
    ConnectorID,
    GeneratePredictionTableJobID,
    GraphIDv2,
    JobSource,
    JobStatus,
    PredictiveQueryID,
    ProjectID,
    TableName,
    TrainingJobID,
)
from statedb.config.timestamp import ResourceTimestamps
from statedb.db import StateDB
from statedb.db.ml_jobs import MLJobService
from statedb.db.schema import STATE_DB_TABLES
from statedb.rest.connector_db import InternalConnectorDBEntry

_PROJECT_ID = ProjectID("project_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")


class _CountingStateDB:
    def __init__(self, state_db: StateDB) -> None:
        self._state_db = state_db
        self.fetch_count = 0

    def __getattr__(self, name):
        return getattr(self._state_db, name)

    @asynccontextmanager
    async def read_tx(self):
        async with self._state_db.read_tx() as tx:
            owner = self

            class _CountingTx:
                backend = tx.backend

                def __getattr__(self, name):
                    return getattr(tx, name)

                async def fetch(self, query, *args):
                    owner.fetch_count += 1
                    return await tx.fetch(query, *args)

            yield _CountingTx()


def _timestamps() -> ResourceTimestamps:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return ResourceTimestamps(created_at=now, updated_at=now)


def _batch_prediction_config(
        job_id: BatchPredictionJobID,
        *,
        pquery_id: PredictiveQueryID = PredictiveQueryID("risk"),
        training_job_id: TrainingJobID | None = None,
        pquery_def_fpid: FPID = FPID("b" * 32),
) -> BatchPredictionJobConfig:
    return BatchPredictionJobConfig(
        predictive_query_id=pquery_id,
        training_job_id=training_job_id or TrainingJobID.make_random_id(),
        task_type=TaskType.REGRESSION,
        generate_pred_table_job_id=GeneratePredictionTableJobID.as_child_of(
            job_id),
        graph_id=GraphIDv2.public(FPID("a" * 32)),
        pquery_def_fpid=pquery_def_fpid,
    )


def test_state_db_schema_contains_rest_metadata_tables() -> None:
    table_by_name = {table.name: table for table in STATE_DB_TABLES}

    assert {
        "connectors",
        "identities",
        "identity_groups",
        "groups",
        "tables",
        "graphs",
        "pquery_definitions",
        "pquery_aliases",
        "api_keys",
        "api_key_secrets",
        "projects",
        "batch_prediction_jobs",
        "job_tags",
    } <= set(table_by_name)

    assert any(index.name == "tables_project_alias_unique_idx" and index.unique
               for index in table_by_name["tables"].indexes)
    assert any(index.name == "graphs_project_alias_unique_idx" and index.unique
               for index in table_by_name["graphs"].indexes)
    assert any(index.name == "batch_prediction_jobs_workflow_id_idx"
               for index in table_by_name["batch_prediction_jobs"].indexes)


@pytest.mark.asyncio
async def test_state_db_runtime_reports_rest_metadata_schema_ready(
        tmp_path) -> None:
    state_db = await StateDB.open_sqlite(str(tmp_path / "state.db"))

    status = await state_db.get_schema_status()

    assert status.ok
    assert not status.missing_table_names
    assert status.existing_state_table_count == status.expected_table_count


@pytest.mark.asyncio
async def test_state_db_repositories_round_trip_rest_metadata_rows(
        tmp_path) -> None:
    from statedb.db.repositories import JsonRepository

    state_db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    connector = InternalConnectorDBEntry(
        id=ConnectorID("connector_00000000000000000000000000000001"),
        project_id=_PROJECT_ID,
        name="uploads",
        normalized_name="uploads",
        config=FileConnectorResourceConfig(name="uploads",
                                           root_dir="/tmp/csv_1"),
        timestamps=_timestamps(),
    )
    table_definition = TableDefinition(
        cols=[Column(name="a", stype=Stype.ID, dtype=Dtype.int)],
        source_table=S3SourceTable(
            connector_id=str(connector.id),
            source={
                'mode': 'relative',
                'path': "table.parquet"
            },
            file_type=FileType.PARQUET,
            data_source_type=DataSourceType.S3,
        ),
    )
    table = TableDBEntryV2(
        id=table_definition.fpid,
        project_id=_PROJECT_ID,
        config=table_definition,
        col_dtypes={},
        alias="A",
        timestamps=_timestamps(),
    )
    graph = GraphDBEntryV2(
        id=FPID("a" * 32),
        project_id=_PROJECT_ID,
        table_name_to_dbid={TableName("ACCOUNT"): table.id},
        col_groups=[],
        alias="G",
        timestamps=_timestamps(),
    )

    async with state_db.write_tx() as tx:
        await JsonRepository("connectors",
                             ConnectorDBEntry).upsert(tx, str(connector.id),
                                                      connector)
        await JsonRepository("tables",
                             TableDBEntryV2).upsert(tx, str(table.id), table)
        await JsonRepository("graphs",
                             GraphDBEntryV2).upsert(tx, str(graph.id), graph)

    async with state_db.read_tx() as tx:
        connector_row = await JsonRepository("connectors",
                                             ConnectorDBEntry).get(
                                                 tx, str(connector.id))
        table_row = await JsonRepository("tables",
                                         TableDBEntryV2).get(tx, str(table.id))
        graph_row = await JsonRepository("graphs",
                                         GraphDBEntryV2).get(tx, str(graph.id))

    assert connector_row is not None
    assert connector_row.payload.id == connector.id
    assert connector_row.payload.project_id == connector.project_id
    assert connector_row.payload.config == connector.config
    assert (connector_row.payload.timestamps.created_at ==
            connector.timestamps.created_at)
    assert table_row is not None
    assert table_row.payload.alias == "A"
    assert graph_row is not None
    assert graph_row.payload.table_name_to_dbid == {
        TableName("ACCOUNT"): table.id,
    }


@pytest.mark.asyncio
async def test_batch_prediction_row_persists_execution_and_launch_metadata(
        tmp_path) -> None:
    state_db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    service = MLJobService(state_db)
    job_id = BatchPredictionJobID.make_random_id()
    config = _batch_prediction_config(job_id)

    await service.create_batch_prediction_job(
        id=job_id,
        config=config,
        now=_timestamps().created_at,
        source=JobSource.SDK,
        project_id=_PROJECT_ID,
        user="user@example.com",
        description="customer scoring",
        batch_prediction_workflow_id=BatchPredictionWorkflowID("workflow_1"),
    )

    rows = await service.get_batch_prediction_rows([job_id], project_ids=None)

    assert len(rows) == 1
    assert rows[0].config == config
    assert rows[0].execution is not None
    assert rows[0].execution.status == JobStatus.QUEUED
    assert rows[0].description == "customer scoring"
    assert rows[0].batch_prediction_workflow_id == "workflow_1"


@pytest.mark.asyncio
async def test_create_batch_prediction_job_is_immediately_gettable(
        tmp_path) -> None:
    """A just-submitted batch-prediction job must be fully GET-able from the
    single create write -- config, execution, and the REST launch metadata --
    without any follow-up metadata write. This is the invariant that prevents
    the read-after-write 404 race on submit."""
    state_db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    service = MLJobService(state_db)
    job_id = BatchPredictionJobID.make_random_id()
    config = _batch_prediction_config(job_id)

    # Single write, mirroring the submit path's `start_batch_prediction`.
    await service.create_batch_prediction_job(
        id=job_id,
        config=config,
        now=_timestamps().created_at,
        source=JobSource.SDK,
        project_id=_PROJECT_ID,
        user="user@example.com",
        batch_prediction_workflow_id=BatchPredictionWorkflowID("workflow_1"),
    )

    # The GET path requires BOTH config (REST gate) and execution (ML gate).
    rows = await service.get_batch_prediction_rows([job_id], project_ids=None)
    assert len(rows) == 1
    row = rows[0]
    assert row.config == config
    assert row.execution is not None
    assert row.execution.status == JobStatus.QUEUED
    assert row.batch_prediction_workflow_id == "workflow_1"

    # The ML-gated read (used by the driver GET) sees the execution row too.
    ml_entries = await service.get_batch_prediction_jobs([job_id],
                                                         project_ids=None)
    assert len(ml_entries) == 1
    assert ml_entries[0].config == config


@pytest.mark.asyncio
async def test_batch_prediction_state_queries_filter_sort_page_and_count(
        tmp_path) -> None:
    state_db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    service = MLJobService(state_db)
    pquery_id = PredictiveQueryID("risk")
    other_pquery_id = PredictiveQueryID("other")
    training_job_id = TrainingJobID.make_random_id()
    other_training_job_id = TrainingJobID.make_random_id()

    async def create_row(
        job_id: BatchPredictionJobID,
        *,
        created_at: datetime,
        pquery_id: PredictiveQueryID,
        training_job_id: TrainingJobID,
        workflow_id: str,
    ) -> None:
        config = _batch_prediction_config(
            job_id,
            pquery_id=pquery_id,
            training_job_id=training_job_id,
        )
        await service.create_batch_prediction_job(
            id=job_id,
            config=config,
            now=created_at,
            project_id=_PROJECT_ID,
            description="customer scoring",
            batch_prediction_workflow_id=BatchPredictionWorkflowID(workflow_id),
        )

    old_id = BatchPredictionJobID("bp-old")
    new_id = BatchPredictionJobID("bp-new")
    other_id = BatchPredictionJobID("bp-other")
    await create_row(
        old_id,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        pquery_id=pquery_id,
        training_job_id=training_job_id,
        workflow_id="workflow_1",
    )
    await create_row(
        new_id,
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        pquery_id=pquery_id,
        training_job_id=training_job_id,
        workflow_id="workflow_1",
    )
    await create_row(
        other_id,
        created_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
        pquery_id=other_pquery_id,
        training_job_id=other_training_job_id,
        workflow_id="workflow_2",
    )

    newest = await service.list_batch_prediction_rows(
        project_ids=None,
        model_id=training_job_id,
        pquery_id=pquery_id,
        status=JobStatus.QUEUED,
        sort_created_at="desc",
        limit=1,
    )
    page_two = await service.list_batch_prediction_rows(
        project_ids=None,
        model_id=training_job_id,
        pquery_id=pquery_id,
        sort_created_at="desc",
        limit=1,
        offset=1,
    )
    counting_state_db = _CountingStateDB(state_db)
    service._state_db = counting_state_db
    current = await service.get_current_batch_prediction_job_ids(
        [pquery_id, other_pquery_id,
         PredictiveQueryID("missing")],
        project_ids=None,
    )
    assert counting_state_db.fetch_count == 1
    prune_candidates = await service.list_batch_prediction_prune_candidate_ids(
        "workflow_1",
        project_ids=None,
        keep_most_recent=1,
    )

    assert [row.id for row in newest] == [new_id]
    assert [row.id for row in page_two] == [old_id]
    assert await service.count_batch_prediction_rows(
        project_ids=None,
        model_id=training_job_id,
        pquery_id=pquery_id,
        status=JobStatus.QUEUED,
    ) == 2
    assert await service.has_batch_prediction_rows(
        project_ids=None,
        model_id=training_job_id,
        pquery_id=pquery_id,
        status=JobStatus.QUEUED,
    )
    public_pquery_id = PredictiveQueryID.compose(fpid=FPID("b" * 32))
    assert await service.has_batch_prediction_rows(project_ids=None,
                                                   pquery_id=public_pquery_id)
    assert not await service.has_batch_prediction_rows(
        project_ids=None,
        pquery_id=public_pquery_id,
        match_pquery_def_fpid=False,
    )
    assert current == {
        pquery_id: new_id,
        other_pquery_id: other_id,
    }
    assert prune_candidates == [old_id]
