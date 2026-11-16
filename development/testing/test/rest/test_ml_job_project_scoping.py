# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for project-level ML job scoping through MLJobService."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from fastapi import Request, Response
from kumo_core.model_plan import ModelPlan, TrainingTableGenerationPlan
from kumo_core.types import TaskType
from kumoapi.data_source import DEFAULT_PROJECT_ID
from kumoapi.model_plan import PredictionTableGenerationPlan

from rest.backends.ml_backend.job_models import (
    GeneratePredTableJobConfig,
    GenerateTrainTableJobConfig,
    GenericJob,
    JobFilterMetadata,
    ListJobsProcessor,
    ListJobsQuery,
)
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.config.batch_prediction_job import (
    BatchPredictionJobConfig,
    BatchPredictionJobOutput,
)
from rest.public.routers.training import TrainingRouterV1
from statedb.config.common import (
    BatchPredictionJobID,
    BatchPredictionWorkflowID,
    GeneratePredictionTableJobID,
    GraphIDv1,
    JobEventLogEntry,
    JobExecution,
    JobSource,
    JobStatus,
    JobType,
    PredictiveQueryID,
    ProjectID,
    TrainingJobID,
)
from statedb.config.job_tag import (
    TAG_KEY_PQUERY_NAME,
    JobTagKey,
    JobTags,
    JobTagValue,
)
from statedb.config.timestamp import get_current_timestamps
from statedb.db import MLJobService, StateDB

TEST_GRAPH_ID = GraphIDv1("graph_id_test")
TEST_PQUERY_ID = PredictiveQueryID("pquery_id_test")

PROJECT_A = ProjectID.generate()
PROJECT_B = ProjectID.generate()


@pytest.fixture(scope="session", autouse=True)
def setup_rest_test_session() -> None:
    pass


@pytest.fixture(autouse=True)
def auto_cleanup_per_func() -> None:
    pass


@pytest_asyncio.fixture
async def ml_jobs(tmp_path) -> MLJobService:
    state_db = await StateDB.open_sqlite(str(tmp_path / "state.sqlite"))
    return MLJobService(state_db)


def _now() -> datetime:
    return datetime.utcnow()


async def _create_training_job(
    service: MLJobService,
    project_id: ProjectID = PROJECT_A,
    now: datetime | None = None,
) -> tuple[TrainingJobID, object]:
    return await service.create_training_job(
        pquery_id=TEST_PQUERY_ID,
        pquery_def_fpid=None,
        graph_id=TEST_GRAPH_ID,
        now=now or _now(),
        advanced_options=ModelPlan(),
        graph_data_version="1.0",
        project_id=project_id,
    )


async def _create_batch_prediction_job(
    service: MLJobService,
    training_job_id: TrainingJobID,
    project_id: ProjectID = PROJECT_A,
    workflow_id: BatchPredictionWorkflowID | None = None,
) -> BatchPredictionJobID:
    batch_id = BatchPredictionJobID.make_random_id()
    await service.create_batch_prediction_job(
        id=batch_id,
        now=_now(),
        config=BatchPredictionJobConfig(
            training_job_id=training_job_id,
            graph_id=TEST_GRAPH_ID,
            predictive_query_id=TEST_PQUERY_ID,
            generate_pred_table_job_id=(
                GeneratePredictionTableJobID.as_child_of(batch_id)),
            task_type=TaskType.BINARY_CLASSIFICATION,
        ),
        source=JobSource.SDK,
        project_id=project_id,
        batch_prediction_workflow_id=workflow_id,
    )
    return batch_id


async def _create_gen_train_table_job(
    service: MLJobService,
    project_id: ProjectID = PROJECT_A,
    now: datetime | None = None,
):
    return await service.create_generate_train_table_job(
        config=GenerateTrainTableJobConfig(
            pquery_id=TEST_PQUERY_ID,
            plan=TrainingTableGenerationPlan(),
            graph_id=TEST_GRAPH_ID,
            graph_data_version=None,
        ),
        now=now or _now(),
        source=JobSource.SDK,
        project_id=project_id,
    )


async def _create_gen_pred_table_job(
    service: MLJobService,
    project_id: ProjectID = PROJECT_A,
):
    return await service.create_generate_pred_table_job(
        id=GeneratePredictionTableJobID.make_random_id(),
        config=GeneratePredTableJobConfig(
            pquery_id=TEST_PQUERY_ID,
            plan=PredictionTableGenerationPlan(),
            graph_id=TEST_GRAPH_ID,
            graph_data_version=None,
        ),
        now=_now(),
        source=JobSource.SDK,
        project_id=project_id,
    )


def _visible_ids(entries) -> set:
    return {entry.id for entry in entries}


def _request_ctx(*project_ids: ProjectID) -> SimpleNamespace:
    return SimpleNamespace(user_authorization=SimpleNamespace(
        is_admin=False,
        project_ids=tuple(str(project_id) for project_id in project_ids),
        external_membership_unavailable=False,
    ))


def _backend_for_service(service: MLJobService) -> MLJobBackend:
    backend = object.__new__(MLJobBackend)
    backend._ml_job_service = service
    backend._settings = SimpleNamespace(ml_job_settings=SimpleNamespace(
        use_fake_job_backend=False))
    backend._now_fn = _now
    return backend


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'listing',
    ['visible-projects', 'exact-project', 'both'],
)
async def test_aggregate_listing_applies_project_scope(ml_jobs: MLJobService,
                                                       listing: str) -> None:
    own_id, _ = await _create_training_job(ml_jobs, project_id=PROJECT_A)
    other_id, _ = await _create_training_job(ml_jobs, project_id=PROJECT_B)

    if listing == 'visible-projects':
        results = await ml_jobs.list_jobs_by_project_visibility([PROJECT_A])
    elif listing == 'exact-project':
        results = await ml_jobs.list_aggregate_job_entries(project_id=PROJECT_A)
    else:
        visible = await ml_jobs.list_jobs_by_project_visibility([PROJECT_A])
        exact = await ml_jobs.list_aggregate_job_entries(project_id=PROJECT_A)
        assert _visible_ids(visible) == _visible_ids(exact)
        results = visible

    assert own_id in _visible_ids(results)
    assert other_id not in _visible_ids(results)


@pytest.mark.asyncio
async def test_project_visibility_applies_to_all_aggregate_job_types(
        ml_jobs: MLJobService) -> None:
    own_training_id, own_gen_train_id = await _create_training_job(
        ml_jobs,
        project_id=PROJECT_A,
    )
    other_training_id, other_gen_train_id = await _create_training_job(
        ml_jobs,
        project_id=PROJECT_B,
    )
    own_batch_id = await _create_batch_prediction_job(
        ml_jobs,
        own_training_id,
        project_id=PROJECT_A,
    )
    other_batch_id = await _create_batch_prediction_job(
        ml_jobs,
        other_training_id,
        project_id=PROJECT_B,
    )
    own_gen_pred_id = await _create_gen_pred_table_job(
        ml_jobs,
        project_id=PROJECT_A,
    )
    other_gen_pred_id = await _create_gen_pred_table_job(
        ml_jobs,
        project_id=PROJECT_B,
    )

    visible = _visible_ids(await
                           ml_jobs.list_jobs_by_project_visibility([PROJECT_A]))

    assert {own_training_id, own_gen_train_id, own_batch_id,
            own_gen_pred_id} <= visible
    assert {
        other_training_id, other_gen_train_id, other_batch_id, other_gen_pred_id
    }.isdisjoint(visible)


@pytest.mark.asyncio
async def test_batch_prediction_workflow_lookup_filters_by_workflow_id(
        ml_jobs: MLJobService) -> None:
    training_id, _ = await _create_training_job(ml_jobs)
    workflow_id = BatchPredictionWorkflowID("workflow_a")
    matching_id = await _create_batch_prediction_job(ml_jobs, training_id,
                                                     workflow_id=workflow_id)
    null_workflow_id = await _create_batch_prediction_job(ml_jobs, training_id)

    assert [
        row.id
        for row in await ml_jobs.list_batch_prediction_rows_by_workflow_id(
            workflow_id, project_ids={PROJECT_A})
    ] == [matching_id]
    assert [
        row.id
        for row in await ml_jobs.list_batch_prediction_rows_by_workflow_id(
            None, project_ids={PROJECT_A})
    ] == [null_workflow_id]


@pytest.mark.asyncio
async def test_batch_prediction_queries_apply_explicit_project_scope_before_sql_operations(
        ml_jobs: MLJobService) -> None:
    """Every BP query treats None/global, visible, exact, and empty distinctly."""
    training_a, _ = await _create_training_job(ml_jobs, project_id=PROJECT_A)
    training_b, _ = await _create_training_job(ml_jobs, project_id=PROJECT_B)
    workflow_id = BatchPredictionWorkflowID("shared_workflow")
    job_a = await _create_batch_prediction_job(ml_jobs, training_a,
                                               project_id=PROJECT_A,
                                               workflow_id=workflow_id)
    job_b = await _create_batch_prediction_job(ml_jobs, training_b,
                                               project_id=PROJECT_B,
                                               workflow_id=workflow_id)

    assert _visible_ids(await ml_jobs.get_batch_prediction_rows(
        [job_a, job_b], project_ids={PROJECT_A})) == {job_a}
    assert _visible_ids(await ml_jobs.get_batch_prediction_rows(
        [job_a, job_b], project_ids=set())) == set()
    assert _visible_ids(await ml_jobs.get_batch_prediction_rows(
        [job_a, job_b], project_ids=None)) == {job_a, job_b}

    # The scope is part of SQL, so a newer inaccessible row cannot consume
    # the limit before the visible row is selected.
    assert [
        row.id for row in await ml_jobs.list_batch_prediction_rows(
            project_ids={PROJECT_A}, sort_created_at="desc", limit=1)
    ] == [job_a]
    assert await ml_jobs.count_batch_prediction_rows(project_ids={PROJECT_A}
                                                     ) == 1
    assert await ml_jobs.count_batch_prediction_rows(project_ids=None) == 2
    assert await ml_jobs.count_batch_prediction_rows(project_ids=set()) == 0
    assert await ml_jobs.has_batch_prediction_rows(project_ids={PROJECT_A},
                                                   job_ids={job_a})
    assert not await ml_jobs.has_batch_prediction_rows(project_ids={PROJECT_A},
                                                       job_ids={job_b})

    assert await ml_jobs.get_current_batch_prediction_job_ids(
        [TEST_PQUERY_ID], project_ids={PROJECT_A}) == {
            TEST_PQUERY_ID: job_a
        }
    assert await ml_jobs.get_current_batch_prediction_job_ids(
        [TEST_PQUERY_ID], project_ids={PROJECT_B}) == {
            TEST_PQUERY_ID: job_b
        }
    assert await ml_jobs.get_current_batch_prediction_job_ids(
        [TEST_PQUERY_ID], project_ids=set()) == {}
    assert [
        row.id
        for row in await ml_jobs.list_batch_prediction_rows_by_workflow_id(
            workflow_id, project_ids={PROJECT_A})
    ] == [job_a]
    assert await ml_jobs.list_batch_prediction_prune_candidate_ids(
        workflow_id, project_ids={PROJECT_A}, keep_most_recent=0) == [job_a]
    assert await ml_jobs.get_running_batch_prediction_job_ids(
        project_ids={PROJECT_A}) == [job_a]


@pytest.mark.asyncio
async def test_generate_prediction_table_bulk_lookup_scopes_before_limit_input(
        ml_jobs: MLJobService) -> None:
    job_a = await _create_gen_pred_table_job(ml_jobs, project_id=PROJECT_A)
    job_b = await _create_gen_pred_table_job(ml_jobs, project_id=PROJECT_B)

    assert _visible_ids(await ml_jobs.get_generate_pred_table_jobs(
        [job_a, job_b], project_ids={PROJECT_A})) == {job_a}
    assert await ml_jobs.get_generate_pred_table_jobs([job_a, job_b],
                                                      project_ids=set()) == []
    assert _visible_ids(await ml_jobs.get_generate_pred_table_jobs(
        [job_a, job_b], project_ids=None)) == {job_a, job_b}


@pytest.mark.asyncio
async def test_training_candidate_ordering_is_scoped_before_limit(
        ml_jobs: MLJobService) -> None:
    older = _now()
    newer = older + timedelta(seconds=1)
    job_a, _ = await _create_training_job(ml_jobs, project_id=PROJECT_A,
                                          now=older)
    job_b, _ = await _create_training_job(ml_jobs, project_id=PROJECT_B,
                                          now=newer)

    assert [
        row.id for row in await ml_jobs.list_training_job_rows(
            job_ids={job_a, job_b},
            project_ids={PROJECT_A},
            sort_created_at="desc",
            limit=1,
        )
    ] == [job_a]
    assert [
        row.id for row in await ml_jobs.list_training_job_rows(
            job_ids={job_a, job_b},
            project_ids=None,
            sort_created_at="desc",
            limit=1,
        )
    ] == [job_b]
    assert await ml_jobs.list_training_job_rows(job_ids={job_a, job_b},
                                                project_ids=set()) == []


@pytest.mark.asyncio
async def test_training_backend_single_and_bulk_share_project_scope(
        ml_jobs: MLJobService) -> None:
    job_a, _ = await _create_training_job(ml_jobs, project_id=PROJECT_A)
    job_b, _ = await _create_training_job(ml_jobs, project_id=PROJECT_B)
    missing = TrainingJobID.make_random_id()
    backend = _backend_for_service(ml_jobs)
    ctx = _request_ctx(PROJECT_A)

    assert (await backend.get_training_job(ctx, job_a)).id == job_a
    assert await backend.get_training_job(ctx, job_b) is None
    assert await backend.get_training_job(ctx, missing) is None
    assert set(await
               backend.get_training_jobs(ctx,
                                         {job_a, job_b, missing})) == {job_a}
    assert not await backend.delete_training_job(ctx, job_b, force=True)
    assert await ml_jobs.get_training_job(job_b) is not None


@pytest.mark.asyncio
async def test_training_backend_replaces_request_context_for_each_bulk_read(
        ml_jobs: MLJobService) -> None:
    job_a, _ = await _create_training_job(ml_jobs, project_id=PROJECT_A)
    job_b, _ = await _create_training_job(ml_jobs, project_id=PROJECT_B)
    backend = _backend_for_service(ml_jobs)

    for ctx, visible, hidden in (
        (_request_ctx(PROJECT_A), job_a, job_b),
        (_request_ctx(PROJECT_B), job_b, job_a),
        (_request_ctx(PROJECT_A), job_a, job_b),
    ):
        jobs = await backend._get_training_jobs(
            ctx,
            {job_a, job_b},
            strip_expensive_fields=False,
        )
        assert jobs[visible] is not None
        assert jobs[hidden] is None


@pytest.mark.asyncio
async def test_public_training_list_scopes_before_conversion_and_limit(
        ml_jobs: MLJobService) -> None:
    older = _now()
    newer = older + timedelta(seconds=1)
    job_a, _ = await _create_training_job(ml_jobs, project_id=PROJECT_A,
                                          now=older)
    job_b, _ = await _create_training_job(ml_jobs, project_id=PROJECT_B,
                                          now=newer)

    # Simulate a malformed inaccessible payload. SQL project scoping must
    # exclude it before deserialization/public conversion and before limit.
    foreign = await ml_jobs.get_training_job(job_b)
    assert foreign is not None
    foreign.execution = None
    async with ml_jobs._state_db.write_tx() as tx:
        await ml_jobs._training_jobs.upsert(tx, str(job_b), foreign)

    tags = {
        job_id:
        JobTags(
            timestamps=get_current_timestamps(),
            tags={TAG_KEY_PQUERY_NAME: JobTagValue(TEST_PQUERY_ID)},
        )
        for job_id in (job_a, job_b)
    }
    router = object.__new__(TrainingRouterV1)
    router._driver = SimpleNamespace(list_job_tags=AsyncMock(return_value=tags))
    router._ml_job_backend = _backend_for_service(ml_jobs)
    request = Request({
        "type": "http",
        "method": "GET",
        "path": "/v1/training_jobs",
        "query_string": b"",
        "headers": [],
    })

    resources = await router.list(request, Response(), limit=1,
                                  ctx=_request_ctx(PROJECT_A))

    assert [resource.job_id for resource in resources] == [job_a]
    assert resources[0].created_at == older


@pytest.mark.asyncio
async def test_generate_train_table_scope_precedes_conversion_and_limit(
        ml_jobs: MLJobService) -> None:
    older = _now()
    newer = older + timedelta(seconds=1)
    job_a = await _create_gen_train_table_job(ml_jobs, project_id=PROJECT_A,
                                              now=older)
    job_b = await _create_gen_train_table_job(ml_jobs, project_id=PROJECT_B,
                                              now=newer)

    assert _visible_ids(await ml_jobs.get_generate_train_table_jobs(
        {job_a, job_b}, project_ids={PROJECT_A})) == {job_a}
    assert await ml_jobs.get_generate_train_table_jobs({job_a, job_b},
                                                       project_ids=set()) == []
    assert _visible_ids(await ml_jobs.get_generate_train_table_jobs(
        {job_a, job_b}, project_ids=None)) == {job_a, job_b}

    # The backend must scope StateDB rows before public-resource conversion
    # and before applying its limit. A newer foreign row cannot shadow A.
    backend = object.__new__(MLJobBackend)
    backend._ml_job_service = ml_jobs
    resources = await backend.get_generate_train_table_jobs(
        {job_a, job_b}, project_ids={PROJECT_A}, limit=1)
    assert [resource.job_id for resource in resources] == [job_a]


@pytest.mark.asyncio
async def test_running_batch_prediction_ids_exclude_finished_jobs(
        ml_jobs: MLJobService) -> None:
    training_id, _ = await _create_training_job(ml_jobs)
    queued_id = await _create_batch_prediction_job(ml_jobs, training_id)
    cancelled_id = await _create_batch_prediction_job(ml_jobs, training_id)
    await ml_jobs.save_batch_prediction_job_output(
        cancelled_id,
        BatchPredictionJobOutput(
            status=JobStatus.CANCELLED,
            log=JobEventLogEntry.cancelled(),
        ),
    )

    assert await ml_jobs.get_running_batch_prediction_job_ids(
        project_ids={PROJECT_A}) == [queued_id]


@pytest.mark.asyncio
async def test_list_job_ids_by_type_filters_job_type(
        ml_jobs: MLJobService) -> None:
    training_id, _ = await _create_training_job(ml_jobs)
    assert await ml_jobs.list_job_ids_by_type(JobType.TRAINING_JOB
                                              ) == [training_id]


@pytest.mark.asyncio
async def test_list_job_tags_filters_by_tag_values(
        ml_jobs: MLJobService) -> None:
    first_id, _ = await _create_training_job(ml_jobs)
    second_id, _ = await _create_training_job(ml_jobs)
    batch_id = await _create_batch_prediction_job(ml_jobs, first_id)
    await ml_jobs.apply_job_tags(
        JobType.TRAINING_JOB,
        first_id,
        {JobTagKey("phase"): JobTagValue("one")},
    )
    await ml_jobs.apply_job_tags(
        JobType.TRAINING_JOB,
        second_id,
        {JobTagKey("phase"): JobTagValue("two")},
    )
    await ml_jobs.apply_job_tags(
        JobType.BATCH_PREDICTION_JOB,
        batch_id,
        {JobTagKey("phase"): JobTagValue("one")},
    )
    listed = await ml_jobs.list_job_tags(JobType.TRAINING_JOB, {"phase": "one"})

    assert set(listed) == {first_id}
    assert listed[first_id].tags[JobTagKey("phase")] == JobTagValue("one")


@pytest.mark.asyncio
async def test_created_jobs_persist_project_id(ml_jobs: MLJobService) -> None:
    training_id, gen_train_id = await _create_training_job(
        ml_jobs,
        project_id=PROJECT_A,
    )
    batch_id = await _create_batch_prediction_job(
        ml_jobs,
        training_id,
        project_id=PROJECT_A,
    )
    gen_pred_id = await _create_gen_pred_table_job(
        ml_jobs,
        project_id=PROJECT_B,
    )

    assert (await ml_jobs.get_training_job(training_id)).project_id == PROJECT_A
    assert (await ml_jobs.get_generate_train_table_job(
        gen_train_id, project_ids=None)).project_id == PROJECT_A
    batch_jobs = await ml_jobs.get_batch_prediction_jobs(
        [batch_id], project_ids={PROJECT_A})
    assert batch_jobs[0].project_id == PROJECT_A
    assert (await ml_jobs.get_generate_pred_table_job(gen_pred_id)
            ).project_id == PROJECT_B


@pytest.mark.asyncio
async def test_default_project_id_persists_as_job_foreign_reference(
        ml_jobs: MLJobService) -> None:
    default_project_id = ProjectID(DEFAULT_PROJECT_ID)

    training_id, generated_id = await _create_training_job(
        ml_jobs, project_id=default_project_id)
    batch_id = await _create_batch_prediction_job(ml_jobs, training_id,
                                                  project_id=default_project_id)

    assert (await ml_jobs.get_training_job(training_id)).project_id == 'default'
    assert (await ml_jobs.get_generate_train_table_job(
        generated_id, project_ids={default_project_id})).project_id == 'default'
    assert (await ml_jobs.get_batch_prediction_jobs([batch_id], project_ids={
        default_project_id
    }))[0].project_id == 'default'


@pytest.mark.asyncio
async def test_job_creation_rejects_cross_project_parent_reference(
        ml_jobs: MLJobService) -> None:
    training_id, _ = await _create_training_job(ml_jobs, project_id=PROJECT_A)

    with pytest.raises(ValueError, match='Training job belongs to project'):
        await _create_batch_prediction_job(
            ml_jobs,
            training_id,
            project_id=PROJECT_B,
        )


@pytest.mark.asyncio
async def test_child_gen_pred_table_inherits_project_id(
        ml_jobs: MLJobService) -> None:
    training_id, _ = await _create_training_job(
        ml_jobs,
        project_id=PROJECT_A,
    )
    batch_id = BatchPredictionJobID.make_random_id()
    child_id = GeneratePredictionTableJobID.as_child_of(batch_id)

    await ml_jobs.create_batch_prediction_job(
        id=batch_id,
        now=_now(),
        config=BatchPredictionJobConfig(
            training_job_id=training_id,
            graph_id=TEST_GRAPH_ID,
            predictive_query_id=TEST_PQUERY_ID,
            generate_pred_table_job_id=child_id,
            task_type=TaskType.BINARY_CLASSIFICATION,
        ),
        source=JobSource.SDK,
        project_id=PROJECT_A,
    )

    child = await ml_jobs.get_generate_pred_table_job(child_id)
    assert child.project_id == PROJECT_A


@pytest.mark.asyncio
async def test_delete_job_generic_removes_only_requested_job(
        ml_jobs: MLJobService) -> None:
    own_id, _ = await _create_training_job(ml_jobs, project_id=PROJECT_A)
    other_id, _ = await _create_training_job(ml_jobs, project_id=PROJECT_B)

    assert await ml_jobs.delete_job_generic(own_id, force=True)

    assert await ml_jobs.get_training_job(own_id) is None
    assert await ml_jobs.get_training_job(other_id) is not None


def test_list_jobs_processor_filters_by_exact_project() -> None:
    now = _now()
    jobs = [
        GenericJob(
            job_id=TrainingJobID.make_random_id(),
            job_type=JobType.TRAINING_JOB,
            execution=JobExecution(
                status=JobStatus.QUEUED,
                log=JobEventLogEntry.not_started(now),
                last_updated_at=now,
            ),
            metadata=JobFilterMetadata(
                start_time=now,
                end_time=None,
                created_at=now,
                job_type=JobType.TRAINING_JOB,
                status=JobStatus.QUEUED,
            ),
            project_id=PROJECT_A,
        ),
        GenericJob(
            job_id=TrainingJobID.make_random_id(),
            job_type=JobType.TRAINING_JOB,
            execution=JobExecution(
                status=JobStatus.QUEUED,
                log=JobEventLogEntry.not_started(now),
                last_updated_at=now,
            ),
            metadata=JobFilterMetadata(
                start_time=now,
                end_time=None,
                created_at=now,
                job_type=JobType.TRAINING_JOB,
                status=JobStatus.QUEUED,
            ),
            project_id=PROJECT_B,
        ),
    ]

    filtered = ListJobsProcessor.apply_filters(
        ListJobsQuery(project_id=PROJECT_A),
        jobs,
    )

    assert [job.project_id for job in filtered] == [PROJECT_A]
