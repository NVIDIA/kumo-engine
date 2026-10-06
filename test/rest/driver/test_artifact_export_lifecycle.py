# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import timedelta
from itertools import chain, repeat
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import kumoapi.data_source as data_source_api
import pytest
from kumoapi.jobs import (
    DatabricksPredictionOutput,
    PredictionArtifactType,
    S3PredictionOutput,
)
from temporalio import workflow
from temporalio.client import WorkflowExecutionStatus
from temporalio.common import WorkflowIDReusePolicy
from temporalio.exceptions import WorkflowAlreadyStartedError

from rest.backends.data_backend.data_backend import DataBackend
from rest.config.artifact_export import ArtifactExportInput
from rest.config.data_backend import ConnectorExecutionRef
from rest.driver.artifact_api_driver import (
    _ARTIFACT_UNSEALED_DISPATCH_WINDOW,
    ArtifactApiDriver,
)
from statedb.config.common import (
    ArtifactExportJobID,
    ConnectorID,
    JobStatus,
    ProjectID,
    UserID,
)
from statedb.config.timestamp import get_current_timestamps, now_utc
from statedb.db import StateDB
from statedb.db.artifact_exports import ArtifactExportJobService
from statedb.utils.errors import ArtifactExportJobNotFoundError

PROJECT_ID = ProjectID('project_' + 'a' * 32)
CONNECTOR_ID = ConnectorID('connector_' + 'b' * 32)
WORKFLOW_ID = ArtifactExportJobID('ae-job-' + 'd' * 32)
REQUEST_ID = '00000000-0000-4000-8000-000000000001'


@workflow.defn(name='ArtifactDuplicateRunningWorkflow')
class ArtifactDuplicateRunningWorkflow:
    @workflow.run
    async def run(self) -> None:
        await workflow.wait_condition(lambda: False)


def _artifact_driver(*, dispatch_sealed=False, created_at=None):
    connectors = SimpleNamespace(
        require_project_access=AsyncMock(),
        get_for_project=AsyncMock(return_value=SimpleNamespace(
            config=data_source_api.S3ConnectorResourceConfig(
                target=data_source_api.S3RootedConnectorTarget(
                    bucket='customer-output')))),
        execution_ref_for_project=AsyncMock(return_value=ConnectorExecutionRef(
            project_id=PROJECT_ID,
            connector_id=CONNECTOR_ID,
        )),
    )
    timestamps = get_current_timestamps()
    if created_at is not None:
        timestamps.created_at = created_at
    entry = SimpleNamespace(
        id=WORKFLOW_ID,
        project_id=PROJECT_ID,
        dispatch_sealed=dispatch_sealed,
        cancel_requested=False,
        timestamps=timestamps,
    )

    first_entry = (entry if dispatch_sealed or created_at is not None else None)
    artifact_exports = SimpleNamespace(
        get=AsyncMock(side_effect=chain((first_entry, ), repeat(entry))),
        create=AsyncMock(
            return_value=(entry, not dispatch_sealed and created_at is None)),
        seal_dispatch=AsyncMock(return_value=entry),
        request_cancel=AsyncMock(return_value=entry),
        get_for_project=AsyncMock(return_value=entry),
    )
    ml_jobs = SimpleNamespace(get_job_if_exists_generic=AsyncMock(
        return_value=SimpleNamespace(project_id=PROJECT_ID)))
    driver = ArtifactApiDriver(MagicMock(), MagicMock())
    driver._configure_state_services(
        SimpleNamespace(
            connectors=connectors,
            artifact_exports=artifact_exports,
            ml_jobs=ml_jobs,
        ))
    output = S3PredictionOutput(
        artifact_type=PredictionArtifactType.PREDICTIONS,
        connector_id=str(CONNECTOR_ID),
        file_path='predictions',
    )
    principal = SimpleNamespace(user_id=UserID('user-test'))
    return driver, artifact_exports, output, principal


@contextmanager
def _request_context(
    principal,
    *,
    dispatch=None,
    status=None,
    cancel=None,
) -> Iterator[None]:
    patches = [
        patch('rest.driver.artifact_api_driver._artifact_export_job_id',
              return_value=WORKFLOW_ID),
        patch('rest.driver.artifact_api_driver._current_user_ctx',
              new=SimpleNamespace(get=lambda: principal)),
    ]
    for method, replacement in (
        ('create_artifact_export_job', dispatch),
        ('get_temporal_job_status', status),
        ('cancel_temporal_job', cancel),
    ):
        if replacement is not None:
            patches.append(patch.object(DataBackend, method, new=replacement))
    with ExitStack() as stack:
        for request_patch in patches:
            stack.enter_context(request_patch)
        yield


async def _create_job(driver, output) -> str:
    return await driver.create_artifact_export_job('bp-job', output, PROJECT_ID,
                                                   REQUEST_ID)


@pytest.mark.asyncio
async def test_same_request_dispatches_once_and_sets_dispatch_seal() -> None:
    driver, artifact_exports, output, principal = _artifact_driver()
    dispatch = AsyncMock(return_value=WORKFLOW_ID)
    with _request_context(principal, dispatch=dispatch):
        result = await _create_job(driver, output)

    assert result == str(WORKFLOW_ID)
    dispatch.assert_awaited_once()
    artifact_exports.seal_dispatch.assert_awaited_once_with(
        WORKFLOW_ID, project_id=PROJECT_ID)


@pytest.mark.asyncio
async def test_conflicting_request_is_rejected_before_connector_resolution(
) -> None:
    driver, artifact_exports, output, principal = _artifact_driver()
    artifact_exports.get.side_effect = None
    artifact_exports.get.return_value = SimpleNamespace()
    artifact_exports.create.side_effect = ValueError('different payload')
    connector_service = driver._required_state_services.connectors
    with _request_context(principal):
        with pytest.raises(ValueError, match='different payload'):
            await _create_job(driver, output)
    connector_service.get_for_project.assert_not_awaited()
    connector_service.execution_ref_for_project.assert_not_awaited()


def test_workflow_history_has_no_artifact_lifecycle_reference() -> None:
    assert 'artifact_export_job' not in ArtifactExportInput.__dataclass_fields__


@pytest.mark.asyncio
async def test_invalid_new_request_does_not_create_idempotency_row() -> None:
    driver, artifact_exports, output, principal = _artifact_driver()
    driver._required_state_services.connectors.get_for_project.side_effect = (
        ValueError('invalid connector'))
    with _request_context(principal):
        with pytest.raises(ValueError, match='invalid connector'):
            await _create_job(driver, output)
    artifact_exports.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_posthoc_databricks_export_is_rejected_before_dispatch() -> None:
    driver, artifact_exports, _, principal = _artifact_driver()
    connector_service = driver._required_state_services.connectors
    connector_service.get_for_project.return_value = SimpleNamespace(
        config=SimpleNamespace(type=data_source_api.ConnectorType.DATABRICKS))
    output = DatabricksPredictionOutput(
        artifact_type=PredictionArtifactType.PREDICTIONS,
        connector_id=str(CONNECTOR_ID),
        table_name='schema.predictions',
    )

    with _request_context(principal), pytest.raises(
            ValueError, match='Post-hoc Databricks'):
        await _create_job(driver, output)

    artifact_exports.create.assert_not_awaited()
    connector_service.execution_ref_for_project.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('dispatch_sealed', 'age', 'status', 'expect_seal'),
    [
        (True, None, None, False),
        (False, _ARTIFACT_UNSEALED_DISPATCH_WINDOW + timedelta(seconds=1),
         JobStatus.MISSING, True),
        (False, timedelta(), JobStatus.RUNNING, True),
    ],
    ids=('sealed-history', 'expired-unsealed', 'accepted-unsealed'),
)
async def test_existing_request_is_closed_without_redispatch(
    dispatch_sealed,
    age,
    status,
    expect_seal,
) -> None:
    created_at = now_utc() - age if age is not None else None
    driver, artifact_exports, output, principal = _artifact_driver(
        dispatch_sealed=dispatch_sealed, created_at=created_at)
    connector_service = driver._required_state_services.connectors
    dispatch = AsyncMock()
    temporal_status = (AsyncMock(
        return_value=status) if status is not None else None)
    with _request_context(principal, dispatch=dispatch, status=temporal_status):
        assert await _create_job(driver, output) == str(WORKFLOW_ID)
    dispatch.assert_not_awaited()
    assert artifact_exports.seal_dispatch.await_count == int(expect_seal)
    connector_service.get_for_project.assert_not_awaited()
    connector_service.execution_ref_for_project.assert_not_awaited()


@pytest.mark.asyncio
async def test_seal_write_failure_recovers_by_describing_accepted_start(
) -> None:
    driver, artifact_exports, output, principal = _artifact_driver()
    artifact_exports.seal_dispatch.side_effect = [
        RuntimeError('db failed'), artifact_exports.create.return_value[0]
    ]
    dispatch = AsyncMock(return_value=WORKFLOW_ID)
    status = AsyncMock(return_value=JobStatus.RUNNING)
    with _request_context(principal, dispatch=dispatch, status=status):
        with pytest.raises(RuntimeError, match='db failed'):
            await _create_job(driver, output)
        assert await _create_job(driver, output) == str(WORKFLOW_ID)
    dispatch.assert_awaited_once()


@pytest.mark.asyncio
async def test_ambiguous_accepted_start_retries_same_id_then_seals() -> None:
    driver, artifact_exports, output, principal = _artifact_driver()
    dispatch = AsyncMock(
        side_effect=[RuntimeError('transport ambiguous'), WORKFLOW_ID])
    status = AsyncMock(return_value=JobStatus.MISSING)
    with _request_context(principal, dispatch=dispatch, status=status):
        with pytest.raises(RuntimeError, match='transport ambiguous'):
            await _create_job(driver, output)
        assert await _create_job(driver, output) == str(WORKFLOW_ID)
    assert dispatch.await_count == 2
    artifact_exports.seal_dispatch.assert_awaited_once()


@pytest.mark.asyncio
async def test_seal_before_final_admission_check_blocks_dispatch() -> None:
    driver, artifact_exports, output, principal = _artifact_driver()
    entry, _ = artifact_exports.create.return_value
    sealed_entry = SimpleNamespace(**vars(entry))
    sealed_entry.dispatch_sealed = True
    artifact_exports.get.side_effect = [None, sealed_entry]
    dispatch = AsyncMock()
    with _request_context(principal, dispatch=dispatch):
        assert await _create_job(driver, output) == str(WORKFLOW_ID)
    dispatch.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('cancel_requested', 'cancel_error', 'expect_cancel'),
    [
        (True, None, True),
        (True, RuntimeError('cancel unavailable'), True),
        (False, None, False),
    ],
    ids=('cancel-race', 'cancel-failure', 'duplicate-seal'),
)
async def test_post_start_seal_honors_cancellation_state(
    cancel_requested,
    cancel_error,
    expect_cancel,
) -> None:
    driver, artifact_exports, output, principal = _artifact_driver()
    entry, _ = artifact_exports.create.return_value
    sealed_entry = SimpleNamespace(**vars(entry))
    sealed_entry.dispatch_sealed = True
    sealed_entry.cancel_requested = cancel_requested
    artifact_exports.seal_dispatch.return_value = sealed_entry
    dispatch = AsyncMock(return_value=WORKFLOW_ID)
    cancel = AsyncMock(side_effect=cancel_error,
                       return_value=JobStatus.CANCELLED)
    with _request_context(principal, dispatch=dispatch, cancel=cancel):
        assert await _create_job(driver, output) == str(WORKFLOW_ID)
    dispatch.assert_awaited_once()
    assert cancel.await_count == int(expect_cancel)


@pytest.mark.asyncio
async def test_concurrent_identical_creates_leave_temporal_workflow_running(
    tmp_path,
    temporal_client,
    data_backend_ctx,
) -> None:
    state_db = await StateDB.open_sqlite(str(tmp_path / 'artifact-state.db'))
    driver, _, output, principal = _artifact_driver()
    artifact_exports = ArtifactExportJobService(state_db)
    driver._state_services.artifact_exports = artifact_exports
    driver._data_backend_ctx = data_backend_ctx
    raw_client = await temporal_client.get()
    both_admitted = asyncio.Event()
    admitted_count = 0

    async def start_workflow(_ctx, _job_id, _output, _connector, workflow_id):
        nonlocal admitted_count
        admitted_count += 1
        if admitted_count == 2:
            both_admitted.set()
        await asyncio.wait_for(both_admitted.wait(), timeout=5)
        try:
            await raw_client.start_workflow(
                ArtifactDuplicateRunningWorkflow.run,
                id=str(workflow_id),
                task_queue='artifact-duplicate-test',
                id_reuse_policy=WorkflowIDReusePolicy.REJECT_DUPLICATE,
            )
        except WorkflowAlreadyStartedError:
            pass
        return workflow_id

    try:
        with _request_context(principal, dispatch=start_workflow):
            results = await asyncio.gather(
                _create_job(driver, output),
                _create_job(driver, output),
            )
        assert results == [str(WORKFLOW_ID), str(WORKFLOW_ID)]
        assert admitted_count == 2
        description = await temporal_client.get_workflow_desc(str(WORKFLOW_ID))
        assert description is not None
        assert description.status == WorkflowExecutionStatus.RUNNING
        entry = await artifact_exports.get(WORKFLOW_ID)
        assert entry is not None
        assert entry.dispatch_sealed
        assert not entry.cancel_requested
        await temporal_client.cancel(str(WORKFLOW_ID))
    finally:
        await state_db.close()


@pytest.mark.asyncio
async def test_status_and_cancel_are_temporal_authoritative_after_project_gate(
) -> None:
    driver, artifact_exports, _, principal = _artifact_driver()
    temporal_status = AsyncMock(return_value=JobStatus.MISSING)
    temporal_cancel = AsyncMock(return_value=JobStatus.MISSING)
    with _request_context(principal, status=temporal_status,
                          cancel=temporal_cancel):
        assert await driver.get_artifact_export_job_status(
            str(WORKFLOW_ID), PROJECT_ID) == JobStatus.MISSING
        assert await driver.cancel_artifact_export_job(
            str(WORKFLOW_ID), PROJECT_ID) == JobStatus.MISSING
    assert artifact_exports.get_for_project.await_count == 2
    artifact_exports.request_cancel.assert_awaited_once_with(
        WORKFLOW_ID, project_id=PROJECT_ID)


@pytest.mark.asyncio
async def test_cancel_missing_seals_before_later_retry() -> None:
    driver, artifact_exports, output, principal = _artifact_driver()
    entry, _ = artifact_exports.create.return_value

    async def request_cancel(_workflow_id, *, project_id):
        assert project_id == PROJECT_ID
        entry.dispatch_sealed = True
        entry.cancel_requested = True
        return entry

    artifact_exports.request_cancel.side_effect = request_cancel
    dispatch = AsyncMock()
    cancel = AsyncMock(return_value=JobStatus.MISSING)
    with _request_context(principal, dispatch=dispatch, cancel=cancel):
        assert await driver.cancel_artifact_export_job(
            str(WORKFLOW_ID), PROJECT_ID) == JobStatus.MISSING
        artifact_exports.get.side_effect = None
        artifact_exports.get.return_value = entry
        assert await _create_job(driver, output) == str(WORKFLOW_ID)
    dispatch.assert_not_awaited()


@pytest.mark.asyncio
async def test_project_isolation_blocks_status_before_temporal() -> None:
    driver, artifact_exports, _, principal = _artifact_driver()
    artifact_exports.get_for_project.side_effect = ArtifactExportJobNotFoundError(
        'not found')
    temporal_status = AsyncMock()
    with _request_context(principal, status=temporal_status):
        with pytest.raises(ArtifactExportJobNotFoundError):
            await driver.get_artifact_export_job_status(str(WORKFLOW_ID),
                                                        PROJECT_ID)
    temporal_status.assert_not_awaited()
