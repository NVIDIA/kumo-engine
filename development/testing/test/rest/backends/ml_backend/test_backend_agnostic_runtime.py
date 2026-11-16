# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
from test.statedb.ml.data_model import (
    case_batch_prediction_job,
    case_training_job,
)
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.deps.authentication import RequestContext, UserAuthorization
from statedb.config.common import (
    BatchPredictionJobID,
    JobStatus,
    ProjectID,
    TrainingJobID,
)


def _backend(service) -> tuple[MLJobBackend, AsyncMock]:
    backend = MLJobBackend.__new__(MLJobBackend)
    backend._ml_job_service = service
    backend._settings = SimpleNamespace(ml_job_settings=SimpleNamespace(
        use_fake_job_backend=False))
    queue = AsyncMock()
    backend._job_queue = queue
    return backend, queue


def _job(job_kind: str, status: JobStatus):
    if job_kind == "training":
        job = case_training_job().value
        job.id = TrainingJobID.make_random_id()
    else:
        job = case_batch_prediction_job().value
        job.id = BatchPredictionJobID.make_random_id()
    job.execution.status = status
    return job


def _delete_runtime(job_kind: str, job_home, *, deleted=True):
    job = _job(job_kind, JobStatus.DONE)
    lookup = ({
        'list_training_job_rows': AsyncMock(return_value=[job])
    } if job_kind == 'training' else {
        'get_batch_prediction_jobs': AsyncMock(return_value=[job])
    })
    service = SimpleNamespace(
        **lookup,
        delete_job_generic=AsyncMock(return_value=deleted),
    )
    backend, queue = _backend(service)
    backend.get_job_home_dir = AsyncMock(return_value=str(job_home))
    return SimpleNamespace(job=job, service=service, backend=backend,
                           queue=queue)


@pytest.mark.asyncio
@pytest.mark.parametrize("job_kind", ["training", "batch_prediction"])
@pytest.mark.parametrize(
    ("status", "force", "raises"),
    [
        (JobStatus.DONE, False, False),
        (JobStatus.RUNNING, False, True),
        (JobStatus.RUNNING, True, False),
    ],
)
async def test_delete_job_releases_only_deletable_jobs(
    job_kind: str,
    status: JobStatus,
    force: bool,
    raises: bool,
) -> None:
    job = _job(job_kind, status)
    service = SimpleNamespace(
        list_training_job_rows=AsyncMock(return_value=[job]),
        get_batch_prediction_jobs=AsyncMock(return_value=[job]),
        delete_job_generic=AsyncMock(return_value=True),
    )
    backend, queue = _backend(service)
    ctx = RequestContext(UserAuthorization.disabled())

    if job_kind == "training":
        delete = backend.delete_training_job(ctx, job.id, force=force)
    else:
        delete = backend.delete_batch_prediction(ctx, job.id, force=force)

    if raises:
        with pytest.raises(ValueError, match="queued/running"):
            await delete
        service.delete_job_generic.assert_not_awaited()
        queue.release_job.assert_not_awaited()
    else:
        assert await delete
        queue.release_job.assert_awaited_once_with(job.id)
    if job_kind == "training":
        service.list_training_job_rows.assert_awaited_once_with(
            job_ids={job.id},
            project_ids=None,
            limit=1,
        )


@pytest.mark.asyncio
async def test_delete_many_batch_prediction_jobs_skips_running_jobs() -> None:
    done = _job("batch_prediction", JobStatus.DONE)
    running = _job("batch_prediction", JobStatus.RUNNING)
    service = SimpleNamespace(
        get_batch_prediction_jobs=AsyncMock(return_value=[done, running]),
        delete_batch_prediction_jobs=AsyncMock(return_value={done.id}),
    )
    backend, queue = _backend(service)

    deleted = await backend.delete_many_batch_prediction_jobs(
        RequestContext(UserAuthorization.disabled()),
        {done.id, running.id},
    )

    assert deleted == {done.id}
    service.delete_batch_prediction_jobs.assert_awaited_once_with({done.id},
                                                                  force=False)
    queue.release_job.assert_awaited_once_with(done.id)


@pytest.mark.asyncio
async def test_delete_batch_prediction_removes_only_internal_job_home(
    tmp_path, ) -> None:
    runtime = _delete_runtime('batch_prediction', tmp_path)
    job_home = tmp_path / "jobs" / str(runtime.job.id)
    runtime.backend.get_job_home_dir.return_value = str(job_home)
    internal_output = job_home / "prediction_output" / "predictions.parquet"
    internal_output.mkdir(parents=True)
    (internal_output / "part-0.parquet").write_bytes(b"internal")
    customer_output = tmp_path / "customer-output" / "predictions.parquet"
    customer_output.mkdir(parents=True)
    (customer_output / "part-0.parquet").write_bytes(b"customer")
    assert await runtime.backend.delete_batch_prediction(
        RequestContext(UserAuthorization.disabled()), runtime.job.id)

    assert not job_home.exists()
    assert (customer_output / "part-0.parquet").read_bytes() == b"customer"
    runtime.service.delete_job_generic.assert_awaited_once_with(
        runtime.job.id, force=False)
    runtime.queue.release_job.assert_awaited_once_with(runtime.job.id)


@pytest.mark.asyncio
async def test_delete_training_removes_only_internal_job_home(tmp_path) -> None:
    runtime = _delete_runtime('training', tmp_path)
    job_home = tmp_path / "jobs" / str(runtime.job.id)
    runtime.backend.get_job_home_dir.return_value = str(job_home)
    internal_model = job_home / "models" / "task" / "automl" / "version_0"
    internal_model.mkdir(parents=True)
    (internal_model / "model.pt").write_bytes(b"internal")
    customer_output = tmp_path / "customer-output" / "model" / "1"
    customer_output.mkdir(parents=True)
    (customer_output / "model.pt").write_bytes(b"customer")
    assert await runtime.backend.delete_training_job(
        RequestContext(UserAuthorization.disabled()), runtime.job.id)

    assert not job_home.exists()
    assert (customer_output / "model.pt").read_bytes() == b"customer"
    runtime.service.delete_job_generic.assert_awaited_once_with(
        runtime.job.id, force=False)
    runtime.queue.release_job.assert_awaited_once_with(runtime.job.id)


@pytest.mark.asyncio
async def test_delete_batch_prediction_is_idempotent_when_job_home_is_absent(
    tmp_path, ) -> None:
    missing_home = tmp_path / "jobs" / "missing"
    runtime = _delete_runtime('batch_prediction', missing_home)

    assert await runtime.backend.delete_batch_prediction(
        RequestContext(UserAuthorization.disabled()), runtime.job.id)
    assert not missing_home.exists()


@pytest.mark.asyncio
async def test_delete_batch_prediction_retains_home_when_metadata_is_not_deleted(
    tmp_path, ) -> None:
    job_home = tmp_path / "jobs" / "retained"
    job_home.mkdir(parents=True)
    runtime = _delete_runtime('batch_prediction', job_home, deleted=False)

    assert not await runtime.backend.delete_batch_prediction(
        RequestContext(UserAuthorization.disabled()), runtime.job.id)

    assert job_home.exists()
    runtime.queue.release_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_many_batch_predictions_removes_only_confirmed_homes(
    tmp_path, ) -> None:
    deleted = _job("batch_prediction", JobStatus.DONE)
    retained = _job("batch_prediction", JobStatus.DONE)
    homes = {
        deleted.id: tmp_path / str(deleted.id),
        retained.id: tmp_path / str(retained.id),
    }
    for home in homes.values():
        home.mkdir()
    service = SimpleNamespace(
        get_batch_prediction_jobs=AsyncMock(return_value=[deleted, retained]),
        delete_batch_prediction_jobs=AsyncMock(return_value={deleted.id}),
    )
    backend, queue = _backend(service)
    backend.get_job_home_dir = AsyncMock(
        side_effect=lambda job_id: str(homes[job_id]))

    assert await backend.delete_many_batch_prediction_jobs(
        RequestContext(UserAuthorization.disabled()),
        set(homes),
    ) == {deleted.id}

    assert not homes[deleted.id].exists()
    assert homes[retained.id].exists()
    queue.release_job.assert_awaited_once_with(deleted.id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "job_home",
    [
        "s3://example-data/jobs/bp-delete",
        "gs://example-data/jobs/bp-delete",
        "abfss://data@account.dfs.core.windows.net/jobs/bp-delete",
    ],
)
async def test_delete_batch_prediction_targets_only_cloud_job_home(
    job_home: str, ) -> None:
    runtime = _delete_runtime('batch_prediction', job_home)

    with (
            patch("rest.backends.ml_backend.ml_job_backend.fsutil.exists",
                  return_value=True),
            patch("rest.backends.ml_backend.ml_job_backend.fsutil.rm") as rm,
    ):
        assert await runtime.backend.delete_batch_prediction(
            RequestContext(UserAuthorization.disabled()), runtime.job.id)

    rm.assert_called_once_with(job_home)


@pytest.mark.asyncio
async def test_delete_batch_prediction_cleanup_failure_is_best_effort(
        tmp_path, caplog) -> None:
    job_home = tmp_path / "jobs" / "cleanup-failure"
    job_home.mkdir(parents=True)
    runtime = _delete_runtime('batch_prediction', job_home)
    provider_detail = "provider-secret-cleanup-detail"

    with patch(
            "rest.backends.ml_backend.ml_job_backend.fsutil.rm",
            side_effect=OSError(provider_detail),
    ):
        assert await runtime.backend.delete_batch_prediction(
            RequestContext(UserAuthorization.disabled()), runtime.job.id)

    runtime.service.delete_job_generic.assert_awaited_once_with(
        runtime.job.id, force=False)
    runtime.queue.release_job.assert_awaited_once_with(runtime.job.id)
    cleanup_record = next(
        record for record in caplog.records
        if "Internal home cleanup failed" in record.getMessage())
    assert cleanup_record.levelno == logging.ERROR
    assert str(runtime.job.id) in cleanup_record.getMessage()
    assert str(job_home) in cleanup_record.getMessage()
    assert "OSError" in cleanup_record.getMessage()
    assert provider_detail not in caplog.text


@pytest.mark.asyncio
async def test_delete_many_batch_predictions_attempts_every_home_on_failure(
    tmp_path, ) -> None:
    first = _job("batch_prediction", JobStatus.DONE)
    second = _job("batch_prediction", JobStatus.DONE)
    homes = {
        first.id: tmp_path / str(first.id),
        second.id: tmp_path / str(second.id),
    }
    for home in homes.values():
        home.mkdir()
    service = SimpleNamespace(
        get_batch_prediction_jobs=AsyncMock(return_value=[first, second]),
        delete_batch_prediction_jobs=AsyncMock(
            return_value={first.id, second.id}),
    )
    backend, queue = _backend(service)
    backend.get_job_home_dir = AsyncMock(
        side_effect=lambda job_id: str(homes[job_id]))
    attempted = set()

    def remove(path):
        attempted.add(path)
        if path == str(homes[first.id]):
            raise OSError("first cleanup failed")

    with patch("rest.backends.ml_backend.ml_job_backend.fsutil.rm", remove):
        assert await backend.delete_many_batch_prediction_jobs(
            RequestContext(UserAuthorization.disabled()),
            set(homes),
        ) == {first.id, second.id}

    assert attempted == {str(home) for home in homes.values()}
    service.delete_batch_prediction_jobs.assert_awaited_once_with(
        {first.id, second.id}, force=False)
    assert {entry.args[0]
            for entry in queue.release_job.await_args_list} == {
                first.id,
                second.id,
            }


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 20])
async def test_in_progress_statuses_use_one_bulk_read(count: int) -> None:
    jobs = []
    for _ in range(count):
        job = _job("training", JobStatus.RUNNING)
        job.generate_train_table_job_id = None
        jobs.append(job)
    statuses = {job.id: JobStatus.RUNNING for job in jobs}
    service = SimpleNamespace(
        list_open_parent_jobs=AsyncMock(return_value=jobs),
        get_job_statuses_by_ids=AsyncMock(return_value=statuses),
        get_job_if_exists_generic=AsyncMock(),
    )
    backend, _ = _backend(service)

    result = await backend.get_in_progress_job_or_job_status(
        RequestContext(UserAuthorization.disabled()),
        only_status_job_ids=set(statuses),
    )

    assert result == statuses
    service.get_job_statuses_by_ids.assert_awaited_once_with(set(statuses))
    service.get_job_if_exists_generic.assert_not_awaited()


@pytest.mark.asyncio
async def test_in_progress_project_filter_precedes_bulk_status_read() -> None:
    job = _job("training", JobStatus.RUNNING)
    job.generate_train_table_job_id = None
    job.project_id = ProjectID("project_" + "a" * 32)
    service = SimpleNamespace(
        list_open_parent_jobs=AsyncMock(return_value=[job]),
        get_job_statuses_by_ids=AsyncMock(),
        get_job_if_exists_generic=AsyncMock(),
    )
    backend, _ = _backend(service)

    result = await backend.get_in_progress_job_or_job_status(
        RequestContext(UserAuthorization.disabled()),
        only_status_job_ids={job.id},
        project_id=ProjectID("project_" + "b" * 32),
    )

    assert result == {}
    service.get_job_statuses_by_ids.assert_not_awaited()
    service.get_job_if_exists_generic.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("job_kind", ["training", "batch_prediction"])
async def test_cancel_job_marks_state_and_releases_capacity(
        job_kind: str) -> None:
    job = _job(job_kind, JobStatus.RUNNING)
    job.execution.workflow_end_state = None
    service = SimpleNamespace(
        get_job_if_exists_generic=AsyncMock(return_value=job),
        cancel_job_if_open_generic=AsyncMock(return_value=True),
    )
    backend, queue = _backend(service)
    backend._cancel_job_workflow = AsyncMock()
    backend._cancellation_tasks = set()

    if job_kind == "training":
        cancelled = await backend.cancel_training_job(job.id)
    else:
        cancelled = await backend.cancel_batch_prediction_job(job.id)
    if backend._cancellation_tasks:
        await asyncio.gather(*backend._cancellation_tasks)

    assert cancelled
    service.cancel_job_if_open_generic.assert_awaited_once_with(job.id)
    queue.release_job.assert_awaited_once_with(job.id)
    backend._cancel_job_workflow.assert_awaited_once()
