# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the StateDB-backed ML job queue
(rest/backends/ml_backend/job_queue.py).

These DB-backed queue tests run entirely against a real SQLite StateDB.

Covered invariants (plan ``Job Queue Design``):
  * FIFO admission from QUEUED -> RUNNING;
  * maximum top-level concurrent jobs;
  * maximum GPU slot usage;
  * queue length limits;
  * idempotent enqueue, release, and drain;
  * terminal job release admits the next queued job(s);
  * concurrent admission (parallel submit) stays within limits;
  * restart behavior (a fresh queue object recovers state from the DB).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from kumo_core.types import TaskType
from temporalio.client import WorkflowExecutionStatus

from rest.backends.ml_backend.job_queue import (
    JobConcurrencyLimitError,
    StateDbJobQueue,
)
from rest.backends.ml_backend.settings import MLJobQueueSettings
from rest.config.batch_prediction_job import (
    BatchPredictionJobConfig,
    BatchPredictionJobOutput,
)
from statedb.config.common import (
    BatchPredictionJobID,
    GeneratePredictionTableJobID,
    GraphIDv1,
    JobEventLogEntry,
    JobSource,
    JobStatus,
    PredictiveQueryID,
    ProjectID,
    TrainingJobID,
)
from statedb.db import MLJobService, StateDB

TEST_GRAPH_ID = GraphIDv1("graph_id_test")
TEST_PQUERY_ID = PredictiveQueryID("pquery_id_test")
TEST_PROJECT_ID = ProjectID("default")


@pytest.fixture(scope="session", autouse=True)
def setup_rest_test_session() -> None:
    """This unit suite does not require a Temporal test server."""


@pytest.fixture(autouse=True)
def auto_cleanup_per_func() -> None:
    """Override the REST integration cleanup fixture for this unit suite."""


@pytest_asyncio.fixture
async def state_db(tmp_path) -> StateDB:
    # The DB-backed queue derives all state from the default ML job tables;
    # no extra table is needed, so the default schema is used as-is.
    return await StateDB.open_sqlite(str(tmp_path / "state.sqlite"))


@pytest_asyncio.fixture
async def ml_jobs(state_db: StateDB) -> MLJobService:
    return MLJobService(state_db)


def _settings(
    *,
    max_jobs: int = 2,
    max_gpu: int = 4,
    max_queue: int = 10,
    use_queue: bool = True,
) -> MLJobQueueSettings:
    return MLJobQueueSettings(
        max_concurrent_jobs_total=max_jobs,
        max_concurrent_gpu_worker_slots=max_gpu,
        use_job_queue=use_queue,
        max_job_queue_length=max_queue,
    )


def _queue(state_db: StateDB, settings: MLJobQueueSettings) -> StateDbJobQueue:
    return StateDbJobQueue(state_db, settings)


async def _make_bp_job(
    ml_jobs: MLJobService,
    *,
    gpu_workers: int = 1,
    created_at: datetime | None = None,
) -> BatchPredictionJobID:
    """Create a QUEUED batch-prediction job.

    ``num_parallel_workers_limit`` controls the GPU slot demand, matching
    ``get_job_gpu_slots``.
    """
    now = created_at or datetime.utcnow()
    job_id = BatchPredictionJobID.make_random_id()
    await ml_jobs.create_batch_prediction_job(
        id=job_id,
        now=now,
        config=BatchPredictionJobConfig(
            training_job_id=TrainingJobID("trainingjob-deadbeef"),
            graph_id=TEST_GRAPH_ID,
            predictive_query_id=TEST_PQUERY_ID,
            generate_pred_table_job_id=(
                GeneratePredictionTableJobID.as_child_of(job_id)),
            task_type=TaskType.BINARY_CLASSIFICATION,
            num_parallel_workers_limit=gpu_workers,
        ),
        source=JobSource.SDK,
        project_id=TEST_PROJECT_ID,
    )
    return job_id


async def _status_of(
    ml_jobs: MLJobService,
    job_id: BatchPredictionJobID,
) -> JobStatus:
    job = await ml_jobs.get_job_if_exists_generic(job_id)
    assert job is not None
    return job.execution.status


async def _terminalize(
    ml_jobs: MLJobService,
    job_id: BatchPredictionJobID,
    status: JobStatus = JobStatus.CANCELLED,
) -> None:
    job = await ml_jobs.get_job_if_exists_generic(job_id)
    assert job is not None
    job.execution.status = status
    from temporalio.client import WorkflowExecutionStatus
    job.execution.workflow_end_state = WorkflowExecutionStatus.COMPLETED
    await ml_jobs.save_batch_prediction_job_output(job_id, job.execution)


# --------------------------------------------------------------------------- #
# FIFO admission + max concurrent jobs
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_fifo_admission_and_max_jobs(
    state_db: StateDB,
    ml_jobs: MLJobService,
) -> None:
    settings = _settings(max_jobs=2, max_gpu=10)
    queue = _queue(state_db, settings)

    base = datetime(2024, 1, 1, 0, 0, 0)
    a = await _make_bp_job(ml_jobs, gpu_workers=1, created_at=base)
    b = await _make_bp_job(ml_jobs, gpu_workers=1,
                           created_at=base + timedelta(seconds=1))
    c = await _make_bp_job(ml_jobs, gpu_workers=1,
                           created_at=base + timedelta(seconds=2))
    # Deterministic FIFO order: a < b < c.

    assert await queue.submit_job(a) == JobStatus.RUNNING
    assert await queue.submit_job(b) == JobStatus.RUNNING
    # Third job exceeds max_jobs=2, stays queued.
    assert await queue.submit_job(c) == JobStatus.QUEUED

    status = await queue.report_status()
    assert status.job_slots_occupied == 2
    assert set(status.running_jobs) == {a, b}
    assert status.queued_jobs == [c]

    # Terminalize the FIFO-oldest running job; release admits c.
    await _terminalize(ml_jobs, a)
    admitted = await queue.release_job(a)
    assert admitted == [c]
    assert await _status_of(ml_jobs, c) == JobStatus.RUNNING


@pytest.mark.asyncio
async def test_runtime_starts_when_admitted(
    state_db: StateDB,
    ml_jobs: MLJobService,
) -> None:
    queue = _queue(state_db, _settings(max_jobs=1, max_gpu=10))
    created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    job_id = await _make_bp_job(ml_jobs, created_at=created_at)

    assert await queue.submit_job(job_id) == JobStatus.RUNNING
    job = await ml_jobs.get_job_if_exists_generic(job_id)
    assert job is not None
    admission_time = job.execution.start_time
    assert admission_time > created_at
    # Submission remains independently persisted for queue-time diagnostics,
    # but is intentionally not used by either runtime surface.
    assert job.timestamps.created_at == created_at
    assert job.to_generic(None).metadata.start_time == admission_time

    await ml_jobs.save_batch_prediction_job_output(
        job_id,
        BatchPredictionJobOutput(
            status=JobStatus.RUNNING,
            log=JobEventLogEntry.starting(),
        ),
    )
    job = await ml_jobs.get_job_if_exists_generic(job_id)
    assert job is not None
    assert job.execution.start_time == admission_time
    assert job.to_generic(None).metadata.start_time == admission_time


@pytest.mark.asyncio
async def test_runtime_excludes_queue_when_terminalized_before_admission(
    ml_jobs: MLJobService,
) -> None:
    created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    job_id = await _make_bp_job(ml_jobs, created_at=created_at)
    await ml_jobs.save_batch_prediction_job_output(
        job_id,
        BatchPredictionJobOutput(
            status=JobStatus.CANCELLED,
            log=JobEventLogEntry.cancelled(),
            workflow_end_state=WorkflowExecutionStatus.CANCELED,
        ),
    )

    job = await ml_jobs.get_job_if_exists_generic(job_id)
    assert job is not None
    assert job.execution.start_time > created_at
    assert job.execution.start_time == job.execution.last_updated_at
    assert job.timestamps.created_at == created_at


@pytest.mark.asyncio
async def test_fifo_order_respected_under_starvation(
    state_db: StateDB,
    ml_jobs: MLJobService,
) -> None:
    settings = _settings(max_jobs=1, max_gpu=10)
    queue = _queue(state_db, settings)
    base = datetime(2024, 1, 1)
    first = await _make_bp_job(ml_jobs, created_at=base)
    second = await _make_bp_job(ml_jobs, created_at=base + timedelta(seconds=1))
    third = await _make_bp_job(ml_jobs, created_at=base + timedelta(seconds=2))

    assert await queue.submit_job(first) == JobStatus.RUNNING
    assert await queue.submit_job(second) == JobStatus.QUEUED
    assert await queue.submit_job(third) == JobStatus.QUEUED

    # Free one slot; only the FIFO head (second) is admitted.
    await _terminalize(ml_jobs, first)
    assert await queue.release_job(first) == [second]
    assert await _status_of(ml_jobs, second) == JobStatus.RUNNING
    assert await _status_of(ml_jobs, third) == JobStatus.QUEUED


# --------------------------------------------------------------------------- #
# GPU slot accounting
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_gpu_slot_accounting(
    state_db: StateDB,
    ml_jobs: MLJobService,
) -> None:
    # Plenty of job slots, but only 4 GPU slots.
    settings = _settings(max_jobs=10, max_gpu=4)
    queue = _queue(state_db, settings)
    base = datetime(2024, 1, 1)
    big = await _make_bp_job(ml_jobs, gpu_workers=3, created_at=base)
    small = await _make_bp_job(ml_jobs, gpu_workers=2,
                               created_at=base + timedelta(seconds=1))

    assert await queue.submit_job(big) == JobStatus.RUNNING  # 3/4 GPU
    # small needs 2 more -> 5 > 4, stays queued.
    assert await queue.submit_job(small) == JobStatus.QUEUED
    assert (await queue.report_status()).gpu_slots_occupied == 3

    await _terminalize(ml_jobs, big)
    assert await queue.release_job(big) == [small]
    assert (await queue.report_status()).gpu_slots_occupied == 2


@pytest.mark.asyncio
async def test_job_exceeding_max_gpu_is_rejected(
    state_db: StateDB,
    ml_jobs: MLJobService,
) -> None:
    settings = _settings(max_jobs=10, max_gpu=2)
    queue = _queue(state_db, settings)
    too_big = await _make_bp_job(ml_jobs, gpu_workers=5)
    with pytest.raises(JobConcurrencyLimitError):
        await queue.submit_job(too_big)


# --------------------------------------------------------------------------- #
# Queue length enforcement
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_queue_length_enforced(
    state_db: StateDB,
    ml_jobs: MLJobService,
) -> None:
    settings = _settings(max_jobs=1, max_gpu=10, max_queue=2)
    queue = _queue(state_db, settings)
    base = datetime(2024, 1, 1)
    running = await _make_bp_job(ml_jobs, created_at=base)
    assert await queue.submit_job(running) == JobStatus.RUNNING

    q1 = await _make_bp_job(ml_jobs, created_at=base + timedelta(seconds=1))
    q2 = await _make_bp_job(ml_jobs, created_at=base + timedelta(seconds=2))
    assert await queue.submit_job(q1) == JobStatus.QUEUED
    assert await queue.submit_job(q2) == JobStatus.QUEUED

    # Queue is now full (2 queued). A third submission is rejected.
    q3 = await _make_bp_job(ml_jobs, created_at=base + timedelta(seconds=3))
    with pytest.raises(JobConcurrencyLimitError):
        await queue.submit_job(q3)


# --------------------------------------------------------------------------- #
# Idempotency
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_submit_release_drain_idempotent(
    state_db: StateDB,
    ml_jobs: MLJobService,
) -> None:
    settings = _settings(max_jobs=2, max_gpu=10)
    queue = _queue(state_db, settings)
    base = datetime(2024, 1, 1)
    a = await _make_bp_job(ml_jobs, created_at=base)

    assert await queue.submit_job(a) == JobStatus.RUNNING
    # Re-submitting a RUNNING job is a no-op.
    assert await queue.submit_job(a) == JobStatus.RUNNING
    assert (await queue.report_status()).job_slots_occupied == 1

    b = await _make_bp_job(ml_jobs, created_at=base + timedelta(seconds=1))
    assert await queue.submit_job(b) == JobStatus.RUNNING
    # Drain with no free capacity admits nothing.
    assert await queue.drain() == []
    # Release a non-terminal job (no capacity freed) admits nothing.
    assert await queue.release_job(a) == []

    # Enqueue c while the queue is full (a, b both running) -> stays QUEUED.
    c = await _make_bp_job(ml_jobs, created_at=base + timedelta(seconds=2))
    assert await queue.submit_job(c) == JobStatus.QUEUED

    # Terminalize a; repeated release only admits c once (idempotent).
    await _terminalize(ml_jobs, a)
    first = await queue.release_job(a)
    second = await queue.release_job(a)
    assert first == [c]
    assert second == []


# --------------------------------------------------------------------------- #
# Concurrent admission
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_concurrent_submit_respects_limit(
    state_db: StateDB,
    ml_jobs: MLJobService,
) -> None:
    settings = _settings(max_jobs=3, max_gpu=100)
    queue = _queue(state_db, settings)
    base = datetime(2024, 1, 1)
    jobs = []
    for i in range(8):
        j = await _make_bp_job(ml_jobs, created_at=base + timedelta(seconds=i))
        jobs.append(j)

    # Fire all submits concurrently; the per-tx lock must serialize them.
    results = await asyncio.gather(*[queue.submit_job(j) for j in jobs])
    running = [s for s in results if s == JobStatus.RUNNING]
    # At most max_jobs may be RUNNING.
    assert len(running) <= 3
    status = await queue.report_status()
    assert status.job_slots_occupied == 3
    assert len(status.running_jobs) == 3
    assert len(status.queued_jobs) == 5


# --------------------------------------------------------------------------- #
# Restart behavior
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_restart_recovers_state_and_admits(
    state_db: StateDB,
    ml_jobs: MLJobService,
) -> None:
    settings = _settings(max_jobs=2, max_gpu=10)
    queue = _queue(state_db, settings)
    base = datetime(2024, 1, 1)
    a = await _make_bp_job(ml_jobs, created_at=base)
    b = await _make_bp_job(ml_jobs, created_at=base + timedelta(seconds=1))
    c = await _make_bp_job(ml_jobs, created_at=base + timedelta(seconds=2))
    await queue.submit_job(a)
    await queue.submit_job(b)
    await queue.submit_job(c)
    assert await _status_of(ml_jobs, c) == JobStatus.QUEUED

    # Simulate a process restart: a brand-new queue object (no memory) over the
    # same DB. State is recovered purely from the job rows.
    fresh_queue = _queue(state_db, settings)
    status = await fresh_queue.report_status()
    assert set(status.running_jobs) == {a, b}
    assert status.queued_jobs == [c]

    # Free a slot out-of-band, then drain on the fresh queue admits c FIFO.
    await _terminalize(ml_jobs, a)
    assert await fresh_queue.drain() == [c]
