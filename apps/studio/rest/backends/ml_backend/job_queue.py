# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""StateDB-backed ML job queue / admission library.

This stateless, transactional admission library derives *all* of its state
from the existing StateDB ML job tables. It replaced the former in-memory,
AutoRPC queue singleton.

Design summary
==============

The legacy ``MLJobQueueManager`` kept the queue (``deque`` of ``JobID``), the
set of running jobs, and the occupied job/GPU slot counts purely in process
memory.  On startup it rebuilt that memory from
``MLJobService.list_open_parent_jobs()`` -- i.e. the authoritative state was
*already* the DB.  Because of that, the queue does **not** need a dedicated
table: a job is "queued" iff its row has ``execution.status == QUEUED`` and it
is a top-level (non-child) job, and it is "running" iff its status is
``RUNNING``.  FIFO ordering is recovered from the immutable submission time
(``timestamps.created_at``).  GPU slot
demand is derived from the job type/config, exactly as
``_get_job_gpu_slots`` does today.

Every admission/release pass therefore:

1. opens a single StateDB ``write_tx``;
2. takes a serialization lock (see "Locking" below);
3. reads the current open top-level jobs from the DB;
4. computes occupied job/GPU slots from the ``RUNNING`` jobs;
5. admits queued jobs FIFO while capacity allows, flipping their status to
   ``RUNNING`` in the *same* transaction.

Because reads and writes happen in one transaction guarded by a lock, the
admission decision can never race another admission/release pass.

Locking
=======

* **SQLite**: ``StateDB.write_tx`` issues ``BEGIN IMMEDIATE`` and sets
  ``busy_timeout=30000``.  ``BEGIN IMMEDIATE`` acquires the database-level
  ``RESERVED`` write lock at the start of the transaction, so two concurrent
  admission passes are fully serialized -- the second blocks (up to the busy
  timeout) until the first commits.  No extra primitive is required.

* **Postgres**: separate connections in ``READ COMMITTED`` do *not* block each
  other just because both write, so we take a transaction-scoped advisory lock
  (``pg_advisory_xact_lock``) keyed on the queue name.  The lock is held until
  the transaction commits/aborts and serializes all admission/release passes
  for that queue across connections and processes.  Row contention is also
  covered because both passes update the same job rows, but the advisory lock
  makes the *decision* (read-compute-admit) atomic, which row locks alone would
  not.

Non-job capacity holds (removed)
================================

The legacy manager exposed ``occupy_slot``/``vacate_slot`` for non-ML
workloads (diskgraph graph refresh) to consume a job slot without a job row.
A caller audit proved there are **zero production callers** (only legacy
AutoRPC unit tests referenced them), so this behavior is intentionally not
modeled in the DB-backed queue.  No extra table is introduced and the default
StateDB schema hash is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from io_utils import logging
from rest.backends.ml_backend.job_models import (
    GeneratePredTableJobDBEntry,
    GenerateTrainTableJobDBEntry,
    TrainingJobDBEntry,
)
from rest.backends.ml_backend.settings import MLJobQueueSettings
from statedb.config.common import (
    BatchPredictionJobID,
    GeneratePredictionTableJobID,
    GenerateTrainTableJobID,
    JobEventLogEntry,
    JobID,
    JobStatus,
    JobType,
    TrainingJobID,
    now_utc,
)
from statedb.db.ml_jobs import MLJobService
from statedb.db.repositories import JsonRepository
from statedb.db.runtime import StateDB, StateTx
from statedb.ml.jobs import BatchPredictionJobDBEntry

log = logging.getLogger(__name__)

# Stable 64-bit-ish key for the Postgres advisory lock.  All admission/release
# passes for the ML job queue use the same key, so they serialize against each
# other (but not against unrelated advisory locks).
_PG_ADVISORY_LOCK_KEY = 0x4B554D4F5F4D4C51  # "KUMO_MLQ"


class JobConcurrencyLimitError(ValueError):
    """Raised when a job cannot be admitted/queued under the limits."""


@dataclass
class JobQueueStatus:
    """Snapshot of queue state, derived from the DB (debugging/tests)."""

    job_slots_occupied: int
    gpu_slots_occupied: int
    running_jobs: list[JobID]
    queued_jobs: list[JobID]
    max_concurrent_jobs_total: int
    max_concurrent_gpu_worker_slots: int
    max_job_queue_length: int
    use_job_queue: bool


def get_job_gpu_slots(job_entry: Any) -> int:
    """GPU slot demand for a job entry.

    Mirrors ``MLJobQueueManager._get_job_gpu_slots`` so the migration is a
    behavior-preserving swap.
    """
    if isinstance(job_entry, TrainingJobDBEntry):
        return 1
    if isinstance(job_entry, BatchPredictionJobDBEntry):
        return (1 if not job_entry.config else
                job_entry.config.num_parallel_workers_limit)
    return 0


@dataclass
class _OpenJob:
    """Lightweight admission view of an open top-level job."""

    id: JobID
    status: JobStatus
    created_at: Optional[datetime]
    gpu_slots: int


class StateDbJobQueue:
    """DB-backed FIFO admission queue over the StateDB ML job tables.

    The queue is *stateless*: every operation reads the authoritative job
    rows inside a single locked transaction, so concurrent REST workers and
    Temporal activities can all drive it safely.

    All public methods are idempotent:

    * :meth:`submit_job` -- enqueue (no-op if already queued/running) and
      attempt admission; returns the resulting :class:`JobStatus`.
    * :meth:`release_job` -- called on terminal/cancel; admits the next jobs.
      Safe to call repeatedly for the same job.
    * :meth:`drain` -- re-run admission (e.g. after a manual capacity change).
    """
    def __init__(
        self,
        state_db: StateDB,
        settings: MLJobQueueSettings,
        *,
        queue_name: str = "ml_jobs",
    ) -> None:
        self._state_db = state_db
        self._settings = settings
        self._queue_name = queue_name

        # Read open top-level jobs through MLJobService, which filters
        # ``workflow_end_state IS NULL`` and detects orphan child jobs in SQL
        # (``list_open_parent_jobs_in_tx``).  This avoids scanning every
        # historical job row on each admission pass / queue poll.
        self._ml_job_service = MLJobService(state_db)

        # Repositories over the *existing* ML job tables.  We construct our own
        # so admission can flip a job's status to RUNNING inside the same
        # transaction it read open jobs in (the public MLJobService write
        # methods each open their own tx).
        self._training_jobs = JsonRepository("training_jobs",
                                             TrainingJobDBEntry)
        self._batch_prediction_jobs = JsonRepository("batch_prediction_jobs",
                                                     BatchPredictionJobDBEntry)
        self._generate_train_table_jobs = JsonRepository(
            "generate_train_table_jobs", GenerateTrainTableJobDBEntry)
        self._generate_pred_table_jobs = JsonRepository(
            "generate_pred_table_jobs", GeneratePredTableJobDBEntry)

    # ------------------------------------------------------------------ #
    # Locking primitive
    # ------------------------------------------------------------------ #
    async def _lock(self, tx: StateTx) -> None:
        """Serialize admission/release across connections and processes.

        SQLite ``write_tx`` already holds a DB-level write lock (BEGIN
        IMMEDIATE), so nothing extra is needed.  Postgres needs an explicit
        transaction-scoped advisory lock.
        """
        if tx.backend == "postgres":
            await tx.execute("SELECT pg_advisory_xact_lock($1)",
                             _PG_ADVISORY_LOCK_KEY)

    # ------------------------------------------------------------------ #
    # Reading open jobs (single-tx)
    # ------------------------------------------------------------------ #
    async def _list_open_top_level_jobs(self, tx: StateTx) -> list[_OpenJob]:
        """Open (non-terminal) top-level jobs, as admission views.

        Delegates to ``MLJobService.list_open_parent_jobs_in_tx``, which already
        filters ``execution.workflow_end_state IS NULL`` and excludes child
        generate-train/pred-table jobs (whose parent exists) entirely in SQL --
        so we only deserialize open jobs, not the full job history.  We then
        keep only the QUEUED / RUNNING rows the queue cares about.
        """
        entries = await self._ml_job_service.list_open_parent_jobs_in_tx(tx)
        open_jobs: list[_OpenJob] = []
        for entry in entries:
            view = self._job_view(entry)
            if view is not None:
                open_jobs.append(view)
        return open_jobs

    def _job_view(self, entry: Any) -> Optional[_OpenJob]:
        execution = getattr(entry, "execution", None)
        if execution is None:
            return None
        status = execution.status
        if status not in (JobStatus.QUEUED, JobStatus.RUNNING):
            return None
        if getattr(execution, "workflow_end_state", None):
            return None
        return _OpenJob(
            id=entry.id,
            status=status,
            created_at=getattr(getattr(entry, "timestamps", None), "created_at",
                               None),
            gpu_slots=get_job_gpu_slots(entry),
        )

    # ------------------------------------------------------------------ #
    # Capacity computation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _running(jobs: list[_OpenJob]) -> list[_OpenJob]:
        return [j for j in jobs if j.status == JobStatus.RUNNING]

    @staticmethod
    def _queued_fifo(jobs: list[_OpenJob]) -> list[_OpenJob]:
        queued = [j for j in jobs if j.status == JobStatus.QUEUED]

        # FIFO: oldest submission time first; missing created_at sorts first.
        # Sort on a numeric epoch (via .timestamp()) rather than the datetime
        # objects directly, so a mix of timezone-aware and naive timestamps
        # (possible across DB round-trips/backends) never raises
        # "can't compare offset-naive and offset-aware datetimes".
        def _key(job: "_OpenJob") -> tuple[int, float]:
            if job.created_at is None:
                return (0, 0.0)
            try:
                return (1, job.created_at.timestamp())
            except (OverflowError, OSError, ValueError):
                return (1, 0.0)

        queued.sort(key=_key)
        return queued

    def _capacity(
        self,
        running: list[_OpenJob],
    ) -> tuple[int, int]:
        job_slots = len(running)
        gpu_slots = sum(j.gpu_slots for j in running)
        return job_slots, gpu_slots

    def _check_can_admit(
        self,
        gpu_slots_required: int,
        job_slots_occupied: int,
        gpu_slots_occupied: int,
    ) -> None:
        if gpu_slots_required > self._settings.max_concurrent_gpu_worker_slots:
            raise JobConcurrencyLimitError(
                f"Unable to launch job that requires {gpu_slots_required} GPU "
                "workers, max GPU worker slots is "
                f"{self._settings.max_concurrent_gpu_worker_slots}")
        job_slots_full = (job_slots_occupied
                          >= self._settings.max_concurrent_jobs_total)
        gpu_slots_full = gpu_slots_required > 0 and (
            gpu_slots_occupied + gpu_slots_required
            > self._settings.max_concurrent_gpu_worker_slots)
        if job_slots_full or gpu_slots_full:
            raise JobConcurrencyLimitError(
                "Unable to start new job since maximum worker limit of "
                f"{self._settings.max_concurrent_jobs_total} has been reached. "
                f"Currently {job_slots_occupied} job slots and "
                f"{gpu_slots_occupied} GPU slots are in use; the new job "
                f"requires {gpu_slots_required} GPU workers.")

    # ------------------------------------------------------------------ #
    # Admission (single locked tx)
    # ------------------------------------------------------------------ #
    async def _admit_in_tx(self, tx: StateTx) -> list[JobID]:
        """Flip as many QUEUED jobs to RUNNING as capacity allows (FIFO)."""
        jobs = await self._list_open_top_level_jobs(tx)
        running = self._running(jobs)
        queue = self._queued_fifo(jobs)
        job_slots, gpu_slots = self._capacity(running)

        admitted: list[JobID] = []
        for job in queue:
            try:
                self._check_can_admit(job.gpu_slots, job_slots, gpu_slots)
            except JobConcurrencyLimitError:
                break
            await self._set_running(tx, job.id)
            job_slots += 1
            gpu_slots += job.gpu_slots
            admitted.append(job.id)
            log.info("Admitted job %s (job_slots=%d, gpu_slots=%d)", job.id,
                     job_slots, gpu_slots)
        return admitted

    async def _set_running(self, tx: StateTx, job_id: JobID) -> None:
        entry = await self._get_job_in_tx(tx, job_id)
        if entry is None:
            return
        now = now_utc()
        entry.execution.status = JobStatus.RUNNING
        entry.execution.start_time = now
        entry.execution.last_updated_at = now
        entry.execution.log = JobEventLogEntry(
            stage_name="Starting", last_updated_at=now,
            detail=("Job has finished waiting in queue and is now starting "
                    "to run."))
        await self._save_job_in_tx(tx, entry)

    async def _get_job_in_tx(self, tx: StateTx, job_id: JobID) -> Any | None:
        job_type = JobType.from_id(job_id)
        if job_type == JobType.TRAINING_JOB:
            row = await self._training_jobs.get(tx, str(TrainingJobID(job_id)))
            return row.payload if row else None
        if job_type == JobType.GENERATE_TRAIN_TABLE_JOB:
            row = await self._generate_train_table_jobs.get(
                tx, str(GenerateTrainTableJobID(job_id)))
            return row.payload if row else None
        if job_type == JobType.GENERATE_PREDICTION_TABLE_JOB:
            row = await self._generate_pred_table_jobs.get(
                tx, str(GeneratePredictionTableJobID(job_id)))
            return row.payload if row else None
        if job_type == JobType.BATCH_PREDICTION_JOB:
            row = await self._batch_prediction_jobs.get(
                tx, str(BatchPredictionJobID(job_id)))
            return row.payload if row else None
        raise AssertionError(f"Unsupported job type {job_type}")

    async def _save_job_in_tx(self, tx: StateTx, entry: Any) -> None:
        if isinstance(entry, BatchPredictionJobDBEntry):
            entry._sync_index_fields()
            await self._batch_prediction_jobs.upsert(tx, str(entry.id), entry)
            return
        repo = self._repo_for(entry)
        await repo.upsert(tx, str(entry.id), entry)

    def _repo_for(self, entry: Any) -> JsonRepository:
        if isinstance(entry, TrainingJobDBEntry):
            return self._training_jobs
        if isinstance(entry, GenerateTrainTableJobDBEntry):
            return self._generate_train_table_jobs
        if isinstance(entry, GeneratePredTableJobDBEntry):
            return self._generate_pred_table_jobs
        raise AssertionError(f"Unsupported job entry type {type(entry)}")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def submit_job(self, job_id: JobID) -> JobStatus:
        """Enqueue ``job_id`` (idempotent) and attempt admission.

        The job row must already exist with status ``QUEUED`` (created via
        ``MLJobService.create_*``).  Returns the resulting status (``QUEUED``
        if still waiting, ``RUNNING`` if admitted, or a terminal status if the
        job already finished).

        Raises :class:`JobConcurrencyLimitError` when the queue is full or the
        job can never fit (GPU demand exceeds the max).
        """
        async with self._state_db.write_tx() as tx:
            await self._lock(tx)
            entry = await self._get_job_in_tx(tx, job_id)
            if entry is None:
                raise KeyError(f"Job with ID '{job_id}' does not exist.")
            status = entry.execution.status
            if status == JobStatus.RUNNING:
                return JobStatus.RUNNING
            if status.is_terminal:
                return status
            if status != JobStatus.QUEUED:
                raise AssertionError(
                    f"Job {job_id} must be QUEUED to submit; got {status}")

            jobs = await self._list_open_top_level_jobs(tx)
            gpu_required = get_job_gpu_slots(entry)

            # Queue-length enforcement (only when queueing is enabled).
            # Count jobs *other than* the one being submitted, mirroring the
            # legacy "check before append" semantics (the row already exists
            # as QUEUED, so it would otherwise be double-counted).
            if self._settings.use_job_queue:
                queued = [j for j in self._queued_fifo(jobs) if j.id != job_id]
                if len(queued) >= self._settings.max_job_queue_length:
                    raise JobConcurrencyLimitError(
                        "Unable to start new job since the maximum job queue "
                        f"length of {self._settings.max_job_queue_length} has "
                        f"been reached ({len(queued)} jobs queued).")
                # Even when the GPU demand can never fit, surface that early.
                if (gpu_required
                        > self._settings.max_concurrent_gpu_worker_slots):
                    raise JobConcurrencyLimitError(
                        f"Unable to launch job that requires {gpu_required} "
                        "GPU workers, max GPU worker slots is "
                        f"{self._settings.max_concurrent_gpu_worker_slots}")
            else:
                # No queue: the job must be admissible right now or rejected.
                running = self._running(jobs)
                job_slots, gpu_slots = self._capacity(running)
                self._check_can_admit(gpu_required, job_slots, gpu_slots)

            # The job row is already QUEUED, so it is part of the FIFO queue.
            # Run an admission pass; it will admit this job if capacity allows
            # (and FIFO order permits).
            await self._admit_in_tx(tx)

            updated = await self._get_job_in_tx(tx, job_id)
            assert updated is not None
            return updated.execution.status

    async def release_job(self, job_id: JobID) -> list[JobID]:
        """Release capacity for a terminal/cancelled/removed job and admit next.

        Idempotent: the job's own status is *not* mutated here (terminalization
        owns that); this only re-runs admission so freed capacity is filled.
        Returns the list of newly admitted job IDs.
        """
        async with self._state_db.write_tx() as tx:
            await self._lock(tx)
            return await self._admit_in_tx(tx)

    async def drain(self) -> list[JobID]:
        """Re-run admission (idempotent). Used on startup and after changes."""
        async with self._state_db.write_tx() as tx:
            await self._lock(tx)
            return await self._admit_in_tx(tx)

    async def report_status(self) -> JobQueueStatus:
        """Derive a queue snapshot from the DB (debugging/tests)."""
        async with self._state_db.read_tx() as tx:
            jobs = await self._list_open_top_level_jobs(tx)
        running = self._running(jobs)
        queued = self._queued_fifo(jobs)
        job_slots, gpu_slots = self._capacity(running)
        return JobQueueStatus(
            job_slots_occupied=job_slots,
            gpu_slots_occupied=gpu_slots,
            running_jobs=[j.id for j in running],
            queued_jobs=[j.id for j in queued],
            max_concurrent_jobs_total=self._settings.max_concurrent_jobs_total,
            max_concurrent_gpu_worker_slots=(
                self._settings.max_concurrent_gpu_worker_slots),
            max_job_queue_length=self._settings.max_job_queue_length,
            use_job_queue=self._settings.use_job_queue,
        )


__all__ = [
    "StateDbJobQueue",
    "JobQueueStatus",
    "JobConcurrencyLimitError",
    "get_job_gpu_slots",
]
