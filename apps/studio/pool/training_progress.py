# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Training progress publication for pool-run kumo-auto jobs."""

import asyncio
from typing import Any

from kumo_core.evaluation.suite import MetricStore
from kumo_core.job_config import JobConfig

from kumo.home import HomeDirManager
from kumo.train.config_io import load_job_automl_config
from pool.dataclasses import JobID
from rest.backends.ml_backend.ml_job_output_store import MLJobOutputStore
from rest.backends.ml_backend.reporters import TrainingJobReporter
from rest.config.training_job import TrainingJobOutput, TrainingJobProgress
from statedb.config.common import JobEventLogEntry, JobStatus

TRAINING_PROGRESS_PUBLISH_TIMEOUT_SECS = 10


def training_progress_output(
    progress: TrainingJobProgress | None,
    training_job_cfg: Any,
) -> TrainingJobOutput | None:
    """Build the running job output represented by AutoTrainer progress."""
    if progress is None:
        return None

    if (progress.iterations_remaining == 0 and training_job_cfg.refit_trainval):
        log_entry = JobEventLogEntry.refit_trainval()
    elif progress.iterations_remaining == 0 and training_job_cfg.refit_full:
        log_entry = JobEventLogEntry.refit_full()
    else:
        log_entry = JobEventLogEntry.training()

    return TrainingJobOutput(
        status=JobStatus.RUNNING,
        log=log_entry,
        progress=progress,
    )


def _read_training_progress_output(
        job_config_path: str) -> TrainingJobOutput | None:
    """Read the running-job output from blob storage.

    Several blocking ``fsutil`` reads; run off the event loop (see
    ``publish_training_progress``).
    """
    job_config = JobConfig.load(job_config_path)
    home_dir_mgr = HomeDirManager(job_config.home_dir)
    log_dir = home_dir_mgr.automl_dir(job_config.task_name)
    automl_cfg = load_job_automl_config(job_config)
    tune_metric = automl_cfg.training_job.tune_metric
    assert isinstance(tune_metric, str)

    reporter = TrainingJobReporter(
        log_dir=log_dir,
        metric=tune_metric,
        metric_mode=MetricStore.tune_mode(tune_metric),
    )
    return training_progress_output(reporter.get_training_job_progress(),
                                    automl_cfg.training_job)


async def publish_training_progress(job_id: JobID, job_config_path: str,
                                    job_store: MLJobOutputStore) -> bool:
    """Publish training progress for a running pool ``kumo-auto`` job.

    ``job_store`` is the self-contained store owned by the pool worker's
    ``SnapshotActivities``. It opens one StateDB on its first write and reuses
    it across publishes (all on the worker's event loop); the store is closed
    by ``SnapshotActivities.stop``.

    Returns True when a running output was saved. If AutoTrainer has not
    emitted progress yet, there is nothing valid to publish.
    """
    # Run the blocking read on a worker thread so a slow/hung S3 read never
    # pins the event loop; on-loop it would block the wait_for timeout and the
    # heartbeat cancellation, hanging the train finalize on py3.14.
    output = await asyncio.wait_for(
        asyncio.to_thread(_read_training_progress_output, job_config_path),
        timeout=TRAINING_PROGRESS_PUBLISH_TIMEOUT_SECS,
    )
    if output is None:
        return False

    await asyncio.wait_for(
        job_store.save_training_job_output(job_id, output),
        timeout=TRAINING_PROGRESS_PUBLISH_TIMEOUT_SECS,
    )
    return True
