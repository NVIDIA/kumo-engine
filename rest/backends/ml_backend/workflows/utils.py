# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from datetime import timedelta
from typing import Optional, Union

from temporalio import workflow
from temporalio.workflow import ActivityCancellationType

from io_utils.exceptions import catch_and_log
from rest.backends.ml_backend.activities import (
    MLBackendActivities,
    persist_job_inputs,
)
from rest.backends.ml_backend.settings import MLJobSettings
from rest.config.batch_prediction_job import BatchPredictionJobInput
from rest.config.ml_backend import TrainingJobInput
from statedb.config.common import JobID


async def enqueue_job(job_id: JobID):
    """
    Execute `add_job_activity` in workflow while ensuring the activity
    execution is complete or canceled, when receiving a cancellation signal
    during the execution (waiting for the job to be enqueued).
    """
    # Queue the job, note that we use `WAIT_CANCELLATION_COMPLETED`.
    enqueue_job_task = asyncio.create_task(
        workflow.execute_activity_method(
            MLBackendActivities.wait_in_job_queue,
            arg=job_id,
            schedule_to_close_timeout=timedelta(hours=24),
            cancellation_type=(
                ActivityCancellationType.WAIT_CANCELLATION_COMPLETED),
        ))
    try:
        # Wait for the job to be enqueued while don't propagate cancellation to
        # the activity.
        await asyncio.shield(enqueue_job_task)
    except asyncio.CancelledError:
        # The workflow receives a cancellation signal while the enqueue
        # task is still running
        enqueue_job_task.cancel(
            f'Workflow {workflow.info().workflow_id} received cancellation.')
        # Make sure the task completes
        await enqueue_job_task
        raise  # re-raise the CancelledError


async def dequeue_job(job_id: JobID):
    """Finalize a job: workflow-owned terminalization (safety-net) + Slack +
    queue-slot release.

    This replaces the external ``_wait_for_job_wf_completion`` watcher.  It is
    invoked from each parent workflow's ``finally`` block, so it runs on BOTH
    the success and failure (incl. cancellation) paths.  ``finalize_job`` is
    idempotent: if the workflow already wrote a terminal status it only
    releases the slot.
    """
    await workflow.execute_activity(
        MLBackendActivities.finalize_job,
        args=[job_id],
        schedule_to_close_timeout=timedelta(hours=24),
        cancellation_type=(
            ActivityCancellationType.WAIT_CANCELLATION_COMPLETED),
    )


async def init_job_generic(job_input: Union[TrainingJobInput,
                                            BatchPredictionJobInput],
                           ml_job_settings: MLJobSettings,
                           job_id: Optional[JobID] = None,
                           home_dir: Optional[str] = None):
    job_id = job_id or job_input.id
    # Registry metadata enriches the primary job row but is not required to
    # execute the job, so an exhausted retry should not fail the workflow.
    with catch_and_log():
        await workflow.execute_activity(
            MLBackendActivities.upsert_job_registry_metadata,
            args=[job_id, job_input, ml_job_settings],
            schedule_to_close_timeout=timedelta(minutes=10),
        )

    await workflow.execute_activity(
        persist_job_inputs, args=[job_input, ml_job_settings, home_dir],
        schedule_to_close_timeout=timedelta(minutes=10))
