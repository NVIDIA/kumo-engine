# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta

from kumo_core.types import PredictionArtifactType
from temporalio import workflow
from temporalio.common import RetryPolicy

from rest.backends import DATA_BACKEND_BP_STATS_TIMEOUT_HOURS
from rest.backends.data_backend.activities import DataBackendActivities
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.ml_backend.settings import MLJobSettings
from rest.config.batch_prediction_job import (
    BatchPredictionJobInput,
    requested_prediction_artifact_types,
)
from rest.constants import DATA_BACKEND_TASK_QUEUE
from rest.settings import DataBackendSettings
from service_framework.deploy_env_config import deploy_env

# BP column statistics are best-effort MLOps data computed after durable BP
# output is published. Bound retries keep persistent failures visible without
# tying them to the already-terminal BP job.
_BP_STATS_RETRY = RetryPolicy(maximum_attempts=3)


@workflow.defn
class BatchPredictionStatsWorkflow:
    """Compute batch-prediction column stats (raw + transformed outputs).

    Detached child of the main BP workflow. Durable outputs use DiskGraph;
    Databricks-only outputs have already persisted sampled stats from their
    managed-volume stage. Failures remain observable on this workflow without
    changing the terminal BP result.

    The DG cutover (PR #28324) deleted the legacy Spark ColStatsWorkflow
    plumbing; this workflow is the replacement entry point.
    """
    @workflow.run
    async def run(
        self,
        job_input: BatchPredictionJobInput,
        ml_job_settings: MLJobSettings,
    ):
        if (PredictionArtifactType.PREDICTIONS
                not in requested_prediction_artifact_types(job_input.config)):
            return

        data_backend_ctx = DataBackendContext(
            deploy_env(), DataBackendSettings(disable_async_ingestion=False))

        # Raw predictions
        await workflow.execute_activity(
            DataBackendActivities.start_bp_stats_activity,
            args=[data_backend_ctx, job_input, False],
            task_queue=DATA_BACKEND_TASK_QUEUE,
            schedule_to_close_timeout=timedelta(
                hours=DATA_BACKEND_BP_STATS_TIMEOUT_HOURS),
            retry_policy=_BP_STATS_RETRY,
        )

        # Transformed predictions (where applicable)
        # TODO(frg100:BPT): Deprecate transformation handling.
        if job_input.transformation is not None:
            await workflow.execute_activity(
                DataBackendActivities.start_bp_stats_activity,
                args=[data_backend_ctx, job_input, True],
                task_queue=DATA_BACKEND_TASK_QUEUE,
                schedule_to_close_timeout=timedelta(
                    hours=DATA_BACKEND_BP_STATS_TIMEOUT_HOURS),
                retry_policy=_BP_STATS_RETRY,
            )
