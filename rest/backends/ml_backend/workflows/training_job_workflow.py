# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

from io_utils import logging
from pool.activities import SnapshotActivities
from pool.dataclasses import SnapshotActivity
from pool.wf_coordinator import run_pooled_activity
from rest.backends import TRAINING_AND_PREDICTION_TIMEOUT_HOURS
from rest.backends.ml_backend.activities import (
    BestTrainerCfgFileMissingError,
    JobOutputActivities,
    PollJobResult,
    TrainingJobActivities,
)
from rest.backends.ml_backend.settings import MLJobSettings
from rest.config.ml_backend import JobState, TrainingJobInput
from rest.config.training_job import TrainingJobOutput
from statedb.config._graph_key_norm import _graph_key_to_v2
from statedb.config.common import DataSnapshot, JobEventLogEntry, JobStatus

log = logging.getLogger(__name__)


# Core workflow: run training job.
@workflow.defn
class TrainingJobWorkflow:
    @staticmethod
    def generate_wf_id(job_id: str) -> str:
        return f"training-job-workflow-{job_id}"

    async def _save_output(self, output: TrainingJobOutput):
        await workflow.execute_activity(
            JobOutputActivities.save_training_job_output,
            args=[self._input.id, output],
            schedule_to_close_timeout=timedelta(minutes=10),
        )

    @workflow.run
    async def run(
        self,
        job_input: TrainingJobInput,
        ml_job_settings: MLJobSettings,
        job_config_path: str,
    ):
        self._input = job_input
        self._settings = ml_job_settings
        graph_compute_input = job_input.graph_data_compute_input
        graph_version_info = job_input.graph_version_info

        # Pool path: dispatch SnapshotActivities.train on the pool task
        # queue. The pool worker subprocess (singlehost daemon) spawns
        # kumo-auto in-process.
        poll_result = await self._run_via_pool(job_input, job_config_path)

        assert poll_result.state in [JobState.FAILED, JobState.SUCCEEDED]
        if poll_result.state == JobState.SUCCEEDED:
            snapshot_id = DataSnapshot.GRAPH.canonicalize(
                f'{_graph_key_to_v2(graph_compute_input.graph_key)}@'
                f'{graph_version_info.graph_data_version}')
            output = await workflow.execute_activity(
                TrainingJobActivities.postprocess_job_output,
                args=[
                    job_input.id,
                    job_config_path,
                    graph_compute_input.connectors,
                    snapshot_id,
                ],
                retry_policy=RetryPolicy(non_retryable_error_types=[
                    BestTrainerCfgFileMissingError.__name__
                ]),
                schedule_to_close_timeout=timedelta(minutes=100),
            )
            # postprocess_job_output computes the result in the (CPU-heavy)
            # activity but no longer persists it inline; the StateDB write is
            # hoisted here onto the worker loop via save_training_job_output.
            # Note: warnings are saved by postprocess_job_output via ValidationResponse.save()
            await self._save_output(output)
        else:  # FAILED
            # log_failed_job also saves warnings if present
            await workflow.execute_activity(
                TrainingJobActivities.log_failed_job,
                args=[job_input.id, job_config_path, poll_result.warning_msg],
                schedule_to_close_timeout=timedelta(minutes=10))

            output = TrainingJobOutput.failed(err_msg=poll_result.err_msg,
                                              stacktrace=poll_result.stacktrace)
            await self._save_output(output)

    async def _run_via_pool(
        self,
        job_input: TrainingJobInput,
        job_config_path: str,
    ) -> PollJobResult:
        """Pool path: dispatch SnapshotActivities.train on the pool task
        queue (singlehost daemon) and block until it returns. The pool
        worker spawns ``kumo-auto`` as a subprocess. Each dispatch has the
        shared 21-day cap and five-minute heartbeat; the coordinator
        allows up to 63 days of activity execution across three attempts, and
        retries do not resume checkpoints."""
        log.info("Pool training_job=%s predictive_query=%s graph=%s",
                 self._input.id, self._input.spec.predictive_query.name,
                 self._input.spec.predictive_query.graph.name)

        if self._settings.raise_fake_error:
            output = TrainingJobOutput.failed(
                err_msg="Encountered fake error",
                exception=Exception("Fake error was raised during training"))
            await self._save_output(output)
            return PollJobResult(state=JobState.FAILED, err_msg="fake error")

        await self._save_output(
            TrainingJobOutput(status=JobStatus.RUNNING,
                              log=JobEventLogEntry.provisioning_resources()))

        # By the time TrainingJobWorkflow runs, refresh has populated
        # graph_version_info on the parent's job_input. The canonical
        # GraphSnapshotID is the lease identity — many concurrent
        # train/predict jobs over the same (graph, version) all hold
        # leases on the *same* snapshot_id and share warm placement.
        graph_compute_input = job_input.graph_data_compute_input
        graph_version_info = job_input.graph_version_info
        assert graph_compute_input is not None, (
            'graph_data_compute_input is required for pool training')
        assert graph_version_info is not None, (
            'graph_version_info must be populated by graph refresh before '
            'TrainingJobWorkflow runs')
        snapshot_id = DataSnapshot.GRAPH.canonicalize(
            f'{_graph_key_to_v2(graph_compute_input.graph_key)}@'
            f'{graph_version_info.graph_data_version}')
        await self._save_output(
            TrainingJobOutput(status=JobStatus.RUNNING,
                              log=JobEventLogEntry.loading_materialized_data()))
        state: JobState = await run_pooled_activity(
            snapshot_id=snapshot_id, activity=SnapshotActivity.TRAIN,
            activity_fn=SnapshotActivities.train, args=[
                snapshot_id,
                job_input.id,
                job_config_path,
            ], schedule_to_close_timeout=timedelta(
                hours=TRAINING_AND_PREDICTION_TIMEOUT_HOURS),
            heartbeat_timeout=timedelta(minutes=5))

        # On failure the subprocess's exit reason is in the pool worker
        # log, not in workflow history. Surface a pointer in err_msg so
        # the stored ``TrainingJobOutput`` distinguishes "pool failure"
        # from "no information at all".
        err_msg = (f'kumo-auto subprocess exited non-zero for job '
                   f'{job_input.id}; see pool worker logs for details'
                   if state == JobState.FAILED else None)
        return PollJobResult(state=state, err_msg=err_msg)
