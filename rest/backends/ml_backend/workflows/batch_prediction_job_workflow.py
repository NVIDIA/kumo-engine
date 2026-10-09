# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Core workflow: run prediction job.
import asyncio
from datetime import timedelta
from typing import Optional

from kumo_core.types import PredictionArtifactType
from temporalio import workflow
from temporalio.client import WorkflowExecutionStatus
from temporalio.common import RetryPolicy
from temporalio.workflow import (
    ChildWorkflowCancellationType,
    ParentClosePolicy,
)

from io_utils import logging
from io_utils.exceptions import catch_and_log
from kumo.train.artifact_manifest import (
    PredictionProducerMismatchError,
    prediction_shard_producer_abi,
)
from pool.activities import SnapshotActivities
from pool.dataclasses import ResourceHint, SnapshotActivity
from pool.wf_coordinator import run_pooled_activity
from rest.backends import (
    DATA_BACKEND_BP_STATS_TIMEOUT_HOURS,
    TRAINING_AND_PREDICTION_TIMEOUT_HOURS,
)
from rest.backends.data_backend.activities import DataBackendActivities
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.ml_backend.activities import (
    JobOutputActivities,
    PredictionOutputFileMissingError,
    cleanup_databricks_volume_outputs,
    collect_prediction_job_output,
    finalize_prediction_job_output,
)
from rest.backends.ml_backend.settings import MLJobSettings
from rest.backends.ml_backend.workflows import ArtifactExportWorkflow
from rest.config.artifact_export import (
    ArtifactExportInput,
    DatabricksManagedVolumeStage,
)
from rest.config.batch_prediction_job import (
    BatchPredictionJobInput,
    BatchPredictionJobOutput,
    BatchPredictionJobResults,
    DatabricksVolumeShard,
    OutputArtifactStats,
    PredictionShardResult,
    direct_databricks_prediction_output_index,
)
from rest.config.ml_backend import JobState
from rest.config.prediction_output import DatabricksPredictionOutput
from rest.constants import DATA_BACKEND_TASK_QUEUE
from rest.settings import DataBackendSettings
from service_framework.deploy_env_config import deploy_env
from statedb.config._graph_key_norm import _graph_key_to_v2
from statedb.config.common import DataSnapshot, JobEventLogEntry, JobStatus
from statedb.utils.errors import (
    BatchPredictionJobError,
    PredictionOutputProcessingError,
    rethrow_or_wrap,
)

log = logging.getLogger(__name__)

# Selection materializes the accepted immutable shard inventory. Production
# outputs can be hundreds of GB, so this bound must cover the provider-side
# copy rather than the prior metadata-only finalizer.
PREDICTION_SELECTION_TIMEOUT_HOURS = 6
PREDICTION_ARTIFACT_OWNERSHIP_PATCH = 'bp-output-attempt-ownership-v1'
PREDICTION_SHARD_ABI_PATCH = 'bp-shard-producer-abi-v1'


def _aggregate_databricks_uploads(
    uploads: list[DatabricksVolumeShard],
    *,
    expected_shards: int,
    check_producer_abi: bool = True,
) -> tuple[
        dict[int, DatabricksManagedVolumeStage],
        dict[PredictionArtifactType, OutputArtifactStats],
]:
    by_output: dict[int, list[DatabricksVolumeShard]] = {}
    for upload in uploads:
        by_output.setdefault(upload.output_index, []).append(upload)

    stages: dict[int, DatabricksManagedVolumeStage] = {}
    stats: dict[PredictionArtifactType, OutputArtifactStats] = {}
    for output_index, output_uploads in by_output.items():
        if len(output_uploads) != expected_shards:
            raise RuntimeError(
                f'Databricks output {output_index} received '
                f'{len(output_uploads)} of {expected_shards} worker uploads')
        stage_paths = {upload.stage_path for upload in output_uploads}
        artifact_types = {upload.artifact_type for upload in output_uploads}
        column_counts = {upload.num_cols for upload in output_uploads}
        producer_abis = {upload.producer_abi for upload in output_uploads}
        if check_producer_abi and producer_abis != {
                prediction_shard_producer_abi(expected_shards)
        }:
            raise PredictionOutputProcessingError(
                message=str(PredictionProducerMismatchError()),
                non_retryable=True)
        if (len(stage_paths) != 1 or len(artifact_types) != 1
                or len(column_counts) != 1):
            raise RuntimeError(
                'Databricks prediction workers returned inconsistent '
                'managed-volume metadata')
        files = [
            file for upload in output_uploads for file in upload.staged_files
        ]
        if len(set(files)) != len(files):
            raise RuntimeError(
                'Databricks prediction workers returned duplicate staged '
                'files')
        stage_path = next(iter(stage_paths))
        artifact_type = next(iter(artifact_types))
        uploaded_bytes = sum(upload.uploaded_bytes for upload in output_uploads)
        slowest_worker_seconds = max(
            (upload.upload_seconds for upload in output_uploads),
            default=0.0,
        )
        log.info(
            'Databricks managed-volume producer phase complete: '
            'output_index=%s workers=%s files=%s bytes=%s '
            'slowest_worker_seconds=%.3f aggregate_mib_s=%.1f',
            output_index,
            len(output_uploads),
            len(files),
            uploaded_bytes,
            slowest_worker_seconds,
            (uploaded_bytes / slowest_worker_seconds /
             (1 << 20) if slowest_worker_seconds > 0 else 0.0),
        )
        stages[output_index] = DatabricksManagedVolumeStage(
            path=stage_path,
            files=files,
        )
        stats.setdefault(
            artifact_type,
            OutputArtifactStats(
                artifact_type=artifact_type,
                num_rows=sum(upload.num_rows for upload in output_uploads),
                num_cols=next(iter(column_counts)),
                size_bytes=sum(upload.size_bytes for upload in output_uploads),
                output_path=stage_path,
            ),
        )
    return stages, stats


async def _cleanup_databricks_uploads(
    job_input: BatchPredictionJobInput,
    uploads: list[DatabricksVolumeShard],
) -> None:
    if not uploads:
        return
    with catch_and_log():
        await workflow.execute_activity(
            cleanup_databricks_volume_outputs,
            args=[job_input, uploads],
            retry_policy=RetryPolicy(maximum_attempts=3),
            schedule_to_close_timeout=timedelta(hours=1),
        )


def _prediction_resource_hint(
    num_shards: int,
    max_durable_gpu_slots: Optional[int] = None,
) -> ResourceHint:
    min_gpu_slots = max(1, num_shards)
    if max_durable_gpu_slots is not None:
        min_gpu_slots = min(min_gpu_slots, max(1, max_durable_gpu_slots))
    return ResourceHint(min_gpu_slots=min_gpu_slots)


def _configured_max_durable_gpu_slots() -> Optional[int]:
    k8s_pool = getattr(deploy_env().data_compute, 'k8s_pool', None)
    node_types = getattr(k8s_pool, 'pool_diskgraph_node_types', None)
    if not node_types:
        return None
    return max(max(node_type.gpu_count, 1) for node_type in node_types.values())


@workflow.defn
class BatchPredictionJobWorkflow:
    @staticmethod
    def generate_wf_id(job_id: str) -> str:
        return f"batch-prediction-job-workflow-{job_id}"

    @workflow.run
    async def run(
        self,
        job_input: BatchPredictionJobInput,
        ml_job_settings: MLJobSettings,
        prediction_config_path: str,
        resource_hint: ResourceHint | None = None,
    ) -> BatchPredictionJobOutput:
        self._input = job_input
        self._settings = ml_job_settings
        try:
            return await self._run(job_input, ml_job_settings,
                                   prediction_config_path, resource_hint)
        except Exception as ex:
            log.exception("Batch prediction job failed")

            # Freeze the row terminal so finalize sees a frozen FAILED row with
            # the real failure_info, not a maskable non-terminal row.
            output = BatchPredictionJobOutput.failed(
                exception=ex, workflow_end_state=WorkflowExecutionStatus.FAILED)
            await self._save_output(output)
            rethrow_or_wrap(
                ex, BatchPredictionJobError,
                "Kumo encountered an error during batch prediction.")

    async def _run(
        self,
        job_input: BatchPredictionJobInput,
        ml_job_settings: MLJobSettings,
        prediction_config_path: str,
        resource_hint: ResourceHint | None = None,
    ) -> BatchPredictionJobOutput:
        use_artifact_ownership = workflow.patched(
            PREDICTION_ARTIFACT_OWNERSHIP_PATCH)

        # Pool path: dispatch SnapshotActivities.predict on the pool task
        # queue (singlehost daemon). The pool worker spawns kumo-pred as a
        # subprocess.
        log.info("Pool batch_prediction_job=%s training_job=%s pq=%s graph=%s",
                 job_input.id, job_input.training_job_id,
                 job_input.predictive_query.name,
                 job_input.predictive_query.graph.name)
        await self._save_output(
            BatchPredictionJobOutput(
                status=JobStatus.RUNNING,
                log=JobEventLogEntry.provisioning_resources()))
        self._launch_start_time = workflow.now()
        self._launch_finish_time = None
        # Canonical snapshot id — see TrainingJobWorkflow / pool/
        # snapshot_id.py for the rationale. By the time the BP job
        # workflow runs, the parent has populated graph_version_info
        # via graph refresh.
        graph_version_info = job_input.graph_version_info
        assert graph_version_info is not None, (
            'graph_version_info must be populated by graph refresh '
            'before BatchPredictionJobWorkflow runs')
        snapshot_id = DataSnapshot.GRAPH.canonicalize(
            f'{_graph_key_to_v2(job_input.graph_data_compute_input.graph_key)}@'
            f'{graph_version_info.graph_data_version}')
        # Sharded prediction: each shard runs its own
        # run_pooled_activity (distinct shard_index → distinct lease) so
        # the controller decides per-shard placement (one queue on
        # singlehost, N different daemons on multi-daemon k8s) and each
        # shard independently re-routes around a dead worker.
        num_shards = max(1, job_input.config.num_parallel_workers_limit)
        durable_resource_hint = (resource_hint or _prediction_resource_hint(
            num_shards,
            _configured_max_durable_gpu_slots(),
        ))
        await self._save_output(
            BatchPredictionJobOutput(
                status=JobStatus.RUNNING,
                log=JobEventLogEntry.loading_materialized_data()))

        # Each shard dispatch has a 21-day cap and five-minute heartbeat.
        # The coordinator can dispatch a shard up to three times, for a
        # theoretical 63 days of activity execution before the shard fails.
        async def _run_shard(shard_index: int) -> PredictionShardResult:
            activity_args = [
                snapshot_id,
                job_input.id,
                prediction_config_path,
                shard_index,
                num_shards,
                job_input,
            ]
            # Keep pre-patch histories byte-for-byte compatible. Existing
            # workflow executions omit this argument and pool workers retain
            # their legacy deterministic output path.
            if use_artifact_ownership:
                activity_args.extend([True, True])
            return await run_pooled_activity(
                snapshot_id=snapshot_id, activity=SnapshotActivity.PREDICT,
                activity_fn=SnapshotActivities.predict, args=activity_args,
                shard_index=shard_index, resource_hint=durable_resource_hint,
                schedule_to_close_timeout=timedelta(
                    hours=TRAINING_AND_PREDICTION_TIMEOUT_HOURS),
                heartbeat_timeout=timedelta(minutes=5))

        # ``return_exceptions=True`` so a shard raise doesn't leave
        # sibling shards running and consuming pool leases (the
        # default ``asyncio.gather`` propagates the first exception
        # but does NOT cancel the other coroutines). Sibling shards
        # run to natural completion (and release their leases via
        # ``run_pooled_activity``'s ``finally`` block) before we re-raise
        # the first exception below.
        shard_results = await asyncio.gather(
            *[_run_shard(i) for i in range(num_shards)], return_exceptions=True)
        successful_results: list[PredictionShardResult] = [
            result for result in shard_results
            if isinstance(result, PredictionShardResult)
        ]
        completed_uploads = [
            upload for result in successful_results
            for upload in result.databricks_uploads
        ]
        exceptions = [r for r in shard_results if isinstance(r, BaseException)]
        if exceptions:
            await _cleanup_databricks_uploads(job_input, completed_uploads)
            raise exceptions[0]
        failed_shards = [
            i for i, result in enumerate(successful_results)
            if result.state == JobState.FAILED
        ]
        final_job_state = (JobState.FAILED
                           if failed_shards else JobState.SUCCEEDED)
        # On failure the subprocess's exit reason is in the pool worker
        # log, not in workflow history. Surface a pointer so the stored
        # ``BatchPredictionJobOutput`` distinguishes "pool failure" from
        # "no information at all".
        err_msg = (f'kumo-pred subprocess exited non-zero for job '
                   f'{job_input.id} on shard(s) {failed_shards}; see '
                   f'pool worker logs for details' if failed_shards else None)
        stacktrace = None

        assert final_job_state in [JobState.FAILED, JobState.SUCCEEDED]
        if final_job_state == JobState.SUCCEEDED:
            try:
                databricks_stages, known_stats = (_aggregate_databricks_uploads(
                    completed_uploads,
                    expected_shards=num_shards,
                    # Replay-safe for pre-deploy histories.
                    check_producer_abi=workflow.patched(
                        PREDICTION_SHARD_ABI_PATCH),
                ))
                expected_databricks_outputs = {
                    index
                    for index, output in enumerate(job_input.config.outputs)
                    if isinstance(output, DatabricksPredictionOutput)
                }
                if set(databricks_stages) != expected_databricks_outputs:
                    raise RuntimeError(
                        'Databricks prediction workers did not return every '
                        'configured output stage')
            except BaseException:
                await _cleanup_databricks_uploads(job_input, completed_uploads)
                raise
            needs_durable_artifact = (not job_input.config.outputs or any(
                not isinstance(output, DatabricksPredictionOutput)
                for output in job_input.config.outputs))
            artifact_paths = {}
            selection = None
            if needs_durable_artifact:
                if use_artifact_ownership:
                    shard_manifests = [
                        result.artifact_manifest
                        for result in successful_results
                    ]
                    if any(value is None for value in shard_manifests):
                        raise RuntimeError(
                            'Prediction worker did not publish its artifact '
                            'manifest')
                    selection = await workflow.execute_activity(
                        finalize_prediction_job_output,
                        args=[
                            job_input,
                            prediction_config_path,
                            str(snapshot_id),
                            workflow.uuid4().hex,
                            shard_manifests,
                            num_shards,
                        ],
                        retry_policy=RetryPolicy(maximum_attempts=2),
                        schedule_to_close_timeout=timedelta(
                            hours=PREDICTION_SELECTION_TIMEOUT_HOURS),
                        heartbeat_timeout=timedelta(minutes=5),
                    )
                else:
                    artifact_paths = await workflow.execute_activity(
                        collect_prediction_job_output,
                        args=[job_input, prediction_config_path],
                        retry_policy=RetryPolicy(
                            non_retryable_error_types=[
                                PredictionOutputFileMissingError.__name__
                            ],
                            maximum_attempts=2,
                        ),
                        schedule_to_close_timeout=timedelta(minutes=15),
                        heartbeat_timeout=timedelta(minutes=5),
                    )

            artifact_input = ArtifactExportInput.from_bp_input(
                job_input,
                artifact_paths,
                prediction_selection=(selection if use_artifact_ownership
                                      and needs_durable_artifact else None),
                databricks_stages=databricks_stages,
                known_artifact_stats=known_stats,
            )
            try:
                direct_stats_output_index = (
                    direct_databricks_prediction_output_index(job_input.config)
                    if not needs_durable_artifact else None)
                if direct_stats_output_index is not None:
                    await workflow.execute_activity(
                        DataBackendActivities.start_bp_stats_activity,
                        args=[
                            DataBackendContext(
                                deploy_env(),
                                DataBackendSettings(
                                    disable_async_ingestion=False),
                            ),
                            job_input,
                            False,
                            direct_stats_output_index,
                            databricks_stages[direct_stats_output_index],
                        ],
                        task_queue=DATA_BACKEND_TASK_QUEUE,
                        schedule_to_close_timeout=timedelta(
                            hours=DATA_BACKEND_BP_STATS_TIMEOUT_HOURS),
                        retry_policy=RetryPolicy(maximum_attempts=3),
                    )
                artifact_stats = await workflow.execute_child_workflow(
                    ArtifactExportWorkflow.run,
                    args=[artifact_input],
                    id=f'{job_input.id}__{workflow.uuid4().hex[:6]}',
                    cancellation_type=(
                        ChildWorkflowCancellationType.TRY_CANCEL),
                    parent_close_policy=ParentClosePolicy.TERMINATE,
                )
            finally:
                await _cleanup_databricks_uploads(job_input, completed_uploads)
            key_cols: list[str] = [job_input.predictive_query.target
                                   ]  # type: ignore
            output = BatchPredictionJobOutput(
                status=JobStatus.DONE, log=JobEventLogEntry.done(),
                results=BatchPredictionJobResults(
                    key_cols=key_cols,
                    output_artifact_stats=artifact_stats,
                ))
            # The parent owns terminal publication after its required cleanup
            # and validation tail. Detailed column statistics are detached and
            # do not delay that publication.
            return output

        else:  # FAILED
            await _cleanup_databricks_uploads(job_input, completed_uploads)
            # Graceful shard failure: freeze the row terminal so finalize sees a
            # frozen FAILED row with the real failure_info.
            output = BatchPredictionJobOutput.failed(
                err_msg=err_msg, stacktrace=stacktrace,
                workflow_end_state=WorkflowExecutionStatus.COMPLETED)
            await self._save_output(output)
            return output

    async def _save_output(self, output: BatchPredictionJobOutput):
        await workflow.execute_activity(
            JobOutputActivities.save_batch_prediction_job_output,
            args=[self._input.id, output],
            schedule_to_close_timeout=timedelta(hours=10),
        )
