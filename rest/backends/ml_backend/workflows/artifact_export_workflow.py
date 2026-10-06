# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.workflow import ActivityCancellationType

from io_utils import logging
from rest.backends.ml_backend.activities import (
    get_artifact_stats,
    write_artifacts_to_customer,
)
from rest.config.artifact_export import ArtifactExportInput, NamedOutputConfig
from rest.config.batch_prediction_job import OutputArtifactStats
from rest.config.jobs import WriteMode
from rest.constants import DISABLE_RETRY_POLICY
from service_framework.env_var import EnvVar

log = logging.getLogger(__name__)

# Env variable used toEnvVarol BP artifact export deadline.
BP_EXPORT_BANDWIDTH = EnvVar(
    'BP_EXPORT_BANDWIDTH',
    type=int,
    required=False,
    default_value=(2 << 20),  # 2MB/sec
    description="""
Expected overall bandwidth (size bytes per seconds, decompressed data) for
exporting batch prediction outputs to customer specified output location.
""")


@workflow.defn
class ArtifactExportWorkflow:
    @workflow.run
    async def run(
        self,
        input: ArtifactExportInput,
    ) -> list[OutputArtifactStats]:
        return await self._run(input)

    async def _run(
        self,
        input: ArtifactExportInput,
    ) -> list[OutputArtifactStats]:
        # Now export the artifacts to output locations:
        # first, check artifact stats to dynamically determine export
        # deadline.
        artifact_stats = dict(input.known_artifact_stats)
        missing_artifacts = set(input.artifact_types) - set(artifact_stats)
        missing_paths = {
            artifact_type: path
            for artifact_type, path in input.artifact_paths.items()
            if artifact_type in missing_artifacts
        }
        if missing_artifacts:
            stats_args = [missing_paths]
            if input.prediction_selection is not None:
                stats_args.extend([
                    input.prediction_selection,
                    list(input.prediction_artifact_types),
                ])
            artifact_stats.update(await workflow.execute_activity(
                get_artifact_stats,
                args=stats_args,
                schedule_to_close_timeout=timedelta(hours=1),
            ))

        export_timeout = timedelta(hours=2)  # default extra timeout
        for artifact_type in input.artifact_types:
            stats = artifact_stats[artifact_type]
            destination_count = len(input.output_config.get(artifact_type, []))
            # Destinations are processed sequentially by one activity.
            export_timeout += timedelta(
                seconds=(stats.size_bytes * destination_count /
                         BP_EXPORT_BANDWIDTH.parse_value()))

        # Allow retry except for APPEND write mode.
        export_retry_allowed = True
        for configs in input.output_config.values():
            if any(
                    isinstance(config, NamedOutputConfig)
                    and config.out_table.write_mode == WriteMode.APPEND
                    for config in configs):
                export_retry_allowed = False
                break

        retry_policy = (RetryPolicy(
            maximum_attempts=3, initial_interval=timedelta(
                minutes=2)) if export_retry_allowed else DISABLE_RETRY_POLICY)

        output = await workflow.execute_activity(
            write_artifacts_to_customer,
            args=[input, artifact_stats],
            retry_policy=retry_policy,
            heartbeat_timeout=timedelta(seconds=30),
            cancellation_type=(
                ActivityCancellationType.WAIT_CANCELLATION_COMPLETED),
            schedule_to_close_timeout=(export_timeout *
                                       3 if export_retry_allowed else
                                       export_timeout),
        )

        if len(output) == 0:
            # Save stats for every artifact produced by a local-download-only
            # BP job. Such jobs can request embeddings without predictions.
            output = []
            for artifact_type in input.artifact_types:
                stats = artifact_stats[artifact_type]
                artifact_name = artifact_type.value.lower()
                stats.output_path = (
                    f'N/A (customer can download {artifact_name})')
                output.append(stats)
        return output
