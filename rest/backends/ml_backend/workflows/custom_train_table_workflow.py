# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
from typing import Optional

from temporalio import workflow

from io_utils import kumo_dataclass
from pool.activities import SnapshotActivities
from pool.dataclasses import SnapshotActivity
from pool.wf_coordinator import run_pooled_activity
from rest.backends import (
    DATA_BACKEND_INGEST_TIMEOUT_HOURS,
    DATA_BACKEND_MATERIALIZE_TIMEOUT_HOURS,
)
from rest.backends.data_backend.activities import (
    ingest_custom_train_table,
    validate_weighted_train_table,
)
from rest.backends.ml_backend.activities import materialize_training_table
from rest.backends.ml_backend.settings import MLJobSettings
from rest.backends.ml_backend.validation import ValidationResponse
from rest.config.ml_backend import TrainingJobInput
from rest.config.training_job import CustomTrainingTable
from rest.constants import (
    COMPUTE_ACTIVITY_RETRY_POLICY,
    DATA_BACKEND_TASK_QUEUE,
    HEARTBEAT_TIMEOUT_COMPUTE_ACTIVITY,
    HEARTBEAT_TIMEOUT_TRAIN_TABLE_SPLIT,
)
from service_framework.deploy_env_config import deploy_env
from statedb.config._graph_key_norm import _graph_key_to_v2
from statedb.config.common import DataSnapshot, TrainingJobID


def _resolve_entity_time_col(predictive_query) -> Optional[str]:
    """Return the entity table's time_col name from the PQ V2 graph schema,
    or ``None`` if absent. Defensive: on lookup failure, synthesis falls back
    to task-shape defaults.
    """
    try:
        validated = predictive_query.validated_predictive_query
        entity_table_name = validated.entity_table
        for table in predictive_query.graph.tables:
            if (table.table_name == entity_table_name
                    or getattr(table, 'name', None) == entity_table_name):
                if table.time_col is not None:
                    return table.time_col.name
                return None
        return None
    except Exception:
        return None


@kumo_dataclass
class CustomTrainTableResult:
    """Result of the CustomTrainTableWorkflow.
    Args:
        train_table_path: Path to custom train table. This is the path to which
            the custom train table is ingested.
        train_table_materialization_path: Path to materialized custom train
            table.
        validation: Validation response.
    """
    train_table_path: str
    train_table_materialization_path: str
    validation: ValidationResponse


@workflow.defn
class CustomTrainTableWorkflow:
    """Workflow for creating and materializing a custom training table."""
    @staticmethod
    def generate_wf_id(job_id: TrainingJobID) -> str:
        """Generate a workflow ID for a particular job."""
        return f"custom-train-table-{job_id}"

    @workflow.run
    async def run(
        self,
        custom_train_table_spec: CustomTrainingTable,
        original_train_table_path: str,
        custom_train_table_path: str,
        job_input: TrainingJobInput,
        ml_job_settings: MLJobSettings,
        already_registered: bool = False,
    ) -> CustomTrainTableResult:
        """Ingest, validate, and materialize a custom training table.

        Args:
            custom_train_table_spec: Custom training table spec.
            original_train_table_path: Path to the original training table --
                or, when `already_registered`, to the registration's own copy
                of the rows this run trains on.
            custom_train_table_path: Destination for the custom training
                table. Unused when `already_registered`: there is nothing to
                ingest into it.
            job_input: Training job input.
            ml_job_settings: ML job settings.
            already_registered: True when `custom_train_table_spec` names a
                table `register_custom_table` already copied into Kumo
                storage (`GenerateTrainTableResult.registered`), rather than
                one supplied against an already-generated job. Ingesting it
                again here would re-read the caller's original source at
                every training run, so this reads the registered copy back
                instead -- unaffected by a source deleted or rewritten since
                registration.
        """
        validation_response = ValidationResponse()

        task_type = job_input.spec.task_type
        assert task_type is not None

        if already_registered:
            new_train_table_path = original_train_table_path
        else:
            # Drives sentinel-TIMESTAMP synthesis in the ingest activity.
            predictive_query = job_input.spec.predictive_query
            query_type = predictive_query.query_type
            list_distinct_target_is_array = bool(
                predictive_query.list_distinct_target_is_array)

            # When the entity table has its own time_col, the sentinel
            # synthesizer skips static node-pred since the customer's
            # parquet already carries real per-row time.
            entity_time_col = _resolve_entity_time_col(predictive_query)

            new_train_table_path, response = (await workflow.execute_activity(
                ingest_custom_train_table,
                args=[
                    custom_train_table_spec,
                    custom_train_table_path,
                    deploy_env(),
                    job_input.id,
                    task_type,
                    query_type,
                    list_distinct_target_is_array,
                    entity_time_col,
                ],
                task_queue=DATA_BACKEND_TASK_QUEUE,
                schedule_to_close_timeout=timedelta(
                    hours=DATA_BACKEND_INGEST_TIMEOUT_HOURS),
                retry_policy=COMPUTE_ACTIVITY_RETRY_POLICY,
            ))
            validation_response.merge(response)
            if not validation_response.ok:
                return CustomTrainTableResult("", "", validation_response)
            validation_response.merge(await workflow.execute_activity(
                validate_weighted_train_table,
                args=[
                    original_train_table_path,
                    new_train_table_path,
                    job_input.weight_col_name,
                    deploy_env(),
                    job_input.id,
                    task_type,
                    custom_train_table_spec.split_col_name,
                ],
                task_queue=DATA_BACKEND_TASK_QUEUE,
                schedule_to_close_timeout=timedelta(
                    hours=DATA_BACKEND_INGEST_TIMEOUT_HOURS),
                retry_policy=COMPUTE_ACTIVITY_RETRY_POLICY,
            ))
            if not validation_response.ok:
                return CustomTrainTableResult("", "", validation_response)

        # Build and persist the custom TT-NG bundle while the pool keeps the
        # parent graph index resident. The following materializer only validates
        # that durable bundle, so it does not need a snapshot lease.
        assert job_input.graph_data_compute_input is not None
        assert job_input.graph_version_info is not None
        snapshot_id = DataSnapshot.GRAPH.canonicalize(
            f'{_graph_key_to_v2(job_input.graph_data_compute_input.graph_key)}'
            f'@{job_input.graph_version_info.graph_data_version}')
        new_train_table_path = await run_pooled_activity(
            snapshot_id=snapshot_id,
            activity=SnapshotActivity.GEN_CUSTOM_TT,
            activity_fn=SnapshotActivities.build_ttng_custom_train_table_bundle,
            args=[
                snapshot_id,
                job_input,
                ml_job_settings,
                new_train_table_path,
                job_input.weight_col_name,
            ],
            schedule_to_close_timeout=timedelta(
                hours=DATA_BACKEND_MATERIALIZE_TIMEOUT_HOURS),
            heartbeat_timeout=HEARTBEAT_TIMEOUT_COMPUTE_ACTIVITY,
            retry_policy=COMPUTE_ACTIVITY_RETRY_POLICY,
        )

        materialized_train_table_data_path, response = (
            await workflow.execute_activity(
                materialize_training_table,
                args=[
                    job_input,
                    ml_job_settings,
                    new_train_table_path,
                    True,
                    job_input.weight_col_name,
                    True,
                ],
                task_queue=DATA_BACKEND_TASK_QUEUE,
                schedule_to_close_timeout=timedelta(
                    hours=DATA_BACKEND_MATERIALIZE_TIMEOUT_HOURS),
                heartbeat_timeout=HEARTBEAT_TIMEOUT_TRAIN_TABLE_SPLIT,
                retry_policy=COMPUTE_ACTIVITY_RETRY_POLICY,
            ))
        validation_response.merge(response)

        if not validation_response.ok:
            return CustomTrainTableResult("", "", validation_response)

        return CustomTrainTableResult(
            new_train_table_path,
            materialized_train_table_data_path,
            validation_response,
        )
