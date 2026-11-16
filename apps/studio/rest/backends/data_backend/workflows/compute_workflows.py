# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
from typing import Optional

from temporalio import workflow
from temporalio.exceptions import FailureError

from io_utils import logging
from kumo.data.backend_cache import AsyncDataTaskStatus, TableDataTaskResult
from kumo.data.common import TableRefreshStage
from kumo.validation import ValidationResponse
from kumo.validation.validation import ValidationError
from rest.backends import (
    DATA_BACKEND_INGEST_TIMEOUT_HOURS,
    DATA_BACKEND_IO_TIMEOUT_HOURS,
)
from rest.backends.data_backend.activities import DataBackendActivities
from rest.backends.data_backend.validation import IngestionValidator
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.table_v2 import SnowflakeSourceTable, TableDefinition
from rest.constants import (
    COMPUTE_ACTIVITY_RETRY_POLICY,
    HEARTBEAT_TIMEOUT_COMPUTE_ACTIVITY,
)
from service_framework.deploy_env_config import deploy_env
from statedb.config.table import TableDataVersion
from statedb.config.timestamp import now_utc
from statedb.utils.errors import maybe_get_public_message

from ..activities import ingest_table_data_activity_v2

log = logging.getLogger(__name__)


@workflow.defn
class TableIngestionWorkflow:
    @workflow.run
    async def run(
        self,
        connector_spec: ConnectorExecutionRef,
        table: TableDefinition,
        job_id: str,
        data_version: TableDataVersion,
        graph_table_count: Optional[int] = None,
    ) -> TableDataTaskResult:
        return await self._run_workflow_impl(
            connector_spec,
            table,
            job_id,
            data_version,
            graph_table_count,
        )

    async def _run_workflow_impl(
        self,
        connector_spec: ConnectorExecutionRef,
        table: TableDefinition,
        job_id: str,
        data_version: TableDataVersion,
        graph_table_count: Optional[int],
    ) -> TableDataTaskResult:
        self._table = table
        self._data_version = data_version
        self._result = TableDataTaskResult(
            AsyncDataTaskStatus.IN_PROGRESS,
            ValidationResponse(),
            TableRefreshStage.INGEST_TABLE,
            error=None,
            data_version=data_version,
            created_at=now_utc(),
            updated_at=now_utc(),
        )

        await self._save_task_result()  # Mark ingestion task is IN PROGRESS

        await workflow.execute_activity(
            DataBackendActivities.create_table_cache_dir,
            args=[
                connector_spec.connector_id,
                table.get_source_table_name(),
                self._data_version,
                table.fingerprint(),
            ],
            schedule_to_close_timeout=timedelta(
                hours=DATA_BACKEND_IO_TIMEOUT_HOURS),
        )

        try:

            async def run_ingest():
                # No ``task_queue``: dispatch to ``workflow.info().task_queue``
                # so the activity runs on the workflow's own (thread) queue,
                # preserving any deploy-specific prefix.
                _, self._result.validation = await workflow.execute_activity(
                    ingest_table_data_activity_v2.ingest_table_data_activity_v2,
                    args=[
                        connector_spec,
                        table,
                        self._data_version,
                        job_id,
                        deploy_env(),
                        graph_table_count,
                    ],
                    schedule_to_close_timeout=timedelta(
                        hours=DATA_BACKEND_INGEST_TIMEOUT_HOURS),
                    heartbeat_timeout=HEARTBEAT_TIMEOUT_COMPUTE_ACTIVITY,
                    retry_policy=COMPUTE_ACTIVITY_RETRY_POLICY,
                )

            await run_ingest()
            # The snowflake connection may have failed on the snowflake end,
            # so reattempt the activity (not raising an error to actually fail
            # so parent handles the outcome of the retry as normal)
            if isinstance(table.source_table, SnowflakeSourceTable):
                val_errors = self._result.validation.errors
                validator = IngestionValidator(deploy_env())
                if len(val_errors) > 0:
                    for error in val_errors:
                        if validator.error_should_retry(error):
                            log.info(f"Ingestion of table {table}"
                                     f" with snowflake connector resulted in 0"
                                     " rows, retrying ingest activity.")
                            await run_ingest()
                            break

            self._result.status = AsyncDataTaskStatus.DONE
            self._result.updated_at = now_utc()
        except FailureError as e:
            self._result.status = AsyncDataTaskStatus.FAILED
            error_message = maybe_get_public_message(e)
            error_title = ("Ingestion failed for table "
                           f"{table.get_source_table_name()}")
            self._result.validation.errors.append(
                ValidationError(error_title, error_message))
            log.error('Ingestion failed (error_type=%s)', type(e).__name__)

        await workflow.execute_activity(
            DataBackendActivities.save_table_version_activity,
            args=[
                job_id,
                connector_spec.connector_id,
                table.get_source_table_name(),
                data_version,
                self._result.ok,
            ],
            schedule_to_close_timeout=timedelta(
                hours=DATA_BACKEND_IO_TIMEOUT_HOURS),
        )

        await self._save_task_result()
        return self._result

    async def _save_task_result(self):
        await workflow.execute_activity(
            DataBackendActivities.save_source_table_ingest_result,
            args=[
                self._table.source_table_id,
                self._data_version,
                self._result,
            ],
            schedule_to_close_timeout=timedelta(
                hours=DATA_BACKEND_IO_TIMEOUT_HOURS),
        )
