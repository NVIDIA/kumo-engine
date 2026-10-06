# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import timedelta

from temporalio import workflow

from io_utils import logging
from statedb.config.timestamp import now_utc

# isort: off
from rest.backends import DATA_BACKEND_MATERIALIZE_TIMEOUT_HOURS
from rest.backends.common.activities.io_activities import (
    save_validation_response, )
# isort: on
from rest.backends.ml_backend.activities.split import (
    SplitWorkflowInput,
    SplitWorkflowStatus,
    execute_split,
    generate_split_fpid,
    save_split_workflow_status,
)
from rest.backends.ml_backend.settings import MLJobSettings
from rest.constants import (
    COMPUTE_ACTIVITY_RETRY_POLICY,
    DATA_BACKEND_TASK_QUEUE,
    HEARTBEAT_TIMEOUT_COMPUTE_ACTIVITY,
    HEARTBEAT_TIMEOUT_TRAIN_TABLE_SPLIT,
)

# isort: off
from rest.backends.ml_backend.workflows.mixins import (
    JobWorkflowCancellationMixin, )
# isort: on
from kumo_core.types.schema import FPID

from statedb.config.common import JobEventLogEntry, JobStatus, SplitJobID

log = logging.getLogger(__name__)


@workflow.defn
class SplitTrainingTableWorkflow(JobWorkflowCancellationMixin):
    @workflow.run
    async def run(
        self,
        job_input: SplitWorkflowInput,
        ml_job_settings: MLJobSettings,
        home_dir: str,
        standalone_mode: bool,
    ) -> SplitWorkflowStatus:
        """Split a training table into multiple splits.

        Args:
            job_input (SplitWorkflowInput): The job input for the split.
            ml_job_settings (MLJobSettings): The ML job settings.
            home_dir (str): The home directory for the job.
            standalone_mode (bool): Whether the workflow is running in
                standalone mode (triggered independently of a training job).

        Returns:
            SplitWorkflowStatus: The status of the split.
        """
        self._job_input = job_input
        self._ml_job_settings = ml_job_settings
        self._standalone_mode = standalone_mode

        self._generate_train_table_job_id = (
            job_input.generate_train_table_job_id)
        self._generate_train_table_home_dir = home_dir
        self._split_config = job_input.split_config
        self._split_fpid = generate_split_fpid(self._split_config)

        self._job_id = SplitJobID.as_child_of(
            self._generate_train_table_job_id,
            self._split_fpid,
        )

        return await self.run_until_canceled(
            self._run(job_input, ml_job_settings, standalone_mode))

    def _get_split_dir(self, split_fpid: FPID) -> str:
        """Get the directory to where this split's data will be stored.

        Args:
            split_fpid (FPID): The FPID of the split (uniquely identifies the
                split).

        Returns:
            str: The directory to where this split's data will be stored.
        """
        return os.path.join(
            self._generate_train_table_home_dir,
            'splits',
            split_fpid,
        )

    def _get_status_path(self, split_fpid: FPID) -> str:
        """Get the path to the status file for this split.

        Args:
            split_fpid (FPID): The FPID of the split (uniquely identifies the
                split).

        Returns:
            str: The path to the status file for this split.
        """
        return os.path.join(self._get_split_dir(split_fpid), 'status.json')

    async def _save_status(self, status: SplitWorkflowStatus):
        """Save the status of the split.

        Args:
            status (SplitWorkflowStatus): The status of the split.
        """
        await workflow.execute_activity(
            save_split_workflow_status,
            args=[
                self._get_status_path(self._split_fpid),
                status,
            ],
            # Async status-write (save_json) — never used the process
            # pool; runs on the thread-executor workflow worker.
            task_queue=DATA_BACKEND_TASK_QUEUE,
            schedule_to_close_timeout=timedelta(
                hours=DATA_BACKEND_MATERIALIZE_TIMEOUT_HOURS),
            heartbeat_timeout=HEARTBEAT_TIMEOUT_COMPUTE_ACTIVITY,
            retry_policy=COMPUTE_ACTIVITY_RETRY_POLICY,
        )
        self.check_cancelled()

    async def _split_training_table(self) -> SplitWorkflowStatus:
        """Split the training table into multiple splits.

        Returns:
            SplitWorkflowStatus: The status of the split.
        """
        log.info("SplitTrainingTableWorkflow._split_training_table: Calling "
                 "execute_split activity")
        return await workflow.execute_activity(
            execute_split,
            args=[
                self._job_input,
                self._get_split_dir(self._split_fpid),
                self._generate_train_table_home_dir,
            ],
            # execute_split was de-globalized and moved to the data-backend
            # thread-pool worker; target its queue instead of the compute
            # process-pool queue.
            task_queue=DATA_BACKEND_TASK_QUEUE,
            schedule_to_close_timeout=timedelta(
                hours=DATA_BACKEND_MATERIALIZE_TIMEOUT_HOURS),
            heartbeat_timeout=HEARTBEAT_TIMEOUT_TRAIN_TABLE_SPLIT,
            retry_policy=COMPUTE_ACTIVITY_RETRY_POLICY,
        )

    async def _run(
        self,
        job_input: SplitWorkflowInput,
        ml_job_settings: MLJobSettings,
        standalone_mode: bool,
    ) -> SplitWorkflowStatus:
        """Run the split workflow and handle errors.

        Args:
            job_input (SplitWorkflowInput): The job input for the split.
            ml_job_settings (MLJobSettings): The ML job settings.
            standalone_mode (bool): Whether the workflow is running in
                standalone mode (triggered independently of a training job).

        Returns:
            SplitWorkflowStatus: The status of the split.
        """
        try:
            log.info(
                f"SplitTrainingTableWorkflow._run: received "
                f"standalone_mode={standalone_mode}, job_id={self._job_id}")

            result = await self._split_training_table()
        except BaseException:
            log.exception(f'SplitWorkflow failed for job id={self._job_id}')
            raise
        else:
            if result.validation.ok:
                status = JobStatus.DONE
                msg = 'Successfully completed splitting training table.'
            else:
                status = JobStatus.FAILED
                msg = ('Split training table failed due to critical '
                       'validation error.')
                await save_validation_response(
                    result.validation,
                    self._generate_train_table_home_dir,
                    self._job_id,
                )
            await self._save_status(
                SplitWorkflowStatus(
                    status=status,
                    log=JobEventLogEntry(
                        'Splitting Training Table',
                        now_utc(),
                        msg,
                    ),
                    validation=result.validation,
                ))
            return result
