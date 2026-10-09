# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import concurrent
import concurrent.futures
import time
import uuid
from functools import cached_property
from typing import List, Optional, Union
from urllib.parse import urlparse, urlunparse

import pandas as pd
from kumoapi.common import JobStatus
from kumoapi.jobs import (
    ArtifactExportRequest,
    AutoTrainerProgress,
    BatchPredictionJobSummary,
    BatchPredictionRequest,
    JobStatusReport,
    ModelEvaluationMetrics,
    PredictionProgress,
    TrainingJobRequest,
)
from kumoapi.model_plan import ModelPlan
from typing_extensions import override

from kumoai_studio import global_state
from kumoai_studio.artifact_export import (
    ArtifactExportJob,
    ArtifactExportResult,
    OutputConfig,
)
from kumoai_studio.backend import (
    PredictionBackend,
    TrainingBackend,
    cloud_backend,
)
from kumoai_studio.client.jobs import (
    BatchPredictionJobID,
    BatchPredictionJobResource,
    TrainingJobID,
    TrainingJobResource,
)
from kumoai_studio.databricks import to_db_table_name
from kumoai_studio.futures import KumoProgressFuture, create_future
from kumoai_studio.jobs import JobInterface
from kumoai_studio.pquery import (
    PredictionTable,
    PredictionTableJob,
    PredictiveQuery,
    TrainingTable,
    TrainingTableJob,
)
from kumoai_studio.trainer.util import (
    build_prediction_output_config,
    validate_output_arguments,
)


def _resolve_project_id(project_id: Optional[str]) -> str:
    if project_id is not None:
        return project_id
    from kumoai_studio import _require_project
    _require_project()
    assert global_state._project_id is not None
    return global_state._project_id


class TrainingJobResult:
    r"""Represents a completed training job.

    A :class:`TrainingJobResult` object can either be obtained by creating a
    :class:`~kumoai_studio.trainer.TrainingJob` object and calling the
    :meth:`~kumoai_studio.trainer.TrainingJob.result` method to await the job's
    completion, or by directly creating the object. The former approach is
    recommended, as it includes verification that the job finished succesfully.

    .. code-block:: python

        import kumoai_studio

        training_job = kumoai_studio.TrainingJob("trainingjob-...")

        # Wait for a training job's completion, and get its result:
        training_job_result = training_job.result()

        # Alternatively, create the result directly, but be sure that the job
        # is completed:
        training_job_result = kumoai_studio.TrainingJobResult("trainingjob-...")

        # Get associated objects:
        pquery = training_job_result.predictive_query
        training_table = training_job_result.training_table

        # Get holdout data:
        holdout_df = training_job_result.holdout_df()

    Example:
        >>> import kumoai_studio  # doctest: +SKIP
        >>> job_future = kumoai_studio.TrainingJob(id=...)  # doctest: +SKIP
        >>> job = job_future.result()  # doctest: +SKIP

    Args:
        job_id: ID of the completed training job.
        project_id: Exact project owning the job. ``None`` snapshots the
            active project.
    """
    def __init__(self, job_id: TrainingJobID,
                 project_id: Optional[str] = None) -> None:
        self.job_id = job_id
        self._kumo_project_id = _resolve_project_id(project_id)

        # A cached completed, finalized job resource:
        self._job_resource: Optional[TrainingJobResource] = None

    @property
    def kumo_project_id(self) -> str:
        """Return the immutable project owning this training result."""
        return self._kumo_project_id

    @property
    def id(self) -> TrainingJobID:
        r"""The unique ID of this training job."""
        return self.job_id

    @property
    def model_plan(self) -> ModelPlan:
        r"""Returns the model plan associated with this training job."""
        return self._get_job_resource(
            require_completed=False).config.model_plan

    @property
    def training_table(self) -> Union[TrainingTableJob, TrainingTable]:
        r"""Returns the training table associated with this training job,
        either as a :class:`~kumoai_studio.pquery.TrainingTable` or a
        :class:`~kumoai_studio.pquery.TrainingTableJob` depending on the status of
        the training table generation job.
        """
        training_table_job_id = self._get_job_resource(
            require_completed=False).config.train_table_job_id
        if training_table_job_id is None:
            raise RuntimeError(
                f"Unable to access the training table generation job ID for "
                f"job {self.job_id}. Did this job fail before generating its "
                f"training table?")
        fut = TrainingTableJob(training_table_job_id,
                               project_id=self._kumo_project_id)
        if fut.status().status == JobStatus.DONE:
            return fut.result()
        return fut

    @property
    def predictive_query(self) -> PredictiveQuery:
        r"""Returns the :class:`~kumoai_studio.pquery.PredictiveQuery` object that
        defined the training table for this training job.
        """
        return PredictiveQuery.load_from_training_job(
            self.job_id, project_id=self._kumo_project_id)

    @property
    def tracking_url(self) -> str:
        r"""Returns a tracking URL pointing to the UI display of this training
        job.
        """
        tracking_url = self._get_job_resource(
            require_completed=False).job_status_report.tracking_url
        return _rewrite_tracking_url(tracking_url)

    def metrics(self) -> ModelEvaluationMetrics:
        r"""Returns the metrics associated with this completed training job,
        or raises an exception if metrics cannot be obtained.
        """
        result = self._get_job_resource(require_completed=True).result
        assert result is not None
        return result.eval_metrics

    def holdout_url(self) -> str:
        r"""Return a safe reference to the model's holdout dataset.

        The server chooses a reference suitable for the active deployment,
        such as a short-lived signed download URL or a customer-facing
        warehouse path. Internal data-plane paths are never returned.
        """
        api: TrainingBackend = cloud_backend().training
        return api.holdout_data_url(self.job_id,
                                    project_id=self._kumo_project_id)

    def holdout_df(self) -> pd.DataFrame:
        r"""Reads the holdout dataset (parquet file) as pandas DataFrame.

        .. note::
            Please note that this function may be memory-intensive, depending
            on the size of your holdout dataframe. Please exercise caution.
        """
        holdout_url = self.holdout_url()

        if holdout_url.startswith("dbfs:"):
            raise ValueError(f"holdout_df is unsupported for "
                             f"Databricks UC Volume path {holdout_url}")

        return pd.read_parquet(holdout_url)

    def _get_job_resource(self,
                          require_completed: bool) -> TrainingJobResource:
        if self._job_resource:
            return self._job_resource

        try:
            api = cloud_backend().training
            resource: TrainingJobResource = api.get(
                self.job_id, project_id=self._kumo_project_id)
        except Exception as e:
            raise RuntimeError(
                f"Training job {self.job_id} was not found in the Kumo "
                f"database. Please contact Kumo for further assistance. "
            ) from e

        if not require_completed:
            return resource

        status = resource.job_status_report.status
        if not status.is_terminal:
            raise RuntimeError(
                f"Training job {self.job_id} has not yet completed. Please "
                f"create a `TrainingJob` class and await its completion "
                f"before attempting to view metrics.")

        if status != JobStatus.DONE:
            # Should never happen, the future will not resolve:
            raise ValueError(
                f"Training job {self.job_id} completed with status {status}, "
                f"and was therefore unable to produce metrics. Please "
                f"re-train the job until it successfully completes.")

        self._job_resource = resource
        return self._job_resource


class BatchPredictionJobResult:
    r"""Represents a completed batch prediction job.

    A :class:`BatchPredictionJobResult` object can either be obtained by
    creating a :class:`~kumoai_studio.trainer.BatchPredictionJob` object and calling
    the :meth:`~kumoai_studio.trainer.BatchPredictionJob.result` method to await the
    job's completion, or by directly creating the object. The former approach
    is recommended, as it includes verification that the job finished
    succesfully.

    .. code-block:: python

        import kumoai_studio

        prediction_job = kumoai_studio.BatchPredictionJob("bp-job-...")

        # Wait for a batch prediction job's completion, and get its result:
        prediction_job_result = prediction_job.result()

        # Alternatively, create the result directly, but be sure that the job
        # is completed:
        prediction_job_result = kumoai_studio.BatchPredictionJobResult("bp-job-...")

        # Get associated objects:
        prediction_table = prediction_job_result.prediction_table

        # Get prediction data (in-memory):
        predictions_df = training_job.predictions_df()

        # Export prediction data to any output connector:
        prediction_job_result.export(
            output_type = ...,
            output_connector = ...,
            output_table_name = ...,
        )

    Args:
        job_id: ID of the completed batch prediction job.
        project_id: Exact project owning the job. ``None`` snapshots the
            active project.
    """  # noqa: E501

    def __init__(self, job_id: BatchPredictionJobID,
                 project_id: Optional[str] = None) -> None:
        self.job_id = job_id
        self._kumo_project_id = _resolve_project_id(project_id)
        self._job_resource: Optional[BatchPredictionJobResource] = None

    @property
    def kumo_project_id(self) -> str:
        """Return the immutable project owning this prediction result."""
        return self._kumo_project_id

    @property
    def id(self) -> BatchPredictionJobID:
        r"""The unique ID of this batch prediction job."""
        return self.job_id

    @property
    def tracking_url(self) -> str:
        r"""Returns a tracking URL pointing to the UI display of this batch
        prediction job.
        """
        tracking_url = self._get_job_resource(
            require_completed=False).job_status_report.tracking_url
        return _rewrite_tracking_url(tracking_url)

    def summary(self) -> BatchPredictionJobSummary:
        r"""Returns summary statistics associated with the batch prediction
        job's output, or raises an exception if summary statistics cannot be
        obtained.
        """
        result = self._get_job_resource(require_completed=True).result
        if result is None:
            raise RuntimeError(
                f"Batch prediction job {self.job_id} has no summary because "
                f"it scored nothing: its output config asked only for "
                f"embeddings. Read those with embeddings_df(), or predict "
                f"again with 'predictions' among the output types.")
        return result

    @property
    def prediction_table(self) -> Union[PredictionTableJob, PredictionTable]:
        r"""Returns the prediction table associated with this prediction job,
        either as a :class:`~kumoai_studio.pquery.PredictionTable` or a
        :class:`~kumoai_studio.pquery.PredictionTableJob` depending on the status
        of the prediction table generation job.
        """
        prediction_table_job_id = self._get_job_resource(
            require_completed=False).config.pred_table_job_id
        if prediction_table_job_id is None:
            raise RuntimeError(
                f"Unable to access the prediction table generation job ID for "
                f"job {self.job_id}. Did this job fail before generating its "
                f"prediction table, or use a custom prediction table?")
        fut = PredictionTableJob(prediction_table_job_id,
                                 project_id=self._kumo_project_id)
        if fut.status().status == JobStatus.DONE:
            return fut.result()
        return fut

    def export(
        self,
        output_config: OutputConfig,
        non_blocking: bool = True,
        *,
        request_id: Optional[str] = None,
    ) -> Union['ArtifactExportJob', 'ArtifactExportResult']:
        r"""Export the prediction output or the embedding output to the
        specific output location.

        Args:
            output_config: The output configuration to be used.
            non_blocking: If ``True``, the method will return a future object
                `ArtifactExportJob` representing the export job.
                If ``False``, the method will block until the export job is
                complete and return `ArtifactExportResult`.
            request_id: Canonical UUID for idempotent submission. ``None``
                generates one. Reuse the same explicit value when retrying the
                same export request.
        """
        output_table_name = to_db_table_name(output_config.output_table_name)
        validate_output_arguments(
            (output_config.output_types),
            output_config.output_connector,
            output_table_name,
            output_paths=output_config.output_paths,
        )
        assert output_config.output_types is not None
        if len(output_config.output_types) > 1:
            raise ValueError(
                'Export one artifact per export() call. Use separate calls '
                'for predictions and embeddings.')
        prediction_output_config = build_prediction_output_config(
            list(output_config.output_types)[0],
            output_config.output_connector,
            output_table_name,
            output_config.output_metadata_fields,
            output_config,
        )

        connector = output_config.output_connector
        if (connector is not None
                and connector.kumo_project_id != self._kumo_project_id):
            raise ValueError('Output connector and prediction job belong to '
                             'different projects.')

        api = cloud_backend().artifact_export
        request = ArtifactExportRequest(
            request_id=(str(uuid.uuid4())
                        if request_id is None else request_id),
            job_id=self.id,
            prediction_output=prediction_output_config,
        )
        job_id = api.create(request, project_id=self._kumo_project_id)
        if non_blocking:
            return ArtifactExportJob(job_id, project_id=self._kumo_project_id)
        return ArtifactExportJob(job_id,
                                 project_id=self._kumo_project_id).attach()

    def predictions_urls(self) -> List[str]:
        r"""Return safe references to the generated predictions.

        The server chooses references suitable for the active deployment,
        such as short-lived signed download URLs or customer-facing Snowflake
        stage and Databricks Unity Catalog volume paths. Internal data-plane
        paths are never returned.
        """
        api: PredictionBackend = cloud_backend().prediction
        return api.get_batch_predictions_url(self.job_id,
                                             project_id=self._kumo_project_id)

    def predictions_df(self) -> pd.DataFrame:
        r"""Returns a :class:`~pandas.DataFrame` object representing the
        generated predictions.

        .. warning::

            This method will load the full prediction output into memory as a
            :class:`~pandas.DataFrame` object. If you are working on a machine
            with limited resources, please use
            :meth:`~kumoai_studio.trainer.BatchPredictionResult.predictions_urls`
            instead to download the data and perform analysis per-partition.
        """
        urls = self.predictions_urls()
        try:
            return pd.concat(pd.read_parquet(x) for x in urls)
        except Exception as e:
            raise ValueError(
                f"Could not create a Pandas DataFrame object from data paths "
                f"{urls}. Please construct the DataFrame manually.") from e

    def embeddings_urls(self) -> List[str]:
        r"""Return safe references to the generated embeddings.

        The server chooses references suitable for the active deployment,
        such as short-lived signed download URLs or customer-facing Snowflake
        stage and Databricks Unity Catalog volume paths. Internal data-plane
        paths are never returned.
        """
        api: PredictionBackend = cloud_backend().prediction
        return api.get_batch_embeddings_url(self.job_id,
                                            project_id=self._kumo_project_id)

    def embeddings_df(self) -> pd.DataFrame:
        r"""Returns a :class:`~pandas.DataFrame` object representing the
        generated embeddings.

        .. warning::

            This method will load the full prediction output into memory as a
            :class:`~pandas.DataFrame` object. If you are working on a machine
            with limited resources, please use
            :meth:`~kumoai_studio.trainer.BatchPredictionResult.embeddings_urls`
            instead to download the data and perform analysis per-partition.
        """
        urls = self.embeddings_urls()
        try:
            return pd.concat(pd.read_parquet(x) for x in urls)
        except Exception as e:
            raise ValueError(
                f"Could not create a Pandas DataFrame object from data paths "
                f"{urls}. Please construct the DataFrame manually.") from e

    def _get_job_resource(
            self, require_completed: bool) -> BatchPredictionJobResource:
        if self._job_resource:
            return self._job_resource

        try:
            api = cloud_backend().prediction
            resource: BatchPredictionJobResource = api.get(
                self.job_id, project_id=self._kumo_project_id)
        except Exception as e:
            raise RuntimeError(
                f"Batch prediction job {self.job_id} was not found in the "
                f"Kumo database. Please contact Kumo for further assistance. "
            ) from e

        if not require_completed:
            return resource

        status = resource.job_status_report.status
        if not status.is_terminal:
            raise RuntimeError(
                f"Batch prediction job {self.job_id} has not yet completed. "
                f"Please create a `BatchPredictionJob` class and await "
                "its completion before attempting to view stats.")

        if status != JobStatus.DONE:
            validation_resp = resource.job_status_report.validation_response
            validation_message = ""
            if validation_resp:
                validation_message = validation_resp.message()
            if len(validation_message) > 0:
                validation_message = f"Details: {validation_message}"

            raise ValueError(
                f"Batch prediction job {self.job_id} completed with status "
                f"{status}, and was therefore unable to produce metrics. "
                f"{validation_message}")

        self._job_resource = resource
        return resource


# Training Job Future #########################################################


class TrainingJob(JobInterface[TrainingJobID, TrainingJobRequest,
                               TrainingJobResource],
                  KumoProgressFuture[TrainingJobResult]):
    r"""Represents an in-progress training job.

    A :class:`TrainingJob` object can either be created as the result of
    :meth:`~kumoai_studio.trainer.Trainer.fit` with ``non_blocking=True``, or
    directly with a training job ID (*e.g.* of a job created asynchronously in
    a different environment).


    .. code-block:: python

        import kumoai_studio

        # See `Trainer` documentation:
        trainer = kumoai_studio.Trainer(...)

        # If a Trainer is `fit` in nonblocking mode, the response will be of
        # type `TrainingJob`:
        training_job = trainer.fit(..., non_blocking=True)

        # You can also construct a `TrainingJob` from a job ID, e.g. one that
        # is present in the Kumo Jobs page:
        training_job = kumoai_studio.TrainingJob("trainingjob-...")

        # Get the status of the job:
        print(training_job.status())

        # Attach to the job, and poll progress updates:
        training_job.attach()
        # Training: 70%|█████████    | [300s<90s, trial=4, train_loss=1.056, val_loss=0.682, val_mae=35.709, val_mse=7906.239, val_rmse=88.917

        # Cancel the job:
        training_job.cancel()

        # Wait for the job to complete, and return a `TrainingJobResult`:
        training_job.result()

    Args:
        job_id: The training job ID to await completion of.
        project_id: Exact project owning the job. ``None`` snapshots the
            active project; later active-project changes do not redirect it.
    """  # noqa

    @override
    @staticmethod
    def _api() -> TrainingBackend:
        return cloud_backend().training

    def __init__(self, job_id: TrainingJobID,
                 project_id: Optional[str] = None) -> None:
        self.job_id = job_id
        self._kumo_project_id = _resolve_project_id(project_id)

    @cached_property
    def _fut(self) -> concurrent.futures.Future:
        return create_future(_poll_training(self.job_id,
                                            self._kumo_project_id))

    @override
    @property
    def id(self) -> TrainingJobID:
        r"""The unique ID of this training job."""
        return self.job_id

    @property
    def kumo_project_id(self) -> str:
        """Return the immutable project owning this training job."""
        return self._kumo_project_id

    @override
    def result(self) -> TrainingJobResult:
        return self._fut.result()

    @override
    def future(self) -> 'concurrent.futures.Future[TrainingJobResult]':
        return self._fut

    @property
    def tracking_url(self) -> str:
        r"""Returns a tracking URL pointing to the UI that can be used to
        monitor the status of an ongoing or completed job.
        """
        return _rewrite_tracking_url(self.status().tracking_url)

    @override
    def _attach_internal(
        self,
        interval_s: float = 20.0,
    ) -> TrainingJobResult:
        r"""Allows a user to attach to a running training job, and view its
        progress inline.

        Args:
            interval_s (float): Time interval (seconds) between polls, minimum
                value allowed is 4 seconds.

        Example:
            >>> job_future = kumoai_studio.TrainingJob(job_id="...")  # doctest: +SKIP
            >>> job_future.attach()  # doctest: +SKIP
            Attaching to training job <id>. To track this job...
            Training: 70%|█████████    | [300s<90s, trial=4, train_loss=1.056, val_loss=0.682, val_mae=35.709, val_mse=7906.239, val_rmse=88.917
        """  # noqa
        assert interval_s >= 4.0
        print(f"Attaching to training job {self.job_id}. To track this job in "
              f"the Kumo UI, please visit {self.tracking_url}. To detach from "
              f"this job, please enter Ctrl+C: the job will continue to run, "
              f"and you can re-attach anytime by calling the `attach()` "
              f"method on the `TrainingJob` object. For example: "
              f"kumoai_studio.TrainingJob(\"{self.job_id}\").attach()")

        # TODO(manan): this is not perfect, the `asyncio.sleep` in the poller
        # may cause a "DONE" status to be printed for up to
        # interval_s*`timeout` seconds before the future resolves.
        # That's probably fine:
        if self.done():
            return self.result()

        # For every non-training stage, just show the stage and status:
        print("Waiting for job to start.")
        current_status = JobStatus.NOT_STARTED
        while current_status == JobStatus.NOT_STARTED:
            report = self.status()
            current_status = report.status
            current_stage = report.event_log[-1].stage_name
            time.sleep(interval_s)

        prev_stage = current_stage
        print(f"Current stage: {current_stage}. In progress...", end="",
              flush=True)
        while not self.done():
            # Print status of stage:
            if current_stage != prev_stage:
                print(" Done.")
                print(f"Current stage: {current_stage}. In progress...",
                      end="", flush=True)
            if current_stage == "Training":
                _time = self.progress().estimated_training_time
                if _time and _time != 0:
                    break
            time.sleep(interval_s)
            report = self.status()
            prev_stage = current_stage
            current_stage = report.event_log[-1].stage_name

        # We are not on Training:
        if self.done():
            return self.result()

        # We are training: print a progress bar
        progress = self.progress()
        bar_format = '{desc}: {percentage:3.0f}%|{bar}|{unit} '
        total = int(progress.estimated_training_time or 0)
        elapsed = int(progress.elapsed_training_time or 0)
        from tqdm.auto import tqdm
        pbar = tqdm(desc="Training", unit="% done", bar_format=bar_format,
                    total=total, dynamic_ncols=True)
        pbar.update(elapsed)

        while not self.done():
            progress = self.progress()
            trial_no = min(progress.completed_trials + 1,
                           progress.total_trials)

            if f'{max(trial_no-1, 0)}' in progress.trial_progress:
                trial_metrics = progress.trial_progress[
                    f'{max(trial_no-1, 0)}'].metrics
            elif f'{max(trial_no-2, 0)}' in progress.trial_progress:
                trial_metrics = progress.trial_progress[
                    f'{max(trial_no-2, 0)}'].metrics
            else:
                trial_metrics = {}

            # If we don't have metrics, wait until we do:
            if len(trial_metrics) == 0:
                continue

            # Show all metrics:
            # TODO(manan): only show tune metric, trial, epoch, and loss:
            last_epoch_metrics = trial_metrics[sorted(
                trial_metrics.keys())[-1]]
            train_metrics_s = ", ".join([
                f"{key_name}={key_val:.3f}" for key_name, key_val in
                last_epoch_metrics.train_metrics.items()
            ])
            val_metrics_s = ", ".join([
                f"{key_name}={key_val:.3f}" for key_name, key_val in
                last_epoch_metrics.validation_metrics.items()
            ])

            # Update numbers:
            delta = int((progress.elapsed_training_time or 0) - pbar.n)
            total = int(progress.estimated_training_time or 0)
            pbar.update(delta)
            pbar.total = total
            if pbar.n > pbar.total:
                pbar.total = pbar.n

            # NOTE we use `unit` here as a hack, instead of `set_postfix`,
            # since `tqdm` defaults to adding a comma before the postfix
            # (https://github.com/tqdm/tqdm/issues/712)
            pbar.unit = (f"[{pbar.n}s<{pbar.total-pbar.n}s, trial={trial_no}, "
                         f"{train_metrics_s}, {val_metrics_s}]")
            pbar.refresh()
            time.sleep(interval_s)
        assert pbar.total is not None
        pbar.update(pbar.total - pbar.n)
        pbar.close()

        # Future is done:
        return self.result()

    def progress(self) -> AutoTrainerProgress:
        r"""Returns the progress of an ongoing or completed training job."""
        return self._api().get_progress(self.job_id,
                                        project_id=self._kumo_project_id)

    @override
    def status(self) -> JobStatusReport:
        r"""Returns the status of a running training job."""
        return _get_training_status(self.job_id, self._kumo_project_id)

    def cancel(self) -> bool:
        r"""Cancels a running training job, and returns ``True`` if
        cancellation succeeded.

        Example:
            >>> job_future = kumoai_studio.TrainingJob(job_id="...")  # doctest: +SKIP
            >>> job_future.cancel()  # doctest: +SKIP
        """  # noqa
        return self._api().cancel(
            self.job_id, project_id=self._kumo_project_id).is_cancelled

    @override
    def load_config(self) -> TrainingJobRequest:
        r"""Load the full configuration for this training job.

        Returns:
            TrainingJobRequest: Complete configuration including model_plan,
            pquery_id, graph_snapshot_id, train_table_job_id, etc.
        """
        return self._api().get_config(self.job_id,
                                      project_id=self._kumo_project_id)


def _get_training_status(job_id: str, project_id: str) -> JobStatusReport:
    api = cloud_backend().training
    resource: TrainingJobResource = api.get(job_id, project_id=project_id)
    return resource.job_status_report


async def _poll_training(job_id: str, project_id: str) -> TrainingJobResult:
    # TODO(manan): make asynchronous natively with aiohttp:
    status = _get_training_status(job_id, project_id).status
    while not status.is_terminal:
        await asyncio.sleep(10)
        status = _get_training_status(job_id, project_id).status

    # TODO(manan, siyang): improve
    if status != JobStatus.DONE:
        api = cloud_backend().training
        job_resource = api.get(job_id, project_id=project_id)
        validation_resp = (job_resource.job_status_report.validation_response)

        validation_message = ""
        if validation_resp:
            validation_message = validation_resp.message()
        if len(validation_message) > 0:
            validation_message = f"Details: {validation_message}"

        raise RuntimeError(f"Training job {job_id} completed with job status "
                           f"{status}. {validation_message}")

    # TODO(manan): improve
    return TrainingJobResult(job_id=job_id, project_id=project_id)


# Batch Prediction Job Future #################################################


class BatchPredictionJob(JobInterface[BatchPredictionJobID,
                                      BatchPredictionRequest,
                                      BatchPredictionJobResource],
                         KumoProgressFuture[BatchPredictionJobResult]):
    r"""Represents an in-progress batch prediction job.

    A :class:`BatchPredictionJob` object can either be created as the
    result of :meth:`~kumoai_studio.trainer.Trainer.predict` with
    ``non_blocking=True``, or directly with a batch prediction job ID (*e.g.*
    of a job created asynchronously in a different environment).

    .. code-block:: python

        import kumoai_studio

        # See `Trainer` documentation:
        trainer = kumoai_studio.Trainer(...)

        # If a Trainer `predict` is called in nonblocking mode, the response
        # will be of type `BatchPredictionJob`:
        prediction_job = trainer.predict(..., non_blocking=True)

        # You can also construct a `BatchPredictionJob` from a job ID, e.g. one
        # that is present in the Kumo Jobs page:
        prediction_job = kumoai_studio.BatchPredictionJob("bp-job-...")

        # Get the status of the job:
        print(prediction_job.status())

        # Attach to the job, and poll progress updates:
        prediction_job.attach()
        # Attaching to batch prediction job <id>. To track this job...
        # Predicting (job_id=..., start=..., elapsed=..., status=...). Stage: ...

        # Cancel the job:
        prediction_job.cancel()

        # Wait for the job to complete, and return a `BatchPredictionJobResult`:
        prediction_job.result()

    Args:
        job_id: The batch prediction job ID to await completion of.
        project_id: Exact project owning the job. ``None`` snapshots the
            active project; later active-project changes do not redirect it.
    """  # noqa

    @override
    @staticmethod
    def _api() -> PredictionBackend:
        return cloud_backend().prediction

    def __init__(self, job_id: BatchPredictionJobID,
                 project_id: Optional[str] = None) -> None:
        self.job_id = job_id
        self._kumo_project_id = _resolve_project_id(project_id)

    @cached_property
    def _fut(self) -> concurrent.futures.Future:
        return create_future(
            _poll_batch_prediction(self.job_id, self._kumo_project_id))

    @override
    @property
    def id(self) -> BatchPredictionJobID:
        r"""The unique ID of this batch prediction job."""
        return self.job_id

    @property
    def kumo_project_id(self) -> str:
        """Return the immutable project owning this prediction job."""
        return self._kumo_project_id

    @override
    def result(self) -> BatchPredictionJobResult:
        return self._fut.result()

    @override
    def future(self) -> 'concurrent.futures.Future[BatchPredictionJobResult]':
        return self._fut

    @property
    def tracking_url(self) -> str:
        r"""Returns a tracking URL pointing to the UI that can be used to
        monitor the status of an ongoing or completed job.
        """
        return _rewrite_tracking_url(self.status().tracking_url)

    @override
    def _attach_internal(
        self,
        interval_s: float = 20.0,
    ) -> BatchPredictionJobResult:
        r"""Allows a user to attach to a running batch prediction job, and view
        its progress inline.

        Args:
            interval_s (float): Time interval (seconds) between polls, minimum
                value allowed is 4 seconds.

        """
        assert interval_s >= 4.0
        print(f"Attaching to batch prediction job {self.job_id}. To track "
              f"this job in the Kumo UI, please visit {self.tracking_url}. To "
              f"detach from this job, please enter Ctrl+C (the job will "
              f"continue to run, and you can re-attach anytime).")
        # TODO(manan): this is not perfect, the `asyncio.sleep` in the poller
        # may cause a "DONE" status to be printed for up to
        # interval_s*`timeout` seconds before the future resolves.
        # That's probably fine:
        if self.done():
            return self.result()

        print("Waiting for job to start.")
        current_status = JobStatus.NOT_STARTED
        while current_status == JobStatus.NOT_STARTED:
            report = self.status()
            current_status = report.status
            current_stage = report.event_log[-1].stage_name
            time.sleep(interval_s)

        prev_stage = current_stage
        print(f"Current stage: {current_stage}. In progress...", end="",
              flush=True)
        while not self.done():
            # Print status of stage:
            if current_stage != prev_stage:
                print(" Done.")
                print(f"Current stage: {current_stage}. In progress...",
                      end="", flush=True)
                if current_stage == "Predicting":
                    _time = self.progress().estimated_prediction_time
                    if _time and _time != 0:
                        break

            time.sleep(interval_s)
            report = self.status()
            prev_stage = current_stage
            current_stage = report.event_log[-1].stage_name

        # We are not on Batch Prediction:
        if self.done():
            return self.result()

        # We are predicting: print a progress bar
        bar_format = '{desc}: {percentage:3.0f}%|{bar} '
        total_iterations, elapsed = 0, 0
        from tqdm.auto import tqdm
        pbar = tqdm(desc="Predicting", unit="% done", bar_format=bar_format,
                    total=100, dynamic_ncols=True)
        pbar.update(elapsed)

        while not self.done():
            progress = self.progress()
            if progress is None:
                time.sleep(interval_s)
                continue
            total_iterations = progress.total_iterations
            completed_iterations = progress.completed_iterations
            pbar.update(
                (completed_iterations - elapsed) / total_iterations * 100)
            elapsed = completed_iterations
            elapsed_pct = completed_iterations / total_iterations
            pbar.refresh()
            time.sleep(interval_s)

        pbar.update(1.0 - elapsed_pct)
        pbar.close()

        # Future is done:
        return self.result()

    def progress(self) -> PredictionProgress:
        r"""Returns the progress of an ongoing or completed batch prediction
        job.
        """
        return self._api().get_progress(self.job_id,
                                        project_id=self._kumo_project_id)

    @override
    def status(self) -> JobStatusReport:
        r"""Returns the status of a running batch prediction job."""
        return _get_batch_prediction_status(self.job_id, self._kumo_project_id)

    def cancel(self) -> bool:
        r"""Cancels a running batch prediction job, and returns ``True`` if
        cancellation succeeded.
        """
        return self._api().cancel(
            self.job_id, project_id=self._kumo_project_id).is_cancelled

    @override
    def load_config(self) -> BatchPredictionRequest:
        r"""Load the full configuration for this batch prediction job.

        Returns:
            BatchPredictionRequest: Complete
            configuration including predict_options,
            outputs, model_training_job_id, etc.
        """
        return self._api().get_config(self.job_id,
                                      project_id=self._kumo_project_id)


def _get_batch_prediction_job(job_id: str,
                              project_id: str) -> BatchPredictionJobResource:
    api = cloud_backend().prediction
    return api.get(job_id, project_id=project_id)


def _get_batch_prediction_status(job_id: str,
                                 project_id: str) -> JobStatusReport:
    api = cloud_backend().prediction
    resource: BatchPredictionJobResource = api.get(job_id,
                                                   project_id=project_id)
    return resource.job_status_report


async def _poll_batch_prediction(job_id: str,
                                 project_id: str) -> BatchPredictionJobResult:
    # TODO(manan): make asynchronous natively with aiohttp:
    job_resource = _get_batch_prediction_job(job_id, project_id)
    status = job_resource.job_status_report.status
    while not status.is_terminal:
        await asyncio.sleep(10)
        job_resource = _get_batch_prediction_job(job_id, project_id)
        status = job_resource.job_status_report.status

    # TODO(manan, siyang): improve
    if status != JobStatus.DONE:
        validation_resp = job_resource.job_status_report.validation_response
        validation_message = ""
        if validation_resp:
            validation_message = validation_resp.message()
        if len(validation_message) > 0:
            validation_message = f"Details: {validation_message}"

        raise ValueError(
            f"Batch prediction job {job_id} completed with status "
            f"{status}, and was therefore unable to produce metrics. "
            f"{validation_message}")

    # TODO(manan): improve
    return BatchPredictionJobResult(job_id=job_id, project_id=project_id)


def _rewrite_tracking_url(tracking_url: str) -> str:
    r"""Rewrites tracking URL to account for deployment subdomains."""
    # TODO(manan): improve...
    if 'http' not in tracking_url:
        return tracking_url
    # A tracking URL points at the Kumo web UI, which only the cloud
    # deployment has; this is deliberately not a backend operation.
    parsed_base = urlparse(global_state.client._url)
    parsed_tracking = urlparse(tracking_url)
    tracking_url = urlunparse((
        parsed_base.scheme,
        parsed_base.netloc,
        parsed_tracking.path,
        parsed_tracking.params,
        parsed_tracking.query,
        parsed_tracking.fragment,
    ))
    return tracking_url
