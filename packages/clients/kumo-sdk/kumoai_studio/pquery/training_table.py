# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
import uuid
from concurrent.futures import Future
from functools import cached_property
from typing import List, Optional, Protocol, Tuple, Union

import pandas as pd
from kumoapi.common import JobStatus
from kumoapi.jobs import (
    ArtifactExportRequest,
    CustomTrainingTable,
    FileTrainingTableOutputConfig,
    GenerateTrainTableJobResource,
    GenerateTrainTableRequest,
    JobStatusReport,
    NamedTrainingTableOutputConfig,
    SourceTableType,
    TrainingTableSpec,
    WriteMode,
)
from kumoapi.model_plan import (
    InferredType,
    MissingType,
    TrainingTableGenerationPlan,
)
from kumoapi.source_table import LocalFileSourceTable, S3SourceTable
from typing_extensions import override

from kumoai_studio import global_state
from kumoai_studio.artifact_export import (
    ArtifactExportJob,
    ArtifactExportResult,
    TrainingTableExportConfig,
)
from kumoai_studio.backend import TrainTableBackend, cloud_backend
from kumoai_studio.client.jobs import GenerateTrainTableJobID
from kumoai_studio.connector import (
    ADLSConnector,
    DatabricksConnector,
    GCSConnector,
    S3Connector,
    SourceTable,
    SQLSourceTable,
)
from kumoai_studio.databricks import to_db_table_name
from kumoai_studio.formatting import pretty_print_error_details
from kumoai_studio.futures import KumoProgressFuture, create_future
from kumoai_studio.jobs import JobInterface


class _SavedQuery(Protocol):
    r"""What registering a supplied table needs of a predictive query.

    Named structurally rather than imported: `kumoai_studio.pquery.predictive_query`
    imports :class:`TrainingTable` from this module, so importing the class
    back is a cycle. These two members are the whole of the dependency, and
    saying so keeps the contract honest -- a caller may pass anything that
    answers them.
    """
    @property
    def kumo_project_id(self) -> str:
        ...

    def save(self, name: Optional[str] = None) -> str:
        ...

    #: Set to the registered table, so the query answers `train_table` with
    #: the registered rows, which is what a caller hands `Trainer.fit`.
    #:
    #: Declared as widely as a query actually holds it -- unset, or a job that
    #: has not finished -- because a protocol attribute is invariant: narrowing
    #: it to what *this* path assigns would refuse every real query.
    _train_table: Optional[Union[TrainingTable, 'TrainingTableJob']]


logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_S = 20


class TrainingTable:
    r"""A training table in the Kumo platform. A training table can be
    initialized from a job ID of a completed training table generation job.

    .. code-block:: python

        import kumoai_studio

        # Create a Training Table from a training table generation job. Note
        # that the job ID passed here must be in a completed state:
        training_table = kumoai_studio.TrainingTable("gen-traintable-job-...")

        # Read the training table as a Pandas DataFrame:
        training_df = training_table.data_df()

        # Get URLs to download the training table:
        training_download_urls = training_table.data_urls()

        # Add weight column to the training table:
        # see `kumo-sdk.examples.datasets.weighted_train_table.py`
        # for a more detailed example
        # 1. Export train table
        connector = kumoai_studio.S3Connector(
            name="training_table_output",
            root_dir="s3://bucket/training-tables/",
        )
        training_table.export(TrainingTableExportConfig(
            output_types={'training_table'},
            output_connector=connector,
            output_table_name="<any_name>"))
        # 2. Assume the weight column was added to the train table
        # and it was saved to the same S3 path as "<mod_name>"
        weighted = kumoai_studio.TrainingTable.from_source_table(
            SourceTable("<mod_table>", connector),
            TrainingTableSpec(weight_col="weight", split_col="SPLIT"),
            pq=pquery)

    Args:
        job_id: ID of the training table generation job which generated this
            training table.
        project_id: Exact project owning the generation job. ``None``
            snapshots the active project.
    """
    def __init__(
        self,
        job_id: GenerateTrainTableJobID,
        project_id: Optional[str] = None,
    ):
        self.job_id = job_id
        self._kumo_project_id = _resolve_project_id(project_id)
        resource = _get_resource(job_id, self._kumo_project_id)
        status = resource.job_status_report.status
        if status != JobStatus.DONE:
            raise ValueError(
                f"Job {job_id} is not yet complete (status: {status}). If you "
                f"would like to create a future (waiting for training table "
                f"generation success), please use `TrainingTableJob`.")
        # A registered job carries the supplied source and its modification
        # spec on the job record itself (`register_custom_table` puts it
        # there); a generated job carries none. Reading it back here -- not
        # trusting whatever the caller's in-memory object happened to have
        # set -- is what makes `TrainingTable(job_id)` for a registered job
        # behave the same as the object `from_source_table` returned, rather
        # than silently training on the wrong rows.
        self._custom_train_table = _recovered_custom_train_table(
            job_id, resource)

    @property
    def kumo_project_id(self) -> str:
        """Return the immutable project owning this generated table."""
        return self._kumo_project_id

    def data_urls(self) -> List[str]:
        r"""Return safe references to the generated training-table data.

        The server returns one or more references for the active deployment.
        The result may be a customer-facing dataset directory or short-lived
        signed URLs for individual parquet partitions. Internal data-plane
        paths are never returned.
        """
        api: TrainTableBackend = cloud_backend().train_table
        return api.get_table_data(self.job_id,
                                  project_id=self._kumo_project_id)

    def data_df(self) -> pd.DataFrame:
        r"""Returns a :class:`~pandas.DataFrame` object representing the
        generated training data.

        .. warning::

            This method will load the full training table into memory as a
            :class:`~pandas.DataFrame` object. If you are working on a machine
            with limited resources, please use
            :meth:`~kumoai_studio.pquery.TrainingTable.data_urls` instead to download
            the data and perform analysis per-partition.
        """
        urls = self.data_urls()
        try:
            return pd.concat([pd.read_parquet(x) for x in urls])
        except Exception as e:
            raise ValueError(
                f"Could not create a Pandas DataFrame object from data paths "
                f"{urls}. Please construct the DataFrame manually.") from e

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(job_id={self.job_id})'

    def export(
        self,
        output_config: TrainingTableExportConfig,
        non_blocking: bool = True,
        *,
        request_id: Optional[str] = None,
    ) -> Union[ArtifactExportJob, ArtifactExportResult]:
        r"""Export the training table to the connector specified in the
        output config. A generated table is exported with a ``SPLIT`` column (``0``
        train, ``1`` validation, ``2`` test). Add a weight column to the
        exported table, then load it with `from_source_table` and
        ``split_col="SPLIT"``.

        Args:
            output_config: The output configuration to write the training
                table.
            non_blocking: If ``True``, the method will return a future object
                `ArtifactExportJob` representing the export job.
                If ``False``, the method will block until the export job is
                complete and return `ArtifactExportResult`.
            request_id: Canonical UUID for idempotent submission. ``None``
                generates one. Reuse the same explicit value when retrying the
                same export request.
        """
        assert output_config.output_connector is not None
        assert output_config.output_types == {'training_table'}
        output_table_name = to_db_table_name(output_config.output_table_name)
        assert output_table_name is not None
        connector_id = output_config.output_connector.id
        table_name = output_table_name
        write_mode = WriteMode.OVERWRITE

        if not isinstance(output_config.output_connector,
                          (S3Connector, GCSConnector, ADLSConnector)):
            if output_config.connector_specific_config:
                write_mode = output_config.connector_specific_config.write_mode

        api = cloud_backend().artifact_export
        connector = output_config.output_connector
        if connector.kumo_project_id != self._kumo_project_id:
            raise ValueError(
                f"Output connector belongs to project "
                f"{connector.kumo_project_id!r}, not training-table project "
                f"{self._kumo_project_id!r}.")
        if isinstance(connector, (S3Connector, GCSConnector, ADLSConnector)):
            training_output = FileTrainingTableOutputConfig(
                connector_id=connector_id,
                path=connector._rooted_destination(table_name),
            )
        else:
            training_output = NamedTrainingTableOutputConfig(
                connector_id=connector_id,
                table_name=table_name,
                write_mode=write_mode,
            )

        request = ArtifactExportRequest(
            request_id=(str(uuid.uuid4())
                        if request_id is None else request_id),
            job_id=self.job_id,
            training_table_output=training_output,
        )
        job_id = api.create(request, project_id=self._kumo_project_id)
        if non_blocking:
            return ArtifactExportJob(job_id, project_id=self._kumo_project_id)
        return ArtifactExportJob(job_id,
                                 project_id=self._kumo_project_id).attach()

    def validate_custom_table(
        self,
        source_table_type: Union[SourceTableType, LocalFileSourceTable],
        train_table_mod: TrainingTableSpec,
        extensive_validation: bool = False,
    ) -> None:
        r"""Validates the modified training table.

        Args:
            source_table_type: The source table to be used as the modified
                training table.
            train_table_mod: The modification specification.
            extensive_validation: Enable extensive validation for custom
                table.

        Raises:
            ValueError: If the modified training table is invalid.

        """
        api: TrainTableBackend = cloud_backend().train_table
        response = api.validate_custom_train_table(
            self.job_id, source_table_type, train_table_mod,
            extensive_validation, project_id=(self._kumo_project_id))
        if not response.ok:
            raise ValueError(
                f"Invalid weighted train table: {response.error_message()}")

    @classmethod
    def from_source_table(
        cls,
        source_table: Union[SourceTable, LocalFileSourceTable],
        spec: TrainingTableSpec,
        *,
        pq: _SavedQuery,
        plan: Optional[TrainingTableGenerationPlan] = None,
    ) -> "TrainingTable":
        r"""Returns a training table over :obj:`source_table`'s rows.

        No generation runs: the rows are the caller's, and the graph, the
        query and the split come from :obj:`pq` and :obj:`plan`. The result is
        an ordinary training table with a job ID, so it names a run the same
        way a generated one does.

        Both backends record the rows. A local backend reads them in this
        process; the cloud mints the job itself, reads the rows into storage
        it owns and completes the job, and this call blocks until it does.
        A Databricks table cannot be registered against the cloud: it is
        read in place and projected onto a generated table's columns, and a
        registered table has none.

        .. code-block:: python

            train_table = kumoai_studio.TrainingTable.from_source_table(
                kumoai_studio.LocalFileSourceTable(path='labels.parquet'),
                TrainingTableSpec(weight_col='weight', split_col='SPLIT'),
                pq=pquery)
            trainer.fit(graph, train_table=train_table)

        The split is resolved in one of two ways, and refused if neither is
        available:

        1. :attr:`spec.split_col` names a column of the supplied table.
           DiskGraph reads it by name, routes each row by its value
           (``0`` train, ``1`` validation, ``2`` test) and drops the column
           from the seed output.
        2. Otherwise :obj:`plan` pins a split, exactly as it does for a
           generated table.

        Args:
            source_table: The table holding the training rows. A
                :class:`~kumoapi.source_table.LocalFileSourceTable` is
                readable only by a backend in this process.
            spec: What the supplied table carries beyond the generated
                columns -- its weight column, and its split column if it has
                one.
            pq: The query the rows are labelled for. Its graph is snapshotted
                and its ID recorded, since no generation job resolved either.
            plan: The generation plan the run reads its split, and the rest of
                its training-table settings, out of.

        Raises:
            :class:`ValueError`: If neither the specification nor the plan
                resolves a split, if the source table belongs to another
                project, or if the supplied rows do not satisfy
                training-table validation.
            :class:`RuntimeError`: If the cloud's registration job failed,
                carrying the errors it reported.
        """
        plan = _plan_with_split(plan, spec.split_col)
        project_id = pq.kumo_project_id
        source_table_type = _api_source_table(source_table, project_id)

        api: TrainTableBackend = cloud_backend().train_table
        job_id = api.register_custom_table(
            GenerateTrainTableRequest({}, pq.save(), plan, None),
            source_table_type,
            spec,
            project_id=project_id,
        )
        # A training table names a completed job. A local mint records rows
        # it has already read, so its job is done on return; the cloud reads
        # them in the job it just started, so this waits on it -- and raises
        # with the platform's own errors if the rows do not check out.
        if _get_status(job_id, project_id).status != JobStatus.DONE:
            TrainingTableJob(job_id, project_id=project_id).attach()
        # Registration records the supplied source and its modification spec
        # on the job itself (`register_custom_table`, per backend), so the
        # constructor below reads the same `CustomTrainingTable` back off the
        # job record rather than this call setting it a second time. Neither
        # backend runs training-table validation here -- there is no
        # generated table to check the rows against -- so the record's
        # `validated` stays `False`, not claimed `True`.
        table = cls(job_id, project_id=project_id)
        # The query now has a training table, exactly as it would after
        # `generate_training_table`. Without this `pq.train_table` still
        # raises, so a caller reaching for the query's table to hand
        # `Trainer.fit` would not find these rows on it.
        pq._train_table = table
        return table


def _plan_with_split(
    plan: Optional[TrainingTableGenerationPlan],
    split_col: Optional[str],
) -> TrainingTableGenerationPlan:
    r"""Returns the plan the run resolves its split out of.

    A named split column becomes the plan's split, so the one channel a
    generated table already carries its split on carries this one's too, and
    the record is honest about what the run will do. The column is the
    specification's alone: it names one table, which is the scope DiskGraph
    names it at too.

    A plan that pins no split is refused rather than defaulted:
    :func:`kumo_core.pquery._split_options` would otherwise substitute a
    ratio split of its own, which resolves -- so nothing downstream
    complains -- while training on rows the caller held out.
    """
    column = (split_col or '').strip()
    if column:
        return dataclasses.replace(
            plan if plan is not None else TrainingTableGenerationPlan(),
            split=_manual_split(column))
    if plan is not None and plan.split not in (InferredType.VALUE,
                                               MissingType.VALUE):
        return plan
    raise ValueError(
        "A supplied training table needs a split, and this one names none. "
        "Either add a split column to the table and name it with "
        "`TrainingTableSpec(split_col=...)`, valued 0 for train, 1 for "
        "validation and 2 for test, or pin one on the plan with "
        "`TrainingTableGenerationPlan(split=...)`. Leaving both unset would "
        "train on a split chosen for you rather than the one you held out.")


def _manual_split(col: str) -> str:
    r"""The split repr a plan carries a column split on.

    Spelled rather than constructed: a plan holds a split as its constructor
    repr, and ``kumo_core`` resolves that repr through its own
    :obj:`PLAN_SPLITS` -- which the SDK reaches this path without.
    """
    return (f'ManualSplit(col={col!r}, train_value=0, val_value=1, '
            f'test_value=2)')


def _api_source_table(
    source_table: Union[SourceTable, LocalFileSourceTable],
    project_id: str,
) -> Union[SourceTableType, LocalFileSourceTable]:
    r"""Resolves a source table to the form a request carries."""
    if isinstance(source_table, LocalFileSourceTable):
        return source_table
    connector = source_table.connector
    if connector.kumo_project_id != project_id:
        raise ValueError(
            f"Source table belongs to project "
            f"{connector.kumo_project_id!r}, not training-table project "
            f"{project_id!r}.")
    if (isinstance(source_table, SQLSourceTable)
            and not isinstance(connector, DatabricksConnector)):
        raise ValueError(
            'Custom SQL training sources are currently supported only '
            'for Databricks connectors.')
    if isinstance(connector, S3Connector):
        source_type = source_table._to_api_source_table()
        assert isinstance(source_type, S3SourceTable)
        return S3SourceTable(
            connector_id=connector.id,
            source=source_type.source,
            file_type=source_type.file_type,
        )
    return source_table._to_api_source_table()


# Training Table Future #######################################################


class TrainingTableJob(JobInterface[GenerateTrainTableJobID,
                                    GenerateTrainTableRequest,
                                    GenerateTrainTableJobResource],
                       KumoProgressFuture[TrainingTable]):
    r"""A representation of an ongoing training table generation job in the
    Kumo platform.

    .. code-block:: python

        import kumoai_studio

        # See `PredictiveQuery` documentation:
        pquery = kumoai_studio.PredictiveQuery(...)

        # If a training table is generated in nonblocking mode, the response
        # will be of type `TrainingTableJob`:
        training_table_job = pquery.generate_training_table(non_blocking=True)

        # You can also construct a `TrainingTableJob` from a job ID, e.g.
        # one that is present in the Kumo Jobs page:
        training_table_job = kumoai_studio.TrainingTableJob("trainingjob-...")

        # Get the status of the job:
        print(training_table_job.status())

        # Attach to the job, and poll progress updates:
        training_table_job.attach()

        # Cancel the job:
        training_table_job.cancel()

        # Wait for the job to complete, and return a `TrainingTable`:
        training_table_job.result()

    Args:
        job_id: ID of the training table generation job.
        project_id: Exact project owning the job. ``None`` snapshots the
            active project; later active-project changes do not redirect it.
    """
    @override
    @staticmethod
    def _api() -> TrainTableBackend:
        return cloud_backend().train_table

    def __init__(
        self,
        job_id: GenerateTrainTableJobID,
        project_id: Optional[str] = None,
    ) -> None:
        self.job_id = job_id
        self._kumo_project_id = _resolve_project_id(project_id)

    @cached_property
    def _fut(self) -> Future[TrainingTable]:
        return create_future(_poll(self.job_id, self._kumo_project_id))

    @override
    @property
    def id(self) -> GenerateTrainTableJobID:
        r"""The unique ID of this training table generation process."""
        return self.job_id

    @property
    def kumo_project_id(self) -> str:
        """Return the immutable project owning this generation job."""
        return self._kumo_project_id

    @override
    def result(self) -> TrainingTable:
        return self._fut.result()

    @override
    def future(self) -> Future[TrainingTable]:
        return self._fut

    @override
    def status(self) -> JobStatusReport:
        r"""Returns the status of a running training table generation job."""
        return _get_status(self.job_id, self._kumo_project_id)

    @override
    def _attach_internal(self, interval_s: float = 20.0) -> TrainingTable:
        assert interval_s >= 4.0
        tracking_url = self.status().tracking_url
        if 'http' in tracking_url:
            tracking_msg = (f"To track this job in the Kumo UI, please "
                            f"visit {tracking_url}. ")
        else:
            tracking_msg = ""
        print(f"Attaching to training table generation job {self.job_id}. "
              f"{tracking_msg}To detach "
              f"from this job, please enter Ctrl+C (the job will continue to "
              f"run, and you can re-attach anytime).")

        def _get_progress() -> Optional[Tuple[int, int]]:
            progress = self._api().get_progress(
                self.job_id, project_id=self._kumo_project_id)
            if len(progress) == 0:
                return None
            expected_iter = progress['num_expected_iterations']
            completed_iter = progress['num_finished_iterations']
            return (expected_iter, completed_iter)

        # Print progress bar:
        print("Training table generation is in progress. If your task is "
              "temporal, progress per timeframe will be loaded shortly.")

        # Wait for either timeframes to become available, or the job is done:
        progress = _get_progress()
        while progress is None:
            progress = _get_progress()
            # Not a temporal task, and it's done:
            if self.status().status.is_terminal:
                return self.result()
            time.sleep(interval_s)

        # Wait for timeframes to become available:
        progress = _get_progress()
        assert progress is not None
        total, prog = progress
        from tqdm.auto import tqdm
        pbar = tqdm(total=total, unit="timeframe",
                    desc="Generating Training Table")
        pbar.update(pbar.n - prog)
        while not self.done():
            progress = _get_progress()
            assert progress is not None
            total, prog = progress
            pbar.reset(total)
            pbar.update(prog)
            time.sleep(interval_s)
        pbar.update(pbar.total)
        pbar.close()

        # Future is done:
        return self.result()

    def cancel(self) -> None:
        r"""Cancels a running training table generation job, and raises an
        error if cancellation failed.
        """
        return self._api().cancel(self.job_id,
                                  project_id=self._kumo_project_id)

    @override
    def load_config(self) -> GenerateTrainTableRequest:
        r"""Load the full configuration for this training table generation job.

        Returns:
            GenerateTrainTableRequest: Complete configuration including plan,
            pquery_id, graph_snapshot_id, etc.
        """
        return self._api().get_config(self.job_id,
                                      project_id=self._kumo_project_id)


def _resolve_project_id(project_id: Optional[str]) -> str:
    if project_id is not None:
        return project_id
    from kumoai_studio import _require_project
    _require_project()
    assert global_state._project_id is not None
    return global_state._project_id


def _get_resource(job_id: str,
                  project_id: str) -> GenerateTrainTableJobResource:
    api = cloud_backend().train_table
    return api.get(job_id, project_id=project_id)


def _get_status(job_id: str, project_id: str) -> JobStatusReport:
    return _get_resource(job_id, project_id).job_status_report


def _recovered_custom_train_table(
    job_id: str,
    resource: GenerateTrainTableJobResource,
) -> Optional[CustomTrainingTable]:
    r"""The custom table a registration recorded on :obj:`resource`, if any.

    Three answers, not two:

    * a spec, for a job that registered supplied rows;
    * ``None``, for a job that generated its own -- it never had one, and
      that is what a generating request records;
    * a refusal, for a record that says it registered a table but cannot
      produce the spec.

    The third is the one worth having. ``config.custom_train_table`` reads
    ``None`` both for a generated job and for a record written by a
    deployment that did not persist the field, and treating those alike is
    what made a registration silently train on generated rows.
    ``custom_train_table_used`` is what separates them: it is set by the
    record itself, on both backends.
    """
    recorded = getattr(resource.config, 'custom_train_table', None)
    if recorded is not None:
        return recorded
    if not getattr(resource, 'custom_train_table_used', False):
        return None
    raise ValueError(
        f"Job {job_id!r} registered a supplied training table, but its job "
        f"record does not carry the table's specification, so which rows it "
        f"was registered against cannot be recovered. Refusing to construct "
        f"a `TrainingTable` that would silently train on generated rows "
        f"instead -- re-register the supplied table with "
        f"`TrainingTable.from_source_table`.")


async def _poll(job_id: str, project_id: str) -> TrainingTable:
    # TODO(manan): make asynchronous natively with aiohttp:
    status = _get_status(job_id, project_id).status
    while not status.is_terminal:
        await asyncio.sleep(_DEFAULT_INTERVAL_S)
        status = _get_status(job_id, project_id).status

    if status != JobStatus.DONE:
        api = cloud_backend().train_table
        error_details = api.get_job_error(job_id, project_id=project_id)
        error_str = pretty_print_error_details(error_details)
        raise RuntimeError(
            f"Training table generation for job {job_id} failed with "
            f"job status {status}. Encountered below error(s):"
            f'{error_str}')

    return TrainingTable(job_id, project_id=project_id)
