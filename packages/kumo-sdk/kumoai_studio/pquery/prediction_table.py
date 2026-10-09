# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Future
from datetime import datetime
from functools import cached_property
from typing import List, Optional

import pandas as pd
from kumoapi.common import JobStatus
from kumoapi.jobs import (
    ADLSPredictionTableSource,
    DatabricksPredictionTableSource,
    GCSPredictionTableSource,
    GeneratePredictionTableJobResource,
    GeneratePredictionTableRequest,
    JobStatusReport,
    PredictionTableSource,
    S3PredictionTableSource,
)
from kumoapi.source_table import (
    ADLSSourceTable,
    ADLSURISource,
    CustomSQLSourceTable,
    DatabricksSourceTable,
    GCSSourceTable,
    GCSURISource,
    S3SourceTable,
    S3URISource,
)
from typing_extensions import override

from kumoai_studio import global_state
from kumoai_studio.backend import PredictionTableBackend, cloud_backend
from kumoai_studio.client.jobs import GeneratePredictionTableJobID
from kumoai_studio.connector import (
    ADLSConnector,
    DatabricksConnector,
    GCSConnector,
    S3Connector,
    SourceTable,
)
from kumoai_studio.connector.file_connector import ObjectStoreConnector
from kumoai_studio.formatting import pretty_print_error_details
from kumoai_studio.futures import KumoFuture, create_future
from kumoai_studio.jobs import JobInterface

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_S = 20


class PredictionTable:
    r"""A prediction table in the Kumo platform. A prediction table can
    either be initialized from a job ID of a completed prediction table
    generation job, or an exact project-owned object-store or Databricks
    source.

    .. warning::
        Custom prediction table is an experimental feature; please work
        with your Kumo POC to ensure you are using it correctly!


    .. code-block:: python

        import kumoai_studio

        # Create a Prediction Table from a prediction table generation job.
        # Note that the job ID passed here must be in a completed state:
        prediction_table = kumoai_studio.PredictionTable("gen-predtable-job-...")

        # Read the prediction table as a Pandas DataFrame:
        prediction_df = prediction_table.data_df()

        # Get URLs to download the prediction table:
        prediction_download_urls = prediction_table.data_urls()

        # Or use a custom table through a rooted or any-URI connector:
        gcs = kumoai_studio.GCSConnector(name="all-gcs")
        source = gcs.table(
            "gs://company-data/prediction/2026/07/table.parquet")
        prediction_table = kumoai_studio.PredictionTable(source_table=source)

        # Databricks physical tables and read-only SQL use the same interface:
        dbx = kumoai_studio.DatabricksConnector.get_by_name("features")
        prediction_table = kumoai_studio.PredictionTable(
            source_table=dbx.table("prediction_entities"))
        prediction_table = kumoai_studio.PredictionTable(
            source_table=dbx.sql(
                "SELECT entity_id AS ENTITY1, event_time AS TIMESTAMP "
                "FROM seeds"))

    Args:
        job_id: ID of the prediction table generation job which
            generated this prediction table. Leave unset for a custom source.
        source_table: Exact S3, GCS, ADLS, or Databricks source returned by a
            project-owned connector. Databricks accepts a physical table from
            :meth:`DatabricksConnector.table` or read-only SQL from
            :meth:`DatabricksConnector.sql`. Raw coordinates are not accepted
            because they do not carry connector identity or project ownership.
            Leave unset when ``job_id`` is supplied.
        project_id: Exact project owning the generation job or custom table.
            For custom sources, ``None`` uses the source connector's project;
            an explicit value must match it. For job IDs, ``None`` snapshots
            the active project.
    """
    def __init__(
        self,
        job_id: Optional[GeneratePredictionTableJobID] = None,
        source_table: Optional[SourceTable] = None,
        project_id: Optional[str] = None,
    ) -> None:
        if (job_id is None) == (source_table is None):
            raise ValueError('Specify exactly one of job_id or source_table.')
        if source_table is not None:
            valid_source = isinstance(
                source_table, SourceTable) and isinstance(
                    source_table.connector,
                    (S3Connector, GCSConnector, ADLSConnector,
                     DatabricksConnector),
                )
            if not valid_source:
                raise ValueError('source_table must belong to an S3Connector, '
                                 'GCSConnector, ADLSConnector, or '
                                 'DatabricksConnector')
            source_project_id = source_table.connector.kumo_project_id
            if (project_id is not None and project_id != source_project_id):
                raise ValueError(
                    f'Source table belongs to project {source_project_id!r}, '
                    f'not {project_id!r}.')
            self._kumo_project_id = source_project_id
        else:
            self._kumo_project_id = _resolve_project_id(project_id)

        self.job_id = job_id
        self.source_table = source_table
        if job_id:
            status = _get_status(job_id, self._kumo_project_id).status
            if status != JobStatus.DONE:
                raise ValueError(
                    f"Job {job_id} is not yet complete (status: {status}). If "
                    f"you would like to create a future (waiting for "
                    f"prediction table generation success), please use "
                    f"`PredictionTableJob`.")

    @property
    def kumo_project_id(self) -> str:
        """Return the immutable project owning this prediction table."""
        return self._kumo_project_id

    def data_urls(self) -> List[str]:
        r"""Return references to the prediction-table data.

        A custom source returns its canonical object-store URI. For a generated
        table, the server returns references for the active deployment.
        The result may be a customer-facing dataset directory or short-lived
        signed URLs for individual parquet partitions. Internal data-plane
        paths are never returned.
        """
        if self.source_table is not None:
            wire_source = self._to_api_custom_source()
            if isinstance(wire_source, DatabricksPredictionTableSource):
                raise ValueError(
                    'Databricks prediction table sources are read directly by '
                    'Kumo and do not expose downloadable data URLs.')
            custom_source = wire_source.source
            if isinstance(custom_source,
                          (S3URISource, GCSURISource, ADLSURISource)):
                return [custom_source.uri]
            connector = self.source_table.connector
            assert isinstance(connector, ObjectStoreConnector)
            return [connector._full_uri_for_relative(custom_source.path)]
        assert self.job_id is not None
        api = cloud_backend().prediction_table
        return api.get_table_data(self.job_id,
                                  project_id=self._kumo_project_id)

    def _data_urls_for_read(self) -> List[str]:
        """Return server-authorized URLs when reading a custom source."""
        if self.source_table is None:
            return self.data_urls()
        if isinstance(self._to_api_custom_source(),
                      DatabricksPredictionTableSource):
            raise ValueError(
                'Databricks prediction table sources are read directly by '
                'Kumo and cannot be downloaded through PredictionTable.')
        connector = self.source_table.connector
        # Source data is read through the connector's own access seam (see
        # `kumoai_studio.connector.access`), not the execution backend.
        return connector._get_download_urls(
            self.source_table.name,
            project_id=self._kumo_project_id,
        )

    def _to_api_custom_source(self) -> PredictionTableSource:
        """Return the secret-free connector-backed custom source contract."""
        if self.source_table is None:
            raise ValueError(
                'Generated prediction tables have no custom source')
        source = self.source_table._to_api_source_table()
        source_types = (
            S3SourceTable,
            GCSSourceTable,
            ADLSSourceTable,
            DatabricksSourceTable,
            CustomSQLSourceTable,
        )
        if not isinstance(source, source_types):
            raise ValueError(
                'Custom prediction source must resolve to S3, GCS, ADLS, or '
                'Databricks')
        connector_id = (source.databricks_connector_id if isinstance(
            source, DatabricksSourceTable) else source.connector_id)
        if connector_id != self.source_table.connector.id:
            raise ValueError('Custom prediction source connector mismatch')
        if isinstance(source, DatabricksSourceTable):
            return DatabricksPredictionTableSource(
                connector_id=connector_id,
                table=source.table,
            )
        if isinstance(source, CustomSQLSourceTable):
            if not isinstance(self.source_table.connector,
                              DatabricksConnector):
                raise ValueError(
                    'Custom SQL prediction sources are currently supported '
                    'only for Databricks connectors')
            return DatabricksPredictionTableSource(
                connector_id=connector_id,
                sql=source.sql,
            )
        if isinstance(source, S3SourceTable):
            return S3PredictionTableSource(connector_id=source.connector_id,
                                           source=source.source)
        if isinstance(source, GCSSourceTable):
            return GCSPredictionTableSource(connector_id=source.connector_id,
                                            source=source.source)
        return ADLSPredictionTableSource(connector_id=source.connector_id,
                                         source=source.source)

    def data_df(self) -> pd.DataFrame:
        r"""Returns a Pandas DataFrame object representing the generated
        or custom-specified prediction table data.

        .. warning::

            This method will load the full prediction table into memory as a
            :class:`~pandas.DataFrame` object. If you are working on a machine
            with limited resources, please use
            :meth:`~kumoai_studio.pquery.PredictionTable.data_urls` instead to
            download the data and perform analysis per-partition.
        """
        urls = self._data_urls_for_read()
        try:
            return pd.concat([pd.read_parquet(x) for x in urls])
        except Exception as e:
            raise ValueError(
                f"Could not create a Pandas DataFrame object from data "
                f"paths {urls}. Please construct the DataFrame manually."
            ) from e

    @property
    def anchor_time(self) -> Optional[datetime]:
        r"""Returns the anchor time corresponding to the generated prediction
        table data, if the data was not custom-specified.
        """
        if self.job_id is None:
            logger.warning(
                "Fetching the anchor time is not supported for a custom "
                "prediction table (source: %s)", self.source_table)
            return None
        api = cloud_backend().prediction_table
        return api.get_anchor_time(self.job_id,
                                   project_id=self._kumo_project_id)


# Prediction Table Future #####################################################


class PredictionTableJob(JobInterface[GeneratePredictionTableJobID,
                                      GeneratePredictionTableRequest,
                                      GeneratePredictionTableJobResource],
                         KumoFuture[PredictionTable]):
    r"""A representation of an ongoing prediction table generation job in the
    Kumo platform.

    .. code-block:: python

        import kumoai_studio

        # See `PredictiveQuery` documentation:
        pquery = kumoai_studio.PredictiveQuery(...)

        # If a prediction table is generated in nonblocking mode, the response
        # will be of type `PredictionTableJob`:
        prediction_table_job = pquery.generate_prediction_table(non_blocking=True)

        # You can also construct a `PredictionTableJob` from a job ID, e.g.
        # one that is present in the Kumo Jobs page:
        prediction_table_job = kumoai_studio.PredictionTableJob("gen-predtable-job-...")

        # Get the status of the job:
        print(prediction_table_job.status())

        # Cancel the job:
        prediction_table_job.cancel()

        # Wait for the job to complete, and return a `PredictionTable`:
        prediction_table_job.result()

    Args:
        job_id: ID of the prediction table generation job.
        project_id: Exact project owning the job. ``None`` snapshots the
            active project; later active-project changes do not redirect it.
    """  # noqa

    @override
    @staticmethod
    def _api() -> PredictionTableBackend:
        return cloud_backend().prediction_table

    def __init__(
        self,
        job_id: GeneratePredictionTableJobID,
        project_id: Optional[str] = None,
    ) -> None:
        self.job_id = job_id
        self._kumo_project_id = _resolve_project_id(project_id)
        self.job: Optional[GeneratePredictionTableJobResource] = None

    @cached_property
    def _fut(self) -> Future:
        return create_future(self._poll())

    @override
    @property
    def id(self) -> GeneratePredictionTableJobID:
        r"""The unique ID of this prediction table generation process."""
        return self.job_id

    @property
    def kumo_project_id(self) -> str:
        """Return the immutable project owning this generation job."""
        return self._kumo_project_id

    @override
    def result(self) -> PredictionTable:
        return self._fut.result()

    @override
    def future(self) -> Future[PredictionTable]:
        return self._fut

    @override
    def status(self) -> JobStatusReport:
        r"""Returns the status of a running prediction table generation job."""
        return self._poll_job().job_status_report

    def cancel(self) -> None:
        r"""Cancels a running prediction table generation job, and raises an
        error if cancellation failed.
        """
        return self._api().cancel(self.job_id,
                                  project_id=self._kumo_project_id)

    # TODO(manan): make asynchronous natively with aiohttp:
    def _poll_job(self) -> GeneratePredictionTableJobResource:
        # Skip polling if job is already in terminal state.
        if not self.job or not self.job.job_status_report.status.is_terminal:
            self.job = self._api().get(self.job_id,
                                       project_id=self._kumo_project_id)
        return self.job

    async def _poll(self) -> PredictionTable:
        while not self.status().status.is_terminal:
            await asyncio.sleep(_DEFAULT_INTERVAL_S)
        status = self.status().status
        if status != JobStatus.DONE:
            error_details = self._api().get_job_error(
                self.job_id, project_id=self._kumo_project_id)
            error_str = pretty_print_error_details(error_details)
            raise RuntimeError(
                f"Prediction table generation for job {self.job_id} failed "
                f"with job status {status}. Encountered below"
                f" errors: {error_str}")
        return PredictionTable(self.job_id, project_id=self._kumo_project_id)

    @override
    def load_config(self) -> GeneratePredictionTableRequest:
        r"""Load the full configuration for this
        prediction table generation job.

        Returns:
            GeneratePredictionTableRequest:
                Complete configuration including plan,
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


def _get_status(job_id: str, project_id: str) -> JobStatusReport:
    api = cloud_backend().prediction_table
    resource: GeneratePredictionTableJobResource = api.get(
        job_id, project_id=project_id)
    return resource.job_status_report
