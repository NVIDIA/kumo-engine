# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""The Kumo cloud platform as a :class:`~kumoai_studio.backend.base.Backend`."""

from typing import List, Mapping, Optional

import kumoapi.source_table as source_table_api
from kumoapi.plugin import PrepareIngestResult, SourceBackend, SourceConnector
from kumoapi.typing import Dtype

from kumoai_studio.backend.base import (
    ArtifactExportBackend,
    ConnectorBackend,
    GraphBackend,
    IdentityBackend,
    PQueryBackend,
    PredictionBackend,
    PredictionTableBackend,
    TableBackend,
    TrainingBackend,
    TrainTableBackend,
)
from kumoai_studio.client.client import KumoClient
from kumoai_studio.client.source_table import SourceTableAPI

__all__ = ['CloudBackend', 'RestSourceBackend']


def _no_query(sql: Optional[str]) -> None:
    r"""Refuses a query this path does not carry.

    The parameter is declared because
    :class:`~kumoapi.plugin.SourceBackend` declares it, which is what lets the
    hosted reader stand in for a local one. Custom SQL against the service
    goes through the table area's own endpoint, which resolves a query
    server-side, so a query arriving here means a caller routed it wrongly.
    """
    if sql is not None:
        raise NotImplementedError(
            "The Kumo service resolves Custom SQL through its own endpoint; "
            "this reader answers for a named table only.")


class RestSourceBackend:
    r"""Resolves source data through the Kumo service.

    Only ``connector.id`` is needed: the server holds the connector's target
    and the credentials of the identity bound to it, so one instance serves
    every connector regardless of the system behind it.

    The client is resolved per call for the same reason as in
    :class:`CloudBackend`.
    """
    @property
    def _api(self) -> SourceTableAPI:
        from kumoai_studio import global_state
        return global_state.client.source_table_api

    @staticmethod
    def _project(connector: SourceConnector, project_id: Optional[str]) -> str:
        return project_id or connector.kumo_project_id

    def list_tables(
        self,
        connector: SourceConnector,
        project_id: Optional[str] = None,
    ) -> source_table_api.SourceTableListResponse:
        return self._api.list_tables(
            connector.id, project_id=self._project(connector, project_id))

    def get_table_config(
        self,
        connector: SourceConnector,
        table_name: str,
        project_id: Optional[str] = None,
    ) -> source_table_api.SourceTableConfigResponse:
        return self._api.get_table_config(
            connector.id, table_name,
            project_id=self._project(connector, project_id))

    def get_columns(
        self,
        connector: SourceConnector,
        table_name: str,
        project_id: Optional[str] = None,
        sql: Optional[str] = None,
    ) -> source_table_api.SourceTableDataResponse:
        # Custom SQL reaches the service through the table area's own
        # endpoint, which resolves a query against a connector server-side;
        # this path answers for a named table only.
        _no_query(sql)
        return self._api.get_columns(
            connector.id, table_name,
            project_id=self._project(connector, project_id))

    def get_sample(
        self,
        connector: SourceConnector,
        table_name: str,
        sample_rows: int,
        project_id: Optional[str] = None,
        sql: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> source_table_api.SourceTableDataResponse:
        _no_query(sql)
        del timeout
        return self._api.get_sample(
            connector.id, table_name, sample_rows,
            project_id=self._project(connector, project_id))

    def get_download_urls(
        self,
        connector: SourceConnector,
        table_name: str,
        project_id: Optional[str] = None,
    ) -> List[str]:
        return self._api._get_download_urls(
            connector.id, table_name,
            project_id=self._project(connector, project_id))

    def get_data_version(
        self,
        connector: SourceConnector,
        table_name: str,
        project_id: Optional[str] = None,
    ) -> Optional[str]:
        raise NotImplementedError(
            "The Kumo service versions source data itself, as part of the "
            "snapshot it builds; a client never establishes a data version. "
            "Call graph.snapshot() and let the platform do it.")

    def prepare_ingest(
        self,
        connector: SourceConnector,
        table_name: str,
        dest_dir: str,
        dtypes: Optional[Mapping[str, Dtype]] = None,
        project_id: Optional[str] = None,
        data_version: Optional[str] = None,
        sql: Optional[str] = None,
    ) -> PrepareIngestResult:
        raise NotImplementedError(
            "The Kumo service stages and ingests source tables itself; a "
            "client never prepares them for ingest. Call graph.snapshot() and "
            "let the platform do it.")

    def __repr__(self) -> str:
        return 'RestSourceBackend()'


class CloudBackend:
    r"""Routes every operation to the Kumo REST service.

    Each property returns the corresponding typed API object from
    :class:`~kumoai_studio.client.client.KumoClient`. Those classes already satisfy
    the backend protocols structurally, so this class adds naming and a single
    place to intercept, not a second layer of translation.

    The client is resolved on each access rather than cached, because
    :class:`~kumoai_studio.GlobalState` stores clients as thread-local values and
    refreshes SPCS tokens between calls.
    """
    @property
    def name(self) -> str:
        return 'cloud'

    @property
    def _client(self) -> KumoClient:
        from kumoai_studio import global_state
        return global_state.client

    #: Stateless and connector-agnostic: the server holds every connector's
    #: target and credentials, so one reader serves them all.
    _source = RestSourceBackend()

    @property
    def identity(self) -> IdentityBackend:
        return self._client.identity_api

    @property
    def connector(self) -> ConnectorBackend:
        return self._client.connector_api

    @property
    def source(self) -> SourceBackend:
        return self._source

    @property
    def table(self) -> TableBackend:
        return self._client.table_api

    @property
    def graph(self) -> GraphBackend:
        return self._client.graph_api

    @property
    def pquery(self) -> PQueryBackend:
        return self._client.pquery_api

    @property
    def train_table(self) -> TrainTableBackend:
        return self._client.generate_train_table_job_api

    @property
    def training(self) -> TrainingBackend:
        return self._client.training_job_api

    @property
    def prediction_table(self) -> PredictionTableBackend:
        return self._client.generate_prediction_table_job_api

    @property
    def prediction(self) -> PredictionBackend:
        return self._client.batch_prediction_job_api

    @property
    def artifact_export(self) -> ArtifactExportBackend:
        return self._client.artifact_export_api

    def __repr__(self) -> str:
        return 'CloudBackend()'
