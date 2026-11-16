# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from abc import ABC, abstractmethod
from http import HTTPStatus
from typing import Any, List, Optional

from kumoapi.data_source import ConnectorResourceConfig, DataSourceType
from kumoapi.source_table import (
    SourceTableConfigResponse,
    SourceTableDataResponse,
    SourceTableListResponse,
    SourceTableValidateResponse,
)

from kumoai_studio.backend import cloud_backend
from kumoai_studio.connector.source_table import SourceTable, SQLSourceTable
from kumoai_studio.exceptions import HTTPException

# Warehouse connectors that can execute a custom SQL query (pushdown):
_SQL_SOURCE_TYPES = frozenset({
    DataSourceType.SNOWFLAKE,
    DataSourceType.DATABRICKS,
    DataSourceType.BIGQUERY,
})


class Connector(ABC):
    r"""A project-owned connection to a data target.

    A connector records an immutable target, such as an object-store prefix or
    warehouse schema, within the active project. External connectors also
    reference one administrator-managed identity. The identity controls
    authentication while the connector identifies the data to use.

    Connectors cannot be shared across projects. Selecting another project
    with :func:`kumoai_studio.set_project` changes the scope for connector creation,
    lookup, discovery, and table access.

    Reading source data goes through the active backend's ``source`` area
    (see :class:`~kumoai_studio.backend.base.SourceBackend`), which receives this
    connector and resolves its target itself. A connector therefore carries
    no session and no credentials, which is what keeps a graph containing it
    serializable.
    """

    _kumo_project_id: str

    @property
    @abstractmethod
    def source_type(self) -> DataSourceType:
        """Return the data-source type exposed by this connector."""
        raise NotImplementedError()

    @property
    def config(self) -> ConnectorResourceConfig:
        r"""Return what this connector points at, in its provider's terms.

        A source reader narrows this to its own provider's member and reads
        the target off it -- database and schema, catalog and namespace,
        bucket and prefix.

        Declared here rather than only on the subclasses that carry one,
        because :class:`~kumoapi.plugin.SourceConnector` is what every reader
        is written against and this is the member it narrows. Not abstract:
        an upload connector has no external target, and refusing is a better
        answer for it than an empty one.
        """
        raise NotImplementedError(
            f"{type(self).__name__} points at no provider-native target.")

    @property
    @abstractmethod
    def id(self) -> str:
        """Return the connector's generated immutable ID."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def name(self) -> str:
        r"""Return the connector's project-local display name.

        Names are unique after normalization within a project. The same name
        may refer to a different connector in another project.
        """
        raise NotImplementedError()

    @property
    def kumo_project_id(self) -> str:
        r"""Return the immutable Kumo project that owns this connector.

        Connector operations keep using this project even if an application
        later changes the process-wide active project.
        """
        return self._kumo_project_id

    def table_names(self) -> List[str]:
        r"""Return source table names available through this connector.

        Discovery uses the connector's persisted identity and is authorized
        against the active project each time. For object stores, discovery is
        intentionally shallow and returns immediate children of the configured
        root. Use :meth:`table` with a connector-relative path to address a
        table at any depth without recursively listing the entire root.
        """
        return self._list_tables().table_names

    def has_table(self, name: str) -> bool:
        r"""Return whether a source table is visible through this connector.

        Args:
            name: A provider-native table name. Object-store connectors accept
                an arbitrary-depth path relative to their root or a same-root
                full provider URI.
        """
        return self._validate_table(name).is_valid

    def table(self, name: str) -> SourceTable:
        r"""Return a source-table view owned by this connector's project.

        A :class:`~kumoai_studio.connector.SourceTable` is a view of raw data. Convert
        it to a :class:`~kumoai_studio.graph.Table` to define semantic types, keys,
        and other metadata for use in a Kumo graph.

        Args:
            name: A provider-native table name. Object-store connectors accept
                an arbitrary-depth path relative to their root or a same-root
                full provider URI. Their :meth:`table_from_uri` method is an
                explicit equivalent for the full-URI form.

        Raises:
            :class:`ValueError`: If the table is not visible through this
                connector.
        """
        try:
            response = self._get_table_config(name)
        except HTTPException as error:
            if self._is_missing_table_error(error, name):
                raise ValueError(
                    f"The table {name!r} does not exist in {self}.") from error
            raise
        return SourceTable(
            name=name,
            connector=self,
            _api_source_table=response.source_table,
        )

    def sql(self, query: str) -> SQLSourceTable:
        r"""Returns a :class:`~kumoai_studio.connector.SQLSourceTable` backed by a
        read-only ``SELECT`` executed in this connector's warehouse
        (pushdown). The query is sent to the warehouse unchanged and only its
        result set is ingested. To use it in Kumo, wrap it in a
        :class:`~kumoai_studio.graph.Table` (e.g. with
        :meth:`~kumoai_studio.graph.Table.from_sql`).

        Args:
            query: The read-only ``SELECT`` statement to run in the warehouse.

        Raises:
            :class:`ValueError`: if this connector cannot execute SQL. Only
                warehouse connectors (Snowflake, Databricks, BigQuery) support
                custom SQL.
        """
        if self.source_type not in _SQL_SOURCE_TYPES:
            raise ValueError(
                f"Custom SQL is only supported for warehouse connectors "
                f"(Snowflake, Databricks, BigQuery); {type(self).__name__} "
                f"({self.source_type.value}) cannot execute SQL.")

        return SQLSourceTable(sql=query, connector=self)

    def _validate_table(self, table_name: str) -> SourceTableValidateResponse:
        try:
            self._get_table_config(table_name)
        except HTTPException as error:
            if self._is_missing_table_error(error, table_name):
                return SourceTableValidateResponse(
                    is_valid=False,
                    msg=str(error.detail),
                )
            raise
        return SourceTableValidateResponse(is_valid=True, msg='')

    @classmethod
    def _is_missing_table_error(cls, error: HTTPException,
                                table_name: str) -> bool:
        if error.status_code == HTTPStatus.NOT_FOUND:
            return True
        if error.status_code != HTTPStatus.UNPROCESSABLE_ENTITY:
            return False
        expected = f'Invalid table {table_name}'
        return expected in cls._error_detail_messages(error.detail)

    @classmethod
    def _error_detail_messages(cls, detail: Any) -> set[str]:
        if isinstance(detail, str):
            messages = {detail}
            try:
                parsed = json.loads(detail)
            except (TypeError, ValueError):
                return messages
            return messages | cls._error_detail_messages(parsed)
        if isinstance(detail, dict):
            messages: set[str] = set()
            for value in detail.values():
                messages.update(cls._error_detail_messages(value))
            return messages
        if isinstance(detail, (list, tuple)):
            messages = set()
            for value in detail:
                messages.update(cls._error_detail_messages(value))
            return messages
        return {str(detail)}

    def _list_tables(self) -> SourceTableListResponse:
        return cloud_backend().source.list_tables(self)

    def _get_table_data(
        self,
        table_names: List[str],
        sample_rows: int,
    ) -> List[SourceTableDataResponse]:
        source = cloud_backend().source
        if sample_rows <= 0:
            return [source.get_columns(self, name) for name in table_names]
        return [
            source.get_sample(self, name, sample_rows) for name in table_names
        ]

    def _get_table_config(self, table_name: str) -> SourceTableConfigResponse:
        return cloud_backend().source.get_table_config(self, table_name)

    def _get_download_urls(
        self,
        table_name: str,
        project_id: Optional[str] = None,
    ) -> List[str]:
        r"""Returns readable references to one source table's Parquet data.

        Args:
            table_name: Provider-native table coordinate.
            project_id: Authorizing project. Defaults to the connector's own
                project; callers that already hold an exact project pass it
                explicitly.
        """
        return cloud_backend().source.get_download_urls(
            self, table_name, project_id=project_id)

    def __contains__(self, name: str) -> bool:
        return self.has_table(name)

    def __getitem__(self, name: str) -> SourceTable:
        return self.table(name)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name!r})'
