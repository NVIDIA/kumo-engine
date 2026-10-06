# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set

from kumo_core.types import ConnectorType
from kumoapi.data_source import DataSourceType
from kumoapi.source_table import CUSTOM_SQL_SOURCE_TABLE_TYPE

from rest.config.connector_db import ConnectorDBEntry
from statedb.config.common import ConnectorID, IdentityID
from statedb.config.connector import PROJECT_UPLOAD_CONNECTOR_NAME
from statedb.db.repositories import (
    JsonRepository,
    _in_predicate,
    _json_value,
    _placeholder,
)
from statedb.db.row_parsing import row_str
from statedb.db.runtime import StateTx
from statedb.rest.connector_db import (
    ExternalConnectorDBEntry,
    InternalConnectorDBEntry,
    is_project_upload_connector,
)
from statedb.rest.project import ProjectDBEntry, ProjectStatus
from statedb.utils.errors import ConnectorDoesNotExistError


@dataclass
class ConnectorAlreadyExistsError(Exception):
    connector_id: ConnectorID

    def __str__(self) -> str:
        return f"Connector with ID '{self.connector_id}' already exists"


class StateDBConnectorRepository:
    def __init__(self) -> None:
        self._connectors = JsonRepository("connectors", ConnectorDBEntry)
        self._projects = JsonRepository("projects", ProjectDBEntry)

    async def _validate_persistence_invariants(
        self,
        tx: StateTx,
        connector: ConnectorDBEntry,
    ) -> None:
        project = await self._projects.get(tx, str(connector.project_id))
        if project is None or project.payload.status != ProjectStatus.ACTIVE:
            raise ValueError(
                f"Connector project '{connector.project_id}' is not active")
        if (connector.normalized_name == PROJECT_UPLOAD_CONNECTOR_NAME
                and isinstance(connector, ExternalConnectorDBEntry)):
            raise ValueError('Connector name is reserved for internal uploads')
        if (isinstance(connector, InternalConnectorDBEntry)
                and connector.normalized_name == PROJECT_UPLOAD_CONNECTOR_NAME):
            if not is_project_upload_connector(connector):
                raise ValueError('Invalid project upload connector')

        existing = await self.get(tx, connector.id)
        if existing is None:
            return
        if existing.project_id != connector.project_id:
            raise ValueError('Connector project_id is immutable')
        if existing.kind != connector.kind:
            raise ValueError('Connector kind is immutable')
        if existing.config != connector.config:
            raise ValueError('Connector target configuration is immutable')
        if (getattr(existing, 'identity_id', None)
                != getattr(connector, 'identity_id', None)):
            raise ValueError('Connector identity_id is immutable')

    async def get(
        self,
        tx: StateTx,
        connector_id: ConnectorID,
    ) -> ConnectorDBEntry | None:
        return await self._connectors.get_payload(tx, connector_id)

    async def get_required(
        self,
        tx: StateTx,
        connector_id: ConnectorID,
    ) -> ConnectorDBEntry:
        connector = await self.get(tx, connector_id)
        if connector is None:
            raise ConnectorDoesNotExistError(connector_id)
        return connector

    async def upsert(self, tx: StateTx, connector: ConnectorDBEntry) -> None:
        await self._validate_persistence_invariants(tx, connector)
        await self._connectors.upsert(tx, str(connector.id), connector)

    async def find_live_by_project_and_name(
        self,
        tx: StateTx,
        project_id,
        normalized_name: str,
    ) -> ConnectorDBEntry | None:
        row = await self._connectors.find_one_by_fields(
            tx,
            {
                'project_id': str(project_id),
                'normalized_name': normalized_name,
                'tombstoned_at': None,
            },
        )
        return row.payload if row is not None else None

    async def list_live_by_identity(
        self,
        tx: StateTx,
        identity_id: IdentityID,
    ) -> list[ExternalConnectorDBEntry]:
        rows = await self._connectors.list_by_fields(
            tx,
            {
                'identity_id': str(identity_id),
                'tombstoned_at': None,
            },
        )
        return [
            row.payload for row in rows
            if isinstance(row.payload, ExternalConnectorDBEntry)
        ]

    async def create(
        self,
        tx: StateTx,
        connector: ConnectorDBEntry,
    ) -> None:
        existing = await self.get(tx, connector.id)
        if existing is not None:
            raise ConnectorAlreadyExistsError(connector.id)
        await self.upsert(tx, connector)

    async def delete(self, tx: StateTx, connector_id: ConnectorID) -> None:
        if await self.get(tx, connector_id) is None:
            raise ConnectorDoesNotExistError(connector_id)
        await self._connectors.delete(tx, connector_id)

    async def list(
        self,
        tx: StateTx,
        *,
        connector_ids: Optional[Set[ConnectorID]] = None,
        data_source_type: Optional[DataSourceType] = None,
        connector_type: ConnectorType | str | None = None,
        root_dir: Optional[str] = None,
        root_dirs: Optional[Set[str]] = None,
        project_id=None,
    ) -> list[ConnectorDBEntry]:
        rows = await self._list_rows(
            tx,
            connector_ids=connector_ids,
            data_source_type=data_source_type,
            connector_type=connector_type,
            root_dir=root_dir,
            root_dirs=root_dirs,
            project_id=project_id,
            select_payload=True,
        )
        return [
            self._connectors._row_from_list_row(row).payload for row in rows
        ]

    async def list_ids(
        self,
        tx: StateTx,
        *,
        connector_ids: Optional[Set[ConnectorID]] = None,
        data_source_type: Optional[DataSourceType] = None,
        connector_type: ConnectorType | str | None = None,
        root_dir: Optional[str] = None,
        root_dirs: Optional[Set[str]] = None,
        project_id=None,
    ) -> list[ConnectorID]:
        rows = await self._list_rows(
            tx,
            connector_ids=connector_ids,
            data_source_type=data_source_type,
            connector_type=connector_type,
            root_dir=root_dir,
            root_dirs=root_dirs,
            project_id=project_id,
            select_payload=False,
        )
        return [ConnectorID(row_str(row, "id", 0)) for row in rows]

    async def _list_rows(
        self,
        tx: StateTx,
        *,
        connector_ids: Optional[Set[ConnectorID]],
        data_source_type: Optional[DataSourceType],
        connector_type: ConnectorType | str | None,
        root_dir: Optional[str],
        root_dirs: Optional[Set[str]],
        project_id,
        select_payload: bool,
    ):
        if connector_ids is not None and not connector_ids:
            return []
        if root_dir is not None and root_dirs is not None:
            raise ValueError("Specify either root_dir or root_dirs, not both")
        if root_dirs is not None and not root_dirs:
            return []

        clauses: list[str] = []
        args: list[object] = []

        if connector_ids is not None:
            clauses.append(
                _in_predicate(tx, "id", sorted(map(str, connector_ids)), args))

        if project_id is not None:
            args.append(str(project_id))
            clauses.append(
                f"{_json_value(tx, 'project_id')} = {_placeholder(tx, len(args))}"
            )

        if data_source_type is not None:
            connector_types = _connector_types_for_data_source_type(
                data_source_type)
            if not connector_types:
                return []
            clauses.append(
                _in_predicate(tx, _connector_json_value(tx, "type"),
                              connector_types, args))

        if connector_type is not None:
            args.append(ConnectorType(connector_type).value)
            clauses.append(
                f"{_connector_json_value(tx, 'type')} = {_placeholder(tx, len(args))}"
            )

        if root_dir is not None:
            args.append(root_dir)
            clauses.append(
                f"{_connector_json_value(tx, 'root_dir')} = {_placeholder(tx, len(args))}"
            )
        if root_dirs is not None:
            clauses.append(
                _in_predicate(tx, _connector_json_value(tx, "root_dir"),
                              sorted(root_dirs), args))

        select = ("id, payload, created_at, updated_at"
                  if select_payload else "id")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return await tx.fetch(
            f"""
            SELECT {select}
            FROM connectors
            {where}
            ORDER BY id
            """,
            *args,
        )


def _connector_json_value(tx: StateTx, field_name: str) -> str:
    return _json_value(tx, ('config', field_name))


def _connector_types_for_data_source_type(
        data_source_type: DataSourceType | str) -> list[str]:
    # Custom SQL is a kind of source table, not a ``DataSourceType`` member, so
    # it cannot be coerced to the enum; match its stable on-the-wire tag first.
    tag = getattr(data_source_type, "value", data_source_type)
    if tag == CUSTOM_SQL_SOURCE_TABLE_TYPE:
        # A Custom SQL table is backed by any of the supported warehouse
        # SQL connectors.
        return [
            ConnectorType.SNOWFLAKE.value,
            ConnectorType.DATABRICKS.value,
            ConnectorType.BIGQUERY.value,
        ]
    data_source_type = DataSourceType(data_source_type)
    if data_source_type == DataSourceType.SNOWFLAKE:
        return [ConnectorType.SNOWFLAKE.value]
    if data_source_type in (DataSourceType.S3, DataSourceType.GCS,
                            DataSourceType.ADLS):
        explicit_type = {
            DataSourceType.S3: ConnectorType.S3.value,
            DataSourceType.GCS: ConnectorType.GCS.value,
            DataSourceType.ADLS: ConnectorType.ADLS.value,
        }[data_source_type]
        return [
            explicit_type,
            ConnectorType.FILE.value,
        ]
    if data_source_type == DataSourceType.DATABRICKS:
        return [ConnectorType.DATABRICKS.value]
    if data_source_type == DataSourceType.BIGQUERY:
        return [ConnectorType.BIGQUERY.value]
    return []
