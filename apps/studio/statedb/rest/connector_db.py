# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Literal, Optional, TypeGuard, Union

from pydantic import Field

from io_utils import kumo_dataclass
from statedb.config.common import ConnectorID, DBEntry, IdentityID, ProjectID
from statedb.config.connector import (
    PROJECT_UPLOAD_CONNECTOR_NAME,
    ExternalConnectorResourceConfigDB,
    FileConnectorResourceConfig,
    InternalConnectorResourceConfigDB,
)


@kumo_dataclass
class ExternalConnectorDBEntry(DBEntry):
    id: ConnectorID
    project_id: ProjectID
    identity_id: IdentityID
    name: str
    normalized_name: str
    created_by: str
    description: str = ''
    tombstoned_at: Optional[datetime] = None
    kind: Literal['external'] = 'external'
    config: ExternalConnectorResourceConfigDB = Field(discriminator='type')


@kumo_dataclass
class InternalConnectorDBEntry(DBEntry):
    id: ConnectorID
    project_id: ProjectID
    name: str
    normalized_name: str
    description: str = ''
    tombstoned_at: Optional[datetime] = None
    kind: Literal['internal'] = 'internal'
    config: InternalConnectorResourceConfigDB = Field(discriminator='type')


ConnectorDBEntry = Union[ExternalConnectorDBEntry, InternalConnectorDBEntry]


def is_project_upload_connector(
        connector: ConnectorDBEntry) -> TypeGuard[InternalConnectorDBEntry]:
    """Return whether this is the exact canonical internal upload connector."""
    return (isinstance(connector, InternalConnectorDBEntry)
            and connector.name == PROJECT_UPLOAD_CONNECTOR_NAME
            and connector.normalized_name == PROJECT_UPLOAD_CONNECTOR_NAME
            and isinstance(connector.config, FileConnectorResourceConfig)
            and connector.config.name == PROJECT_UPLOAD_CONNECTOR_NAME)


def is_active_project_upload_connector(
        connector: ConnectorDBEntry) -> TypeGuard[InternalConnectorDBEntry]:
    """Return whether this is the exact live internal upload connector."""
    return (connector.tombstoned_at is None
            and is_project_upload_connector(connector))
