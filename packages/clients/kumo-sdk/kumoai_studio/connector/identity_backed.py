# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from kumoapi.data_source import (
    ConnectorResourceConfig,
    ConnectorResponse,
    CreateConnectorArgs,
    IdentityType,
)

from kumoai_studio import _project_scope, _require_project, global_state
from kumoai_studio.backend import cloud_backend
from kumoai_studio.connector.base import Connector
from kumoai_studio.identity import Identity


class IdentityBackedConnector(Connector):
    r"""Base class for project-owned connectors backed by an identity.

    Customer code normally constructs one of the six concrete connector
    classes. Each constructor creates or idempotently retrieves a connector in
    the active project. The persisted connector always has one exact identity,
    even when the caller delegates identity selection to the server.
    """
    def _bind_connector(
        self,
        name: str,
        identity: str | Identity | None,
        config: ConnectorResourceConfig,
        description: str,
        response: Optional[ConnectorResponse] = None,
        project_id: Optional[str] = None,
    ) -> None:
        resolved_project_id = (self._resolve_project_id(project_id)
                               if project_id is not None else None)
        if resolved_project_id is None and response is None:
            # Only a project-scoped backend needs one; an in-process runtime
            # has no projects to be missing.
            _project_scope()
        expected = IdentityType(config.type.value)
        identity_id = (identity.id
                       if isinstance(identity, Identity) else identity)
        if isinstance(identity, Identity):
            if identity.type != expected:
                raise ValueError(f"Identity {identity.id!r} has type "
                                 f"{identity.type.value!r}, "
                                 f"not {expected.value!r}.")
        if identity is not None and not identity_id:
            raise ValueError('identity must be an immutable identity ID')
        if response is None:
            response = cloud_backend().connector.create_if_not_exist(
                CreateConnectorArgs(
                    identity_id=identity_id,
                    name=name,
                    config=config,
                    description=description,
                ),
                project_id=resolved_project_id,
            )
        if (resolved_project_id is not None
                and response.project_id != resolved_project_id):
            raise ValueError(
                f"Connector {response.id!r} belongs to project "
                f"{response.project_id!r}, not {resolved_project_id!r}.")
        #: What this connector points at, in its provider's own terms:
        #: database and schema, catalog and namespace, bucket and prefix.
        #: Public because it *is* the target contract a source reader is
        #: written against -- a reader asks for ``connector.config`` and
        #: narrows it, rather than for a set of fields that differs per
        #: provider. A connector the local backend rebuilds from a stored
        #: response carries the same object, so both ways into a reader
        #: present one surface.
        self._config = config
        self._id = str(response.id)
        self._kumo_project_id = str(response.project_id)
        self._name = response.name
        self.identity_id = str(response.identity_id)
        self.description = response.description

    @property
    def config(self) -> ConnectorResourceConfig:
        """Return the provider-native target this connector points at."""
        return self._config

    @property
    def id(self) -> str:
        """Return the connector's immutable ID."""
        return self._id

    @property
    def name(self) -> str:
        """Return the connector's project-local display name."""
        return self._name

    @staticmethod
    def _resolve_project_id(project_id: Optional[str]) -> str:
        if project_id is not None:
            if not isinstance(project_id, str) or not project_id.strip():
                raise ValueError('project_id must be a non-empty project ID')
            return project_id
        _require_project()
        assert global_state._project_id is not None
        return global_state._project_id

    @classmethod
    def _response_by_name(
        cls,
        name: str,
        project_id: Optional[str] = None,
    ) -> ConnectorResponse:
        resolved_project_id = (cls._resolve_project_id(project_id)
                               if project_id is not None else None)
        if resolved_project_id is None:
            _project_scope()
        response = cloud_backend().connector.get_by_name(
            name, project_id=resolved_project_id)
        if response is None:
            scope = (f"project {resolved_project_id!r}" if resolved_project_id
                     is not None else 'the active project')
            raise ValueError(f"Connector {name!r} does not exist in {scope}.")
        return response

    def _delete_connector(self) -> None:
        cloud_backend().connector.delete_if_exists(
            self.id, project_id=self.kumo_project_id)
