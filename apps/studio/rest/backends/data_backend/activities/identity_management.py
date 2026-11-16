# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import kumoapi.data_source as api

from common_utils.error import PublicException
from rest.backends.data_backend.connector_activity_registry import (
    connector_authorized_activity,
)
from rest.backends.data_backend.connector_broker import (
    connector_credential_broker,
)
from rest.config.data_backend import ActivityAuthorizationRef
from statedb.config.common import IdentityID, ProjectID, UserID
from statedb.db.connector_authorizations import canonical_request_digest
from statedb.db.services import StateServices
from statedb.rest.connector_authorization import (
    ConnectorAuthorizationResourceKind,
    ConnectorCapabilityAction,
)
from statedb.utils.errors import ConnectorNameConflictError


@connector_authorized_activity
def create_project_connector_activity(
    project_id: ProjectID,
    args: api.CreateConnectorArgs,
    authorizations: dict[IdentityID, ActivityAuthorizationRef],
) -> api.ConnectorResponse:
    """Validate and insert a connector without transporting credentials."""
    try:
        digest = canonical_request_digest(args)
        authorizations_by_identity = {
            identity_id:
            connector_credential_broker.authorize_project_activity_scope(
                reference,
                resource_kind=(ConnectorAuthorizationResourceKind.
                               PROJECT_CONNECTOR_CREATE),
                resource_id=str(identity_id),
                project_id=str(project_id),
                request_digest=digest,
                action=ConnectorCapabilityAction.CREATE_PROJECT_CONNECTOR,
            )
            for identity_id, reference in authorizations.items()
        }
        owners = {(authorization.principal_id, authorization.auth_backend)
                  for authorization in authorizations_by_identity.values()}
        if len(owners) != 1:
            raise PermissionError(
                'Connector authorizations have different owners')
        accepted_submitter_id, _ = next(iter(owners))

        async def create(services: StateServices) -> api.ConnectorResponse:
            return await services.connectors.create_from_activity(
                project_id,
                args,
                UserID(accepted_submitter_id),
                set(authorizations),
            )

        return connector_credential_broker.run(create)
    except PublicException as error:
        # Only the repository's concrete type may keep a name-conflict payload;
        # strip ValueError-compatible and same-name impostors before Temporal.
        if type(error) is ConnectorNameConflictError:
            raise
        if (not isinstance(error, ValueError) and type(error).__name__
                != ConnectorNameConflictError.__name__):
            raise
    raise ValueError


@connector_authorized_activity
def validate_identity_credentials_activity(
    identity_id: IdentityID,
    authorization: ActivityAuthorizationRef,
) -> api.IdentityResponse:
    """Load, decrypt or exchange, and test a stored identity in an activity."""
    digest = canonical_request_digest({
        'operation': 'validate_identity',
        'identity_id': str(identity_id),
    })
    principal = connector_credential_broker.authorize_activity_scope(
        authorization,
        resource_kind=ConnectorAuthorizationResourceKind.IDENTITY_VALIDATE,
        resource_id=str(identity_id),
        project_id='',
        request_digest=digest,
        action=ConnectorCapabilityAction.VALIDATE_IDENTITY,
    )

    async def test(services: StateServices) -> api.IdentityResponse:
        return await services.identities.test_credentials_from_activity(
            identity_id, principal)

    return connector_credential_broker.run(test)
