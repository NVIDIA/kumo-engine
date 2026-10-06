# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

from kumoapi.data_source import (
    CreateIdentityArgs,
    IdentityGroupsResponse,
    IdentityResponse,
    ReplaceIdentityGroupsArgs,
    RotateIdentityCredentialsArgs,
    SnowflakeAuthenticationKind,
    UpdateIdentityArgs,
)
from kumoapi.json_serde import to_json_dict

from kumoai_studio.client.endpoints import HTTPMethod, IdentityEndpoints
from kumoai_studio.client.transport import Transport
from kumoai_studio.client.utils import parse_response, raise_on_error
from kumoai_studio.identity import Identity


class IdentityAPI:
    r"""Identity discovery plus administrator lifecycle management.

    Project-scoped discovery (:meth:`list`, :meth:`get_by_slug`) is available
    to any project member. The lifecycle methods (:meth:`create`,
    :meth:`update`, :meth:`delete`, :meth:`rotate_credentials`,
    :meth:`test_credentials`, :meth:`list_groups`, :meth:`replace_groups`, and
    :meth:`list_admin`) mirror the UI's administrator surface and require a
    customer-admin API key; the server enforces the gate.

    In all cases this API returns only safe identity metadata: it never exposes
    stored credentials or short-lived provider tokens. BigQuery and GCS results
    expose ``config.authentication.mode`` so users can distinguish preferred
    workload identity from server-stored service-account-key authentication
    without receiving either mode's credentials.
    """
    def __init__(self, client: Transport) -> None:
        self._client = client

    @staticmethod
    def _project_id(project_id: Optional[str]) -> str:
        if project_id is not None:
            return project_id
        from kumoai_studio import _require_project, global_state
        _require_project()
        assert global_state._project_id is not None
        return global_state._project_id

    def list(self, project_id: Optional[str] = None) -> List[Identity]:
        r"""List identities available through a project's owning group.

        Args:
            project_id: Immutable ID of a project visible to the caller, or
                ``None`` for the active project. The service scopes identities
                to a project and cannot answer without one, so ``None`` with no
                project set raises rather than reaching the server.

        Returns:
            Immutable, credential-free :class:`~kumoai_studio.identity.Identity`
            values that may be used by connectors in the project.
        """
        response = self._client._request(
            IdentityEndpoints.list(self._project_id(project_id)))
        raise_on_error(response)
        identities = parse_response(List[IdentityResponse], response)
        return [Identity._from_response(identity) for identity in identities]

    def get_by_slug(self, project_id: Optional[str], slug: str) -> Identity:
        r"""Get one available identity by exact canonical slug.

        ``project_id`` follows :meth:`list`: ``None`` means the active project.
        """
        response = self._client._request(
            IdentityEndpoints.by_slug(self._project_id(project_id), slug))
        raise_on_error(response)
        return Identity._from_response(
            parse_response(IdentityResponse, response))

    # -- Administrator lifecycle (customer-admin API key only) -------------

    def create(self, args: CreateIdentityArgs) -> Identity:
        r"""Create an identity and its initial group bindings.

        Requires a customer-admin API key. The returned :class:`Identity` is
        credential-free; stored secrets never leave the server.
        """
        _refuse_ambient_container_identity(args)
        response = self._client._request(
            IdentityEndpoints.collection(HTTPMethod.POST),
            json=to_json_dict(args, insecure=True),
        )
        raise_on_error(response)
        return Identity._from_response(
            parse_response(IdentityResponse, response))

    def list_admin(self) -> List[Identity]:
        r"""List every identity in the customer tenant (admin only)."""
        response = self._client._request(
            IdentityEndpoints.collection(HTTPMethod.GET))
        raise_on_error(response)
        identities = parse_response(List[IdentityResponse], response)
        return [Identity._from_response(identity) for identity in identities]

    def get(self, identity_id: str) -> Identity:
        r"""Get any identity in the tenant by immutable ID (admin only)."""
        response = self._client._request(
            IdentityEndpoints.item(identity_id, HTTPMethod.GET))
        raise_on_error(response)
        return Identity._from_response(
            parse_response(IdentityResponse, response))

    def update(self, identity_id: str, args: UpdateIdentityArgs) -> Identity:
        r"""Update an identity's display name and/or description."""
        response = self._client._request(
            IdentityEndpoints.item(identity_id, HTTPMethod.PATCH),
            json=to_json_dict(args),
        )
        raise_on_error(response)
        return Identity._from_response(
            parse_response(IdentityResponse, response))

    def delete(self, identity_id: str) -> Identity:
        r"""Delete an identity. Fails if a connector still references it."""
        response = self._client._request(
            IdentityEndpoints.item(identity_id, HTTPMethod.DELETE))
        raise_on_error(response)
        return Identity._from_response(
            parse_response(IdentityResponse, response))

    def rotate_credentials(self, identity_id: str,
                           args: RotateIdentityCredentialsArgs) -> Identity:
        r"""Replace an identity's stored secret material."""
        response = self._client._request(
            IdentityEndpoints.credentials(identity_id),
            json=to_json_dict(args, insecure=True),
        )
        raise_on_error(response)
        return Identity._from_response(
            parse_response(IdentityResponse, response))

    def test_credentials(self, identity_id: str) -> Identity:
        r"""Validate an identity's credentials against its provider."""
        response = self._client._request(IdentityEndpoints.test(identity_id))
        raise_on_error(response)
        return Identity._from_response(
            parse_response(IdentityResponse, response))

    def list_groups(self, identity_id: str) -> IdentityGroupsResponse:
        r"""List the Kumo group IDs bound to an identity."""
        response = self._client._request(
            IdentityEndpoints.groups(identity_id, HTTPMethod.GET))
        raise_on_error(response)
        return parse_response(IdentityGroupsResponse, response)

    def replace_groups(
            self, identity_id: str,
            args: ReplaceIdentityGroupsArgs) -> IdentityGroupsResponse:
        r"""Replace the complete set of groups bound to an identity."""
        response = self._client._request(
            IdentityEndpoints.groups(identity_id, HTTPMethod.PUT),
            json=to_json_dict(args),
        )
        raise_on_error(response)
        return parse_response(IdentityGroupsResponse, response)


def _refuse_ambient_container_identity(args: CreateIdentityArgs) -> None:
    r"""Refuses a kind the Kumo platform can never authenticate with.

    A Snowflake ``OAUTH_TOKEN_FILE`` identity reads a token that Snowpark
    Container Services mounts into the *running container*. The Kumo platform
    is not that container and has no such file, so an identity created this
    way could never connect. The server refuses it too -- this check exists to
    say so before a round trip.
    """
    authentication = getattr(args.config, 'authentication', None)
    if authentication is SnowflakeAuthenticationKind.OAUTH_TOKEN_FILE:
        raise ValueError(
            f"Identity {args.display_name!r} authenticates with a "
            f"container-mounted Snowflake OAuth token, which only a runtime "
            f"inside Snowpark Container Services can read. The Kumo platform "
            f"cannot, so it does not accept this kind. Create a key-pair or "
            f"password identity for the platform instead.")
