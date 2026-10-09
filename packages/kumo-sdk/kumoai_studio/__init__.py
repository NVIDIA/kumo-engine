# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# isort: skip_file
# NOTE Import order below is load-bearing (submodule init side effects); keep
# it as-is rather than letting isort reorder.
from functools import lru_cache
import warnings
import os
import re
import sys
import threading
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Union
import logging

from kumoapi.typing import Dtype, Stype
from kumoapi.data_source import (
    DEFAULT_GROUP_ID,
    DEFAULT_GROUP_SLUG,
    DEFAULT_PROJECT_ID,
    CreateIdentityArgs,
    DataSourceType,
    IdentityGroupsResponse,
    ReplaceIdentityGroupsArgs,
    RotateIdentityCredentialsArgs,
    S3AccessKeyAuthentication,
    S3AccessKeyCredentials,
    S3AddressingStyle,
    S3ConnectionConfig,
    S3IdentityConfig,
    UpdateIdentityArgs,
    validate_resource_slug,
)

from kumoai_studio.identity import Identity, s3_access_key_credentials
from kumoai_studio.client.client import KumoClient
from kumoai_studio._logging import initialize_logging, _ENV_KUMO_LOG
from kumoai_studio._singleton import Singleton
from kumoai_studio._version import __version__
from kumoai_studio.futures import initialize_event_loop

initialize_logging()
initialize_event_loop()


@dataclass
class GlobalState(metaclass=Singleton):
    r"""Global storage of the state needed to create a Kumo client object. A
    singleton so its initialized state can be referenced elsewhere for free.
    """

    # NOTE fork semantics: CoW on Linux, and re-execed on Windows. So this will
    # likely not work on Windows unless we have special handling for the shared
    # state:
    _url: Optional[str] = None
    _api_key: Optional[str] = None

    # Every SDK operation runs in one exact project context. The group is
    # retained only as safe project metadata and is never sent as connector
    # authorization input.
    _project_id: Optional[str] = None
    _project_slug: Optional[str] = None
    _project_name: Optional[str] = None
    _project_group_id: Optional[str] = None

    thread_local: threading.local = field(default_factory=threading.local)

    def clear(self) -> None:
        if hasattr(self.thread_local, '_client'):
            del self.thread_local._client
        self._url = None
        self._api_key = None
        self._project_id = None
        self._project_slug = None
        self._project_name = None
        self._project_group_id = None

    @property
    def initialized(self) -> bool:
        return self._url is not None and self._api_key is not None

    @property
    def client(self) -> KumoClient:
        r"""Accessor for the Kumo client. Note that clients are stored as
        thread-local variables as the requests Session library is not
        guaranteed to be thread-safe.

        For more information, see https://github.com/psf/requests/issues/1871.
        """
        if self._url is None or self._api_key is None:
            raise ValueError("Client creation or authentication failed. "
                             "Please re-create your client before proceeding.")

        if hasattr(self.thread_local, '_client'):
            return self.thread_local._client

        client = KumoClient(self._url, self._api_key)
        self.thread_local._client = client
        return client


global_state: GlobalState = GlobalState()


def init(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    log_level: str = "INFO",
    project: Optional[str] = None,
) -> None:
    r"""Initialize the SDK and select the project for subsequent operations.

    Initialization authenticates the caller and resolves exactly one visible
    project. Connector, table, graph, predictive-query, and job operations are
    scoped to that project. If ``project`` is omitted, Kumo selects the
    canonical Default project by its literal immutable ID, ``"default"``; it
    never selects a project based on how many projects the caller can see.

    Example:
        >>> import kumoai_studio
        >>> kumoai_studio.init(url="<api_url>", api_key="<api_key>")  # doctest: +SKIP
        >>> # Select a non-default project by stable slug:
        >>> kumoai_studio.init(  # doctest: +SKIP
        ...     url="<api_url>",
        ...     api_key="<api_key>",
        ...     project="fraud-detection",
        ... )

    Args:
        url: The Kumo API endpoint. Can also be provided via the
            ``KUMO_API_ENDPOINT`` envronment variable. Will be inferred from
            the provided API key, if not provided.
        api_key: The Kumo API key. Can also be provided via the
            ``KUMO_API_KEY`` environment variable.
        log_level: The logging level that Kumo operates under. Defaults to
            INFO; for more information, please see
            :class:`~kumoai_studio.set_log_level`. Can also be set with the
            environment variable ``KUMOAI_LOG``.
        project: Project immutable ID, exact slug, or display name. When
            omitted, the canonical Kumo Default project is selected by its
            literal immutable ID, ``"default"``.

    Raises:
        :class:`ValueError`: If authentication fails, the endpoint cannot be
            reached, or a requested project name is missing.
        :class:`~kumoai_studio.exceptions.HTTPException`: If the project lookup is
            rejected by the server, including when an ID is inaccessible.
        :class:`RuntimeError`: If the server returns duplicate normalized
            project names, or the selector is ambiguous because it is a slug
            for one visible project and the display name of another.

    Note:
        Authentication and project selection are atomic from the caller's
        perspective. If either step fails, no partial SDK session is retained,
        so the caller may correct the input and call :func:`init` again.
    """  # noqa
    # Avoid mutations to the global state after it is set:
    if global_state.initialized:
        warnings.warn("Kumo SDK already initialized. To re-initialize the "
                      "SDK, please start a new interpreter. No changes will "
                      "be made to the current session.")
        return

    set_log_level(os.getenv(_ENV_KUMO_LOG, log_level))

    # Get API key:
    api_key = api_key or os.getenv("KUMO_API_KEY")

    if api_key is None:
        raise ValueError(
            "Kumo SDK initialization failed since no API key was provided. "
            "Please either set the 'KUMO_API_KEY' environment variable or "
            "explicitly call `kumoai_studio.init(...)`.")

    # Get or infer URL:
    url = url or os.getenv("KUMO_API_ENDPOINT")
    url = url or f"https://{api_key.split(':')[0]}.kumoai.cloud/api"
    if url is None:
        raise ValueError("Kumo SDK initialization failed since no endpoint "
                         "URL was provided. Please either set the "
                         "'KUMO_API_ENDPOINT' environment variable or "
                         "explicitly call `kumoai_studio.init(...)`.")

    # Assign global state after verification that client can be created and
    # authenticated successfully:
    client = KumoClient(url=url, api_key=api_key)
    try:
        client.authenticate()
        selected_project = _resolve_project(client, project)
    except Exception:
        # Authentication and project selection are one public operation. A
        # failed lookup must never leave a partially initialized SDK that
        # cannot be retried in the same interpreter.
        global_state.clear()
        raise

    global_state._url = client._url
    global_state._api_key = client._api_key
    _set_active_project(selected_project)

    logger = logging.getLogger('kumoai_studio')
    log_level = logging.getLevelName(logger.getEffectiveLevel())

    # Endpoints can contain HTTP user info, signed query parameters, or other
    # deployment-specific secrets. Do not include the caller-provided URL in
    # logs, even after a successful authentication.
    logger.info("Initialized Kumo SDK v%s", __version__)


def _init_project(
    client: KumoClient,
    project: Optional[str],
) -> None:
    r"""Resolve one exact visible project without cardinality fallbacks."""
    _set_active_project(_resolve_project(client, project))


def _resolve_project(client: KumoClient, project: Optional[str]) -> 'Project':
    if project is None:
        match = client.projects_api.get_project(DEFAULT_PROJECT_ID)
        logging.getLogger('kumoai_studio').info(
            "Selected canonical default project: '%s'", match.name)
    elif not project.strip():
        raise ValueError(
            'project must be a non-empty ID, slug, or display name')
    elif _is_project_id(project):
        match = client.projects_api.get_project(project)
    else:
        slug_match = None
        if _is_canonical_slug(project):
            from kumoai_studio.exceptions import HTTPException
            try:
                slug_match = client.projects_api.get_project_by_slug(project)
            except HTTPException as error:
                if error.status_code != 404:
                    raise
        normalized_project = _normalize_name(project)
        name_matches = [
            candidate for candidate in client.projects_api.list_projects()
            if _normalize_name(candidate.name) == normalized_project
        ]
        if len(name_matches) > 1:
            raise RuntimeError(
                f"Server returned duplicate project name {project!r}.")
        if (slug_match is not None and name_matches
                and name_matches[0].id != slug_match.id):
            raise RuntimeError(
                f"Project selector {project!r} is ambiguous between a slug "
                "and display name.")
        if slug_match is not None:
            match = slug_match
        elif name_matches:
            match = name_matches[0]
        else:
            raise ValueError(
                f"Project {project!r} was not found or is not accessible.")

    return match


def _set_active_project(project: 'Project') -> None:
    global_state._project_id = project.id
    global_state._project_slug = project.slug
    global_state._project_name = project.name
    global_state._project_group_id = project.group_id


def _normalize_name(value: str) -> str:
    return unicodedata.normalize('NFKC', value.strip()).casefold()


def _is_project_id(value: str) -> bool:
    return (value == DEFAULT_PROJECT_ID
            or re.fullmatch(r'project_[a-f0-9]{32}', value) is not None)


def _is_group_id(value: str) -> bool:
    return re.fullmatch(r'group_[a-f0-9]{32}', value) is not None


def _is_identity_id(value: str) -> bool:
    return re.fullmatch(r'identity_[a-f0-9]{32}', value) is not None


def _is_canonical_slug(value: str) -> bool:
    try:
        validate_resource_slug(value, allow_reserved=True)
    except ValueError:
        return False
    return True


def _require_project() -> None:
    from kumoai_studio.exceptions import ProjectNotSetError
    if global_state._project_id is None:
        raise ProjectNotSetError()


def _project_scope() -> Optional[str]:
    r"""Returns the project the active backend scopes resources to.

    The Kumo platform scopes identities to a project and cannot answer without
    one, so the familiar :class:`~kumoai_studio.exceptions.ProjectNotSetError` is
    still raised there. An in-process backend has no projects at all, and
    demanding one would report a missing scope that does not exist rather than
    the real state of the runtime.
    """
    from kumoai_studio.backend import CloudBackend, cloud_backend
    if isinstance(cloud_backend(), CloudBackend):
        _require_project()
    return global_state._project_id


def list_groups() -> 'List[Group]':
    r"""List native Kumo groups visible to the authenticated user.

    Groups determine project membership and which administrator-managed
    identities are available to a project. Callers do not select a group when
    creating a connector; Kumo derives it from the connector's project.

    Returns:
        A list of credential-free :class:`~kumoai_studio.project.Group` values.
    """
    return global_state.client.groups_api.list_groups()


def list_projects() -> 'List[Project]':
    r"""List every project visible to the authenticated user.

    Returns:
        A list of :class:`~kumoai_studio.project.Project` values. This includes the
        canonical Default project and any shared or private projects the user
        can access.
    """
    return global_state.client.projects_api.list_projects()


def get_project(project: str) -> 'Project':
    r"""Return one visible project by immutable ID, slug, or display name.

    This is a non-mutating lookup: it does not change the project returned by
    :func:`current_project`. Immutable IDs—including the canonical Default ID
    ``"default"``—and exact canonical slugs use explicit lookup endpoints.
    Display names use the same whitespace, Unicode, and case normalization as
    :func:`init` and :func:`set_project`. If a selector is a slug for one
    project and display name for another, lookup fails as ambiguous.

    Args:
        project: Immutable Kumo project ID, exact canonical slug, or visible
            display name. ``None`` and empty strings are not accepted.

    Returns:
        The matching credential-free :class:`~kumoai_studio.project.Project`.

    Raises:
        :class:`ValueError`: If ``project`` is empty, or a display name is not
            visible to the authenticated caller.
        :class:`~kumoai_studio.exceptions.HTTPException`: If an immutable project ID
            is nonexistent or inaccessible.
        :class:`RuntimeError`: If the server unexpectedly returns multiple
            projects with the same normalized display name, or the selector is
            ambiguous because it is a slug for one visible project and the
            display name of another.

    Example:
        >>> import kumoai_studio
        >>> kumoai_studio.init()  # The canonical Default project stays active.  # doctest: +SKIP
        >>> fraud = kumoai_studio.get_project("fraud-detection")  # doctest: +SKIP
        >>> assert kumoai_studio.current_project().name == "Default"  # doctest: +SKIP
        >>> connector = kumoai_studio.BigQueryConnector(  # doctest: +SKIP
        ...     name="events",
        ...     project_id="company-data-prod",
        ...     dataset_id="events",
        ...     kumo_project_id=fraud.id,
        ... )
    """  # noqa
    if project is None or not project.strip():
        raise ValueError(
            'project must be a non-empty ID, slug, or display name')
    return _resolve_project(global_state.client, project)


def set_project(project: 'Union[str, Project]') -> None:
    r"""Sets the active project context. All subsequent object operations will
    be scoped to this project.

    Args:
        project: A :class:`~kumoai_studio.project.Project` object, immutable ID,
            exact slug, or display name. IDs and slugs use direct endpoints;
            names are matched after whitespace, Unicode, and case
            normalization.

    Raises:
        :class:`ValueError`: If a project name cannot be found.
        :class:`~kumoai_studio.exceptions.HTTPException`: If lookup by project ID is
            rejected by the server, including when the project is inaccessible.
        :class:`RuntimeError`: If the server returns duplicate normalized
            project names, or the selector is ambiguous because it is a slug
            for one visible project and the display name of another.

    Example:
        >>> import kumoai_studio
        >>> kumoai_studio.set_project("fraud-detection")  # doctest: +SKIP
        >>> print(kumoai_studio.current_project().name)  # doctest: +SKIP
        Fraud Detection
    """
    from kumoai_studio.project import Project
    if isinstance(project, Project):
        match = project
    elif isinstance(project, str):
        match = get_project(project)
    else:
        raise TypeError('project must be a project ID, slug, name, or Project')
    _set_active_project(match)


def current_project() -> 'Project':
    r"""Return the project that scopes the current SDK session.

    Returns:
        The active :class:`~kumoai_studio.project.Project`, including its immutable
        human-facing :attr:`~kumoai_studio.project.Project.slug`.

    Raises:
        :class:`~kumoai_studio.exceptions.ProjectNotSetError`: if no project is set.
    """
    _require_project()
    from kumoai_studio.project import Project
    assert global_state._project_id is not None
    assert global_state._project_slug is not None
    assert global_state._project_name is not None
    assert global_state._project_group_id is not None
    return Project(
        id=global_state._project_id,
        slug=global_state._project_slug,
        name=global_state._project_name,
        group_id=global_state._project_group_id,
    )


def _resolve_group(group: str) -> 'Group':
    if not group.strip():
        raise ValueError('group must be a non-empty ID, slug, or display name')
    if _is_group_id(group):
        id_matches = [
            candidate for candidate in list_groups() if candidate.id == group
        ]
        if id_matches:
            return id_matches[0]
        raise ValueError(
            f"Group {group!r} was not found or is not accessible.")

    slug_match = None
    if _is_canonical_slug(group):
        from kumoai_studio.exceptions import HTTPException
        try:
            slug_match = global_state.client.groups_api.get_group_by_slug(
                group)
        except HTTPException as error:
            if error.status_code != 404:
                raise
    normalized_group = _normalize_name(group)
    name_matches = [
        candidate for candidate in list_groups()
        if _normalize_name(candidate.name) == normalized_group
    ]
    if len(name_matches) > 1:
        raise RuntimeError(f"Server returned duplicate group name {group!r}.")
    if (slug_match is not None and name_matches
            and name_matches[0].id != slug_match.id):
        raise RuntimeError(
            f"Group selector {group!r} is ambiguous between a slug and "
            "display name.")
    if slug_match is not None:
        return slug_match
    if name_matches:
        return name_matches[0]
    raise ValueError(f"Group {group!r} was not found or is not accessible.")


def create_project(name: str, group: 'Union[str, Group, None]' = None,
                   is_private: bool = False,
                   slug: Optional[str] = None) -> 'Project':
    r"""Create a project owned by a native Kumo group.

    Omitting ``group`` always creates the project in the canonical Default
    group. To create it in another group, pass a visible group object, display
    name, slug, or immutable ID explicitly. Creating a project does not switch
    the active project; call :func:`set_project` when desired.

    Args:
        name: The name for the new project.
        group: A visible group display name, exact slug, immutable ID, or
            :class:`~kumoai_studio.project.Group` object. When omitted, the canonical
            Default group is used by immutable ID.
        is_private: If ``True``, the project is visible only to its creator.
            If ``False`` (default), it is shared with all members of the group.
        slug: Optional immutable project short ID, such as
            ``"fraud-detection"``. When omitted, the server derives it once
            from ``name``. It never changes when the display name changes.

    Returns:
        The newly created :class:`~kumoai_studio.project.Project`.

    Raises:
        :class:`ValueError`: If the target group is missing or inaccessible.
        :class:`~kumoai_studio.exceptions.HTTPException`: If the server rejects the
            project name, group, visibility setting, or caller permissions.
        :class:`RuntimeError`: If the group selector is ambiguous between a
            slug and display name.

    Example:
        >>> project = kumoai_studio.create_project(  # doctest: +SKIP
        ...     "Fraud Detection",
        ...     group="risk-team",
        ...     slug="fraud-detection",
        ... )
        >>> kumoai_studio.set_project(project)  # doctest: +SKIP
    """
    if isinstance(group, Group):
        selected = group
    elif group is not None:
        selected = _resolve_group(group)
    else:
        selected = Group(id=DEFAULT_GROUP_ID, slug=DEFAULT_GROUP_SLUG,
                         name='Default')
    return global_state.client.projects_api.create_project(
        name=name,
        group_id=selected.id,
        is_private=is_private,
        slug=slug,
    )


def list_identities() -> List[Identity]:
    r"""List identities available to the active project.

    An identity represents an administrator-managed external principal, such
    as an AWS role or warehouse service account. The returned values contain
    safe configuration and credential status only; passwords, private keys,
    external IDs, access tokens, and temporary cloud credentials are never
    returned to the SDK.

    Returns:
        A list of immutable, credential-free :class:`~kumoai_studio.identity.Identity`
        values bound to the active project's group.

    Raises:
        :class:`~kumoai_studio.exceptions.ProjectNotSetError`: If no project is
            active.
    """
    return cloud_backend().identity.list(_project_scope())


def get_identity(name: str) -> Identity:
    r"""Get one identity available to the active project by selector.

    Opaque IDs are resolved directly, canonical slugs use the project-scoped
    slug endpoint, and other values are matched as normalized display names.
    If the same selector is a slug for one identity and display name for
    another, resolution fails safely as ambiguous. Credentials are never
    returned.

    Use this helper when a connector should select a specific identity by
    slug or display name. Passing a raw string to a connector's ``identity``
    argument remains an immutable identity ID, never a slug or display name.

    Args:
        name: Immutable ID, exact canonical slug, or display name of an
            identity available to the active project.

    Returns:
        The matching immutable, credential-free
        :class:`~kumoai_studio.identity.Identity` value.

    Raises:
        :class:`ValueError`: If the selector is empty, missing, or
            inaccessible.
        :class:`RuntimeError`: If the server returns duplicate normalized
            identity names, or the selector is ambiguous because it is a slug
            for one available identity and the display name of another.
        :class:`~kumoai_studio.exceptions.HTTPException`: If identity discovery is
            rejected by the server.

    Example:
        >>> identity = kumoai_studio.get_identity("prod-aws")  # doctest: +SKIP
        >>> connector = kumoai_studio.S3Connector(  # doctest: +SKIP
        ...     name="events",
        ...     root_dir="s3://company-data/events/",
        ...     identity=identity,
        ... )
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError('Identity selector cannot be empty.')
    _project_scope()
    if _is_identity_id(name):
        id_matches = [
            identity for identity in list_identities() if identity.id == name
        ]
        if id_matches:
            return id_matches[0]
        raise ValueError(
            f"Identity {name!r} is not available to the active project.")

    slug_match = None
    if _is_canonical_slug(name):
        from kumoai_studio.exceptions import HTTPException
        try:
            slug_match = cloud_backend().identity.get_by_slug(
                global_state._project_id, name)
        except HTTPException as error:
            if error.status_code != 404:
                raise
    normalized_name = _normalize_name(name)
    name_matches = [
        identity for identity in list_identities()
        if _normalize_name(identity.name) == normalized_name
    ]
    if len(name_matches) > 1:
        raise RuntimeError(
            f"Server returned duplicate identity name {name!r}.")
    if (slug_match is not None and name_matches
            and name_matches[0].id != slug_match.id):
        raise RuntimeError(
            f"Identity selector {name!r} is ambiguous between a slug and "
            "display name.")
    if slug_match is not None:
        return slug_match
    if name_matches:
        return name_matches[0]
    raise ValueError(
        f"Identity {name!r} is not available to the active project.")


def create_identity(args: CreateIdentityArgs) -> Identity:
    r"""Create an administrator-managed identity (admin API key required).

    Identities hold the external authentication that connectors reference.
    Creating one is an administrator action; the API key must belong to a
    customer administrator. The returned :class:`~kumoai_studio.identity.Identity`
    is credential-free.

    Args:
        args: Display name, provider ``config``, optional ``credentials``,
            optional ``group_ids`` bindings, and optional ``slug``.

    Returns:
        The persisted, credential-free identity.

    Raises:
        :class:`~kumoai_studio.exceptions.HTTPException`: If the caller is not a
            customer administrator, or the identity conflicts with an existing
            one.

    Example:
        >>> from kumoapi.data_source import (  # doctest: +SKIP
        ...     CreateIdentityArgs,
        ...     S3IdentityConfig,
        ...     S3AssumeRoleAuthentication,
        ... )
        >>> args = CreateIdentityArgs(  # doctest: +SKIP
        ...     display_name="prod-aws",
        ...     config=S3IdentityConfig(
        ...         authentication=S3AssumeRoleAuthentication(
        ...             partition="aws",
        ...             role_arn="arn:aws:iam::123456789012:role/prod")),
        ... )
        >>> identity = kumoai_studio.create_identity(args)  # doctest: +SKIP
    """
    return cloud_backend().identity.create(args)


def list_all_identities() -> List[Identity]:
    r"""List every identity in the customer tenant (admin API key required).

    Unlike :func:`list_identities`, which is scoped to the active project's
    group, this returns all identities and requires a customer administrator
    API key. Returned values are credential-free.
    """
    return cloud_backend().identity.list_admin()


def update_identity(
    identity_id: str,
    *,
    display_name: Optional[str] = None,
    description: Optional[str] = None,
) -> Identity:
    r"""Update an identity's display name and/or description (admin only).

    Args:
        identity_id: Immutable identity ID.
        display_name: New display name, or ``None`` to leave unchanged.
        description: New description, or ``None`` to leave unchanged.
    """
    return cloud_backend().identity.update(
        identity_id,
        UpdateIdentityArgs(display_name=display_name, description=description),
    )


def delete_identity(identity_id: str) -> Identity:
    r"""Delete an identity by immutable ID (admin only).

    The server rejects deletion while any connector still references the
    identity.

    Args:
        identity_id: Immutable identity ID.
    """
    return cloud_backend().identity.delete(identity_id)


def rotate_identity_credentials(
    identity_id: str,
    args: RotateIdentityCredentialsArgs,
) -> Identity:
    r"""Replace the authentication behind an identity (admin only).

    Rotation replaces the secret material behind a name and leaves the
    principal and authentication kind it was created with alone. It means the
    same thing on both backends; only where the material comes to rest
    differs, encrypted on the Kumo platform and in memory in process.

    An identity that stores no material -- a Snowflake ``OAUTH_TOKEN_FILE`` or
    a Google workload identity -- has nothing to rotate, since the token it
    authenticates with is ambient.

    Args:
        identity_id: Immutable identity ID.
        args: New credentials, plus Google key metadata when rotating a Google
            service-account key.
    """
    return cloud_backend().identity.rotate_credentials(identity_id, args)


def test_identity_credentials(identity_id: str) -> Identity:
    r"""Validate an identity's credentials against its provider (admin only).

    Args:
        identity_id: Immutable identity ID.

    Returns:
        The identity with refreshed credential status.
    """
    return cloud_backend().identity.test_credentials(identity_id)


def get_identity_groups(identity_id: str) -> IdentityGroupsResponse:
    r"""List the Kumo group IDs bound to an identity (admin only).

    Args:
        identity_id: Immutable identity ID.
    """
    return cloud_backend().identity.list_groups(identity_id)


def set_identity_groups(
    identity_id: str,
    group_ids: List[str],
) -> IdentityGroupsResponse:
    r"""Replace the complete set of groups bound to an identity (admin only).

    Args:
        identity_id: Immutable identity ID.
        group_ids: The full desired set of group IDs; existing bindings not in
            this set are removed.
    """
    return cloud_backend().identity.replace_groups(
        identity_id, ReplaceIdentityGroupsArgs(group_ids=group_ids))


def set_log_level(level: str) -> None:
    r"""Sets the Kumo logging level, which defines the amount of output that
    methods produce.

    Example:
        >>> import kumoai_studio
        >>> kumoai_studio.set_log_level("INFO")  # doctest: +SKIP

    Args:
        level: the logging level. Can be one of (in order of lowest to highest
            log output) :obj:`DEBUG`, :obj:`INFO`, :obj:`WARNING`,
            :obj:`ERROR`, :obj:`FATAL`, :obj:`CRITICAL`.
    """
    # logging library will ensure `level` is a valid string, and raise a
    # warning if not:
    logging.getLogger('kumoai_studio').setLevel(level)


# Try to initialize purely with environment variables:
if ("pytest" not in sys.modules and "KUMO_API_KEY" in os.environ
        and "KUMO_API_ENDPOINT" in os.environ):
    init()

import kumoai_studio.connector  # noqa
import kumoai_studio.encoder  # noqa
import kumoai_studio.graph  # noqa
import kumoai_studio.pquery  # noqa
import kumoai_studio.trainer  # noqa
import kumoai_studio.utils  # noqa
import kumoai_studio.databricks  # noqa

from kumoapi.source_table import LocalFileSourceTable  # noqa
from kumoai_studio.connector import (  # noqa
    SourceTable, SQLSourceTable, S3Connector, GCSConnector, ADLSConnector,
    SnowflakeConnector, DatabricksConnector, BigQueryConnector,
    FileUploadConnector)
# Backend selection. Exported here because choosing where a program executes
# is a top-level concern.
from kumoai_studio.backend import Backend, CloudBackend, cloud_backend  # noqa
# The source and sink backends are deliberately absent above: Snowflake,
# Databricks, BigQuery, S3 and GCS each ship their reader and sink from their
# own distribution now (`kumo_snowflake.SnowflakeSourceBackend`,
# `kumo_s3.S3SourceBackend`, ...), not from `kumoai_studio`, so there is nothing
# left here to resolve lazily -- import the one a caller needs from its own
# package instead.
from kumoai_studio.project import Group, Project  # noqa
from kumoai_studio.exceptions import ProjectNotSetError  # noqa
from kumoai_studio.graph import Column, Edge, Graph, Table  # noqa
from kumoai_studio.pquery import (  # noqa
    PredictionTableGenerationPlan, PredictiveQuery,
    TrainingTableGenerationPlan, TrainingTable, TrainingTableJob,
    PredictionTable, PredictionTableJob)
from kumoai_studio.trainer import (  # noqa
    ModelPlan, Trainer, TrainingJobResult, TrainingJob,
    BatchPredictionJobResult, BatchPredictionJob, ModelExportConfig,
    ModelImportConfig, ModelRegistryLocation, export_model)
# Last, and from its own module: importing a model produces a `Trainer` and a
# `PredictiveQuery`, so it can only be defined once both exist.
from kumoai_studio.model_import import import_model, list_published_models  # noqa: E402

__all__ = [
    'Dtype',
    'Stype',
    # Named when registering a reader or a sink on a local backend, which is
    # what makes the mapping's key the caller's rather than the reader's.
    'DataSourceType',
    'SourceTable',
    'SQLSourceTable',
    'LocalFileSourceTable',
    'S3Connector',
    'GCSConnector',
    'ADLSConnector',
    'SnowflakeConnector',
    'DatabricksConnector',
    'BigQueryConnector',
    'FileUploadConnector',
    'Identity',
    # Backend selection
    'Backend',
    'CloudBackend',
    'cloud_backend',
    'Column',
    'Table',
    'Graph',
    'Edge',
    'PredictiveQuery',
    'TrainingTable',
    'TrainingTableJob',
    'TrainingTableGenerationPlan',
    'PredictionTable',
    'PredictionTableJob',
    'PredictionTableGenerationPlan',
    'Trainer',
    'TrainingJobResult',
    'TrainingJob',
    'BatchPredictionJobResult',
    'BatchPredictionJob',
    'ModelPlan',
    'ModelExportConfig',
    'ModelImportConfig',
    'import_model',
    'list_published_models',
    'ModelRegistryLocation',
    'export_model',
    # Project and group authorization
    'Group',
    'Project',
    'ProjectNotSetError',
    'CreateIdentityArgs',
    'RotateIdentityCredentialsArgs',
    'S3AccessKeyAuthentication',
    'S3AccessKeyCredentials',
    'S3AddressingStyle',
    'S3ConnectionConfig',
    'S3IdentityConfig',
    's3_access_key_credentials',
    'list_groups',
    'list_projects',
    'get_project',
    'set_project',
    'create_project',
    'current_project',
    'list_identities',
    'get_identity',
    'create_identity',
    'list_all_identities',
    'update_identity',
    'delete_identity',
    'rotate_identity_credentials',
    'test_identity_credentials',
    'get_identity_groups',
    'set_identity_groups',
    '__version__',
]


@lru_cache
def in_streamlit_notebook() -> bool:
    try:
        from snowflake.snowpark.context import get_active_session
        import streamlit  # noqa: F401
        get_active_session()
        return True
    except Exception:
        return False


@lru_cache
def in_jupyter_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if 'google.colab' in str(shell.__class__):
            return True
        if 'DatabricksShell' in str(shell.__class__):
            return True
        return shell.__class__.__name__ == 'ZMQInteractiveShell'
    except Exception:
        return False


@lru_cache
def in_vnext_notebook() -> bool:
    try:
        from snowflake.snowpark.context import get_active_session
        get_active_session()
        return in_jupyter_notebook()
    except Exception:
        return False


@lru_cache
def in_notebook() -> bool:
    return in_streamlit_notebook() or in_jupyter_notebook()


@lru_cache
def in_tmux() -> bool:
    return 'TMUX' in os.environ or 'HERDR_ENV' in os.environ


ascii_logo = """
@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@  @@@@@@@@@@@
@@@@@@@@  @@@@@@@@@@@
@@@@@@@@  @%  %  @@@@
@@@@@@@@  %  %@  @@@@
@@@@@@@@    .@@  @@@@
@@@@@@@@  @. .@: .@@.
@@@@@@@@  @@. .@+
@@@@@@@@@@@@@@@@@@@@@
"""[1:-1]
