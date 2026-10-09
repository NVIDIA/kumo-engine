# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from dataclasses import replace
from enum import Enum
from http import HTTPStatus
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from fastapi import Request, Response
from fastapi.exceptions import HTTPException

from async_utils.instrument import measure_blocking_code
from io_utils import kumo_dataclass, logging
from rest.config.api_key import ParsedAPIKey
from rest.config.authentication import AuthenticationSettings
from rest.config.status import RestStatusCode
from rest.deps.context import _current_user_ctx
from rest.deps.group_membership import ResolvedGroupMembership
from rest.deps.project_resolver import ProjectResolver
from rest.driver.api_key_driver import APIKeyDriver
from rest.driver.direct_connect_auth_storage import DirectConnectAuthStore
from statedb.config.common import ProjectID, UserID
from statedb.config.timestamp import now_utc

log = logging.getLogger(__name__)


class Permissions(Enum):
    READ = 'read'
    WRITE = 'write'
    ADMIN = 'admin'

    def compose(self) -> str:
        return self.value


class GroupResolverProtocol(Protocol):
    async def resolve_groups(
        self,
        user_id: str,
        auth_context: Optional[Dict[str, Any]],
    ) -> ResolvedGroupMembership:
        ...


AUTHORIZATION_DISABLED = '_AUTHORIZATION_DISABLED_'
AUTHORIZATION_PROXIED = '_AUTHORIZATION_PROXIED_'
AUTHORIZATION_API_KEY = '_AUTHORIZATION_API_KEY_'
AUTHORIZATION_DIRECT_CONNECT = '_AUTHORIZATION_DIRECT_CONNECT_'
DIRECT_CONNECT_SESSION_COOKIE = 'kumo_direct_connect_session'

_CACHE_IDENTITY_KEYS = ("auth_source", )


class _AuthorizationMode(Enum):
    AUTHENTICATED = 'authenticated'
    DISABLED = 'disabled'
    PROXIED = 'proxied'


@kumo_dataclass(frozen=True)
class UserAuthorization:
    user_id: UserID
    access_token: str
    permissions: Sequence[str]
    # Resolved Kumo group IDs for this request context.
    kumo_group_ids: Tuple[str, ...] = ()
    # True only when a configured external membership adapter could not
    # provide a fresh or unexpired cached result for this request.
    external_membership_unavailable: bool = False
    # Resolved project IDs visible to this user for this request context.
    project_ids: Tuple[str, ...] = ()
    # Human-readable display name.
    user_name: str = ""

    # If there was an error logging in the user, it is listed here.
    error: Optional[str] = None

    @classmethod
    def disabled(cls) -> 'UserAuthorization':
        """
        Represets a UserAuthorization object when Authentication/Authorization
        is disabled (we want to bypass). The role should have all permissions
        enabled.
        """
        perms = [
            Permissions.READ.compose(),
            Permissions.WRITE.compose(),
            Permissions.ADMIN.compose(),
        ]
        return _DisabledUserAuthorization(
            user_id=UserID(AUTHORIZATION_DISABLED),
            access_token=AUTHORIZATION_DISABLED,
            permissions=perms,
        )

    @classmethod
    def disabled_request(cls) -> 'UserAuthorization':
        """Principal for auth-disabled requests."""
        return _DisabledUserAuthorization(
            user_id=UserID(AUTHORIZATION_DISABLED),
            access_token=AUTHORIZATION_DISABLED,
            permissions=[
                Permissions.READ.compose(),
                Permissions.WRITE.compose(),
                Permissions.ADMIN.compose(),
            ],
        )

    @classmethod
    def proxied(cls) -> 'UserAuthorization':
        """
        Represets a UserAuthorization object when Authentication/Authorization
        is disabled (we want to bypass). The role should have basic permissions
        enabled.
        """
        perms = [
            Permissions.READ.compose(),
            Permissions.WRITE.compose(),
        ]
        return _ProxiedUserAuthorization(
            user_id=UserID(AUTHORIZATION_PROXIED),
            access_token=AUTHORIZATION_PROXIED,
            permissions=perms,
        )

    @classmethod
    def not_authorized(cls, error: str) -> 'UserAuthorization':
        """
        Represets a UserAuthorization object for a user that is not logged in
        (authentication failed).
        """
        return cls(
            user_id=UserID(""),
            access_token="",
            permissions=[],
            error=error,
        )

    @property
    def is_disabled(self) -> bool:
        return self._authorization_mode is _AuthorizationMode.DISABLED

    @property
    def is_proxied(self) -> bool:
        return self._authorization_mode is _AuthorizationMode.PROXIED

    @property
    def _authorization_mode(self) -> _AuthorizationMode:
        return _AuthorizationMode.AUTHENTICATED

    @property
    def is_admin(self) -> bool:
        return Permissions.ADMIN.compose() in self.permissions

    @property
    def parsed_permissions(self) -> List[Permissions]:
        return [Permissions(p) for p in self.permissions]


class _DisabledUserAuthorization(UserAuthorization):
    @property
    def _authorization_mode(self) -> _AuthorizationMode:
        return _AuthorizationMode.DISABLED


class _ProxiedUserAuthorization(UserAuthorization):
    @property
    def _authorization_mode(self) -> _AuthorizationMode:
        return _AuthorizationMode.PROXIED


@kumo_dataclass(frozen=True)
class RequestContext:
    user_authorization: UserAuthorization
    trace_id: str = ''


def visible_project_ids_or_none(
    user_authorization: UserAuthorization, ) -> set[ProjectID] | None:
    """Return the explicit StateDB project scope for a request principal.

    ``None`` is reserved for administrators and deliberately means global
    access. Non-admin users always receive a concrete set; an empty set must
    remain empty so query layers return no rows.
    """
    if user_authorization.is_admin:
        return None
    return {ProjectID(id) for id in user_authorization.project_ids}


class Authentication:
    def __init__(self, settings: AuthenticationSettings,
                 is_unit_test: bool = False):
        self._settings = settings
        self._is_unit_test = is_unit_test

    def _require_direct_connect_request(
        self,
        request: Request,
        *,
        admin: bool = False,
    ) -> None:
        user_authorization = get_user_authorization_from_request(request)
        if user_authorization.error:
            raise HTTPException(HTTPStatus.UNAUTHORIZED,
                                user_authorization.error)
        if admin and not user_authorization.is_admin:
            raise HTTPException(HTTPStatus.FORBIDDEN,
                                RestStatusCode.AUTH_NOT_ADMIN.value)

    async def authenticate_request_header(
        self,
        request: Request,
    ) -> None:
        user_authorization = getattr(request.state, "user_authorization", None)
        if (user_authorization is not None
                and user_authorization.access_token == AUTHORIZATION_API_KEY):
            if user_authorization.error:
                raise HTTPException(HTTPStatus.UNAUTHORIZED,
                                    user_authorization.error)
            return
        self._require_direct_connect_request(request)

    # TODO(frg100): Refactor to more naturally specify read, write, admin in
    # router
    async def require_customer_admin(
        self,
        request: Request,
    ) -> None:
        self._require_direct_connect_request(request, admin=True)

    async def require_admin_or_kumo_employee(
        self,
        request: Request,
    ) -> None:
        self._require_direct_connect_request(request, admin=True)

    @property
    def settings(self) -> AuthenticationSettings:
        return self._settings


def get_secret_from_api_key(api_key: str):
    try:
        return ParsedAPIKey.parse(api_key).secret
    except ValueError as error:
        raise ValueError('The API Key must be of the form '
                         '<user_hash>:<name>:<secret_value>') from error


@measure_blocking_code
async def resolve_user_authorization_from_api_key(
    api_key: str,
    auth_disabled: bool = False,
    auth_proxied: bool = False,
    api_key_driver: Optional[APIKeyDriver] = None,
) -> UserAuthorization:
    """Resolve a key, preserving the explicit auth-disabled bypass."""
    # Disabled-auth mode may ignore a non-key header value, but a value that
    # parses as an owner-bound API key must still be verified normally.  In
    # particular, never turn a validly-shaped key owned by a synthetic mode
    # sentinel into an auth-disabled request after its mode has changed.
    try:
        ParsedAPIKey.parse(api_key)
    except ValueError:
        if auth_disabled:
            return UserAuthorization.disabled_request()
        raise HTTPException(
            HTTPStatus.UNAUTHORIZED,
            RestStatusCode.AUTH_INVALID_API_KEY.value,
        )

    return await _resolve_user_authorization_from_valid_api_key(
        api_key=api_key,
        auth_disabled=auth_disabled,
        auth_proxied=auth_proxied,
        api_key_driver=api_key_driver,
    )


async def _resolve_user_authorization_from_valid_api_key(
    api_key: str,
    auth_disabled: bool = False,
    auth_proxied: bool = False,
    api_key_driver: Optional[APIKeyDriver] = None,
) -> UserAuthorization:
    """Verify an owner-bound key without mode-based bypasses."""
    try:
        ParsedAPIKey.parse(api_key)
    except ValueError as error:
        raise HTTPException(
            HTTPStatus.UNAUTHORIZED,
            RestStatusCode.AUTH_INVALID_API_KEY.value,
        ) from error

    if api_key_driver is None:
        raise HTTPException(
            HTTPStatus.SERVICE_UNAVAILABLE,
            'Owner-bound API key verification is unavailable; please retry.',
        )

    from rest.driver.api_key_driver import APIKeyVerificationError
    from temporalio_commons.auto_rpc_exception import RpcError

    try:
        identity = await api_key_driver.verify_api_key(api_key)
    except APIKeyVerificationError as error:
        raise HTTPException(HTTPStatus.UNAUTHORIZED, error.detail) from error
    except RpcError as error:
        if error.error_type != APIKeyVerificationError.__name__:
            raise
        raise HTTPException(HTTPStatus.UNAUTHORIZED, error.message) from error

    try:
        if identity is None:
            raise HTTPException(
                HTTPStatus.UNAUTHORIZED,
                RestStatusCode.AUTH_INVALID_API_KEY.value,
            )

        synthetic_owner_mode_matches = {
            AUTHORIZATION_DISABLED: auth_disabled and not auth_proxied,
            AUTHORIZATION_PROXIED: auth_proxied and not auth_disabled,
        }
        if (identity.user_id in synthetic_owner_mode_matches
                and not synthetic_owner_mode_matches[identity.user_id]):
            raise HTTPException(
                HTTPStatus.UNAUTHORIZED,
                RestStatusCode.AUTH_INVALID_API_KEY.value,
            )

        permissions = [
            Permissions.READ.compose(),
            Permissions.WRITE.compose(),
        ]
        if identity.is_admin:
            permissions.append(Permissions.ADMIN.compose())
        # Membership is deliberately resolved live below; API-key metadata
        # never carries a persisted authorization snapshot. An API-key
        # principal is at most an API-key admin.
        return UserAuthorization(
            user_id=UserID(identity.user_id),
            access_token=AUTHORIZATION_API_KEY,
            permissions=permissions,
        )
    except HTTPException:
        raise
    except Exception as error:
        log.error(
            'Failed to resolve owner authorization for API key auth '
            '(error_type=%s)',
            type(error).__name__,
            exc_info=False,
        )
        raise HTTPException(
            HTTPStatus.UNAUTHORIZED,
            RestStatusCode.AUTH_INVALID_API_KEY.value,
        ) from error


def get_user_authorization_from_request(request: Request) -> UserAuthorization:
    """Return a defensive copy of the request-local authorization."""
    return deepcopy(request.state.user_authorization)


@measure_blocking_code
async def get_ctx(request: Request) -> RequestContext:
    user_authorization = get_user_authorization_from_request(request)
    auth_settings: AuthenticationSettings = request.state.auth_settings

    if user_authorization.access_token in (AUTHORIZATION_API_KEY,
                                           AUTHORIZATION_DIRECT_CONNECT):
        return RequestContext(user_authorization=user_authorization,
                              trace_id='')
    if auth_settings.disable and user_authorization.is_disabled:
        return RequestContext(user_authorization=user_authorization,
                              trace_id='')
    if auth_settings.proxy and user_authorization.is_proxied:
        return RequestContext(user_authorization=user_authorization,
                              trace_id='')
    raise HTTPException(HTTPStatus.UNAUTHORIZED,
                        RestStatusCode.AUTH_INVALID_TOKEN.value)


def _build_auth_context(
    user_authorization: UserAuthorization,
    auth_source: str,
) -> Dict[str, Any]:
    """Build the auth context dict passed to the group resolver.

    Includes request provenance for diagnostics.
    """
    return {
        'auth_source': auth_source,
    }


async def _add_authorization_context(
    user_authorization: UserAuthorization,
    group_resolver: Optional[GroupResolverProtocol],
    project_resolver: Optional[ProjectResolver],
    auth_source: str,
    auth_disabled: bool,
    auth_proxied: bool,
    group_auth_context: Optional[Dict[str, Any]] = None,
) -> UserAuthorization:
    """Resolve and add group and project IDs to the given UserAuthorization.

    Groups are resolved first since project visibility depends on them.
    """
    user_authorization = await _add_groups_to_authorization(
        user_authorization=user_authorization,
        group_resolver=group_resolver,
        auth_source=auth_source,
        auth_disabled=auth_disabled,
        auth_proxied=auth_proxied,
        group_auth_context=group_auth_context,
    )
    return await _add_projects_to_authorization(
        user_authorization=user_authorization,
        project_resolver=project_resolver,
    )


async def _add_projects_to_authorization(
    user_authorization: UserAuthorization,
    project_resolver: Optional[ProjectResolver],
) -> UserAuthorization:
    """Resolve and add visible project IDs to the given UserAuthorization.

    Called after group resolution so kumo_group_ids is already populated.
    Raises 503 on resolver failure. Native projects remain active in disabled
    and proxied deployments, so only a missing resolver or auth error skips
    resolution.
    """
    if project_resolver is None or user_authorization.error:
        return user_authorization

    try:
        resolved_projects = await project_resolver.resolve_projects(
            user_authorization)
    except Exception as exc:
        log.error(
            "Project resolution failed for user_id=%s: %s",
            user_authorization.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            HTTPStatus.SERVICE_UNAVAILABLE,
            detail='Project resolution unavailable, please retry.',
        ) from exc

    return replace(
        user_authorization,
        project_ids=tuple(resolved_projects),
    )


async def _add_groups_to_authorization(
    user_authorization: UserAuthorization,
    group_resolver: Optional[GroupResolverProtocol],
    auth_source: str,
    auth_disabled: bool,
    auth_proxied: bool,
    group_auth_context: Optional[Dict[str, Any]] = None,
) -> UserAuthorization:
    """Resolve and add Kumo group IDs to the given UserAuthorization.

    Native membership is resolved in every deployment, including disabled and
    proxied authentication modes. Provider adapters themselves decide whether
    an external lookup applies. On success, returns a new UserAuthorization
    with kumo_group_ids populated. Raises 503 if the resolver fails so the
    client knows to retry rather than treating the result as a permission
    denial.
    """
    del auth_disabled, auth_proxied
    if group_resolver is None or user_authorization.error:
        return user_authorization
    auth_context = _build_auth_context(user_authorization, auth_source)
    if group_auth_context:
        auth_context.update(group_auth_context)
    try:
        resolved_membership = await group_resolver.resolve_groups(
            user_authorization.user_id,
            auth_context,
        )
    except Exception as exc:
        log.error(
            "Group resolution failed for user_id=%s (error_type=%s)",
            user_authorization.user_id,
            type(exc).__name__,
            exc_info=False,
        )
        raise HTTPException(
            HTTPStatus.SERVICE_UNAVAILABLE,
            detail='Group resolution unavailable, please retry.',
        ) from exc

    return replace(
        user_authorization,
        kumo_group_ids=resolved_membership.group_ids,
        external_membership_unavailable=(
            resolved_membership.external_membership_unavailable),
    )


async def _call_next_with_user_context(
    request: Request,
    call_next: Callable,
    user_authorization: UserAuthorization,
) -> Response:
    """Store user authorization on the request state and call the next handler."""
    set_request_user_authorization(request, user_authorization)
    return await call_next(request)


def set_request_user_authorization(
    request: Request,
    user_authorization: UserAuthorization,
) -> None:
    """Store request authorization for route dependencies and RPC propagation."""
    request.state.user_authorization = user_authorization
    _current_user_ctx.set(user_authorization)


def _permissions(is_admin: bool) -> List[str]:
    permissions = [
        Permissions.READ.compose(),
        Permissions.WRITE.compose(),
    ]
    if is_admin:
        permissions.append(Permissions.ADMIN.compose())
    return permissions


async def authorization_middleware(
    request: Request,
    call_next: Callable,
    auth_disabled: bool,
    auth_proxied: bool,
    is_unit_test: bool = False,
    group_resolver: Optional[Any] = None,
    project_resolver: Optional[ProjectResolver] = None,
    api_key_driver: Optional[APIKeyDriver] = None,
    direct_connect_auth_store: Optional[DirectConnectAuthStore] = None,
) -> Response:
    """Resolve request authorization and attach it to request state."""
    request.state.auth_settings = AuthenticationSettings(
        disable=auth_disabled,
        is_unit_test=is_unit_test,
        proxy=auth_proxied,
    )

    authorization = request.headers.get('X-API-KEY')
    if authorization is not None:
        try:
            user_authorization = await resolve_api_key_user_authorization_context(
                api_key=authorization,
                auth_disabled=auth_disabled,
                auth_proxied=auth_proxied,
                api_key_driver=api_key_driver,
                group_resolver=group_resolver,
                project_resolver=project_resolver,
            )
        except HTTPException as e:
            user_authorization = UserAuthorization.not_authorized(e.detail)
        except Exception:
            log.exception(
                "Unexpected error while resolving Product API key authorization"
            )
            user_authorization = UserAuthorization.not_authorized(
                RestStatusCode.AUTH_INVALID_API_KEY.value)
        return await _call_next_with_user_context(request, call_next,
                                                  user_authorization)

    if auth_disabled:
        user_authorization = await _add_authorization_context(
            user_authorization=UserAuthorization.disabled_request(),
            group_resolver=group_resolver,
            project_resolver=project_resolver,
            auth_source='v0_auth_disabled',
            auth_disabled=auth_disabled,
            auth_proxied=auth_proxied,
        )
        return await _call_next_with_user_context(request, call_next,
                                                  user_authorization)
    if auth_proxied:
        user_authorization = await _add_authorization_context(
            user_authorization=UserAuthorization.proxied(),
            group_resolver=group_resolver,
            project_resolver=project_resolver,
            auth_source='v0_auth_proxied',
            auth_disabled=auth_disabled,
            auth_proxied=auth_proxied,
        )
        return await _call_next_with_user_context(request, call_next,
                                                  user_authorization)

    try:
        if direct_connect_auth_store is None:
            raise HTTPException(HTTPStatus.UNAUTHORIZED,
                                RestStatusCode.AUTH_INVALID_TOKEN.value)
        session_id = request.cookies.get(DIRECT_CONNECT_SESSION_COOKIE)
        if not session_id:
            raise HTTPException(HTTPStatus.UNAUTHORIZED,
                                RestStatusCode.AUTH_INVALID_TOKEN.value)
        session = await direct_connect_auth_store.get_session(session_id)
        if session is None or session.expires_time <= now_utc():
            if session is not None:
                await direct_connect_auth_store.delete_session(session_id)
            raise HTTPException(HTTPStatus.UNAUTHORIZED,
                                RestStatusCode.AUTH_INVALID_TOKEN.value)
        user = await direct_connect_auth_store.get_user(session.user_id)
        if user is None or user.disabled:
            raise HTTPException(HTTPStatus.FORBIDDEN,
                                RestStatusCode.INVALID_PERMISSIONS.value)

        user_authorization = UserAuthorization(
            user_id=UserID(session.user_id),
            access_token=AUTHORIZATION_DIRECT_CONNECT,
            permissions=_permissions(user.role == 'admin'),
            user_name=user.display_name,
        )
        user_authorization = await _add_authorization_context(
            user_authorization=user_authorization,
            group_resolver=group_resolver,
            project_resolver=project_resolver,
            auth_source='direct_connect_session',
            auth_disabled=auth_disabled,
            auth_proxied=auth_proxied,
        )
    except HTTPException as e:
        user_authorization = UserAuthorization.not_authorized(e.detail)
    except Exception:
        log.exception(
            "Unexpected error while resolving Direct Connect authorization")
        user_authorization = UserAuthorization.not_authorized(
            RestStatusCode.AUTH_INVALID_TOKEN.value)

    return await _call_next_with_user_context(request, call_next,
                                              user_authorization)


async def authorization_middleware_public(
    request: Request,
    call_next: Callable,
    auth_disabled: bool,
    auth_proxied: bool,
    is_unit_test: bool = False,
    group_resolver: Optional[GroupResolverProtocol] = None,
    project_resolver: Optional[ProjectResolver] = None,
    api_key_driver: Optional[APIKeyDriver] = None,
) -> Response:
    """Resolve API-key authorization and attach it to request state."""
    # Add AuthenticationSettings to the request
    request.state.auth_settings = AuthenticationSettings(
        disable=auth_disabled,
        is_unit_test=is_unit_test,
        proxy=auth_proxied,
    )

    # Add the AuthInfo to the request
    authorization = request.headers.get('X-API-KEY')
    if authorization is None:
        if not auth_disabled:
            user_authorization = UserAuthorization.not_authorized(
                RestStatusCode.AUTH_INVALID_API_KEY.value)
            return await _call_next_with_user_context(request, call_next,
                                                      user_authorization)
        user_authorization = await _add_authorization_context(
            user_authorization=UserAuthorization.disabled_request(),
            group_resolver=group_resolver,
            project_resolver=project_resolver,
            auth_source='v1_auth_disabled',
            auth_disabled=auth_disabled,
            auth_proxied=auth_proxied,
        )
        return await _call_next_with_user_context(request, call_next,
                                                  user_authorization)

    try:
        api_key = authorization
        user_authorization = await resolve_api_key_user_authorization_context(
            api_key=api_key,
            auth_disabled=auth_disabled,
            auth_proxied=auth_proxied,
            api_key_driver=api_key_driver,
            group_resolver=group_resolver,
            project_resolver=project_resolver,
        )
    except HTTPException as e:
        user_authorization = UserAuthorization.not_authorized(e.detail)
    except Exception:
        log.exception("Unexpected error while resolving API key authorization")
        user_authorization = UserAuthorization.not_authorized(
            RestStatusCode.AUTH_INVALID_API_KEY.value)

    return await _call_next_with_user_context(request, call_next,
                                              user_authorization)


async def resolve_api_key_user_authorization_context(
    api_key: str,
    auth_disabled: bool = False,
    auth_proxied: bool = False,
    api_key_driver: Optional[APIKeyDriver] = None,
    group_resolver: Optional[GroupResolverProtocol] = None,
    project_resolver: Optional[ProjectResolver] = None,
) -> UserAuthorization:
    """Resolve API-key authorization with current groups and projects."""
    user_authorization = await resolve_user_authorization_from_api_key(
        api_key=api_key,
        auth_disabled=auth_disabled,
        auth_proxied=auth_proxied,
        api_key_driver=api_key_driver,
    )
    return await _add_authorization_context(
        user_authorization=user_authorization,
        group_resolver=group_resolver,
        project_resolver=project_resolver,
        auth_source='v1_api_key',
        auth_disabled=auth_disabled,
        auth_proxied=auth_proxied,
    )
