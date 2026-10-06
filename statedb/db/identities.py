# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import sqlite3
import threading
import unicodedata
import uuid
from collections.abc import Callable, Iterable
from dataclasses import fields, replace

import asyncpg
import kumoapi.data_source as api

from io_utils import logging
from rest.config.connector import (
    ADLSConnectorResourceConfig,
    BigQueryConnectorResourceConfig,
    DatabricksConnectorResourceConfig,
    FileConnectorResourceConfig,
    GCSConnectorResourceConfig,
    S3ConnectorResourceConfig,
    SnowflakeConnectorResourceConfig,
)
from rest.config.connector_db import (
    ExternalConnectorDBEntry,
    InternalConnectorDBEntry,
)
from rest.config.data_backend import (
    ActivityAuthorizationRef,
    ConnectorExecutionRef,
    ConnectorOpenSpec,
    ConnectorSpec,
    ResolvedConnectorSpec,
)
from rest.config.encryption import CredentialsEncryptor
from rest.config.identity import (
    IdentityAuthenticationCategory,
    IdentityAuthenticationCategoryError,
    NativeWorkloadIdentityExchanger,
    WorkloadExchangeConfigurationError,
    WorkloadIdentityExchanger,
    ambient_adls_identity_config,
    compose_resolved_connector_spec,
    identity_key,
    is_ambient_s3_identity,
    require_ambient_s3_workload_config,
    validate_google_key_material,
    validate_identity_authentication,
    validate_resolved_connector,
)
from statedb.config.common import (
    ConnectorID,
    GroupID,
    IdentityGroupID,
    IdentityID,
    ProjectID,
    UserID,
)
from statedb.config.connector import PROJECT_UPLOAD_CONNECTOR_NAME
from statedb.db.auth import AuthPrincipal
from statedb.db.connector_admission import lock_connector_admission_and_delete
from statedb.db.connector_authorizations import (
    ConnectorAuthorizationService,
    canonical_request_digest,
)
from statedb.db.connectors import StateDBConnectorRepository
from statedb.db.graphs import StateDBGraphRepository
from statedb.db.group_relations import lock_group_relations
from statedb.db.identity_relations import lock_identity_mutations
from statedb.db.repositories import JsonRepository
from statedb.db.runtime import StateDB, StateTx
from statedb.db.tables import StateDBTableRepository
from statedb.rest.connector_authorization import (
    ConnectorAuthorizationResourceKind,
    ConnectorCapabilityAction,
)
from statedb.rest.connector_db import is_active_project_upload_connector
from statedb.rest.group import GroupStatus, KumoGroupDBEntry
from statedb.rest.identity import IdentityDBEntry, IdentityGroupEntry
from statedb.rest.project import ProjectDBEntry, ProjectStatus
from statedb.utils.errors import (
    ConnectorDoesNotExistError,
    ConnectorInUseError,
    ConnectorNameConflictError,
    ConnectorValidationError,
    ExternalMembershipUnavailableError,
    IdentityConflictError,
    IdentityDoesNotExistError,
    IdentityForbiddenError,
    IdentityInUseError,
    IdentityResolutionUnavailableError,
    IdentityValidationError,
    MultipleIdentitiesValidateTargetError,
    NoIdentityValidatesTargetError,
    ProjectNotFoundError,
    ResourceSlugConflictError,
    connector_validation_diagnostic_value,
)

logger = logging.getLogger(__name__)

IdentityValidator = Callable[[
    api.IdentityConfig, api.IdentityCredentials
    | None, WorkloadIdentityExchanger
], None]
ConnectorValidator = Callable[[ResolvedConnectorSpec], None]


def _authorization_issue_context(_principal: AuthPrincipal):
    """Describe request-user authority without granting service identity."""
    from rest.config.direct_connect import resolve_auth_settings_config
    from service_framework.deploy_env_config import deploy_env

    deploy_config = deploy_env()
    configured = resolve_auth_settings_config(deploy_config)
    if not configured.enabled:
        auth_backend = 'native_disabled'
    elif configured.proxy:
        auth_backend = 'native_proxied'
    else:
        auth_backend = 'direct_connect'
    return auth_backend, deploy_config


# Identity authentication probes are management operations, but they still
# invoke external IdP/cloud endpoints and must not be allowed to hang an API
# request indefinitely.
IDENTITY_AUTHENTICATION_VALIDATION_TIMEOUT_SECONDS = 60.0

# This bounds the complete credential exchange, native-client open, and exact
# target probe for every candidate.  Warehouse connectors also have narrower
# provider-specific timeouts, but object-store and Databricks SDK calls do not
# all provide a reliable request deadline of their own.
CONNECTOR_TARGET_VALIDATION_TIMEOUT_SECONDS = 60.0

_UNIQUE_VIOLATION_ERRORS = (
    sqlite3.IntegrityError,
    asyncpg.UniqueViolationError,
)

MAX_CONNECTOR_NAME_LENGTH = 256

AMBIENT_S3_IDENTITY_ID = IdentityID('identity_00000000000000000000000000000001')
AMBIENT_S3_IDENTITY_DISPLAY_NAME = 'Kumo ambient S3 workload identity'
AMBIENT_S3_IDENTITY_DESCRIPTION = (
    'System-managed AWS workload identity for ambient S3 access')
_AMBIENT_S3_BOOTSTRAP_LOCK_KEY = int.from_bytes(b'kumo-s3a', byteorder='big')
AMBIENT_ADLS_IDENTITY_ID = IdentityID(
    'identity_00000000000000000000000000000002')
AMBIENT_ADLS_IDENTITY_SLUG = 'kumo-ambient-adls'
AMBIENT_ADLS_IDENTITY_DISPLAY_NAME = 'Kumo ambient ADLS workload identity'
AMBIENT_ADLS_IDENTITY_DESCRIPTION = (
    'System-managed Azure workload identity for ambient ADLS access')
_AMBIENT_ADLS_BOOTSTRAP_LOCK_KEY = int.from_bytes(b'kumo-adl',
                                                  byteorder='big')


def _ambient_s3_config() -> api.S3IdentityConfig:
    return api.S3IdentityConfig(authentication=api.S3AmbientAuthentication())


def _ambient_s3_enabled() -> bool:
    try:
        require_ambient_s3_workload_config()
    except WorkloadExchangeConfigurationError:
        return False
    return True


def _require_ambient_s3_enabled() -> None:
    try:
        require_ambient_s3_workload_config()
    except WorkloadExchangeConfigurationError:
        raise IdentityResolutionUnavailableError(
            'Ambient S3 workload identity is disabled') from None


def _ambient_adls_config() -> api.ADLSIdentityConfig | None:
    return ambient_adls_identity_config()


def _ambient_adls_enabled() -> bool:
    return _ambient_adls_config() is not None


def _require_ambient_adls_enabled() -> None:
    if not _ambient_adls_enabled():
        raise IdentityResolutionUnavailableError(
            'Ambient ADLS workload identity is disabled')


def _is_system_ambient_identity(identity: IdentityDBEntry) -> bool:
    return ((identity.id == AMBIENT_S3_IDENTITY_ID
             and is_ambient_s3_identity(identity.config))
            or (identity.id == AMBIENT_ADLS_IDENTITY_ID
                and isinstance(identity.config, api.ADLSIdentityConfig)))


def _system_ambient_identity_is_available(identity: IdentityDBEntry) -> bool:
    if is_ambient_s3_identity(identity.config):
        return (identity.id == AMBIENT_S3_IDENTITY_ID
                and _ambient_s3_enabled())
    if identity.id == AMBIENT_ADLS_IDENTITY_ID:
        return (isinstance(identity.config, api.ADLSIdentityConfig)
                and _ambient_adls_enabled())
    return True


def _require_mutable_identity(identity: IdentityDBEntry) -> None:
    if _is_system_ambient_identity(identity):
        raise IdentityForbiddenError(
            'The ambient cloud identity is system-managed and immutable')


async def _run_blocking(function: Callable, *args) -> None:
    errors: list[BaseException] = []

    def invoke() -> None:
        try:
            function(*args)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=invoke, daemon=True)
    thread.start()
    while thread.is_alive():
        await asyncio.sleep(0.001)
    thread.join()
    if errors:
        raise errors[0]


def normalize_display_name(value: str) -> str:
    return unicodedata.normalize('NFKC', value.strip()).casefold()


def _validated_connector_name(raw_name: str) -> tuple[str, str]:
    name = raw_name.strip()
    normalized_name = normalize_display_name(name)
    if not normalized_name:
        raise ValueError('Connector name cannot be empty')
    if len(name) > MAX_CONNECTOR_NAME_LENGTH:
        raise ValueError(
            f'Connector name must be at most {MAX_CONNECTOR_NAME_LENGTH} '
            'characters long')
    if normalized_name == PROJECT_UPLOAD_CONNECTOR_NAME:
        raise ValueError('Connector name is reserved for internal uploads')
    return name, normalized_name


def _require_principal(principal: AuthPrincipal | None) -> AuthPrincipal:
    if principal is None:
        raise IdentityForbiddenError('Authenticated principal is required')
    return principal


def _require_admin(principal: AuthPrincipal | None) -> AuthPrincipal:
    principal = _require_principal(principal)
    if not principal.is_admin:
        raise IdentityForbiddenError('Administrator access required')
    return principal


def _refuse_container_mounted_token(config: api.IdentityConfig) -> None:
    """Refuses a kind only an in-process runtime can authenticate with.

    A Snowflake ``OAUTH_TOKEN_FILE`` identity reads a token that Snowpark
    Container Services mounts into the *running container*. This service is
    not that container and has no such file, so an identity created here could
    never connect -- it would be accepted, stored, and fail at every use. Both
    identity routers reach this one method, so refusing here covers both and
    they cannot drift apart.
    """
    authentication = getattr(config, 'authentication', None)
    if authentication is api.SnowflakeAuthenticationKind.OAUTH_TOKEN_FILE:
        raise IdentityForbiddenError(
            'A Snowflake oauth_token_file identity authenticates with a token '
            'mounted into a Snowpark Container Services container, which Kumo '
            'cannot read. Create it against an in-process runtime running '
            'inside Snowflake, or use a key pair or password here.')
    if authentication is api.DatabricksAuthenticationKind.AMBIENT:
        raise IdentityForbiddenError(
            'A Databricks ambient identity authenticates through the '
            "Databricks SDK's default auth chain in the running process, "
            'which Kumo cannot supply. Use it with a local kumo-engine '
            'DatabricksConnector in an environment where that chain resolves, '
            'or use OAuth service-principal credentials here.')


def _connector_blocks_identity_delete(
    connector: ExternalConnectorDBEntry,
    projects_by_id: dict[ProjectID, ProjectDBEntry],
) -> bool:
    """A live connector blocks identity deletion when its project is ACTIVE or
    its project row is missing. Connectors in archived projects don't block
    (archived projects can't run activities)."""
    project = projects_by_id.get(connector.project_id)
    return project is None or project.status == ProjectStatus.ACTIVE


def _api_timestamps(entry) -> api.ResourceTimestamps:
    return api.ResourceTimestamps(
        created_at=entry.timestamps.created_at,
        updated_at=entry.timestamps.updated_at,
    )


def _identity_response(entry: IdentityDBEntry) -> api.IdentityResponse:
    google_workload_identity = (
        isinstance(entry.config,
                   (api.BigQueryIdentityConfig, api.GCSIdentityConfig))
        and isinstance(entry.config.authentication,
                       api.GoogleWorkloadIdentityAuthentication))
    return api.IdentityResponse(
        id=str(entry.id),
        slug=entry.slug,
        display_name=entry.display_name,
        description=entry.description,
        timestamps=_api_timestamps(entry),
        credential_status=api.IdentityCredentialStatus(
            configured=(google_workload_identity
                        or entry.encrypted_credentials is not None),
            last_rotated_at=(None if google_workload_identity else
                             entry.credentials_rotated_at),
        ),
        config=entry.config,
    )


def _credentials(
    encryptor: CredentialsEncryptor,
    identity: IdentityDBEntry,
) -> api.IdentityCredentials | None:
    if identity.encrypted_credentials is None:
        return None
    return encryptor.decrypt_identity_credentials(
        identity.encrypted_credentials)


class IdentityService:
    def __init__(
        self,
        state_db: StateDB,
        connector_authorizations: ConnectorAuthorizationService,
        *,
        identity_validator: IdentityValidator = validate_identity_authentication,
        workload_exchanger: WorkloadIdentityExchanger | None = None,
    ) -> None:
        self._state_db = state_db
        self._identities = JsonRepository('identities', IdentityDBEntry)
        self._bindings = JsonRepository('identity_groups', IdentityGroupEntry)
        self._groups = JsonRepository('groups', KumoGroupDBEntry)
        self._projects = JsonRepository('projects', ProjectDBEntry)
        self._connectors = StateDBConnectorRepository()
        self._encryptor = CredentialsEncryptor()
        self._identity_validator = identity_validator
        self._workload_exchanger = (workload_exchanger
                                    or NativeWorkloadIdentityExchanger())
        self._connector_authorizations = connector_authorizations

    async def _ensure_system_ambient_identity(
        self,
        *,
        identity_id: IdentityID,
        slug: str,
        display_name: str,
        description: str,
        config: api.IdentityConfig,
        bootstrap_lock_key: int,
        initial_group_id: GroupID | None = None,
    ) -> api.IdentityResponse:
        from statedb.config.timestamp import get_current_timestamps

        canonical_key = identity_key(config)
        async with self._state_db.write_tx() as tx:
            if tx.backend == 'postgres':
                await tx.fetchrow(
                    'SELECT pg_advisory_xact_lock($1)',
                    bootstrap_lock_key,
                )
            row = await self._identities.get(tx, str(identity_id))
            keyed = await self._identities.find_one_by_fields(
                tx, {'identity_key': canonical_key})
            if keyed is not None and keyed.id != str(identity_id):
                raise RuntimeError(
                    f'A non-canonical ambient identity already uses {slug}')
            if row is None:
                entry = IdentityDBEntry(
                    id=identity_id,
                    slug=api.validate_resource_slug(slug,
                                                    allow_reserved=True),
                    display_name=display_name,
                    normalized_display_name=normalize_display_name(
                        display_name),
                    description=description,
                    identity_key=canonical_key,
                    credentials_rotated_at=None,
                    config=config,
                    encrypted_credentials=None,
                    timestamps=get_current_timestamps(),
                )
                await self._identities.upsert(tx, str(entry.id), entry)
                if initial_group_id is not None:
                    await lock_group_relations(tx, [initial_group_id])
                    group_row = await self._groups.get(
                        tx, str(initial_group_id))
                    if (group_row is None or group_row.payload.status
                            != GroupStatus.ACTIVE):
                        raise ValueError(
                            f'Unknown or inactive group: {initial_group_id}')
                    binding = IdentityGroupEntry(
                        id=IdentityGroupID.generate(),
                        identity_id=identity_id,
                        group_id=initial_group_id,
                        timestamps=get_current_timestamps(),
                    )
                    await self._bindings.upsert(tx, str(binding.id), binding)
            else:
                entry = row.payload
                if (entry.slug != slug
                        or entry.tombstoned_at is not None or entry.display_name
                        != display_name
                        or entry.normalized_display_name !=
                        normalize_display_name(display_name)
                        or entry.description != description
                        or entry.identity_key != canonical_key
                        or entry.config != config
                        or entry.credentials_rotated_at is not None
                        or entry.encrypted_credentials is not None):
                    raise RuntimeError(
                        f'Canonical ambient identity {slug} violates '
                        'bootstrap invariants')
        return _identity_response(entry)

    async def ensure_ambient_s3_identity(self) -> api.IdentityResponse | None:
        if not _ambient_s3_enabled():
            return None
        return await self._ensure_system_ambient_identity(
            identity_id=AMBIENT_S3_IDENTITY_ID,
            slug=api.AMBIENT_S3_IDENTITY_SLUG,
            display_name=AMBIENT_S3_IDENTITY_DISPLAY_NAME,
            description=AMBIENT_S3_IDENTITY_DESCRIPTION,
            config=_ambient_s3_config(),
            bootstrap_lock_key=_AMBIENT_S3_BOOTSTRAP_LOCK_KEY,
        )

    async def ensure_ambient_adls_identity_bound_on_create(
            self, group_id: GroupID) -> api.IdentityResponse | None:
        config = _ambient_adls_config()
        if config is None:
            return None
        return await self._ensure_system_ambient_identity(
            identity_id=AMBIENT_ADLS_IDENTITY_ID,
            slug=AMBIENT_ADLS_IDENTITY_SLUG,
            display_name=AMBIENT_ADLS_IDENTITY_DISPLAY_NAME,
            description=AMBIENT_ADLS_IDENTITY_DESCRIPTION,
            config=config,
            bootstrap_lock_key=_AMBIENT_ADLS_BOOTSTRAP_LOCK_KEY,
            initial_group_id=group_id,
        )

    async def ensure_ambient_s3_identity_bound_to_group(
            self, group_id: GroupID) -> api.IdentityGroupsResponse | None:
        """Add the system ambient S3 identity to one explicit active group.

        ``ensure_ambient_s3_identity`` intentionally leaves the identity
        unbound because authenticated deployments require an administrator to
        choose each project group that may use the workload role. Auth-disabled
        application and devserver bootstrap call this for their canonical
        Default group. The operation is additive: it never replaces an existing
        administrator-managed binding.
        """
        identity = await self.ensure_ambient_s3_identity()
        if identity is None:
            return None
        from statedb.config.timestamp import get_current_timestamps

        async with self._state_db.write_tx() as tx:
            await lock_identity_mutations(tx, [AMBIENT_S3_IDENTITY_ID])
            await lock_group_relations(tx, [group_id])
            await self._get_required(tx, AMBIENT_S3_IDENTITY_ID)
            group_row = await self._groups.get(tx, str(group_id))
            if (group_row is None
                    or group_row.payload.status != GroupStatus.ACTIVE):
                raise ValueError(f'Unknown or inactive group: {group_id}')

            existing = await self._bindings.find_one_by_fields(
                tx,
                {
                    'identity_id': str(AMBIENT_S3_IDENTITY_ID),
                    'group_id': str(group_id),
                },
            )
            if existing is None:
                binding = IdentityGroupEntry(
                    id=IdentityGroupID.generate(),
                    identity_id=AMBIENT_S3_IDENTITY_ID,
                    group_id=group_id,
                    timestamps=get_current_timestamps(),
                )
                await self._bindings.upsert(tx, str(binding.id), binding)

            bindings = await self._bindings.list_by_field(
                tx, 'identity_id', str(AMBIENT_S3_IDENTITY_ID))
        return api.IdentityGroupsResponse(
            identity_id=str(AMBIENT_S3_IDENTITY_ID),
            group_ids=[str(row.payload.group_id) for row in bindings],
        )

    async def _validate_credentials(
        self,
        config: api.IdentityConfig,
        credentials: api.IdentityCredentials | None,
    ) -> None:
        try:
            await asyncio.wait_for(
                _run_blocking(
                    self._identity_validator,
                    config,
                    credentials,
                    self._workload_exchanger,
                ),
                timeout=IDENTITY_AUTHENTICATION_VALIDATION_TIMEOUT_SECONDS,
            )
        except IdentityAuthenticationCategoryError as error:
            raise IdentityValidationError(str(error)) from None
        except Exception:
            # Provider SDKs routinely include request bodies, credential
            # fragments, signed assertions, or temporary tokens in exception
            # text.  Authentication failures cross the REST boundary, so the
            # original exception must never become part of the public error or
            # its displayed exception chain.
            raise IdentityValidationError(
                'Identity authentication failed') from None

    async def create(
        self,
        args: api.CreateIdentityArgs,
        principal: AuthPrincipal | None,
    ) -> api.IdentityResponse:
        _require_admin(principal)
        if is_ambient_s3_identity(args.config):
            raise IdentityForbiddenError(
                'The ambient S3 identity is created and managed by Kumo')
        _refuse_container_mounted_token(args.config)
        api.validate_identity_credentials_match(args.config, args.credentials)
        await self._validate_credentials(args.config, args.credentials)
        from statedb.config.timestamp import get_current_timestamps

        display_name = args.display_name.strip()
        normalized_name = normalize_display_name(display_name)
        if not normalized_name:
            raise ValueError('display_name cannot be empty')
        slug = (api.validate_resource_slug(str(args.slug)) if args.slug
                is not None else api.derive_resource_slug(display_name))
        group_ids = [GroupID(str(group_id)) for group_id in args.group_ids]
        if len(set(group_ids)) != len(group_ids):
            raise ValueError('group_ids must not contain duplicates')
        timestamps = get_current_timestamps()
        entry = IdentityDBEntry(
            id=IdentityID.generate(),
            slug=slug,
            display_name=display_name,
            normalized_display_name=normalized_name,
            description=args.description,
            identity_key=identity_key(args.config, args.credentials),
            credentials_rotated_at=(timestamps.updated_at
                                    if args.credentials is not None else None),
            config=args.config,
            encrypted_credentials=(self._encryptor.encrypt_identity_credentials(
                args.credentials) if args.credentials is not None else None),
            timestamps=timestamps,
        )
        try:
            async with self._state_db.write_tx() as tx:
                # Group deletion and every group-relation mutation take these
                # same stable locks. Validate after acquiring them so an
                # identity can never commit with bindings to a concurrently
                # deleted group.
                await lock_group_relations(tx, group_ids)
                group_rows = await self._groups.list_by_ids(
                    tx, [str(group_id) for group_id in group_ids])
                active_groups = {
                    row.id
                    for row in group_rows
                    if row.payload.status == GroupStatus.ACTIVE
                }
                missing = [
                    str(group_id) for group_id in group_ids
                    if str(group_id) not in active_groups
                ]
                if missing:
                    raise ValueError(f'Unknown or inactive groups: {missing}')
                if await self._identities.exists_by_fields(
                        tx, {'identity_key': entry.identity_key}):
                    raise IdentityConflictError(
                        'An identity already exists for this external principal'
                    )
                if await self._identities.exists_by_fields(
                        tx, {'normalized_display_name': normalized_name}):
                    raise IdentityConflictError(
                        'An identity with this display name already exists')
                if (await self._identities.exists_by_fields(
                        tx, {'normalized_display_name': slug})
                        or await self._identities.exists_by_fields(
                            tx, {'slug': normalized_name})):
                    raise IdentityConflictError(
                        'Identity display name conflicts with an existing slug')
                await self._identities.upsert(tx, str(entry.id), entry)
                for group_id in group_ids:
                    binding = IdentityGroupEntry(
                        id=IdentityGroupID.generate(),
                        identity_id=entry.id,
                        group_id=group_id,
                        timestamps=timestamps,
                    )
                    await self._bindings.upsert(tx, str(binding.id), binding)
        except _UNIQUE_VIOLATION_ERRORS as error:
            async with self._state_db.read_tx() as tx:
                if await self._identities.exists_by_fields(tx, {'slug': slug}):
                    raise ResourceSlugConflictError(
                        f"Identity slug '{slug}' is already in use.") from error
            raise IdentityConflictError(
                'Identity principal and display name must be unique') from error
        return _identity_response(entry)

    async def list_admin(
        self,
        principal: AuthPrincipal | None,
    ) -> list[api.IdentityResponse]:
        _require_admin(principal)
        async with self._state_db.read_tx() as tx:
            rows = await self._identities.list_by_fields(
                tx, {'tombstoned_at': None})
        return [_identity_response(row.payload) for row in rows]

    async def get_admin(
        self,
        identity_id: IdentityID,
        principal: AuthPrincipal | None,
    ) -> api.IdentityResponse:
        _require_admin(principal)
        async with self._state_db.read_tx() as tx:
            identity = await self._get_required(tx, identity_id)
        return _identity_response(identity)

    async def update_metadata(
        self,
        identity_id: IdentityID,
        args: api.UpdateIdentityArgs,
        principal: AuthPrincipal | None,
    ) -> api.IdentityResponse:
        _require_admin(principal)
        from statedb.config.timestamp import update_timestamps

        try:
            async with self._state_db.write_tx() as tx:
                await lock_identity_mutations(tx, [identity_id])
                identity = await self._get_required(tx, identity_id)
                _require_mutable_identity(identity)
                if args.display_name is not None:
                    display_name = args.display_name.strip()
                    normalized_name = normalize_display_name(display_name)
                    if not normalized_name:
                        raise ValueError('display_name cannot be empty')
                    if await self._identities.exists_by_fields(
                            tx, {'normalized_display_name': normalized_name},
                            exclude_id=str(identity_id)):
                        raise IdentityConflictError(
                            'An identity with this display name already exists')
                    if await self._identities.exists_by_fields(
                            tx, {'slug': normalized_name},
                            exclude_id=str(identity_id)):
                        raise IdentityConflictError(
                            'Identity display name conflicts with an existing '
                            'slug')
                    identity.display_name = display_name
                    identity.normalized_display_name = normalized_name
                if args.description is not None:
                    identity.description = args.description
                identity.timestamps = update_timestamps(identity.timestamps)
                await self._identities.upsert(tx, str(identity.id), identity)
        except _UNIQUE_VIOLATION_ERRORS as error:
            # The precheck provides a useful early error, but the database is
            # authoritative when concurrent normalized renames race it.
            raise IdentityConflictError(
                'An identity with this display name already exists') from error
        return _identity_response(identity)

    async def rotate_credentials(
        self,
        identity_id: IdentityID,
        args: api.RotateIdentityCredentialsArgs,
        principal: AuthPrincipal | None,
        *,
        preserve_s3_ca_certificate: bool = False,
    ) -> api.IdentityResponse:
        _require_admin(principal)
        async with self._state_db.read_tx() as tx:
            current = await self._get_required(tx, identity_id)
            _require_mutable_identity(current)
        credentials = args.credentials
        if (preserve_s3_ca_certificate
                and isinstance(current.config, api.S3IdentityConfig)
                and isinstance(current.config.authentication,
                               api.S3AccessKeyAuthentication)
                and isinstance(credentials, api.S3AccessKeyCredentials)
                and credentials.ca_certificate is None):
            current_credentials = _credentials(self._encryptor, current)
            if (isinstance(current_credentials, api.S3AccessKeyCredentials)
                    and current_credentials.ca_certificate is not None):
                # The CA is write-only because returning trust material from an
                # identity response would expose a stored secret. The router
                # requests preservation only when the JSON field was omitted;
                # an explicit null continues to clear the certificate.
                credentials = replace(
                    credentials,
                    ca_certificate=current_credentials.ca_certificate,
                )
        if (isinstance(current.config,
                       (api.BigQueryIdentityConfig, api.GCSIdentityConfig))
                and isinstance(
                    current.config.authentication,
                    api.GoogleWorkloadIdentityAuthentication,
                )):
            raise ValueError(
                'Workload identity credentials are managed by the deployment')
        if (isinstance(current.config,
                       (api.BigQueryIdentityConfig, api.GCSIdentityConfig))
                and args.authentication != current.config.authentication):
            raise ValueError(
                'Google key rotation must match the configured service account')
        if (not isinstance(current.config,
                           (api.BigQueryIdentityConfig, api.GCSIdentityConfig))
                and args.authentication is not None):
            raise ValueError(
                'Google authentication metadata is only valid for Google keys')
        api.validate_identity_credentials_match(current.config, credentials)
        google_key_mode = (
            isinstance(current.config,
                       (api.BigQueryIdentityConfig, api.GCSIdentityConfig))
            and isinstance(current.config.authentication,
                           api.GoogleServiceAccountKeyAuthentication))
        if google_key_mode:
            assert args.authentication is not None
            assert isinstance(credentials,
                              api.GoogleServiceAccountKeyCredentials)
            try:
                validate_google_key_material(args.authentication,
                                             credentials)
            except Exception:
                raise IdentityValidationError(
                    'Google service-account key is invalid') from None
        else:
            await self._validate_credentials(current.config, credentials)
        from statedb.config.timestamp import now_utc, update_timestamps

        rotated_identity_key = identity_key(current.config, credentials)

        try:
            async with self._state_db.write_tx() as tx:
                await lock_identity_mutations(tx, [identity_id])
                identity = await self._get_required(tx, identity_id)
                _require_mutable_identity(identity)
                if (identity.identity_key != current.identity_key
                        or identity.config != current.config):
                    raise ValueError('Identity principal changed')
                if await self._identities.exists_by_fields(
                        tx, {'identity_key': rotated_identity_key},
                        exclude_id=str(identity_id)):
                    raise IdentityConflictError(
                        'An identity already exists for this external principal'
                    )
                identity.identity_key = rotated_identity_key
                identity.encrypted_credentials = (
                    self._encryptor.encrypt_identity_credentials(
                        credentials)
                    if credentials is not None else None)
                identity.credentials_rotated_at = (
                    now_utc() if credentials is not None else None)
                identity.timestamps = update_timestamps(identity.timestamps)
                await self._identities.upsert(tx, str(identity.id), identity)
        except _UNIQUE_VIOLATION_ERRORS as error:
            raise IdentityConflictError(
                'An identity already exists for this external principal'
            ) from error
        return _identity_response(identity)

    async def authorize_test_credentials(
        self,
        identity_id: IdentityID,
        principal: AuthPrincipal | None,
    ) -> ActivityAuthorizationRef:
        principal = _require_admin(principal)
        auth_backend, deploy_config = _authorization_issue_context(principal)
        digest = canonical_request_digest({
            'operation': 'validate_identity',
            'identity_id': str(identity_id),
        })
        async with self._state_db.write_tx() as tx:
            await self._get_required(tx, identity_id)
            _, token = await self._connector_authorizations.issue_in_tx(
                tx,
                principal=principal,
                auth_backend=auth_backend,
                resource_kind=(
                    ConnectorAuthorizationResourceKind.IDENTITY_VALIDATE),
                resource_id=str(identity_id),
                project_id='',
                request_digest=digest,
                allowed_actions=(ConnectorCapabilityAction.VALIDATE_IDENTITY, ),
                deployment_id=(f'{deploy_config.instance_id}:'
                               f'{deploy_config.client_id}'),
                temporal_namespace=deploy_config.temporal_namespace,
            )
        return ActivityAuthorizationRef(token)

    async def test_credentials_from_activity(
        self,
        identity_id: IdentityID,
        principal: AuthPrincipal | None,
    ) -> api.IdentityResponse:
        """Decrypt and test a stored identity only inside an activity."""
        _require_admin(principal)
        async with self._state_db.read_tx() as tx:
            identity = await self._get_required(tx, identity_id)
        await self._validate_credentials(
            identity.config, _credentials(self._encryptor, identity))
        return _identity_response(identity)

    async def replace_groups(
        self,
        identity_id: IdentityID,
        args: api.ReplaceIdentityGroupsArgs,
        principal: AuthPrincipal | None,
    ) -> api.IdentityGroupsResponse:
        _require_admin(principal)
        from statedb.config.timestamp import get_current_timestamps

        group_ids = list(
            dict.fromkeys(
                GroupID(str(group_id)) for group_id in args.group_ids))
        async with self._state_db.write_tx() as tx:
            await lock_identity_mutations(tx, [identity_id])
            await self._get_required(tx, identity_id)
            existing = await self._bindings.list_by_field(
                tx, 'identity_id', str(identity_id))
            relation_group_ids = list(group_ids) + [
                row.payload.group_id for row in existing
            ]
            await lock_group_relations(tx, relation_group_ids)
            group_rows = await self._groups.list_by_ids(
                tx, [str(group_id) for group_id in group_ids])
            active_groups = {
                row.id
                for row in group_rows
                if row.payload.status == GroupStatus.ACTIVE
            }
            missing = [
                str(group_id) for group_id in group_ids
                if str(group_id) not in active_groups
            ]
            if missing:
                raise ValueError(f'Unknown or inactive groups: {missing}')
            existing = await self._bindings.list_by_field(
                tx, 'identity_id', str(identity_id))
            for row in existing:
                await self._bindings.delete(tx, row.id)
            for group_id in group_ids:
                binding = IdentityGroupEntry(
                    id=IdentityGroupID.generate(),
                    identity_id=identity_id,
                    group_id=group_id,
                    timestamps=get_current_timestamps(),
                )
                await self._bindings.upsert(tx, str(binding.id), binding)
        return api.IdentityGroupsResponse(
            identity_id=str(identity_id),
            group_ids=[str(group_id) for group_id in group_ids],
        )

    async def list_groups(
        self,
        identity_id: IdentityID,
        principal: AuthPrincipal | None,
    ) -> api.IdentityGroupsResponse:
        _require_admin(principal)
        async with self._state_db.read_tx() as tx:
            await self._get_required(tx, identity_id)
            rows = await self._bindings.list_by_field(tx, 'identity_id',
                                                      str(identity_id))
        return api.IdentityGroupsResponse(
            identity_id=str(identity_id),
            group_ids=[str(row.payload.group_id) for row in rows],
        )

    async def available_for_project(
        self,
        project_id: ProjectID,
        principal: AuthPrincipal | None,
    ) -> list[api.IdentityResponse]:
        principal = _require_principal(principal)
        async with self._state_db.read_tx() as tx:
            project = await _get_project_for_principal(tx, self._projects,
                                                       project_id, principal)
            await _require_active_group(tx, self._groups, project.group_id)
            bindings = await self._bindings.list_by_field(
                tx, 'group_id', str(project.group_id))
            rows = await self._identities.list_by_ids(
                tx, [str(row.payload.identity_id) for row in bindings])
        return [
            _identity_response(row.payload) for row in rows
            if (row.payload.tombstoned_at is None
                and _system_ambient_identity_is_available(row.payload))
        ]

    async def available_for_project_by_slug(
        self,
        project_id: ProjectID,
        slug: str,
        principal: AuthPrincipal | None,
    ) -> api.IdentityResponse:
        """Resolve a live identity only through the project's group binding."""
        slug = api.validate_resource_slug(slug, allow_reserved=True)
        principal = _require_principal(principal)
        async with self._state_db.read_tx() as tx:
            project = await _get_project_for_principal(tx, self._projects,
                                                       project_id, principal)
            await _require_active_group(tx, self._groups, project.group_id)
            identity = await self._available_identity_for_project(
                tx, project.group_id, slug=slug)
        return _identity_response(identity)

    async def available_for_project_by_id(
        self,
        project_id: ProjectID,
        identity_id: IdentityID,
        principal: AuthPrincipal | None,
    ) -> api.IdentityResponse:
        """Resolve a live identity ID only through the project's binding."""
        principal = _require_principal(principal)
        async with self._state_db.read_tx() as tx:
            project = await _get_project_for_principal(tx, self._projects,
                                                       project_id, principal)
            await _require_active_group(tx, self._groups, project.group_id)
            identity = await self._available_identity_for_project(
                tx, project.group_id, identity_id=identity_id)
        return _identity_response(identity)

    async def _available_identity_for_project(
        self,
        tx: StateTx,
        group_id: GroupID,
        *,
        identity_id: IdentityID | None = None,
        slug: str | None = None,
    ) -> IdentityDBEntry:
        if (identity_id is None) == (slug is None):
            raise RuntimeError('Exactly one identity selector is required')
        selector = str(identity_id) if identity_id is not None else str(slug)
        if identity_id is not None:
            identity_row = await self._identities.get(tx, str(identity_id))
            if (identity_row is not None
                    and identity_row.payload.tombstoned_at is not None):
                identity_row = None
        else:
            identity_row = await self._identities.find_one_by_fields(
                tx, {
                    'slug': slug,
                    'tombstoned_at': None,
                })
        if identity_row is None:
            raise IdentityDoesNotExistError(selector)
        binding = await self._bindings.find_one_by_fields(
            tx, {
                'identity_id': identity_row.id,
                'group_id': str(group_id),
            })
        identity = identity_row.payload
        if (binding is None
                or not _system_ambient_identity_is_available(identity)):
            raise IdentityDoesNotExistError(selector)
        return identity

    async def delete(
        self,
        identity_id: IdentityID,
        principal: AuthPrincipal | None,
    ) -> api.IdentityResponse:
        _require_admin(principal)
        from statedb.config.timestamp import now_utc, update_timestamps

        async with self._state_db.write_tx() as tx:
            await lock_identity_mutations(tx, [identity_id])
            identity = await self._get_required(tx, identity_id)
            _require_mutable_identity(identity)
            bindings = await self._bindings.list_by_field(
                tx, 'identity_id', str(identity_id))
            # Connector creation takes the same group-relation lock before
            # its final binding recheck and insert. Without this lock,
            # Postgres could commit an identity deletion concurrently after
            # that recheck and leave the new connector orphaned.
            await lock_group_relations(
                tx, [row.payload.group_id for row in bindings])
            identity = await self._get_required(tx, identity_id)
            identity_connectors = await self._connectors.list_live_by_identity(
                tx, identity_id)
            project_rows = await self._projects.list_by_ids(
                tx, [
                    str(connector.project_id)
                    for connector in identity_connectors
                ])
            projects_by_id = {
                ProjectID(row.id): row.payload
                for row in project_rows
            }
            if any(
                    _connector_blocks_identity_delete(connector, projects_by_id)
                    for connector in identity_connectors):
                raise IdentityInUseError(
                    f'Identity {identity_id} is used by connectors')
            bindings = await self._bindings.list_by_field(
                tx, 'identity_id', str(identity_id))
            for row in bindings:
                await self._bindings.delete(tx, row.id)
            response = _identity_response(identity)
            identity.encrypted_credentials = None
            identity.credentials_rotated_at = None
            identity.tombstoned_at = now_utc()
            identity.timestamps = update_timestamps(identity.timestamps)
            await self._identities.upsert(tx, str(identity_id), identity)
        return response

    async def list_connectors_using_identity(
        self,
        identity_id: IdentityID,
        principal: AuthPrincipal | None,
    ) -> api.IdentityConnectorsResponse:
        """List the live connectors that block deletion of this identity.

        Returns exactly the set the delete guard would 409 on: connectors
        whose project is ACTIVE or whose project row is missing. Connectors in
        archived projects are omitted because they do not block deletion.
        """
        _require_admin(principal)
        async with self._state_db.read_tx() as tx:
            await self._get_required(tx, identity_id)
            identity_connectors = await self._connectors.list_live_by_identity(
                tx, identity_id)
            project_rows = await self._projects.list_by_ids(
                tx, [
                    str(connector.project_id)
                    for connector in identity_connectors
                ])
        projects_by_id = {
            ProjectID(row.id): row.payload
            for row in project_rows
        }
        connectors = [
            api.IdentityConnectorUsage(
                connector_id=str(connector.id),
                connector_name=connector.name,
                project_id=str(connector.project_id),
                project_display_name=(
                    projects_by_id[connector.project_id].display_name
                    if connector.project_id in projects_by_id else None),
            )
            for connector in identity_connectors
            if _connector_blocks_identity_delete(connector, projects_by_id)
        ]
        return api.IdentityConnectorsResponse(
            identity_id=str(identity_id),
            connectors=connectors,
        )

    async def _get_required(
        self,
        tx: StateTx,
        identity_id: IdentityID,
    ) -> IdentityDBEntry:
        row = await self._identities.get(tx, str(identity_id))
        if row is None or row.payload.tombstoned_at is not None:
            raise IdentityDoesNotExistError(identity_id)
        return row.payload


def _project_visible(project: ProjectDBEntry, principal: AuthPrincipal) -> bool:
    if principal.is_admin:
        return True
    groups = {str(group_id) for group_id in principal.kumo_group_ids}
    if str(project.group_id) not in groups:
        return False
    if project.is_private:
        return project.created_by == str(principal.user_id)
    return True


async def _get_project_for_principal(
    tx: StateTx,
    projects: JsonRepository,
    project_id: ProjectID,
    principal: AuthPrincipal,
    *,
    require_active: bool = True,
) -> ProjectDBEntry:
    row = await projects.get(tx, str(project_id))
    if row is None:
        raise ProjectNotFoundError(f"Project '{project_id}' not found")
    project = row.payload
    if not _project_visible(project, principal):
        membership_could_authorize = (not project.is_private
                                      or project.created_by == str(
                                          principal.user_id))
        if (membership_could_authorize and getattr(
                principal, 'external_membership_unavailable', False)):
            raise ExternalMembershipUnavailableError(
                'External group membership is unavailable; please retry')
        raise ProjectNotFoundError(f"Project '{project_id}' not found")
    if require_active and project.status != ProjectStatus.ACTIVE:
        raise ProjectNotFoundError(f"Project '{project_id}' not found")
    return project


async def _require_active_group(
    tx: StateTx,
    groups: JsonRepository,
    group_id: GroupID,
) -> KumoGroupDBEntry:
    row = await groups.get(tx, str(group_id))
    if row is None or row.payload.status != GroupStatus.ACTIVE:
        raise IdentityForbiddenError('Group is not active')
    return row.payload


def _db_config(name: str, config: api.ConnectorResourceConfig):
    config_types = {
        api.ConnectorType.DATABRICKS: DatabricksConnectorResourceConfig,
        api.ConnectorType.SNOWFLAKE: SnowflakeConnectorResourceConfig,
        api.ConnectorType.BIGQUERY: BigQueryConnectorResourceConfig,
        api.ConnectorType.S3: S3ConnectorResourceConfig,
        api.ConnectorType.GCS: GCSConnectorResourceConfig,
        api.ConnectorType.ADLS: ADLSConnectorResourceConfig,
    }
    values = {
        field.name: getattr(config, field.name)
        for field in fields(config) if field.name != 'type'
    }
    return config_types[config.type](name=name, **values)


class ConnectorService:
    def __init__(
        self,
        state_db: StateDB,
        connector_authorizations: ConnectorAuthorizationService,
        *,
        connector_validator: ConnectorValidator = validate_resolved_connector,
        workload_exchanger: WorkloadIdentityExchanger | None = None,
    ) -> None:
        self._state_db = state_db
        self._connectors = StateDBConnectorRepository()
        self._identities = JsonRepository('identities', IdentityDBEntry)
        self._bindings = JsonRepository('identity_groups', IdentityGroupEntry)
        self._groups = JsonRepository('groups', KumoGroupDBEntry)
        self._projects = JsonRepository('projects', ProjectDBEntry)
        self._tables = StateDBTableRepository()
        self._graphs = StateDBGraphRepository()
        self._connector_authorizations = connector_authorizations
        self._encryptor = CredentialsEncryptor()
        self._connector_validator = connector_validator
        self._workload_exchanger = (workload_exchanger
                                    or NativeWorkloadIdentityExchanger())

    async def authorize_create(
        self,
        project_id: ProjectID,
        args: api.CreateConnectorArgs,
        principal: AuthPrincipal | None,
    ) -> dict[IdentityID, ActivityAuthorizationRef]:
        """Authorize a connector create without resolving its identity."""
        principal = _require_principal(principal)
        _validated_connector_name(args.name)
        auth_backend, deploy_config = _authorization_issue_context(principal)
        request_digest = canonical_request_digest(args)
        async with self._state_db.write_tx() as tx:
            project = await _get_project_for_principal(tx, self._projects,
                                                       project_id, principal)
            await _require_active_group(tx, self._groups, project.group_id)
            if args.identity_id is not None:
                identity = await self._load_bound_identity(
                    tx, project.group_id, IdentityID(str(args.identity_id)))
                if identity.config.type.value != args.config.type.value:
                    raise ValueError('Connector and identity types must match')
                identities = [identity]
            elif args.config.type == api.ConnectorType.S3:
                identities = [
                    await self._load_automatic_ambient_s3_identity(
                        tx, project.group_id)
                ]
            else:
                identities = await self._bound_identities_for_type(
                    tx, project.group_id, args.config.type)
            if not identities:
                raise NoIdentityValidatesTargetError(message=(
                    f"No {args.config.type.value} identity is assigned to this "
                    "project's group. Assign an existing "
                    f"{args.config.type.value} identity to the group under "
                    "Admin -> Identity Settings, or create one and add it to "
                    "the group."))
            authorizations = {}
            for identity in identities:
                _, token = await self._connector_authorizations.issue_in_tx(
                    tx,
                    principal=principal,
                    auth_backend=auth_backend,
                    resource_kind=(ConnectorAuthorizationResourceKind.
                                   PROJECT_CONNECTOR_CREATE),
                    resource_id=str(identity.id),
                    project_id=str(project_id),
                    request_digest=request_digest,
                    allowed_actions=(
                        ConnectorCapabilityAction.CREATE_PROJECT_CONNECTOR, ),
                    deployment_id=(f'{deploy_config.instance_id}:'
                                   f'{deploy_config.client_id}'),
                    temporal_namespace=deploy_config.temporal_namespace,
                )
                authorizations[identity.id] = ActivityAuthorizationRef(token)
        return authorizations

    async def create_from_activity(
        self,
        project_id: ProjectID,
        args: api.CreateConnectorArgs,
        created_by: UserID,
        authorized_identity_ids: set[IdentityID],
    ) -> api.ConnectorResponse:
        """Validate and persist an authorized connector inside an activity.

        The activity reloads the project, group, identity, and binding before
        credential resolution and again before insertion.  Its inputs contain
        only durable IDs and safe target configuration.
        """
        name, normalized_name = _validated_connector_name(args.name)
        async with self._state_db.read_tx() as tx:
            project = await self._get_active_project_for_activity(
                tx, project_id)
            existing = await self._connectors.find_live_by_project_and_name(
                tx, project_id, normalized_name)
            if existing is not None:
                expected = _db_config(str(existing.id), args.config)
                if (not isinstance(existing, ExternalConnectorDBEntry)
                        or existing.config != expected):
                    raise ConnectorNameConflictError(
                        f"Connector named '{name}' already exists in this "
                        'project')
                persisted_identity = await self._load_bound_identity(
                    tx, project.group_id, existing.identity_id)
                if (persisted_identity.config.type.value
                        != args.config.type.value):
                    raise ConnectorNameConflictError(
                        f"Connector named '{name}' already exists in this "
                        'project')
                if (args.identity_id is not None
                        and existing.identity_id != IdentityID(
                            str(args.identity_id))):
                    raise ConnectorNameConflictError(
                        f"Connector named '{name}' already exists in this "
                        'project')
                if persisted_identity.id not in authorized_identity_ids:
                    raise IdentityForbiddenError(
                        'Connector identity is outside the authorization scope')
                return existing.to_connector_response()
            automatic_selection = False
            if args.identity_id is not None:
                identity = await self._load_bound_identity(
                    tx, project.group_id, IdentityID(str(args.identity_id)))
                if identity.id not in authorized_identity_ids:
                    raise IdentityForbiddenError(
                        'Connector identity is outside the authorization scope')
                if identity.config.type.value != args.config.type.value:
                    raise ValueError('Connector and identity types must match')
                candidates = [identity]
            elif args.config.type == api.ConnectorType.S3:
                candidates = [
                    await self._load_automatic_ambient_s3_identity(
                        tx, project.group_id)
                ]
            else:
                automatic_selection = True
                candidates = await self._bound_identities_for_type(
                    tx, project.group_id, args.config.type)
            candidates = [
                identity for identity in candidates
                if identity.id in authorized_identity_ids
            ]
            candidate_identity_ids = frozenset(candidate.id
                                               for candidate in candidates)

        rootless = isinstance(
            getattr(args.config, 'target', None),
            (api.S3AnyURIConnectorTarget, api.GCSAnyURIConnectorTarget,
             api.ADLSAnyURIConnectorTarget),
        )

        from statedb.config.timestamp import get_current_timestamps

        connector_id = ConnectorID.generate()
        timestamps = get_current_timestamps()

        def entry_for(identity_id: IdentityID) -> ExternalConnectorDBEntry:
            return ExternalConnectorDBEntry(
                id=connector_id,
                project_id=project_id,
                identity_id=identity_id,
                name=name,
                normalized_name=normalized_name,
                created_by=str(created_by),
                description=args.description,
                config=_db_config(str(connector_id), args.config),
                timestamps=timestamps,
            )

        semaphore = asyncio.Semaphore(4)
        # Automatic selection swallows each candidate's provider error, so
        # sanitized (diagnostic_code, correlation_id) pairs are accumulated here
        # (asyncio is single-threaded, so the gathered appends need no lock) to
        # let a single-candidate failure surface the same safe aggregate.
        automatic_diagnostics: list[tuple[str, str]] = []

        async def validate_candidate(
            identity: IdentityDBEntry, ) -> IdentityDBEntry | None:
            async with semaphore:
                entry = entry_for(identity.id)

                def validate() -> None:
                    self._connector_validator(self._compose(identity, entry))

                try:
                    await asyncio.wait_for(
                        _run_blocking(validate),
                        timeout=CONNECTOR_TARGET_VALIDATION_TIMEOUT_SECONDS,
                    )
                except TimeoutError:
                    # The worker thread is deliberately daemonized because
                    # Python cannot interrupt a provider SDK call.  The
                    # request still fails closed at the bounded deadline and
                    # no provider detail crosses the activity boundary.
                    if not automatic_selection:
                        raise ConnectorValidationError(
                            'Connector target validation timed out') from None
                    return None
                except IdentityResolutionUnavailableError:
                    # This is deployment infrastructure, not a candidate IAM
                    # denial.  Retrying other identities that depend on the
                    # same broker cannot produce a trustworthy selection.
                    raise
                except IdentityValidationError:
                    # Credential decryption/exchange failures are already
                    # converted to a sanitized, Temporal-safe error.  Preserve
                    # that distinction for an explicitly requested identity;
                    # in automatic mode it is simply a failed candidate.
                    if not automatic_selection:
                        raise
                    return None
                except Exception as error:
                    # Provider exceptions can contain tokens, signed request
                    # material, or account details. Only the sanitized
                    # diagnostic category and a correlation id ever leave this
                    # boundary; the failed candidate and its error never do.
                    # Both selection modes log so an automatic-mode failure is
                    # still traceable by its correlation id.
                    correlation_id = uuid.uuid4().hex
                    diagnostic_code = connector_validation_diagnostic_value(
                        error)
                    logger.warning(
                        'Connector target validation failed '
                        '(correlation_id=%s error_type=%s '
                        'diagnostic_code=%s)',
                        correlation_id,
                        type(error).__name__,
                        diagnostic_code or 'unclassified',
                    )
                    if not automatic_selection:
                        category = (f' [{diagnostic_code}]'
                                    if diagnostic_code else '')
                        raise ConnectorValidationError(
                            'Connector target validation failed'
                            f'{category}; reference: {correlation_id}'
                        ) from None
                    # Record only classified categories so a single-candidate
                    # typo can surface an aggregate diagnostic; unclassified
                    # failures stay generic but remain traceable through the
                    # logged correlation id.
                    if diagnostic_code is not None:
                        automatic_diagnostics.append(
                            (diagnostic_code, correlation_id))
                    return None
                return identity

        validation_results = await asyncio.gather(
            *(validate_candidate(candidate) for candidate in candidates),
            return_exceptions=True,
        )
        failures = [
            result for result in validation_results
            if isinstance(result, BaseException)
        ]
        if failures:
            # Explicit validation has one candidate.  Automatic validation
            # only lets a sanitized infrastructure failure escape the
            # candidate function.  Awaiting every candidate above avoids
            # leaving detached validation tasks after the request fails.
            raise failures[0]
        successful = [
            identity for identity in validation_results if identity is not None
        ]
        if not successful:
            detail = ('could resolve credentials for rootless operations'
                      if rootless else 'could access this target')
            message = (
                f'No available {args.config.type.value} identity {detail}. '
                'Ask an administrator to grant access or pass an explicit '
                'identity.')
            # When exactly one candidate was checked and it failed with a
            # classified diagnostic, surface the same safe aggregate the
            # explicit path emits. Multi-candidate selection stays generic so
            # it never reveals which bound identity failed.
            if len(candidates) == 1 and len(automatic_diagnostics) == 1:
                code, correlation_id = automatic_diagnostics[0]
                message += f' [{code}]; reference: {correlation_id}'
            raise NoIdentityValidatesTargetError(message=message)
        if automatic_selection and len(successful) > 1:
            choices = ', '.join(
                f'{identity.display_name} ({identity.id})'
                for identity in sorted(
                    successful,
                    key=lambda value:
                    (value.normalized_display_name, str(value.id)),
                ))
            detail = ('are compatible with this rootless connector'
                      if rootless else 'can access this target')
            raise MultipleIdentitiesValidateTargetError(
                message=(f'Multiple identities {detail}; pass an explicit '
                         f'identity: {choices}'))

        identity = successful[0]
        identity_id = identity.id

        async with self._state_db.read_tx() as tx:
            existing = await self._connectors.find_live_by_project_and_name(
                tx, project_id, normalized_name)
            if existing is not None:
                return self._require_idempotent_create_match(
                    existing, args, identity_id)

        entry = entry_for(identity_id)
        try:
            async with self._state_db.write_tx() as tx:
                project = await self._get_active_project_for_activity(
                    tx, project_id)
                # Serialize the final project/binding recheck and connector
                # insert with identity binding replacement/deletion and
                # project archival. JSON-backed StateDB relations have no
                # foreign keys that would provide this protection for us.
                await lock_group_relations(tx, [project.group_id])
                project = await self._get_active_project_for_activity(
                    tx, project_id)
                live_identity = await self._load_bound_identity(
                    tx, project.group_id, identity_id)
                if (live_identity.config.type.value != args.config.type.value):
                    raise ValueError('Connector and identity types must match')
                if automatic_selection:
                    current_candidates = await self._bound_identities_for_type(
                        tx, project.group_id, args.config.type)
                    if (frozenset(candidate.id
                                  for candidate in current_candidates)
                            != candidate_identity_ids):
                        raise IdentityResolutionUnavailableError(
                            'Connector identity bindings changed during '
                            'automatic selection; retry the request')
                existing = await self._connectors.find_live_by_project_and_name(
                    tx, project_id, normalized_name)
                if existing is not None:
                    return self._require_idempotent_create_match(
                        existing, args, identity_id)
                await self._connectors.create(tx, entry)
        except _UNIQUE_VIOLATION_ERRORS as error:
            async with self._state_db.read_tx() as tx:
                existing = await self._connectors.find_live_by_project_and_name(
                    tx, project_id, normalized_name)
            if existing is not None:
                return self._require_idempotent_create_match(
                    existing, args, identity_id)
            raise ConnectorNameConflictError(
                f"Connector named '{name}' already exists in this project"
            ) from error
        return entry.to_connector_response()

    @staticmethod
    def _require_idempotent_create_match(
        existing: ExternalConnectorDBEntry | InternalConnectorDBEntry,
        args: api.CreateConnectorArgs,
        identity_id: IdentityID,
    ) -> api.ConnectorResponse:
        expected = _db_config(str(existing.id), args.config)
        if (isinstance(existing, ExternalConnectorDBEntry)
                and existing.identity_id == identity_id
                and existing.config == expected):
            return existing.to_connector_response()
        raise ConnectorNameConflictError(
            f"Connector named '{args.name.strip()}' already exists in this "
            'project')

    async def _bound_identities_for_type(
        self,
        tx: StateTx,
        group_id: GroupID,
        connector_type: api.ConnectorType,
    ) -> list[IdentityDBEntry]:
        await _require_active_group(tx, self._groups, group_id)
        bindings = await self._bindings.list_by_field(tx, 'group_id',
                                                      str(group_id))
        rows = await self._identities.list_by_ids(
            tx, [str(binding.payload.identity_id) for binding in bindings])
        return [
            row.payload for row in rows
            if (row.payload.tombstoned_at is None
                and row.payload.config.type.value == connector_type.value
                and _system_ambient_identity_is_available(row.payload))
        ]

    async def _get_active_project_for_activity(
        self,
        tx: StateTx,
        project_id: ProjectID,
    ) -> ProjectDBEntry:
        row = await self._projects.get(tx, str(project_id))
        if row is None or row.payload.status != ProjectStatus.ACTIVE:
            raise ProjectNotFoundError(
                f"Active project '{project_id}' not found")
        project = row.payload
        await _require_active_group(tx, self._groups, project.group_id)
        return project

    async def list_for_project(
        self,
        project_id: ProjectID,
        principal: AuthPrincipal | None,
    ) -> list[api.ConnectorResponse]:
        principal = _require_principal(principal)
        async with self._state_db.read_tx() as tx:
            project = await _get_project_for_principal(
                tx,
                self._projects,
                project_id,
                principal,
                require_active=False,
            )
            connectors = await self._connectors.list(
                tx,
                project_id=project_id,
            )
            connectors = await self._visible_external_connectors(
                tx, project.group_id, connectors)
        return [connector.to_connector_response() for connector in connectors]

    async def get_for_project(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        principal: AuthPrincipal | None,
    ) -> api.ConnectorResponse:
        principal = _require_principal(principal)
        async with self._state_db.read_tx() as tx:
            project = await _get_project_for_principal(
                tx,
                self._projects,
                project_id,
                principal,
                require_active=False,
            )
            connector = await self._get_visible_external(
                tx, project, connector_id)
        return connector.to_connector_response()

    async def require_delegatable_for_project(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
    ) -> None:
        """Recheck one connector before granting authority to a child.

        External connectors must still be visible through the owning
        project's current identity-group binding. The only internal connector
        that may cross a workflow boundary is the exact live canonical project
        upload connector. Public connector discovery remains external-only.
        """
        async with self._state_db.read_tx() as tx:
            project = await self._get_active_project_for_activity(
                tx, project_id)
            connector = await self._get_connector(tx, project_id, connector_id)
            if isinstance(connector, InternalConnectorDBEntry):
                if not is_active_project_upload_connector(connector):
                    raise ConnectorDoesNotExistError(connector_id)
                return
            await self._require_visible_external_connector(
                tx, project.group_id, connector)

    async def source_table_connector_for_project(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        principal: AuthPrincipal | None,
    ) -> api.ConnectorResponse | None:
        """Return safe source-table metadata for a project connector.

        External connectors return their credential-free API response. The
        one hidden project upload connector returns ``None`` so callers can
        construct a file-backed source-table reference without receiving its
        internal root. All other internal connectors remain unavailable from
        public request paths.
        """
        principal = _require_principal(principal)
        async with self._state_db.read_tx() as tx:
            project = await _get_project_for_principal(
                tx,
                self._projects,
                project_id,
                principal,
            )
            await _require_active_group(tx, self._groups, project.group_id)
            connector = await self._get_connector(tx, project_id, connector_id)
            if isinstance(connector, InternalConnectorDBEntry):
                if not is_active_project_upload_connector(connector):
                    raise ConnectorDoesNotExistError(connector_id)
                return None
            connector = await self._require_visible_external_connector(
                tx, project.group_id, connector)
        return connector.to_connector_response()

    async def update_metadata(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        args: api.UpdateConnectorMetadataArgs,
        principal: AuthPrincipal | None,
    ) -> api.ConnectorResponse:
        principal = _require_principal(principal)
        from statedb.config.timestamp import update_timestamps

        try:
            async with self._state_db.write_tx() as tx:
                project = await _get_project_for_principal(
                    tx, self._projects, project_id, principal)
                await lock_group_relations(tx, [project.group_id])
                project = await _get_project_for_principal(
                    tx, self._projects, project_id, principal)
                connector = await self._get_visible_external(
                    tx, project, connector_id)
                if args.name is not None:
                    name, normalized_name = _validated_connector_name(args.name)
                    connector.name = name
                    connector.normalized_name = normalized_name
                if args.description is not None:
                    connector.description = args.description
                connector.timestamps = update_timestamps(connector.timestamps)
                await self._connectors.upsert(tx, connector)
        except _UNIQUE_VIOLATION_ERRORS as error:
            raise ConnectorNameConflictError(
                f"Connector named '{args.name}' already exists in this "
                'project') from error
        return connector.to_connector_response()

    async def delete(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        principal: AuthPrincipal | None,
    ) -> api.ConnectorResponse:
        principal = _require_principal(principal)
        from statedb.config.timestamp import now_utc, update_timestamps

        async with self._state_db.write_tx() as tx:
            await lock_connector_admission_and_delete(tx, project_id,
                                                      connector_id)
            project = await _get_project_for_principal(tx, self._projects,
                                                       project_id, principal)
            await lock_group_relations(tx, [project.group_id])
            project = await _get_project_for_principal(tx, self._projects,
                                                       project_id, principal)
            connector = (await self._get_external(tx, project.id, connector_id)
                         if principal.is_admin else await
                         self._get_visible_external(tx, project, connector_id))
            live, orphaned, stale_graphs = (
                await self._tables.partition_connector_references(
                    tx, project_id, connector_id))
            if live:
                names = ', '.join(
                    sorted(entry.alias or str(entry.id) for entry in live))
                raise ConnectorInUseError(
                    f'Connector {connector_id} is referenced by a live table: '
                    f'{names}')
            # Rows nothing reaches are collected here rather than left behind
            # pointing at a tombstoned connector. The admission lock taken
            # above fences concurrent reference writes, so this set is stable.
            #
            # A graph that pinned one of them goes in the same transaction when
            # nothing can reach it either. Keeping it would hold the wedge one
            # level up, since the only graph delete route is keyed by alias;
            # dropping only its tables would leave it readable by
            # ``graph-<fpid>`` and raising on the missing fingerprint.
            for graph in stale_graphs:
                # Graph definitions hold a claim on every table they name.
                # Release those claims before collecting the graph, including
                # claims on tables belonging to other connectors. Otherwise a
                # stale multi-connector graph leaves alias-less residue behind
                # when only one of its connectors is deleted.
                for table_fpid in graph.table_name_to_dbid.values():
                    await self._tables.decrement_or_delete(
                        tx, project_id, table_fpid)
                await self._graphs.delete(tx, project_id, graph.id)
            for entry in orphaned:
                await self._tables.delete(tx, project_id, entry.id)
            connector.tombstoned_at = now_utc()
            connector.timestamps = update_timestamps(connector.timestamps)
            await self._connectors.upsert(tx, connector)
        return connector.to_connector_response()

    async def config_for_project(
        self,
        principal: AuthPrincipal | None,
        project_id: ProjectID,
        connector_id: ConnectorID,
    ) -> ConnectorSpec:
        """Return secret-free connector configuration for request parsing."""
        principal = _require_principal(principal)
        async with self._state_db.read_tx() as tx:
            project = await _get_project_for_principal(tx, self._projects,
                                                       project_id, principal)
            connector = await self._get_connector(tx, project_id, connector_id)
            if isinstance(connector, InternalConnectorDBEntry):
                return self._compose_internal(connector)
            identity = await self._load_bound_identity(tx, project.group_id,
                                                       connector.identity_id)
            if connector.config.type.value != identity.config.type.value:
                raise ValueError('Connector and identity types differ')
            if connector.config.type == api.ConnectorType.DATABRICKS:
                native = connector.config.as_connector(
                    identity.config.host,
                    require_explicit_credentials=True,
                )
            elif connector.config.type == api.ConnectorType.SNOWFLAKE:
                native = connector.config.as_connector(
                    identity.config.account,
                    require_explicit_credentials=True,
                )
            else:
                native = connector.config.as_connector(
                    require_explicit_credentials=True)
            return ConnectorSpec(
                connector_id=connector.id,
                connector_config=native.config(),
            )

    async def resolve_for_activity(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
    ) -> ConnectorOpenSpec:
        """Resolve project-owned credentials only inside a trusted activity."""
        async with self._state_db.read_tx() as tx:
            project = await self._get_active_project_for_activity(
                tx, project_id)
            connector = await self._get_connector(tx, project_id, connector_id)
            if isinstance(connector, InternalConnectorDBEntry):
                return self._compose_internal(connector)
            identity = await self._load_bound_identity(tx, project.group_id,
                                                       connector.identity_id)
            if connector.config.type.value != identity.config.type.value:
                raise ValueError('Connector and identity types differ')
            return self._compose(identity, connector)

    async def require_project_access(
        self,
        principal: AuthPrincipal | None,
        project_id: ProjectID,
    ) -> ProjectDBEntry:
        principal = _require_principal(principal)
        async with self._state_db.read_tx() as tx:
            return await _get_project_for_principal(tx, self._projects,
                                                    project_id, principal)

    async def execution_ref_for_project(
        self,
        principal: AuthPrincipal | None,
        project_id: ProjectID,
        connector_id: ConnectorID,
        allowed_actions: Iterable[ConnectorCapabilityAction],
        expected_type: api.ConnectorType | None = None,
    ) -> ConnectorExecutionRef:
        principal = _require_principal(principal)
        auth_backend, deploy_config = _authorization_issue_context(principal)
        async with self._state_db.write_tx() as tx:
            project = await _get_project_for_principal(tx, self._projects,
                                                       project_id, principal)
            connector = await self._get_connector(tx, project_id, connector_id)
            if isinstance(connector, ExternalConnectorDBEntry):
                identity = await self._load_bound_identity(
                    tx, project.group_id, connector.identity_id)
                if connector.config.type.value != identity.config.type.value:
                    raise ValueError('Connector and identity types differ')
            if (expected_type is not None
                    and connector.config.type.value != expected_type.value):
                raise ValueError(
                    f'Connector {connector_id} is not a {expected_type.value} '
                    'connector')
            _, token = await self._connector_authorizations.issue_in_tx(
                tx,
                principal=principal,
                auth_backend=auth_backend,
                resource_kind=ConnectorAuthorizationResourceKind.CONNECTOR,
                resource_id=str(connector_id),
                project_id=str(project_id),
                request_digest='',
                allowed_actions=allowed_actions,
                deployment_id=(f'{deploy_config.instance_id}:'
                               f'{deploy_config.client_id}'),
                temporal_namespace=deploy_config.temporal_namespace,
            )
        return ConnectorExecutionRef(project_id=project_id,
                                     connector_id=connector_id,
                                     authorization_token=token)

    async def fresh_execution_refs_for_workflow(
        self,
        principal: AuthPrincipal | None,
        references: Iterable[ConnectorExecutionRef],
        allowed_actions: Iterable[ConnectorCapabilityAction],
    ) -> tuple[ConnectorExecutionRef, ...]:
        """Mint unbound capabilities for one subsequent workflow start.

        A connector capability becomes permanently bound to the first Temporal
        workflow ID that receives it.  Callers crossing another workflow
        boundary must mint from the authenticated principal and the original
        project/connector resource scope; copying or rebinding the opaque token
        would defeat replay protection.
        """
        principal = _require_principal(principal)
        actions = tuple(allowed_actions)
        fresh = []
        for reference in references:
            fresh.append(await self.execution_ref_for_project(
                principal,
                reference.project_id,
                reference.connector_id,
                actions,
            ))
        return tuple(fresh)

    async def get_internal(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
    ) -> InternalConnectorDBEntry:
        async with self._state_db.read_tx() as tx:
            connector = await self._get_connector(tx, project_id, connector_id)
        if not isinstance(connector, InternalConnectorDBEntry):
            raise IdentityForbiddenError(
                'External connector requires project authorization')
        return connector

    async def register_project_upload_connector(
        self,
        project_id: ProjectID,
        root_dir: str,
    ) -> ConnectorID:
        from statedb.config.timestamp import get_current_timestamps

        name = PROJECT_UPLOAD_CONNECTOR_NAME
        async with self._state_db.write_tx() as tx:
            project = await self._projects.get(tx, str(project_id))
            if project is None or project.payload.status != ProjectStatus.ACTIVE:
                raise ProjectNotFoundError(
                    f"Active project '{project_id}' not found")
            existing = await self._connectors.find_live_by_project_and_name(
                tx, project_id, normalize_display_name(name))
            if existing is not None:
                if (not is_active_project_upload_connector(existing)
                        or existing.config.root_dir != root_dir):
                    raise ValueError(
                        'Invalid existing project upload connector')
                return existing.id
            connector = InternalConnectorDBEntry(
                id=ConnectorID.generate(),
                project_id=project_id,
                name=name,
                normalized_name=normalize_display_name(name),
                description='Project-scoped CSV and Parquet uploads',
                config=FileConnectorResourceConfig(name=name,
                                                   root_dir=root_dir),
                timestamps=get_current_timestamps(),
            )
            await self._connectors.create(tx, connector)
            return connector.id

    @staticmethod
    def connector_ids_for_graph(graph_definition) -> set[ConnectorID]:
        return {
            table.get_connector_id()
            for table in graph_definition.tables.values()
        }

    async def _get_external(
        self,
        tx: StateTx,
        project_id: ProjectID,
        connector_id: ConnectorID,
    ) -> ExternalConnectorDBEntry:
        connector = await self._get_connector(tx, project_id, connector_id)
        if not isinstance(connector, ExternalConnectorDBEntry):
            raise ConnectorDoesNotExistError(connector_id)
        return connector

    async def _get_visible_external(
        self,
        tx: StateTx,
        project: ProjectDBEntry,
        connector_id: ConnectorID,
    ) -> ExternalConnectorDBEntry:
        connector = await self._get_external(tx, project.id, connector_id)
        return await self._require_visible_external_connector(
            tx, project.group_id, connector)

    async def _require_visible_external_connector(
        self,
        tx: StateTx,
        group_id: GroupID,
        connector: ExternalConnectorDBEntry,
    ) -> ExternalConnectorDBEntry:
        visible = await self._visible_external_connectors(
            tx, group_id, [connector])
        if not visible:
            raise IdentityForbiddenError(
                'Connector identity is not mapped to the project group')
        return visible[0]

    async def _visible_external_connectors(
        self,
        tx: StateTx,
        group_id: GroupID,
        connectors: list[ExternalConnectorDBEntry | InternalConnectorDBEntry],
    ) -> list[ExternalConnectorDBEntry]:
        """Return discoverable external connectors with constant query count.

        Connector discovery is conditional on the identity's current binding
        to the owning project's group. Load the group's bindings and all
        referenced identities in two batched reads so project listing does not
        issue one identity/binding lookup per connector.
        """
        await _require_active_group(tx, self._groups, group_id)
        external = [
            connector for connector in connectors
            if isinstance(connector, ExternalConnectorDBEntry)
            and connector.tombstoned_at is None
        ]
        if not external:
            return []
        bindings = await self._bindings.list_by_field(tx, 'group_id',
                                                      str(group_id))
        bound_identity_ids = {
            binding.payload.identity_id
            for binding in bindings
        }
        referenced_identity_ids = {
            connector.identity_id
            for connector in external
            if connector.identity_id in bound_identity_ids
        }
        identity_rows = await self._identities.list_by_ids(
            tx, [str(identity_id) for identity_id in referenced_identity_ids])
        identities = {row.payload.id: row.payload for row in identity_rows}
        return [
            connector for connector in external
            if (identity := identities.get(connector.identity_id)) is not None
            and identity.tombstoned_at is None
            and connector.config.type.value == identity.config.type.value
            and _system_ambient_identity_is_available(identity)
        ]

    async def _get_connector(
        self,
        tx: StateTx,
        project_id: ProjectID,
        connector_id: ConnectorID,
    ) -> ExternalConnectorDBEntry | InternalConnectorDBEntry:
        connector = await self._connectors.get(tx, connector_id)
        if (connector is None or connector.project_id != project_id
                or connector.tombstoned_at is not None):
            raise ConnectorDoesNotExistError(connector_id)
        return connector

    async def _load_bound_identity(
        self,
        tx: StateTx,
        group_id: GroupID,
        identity_id: IdentityID,
    ) -> IdentityDBEntry:
        await _require_active_group(tx, self._groups, group_id)
        row = await self._identities.get(tx, str(identity_id))
        if row is None or row.payload.tombstoned_at is not None:
            raise IdentityDoesNotExistError(identity_id)
        binding = await self._bindings.find_one_by_fields(
            tx,
            {
                'identity_id': str(identity_id),
                'group_id': str(group_id),
            },
        )
        if binding is None:
            raise IdentityForbiddenError(
                'Identity is not mapped to the project group')
        identity = row.payload
        if is_ambient_s3_identity(identity.config):
            if identity.id != AMBIENT_S3_IDENTITY_ID:
                raise IdentityForbiddenError(
                    'Ambient S3 identity is not canonical')
            _require_ambient_s3_enabled()
        elif identity.id == AMBIENT_ADLS_IDENTITY_ID:
            if not isinstance(identity.config, api.ADLSIdentityConfig):
                raise IdentityForbiddenError(
                    'Ambient ADLS identity is not canonical')
            _require_ambient_adls_enabled()
        return identity

    async def _load_automatic_ambient_s3_identity(
        self,
        tx: StateTx,
        group_id: GroupID,
    ) -> IdentityDBEntry:
        """Resolve omitted S3 identity without exposing the system row ID."""
        try:
            if not _ambient_s3_enabled():
                raise IdentityResolutionUnavailableError(
                    'Ambient S3 workload identity is disabled')
            return await self._load_bound_identity(tx, group_id,
                                                   AMBIENT_S3_IDENTITY_ID)
        except (IdentityDoesNotExistError, IdentityForbiddenError,
                IdentityResolutionUnavailableError):
            raise NoIdentityValidatesTargetError(
                message=('Ambient S3 access is disabled or unavailable for '
                         'this project. Ask an administrator to enable and '
                         'assign ambient S3 access.')) from None

    def _compose(
        self,
        identity: IdentityDBEntry,
        connector: ExternalConnectorDBEntry,
    ) -> ResolvedConnectorSpec:
        try:
            return compose_resolved_connector_spec(
                identity,
                connector,
                _credentials(self._encryptor, identity),
                self._workload_exchanger,
            )
        except IdentityAuthenticationCategoryError as error:
            if error.category == IdentityAuthenticationCategory.STORED_KEY_INVALID:
                raise IdentityValidationError(str(error)) from None
            raise IdentityResolutionUnavailableError(str(error)) from None
        except WorkloadExchangeConfigurationError:
            raise IdentityResolutionUnavailableError(
                'Connector identity resolution is temporarily unavailable; '
                'please retry or contact an administrator') from None
        except Exception:
            raise IdentityValidationError(
                'Identity credential resolution failed') from None

    @staticmethod
    def _compose_internal(
        connector: InternalConnectorDBEntry, ) -> ConnectorSpec:
        # Display names are mutable project metadata.  Runtime graph/table
        # references use the immutable connector ID, so the native connector
        # must be keyed by that ID just like an external connector.  Keeping
        # the persisted config's human name here makes Graph.connect() index
        # the connector under (for example) ``uploads`` while tables refer to
        # ``connector_<id>``, which fails with a misleading KeyError.
        runtime_config = replace(connector.config, name=str(connector.id))
        return ConnectorSpec(
            connector_id=connector.id,
            connector_config=runtime_config.as_connector().config(),
            is_project_upload=is_active_project_upload_connector(connector),
        )
