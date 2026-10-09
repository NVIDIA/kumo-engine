# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import re
import threading
import time
from dataclasses import dataclass, replace
from datetime import timedelta
from functools import lru_cache
from test.statedb.db.connector_fixtures import seed_connector
from test.statedb.db.matrix_assertions import only_instance
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import kumoapi.data_source as api
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from kumo_core.model_plan import ModelPlan
from kumo_core.types.schema import FPID
from kumoapi.typing import Dtype, Stype
from pydantic import SecretStr

from rest.config.connector import FileConnectorResourceConfig
from rest.config.connector_db import InternalConnectorDBEntry
from rest.config.credentials import (
    AWSAccessKeyCredentials,
    AWSFederatedCredentials,
    AzureFederatedCredentials,
    DatabricksCredentials,
    GoogleCloudCredentials,
)
from rest.config.data_backend import ConnectorExecutionRef, ConnectorSpec
from rest.config.encryption import CredentialsEncryptor
from rest.config.graph_v2 import GraphDBEntryV2
from rest.config.identity import (
    IdentityAuthenticationCategory,
    IdentityAuthenticationCategoryError,
    WorkloadExchangeConfigurationError,
    identity_key,
)
from rest.config.table_v2 import (
    Column,
    DatabricksSourceTable,
    DataSourceType,
    TableDBEntryV2,
    TableDefinition,
)
from rest.config.training_job import CustomTrainingTable
from rest.utils.ambient_identity import ensure_startup_ambient_identities
from statedb.config.common import (
    ArtifactExportJobID,
    ConnectorID,
    GraphIDv1,
    GroupID,
    PredictiveQueryID,
    ProjectID,
    TableName,
    UserID,
    now_utc,
)
from statedb.config.connector import PROJECT_UPLOAD_CONNECTOR_NAME
from statedb.config.predictive_query import PQueryDefinition
from statedb.config.timestamp import get_current_timestamps
from statedb.db import MLJobService, StateDB
from statedb.db.artifact_exports import ArtifactExportJobService
from statedb.db.connector_authorizations import (
    ConnectorAuthorizationError,
    ConnectorAuthorizationService,
)
from statedb.db.connectors import StateDBConnectorRepository
from statedb.db.graphs import StateDBGraphRepository
from statedb.db.identities import (
    AMBIENT_ADLS_IDENTITY_DISPLAY_NAME,
    AMBIENT_ADLS_IDENTITY_ID,
    AMBIENT_ADLS_IDENTITY_SLUG,
    AMBIENT_S3_IDENTITY_DISPLAY_NAME,
    AMBIENT_S3_IDENTITY_ID,
    MAX_CONNECTOR_NAME_LENGTH,
    ConnectorService,
    IdentityService,
)
from statedb.db.pqueries import StateDBPQueryRepository
from statedb.db.repositories import JsonRepository
from statedb.db.services import GraphService, ProjectService, TableService
from statedb.db.tables import StateDBTableRepository
from statedb.rest.connector_authorization import (
    CONNECTOR_GRAPH_ACTIONS,
    CONNECTOR_OUTPUT_ACTIONS,
)
from statedb.rest.group import GroupStatus, KumoGroupDBEntry
from statedb.rest.identity import IdentityDBEntry
from statedb.rest.project import ProjectDBEntry, ProjectStatus
from statedb.utils.errors import (
    ConnectorDoesNotExistError,
    ConnectorInUseError,
    ConnectorNameConflictError,
    ConnectorValidationDiagnosticCode,
    ConnectorValidationDiagnosticError,
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
)


@dataclass(frozen=True)
class _Principal:
    user_id: UserID
    user_name: str
    kumo_group_ids: tuple[str, ...]
    is_admin: bool = False
    external_membership_unavailable: bool = False


ADMIN = _Principal(UserID('admin'), 'Admin', (), True)


class _TestWorkloadExchanger:
    def exchange(self, config, credentials):
        if isinstance(config, api.S3IdentityConfig):
            if isinstance(config.authentication,
                          api.S3AccessKeyAuthentication):
                assert isinstance(credentials, api.S3AccessKeyCredentials)
                return AWSAccessKeyCredentials(
                    access_key_id=credentials.access_key_id,
                    secret_access_key=credentials.secret_access_key,
                    session_token=credentials.session_token,
                    region=(config.connection.region
                            if config.connection else None),
                )
            return AWSFederatedCredentials(
                access_key_id=SecretStr('temporary-access-key'),
                secret_access_key=SecretStr('temporary-secret-key'),
                session_token=SecretStr('temporary-session-token'),
            )
        if isinstance(config,
                      (api.BigQueryIdentityConfig, api.GCSIdentityConfig)):
            return GoogleCloudCredentials(
                provider=config.type.value,
                credential=MagicMock(name='gcs-credential'),
                project='workload-project',
            )
        if isinstance(config, api.ADLSIdentityConfig):
            return AzureFederatedCredentials(credential=MagicMock(
                name='azure-credential'))
        raise TypeError(type(config))


@pytest.fixture
def db(tmp_path):
    state_db = asyncio.run(
        StateDB.open_sqlite(str(tmp_path / 'state.db'), pool_size=1))
    try:
        yield state_db
    finally:
        asyncio.run(state_db.close())


async def _create_group(db: StateDB, name: str) -> GroupID:
    group_id = GroupID.generate()
    entry = KumoGroupDBEntry(
        id=group_id,
        slug=f"group-{str(group_id)[-8:]}",
        display_name=name,
        normalized_display_name=name.casefold(),
        description='',
        status=GroupStatus.ACTIVE,
        timestamps=get_current_timestamps(),
    )
    repo = JsonRepository('groups', KumoGroupDBEntry)
    async with db.write_tx() as tx:
        await repo.upsert(tx, str(group_id), entry)
    return group_id


async def _create_project(
    db: StateDB,
    group_id: GroupID,
    name: str,
    *,
    owner: str = 'owner',
    private: bool = False,
) -> ProjectID:
    project_id = ProjectID.generate()
    entry = ProjectDBEntry(
        id=project_id,
        group_id=group_id,
        slug=f"project-{str(project_id)[-8:]}",
        display_name=name,
        normalized_display_name=name.casefold(),
        created_by=owner,
        created_by_name=owner,
        is_private=private,
        status=ProjectStatus.ACTIVE,
        timestamps=get_current_timestamps(),
    )
    repo = JsonRepository('projects', ProjectDBEntry)
    async with db.write_tx() as tx:
        await repo.upsert(tx, str(project_id), entry)
    return project_id


def _identity_args(name: str, secret: str) -> api.CreateIdentityArgs:
    return api.CreateIdentityArgs(
        display_name=name,
        config=api.DatabricksIdentityConfig(
            host='https://workspace.cloud.databricks.com',
            client_id=f'{name}-service-principal',
        ),
        credentials=api.DatabricksOAuthCredentials(
            client_secret=SecretStr(secret)),
    )


def _s3_access_key_args(
    name: str,
    access_key_id: str,
    secret_access_key: str,
    *,
    endpoint_url: str = 'https://minio.example.test:9000',
) -> api.CreateIdentityArgs:
    return api.CreateIdentityArgs(
        display_name=name,
        config=api.S3IdentityConfig(
            authentication=api.S3AccessKeyAuthentication(),
            connection=api.S3ConnectionConfig(
                endpoint_url=endpoint_url,
                region='us-east-1',
                addressing_style=api.S3AddressingStyle.PATH,
            ),
        ),
        credentials=api.S3AccessKeyCredentials(
            access_key_id=SecretStr(access_key_id),
            secret_access_key=SecretStr(secret_access_key),
            session_token=SecretStr(f'{name}-session-token'),
        ),
    )


@lru_cache(maxsize=1)
def _google_private_key_pem() -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode('utf-8')


def _google_key_authentication(
    email: str = 'kumo@service-project.iam.gserviceaccount.com',
) -> api.GoogleServiceAccountKeyAuthentication:
    return api.GoogleServiceAccountKeyAuthentication(
        project_id='service-project',
        client_email=email,
        client_id='1234567890',
        auth_uri='https://accounts.google.com/o/oauth2/auth',
        token_uri='https://oauth2.googleapis.com/token',
    )


def _google_identity_args(
    provider: str,
    mode: str,
    *,
    name: str,
    email: str = 'kumo@service-project.iam.gserviceaccount.com',
) -> api.CreateIdentityArgs:
    config_type = (api.BigQueryIdentityConfig
                   if provider == 'bigquery' else api.GCSIdentityConfig)
    if mode == 'service_account_key':
        return api.CreateIdentityArgs(
            display_name=name,
            config=config_type(
                authentication=_google_key_authentication(email)),
            credentials=api.GoogleServiceAccountKeyCredentials(
                private_key_id=SecretStr(f'{name}-key-id'),
                private_key=SecretStr(_google_private_key_pem()),
            ),
        )
    return api.CreateIdentityArgs(
        display_name=name,
        config=config_type(
            authentication=api.GoogleWorkloadIdentityAuthentication(
                service_account_email=email)),
    )


def _connector_args(
    name: str,
    identity_id: str | None,
    schema_name: str,
) -> api.CreateConnectorArgs:
    return api.CreateConnectorArgs(
        identity_id=identity_id,
        name=name,
        config=api.DatabricksConnectorResourceConfig(
            warehouse_id=f'{name}-warehouse',
            catalog='sales',
            schema_name=schema_name,
        ),
    )


def _new_identity_service(
    db: StateDB,
    connector_authorizations: ConnectorAuthorizationService | None = None,
    **kwargs,
) -> IdentityService:
    return IdentityService(
        db, connector_authorizations or ConnectorAuthorizationService(db),
        **kwargs)


def _new_connector_service(
    db: StateDB,
    connector_authorizations: ConnectorAuthorizationService | None = None,
    **kwargs,
) -> ConnectorService:
    return ConnectorService(
        db, connector_authorizations or ConnectorAuthorizationService(db),
        **kwargs)


def _services(db: StateDB):
    exchanger = _TestWorkloadExchanger()
    connector_authorizations = ConnectorAuthorizationService(db)
    identities = _new_identity_service(
        db,
        connector_authorizations,
        identity_validator=lambda *_: None,
        workload_exchanger=exchanger,
    )
    connectors = _new_connector_service(
        db,
        connector_authorizations,
        connector_validator=lambda *_: None,
        workload_exchanger=exchanger,
    )
    return identities, connectors


@pytest.mark.asyncio
async def test_s3_access_key_identity_is_encrypted_and_fingerprinted(
        db: StateDB) -> None:
    identities, _ = _services(db)
    args = _s3_access_key_args(
        'minio-identity',
        'minio-access-key',
        'minio-secret-key',
    )

    response = await identities.create(args, ADMIN)

    assert response.config == args.config
    assert response.credential_status.configured is True
    assert response.credential_status.last_rotated_at is not None
    assert not hasattr(response, 'credentials')
    repository = JsonRepository('identities', IdentityDBEntry)
    async with db.read_tx() as tx:
        row = await repository.get(tx, response.id)
    assert row is not None
    stored = row.payload
    assert stored.identity_key == identity_key(args.config, args.credentials)
    assert stored.identity_key.startswith('s3:access_key:')
    serialized = repr(stored.encrypted_credentials)
    for secret in (
        'minio-access-key',
        'minio-secret-key',
        'minio-identity-session-token',
    ):
        assert secret not in serialized


@pytest.mark.asyncio
async def test_s3_access_key_rotation_updates_identity_key_atomically(
        db: StateDB) -> None:
    identities, _ = _services(db)
    first_args = _s3_access_key_args('first-minio', 'first-key', 'first-secret')
    second_args = _s3_access_key_args(
        'second-minio', 'second-key', 'second-secret')
    first = await identities.create(first_args, ADMIN)
    await identities.create(second_args, ADMIN)
    repository = JsonRepository('identities', IdentityDBEntry)

    conflicting = api.S3AccessKeyCredentials(
        access_key_id=SecretStr('second-key'),
        secret_access_key=SecretStr('replacement-secret'),
    )
    with pytest.raises(IdentityConflictError):
        await identities.rotate_credentials(
            first.id,
            api.RotateIdentityCredentialsArgs(credentials=conflicting),
            ADMIN,
        )
    async with db.read_tx() as tx:
        unchanged = await repository.get(tx, first.id)
    assert unchanged is not None
    assert unchanged.payload.identity_key == identity_key(
        first_args.config, first_args.credentials)

    rotated_credentials = api.S3AccessKeyCredentials(
        access_key_id=SecretStr('rotated-key'),
        secret_access_key=SecretStr('rotated-secret'),
        session_token=SecretStr('rotated-token'),
    )
    rotated = await identities.rotate_credentials(
        first.id,
        api.RotateIdentityCredentialsArgs(credentials=rotated_credentials),
        ADMIN,
    )
    async with db.read_tx() as tx:
        updated = await repository.get(tx, first.id)
    assert updated is not None
    assert updated.payload.identity_key == identity_key(
        first_args.config, rotated_credentials)
    decrypted = CredentialsEncryptor().decrypt_identity_credentials(
        updated.payload.encrypted_credentials)
    assert decrypted == rotated_credentials
    assert rotated.credential_status.last_rotated_at is not None


@pytest.mark.parametrize(
    ('submitted_ca', 'preserve_ca', 'expected_ca'),
    [
        (None, True, 'initial-ca'),
        ('replacement-ca', False, 'replacement-ca'),
        (None, False, None),
    ],
)
@pytest.mark.asyncio
async def test_s3_access_key_rotation_ca_certificate_tri_state(
        db: StateDB, submitted_ca: str | None, preserve_ca: bool,
        expected_ca: str | None) -> None:
    validator = MagicMock()
    identities = _new_identity_service(
        db,
        identity_validator=validator,
        workload_exchanger=_TestWorkloadExchanger(),
    )
    certificates = {
        'initial-ca': SecretStr(
            '-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----'),
        'replacement-ca': SecretStr(
            '-----BEGIN CERTIFICATE-----\nMIIC\n-----END CERTIFICATE-----'),
    }
    create_args = _s3_access_key_args(
        'private-ca-minio',
        'initial-key',
        'initial-secret',
    )
    assert isinstance(create_args.credentials, api.S3AccessKeyCredentials)
    create_args = replace(
        create_args,
        credentials=replace(
            create_args.credentials,
            ca_certificate=certificates['initial-ca'],
        ),
    )
    created = await identities.create(create_args, ADMIN)
    validator.reset_mock()

    submitted = api.S3AccessKeyCredentials(
        access_key_id=SecretStr('rotated-key'),
        secret_access_key=SecretStr('rotated-secret'),
        ca_certificate=(certificates[submitted_ca]
                        if submitted_ca is not None else None),
    )
    await identities.rotate_credentials(
        created.id,
        api.RotateIdentityCredentialsArgs(credentials=submitted),
        ADMIN,
        preserve_s3_ca_certificate=preserve_ca,
    )

    expected = replace(
        submitted,
        ca_certificate=(certificates[expected_ca]
                        if expected_ca is not None else None),
    )
    validator.assert_called_once_with(
        create_args.config,
        expected,
        identities._workload_exchanger,
    )
    repository = JsonRepository('identities', IdentityDBEntry)
    async with db.read_tx() as tx:
        updated = await repository.get(tx, created.id)
    assert updated is not None
    decrypted = CredentialsEncryptor().decrypt_identity_credentials(
        updated.payload.encrypted_credentials)
    assert decrypted == expected


@pytest.mark.asyncio
async def test_s3_connector_resolution_carries_identity_owned_connection(
        db: StateDB) -> None:
    group_id = await _create_group(db, 'minio-team')
    project_id = await _create_project(db, group_id, 'minio-project')
    member = _Principal(UserID('minio-member'), 'Member', (str(group_id), ))
    identities, connectors = _services(db)
    args = replace(
        _s3_access_key_args(
            'connector-minio',
            'connector-access',
            'connector-secret',
        ),
        group_ids=[str(group_id)],
    )
    identity = await identities.create(args, ADMIN)
    connector = await _create_connector(
        connectors,
        project_id,
        api.CreateConnectorArgs(
            identity_id=identity.id,
            name='customer-data',
            config=api.S3ConnectorResourceConfig(
                target=api.S3RootedConnectorTarget(
                    bucket='customer-data',
                    prefix='ltv',
                ),
            ),
        ),
        member,
    )

    resolved = await connectors.resolve_for_activity(project_id, connector.id)

    assert isinstance(resolved.credentials, AWSAccessKeyCredentials)
    connection = resolved.credentials.connection
    assert connection is not None
    assert connection.owner == f'connector:{identity.id}'
    assert connection.endpoint_url == 'https://minio.example.test:9000'
    assert connection.region == 'us-east-1'
    assert connection.addressing_style == 'path'
    assert connection.send_bucket_owner_full_control is False
    assert connection.access_key == 'connector-access'
    assert connection.secret_key == 'connector-secret'


@pytest.mark.parametrize('provider', ['bigquery', 'gcs'])
@pytest.mark.parametrize('mode', ['service_account_key', 'workload_identity'])
@pytest.mark.asyncio
async def test_google_identity_mode_persistence_and_status(
        db: StateDB, provider: str, mode: str) -> None:
    identities, _ = _services(db)
    args = _google_identity_args(
        provider,
        mode,
        name=f'{provider}-{mode}',
    )

    response = await identities.create(args, ADMIN)

    assert response.credential_status.configured is True
    assert (response.credential_status.last_rotated_at
            is not None) == (mode == 'service_account_key')
    assert identity_key(args.config) == (
        f'{provider}:{mode}:kumo@service-project.iam.gserviceaccount.com')
    repository = JsonRepository('identities', IdentityDBEntry)
    async with db.read_tx() as tx:
        row = await repository.get(tx, response.id)
    assert row is not None
    encrypted = row.payload.encrypted_credentials
    assert (encrypted is not None) == (mode == 'service_account_key')
    if encrypted is not None:
        assert _google_private_key_pem() not in repr(encrypted)
        assert f'{provider}-{mode}-key-id' not in repr(encrypted)


@pytest.mark.parametrize('provider', ['bigquery', 'gcs'])
@pytest.mark.asyncio
async def test_activity_google_key_revalidation_observes_revocation_without_cached_success(
        db: StateDB, provider: str) -> None:
    poison = 'provider-revocation-response-with-private-material'

    class _MutableValidator:
        revoked = False
        calls = 0

        def __call__(self, config, credentials, exchanger) -> None:
            del config, credentials, exchanger
            self.calls += 1
            if self.revoked:
                try:
                    raise RuntimeError(poison)
                except RuntimeError as provider_error:
                    raise IdentityAuthenticationCategoryError(
                        IdentityAuthenticationCategory.STORED_KEY_INVALID
                    ) from provider_error

    validator = _MutableValidator()
    identities = _new_identity_service(db, identity_validator=validator)
    created = await identities.create(
        _google_identity_args(
            provider,
            'service_account_key',
            name=f'{provider}-activity-revalidation',
        ),
        ADMIN,
    )

    tested = await identities.test_credentials_from_activity(created.id, ADMIN)
    assert tested.id == created.id

    demoted_admin = _Principal(UserID('admin'), 'Admin', (), False)
    with pytest.raises(IdentityForbiddenError,
                       match='Administrator access required'):
        await identities.test_credentials_from_activity(created.id,
                                                        demoted_admin)

    validator.revoked = True
    with pytest.raises(
            IdentityValidationError,
            match='stored Google service-account key is invalid') as caught:
        await identities.test_credentials_from_activity(created.id, ADMIN)

    assert validator.calls == 3
    assert poison not in str(caught.value)
    assert poison not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


@pytest.mark.parametrize('provider', ['bigquery', 'gcs'])
@pytest.mark.asyncio
async def test_same_google_principal_modes_coexist_and_are_ambiguous_for_auto_selection(
        db: StateDB, provider: str) -> None:
    group_id = await _create_group(db, f'{provider}-team')
    project_id = await _create_project(db, group_id, f'{provider}-project')
    member = _Principal(UserID('member'), 'Member', (str(group_id), ))
    identities, _ = _services(db)
    email = 'shared@service-project.iam.gserviceaccount.com'
    created = []
    for mode in ('service_account_key', 'workload_identity'):
        identity = await identities.create(
            _google_identity_args(
                provider,
                mode,
                name=f'{provider}-{mode.replace("_", "-")}',
                email=email,
            ),
            ADMIN,
        )
        await identities.replace_groups(
            identity.id,
            api.ReplaceIdentityGroupsArgs(group_ids=[str(group_id)]),
            ADMIN,
        )
        created.append(identity)

    assert created[0].id != created[1].id
    assert created[0].config.authentication.mode == 'service_account_key'
    assert created[1].config.authentication.mode == 'workload_identity'
    resource_config = (api.BigQueryConnectorResourceConfig(
        project_id='data-project',
        dataset_id='events',
        location='US',
    ) if provider == 'bigquery' else api.GCSConnectorResourceConfig(
        bucket='data-bucket',
        prefix='events',
    ))
    connectors = _new_connector_service(
        db,
        connector_validator=lambda *_: None,
        workload_exchanger=_TestWorkloadExchanger(),
    )

    with pytest.raises(MultipleIdentitiesValidateTargetError):
        await _create_connector(
            connectors,
            project_id,
            api.CreateConnectorArgs(
                name='automatic',
                config=resource_config,
            ),
            member,
        )

    explicit = await _create_connector(
        connectors,
        project_id,
        api.CreateConnectorArgs(
            name='explicit',
            identity_id=created[0].id,
            config=resource_config,
        ),
        member,
    )
    assert explicit.identity_id == created[0].id


@pytest.mark.parametrize('provider', ['bigquery', 'gcs'])
@pytest.mark.asyncio
async def test_google_key_rotation_is_local_and_requires_same_principal(
        db: StateDB, provider: str) -> None:
    validator = MagicMock()
    exchanger = MagicMock()
    identities = _new_identity_service(
        db,
        identity_validator=validator,
        workload_exchanger=exchanger,
    )
    args = _google_identity_args(
        provider,
        'service_account_key',
        name=f'{provider}-rotation',
    )
    created = await identities.create(args, ADMIN)
    validator.reset_mock()
    exchanger.reset_mock()
    credentials = api.GoogleServiceAccountKeyCredentials(
        private_key_id=SecretStr('rotated-key-id'),
        private_key=SecretStr(_google_private_key_pem()),
    )

    rotated = await identities.rotate_credentials(
        created.id,
        api.RotateIdentityCredentialsArgs(
            authentication=args.config.authentication,
            credentials=credentials,
        ),
        ADMIN,
    )

    assert rotated.credential_status.configured is True
    assert rotated.credential_status.last_rotated_at is not None
    validator.assert_not_called()
    exchanger.exchange.assert_not_called()

    mismatched = replace(
        args.config.authentication,
        client_email='other@service-project.iam.gserviceaccount.com',
    )
    with pytest.raises(ValueError, match='configured service account'):
        await identities.rotate_credentials(
            created.id,
            api.RotateIdentityCredentialsArgs(
                authentication=mismatched,
                credentials=credentials,
            ),
            ADMIN,
        )
    with pytest.raises(IdentityValidationError,
                       match='service-account key is invalid'):
        await identities.rotate_credentials(
            created.id,
            api.RotateIdentityCredentialsArgs(
                authentication=args.config.authentication,
                credentials=api.GoogleServiceAccountKeyCredentials(
                    private_key_id=SecretStr('invalid-key-id'),
                    private_key=SecretStr('not a PEM private key'),
                ),
            ),
            ADMIN,
        )
    validator.assert_not_called()
    exchanger.exchange.assert_not_called()


@pytest.mark.parametrize('provider', ['bigquery', 'gcs'])
@pytest.mark.asyncio
async def test_subsequent_resolution_observes_rotated_google_key(
        db: StateDB, provider: str) -> None:
    group_id = await _create_group(db, f'{provider}-rotation-team')
    project_id = await _create_project(db, group_id,
                                       f'{provider}-rotation-project')
    member = _Principal(UserID('member'), 'Member', (str(group_id), ))
    identities, _ = _services(db)
    args = _google_identity_args(
        provider,
        'service_account_key',
        name=f'{provider}-rotating-key',
    )
    identity = await identities.create(args, ADMIN)
    await identities.replace_groups(
        identity.id,
        api.ReplaceIdentityGroupsArgs(group_ids=[str(group_id)]),
        ADMIN,
    )
    config = (api.BigQueryConnectorResourceConfig(
        project_id='data-project', dataset_id='events', location='US')
              if provider == 'bigquery' else api.GCSConnectorResourceConfig(
                  bucket='data-bucket', prefix='events'))
    connectors = _new_connector_service(
        db,
        connector_validator=lambda *_: None,
        workload_exchanger=_TestWorkloadExchanger(),
    )
    connector = await _create_connector(
        connectors,
        project_id,
        api.CreateConnectorArgs(
            identity_id=identity.id,
            name='target',
            config=config,
        ),
        member,
    )

    before = await connectors.resolve_for_activity(project_id, connector.id)
    assert before.credentials.credential.signer.key_id == (
        f'{provider}-rotating-key-key-id')
    await identities.rotate_credentials(
        identity.id,
        api.RotateIdentityCredentialsArgs(
            authentication=args.config.authentication,
            credentials=api.GoogleServiceAccountKeyCredentials(
                private_key_id=SecretStr('rotated-key-id'),
                private_key=SecretStr(_google_private_key_pem()),
            ),
        ),
        ADMIN,
    )

    after = await connectors.resolve_for_activity(project_id, connector.id)
    assert after.credentials.credential.signer.key_id == 'rotated-key-id'


@pytest.mark.parametrize('provider', ['bigquery', 'gcs'])
@pytest.mark.asyncio
async def test_subsequent_workload_resolution_observes_impersonation_failure(
        db: StateDB, provider: str) -> None:
    group_id = await _create_group(db, f'{provider}-workload-team')
    project_id = await _create_project(db, group_id,
                                       f'{provider}-workload-project')
    member = _Principal(UserID('member'), 'Member', (str(group_id), ))
    available = _TestWorkloadExchanger()
    identities = _new_identity_service(
        db,
        identity_validator=lambda *_: None,
        workload_exchanger=available,
    )
    identity = await identities.create(
        _google_identity_args(
            provider,
            'workload_identity',
            name=f'{provider}-workload',
        ),
        ADMIN,
    )
    await identities.replace_groups(
        identity.id,
        api.ReplaceIdentityGroupsArgs(group_ids=[str(group_id)]),
        ADMIN,
    )

    class _MutableExchanger(_TestWorkloadExchanger):
        denied = False

        def exchange(self, config, credentials):
            if self.denied:
                raise IdentityAuthenticationCategoryError(
                    IdentityAuthenticationCategory.IMPERSONATION_DENIED)
            return super().exchange(config, credentials)

    exchanger = _MutableExchanger()
    connectors = _new_connector_service(
        db,
        connector_validator=lambda *_: None,
        workload_exchanger=exchanger,
    )
    config = (api.BigQueryConnectorResourceConfig(
        project_id='data-project', dataset_id='events', location='US')
              if provider == 'bigquery' else api.GCSConnectorResourceConfig(
                  bucket='data-bucket', prefix='events'))
    connector = await _create_connector(
        connectors,
        project_id,
        api.CreateConnectorArgs(
            identity_id=identity.id,
            name='target',
            config=config,
        ),
        member,
    )

    assert (await connectors.resolve_for_activity(
        project_id, connector.id)).credentials.provider == provider
    exchanger.denied = True
    with pytest.raises(IdentityResolutionUnavailableError,
                       match='impersonation was denied') as caught:
        await connectors.resolve_for_activity(project_id, connector.id)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize('provider', ['bigquery', 'gcs'])
@pytest.mark.asyncio
async def test_google_workload_identity_rotation_is_not_applicable(
        db: StateDB, provider: str) -> None:
    identities, _ = _services(db)
    created = await identities.create(
        _google_identity_args(
            provider,
            'workload_identity',
            name=f'{provider}-workload-no-rotation',
        ),
        ADMIN,
    )

    with pytest.raises(ValueError, match='managed by the deployment'):
        await identities.rotate_credentials(
            created.id,
            api.RotateIdentityCredentialsArgs(credentials=None),
            ADMIN,
        )


async def _create_bound_databricks_identity(
    identities: IdentityService,
    group_id: GroupID,
    name: str,
    secret: str,
):
    identity = await identities.create(_identity_args(name, secret), ADMIN)
    await identities.replace_groups(
        identity.id,
        api.ReplaceIdentityGroupsArgs(group_ids=[str(group_id)]),
        ADMIN,
    )
    return identity


async def _wait_for_thread_event(event: threading.Event) -> bool:
    for _ in range(5000):
        if event.is_set():
            return True
        await asyncio.sleep(0.001)
    return False


def _all_connector_cases():
    return [
        (
            api.CreateIdentityArgs(
                display_name='databricks',
                config=api.DatabricksIdentityConfig(
                    host='https://workspace.cloud.databricks.com',
                    client_id='databricks-client',
                ),
                credentials=api.DatabricksOAuthCredentials(
                    client_secret=SecretStr('databricks-secret')),
            ),
            api.DatabricksConnectorResourceConfig(
                warehouse_id='warehouse-id',
                catalog='main',
                schema_name='features',
            ),
        ),
        (
            api.CreateIdentityArgs(
                display_name='snowflake',
                config=api.SnowflakeIdentityConfig(
                    account='organization-account',
                    username='KUMO_READER',
                    authentication=api.SnowflakeAuthenticationKind.PASSWORD,
                ),
                credentials=api.SnowflakePasswordCredentials(
                    password=SecretStr('snowflake-secret')),
            ),
            api.SnowflakeConnectorResourceConfig(
                role='KUMO_READER',
                warehouse='COMPUTE_WH',
                database='PRODUCTION',
                schema_name='ANALYTICS',
            ),
        ),
        (
            api.CreateIdentityArgs(
                display_name='bigquery',
                config=api.BigQueryIdentityConfig(
                    authentication=_google_key_authentication(), ),
                credentials=api.GoogleServiceAccountKeyCredentials(
                    private_key_id=SecretStr('private-key-id'),
                    private_key=SecretStr(_google_private_key_pem()),
                ),
            ),
            api.BigQueryConnectorResourceConfig(
                project_id='data-project',
                dataset_id='events',
                location='US',
                billing_project_id='billing-project',
            ),
        ),
        (
            _google_identity_args(
                'bigquery',
                'workload_identity',
                name='bigquery-workload',
            ),
            api.BigQueryConnectorResourceConfig(
                project_id='data-project',
                dataset_id='events',
                location='US',
                billing_project_id='billing-project',
            ),
        ),
        (
            api.CreateIdentityArgs(
                display_name='s3',
                slug='s3-identity',
                config=api.S3IdentityConfig(
                    authentication=api.S3AssumeRoleAuthentication(
                        partition='aws',
                        role_arn=('arn:aws:iam::123456789012:role/kumo-reader'),
                    ), ),
            ),
            api.S3ConnectorResourceConfig(
                target=api.S3RootedConnectorTarget(
                    bucket='company-data',
                    prefix='raw/events',
                    region='us-west-2',
                ), ),
        ),
        (
            api.CreateIdentityArgs(
                display_name='gcs-key',
                config=api.GCSIdentityConfig(
                    authentication=_google_key_authentication(), ),
                credentials=api.GoogleServiceAccountKeyCredentials(
                    private_key_id=SecretStr('gcs-private-key-id'),
                    private_key=SecretStr(_google_private_key_pem()),
                ),
            ),
            api.GCSConnectorResourceConfig(
                bucket='training-data',
                prefix='features',
                requester_pays_project='billing-project',
            ),
        ),
        (
            api.CreateIdentityArgs(
                display_name='gcs-workload',
                config=api.GCSIdentityConfig(
                    authentication=api.GoogleWorkloadIdentityAuthentication(
                        service_account_email=(
                            'kumo-reader@example.iam.gserviceaccount.com'
                        ), ), ),
            ),
            api.GCSConnectorResourceConfig(
                bucket='training-data',
                prefix='features',
                requester_pays_project='billing-project',
            ),
        ),
        (
            api.CreateIdentityArgs(
                display_name='adls',
                config=api.ADLSIdentityConfig(
                    tenant_id='tenant-id',
                    client_id='application-id',
                    cloud=api.AzureCloud.PUBLIC,
                ),
            ),
            api.ADLSConnectorResourceConfig(
                storage_account='companylake',
                filesystem='curated',
                prefix='orders',
                endpoint_suffix='dfs.core.windows.net',
            ),
        ),
    ]


async def _create_connector(
    connectors: ConnectorService,
    project_id: ProjectID,
    args: api.CreateConnectorArgs,
    principal: _Principal,
):
    authorizations = await connectors.authorize_create(project_id, args,
                                                       principal)
    return await connectors.create_from_activity(project_id, args,
                                                 principal.user_id,
                                                 set(authorizations))


async def _bound_databricks_connector(
    db: StateDB,
    *,
    group_name: str = 'team',
    project_name: str = 'project',
    user_id: str = 'member',
    user_name: str = 'Member',
    identity_name: str = 'identity',
    identity_secret: str = 'secret',
    connector_name: str = 'data',
    schema_name: str = 'schema',
):
    group_id = await _create_group(db, group_name)
    project_id = await _create_project(db, group_id, project_name)
    user = _Principal(UserID(user_id), user_name, (str(group_id), ))
    identities, connectors = _services(db)
    identity = await _create_bound_databricks_identity(identities, group_id,
                                                       identity_name,
                                                       identity_secret)
    connector = await _create_connector(
        connectors,
        project_id,
        _connector_args(connector_name, identity.id, schema_name),
        user,
    )
    return (group_id, project_id, user, identities, connectors, identity,
            connector)


async def _automatic_resolution_project(db: StateDB):
    group_id = await _create_group(db, 'team')
    project_id = await _create_project(db, group_id, 'project')
    user = _Principal(UserID('member'), 'Member', (str(group_id), ))
    identities = _new_identity_service(db, identity_validator=lambda *_: None)
    return group_id, project_id, user, identities


def _enable_ambient_s3(monkeypatch) -> None:
    monkeypatch.setattr('statedb.db.identities._ambient_s3_enabled',
                        lambda: True)
    monkeypatch.setattr('statedb.db.identities._require_ambient_s3_enabled',
                        lambda: None)


def _disable_ambient_s3(monkeypatch) -> None:
    monkeypatch.setattr('statedb.db.identities._ambient_s3_enabled',
                        lambda: False)

    def disabled() -> None:
        raise IdentityResolutionUnavailableError(
            'Ambient S3 workload identity is disabled')

    monkeypatch.setattr('statedb.db.identities._require_ambient_s3_enabled',
                        disabled)


def _enable_ambient_adls(monkeypatch) -> api.ADLSIdentityConfig:
    config = api.ADLSIdentityConfig(
        tenant_id='tenant-id',
        client_id='client-id',
        cloud=api.AzureCloud.PUBLIC,
    )
    monkeypatch.setattr('statedb.db.identities._ambient_adls_config',
                        lambda: config)
    monkeypatch.setattr('statedb.db.identities._ambient_adls_enabled',
                        lambda: True)
    monkeypatch.setattr('statedb.db.identities._require_ambient_adls_enabled',
                        lambda: None)
    return config


def _disable_ambient_adls(monkeypatch) -> None:
    monkeypatch.setattr('statedb.db.identities._ambient_adls_config',
                        lambda: None)
    monkeypatch.setattr('statedb.db.identities._ambient_adls_enabled',
                        lambda: False)

    def disabled() -> None:
        raise IdentityResolutionUnavailableError(
            'Ambient ADLS workload identity is disabled')

    monkeypatch.setattr('statedb.db.identities._require_ambient_adls_enabled',
                        disabled)


def _assume_role_identity_args(name: str) -> api.CreateIdentityArgs:
    return api.CreateIdentityArgs(
        display_name=name,
        config=api.S3IdentityConfig(
            authentication=api.S3AssumeRoleAuthentication(
                partition='aws',
                role_arn=f'arn:aws:iam::123456789012:role/{name}',
            ), ),
    )


@pytest.mark.parametrize('auth_disabled', [True, False])
@pytest.mark.asyncio
async def test_startup_ambient_s3_binding_depends_on_auth(
        db: StateDB, monkeypatch, *, auth_disabled: bool) -> None:
    _enable_ambient_s3(monkeypatch)
    _disable_ambient_adls(monkeypatch)
    group_id = await _create_group(db, 'default-team')
    project_id = await _create_project(db, group_id, 'default-project')
    member = _Principal(UserID('member'), 'Member', (str(group_id), ))
    identities, _ = _services(db)

    await ensure_startup_ambient_identities(
        identities,
        group_id,
        auth_disabled=auth_disabled,
    )

    async with db.read_tx() as tx:
        assert await identities._identities.get(
            tx, str(AMBIENT_S3_IDENTITY_ID)) is not None

    available = await identities.available_for_project(project_id, member)
    assert [identity.id for identity in available
            ] == ([AMBIENT_S3_IDENTITY_ID] if auth_disabled else [])


@pytest.mark.parametrize('auth_disabled', [True, False])
@pytest.mark.asyncio
async def test_startup_ambient_s3_disabled_is_noop(db: StateDB, monkeypatch, *,
                                                   auth_disabled: bool) -> None:
    _disable_ambient_s3(monkeypatch)
    _disable_ambient_adls(monkeypatch)
    group_id = await _create_group(db, 'default-team')
    identities, _ = _services(db)

    await ensure_startup_ambient_identities(
        identities,
        group_id,
        auth_disabled=auth_disabled,
    )

    async with db.read_tx() as tx:
        assert await identities._identities.get(
            tx, str(AMBIENT_S3_IDENTITY_ID)) is None


@pytest.mark.asyncio
async def test_startup_ambient_adls_binds_default_only_when_first_created(
        db: StateDB, monkeypatch) -> None:
    _disable_ambient_s3(monkeypatch)
    config = _enable_ambient_adls(monkeypatch)
    default_group_id = await _create_group(db, 'default-team')
    narrow_group_id = await _create_group(db, 'narrow-team')
    default_project_id = await _create_project(db, default_group_id,
                                               'default-project')
    narrow_project_id = await _create_project(db, narrow_group_id,
                                              'narrow-project')
    default_member = _Principal(UserID('default-member'), 'Default member',
                                (str(default_group_id), ))
    narrow_member = _Principal(UserID('narrow-member'), 'Narrow member',
                               (str(narrow_group_id), ))
    identities, connectors = _services(db)

    await asyncio.gather(*[
        ensure_startup_ambient_identities(
            identities,
            default_group_id,
            auth_disabled=False,
        ) for _ in range(2)
    ])

    available = await identities.available_for_project(default_project_id,
                                                       default_member)
    assert [identity.id for identity in available] == [AMBIENT_ADLS_IDENTITY_ID]
    identity = available[0]
    assert identity.slug == AMBIENT_ADLS_IDENTITY_SLUG
    assert identity.display_name == AMBIENT_ADLS_IDENTITY_DISPLAY_NAME
    assert identity.config == config
    async with db.read_tx() as tx:
        row = await identities._identities.get(tx,
                                               str(AMBIENT_ADLS_IDENTITY_ID))
        assert row is not None
        assert row.payload.encrypted_credentials is None
        assert row.payload.credentials_rotated_at is None
    created = await _create_connector(
        connectors,
        default_project_id,
        api.CreateConnectorArgs(
            name='ambient-adls',
            config=api.ADLSConnectorResourceConfig(
                storage_account='companylake',
                filesystem='curated',
                prefix='orders',
                endpoint_suffix='dfs.core.windows.net',
            ),
        ),
        default_member,
    )
    assert created.identity_id == AMBIENT_ADLS_IDENTITY_ID

    await identities.replace_groups(
        AMBIENT_ADLS_IDENTITY_ID,
        api.ReplaceIdentityGroupsArgs(group_ids=[str(narrow_group_id)]),
        ADMIN,
    )
    await ensure_startup_ambient_identities(
        identities,
        default_group_id,
        auth_disabled=False,
    )

    assert await identities.available_for_project(default_project_id,
                                                  default_member) == []
    assert [
        identity.id for identity in await identities.available_for_project(
            narrow_project_id, narrow_member)
    ] == [AMBIENT_ADLS_IDENTITY_ID]
    with pytest.raises(IdentityForbiddenError, match='immutable'):
        await identities.update_metadata(
            AMBIENT_ADLS_IDENTITY_ID,
            api.UpdateIdentityArgs(display_name='renamed'),
            ADMIN,
        )
    with pytest.raises(IdentityForbiddenError, match='immutable'):
        await identities.delete(AMBIENT_ADLS_IDENTITY_ID, ADMIN)


@pytest.mark.asyncio
async def test_startup_ambient_adls_disabled_is_noop(db: StateDB,
                                                     monkeypatch) -> None:
    _disable_ambient_s3(monkeypatch)
    _disable_ambient_adls(monkeypatch)
    group_id = await _create_group(db, 'default-team')
    identities, _ = _services(db)

    await ensure_startup_ambient_identities(
        identities,
        group_id,
        auth_disabled=False,
    )

    async with db.read_tx() as tx:
        assert await identities._identities.get(
            tx, str(AMBIENT_ADLS_IDENTITY_ID)) is None


@pytest.mark.asyncio
async def test_ambient_s3_identity_is_canonical_unbound_and_immutable(
        db: StateDB, monkeypatch) -> None:
    _enable_ambient_s3(monkeypatch)
    group_id = await _create_group(db, 'ambient-team')
    project_id = await _create_project(db, group_id, 'ambient-project')
    member = _Principal(UserID('member'), 'Member', (str(group_id), ))
    identities, _ = _services(db)

    first, second = await asyncio.gather(
        identities.ensure_ambient_s3_identity(),
        identities.ensure_ambient_s3_identity(),
    )

    assert first is not None and second is not None
    assert first.id == second.id == AMBIENT_S3_IDENTITY_ID
    assert first.slug == second.slug == api.AMBIENT_S3_IDENTITY_SLUG
    assert first.display_name == AMBIENT_S3_IDENTITY_DISPLAY_NAME
    assert await identities.available_for_project(project_id, member) == []
    with pytest.raises(IdentityForbiddenError, match='created and managed'):
        await identities.create(
            api.CreateIdentityArgs(
                display_name='another ambient identity',
                config=api.S3IdentityConfig(
                    authentication=api.S3AmbientAuthentication()),
            ),
            ADMIN,
        )

    await identities.replace_groups(
        AMBIENT_S3_IDENTITY_ID,
        api.ReplaceIdentityGroupsArgs(group_ids=[str(group_id)]),
        ADMIN,
    )
    assert [
        identity.id for identity in await identities.available_for_project(
            project_id, member)
    ] == [AMBIENT_S3_IDENTITY_ID]
    with pytest.raises(IdentityForbiddenError, match='immutable'):
        await identities.update_metadata(
            AMBIENT_S3_IDENTITY_ID,
            api.UpdateIdentityArgs(display_name='renamed'),
            ADMIN,
        )
    with pytest.raises(IdentityForbiddenError, match='immutable'):
        await identities.rotate_credentials(
            AMBIENT_S3_IDENTITY_ID,
            api.RotateIdentityCredentialsArgs(credentials=None),
            ADMIN,
        )
    with pytest.raises(IdentityForbiddenError, match='immutable'):
        await identities.delete(AMBIENT_S3_IDENTITY_ID, ADMIN)


@pytest.mark.parametrize(
    'target',
    [
        api.S3RootedConnectorTarget(bucket='company-data', prefix='root',
                                    region='us-west-2'),
        api.S3AnyURIConnectorTarget(),
    ],
)
@pytest.mark.asyncio
async def test_omitted_s3_identity_selects_only_bound_ambient_identity(
        db: StateDB, monkeypatch, target) -> None:
    _enable_ambient_s3(monkeypatch)
    group_id = await _create_group(db, 'ambient-team')
    project_id = await _create_project(db, group_id, 'ambient-project')
    member = _Principal(UserID('member'), 'Member', (str(group_id), ))
    identities, _ = _services(db)
    await identities.ensure_ambient_s3_identity()
    assume_role = await identities.create(
        _assume_role_identity_args('explicit-role'), ADMIN)
    for identity_id in (AMBIENT_S3_IDENTITY_ID, assume_role.id):
        await identities.replace_groups(
            identity_id,
            api.ReplaceIdentityGroupsArgs(group_ids=[str(group_id)]),
            ADMIN,
        )
    validated = []
    connectors = _new_connector_service(
        db,
        connector_validator=validated.append,
        workload_exchanger=_TestWorkloadExchanger(),
    )
    args = api.CreateConnectorArgs(
        name=f's3-{target.mode}',
        config=api.S3ConnectorResourceConfig(target=target),
    )

    created = await _create_connector(connectors, project_id, args, member)

    assert created.identity_id == AMBIENT_S3_IDENTITY_ID
    assert len(validated) == 1
    assert validated[0].connector_id == created.id

    # An exact retry is returned without another provider validation. A
    # target mismatch under the same normalized name remains a conflict.
    retried = await _create_connector(connectors, project_id, args, member)
    assert retried.id == created.id
    assert len(validated) == 1
    different_target = (api.S3AnyURIConnectorTarget() if target.mode == 'rooted'
                        else api.S3RootedConnectorTarget(bucket='other-bucket'))
    with pytest.raises(ConnectorNameConflictError,
                       match='already exists in this project'):
        await _create_connector(
            connectors,
            project_id,
            api.CreateConnectorArgs(
                name=args.name,
                config=api.S3ConnectorResourceConfig(target=different_target),
            ),
            member,
        )


@pytest.mark.asyncio
async def test_omitted_s3_disabled_or_unbound_is_safe_422_and_persists_nothing(
        db: StateDB, monkeypatch) -> None:
    group_id = await _create_group(db, 'ambient-team')
    project_id = await _create_project(db, group_id, 'ambient-project')
    member = _Principal(UserID('member'), 'Member', (str(group_id), ))
    identities, connectors = _services(db)
    args = api.CreateConnectorArgs(
        name='rootless-s3',
        config=api.S3ConnectorResourceConfig(
            target=api.S3AnyURIConnectorTarget()),
    )

    _disable_ambient_s3(monkeypatch)
    with pytest.raises(NoIdentityValidatesTargetError) as disabled:
        await connectors.authorize_create(project_id, args, member)
    assert str(AMBIENT_S3_IDENTITY_ID) not in str(disabled.value)

    _enable_ambient_s3(monkeypatch)
    await identities.ensure_ambient_s3_identity()
    with pytest.raises(NoIdentityValidatesTargetError) as unbound:
        await connectors.authorize_create(project_id, args, member)
    assert str(AMBIENT_S3_IDENTITY_ID) not in str(unbound.value)

    explicit_missing = api.CreateConnectorArgs(
        name='explicit-missing',
        identity_id='identity_' + 'f' * 32,
        config=args.config,
    )
    with pytest.raises(IdentityDoesNotExistError):
        await connectors.authorize_create(project_id, explicit_missing, member)

    async with db.read_tx() as tx:
        assert await connectors._connectors.list(tx,
                                                 project_id=project_id) == []


@pytest.mark.asyncio
async def test_ambient_connector_disable_unbind_and_rebind_take_effect_immediately(
        db: StateDB, monkeypatch) -> None:
    _enable_ambient_s3(monkeypatch)
    group_id = await _create_group(db, 'ambient-team')
    project_id = await _create_project(db, group_id, 'ambient-project')
    member = _Principal(UserID('member'), 'Member', (str(group_id), ))
    identities, connectors = _services(db)
    await identities.ensure_ambient_s3_identity()
    await identities.replace_groups(
        AMBIENT_S3_IDENTITY_ID,
        api.ReplaceIdentityGroupsArgs(group_ids=[str(group_id)]),
        ADMIN,
    )
    args = api.CreateConnectorArgs(
        name='ambient-root',
        config=api.S3ConnectorResourceConfig(target=api.S3RootedConnectorTarget(
            bucket='company-data')),
    )
    connector = await _create_connector(connectors, project_id, args, member)

    await identities.replace_groups(
        AMBIENT_S3_IDENTITY_ID,
        api.ReplaceIdentityGroupsArgs(group_ids=[]),
        ADMIN,
    )
    assert await connectors.list_for_project(project_id, member) == []
    with pytest.raises(IdentityForbiddenError):
        await connectors.get_for_project(project_id, connector.id, member)
    with pytest.raises(IdentityForbiddenError):
        await connectors.resolve_for_activity(project_id, connector.id)

    await identities.replace_groups(
        AMBIENT_S3_IDENTITY_ID,
        api.ReplaceIdentityGroupsArgs(group_ids=[str(group_id)]),
        ADMIN,
    )
    assert [
        item.id
        for item in await connectors.list_for_project(project_id, member)
    ] == [connector.id]
    assert (await connectors.get_for_project(project_id, connector.id,
                                             member)).id == connector.id

    _disable_ambient_s3(monkeypatch)
    assert await connectors.list_for_project(project_id, member) == []
    assert await identities.available_for_project(project_id, member) == []
    with pytest.raises(IdentityForbiddenError):
        await connectors.get_for_project(project_id, connector.id, member)
    with pytest.raises(IdentityResolutionUnavailableError):
        await connectors.config_for_project(member, project_id, connector.id)
    with pytest.raises(IdentityResolutionUnavailableError):
        await connectors.execution_ref_for_project(member, project_id,
                                                   connector.id,
                                                   CONNECTOR_GRAPH_ACTIONS)
    with pytest.raises(IdentityResolutionUnavailableError):
        await connectors.resolve_for_activity(project_id, connector.id)
    with pytest.raises(NoIdentityValidatesTargetError):
        await connectors.authorize_create(project_id, args, member)


@pytest.mark.asyncio
async def test_identity_validation_never_exposes_provider_error_or_secret(
        db: StateDB) -> None:
    secret = 'provider-returned-client-secret'

    def rejected(*_args) -> None:
        raise RuntimeError(f'invalid assertion contains {secret}')

    service = _new_identity_service(db, identity_validator=rejected)
    with pytest.raises(IdentityValidationError) as caught:
        await service.create(_identity_args('rejected', secret), ADMIN)

    assert str(caught.value) == 'Identity authentication failed'
    assert secret not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    ('category', 'message'),
    [
        (
            IdentityAuthenticationCategory.EXCHANGE_DISABLED,
            'Google workload exchange is disabled for this deployment.',
        ),
        (
            IdentityAuthenticationCategory.SOURCE_UNAVAILABLE,
            'The deployment Google workload credential is unavailable.',
        ),
        (
            IdentityAuthenticationCategory.DISALLOWED_SOURCE,
            'The deployment Google credential source is not an approved '
            'workload credential.',
        ),
        (
            IdentityAuthenticationCategory.IMPERSONATION_DENIED,
            'Google service-account impersonation was denied or unavailable.',
        ),
        (
            IdentityAuthenticationCategory.STORED_KEY_INVALID,
            'The stored Google service-account key is invalid.',
        ),
    ],
)
@pytest.mark.asyncio
async def test_identity_validation_preserves_only_trusted_google_category(
        db: StateDB, category, message) -> None:
    def rejected(*_args) -> None:
        raise IdentityAuthenticationCategoryError(category)

    service = _new_identity_service(db, identity_validator=rejected)

    with pytest.raises(IdentityValidationError) as caught:
        await service.create(
            _google_identity_args(
                'gcs',
                'workload_identity',
                name=f'gcs-{category.value.replace("_", "-")}',
            ),
            ADMIN,
        )

    assert str(caught.value) == message
    assert caught.value.__cause__ is None


@pytest.mark.asyncio
async def test_identity_validation_enforces_provider_deadline(
        db: StateDB, monkeypatch) -> None:
    started = threading.Event()
    release = threading.Event()

    def blocked(*_args) -> None:
        started.set()
        release.wait(timeout=5)

    monkeypatch.setattr(
        'statedb.db.identities.'
        'IDENTITY_AUTHENTICATION_VALIDATION_TIMEOUT_SECONDS',
        0.01,
    )
    service = _new_identity_service(db, identity_validator=blocked)
    try:
        with pytest.raises(IdentityValidationError) as caught:
            await service.create(
                _identity_args('blocked-identity', 'identity-provider-secret'),
                ADMIN,
            )
    finally:
        release.set()

    assert started.is_set()
    assert str(caught.value) == 'Identity authentication failed'
    assert 'identity-provider-secret' not in str(caught.value)


@pytest.mark.asyncio
async def test_connector_validation_never_exposes_provider_error_or_secret(
        db: StateDB, caplog) -> None:
    secret = 'provider-returned-temporary-token'

    def rejected(*_args) -> None:
        raise RuntimeError(f'permission denied with token {secret}')

    group_id = await _create_group(db, 'sales')
    project_id = await _create_project(db, group_id, 'analytics')
    user = _Principal(UserID('sales-user'), 'Sales User', (str(group_id), ))
    identities = _new_identity_service(db, identity_validator=lambda *_: None)
    connectors = _new_connector_service(db, connector_validator=rejected)
    identity = await identities.create(_identity_args('sales', 'secret-1'),
                                       ADMIN)
    await identities.replace_groups(
        identity.id,
        api.ReplaceIdentityGroupsArgs(group_ids=[str(group_id)]),
        ADMIN,
    )

    with pytest.raises(ConnectorValidationError) as caught:
        await _create_connector(
            connectors,
            project_id,
            _connector_args('warehouse', identity.id, 'analytics'),
            user,
        )

    match = re.fullmatch(
        r'Connector target validation failed; reference: ([0-9a-f]{32})',
        str(caught.value),
    )
    assert match is not None
    assert match.group(1) in caplog.text
    assert 'error_type=RuntimeError' in caplog.text
    assert 'diagnostic_code=unclassified' in caplog.text
    assert secret not in str(caught.value)
    assert secret not in caplog.text
    assert caught.value.__cause__ is None


@pytest.mark.asyncio
async def test_explicit_connector_validation_surfaces_only_safe_diagnostic(
        db: StateDB, caplog) -> None:
    code = (
        ConnectorValidationDiagnosticCode.DATABRICKS_SAMPLE_TABLE_PERMISSIONS)

    def rejected(*_args) -> None:
        raise ConnectorValidationDiagnosticError(code)

    group_id = await _create_group(db, 'safe-diagnostic-team')
    project_id = await _create_project(db, group_id, 'safe-diagnostic-project')
    user = _Principal(UserID('member'), 'Member', (str(group_id), ))
    identities = _new_identity_service(db, identity_validator=lambda *_: None)
    identity = await _create_bound_databricks_identity(
        identities,
        group_id,
        'candidate',
        'provider-secret-never-logged',
    )
    connectors = _new_connector_service(db, connector_validator=rejected)

    with pytest.raises(ConnectorValidationError) as caught:
        await _create_connector(
            connectors,
            project_id,
            _connector_args('warehouse', identity.id, 'analytics'),
            user,
        )

    match = re.fullmatch(
        rf'Connector target validation failed \[{code.value}\]; '
        r'reference: ([0-9a-f]{32})',
        str(caught.value),
    )
    assert match is not None
    assert match.group(1) in caplog.text
    assert f'diagnostic_code={code.value}' in caplog.text
    assert 'provider-secret-never-logged' not in caplog.text


@pytest.mark.asyncio
async def test_automatic_resolution_multi_candidate_hides_safe_diagnostics(
        db: StateDB, caplog) -> None:
    # More than one candidate was checked, so the aggregate stays generic: the
    # message must not reveal which bound identity failed or its category.
    code = (
        ConnectorValidationDiagnosticCode.DATABRICKS_SAMPLE_TABLE_PERMISSIONS)

    def rejected(*_args) -> None:
        raise ConnectorValidationDiagnosticError(code)

    group_id = await _create_group(db, 'hidden-diagnostic-team')
    project_id = await _create_project(db, group_id,
                                       'hidden-diagnostic-project')
    user = _Principal(UserID('member'), 'Member', (str(group_id), ))
    identities = _new_identity_service(db, identity_validator=lambda *_: None)
    await _create_bound_databricks_identity(
        identities,
        group_id,
        'private-candidate-a',
        'private-candidate-secret-a',
    )
    await _create_bound_databricks_identity(
        identities,
        group_id,
        'private-candidate-b',
        'private-candidate-secret-b',
    )
    connectors = _new_connector_service(db, connector_validator=rejected)

    with pytest.raises(NoIdentityValidatesTargetError) as caught:
        await _create_connector(
            connectors,
            project_id,
            _connector_args('warehouse', None, 'analytics'),
            user,
        )

    message = str(caught.value)
    assert code.value not in message
    assert 'reference:' not in message
    assert 'private-candidate-a' not in message
    assert 'private-candidate-b' not in message
    assert 'private-candidate-secret-a' not in caplog.text
    assert 'private-candidate-secret-b' not in caplog.text


@pytest.mark.asyncio
async def test_automatic_resolution_single_candidate_surfaces_safe_diagnostic(
        db: StateDB, caplog) -> None:
    # Exactly one candidate was checked and it failed with a classified
    # diagnostic, so identity-less creation surfaces the same safe aggregate
    # the explicit path emits instead of a bare "no identity" message.
    code = (
        ConnectorValidationDiagnosticCode.DATABRICKS_SAMPLE_TABLE_PERMISSIONS)

    def rejected(*_args) -> None:
        raise ConnectorValidationDiagnosticError(code)

    group_id = await _create_group(db, 'single-diagnostic-team')
    project_id = await _create_project(db, group_id,
                                       'single-diagnostic-project')
    user = _Principal(UserID('member'), 'Member', (str(group_id), ))
    identities = _new_identity_service(db, identity_validator=lambda *_: None)
    await _create_bound_databricks_identity(
        identities,
        group_id,
        'private-candidate',
        'private-candidate-secret',
    )
    connectors = _new_connector_service(db, connector_validator=rejected)

    with pytest.raises(NoIdentityValidatesTargetError) as caught:
        await _create_connector(
            connectors,
            project_id,
            _connector_args('warehouse', None, 'analytics'),
            user,
        )

    match = re.search(
        rf'\[{code.value}\]; reference: ([0-9a-f]{{32}})$',
        str(caught.value),
    )
    assert match is not None
    assert match.group(1) in caplog.text
    assert 'private-candidate' not in str(caught.value)
    assert 'private-candidate-secret' not in caplog.text


@pytest.mark.asyncio
async def test_activity_credential_resolution_never_exposes_exchange_secret(
        db: StateDB) -> None:
    secret = 'provider-returned-signed-assertion'

    class _RejectedExchange:
        def exchange(self, *_args):
            raise RuntimeError(f'failed assertion {secret}')

    group_id = await _create_group(db, 'storage')
    project_id = await _create_project(db, group_id, 'lake')
    user = _Principal(UserID('storage-user'), 'Storage User', (str(group_id), ))
    identities = _new_identity_service(
        db,
        identity_validator=lambda *_: None,
        workload_exchanger=_RejectedExchange(),
    )
    connectors = _new_connector_service(
        db,
        connector_validator=lambda *_: None,
        workload_exchanger=_RejectedExchange(),
    )
    identity = await identities.create(
        api.CreateIdentityArgs(
            display_name='storage-role',
            config=api.S3IdentityConfig(
                authentication=api.S3AssumeRoleAuthentication(
                    partition='aws',
                    role_arn='arn:aws:iam::123456789012:role/storage',
                ), ),
            credentials=None,
        ),
        ADMIN,
    )
    await identities.replace_groups(
        identity.id,
        api.ReplaceIdentityGroupsArgs(group_ids=[str(group_id)]),
        ADMIN,
    )
    args = api.CreateConnectorArgs(
        identity_id=identity.id,
        name='storage',
        config=api.S3ConnectorResourceConfig(
            target=api.S3RootedConnectorTarget(
                bucket='company-data',
                prefix='events',
                region='us-west-2',
            ), ),
    )
    authorizations = await connectors.authorize_create(project_id, args, user)

    with pytest.raises(IdentityValidationError) as caught:
        await connectors.create_from_activity(project_id, args, user.user_id,
                                              set(authorizations))

    assert str(caught.value) == 'Identity credential resolution failed'
    assert secret not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.asyncio
async def test_automatic_resolution_with_no_candidates_returns_sanitized_422(
        db: StateDB) -> None:
    group_id = await _create_group(db, 'team')
    project_id = await _create_project(db, group_id, 'project')
    user = _Principal(UserID('member'), 'Member', (str(group_id), ))
    connector_validator = MagicMock()
    connectors = _new_connector_service(db,
                                        connector_validator=connector_validator)
    args = _connector_args('warehouse', None, 'analytics')

    with pytest.raises(NoIdentityValidatesTargetError) as caught:
        await connectors.authorize_create(project_id, args, user)

    assert caught.value.http_code == 422
    assert str(caught.value) == (
        "No databricks identity is assigned to this project's group. "
        'Assign an existing databricks identity to the group under '
        'Admin -> Identity Settings, or create one and add it to the group.')
    connector_validator.assert_not_called()
    assert await connectors.list_for_project(project_id, user) == []


@pytest.mark.asyncio
async def test_automatic_resolution_filters_by_type_and_persists_the_only_success(
        db: StateDB) -> None:
    group_id, project_id, user, identities = await _automatic_resolution_project(
        db)
    selected = await _create_bound_databricks_identity(identities, group_id,
                                                       'selected',
                                                       'selected-secret')
    bigquery_args, _ = _all_connector_cases()[2]
    other_type = await identities.create(bigquery_args, ADMIN)
    await identities.replace_groups(
        other_type.id,
        api.ReplaceIdentityGroupsArgs(group_ids=[str(group_id)]),
        ADMIN,
    )
    validated = []
    connectors = _new_connector_service(db,
                                        connector_validator=validated.append)

    created = await _create_connector(
        connectors,
        project_id,
        _connector_args('warehouse', None, 'analytics'),
        user,
    )

    assert created.identity_id == selected.id
    assert len(validated) == 1
    assert isinstance(validated[0].credentials, DatabricksCredentials)
    assert (validated[0].credentials.client_secret.get_secret_value() ==
            'selected-secret')
    async with db.read_tx() as tx:
        row = await tx.fetchrow('SELECT payload FROM connectors WHERE id = ?',
                                created.id)
    assert row is not None
    payload = json.loads(str(row['payload']))
    assert payload['identity_id'] == selected.id
    assert payload['identity_id'] is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'retry_change',
    ['provider-unavailable', 'new-valid-candidate'],
)
async def test_automatic_resolution_exact_retry_reuses_persisted_selection(
        db: StateDB, retry_change: str) -> None:
    group_id, project_id, user, identities = await _automatic_resolution_project(
        db)
    await _create_bound_databricks_identity(identities, group_id, 'selected',
                                            'selected-secret')
    validator = MagicMock()
    connectors = _new_connector_service(db, connector_validator=validator)
    args = _connector_args('warehouse', None, 'analytics')

    created = await _create_connector(connectors, project_id, args, user)
    if retry_change == 'new-valid-candidate':
        await _create_bound_databricks_identity(identities, group_id,
                                                'new-valid', 'new-valid-secret')
    validator.reset_mock()
    if retry_change == 'provider-unavailable':
        validator.side_effect = RuntimeError('provider is now unavailable')

    authorizations = await connectors.authorize_create(project_id, args, user)
    retried = await connectors.create_from_activity(project_id, args,
                                                    user.user_id,
                                                    set(authorizations))

    assert retried.id == created.id
    assert retried.identity_id == created.identity_id
    validator.assert_not_called()


@pytest.mark.asyncio
async def test_automatic_resolution_retry_denies_unbound_persisted_identity(
        db: StateDB) -> None:
    group_id, project_id, user, identities = await _automatic_resolution_project(
        db)
    identity = await _create_bound_databricks_identity(identities, group_id,
                                                       'selected',
                                                       'selected-secret')
    validator = MagicMock()
    connectors = _new_connector_service(db, connector_validator=validator)
    args = _connector_args('warehouse', None, 'analytics')
    await _create_connector(connectors, project_id, args, user)
    await identities.replace_groups(
        identity.id,
        api.ReplaceIdentityGroupsArgs(group_ids=[]),
        ADMIN,
    )
    validator.reset_mock()

    with pytest.raises(IdentityForbiddenError, match='not mapped'):
        await connectors.create_from_activity(project_id, args, user.user_id,
                                              {identity.id})

    validator.assert_not_called()


@pytest.mark.asyncio
async def test_automatic_resolution_selects_exactly_one_of_several_candidates(
        db: StateDB) -> None:
    group_id, project_id, user, identities = await _automatic_resolution_project(
        db)
    await _create_bound_databricks_identity(identities, group_id, 'denied-one',
                                            'denied-secret-one')
    expected = await _create_bound_databricks_identity(identities, group_id,
                                                       'allowed',
                                                       'allowed-secret')
    await _create_bound_databricks_identity(identities, group_id, 'denied-two',
                                            'denied-secret-two')

    def validate(spec) -> None:
        secret = spec.credentials.client_secret.get_secret_value()
        if secret != 'allowed-secret':
            raise RuntimeError(f'provider denied temporary token {secret}')

    connectors = _new_connector_service(db, connector_validator=validate)

    created = await _create_connector(
        connectors,
        project_id,
        _connector_args('warehouse', None, 'analytics'),
        user,
    )

    assert created.identity_id == expected.id
    assert [
        value.id
        for value in await connectors.list_for_project(project_id, user)
    ] == [created.id]


@pytest.mark.asyncio
async def test_automatic_resolution_multiple_successes_returns_safe_409_only(
        db: StateDB) -> None:
    group_id, project_id, user, identities = await _automatic_resolution_project(
        db)
    first = await _create_bound_databricks_identity(identities, group_id,
                                                    'Production A', 'allowed-a')
    second = await _create_bound_databricks_identity(identities, group_id,
                                                     'Production B',
                                                     'allowed-b')
    denied = await _create_bound_databricks_identity(identities, group_id,
                                                     'Private Failure',
                                                     'failed-provider-secret')

    def validate(spec) -> None:
        secret = spec.credentials.client_secret.get_secret_value()
        if secret == 'failed-provider-secret':
            raise RuntimeError(f'failed identity and provider payload {secret}')

    connectors = _new_connector_service(db, connector_validator=validate)

    with pytest.raises(MultipleIdentitiesValidateTargetError) as caught:
        await _create_connector(
            connectors,
            project_id,
            _connector_args('warehouse', None, 'analytics'),
            user,
        )

    message = str(caught.value)
    assert caught.value.http_code == 409
    assert first.display_name in message and first.id in message
    assert second.display_name in message and second.id in message
    assert denied.display_name not in message and denied.id not in message
    assert 'failed-provider-secret' not in message
    assert await connectors.list_for_project(project_id, user) == []


@pytest.mark.asyncio
async def test_automatic_resolution_caps_candidate_validation_concurrency_at_four(
        db: StateDB) -> None:
    group_id, project_id, user, identities = await _automatic_resolution_project(
        db)
    for index in range(9):
        await _create_bound_databricks_identity(identities, group_id,
                                                f'candidate-{index}',
                                                f'secret-{index}')

    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def validate(_spec) -> None:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            time.sleep(0.03)
        finally:
            with lock:
                active -= 1

    connectors = _new_connector_service(db, connector_validator=validate)

    with pytest.raises(MultipleIdentitiesValidateTargetError):
        await _create_connector(
            connectors,
            project_id,
            _connector_args('warehouse', None, 'analytics'),
            user,
        )

    assert maximum_active == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('concurrent_change', 'error_type', 'message'),
    [
        ('remove-binding', IdentityForbiddenError, 'not mapped'),
        ('add-binding', IdentityResolutionUnavailableError, 'bindings changed'),
        ('archive-project', ProjectNotFoundError, 'not found'),
    ],
)
async def test_automatic_resolution_rechecks_authority_before_insert(
        db: StateDB, concurrent_change, error_type, message) -> None:
    group_id, project_id, user, identities = await _automatic_resolution_project(
        db)
    identity = await _create_bound_databricks_identity(identities, group_id,
                                                       'candidate', 'secret')
    validation_started = threading.Event()
    finish_validation = threading.Event()

    def validate(_spec) -> None:
        validation_started.set()
        if not finish_validation.wait(timeout=5):
            raise RuntimeError('test failed to release validator')

    connectors = _new_connector_service(db, connector_validator=validate)
    args = _connector_args('warehouse', None, 'analytics')
    authorizations = await connectors.authorize_create(project_id, args, user)
    create_task = asyncio.create_task(
        connectors.create_from_activity(project_id, args, user.user_id,
                                        set(authorizations)))
    assert await _wait_for_thread_event(validation_started)
    if concurrent_change == 'remove-binding':
        await identities.replace_groups(
            identity.id,
            api.ReplaceIdentityGroupsArgs(group_ids=[]),
            ADMIN,
        )
    elif concurrent_change == 'add-binding':
        await _create_bound_databricks_identity(identities, group_id,
                                                'new-valid', 'new-valid-secret')
    else:
        projects = JsonRepository('projects', ProjectDBEntry)
        async with db.write_tx() as tx:
            row = await projects.get(tx, str(project_id))
            assert row is not None
            row.payload.status = ProjectStatus.ARCHIVED
            await projects.upsert(tx, str(project_id), row.payload)
    finish_validation.set()

    with pytest.raises(error_type, match=message):
        await create_task
    if concurrent_change == 'archive-project':
        async with db.read_tx() as tx:
            assert await tx.fetch('SELECT payload FROM connectors') == []
    else:
        assert await connectors.list_for_project(project_id, user) == []


@pytest.mark.asyncio
async def test_automatic_resolution_provider_timeout_fails_closed_and_redacts(
        db: StateDB, caplog) -> None:
    group_id, project_id, user, identities = await _automatic_resolution_project(
        db)
    await _create_bound_databricks_identity(identities, group_id, 'candidate',
                                            'provider-timeout-secret')

    def timeout(_spec) -> None:
        raise TimeoutError('provider-timeout-secret signed request timed out')

    connectors = _new_connector_service(db, connector_validator=timeout)

    with pytest.raises(NoIdentityValidatesTargetError) as caught:
        await _create_connector(
            connectors,
            project_id,
            _connector_args('warehouse', None, 'analytics'),
            user,
        )

    assert caught.value.http_code == 422
    assert 'provider-timeout-secret' not in str(caught.value)
    assert 'signed request' not in str(caught.value)
    assert 'provider-timeout-secret' not in caplog.text


@pytest.mark.asyncio
async def test_automatic_resolution_enforces_provider_deadline(
        db: StateDB, monkeypatch) -> None:
    group_id = await _create_group(db, 'deadline-team')
    project_id = await _create_project(db, group_id, 'deadline-project')
    user = _Principal(UserID('member'), 'Member', (str(group_id), ))
    identities = _new_identity_service(db, identity_validator=lambda *_: None)
    await _create_bound_databricks_identity(
        identities,
        group_id,
        'blocked-candidate',
        'blocked-provider-secret',
    )

    started = threading.Event()
    release = threading.Event()

    def blocked(_spec) -> None:
        started.set()
        release.wait(timeout=5)

    monkeypatch.setattr(
        'statedb.db.identities.'
        'CONNECTOR_TARGET_VALIDATION_TIMEOUT_SECONDS',
        0.01,
    )
    connectors = _new_connector_service(db, connector_validator=blocked)

    try:
        with pytest.raises(NoIdentityValidatesTargetError) as caught:
            await _create_connector(
                connectors,
                project_id,
                _connector_args('deadline-warehouse', None, 'analytics'),
                user,
            )
    finally:
        release.set()

    assert started.is_set()
    assert 'blocked-provider-secret' not in str(caught.value)


@pytest.mark.asyncio
async def test_explicit_s3_resolution_redacts_workload_exchange_failures(
        db: StateDB, caplog) -> None:
    provider_secret = 'provider-signed-assertion'

    class _RejectedExchange:
        def exchange(self, *_args):
            raise RuntimeError(f'exchange failed with {provider_secret}')

    group_id = await _create_group(db, 'storage')
    project_id = await _create_project(db, group_id, 'lake')
    user = _Principal(UserID('member'), 'Member', (str(group_id), ))
    identities = _new_identity_service(
        db,
        identity_validator=lambda *_: None,
        workload_exchanger=_RejectedExchange(),
    )
    for index in range(2):
        identity = await identities.create(
            api.CreateIdentityArgs(
                display_name=f'storage-{index}',
                config=api.S3IdentityConfig(
                    authentication=api.S3AssumeRoleAuthentication(
                        partition='aws',
                        role_arn=('arn:aws:iam::123456789012:role/'
                                  f'storage-{index}'),
                    ), ),
                credentials=None,
            ),
            ADMIN,
        )
        await identities.replace_groups(
            identity.id,
            api.ReplaceIdentityGroupsArgs(group_ids=[str(group_id)]),
            ADMIN,
        )
    connectors = _new_connector_service(
        db,
        connector_validator=lambda *_: None,
        workload_exchanger=_RejectedExchange(),
    )
    args = api.CreateConnectorArgs(
        identity_id=identity.id,
        name='storage',
        config=api.S3ConnectorResourceConfig(
            target=api.S3RootedConnectorTarget(
                bucket='company-data',
                prefix='events',
                region='us-west-2',
            ), ),
    )

    with pytest.raises(IdentityValidationError) as caught:
        await _create_connector(connectors, project_id, args, user)

    assert provider_secret not in str(caught.value)
    assert caught.value.__cause__ is None
    assert provider_secret not in caplog.text


@pytest.mark.asyncio
async def test_automatic_resolution_returns_sanitized_503_when_workload_broker_is_unavailable(
        db: StateDB) -> None:
    provider_secret = 'deployment-workload-secret'

    class _UnavailableExchange:
        def exchange(self, *_args):
            raise WorkloadExchangeConfigurationError(
                f'broker configuration includes {provider_secret}')

    group_id = await _create_group(db, 'storage')
    project_id = await _create_project(db, group_id, 'lake')
    user = _Principal(UserID('member'), 'Member', (str(group_id), ))
    identities = _new_identity_service(
        db,
        identity_validator=lambda *_: None,
        workload_exchanger=_UnavailableExchange(),
    )
    identity = await identities.create(
        api.CreateIdentityArgs(
            display_name='storage',
            config=api.S3IdentityConfig(
                authentication=api.S3AssumeRoleAuthentication(
                    partition='aws',
                    role_arn='arn:aws:iam::123456789012:role/storage',
                ), ),
            credentials=None,
        ),
        ADMIN,
    )
    await identities.replace_groups(
        identity.id,
        api.ReplaceIdentityGroupsArgs(group_ids=[str(group_id)]),
        ADMIN,
    )
    connectors = _new_connector_service(
        db,
        connector_validator=lambda *_: None,
        workload_exchanger=_UnavailableExchange(),
    )
    args = api.CreateConnectorArgs(
        identity_id=identity.id,
        name='storage',
        config=api.S3ConnectorResourceConfig(
            target=api.S3RootedConnectorTarget(
                bucket='company-data',
                prefix='events',
                region='us-west-2',
            ), ),
    )

    with pytest.raises(IdentityResolutionUnavailableError) as caught:
        await _create_connector(connectors, project_id, args, user)

    assert caught.value.http_code == 503
    assert str(caught.value) == (
        'Connector identity resolution is temporarily unavailable; please '
        'retry or contact an administrator')
    assert provider_secret not in str(caught.value)


@pytest.mark.parametrize(
    ('identity_args', 'resource_config'),
    _all_connector_cases(),
    # Keep the Snowflake case ID provider-neutral: the repository's opt-in
    # integration-test filter skips any node ID containing "snowflake", even
    # though this is a fully mocked StateDB unit test.
    ids=[
        'databricks',
        'sf',
        'bigquery-key',
        'bigquery-workload',
        's3',
        'gcs-key',
        'gcs-workload',
        'adls',
    ],
)
@pytest.mark.asyncio
async def test_all_external_connector_types_use_the_project_identity_lifecycle(
        db: StateDB, identity_args, resource_config) -> None:
    group_id = await _create_group(db, f'{resource_config.type.value}-team')
    outsider_group_id = await _create_group(
        db, f'{resource_config.type.value}-outsiders')
    project_id = await _create_project(db, group_id,
                                       f'{resource_config.type.value}-project')
    member = _Principal(UserID('member'), 'Member', (str(group_id), ))
    outsider = _Principal(UserID('outsider'), 'Outsider',
                          (str(outsider_group_id), ))
    exchanger = _TestWorkloadExchanger()
    validated = []
    identities = _new_identity_service(
        db,
        identity_validator=lambda *_: None,
        workload_exchanger=exchanger,
    )
    connectors = _new_connector_service(
        db,
        connector_validator=validated.append,
        workload_exchanger=exchanger,
    )
    identity = await identities.create(identity_args, ADMIN)
    await identities.replace_groups(
        identity.id,
        api.ReplaceIdentityGroupsArgs(group_ids=[str(group_id)]),
        ADMIN,
    )
    create_args = api.CreateConnectorArgs(
        identity_id=identity.id,
        name='primary',
        description='Project target',
        config=resource_config,
    )

    created = await _create_connector(connectors, project_id, create_args,
                                      member)

    assert created.project_id == str(project_id)
    assert created.identity_id == identity.id
    assert created.config == resource_config
    assert len(validated) == 1
    assert validated[0].connector_id == created.id
    assert [
        item.id
        for item in await identities.available_for_project(project_id, member)
    ] == [identity.id]
    assert [
        item.id
        for item in await connectors.list_for_project(project_id, member)
    ] == [created.id]
    assert (await connectors.get_for_project(project_id, created.id,
                                             member)).id == created.id
    assert (await connectors.resolve_for_activity(
        project_id, created.id)).connector_id == created.id

    with pytest.raises(ProjectNotFoundError):
        await connectors.get_for_project(project_id, created.id, outsider)
    with pytest.raises(ProjectNotFoundError):
        await _create_connector(
            connectors,
            project_id,
            api.CreateConnectorArgs(
                identity_id=identity.id,
                name='outsider-target',
                config=resource_config,
            ),
            outsider,
        )

    connectors._tables.partition_connector_references = AsyncMock(
        return_value=([_live_table_reference('customers')], [], []))
    with pytest.raises(ConnectorInUseError, match='live table: customers'):
        await connectors.delete(project_id, created.id, member)

    connectors._tables.partition_connector_references = AsyncMock(
        return_value=([], [], []))
    deleted = await connectors.delete(project_id, created.id, member)
    assert deleted.id == created.id
    with pytest.raises(ConnectorDoesNotExistError):
        await connectors.get_for_project(project_id, created.id, member)


@pytest.mark.asyncio
async def test_identity_create_atomically_binds_multiple_groups(
        db: StateDB) -> None:
    group_a = await _create_group(db, 'group-a')
    group_b = await _create_group(db, 'group-b')
    identities, _ = _services(db)
    args = _identity_args('shared-identity', 'provider-secret')
    args.group_ids = [str(group_a), str(group_b)]

    identity = await identities.create(args, ADMIN)

    with pytest.raises(IdentityConflictError):
        await identities.create(args, ADMIN)

    bound_group_ids = (await identities.list_groups(identity.id,
                                                    ADMIN)).group_ids
    assert set(bound_group_ids) == {str(group_a), str(group_b)}
    async with db.read_tx() as tx:
        rows = await identities._identities.list_all(tx)
        bindings = await identities._bindings.list_by_field(
            tx, 'identity_id', str(identity.id))
    assert [row.payload.id for row in rows] == [identity.id]
    binding_group_ids = {str(row.payload.group_id) for row in bindings}
    assert binding_group_ids == {str(group_a), str(group_b)}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('failure_mode', 'error_type', 'message'),
    [
        ('inactive-groups', ValueError, 'Unknown or inactive groups'),
        ('binding-write', RuntimeError, 'binding write failure'),
        ('non-admin', IdentityForbiddenError, 'Administrator access required'),
    ],
)
async def test_identity_create_with_groups_rolls_back_on_failure(
        db: StateDB, failure_mode, error_type, message) -> None:
    group_id = await _create_group(db, 'group')
    identities, _ = _services(db)
    args = _identity_args('must-rollback', 'provider-secret')
    args.group_ids = [str(group_id)]
    principal = ADMIN
    if failure_mode == 'inactive-groups':
        groups = JsonRepository('groups', KumoGroupDBEntry)
        async with db.write_tx() as tx:
            row = await groups.get(tx, str(group_id))
            assert row is not None
            row.payload.status = GroupStatus.DELETED
            await groups.upsert(tx, row.id, row.payload)
        args.group_ids.append(str(GroupID.generate()))
    elif failure_mode == 'binding-write':
        identities._bindings.upsert = AsyncMock(
            side_effect=RuntimeError('simulated binding write failure'))
    else:
        principal = _Principal(UserID('member'), 'Member', (str(group_id), ))

    with pytest.raises(error_type, match=message):
        await identities.create(args, principal)

    async with db.read_tx() as tx:
        assert await identities._identities.list_all(tx) == []
        assert await identities._bindings.list_all(tx) == []
        assert await tx.fetch('SELECT payload FROM identities') == []


@pytest.mark.asyncio
async def test_concurrent_identity_creates_cannot_bypass_normalized_name_or_principal_uniqueness(
        db: StateDB) -> None:
    identities, _ = _services(db)
    same_normalized_name = (
        _identity_args('Production', 'secret-1'),
        api.CreateIdentityArgs(
            display_name='  production  ',
            config=api.DatabricksIdentityConfig(
                host='https://other.cloud.databricks.com',
                client_id='other-client',
            ),
            credentials=api.DatabricksOAuthCredentials(
                client_secret=SecretStr('secret-2')),
        ),
    )

    results = await asyncio.gather(
        *(identities.create(args, ADMIN) for args in same_normalized_name),
        return_exceptions=True,
    )

    first = only_instance(results, api.IdentityResponse)
    only_instance(results, IdentityConflictError)
    same_principal = api.CreateIdentityArgs(
        display_name='different display name',
        config=first.config,
        credentials=api.DatabricksOAuthCredentials(
            client_secret=SecretStr('replacement-secret')),
    )
    with pytest.raises(IdentityConflictError, match='external principal'):
        await identities.create(same_principal, ADMIN)


@pytest.mark.asyncio
async def test_concurrent_identity_renames_translate_database_name_conflict(
        db: StateDB) -> None:
    identities, _ = _services(db)
    first = await identities.create(_identity_args('first', 'secret-1'), ADMIN)
    second = await identities.create(_identity_args('second', 'secret-2'),
                                     ADMIN)

    # Model two transactions that both observed the name as available before
    # the database unique index selected the winner. The precheck is only an
    # optimization; the authoritative index violation must still become the
    # same sanitized 409-domain error.
    identities._identities.exists_by_fields = AsyncMock(return_value=False)
    results = await asyncio.gather(
        identities.update_metadata(
            first.id,
            api.UpdateIdentityArgs(display_name='  Production  '),
            ADMIN,
        ),
        identities.update_metadata(
            second.id,
            api.UpdateIdentityArgs(display_name='production'),
            ADMIN,
        ),
        return_exceptions=True,
    )

    only_instance(results, api.IdentityResponse)
    conflict = only_instance(results, IdentityConflictError)
    assert str(conflict) == 'An identity with this display name already exists'


@pytest.mark.asyncio
async def test_concurrent_connector_names_are_unique_per_project_but_reusable_across_projects(
        db: StateDB) -> None:
    group_id = await _create_group(db, 'team')
    project_a = await _create_project(db, group_id, 'a')
    project_b = await _create_project(db, group_id, 'b')
    member = _Principal(UserID('member'), 'Member', (str(group_id), ))
    identities, connectors = _services(db)
    identity = await identities.create(_identity_args('identity', 'secret'),
                                       ADMIN)
    await identities.replace_groups(
        identity.id,
        api.ReplaceIdentityGroupsArgs(group_ids=[str(group_id)]),
        ADMIN,
    )
    first_args = _connector_args('Production Data', identity.id, 'one')
    second_args = _connector_args('  production data  ', identity.id, 'two')
    authorizations = await asyncio.gather(
        connectors.authorize_create(project_a, first_args, member),
        connectors.authorize_create(project_a, second_args, member),
    )

    results = await asyncio.gather(
        connectors.create_from_activity(project_a, first_args, member.user_id,
                                        set(authorizations[0])),
        connectors.create_from_activity(project_a, second_args, member.user_id,
                                        set(authorizations[1])),
        return_exceptions=True,
    )

    only_instance(results, api.ConnectorResponse)
    only_instance(results, ConnectorNameConflictError)
    other_project = await _create_connector(
        connectors,
        project_b,
        _connector_args('production data', identity.id, 'three'),
        member,
    )
    assert other_project.project_id == str(project_b)


@pytest.mark.asyncio
async def test_concurrent_identical_connector_create_is_idempotent_and_mismatches_conflict(
        db: StateDB) -> None:
    group_id = await _create_group(db, 'idempotent-team')
    project_id = await _create_project(db, group_id, 'idempotent-project')
    member = _Principal(UserID('member'), 'Member', (str(group_id), ))
    identities, _ = _services(db)
    identity = await _create_bound_databricks_identity(identities, group_id,
                                                       'primary',
                                                       'primary-secret')
    other_identity = await _create_bound_databricks_identity(
        identities, group_id, 'secondary', 'secondary-secret')

    validation_barrier = threading.Barrier(2)

    def validate(_resolved) -> None:
        validation_barrier.wait(timeout=10)

    connectors = _new_connector_service(db, connector_validator=validate)
    args = _connector_args('Production Data', identity.id, 'analytics')
    authorizations = await connectors.authorize_create(project_id, args, member)

    results = await asyncio.gather(
        connectors.create_from_activity(project_id, args, member.user_id,
                                        set(authorizations)),
        connectors.create_from_activity(project_id, args, member.user_id,
                                        set(authorizations)),
    )

    assert results[0].id == results[1].id
    async with db.read_tx() as tx:
        rows = await connectors._connectors.list(tx, project_id=project_id)
    assert [row.id for row in rows] == [results[0].id]

    different_target = _connector_args('Production Data', identity.id,
                                       'different-schema')
    different_identity = _connector_args('Production Data', other_identity.id,
                                         'analytics')
    conflict_authorizations = await asyncio.gather(
        connectors.authorize_create(project_id, different_target, member),
        connectors.authorize_create(project_id, different_identity, member),
    )
    conflicts = await asyncio.gather(
        connectors.create_from_activity(project_id, different_target,
                                        member.user_id,
                                        set(conflict_authorizations[0])),
        connectors.create_from_activity(project_id, different_identity,
                                        member.user_id,
                                        set(conflict_authorizations[1])),
        return_exceptions=True,
    )

    assert all(
        isinstance(error, ConnectorNameConflictError) for error in conflicts)
    assert all(
        "Connector named 'Production Data' already exists in this project"
        == str(error) for error in conflicts)
    async with db.read_tx() as tx:
        rows = await connectors._connectors.list(tx, project_id=project_id)
    assert [row.id for row in rows] == [results[0].id]


@pytest.mark.asyncio
async def test_one_identity_backs_multiple_project_connectors_without_secret_duplication(
        db: StateDB) -> None:
    (_, project_id, user, identities, connectors, identity,
     first) = await _bound_databricks_connector(
         db,
         group_name='sales',
         project_name='analytics',
         user_id='sales-user',
         user_name='Sales User',
         identity_name='sales',
         identity_secret='secret-1',
         connector_name='sales-eu',
         schema_name='eu',
     )
    assert [
        available.id for available in await identities.available_for_project(
            project_id, user)
    ] == [identity.id]
    second = await _create_connector(
        connectors, project_id, _connector_args('sales-us', identity.id, 'us'),
        user)

    assert first.identity_id == identity.id
    assert second.identity_id == identity.id
    assert first.project_id == str(project_id)
    assert {
        item.name
        for item in await connectors.list_for_project(project_id, user)
    } == {'sales-eu', 'sales-us'}

    resolved = await connectors.resolve_for_activity(project_id, first.id)
    assert isinstance(resolved.credentials, DatabricksCredentials)
    assert resolved.credentials.client_secret.get_secret_value() == 'secret-1'

    async with db.read_tx() as tx:
        identity_rows = await tx.fetch('SELECT payload FROM identities')
        connector_rows = await tx.fetch('SELECT payload FROM connectors')
    assert len(identity_rows) == 1
    assert len(connector_rows) == 2
    assert 'secret-1' not in str(identity_rows[0]['payload'])
    for row in connector_rows:
        payload = json.loads(str(row['payload']))
        assert 'encrypted_credentials' not in payload
        assert 'group_id' not in payload


@pytest.mark.asyncio
async def test_identity_group_replacement_deduplicates_group_ids(
        db: StateDB) -> None:
    group_id = await _create_group(db, 'team')
    identities, _ = _services(db)
    identity = await identities.create(_identity_args('identity', 'secret'),
                                       ADMIN)

    response = await identities.replace_groups(
        identity.id,
        SimpleNamespace(group_ids=[str(group_id),
                                   str(group_id),
                                   str(group_id)]),
        ADMIN,
    )

    assert response.group_ids == [str(group_id)]
    assert (await identities.list_groups(identity.id,
                                         ADMIN)).group_ids == [str(group_id)]


@pytest.mark.asyncio
async def test_connector_is_exclusively_owned_by_exact_project(
        db: StateDB) -> None:
    (group_id, _, user, _, connectors, _,
     connector) = await _bound_databricks_connector(db, group_name='sales',
                                                    project_name='a',
                                                    connector_name='warehouse')
    project_b = await _create_project(db, group_id, 'b')

    with pytest.raises(ConnectorDoesNotExistError):
        await connectors.get_for_project(project_b, connector.id, user)
    with pytest.raises(ConnectorDoesNotExistError):
        await connectors.resolve_for_activity(project_b, connector.id)


@pytest.mark.asyncio
async def test_shared_members_mutate_and_private_project_is_owner_only(
        db: StateDB) -> None:
    group_id = await _create_group(db, 'team')
    shared = await _create_project(db, group_id, 'shared', owner='owner')
    private = await _create_project(db, group_id, 'private', owner='owner',
                                    private=True)
    member = _Principal(UserID('member'), 'Member', (str(group_id), ))
    owner = _Principal(UserID('owner'), 'Owner', (str(group_id), ))
    identities, connectors = _services(db)
    identity = await identities.create(_identity_args('identity', 'secret'),
                                       ADMIN)
    await identities.replace_groups(
        identity.id, api.ReplaceIdentityGroupsArgs(group_ids=[str(group_id)]),
        ADMIN)

    created = await _create_connector(
        connectors, shared, _connector_args('shared-data', identity.id,
                                            'schema'), member)
    updated = await connectors.update_metadata(
        shared, created.id,
        api.UpdateConnectorMetadataArgs(description='member update'), member)
    assert updated.description == 'member update'

    with pytest.raises(ProjectNotFoundError):
        await _create_connector(
            connectors, private, _connector_args('denied', identity.id,
                                                 'schema'), member)
    private_connector = await _create_connector(
        connectors, private, _connector_args('owner-data', identity.id,
                                             'schema'), owner)
    assert private_connector.project_id == str(private)

    removed_owner = _Principal(UserID('owner'), 'Owner', ())
    with pytest.raises(ProjectNotFoundError):
        await connectors.get_for_project(private, private_connector.id,
                                         removed_owner)
    unavailable_owner = _Principal(UserID('owner'), 'Owner', (),
                                   external_membership_unavailable=True)
    with pytest.raises(ExternalMembershipUnavailableError):
        await connectors.get_for_project(private, private_connector.id,
                                         unavailable_owner)
    assert (await connectors.get_for_project(private, private_connector.id,
                                             ADMIN)).id == private_connector.id


@pytest.mark.asyncio
async def test_activity_execution_is_project_owned_not_submitter_owned(
        db: StateDB) -> None:
    (_, project_id, submitter, _, connectors, identity,
     connector) = await _bound_databricks_connector(db, user_id='submitter',
                                                    user_name='Submitter')
    reference = await connectors.execution_ref_for_project(
        submitter, project_id, connector.id, CONNECTOR_GRAPH_ACTIONS)
    accepted_args = _connector_args('accepted-before-membership-removal',
                                    identity.id, 'accepted_schema')
    accepted_authorizations = await connectors.authorize_create(
        project_id, accepted_args, submitter)

    removed_submitter = _Principal(UserID('submitter'), 'Submitter', ())
    with pytest.raises(ProjectNotFoundError):
        await connectors.execution_ref_for_project(removed_submitter,
                                                   project_id, connector.id,
                                                   CONNECTOR_GRAPH_ACTIONS)
    with pytest.raises(ProjectNotFoundError):
        await connectors.authorize_create(
            project_id,
            _connector_args('new-after-removal', identity.id, 'new_schema'),
            removed_submitter,
        )
    accepted = await connectors.create_from_activity(
        project_id, accepted_args, submitter.user_id,
        set(accepted_authorizations))
    assert accepted.project_id == str(project_id)
    assert (await connectors.resolve_for_activity(
        reference.project_id,
        reference.connector_id)).connector_id == connector.id
    await connectors.require_delegatable_for_project(reference.project_id,
                                                     reference.connector_id)

    admin_args = _connector_args('accepted-before-admin-demotion', identity.id,
                                 'admin_schema')
    admin_authorizations = await connectors.authorize_create(
        project_id, admin_args, ADMIN)
    demoted_admin = _Principal(UserID('admin'), 'Admin', ())
    with pytest.raises(ProjectNotFoundError):
        await connectors.authorize_create(
            project_id,
            _connector_args('new-after-demotion', identity.id, 'denied_schema'),
            demoted_admin,
        )
    accepted_admin = await connectors.create_from_activity(
        project_id, admin_args, ADMIN.user_id, set(admin_authorizations))
    assert accepted_admin.project_id == str(project_id)


@pytest.mark.asyncio
async def test_fresh_workflow_references_preserve_scope_and_replay_protection(
        db: StateDB) -> None:
    (group_id, project_id, user, identities, connectors, identity,
     connector) = await _bound_databricks_connector(db)
    other_project_id = await _create_project(db, group_id, 'other-project')
    version_reference = await connectors.execution_ref_for_project(
        user, project_id, connector.id, CONNECTOR_GRAPH_ACTIONS)
    refresh_reference, = await connectors.fresh_execution_refs_for_workflow(
        user,
        (version_reference, ),
        CONNECTOR_GRAPH_ACTIONS,
    )

    assert refresh_reference.project_id == version_reference.project_id
    assert refresh_reference.connector_id == version_reference.connector_id
    assert (refresh_reference.authorization_token
            != version_reference.authorization_token)

    authorizations = connectors._connector_authorizations
    async with db.read_tx() as tx:
        version_authorization = await authorizations._get_by_token_in_tx(
            tx, version_reference.authorization_token)
        refresh_authorization = await authorizations._get_by_token_in_tx(
            tx, refresh_reference.authorization_token)
    expected_actions = tuple(
        sorted(CONNECTOR_GRAPH_ACTIONS, key=lambda action: action.value))
    assert version_authorization.allowed_actions == expected_actions
    assert refresh_authorization.allowed_actions == expected_actions
    assert version_authorization.workflow_id == ''
    assert refresh_authorization.workflow_id == ''

    await authorizations.bind_workflow(
        version_reference.authorization_token,
        deployment_id=version_authorization.deployment_id,
        temporal_namespace=version_authorization.temporal_namespace,
        workflow_id='version-workflow',
        expires_at=now_utc() + timedelta(hours=1),
    )
    await authorizations.bind_workflow(
        refresh_reference.authorization_token,
        deployment_id=refresh_authorization.deployment_id,
        temporal_namespace=refresh_authorization.temporal_namespace,
        workflow_id='refresh-workflow',
        expires_at=now_utc() + timedelta(hours=1),
    )
    with pytest.raises(ConnectorAuthorizationError, match='workflow mismatch'):
        await authorizations.bind_workflow(
            version_reference.authorization_token,
            deployment_id=version_authorization.deployment_id,
            temporal_namespace=version_authorization.temporal_namespace,
            workflow_id='refresh-workflow',
            expires_at=now_utc() + timedelta(hours=1),
        )

    with pytest.raises(ConnectorDoesNotExistError):
        await connectors.fresh_execution_refs_for_workflow(
            user,
            (replace(version_reference, project_id=other_project_id), ),
            CONNECTOR_GRAPH_ACTIONS,
        )

    await identities.replace_groups(
        identity.id,
        api.ReplaceIdentityGroupsArgs(group_ids=[]),
        ADMIN,
    )
    with pytest.raises(IdentityForbiddenError, match='not mapped'):
        await connectors.fresh_execution_refs_for_workflow(
            user,
            (refresh_reference, ),
            CONNECTOR_GRAPH_ACTIONS,
        )


@pytest.mark.asyncio
async def test_binding_removal_immediately_hides_connector_discovery_and_rebind_restores_it(
        db: StateDB) -> None:
    group_a = await _create_group(db, 'team-a')
    group_b = await _create_group(db, 'team-b')
    project_a = await _create_project(db, group_a, 'project-a')
    project_b = await _create_project(db, group_b, 'project-b')
    user = _Principal(UserID('member'), 'Member', (str(group_a), str(group_b)))
    identities, connectors = _services(db)
    identity = await identities.create(_identity_args('identity', 'secret'),
                                       ADMIN)
    await identities.replace_groups(
        identity.id,
        api.ReplaceIdentityGroupsArgs(
            group_ids=[str(group_a), str(group_b)]),
        ADMIN,
    )
    connector_a = await _create_connector(
        connectors,
        project_a,
        _connector_args('data-a', identity.id, 'schema-a'),
        user,
    )
    connector_b = await _create_connector(
        connectors,
        project_b,
        _connector_args('data-b', identity.id, 'schema-b'),
        user,
    )

    assert [
        connector.id
        for connector in await connectors.list_for_project(project_a, user)
    ] == [connector_a.id]
    assert (await connectors.get_for_project(project_a, connector_a.id,
                                             ADMIN)).id == connector_a.id
    assert (await connectors.source_table_connector_for_project(
        project_a, connector_a.id, user)).id == connector_a.id
    assert (await connectors.execution_ref_for_project(
        user, project_a, connector_a.id,
        CONNECTOR_GRAPH_ACTIONS)).connector_id == connector_a.id

    await identities.replace_groups(
        identity.id,
        api.ReplaceIdentityGroupsArgs(group_ids=[str(group_b)]),
        ADMIN,
    )

    assert await connectors.list_for_project(project_a, user) == []
    assert await connectors.list_for_project(project_a, ADMIN) == []
    for principal in (user, ADMIN):
        with pytest.raises(IdentityForbiddenError, match='not mapped'):
            await connectors.get_for_project(project_a, connector_a.id,
                                             principal)
        with pytest.raises(IdentityForbiddenError, match='not mapped'):
            await connectors.update_metadata(
                project_a,
                connector_a.id,
                api.UpdateConnectorMetadataArgs(description='must not change'),
                principal,
            )
    with pytest.raises(IdentityForbiddenError, match='not mapped'):
        await connectors.delete(project_a, connector_a.id, user)
    with pytest.raises(IdentityForbiddenError, match='not mapped'):
        await connectors.source_table_connector_for_project(
            project_a, connector_a.id, user)
    # Refresh and source-table discovery both authorize their activity
    # reference through this method before queuing external access.
    with pytest.raises(IdentityForbiddenError, match='not mapped'):
        await connectors.execution_ref_for_project(user, project_a,
                                                   connector_a.id,
                                                   CONNECTOR_GRAPH_ACTIONS)
    with pytest.raises(IdentityForbiddenError, match='not mapped'):
        await connectors.config_for_project(user, project_a, connector_a.id)
    with pytest.raises(IdentityForbiddenError, match='not mapped'):
        await connectors.resolve_for_activity(project_a, connector_a.id)
    with pytest.raises(IdentityForbiddenError, match='not mapped'):
        await connectors.require_delegatable_for_project(
            project_a, connector_a.id)

    # Removing one group binding neither exposes a connector cross-project nor
    # disrupts the same reusable identity where it remains bound.
    assert [
        connector.id
        for connector in await connectors.list_for_project(project_b, user)
    ] == [connector_b.id]
    assert (await connectors.get_for_project(project_b, connector_b.id,
                                             ADMIN)).id == connector_b.id
    assert (await connectors.source_table_connector_for_project(
        project_b, connector_b.id, user)).id == connector_b.id
    with pytest.raises(ConnectorDoesNotExistError):
        await connectors.get_for_project(project_b, connector_a.id, user)

    await identities.replace_groups(
        identity.id,
        api.ReplaceIdentityGroupsArgs(
            group_ids=[str(group_a), str(group_b)]),
        ADMIN,
    )

    assert [
        connector.id
        for connector in await connectors.list_for_project(project_a, user)
    ] == [connector_a.id]
    assert (await connectors.get_for_project(project_a, connector_a.id,
                                             ADMIN)).id == connector_a.id
    assert (await connectors.source_table_connector_for_project(
        project_a, connector_a.id, user)).id == connector_a.id
    assert (await connectors.resolve_for_activity(
        project_a, connector_a.id)).connector_id == connector_a.id
    updated = await connectors.update_metadata(
        project_a,
        connector_a.id,
        api.UpdateConnectorMetadataArgs(description='binding restored'),
        user,
    )
    assert updated.description == 'binding restored'
    deleted = await connectors.delete(project_a, connector_a.id, ADMIN)
    assert deleted.id == connector_a.id


@pytest.mark.asyncio
async def test_tombstoned_connector_retains_safe_history_and_allows_identity_delete(
        db: StateDB, monkeypatch) -> None:
    (_, project_id, user, identities, connectors, identity,
     connector) = await _bound_databricks_connector(db)

    await connectors.delete(project_id, connector.id, user)
    with pytest.raises(ConnectorDoesNotExistError):
        await connectors.resolve_for_activity(project_id, connector.id)

    async def fail_full_connector_scan(*_args, **_kwargs):
        raise AssertionError('identity deletion must use the indexed lookup')

    monkeypatch.setattr(StateDBConnectorRepository, 'list',
                        fail_full_connector_scan)
    deleted_identity = await identities.delete(identity.id, ADMIN)
    assert deleted_identity.id == identity.id

    async with db.read_tx() as tx:
        row = await tx.fetchrow('SELECT payload FROM connectors WHERE id = ?',
                                connector.id)
        identity_row = await tx.fetchrow(
            'SELECT payload FROM identities WHERE id = ?', identity.id)
        bindings = await identities._bindings.list_by_field(
            tx, 'identity_id', str(identity.id))
    assert row is not None
    payload = json.loads(str(row['payload']))
    assert payload['tombstoned_at'] is not None
    assert payload['identity_id'] == str(identity.id)
    assert 'credentials' not in payload
    assert 'secret' not in str(row['payload'])
    assert identity_row is not None
    identity_payload = json.loads(str(identity_row['payload']))
    assert identity_payload['slug'] == identity.slug
    assert identity_payload['tombstoned_at'] is not None
    assert identity_payload['encrypted_credentials'] is None
    assert bindings == []


@pytest.mark.asyncio
async def test_deleted_identity_can_be_recreated(db: StateDB) -> None:
    identities, _ = _services(db)
    group_id = await _create_group(db, 'Identity group')
    args = _identity_args('Recreated identity', 'original-secret')
    args.group_ids = [str(group_id)]
    first = await identities.create(args, ADMIN)
    await identities.delete(first.id, ADMIN)
    assert await identities.list_admin(ADMIN) == []
    with pytest.raises(IdentityDoesNotExistError):
        await identities.get_admin(first.id, ADMIN)

    args.credentials = api.DatabricksOAuthCredentials(
        client_secret=SecretStr('replacement-secret'))
    second = await identities.create(args, ADMIN)
    assert second.id != first.id
    assert second.slug == first.slug
    assert second.config == first.config
    assert [row.id for row in await identities.list_admin(ADMIN)] == [second.id]
    with pytest.raises(IdentityConflictError):
        await identities.create(args, ADMIN)
    with pytest.raises(IdentityDoesNotExistError):
        await identities.get_admin(first.id, ADMIN)
    async with db.read_tx() as tx:
        old = await identities._identities.get(tx, str(first.id))
        new = await identities._identities.get(tx, str(second.id))
        assert old.payload.tombstoned_at is not None
        assert old.payload.encrypted_credentials is None
        assert new.payload.tombstoned_at is None
        assert new.payload.encrypted_credentials is not None
        assert await identities._bindings.list_by_field(
            tx, 'identity_id', str(first.id)) == []
        bindings = await identities._bindings.list_by_field(
            tx, 'identity_id', str(second.id))
        assert [row.payload.group_id for row in bindings] == [group_id]


@pytest.mark.asyncio
async def test_tombstoned_connector_name_can_be_recreated(db: StateDB) -> None:
    group_id = await _create_group(db, 'team')
    project_id = await _create_project(db, group_id, 'project')
    user = _Principal(UserID('member'), 'Member', (str(group_id), ))
    identities, connectors = _services(db)
    identity = await _create_bound_databricks_identity(identities, group_id,
                                                       'identity', 'secret')
    args = _connector_args('Data Warehouse', identity.id, 'schema')

    first = await _create_connector(connectors, project_id, args, user)
    await connectors.delete(project_id, first.id, user)
    second = await _create_connector(
        connectors,
        project_id,
        replace(args, name='  DATA WAREHOUSE  '),
        user,
    )

    assert second.id != first.id
    assert [
        connector.id
        for connector in await connectors.list_for_project(project_id, user)
    ] == [second.id]
    with pytest.raises(ConnectorDoesNotExistError):
        await connectors.get_for_project(project_id, first.id, user)
    async with db.read_tx() as tx:
        rows = await tx.fetch(
            "SELECT payload FROM connectors ORDER BY created_at, id")
    payloads = [json.loads(str(row['payload'])) for row in rows]
    assert [payload['id'] for payload in payloads] == [first.id, second.id]
    assert payloads[0]['tombstoned_at'] is not None
    assert payloads[1]['tombstoned_at'] is None


@pytest.mark.asyncio
async def test_archived_project_connector_does_not_block_identity_delete(
        db: StateDB) -> None:
    (_, project_id, _, identities, connectors, identity,
     connector) = await _bound_databricks_connector(db)

    archived = await ProjectService(db).archive_project(project_id, ADMIN)
    deleted = await identities.delete(identity.id, ADMIN)

    assert archived.status == ProjectStatus.ARCHIVED
    assert deleted.id == identity.id
    async with db.read_tx() as tx:
        retained = await connectors._connectors.get(tx, connector.id)
        assert retained is not None
        assert retained.tombstoned_at is None
        tombstone = await identities._identities.get(tx, str(identity.id))
        assert tombstone is not None
        assert tombstone.payload.slug == identity.slug
        assert tombstone.payload.tombstoned_at is not None
        assert tombstone.payload.encrypted_credentials is None


@pytest.mark.asyncio
async def test_list_connectors_using_identity_reports_active_project_blocker(
        db: StateDB) -> None:
    (_, project_id, _, identities, _, identity,
     connector) = await _bound_databricks_connector(
         db, project_name='Sales', connector_name='Data Warehouse')

    usage = await identities.list_connectors_using_identity(identity.id, ADMIN)

    assert usage.identity_id == str(identity.id)
    assert len(usage.connectors) == 1
    entry = usage.connectors[0]
    assert entry.connector_id == str(connector.id)
    assert entry.connector_name == 'Data Warehouse'
    assert entry.project_id == str(project_id)
    assert entry.project_display_name == 'Sales'


@pytest.mark.asyncio
async def test_list_connectors_using_identity_reports_missing_project(
        db: StateDB) -> None:
    (_, project_id, _, identities, _, identity,
     connector) = await _bound_databricks_connector(db)

    async with db.write_tx() as tx:
        await identities._projects.delete(tx, str(project_id))

    usage = await identities.list_connectors_using_identity(identity.id, ADMIN)

    assert len(usage.connectors) == 1
    entry = usage.connectors[0]
    assert entry.connector_id == str(connector.id)
    assert entry.project_id == str(project_id)
    assert entry.project_display_name is None


@pytest.mark.asyncio
async def test_list_connectors_using_identity_excludes_archived_project(
        db: StateDB) -> None:
    (_, project_id, _, identities, _, identity,
     _) = await _bound_databricks_connector(db)

    await ProjectService(db).archive_project(project_id, ADMIN)

    usage = await identities.list_connectors_using_identity(identity.id, ADMIN)

    assert usage.identity_id == str(identity.id)
    assert usage.connectors == []


@pytest.mark.asyncio
async def test_list_connectors_using_identity_empty_for_unused_identity(
        db: StateDB) -> None:
    group_id = await _create_group(db, 'team')
    identities, _ = _services(db)
    identity = await _create_bound_databricks_identity(identities, group_id,
                                                       'identity', 'secret')

    usage = await identities.list_connectors_using_identity(identity.id, ADMIN)

    assert usage.connectors == []


@pytest.mark.asyncio
async def test_list_connectors_using_identity_requires_admin(
        db: StateDB) -> None:
    (group_id, _, _, identities, _, identity,
     _) = await _bound_databricks_connector(db)
    member = _Principal(UserID('member'), 'Member', (str(group_id), ))

    with pytest.raises(IdentityForbiddenError):
        await identities.list_connectors_using_identity(identity.id, member)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'live_reference',
    [False, True],
    ids=('decommission-and-delete', 'live-reference-blocks-delete'),
)
async def test_unbound_connector_admin_delete_semantics(
        db: StateDB, live_reference: bool) -> None:
    (_, project_id, user, identities, connectors, identity,
     connector) = await _bound_databricks_connector(db)
    await identities.replace_groups(identity.id,
                                    api.ReplaceIdentityGroupsArgs(group_ids=[]),
                                    ADMIN)

    if live_reference:
        connectors._tables.partition_connector_references = AsyncMock(
            return_value=([_live_table_reference('customers')], [], []))
        with pytest.raises(ConnectorInUseError, match='live table'):
            await connectors.delete(project_id, connector.id, ADMIN)
        with pytest.raises(IdentityInUseError):
            await identities.delete(identity.id, ADMIN)
    else:
        with pytest.raises(IdentityForbiddenError, match='not mapped'):
            await connectors.delete(project_id, connector.id, user)
        deleted_connector = await connectors.delete(project_id, connector.id,
                                                    ADMIN)
        deleted_identity = await identities.delete(identity.id, ADMIN)
        assert deleted_connector.id == connector.id
        assert deleted_identity.id == identity.id


def _live_table_reference(alias: str) -> SimpleNamespace:
    """A stand-in for a table row that reaches the user, for the tests that
    exercise the delete guard without seeding real tables."""
    return SimpleNamespace(alias=alias, id=FPID('a' * 32))


def _connector_table_definition(connector_id: ConnectorID) -> TableDefinition:
    return TableDefinition(
        cols=[Column(name='id', stype=Stype.ID, dtype=Dtype.int64)],
        source_table=DatabricksSourceTable(
            databricks_connector_id=connector_id,
            table='sales.analytics.customers',
            data_source_type=DataSourceType.DATABRICKS,
        ),
    )


@pytest.mark.asyncio
async def test_unbound_connector_allows_table_alias_cleanup(
        db: StateDB) -> None:
    (_, project_id, _, identities, connectors, identity,
     connector) = await _bound_databricks_connector(db)
    tables = TableService(db)
    created = await tables.save_table_definition(
        project_id,
        _connector_table_definition(connector.id),
        alias='customers',
    )
    await identities.replace_groups(identity.id,
                                    api.ReplaceIdentityGroupsArgs(group_ids=[]),
                                    ADMIN)

    cleared = await tables.clear_table_alias(project_id, 'customers')

    assert cleared is not None
    assert cleared.id == created.id
    assert cleared.alias is None
    assert cleared.ref_count == created.ref_count
    assert await tables.get_table_entry(project_id, created.id) is None
    assert await tables.clear_table_alias(project_id, 'customers') is None
    deleted = await connectors.delete(project_id, connector.id, ADMIN)
    assert deleted.id == connector.id


@pytest.mark.asyncio
async def test_unbound_connector_allows_graph_and_table_reference_cleanup(
        db: StateDB) -> None:
    (_, project_id, _, identities, connectors, identity,
     connector) = await _bound_databricks_connector(db)
    tables = TableService(db)
    table = await tables.save_table_definition(
        project_id,
        _connector_table_definition(connector.id),
        alias='customers',
    )
    table.ref_count += 1
    graph_id = FPID('a' * 32)
    graph = GraphDBEntryV2(
        id=graph_id,
        project_id=project_id,
        table_name_to_dbid={TableName('customers'): table.id},
        col_groups=[],
        alias='customer graph',
        ref_count=1,
        timestamps=get_current_timestamps(),
    )
    async with db.write_tx() as tx:
        await StateDBTableRepository().upsert(tx, table.id, table)
        await StateDBGraphRepository().upsert(tx, graph.id, graph)
    await identities.replace_groups(identity.id,
                                    api.ReplaceIdentityGroupsArgs(group_ids=[]),
                                    ADMIN)

    graphs = GraphService(db, connectors)
    with pytest.raises(ConnectorInUseError, match='live table'):
        await connectors.delete(project_id, connector.id, ADMIN)
    cleared = await tables.clear_table_alias(project_id, 'customers')
    assert cleared is not None
    assert cleared.alias is None
    assert cleared.ref_count == 1
    with pytest.raises(ConnectorInUseError, match='live table'):
        await connectors.delete(project_id, connector.id, ADMIN)
    assert await graphs.decrement_or_delete_graph_definition(
        project_id, graph.id)
    assert await tables.get_table_entry(project_id, table.id) is None
    deleted_connector = await connectors.delete(project_id, connector.id, ADMIN)
    deleted_identity = await identities.delete(identity.id, ADMIN)

    assert deleted_connector.id == connector.id
    assert deleted_identity.id == identity.id


@pytest.mark.asyncio
@pytest.mark.parametrize('queued_job', ['training', 'artifact-export'])
async def test_connector_delete_during_queued_job_makes_resolution_fail(
        db: StateDB, queued_job: str) -> None:
    (_, project_id, user, _, connectors, _,
     connector) = await _bound_databricks_connector(db)

    if queued_job == 'training':
        await MLJobService(db).create_training_job(
            pquery_id=PredictiveQueryID('risk'),
            pquery_def_fpid=None,
            graph_id=GraphIDv1('graph'),
            now=now_utc(),
            advanced_options=ModelPlan(),
            custom_train_table=CustomTrainingTable(
                connector_spec=ConnectorExecutionRef(project_id=project_id,
                                                     connector_id=connector.id),
                source_table=DatabricksSourceTable(
                    databricks_connector_id=str(connector.id),
                    table='training_data',
                ),
                weight_col_name='weight',
            ),
            project_id=project_id,
        )
        actions = CONNECTOR_GRAPH_ACTIONS
    else:
        await ArtifactExportJobService(db).create(
            id=ArtifactExportJobID.make_random_id(),
            project_id=project_id,
            created_by=user.user_id,
            request_id='00000000-0000-4000-8000-000000000001',
            request_digest='request-digest',
        )
        actions = CONNECTOR_OUTPUT_ACTIONS

    reference = await connectors.execution_ref_for_project(
        user, project_id, connector.id, actions)
    deleted = await connectors.delete(project_id, connector.id, user)
    assert deleted.id == connector.id
    assert reference.authorization_token
    with pytest.raises(ConnectorDoesNotExistError):
        await connectors.resolve_for_activity(project_id, connector.id)


@pytest.mark.asyncio
async def test_project_scoped_upload_connector_is_not_reused_across_projects(
        db: StateDB) -> None:
    group_id = await _create_group(db, 'team')
    project_a = await _create_project(db, group_id, 'a')
    project_b = await _create_project(db, group_id, 'b')
    user = _Principal(UserID('member'), 'Member', (str(group_id), ))
    _, connectors = _services(db)

    connector_a = await connectors.register_project_upload_connector(
        project_a, '/tmp/project-a/uploads')
    connector_b = await connectors.register_project_upload_connector(
        project_b, '/tmp/project-b/uploads')

    assert connector_a != connector_b
    assert (await connectors.get_internal(project_a,
                                          connector_a)).project_id == project_a
    resolved = await connectors.config_for_project(user, project_a, connector_a)
    assert resolved.connector_id == connector_a
    assert resolved.connector_config.name == str(connector_a)
    activity_resolved = await connectors.resolve_for_activity(
        project_a, connector_a)
    assert isinstance(activity_resolved, ConnectorSpec)
    assert activity_resolved.is_project_upload is True
    with pytest.raises(ConnectorDoesNotExistError):
        await connectors.get_internal(project_b, connector_a)


@pytest.mark.asyncio
async def test_child_delegation_recheck_accepts_only_visible_external_or_canonical_upload(
        db: StateDB) -> None:
    (group_id, project_id, user, identities, connectors, identity,
     external) = await _bound_databricks_connector(
         db,
         group_name='delegation-team',
         project_name='delegation-project',
         identity_name='delegation-identity',
         connector_name='warehouse',
     )
    upload_id = await connectors.register_project_upload_connector(
        project_id, '/tmp/delegation-project/uploads')
    arbitrary_internal = InternalConnectorDBEntry(
        id=ConnectorID.generate(),
        project_id=project_id,
        name='fixture',
        normalized_name='fixture',
        description='test-only internal connector',
        config=FileConnectorResourceConfig(name='fixture',
                                           root_dir='/tmp/fixture'),
        timestamps=get_current_timestamps(),
    )
    await seed_connector(db, arbitrary_internal)

    await connectors.require_delegatable_for_project(project_id, external.id)
    await connectors.require_delegatable_for_project(project_id, upload_id)
    # Public discovery deliberately remains external-only.
    with pytest.raises(ConnectorDoesNotExistError):
        await connectors.get_for_project(project_id, upload_id, user)
    with pytest.raises(ConnectorDoesNotExistError):
        await connectors.require_delegatable_for_project(
            project_id, arbitrary_internal.id)

    await identities.replace_groups(
        identity.id,
        api.ReplaceIdentityGroupsArgs(group_ids=[]),
        ADMIN,
    )
    with pytest.raises(IdentityForbiddenError, match='not mapped'):
        await connectors.require_delegatable_for_project(
            project_id, external.id)

    await identities.replace_groups(
        identity.id,
        api.ReplaceIdentityGroupsArgs(group_ids=[str(group_id)]),
        ADMIN,
    )
    groups = JsonRepository('groups', KumoGroupDBEntry)
    async with db.write_tx() as tx:
        group = await groups.get(tx, str(group_id))
        assert group is not None
        group.payload.status = GroupStatus.DELETED
        await groups.upsert(tx, group.id, group.payload)
    with pytest.raises(IdentityForbiddenError, match='Group is not active'):
        await connectors.resolve_for_activity(project_id, external.id)
    with pytest.raises(IdentityForbiddenError, match='Group is not active'):
        await connectors.require_delegatable_for_project(
            project_id, external.id)
    async with db.write_tx() as tx:
        group = await groups.get(tx, str(group_id))
        assert group is not None
        group.payload.status = GroupStatus.ACTIVE
        await groups.upsert(tx, group.id, group.payload)
    await connectors.delete(project_id, external.id, user)
    with pytest.raises(ConnectorDoesNotExistError):
        await connectors.require_delegatable_for_project(
            project_id, external.id)

    async with db.write_tx() as tx:
        upload = await connectors._connectors.get_required(tx, upload_id)
        upload.tombstoned_at = now_utc()
        await connectors._connectors.upsert(tx, upload)
    with pytest.raises(ConnectorDoesNotExistError):
        await connectors.require_delegatable_for_project(project_id, upload_id)


@pytest.mark.asyncio
async def test_source_table_connector_metadata_enforces_project_lifecycle_and_internal_kind(
        db: StateDB) -> None:
    (group_id, project_a, user, _, connectors, _,
     external) = await _bound_databricks_connector(db, project_name='a',
                                                   connector_name='warehouse')
    project_b = await _create_project(db, group_id, 'b')
    archived_project = await _create_project(db, group_id, 'archived')
    upload = await connectors.register_project_upload_connector(
        project_a, '/tmp/project-a/uploads')
    archived_upload = await connectors.register_project_upload_connector(
        archived_project, '/tmp/archived/uploads')
    generic_internal = InternalConnectorDBEntry(
        id=ConnectorID.generate(),
        project_id=project_a,
        name='fixture',
        normalized_name='fixture',
        description='test-only internal connector',
        config=FileConnectorResourceConfig(name='fixture',
                                           root_dir='/tmp/fixture'),
        timestamps=get_current_timestamps(),
    )
    await seed_connector(db, generic_internal)
    generic_resolved = await connectors.resolve_for_activity(
        project_a, generic_internal.id)
    assert isinstance(generic_resolved, ConnectorSpec)
    assert generic_resolved.is_project_upload is False

    assert await connectors.source_table_connector_for_project(
        project_a, external.id, user) == external
    assert await connectors.source_table_connector_for_project(
        project_a, upload, user) is None
    with pytest.raises(ConnectorDoesNotExistError):
        await connectors.source_table_connector_for_project(
            project_b, upload, user)
    with pytest.raises(ConnectorDoesNotExistError):
        await connectors.source_table_connector_for_project(
            project_a, generic_internal.id, user)

    async with db.write_tx() as tx:
        persisted_upload = await connectors._connectors.get(tx, upload)
        assert persisted_upload is not None
        persisted_upload.tombstoned_at = now_utc()
        await connectors._connectors.upsert(tx, persisted_upload)
    with pytest.raises(ConnectorDoesNotExistError):
        await connectors.source_table_connector_for_project(
            project_a, upload, user)

    projects = JsonRepository('projects', ProjectDBEntry)
    async with db.write_tx() as tx:
        archived = await projects.get(tx, str(archived_project))
        assert archived is not None
        archived.payload.status = ProjectStatus.ARCHIVED
        await projects.upsert(tx, str(archived_project), archived.payload)
    with pytest.raises(ProjectNotFoundError, match='not found'):
        await connectors.source_table_connector_for_project(
            archived_project, archived_upload, user)


@pytest.mark.asyncio
async def test_upload_connector_uses_reserved_namespace_and_exact_root(
        db: StateDB) -> None:
    (_, project_id, user, _, connectors, identity,
     external) = await _bound_databricks_connector(db, connector_name='uploads')

    # The ordinary name remains distinct from the hidden connector namespace.
    internal_id = await connectors.register_project_upload_connector(
        project_id, '/tmp/project/uploads')
    assert external.id != internal_id

    # Reuse is permitted only for the exact canonical project/root config.
    assert await connectors.register_project_upload_connector(
        project_id, '/tmp/project/uploads') == internal_id
    with pytest.raises(ValueError, match='Invalid existing'):
        await connectors.register_project_upload_connector(
            project_id, '/tmp/other/uploads')

    reserved_args = _connector_args(PROJECT_UPLOAD_CONNECTOR_NAME, identity.id,
                                    'schema')
    with pytest.raises(ValueError, match='reserved for internal uploads'):
        await connectors.authorize_create(project_id, reserved_args, user)


@pytest.mark.asyncio
async def test_connector_name_length_is_bounded_on_every_write_path(
        db: StateDB) -> None:
    (_, project_id, user, _, connectors, identity,
     connector) = await _bound_databricks_connector(db)
    over_long = 'n' * (MAX_CONNECTOR_NAME_LENGTH + 1)
    args = _connector_args(over_long, identity.id, 'schema')

    with pytest.raises(ValueError, match='must be at most'):
        await connectors.authorize_create(project_id, args, user)
    with pytest.raises(ValueError, match='must be at most'):
        await connectors.create_from_activity(project_id, args, user.user_id,
                                              set())
    with pytest.raises(ValueError, match='must be at most') as caught:
        await connectors.update_metadata(
            project_id, connector.id,
            api.UpdateConnectorMetadataArgs(name=over_long), user)

    assert over_long not in str(caught.value)
    at_bound = 'b' * MAX_CONNECTOR_NAME_LENGTH
    renamed = await connectors.update_metadata(
        project_id, connector.id,
        api.UpdateConnectorMetadataArgs(name=at_bound), user)
    assert renamed.name == at_bound


@pytest.mark.asyncio
async def test_provider_outage_allows_native_membership_without_leaking_hidden_project(
        db: StateDB) -> None:
    group_id = await _create_group(db, 'team')
    project_id = await _create_project(db, group_id, 'project')
    private_id = await _create_project(db, group_id, 'private', owner='owner',
                                       private=True)
    _, connectors = _services(db)
    native_member = _Principal(UserID('native'), 'Native', (str(group_id), ),
                               external_membership_unavailable=True)
    unresolved_user = _Principal(UserID('external'), 'External', (),
                                 external_membership_unavailable=True)

    assert (await
            connectors.require_project_access(native_member,
                                              project_id)).id == project_id
    with pytest.raises(ExternalMembershipUnavailableError) as shared_error:
        await connectors.require_project_access(unresolved_user, project_id)
    assert str(shared_error.value) == (
        'External group membership is unavailable; please retry')

    unresolved_owner = _Principal(
        UserID('owner'),
        'Owner',
        (),
        external_membership_unavailable=True,
    )
    with pytest.raises(ExternalMembershipUnavailableError) as private_error:
        await connectors.require_project_access(unresolved_owner, private_id)
    assert str(private_error.value) == str(shared_error.value)

    with pytest.raises(ProjectNotFoundError):
        await connectors.require_project_access(unresolved_user, private_id)


@pytest.mark.asyncio
async def test_a_container_mounted_snowflake_token_is_refused(
        db: StateDB) -> None:
    """The token lives on the filesystem of a running Snowpark Container
    Services container. This service is not that container and has no such
    file, so an identity created here would be stored and then fail at every
    use. Both identity routers reach this one method, so refusing here covers
    both.
    """
    identities, _ = _services(db)
    args = api.CreateIdentityArgs(
        display_name='snowflake-spcs',
        config=api.SnowflakeIdentityConfig(
            account='organization-account',
            authentication=(
                api.SnowflakeAuthenticationKind.OAUTH_TOKEN_FILE),
        ),
    )

    with pytest.raises(IdentityForbiddenError, match='Kumo cannot read'):
        await identities.create(args, ADMIN)

    assert await identities.list_admin(ADMIN) == []


def _retry_definition(connector_id: ConnectorID,
                      num_cols: int) -> TableDefinition:
    """One attempt at adding the same source table, with the column the stats
    step choked on progressively dropped, so each retry is a new fingerprint.
    """
    return TableDefinition(
        cols=[
            Column(
                name=f'c{index}',
                stype=Stype.ID if index == 0 else Stype.numerical,
                dtype=Dtype.int64 if index == 0 else Dtype.float64,
            ) for index in range(num_cols)
        ],
        source_table=DatabricksSourceTable(
            databricks_connector_id=connector_id,
            table='sales.analytics.dim_cr_merch_acct',
            data_source_type=DataSourceType.DATABRICKS,
        ),
    )


async def _all_table_rows(db: StateDB, project_id: ProjectID):
    async with db.read_tx() as tx:
        return await StateDBTableRepository().list(tx, project_id)


@pytest.mark.asyncio
async def test_renaming_over_an_unreferenced_table_deletes_it(
        db: StateDB) -> None:
    """Retried table adds must not accumulate rows nothing can reach.

    Each retry claims the same display name under a new fingerprint, which
    unnames the previous row. Left behind, those rows are absent from the
    alias-only Tables list and have no delete handle, yet still count as live
    connector references.
    """
    (_, project_id, _, _, connectors, _,
     connector) = await _bound_databricks_connector(db)
    tables = TableService(db)

    fpids = [(await tables.save_table_definition(
        project_id,
        _retry_definition(connector.id, num_cols),
        alias='dim_cr_merch_acct',
        force_rename=True,
    )).id for num_cols in (5, 4, 3, 2)]
    assert len(set(fpids)) == 4

    rows = await _all_table_rows(db, project_id)
    assert [row.id for row in rows] == [fpids[-1]]

    await tables.clear_table_alias(project_id, 'dim_cr_merch_acct')
    assert await _all_table_rows(db, project_id) == []
    assert (await connectors.delete(project_id, connector.id,
                                    ADMIN)).id == connector.id


@pytest.mark.asyncio
async def test_renaming_over_a_graph_referenced_table_keeps_it(
        db: StateDB) -> None:
    """The row is only dropped when nothing reaches it: a graph pinning the
    fingerprint keeps it, unnamed, exactly as before."""
    (_, project_id, _, _, connectors, _,
     connector) = await _bound_databricks_connector(db)
    tables = TableService(db)
    pinned = await tables.save_table_definition(
        project_id,
        _retry_definition(connector.id, 5),
        alias='dim_cr_merch_acct',
    )
    graph = GraphDBEntryV2(
        id=FPID('b' * 32),
        project_id=project_id,
        table_name_to_dbid={TableName('dim_cr_merch_acct'): pinned.id},
        col_groups=[],
        alias='merchant graph',
        ref_count=1,
        timestamps=get_current_timestamps(),
    )
    async with db.write_tx() as tx:
        await StateDBGraphRepository().upsert(tx, graph.id, graph)

    replacement = await tables.save_table_definition(
        project_id,
        _retry_definition(connector.id, 4),
        alias='dim_cr_merch_acct',
        force_rename=True,
    )

    rows = {row.id: row for row in await _all_table_rows(db, project_id)}
    assert set(rows) == {pinned.id, replacement.id}
    assert rows[pinned.id].alias is None
    with pytest.raises(ConnectorInUseError, match='live table'):
        await connectors.delete(project_id, connector.id, ADMIN)


@pytest.mark.asyncio
async def test_renaming_releases_the_alias_reference(db: StateDB) -> None:
    """The alias holds one reference, so losing it must decrement the count.

    Left alone the row keeps looking claimed by a name it no longer has, and a
    later graph delete strands it at a count nothing accounts for.
    """
    (_, project_id, _, _, connectors, _,
     connector) = await _bound_databricks_connector(db)
    tables = TableService(db)
    definition = _retry_definition(connector.id, 5)
    # ``ref_count`` counts claims on one fingerprint, and saving the same
    # definition again takes another. Two claims, one of which is the alias.
    for _ in range(2):
        pinned = await tables.save_table_definition(
            project_id, definition, alias='dim_cr_merch_acct')
    async with db.write_tx() as tx:
        before = await StateDBTableRepository().get(tx, project_id, pinned.id)
    assert before is not None and before.ref_count == 2

    await tables.save_table_definition(
        project_id,
        _retry_definition(connector.id, 4),
        alias='dim_cr_merch_acct',
        force_rename=True,
    )

    async with db.write_tx() as tx:
        after = await StateDBTableRepository().get(tx, project_id, pinned.id)
    assert after is not None
    assert after.alias is None
    # One claim released with the alias, one still held.
    assert after.ref_count == 1


@pytest.mark.asyncio
async def test_connector_delete_collects_the_graph_it_orphans(
        db: StateDB) -> None:
    """A graph is readable by ``graph-<fpid>`` without an alias, so its tables
    cannot be dropped from under it. It has no delete route either, so leaving
    it would move the wedge up a level -- it goes in the same transaction.
    """
    (_, project_id, _, _, connectors, _,
     connector) = await _bound_databricks_connector(db)
    tables = StateDBTableRepository()
    graphs = StateDBGraphRepository()
    definition = _retry_definition(connector.id, 5)
    async with db.write_tx() as tx:
        await tables.upsert(
            tx, definition.fpid,
            TableDBEntryV2(
                id=definition.fpid,
                project_id=project_id,
                config=definition,
                col_dtypes={},
                alias=None,
                timestamps=get_current_timestamps(),
            ))
        stale = GraphDBEntryV2(
            id=FPID('d' * 32),
            project_id=project_id,
            table_name_to_dbid={TableName('t'): definition.fpid},
            col_groups=[],
            alias=None,
            ref_count=1,
            timestamps=get_current_timestamps(),
        )
        await graphs.upsert(tx, stale.id, stale)

    assert (await connectors.delete(project_id, connector.id,
                                    ADMIN)).id == connector.id
    assert await _all_table_rows(db, project_id) == []
    async with db.write_tx() as tx:
        # The graph went with its tables: nothing is left to read and break.
        assert await graphs.get(tx, project_id, FPID('d' * 32)) is None


@pytest.mark.asyncio
async def test_connector_delete_collects_unreachable_table_rows(
        db: StateDB) -> None:
    """Rows orphaned before this fix shipped are collected on connector
    delete, so an already-wedged deployment recovers without a migration."""
    (_, project_id, _, _, connectors, _,
     connector) = await _bound_databricks_connector(db)
    tables = StateDBTableRepository()
    orphans = []
    async with db.write_tx() as tx:
        for num_cols in (5, 4, 3):
            definition = _retry_definition(connector.id, num_cols)
            entry = TableDBEntryV2(
                id=definition.fpid,
                project_id=project_id,
                config=definition,
                col_dtypes={},
                alias=None,
                timestamps=get_current_timestamps(),
            )
            await tables.upsert(tx, entry.id, entry)
            orphans.append(entry.id)

    assert len(await _all_table_rows(db, project_id)) == 3
    assert await TableService(db).list_named_tables(project_id) == []

    assert (await connectors.delete(project_id, connector.id,
                                    ADMIN)).id == connector.id
    assert await _all_table_rows(db, project_id) == []


@pytest.mark.asyncio
async def test_connector_delete_names_the_blocking_tables(db: StateDB) -> None:
    """The error has to say what to remove; the old message named nothing."""
    (_, project_id, _, _, connectors, _,
     connector) = await _bound_databricks_connector(db)
    tables = TableService(db)
    for num_cols, alias in ((5, 'merchants'), (4, 'accounts')):
        await tables.save_table_definition(
            project_id, _retry_definition(connector.id, num_cols), alias=alias)

    with pytest.raises(ConnectorInUseError,
                       match='live table: accounts, merchants'):
        await connectors.delete(project_id, connector.id, ADMIN)


async def _graph_over(db: StateDB, project_id: ProjectID, fpid: str,
                      table_fpid: FPID, alias: str | None) -> FPID:
    entry = GraphDBEntryV2(
        id=FPID(fpid),
        project_id=project_id,
        table_name_to_dbid={TableName('dim_cr_merch_acct'): table_fpid},
        col_groups=[],
        alias=alias,
        ref_count=1,
        timestamps=get_current_timestamps(),
    )
    async with db.write_tx() as tx:
        await StateDBGraphRepository().upsert(tx, entry.id, entry,
                                              force_rename=True)
    return entry.id


@pytest.mark.asyncio
async def test_connector_delete_ignores_a_graph_nothing_can_reach(
        db: StateDB) -> None:
    """Re-saving an edited graph under the same name unnames the previous
    graph row, which keeps pinning its tables. That chain is invisible end to
    end -- the Graphs list is alias-only too -- so it must not hold the
    connector open, or the fix just moves the wedge one level up.
    """
    (_, project_id, _, _, connectors, _,
     connector) = await _bound_databricks_connector(db)
    tables = TableService(db)
    superseded = await tables.save_table_definition(
        project_id, _retry_definition(connector.id, 5))
    current = await tables.save_table_definition(
        project_id, _retry_definition(connector.id, 4))
    await _graph_over(db, project_id, 'a' * 32, superseded.id, 'merchants')
    await _graph_over(db, project_id, 'b' * 32, current.id, 'merchants')

    async with db.read_tx() as tx:
        graphs = await StateDBGraphRepository().list(tx, project_id)
    assert sorted(graph.alias or '' for graph in graphs) == ['', 'merchants']

    # The named graph still uses its table, so the connector stays blocked.
    with pytest.raises(ConnectorInUseError, match='live table'):
        await connectors.delete(project_id, connector.id, ADMIN)

    # Once the user removes the only graph they can see, nothing reachable is
    # left and both table rows are collected.
    graphs = GraphService(db, connectors)
    await graphs.clear_graph_alias(project_id, 'merchants')
    assert (await connectors.delete(project_id, connector.id,
                                    ADMIN)).id == connector.id
    assert await _all_table_rows(db, project_id) == []


@pytest.mark.asyncio
async def test_connector_delete_keeps_a_table_pinned_by_a_surviving_graph(
        db: StateDB) -> None:
    """A stale graph sharing a table pin cannot unpin the named graph."""
    (_, project_id, _, _, connectors, _,
     connector) = await _bound_databricks_connector(db)
    table = await TableService(db).save_table_definition(
        project_id, _retry_definition(connector.id, 5))

    # Giving the second graph the same alias makes the first graph stale.
    # Both nonetheless reference the same table fingerprint.
    stale_fpid = await _graph_over(db, project_id, 'e' * 32, table.id,
                                   'merchants')
    reachable_fpid = await _graph_over(db, project_id, 'f' * 32, table.id,
                                       'merchants')

    with pytest.raises(ConnectorInUseError, match='live table'):
        await connectors.delete(project_id, connector.id, ADMIN)

    async with db.read_tx() as tx:
        graphs = StateDBGraphRepository()
        assert await graphs.get(tx, project_id, stale_fpid) is not None
        assert await graphs.get(tx, project_id, reachable_fpid) is not None
        assert await StateDBTableRepository().get(tx, project_id,
                                                  table.id) is not None


@pytest.mark.asyncio
async def test_connector_delete_blocked_by_a_query_held_graph(
        db: StateDB) -> None:
    """An unnamed graph a stored query holds is still reachable -- the SDK
    saves the graph anonymously and the query pins it -- so its tables remain
    live references."""
    (_, project_id, _, _, connectors, _,
     connector) = await _bound_databricks_connector(db)
    table = await TableService(db).save_table_definition(
        project_id, _retry_definition(connector.id, 5))
    graph_fpid = await _graph_over(db, project_id, 'c' * 32, table.id, None)
    async with db.write_tx() as tx:
        await StateDBPQueryRepository().add_definition(
            tx,
            PQueryDefinition(
                pquery_name='churn',
                query_str='PREDICT COUNT(*) FOR EACH dim_cr_merch_acct.c0',
                graph_fpid=graph_fpid,
                timestamps=get_current_timestamps(),
            ),
        )

    with pytest.raises(ConnectorInUseError, match='live table'):
        await connectors.delete(project_id, connector.id, ADMIN)
    assert len(await _all_table_rows(db, project_id)) == 1
