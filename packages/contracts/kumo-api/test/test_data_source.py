# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from kumoapi.data_source import (
    AMBIENT_S3_IDENTITY_SLUG,
    DEFAULT_GROUP_ID,
    DEFAULT_GROUP_SLUG,
    DEFAULT_PROJECT_ID,
    DEFAULT_PROJECT_SLUG,
    ADLSAnyURIConnectorTarget,
    ADLSConnectorResourceConfig,
    ADLSIdentityConfig,
    ADLSRootedConnectorTarget,
    AzureCloud,
    BigQueryConnectorResourceConfig,
    BigQueryIdentityConfig,
    CompleteFileUploadRequest,
    ConnectorResponse,
    CreateConnectorArgs,
    CreateGroupArgs,
    CreateIdentityArgs,
    CreateProjectArgs,
    DatabricksConnectorResourceConfig,
    DatabricksIdentityConfig,
    DatabricksOAuthCredentials,
    DeleteUploadedFileRequest,
    GCSAnyURIConnectorTarget,
    GCSConnectorResourceConfig,
    GCSIdentityConfig,
    GCSRootedConnectorTarget,
    GoogleServiceAccountKeyAuthentication,
    GoogleServiceAccountKeyCredentials,
    GoogleWorkloadIdentityAuthentication,
    GroupResponse,
    IdentityCredentialStatus,
    IdentityResponse,
    ProjectResponse,
    ProjectUploadConnectorResponse,
    ReplaceIdentityGroupsArgs,
    ResourceTimestamps,
    RotateIdentityCredentialsArgs,
    S3AccessKeyAuthentication,
    S3AccessKeyCredentials,
    S3AddressingStyle,
    S3AmbientAuthentication,
    S3AnyURIConnectorTarget,
    S3AssumeRoleAuthentication,
    S3AssumeRoleCredentials,
    S3ConnectionConfig,
    S3ConnectorResourceConfig,
    S3IdentityConfig,
    S3RootedConnectorTarget,
    SnowflakeAuthenticationKind,
    SnowflakeConnectorResourceConfig,
    SnowflakeIdentityConfig,
    SnowflakeKeyPairCredentials,
    SnowflakePasswordCredentials,
    StartFileUploadRequest,
    UploadFileType,
    derive_resource_slug,
    validate_identity_credentials_match,
    validate_resource_slug,
)
from kumoapi.json_serde import from_json, to_json, to_json_dict


def test_canonical_default_ids_are_stable_api_constants():
    assert DEFAULT_GROUP_ID == 'group_00000000000000000000000000000000'
    assert DEFAULT_PROJECT_ID == 'default'


def test_canonical_bootstrap_slugs_are_stable_api_constants():
    assert DEFAULT_GROUP_SLUG == 'default'
    assert DEFAULT_PROJECT_SLUG == 'default'
    assert AMBIENT_S3_IDENTITY_SLUG == 'ambient-s3'


def test_resource_slug_validation_and_deterministic_derivation():
    assert validate_resource_slug('fraud-detection') == 'fraud-detection'
    assert derive_resource_slug('  Fraud & Risk / 2026  ') == 'fraud-risk-2026'
    assert derive_resource_slug('ＦＲＡＵＤ　Detection') == 'fraud-detection'

    for invalid in ('ab', 'Uppercase', 'two--hyphens', '-leading', 'trailing-',
                    'with_underscore', 'équipe'):
        with pytest.raises(ValueError):
            validate_resource_slug(invalid)
    with pytest.raises(ValueError, match='Reserved'):
        validate_resource_slug('default')
    with pytest.raises(ValueError, match='Reserved'):
        validate_resource_slug(AMBIENT_S3_IDENTITY_SLUG)
    assert validate_resource_slug('default', allow_reserved=True) == 'default'
    assert validate_resource_slug(
        AMBIENT_S3_IDENTITY_SLUG,
        allow_reserved=True) == AMBIENT_S3_IDENTITY_SLUG
    with pytest.raises(ValueError, match='Reserved'):
        derive_resource_slug('Default')


def test_resource_creation_slugs_are_optional_but_strict():
    assert CreateProjectArgs(name='Fraud Detection',
                             group_id='group-risk').slug is None
    assert CreateGroupArgs(display_name='Risk Team',
                           slug='risk-team').slug == ('risk-team')
    identity = CreateIdentityArgs(
        display_name='Production AWS',
        slug='prod-aws',
        config=S3IdentityConfig(authentication=S3AmbientAuthentication()),
    )
    assert identity.slug == 'prod-aws'

    with pytest.raises((ValidationError, ValueError), match='Reserved'):
        CreateProjectArgs(name='Default', group_id='group-risk',
                          slug='default')


def test_resource_responses_require_slug_and_accept_bootstrap_reserved_values(
):
    project = ProjectResponse(id=DEFAULT_PROJECT_ID, slug='default',
                              display_name='Default',
                              group_id=DEFAULT_GROUP_ID)
    group = GroupResponse(id=DEFAULT_GROUP_ID, slug='default',
                          display_name='Default')
    assert project.slug == group.slug == 'default'


def _identity_create_payloads():
    return [
        CreateIdentityArgs(
            display_name='databricks-production',
            group_ids=['group_sales', 'group_finance'],
            config=DatabricksIdentityConfig(
                host='https://workspace.cloud.databricks.com',
                client_id='service-principal-id',
            ),
            credentials=DatabricksOAuthCredentials(
                client_secret=SecretStr('databricks-secret')),
        ),
        CreateIdentityArgs(
            display_name='snowflake-password',
            config=SnowflakeIdentityConfig(
                account='org-account',
                username='kumo_service_user',
                authentication=SnowflakeAuthenticationKind.PASSWORD,
            ),
            credentials=SnowflakePasswordCredentials(
                password=SecretStr('sf-password-secret-value')),
        ),
        CreateIdentityArgs(
            display_name='snowflake-key-pair',
            config=SnowflakeIdentityConfig(
                account='org-account',
                username='kumo_key_user',
                authentication=SnowflakeAuthenticationKind.KEY_PAIR,
            ),
            credentials=SnowflakeKeyPairCredentials(
                private_key=SecretStr('snowflake-private-key'),
                private_key_passphrase=SecretStr('snowflake-passphrase'),
            ),
        ),
        CreateIdentityArgs(
            display_name='bigquery-production',
            config=BigQueryIdentityConfig(
                authentication=GoogleServiceAccountKeyAuthentication(
                    project_id='service-account-project',
                    client_email=(
                        '847355853533-compute@developer.gserviceaccount.com'),
                    client_id='1234567890',
                    auth_uri='https://accounts.google.com/o/oauth2/auth',
                    token_uri='https://oauth2.googleapis.com/token',
                ), ),
            credentials=GoogleServiceAccountKeyCredentials(
                private_key_id=SecretStr('private-key-id'),
                private_key=SecretStr('bigquery-private-key'),
            ),
        ),
        CreateIdentityArgs(
            display_name='s3-role',
            config=S3IdentityConfig(
                authentication=S3AssumeRoleAuthentication(
                    partition='aws',
                    role_arn='arn:aws:iam::123456789012:role/kumo-reader',
                ), ),
            credentials=S3AssumeRoleCredentials(
                external_id=SecretStr('customer-external-id')),
        ),
        CreateIdentityArgs(
            display_name='s3-ambient',
            config=S3IdentityConfig(authentication=S3AmbientAuthentication()),
        ),
        CreateIdentityArgs(
            display_name='s3-compatible-access-key',
            config=S3IdentityConfig(
                authentication=S3AccessKeyAuthentication(),
                connection=S3ConnectionConfig(
                    endpoint_url='https://minio.example.test:9000',
                    region='us-east-1',
                    addressing_style=S3AddressingStyle.PATH,
                ),
            ),
            credentials=S3AccessKeyCredentials(
                access_key_id=SecretStr('s3-access-key-id'),
                secret_access_key=SecretStr('s3-secret-access-key'),
                session_token=SecretStr('s3-session-token'),
                ca_certificate=SecretStr('-----BEGIN CERTIFICATE-----\n'
                                         'MIIB\n'
                                         '-----END CERTIFICATE-----'),
            ),
        ),
        CreateIdentityArgs(
            display_name='gcs-service-account',
            config=GCSIdentityConfig(
                authentication=GoogleWorkloadIdentityAuthentication(
                    service_account_email=(
                        'kumo-reader@example.iam.gserviceaccount.com'))),
        ),
        CreateIdentityArgs(
            display_name='adls-application',
            config=ADLSIdentityConfig(
                tenant_id='tenant-id',
                client_id='application-id',
                cloud=AzureCloud.PUBLIC,
            ),
        ),
    ]


@pytest.mark.parametrize('payload', _identity_create_payloads())
def test_identity_create_roundtrip_and_secret_redaction(payload):
    safe_json = to_json(payload)
    insecure_json = to_json(payload, insecure=True)

    assert from_json(insecure_json, CreateIdentityArgs) == payload
    for secret in ('databricks-secret', 'sf-password-secret-value',
                   'snowflake-private-key', 'snowflake-passphrase',
                   'private-key-id', 'bigquery-private-key',
                   'customer-external-id', 's3-access-key-id',
                   's3-secret-access-key', 's3-session-token',
                   'BEGIN CERTIFICATE'):
        assert secret not in safe_json


def test_identity_create_rejects_duplicate_initial_groups():
    with pytest.raises((ValidationError, ValueError),
                       match='group_ids must not contain duplicates'):
        CreateIdentityArgs(
            display_name='databricks-production',
            group_ids=['group_sales', 'group_sales'],
            config=DatabricksIdentityConfig(
                host='https://workspace.cloud.databricks.com',
                client_id='service-principal-id',
            ),
            credentials=DatabricksOAuthCredentials(
                client_secret=SecretStr('databricks-secret')),
        )


@pytest.mark.parametrize(
    ('config', 'credentials', 'error'),
    [
        (
            DatabricksIdentityConfig(host='https://workspace',
                                     client_id='client-id'),
            SnowflakePasswordCredentials(password=SecretStr('wrong')),
            'config and credential types',
        ),
        (
            SnowflakeIdentityConfig(
                account='account',
                username='user',
                authentication=SnowflakeAuthenticationKind.KEY_PAIR,
            ),
            SnowflakePasswordCredentials(password=SecretStr('wrong')),
            'config and credential types',
        ),
        (
            BigQueryIdentityConfig(
                authentication=GoogleServiceAccountKeyAuthentication(
                    project_id='service-project',
                    client_email=(
                        'service@service-project.iam.gserviceaccount.com'),
                    client_id='123',
                    auth_uri='https://accounts.google.com/o/oauth2/auth',
                    token_uri='https://oauth2.googleapis.com/token',
                ), ),
            DatabricksOAuthCredentials(client_secret=SecretStr('wrong')),
            'Google service-account key credentials are missing or invalid',
        ),
        (
            GCSIdentityConfig(
                authentication=GoogleWorkloadIdentityAuthentication(
                    service_account_email=(
                        'service@service-project.iam.gserviceaccount.com'))),
            S3AssumeRoleCredentials(external_id=SecretStr('wrong')),
            'Google workload identities cannot store credentials',
        ),
    ],
)
def test_identity_rejects_mixed_config_and_credentials(config, credentials,
                                                       error):
    with pytest.raises((ValidationError, ValueError), match=error):
        CreateIdentityArgs(
            display_name='invalid',
            config=config,
            credentials=credentials,
        )


@pytest.mark.parametrize('config_type',
                         [BigQueryIdentityConfig, GCSIdentityConfig])
@pytest.mark.parametrize('service_account_email', [
    'reader@service-project.iam.gserviceaccount.com',
    '847355853533-compute@developer.gserviceaccount.com',
])
def test_google_identity_modes_require_exact_credential_shape(
        config_type, service_account_email):
    key_authentication = GoogleServiceAccountKeyAuthentication(
        project_id='service-project',
        client_email=service_account_email,
        client_id='1234567890',
        auth_uri='https://accounts.google.com/o/oauth2/auth',
        token_uri='https://oauth2.googleapis.com/token',
    )
    key_credentials = GoogleServiceAccountKeyCredentials(
        private_key_id=SecretStr('private-key-id'),
        private_key=SecretStr('private-key'),
    )
    workload_authentication = GoogleWorkloadIdentityAuthentication(
        service_account_email=service_account_email)

    key_identity = CreateIdentityArgs(
        display_name='key identity',
        config=config_type(authentication=key_authentication),
        credentials=key_credentials,
    )
    workload_identity = CreateIdentityArgs(
        display_name='workload identity',
        config=config_type(authentication=workload_authentication),
    )
    assert key_identity.credentials == key_credentials
    assert workload_identity.credentials is None

    with pytest.raises((
            ValidationError, ValueError
    ), match='Google service-account key credentials are missing or invalid'):
        CreateIdentityArgs(
            display_name='missing key',
            config=config_type(authentication=key_authentication),
        )
    with pytest.raises(
        (ValidationError, ValueError),
            match='Google workload identities cannot store credentials'):
        CreateIdentityArgs(
            display_name='workload with key',
            config=config_type(authentication=workload_authentication),
            credentials=key_credentials,
        )


@pytest.mark.parametrize('email', [
    'user@example.com',
    'authorized-user@gmail.com',
    'service-account-project.iam.gserviceaccount.com',
    'reader@service_project.iam.gserviceaccount.com',
    'attacker@developer.gserviceaccount.com',
    '847355853533-compute@developer.gserviceaccount.com.evil.example',
])
def test_google_authentication_rejects_non_service_account_email(email):
    with pytest.raises((ValidationError, ValueError),
                       match='service-account email'):
        GoogleWorkloadIdentityAuthentication(service_account_email=email)


@pytest.mark.parametrize(
    ('field_name', 'value'),
    [
        ('auth_uri', 'http://169.254.169.254/latest/meta-data'),
        ('auth_uri', 'https://user@accounts.google.com/o/oauth2/auth'),
        ('auth_uri', 'https://accounts.google.com/o/oauth2/auth?next=evil'),
        ('token_uri', 'http://127.0.0.1/token'),
        ('token_uri', 'https://oauth2.googleapis.com/token#fragment'),
        ('token_uri', 'https://oauth2.evil.example/token'),
        ('universe_domain', 'localhost'),
        ('universe_domain', 'googleapis.com.evil.example'),
    ],
)
def test_google_key_authentication_rejects_noncanonical_endpoints(
        field_name, value):
    fields: dict[str, Any] = {
        'project_id': 'service-project',
        'client_email': 'reader@service-project.iam.gserviceaccount.com',
        'client_id': '1234567890',
        'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
        'token_uri': 'https://oauth2.googleapis.com/token',
        'universe_domain': 'googleapis.com',
        field_name: value,
    }

    with pytest.raises((ValidationError, ValueError), match='canonical'):
        GoogleServiceAccountKeyAuthentication(**fields)


def test_google_key_authentication_strips_canonical_endpoint_whitespace():
    authentication = GoogleServiceAccountKeyAuthentication(
        project_id='service-project',
        client_email='reader@service-project.iam.gserviceaccount.com',
        client_id='1234567890',
        auth_uri='  https://accounts.google.com/o/oauth2/auth  ',
        token_uri='\thttps://oauth2.googleapis.com/token\n',
        universe_domain=' googleapis.com ',
    )

    assert authentication.auth_uri == (
        'https://accounts.google.com/o/oauth2/auth')
    assert authentication.token_uri == 'https://oauth2.googleapis.com/token'
    assert authentication.universe_domain == 'googleapis.com'


def test_google_managed_service_account_email_strips_whitespace():
    email = '847355853533-compute@developer.gserviceaccount.com'
    authentication = GoogleWorkloadIdentityAuthentication(
        service_account_email=f'  {email}\n')

    assert authentication.service_account_email == email


def test_google_key_credentials_never_serialize_or_repr_secrets():
    credentials = GoogleServiceAccountKeyCredentials(
        private_key_id=SecretStr('key-id-secret'),
        private_key=SecretStr('private-key-secret'),
    )
    safe_json = to_json(credentials)
    safe_repr = repr(credentials)
    assert 'key-id-secret' not in safe_json
    assert 'private-key-secret' not in safe_json
    assert 'key-id-secret' not in safe_repr
    assert 'private-key-secret' not in safe_repr


def test_google_key_rotation_requires_safe_metadata_and_secret_dto_together():
    authentication = GoogleServiceAccountKeyAuthentication(
        project_id='service-project',
        client_email='reader@service-project.iam.gserviceaccount.com',
        client_id='1234567890',
        auth_uri='https://accounts.google.com/o/oauth2/auth',
        token_uri='https://oauth2.googleapis.com/token',
    )
    credentials = GoogleServiceAccountKeyCredentials(
        private_key_id=SecretStr('private-key-id'),
        private_key=SecretStr('private-key'),
    )
    request = RotateIdentityCredentialsArgs(
        authentication=authentication,
        credentials=credentials,
    )
    assert request.authentication == authentication
    assert request.credentials == credentials

    with pytest.raises((ValidationError, ValueError), match='together'):
        RotateIdentityCredentialsArgs(credentials=credentials)
    with pytest.raises((ValidationError, ValueError), match='together'):
        RotateIdentityCredentialsArgs(authentication=authentication)


def test_ambient_s3_identity_rejects_assume_role_credentials():
    with pytest.raises((ValidationError, ValueError),
                       match='config and credential types'):
        CreateIdentityArgs(
            display_name='ambient',
            config=S3IdentityConfig(authentication=S3AmbientAuthentication()),
            credentials=S3AssumeRoleCredentials(
                external_id=SecretStr('must-not-be-used')),
        )


def test_s3_identity_authentication_requires_discriminator_selected_fields():
    with pytest.raises(ValidationError):
        from_json(
            {
                'type': 's3',
                'authentication': {
                    'mode': 'assume_role',
                    'partition': 'aws',
                },
            }, S3IdentityConfig)


def test_s3_ambient_authentication_rejects_assume_role_fields():
    with pytest.raises(ValidationError, match='(partition|role_arn)'):
        from_json(
            {
                'type': 's3',
                'authentication': {
                    'mode': 'ambient',
                    'partition': 'aws',
                    'role_arn': 'arn:aws:iam::123456789012:role/typo',
                },
            }, S3IdentityConfig)


@pytest.mark.parametrize(
    'payload,authentication_type',
    [
        ({
            'mode': 'service_account_key',
            'project_id': 'service-project',
            'client_email': 'reader@service-project.iam.gserviceaccount.com',
            'client_id': '1234567890',
            'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
            'token_uri': 'https://oauth2.googleapis.com/token',
            'service_account_email': 'typo@example.com',
        }, GoogleServiceAccountKeyAuthentication),
        ({
            'mode': 'workload_identity',
            'service_account_email':
            'reader@service-project.iam.gserviceaccount.com',
            'project_id': 'typo-project',
        }, GoogleWorkloadIdentityAuthentication),
    ],
)
def test_google_authentication_modes_reject_other_mode_fields(
        payload, authentication_type):
    with pytest.raises(ValidationError):
        from_json(payload, authentication_type)


@pytest.mark.parametrize(
    'payload',
    [
        {
            'type': 's3',
            'bucket': 'legacy-shape-without-required-target',
        },
        {
            'type': 's3',
            'target': {
                'mode': 'rooted',
                'prefix': 'events',
            },
        },
    ],
)
def test_s3_connector_target_requires_discriminator_selected_fields(payload):
    with pytest.raises(ValidationError):
        from_json(payload, S3ConnectorResourceConfig)


@pytest.mark.parametrize(
    'payload',
    [
        {
            'type': 's3',
            'target': {
                'mode': 'any_uri',
                'bucket': 'typo-bucket',
            },
        },
        {
            'type': 's3',
            'target': {
                'mode': 'any_uri',
            },
            'bucket': 'legacy-typo-bucket',
        },
    ],
)
def test_s3_any_uri_target_rejects_rooted_and_legacy_coordinates(payload):
    with pytest.raises(ValidationError, match='bucket'):
        from_json(payload, S3ConnectorResourceConfig)


@pytest.mark.parametrize(
    'payload,config_type,unknown_field',
    [
        ({
            'type': 's3',
            'target': {
                'mode': 'any_uri',
                'region_strategy': 'typo',
            },
        }, S3ConnectorResourceConfig, 'region_strategy'),
        ({
            'type': 's3',
            'authentication': {
                'mode': 'ambient',
                'provider_metadata': 'typo',
            },
        }, S3IdentityConfig, 'provider_metadata'),
    ],
)
def test_connector_and_identity_configs_reject_unknown_fields(
        payload, config_type, unknown_field):
    with pytest.raises(ValidationError, match=unknown_field):
        from_json(payload, config_type)


def test_databricks_identity_rejects_unknown_credential_fields():
    with pytest.raises(ValidationError, match='pat'):
        from_json(
            {
                'display_name': 'oauth-identity',
                'config': {
                    'type': 'databricks',
                    'host': 'https://workspace',
                    'client_id': 'client-id',
                },
                'credentials': {
                    'kind': 'databricks_oauth',
                    'client_secret': 'secret',
                    'pat': 'typo',
                },
            },
            CreateIdentityArgs,
        )


@pytest.mark.parametrize(
    'credential_type,payload',
    [
        (DatabricksOAuthCredentials, {
            'kind': 'databricks_oauth',
            'client_secret': 'secret',
        }),
        (SnowflakePasswordCredentials, {
            'kind': 'snowflake_password',
            'password': 'secret',
        }),
        (SnowflakeKeyPairCredentials, {
            'kind': 'snowflake_key_pair',
            'private_key': 'secret',
        }),
        (GoogleServiceAccountKeyCredentials, {
            'kind': 'google_service_account_key',
            'private_key_id': 'secret-id',
            'private_key': 'secret',
        }),
        (S3AssumeRoleCredentials, {
            'kind': 's3_assume_role',
            'external_id': 'secret',
        }),
        (S3AccessKeyCredentials, {
            'kind': 's3_access_key',
            'access_key_id': 'access-key',
            'secret_access_key': 'secret-key',
        }),
    ],
)
def test_credential_dtos_reject_unknown_fields(credential_type, payload):
    with pytest.raises(ValidationError, match='credential_typo'):
        from_json({
            **payload,
            'credential_typo': 'secret',
        }, credential_type)


@pytest.mark.parametrize(
    'payload,command_type,unknown_field',
    [
        ({
            'name': 'Project',
            'group_id': 'group-a',
            'group_slug': 'typo',
        }, CreateProjectArgs, 'group_slug'),
        ({
            'name': 'events',
            'config': {
                'type': 's3',
                'target': {
                    'mode': 'any_uri'
                },
            },
            'identity_slug': 'typo',
        }, CreateConnectorArgs, 'identity_slug'),
        ({
            'group_ids': ['group-a'],
            'binding_mode': 'typo',
        }, ReplaceIdentityGroupsArgs, 'binding_mode'),
        ({
            'secret_kind': 'typo',
        }, RotateIdentityCredentialsArgs, 'secret_kind'),
    ],
)
def test_command_dtos_reject_unknown_authority_fields(payload, command_type,
                                                      unknown_field):
    with pytest.raises(ValidationError, match=unknown_field):
        from_json(payload, command_type)


def test_safe_identity_response_has_no_credentials():
    now = datetime.now(timezone.utc)
    response = IdentityResponse(
        id='identity_123',
        slug='databricks-production',
        display_name='databricks-production',
        description='Production service principal',
        timestamps=ResourceTimestamps(created_at=now, updated_at=now),
        credential_status=IdentityCredentialStatus(configured=True,
                                                   last_rotated_at=now),
        config=DatabricksIdentityConfig(host='https://workspace',
                                        client_id='client-id'),
    )

    serialized = to_json_dict(response)

    assert serialized['id'] == 'identity_123'
    assert serialized['slug'] == 'databricks-production'
    assert 'credentials' not in serialized
    assert 'client_secret' not in str(serialized)


@pytest.mark.parametrize(
    'config',
    [
        DatabricksConnectorResourceConfig(
            warehouse_id='warehouse-id',
            catalog='sales',
            schema_name='europe',
        ),
        SnowflakeConnectorResourceConfig(
            role='SALES_READER',
            warehouse='ANALYTICS',
            database='SALES',
            schema_name='EUROPE',
        ),
        BigQueryConnectorResourceConfig(
            project_id='sales-data',
            dataset_id='europe',
            location='EU',
            billing_project_id='central-billing',
        ),
        S3ConnectorResourceConfig(target=S3RootedConnectorTarget(
            bucket='company-data',
            prefix='raw/events',
            region='us-west-2',
        )),
        S3ConnectorResourceConfig(target=S3AnyURIConnectorTarget()),
        GCSConnectorResourceConfig(
            target=GCSRootedConnectorTarget(bucket='ml-data',
                                            prefix='training'),
            requester_pays_project='central-billing',
        ),
        ADLSConnectorResourceConfig(target=ADLSRootedConnectorTarget(
            storage_account='companylake',
            filesystem='curated',
            prefix='orders',
        )),
    ],
)
def test_connector_create_roundtrip_with_explicit_identity(config):
    payload = CreateConnectorArgs(
        identity_id='identity_123',
        name='sales',
        config=config,
    )

    serialized = to_json_dict(payload)

    assert 'credentials' not in serialized
    assert 'group_id' not in serialized
    assert 'project_id' not in serialized
    assert from_json(serialized, CreateConnectorArgs) == payload


@pytest.mark.parametrize(
    'config',
    [
        GCSConnectorResourceConfig(
            target=GCSRootedConnectorTarget(bucket='company-data',
                                            prefix='raw/events'),
            requester_pays_project='central-billing',
        ),
        GCSConnectorResourceConfig(
            target=GCSAnyURIConnectorTarget(),
            requester_pays_project='central-billing',
        ),
        ADLSConnectorResourceConfig(target=ADLSRootedConnectorTarget(
            storage_account='companylake',
            filesystem='curated',
            prefix='raw/events',
            endpoint_suffix='dfs.core.usgovcloudapi.net',
        )),
        ADLSConnectorResourceConfig(target=ADLSAnyURIConnectorTarget()),
    ],
)
def test_gcs_and_adls_connector_target_unions_roundtrip(config):
    serialized = to_json_dict(config)
    assert serialized['target']['mode'] in {'rooted', 'any_uri'}
    assert from_json(serialized, type(config)) == config


@pytest.mark.parametrize(
    'config_type,payload,unexpected_field',
    [
        (GCSConnectorResourceConfig, {
            'type': 'gcs',
            'target': {
                'mode': 'any_uri',
                'bucket': 'company-data',
            },
        }, 'bucket'),
        (ADLSConnectorResourceConfig, {
            'type': 'adls',
            'target': {
                'mode': 'any_uri',
                'filesystem': 'curated',
            },
        }, 'filesystem'),
    ],
)
def test_any_uri_targets_reject_rooted_coordinates(config_type, payload,
                                                   unexpected_field):
    with pytest.raises(ValidationError, match=unexpected_field):
        from_json(payload, config_type)


@pytest.mark.parametrize(
    'config_type,payload,expected_target',
    [
        (GCSConnectorResourceConfig, {
            'type': 'gcs',
            'bucket': 'company-data',
            'prefix': 'legacy',
        }, GCSRootedConnectorTarget(bucket='company-data', prefix='legacy')),
        (ADLSConnectorResourceConfig, {
            'type': 'adls',
            'storage_account': 'companylake',
            'filesystem': 'curated',
            'prefix': 'legacy',
        },
         ADLSRootedConnectorTarget(
             storage_account='companylake',
             filesystem='curated',
             prefix='legacy',
         )),
    ],
)
def test_legacy_flat_rooted_connector_records_migrate_to_target_union(
        config_type, payload, expected_target):
    original_payload = dict(payload)
    restored = from_json(payload, config_type)
    assert restored.target == expected_target
    assert to_json_dict(restored)['target']['mode'] == 'rooted'
    assert payload == original_payload


def test_connector_create_accepts_omitted_and_null_identity():
    config = S3ConnectorResourceConfig(
        target=S3RootedConnectorTarget(bucket='company-data', prefix='events'))

    omitted = {
        'name': 'events',
        'config': to_json_dict(config),
    }
    explicit_null = {**omitted, 'identity_id': None}

    expected = CreateConnectorArgs(name='events', config=config)
    assert from_json(omitted, CreateConnectorArgs) == expected
    assert from_json(explicit_null, CreateConnectorArgs) == expected
    assert to_json_dict(expected)['identity_id'] is None


def test_connector_response_still_requires_identity():
    now = datetime.now(timezone.utc)
    payload = {
        'id':
        'connector_123',
        'project_id':
        'project_123',
        'name':
        'events',
        'timestamps':
        to_json_dict(ResourceTimestamps(created_at=now, updated_at=now)),
        'config':
        to_json_dict(
            S3ConnectorResourceConfig(target=S3RootedConnectorTarget(
                bucket='company-data', prefix='events'))),
    }

    with pytest.raises(ValidationError):
        from_json(payload, ConnectorResponse)


@pytest.mark.parametrize(
    ('cluster_id', 'warehouse_id'),
    [
        (None, None),
        ('  ', None),
        (None, '\t'),
        ('cluster-id', 'warehouse-id'),
    ],
)
def test_databricks_connector_requires_exactly_one_compute_target(
        cluster_id, warehouse_id):
    with pytest.raises((ValidationError, ValueError), match='Exactly one'):
        DatabricksConnectorResourceConfig(
            catalog='sales',
            schema_name='europe',
            cluster_id=cluster_id,
            warehouse_id=warehouse_id,
        )


@pytest.mark.parametrize(
    ('config_type', 'valid_fields', 'required_field'),
    [
        (SnowflakeConnectorResourceConfig, {
            'warehouse': 'ANALYTICS',
            'database': 'SALES',
            'schema_name': 'EUROPE',
        }, 'warehouse'),
        (SnowflakeConnectorResourceConfig, {
            'warehouse': 'ANALYTICS',
            'database': 'SALES',
            'schema_name': 'EUROPE',
        }, 'database'),
        (SnowflakeConnectorResourceConfig, {
            'warehouse': 'ANALYTICS',
            'database': 'SALES',
            'schema_name': 'EUROPE',
        }, 'schema_name'),
        (DatabricksConnectorResourceConfig, {
            'warehouse_id': 'warehouse-id',
            'catalog': 'main',
            'schema_name': 'sales',
        }, 'catalog'),
        (DatabricksConnectorResourceConfig, {
            'warehouse_id': 'warehouse-id',
            'catalog': 'main',
            'schema_name': 'sales',
        }, 'schema_name'),
        (BigQueryConnectorResourceConfig, {
            'project_id': 'sales-data',
            'dataset_id': 'europe',
            'location': 'EU',
        }, 'project_id'),
        (BigQueryConnectorResourceConfig, {
            'project_id': 'sales-data',
            'dataset_id': 'europe',
            'location': 'EU',
        }, 'dataset_id'),
        (BigQueryConnectorResourceConfig, {
            'project_id': 'sales-data',
            'dataset_id': 'europe',
            'location': 'EU',
        }, 'location'),
        (S3RootedConnectorTarget, {
            'bucket': 'company-data'
        }, 'bucket'),
        (GCSRootedConnectorTarget, {
            'bucket': 'company-data'
        }, 'bucket'),
        (ADLSRootedConnectorTarget, {
            'storage_account': 'companylake',
            'filesystem': 'curated',
        }, 'storage_account'),
        (ADLSRootedConnectorTarget, {
            'storage_account': 'companylake',
            'filesystem': 'curated',
        }, 'filesystem'),
    ],
)
def test_connector_targets_reject_blank_required_provider_fields(
        config_type, valid_fields, required_field):
    fields = {**valid_fields, required_field: '  \t '}

    with pytest.raises((ValidationError, ValueError),
                       match='must not be blank'):
        config_type(**fields)


@pytest.mark.parametrize(
    ('config_type', 'valid_fields', 'required_field'),
    [
        (DatabricksIdentityConfig, {
            'host': 'https://workspace',
            'client_id': 'client-id',
        }, 'host'),
        (DatabricksIdentityConfig, {
            'host': 'https://workspace',
            'client_id': 'client-id',
        }, 'client_id'),
        (SnowflakeIdentityConfig, {
            'account': 'org-account',
            'username': 'service-user',
            'authentication': SnowflakeAuthenticationKind.PASSWORD,
        }, 'account'),
        (SnowflakeIdentityConfig, {
            'account': 'org-account',
            'username': 'service-user',
            'authentication': SnowflakeAuthenticationKind.PASSWORD,
        }, 'username'),
        (GoogleServiceAccountKeyAuthentication, {
            'project_id': 'service-project',
            'client_email': 'service@service-project.iam.gserviceaccount.com',
            'client_id': '123',
            'auth_uri': 'https://example.com/auth',
            'token_uri': 'https://example.com/token',
        }, 'project_id'),
        (GoogleServiceAccountKeyAuthentication, {
            'project_id': 'service-project',
            'client_email': 'service@service-project.iam.gserviceaccount.com',
            'client_id': '123',
            'auth_uri': 'https://example.com/auth',
            'token_uri': 'https://example.com/token',
        }, 'client_email'),
        (GoogleServiceAccountKeyAuthentication, {
            'project_id': 'service-project',
            'client_email': 'service@service-project.iam.gserviceaccount.com',
            'client_id': '123',
            'auth_uri': 'https://example.com/auth',
            'token_uri': 'https://example.com/token',
        }, 'client_id'),
        (GoogleServiceAccountKeyAuthentication, {
            'project_id': 'service-project',
            'client_email': 'service@service-project.iam.gserviceaccount.com',
            'client_id': '123',
            'auth_uri': 'https://example.com/auth',
            'token_uri': 'https://example.com/token',
        }, 'auth_uri'),
        (GoogleServiceAccountKeyAuthentication, {
            'project_id': 'service-project',
            'client_email': 'service@service-project.iam.gserviceaccount.com',
            'client_id': '123',
            'auth_uri': 'https://example.com/auth',
            'token_uri': 'https://example.com/token',
        }, 'token_uri'),
        (GoogleServiceAccountKeyAuthentication, {
            'project_id': 'service-project',
            'client_email': 'service@service-project.iam.gserviceaccount.com',
            'client_id': '123',
            'auth_uri': 'https://example.com/auth',
            'token_uri': 'https://example.com/token',
        }, 'universe_domain'),
        (S3AssumeRoleAuthentication, {
            'partition': 'aws',
            'role_arn': 'arn:aws:iam::123456789012:role/kumo-reader',
        }, 'partition'),
        (S3AssumeRoleAuthentication, {
            'partition': 'aws',
            'role_arn': 'arn:aws:iam::123456789012:role/kumo-reader',
        }, 'role_arn'),
        (GoogleWorkloadIdentityAuthentication, {
            'service_account_email':
            ('service@service-project.iam.gserviceaccount.com'),
        }, 'service_account_email'),
        (ADLSIdentityConfig, {
            'tenant_id': 'tenant-id',
            'client_id': 'client-id',
            'cloud': AzureCloud.PUBLIC,
        }, 'tenant_id'),
        (ADLSIdentityConfig, {
            'tenant_id': 'tenant-id',
            'client_id': 'client-id',
            'cloud': AzureCloud.PUBLIC,
        }, 'client_id'),
    ],
)
def test_identities_reject_blank_required_provider_fields(
        config_type, valid_fields, required_field):
    if config_type is GoogleServiceAccountKeyAuthentication:
        valid_fields = {
            **valid_fields,
            'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
            'token_uri': 'https://oauth2.googleapis.com/token',
        }
    fields = {**valid_fields, required_field: ' \n '}

    with pytest.raises((ValidationError, ValueError),
                       match='must not be blank'):
        config_type(**fields)


@pytest.mark.parametrize(
    ('config_type', 'fields'),
    [
        (S3RootedConnectorTarget, {
            'bucket': 's3://company-data'
        }),
        (S3RootedConnectorTarget, {
            'bucket': 'company-data/events'
        }),
        (GCSRootedConnectorTarget, {
            'bucket': 'gs://company-data'
        }),
        (GCSRootedConnectorTarget, {
            'bucket': 'company-data\\events'
        }),
        (ADLSRootedConnectorTarget, {
            'storage_account': 'companylake.dfs.core.windows.net',
            'filesystem': 'curated',
        }),
        (ADLSRootedConnectorTarget, {
            'storage_account': 'companylake@other',
            'filesystem': 'curated',
        }),
        (ADLSRootedConnectorTarget, {
            'storage_account': 'companylake',
            'filesystem': 'curated/orders',
        }),
        (ADLSRootedConnectorTarget, {
            'storage_account': 'companylake',
            'filesystem': 'curated@other',
        }),
        (ADLSRootedConnectorTarget, {
            'storage_account': 'companylake',
            'filesystem': 'curated',
            'prefix': '../escape',
        }),
        (ADLSRootedConnectorTarget, {
            'storage_account': 'companylake',
            'filesystem': 'curated',
            'prefix': 'orders/./daily',
        }),
        (BigQueryConnectorResourceConfig, {
            'project_id': 'sales.project',
            'dataset_id': 'europe',
            'location': 'EU',
        }),
        (BigQueryConnectorResourceConfig, {
            'project_id': 'sales-project',
            'dataset_id': 'europe`other',
            'location': 'EU',
        }),
        (BigQueryConnectorResourceConfig, {
            'project_id': 'sales-project',
            'dataset_id': 'europe.other',
            'location': 'EU',
        }),
        (BigQueryConnectorResourceConfig, {
            'project_id': 'sales-project',
            'dataset_id': 'europe',
            'location': 'EU',
            'billing_project_id': 'billing.project',
        }),
    ],
)
def test_connector_targets_reject_structural_target_delimiters(
        config_type, fields):
    with pytest.raises((ValidationError, ValueError)):
        config_type(**fields)


def test_s3_identity_partition_and_role_arn_must_match():
    with pytest.raises((ValidationError, ValueError), match='partition'):
        S3AssumeRoleAuthentication(
            partition='aws-us-gov',
            role_arn='arn:aws:iam::123456789012:role/kumo-reader',
        )

    with pytest.raises((ValidationError, ValueError), match='Unsupported'):
        S3AssumeRoleAuthentication(
            partition='aws-invalid',
            role_arn='arn:aws-invalid:iam::123456789012:role/kumo-reader',
        )

    with pytest.raises((ValidationError, ValueError), match='role ARN'):
        S3AssumeRoleAuthentication(
            partition='aws',
            role_arn='arn:aws:s3:::company-data',
        )


def test_optional_target_fields_canonicalize_blank_without_requiring_prefix():
    snowflake = SnowflakeConnectorResourceConfig(warehouse='ANALYTICS',
                                                 database='SALES',
                                                 schema_name='EUROPE',
                                                 role=' ')
    s3 = S3ConnectorResourceConfig(target=S3RootedConnectorTarget(
        bucket='company-data', prefix='', region=' '))
    gcs = GCSConnectorResourceConfig(
        target=GCSRootedConnectorTarget(bucket='company-data', prefix=''),
        requester_pays_project=' ')
    bigquery = BigQueryConnectorResourceConfig(project_id='sales-project',
                                               dataset_id='europe',
                                               location='EU',
                                               billing_project_id=' ')

    assert snowflake.role is None
    assert isinstance(s3.target, S3RootedConnectorTarget)
    assert s3.target.prefix == '' and s3.target.region is None
    assert gcs.prefix == '' and gcs.requester_pays_project is None
    assert bigquery.billing_project_id is None


@pytest.mark.parametrize(
    'config_type,base_fields',
    [
        (GCSRootedConnectorTarget, {
            'bucket': 'company-data',
        }),
        (ADLSRootedConnectorTarget, {
            'storage_account': 'companylake',
            'filesystem': 'curated',
        }),
    ],
)
@pytest.mark.parametrize(
    'prefix',
    [
        '/leading',
        'trailing/',
        'a//b',
        'a/./b',
        'a/../b',
        'a\\b',
        'gs://company-data/root',
        'a/query?value',
        'a/fragment#value',
        'a/%2f/b',
        'a/%2e%2e/b',
        'a/%252f/b',
        ' a/b',
        'a/b ',
        'a/\x00/b',
        'a/\x1f/b',
        'a/\x7f/b',
    ],
)
def test_gcs_and_adls_prefixes_reject_noncanonical_or_ambiguous_values(
        config_type, base_fields, prefix):
    with pytest.raises((ValidationError, ValueError)):
        config_type(**base_fields, prefix=prefix)


@pytest.mark.parametrize(
    'config',
    [
        GCSConnectorResourceConfig(target=GCSRootedConnectorTarget(
            bucket='company-data',
            prefix='tenant/year=2026/month=07/events',
        )),
        ADLSConnectorResourceConfig(target=ADLSRootedConnectorTarget(
            storage_account='companylake',
            filesystem='curated',
            prefix='tenant/year=2026/month=07/events',
        )),
    ],
)
def test_gcs_and_adls_prefixes_preserve_valid_deep_paths(config):
    assert config.prefix == 'tenant/year=2026/month=07/events'


def test_identity_group_replacement_rejects_duplicates():
    with pytest.raises((ValidationError, ValueError), match='duplicates'):
        ReplaceIdentityGroupsArgs(group_ids=['group_a', 'group_a'])


@pytest.mark.parametrize(
    'request_type,payload',
    [
        (StartFileUploadRequest, {
            'source_table_name': 'events',
            'file_type': UploadFileType.CSV,
            'num_parts': 2,
        }),
        (CompleteFileUploadRequest, {
            'source_table_name': 'events',
            'file_type': UploadFileType.PARQUET,
            'temp_upload_path': 's3://data/tmp/events.parquet',
            'upload_id': 'upload-id',
        }),
        (DeleteUploadedFileRequest, {
            'source_table_name': 'events',
            'file_type': UploadFileType.CSV,
        }),
    ],
)
def test_project_upload_requests_require_format_and_reject_unknown_fields(
        request_type, payload):
    request = request_type(**payload)
    assert from_json(to_json_dict(request), request_type) == request

    with pytest.raises(ValidationError, match='upload_metadata'):
        from_json({
            **payload,
            'upload_metadata': 'typo',
        }, request_type)

    without_format = dict(payload)
    without_format.pop('file_type')
    with pytest.raises((ValidationError, TypeError)):
        from_json(without_format, request_type)


def test_project_upload_connector_response_contains_only_safe_reference():
    response = ProjectUploadConnectorResponse(
        connector_id='connector_upload_project_a')

    assert to_json_dict(response) == {
        'connector_id': 'connector_upload_project_a'
    }
    assert not hasattr(response, 'credentials')


def test_snowflake_oauth_token_file_identity_stores_no_credential():
    # The token is a file the container runtime mounts, so the identity holds
    # only where to read it -- the shape a Google workload identity already
    # has, and the reason this kind is usable in-process but not by the
    # platform.
    config = SnowflakeIdentityConfig(
        account='org-account',
        authentication=SnowflakeAuthenticationKind.OAUTH_TOKEN_FILE,
        host='snowflake-host.internal',
        token_path='/snowflake/session/token',
    )
    assert config.username is None
    validate_identity_credentials_match(config, None)

    args = CreateIdentityArgs(display_name='snowflake-spcs', config=config)
    assert from_json(to_json_dict(args), CreateIdentityArgs) == args

    with pytest.raises(ValueError, match='ambient'):
        validate_identity_credentials_match(
            config, SnowflakePasswordCredentials(password=SecretStr('unused')))


@pytest.mark.parametrize('kind', [
    SnowflakeAuthenticationKind.PASSWORD,
    SnowflakeAuthenticationKind.KEY_PAIR,
])
def test_snowflake_credentialed_identities_require_a_username(kind):
    with pytest.raises((ValidationError, ValueError), match='username'):
        SnowflakeIdentityConfig(account='org-account', authentication=kind)


def _key_pair_config_with_host() -> SnowflakeIdentityConfig:
    return SnowflakeIdentityConfig(
        account='org-account',
        username='kumo_service_user',
        authentication=SnowflakeAuthenticationKind.KEY_PAIR,
        host='value',
    )


def _key_pair_config_with_token_path() -> SnowflakeIdentityConfig:
    return SnowflakeIdentityConfig(
        account='org-account',
        username='kumo_service_user',
        authentication=SnowflakeAuthenticationKind.KEY_PAIR,
        token_path='value',
    )


@pytest.mark.parametrize('field,build', [
    ('host', _key_pair_config_with_host),
    ('token_path', _key_pair_config_with_token_path),
])
def test_snowflake_token_file_fields_are_refused_by_other_kinds(field, build):
    # Accepting them would let an identity carry a setting that silently does
    # nothing, which reads as configured but is not in force. Each case names
    # its field rather than unpacking one, which no type checker can follow.
    with pytest.raises((ValidationError, ValueError), match=field):
        build()


def test_snowflake_oauth_token_file_identity_names_no_username():
    with pytest.raises((ValidationError, ValueError), match='username'):
        SnowflakeIdentityConfig(
            account='org-account',
            username='kumo_service_user',
            authentication=SnowflakeAuthenticationKind.OAUTH_TOKEN_FILE,
        )
