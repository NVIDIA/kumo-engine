# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from functools import partial
from http import HTTPStatus
from typing import Any, cast
from unittest.mock import MagicMock, call, patch

import pytest
from kumoapi.data_source import (
    ADLSAnyURIConnectorTarget,
    ADLSConnectorResourceConfig,
    ADLSIdentityConfig,
    ADLSRootedConnectorTarget,
    AzureCloud,
    BigQueryConnectorResourceConfig,
    BigQueryIdentityConfig,
    ConnectorResponse,
    ConnectorType,
    CreateConnectorArgs,
    DatabricksConnectorResourceConfig,
    DatabricksIdentityConfig,
    GCSAnyURIConnectorTarget,
    GCSConnectorResourceConfig,
    GCSIdentityConfig,
    GCSRootedConnectorTarget,
    GoogleServiceAccountKeyAuthentication,
    GoogleWorkloadIdentityAuthentication,
    IdentityCredentialStatus,
    IdentityResponse,
    ResourceTimestamps,
    S3AccessKeyAuthentication,
    S3AddressingStyle,
    S3AnyURIConnectorTarget,
    S3AssumeRoleAuthentication,
    S3ConnectionConfig,
    S3ConnectorResourceConfig,
    S3IdentityConfig,
    S3RootedConnectorTarget,
    SnowflakeAuthenticationKind,
    SnowflakeConnectorResourceConfig,
    SnowflakeIdentityConfig,
)
from kumoapi.source_table import (
    ADLSSourceTable,
    ADLSURISource,
    FileType,
    GCSSourceTable,
    GCSURISource,
    S3RelativeSource,
    S3SourceTable,
    SourceTableConfigResponse,
)
from pydantic import ValidationError

from kumoai_studio.client.connector import ConnectorAPI
from kumoai_studio.connector import (
    ADLSConnector,
    BigQueryConnector,
    DatabricksConnector,
    GCSConnector,
    S3Connector,
    SnowflakeConnector,
    SourceTable,
)
from kumoai_studio.connector.adls_connector import ADLSURI
from kumoai_studio.connector.gcs_connector import GCSURI
from kumoai_studio.connector.s3_connector import S3URI
from kumoai_studio.exceptions import HTTPException
from kumoai_studio.graph import Table
from kumoai_studio.identity import Identity

TIMESTAMPS = ResourceTimestamps(
    created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
)


def _s3_rooted_config(
    bucket: str = 'bucket',
    prefix: str = '',
    region: str | None = None,
) -> S3ConnectorResourceConfig:
    return S3ConnectorResourceConfig(target=S3RootedConnectorTarget(
        bucket=bucket, prefix=prefix, region=region))


def _gcs_rooted_config(
    bucket: str = 'bucket',
    prefix: str = '',
    requester_pays_project: str | None = None,
) -> GCSConnectorResourceConfig:
    return GCSConnectorResourceConfig(
        target=GCSRootedConnectorTarget(bucket=bucket, prefix=prefix),
        requester_pays_project=requester_pays_project,
    )


def _adls_rooted_config(
    storage_account: str = 'lake',
    filesystem: str = 'curated',
    prefix: str = '',
    endpoint_suffix: str = 'dfs.core.windows.net',
) -> ADLSConnectorResourceConfig:
    return ADLSConnectorResourceConfig(target=ADLSRootedConnectorTarget(
        storage_account=storage_account,
        filesystem=filesystem,
        prefix=prefix,
        endpoint_suffix=endpoint_suffix,
    ))


def _google_identity_config(provider: str, mode: str):
    authentication = (
        GoogleWorkloadIdentityAuthentication(service_account_email=(
            f'{provider}@identity-project.iam.gserviceaccount.com')) if mode
        == 'workload_identity' else GoogleServiceAccountKeyAuthentication(
            project_id='identity-project',
            client_email=(
                f'{provider}@identity-project.iam.gserviceaccount.com'),
            client_id=f'{provider}-client-id',
            auth_uri='https://accounts.google.com/o/oauth2/auth',
            token_uri='https://oauth2.googleapis.com/token',
            universe_domain='googleapis.com',
        ))
    if provider == 'bigquery':
        return BigQueryIdentityConfig(authentication=authentication)
    return GCSIdentityConfig(authentication=authentication)


def _google_identity(provider: str, mode: str) -> Identity:
    slug_mode = mode.replace('_', '-')
    response = IdentityResponse(
        id=f'identity-{provider}-{mode}',
        slug=f'{provider}-{slug_mode}',
        display_name=f'{provider} {mode}',
        description='',
        timestamps=TIMESTAMPS,
        credential_status=IdentityCredentialStatus(configured=True),
        config=_google_identity_config(provider, mode),
    )
    return Identity._from_response(response)


def _factories():
    return [
        (
            lambda: DatabricksConnector(
                name='databricks',
                identity='identity-databricks',
                warehouse_id='warehouse',
                catalog='main',
                schema_name='sales',
            ),
            DatabricksConnectorResourceConfig(
                warehouse_id='warehouse',
                catalog='main',
                schema_name='sales',
            ),
        ),
        (
            lambda: SnowflakeConnector(
                name='snowflake',
                identity='identity-snowflake',
                role='KUMO_READER',
                warehouse='COMPUTE_WH',
                database='PROD',
                schema_name='SALES',
            ),
            SnowflakeConnectorResourceConfig(
                role='KUMO_READER',
                warehouse='COMPUTE_WH',
                database='PROD',
                schema_name='SALES',
            ),
        ),
        (
            lambda: BigQueryConnector(
                name='bigquery',
                identity='identity-bigquery',
                project_id='data-project',
                dataset_id='sales',
                location='US',
            ),
            BigQueryConnectorResourceConfig(
                project_id='data-project',
                dataset_id='sales',
                location='US',
            ),
        ),
        (
            lambda: S3Connector(
                name='s3',
                identity='identity-s3',
                root_dir='s3://bucket/raw/events/',
                region='us-west-2',
            ),
            _s3_rooted_config(prefix='raw/events', region='us-west-2'),
        ),
        (
            lambda: GCSConnector(
                name='gcs',
                identity='identity-gcs',
                root_dir='gs://bucket/training/',
                requester_pays_project='billing',
            ),
            _gcs_rooted_config(prefix='training',
                               requester_pays_project='billing'),
        ),
        (
            lambda: ADLSConnector(
                name='adls',
                identity='identity-adls',
                root_dir=('abfss://curated@lake.dfs.core.windows.net/orders/'),
            ),
            _adls_rooted_config(prefix='orders'),
        ),
    ]


def _automatic_factories():
    return [
        (
            lambda: DatabricksConnector(
                name='databricks',
                warehouse_id='warehouse',
                catalog='main',
                schema_name='sales',
            ),
            DatabricksConnectorResourceConfig(warehouse_id='warehouse',
                                              catalog='main',
                                              schema_name='sales'),
        ),
        (
            lambda: SnowflakeConnector(
                name='snowflake',
                role='KUMO_READER',
                warehouse='COMPUTE_WH',
                database='PROD',
                schema_name='SALES',
            ),
            SnowflakeConnectorResourceConfig(role='KUMO_READER',
                                             warehouse='COMPUTE_WH',
                                             database='PROD',
                                             schema_name='SALES'),
        ),
        (
            lambda: BigQueryConnector(
                name='bigquery',
                project_id='data-project',
                dataset_id='sales',
                location='US',
            ),
            BigQueryConnectorResourceConfig(project_id='data-project',
                                            dataset_id='sales', location='US'),
        ),
        (
            lambda: S3Connector(name='s3', root_dir='s3://bucket/raw/events/',
                                region='us-west-2'),
            _s3_rooted_config(prefix='raw/events', region='us-west-2'),
        ),
        (
            lambda: GCSConnector(name='gcs', root_dir='gs://bucket/training/',
                                 requester_pays_project='billing'),
            _gcs_rooted_config(prefix='training',
                               requester_pays_project='billing'),
        ),
        (
            lambda: ADLSConnector(
                name='adls',
                root_dir=('abfss://curated@lake.dfs.core.windows.net/orders/'),
            ),
            _adls_rooted_config(prefix='orders'),
        ),
    ]


def _project_override_factories():
    return [
        (
            lambda: DatabricksConnector(
                name='databricks', warehouse_id='warehouse', catalog='main',
                schema_name='sales', project_id='project-b'),
            partial(DatabricksConnector.get_by_name, 'databricks',
                    project_id='project-b'),
            DatabricksConnectorResourceConfig(warehouse_id='warehouse',
                                              catalog='main',
                                              schema_name='sales'),
        ),
        (
            lambda: SnowflakeConnector(
                name='snowflake', warehouse='COMPUTE_WH', database='PROD',
                schema_name='SALES', project_id='project-b'),
            partial(SnowflakeConnector.get_by_name, 'snowflake',
                    project_id='project-b'),
            SnowflakeConnectorResourceConfig(warehouse='COMPUTE_WH',
                                             database='PROD',
                                             schema_name='SALES'),
        ),
        (
            lambda: BigQueryConnector(
                name='bigquery', project_id='data-project', dataset_id='sales',
                location='US', kumo_project_id='project-b'),
            partial(BigQueryConnector.get_by_name, 'bigquery',
                    kumo_project_id='project-b'),
            BigQueryConnectorResourceConfig(project_id='data-project',
                                            dataset_id='sales', location='US'),
        ),
        (
            lambda: S3Connector(name='s3', root_dir='s3://bucket/events/',
                                project_id='project-b'),
            lambda: S3Connector.get_by_name('s3', project_id='project-b'),
            _s3_rooted_config(prefix='events'),
        ),
        (
            lambda: GCSConnector(name='gcs', root_dir='gs://bucket/training/',
                                 project_id='project-b'),
            lambda: GCSConnector.get_by_name('gcs', project_id='project-b'),
            _gcs_rooted_config(prefix='training'),
        ),
        (
            lambda: ADLSConnector(
                name='adls', root_dir=(
                    'abfss://curated@lake.dfs.core.windows.net/orders/'),
                project_id='project-b'),
            lambda: ADLSConnector.get_by_name('adls', project_id='project-b'),
            _adls_rooted_config(prefix='orders'),
        ),
    ]


@pytest.mark.parametrize('factory,expected_config', _factories())
def test_all_six_connectors_send_identity_and_target_only(
        factory, expected_config):
    response = ConnectorResponse(
        id='connector-id',
        project_id='project-id',
        identity_id=f'identity-{expected_config.type.value}',
        name=expected_config.type.value,
        timestamps=TIMESTAMPS,
        config=expected_config,
    )
    state = MagicMock()
    state.client.connector_api.create_if_not_exist.return_value = response
    with patch('kumoai_studio.global_state', state), \
            patch('kumoai_studio._require_project'):
        connector = factory()

    args = state.client.connector_api.create_if_not_exist.call_args.args[0]
    assert args.identity_id == f'identity-{expected_config.type.value}'
    assert args.config == expected_config
    assert args.name == expected_config.type.value
    assert not hasattr(args, 'group_id')
    assert not hasattr(args, 'credentials')
    assert connector.id == 'connector-id'


@pytest.mark.parametrize('factory,expected_config', _automatic_factories())
def test_all_six_connectors_delegate_omitted_identity_to_server(
        factory, expected_config):
    selected_identity_id = f'identity-{expected_config.type.value}'
    response = ConnectorResponse(
        id='connector-id',
        project_id='project-id',
        identity_id=selected_identity_id,
        name=expected_config.type.value,
        timestamps=TIMESTAMPS,
        config=expected_config,
    )
    state = MagicMock()
    state.client.connector_api.create_if_not_exist.return_value = response
    with patch('kumoai_studio.global_state', state), \
            patch('kumoai_studio._require_project'), \
            patch('kumoai_studio.list_identities') as list_identities:
        connector = factory()

    args = state.client.connector_api.create_if_not_exist.call_args.args[0]
    assert args.identity_id is None
    assert args.config == expected_config
    assert connector.identity_id == selected_identity_id
    list_identities.assert_not_called()


@pytest.mark.parametrize('factory,getter,config',
                         _project_override_factories())
def test_all_six_connectors_keep_explicit_project_after_global_switch(
        factory, getter, config):
    response = ConnectorResponse(id='connector-id', project_id='project-b',
                                 identity_id=f'identity-{config.type.value}',
                                 name=config.type.value, timestamps=TIMESTAMPS,
                                 config=config)
    state = MagicMock()
    state._project_id = 'project-a'
    state.client.connector_api.create_if_not_exist.return_value = response
    state.client.connector_api.get_by_name.return_value = response

    with patch('kumoai_studio.global_state', state), \
            patch('kumoai_studio.global_state', state):
        connector = factory()
        connector.table_names()
        connector._delete_connector()
        loaded = getter()

    state.client.connector_api.create_if_not_exist.assert_called_once()
    assert state.client.connector_api.create_if_not_exist.call_args.kwargs == {
        'project_id': 'project-b'
    }
    state.client.source_table_api.list_tables.assert_called_once_with(
        'connector-id', project_id='project-b')
    state.client.connector_api.delete_if_exists.assert_called_once_with(
        'connector-id', project_id='project-b')
    state.client.connector_api.get_by_name.assert_called_once_with(
        config.type.value, project_id='project-b')
    assert connector.kumo_project_id == 'project-b'
    assert loaded.kumo_project_id == 'project-b'


@pytest.mark.parametrize(
    ('connector_type', 'kwargs'),
    [
        (DatabricksConnector, {
            'name': 'databricks',
            'warehouse_id': 'warehouse',
            'catalog': ' ',
            'schema_name': 'sales',
        }),
        (DatabricksConnector, {
            'name': 'databricks',
            'warehouse_id': 'warehouse',
            'catalog': 'main',
            'schema_name': '\t',
        }),
        (SnowflakeConnector, {
            'name': 'snowflake',
            'warehouse': ' ',
            'database': 'PROD',
            'schema_name': 'SALES',
        }),
        (SnowflakeConnector, {
            'name': 'snowflake',
            'warehouse': 'WH',
            'database': ' ',
            'schema_name': 'SALES',
        }),
        (SnowflakeConnector, {
            'name': 'snowflake',
            'warehouse': 'WH',
            'database': 'PROD',
            'schema_name': '\n',
        }),
        (BigQueryConnector, {
            'name': 'bigquery',
            'project_id': ' ',
            'dataset_id': 'sales',
        }),
        (BigQueryConnector, {
            'name': 'bigquery',
            'project_id': 'data-project',
            'dataset_id': '\t',
        }),
        (BigQueryConnector, {
            'name': 'bigquery',
            'project_id': 'data-project',
            'dataset_id': 'sales',
            'location': ' ',
        }),
        (S3Connector, {
            'name': 's3',
            'root_dir': 's3:///events'
        }),
        (GCSConnector, {
            'name': 'gcs',
            'root_dir': 'gs:///events'
        }),
        (ADLSConnector, {
            'name': 'adls',
            'root_dir': 'abfss://@lake.dfs.core.windows.net/events',
        }),
        (ADLSConnector, {
            'name': 'adls',
            'root_dir': 'abfss://curated@.dfs.core.windows.net/events',
        }),
    ],
)
def test_sdk_connectors_reject_blank_exact_targets_before_request(
        connector_type, kwargs):
    state = MagicMock()
    with patch('kumoai_studio.global_state', state), \
            patch('kumoai_studio._require_project'):
        with pytest.raises((ValidationError, ValueError)):
            connector_type(**kwargs)

    state.client.connector_api.create_if_not_exist.assert_not_called()


@pytest.mark.parametrize(
    ('config_type', 'kwargs'),
    [
        (DatabricksIdentityConfig, {
            'host': ' ',
            'client_id': 'client-id',
        }),
        (SnowflakeIdentityConfig, {
            'account': ' ',
            'username': 'service-user',
            'authentication': SnowflakeAuthenticationKind.PASSWORD,
        }),
        (BigQueryIdentityConfig, {
            'authentication': {
                'mode': 'service_account_key',
                'project_id': ' ',
                'client_email':
                'service@identity-project.iam.gserviceaccount.com',
                'client_id': '123',
                'auth_uri': 'https://example.com/auth',
                'token_uri': 'https://example.com/token',
            },
        }),
        (S3IdentityConfig, {
            'authentication': {
                'mode': 'assume_role',
                'partition': 'aws',
                'role_arn': ' ',
            }
        }),
        (GCSIdentityConfig, {
            'authentication': {
                'mode': 'workload_identity',
                'service_account_email': ' ',
            }
        }),
        (ADLSIdentityConfig, {
            'tenant_id': ' ',
            'client_id': 'client-id',
            'cloud': AzureCloud.PUBLIC,
        }),
    ],
)
def test_sdk_identity_values_reject_blank_provider_identifiers(
        config_type, kwargs):
    with pytest.raises((ValidationError, ValueError),
                       match='must not be blank'):
        config_type(**kwargs)


@pytest.mark.parametrize(
    'uri,config',
    [
        ('s3://bucket/a/b/', _s3_rooted_config(prefix='a/b')),
        ('gs://bucket/a/b/', _gcs_rooted_config(prefix='a/b')),
        (
            'abfss://raw@lake.dfs.core.windows.net/a/b/',
            _adls_rooted_config(filesystem='raw', prefix='a/b'),
        ),
    ],
)
def test_equivalent_object_store_uris_have_canonical_config(uri, config):
    state = MagicMock()
    state.client.connector_api.create_if_not_exist.return_value = (
        ConnectorResponse(
            id='connector-id',
            project_id='project-id',
            identity_id=f'identity-{config.type.value}',
            name='files',
            timestamps=TIMESTAMPS,
            config=config,
        ))
    classes = {
        ConnectorType.S3: S3Connector,
        ConnectorType.GCS: GCSConnector,
        ConnectorType.ADLS: ADLSConnector,
    }
    with patch('kumoai_studio.global_state', state), \
            patch('kumoai_studio._require_project'):
        classes[config.type](name='files',
                             identity=f'identity-{config.type.value}',
                             root_dir=uri)

    sent = state.client.connector_api.create_if_not_exist.call_args.args[0]
    assert sent.config == config


def _backend_of(state):
    """`SourceTable._from_api_table_definition` resolves a connector through
    the active backend rather than the REST client, so a definition can be
    rebuilt where there is no service; on the cloud backend the two are the
    same object.
    """
    backend = MagicMock()
    backend.connector = state.client.connector_api
    return backend


def test_identity_object_type_must_match_connector():
    response = IdentityResponse(
        id='identity-s3',
        slug='aws-role',
        display_name='AWS role',
        description='',
        timestamps=TIMESTAMPS,
        credential_status=IdentityCredentialStatus(configured=True),
        config=S3IdentityConfig(authentication=S3AssumeRoleAuthentication(
            partition='aws', role_arn='arn:aws:iam::123456789012:role/kumo')),
    )
    identity = Identity._from_response(response)

    with patch('kumoai_studio._require_project'):
        with pytest.raises(ValueError, match='not.*snowflake'):
            SnowflakeConnector(
                name='sales',
                identity=identity,
                warehouse='WH',
                database='DB',
                schema_name='SCHEMA',
            )


def test_minio_identity_connection_is_not_duplicated_in_connector_request():
    connection = S3ConnectionConfig(
        endpoint_url='https://minio.example.com:9000',
        region='us-east-1',
        addressing_style=S3AddressingStyle.PATH,
    )
    identity = Identity._from_response(
        IdentityResponse(
            id='identity-minio',
            slug='private-minio',
            display_name='Private MinIO',
            description='',
            timestamps=TIMESTAMPS,
            credential_status=IdentityCredentialStatus(configured=True),
            config=S3IdentityConfig(
                authentication=S3AccessKeyAuthentication(),
                connection=connection,
            ),
        ))
    target = _s3_rooted_config(bucket='customer-data', prefix='events')
    response = ConnectorResponse(
        id='connector-minio',
        project_id='project-id',
        identity_id=identity.id,
        name='events',
        timestamps=TIMESTAMPS,
        config=target,
    )
    state = MagicMock()
    state.client.connector_api.create_if_not_exist.return_value = response

    with patch('kumoai_studio.connector.identity_backed.global_state', state), \
            patch('kumoai_studio.global_state', state), \
            patch('kumoai_studio.connector.identity_backed._project_scope'):
        # `_bind_connector` guards on `_project_scope`, not `_require_project`
        # directly: only a project-scoped backend needs a project, so the
        # check now routes through the backend. Patch the guard the caller
        # actually reaches.
        connector = S3Connector(
            name='events',
            root_dir='s3://customer-data/events',
            identity=identity,
        )

    args = state.client.connector_api.create_if_not_exist.call_args.args[0]
    assert args.identity_id == 'identity-minio'
    assert args.config == target
    assert not hasattr(args.config, 'connection')
    assert isinstance(identity.config, S3IdentityConfig)
    assert identity.config.connection == connection
    assert connector.identity_id == 'identity-minio'


@pytest.mark.parametrize('provider', ['bigquery', 'gcs'])
@pytest.mark.parametrize('mode', ['service_account_key', 'workload_identity'])
def test_google_connector_accepts_credential_free_explicit_identity_for_mode(
        provider, mode):
    identity = _google_identity(provider, mode)
    target = (BigQueryConnectorResourceConfig(
        project_id='data-project', dataset_id='events', location='US')
              if provider == 'bigquery' else _gcs_rooted_config(
                  bucket='data-bucket', prefix='events'))
    response = ConnectorResponse(
        id=f'connector-{provider}-{mode}',
        project_id='project-id',
        identity_id=identity.id,
        name=provider,
        timestamps=TIMESTAMPS,
        config=target,
    )
    state = MagicMock()
    state.client.connector_api.create_if_not_exist.return_value = response

    with patch('kumoai_studio.global_state', state), \
            patch('kumoai_studio._require_project'):
        connector = (BigQueryConnector(
            name=provider,
            project_id='data-project',
            dataset_id='events',
            identity=identity,
        ) if provider == 'bigquery' else GCSConnector(
            name=provider,
            root_dir='gs://data-bucket/events/',
            identity=identity,
        ))

    args = state.client.connector_api.create_if_not_exist.call_args.args[0]
    assert args.identity_id == identity.id
    assert args.config == target
    assert not hasattr(args, 'credentials')
    assert connector.identity_id == identity.id
    assert isinstance(identity.config,
                      (BigQueryIdentityConfig, GCSIdentityConfig))
    assert isinstance(identity.config.authentication,
                      (GoogleServiceAccountKeyAuthentication,
                       GoogleWorkloadIdentityAuthentication))
    assert identity.config.authentication.mode == mode


@pytest.mark.parametrize('provider', ['bigquery', 'gcs'])
@pytest.mark.parametrize('mode', ['service_account_key', 'workload_identity'])
def test_google_connector_automatic_selection_persists_server_mode_identity(
        provider, mode):
    identity_id = f'identity-{provider}-{mode}'
    target = (BigQueryConnectorResourceConfig(
        project_id='data-project', dataset_id='events', location='US')
              if provider == 'bigquery' else _gcs_rooted_config(
                  bucket='data-bucket', prefix='events'))
    response = ConnectorResponse(
        id=f'connector-{provider}-{mode}',
        project_id='project-id',
        identity_id=identity_id,
        name=provider,
        timestamps=TIMESTAMPS,
        config=target,
    )
    state = MagicMock()
    state.client.connector_api.create_if_not_exist.return_value = response

    with patch('kumoai_studio.global_state', state), \
            patch('kumoai_studio._require_project'), \
            patch('kumoai_studio.list_identities') as list_identities:
        connector = (BigQueryConnector(
            name=provider,
            project_id='data-project',
            dataset_id='events',
        ) if provider == 'bigquery' else GCSConnector(
            name=provider,
            root_dir='gs://data-bucket/events/',
        ))

    args = state.client.connector_api.create_if_not_exist.call_args.args[0]
    assert args.identity_id is None
    assert connector.identity_id == identity_id
    list_identities.assert_not_called()


@pytest.mark.parametrize(
    ('identity_provider', 'connector_provider'),
    [('bigquery', 'gcs'), ('gcs', 'bigquery')],
)
@pytest.mark.parametrize('mode', ['service_account_key', 'workload_identity'])
def test_google_connector_rejects_cross_provider_identity(
        identity_provider, connector_provider, mode):
    identity = _google_identity(identity_provider, mode)

    with patch('kumoai_studio._require_project'):
        with pytest.raises(ValueError, match='not'):
            if connector_provider == 'bigquery':
                BigQueryConnector(
                    name='events',
                    project_id='data-project',
                    dataset_id='events',
                    identity=identity,
                )
            else:
                GCSConnector(
                    name='events',
                    root_dir='gs://data-bucket/events/',
                    identity=identity,
                )


def test_omitted_identity_uses_only_server_selected_response_identity():
    config = _s3_rooted_config(prefix='events')
    response = ConnectorResponse(
        id='connector-id',
        project_id='project-id',
        identity_id='identity-production',
        name='events',
        timestamps=TIMESTAMPS,
        config=config,
    )
    state = MagicMock()
    state.client.connector_api.create_if_not_exist.return_value = response
    with patch('kumoai_studio.global_state', state), \
            patch('kumoai_studio._require_project'), \
            patch('kumoai_studio.list_identities') as list_identities:
        connector = S3Connector(name='events', root_dir='s3://bucket/events/')

    args = state.client.connector_api.create_if_not_exist.call_args.args[0]
    assert args.identity_id is None
    assert connector.identity_id == 'identity-production'
    list_identities.assert_not_called()


def test_create_if_not_exist_posts_directly_without_list_race():
    config = _s3_rooted_config(prefix='events')
    response = ConnectorResponse(id='connector-id', project_id='project-id',
                                 identity_id='identity-persisted',
                                 name='events', timestamps=TIMESTAMPS,
                                 config=config)
    api = ConnectorAPI(MagicMock())
    args = CreateConnectorArgs(name='events', config=config)
    with patch.object(api, 'get_by_name') as get_by_name, \
            patch.object(api, 'list') as list_connectors, \
            patch.object(api, 'create', return_value=response) as create:
        result = api.create_if_not_exist(args, project_id='project-id')

    assert result is response
    create.assert_called_once_with(args, project_id='project-id')
    get_by_name.assert_not_called()
    list_connectors.assert_not_called()


def test_create_if_not_exist_propagates_server_conflict():
    config = _s3_rooted_config(prefix='events')
    api = ConnectorAPI(MagicMock())
    conflict = HTTPException(HTTPStatus.CONFLICT, 'different target')
    with patch.object(api, 'create', side_effect=conflict):
        with pytest.raises(HTTPException) as exc_info:
            api.create_if_not_exist(
                CreateConnectorArgs(name='events', config=config,
                                    identity_id='identity-other'),
                project_id='project-id',
            )

    assert exc_info.value is conflict


@pytest.mark.parametrize(
    'parser,valid,invalid',
    [
        (S3URI, 's3://bucket/prefix', 'https://bucket/prefix'),
        (GCSURI, 'gs://bucket/prefix', 's3://bucket/prefix'),
        (
            ADLSURI,
            'abfss://raw@lake.dfs.core.windows.net/prefix',
            'abfs://raw@lake.dfs.core.windows.net/prefix',
        ),
    ],
)
def test_object_store_uri_schemes_are_strict(parser, valid, invalid):
    assert parser(valid).is_valid
    assert not parser(invalid).is_valid


@pytest.mark.parametrize(
    ('parser', 'path', 'expected'),
    [
        (S3URI, '2026/07/events.parquet', '2026/07/events.parquet'),
        (GCSURI, '2026/07/events.CSV/', '2026/07/events.CSV/'),
        (ADLSURI, 'datasets/customer%2Devents/', 'datasets/customer-events/'),
    ],
)
def test_object_store_relative_table_name_is_canonical(parser, path, expected):
    assert parser.normalize_relative_table_name(path) == expected


@pytest.mark.parametrize(
    ('parser', 'path'),
    [
        (S3URI, ''),
        (S3URI, '/absolute/table.parquet'),
        (GCSURI, 'nested//table.parquet'),
        (GCSURI, 'nested/table.parquet//'),
        (GCSURI, 'nested/../table.parquet'),
        (GCSURI, 'nested%5Ctable.parquet'),
        (ADLSURI, 'nested/%2E%2E/table.parquet'),
        (ADLSURI, 'nested/%00table.parquet'),
        (ADLSURI, 'nested/table.parquet?version=1'),
        (S3URI, 'nested/table.parquet#fragment'),
        (S3URI, 'nested\\table.parquet'),
        (S3URI, 'nested%5Ctable.parquet'),
        (S3URI, 'nested/%00table.parquet'),
        (S3URI, 'foo:bar.parquet'),
        (S3URI, 'datasets%2F2026/events.parquet'),
        (S3URI, 'datasets%252F2026/events.parquet'),
    ],
)
def test_object_store_relative_table_name_rejects_unsafe_paths(parser, path):
    with pytest.raises(ValueError):
        parser.normalize_relative_table_name(path)


@pytest.mark.parametrize(
    ('parser', 'root', 'uri', 'expected'),
    [
        (
            S3URI,
            's3://bucket/project/root',
            's3://bucket/project/root/2026/07/events.parquet',
            '2026/07/events.parquet',
        ),
        (
            GCSURI,
            'gs://bucket/project/root/',
            'gs://bucket/project/root/datasets/events/',
            'datasets/events/',
        ),
        (
            ADLSURI,
            'abfss://raw@lake.dfs.core.windows.net/project/root',
            ('abfss://raw@lake.dfs.core.windows.net/project/root/'
             'nested/2026/events.csv'),
            'nested/2026/events.csv',
        ),
    ],
)
def test_full_uri_becomes_nested_connector_relative_table(
        parser, root, uri, expected):
    assert parser(uri).relative_table_name(root) == expected


@pytest.mark.parametrize('parser', [S3URI, GCSURI, ADLSURI])
@pytest.mark.parametrize('suffix', ['csv', 'parquet'])
def test_object_store_suffix_file_and_directory_coordinates_stay_distinct(
        parser, suffix):
    file_name = f'deep/2026/dataset.{suffix}'
    directory_name = f'{file_name}/'

    assert parser.normalize_relative_table_name(file_name) == file_name
    assert parser.normalize_relative_table_name(
        directory_name) == directory_name


@pytest.mark.parametrize('parser', [S3URI, GCSURI, ADLSURI])
@pytest.mark.parametrize(
    'path',
    [
        'release.v1/',
        'deep/archive.v2/2026/events/',
    ],
)
def test_object_store_dotted_and_deep_directory_slashes_are_preserved(
        parser, path):
    assert parser.normalize_relative_table_name(path) == path


@pytest.mark.parametrize(
    ('parser', 'root', 'file_uri', 'directory_uri', 'expected_file',
     'expected_directory'),
    [
        (
            S3URI,
            's3://bucket/root',
            's3://bucket/root/dataset.csv',
            's3://bucket/root/dataset.csv/',
            'dataset.csv',
            'dataset.csv/',
        ),
        (
            GCSURI,
            'gs://bucket/root',
            'gs://bucket/root/deep/dataset.parquet',
            'gs://bucket/root/deep/dataset.parquet/',
            'deep/dataset.parquet',
            'deep/dataset.parquet/',
        ),
        (
            ADLSURI,
            'abfss://raw@lake.dfs.core.windows.net/root',
            'abfss://raw@lake.dfs.core.windows.net/root/release.v1',
            'abfss://raw@lake.dfs.core.windows.net/root/release.v1/',
            'release.v1',
            'release.v1/',
        ),
    ],
)
def test_rooted_full_uris_preserve_file_vs_directory_intent(
        parser, root, file_uri, directory_uri, expected_file,
        expected_directory):
    assert parser(file_uri).relative_table_name(root) == expected_file
    assert parser(directory_uri).relative_table_name(
        root) == expected_directory


@pytest.mark.parametrize(
    ('parser', 'root', 'uri'),
    [
        (S3URI, 's3://bucket/root', 's3://bucket/root2/table.parquet'),
        (S3URI, 's3://bucket/root', 's3://other/root/table.parquet'),
        (S3URI, 's3://bucket/root', 'gs://bucket/root/table.parquet'),
        (GCSURI, 'gs://bucket/root', 'gs://bucket/root'),
        (GCSURI, 'gs://bucket/root', 'gs://bucket/root/../table.parquet'),
        (
            ADLSURI,
            'abfss://raw@lake.dfs.core.windows.net/root',
            'abfss://other@lake.dfs.core.windows.net/root/table.parquet',
        ),
        (
            ADLSURI,
            'abfss://raw@lake.dfs.core.windows.net/root',
            'abfss://raw@other.dfs.core.windows.net/root/table.parquet',
        ),
        (S3URI, 's3://bucket/root',
         's3://bucket/root/table.parquet?version=1'),
        (S3URI, 's3://bucket/root', 's3://bucket/root/table.parquet#fragment'),
        (S3URI, 's3://bucket/root', 's3://bucket/root/table.parquet//'),
        (GCSURI, 'gs://bucket/root',
         'gs://bucket/root/events%2Fsecret.parquet'),
        (
            ADLSURI,
            'abfss://raw@lake.dfs.core.windows.net/root',
            ('abfss://raw@lake.dfs.core.windows.net/root/'
             'events%5Csecret.parquet'),
        ),
    ],
)
def test_full_uri_rejects_outside_or_unsafe_target(parser, root, uri):
    with pytest.raises(ValueError):
        parser(uri).relative_table_name(root)


def _unbound_s3_connector() -> S3Connector:
    connector = object.__new__(S3Connector)
    connector._id = 'connector-s3'
    connector._name = 'project-data'
    connector._kumo_project_id = 'project-id'
    connector.root_dir = 's3://bucket/root'
    connector.target = S3RootedConnectorTarget(bucket='bucket', prefix='root')
    connector.any_uri = False
    connector.bucket = 'bucket'
    connector.prefix = 'root'
    connector.region = None
    connector.identity_id = 'identity-s3'
    connector.description = ''
    return connector


def _unbound_gcs_connector() -> GCSConnector:
    connector = object.__new__(GCSConnector)
    connector._id = 'connector-gcs'
    connector._name = 'project-gcs-data'
    connector._kumo_project_id = 'project-id'
    connector.root_dir = 'gs://bucket/root'
    connector.identity_id = 'identity-gcs'
    connector.description = ''
    return connector


def _unbound_adls_connector() -> ADLSConnector:
    connector = object.__new__(ADLSConnector)
    connector._id = 'connector-adls'
    connector._name = 'project-adls-data'
    connector._kumo_project_id = 'project-id'
    connector.root_dir = ('abfss://raw@lake.dfs.core.windows.net/root')
    connector.identity_id = 'identity-adls'
    connector.description = ''
    return connector


@pytest.mark.parametrize(
    ('factory', 'relative_name', 'full_uri', 'connector_id'),
    [
        (
            _unbound_s3_connector,
            'deep/dataset.csv/',
            's3://bucket/root/deep/dataset.parquet/',
            'connector-s3',
        ),
        (
            _unbound_gcs_connector,
            'release.v1/',
            'gs://bucket/root/deep/dataset.csv/',
            'connector-gcs',
        ),
        (
            _unbound_adls_connector,
            'deep/dataset.parquet/',
            ('abfss://raw@lake.dfs.core.windows.net/root/'
             'deep/release.v2/'),
            'connector-adls',
        ),
    ],
)
def test_object_store_membership_and_table_preserve_directory_coordinates(
        factory, relative_name, full_uri, connector_id):
    connector = factory()
    state = MagicMock()

    with patch('kumoai_studio.global_state', state):
        assert relative_name in connector
        relative_table = connector.table(relative_name)
        full_uri_table = connector.table(full_uri)

    expected_full_name = connector._normalize_table_reference(full_uri)
    assert relative_table.name == relative_name
    assert full_uri_table.name == expected_full_name
    assert state.client.source_table_api.get_table_config.call_args_list == [
        call(connector_id, relative_name, project_id='project-id'),
        call(connector_id, relative_name, project_id='project-id'),
        call(connector_id, expected_full_name, project_id='project-id'),
    ]


def _unbound_snowflake_connector() -> SnowflakeConnector:
    connector = object.__new__(SnowflakeConnector)
    connector._id = 'connector-snowflake'
    connector._name = 'warehouse-data'
    connector._kumo_project_id = 'project-id'
    connector.identity_id = 'identity-snowflake'
    connector.role = 'KUMO_READER'
    connector.warehouse = 'COMPUTE_WH'
    connector.database = 'PROD'
    connector.schema_name = 'SALES'
    connector.description = ''
    return connector


def test_redesigned_s3_and_snowflake_constructors_are_keyword_only():
    with pytest.raises(TypeError, match='positional'):
        cast(Any, S3Connector)('events', 's3://bucket/events/')
    with pytest.raises(TypeError, match='positional'):
        cast(Any, SnowflakeConnector)('warehouse-data', 'COMPUTE_WH', 'PROD',
                                      'SALES')


def test_warehouse_has_table_uses_exact_authorized_lookup_every_time():
    connector = _unbound_snowflake_connector()
    state = MagicMock()

    with patch('kumoai_studio.global_state', state):
        assert connector.has_table('CUSTOMERS')
        assert connector.has_table('CUSTOMERS')

    assert state.client.source_table_api.get_table_config.call_args_list == [
        call('connector-snowflake', 'CUSTOMERS', project_id='project-id'),
        call('connector-snowflake', 'CUSTOMERS', project_id='project-id'),
    ]
    state.client.source_table_api.list_tables.assert_not_called()


def test_nested_table_uses_exact_config_validation_instead_of_shallow_list():
    connector = _unbound_s3_connector()
    state = MagicMock()
    state._project_id = 'project-id'
    with patch('kumoai_studio.global_state', state):
        table = connector.table('2026/07/events.parquet')

    assert isinstance(table, SourceTable)
    assert table.name == '2026/07/events.parquet'
    state.client.source_table_api.get_table_config.assert_called_once_with(
        'connector-s3', '2026/07/events.parquet', project_id='project-id')
    state.client.source_table_api.list_tables.assert_not_called()


def test_full_uri_and_alias_use_same_exact_authorized_validation_each_time():
    connector = _unbound_s3_connector()
    state = MagicMock()
    state._project_id = 'project-id'
    with patch('kumoai_studio.global_state', state):
        table = connector.table('s3://bucket/root/nested/events.parquet')
        alias = connector.table_from_uri(
            's3://bucket/root/nested/events.parquet/')

    assert table.name == 'nested/events.parquet'
    assert alias.name == 'nested/events.parquet/'
    assert state.client.source_table_api.get_table_config.call_args_list == [
        call('connector-s3', 'nested/events.parquet', project_id='project-id'),
        call('connector-s3', 'nested/events.parquet/',
             project_id='project-id'),
    ]


@pytest.mark.parametrize(
    ('status', 'detail'),
    [
        (HTTPStatus.NOT_FOUND, 'Unable to find table nested/missing.parquet'),
        (HTTPStatus.UNPROCESSABLE_ENTITY,
         'Invalid table nested/missing.parquet'),
        (HTTPStatus.UNPROCESSABLE_ENTITY, {
            'detail': 'Invalid table nested/missing.parquet'
        }),
        (HTTPStatus.UNPROCESSABLE_ENTITY,
         '{"detail": "Invalid table nested/missing.parquet"}'),
    ],
)
def test_exact_table_known_missing_or_invalid_returns_false(status, detail):
    connector = _unbound_s3_connector()
    state = MagicMock()
    state._project_id = 'project-id'
    state.client.source_table_api.get_table_config.side_effect = HTTPException(
        status, detail)

    with patch('kumoai_studio.global_state', state):
        assert not connector.has_table('nested/missing.parquet')


@pytest.mark.parametrize(
    'name',
    [
        'gs://bucket/root/events.parquet',
        's3://other/root/events.parquet',
        's3://bucket/root/../events.parquet',
        'nested//events.parquet',
        'nested/%2E%2E/events.parquet',
    ],
)
def test_object_store_membership_is_false_for_invalid_coordinates(name):
    connector = _unbound_s3_connector()
    state = MagicMock()

    with patch('kumoai_studio.global_state', state):
        assert not connector.has_table(name)
        assert name not in connector
        with pytest.raises(ValueError):
            connector.table(name)

    state.client.source_table_api.get_table_config.assert_not_called()


@pytest.mark.parametrize('status', [
    HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN,
    HTTPStatus.UNPROCESSABLE_ENTITY, HTTPStatus.SERVICE_UNAVAILABLE
])
def test_exact_table_auth_and_outage_errors_propagate(status):
    connector = _unbound_s3_connector()
    state = MagicMock()
    state._project_id = 'project-id'
    state.client.source_table_api.get_table_config.side_effect = HTTPException(
        status, 'authorization or infrastructure failure')
    with patch('kumoai_studio.global_state', state):
        with pytest.raises(HTTPException) as error:
            connector.has_table('nested/events.parquet')
    assert error.value.status_code == status


def test_encoded_connector_prefix_is_decoded_and_canonically_rendered():
    config = _s3_rooted_config(prefix='project data')
    response = ConnectorResponse(id='connector-id', project_id='project-id',
                                 identity_id='identity-s3', name='files',
                                 timestamps=TIMESTAMPS, config=config)
    state = MagicMock()
    state.client.connector_api.create_if_not_exist.return_value = response

    with patch('kumoai_studio.global_state', state), \
            patch('kumoai_studio._require_project'):
        connector = S3Connector(name='files', identity='identity-s3',
                                root_dir='s3://bucket/project%20data/')

    sent = state.client.connector_api.create_if_not_exist.call_args.args[0]
    assert sent.config == config
    assert connector.root_dir == 's3://bucket/project%20data'


def test_nested_source_table_round_trips_through_table_definition():
    connector = _unbound_s3_connector()
    nested_name = 'archive/2026/07/events.parquet'
    source_config = S3SourceTable(
        connector_id=connector.id,
        source=S3RelativeSource(path=nested_name),
        file_type=FileType.PARQUET,
    )
    table = Table(SourceTable(nested_name, connector), columns=[])

    with patch.object(
            connector, '_get_table_config',
            return_value=SourceTableConfigResponse(
                source_table=source_config)):
        definition = table._to_api_table_definition()

    assert isinstance(definition.source_table, S3SourceTable)
    assert isinstance(definition.source_table.source, S3RelativeSource)
    assert definition.source_table.source.path == nested_name
    state = MagicMock()
    upload = state.client.connector_api.get_upload_connector.return_value
    upload.connector_id = 'connector-upload'
    state.client.connector_api.get.return_value = ConnectorResponse(
        id=connector.id,
        project_id='project-id',
        identity_id='identity-s3',
        name='project-data',
        timestamps=TIMESTAMPS,
        config=_s3_rooted_config(prefix='root'),
    )
    with patch('kumoai_studio.backend.cloud_backend',
               return_value=_backend_of(state)), \
            patch('kumoai_studio._require_project'):
        restored = Table._from_api_table_definition(definition)

    assert restored.source_table.name == nested_name
    assert restored.source_table.connector.id == connector.id


@pytest.mark.parametrize(
    ('connector_type', 'config', 'default_name'),
    [
        (
            S3Connector,
            S3ConnectorResourceConfig(target=S3AnyURIConnectorTarget()),
            's3',
        ),
        (
            GCSConnector,
            GCSConnectorResourceConfig(target=GCSAnyURIConnectorTarget()),
            'gcs',
        ),
        (
            ADLSConnector,
            ADLSConnectorResourceConfig(target=ADLSAnyURIConnectorTarget()),
            'adls',
        ),
    ],
)
def test_object_store_connectors_use_provider_default_name(
        connector_type, config, default_name):
    response = ConnectorResponse(
        id=f'connector-{default_name}',
        project_id='project-b',
        identity_id=f'identity-{default_name}',
        name=default_name,
        timestamps=TIMESTAMPS,
        config=config,
    )
    state = MagicMock()
    state.client.connector_api.create_if_not_exist.return_value = response
    state.client.connector_api.get_by_name.return_value = response

    with patch('kumoai_studio.global_state', state):
        connector = connector_type(project_id='project-b')
        loaded = connector_type.get_by_name(project_id='project-b')

    request = state.client.connector_api.create_if_not_exist.call_args
    assert request.args[0] == CreateConnectorArgs(name=default_name,
                                                  config=config)
    assert request.kwargs == {'project_id': 'project-b'}
    assert connector.name == default_name
    assert connector.any_uri
    assert connector.root_dir is None
    assert connector.kumo_project_id == 'project-b'
    assert connector.identity_id == f'identity-{default_name}'
    state.client.connector_api.get_by_name.assert_called_once_with(
        default_name, project_id='project-b')
    assert loaded.name == default_name
    assert loaded.any_uri
    assert loaded.root_dir is None


@pytest.mark.parametrize(
    ('connector_type', 'config', 'uri', 'expected_uri'),
    [
        (
            S3Connector,
            S3ConnectorResourceConfig(target=S3AnyURIConnectorTarget()),
            'S3://customer-bucket/deep/events%20data.parquet',
            's3://customer-bucket/deep/events%20data.parquet',
        ),
        (
            GCSConnector,
            GCSConnectorResourceConfig(
                target=GCSAnyURIConnectorTarget(),
                requester_pays_project='billing-project',
            ),
            'GS://customer-bucket/deep/events%20data.parquet',
            'gs://customer-bucket/deep/events%20data.parquet',
        ),
        (
            ADLSConnector,
            ADLSConnectorResourceConfig(target=ADLSAnyURIConnectorTarget()),
            ('ABFSS://curated@companylake.dfs.core.usgovcloudapi.net/'
             'deep/events%20data.parquet'),
            ('abfss://curated@companylake.dfs.core.usgovcloudapi.net/'
             'deep/events%20data.parquet'),
        ),
    ],
)
def test_object_store_any_uri_create_lookup_and_exact_table_only(
        connector_type, config, uri, expected_uri):
    response = ConnectorResponse(
        id=f'connector-{config.type.value}',
        project_id='project-b',
        identity_id=f'identity-{config.type.value}',
        name=f'all-{config.type.value}',
        timestamps=TIMESTAMPS,
        config=config,
    )
    state = MagicMock()
    state.client.connector_api.create_if_not_exist.return_value = response
    state.client.connector_api.get_by_name.return_value = response

    kwargs = {
        'name': response.name,
        'identity': response.identity_id,
        'project_id': response.project_id,
    }
    if connector_type is GCSConnector:
        kwargs['requester_pays_project'] = 'billing-project'
    with patch('kumoai_studio.global_state', state), \
            patch('kumoai_studio.global_state', state):
        connector = connector_type(**kwargs)
        table = connector.table(uri)
        loaded = connector_type.get_by_name(response.name,
                                            project_id='project-b')
        with pytest.raises(ValueError):
            connector.table('deep/events.parquet')
        with pytest.raises(NotImplementedError, match='cannot list'):
            connector.table_names()

    request = state.client.connector_api.create_if_not_exist.call_args.args[0]
    assert request.config == config
    assert request.identity_id == response.identity_id
    assert connector.any_uri and connector.root_dir is None
    assert connector.identity_id == response.identity_id
    assert connector.kumo_project_id == response.project_id
    assert loaded.any_uri and loaded.root_dir is None
    assert loaded.identity_id == response.identity_id
    assert loaded.kumo_project_id == response.project_id
    assert table.name == expected_uri
    if isinstance(connector, GCSConnector):
        assert connector.requester_pays_project == 'billing-project'
        assert loaded.requester_pays_project == 'billing-project'
    state.client.source_table_api.get_table_config.assert_called_once_with(
        response.id, expected_uri, project_id='project-b')


@pytest.mark.parametrize(
    ('connector_type', 'config', 'api_source', 'uri'),
    [
        (
            GCSConnector,
            GCSConnectorResourceConfig(target=GCSAnyURIConnectorTarget()),
            GCSSourceTable(
                connector_id='connector-gcs',
                source=GCSURISource(
                    uri='gs://customer-bucket/deep/events.parquet'),
                file_type=FileType.PARQUET,
            ),
            'gs://customer-bucket/deep/events.parquet',
        ),
        (
            ADLSConnector,
            ADLSConnectorResourceConfig(target=ADLSAnyURIConnectorTarget()),
            ADLSSourceTable(
                connector_id='connector-adls',
                source=ADLSURISource(
                    uri=('abfss://curated@lake.dfs.core.windows.net/'
                         'deep/events.parquet')),
                file_type=FileType.PARQUET,
            ),
            ('abfss://curated@lake.dfs.core.windows.net/'
             'deep/events.parquet'),
        ),
    ],
)
def test_gcs_adls_any_uri_source_table_hydrates_without_a_root(
        connector_type, config, api_source, uri):
    response = ConnectorResponse(
        id=api_source.connector_id,
        project_id='project-b',
        identity_id=f'identity-{config.type.value}',
        name=f'all-{config.type.value}',
        timestamps=TIMESTAMPS,
        config=config,
    )
    state = MagicMock()
    state.client.connector_api.get.return_value = response
    definition = MagicMock(source_table=api_source)

    with patch('kumoai_studio.backend.cloud_backend',
               return_value=_backend_of(state)):
        restored = SourceTable._from_api_table_definition(
            definition, project_id='project-b')

    assert isinstance(restored.connector, connector_type)
    assert restored.connector.any_uri
    assert restored.connector.root_dir is None
    assert restored.name == uri
    assert restored._to_api_source_table() == api_source


def test_connector_raw_slug_like_identity_is_never_resolved_by_sdk():
    config = S3ConnectorResourceConfig(target=S3AnyURIConnectorTarget())
    response = ConnectorResponse(
        id='connector-s3',
        project_id='project-b',
        identity_id='prod-aws',
        name='all-s3',
        timestamps=TIMESTAMPS,
        config=config,
    )
    state = MagicMock()
    state.client.connector_api.create_if_not_exist.return_value = response

    with patch('kumoai_studio.global_state', state), \
            patch('kumoai_studio.get_identity') as get_identity:
        connector = S3Connector(name='all-s3', identity='prod-aws',
                                project_id='project-b')

    args = state.client.connector_api.create_if_not_exist.call_args.args[0]
    assert args.identity_id == 'prod-aws'
    assert connector.identity_id == 'prod-aws'
    get_identity.assert_not_called()
    state.client.identity_api.get_by_slug.assert_not_called()


def test_any_uri_table_uses_full_canonical_uri_and_exact_project_once():
    connector = object.__new__(S3Connector)
    connector._id = 'connector-s3'
    connector._name = 's3'
    connector._kumo_project_id = 'project-b'
    connector.identity_id = 'identity-ambient-s3'
    connector.description = ''
    connector.target = S3AnyURIConnectorTarget()
    connector.any_uri = True
    connector.bucket = connector.prefix = connector.region = None
    connector.root_dir = None
    state = MagicMock()

    with patch('kumoai_studio.global_state', state):
        table = connector.table(
            's3://customer-data/deep/many/layers/events%20data.parquet')

    assert table.name == (
        's3://customer-data/deep/many/layers/events%20data.parquet')
    state.client.source_table_api.get_table_config.assert_called_once_with(
        'connector-s3',
        's3://customer-data/deep/many/layers/events%20data.parquet',
        project_id='project-b',
    )
    with pytest.raises(ValueError, match='S3 URI'):
        connector.table('deep/many/layers/events.parquet')
    with pytest.raises(NotImplementedError, match='cannot list'):
        connector.table_names()


def test_connector_operations_keep_response_project_after_global_switch():
    connector = _unbound_s3_connector()
    state = MagicMock()
    state._project_id = 'project-c'

    with patch('kumoai_studio.global_state', state), \
            patch('kumoai_studio.global_state', state):
        connector.table_names()
        connector.table('nested/events.parquet')
        connector._delete_connector()

    state.client.source_table_api.list_tables.assert_called_once_with(
        connector.id, project_id='project-id')
    state.client.source_table_api.get_table_config.assert_called_once_with(
        connector.id,
        'nested/events.parquet',
        project_id='project-id',
    )
    state.client.connector_api.delete_if_exists.assert_called_once_with(
        connector.id, project_id='project-id')


@pytest.mark.parametrize(
    'root_dir',
    [
        's3://bucket//root',
        's3://bucket/root//',
        's3://bucket/events%2Fsecret',
        's3://bucket/events%5Csecret',
    ],
)
def test_rooted_s3_connector_rejects_ambiguous_prefix_separators(root_dir):
    state = MagicMock()
    with patch('kumoai_studio.global_state', state):
        with pytest.raises(ValueError, match='separator'):
            S3Connector(root_dir=root_dir, project_id='project-b')
    state.client.connector_api.create_if_not_exist.assert_not_called()


@pytest.mark.parametrize(
    'name',
    [
        'nested/events%2Fsecret.parquet',
        's3://bucket/root/nested/events%2Fsecret.parquet',
        's3://bucket/root//nested/events.parquet',
    ],
)
def test_rooted_s3_table_rejects_encoded_or_ambiguous_separators(name):
    connector = _unbound_s3_connector()
    with pytest.raises(ValueError):
        connector.table(name)


def test_source_table_conversion_propagates_error_without_retry_or_mutation():
    connector = _unbound_s3_connector()
    source = SourceTable('nested/events.parquet', connector)
    error = HTTPException(HTTPStatus.FORBIDDEN, 'identity was unbound')

    with patch.object(connector, '_get_table_config',
                      side_effect=error) as get:
        with pytest.raises(HTTPException) as raised:
            source._to_api_source_table()

    assert raised.value is error
    get.assert_called_once_with('nested/events.parquet')
    assert source.name == 'nested/events.parquet'
