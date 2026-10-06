# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for DeployEnvConfig -- centralized deployment configuration.

End-to-end scenarios:
  a) Legacy env vars -> ConfigMigrationHelper -> RawDeployConfig
  b) Structured config -> DeployEnvConfig:
     b1) preset only
     b2) preset + override file (DEPLOY_ENV_CONFIG_FILE)
     b3) file only (no preset)
"""

import dataclasses
import getpass
import json
import os
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from pydantic_core import to_jsonable_python

from service_framework._config_migration import ConfigMigrationHelper
from service_framework._raw_deploy_config import (
    PLATFORM_DEFAULTS,
    AuthConfig,
    CustomTemporalConfig,
    DirectConnectAuthConfig,
    DirectConnectPasswordAuthConfig,
    DiskgraphConfig,
    FileStorageConfig,
    LoggingBackend,
    LoggingConfig,
    LoggingOutputFormat,
    PlatformPreset,
    PostgresMetadataStorageConfig,
    S3AddressingStyle,
    S3StorageConfig,
    SecretBackend,
    SecretConfig,
    SecretKeyRef,
    SqliteMetadataStorageConfig,
    TemporalCloudConfig,
    load_raw_deploy_config,
)
from service_framework.deploy_env import DEVTEST_NAMESPACE, ENV_KUMO_HOME
from service_framework.deploy_env_config import DeployEnvConfig
from service_framework.deploy_stage import DeployStage

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')


@pytest.fixture(autouse=True)
def _clear_pod_namespace(monkeypatch):  # pyright: ignore[reportUnusedFunction]
    # The k8s_namespace tests assert the legacy derivation unless a test
    # sets POD_NAMESPACE itself; guarantee a clean slate even when the
    # test runner itself executes inside a pod with the downward API.
    monkeypatch.delenv('POD_NAMESPACE', raising=False)


# --- Helpers ---


def _to_dict(obj):
    """Serialize a dataclass to a JSON-round-tripped dict for comparison."""
    serialized = json.loads(
        json.dumps(dataclasses.asdict(obj), default=to_jsonable_python))
    # Keep legacy fixture coverage focused on pre-existing config surfaces.
    serialized.pop('metadata_storage', None)
    serialized.pop('ambient_s3', None)
    serialized.pop('ambient_gcp', None)
    serialized.pop('external_host', None)
    # Older golden fixtures intentionally omit newly defaulted deploy fields.
    if not serialized.get('storage', {}).get('lifecycle_ready', False):
        serialized.get('storage', {}).pop('lifecycle_ready', None)
    if serialized.get('storage', {}).get('s3') is None:
        serialized.get('storage', {}).pop('s3', None)
    return serialized


def test_storage_lifecycle_readiness_defaults_false_and_parses(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv(
        'DEPLOY_ENV_CONFIG_JSON',
        json.dumps({'storage': {
            'lifecycle_ready': True
        }}),
    )
    assert load_raw_deploy_config().storage.lifecycle_ready is True

    monkeypatch.setenv('DEPLOY_ENV_CONFIG_JSON', json.dumps({'storage': {}}))
    assert load_raw_deploy_config().storage.lifecycle_ready is False


def test_s3_compatible_storage_config_parses_without_secret_values(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv(
        'DEPLOY_ENV_CONFIG_JSON',
        json.dumps({
            'storage': {
                'data_plane_storage_root': 's3://example-dataplane/customer-a/',
                'artifact_root': 's3://example-artifacts/',
                's3': {
                    'endpoint_url': 'https://minio.example.test:9000',
                    'region': 'us-east-1',
                    'addressing_style': 'path',
                    'credential_secret_name': 'MINIO_KUMO_CREDENTIALS',
                    'ca_bundle_path': '/etc/kumo/minio-ca/ca.crt',
                },
            },
        }),
    )

    storage = load_raw_deploy_config().storage

    assert storage.s3 == S3StorageConfig(
        endpoint_url='https://minio.example.test:9000',
        region='us-east-1',
        addressing_style=S3AddressingStyle.PATH,
        credential_secret_name='MINIO_KUMO_CREDENTIALS',
        ca_bundle_path='/etc/kumo/minio-ca/ca.crt',
    )
    serialized = json.dumps(dataclasses.asdict(storage))
    assert 'aws_access_key_id' not in serialized
    assert 'aws_secret_access_key' not in serialized


@pytest.mark.parametrize(
    'endpoint',
    [
        'minio.example.test:9000',
        'ftp://minio.example.test',
        'https://user:password@minio.example.test',
        'https://minio.example.test/console',
        'https://minio.example.test?query=yes',
        'https://minio.example.test#fragment',
        ' https://minio.example.test',
    ],
)
def test_s3_compatible_storage_rejects_unsafe_endpoint(monkeypatch, endpoint):
    _clear_env(monkeypatch)
    monkeypatch.setenv(
        'DEPLOY_ENV_CONFIG_JSON',
        json.dumps({
            'storage': {
                'data_plane_storage_root': 's3://example-dataplane/',
                's3': {
                    'endpoint_url': endpoint,
                    'region': 'us-east-1',
                    'addressing_style': 'path',
                    'credential_secret_name': 'MINIO_KUMO_CREDENTIALS',
                },
            },
        }),
    )

    with pytest.raises(ValidationError, match='endpoint_url'):
        load_raw_deploy_config()


def test_s3_compatible_storage_requires_s3_root(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv(
        'DEPLOY_ENV_CONFIG_JSON',
        json.dumps({
            'storage': {
                'data_plane_storage_root': '/mnt/localssd',
                's3': {
                    'endpoint_url': 'https://minio.example.test:9000',
                    'region': 'us-east-1',
                    'addressing_style': 'path',
                    'credential_secret_name': 'MINIO_KUMO_CREDENTIALS',
                },
            },
        }),
    )

    with pytest.raises(ValidationError, match='at least one s3://'):
        load_raw_deploy_config()


def test_s3_compatible_storage_requires_https_in_production(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv('DEPLOY_STAGE', 'prod')
    monkeypatch.setenv(
        'DEPLOY_ENV_CONFIG_JSON',
        json.dumps({
            'storage': {
                'data_plane_storage_root': 's3://example-dataplane/',
                's3': {
                    'endpoint_url': 'http://minio:9000',
                    'region': 'us-east-1',
                    'addressing_style': 'path',
                    'credential_secret_name': 'MINIO_KUMO_CREDENTIALS',
                },
            },
        }),
    )

    with pytest.raises(ValidationError, match='must use https'):
        DeployEnvConfig.parse_from_env()


def test_s3_compatible_storage_survives_root_resolution(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv('DEPLOY_STAGE', 'dev_local')
    monkeypatch.setenv(
        'DEPLOY_ENV_CONFIG_JSON',
        json.dumps({
            'storage': {
                'data_plane_storage_root': 's3://example-dataplane/',
                'lifecycle_ready': True,
                's3': {
                    'endpoint_url': 'http://minio:9000',
                    'region': 'us-east-1',
                    'addressing_style': 'path',
                    'credential_secret_name': 'MINIO_KUMO_CREDENTIALS',
                },
            },
        }),
    )

    storage = DeployEnvConfig.parse_from_env().storage

    assert storage.s3 is not None
    assert storage.s3.addressing_style == S3AddressingStyle.PATH
    assert storage.lifecycle_ready is True


def _load_fixture(name: str) -> dict:
    with open(os.path.join(_TEST_DATA_DIR, name)) as f:
        return json.load(f)


_ALL_ENV_VARS = [
    'IS_GCP_DEPLOY',
    'DATABRICKS_HOST',
    'DATABRICKS_CLUSTER_ID',
    'DATABRICKS_WAREHOUSE_ID',
    'SNOWFLAKE_ACCOUNT',
    'SNOWFLAKE_HOST',
    'SNOWFLAKE_WAREHOUSE',
    'SNOWFLAKE_DATABASE',
    'SNOWFLAKE_SCHEMA',
    'TEMPORAL_SERVER',
    'TEMPORAL_ADDRESS',
    'TEMPORAL_CERT_DIR',
    'TEMPORAL_NAMESPACE',
    'CUSTOM_S3_DATA_PLANE_BUCKET',
    'CUSTOM_REST_PERSIST_DIR',
    'CUSTOM_ARTIFACT_ROOT',
    'CUSTOMER_ROLE',
    'DISABLE_REMOTE_INGESTION',
    'PLATFORM_PRESET',
    'DEPLOY_ENV_CONFIG_JSON',
    'DEPLOY_ENV_CONFIG_FILE',
    'DEPLOY_STAGE',
    ENV_KUMO_HOME,
    'DEVSERVER_INSTANCE_SUFFIX',
    'INSTANCE_ID',
    'REGION_SUFFIX',
    'AZURE_CLIENT_ID',
    'AZURE_CLIENT_SECRET',
    'AZURE_TENANT_ID',
    'AZURE_STORAGE_ACCOUNT',
]


def _clear_env(monkeypatch):
    for var in _ALL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# ===================================================================
# Section 1: Legacy Migration (env vars -> RawDeployConfig)
# ===================================================================

_DEV = DeployStage.DEV
_PROD = DeployStage.PROD

_MIGRATION_CASES = [
    pytest.param({}, _DEV, 'test', 'inst', 'migration_default.json',
                 id='default'),
    pytest.param({}, _DEV, 'test', 'inst', 'migration_aws_remote.json',
                 id='aws-remote'),
    pytest.param({'IS_GCP_DEPLOY': 'true'}, _DEV, 'test', 'inst',
                 'migration_gcp_remote.json', id='gcp-remote'),
    pytest.param({'TEMPORAL_SERVER': 'localhost'}, _DEV, 'test', 'inst',
                 'migration_dev_local.json', id='dev-local'),
    pytest.param({'TEMPORAL_SERVER': 'cloud'}, _PROD, 'acme', 'prod',
                 'migration_prod_cloud.json', id='prod-cloud'),
]


class TestLegacyMigration:
    """Legacy env vars -> ConfigMigrationHelper -> matches fixture JSON."""
    @pytest.mark.parametrize(
        'env_vars,deploy_mode,_legacy_tenant,instance_id,fixture',
        _MIGRATION_CASES,
    )
    def test_env_to_raw_config(self, monkeypatch, env_vars, deploy_mode,
                               _legacy_tenant, instance_id, fixture):
        _clear_env(monkeypatch)
        for k, v in env_vars.items():
            monkeypatch.setenv(k, v)

        raw = ConfigMigrationHelper(deploy_mode, instance_id).build()

        assert _to_dict(raw) == _load_fixture(fixture)

    # -- Edge cases --

    def test_temporal_custom_address(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv('TEMPORAL_SERVER', 'localhost')
        monkeypatch.setenv('TEMPORAL_ADDRESS', '10.0.0.5')

        raw = ConfigMigrationHelper(_DEV, 'inst').build()
        assert raw.orchestration.server_address == '10.0.0.5:7233'

    def test_temporal_cert_dir(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv('TEMPORAL_SERVER', 'localhost')
        monkeypatch.setenv('TEMPORAL_CERT_DIR', '/etc/temporal/certs')

        raw = ConfigMigrationHelper(_DEV, 'inst').build()
        assert raw.orchestration.tls_cert_dir == '/etc/temporal/certs'

    def test_custom_namespace_override(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv('TEMPORAL_NAMESPACE', 'custom-ns')

        raw = ConfigMigrationHelper(_PROD, 'prod').build()
        assert raw.orchestration.namespace == 'custom-ns'

    def test_custom_bucket(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv('CUSTOM_S3_DATA_PLANE_BUCKET', 's3://custom-bucket/')

        raw = ConfigMigrationHelper(_DEV, 'inst').build()
        assert raw.storage.data_plane_storage_root == 's3://custom-bucket/'

    def test_custom_persist_dir(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv('CUSTOM_REST_PERSIST_DIR', 's3://persist-bucket')

        raw = ConfigMigrationHelper(_DEV, 'inst').build()
        assert raw.storage.control_plane_storage_root == 's3://persist-bucket'


# ===================================================================
# Section 2: Structured Config E2E (-> DeployEnvConfig)
# ===================================================================

_PROD_ENV = {'DEPLOY_STAGE': 'prod'}
_DEV_ENV = {'DEPLOY_STAGE': 'dev', 'INSTANCE_ID': 'abc123'}


class TestStructuredConfigE2E:
    """Structured config (preset/file) -> DeployEnvConfig -> fixture JSON."""
    @pytest.mark.parametrize(
        'preset,config_file,env_vars,fixture',
        [
            # b1: preset only -> DeployEnvConfig
            ('aws', None, _PROD_ENV, 'resolved_aws_minimal.json'),
            ('gcp', None, _PROD_ENV, 'resolved_gcp_minimal.json'),
            # b2: preset + override file -> DeployEnvConfig
            ('aws', 'raw_aws_overrides.json', _PROD_ENV,
             'resolved_aws_overrides.json'),
            ('gcp', 'raw_gcp_overrides.json', _PROD_ENV,
             'resolved_gcp_overrides.json'),
            # b3: file only (no preset) -> DeployEnvConfig
            (None, 'raw_dev_overrides.json', _DEV_ENV,
             'resolved_dev_overrides.json'),
        ],
        ids=[
            'aws-minimal',
            'gcp-minimal',
            'aws-overrides',
            'gcp-overrides',
            'dev-overrides',
        ],
    )
    def test_config_e2e(self, monkeypatch, preset, config_file, env_vars,
                        fixture):
        _clear_env(monkeypatch)
        if preset:
            monkeypatch.setenv('PLATFORM_PRESET', preset)
        if config_file:
            monkeypatch.setenv('DEPLOY_ENV_CONFIG_FILE',
                               os.path.join(_TEST_DATA_DIR, config_file))
        for k, v in env_vars.items():
            monkeypatch.setenv(k, v)

        with patch('getpass.getuser', return_value='testuser'):
            cfg = DeployEnvConfig.parse_from_env(app_name='rest-api')

        assert _to_dict(cfg) == _load_fixture(fixture)

    def test_ambient_s3_structured_config_is_preserved(self, monkeypatch):
        _clear_env(monkeypatch)
        role_arn = 'arn:aws:iam::123456789012:role/kumo-ambient-test'
        monkeypatch.setenv('PLATFORM_PRESET', 'aws')
        monkeypatch.setenv('DEPLOY_STAGE', 'dev')
        monkeypatch.setenv('INSTANCE_ID', 'ambient-test')
        monkeypatch.setenv(
            'DEPLOY_ENV_CONFIG_JSON',
            json.dumps({
                'ambient_s3': {
                    'approved_role_arns': [role_arn],
                },
            }),
        )

        cfg = DeployEnvConfig.parse_from_env(app_name='rest-api')

        assert cfg.ambient_s3 is not None
        assert cfg.ambient_s3.approved_role_arns == (role_arn, )

    def test_external_host_structured_config_is_preserved(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv('PLATFORM_PRESET', 'aws')
        monkeypatch.setenv('DEPLOY_STAGE', 'dev')
        monkeypatch.setenv('INSTANCE_ID', 'ext-host-test')
        monkeypatch.setenv(
            'DEPLOY_ENV_CONFIG_JSON',
            json.dumps({'external_host': 'kumo-acme.dev.kumoai.cloud'}),
        )

        cfg = DeployEnvConfig.parse_from_env(app_name='rest-api')

        assert cfg.external_host == 'kumo-acme.dev.kumoai.cloud'

    @pytest.mark.parametrize('preset', ['aws', 'gcp'])
    def test_external_host_presets_default_to_none(self, monkeypatch, preset):
        _clear_env(monkeypatch)
        monkeypatch.setenv('PLATFORM_PRESET', preset)
        monkeypatch.setenv('DEPLOY_STAGE', 'prod')

        cfg = DeployEnvConfig.parse_from_env(app_name='rest-api')

        assert cfg.external_host is None

    @pytest.mark.parametrize('preset', ['aws', 'gcp'])
    def test_ambient_s3_presets_are_disabled_by_default(self, monkeypatch,
                                                        preset):
        _clear_env(monkeypatch)
        monkeypatch.setenv('PLATFORM_PRESET', preset)
        monkeypatch.setenv('DEPLOY_STAGE', 'prod')

        cfg = DeployEnvConfig.parse_from_env(app_name='rest-api')

        assert cfg.ambient_s3 is None

    def test_ambient_gcp_structured_config_is_preserved(self, monkeypatch):
        _clear_env(monkeypatch)
        email = 'kumo-workload@example-project.iam.gserviceaccount.com'
        monkeypatch.setenv('PLATFORM_PRESET', 'gcp')
        monkeypatch.setenv('DEPLOY_STAGE', 'dev')
        monkeypatch.setenv('INSTANCE_ID', 'ambient-gcp-test')
        monkeypatch.setenv(
            'DEPLOY_ENV_CONFIG_JSON',
            json.dumps({
                'ambient_gcp': {
                    'approved_service_account_emails': [email],
                },
            }),
        )

        cfg = DeployEnvConfig.parse_from_env(app_name='rest-api')

        assert cfg.ambient_gcp is not None
        assert cfg.ambient_gcp.approved_service_account_emails == (email, )

    @pytest.mark.parametrize('preset', ['aws', 'gcp'])
    def test_ambient_gcp_presets_are_disabled_by_default(
            self, monkeypatch, preset):
        _clear_env(monkeypatch)
        monkeypatch.setenv('PLATFORM_PRESET', preset)
        monkeypatch.setenv('DEPLOY_STAGE', 'prod')

        cfg = DeployEnvConfig.parse_from_env(app_name='rest-api')

        assert cfg.ambient_gcp is None


# ===================================================================
# Section 3: Properties & Computed Fields
# ===================================================================


class TestConstruction:
    """Test DeployEnvConfig constructed directly (not from env)."""
    def test_basic_fields(self):
        cfg = DeployEnvConfig(
            client_id='test-app',
            deploy_mode=DeployStage.DEV,
            instance_id='user-inst',
        )
        assert cfg.client_id == 'test-app'
        assert cfg.deploy_mode == DeployStage.DEV
        assert cfg.instance_id == 'user-inst'

    def test_sub_config_fields(self):
        compute = DiskgraphConfig()
        orch = TemporalCloudConfig()
        secrets = SecretConfig(backend=SecretBackend.AWS_SM)
        storage = FileStorageConfig()
        logging_cfg = LoggingConfig(backend=LoggingBackend.CLOUDWATCH)

        cfg = DeployEnvConfig(
            data_compute=compute,
            orchestration=orch,
            secrets=secrets,
            storage=storage,
            logging_config=logging_cfg,
        )
        assert cfg.data_compute is compute
        assert cfg.orchestration is orch
        assert cfg.secrets is secrets
        assert cfg.storage is storage
        assert cfg.logging_config is logging_cfg

    def test_defaults(self):
        cfg = DeployEnvConfig()
        assert cfg.client_id == ''
        assert cfg.deploy_mode == DeployStage.DEV
        assert isinstance(cfg.data_compute, DiskgraphConfig)
        assert cfg.logging_config.output_format == LoggingOutputFormat.TEXT
        assert cfg.ambient_s3 is None


class TestMetadataStorageConfig:
    def test_defaults_to_sqlite(self):
        cfg = DeployEnvConfig()
        assert isinstance(cfg.metadata_storage, SqliteMetadataStorageConfig)

    def test_parse_postgres_metadata_storage(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv(
            'DEPLOY_ENV_CONFIG_JSON',
            json.dumps({
                'metadata_storage': {
                    'type': 'postgres',
                    'dsn': 'postgresql://user:pass@localhost/db',
                }
            }))
        monkeypatch.setenv('DEPLOY_STAGE', 'dev')
        monkeypatch.setenv('INSTANCE_ID', 'test')

        cfg = DeployEnvConfig.parse_from_env()
        assert isinstance(cfg.metadata_storage, PostgresMetadataStorageConfig)
        assert cfg.metadata_storage.dsn == 'postgresql://user:pass@localhost/db'

    def test_parse_postgres_dsn_secret_name(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv(
            'DEPLOY_ENV_CONFIG_JSON',
            json.dumps({
                'metadata_storage': {
                    'type': 'postgres',
                    'dsn_secret_name': 'STATE_DB_DSN',
                }
            }))
        monkeypatch.setenv('DEPLOY_STAGE', 'dev')
        monkeypatch.setenv('INSTANCE_ID', 'test')

        cfg = DeployEnvConfig.parse_from_env()
        assert isinstance(cfg.metadata_storage, PostgresMetadataStorageConfig)
        assert cfg.metadata_storage.dsn is None
        assert cfg.metadata_storage.dsn_secret_name == 'STATE_DB_DSN'

    def test_parse_passwordless_postgres(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv(
            'DEPLOY_ENV_CONFIG_JSON',
            json.dumps({
                'metadata_storage': {
                    'type': 'postgres',
                    'auth_mode': 'azure_entra',
                    'username': 'kumo-wi',
                    'host': 'kumo.postgres.database.azure.com',
                    'database': 'postgres',
                }
            }))
        monkeypatch.setenv('DEPLOY_STAGE', 'dev')
        monkeypatch.setenv('INSTANCE_ID', 'test')

        storage = DeployEnvConfig.parse_from_env().metadata_storage
        assert isinstance(storage, PostgresMetadataStorageConfig)
        assert storage.auth_mode == 'azure_entra'
        assert storage.port == 5432

    def test_postgres_rejects_both_dsn_sources(self):
        with pytest.raises(ValidationError):
            PostgresMetadataStorageConfig(
                dsn='postgresql://user:pass@localhost/db',
                dsn_secret_name='STATE_DB_DSN')

    def test_postgres_rejects_no_dsn_source(self):
        with pytest.raises(ValidationError):
            PostgresMetadataStorageConfig()


class TestAuthConfig:
    """Auth config is part of the structured deploy-config contract."""
    def test_parse_direct_connect_auth_config(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv('DEPLOY_STAGE', 'dev')
        monkeypatch.setenv('INSTANCE_ID', 'test')
        monkeypatch.setenv(
            'DEPLOY_ENV_CONFIG_JSON',
            json.dumps({
                'auth': {
                    'enabled': True,
                    'backend': 'direct_connect',
                    'direct_connect': {
                        'login_method': 'password',
                        'allowed_return_origins':
                        'https://kumo.example.com, http://localhost:5173',
                        'admin_emails': ['admin@kumo.ai', ' ops@kumo.ai '],
                        'password': {
                            'temporary_password_ref': {
                                'secret_name': 'direct-connect-auth',
                                'secret_key': 'temporary-password',
                            },
                        },
                    },
                },
            }))

        cfg = DeployEnvConfig.parse_from_env()

        assert isinstance(cfg.auth, AuthConfig)
        assert cfg.auth.enabled is True
        assert cfg.auth.backend == 'direct_connect'

        direct_connect = cfg.auth.direct_connect
        assert isinstance(direct_connect, DirectConnectAuthConfig)
        assert direct_connect.login_method == 'password'
        assert direct_connect.allowed_return_origins == (
            'https://kumo.example.com', 'http://localhost:5173')
        assert direct_connect.admin_emails == ('admin@kumo.ai', 'ops@kumo.ai')

        password = direct_connect.password
        assert isinstance(password, DirectConnectPasswordAuthConfig)
        assert password.temporary_password_ref == SecretKeyRef(
            secret_name='direct-connect-auth',
            secret_key='temporary-password',
        )


class TestTemporalProperties:
    """Temporal-related properties derived from orchestration config."""
    def test_cloud_temporal_namespace(self):
        cfg = DeployEnvConfig(
            deploy_mode=DeployStage.PROD,
            instance_id='prod',
            orchestration=TemporalCloudConfig(namespace='prod'),
        )
        assert cfg.temporal_namespace == 'prod.put0h'
        assert cfg.temporal_server_address == 'prod.put0h.tmprl.cloud:7233'

    def test_cloud_temporal_namespace_none(self):
        cfg = DeployEnvConfig(
            deploy_mode=DeployStage.PROD,
            orchestration=TemporalCloudConfig(),
        )
        assert cfg.temporal_namespace == '.put0h'

    def test_custom_temporal_namespace(self):
        cfg = DeployEnvConfig(
            orchestration=CustomTemporalConfig(server_address='10.0.0.5:7233',
                                               namespace=DEVTEST_NAMESPACE),
            instance_id='user-inst',
        )
        assert cfg.temporal_namespace == DEVTEST_NAMESPACE
        assert cfg.temporal_server_address == '10.0.0.5:7233'

    def test_tls_cert_dir_cloud(self):
        cfg = DeployEnvConfig()
        assert cfg.tls_cert_dir == ''

    def test_tls_cert_dir_custom(self):
        cfg = DeployEnvConfig(
            orchestration=CustomTemporalConfig(server_address='10.0.0.5:7233',
                                               tls_cert_dir='/etc/certs'), )
        assert cfg.tls_cert_dir == '/etc/certs'

    def test_web_url_cloud(self):
        cfg = DeployEnvConfig(
            deploy_mode=DeployStage.PROD,
            instance_id='prod',
            orchestration=TemporalCloudConfig(namespace='prod'),
        )
        assert 'prod.put0h' in cfg.web_url

    def test_web_url_localhost(self):
        cfg = DeployEnvConfig(
            orchestration=CustomTemporalConfig(
                server_address='127.0.0.1:7233'), )
        assert cfg.web_url == 'localhost:7233'


class TestK8sProperties:
    """k8s_namespace and k8s_deploy_env_label."""
    def test_prod_namespace(self):
        cfg = DeployEnvConfig(
            deploy_mode=DeployStage.PROD,
            instance_id='prod',
        )
        assert cfg.k8s_namespace == 'prod'
        assert cfg.k8s_deploy_env_label == 'prod'

    def test_preprod_namespace(self):
        cfg = DeployEnvConfig(
            deploy_mode=DeployStage.PREPROD,
            instance_id='i-123',
        )
        assert cfg.k8s_namespace == 'preprod'
        assert cfg.k8s_deploy_env_label == 'preprod-i-123'

    def test_dev_namespace(self):
        cfg = DeployEnvConfig(
            deploy_mode=DeployStage.DEV,
            instance_id='user-inst',
        )
        assert cfg.k8s_namespace == 'kumo-dataplane-devtest'
        assert cfg.k8s_deploy_env_label == 'dev-user-inst'

    def test_pod_namespace_env_overrides(self, monkeypatch):
        monkeypatch.setenv('POD_NAMESPACE', 'kumo')
        cfg = DeployEnvConfig(
            deploy_mode=DeployStage.PROD,
            instance_id='prod',
        )
        assert cfg.k8s_namespace == 'kumo'

    def test_pod_namespace_env_empty_falls_back(self, monkeypatch):
        monkeypatch.setenv('POD_NAMESPACE', '')
        cfg = DeployEnvConfig(
            deploy_mode=DeployStage.PROD,
            instance_id='prod',
        )
        assert cfg.k8s_namespace == 'prod'


class TestIdSuffix:
    """id_suffix, task_queue, workflow_id, strip_suffix."""
    def test_prod_suffix(self):
        cfg = DeployEnvConfig(
            deploy_mode=DeployStage.PROD,
            instance_id='prod',
        )
        assert cfg.id_suffix == '-prod'
        assert cfg.task_queue('ml-train') == 'ml-train-prod'
        assert cfg.workflow_id('wf-123') == 'wf-123-prod'

    def test_dev_suffix(self):
        cfg = DeployEnvConfig(
            deploy_mode=DeployStage.DEV,
            instance_id='user-inst',
        )
        assert cfg.id_suffix == '-user-inst'

    def test_test_suffix(self):
        cfg = DeployEnvConfig(
            deploy_mode=DeployStage.TEST,
            instance_id='ci',
        )
        assert cfg.id_suffix == '-ci-test'

    def test_strip_suffix(self):
        cfg = DeployEnvConfig(
            deploy_mode=DeployStage.PROD,
            instance_id='acme',
        )
        assert cfg.strip_suffix('ml-train-acme') == 'ml-train'
        assert cfg.strip_suffix('ml-train') == 'ml-train'

    def test_idempotent_suffix(self):
        cfg = DeployEnvConfig(
            deploy_mode=DeployStage.PROD,
            instance_id='acme',
        )
        assert cfg.task_queue('ml-train-acme') == 'ml-train-acme'


class TestStorageResolution:
    """Storage roots are eagerly resolved in parse_from_env."""
    def test_default_s3_root(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv('DEPLOY_STAGE', 'prod')
        cfg = DeployEnvConfig.parse_from_env()

        assert cfg.storage.data_plane_storage_root == (
            's3://kumo-prod-dataplane-bucket/prod/')
        assert cfg.storage.control_plane_storage_root == (
            cfg.storage.data_plane_storage_root)

    def test_custom_bucket(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv('DEPLOY_STAGE', 'prod')
        monkeypatch.setenv('CUSTOM_S3_DATA_PLANE_BUCKET',
                           's3://customer-bucket')

        cfg = DeployEnvConfig.parse_from_env()

        assert cfg.storage.data_plane_storage_root == 's3://customer-bucket'

    def test_custom_bucket_with_template(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv('DEPLOY_STAGE', 'preprod')
        monkeypatch.setenv('INSTANCE_ID', 'i-123')
        monkeypatch.setenv('CUSTOM_S3_DATA_PLANE_BUCKET',
                           's3://customer-bucket/{instance_id}')

        cfg = DeployEnvConfig.parse_from_env()

        assert cfg.storage.data_plane_storage_root == (
            's3://customer-bucket/i-123')

    def test_preprod_legacy_append(self, monkeypatch):
        """PREPROD without {instance_id} template gets instance_id appended."""
        _clear_env(monkeypatch)
        monkeypatch.setenv('DEPLOY_STAGE', 'preprod')
        monkeypatch.setenv('INSTANCE_ID', 'i-123')
        monkeypatch.setenv('CUSTOM_S3_DATA_PLANE_BUCKET',
                           's3://customer-bucket')

        cfg = DeployEnvConfig.parse_from_env()

        assert cfg.storage.data_plane_storage_root == (
            's3://customer-bucket/i-123')

    def test_custom_persist_dir(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv('DEPLOY_STAGE', 'prod')
        monkeypatch.setenv('CUSTOM_REST_PERSIST_DIR', 's3://persist-bucket')

        cfg = DeployEnvConfig.parse_from_env()

        assert cfg.storage.control_plane_storage_root == (
            's3://persist-bucket/prod')

    def test_region_suffix(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv('DEPLOY_STAGE', 'prod')
        monkeypatch.setenv('REGION_SUFFIX', '-eu')

        cfg = DeployEnvConfig.parse_from_env()

        assert cfg.storage.data_plane_storage_root == (
            's3://kumo-prod-eu-dataplane-bucket/prod/')

    def test_kumo_home_placeholder_uses_repo_cache(self, monkeypatch, tmp_path):
        _clear_env(monkeypatch)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('DEPLOY_STAGE', 'dev')
        monkeypatch.setenv('INSTANCE_ID', 'abc123')
        monkeypatch.setenv(
            'DEPLOY_ENV_CONFIG_JSON',
            json.dumps({'storage': {
                'data_plane_storage_root': '{kumo_home}'
            }}))

        with patch('getpass.getuser', return_value='testuser'):
            cfg = DeployEnvConfig.parse_from_env()

        repo_root = Path(__file__).resolve().parents[2]
        expected = str(repo_root / '.cache' / 'kumo')
        assert cfg.storage.data_plane_storage_root == expected
        assert cfg.storage.control_plane_storage_root == expected

    def test_non_kumo_home_template_does_not_resolve_home(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv('DEPLOY_STAGE', 'dev')
        monkeypatch.setenv('INSTANCE_ID', 'abc123')
        monkeypatch.setenv(
            'DEPLOY_ENV_CONFIG_JSON',
            json.dumps({
                'storage': {
                    'data_plane_storage_root': 's3://bucket/{instance_id}/'
                }
            }))

        with patch('getpass.getuser', return_value='testuser'), patch(
                'service_framework.deploy_env_config.get_default_home_dir'
        ) as get_home_dir:
            cfg = DeployEnvConfig.parse_from_env()

        get_home_dir.assert_not_called()
        assert cfg.storage.data_plane_storage_root == (
            's3://bucket/testuser-abc123/')


class TestIsDataplanePath:
    """``is_dataplane_path`` is the activity-boundary contract check.
    Bugs here surface only at integration time, so unit-cover the URI
    scheme + prefix-comparison semantics explicitly.
    """
    def _cfg(self, root: str) -> DeployEnvConfig:
        return DeployEnvConfig(
            deploy_mode=DeployStage.PROD,
            instance_id='prod',
            storage=FileStorageConfig(
                data_plane_storage_root=root,
                control_plane_storage_root=root,
            ),
        )

    def test_exact_root_match(self):
        cfg = self._cfg('s3://example-bucket/jobs/')
        assert cfg.is_dataplane_path('s3://example-bucket/jobs')
        assert cfg.is_dataplane_path('s3://example-bucket/jobs/')

    def test_path_under_root(self):
        cfg = self._cfg('s3://example-bucket/jobs/')
        assert cfg.is_dataplane_path(
            's3://example-bucket/jobs/job-1/train_table/x.parquet')

    def test_path_outside_root_rejected(self):
        cfg = self._cfg('s3://example-bucket/jobs/')
        assert not cfg.is_dataplane_path(
            's3://example-other-bucket/jobs/job-1/x.parquet')
        assert not cfg.is_dataplane_path('/mnt/localssd/edge_persist/job-1/x')
        assert not cfg.is_dataplane_path('/tmp/staging_abc/tt.parquet')

    def test_scheme_must_match(self):
        # Post-#28324 PySpark is removed and the codebase no longer
        # emits s3a:// paths. is_dataplane_path is a pure prefix check;
        # scheme mismatches are rejected (no implicit s3a<->s3
        # normalization).
        cfg = self._cfg('s3://example-bucket/jobs/')
        assert not cfg.is_dataplane_path('s3a://example-bucket/jobs/job-1/x')

    def test_different_bucket_rejected_regardless_of_scheme(self):
        # s3 vs s3a normalization must NOT mask a real bucket mismatch.
        cfg = self._cfg('s3://example-bucket/jobs/')
        assert not cfg.is_dataplane_path('s3a://other-bucket/jobs/x')

    def test_local_root_with_local_path(self):
        cfg = self._cfg('/var/lib/kumo/test/')
        assert cfg.is_dataplane_path('/var/lib/kumo/test/job-1/x.parquet')
        assert not cfg.is_dataplane_path('/var/lib/kumo/other/job-1/x')
        assert not cfg.is_dataplane_path('s3://example-bucket/jobs/x')


# ===================================================================
# Section 4: parse_from_env Behaviors
# ===================================================================


class TestParseFromEnv:
    """Test parse_from_env with various env var combinations."""
    def test_default_dev(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv('DEPLOY_STAGE', 'dev')
        monkeypatch.setenv('INSTANCE_ID', 'test')

        cfg = DeployEnvConfig.parse_from_env(app_name='my-app')

        assert cfg.client_id == 'my-app'
        assert cfg.deploy_mode == DeployStage.DEV
        assert isinstance(cfg.data_compute, DiskgraphConfig)
        assert isinstance(cfg.orchestration, TemporalCloudConfig)
        username = getpass.getuser()
        assert cfg.instance_id == f'{username}-test'
        assert cfg.orchestration.namespace == DEVTEST_NAMESPACE

    def test_prod_defaults(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv('DEPLOY_STAGE', 'prod')
        monkeypatch.setenv('TEMPORAL_SERVER', 'cloud')

        cfg = DeployEnvConfig.parse_from_env()

        assert cfg.deploy_mode == DeployStage.PROD
        assert cfg.instance_id == 'prod'
        assert cfg.orchestration.namespace == 'prod'

    def test_staging_instance_id_override(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv('DEPLOY_STAGE', 'staging')
        monkeypatch.setenv('INSTANCE_ID', 'should-be-overridden')

        cfg = DeployEnvConfig.parse_from_env()

        assert cfg.instance_id == 'prod'

    def test_dev_local_with_localhost_temporal(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv('DEPLOY_STAGE', 'dev_local')
        monkeypatch.setenv('TEMPORAL_SERVER', 'localhost')
        monkeypatch.setenv('INSTANCE_ID', 'test')

        cfg = DeployEnvConfig.parse_from_env()

        assert cfg.deploy_mode == DeployStage.DEV_LOCAL
        assert isinstance(cfg.orchestration, CustomTemporalConfig)
        assert cfg.orchestration.server_address == '127.0.0.1:7233'
        assert cfg.orchestration.namespace == DEVTEST_NAMESPACE

    def test_custom_temporal_namespace(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv('DEPLOY_STAGE', 'dev')
        monkeypatch.setenv('INSTANCE_ID', 'test')
        monkeypatch.setenv('TEMPORAL_NAMESPACE', 'custom-ns')

        cfg = DeployEnvConfig.parse_from_env()

        assert cfg.orchestration.namespace == 'custom-ns'
        assert cfg.temporal_namespace == 'custom-ns.put0h'

    def test_deploy_mode_argument(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv('INSTANCE_ID', 'test')

        cfg = DeployEnvConfig.parse_from_env(deploy_mode=DeployStage.DEV)

        assert cfg.deploy_mode == DeployStage.DEV


# ===================================================================
# Section 5: Error Cases
# ===================================================================


class TestErrors:
    """Validation and error handling."""

    # -- Direct construction warnings --

    def test_dev_local_warns_cloud_temporal(self):
        with pytest.warns(UserWarning, match='misconfiguration'):
            cfg = DeployEnvConfig(
                deploy_mode=DeployStage.DEV_LOCAL,
                orchestration=TemporalCloudConfig(),
            )
        assert cfg.deploy_mode == DeployStage.DEV_LOCAL

    # -- load_raw_deploy_config errors --

    def test_no_structured_config_returns_none(self, monkeypatch):
        _clear_env(monkeypatch)
        assert load_raw_deploy_config() is None

    def test_file_and_json_cannot_coexist(self, monkeypatch, tmp_path):
        _clear_env(monkeypatch)
        config_file = tmp_path / 'cfg.json'
        config_file.write_text('{}')
        monkeypatch.setenv('DEPLOY_ENV_CONFIG_FILE', str(config_file))
        monkeypatch.setenv('DEPLOY_ENV_CONFIG_JSON', '{}')
        with pytest.raises(ValueError, match='cannot both be set'):
            load_raw_deploy_config()

    def test_invalid_preset(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv('PLATFORM_PRESET', 'azure')
        with pytest.raises(ValueError):
            load_raw_deploy_config()

    def test_missing_required_field_in_sub_config(self, monkeypatch):
        # CustomTemporalConfig requires `server_address` (no default).
        # Omitting it should raise.
        _clear_env(monkeypatch)
        bad = {
            'data_compute': {
                'type': 'DISKGRAPH'
            },
            'orchestration': {
                'type': 'custom'
            },
            'secrets': {
                'backend': 'k8s_mount'
            },
            'storage': {},
            'logging': {},
        }
        monkeypatch.setenv('DEPLOY_ENV_CONFIG_JSON', json.dumps(bad))
        with pytest.raises(Exception):
            load_raw_deploy_config()

    def test_unknown_discriminator_value(self, monkeypatch):
        _clear_env(monkeypatch)
        bad = _load_fixture('raw_aws_minimal.json')
        bad['data_compute'] = {'type': 'NONEXISTENT'}
        monkeypatch.setenv('DEPLOY_ENV_CONFIG_JSON', json.dumps(bad))
        with pytest.raises(Exception):
            load_raw_deploy_config()

    # -- parse_from_env errors --

    def test_dev_local_warns_cloud_temporal_env(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv('DEPLOY_STAGE', 'dev_local')
        monkeypatch.setenv('TEMPORAL_SERVER', 'cloud')
        monkeypatch.setenv('INSTANCE_ID', 'test')

        with pytest.warns(UserWarning, match='misconfiguration'):
            cfg = DeployEnvConfig.parse_from_env()
        assert cfg.deploy_mode == DeployStage.DEV_LOCAL


class TestLoggingOutputFormat:
    def test_platform_defaults(self):
        assert (PLATFORM_DEFAULTS[PlatformPreset.AWS].logging.output_format ==
                LoggingOutputFormat.TEXT)
        assert (PLATFORM_DEFAULTS[PlatformPreset.GCP].logging.output_format ==
                LoggingOutputFormat.TEXT)


# ===================================================================
# Section 6: Repr
# ===================================================================


class TestRepr:
    def test_repr(self):
        cfg = DeployEnvConfig(
            deploy_mode=DeployStage.DEV,
            instance_id='user-inst',
        )
        r = repr(cfg)
        assert r.startswith('DeployEnvConfig{')
        assert '"deploy_mode": "dev"' in r
        assert '"instance_id": "user-inst"' in r


# ===================================================================
# Section 7: Singleton accessor
# ===================================================================


class TestFrozen:
    def test_immutable(self):
        """DeployEnvConfig instances cannot be mutated after construction."""
        cfg = DeployEnvConfig(
            deploy_mode=DeployStage.DEV,
            instance_id='test-inst',
        )
        with pytest.raises(FrozenInstanceError):
            cfg.instance_id = 'other'
