# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bridge from legacy scattered env vars to RawDeployConfig.

The single public entry point is :class:`ConfigMigrationHelper`.
"""

import os
from typing import Optional

from service_framework._raw_deploy_config import (
    AzureStorageConfig,
    CustomTemporalConfig,
    DiskgraphConfig,
    FileStorageConfig,
    LoggingBackend,
    LoggingConfig,
    OrchestrationConfig,
    RawDeployConfig,
    SecretBackend,
    SecretConfig,
    TemporalCloudConfig,
)
from service_framework.deploy_stage import DeployStage
from service_framework.env_var import EnvVar

_IS_GCP_DEPLOY = EnvVar('IS_GCP_DEPLOY', bool, default_value=False)
_DISABLE_REMOTE_INGESTION = EnvVar('DISABLE_REMOTE_INGESTION', bool,
                                   default_value=True)


class ConfigMigrationHelper:
    """Converts legacy scattered env vars into a :class:`RawDeployConfig`.

    Takes the resolved deployment mode and instance ID so legacy environment
    variables can be translated without carrying the retired SaaS tenant key.

    The resulting RawDeployConfig is *sparse* — storage roots may be None
    when no custom bucket is set.  Final resolution of None values happens
    in ``DeployEnvConfig.parse_from_env()``.
    """
    def __init__(self, deploy_mode: DeployStage, instance_id: str) -> None:
        self._deploy_mode = deploy_mode
        self._instance_id = instance_id

    def build(self) -> RawDeployConfig:
        """Build a sparse :class:`RawDeployConfig` from legacy env vars."""
        data_compute = self._resolve_data_compute()
        return RawDeployConfig(
            data_compute=data_compute,
            orchestration=self._resolve_orchestration(),
            secrets=self._resolve_secrets(),
            storage=self._resolve_storage(),
            logging=self._resolve_logging(),
            azure_storage=self._resolve_azure_storage(),
        )

    # ------------------------------------------------------------------
    # Private resolvers
    # ------------------------------------------------------------------

    def _resolve_data_compute(self) -> DiskgraphConfig:
        return DiskgraphConfig()

    def _resolve_azure_storage(self) -> 'Optional[AzureStorageConfig]':
        """Read Azure SP credentials for ADLS access from env vars.

        Returns None when any credential is absent — all four are required
        to configure the Hadoop ABFS connector.
        """
        client_id = os.environ.get('AZURE_CLIENT_ID') or None
        client_secret = os.environ.get('AZURE_CLIENT_SECRET') or None
        tenant_id = os.environ.get('AZURE_TENANT_ID') or None
        storage_account = os.environ.get('AZURE_STORAGE_ACCOUNT') or None
        if client_id and client_secret and tenant_id and storage_account:
            return AzureStorageConfig(
                client_id=client_id,
                client_secret=client_secret,
                tenant_id=tenant_id,
                storage_account=storage_account,
            )
        return None

    def _resolve_orchestration(self) -> OrchestrationConfig:
        default = ('localhost' if self._deploy_mode
                   in (DeployStage.DEV_LOCAL, DeployStage.TEST) else 'cloud')
        temporal_server = os.environ.get('TEMPORAL_SERVER', default).lower()
        namespace = self._resolve_temporal_namespace()

        if temporal_server == 'cloud':
            return TemporalCloudConfig(namespace=namespace)

        # localhost or any non-cloud value → custom / self-hosted
        address = os.environ.get('TEMPORAL_ADDRESS', '')
        if address:
            server_address = address if ':' in address else address + ':7233'
        else:
            server_address = '127.0.0.1:7233'

        tls_cert_dir = os.environ.get('TEMPORAL_CERT_DIR', '') or None
        return CustomTemporalConfig(
            server_address=server_address,
            namespace=namespace,
            tls_cert_dir=tls_cert_dir,
        )

    def _resolve_temporal_namespace(self) -> str:
        """Resolve temporal namespace from TEMPORAL_NAMESPACE env var or
        deploy stage default."""
        from service_framework.deploy_env import DEVTEST_NAMESPACE

        custom_ns = os.environ.get('TEMPORAL_NAMESPACE', '')
        if custom_ns:
            return custom_ns

        defaults = {
            DeployStage.PROD: 'prod',
            DeployStage.STAGING: 'staging',
            DeployStage.PREPROD: 'preprod',
            DeployStage.DEV: DEVTEST_NAMESPACE,
            DeployStage.DEV_LOCAL: DEVTEST_NAMESPACE,
            DeployStage.TEST: DEVTEST_NAMESPACE,
        }
        return defaults.get(self._deploy_mode, DEVTEST_NAMESPACE)

    def _resolve_secrets(self) -> SecretConfig:
        is_gcp = _IS_GCP_DEPLOY.parse_value()

        if is_gcp:
            return SecretConfig(backend=SecretBackend.GCP_SM)
        return SecretConfig(backend=SecretBackend.AWS_SM)

    def _resolve_storage(self) -> FileStorageConfig:
        """Read custom bucket/persist dir env vars.  Returns sparse config
        — None values are resolved later by DeployEnvConfig."""
        bucket = os.environ.get('CUSTOM_S3_DATA_PLANE_BUCKET', '') or None
        persist_dir = os.environ.get('CUSTOM_REST_PERSIST_DIR', '') or None
        artifact_root = os.environ.get('CUSTOM_ARTIFACT_ROOT', '') or None
        return FileStorageConfig(
            data_plane_storage_root=bucket,
            control_plane_storage_root=persist_dir,
            artifact_root=artifact_root,
        )

    def _resolve_logging(self) -> LoggingConfig:
        is_gcp = _IS_GCP_DEPLOY.parse_value()

        if is_gcp:
            return LoggingConfig(backend=LoggingBackend.CLOUD_LOGGING)
        return LoggingConfig(backend=LoggingBackend.CLOUDWATCH)
