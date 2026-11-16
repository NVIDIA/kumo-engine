# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Centralized deployment environment configuration (v2).

Public API for application code. Mirrors the RawDeployConfig sub-config
structure with deployment identity fields that are orthogonal to platform
configuration.

Application code should import from this module::

    from service_framework.deploy_env_config import DeployEnvConfig

Migration from the legacy ``DeployEnvConfig`` in ``deploy_env.py`` is
done incrementally — callsites switch one at a time in subsequent PRs.
"""

import dataclasses
import getpass
import json
import logging
import os
import os.path as osp
import sqlite3
import warnings
from dataclasses import asdict, field
from typing import Optional
from urllib.parse import urlparse

from pydantic import Field, ValidationInfo, field_validator
from pydantic.dataclasses import dataclass

from service_framework._config_migration import ConfigMigrationHelper
from service_framework._raw_deploy_config import (
    AmbientGCPWorkloadConfig,
    AmbientS3WorkloadConfig,
    AuthConfig,
    AzureStorageConfig,
    CustomTemporalConfig,
    DiskgraphConfig,
    FileStorageConfig,
    LoggingBackend,
    LoggingConfig,
    MetadataStorageConfig,
    OrchestrationConfig,
    SecretBackend,
    SecretConfig,
    SqliteMetadataStorageConfig,
    TemporalCloudConfig,
    load_raw_deploy_config,
)
from service_framework.deploy_env import (
    DEVTEST_NAMESPACE,
    MACHINE_INSTANCE_ID,
    K8sDeployEnvLabel,
    get_default_home_dir,
)
from service_framework.deploy_stage import _DEPLOY_STAGE, DeployStage
from service_framework.env_var import EnvVar

log = logging.getLogger(__name__)

# Identity env vars — not part of RawDeployConfig, read at construction.
_INSTANCE_ID = EnvVar('INSTANCE_ID', str, required=False, default_value='')
_REGION_SUFFIX = EnvVar('REGION_SUFFIX', str, required=False, default_value='')
_DEVSERVER_INSTANCE_SUFFIX = EnvVar('DEVSERVER_INSTANCE_SUFFIX', str,
                                    required=False, default_value='')


def _append_devserver_suffix(path: str, suffix: str) -> str:
    """Append the child suffix once; a child's KUMO_HOME is already
    child-scoped, so a {kumo_home} template must not double it.
    """
    normalized_suffix = '/' + suffix.strip('/')
    root = path.rstrip('/')
    if root.endswith(normalized_suffix):
        return root
    return root + normalized_suffix


# Default S3 root template (AWS). GCP and other platforms should set
# CUSTOM_S3_DATA_PLANE_BUCKET explicitly or use PLATFORM_PRESET.
_S3_ROOT_TEMPLATE = ('s3://kumo-{deploy_mode}{region_suffix}-dataplane-bucket/'
                     '{instance_id}/')


def _resolve_storage_roots(
    storage: FileStorageConfig,
    deploy_mode: DeployStage,
    instance_id: str,
    is_legacy: bool,
) -> FileStorageConfig:
    """Resolve storage root paths using identity fields.

    When ``is_legacy`` is True (config came from legacy env vars),
    the data_plane_storage_root may contain ``{instance_id}`` templates
    or need PREPROD implicit appending.  Structured configs from infra
    are used as-is when present.

    Absent (None) roots are always derived from the S3 root template
    or local home dir.
    """
    data_root = storage.data_plane_storage_root
    control_root = storage.control_plane_storage_root
    artifact_root = storage.artifact_root
    devserver_suffix = _DEVSERVER_INSTANCE_SUFFIX.parse_value()

    if data_root is not None:
        # Expand single-tenant storage path templates.
        try:
            format_args: dict[str, object] = {
                'instance_id':
                instance_id,
                'devserver_instance_suffix':
                _DEVSERVER_INSTANCE_SUFFIX.parse_value(),
            }
            if '{kumo_home}' in data_root:
                format_args['kumo_home'] = get_default_home_dir()
            expanded = data_root.format(**format_args)
        except KeyError as e:
            raise KeyError(
                f'data_plane_storage_root contains unsupported '
                f'placeholder: {e}; '
                f'only {{instance_id}}, {{devserver_instance_suffix}}, '
                f'and {{kumo_home}} '
                f'are supported') from e
        if expanded != data_root:
            data_root = expanded
        elif is_legacy and deploy_mode == DeployStage.PREPROD:
            # Legacy: PREPROD configs not yet migrated to
            # {instance_id} placeholder get instance_id appended.
            data_root = osp.join(data_root, instance_id)
    else:
        if deploy_mode in (DeployStage.TEST, DeployStage.DEV_LOCAL):
            data_root = get_default_home_dir()
        else:
            region_suffix = _REGION_SUFFIX.parse_value()
            data_root = _S3_ROOT_TEMPLATE.format(
                deploy_mode=deploy_mode.value,
                region_suffix=region_suffix,
                instance_id=instance_id,
            )

    if devserver_suffix:
        data_root = _append_devserver_suffix(data_root, devserver_suffix)

    if control_root is None:
        control_root = data_root
    else:
        control_root = osp.join(control_root, instance_id)

    if artifact_root is not None:
        try:
            format_args = {
                'instance_id':
                instance_id,
                'devserver_instance_suffix':
                _DEVSERVER_INSTANCE_SUFFIX.parse_value(),
            }
            if '{kumo_home}' in artifact_root:
                format_args['kumo_home'] = get_default_home_dir()
            artifact_root = artifact_root.format(**format_args)
        except KeyError as e:
            raise KeyError(
                f'artifact_root contains unsupported placeholder: {e}; '
                f'only {{instance_id}}, {{devserver_instance_suffix}}, '
                f'and {{kumo_home}} '
                f'are supported') from e
        if devserver_suffix:
            artifact_root = _append_devserver_suffix(
                artifact_root,
                devserver_suffix,
            )

    # Expand ~ so downstream code (including non-Python like diskgraph Rust)
    # always sees absolute paths.
    data_root = osp.expanduser(data_root)
    control_root = osp.expanduser(control_root)
    if artifact_root is not None:
        artifact_root = osp.expanduser(artifact_root)

    return FileStorageConfig(
        data_plane_storage_root=data_root,
        control_plane_storage_root=control_root,
        artifact_root=artifact_root,
        lifecycle_ready=storage.lifecycle_ready,
        s3=storage.s3,
    )


def _resolve_data_compute_roots(
    data_compute: DiskgraphConfig,
    instance_id: str,
) -> DiskgraphConfig:
    """Resolve local path templates embedded in data-compute config."""
    def resolve_path(path: Optional[str], field_name: str) -> Optional[str]:
        if path is None:
            return None
        try:
            format_args: dict[str, object] = {
                'instance_id':
                instance_id,
                'devserver_instance_suffix':
                _DEVSERVER_INSTANCE_SUFFIX.parse_value(),
            }
            if '{kumo_home}' in path:
                format_args['kumo_home'] = get_default_home_dir()
            resolved = path.format(**format_args)
        except KeyError as e:
            raise KeyError(
                f'{field_name} contains unsupported placeholder: {e}; '
                f'only {{instance_id}}, {{devserver_instance_suffix}}, '
                f'and {{kumo_home}} '
                f'are supported') from e
        return osp.expanduser(resolved)

    devserver_suffix = _DEVSERVER_INSTANCE_SUFFIX.parse_value()
    data_root = resolve_path(data_compute.data_root, 'data_compute.data_root')
    local_ingest_root = resolve_path(data_compute.local_ingest_root,
                                     'data_compute.local_ingest_root')
    if devserver_suffix:
        if data_root is not None:
            data_root = _append_devserver_suffix(data_root, devserver_suffix)
        if local_ingest_root is not None:
            local_ingest_root = _append_devserver_suffix(
                local_ingest_root,
                devserver_suffix,
            )
    return dataclasses.replace(data_compute, data_root=data_root,
                               local_ingest_root=local_ingest_root)


def _resolve_secret_mount_path(
    secrets: SecretConfig,
    instance_id: str,
) -> SecretConfig:
    """Resolve local path templates embedded in secret mount config."""
    mount_path = secrets.mount_path
    if mount_path is None:
        return secrets
    try:
        format_args: dict[str, object] = {
            'instance_id': instance_id,
            'devserver_instance_suffix':
            _DEVSERVER_INSTANCE_SUFFIX.parse_value(),
        }
        if '{kumo_home}' in mount_path:
            format_args['kumo_home'] = get_default_home_dir()
        resolved = mount_path.format(**format_args)
    except KeyError as e:
        raise KeyError(
            f'secrets.mount_path contains unsupported placeholder: {e}; '
            f'only {{instance_id}}, {{devserver_instance_suffix}}, '
            f'and {{kumo_home}} '
            f'are supported') from e
    return dataclasses.replace(secrets, mount_path=osp.expanduser(resolved))


@dataclass(frozen=True)
class DeployEnvConfig:
    """Centralized deployment environment configuration (v2).

    Mirrors the RawDeployConfig sub-config structure with additional
    deployment identity fields that are orthogonal to platform config.

    Sub-config fields are typed config objects from :mod:`_raw_deploy_config`.
    Derived properties expose convenience values computed from the sub-configs
    and identity fields.
    """

    # ------------------------------------------------------------------
    # Deployment identity (orthogonal to platform config)
    # ------------------------------------------------------------------

    client_id: str = ''
    deploy_mode: DeployStage = DeployStage.DEV
    instance_id: str = ''

    # ------------------------------------------------------------------
    # Platform config (mirrors RawDeployConfig)
    # ------------------------------------------------------------------

    data_compute: DiskgraphConfig = Field(default_factory=DiskgraphConfig)
    orchestration: OrchestrationConfig = Field(
        default_factory=TemporalCloudConfig, discriminator='type')
    secrets: SecretConfig = field(
        default_factory=lambda: SecretConfig(backend=SecretBackend.AWS_SM))
    storage: FileStorageConfig = field(default_factory=FileStorageConfig)
    metadata_storage: MetadataStorageConfig = Field(
        default_factory=SqliteMetadataStorageConfig, discriminator='type')
    logging_config: LoggingConfig = field(default_factory=lambda: LoggingConfig(
        backend=LoggingBackend.CLOUDWATCH))
    azure_storage: Optional[AzureStorageConfig] = None
    auth: Optional[AuthConfig] = None
    ambient_s3: Optional[AmbientS3WorkloadConfig] = None
    ambient_gcp: Optional[AmbientGCPWorkloadConfig] = None
    # Externally reachable host of this deployment (no scheme).
    external_host: Optional[str] = None

    # ------------------------------------------------------------------
    # Validators (structural invariants for any DeployEnvConfig instance)
    # ------------------------------------------------------------------

    @field_validator('orchestration')
    @classmethod
    def _dev_local_requires_local_temporal(cls, v, info: ValidationInfo):
        mode = info.data.get('deploy_mode')
        if mode == DeployStage.DEV_LOCAL and isinstance(v, TemporalCloudConfig):
            warnings.warn(
                'DEV_LOCAL with TemporalCloudConfig — this is likely a '
                'misconfiguration. Set TEMPORAL_SERVER=localhost.',
                stacklevel=2)
        return v

    @field_validator('storage')
    @classmethod
    def _production_s3_requires_https(cls, value: FileStorageConfig,
                                      info: ValidationInfo):
        s3 = value.s3
        mode = info.data.get('deploy_mode')
        if (s3 is not None and urlparse(s3.endpoint_url).scheme != 'https'
                and mode in (DeployStage.PROD, DeployStage.STAGING,
                             DeployStage.PREPROD)):
            raise ValueError(
                'storage.s3.endpoint_url must use https in production '
                'deployment stages')
        return value

    # ------------------------------------------------------------------
    # Construction from env
    # ------------------------------------------------------------------

    @classmethod
    def parse_from_env(
        cls,
        app_name: str = '',
        deploy_mode: Optional[DeployStage] = None,
    ) -> 'DeployEnvConfig':
        """Construct from environment variables.

        1. Resolve identity (deploy_mode and instance_id).
        2. Load structured config (PLATFORM_PRESET / JSON) or use
           ConfigMigrationHelper to bridge legacy env vars.
        3. Resolve storage roots using identity fields.
        4. Return fully resolved DeployEnvConfig.
        """
        # 1. Identity
        resolved_mode = _DEPLOY_STAGE.parse_value(default=deploy_mode)
        instance_id = _INSTANCE_ID.parse_value()

        if resolved_mode in (DeployStage.PROD, DeployStage.STAGING):
            instance_id = 'prod'
        else:
            instance_id = instance_id or MACHINE_INSTANCE_ID
            if resolved_mode.is_dev:
                username = getpass.getuser()
                instance_id = f'{username}-{instance_id}'

        # 2. Platform config
        raw = load_raw_deploy_config()
        is_legacy = raw is None
        if is_legacy:
            helper = ConfigMigrationHelper(resolved_mode, instance_id)
            raw = helper.build()
        else:
            temporal_address = os.environ.get('TEMPORAL_ADDRESS', '')
            if temporal_address and isinstance(raw.orchestration,
                                               CustomTemporalConfig):
                server_address = (temporal_address if ':' in temporal_address
                                  else temporal_address + ':7233')
                raw = dataclasses.replace(
                    raw,
                    orchestration=dataclasses.replace(
                        raw.orchestration,
                        server_address=server_address,
                    ),
                )

        # 3. Storage resolution
        storage = _resolve_storage_roots(raw.storage, resolved_mode,
                                         instance_id, is_legacy)
        data_compute = _resolve_data_compute_roots(raw.data_compute,
                                                   instance_id)
        secrets = _resolve_secret_mount_path(raw.secrets, instance_id)

        # 4. Construct
        # Resolve orchestration namespace if not set in config.
        orchestration = raw.orchestration
        if not orchestration.namespace:
            _NS_DEFAULTS = {
                DeployStage.PROD: 'prod',
                DeployStage.STAGING: 'staging',
                DeployStage.PREPROD: 'preprod',
            }
            ns = _NS_DEFAULTS.get(resolved_mode, DEVTEST_NAMESPACE)
            orchestration = dataclasses.replace(orchestration, namespace=ns)

        return cls(
            client_id=app_name,
            deploy_mode=resolved_mode,
            instance_id=instance_id,
            data_compute=data_compute,
            orchestration=orchestration,
            secrets=secrets,
            storage=storage,
            metadata_storage=raw.metadata_storage,
            logging_config=raw.logging,
            azure_storage=raw.azure_storage,
            auth=raw.auth,
            ambient_s3=raw.ambient_s3,
            ambient_gcp=raw.ambient_gcp,
            external_host=raw.external_host,
        )

    @classmethod
    def for_test(
        cls,
        temporal_target_host: Optional[str] = None,
    ) -> 'DeployEnvConfig':
        """Create a DeployEnvConfig for testing.

        Creates a test SQLite database (``kumo_rest_test.db``) in the cwd
        if absent, and installs the resulting config as the process-global
        ``_deploy_env`` singleton so subsequent ``deploy_env()`` callers
        see the same object.
        """
        config = cls.parse_from_env(deploy_mode=DeployStage.TEST)
        assert config.deploy_mode == DeployStage.TEST

        if temporal_target_host is not None and isinstance(
                config.orchestration, CustomTemporalConfig):
            config = dataclasses.replace(
                config,
                orchestration=dataclasses.replace(
                    config.orchestration,
                    server_address=temporal_target_host,
                ),
            )

        # Create SQLite test database if it doesn't exist
        test_db_path = os.path.join(os.getcwd(), 'kumo_rest_test.db')
        if not os.path.exists(test_db_path):
            conn = sqlite3.connect(test_db_path)
            conn.close()
            log.info(f"Created SQLite test database at {test_db_path}")

        # Install as the process-global singleton so subsequent
        # deploy_env() callers observe the same object.
        global _deploy_env
        _deploy_env = config

        return config

    # ------------------------------------------------------------------
    # Schema migration
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Convenience properties (derived from sub-configs + identity)
    # ------------------------------------------------------------------

    @property
    def data_plane_storage_root(self) -> str:
        """Resolved data-plane storage root (guaranteed non-None after
        construction via parse_from_env)."""
        root = self.storage.data_plane_storage_root
        assert root is not None, 'data_plane_storage_root not resolved'
        return root

    @property
    def control_plane_storage_root(self) -> str:
        """Resolved control-plane storage root (guaranteed non-None after
        construction via parse_from_env)."""
        root = self.storage.control_plane_storage_root
        assert root is not None, 'control_plane_storage_root not resolved'
        return root

    @property
    def pool_state_path(self) -> str:
        """Path to the diskgraph pool controller's persisted state file
        (``pool_state.json``) under the dataplane storage root."""
        return osp.join(self.data_plane_storage_root, 'pool', 'pool_state.json')

    def is_dataplane_path(self, path: str) -> bool:
        """Return True iff ``path`` lives under ``data_plane_storage_root``."""
        root = self.data_plane_storage_root.rstrip('/')
        if not root:
            return False
        return path == root or path.startswith(root + '/')

    @property
    def is_diskgraph(self) -> bool:
        return isinstance(self.data_compute, DiskgraphConfig)

    @property
    def temporal_namespace(self) -> str:
        """Resolved Temporal namespace.

        For Temporal Cloud, appends the Cloud account suffix (.put0h).
        The base namespace is always resolved at construction time in
        parse_from_env() — orchestration.namespace is never None after
        construction.
        """
        ns = self.orchestration.namespace or ''
        if isinstance(self.orchestration, TemporalCloudConfig):
            return ns + '.put0h'
        return ns

    @property
    def temporal_server_address(self) -> str:
        if isinstance(self.orchestration, TemporalCloudConfig):
            return self.temporal_namespace + '.tmprl.cloud:7233'
        return self.orchestration.server_address

    @property
    def tls_cert_dir(self) -> str:
        return getattr(self.orchestration, 'tls_cert_dir', '') or ''

    @property
    def web_url(self) -> str:
        if isinstance(self.orchestration, TemporalCloudConfig):
            return ('https://cloud.temporal.io/namespaces/'
                    f'{self.temporal_namespace}/workflows'
                    f'?search=advanced&query=instance_id%3D"{self.instance_id}"'
                    '+and+rpc_service+is+null')
        return 'localhost:7233'

    @property
    def k8s_namespace(self) -> str:
        r"""Namespace used for K8s jobs launched by the REST API.

        Prefers the POD_NAMESPACE downward-API env var (set by the OSS/BYOC
        Helm chart, which deploys everything into the release namespace);
        falls back to a stable single-tenant namespace when unset.
        """
        pod_namespace = os.environ.get('POD_NAMESPACE')
        if pod_namespace:
            return pod_namespace
        if self.deploy_mode in [
                DeployStage.PROD,
                DeployStage.PREPROD,
                DeployStage.STAGING,
        ]:
            return self.deploy_mode.value
        return 'kumo-dataplane-devtest'

    @property
    def k8s_deploy_env_label(self) -> K8sDeployEnvLabel:
        r"""Label for K8s jobs for fine-grained visibility."""
        if self.deploy_mode in [DeployStage.PROD, DeployStage.STAGING]:
            return K8sDeployEnvLabel(self.deploy_mode.value)
        if self.deploy_mode == DeployStage.PREPROD:
            return K8sDeployEnvLabel(
                f'{self.deploy_mode.value}-{self.instance_id}')
        return K8sDeployEnvLabel(f'{self.deploy_mode.value}-{self.instance_id}')

    @property
    def id_suffix(self) -> str:
        if self.deploy_mode in (DeployStage.STAGING, DeployStage.PROD):
            return f'-{self.instance_id}'
        elif (self.deploy_mode == DeployStage.PREPROD
              or self.deploy_mode.is_dev):
            return f'-{self.instance_id}'
        else:
            return f'-{self.instance_id}-test'

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def task_queue(self, task_queue: str) -> str:
        return self._append_suffix_to_id(task_queue)

    def workflow_id(self, workflow_id: str) -> str:
        return self._append_suffix_to_id(workflow_id)

    def strip_suffix(self, id: str) -> str:
        suffix = self.id_suffix
        return id if not suffix or not id.endswith(
            suffix) else id[:-len(suffix)]

    def _append_suffix_to_id(self, id: str) -> str:
        suffix = self.id_suffix
        return id if id.endswith(suffix) else id + suffix

    def __repr__(self) -> str:
        return 'DeployEnvConfig' + json.dumps(asdict(self), indent=2,
                                              default=str)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_deploy_env: Optional[DeployEnvConfig] = None


def deploy_env() -> DeployEnvConfig:
    """Return the process-global DeployEnvConfig singleton.

    On first call, lazily constructs via ``DeployEnvConfig.parse_from_env()``.
    """
    global _deploy_env
    if _deploy_env is None:
        _deploy_env = DeployEnvConfig.parse_from_env()
        # Announce so each process self-reports its resolved config —
        # debug aid for cross-process mismatches (entrypoints must
        # initialize_logging() first, else the record is dropped).
        log.info(
            'deploy_env() initialized: deploy_mode=%s instance_id=%s '
            'id_suffix=%s data_plane_storage_root=%s '
            'orchestration=%s', _deploy_env.deploy_mode.value,
            _deploy_env.instance_id, _deploy_env.id_suffix,
            _deploy_env.data_plane_storage_root,
            type(_deploy_env.orchestration).__name__)
    return _deploy_env


def _reset_deploy_env() -> None:  # pyright: ignore[reportUnusedFunction]
    """Reset the singleton to ``None`` (for tests only)."""
    global _deploy_env
    _deploy_env = None
