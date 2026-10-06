# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Centralized deployment environment configuration types.

Architecture
------------
The config system has two layers:

1. **RawDeployConfig** (this module) — the structured config that lives at
   the boundary between infrastructure and application code.  Infra produces
   it (via JSON file, env vars, or PLATFORM_PRESET defaults); application
   code consumes it only through DeployEnvConfig.

2. **DeployEnvConfig** (deploy_env.py) — wraps a RawDeployConfig, adds
   derived / inferred fields (e.g. k8s_namespace, persist_dir templates),
   cross-group validation, and the public API that application code imports.

Application code should never import or depend on RawDeployConfig directly.
"""

import dataclasses
import re
from decimal import Decimal, DecimalException
from enum import Enum
from typing import Dict, Literal, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

from pydantic import ConfigDict, Field, field_validator, model_validator
from pydantic.dataclasses import dataclass

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PlatformPreset(str, Enum):
    """High-level deployment platform.

    Set via the PLATFORM_PRESET env var. Each preset populates a full set of
    sub-config defaults so that a single env var is sufficient to configure a
    new environment.
    """
    AWS = 'aws'
    GCP = 'gcp'


class SecretBackend(str, Enum):
    """Where secrets are stored and retrieved from."""
    K8S_MOUNT = 'k8s_mount'  # Mounted as files by K8s (legacy)
    AWS_SM = 'aws_sm'  # AWS Secrets Manager
    GCP_SM = 'gcp_sm'  # GCP Secret Manager
    ENV_VAR = 'env_var'  # Plain env vars (dev/test)


class LoggingBackend(str, Enum):
    """Where structured logs are shipped."""
    CLOUDWATCH = 'cloudwatch'
    CLOUD_LOGGING = 'cloud_logging'  # GCP Cloud Logging
    SNOWFLAKE = 'snowflake'  # Snowflake event tables
    STDOUT = 'stdout'  # Local / test


class LoggingOutputFormat(str, Enum):
    """Log output serialization format."""
    TEXT = 'text'
    JSON = 'json'


# ---------------------------------------------------------------------------
# Metadata persistence sub-configs (discriminated union)
# ---------------------------------------------------------------------------


class MetadataStorageBackend(str, Enum):
    """How REST/ML metadata state is persisted at runtime."""
    SQLITE = 'sqlite'
    POSTGRES = 'postgres'


@dataclass
class SqliteMetadataStorageConfig:
    """Persist metadata in local SQLite files."""
    type: Literal[MetadataStorageBackend.SQLITE] = MetadataStorageBackend.SQLITE


@dataclass
class PostgresMetadataStorageConfig:
    """Persist metadata in Postgres tables.

    Connection config must be one of:
    - ``dsn``: a literal connection string. Used for local devserver,
      docker-compose, single-host/AMI.
    - ``dsn_secret_name``: the name of a secret resolved at runtime, which
      contains structured Postgres connection fields.
    - ``auth_mode`` plus connection fields: passwordless cloud authentication.
    """
    dsn: Optional[str] = None
    dsn_secret_name: Optional[str] = None
    auth_mode: Optional[Literal['aws_iam', 'azure_entra']] = None
    username: Optional[str] = None
    host: Optional[str] = None
    port: int = 5432
    database: Optional[str] = None
    region: Optional[str] = None
    sslrootcert: Optional[str] = None
    type: Literal[
        MetadataStorageBackend.POSTGRES] = MetadataStorageBackend.POSTGRES

    @model_validator(mode='after')
    def _validate_connection_source(self) -> 'PostgresMetadataStorageConfig':
        sources = sum(
            bool(value) for value in (
                self.dsn,
                self.dsn_secret_name,
                self.auth_mode,
            ))
        if sources != 1:
            raise ValueError(
                'PostgresMetadataStorageConfig requires exactly one of '
                '"dsn", "dsn_secret_name", or "auth_mode".')
        if self.auth_mode:
            missing = [
                name for name in ('username', 'host', 'database')
                if not getattr(self, name)
            ]
            if missing:
                raise ValueError('Passwordless Postgres config is missing: ' +
                                 ', '.join(missing))
            if self.auth_mode == 'aws_iam' and not self.region:
                raise ValueError(
                    'Passwordless AWS Postgres config requires "region".')
        return self


MetadataStorageConfig = Union[SqliteMetadataStorageConfig,
                              PostgresMetadataStorageConfig]

# ---------------------------------------------------------------------------
# Data compute sub-config
# ---------------------------------------------------------------------------


class DataComputeBackend(str, Enum):
    """How data ingestion and transformation workloads are run."""
    DISKGRAPH = 'DISKGRAPH'


@dataclass
class DiskgraphNodeType:
    """Capacity declared by a deployment for one DiskGraph host class.

    The controller does not infer SSD capacity from a cloud instance name or
    maintain a provider catalog. These values must match the storage and GPUs
    provisioned by the deployment. ``gpu_count`` defaults to one because
    single-GPU cache nodes are the standard configuration.
    """
    # Provisioned local SSD/PVC capacity. Placement applies
    # pool_ssd_max_utilization before admitting snapshot bytes.
    ssd_gib: int = Field(gt=0)
    gpu_count: int = Field(default=1, gt=0)  # GPUs per durable cache node


def _pool_resource_quantity(value: str, resource: str) -> Decimal:
    """Validate a positive Kubernetes quantity without importing the K8s SDK."""
    match = re.fullmatch(
        r'([+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))'
        r'([eE][+-]?[0-9]+|[numkMGTPE]|[KMGTPE]i)?', value)
    if not match:
        raise ValueError(f'{resource}: invalid Kubernetes quantity {value!r}')
    number, suffix = match.groups()
    suffix = suffix or ''
    try:
        if len(suffix) > 1 and suffix[0] in 'eE' and suffix != 'Ei':
            quantity = Decimal(value)
        elif suffix.endswith('i'):
            quantity = Decimal(number) * (1024**('KMGTPE'.index(suffix[0]) + 1))
        else:
            exponent = {
                'n': -9,
                'u': -6,
                'm': -3,
                '': 0,
                'k': 3,
                'M': 6,
                'G': 9,
                'T': 12,
                'P': 15,
                'E': 18
            }[suffix]
            quantity = Decimal(number) * (Decimal(10)**exponent)
        if not 0 < quantity <= (2**63 - 1):
            raise ValueError(
                f'{resource}: quantity must be positive and <= 2^63-1')
        if resource == 'cpu' and quantity % Decimal('0.001'):
            raise ValueError('cpu: quantity must be a multiple of 1m')
        return quantity
    except DecimalException as exc:
        raise ValueError(
            f'{resource}: invalid Kubernetes quantity {value!r}') from exc


@dataclass(config=ConfigDict(extra='forbid'))
class PoolPodResources:
    """Optional CPU/memory entries, in Kubernetes requests/limits shape."""
    requests: Dict[Literal['cpu', 'memory'],
                   str] = dataclasses.field(default_factory=dict)
    limits: Dict[Literal['cpu', 'memory'],
                 str] = dataclasses.field(default_factory=dict)

    @field_validator('requests', 'limits', mode='before')
    @classmethod
    def _normalize_numbers(cls, value):
        # YAML commonly uses a number for CPU and a string for memory.
        if isinstance(value, dict):
            return {
                key: str(amount) if type(amount) in (int, float) else amount
                for key, amount in value.items()
            }
        return value

    @model_validator(mode='after')
    def _validate_quantities(self) -> 'PoolPodResources':
        requests = {
            key: _pool_resource_quantity(value, key)
            for key, value in self.requests.items()
        }
        limits = {
            key: _pool_resource_quantity(value, key)
            for key, value in self.limits.items()
        }
        for key in requests.keys() & limits.keys():
            if requests[key] > limits[key]:
                raise ValueError(f'{key} request must not exceed limit')
        return self

    def overlay(self, override: 'PoolPodResources') -> 'PoolPodResources':
        """Merge each resource independently, validating the effective result."""
        return PoolPodResources(requests={
            **self.requests,
            **override.requests
        }, limits={
            **self.limits,
            **override.limits
        })


@dataclass(config=ConfigDict(extra='forbid'))
class PoolResourcesConfig:
    """Daemon defaults and per-activity CPU/memory overrides for K8s pools.

    Empty maps preserve existing behavior. Training/prediction overrides overlay
    activity_defaults per field; GPUs remain controlled by the placement code.
    """
    daemon: PoolPodResources = dataclasses.field(
        default_factory=PoolPodResources)
    activity_defaults: PoolPodResources = dataclasses.field(
        default_factory=PoolPodResources)
    training_overrides: PoolPodResources = dataclasses.field(
        default_factory=PoolPodResources)
    prediction_overrides: PoolPodResources = dataclasses.field(
        default_factory=PoolPodResources)

    def effective_daemon(self) -> PoolPodResources:
        return PoolPodResources(requests={
            'cpu': '1',
            'memory': '2Gi'
        }).overlay(self.daemon)

    def effective_activity(
            self, kind: Optional[Literal['training',
                                         'prediction']]) -> PoolPodResources:
        override = (getattr(self, f'{kind}_overrides')
                    if kind is not None else PoolPodResources())
        return self.activity_defaults.overlay(override)

    @model_validator(mode='after')
    def _validate_effective_resources(self) -> 'PoolResourcesConfig':
        # Fail during config loading, not hours later at the first TRAIN pod.
        self.effective_daemon()
        self.effective_activity('training')
        self.effective_activity('prediction')
        return self


@dataclass(config=ConfigDict(extra='forbid', populate_by_name=True))
class PoolToleration:
    """An additional pod toleration, using Kubernetes field names in JSON."""
    key: str = ''
    operator: Literal['Equal', 'Exists', ''] = 'Equal'
    value: str = ''
    effect: Literal['NoSchedule', 'PreferNoSchedule', 'NoExecute', ''] = ''
    toleration_seconds: Optional[int] = Field(default=None,
                                              alias='tolerationSeconds',
                                              strict=True, ge=-(2**63),
                                              le=2**63 - 1)

    @model_validator(mode='after')
    def _validate_toleration(self) -> 'PoolToleration':
        if not self.key and self.operator != 'Exists':
            raise ValueError(
                'a toleration with no key requires operator Exists')
        if self.operator == 'Exists' and self.value:
            raise ValueError('an Exists toleration must not specify a value')
        if self.toleration_seconds is not None and self.effect != 'NoExecute':
            raise ValueError('tolerationSeconds requires effect NoExecute')
        return self


@dataclass
class K8sPoolStrategyConfig:
    """K8s-specific pool controller settings.

    These only apply when ``DiskgraphConfig.pool_strategy == 'k8s'``.
    Singlehost deployments ignore this sub-config entirely — the host
    itself is the single worker, so there are no daemon pods, PVCs, or
    heartbeat-based worker health to configure.
    """
    # DG node types: a ``{instance-type: DiskgraphNodeType}`` map shared by the
    # cache and ephemeral pools. The controller treats each value as a durable
    # capacity tuple ``(ssd_gib, gpu_count)``: it picks the smallest tuple that
    # fits the snapshot estimate and requested durable GPU slots, then targets
    # cache nodes with matching ``kumo.ai/diskgraph-ssd-size`` and
    # ``kumo.ai/diskgraph-gpu-count`` labels. Keys are opaque platform names;
    # values with the same tuple are interchangeable.
    # REQUIRED (no default): a k8s pool with no node types can't schedule
    # anything, and a silent default would mismatch a BYOC cluster's real
    # NodePool labels and strand daemon pods Pending. Failing at config-parse
    # forces explicit controller↔NodePool alignment. SaaS gets it from
    # lib/default.deploy_env_config.json (merged into every env); BYOC must set
    # it. Example: {"standard": {"ssd_gib": 800}, "large": {"ssd_gib": 1750, "gpu_count": 1}}.
    # (Declared first: a required field can't follow defaulted ones in a
    # dataclass.)
    pool_diskgraph_node_types: Dict[str, DiskgraphNodeType] = Field(...)
    pool_resources: PoolResourcesConfig = dataclasses.field(
        default_factory=PoolResourcesConfig)
    # Additive for daemon, cache/eph activities, and their startup canaries.
    # Built-in dedicated=diskgraph[-eph]-<tenant>:NoSchedule tolerations remain.
    pool_extra_tolerations: list[PoolToleration] = dataclasses.field(
        default_factory=list)
    # Fraction (0, 1] of physical SSD the pool may use for snapshot storage.
    # Serves double duty: (1) admission — smallest tier where
    # tier × utilization ≥ snapshot_estimate is selected; (2) pressure
    # eviction — workers above this threshold trigger LRU eviction.
    # None → controller default (0.85); the remaining (1 − utilization)
    # fraction is implicitly reserved for staging tables, temp files, OS.
    pool_ssd_max_utilization: Optional[float] = Field(default=None, gt=0, le=1)
    # How long since the last heartbeat before a worker is marked STALE / DEAD.
    pool_worker_stale_threshold_secs: Optional[int] = Field(default=None, ge=0)
    pool_worker_dead_threshold_secs: Optional[int] = Field(default=None, ge=0)

    @field_validator('pool_extra_tolerations', mode='before')
    @classmethod
    def _normalize_extra_tolerations(cls, value):
        return [] if value is None else value

    @field_validator('pool_diskgraph_node_types', mode='before')
    @classmethod
    def _validate_node_types(cls, v: Optional[dict]) -> Optional[dict]:
        # Per-value validation (ssd_gib > 0, shape) is handled by
        # DiskgraphNodeType during coercion; here we only guard the map itself.
        if v is None:
            return None
        if not v:
            raise ValueError('pool_diskgraph_node_types must not be empty')
        if any(not k for k in v):
            raise ValueError(
                'pool_diskgraph_node_types: instance types must be non-empty')
        return v


@dataclass(frozen=True)
class DiskgraphConfig:
    """DiskGraph compute config.

    Covers graph building, materialization, and feature serving — the
    roles previously split between PySpark (data operations) and the
    legacy in-process graph backend (storage/sampling).

    **Pool controller settings** live on this config because the diskgraph
    pool is intrinsically a diskgraph concern. K8s-specific settings
    (PVC sizing, daemon heartbeat thresholds, SSD pressure) are grouped
    under ``k8s_pool``; settings that apply to both k8s and singlehost
    remain here.
    """
    type: Literal[DataComputeBackend.DISKGRAPH] = DataComputeBackend.DISKGRAPH
    data_root: str = '/mnt/localssd'
    local_ingest_root: Optional[str] = None
    pool_strategy: Literal['k8s', 'singlehost'] = 'k8s'
    pool_min_ssd_idle_secs: Optional[int] = Field(default=None, ge=0)
    pool_max_ssd_idle_secs: Optional[int] = Field(default=None, ge=0)
    pool_max_dataplane_retention_secs: Optional[int] = Field(default=None, ge=0)
    pool_lease_sweep_interval_secs: Optional[int] = Field(default=None, ge=0)
    pool_maintenance_interval_secs: Optional[int] = Field(default=None, ge=0)
    pool_tombstone_gc_grace_secs: Optional[int] = Field(default=None, ge=0)
    # Snapshot estimate = staged parquet bytes × ingest_space_multiplier
    #                     + snapshot_estimate_buffer_gib.
    # Direct sources have no staged parquet and retain their native estimate.
    # The fixed buffer adds a per-snapshot safety margin on top (None → 0).
    ingest_space_multiplier: int = Field(default=4, ge=1)
    snapshot_estimate_buffer_gib: Optional[int] = Field(default=None, ge=0)
    # K8s-specific pool settings (ignored for singlehost).
    k8s_pool: Optional[K8sPoolStrategyConfig] = None


# ---------------------------------------------------------------------------
# Orchestration sub-configs (discriminated union)
# ---------------------------------------------------------------------------


class TemporalServerLocation(str, Enum):
    """Where the Temporal server runs."""
    CLOUD = 'cloud'  # Temporal Cloud (*.tmprl.cloud)
    CUSTOM = 'custom'  # Self-hosted or localhost


@dataclass
class TemporalCloudConfig:
    """Connect to Temporal Cloud.

    Server address is derived from the Temporal namespace
    (<namespace>.put0h.tmprl.cloud:7233). TLS cert is retrieved from
    the secret backend using 'kumo-{namespace}-temporal' convention.
    """
    type: Literal[TemporalServerLocation.CLOUD] = TemporalServerLocation.CLOUD
    namespace: Optional[str] = None  # None → derived from deploy_mode


@dataclass
class CustomTemporalConfig:
    """Connect to a self-hosted or localhost Temporal server.

    Maps to env vars: TEMPORAL_ADDRESS, TEMPORAL_NAMESPACE,
    TEMPORAL_CERT_DIR.
    """
    server_address: str  # e.g. '127.0.0.1:7233'
    type: Literal[TemporalServerLocation.CUSTOM] = TemporalServerLocation.CUSTOM
    namespace: Optional[str] = None  # None → derived from deploy_mode
    tls_cert_dir: Optional[str] = None  # Dir with tls.crt, tls.key, ca.crt


OrchestrationConfig = Union[TemporalCloudConfig, CustomTemporalConfig]

# ---------------------------------------------------------------------------
# File storage sub-config
# ---------------------------------------------------------------------------


class StorageScheme(str, Enum):
    """Scheme of the underlying file storage system.

    Values match the URI scheme strings dispatched by ``io_utils.fsutil``.
    """
    S3 = 's3'
    GCS = 'gs'
    DBFS = 'dbfs'
    ADLS = 'abfss'
    LOCAL = 'local'


class S3AddressingStyle(str, Enum):
    """Bucket addressing used by an S3-compatible deployment endpoint."""
    PATH = 'path'
    VIRTUAL = 'virtual'
    AUTO = 'auto'


@dataclass
class S3StorageConfig:
    """Non-secret settings for deployment-owned S3-compatible storage.

    Credentials are represented only by a secret name and are resolved at
    runtime through the configured secret backend.
    """
    endpoint_url: str
    region: str
    addressing_style: S3AddressingStyle
    credential_secret_name: str
    ca_bundle_path: Optional[str] = None
    send_bucket_owner_full_control: bool = False

    @model_validator(mode='after')
    def _validate_connection(self) -> 'S3StorageConfig':
        endpoint = self.endpoint_url.strip()
        parsed = urlparse(endpoint)
        valid_scheme = parsed.scheme in ('http', 'https')
        if (endpoint != self.endpoint_url or not valid_scheme
                or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.query
                or parsed.fragment or parsed.path not in ('', '/')):
            raise ValueError(
                'storage.s3.endpoint_url must be an absolute HTTP(S) S3 API '
                'origin without credentials, a path, query, or fragment')
        if not self.region.strip() or self.region != self.region.strip():
            raise ValueError('storage.s3.region must be non-empty')
        if (not self.credential_secret_name.strip()
                or self.credential_secret_name
                != self.credential_secret_name.strip()):
            raise ValueError(
                'storage.s3.credential_secret_name must be non-empty')
        if self.ca_bundle_path is not None:
            if (not self.ca_bundle_path.startswith('/')
                    or self.ca_bundle_path.endswith('/')):
                raise ValueError(
                    'storage.s3.ca_bundle_path must be an absolute file path')
        return self


@dataclass
class AzureStorageConfig:
    """Azure Service Principal credentials for ABFS (ADLS) access.

    Used by the ADLS filesystem implementation when the data-plane storage
    root is an ABFS path. All fields are required together.
    """
    client_id: str
    client_secret: str
    tenant_id: str
    storage_account: str


@dataclass
class FileStorageConfig:
    """File storage settings.

    Root paths for the three storage concerns:
    - data_plane: customer data, model artifacts produced by training
    - control_plane: REST persist state, controlplane metadata
    - artifact: pre-built shared inputs (e.g., GloVe embeddings)

    The storage scheme is parsed from the path itself (s3://, gs://,
    /opt/..., etc.).

    None means the root is derived at runtime from deploy_mode and
    instance_id using a platform-specific template.
    """
    data_plane_storage_root: Optional[str] = None
    control_plane_storage_root: Optional[str] = None
    artifact_root: Optional[str] = None
    # True only when deployment infrastructure has installed expiration rules
    # for Kumo's configured temporary prefixes.
    lifecycle_ready: bool = False
    s3: Optional[S3StorageConfig] = None

    @model_validator(mode='after')
    def _validate_s3_roots(self) -> 'FileStorageConfig':
        if self.s3 is None:
            return self
        roots = (self.data_plane_storage_root, self.control_plane_storage_root,
                 self.artifact_root)
        if not any(root and root.startswith('s3://') for root in roots):
            raise ValueError(
                'storage.s3 requires at least one s3:// deployment root')
        return self


# ---------------------------------------------------------------------------
# Simple sub-configs
# ---------------------------------------------------------------------------


@dataclass
class SecretConfig:
    """Secret retrieval settings.

    Secrets are fetched from the primary backend first. If the secret is
    not found, fallback backends are tried in order.

    Read-write capability is inherent to the backend type:
    - AWS_SM, GCP_SM: read + write
    - K8S_MOUNT, ENV_VAR: read-only
    """
    backend: SecretBackend  # Primary secret backend (required)
    fallbacks: Sequence[SecretBackend] = ()
    mount_path: Optional[str] = None  # Base dir for K8S_MOUNT secrets


@dataclass
class LoggingConfig:
    """Logging/observability settings."""
    backend: LoggingBackend = LoggingBackend.CLOUDWATCH
    output_format: LoggingOutputFormat = LoggingOutputFormat.TEXT


@dataclass(frozen=True)
class SecretKeyRef:
    """Reference to a value inside the configured secret backend."""
    secret_name: str
    secret_key: str


@dataclass(frozen=True)
class DirectConnectGoogleGroupsConfig:
    """Google Workspace group lookup via a delegated service account.

    Google's OIDC tokens never carry group membership, so groups are
    fetched from the Directory API using a customer-provided service
    account with domain-wide delegation, impersonating ``admin_email``.
    """
    service_account_key_ref: Optional[SecretKeyRef] = None
    admin_email: Optional[str] = None


@dataclass(frozen=True)
class DirectConnectOIDCAuthConfig:
    """Direct Connect OIDC bootstrap settings.

    Secret values are referenced by name/key so deploy_env_config.json can stay
    non-sensitive.
    """
    issuer: Optional[str] = None
    client_id: Optional[str] = None
    client_secret_ref: Optional[SecretKeyRef] = None
    groups_claim: Optional[str] = None
    google_groups: Optional[DirectConnectGoogleGroupsConfig] = None


@dataclass(frozen=True)
class DirectConnectPasswordAuthConfig:
    """Direct Connect password-login bootstrap settings."""
    temporary_password_ref: Optional[SecretKeyRef] = None


@dataclass(frozen=True)
class DirectConnectAuthConfig:
    """Direct Connect auth settings consumed by REST/UI."""
    login_method: Optional[Literal['oidc', 'password']] = None
    allowed_return_origins: Tuple[str, ...] = ()
    admin_emails: Tuple[str, ...] = ()
    oidc: Optional[DirectConnectOIDCAuthConfig] = None
    password: Optional[DirectConnectPasswordAuthConfig] = None

    @field_validator('allowed_return_origins', 'admin_emails', mode='before')
    @classmethod
    def _normalize_string_tuple(cls, v) -> Tuple[str, ...]:
        if v is None:
            return ()
        if isinstance(v, str):
            return tuple(item.strip() for item in v.split(',') if item.strip())
        return tuple(str(item).strip() for item in v if str(item).strip())


@dataclass(frozen=True)
class AuthConfig:
    """Authentication settings at the deploy-config boundary.

    All fields are optional so older deploy_env_config.json files remain valid.
    Direct Connect is the only supported auth backend.
    """
    enabled: Optional[bool] = None
    # Kept only to reject stale non-Direct-Connect configs with an explicit
    # error during auth resolution.
    backend: Optional[str] = None
    direct_connect: Optional[DirectConnectAuthConfig] = None


@dataclass(frozen=True)
class AmbientS3WorkloadConfig:
    """Explicit allowlist for Kumo's ambient S3 workload identity.

    Absence of this object disables ambient S3 completely.  Every ARN must be
    a concrete IAM role; assumed-role session ARNs are normalized back to an
    IAM role before comparison at the activity boundary.
    """
    approved_role_arns: Tuple[str, ...]

    @field_validator('approved_role_arns', mode='before')
    @classmethod
    def _normalize_role_arns(cls, value) -> Tuple[str, ...]:
        import re

        if isinstance(value, str):
            values = (value, )
        else:
            values = tuple(value or ())
        normalized = tuple(
            str(item).strip() for item in values if str(item).strip())
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                'ambient_s3.approved_role_arns must not contain duplicates')
        role_pattern = re.compile(
            r'^arn:(aws|aws-us-gov|aws-cn):iam::[0-9]{12}:'
            r'role/[A-Za-z0-9+=,.@_-]+$')
        if not normalized:
            raise ValueError('ambient_s3.approved_role_arns must not be empty')
        invalid = [arn for arn in normalized if not role_pattern.fullmatch(arn)]
        if invalid:
            raise ValueError(
                'ambient_s3.approved_role_arns must contain concrete, '
                'pathless IAM role ARNs; invalid values: '
                f'{invalid}')
        return normalized


@dataclass(frozen=True)
class AmbientGCPWorkloadConfig:
    """Explicit allowlist for Kumo's ambient GCP workload identity.

    The GCP analogue of :class:`AmbientS3WorkloadConfig`.  Absence of this
    object disables ambient GCP completely.  Every entry must be a concrete
    Google service-account email; the deployment's own Workload Identity
    service account is attested against this list before it is used to read a
    customer's GCS bucket or BigQuery dataset.
    """
    approved_service_account_emails: Tuple[str, ...]

    @field_validator('approved_service_account_emails', mode='before')
    @classmethod
    def _normalize_service_account_emails(cls, value) -> Tuple[str, ...]:
        import re

        if isinstance(value, str):
            values = (value, )
        else:
            values = tuple(value or ())
        normalized = tuple(
            str(item).strip().lower() for item in values if str(item).strip())
        if not normalized:
            raise ValueError('ambient_gcp.approved_service_account_emails '
                             'must not be empty')
        if len(set(normalized)) != len(normalized):
            raise ValueError('ambient_gcp.approved_service_account_emails '
                             'must not contain duplicates')
        email_pattern = re.compile(
            r'^[a-z0-9._-]+@[a-z0-9.-]+\.gserviceaccount\.com$')
        invalid = [e for e in normalized if not email_pattern.fullmatch(e)]
        if invalid:
            raise ValueError(
                'ambient_gcp.approved_service_account_emails must contain '
                f'Google service-account emails; invalid values: {invalid}')
        return normalized


# ---------------------------------------------------------------------------
# RawDeployConfig — infra → app boundary
# ---------------------------------------------------------------------------


@dataclass
class RawDeployConfig:
    """Structured deployment config at the infra → app boundary.

    Infra produces this (JSON file, env vars, or PLATFORM_PRESET); the
    app-facing DeployEnvConfig wraps it with derived fields and validation.
    Application code should not depend on this type directly.

    Fields are intentionally optional / use defaults so that infra configs
    only need to specify what they override.  Resolution of None values
    (e.g. storage roots) happens in DeployEnvConfig.parse_from_env().
    """
    data_compute: DiskgraphConfig = Field(default_factory=DiskgraphConfig)
    orchestration: OrchestrationConfig = Field(
        default_factory=TemporalCloudConfig, discriminator='type')
    secrets: SecretConfig = Field(
        default_factory=lambda: SecretConfig(backend=SecretBackend.AWS_SM))
    storage: FileStorageConfig = Field(default_factory=FileStorageConfig)
    metadata_storage: MetadataStorageConfig = Field(
        default_factory=SqliteMetadataStorageConfig, discriminator='type')
    logging: LoggingConfig = Field(default_factory=lambda: LoggingConfig(
        backend=LoggingBackend.CLOUDWATCH))
    azure_storage: Optional[AzureStorageConfig] = None
    auth: Optional[AuthConfig] = None
    ambient_s3: Optional[AmbientS3WorkloadConfig] = None
    ambient_gcp: Optional[AmbientGCPWorkloadConfig] = None
    # Externally reachable host of this deployment (no scheme), e.g.
    # 'kumo-acme.dev.kumoai.cloud'. Used to build customer-facing URLs
    # (job tracking links and failure notifications).
    external_host: Optional[str] = None


PLATFORM_DEFAULTS: dict[PlatformPreset, RawDeployConfig] = {
    PlatformPreset.AWS:
    RawDeployConfig(
        data_compute=DiskgraphConfig(),
        orchestration=TemporalCloudConfig(),
        secrets=SecretConfig(backend=SecretBackend.AWS_SM),
        storage=FileStorageConfig(),
        logging=LoggingConfig(backend=LoggingBackend.CLOUDWATCH,
                              output_format=LoggingOutputFormat.TEXT),
    ),
    PlatformPreset.GCP:
    RawDeployConfig(
        data_compute=DiskgraphConfig(),
        orchestration=TemporalCloudConfig(),
        secrets=SecretConfig(backend=SecretBackend.GCP_SM),
        storage=FileStorageConfig(),
        logging=LoggingConfig(backend=LoggingBackend.CLOUD_LOGGING,
                              output_format=LoggingOutputFormat.TEXT),
    ),
}

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

# Env var names for config loading (constants, not parsed at import time)
_PLATFORM_PRESET_ENV = 'PLATFORM_PRESET'
_DEPLOY_CONFIG_JSON_ENV = 'DEPLOY_ENV_CONFIG_JSON'
_DEPLOY_CONFIG_FILE_ENV = 'DEPLOY_ENV_CONFIG_FILE'


def load_raw_deploy_config() -> Optional[RawDeployConfig]:
    """Load RawDeployConfig from the environment.

    Two layers:
    1. PLATFORM_PRESET — base defaults from the preset table (e.g. 'aws')
    2. DEPLOY_ENV_CONFIG_FILE *or* DEPLOY_ENV_CONFIG_JSON — override dict
       merged on top of the base.  These two are mutually exclusive.

    If both a preset and an override are present, the override's fields
    replace the corresponding base fields (shallow merge at the
    RawDeployConfig field level).  If only an override is present it must
    be a complete config.

    Returns None if no config source is found in the environment.
    """
    import json
    import os

    # --- base from preset ---
    base: Optional[RawDeployConfig] = None
    preset_str = os.environ.get(_PLATFORM_PRESET_ENV)
    if preset_str:
        base = PLATFORM_DEFAULTS[PlatformPreset(preset_str)]

    # --- override from JSON file or inline JSON (mutually exclusive) ---
    config_file = os.environ.get(_DEPLOY_CONFIG_FILE_ENV)
    config_json = os.environ.get(_DEPLOY_CONFIG_JSON_ENV)
    if config_file and config_json:
        raise ValueError(
            f'{_DEPLOY_CONFIG_FILE_ENV} and {_DEPLOY_CONFIG_JSON_ENV} '
            f'cannot both be set')

    override_data: Optional[dict] = None
    if config_file:
        from io_utils.json import load_json
        override_data = load_json(config_file, dict)
    elif config_json:
        override_data = json.loads(config_json)

    if override_data is not None and base is not None:
        return _merge_override(base, override_data)
    if override_data is not None:
        return _parse_raw_deploy_config(override_data)
    return base


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override wins)."""
    merged = dict(base)
    for key, val in override.items():
        if (key in merged and isinstance(merged[key], dict)
                and isinstance(val, dict)):
            merged[key] = _deep_merge_dicts(merged[key], val)
        else:
            merged[key] = val
    return merged


def _merge_override(base: RawDeployConfig, override: dict) -> RawDeployConfig:
    """Deep-merge override dict on top of a base RawDeployConfig.

    Serializes *base* to a dict, recursively merges *override* on top,
    then deserializes the result back to a RawDeployConfig.
    """
    base_dict = dataclasses.asdict(base)
    merged = _deep_merge_dicts(base_dict, override)
    return _parse_raw_deploy_config(merged)


def _parse_raw_deploy_config(data: dict) -> RawDeployConfig:
    """Parse a dict (from JSON) into a RawDeployConfig.

    Pydantic handles discriminated union dispatch automatically via the
    ``Field(discriminator='type')`` annotations on RawDeployConfig fields.
    """
    from pydantic import TypeAdapter
    return TypeAdapter(RawDeployConfig).validate_python(data)
