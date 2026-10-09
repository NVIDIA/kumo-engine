# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT

import re
import ssl
import string
import unicodedata
from dataclasses import field
from datetime import datetime
from typing import Dict, List, Optional, Union
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field, SecretStr
from pydantic.dataclasses import dataclass
from typing_extensions import Literal

from kumoapi._legacy_object_store import (
    migrate_connector_config,
    migrate_validator_values,
)
from kumoapi.adls import (
    normalize_adls_endpoint_suffix,
    normalize_adls_filesystem,
    normalize_adls_storage_account,
)
from kumoapi.common import StrEnum
from kumoapi.gcs import normalize_gcs_bucket
from kumoapi.object_store import normalize_object_store_root_prefix
from kumoapi.s3 import normalize_s3_bucket, normalize_s3_root_prefix
from kumoapi.typing import (
    _FORBID_EXTRA_CONFIG,
    compatible_field_validator,
    compatible_model_validator,
)


def _normalize_required_provider_field(value: str) -> str:
    """Trim a required provider identifier and reject an implicit target."""
    normalized = value.strip()
    if not normalized:
        raise ValueError('Required provider field must not be blank')
    return normalized


_BIGQUERY_PROJECT_ID_RE = re.compile(r'^[a-z][a-z0-9-]{4,28}[a-z0-9]$')
_BIGQUERY_DATASET_ID_RE = re.compile(r'^[A-Za-z0-9_]{1,1024}$')
_BIGQUERY_LOCATION_RE = re.compile(r'^[A-Za-z0-9-]+$')
_GOOGLE_USER_MANAGED_SERVICE_ACCOUNT_EMAIL_RE = re.compile(
    r'^[A-Za-z0-9][A-Za-z0-9._-]*@'
    r'[a-z][a-z0-9-]{4,28}[a-z0-9]\.iam\.gserviceaccount\.com$')
_GOOGLE_COMPUTE_SERVICE_ACCOUNT_EMAIL_RE = re.compile(
    r'^[1-9][0-9]{0,18}-compute@developer\.gserviceaccount\.com$')
_AWS_ROLE_ARN_RE = re.compile(r'^arn:(?P<partition>[a-z0-9-]+):iam::[0-9]{12}:'
                              r'role/[A-Za-z0-9+=,.@_/-]+$')
_AWS_PARTITIONS = frozenset({
    'aws',
    'aws-cn',
    'aws-iso',
    'aws-iso-b',
    'aws-iso-e',
    'aws-iso-f',
    'aws-us-gov',
})


def _normalize_bigquery_project_id(value: str) -> str:
    normalized = _normalize_required_provider_field(value)
    if _BIGQUERY_PROJECT_ID_RE.fullmatch(normalized) is None:
        raise ValueError('Invalid BigQuery project ID')
    return normalized


def _normalize_bigquery_dataset_id(value: str) -> str:
    normalized = _normalize_required_provider_field(value)
    if _BIGQUERY_DATASET_ID_RE.fullmatch(normalized) is None:
        raise ValueError('Invalid BigQuery dataset ID')
    return normalized


def _normalize_bigquery_location(value: str) -> str:
    normalized = _normalize_required_provider_field(value)
    if _BIGQUERY_LOCATION_RE.fullmatch(normalized) is None:
        raise ValueError('Invalid BigQuery location')
    return normalized


def _normalize_google_service_account_email(value: str) -> str:
    normalized = _normalize_required_provider_field(value)
    if not (_GOOGLE_USER_MANAGED_SERVICE_ACCOUNT_EMAIL_RE.fullmatch(normalized)
            or _GOOGLE_COMPUTE_SERVICE_ACCOUNT_EMAIL_RE.fullmatch(normalized)):
        raise ValueError('Invalid Google service-account email')
    return normalized


_GOOGLE_SERVICE_ACCOUNT_AUTH_URI = (
    'https://accounts.google.com/o/oauth2/auth')
_GOOGLE_SERVICE_ACCOUNT_TOKEN_URI = 'https://oauth2.googleapis.com/token'
_GOOGLE_SERVICE_ACCOUNT_UNIVERSE_DOMAIN = 'googleapis.com'


def _require_canonical_google_value(value: str, expected: str,
                                    field_name: str) -> str:
    normalized = _normalize_required_provider_field(value)
    if normalized != expected:
        raise ValueError(
            f'Google service-account {field_name} must use the canonical '
            'Google value')
    return normalized


# File uploads ----------------------------------------------------------------


class UploadFileType(StrEnum):
    """Supported formats for a project-owned uploaded table."""

    CSV = "CSV"
    PARQUET = "PARQUET"


@dataclass(config=_FORBID_EXTRA_CONFIG)
class StartFileUploadRequest:
    r"""A request to begin an AWS S3 multipart file upload."""
    source_table_name: str
    file_type: UploadFileType
    num_parts: int = 1

    @compatible_field_validator('num_parts')
    def check_num_parts(cls, num_parts: int) -> int:
        if num_parts < 1 or num_parts > 1000:
            raise ValueError("num_parts must be between 1 and 1000")
        return num_parts


@dataclass
class StartFileUploadResponse:
    r"""The server response to an initiation of a file upload."""
    temp_upload_path: str
    upload_id: str
    presigned_part_urls: Dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ProjectUploadConnectorResponse:
    """Safe reference to the hidden upload connector for one project."""

    connector_id: str


@dataclass(config=_FORBID_EXTRA_CONFIG)
class PartUploadMetadata:
    r"""Metadata for a single part of a multipart upload."""
    part_number: int
    etag: str


@dataclass(config=_FORBID_EXTRA_CONFIG)
class CompleteFileUploadRequest:
    r"""Request to process an uploaded file into a project-owned table."""
    source_table_name: str
    file_type: UploadFileType
    temp_upload_path: str
    upload_id: str
    parts_metadata: List[PartUploadMetadata] = field(default_factory=list)
    validate_data: bool = True


@dataclass(config=_FORBID_EXTRA_CONFIG)
class DeleteUploadedFileRequest:
    r"""Request to delete a project-owned uploaded file."""
    source_table_name: str
    file_type: UploadFileType


# Shared identifiers and enums ------------------------------------------------


class DataSourceType(StrEnum):
    SNOWFLAKE = "SNOWFLAKE"
    S3 = "S3"
    GCS = "GCS"
    ADLS = "ADLS"
    DATABRICKS = "DATABRICKS"
    BIGQUERY = "BIGQUERY"
    UNAVAILABLE = "UNAVAILABLE"


class ConnectorType(StrEnum):
    SNOWFLAKE = "snowflake"
    DATABRICKS = "databricks"
    BIGQUERY = "bigquery"
    S3 = "s3"
    GCS = "gcs"
    ADLS = "adls"


class IdentityType(StrEnum):
    SNOWFLAKE = "snowflake"
    DATABRICKS = "databricks"
    BIGQUERY = "bigquery"
    S3 = "s3"
    GCS = "gcs"
    ADLS = "adls"


class AzureCloud(StrEnum):
    PUBLIC = "public"
    US_GOVERNMENT = "us_government"
    CHINA = "china"


IdentityID = str
IdentityGroupID = str
ConnectorID = str
GroupID = str
ProjectID = str
ResourceSlug = str

# Canonical bootstrap resources. These IDs are part of the API contract so
# clients never infer the default project from mutable names or list contents.
DEFAULT_GROUP_ID: GroupID = 'group_00000000000000000000000000000000'
DEFAULT_PROJECT_ID: ProjectID = 'default'
DEFAULT_GROUP_SLUG: ResourceSlug = 'default'
DEFAULT_PROJECT_SLUG: ResourceSlug = 'default'
AMBIENT_S3_IDENTITY_SLUG: ResourceSlug = 'ambient-s3'

_RESOURCE_SLUG_RE = re.compile(
    r'^[a-z](?:[a-z0-9]|-(?=[a-z0-9])){1,46}[a-z0-9]$')
RESERVED_RESOURCE_SLUGS = frozenset({
    AMBIENT_S3_IDENTITY_SLUG,
    'api',
    'by-id',
    'by-slug',
    'default',
    'new',
    'settings',
    'system',
})
_SLUG_SEPARATORS = frozenset(string.whitespace + string.punctuation)


def validate_resource_slug(value: str, *, allow_reserved: bool = False) -> str:
    r"""Validate an explicit immutable project, group, or identity slug.

    The value is returned unchanged. Explicit caller input is never silently
    normalized. ``allow_reserved`` exists only for canonical bootstrap rows;
    public creation requests must leave it ``False``.
    """
    if not isinstance(value,
                      str) or _RESOURCE_SLUG_RE.fullmatch(value) is None:
        raise ValueError(
            'Slug must be 3-48 lowercase ASCII letters, digits, or single '
            'hyphens; start with a letter; and end with a letter or digit')
    if not allow_reserved and value in RESERVED_RESOURCE_SLUGS:
        raise ValueError(f'Reserved resource slug: {value!r}')
    return value


def derive_resource_slug(display_name: str, *,
                         allow_reserved: bool = False) -> str:
    r"""Derive one deterministic slug from a display name.

    NFKC-normalized lowercase text is processed once: runs of ASCII
    whitespace and punctuation become one hyphen, leading/trailing hyphens
    are removed, and the result is validated. No database-dependent suffix is
    ever appended.
    """
    if not isinstance(display_name, str):
        raise ValueError('Display name must be a string')
    normalized = unicodedata.normalize('NFKC', display_name).lower()
    pieces: list[str] = []
    pending_separator = False
    for character in normalized:
        if character in _SLUG_SEPARATORS:
            pending_separator = bool(pieces)
            continue
        if pending_separator:
            pieces.append('-')
            pending_separator = False
        pieces.append(character)
    derived = ''.join(pieces).strip('-')
    return validate_resource_slug(derived, allow_reserved=allow_reserved)


@dataclass(config=_FORBID_EXTRA_CONFIG)
class CreateProjectArgs:
    r"""Create a project with an immutable group authorization binding."""
    name: str
    group_id: GroupID
    slug: Optional[ResourceSlug] = None
    description: str = ''
    is_private: bool = False

    @compatible_field_validator('slug')
    def validate_slug(cls, value: Optional[str]) -> Optional[str]:
        return None if value is None else validate_resource_slug(value)


@dataclass(frozen=True)
class ProjectResponse:
    r"""Safe project metadata returned by project lookup endpoints."""
    id: ProjectID
    slug: ResourceSlug
    display_name: str
    group_id: GroupID

    @compatible_field_validator('slug')
    def validate_slug(cls, value: str) -> str:
        return validate_resource_slug(value, allow_reserved=True)


@dataclass(config=_FORBID_EXTRA_CONFIG)
class CreateGroupArgs:
    r"""Create a native group with an optional immutable slug."""
    display_name: str
    slug: Optional[ResourceSlug] = None
    description: str = ''

    @compatible_field_validator('slug')
    def validate_slug(cls, value: Optional[str]) -> Optional[str]:
        return None if value is None else validate_resource_slug(value)


@dataclass(frozen=True)
class GroupResponse:
    r"""Safe native-group metadata returned by group lookup endpoints."""
    id: GroupID
    slug: ResourceSlug
    display_name: str

    @compatible_field_validator('slug')
    def validate_slug(cls, value: str) -> str:
        return validate_resource_slug(value, allow_reserved=True)


class SnowflakeAuthenticationKind(StrEnum):
    PASSWORD = "password"
    KEY_PAIR = "key_pair"
    #: The OAuth token Snowpark Container Services mounts into a running
    #: container. Ambient: the token is a file on the container filesystem and
    #: never reaches this schema, so an identity using it stores no credential
    #: at all -- the same shape as a Google workload identity. It is usable
    #: only by an in-process runtime running *inside* Snowflake; the Kumo
    #: platform rejects it, because a server outside the container has no such
    #: file to read.
    OAUTH_TOKEN_FILE = "oauth_token_file"


class DatabricksAuthenticationKind(StrEnum):
    #: A service principal's OAuth client ID and secret.
    OAUTH = "oauth"
    #: Defers entirely to the Databricks SDK's own ambient auth-resolution
    #: chain (`DATABRICKS_HOST`/`DATABRICKS_TOKEN`, a configured CLI profile,
    #: a notebook's attached execution context, ...). Nothing reaches this
    #: schema, so an identity using it stores no credential at all -- the
    #: same shape as `SnowflakeAuthenticationKind.OAUTH_TOKEN_FILE`. It is
    #: usable only by an in-process runtime whose environment that chain
    #: resolves; the Kumo platform rejects it, because a server has nothing
    #: for the chain to find.
    AMBIENT = "ambient"


class IdentityCredentialKind(StrEnum):
    DATABRICKS_OAUTH = "databricks_oauth"
    SNOWFLAKE_PASSWORD = "snowflake_password"
    SNOWFLAKE_KEY_PAIR = "snowflake_key_pair"
    GOOGLE_SERVICE_ACCOUNT_KEY = "google_service_account_key"
    S3_ASSUME_ROLE = "s3_assume_role"
    S3_ACCESS_KEY = "s3_access_key"


@dataclass(frozen=True)
class ResourceTimestamps:
    created_at: datetime
    updated_at: datetime


# Project-owned connector targets --------------------------------------------


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class SnowflakeConnectorResourceConfig:
    warehouse: str
    database: str
    schema_name: str
    role: Optional[str] = None
    type: Literal[ConnectorType.SNOWFLAKE] = ConnectorType.SNOWFLAKE

    @compatible_field_validator('warehouse', 'database', 'schema_name')
    def strip_inputs(cls, value: str) -> str:
        return _normalize_required_provider_field(value)

    @compatible_field_validator('role')
    def strip_optional_inputs(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value.strip() or None


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class DatabricksConnectorResourceConfig:
    catalog: str
    schema_name: str
    cluster_id: Optional[str] = None
    warehouse_id: Optional[str] = None
    type: Literal[ConnectorType.DATABRICKS] = ConnectorType.DATABRICKS

    @compatible_field_validator('catalog', 'schema_name')
    def strip_inputs(cls, value: str) -> str:
        return _normalize_required_provider_field(value)

    @compatible_field_validator('cluster_id', 'warehouse_id')
    def strip_optional_inputs(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value.strip() or None

    def _validate_compute_target(self) -> None:
        if (self.cluster_id is None) == (self.warehouse_id is None):
            raise ValueError(
                "Exactly one of 'cluster_id' or 'warehouse_id' is required.")

    def __post_init__(self) -> None:
        self._validate_compute_target()


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class BigQueryConnectorResourceConfig:
    project_id: str
    dataset_id: str
    location: str
    billing_project_id: Optional[str] = None
    type: Literal[ConnectorType.BIGQUERY] = ConnectorType.BIGQUERY

    @compatible_field_validator('project_id')
    def normalize_project_id(cls, value: str) -> str:
        return _normalize_bigquery_project_id(value)

    @compatible_field_validator('dataset_id')
    def normalize_dataset_id(cls, value: str) -> str:
        return _normalize_bigquery_dataset_id(value)

    @compatible_field_validator('location')
    def normalize_location(cls, value: str) -> str:
        return _normalize_bigquery_location(value)

    @compatible_field_validator('billing_project_id')
    def strip_optional_inputs(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        return (_normalize_bigquery_project_id(normalized)
                if normalized else None)


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class S3RootedConnectorTarget:
    r"""Restrict every S3 operation to one bucket and optional prefix."""
    bucket: str
    prefix: str = ''
    region: Optional[str] = None
    mode: Literal['rooted'] = 'rooted'

    @compatible_field_validator('bucket')
    def strip_inputs(cls, value: str) -> str:
        return normalize_s3_bucket(value)

    @compatible_field_validator('prefix')
    def normalize_prefix(cls, value: str) -> str:
        return normalize_s3_root_prefix(value)

    @compatible_field_validator('region')
    def strip_optional_inputs(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value.strip() or None


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class S3AnyURIConnectorTarget:
    r"""Require a canonical full S3 URI on every object operation.

    Region selection is intentionally per request; connector-wide region
    configuration is forbidden in this mode. Discovery remains unavailable.
    """
    mode: Literal['any_uri'] = 'any_uri'


S3ConnectorTarget = Union[S3RootedConnectorTarget, S3AnyURIConnectorTarget]


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class S3ConnectorResourceConfig:
    r"""Project-owned S3 connector with rooted or rootless any-URI mode."""
    target: S3ConnectorTarget = Field(discriminator='mode')
    type: Literal[ConnectorType.S3] = ConnectorType.S3


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class GCSRootedConnectorTarget:
    r"""Restrict every GCS operation to one bucket and optional prefix."""
    bucket: str
    prefix: str = ''
    mode: Literal['rooted'] = 'rooted'

    @compatible_field_validator('bucket')
    def strip_inputs(cls, value: str) -> str:
        return normalize_gcs_bucket(value)

    @compatible_field_validator('prefix')
    def normalize_prefix(cls, value: str) -> str:
        return normalize_object_store_root_prefix(value, provider_name='GCS')


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class GCSAnyURIConnectorTarget:
    r"""Require one canonical full GCS URI on every object operation."""
    mode: Literal['any_uri'] = 'any_uri'


GCSConnectorTarget = Union[GCSRootedConnectorTarget, GCSAnyURIConnectorTarget]


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class GCSConnectorResourceConfig:
    target: GCSConnectorTarget = Field(discriminator='mode')
    requester_pays_project: Optional[str] = None
    type: Literal[ConnectorType.GCS] = ConnectorType.GCS

    @compatible_model_validator(mode='before')
    @classmethod
    def migrate_legacy_rooted_target(cls, values):
        return migrate_validator_values(
            values, lambda data: migrate_connector_config(data, 'gcs'))

    @compatible_field_validator('requester_pays_project')
    def strip_optional_inputs(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value.strip() or None

    @property
    def bucket(self) -> str:
        if not isinstance(self.target, GCSRootedConnectorTarget):
            raise AttributeError('Any-URI GCS connectors do not have a bucket')
        return self.target.bucket

    @property
    def prefix(self) -> str:
        if not isinstance(self.target, GCSRootedConnectorTarget):
            raise AttributeError('Any-URI GCS connectors do not have a prefix')
        return self.target.prefix


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class ADLSRootedConnectorTarget:
    r"""Restrict every ADLS operation to one filesystem and prefix."""
    storage_account: str
    filesystem: str
    prefix: str = ''
    endpoint_suffix: str = 'dfs.core.windows.net'
    mode: Literal['rooted'] = 'rooted'

    @compatible_field_validator('storage_account')
    def normalize_storage_account(cls, value: str) -> str:
        return normalize_adls_storage_account(value)

    @compatible_field_validator('filesystem')
    def normalize_filesystem(cls, value: str) -> str:
        return normalize_adls_filesystem(value)

    @compatible_field_validator('prefix')
    def normalize_prefix(cls, value: str) -> str:
        return normalize_object_store_root_prefix(
            value,
            provider_name='Azure Data Lake',
        )

    @compatible_field_validator('endpoint_suffix')
    def validate_endpoint_suffix(cls, value: str) -> str:
        return normalize_adls_endpoint_suffix(value)


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class ADLSAnyURIConnectorTarget:
    r"""Require one canonical full ADLS URI on every object operation."""
    mode: Literal['any_uri'] = 'any_uri'


ADLSConnectorTarget = Union[
    ADLSRootedConnectorTarget,
    ADLSAnyURIConnectorTarget,
]


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class ADLSConnectorResourceConfig:
    target: ADLSConnectorTarget = Field(discriminator='mode')
    type: Literal[ConnectorType.ADLS] = ConnectorType.ADLS

    @compatible_model_validator(mode='before')
    @classmethod
    def migrate_legacy_rooted_target(cls, values):
        return migrate_validator_values(
            values, lambda data: migrate_connector_config(data, 'adls'))

    @property
    def storage_account(self) -> str:
        if not isinstance(self.target, ADLSRootedConnectorTarget):
            raise AttributeError(
                'Any-URI ADLS connectors do not have a storage account')
        return self.target.storage_account

    @property
    def filesystem(self) -> str:
        if not isinstance(self.target, ADLSRootedConnectorTarget):
            raise AttributeError(
                'Any-URI ADLS connectors do not have a filesystem')
        return self.target.filesystem

    @property
    def prefix(self) -> str:
        if not isinstance(self.target, ADLSRootedConnectorTarget):
            raise AttributeError(
                'Any-URI ADLS connectors do not have a prefix')
        return self.target.prefix

    @property
    def endpoint_suffix(self) -> str:
        if not isinstance(self.target, ADLSRootedConnectorTarget):
            raise AttributeError(
                'Any-URI ADLS connectors do not have one endpoint suffix')
        return self.target.endpoint_suffix


ConnectorResourceConfig = Union[
    SnowflakeConnectorResourceConfig,
    DatabricksConnectorResourceConfig,
    BigQueryConnectorResourceConfig,
    S3ConnectorResourceConfig,
    GCSConnectorResourceConfig,
    ADLSConnectorResourceConfig,
]

# Administrator-owned identities --------------------------------------------


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class DatabricksIdentityConfig:
    #: Absent only for
    #: :attr:`DatabricksAuthenticationKind.AMBIENT`, where the SDK's
    #: own ambient chain resolves the workspace there is no need for a
    #: caller to name.
    host: Optional[str] = None
    #: Absent only for ``AMBIENT``, for the same reason.
    client_id: Optional[str] = None
    authentication: DatabricksAuthenticationKind = (
        DatabricksAuthenticationKind.OAUTH)
    type: Literal[IdentityType.DATABRICKS] = IdentityType.DATABRICKS

    @compatible_field_validator('host', 'client_id')
    def strip_optional_inputs(cls, value: Optional[str]) -> Optional[str]:
        return None if value is None else _normalize_required_provider_field(
            value)

    def _validate(self) -> None:
        ambient = self.authentication is (DatabricksAuthenticationKind.AMBIENT)
        if ambient:
            if self.host is not None or self.client_id is not None:
                raise ValueError(
                    'a databricks ambient identity names no host or '
                    'client_id; the ambient auth chain resolves both')
        else:
            if self.host is None or self.client_id is None:
                raise ValueError(
                    f'a databricks {self.authentication.value} identity '
                    f'requires both host and client_id')

    def __post_init__(self) -> None:
        self._validate()


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class SnowflakeIdentityConfig:
    account: str
    authentication: SnowflakeAuthenticationKind
    #: Absent only for :attr:`SnowflakeAuthenticationKind.OAUTH_TOKEN_FILE`,
    #: where the principal is whoever the mounted token names and there is no
    #: user for a caller to supply.
    username: Optional[str] = None
    #: The container-internal endpoint Snowpark Container Services exports as
    #: ``SNOWFLAKE_HOST``. Stored rather than read from the environment at
    #: connect time so that an identity describes itself: it can be tested
    #: without mutating ``os.environ``, and pointed at a private-link
    #: endpoint. Meaningful only with ``OAUTH_TOKEN_FILE``; every other kind
    #: derives its host from :attr:`account`.
    host: Optional[str] = None
    #: Where the mounted OAuth token is read from. Meaningful only with
    #: ``OAUTH_TOKEN_FILE``; ``None`` takes the platform default.
    token_path: Optional[str] = None
    type: Literal[IdentityType.SNOWFLAKE] = IdentityType.SNOWFLAKE

    @compatible_field_validator('account')
    def strip_inputs(cls, value: str) -> str:
        return _normalize_required_provider_field(value)

    @compatible_field_validator('username', 'host', 'token_path')
    def strip_optional_inputs(cls, value: Optional[str]) -> Optional[str]:
        return None if value is None else _normalize_required_provider_field(
            value)

    def _validate(self) -> None:
        ambient = self.authentication is (
            SnowflakeAuthenticationKind.OAUTH_TOKEN_FILE)
        if ambient:
            if self.username is not None:
                raise ValueError(
                    'a snowflake oauth_token_file identity names no username; '
                    'the mounted token identifies the principal')
        else:
            if self.username is None:
                raise ValueError(
                    f'a snowflake {self.authentication.value} identity '
                    f'requires a username')
            # `host` and `token_path` describe how to reach a container-mounted
            # token. Accepting them elsewhere would let an identity carry
            # settings that silently do nothing.
            for name in ('host', 'token_path'):
                if getattr(self, name) is not None:
                    raise ValueError(f'{name} applies only to a snowflake '
                                     f'oauth_token_file identity, not to '
                                     f'{self.authentication.value}')

    def __post_init__(self) -> None:
        self._validate()


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class GoogleServiceAccountKeyAuthentication:
    project_id: str
    client_email: str
    client_id: str
    auth_uri: str
    token_uri: str
    universe_domain: str = 'googleapis.com'
    mode: Literal['service_account_key'] = 'service_account_key'

    @compatible_field_validator('project_id')
    def normalize_project_id(cls, value: str) -> str:
        return _normalize_bigquery_project_id(value)

    @compatible_field_validator('client_email')
    def normalize_client_email(cls, value: str) -> str:
        return _normalize_google_service_account_email(value)

    @compatible_field_validator('client_id')
    def strip_inputs(cls, value: str) -> str:
        return _normalize_required_provider_field(value)

    @compatible_field_validator('auth_uri')
    def normalize_auth_uri(cls, value: str) -> str:
        return _require_canonical_google_value(
            value, _GOOGLE_SERVICE_ACCOUNT_AUTH_URI, 'auth_uri')

    @compatible_field_validator('token_uri')
    def normalize_token_uri(cls, value: str) -> str:
        return _require_canonical_google_value(
            value, _GOOGLE_SERVICE_ACCOUNT_TOKEN_URI, 'token_uri')

    @compatible_field_validator('universe_domain')
    def normalize_universe_domain(cls, value: str) -> str:
        return _require_canonical_google_value(
            value, _GOOGLE_SERVICE_ACCOUNT_UNIVERSE_DOMAIN, 'universe_domain')


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class GoogleWorkloadIdentityAuthentication:
    service_account_email: str
    mode: Literal['workload_identity'] = 'workload_identity'

    @compatible_field_validator('service_account_email')
    def normalize_service_account_email(cls, value: str) -> str:
        return _normalize_google_service_account_email(value)


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class GoogleAmbientAuthentication:
    r"""Use only Kumo's verified ambient GCP workload identity chain.

    The GCP analogue of :class:`S3AmbientAuthentication`. This public
    contract has no service-account or credential fields: when Kumo is
    deployed on GCP, its pods already run as a Workload Identity service
    account, so the customer grants that service account IAM access to their
    bucket/dataset and Kumo authenticates via Application Default
    Credentials with NO impersonation and NO uploaded key. The deployment's
    launch configuration supplies the approved service-account allowlist and
    may disable ambient GCP entirely. Static, environment, shared-file, and
    metadata-less credentials are never selected by this mode.
    """
    mode: Literal['ambient'] = 'ambient'


GoogleIdentityAuthentication = Union[
    GoogleServiceAccountKeyAuthentication,
    GoogleWorkloadIdentityAuthentication,
    GoogleAmbientAuthentication,
]


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class BigQueryIdentityConfig:
    authentication: GoogleIdentityAuthentication = Field(discriminator='mode')
    type: Literal[IdentityType.BIGQUERY] = IdentityType.BIGQUERY


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class S3AssumeRoleAuthentication:
    partition: str
    role_arn: str
    mode: Literal['assume_role'] = 'assume_role'

    @compatible_field_validator('partition')
    def normalize_partition(cls, value: str) -> str:
        value = _normalize_required_provider_field(value).lower()
        if value not in _AWS_PARTITIONS:
            raise ValueError('Unsupported AWS partition')
        return value

    @compatible_field_validator('role_arn')
    def normalize_role_arn(cls, value: str) -> str:
        return _normalize_required_provider_field(value)

    def _validate_role_partition(self) -> None:
        match = _AWS_ROLE_ARN_RE.fullmatch(self.role_arn)
        if match is None:
            raise ValueError('Invalid AWS IAM role ARN')
        if match.group('partition') != self.partition:
            raise ValueError('AWS partition must match the IAM role ARN')

    def __post_init__(self) -> None:
        self._validate_role_partition()


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class S3AmbientAuthentication:
    r"""Use only Kumo's verified ambient AWS workload identity chain.

    This public contract has no role or credential fields. The deployment's
    launch configuration supplies the approved role allowlist and may disable
    ambient S3 entirely. Static, environment, shared-file, SSO, and anonymous
    credentials are never selected by this mode.
    """
    mode: Literal['ambient'] = 'ambient'


class S3AddressingStyle(StrEnum):
    AUTO = 'auto'
    PATH = 'path'
    VIRTUAL = 'virtual'


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class S3ConnectionConfig:
    r"""Optional transport settings for an S3 identity.

    Omitting this object preserves the AWS SDK endpoint, region, addressing,
    and system TLS defaults used by existing identities.  A custom endpoint
    requires an explicit SigV4 signing region.
    """
    endpoint_url: Optional[str] = None
    region: Optional[str] = None
    addressing_style: S3AddressingStyle = S3AddressingStyle.AUTO

    @compatible_field_validator('endpoint_url')
    def normalize_endpoint(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if value != value.strip():
            raise ValueError('S3 endpoint URL must not contain whitespace')
        parsed = urlsplit(value)
        try:
            parsed.port
        except ValueError:
            raise ValueError('S3 endpoint URL has an invalid port') from None
        if (parsed.scheme.lower() not in ('http', 'https')
                or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.path not in ('', '/')
                or parsed.query or parsed.fragment):
            raise ValueError(
                'S3 endpoint URL must be an absolute HTTP(S) API origin '
                'without credentials, path, query, or fragment')
        return urlunsplit(
            (parsed.scheme.lower(), parsed.netloc.lower(), '', '', ''))

    @compatible_field_validator('region')
    def normalize_region(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return _normalize_required_provider_field(value)

    def _validate(self) -> None:
        if self.endpoint_url is not None and self.region is None:
            raise ValueError(
                'S3 custom endpoints require an explicit signing region')

    def __post_init__(self) -> None:
        self._validate()


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class S3AccessKeyAuthentication:
    r"""Use an explicitly stored S3 access key instead of AWS STS."""
    mode: Literal['access_key'] = 'access_key'


S3IdentityAuthentication = Union[
    S3AssumeRoleAuthentication,
    S3AmbientAuthentication,
    S3AccessKeyAuthentication,
]


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class S3IdentityConfig:
    r"""Administrator-owned S3 identity authentication policy."""
    authentication: S3IdentityAuthentication = Field(discriminator='mode')
    connection: Optional[S3ConnectionConfig] = None
    type: Literal[IdentityType.S3] = IdentityType.S3


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class GCSIdentityConfig:
    authentication: GoogleIdentityAuthentication = Field(discriminator='mode')
    type: Literal[IdentityType.GCS] = IdentityType.GCS


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class ADLSIdentityConfig:
    tenant_id: str
    client_id: str
    cloud: AzureCloud
    type: Literal[IdentityType.ADLS] = IdentityType.ADLS

    @compatible_field_validator('tenant_id', 'client_id')
    def strip_inputs(cls, value: str) -> str:
        return _normalize_required_provider_field(value)


IdentityConfig = Union[
    DatabricksIdentityConfig,
    SnowflakeIdentityConfig,
    BigQueryIdentityConfig,
    S3IdentityConfig,
    GCSIdentityConfig,
    ADLSIdentityConfig,
]


class _SecretSerializationMixin:
    """Marker for credential inputs with explicit secret serializers."""


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class DatabricksOAuthCredentials(_SecretSerializationMixin):
    client_secret: SecretStr
    kind: Literal[IdentityCredentialKind.DATABRICKS_OAUTH] = (
        IdentityCredentialKind.DATABRICKS_OAUTH)

    from pydantic import SerializationInfo, field_serializer

    @field_serializer('client_secret')
    def serialize_secret(self, value: SecretStr, info: SerializationInfo):
        if info.context and info.context.get('insecure'):
            return value.get_secret_value()
        return value


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class SnowflakePasswordCredentials(_SecretSerializationMixin):
    password: SecretStr
    kind: Literal[IdentityCredentialKind.SNOWFLAKE_PASSWORD] = (
        IdentityCredentialKind.SNOWFLAKE_PASSWORD)

    from pydantic import SerializationInfo, field_serializer

    @field_serializer('password')
    def serialize_secret(self, value: SecretStr, info: SerializationInfo):
        if info.context and info.context.get('insecure'):
            return value.get_secret_value()
        return value


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class SnowflakeKeyPairCredentials(_SecretSerializationMixin):
    private_key: SecretStr
    private_key_passphrase: Optional[SecretStr] = None
    kind: Literal[IdentityCredentialKind.SNOWFLAKE_KEY_PAIR] = (
        IdentityCredentialKind.SNOWFLAKE_KEY_PAIR)

    from pydantic import SerializationInfo, field_serializer

    @field_serializer('private_key', 'private_key_passphrase',
                      when_used='unless-none')
    def serialize_secret(self, value: SecretStr, info: SerializationInfo):
        if info.context and info.context.get('insecure'):
            return value.get_secret_value()
        return value


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class GoogleServiceAccountKeyCredentials(_SecretSerializationMixin):
    private_key_id: SecretStr
    private_key: SecretStr
    kind: Literal[IdentityCredentialKind.GOOGLE_SERVICE_ACCOUNT_KEY] = (
        IdentityCredentialKind.GOOGLE_SERVICE_ACCOUNT_KEY)

    from pydantic import SerializationInfo, field_serializer

    @field_serializer('private_key_id', 'private_key')
    def serialize_secret(self, value: SecretStr, info: SerializationInfo):
        if info.context and info.context.get('insecure'):
            return value.get_secret_value()
        return value


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class S3AssumeRoleCredentials(_SecretSerializationMixin):
    external_id: Optional[SecretStr] = None
    kind: Literal[IdentityCredentialKind.S3_ASSUME_ROLE] = (
        IdentityCredentialKind.S3_ASSUME_ROLE)

    from pydantic import SerializationInfo, field_serializer

    @field_serializer('external_id', when_used='unless-none')
    def serialize_secret(self, value: SecretStr, info: SerializationInfo):
        if info.context and info.context.get('insecure'):
            return value.get_secret_value()
        return value


_PEM_CERTIFICATE_RE = re.compile(
    r'-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----', re.DOTALL)


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class S3AccessKeyCredentials(_SecretSerializationMixin):
    access_key_id: SecretStr
    secret_access_key: SecretStr
    session_token: Optional[SecretStr] = None
    ca_certificate: Optional[SecretStr] = None
    kind: Literal[IdentityCredentialKind.S3_ACCESS_KEY] = (
        IdentityCredentialKind.S3_ACCESS_KEY)

    @compatible_field_validator('access_key_id')
    def validate_access_key(cls, value: SecretStr) -> SecretStr:
        normalized = value.get_secret_value().strip()
        if not normalized:
            raise ValueError('S3 access key ID must not be blank')
        return SecretStr(normalized)

    @compatible_field_validator('secret_access_key')
    def validate_secret_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError('S3 secret access key must not be blank')
        return value

    @compatible_field_validator('session_token')
    def validate_session_token(
            cls, value: Optional[SecretStr]) -> Optional[SecretStr]:
        if value is not None and not value.get_secret_value().strip():
            raise ValueError('S3 session token must not be blank')
        return value

    @compatible_field_validator('ca_certificate')
    def validate_ca_certificate(
            cls, value: Optional[SecretStr]) -> Optional[SecretStr]:
        if value is None:
            return None
        pem = value.get_secret_value().strip()
        if not pem:
            raise ValueError('S3 CA certificate must not be blank')
        certificates = _PEM_CERTIFICATE_RE.findall(pem)
        remainder = _PEM_CERTIFICATE_RE.sub('', pem)
        if not certificates or remainder.strip():
            raise ValueError(
                'S3 CA certificate must contain only PEM certificates')
        try:
            for certificate in certificates:
                ssl.PEM_cert_to_DER_cert(certificate)
        except ValueError:
            raise ValueError('S3 CA certificate is invalid PEM') from None
        return SecretStr(pem)

    from pydantic import SerializationInfo, field_serializer

    @field_serializer('access_key_id', 'secret_access_key', 'session_token',
                      'ca_certificate', when_used='unless-none')
    def serialize_secret(self, value: SecretStr, info: SerializationInfo):
        if info.context and info.context.get('insecure'):
            return value.get_secret_value()
        return value


IdentityCredentials = Union[
    DatabricksOAuthCredentials,
    SnowflakePasswordCredentials,
    SnowflakeKeyPairCredentials,
    GoogleServiceAccountKeyCredentials,
    S3AssumeRoleCredentials,
    S3AccessKeyCredentials,
]


def validate_identity_credentials_match(
    config: IdentityConfig,
    credentials: Optional[IdentityCredentials],
) -> None:
    if isinstance(config, DatabricksIdentityConfig):
        if config.authentication == DatabricksAuthenticationKind.OAUTH:
            valid = isinstance(credentials, DatabricksOAuthCredentials)
        else:
            # The ambient chain resolves its own token, so there is nothing
            # to store and storing something would mean the caller believes
            # an unused secret is in force.
            if credentials is not None:
                raise ValueError(
                    'a databricks ambient identity cannot store '
                    'credentials; the ambient auth chain is not a secret '
                    'to keep')
            valid = True
    elif isinstance(config, SnowflakeIdentityConfig):
        if config.authentication == SnowflakeAuthenticationKind.PASSWORD:
            valid = isinstance(credentials, SnowflakePasswordCredentials)
        elif config.authentication == SnowflakeAuthenticationKind.KEY_PAIR:
            valid = isinstance(credentials, SnowflakeKeyPairCredentials)
        else:
            # The token is a file on the container filesystem, so there is
            # nothing to store and storing something would mean the caller
            # believes an unused secret is in force.
            if credentials is not None:
                raise ValueError(
                    'a snowflake oauth_token_file identity cannot store '
                    'credentials; the mounted token is ambient')
            valid = True
    elif isinstance(config, (BigQueryIdentityConfig, GCSIdentityConfig)):
        if isinstance(config.authentication,
                      GoogleServiceAccountKeyAuthentication):
            if not isinstance(credentials, GoogleServiceAccountKeyCredentials):
                raise ValueError(
                    'Google service-account key credentials are missing or '
                    'invalid')
            valid = True
        else:
            if credentials is not None:
                raise ValueError(
                    'Google workload identities cannot store credentials; '
                    'ambient identities cannot store credentials')
            valid = True
    elif isinstance(config, S3IdentityConfig):
        if isinstance(config.authentication, S3AmbientAuthentication):
            valid = credentials is None
        elif isinstance(config.authentication, S3AccessKeyAuthentication):
            valid = isinstance(credentials, S3AccessKeyCredentials)
        else:
            valid = credentials is None or isinstance(credentials,
                                                      S3AssumeRoleCredentials)
    else:
        valid = credentials is None
    if not valid:
        raise ValueError('identity config and credential types must match')


@dataclass(config=_FORBID_EXTRA_CONFIG)
class CreateIdentityArgs:
    """Create an identity and its initial authorization bindings."""
    display_name: str
    config: IdentityConfig = Field(discriminator='type')
    group_ids: List[GroupID] = field(default_factory=list)
    credentials: Optional[IdentityCredentials] = Field(default=None,
                                                       discriminator='kind')
    description: str = ''
    slug: Optional[ResourceSlug] = None

    @compatible_field_validator('slug')
    def validate_slug(cls, value: Optional[str]) -> Optional[str]:
        return None if value is None else validate_resource_slug(value)

    def _validate(self) -> None:
        validate_identity_credentials_match(self.config, self.credentials)
        if len(set(self.group_ids)) != len(self.group_ids):
            raise ValueError('group_ids must not contain duplicates')

    def __post_init__(self) -> None:
        self._validate()


@dataclass(config=_FORBID_EXTRA_CONFIG)
class UpdateIdentityArgs:
    display_name: Optional[str] = None
    description: Optional[str] = None


@dataclass(config=_FORBID_EXTRA_CONFIG)
class RotateIdentityCredentialsArgs:
    """Replace identity secret material."""
    credentials: Optional[IdentityCredentials] = Field(default=None,
                                                       discriminator='kind')
    authentication: Optional[GoogleServiceAccountKeyAuthentication] = None

    def _validate(self) -> None:
        has_google_key = isinstance(self.credentials,
                                    GoogleServiceAccountKeyCredentials)
        if has_google_key != (self.authentication is not None):
            raise ValueError(
                'Google key rotation requires authentication metadata and '
                'Google key credentials together')

    def __post_init__(self) -> None:
        self._validate()


@dataclass(config=_FORBID_EXTRA_CONFIG)
class ReplaceIdentityGroupsArgs:
    """Replace the complete identity authorization binding set."""
    group_ids: List[GroupID] = field(default_factory=list)

    def _validate_group_ids(self) -> None:
        if len(set(self.group_ids)) != len(self.group_ids):
            raise ValueError('group_ids must not contain duplicates')

    def __post_init__(self) -> None:
        self._validate_group_ids()


@dataclass(frozen=True)
class IdentityCredentialStatus:
    configured: bool
    last_rotated_at: Optional[datetime] = None


@dataclass(frozen=True)
class IdentityResponse:
    id: IdentityID
    slug: ResourceSlug
    display_name: str
    description: str
    timestamps: ResourceTimestamps
    credential_status: IdentityCredentialStatus
    config: IdentityConfig = Field(discriminator='type')

    @compatible_field_validator('slug')
    def validate_slug(cls, value: str) -> str:
        return validate_resource_slug(value, allow_reserved=True)


@dataclass(frozen=True)
class IdentityGroupsResponse:
    identity_id: IdentityID
    group_ids: List[GroupID] = field(default_factory=list)


@dataclass(frozen=True)
class IdentityConnectorUsage:
    connector_id: ConnectorID
    connector_name: str
    project_id: ProjectID
    # None when the connector's project row no longer exists.
    project_display_name: Optional[str] = None


@dataclass(frozen=True)
class IdentityConnectorsResponse:
    r"""Connectors that currently block deletion of an identity.

    Only connectors that keep the identity in use are returned. Connectors
    in archived projects do not block deletion and are omitted.
    """
    identity_id: IdentityID
    connectors: List[IdentityConnectorUsage] = field(default_factory=list)


# Public connector shapes -----------------------------------------------------


@dataclass(config=_FORBID_EXTRA_CONFIG)
class CreateConnectorArgs:
    r"""Create one project-owned connector.

    For S3, ``identity_id=None`` requests the canonical ambient workload
    identity for both rooted and rootless any-URI targets. The server
    rejects the request when ambient S3 is disabled. Supplying an identity ID
    explicitly is supported in either S3 target mode.

    """
    name: str
    config: ConnectorResourceConfig = Field(discriminator='type')
    identity_id: Optional[IdentityID] = None
    description: str = ''


@dataclass(config=_FORBID_EXTRA_CONFIG)
class UpdateConnectorMetadataArgs:
    name: Optional[str] = None
    description: Optional[str] = None


@dataclass(frozen=True)
class ConnectorResponse:
    id: ConnectorID
    project_id: ProjectID
    identity_id: IdentityID
    name: str
    timestamps: ResourceTimestamps
    config: ConnectorResourceConfig = Field(discriminator='type')
    description: str = ''
    created_by: Optional[str] = None
