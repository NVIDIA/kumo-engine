# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from typing import List, Optional, Union
from xmlrpc.client import Boolean

import kumoapi.data_source as api
import kumoapi.source_table as source_api
from kumoapi.adls import (
    normalize_adls_source_relative_path,
    normalize_adls_source_uri,
)
from kumoapi.gcs import (
    normalize_gcs_source_relative_path,
    normalize_gcs_source_uri,
)
from kumoapi.s3 import (
    normalize_s3_source_relative_path,
    normalize_s3_source_uri,
)

from io_utils import kumo_dataclass
from io_utils.fsutil.uri import ADLSURI, GCSURI, S3URI, ParseURI
from kumo.connector import (
    AnyURIADLSConnector,
    AnyURIGCSConnector,
    AnyURIS3Connector,
    BigQueryConnector,
    CompositeFileConnector,
    DatabricksConnector,
    SnowflakeConnector,
)
from kumo.validation import ValidationResponse
from statedb.config.common import SourceColumn, TableLink
from statedb.config.connector import \
    ADLSConnectorResourceConfig as SerializedADLSConnectorResourceConfig
from statedb.config.connector import \
    BigQueryConnectorResourceConfig as \
    SerializedBigQueryConnectorResourceConfig
from statedb.config.connector import \
    DatabricksConnectorResourceConfig as \
    SerializedDatabricksConnectorResourceConfig
from statedb.config.connector import \
    FileConnectorResourceConfig as SerializedFileConnectorResourceConfig
from statedb.config.connector import \
    GCSConnectorResourceConfig as SerializedGCSConnectorResourceConfig
from statedb.config.connector import \
    S3ConnectorResourceConfig as SerializedS3ConnectorResourceConfig
from statedb.config.connector import \
    SnowflakeConnectorResourceConfig as \
    SerializedSnowflakeConnectorResourceConfig


def _relative_path_within_root(path: str, prefix: str) -> str:
    root = prefix.rstrip('/')
    if not root:
        return path
    if not path.startswith(root + '/'):
        raise ValueError('Full URI is outside the selected connector target')
    relative = path[len(root) + 1:]
    if not relative:
        raise ValueError('Object coordinate must identify a path below root')
    return relative


def object_store_source_for_connector(
    config: api.S3ConnectorResourceConfig | api.GCSConnectorResourceConfig
    | api.ADLSConnectorResourceConfig,
    source: str | source_api.S3Source | source_api.GCSSource
    | source_api.ADLSSource,
) -> source_api.S3Source | source_api.GCSSource | source_api.ADLSSource:
    """Normalize one source coordinate against its authorized target mode."""
    coordinate = (source if isinstance(source, str) else getattr(
        source, 'path', getattr(source, 'uri', None)))
    if coordinate is None:
        raise TypeError(f'Unsupported object-store source: {source}')
    target = config.target
    if isinstance(target, api.S3AnyURIConnectorTarget):
        return source_api.S3URISource(uri=coordinate)
    if isinstance(target, api.GCSAnyURIConnectorTarget):
        return source_api.GCSURISource(uri=coordinate)
    if isinstance(target, api.ADLSAnyURIConnectorTarget):
        return source_api.ADLSURISource(uri=coordinate)
    if isinstance(target, api.S3RootedConnectorTarget):
        if isinstance(source, source_api.S3URISource):
            raise ValueError('A rooted S3 connector requires a relative source')
        if '://' not in coordinate:
            return source_api.S3RelativeSource(path=coordinate)
        uri = ParseURI(normalize_s3_source_uri(coordinate))
        if not isinstance(uri, S3URI) or uri.bucket != target.bucket:
            raise ValueError('Full S3 URI is outside the selected connector')
        relative = _relative_path_within_root(uri.path, target.prefix)
        return source_api.S3RelativeSource(path=relative)
    if isinstance(target, api.GCSRootedConnectorTarget):
        if '://' not in coordinate:
            return source_api.GCSRelativeSource(path=coordinate)
        uri = ParseURI(normalize_gcs_source_uri(coordinate))
        if not isinstance(uri, GCSURI) or uri.bucket != target.bucket:
            raise ValueError('Full GCS URI is outside the selected connector')
        relative = _relative_path_within_root(uri.path, target.prefix)
        return source_api.GCSRelativeSource(path=relative)
    if isinstance(target, api.ADLSRootedConnectorTarget):
        if '://' not in coordinate:
            return source_api.ADLSRelativeSource(path=coordinate)
        uri = ParseURI(normalize_adls_source_uri(coordinate))
        if (not isinstance(uri, ADLSURI)
                or uri.storage_account != target.storage_account
                or uri.container != target.filesystem
                or uri.endpoint_suffix != target.endpoint_suffix):
            raise ValueError('Full ADLS URI is outside the selected connector')
        relative = _relative_path_within_root(uri.path, target.prefix)
        return source_api.ADLSRelativeSource(path=relative)
    raise TypeError(f'Unsupported object-store connector target: {target}')


def object_store_source_for_runtime_connector(
    connector: CompositeFileConnector,
    source: source_api.S3Source | source_api.GCSSource | source_api.ADLSSource,
) -> str:
    """Validate a durable source against its hydrated runtime connector."""
    coordinate = getattr(source, 'path', getattr(source, 'uri', None))
    if coordinate is None:
        raise TypeError(f'Unsupported object-store source: {source}')

    if isinstance(connector, AnyURIS3Connector):
        config = api.S3ConnectorResourceConfig(
            target=api.S3AnyURIConnectorTarget())
    elif isinstance(connector, AnyURIGCSConnector):
        config = api.GCSConnectorResourceConfig(
            target=api.GCSAnyURIConnectorTarget())
    elif isinstance(connector, AnyURIADLSConnector):
        config = api.ADLSConnectorResourceConfig(
            target=api.ADLSAnyURIConnectorTarget())
    else:
        root = ParseURI(connector.root_dir)
        if isinstance(root, S3URI):
            config = api.S3ConnectorResourceConfig(
                target=api.S3RootedConnectorTarget(bucket=root.bucket,
                                                   prefix=root.path))
        elif isinstance(root, GCSURI):
            config = api.GCSConnectorResourceConfig(
                target=api.GCSRootedConnectorTarget(bucket=root.bucket,
                                                    prefix=root.path))
        elif isinstance(root, ADLSURI):
            config = api.ADLSConnectorResourceConfig(
                target=api.ADLSRootedConnectorTarget(
                    storage_account=root.storage_account,
                    filesystem=root.container,
                    prefix=root.path,
                    endpoint_suffix=root.endpoint_suffix,
                ))
        else:
            raise ValueError('Custom prediction source requires an '
                             'object-store connector')

    try:
        normalized = object_store_source_for_connector(config, source)
    except ValueError:
        raise ValueError(
            'Prediction source type or mode does not match its connector'
        ) from None
    if normalized != source:
        raise ValueError(
            'Prediction source type or mode does not match its connector')
    return coordinate


def normalize_file_destination(
    config: api.S3ConnectorResourceConfig | api.GCSConnectorResourceConfig
    | api.ADLSConnectorResourceConfig,
    coordinate: str,
) -> str:
    """Normalize an object-store coordinate against its connector target."""
    target = config.target
    rootless_targets = (api.S3AnyURIConnectorTarget,
                        api.GCSAnyURIConnectorTarget,
                        api.ADLSAnyURIConnectorTarget)
    if isinstance(target, rootless_targets):
        if '://' not in coordinate:
            raise ValueError(
                'Rootless connectors require an exact full destination URI')
        source = object_store_source_for_connector(config, coordinate)
        return source.uri
    source = object_store_source_for_connector(config, coordinate)
    if isinstance(source, source_api.S3RelativeSource):
        return normalize_s3_source_relative_path(source.path)
    elif isinstance(source, source_api.GCSRelativeSource):
        return normalize_gcs_source_relative_path(source.path)
    elif isinstance(source, source_api.ADLSRelativeSource):
        return normalize_adls_source_relative_path(source.path)
    raise ValueError('Rooted connectors require a relative destination')


@kumo_dataclass
class SnowflakeConnectorResourceConfig(
        SerializedSnowflakeConnectorResourceConfig):
    def as_connector(
        self,
        account: str,
        require_explicit_credentials: bool = False,
    ) -> SnowflakeConnector:
        connector = SnowflakeConnector(name=self.name, account=account,
                                       warehouse=self.warehouse,
                                       database=self.database,
                                       schema_name=self.schema_name,
                                       role=self.role)
        if require_explicit_credentials:
            connector.require_explicit_credentials()
        return connector

    def get_snowflake_table_name(self, table_name: str) -> str:
        return f"{self.database}.{self.schema_name}.{table_name}"


@kumo_dataclass
class DatabricksConnectorResourceConfig(
        SerializedDatabricksConnectorResourceConfig):
    def as_connector(
        self,
        host: str,
        require_explicit_credentials: bool = False,
    ) -> DatabricksConnector:
        connector = DatabricksConnector(name=self.name, host=host,
                                        cluster_id=self.cluster_id or '',
                                        warehouse_id=self.warehouse_id or '',
                                        catalog=self.catalog,
                                        schema_name=self.schema_name)
        if require_explicit_credentials:
            connector.require_explicit_credentials()
        return connector


@kumo_dataclass
class BigQueryConnectorResourceConfig(SerializedBigQueryConnectorResourceConfig
                                      ):
    def as_connector(
        self,
        require_explicit_credentials: bool = False,
    ) -> BigQueryConnector:
        connector = BigQueryConnector(
            name=self.name,
            project_id=self.project_id,
            dataset_id=self.dataset_id,
            location=self.location,
            billing_project_id=self.billing_project_id,
        )
        if require_explicit_credentials:
            connector.require_explicit_credentials()
        return connector

    def get_bigquery_table_name(self, table_name: str) -> str:
        return f"{self.project_id}.{self.dataset_id}.{table_name}"


@kumo_dataclass
class S3ConnectorResourceConfig(SerializedS3ConnectorResourceConfig):
    def as_connector(
        self,
        require_explicit_credentials: bool = False,
    ) -> CompositeFileConnector:
        if isinstance(self.target, api.S3RootedConnectorTarget):
            root_dir = f"s3://{self.target.bucket}"
            if self.target.prefix:
                root_dir = f"{root_dir}/{self.target.prefix.strip('/')}"
            connector = CompositeFileConnector(name=self.name,
                                               root_dir=root_dir)
        elif isinstance(self.target, api.S3AnyURIConnectorTarget):
            connector = AnyURIS3Connector(name=self.name)
        else:
            raise TypeError(f'Unsupported S3 connector target: {self.target}')
        if require_explicit_credentials:
            connector.require_explicit_credentials()
        return connector

    @property
    def region(self) -> Optional[str]:
        if isinstance(self.target, api.S3RootedConnectorTarget):
            return self.target.region
        return None


@kumo_dataclass
class GCSConnectorResourceConfig(SerializedGCSConnectorResourceConfig):
    def as_connector(
        self,
        require_explicit_credentials: bool = False,
    ) -> CompositeFileConnector:
        if isinstance(self.target, api.GCSRootedConnectorTarget):
            root_dir = f"gs://{self.target.bucket}"
            if self.target.prefix:
                root_dir = f"{root_dir}/{self.target.prefix.strip('/')}"
            connector = CompositeFileConnector(name=self.name,
                                               root_dir=root_dir)
        elif isinstance(self.target, api.GCSAnyURIConnectorTarget):
            connector = AnyURIGCSConnector(name=self.name)
        else:
            raise TypeError(f'Unsupported GCS connector target: {self.target}')
        if require_explicit_credentials:
            connector.require_explicit_credentials()
        return connector


@kumo_dataclass
class ADLSConnectorResourceConfig(SerializedADLSConnectorResourceConfig):
    def as_connector(
        self,
        require_explicit_credentials: bool = False,
    ) -> CompositeFileConnector:
        if isinstance(self.target, api.ADLSRootedConnectorTarget):
            root_dir = (f"abfss://{self.target.filesystem}@"
                        f"{self.target.storage_account}."
                        f"{self.target.endpoint_suffix}")
            if self.target.prefix:
                root_dir = f"{root_dir}/{self.target.prefix.strip('/')}"
            connector = CompositeFileConnector(name=self.name,
                                               root_dir=root_dir)
        elif isinstance(self.target, api.ADLSAnyURIConnectorTarget):
            connector = AnyURIADLSConnector(name=self.name)
        else:
            raise TypeError(f'Unsupported ADLS connector target: {self.target}')
        if require_explicit_credentials:
            connector.require_explicit_credentials()
        return connector


_CSV_SUFFIX = 'csv'
_PARQUET_SUFFIX = 'parquet'


@kumo_dataclass(frozen=True)
class FileWithDepth:
    path: str
    depth: int


@kumo_dataclass
class FileConnectorResourceConfig(SerializedFileConnectorResourceConfig):
    """Configuration for a project-owned internal file connector."""
    def as_connector(self) -> CompositeFileConnector:
        return CompositeFileConnector(name=self.name, root_dir=self.root_dir)


ConnectorResourceConfigDB = Union[
    SnowflakeConnectorResourceConfig,
    DatabricksConnectorResourceConfig,
    S3ConnectorResourceConfig,
    GCSConnectorResourceConfig,
    ADLSConnectorResourceConfig,
    FileConnectorResourceConfig,
    BigQueryConnectorResourceConfig,
]

ExternalConnectorResourceConfigDB = Union[
    SnowflakeConnectorResourceConfig,
    DatabricksConnectorResourceConfig,
    BigQueryConnectorResourceConfig,
    S3ConnectorResourceConfig,
    GCSConnectorResourceConfig,
    ADLSConnectorResourceConfig,
]

InternalConnectorResourceConfigDB = FileConnectorResourceConfig

ConnectorResourceConfigAPI = Union[
    SnowflakeConnectorResourceConfig,
    DatabricksConnectorResourceConfig,
    S3ConnectorResourceConfig,
    GCSConnectorResourceConfig,
    ADLSConnectorResourceConfig,
    FileConnectorResourceConfig,
    BigQueryConnectorResourceConfig,
]


@kumo_dataclass
class SampleRowsResponse:
    columns: List[SourceColumn] = field(default_factory=list)
    rows: List[List[Optional[str]]] = field(default_factory=list)
    # Serialized df of sample rows, as of now df.to_json(orient="table")
    sample_df_serialized: Optional[str] = None
    # The response token for retrieving asynchronous sample rows response
    response_token: str = ''


ConnectorResponse = api.ConnectorResponse
CreateConnectorArgs = api.CreateConnectorArgs
UpdateConnectorMetadataArgs = api.UpdateConnectorMetadataArgs


@kumo_dataclass
class SourceTable:
    name: str
    # Deprecated as we don't display it on UI.
    num_cols: int
    downstream_table_ids: List[TableLink] = field(default_factory=list)
    # A source table can be invalid in case it violate our limits.
    valid: bool = True
    reason: str = ""
    path: Optional[str] = None


@kumo_dataclass
class SourceTablesResponse:
    source_tables: List[SourceTable] = field(default_factory=list)
    response_token: str = ''


@kumo_dataclass
class S3Policy:
    policy: str


TEST_KUMO_FILE = "_TEST_KUMO_FILE"


# S3PathArgs
@kumo_dataclass
class S3PathArgs:
    root_dir: str


# GCSPathArgs
@kumo_dataclass
class GCSPathArgs:
    root_dir: str


# Data class to store file details for files in S3 path.
# Currently used only to store "Ignore" details.
@kumo_dataclass
class S3File:
    file_name: str
    details: str  # Used to send reason for ignoring this file

    def startswith(self, prefix: str) -> Boolean:
        return (self.file_name.startswith(prefix))


# Data class to store Table name and other details for files in S3 path.
@kumo_dataclass
class S3Table:
    table_name: str
    is_dir: bool
    table_files: List[S3File]
    file_type: str


# Data class to store information about files in S3 during validation.
@kumo_dataclass
class S3FileList:
    files_selected: List[S3Table]
    files_ignored: List[S3Table]
    count_total: int
    count_selected: int
    count_ignored: int
    warnings: List[str] = field(default_factory=list)


@kumo_dataclass
class SourceColumnsResource:
    source_cols: List[SourceColumn]
    # Validation message to notify the user when dtypes are coerced to a different type automatically
    validation_message: Optional[ValidationResponse] = None
