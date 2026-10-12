# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import field
from typing import List, Optional, Union
from urllib.parse import unquote, urlsplit

from pydantic import Field
from pydantic.dataclasses import dataclass
from typing_extensions import Literal

from kumoapi._legacy_object_store import (
    migrate_source_table,
    migrate_validator_values,
)
from kumoapi.adls import (
    normalize_adls_source_relative_path,
    normalize_adls_source_uri,
)
from kumoapi.common import StrEnum
from kumoapi.data_source import ConnectorType, DataSourceType
from kumoapi.gcs import (
    normalize_gcs_source_relative_path,
    normalize_gcs_source_uri,
)
from kumoapi.s3 import (
    normalize_s3_source_relative_path,
    normalize_s3_source_uri,
)
from kumoapi.typing import (
    _FORBID_EXTRA_CONFIG,
    Dtype,
    MulticategoricalDelimiter,
    Stype,
    compatible_field_validator,
    compatible_model_validator,
    validate_multicategorical_delimiter,
)

TableName = str

# Source Table ================================================================


class FileType(StrEnum):
    r"""Supported file types for file-based source tables."""
    CSV = "CSV"
    PARQUET = "PARQUET"


def _table_name_from_object_store_path(path: str) -> TableName:
    """Derive a display name without truncating dotted directory names."""
    is_directory = path.endswith('/')
    filename = path.rstrip('/').rsplit('/', 1)[-1]
    if is_directory:
        return TableName(filename)
    lowercase_filename = filename.casefold()
    for file_type in FileType:
        suffix = f'.{file_type.value.casefold()}'
        if (len(filename) > len(suffix)
                and lowercase_filename.endswith(suffix)):
            return TableName(filename[:-len(suffix)])
    return TableName(filename)


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class UnavailableSourceTable:
    r"""A source table not available during processing, e.g. because it is
    processed by client-side."""
    table: TableName
    data_source_type: Literal[
        DataSourceType.UNAVAILABLE] = DataSourceType.UNAVAILABLE


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class S3RelativeSource:
    r"""A table path beneath a rooted S3 connector target."""
    path: TableName
    mode: Literal['relative'] = 'relative'

    @compatible_field_validator('path')
    def normalize_path(cls, value: str) -> str:
        return normalize_s3_source_relative_path(value)


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class S3URISource:
    r"""A canonical full S3 URI used by an any-URI connector."""
    uri: str
    mode: Literal['uri'] = 'uri'

    @compatible_field_validator('uri')
    def normalize_uri(cls, value: str) -> str:
        return normalize_s3_source_uri(value)


S3Source = Union[S3RelativeSource, S3URISource]


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class GCSRelativeSource:
    r"""A table path beneath a rooted GCS connector target."""
    path: TableName
    mode: Literal['relative'] = 'relative'

    @compatible_field_validator('path')
    def normalize_path(cls, value: str) -> str:
        return normalize_gcs_source_relative_path(value)


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class GCSURISource:
    r"""A canonical full GCS URI used by an any-URI connector."""
    uri: str
    mode: Literal['uri'] = 'uri'

    @compatible_field_validator('uri')
    def normalize_uri(cls, value: str) -> str:
        return normalize_gcs_source_uri(value)


GCSSource = Union[GCSRelativeSource, GCSURISource]


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class ADLSRelativeSource:
    r"""A table path beneath a rooted ADLS connector target."""
    path: TableName
    mode: Literal['relative'] = 'relative'

    @compatible_field_validator('path')
    def normalize_path(cls, value: str) -> str:
        return normalize_adls_source_relative_path(value)


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class ADLSURISource:
    r"""A canonical full ADLS URI used by an any-URI connector."""
    uri: str
    mode: Literal['uri'] = 'uri'

    @compatible_field_validator('uri')
    def normalize_uri(cls, value: str) -> str:
        return normalize_adls_source_uri(value)


ADLSSource = Union[ADLSRelativeSource, ADLSURISource]


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class S3SourceTable:
    r"""An S3 table with an explicit rooted-relative or any-URI source.

    Servers enforce ``S3RelativeSource`` only with rooted connectors and
    ``S3URISource`` only with any-URI connectors.
    """
    connector_id: str
    file_type: FileType
    source: S3Source = Field(discriminator='mode')

    data_source_type: Literal[DataSourceType.S3] = DataSourceType.S3

    @property
    def table(self) -> TableName:
        path = (self.source.path if isinstance(self.source, S3RelativeSource)
                else unquote(urlsplit(self.source.uri).path))
        return _table_name_from_object_store_path(path)


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class SnowflakeSourceTable:
    r"""A source table located in the Snowflake data warehouse."""
    snowflake_connector_id: str
    database: str
    schema_name: str
    table: TableName
    data_source_type: Literal[
        DataSourceType.SNOWFLAKE] = DataSourceType.SNOWFLAKE


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class DatabricksSourceTable:
    r"""A source table located in the Databricks data warehouse."""
    databricks_connector_id: str
    table: TableName
    data_source_type: Literal[
        DataSourceType.DATABRICKS] = DataSourceType.DATABRICKS


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class BigQuerySourceTable:
    r"""A source table loated in the BigQuery data warehouse."""
    bigquery_connector_id: str
    table_name: TableName
    project_id: str
    dataset_id: str
    data_source_type: Literal[
        DataSourceType.BIGQUERY] = DataSourceType.BIGQUERY


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class GCSSourceTable:
    r"""A GCS table with an explicit rooted-relative or any-URI source."""
    connector_id: str
    file_type: FileType
    source: GCSSource = Field(discriminator='mode')

    data_source_type: Literal[DataSourceType.GCS] = DataSourceType.GCS

    @compatible_model_validator(mode='before')
    @classmethod
    def migrate_legacy_relative_source(cls, values):
        return migrate_validator_values(
            values, lambda data: migrate_source_table(data, 'gcs'))

    @property
    def source_table_name(self) -> TableName:
        if not isinstance(self.source, GCSRelativeSource):
            raise AttributeError(
                'Any-URI GCS tables do not have a relative name')
        return self.source.path

    @property
    def table(self) -> TableName:
        path = (self.source.path if isinstance(self.source, GCSRelativeSource)
                else unquote(urlsplit(self.source.uri).path))
        return _table_name_from_object_store_path(path)


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class ADLSSourceTable:
    r"""An ADLS table with an explicit rooted-relative or any-URI source."""
    connector_id: str
    file_type: FileType
    source: ADLSSource = Field(discriminator='mode')

    data_source_type: Literal[DataSourceType.ADLS] = DataSourceType.ADLS

    @compatible_model_validator(mode='before')
    @classmethod
    def migrate_legacy_relative_source(cls, values):
        return migrate_validator_values(
            values, lambda data: migrate_source_table(data, 'adls'))

    @property
    def source_table_name(self) -> TableName:
        if not isinstance(self.source, ADLSRelativeSource):
            raise AttributeError(
                'Any-URI ADLS tables do not have a relative name')
        return self.source.path

    @property
    def table(self) -> TableName:
        path = (self.source.path if isinstance(self.source, ADLSRelativeSource)
                else unquote(urlsplit(self.source.uri).path))
        return _table_name_from_object_store_path(path)


# The on-the-wire discriminator tag for a Custom SQL source table. It is a
# plain string that is deliberately *not* a ``DataSourceType`` member: Custom
# SQL is a kind of source table, not a data source, so it discriminates the
# union without polluting the data-source taxonomy.
CUSTOM_SQL_SOURCE_TABLE_TYPE = "CUSTOM_SQL"
CustomSQLSourceTableType = Literal["CUSTOM_SQL"]


@dataclass(frozen=True)
class CustomSQLSourceTable:
    r"""A source table whose rows are produced by a read-only SELECT that runs
    in the customer's data warehouse.

    The query is pushed down and executed unchanged; only its result set is
    ingested. As the query has no underlying physical table, ``table``
    carries a stable synthetic identifier that downstream cache, staging, and
    data-versioning pipelines key on.

    ``connector_type`` and ``referenced_tables`` are server-owned. The server
    derives ``connector_type`` from ``connector_id`` (recording which
    warehouse backs the query --
    snowflake/databricks/bigquery -- so the persisted def is self-describing;
    file-based connectors are rejected) and parses ``referenced_tables`` from
    ``sql`` (the external tables the query reads, used for lineage and change
    detection). Unresolved requests omit both fields; resolved definitions
    carry the authoritative values through later lifecycle operations.
    """
    connector_id: str
    sql: str
    table: TableName
    connector_type: Optional[ConnectorType] = None
    referenced_tables: List[str] = field(default_factory=list)
    data_source_type: CustomSQLSourceTableType = CUSTOM_SQL_SOURCE_TABLE_TYPE


LOCAL_FILE_SOURCE_TABLE_TYPE = "LOCAL_FILE"
LocalFileSourceTableType = Literal["LOCAL_FILE"]


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class LocalFileSourceTable:
    r"""A source table that is a file on the filesystem of the reading process.

    Deliberately *not* a member of :obj:`SourceTableType`: it travels only on
    :attr:`~kumoapi.jobs.CustomTrainingTable.source_table`.
    """
    path: str
    file_type: FileType = FileType.PARQUET
    data_source_type: LocalFileSourceTableType = LOCAL_FILE_SOURCE_TABLE_TYPE

    @compatible_field_validator('path')
    def normalize_path(cls, value: str) -> str:
        path = str(value).strip()
        if not path:
            raise ValueError('local file source requires a non-empty path')
        return os.path.abspath(os.path.expanduser(path))

    @property
    def table(self) -> TableName:
        return _table_name_from_object_store_path(self.path)


SourceTableType = Union[
    S3SourceTable,
    SnowflakeSourceTable,
    DatabricksSourceTable,
    BigQuerySourceTable,
    GCSSourceTable,
    ADLSSourceTable,
    CustomSQLSourceTable,
    UnavailableSourceTable,
]

# Method: Configuration =======================================================


@dataclass
class SourceTableConfigResponse:
    source_table: SourceTableType = Field(discriminator='data_source_type')


# Method: Validate ============================================================


@dataclass
class SourceTableValidateResponse:
    is_valid: bool
    # In case there's a suggestion, a warning, or an error:
    msg: str


# Method: List ================================================================


@dataclass
class SourceTableListResponse:
    table_names: List[str]


# Method: Get Data ============================================================


@dataclass(frozen=True)
class SourceColumn:
    r"""The metadata of a column in a source table. Note that a source column
    simply provides a view into the metadata of a source table. To modify
    metadata, please create a Kumo Table and adjust the table's data and
    semantic types.

    .. note::
        Semantic types are inferred based on data types only, and thus may not
        be accurate.

    Args:
        name (str): The name of the column.
        stype (Stype, optional): The semantic type of the column.
        dtype (Dtype): The data type of the column
        is_primary (bool): Whether the column refers to a primary key.
        multicategorical_delimiter: The proven separator for a scalar string
            multi-categorical source column, if one was inferred.
    """
    name: str
    stype: Optional[Stype]  # Kumo-inferred.
    dtype: Dtype
    is_primary: bool
    multicategorical_delimiter: Optional[MulticategoricalDelimiter] = None

    def _validate_multicategorical_delimiter(self) -> None:
        validate_multicategorical_delimiter(
            self.stype,
            self.dtype,
            self.multicategorical_delimiter,
        )

    def __post_init__(self) -> None:
        self._validate_multicategorical_delimiter()


@dataclass
class SourceTableDataResponse:
    table_name: TableName
    cols: List[SourceColumn] = field(default_factory=list)

    # Serialized (json) data of sample rows dataframe, if requested:
    # TODO(siyang,manan): figure out the ser/de protocol for pandas dataframe
    sample_rows: Optional[str] = None


@dataclass
class TableStats:
    r"""Minimal size statistics for a source table or table snapshot.

    Args:
        size_bytes (int): The size of the table in bytes.
        num_rows (int): The number of rows in the table.
    """
    size_bytes: int
    num_rows: int
    # TODO(siyang): add a flag to indicate if stats are exact or approx?
