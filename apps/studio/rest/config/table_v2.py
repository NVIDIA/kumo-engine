# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import hashlib
from enum import Enum
from functools import cached_property
from hashlib import md5
from typing import Any, Dict, List, Optional, Tuple

import kumoapi.data_snapshot as snapshot_api
import kumoapi.data_source as data_source_api
import kumoapi.source_table as source_table_api
import kumoapi.table as table_api
from kumo_core.types import ConnectorType, persisted_type_token
from kumo_core.types.schema import (
    FPID,
    TS_FORMAT_METADATA,
    BaseColumnConfig,
    IngestedTableFingerprint,
    SourceTableID,
    SqlTableFingerprint,
    TableFingerprint,
)
from kumoapi.data_source import DataSourceType as ApiDataSourceType
from kumoapi.typing import Dtype, Stype
from pydantic import field_validator, model_validator

from common_utils.enum import extends_enum
from io_utils import kumo_dataclass, logging
from io_utils.json import from_json, to_json
from kumo.connector.composite_file_connector import CompositeFileConnector
from kumo.graph.graph import BaseColumn
from kumo.graph.graph import Column as KumoColumn
from kumo.graph.graph import (
    Connector,
    ForeignKey,
    PrimaryKey,
    Table,
    TimeColumn,
)
from kumo.validation import ValidationException
from monitoring.tracing import TraceOptions, trace
from rest.config.connector import (
    ADLSConnectorResourceConfig,
    ConnectorResourceConfigDB,
    FileConnectorResourceConfig,
    GCSConnectorResourceConfig,
    S3ConnectorResourceConfig,
    object_store_source_for_connector,
)
from rest.config.file_type import FileType
from rest.config.table import FactTableTimeRange, TableStats
from statedb.config.common import (
    ConnectorID,
    ConnectorLink,
    SourceTableName,
    TableName,
)
from statedb.config.timestamp import ResourceTimestamps
from statedb.rest.table_v2 import TableDBEntryV2 as SerializedTableDBEntryV2

logger = logging.getLogger(__name__)

TimestampUnit = table_api.TimestampUnit

# ---------------------------------------------------------------------------
# Module-level helpers shared by Column/TableDefinition methods.
#
# These are package-private (`_`-prefixed). The Stype/Dtype coercion helpers
# exist only because v2 ``Column.stype``/``Column.dtype`` natively hold the
# kumoapi enums while the rest of the kumo codebase uses the extended
# ``statedb.config.typing`` enums (extra methods like
# ``supports_cast_from_dtype``/``title_case``, plus ``__str__`` returning
# ``'kumo.<value>'``). When ``Column`` is upgraded to natively hold the
# extended types, every call to these helpers becomes a no-op coercion and
# the helpers can be deleted in one sweep.
# ---------------------------------------------------------------------------


def _coerce_stype(stype: Any) -> Optional[Stype]:
    r"""Coerce a stype value (kumoapi enum, extended enum, or value string)
    to the extended ``Stype``. ``None`` passes through.
    """
    if stype is None:
        return None
    if isinstance(stype, Stype):
        return stype
    return Stype(str(stype.value) if hasattr(stype, 'value') else stype)


def _coerce_dtype(dtype: Any) -> Optional[Dtype]:
    r"""Coerce a dtype value (kumoapi enum, extended enum, or value string)
    to the extended ``Dtype``. ``None`` passes through.
    """
    if dtype is None:
        return None
    if isinstance(dtype, Dtype):
        return dtype
    return Dtype(str(dtype.value) if hasattr(dtype, 'value') else dtype)


def _validate_relative_source_table_name(value: str) -> str:
    """Validate a canonical connector-relative object or dataset name."""
    name = str(value)
    if (not name or name.startswith('/') or name.endswith('/') or '\\' in name
            or '://' in name):
        raise ValueError(
            'source_table_name must be a non-empty connector-relative path')
    if any(part in ('', '.', '..') for part in name.split('/')):
        raise ValueError(
            'source_table_name must not contain empty, dot, or parent parts')
    return name


def _object_source_fpid(connector_id: str, coordinate: str, file_type: FileType,
                        data_source_type: 'DataSourceType') -> FPID:
    md5_hasher = md5()
    for value in (connector_id, coordinate, file_type.name, data_source_type):
        md5_hasher.update(value.encode())
    return FPID(md5_hasher.hexdigest())


def _fixture_file_type(coordinate: str, source_kind: str) -> FileType:
    lower = coordinate.rstrip('/').lower()
    if lower.endswith('.parquet'):
        return FileType.PARQUET
    if lower.endswith('.csv'):
        return FileType.CSV
    raise ValueError('Fixture conversion requires an exact .csv or .parquet '
                     f'{source_kind} source coordinate')


def _decompose_timestamp_format(
    col: 'Column', ) -> Tuple[Dict[str, str], Optional[str]]:
    r"""Decompose v2 ``Column.timestamp_format`` into the v1
    ``(metadata, timestamp_unit)`` pair using the dtype-conditional rule
    shared by every v2-to-v1 conversion site (``col_to_base_config``, the
    kumo-engine column converters, and the inline conversions in
    ``update_graph_config_from_graph_input``).
    """
    metadata: Dict[str, str] = {}
    timestamp_unit: Optional[str] = None
    if col.timestamp_format is None:
        return metadata, timestamp_unit
    if isinstance(col.timestamp_format, table_api.TimestampUnit):
        timestamp_unit = str(col.timestamp_format.value)
        if (Dtype.is_string(col.dtype) or Dtype.is_timestamp(col.dtype)):
            metadata[TS_FORMAT_METADATA] = str(col.timestamp_format.value)
    else:
        metadata[TS_FORMAT_METADATA] = str(col.timestamp_format)
    return metadata, timestamp_unit


def column_def_to_foreign_key(
    col: 'Column',
    dst_connector_id: ConnectorID,
    dst_table_name: TableName,
    dst_col_name: str,
) -> ForeignKey:
    r"""Build a kumo-engine ``ForeignKey`` from a v2 ``Column`` plus the
    destination context. Free function (rather than a method) because the
    destination context isn't part of the column itself. Byte-equivalent
    to what the v1 ``column_args_to_foreign_key`` would produce for the
    same column.
    """
    metadata, timestamp_unit = _decompose_timestamp_format(col)
    out = ForeignKey(
        name=col.name,
        dtype=_coerce_dtype(col.dtype),
        dst_table=dst_table_name,
        dst_data_source=dst_connector_id,
        dst_key=dst_col_name,
    )
    out.stype = _coerce_stype(col.stype)
    out.metadata = metadata
    out.timestamp_unit = timestamp_unit
    return out


@kumo_dataclass(frozen=True)
class Column(table_api.Column):
    @field_validator("timestamp_format", mode="before")
    @classmethod
    def _normalize_timestamp_format(cls, v: Any) -> Any:
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @cached_property
    def fpid(self) -> FPID:
        md5_hasher = md5()
        md5_hasher.update(self.name.encode())
        md5_hasher.update(self.stype.name.encode())
        md5_hasher.update(self.dtype.name.encode())
        if self.timestamp_format is not None:
            md5_hasher.update(str(self.timestamp_format).encode())
        # Preserve all existing fingerprints when the optional delimiter is
        # absent, while making distinct scalar encodings non-equivalent.
        if self.multicategorical_delimiter is not None:
            md5_hasher.update(b'multicategorical_delimiter=')
            md5_hasher.update(self.multicategorical_delimiter.encode())
        return FPID(md5_hasher.hexdigest())

    def to_feature_column(self) -> KumoColumn:
        r"""Build a kumo-engine ``Column`` from this v2 column."""
        metadata, timestamp_unit = _decompose_timestamp_format(self)
        return KumoColumn(
            name=self.name,
            stype=_coerce_stype(self.stype),
            dtype=_coerce_dtype(self.dtype),
            metadata=metadata,
            timestamp_unit=timestamp_unit,
            multicategorical_delimiter=self.multicategorical_delimiter,
        )

    def to_primary_key(self) -> PrimaryKey:
        r"""Build a kumo-engine ``PrimaryKey`` from this v2 column. v1
        discards column metadata for pkeys, so we mirror that here.
        """
        _, timestamp_unit = _decompose_timestamp_format(self)
        out = PrimaryKey(name=self.name, dtype=_coerce_dtype(self.dtype))
        out.metadata = {}
        out.timestamp_unit = timestamp_unit
        out.stype = _coerce_stype(self.stype)
        return out

    def to_time_column(self) -> TimeColumn:
        r"""Build a kumo-engine ``TimeColumn`` from this v2 column."""
        metadata, timestamp_unit = _decompose_timestamp_format(self)
        if self.dtype is not None:
            out = TimeColumn(name=self.name, dtype=_coerce_dtype(self.dtype))
        else:
            out = TimeColumn(name=self.name)
        out.metadata = metadata
        out.timestamp_unit = timestamp_unit
        return out


@extends_enum(ApiDataSourceType)
class DataSourceType(str, Enum):
    SNOWFLAKE = "SNOWFLAKE"
    S3 = "S3"
    GCS = "GCS"
    ADLS = "ADLS"
    DATABRICKS = "DATABRICKS"
    BIGQUERY = "BIGQUERY"
    CUSTOM_SQL = "CUSTOM_SQL"
    UNAVAILABLE = "UNAVAILABLE"

    def match_connector_type(self, connector_type: ConnectorType) -> bool:
        if self == DataSourceType.SNOWFLAKE:
            return connector_type == ConnectorType.SNOWFLAKE
        if self == DataSourceType.S3:
            return connector_type == ConnectorType.S3
        if self == DataSourceType.GCS:
            return connector_type == ConnectorType.GCS
        if self == DataSourceType.ADLS:
            return connector_type == ConnectorType.ADLS
        if self == DataSourceType.DATABRICKS:
            return connector_type == ConnectorType.DATABRICKS
        if self == DataSourceType.BIGQUERY:
            return connector_type == ConnectorType.BIGQUERY
        if self == DataSourceType.CUSTOM_SQL:
            # A Custom SQL table runs its query against a warehouse SQL
            # connector; any of the supported warehouses is valid.
            return connector_type in (ConnectorType.SNOWFLAKE,
                                      ConnectorType.DATABRICKS,
                                      ConnectorType.BIGQUERY)
        return False


@kumo_dataclass(frozen=True)
class S3SourceTable(source_table_api.S3SourceTable):
    @field_validator("data_source_type")
    @classmethod
    def _validate_data_source_type(cls, v):
        return DataSourceType(v)

    @model_validator(mode='after')
    def validate_source(self) -> 'S3SourceTable':
        if self.data_source_type != DataSourceType.S3:
            return self
        if not self.connector_id:
            raise ValueError('connector_id is required')
        return self

    @property
    def table_name(self) -> str:
        if isinstance(self.source, source_table_api.S3RelativeSource):
            return self.source.path
        if isinstance(self.source, source_table_api.S3URISource):
            return self.source.uri
        raise TypeError(f'Unsupported S3 source: {self.source}')

    @property
    def table_location(self):
        return self.table_name

    @cached_property
    def fpid(self) -> FPID:
        return _object_source_fpid(self.connector_id, self.table_name,
                                   self.file_type, self.data_source_type)

    def require_connector(
        self,
        connector_config: ConnectorResourceConfigDB,
    ) -> bool:
        if self.connector_id != connector_config.name:
            return False
        if isinstance(connector_config, FileConnectorResourceConfig):
            return isinstance(self.source, source_table_api.S3RelativeSource)
        if not isinstance(connector_config, S3ConnectorResourceConfig):
            return False
        if isinstance(connector_config.target,
                      data_source_api.S3RootedConnectorTarget):
            return isinstance(self.source, source_table_api.S3RelativeSource)
        if isinstance(connector_config.target,
                      data_source_api.S3AnyURIConnectorTarget):
            return isinstance(self.source, source_table_api.S3URISource)
        return False


@kumo_dataclass(frozen=True)
class GCSSourceTable(source_table_api.GCSSourceTable):
    @field_validator("data_source_type")
    @classmethod
    def _validate_data_source_type(cls, v):
        return DataSourceType(v)

    @model_validator(mode='after')
    def validate_source(self) -> 'GCSSourceTable':
        if self.data_source_type != DataSourceType.GCS:
            return self
        if not self.connector_id:
            raise ValueError('connector_id is required')
        return self

    @property
    def source_coordinate(self) -> str:
        if isinstance(self.source, source_table_api.GCSRelativeSource):
            return self.source.path
        if isinstance(self.source, source_table_api.GCSURISource):
            return self.source.uri
        raise TypeError(f'Unsupported GCS source: {self.source}')

    @property
    def table_name(self) -> str:
        return self.source_coordinate

    @property
    def table_location(self):
        return self.source_coordinate

    @cached_property
    def fpid(self) -> FPID:
        return _object_source_fpid(self.connector_id, self.source_coordinate,
                                   self.file_type, self.data_source_type)

    def require_connector(
        self,
        connector_config: ConnectorResourceConfigDB,
    ) -> bool:
        if (self.connector_id != connector_config.name or
                not isinstance(connector_config, GCSConnectorResourceConfig)):
            return False
        if isinstance(connector_config.target,
                      data_source_api.GCSRootedConnectorTarget):
            return isinstance(self.source, source_table_api.GCSRelativeSource)
        if isinstance(connector_config.target,
                      data_source_api.GCSAnyURIConnectorTarget):
            return isinstance(self.source, source_table_api.GCSURISource)
        return False


@kumo_dataclass(frozen=True)
class ADLSSourceTable(source_table_api.ADLSSourceTable):
    @field_validator("data_source_type")
    @classmethod
    def _validate_data_source_type(cls, v):
        return DataSourceType(v)

    @model_validator(mode='after')
    def validate_source(self) -> 'ADLSSourceTable':
        if self.data_source_type != DataSourceType.ADLS:
            return self
        if not self.connector_id:
            raise ValueError('connector_id is required')
        return self

    @property
    def source_coordinate(self) -> str:
        if isinstance(self.source, source_table_api.ADLSRelativeSource):
            return self.source.path
        if isinstance(self.source, source_table_api.ADLSURISource):
            return self.source.uri
        raise TypeError(f'Unsupported ADLS source: {self.source}')

    @property
    def table_name(self) -> str:
        return self.source_coordinate

    @property
    def table_location(self):
        return self.source_coordinate

    @cached_property
    def fpid(self) -> FPID:
        return _object_source_fpid(self.connector_id, self.source_coordinate,
                                   self.file_type, self.data_source_type)

    def require_connector(
        self,
        connector_config: ConnectorResourceConfigDB,
    ) -> bool:
        if (self.connector_id != connector_config.name or
                not isinstance(connector_config, ADLSConnectorResourceConfig)):
            return False
        if isinstance(connector_config.target,
                      data_source_api.ADLSRootedConnectorTarget):
            return isinstance(self.source, source_table_api.ADLSRelativeSource)
        if isinstance(connector_config.target,
                      data_source_api.ADLSAnyURIConnectorTarget):
            return isinstance(self.source, source_table_api.ADLSURISource)
        return False


@kumo_dataclass(frozen=True)
class SnowflakeSourceTable(source_table_api.SnowflakeSourceTable):
    @field_validator("data_source_type")
    @classmethod
    def _validate_data_source_type(cls, v):
        return DataSourceType(v)

    def require_connector(self,
                          connector_config: ConnectorResourceConfigDB) -> bool:
        if connector_config.name == self.snowflake_connector_id:
            if connector_config.type != ConnectorType.SNOWFLAKE:
                raise ValueError(
                    f'Connector {connector_config.name} is expected to be '
                    f'Snowflake but got {connector_config.type}')
            return True
        return False

    @cached_property
    def fpid(self) -> FPID:
        md5_hasher = md5()
        md5_hasher.update(self.snowflake_connector_id.encode())
        # See https://docs.snowflake.com/en/sql-reference/name-resolution
        full_qualified_object_name = '.'.join(
            [self.database, self.schema_name, self.table])
        md5_hasher.update(full_qualified_object_name.encode())
        md5_hasher.update(self.data_source_type.encode())
        return FPID(md5_hasher.hexdigest())

    @property
    def table_location(self):
        return self.database + '.' + self.schema_name + '.' + self.table


@kumo_dataclass(frozen=True)
class DatabricksSourceTable(source_table_api.DatabricksSourceTable):
    @field_validator("data_source_type")
    @classmethod
    def _validate_data_source_type(cls, v):
        return DataSourceType(v)

    def require_connector(self,
                          connector_config: ConnectorResourceConfigDB) -> bool:
        if connector_config.name == self.databricks_connector_id:
            if connector_config.type != ConnectorType.DATABRICKS:
                raise ValueError(
                    f'Connector {connector_config.name} is expected to be '
                    f'Databricks but got {connector_config.type}')
            return True
        return False

    @cached_property
    def fpid(self) -> FPID:
        md5_hasher = md5()
        md5_hasher.update(self.databricks_connector_id.encode())
        md5_hasher.update(self.table.encode())
        md5_hasher.update(self.data_source_type.encode())
        return FPID(md5_hasher.hexdigest())

    @property
    def unqualified_table_name(self) -> SourceTableName:
        r"""Trailing ``table`` component of a possibly-qualified
        ``catalog.schema.table``. ``SourceTableName`` only allows
        alphanumerics/underscores/dashes, so qualifying dots are stripped.
        """
        return SourceTableName(self.table.split('.')[-1])

    # TODO(min) Set the proper table location attribute.
    @property
    def table_location(self):
        return self.table


@kumo_dataclass(frozen=True)
class BigQuerySourceTable(source_table_api.BigQuerySourceTable):
    r"""V2 TableDefinition for BigQuery tables."""
    @field_validator("data_source_type")
    @classmethod
    def _validate_data_source_type(cls, v):
        return DataSourceType(v)

    # TODO(REST): Why do we even need this complex validation instead of
    # just checking the connector name?
    def require_connector(
        self,
        connector_config: ConnectorResourceConfigDB,
    ) -> bool:
        r"""Determine whether the provided connector is required for this
        table. It's currently used to determine whether a connector can be
        deleted safely without affecting any tables."""
        if connector_config.name == self.bigquery_connector_id:
            if connector_config.type != ConnectorType.BIGQUERY:
                raise ValueError(
                    f'Connector {connector_config.name} is expected to be '
                    f'BigQuery but got {connector_config.type}')
            return True
        return False

    @cached_property
    def fpid(self) -> FPID:
        md5_hasher = md5()
        md5_hasher.update(self.bigquery_connector_id.encode())
        fully_qualified_table_name = ".".join([
            self.project_id,
            self.dataset_id,
            self.table_name,
        ])
        md5_hasher.update(fully_qualified_table_name.encode())
        md5_hasher.update(self.data_source_type.encode())
        return FPID(md5_hasher.hexdigest())

    @property
    def table_location(self):
        return ".".join([self.project_id, self.dataset_id, self.table_name])


@kumo_dataclass(frozen=True)
class UnavailableSourceTable(source_table_api.UnavailableSourceTable):
    r"""V2 TableDefinition source for client-side/unavailable tables.

    These tables do not require a connector, but they still participate in
    graph/table fingerprints when public API requests include inline graph
    definitions. Without a stable fingerprint, validation/create requests can
    fail with an internal AttributeError before input validation can run.
    """
    def require_connector(
        self,
        connector_config: ConnectorResourceConfigDB,
    ) -> bool:
        return False

    @cached_property
    def fpid(self) -> FPID:
        md5_hasher = md5()
        data_source_type = self.data_source_type
        if hasattr(data_source_type, 'value'):
            data_source_type = data_source_type.value
        md5_hasher.update(self.table.encode())
        md5_hasher.update(str(data_source_type).encode())
        return FPID(md5_hasher.hexdigest())

    @property
    def table_location(self):
        return self.table


@kumo_dataclass(frozen=True)
class CustomSQLSourceTable(source_table_api.CustomSQLSourceTable):
    r"""V2 TableDefinition source for a Custom SQL Table.

    Rows are produced by a read-only SELECT pushed down to the warehouse; the
    query has no physical source table, so ``table`` is the stable synthetic
    name that cache/staging/data-versioning pipelines key on. The SQL itself is
    not part of the identity here — edits are tracked separately as a data
    version — so the fingerprint stays stable across formatting-only changes.
    """
    @field_validator("data_source_type")
    @classmethod
    def _validate_data_source_type(cls, v):
        return DataSourceType(v)

    def require_connector(self,
                          connector_config: ConnectorResourceConfigDB) -> bool:
        # The backing connector is a warehouse SQL connector
        # (Snowflake/Databricks/BigQuery); its concrete type is enforced when
        # the query is validated, not here.
        return connector_config.name == self.connector_id

    @cached_property
    def fpid(self) -> FPID:
        md5_hasher = md5()
        md5_hasher.update(self.connector_id.encode())
        md5_hasher.update(self.table.encode())
        md5_hasher.update(self.data_source_type.encode())
        return FPID(md5_hasher.hexdigest())

    @property
    def table_location(self):
        return self.table


# TODO(REST): Why do the SourceTable definitions above not follow
# inheritance property?
_FILE_SOURCE_TABLE_TYPES = (S3SourceTable, GCSSourceTable, ADLSSourceTable)

_SOURCE_TABLE_TYPES = (*_FILE_SOURCE_TABLE_TYPES, BigQuerySourceTable,
                       SnowflakeSourceTable, DatabricksSourceTable,
                       CustomSQLSourceTable, UnavailableSourceTable)


def is_file_source_table(table: 'TableDefinition') -> bool:
    """Whether a table is backed by an S3, GCS, or ADLS file source."""
    return isinstance(table.source_table, _FILE_SOURCE_TABLE_TYPES)


@kumo_dataclass
class TableDefinition(table_api.TableDefinition):
    @field_validator("source_table")
    @classmethod
    def _validate_source_table(cls, v):
        # Adapt API source table type.
        if isinstance(v, _SOURCE_TABLE_TYPES):
            return v

        if isinstance(v, source_table_api.S3SourceTable):
            return S3SourceTable(**dataclasses.asdict(v))
        elif isinstance(v, source_table_api.GCSSourceTable):
            return GCSSourceTable(**dataclasses.asdict(v))
        elif isinstance(v, source_table_api.ADLSSourceTable):
            return ADLSSourceTable(**dataclasses.asdict(v))
        elif isinstance(v, source_table_api.BigQuerySourceTable):
            return BigQuerySourceTable(**dataclasses.asdict(v))
        elif isinstance(v, source_table_api.SnowflakeSourceTable):
            return SnowflakeSourceTable(**dataclasses.asdict(v))
        elif isinstance(v, source_table_api.DatabricksSourceTable):
            return DatabricksSourceTable(**dataclasses.asdict(v))
        elif isinstance(v, source_table_api.CustomSQLSourceTable):
            return CustomSQLSourceTable(**dataclasses.asdict(v))
        elif isinstance(v, source_table_api.UnavailableSourceTable):
            return UnavailableSourceTable(**dataclasses.asdict(v))

        # A net for the next `SourceTableType` member: stored raw, it would
        # 500 in the first accessor rather than 422 here.
        raise ValueError(
            f'Source table type {type(v).__name__} is not supported for a '
            f'Kumo table definition.')

    @field_validator("cols")
    @classmethod
    def _validate_cols(cls, cols):
        # Adapt api.Column to Column
        return [
            Column(name=col.name, stype=col.stype, dtype=col.dtype,
                   timestamp_format=col.timestamp_format,
                   multicategorical_delimiter=(col.multicategorical_delimiter))
            for col in cols
        ]

    @cached_property
    def fpid(self) -> FPID:
        # Generate a stable & unique fingerprint id based on the table
        # definition, discarding inferred information (such as dtype)
        md5_hasher = md5()
        md5_hasher.update(self.source_table.fpid.encode())

        # dedupe & sort cols by fpid
        col_fps = [col.fpid for col in self.cols]
        col_fps = dict.fromkeys(sorted(col_fps)).keys()
        md5_hasher.update('|'.join(col_fps).encode())

        md5_hasher.update(str(self.pkey).encode())
        md5_hasher.update(str(self.time_col).encode())
        if self.end_time_col is not None:
            md5_hasher.update(str(self.end_time_col).encode())

        return FPID(md5_hasher.hexdigest())

    @field_validator("time_col", "end_time_col", "pkey")
    @classmethod
    def empty_str_to_none(cls, v: Optional[str]) -> Optional[str]:
        if v == '':
            return None
        return v

    @staticmethod
    def _build_from_inputs(
        *,
        connector_id: ConnectorID,
        connector_config: ConnectorResourceConfigDB,
        source_table_name: str,
        cols: List[Column],
        pkey: Optional[str],
        time_col: Optional[str],
        end_time_col: Optional[str],
    ) -> Optional['TableDefinition']:
        """Connector-aware builder for ``TableDefinition``.

        Encapsulates the connector-type switch, file-path resolution,
        and source-table assembly shared by v2-native fixture
        construction. Callers must already be in a blocking
        context: the FILE branch issues ``fsutil`` I/O via
        ``CompositeFileConnector._get_path`` / ``fsutil.isdir`` /
        ``fsutil.listdir``.

        Returns ``None`` only when the FILE connector cannot resolve the
        source path; raises for unsupported connector types.
        """
        if connector_config.type == ConnectorType.FILE:
            connector_config = FileConnectorResourceConfig(
                name=connector_config.name,
                root_dir=connector_config.root_dir,
            )
            connector = connector_config.as_connector()
            assert isinstance(connector, CompositeFileConnector)
            try:
                source_path = connector._get_path(source_table_name)
            except FileNotFoundError:
                logger.warning(
                    f"Source table {source_table_name} not found in "
                    f"connector {connector_config.name}. Skipping table.")
                return None
            file_type = FileType(connector._get_path_type(source_path).upper())
            src_table = S3SourceTable(
                file_type=file_type,
                connector_id=str(connector_id),
                source=source_table_api.S3RelativeSource(
                    path=source_table_name),
            )
        elif connector_config.type in {
                ConnectorType.S3, ConnectorType.GCS, ConnectorType.ADLS
        }:
            if connector_config.type == ConnectorType.S3:
                if not isinstance(connector_config, S3ConnectorResourceConfig):
                    raise ValueError('Invalid S3 connector configuration')
                if not isinstance(connector_config.target,
                                  (data_source_api.S3RootedConnectorTarget,
                                   data_source_api.S3AnyURIConnectorTarget)):
                    raise ValueError('Unsupported S3 connector target')
                source_cls = S3SourceTable
                source_kind = 'S3'
            else:
                if not isinstance(connector_config, (
                        GCSConnectorResourceConfig,
                        ADLSConnectorResourceConfig,
                )):
                    raise ValueError(
                        'Invalid object-store connector configuration')
                source_cls = (GCSSourceTable if connector_config.type
                              == ConnectorType.GCS else ADLSSourceTable)
                source_kind = 'object'
            source = object_store_source_for_connector(connector_config,
                                                       source_table_name)
            coordinate = getattr(source, 'path', getattr(source, 'uri', None))
            assert coordinate is not None
            src_table = source_cls(
                file_type=_fixture_file_type(coordinate, source_kind),
                connector_id=str(connector_id),
                source=source,
            )
        elif connector_config.type == ConnectorType.SNOWFLAKE:
            src_table = SnowflakeSourceTable(
                snowflake_connector_id=connector_id,
                database=connector_config.database,
                schema_name=connector_config.schema_name,
                table=SourceTableName(source_table_name),
            )
        elif connector_config.type == ConnectorType.DATABRICKS:
            src_table = DatabricksSourceTable(
                databricks_connector_id=connector_id,
                table=SourceTableName(source_table_name),
            )
        elif connector_config.type == ConnectorType.BIGQUERY:
            src_table = BigQuerySourceTable(
                bigquery_connector_id=connector_id,
                table_name=SourceTableName(source_table_name),
                project_id=connector_config.project_id,
                dataset_id=connector_config.dataset_id,
            )
        else:
            raise Exception(
                f"Unsupported connector type: {connector_config.type}")

        return TableDefinition(
            cols=tuple(cols),
            source_table=src_table,
            pkey=pkey,
            time_col=time_col,
            end_time_col=end_time_col,
        )

    @property
    def is_sql(self) -> bool:
        r"""True when this table's rows come from a Custom SQL query pushed
        down to the warehouse rather than from a physical source table."""
        return isinstance(self.source_table, CustomSQLSourceTable)

    def get_connector_id(self) -> ConnectorID:
        r"""Return the durable connector ID for this table definition."""
        src = self.source_table
        if isinstance(src, SnowflakeSourceTable):
            return src.snowflake_connector_id
        if isinstance(src, DatabricksSourceTable):
            return src.databricks_connector_id
        if isinstance(src, BigQuerySourceTable):
            return src.bigquery_connector_id
        if isinstance(src, CustomSQLSourceTable):
            return ConnectorID(src.connector_id)
        if isinstance(src, _FILE_SOURCE_TABLE_TYPES):
            return ConnectorID(src.connector_id)
        raise ValueError(f"Unknown source_table type: {type(src).__name__}")

    def get_source_table_name(self) -> str:
        r"""Return the source-side table name for this TableDefinition,
        matching the legacy v1 source-table name.
        """
        src = self.source_table
        if isinstance(src, SnowflakeSourceTable):
            return str(src.table)
        if isinstance(src, DatabricksSourceTable):
            return str(src.unqualified_table_name)
        if isinstance(src, BigQuerySourceTable):
            return str(src.table_name)
        if isinstance(src, CustomSQLSourceTable):
            return str(src.table)
        if isinstance(src, _FILE_SOURCE_TABLE_TYPES):
            return str(src.table_name)
        raise ValueError(f"Unknown source_table type: {type(src).__name__}")

    @property
    def source_table_id(self) -> SourceTableID:
        r"""Byte-identical to the legacy v1 table ``source_table_id``."""
        return SourceTableID(
            connector_name=self.get_connector_id(),
            source_table_name=self.get_source_table_name(),
        )

    def fingerprint(self) -> TableFingerprint:
        r"""Build the table fingerprint used for cache paths and workflow IDs
        in the data backend.

        Physical tables get an `IngestedTableFingerprint` byte-identical to the
        one the legacy v1 table fingerprint would produce. Custom SQL tables get
        a `SqlTableFingerprint`, whose path-id lives in a separate ``sql``
        namespace so a SQL table can never collide with a same-named physical
        table. Views are not supported in v2.
        """
        cols: Dict[str, Dtype] = {}
        missing: List[str] = []
        for c in self.cols:
            if c.dtype is None:
                missing.append(c.name)
            else:
                cols[c.name] = c.dtype
        if missing:
            raise ValueError(
                f"Cannot fingerprint TableDefinition for "
                f"'{self.get_source_table_name()}' — these columns have no "
                f"dtype: {missing}. Run column dtype inference or set the "
                f"dtype explicitly before fingerprinting.")
        if self.is_sql:
            # The change-detection ``data_version`` is resolved by the
            # materialization path (it needs the referenced tables' versions),
            # not from the static definition, so it is left unset here.
            return SqlTableFingerprint(
                connector_name=self.get_connector_id(),
                source_name=self.get_source_table_name(),
                cols=cols,
                pkey=self.pkey,
                time_col=self.col_to_base_config(self.time_col),
                end_time_col=self.col_to_base_config(self.end_time_col),
            )
        multicategorical_delimiters = {
            col.name: col.multicategorical_delimiter
            for col in self.cols if col.multicategorical_delimiter is not None
        }
        return IngestedTableFingerprint(
            connector_name=self.get_connector_id(),
            source_name=self.get_source_table_name(),
            cols=cols,
            pkey=self.pkey,
            time_col=self.col_to_base_config(self.time_col),
            end_time_col=self.col_to_base_config(self.end_time_col),
            multicategorical_delimiters=multicategorical_delimiters,
        )

    def _dedupe_hash(self) -> str:
        r"""Byte-identical to `table_args_to_dedupe_hash` for the equivalent
        legacy v1 table args. Column names are unique within a table, so sorting
        by name produces the same iteration order as `sorted(ColumnArgs...)`
        (which orders by all fields). Views are not supported in v2.

        Package-private; used by ``cache_key`` and ``wf_key``. The legacy
        ``table_args_to_dedupe_hash`` stringifies pydantic-coerced extended
        ``Stype``/``Dtype`` (``'kumo.<value>'``); v2 cols carry
        the bare kumoapi enums whose ``__str__`` is ``'<value>'``. Funnel
        through ``_coerce_stype``/``_coerce_dtype`` to keep the byte
        sequence identical until v2 ``Column`` natively holds the
        extended types.
        """
        config_hasher = md5()
        config_hasher.update(f'connector={self.get_connector_id()}'.encode())
        config_hasher.update(
            f'source_table={self.get_source_table_name()}'.encode())
        for col in sorted(self.cols, key=lambda c: c.name):
            config_hasher.update(col.name.encode())
            config_hasher.update(
                persisted_type_token(_coerce_stype(col.stype)).encode())
            config_hasher.update(
                persisted_type_token(_coerce_dtype(col.dtype)).encode())
            if col.multicategorical_delimiter is not None:
                config_hasher.update(b'multicategorical_delimiter=')
                config_hasher.update(col.multicategorical_delimiter.encode())
        config_hasher.update(f'pkey={self.pkey}'.encode())
        config_hasher.update(f'time_col={self.time_col}'.encode())
        config_hasher.update(f'end_time_col={self.end_time_col}'.encode())
        return config_hasher.hexdigest()

    def cache_key(self) -> str:
        r"""Unique table key for the data-backend cache/persist dir."""
        source_hash = hashlib.sha256(
            self.get_source_table_name().encode()).hexdigest()
        return f'table_{source_hash}_{self._dedupe_hash()}'

    def wf_key(self) -> str:
        r"""Unique table key for the table-ingestion workflow id."""
        return '.'.join([
            self.fingerprint().fpid(),
            self._dedupe_hash()[:8],
        ])

    def col_to_base_config(
            self, col_name: Optional[str]) -> Optional[BaseColumnConfig]:
        if col_name is None:
            return None
        col = next((c for c in self.cols if c.name == col_name), None)
        if col is None:
            return None
        metadata, timestamp_unit = _decompose_timestamp_format(col)
        return BaseColumnConfig(
            name=col.name,
            stype=_coerce_stype(col.stype),
            dtype=_coerce_dtype(col.dtype),
            metadata=metadata,
            timestamp_unit=timestamp_unit,
            multicategorical_delimiter=col.multicategorical_delimiter,
        )

    def to_base_columns(self) -> List[BaseColumn]:
        r"""Build kumo-engine feature columns for every column in this
        table. v2 equivalent of ``table_args_to_base_columns``.
        """
        return [col.to_feature_column() for col in self.cols]

    def to_col_objects(self) -> List[BaseColumn]:
        r"""Build kumo-engine columns: ``PrimaryKey`` for the pkey col,
        feature columns for the rest. v2 equivalent of
        ``table_args_to_col_objects``.
        """
        out: List[BaseColumn] = []
        for col in self.cols:
            if self.pkey == col.name:
                out.append(col.to_primary_key())
            else:
                out.append(col.to_feature_column())
        return out

    def primary_key_col(self) -> Optional[PrimaryKey]:
        r"""Return the kumo-engine ``PrimaryKey`` for this table, or
        ``None`` if no pkey is set. Raises ``ValidationException`` if
        the pkey name is not in ``cols``.
        """
        if self.pkey is None:
            return None
        for col in self.cols:
            if self.pkey == col.name:
                return col.to_primary_key()
        raise ValidationException(
            message=f"pkey with name '{self.pkey}' is missing from cols")

    def time_column(self) -> Optional[TimeColumn]:
        r"""Return the kumo-engine ``TimeColumn`` for this table's
        ``time_col``, or ``None`` if not set. Raises
        ``ValidationException`` if the named column is not in ``cols``.
        Method (not property) to mirror ``cols`` / ``primary_key_col``
        and to avoid shadowing the existing ``time_col`` string field.
        """
        if self.time_col is None:
            return None
        for col in self.cols:
            if self.time_col == col.name:
                return col.to_time_column()
        raise ValidationException(
            message=f"time_col with name '{self.time_col}' is missing "
            f"from cols")

    def end_time_column(self) -> Optional[TimeColumn]:
        r"""Return the kumo-engine ``TimeColumn`` for this table's
        ``end_time_col``, or ``None`` if not set. Raises
        ``ValidationException`` if the named column is not in ``cols``.
        """
        if self.end_time_col is None:
            return None
        for col in self.cols:
            if self.end_time_col == col.name:
                return col.to_time_column()
        raise ValidationException(
            message=f"end_time_col with name '{self.end_time_col}' is "
            f"missing from cols")


@kumo_dataclass
class TableDBEntryV2(SerializedTableDBEntryV2):
    config: TableDefinition

    # Historical: column dtype information inferred at table creation time.
    # ``config.cols[].dtype`` is the source of truth — current write paths
    # populate ``col_dtypes`` directly from ``config.cols[].dtype``, so the
    # two are guaranteed in sync for entries written by current code.
    # Legacy entries with divergent state are out of scope per Decision 1.
    # Do not read this field; do not write it independently of ``config``.
    col_dtypes: Dict[str, Dtype]

    # v1 @root_validator (post-validation) -> @model_validator(mode='after').
    @model_validator(mode='after')
    def _check_id_invariance(self) -> 'TableDBEntryV2':
        id = FPID(self.id)
        config_value = self.config
        if isinstance(config_value, TableDefinition):
            config = config_value
        elif isinstance(config_value, str):
            config = from_json(config_value, TableDefinition)
        else:
            config = from_json(to_json(config_value), TableDefinition)

        if id != config.fpid:
            raise RuntimeError(
                f"TableV2 DB ID {id} does not match its definition "
                f"fingerprint {config.fpid}.")

        return self


class TableKind(str, Enum):
    r"""Wire discriminator telling the UI whether a table's rows come from a
    physical warehouse/file source (``RAW``) or from a user-defined read-only
    Custom SQL ``SELECT`` (``SQL``).

    The string values match the UI ``TableKind`` enum
    (``ui/src/types/tables.ts``) exactly, so the Tables-list "Kind" filter and
    the detail card classify SQL tables straight off the wire.
    """
    RAW = "raw"
    SQL = "sql"


def derive_table_kind(
        table_def: 'table_api.TableDefinition | TableDefinition') -> TableKind:
    r"""Classify a table as ``SQL`` or ``RAW`` from its source-table
    discriminator.

    Uses a string comparison on ``source_table.data_source_type`` rather than
    ``isinstance`` so the classification is robust across both the kumoapi
    ``CustomSQLSourceTable`` and the REST-mirror subclass, and whether the
    discriminator is the ``DataSourceType`` enum or its bare string value.
    """
    source_table = getattr(table_def, 'source_table', None)
    if getattr(source_table, 'data_source_type',
               None) == source_table_api.CUSTOM_SQL_SOURCE_TABLE_TYPE:
        return TableKind.SQL
    return TableKind.RAW


@kumo_dataclass
class TableResourceResponse:
    """
    A config object containing table template metadata.
    Used as response for `get_table_v2`.
    """
    # Table resource identifier in the format of
    # <16digit hex id>, generated in the backend.
    id: FPID

    # Immutable metadata definition for a table.
    table: table_api.TableDefinition | TableDefinition

    # Human-friendly name alias for the table template.
    # For table with name alias, we can operate the
    # tables resource by either id or alias, e.g.:
    #   GET /tables/<TableResource.id>
    #   GET /tables/<table_name>
    name_alias: Optional[str]

    # Created and last updated timestamps. If table is a project entry,
    # populate project table entry related timestamps.
    timestamps: ResourceTimestamps

    # Table size and number of rows
    stats: Optional[TableStats] = None

    # Data time range for a table
    time_range: Optional[FactTableTimeRange] = None

    # Connector ID and name
    connector: Optional[ConnectorLink] = None

    created_by: str = ""
    created_by_name: str = ""

    # Whether this is a raw (physical source) table or a Custom SQL table.
    # Stamped by the response builders via ``derive_table_kind`` so the UI can
    # badge/filter SQL tables straight off the wire; defaults to RAW.
    table_kind: TableKind = TableKind.RAW


@kumo_dataclass(frozen=True)
class SqlReferenceStypes:
    """Referenced-Kumo-table stypes threaded from the REST/driver layer into a
    Custom SQL metadata-inference activity for the stype-inheritance pre-pass.

    A Custom SQL Table inherits each pass-through output column's stype from the
    external table it reads when that table is independently a Kumo Table. Those
    stypes come from project state (the state DB), which only the request/driver
    process can read -- a bare connector activity cannot. The driver resolves
    them (``_sql_reference_stype_hints``) and threads this pydantic dataclass
    into the activity so the ``Stype``/``ConnectorType`` enums survive Temporal
    serialization intact: a bare ``dict`` of enums degrades to plain strings
    across the single-activity workflow boundary, whereas a class-tagged payload
    round-trips its declared field types.

    ``stypes`` maps a dialect-normalized referenced-table name to its
    dialect-normalized column stypes; ``connector_type`` is the warehouse dialect
    those names were normalized under.
    """
    connector_type: ConnectorType
    stypes: Dict[str, Dict[str, Stype]]


@kumo_dataclass
class SnapshotStat:
    value: Any
    approx: bool
    data_version: Optional[Any] = None
    num_entries: Optional[int] = None
    timestamp: Optional[float] = None


@kumo_dataclass
class TableSnapshotColumnStats(snapshot_api.ColumnStats):
    stats: Dict[str, SnapshotStat]


@kumo_dataclass
class TableSnapshotResource(snapshot_api.TableSnapshotResource):
    # TODO: Unify TableSnapshotResource classes, which enables removing these
    # three defaults and making timestamps and connector non-optional
    timestamps: ResourceTimestamps | None = None
    connector: ConnectorLink | None = None
    name_alias: str | None = None
    column_stats: Optional[List[TableSnapshotColumnStats]] = None

    stats: TableStats | None = None
    time_range: FactTableTimeRange | None = None


@kumo_dataclass
class TableSnapshotResponse:
    snapshot: TableSnapshotResource
    definition: table_api.TableDefinition


@kumo_dataclass
class TableReferenceMetadata:
    """
    Explicitly used for UI to SDK.
    Differs from TableConfig by
    retaining all the different ways a table can be
    aliased.
    This is particularly important for unnamed tables
    because they can be aliased a few different ways but still
    point to the same table in the end.
    """
    source_table_name: str
    table_fpid: FPID
    table_config: Dict[str, Any]
    table_alias: Optional[str] = None
    # If table alias doesn't match the table alias in the graph.
    # Applicable mainly for unnamed tables. Field has to be
    # present for tables in graphs.
    table_alias_in_graph: Optional[str] = None


@trace(TraceOptions(trace_params=True))
def table_args_to_kumo_table(
    args: TableDefinition,
    connector: Connector,
) -> Table:
    r"""Convert a v2 :obj:`TableDefinition` to a Kumo :obj:`Table` backend
    data structure. (v2 does not model views.)
    """
    cols = args.to_base_columns()
    table = Table(
        table_name=TableName('NA'),
        data_source="",
        source_name=args.get_source_table_name(),
        cols=cols,
        data_version=None,
        custom_sql=args.source_table.sql if args.is_sql else None,
    )
    table.connector = connector
    if args.pkey:
        table.set_pkey(args.pkey, dtype=table[args.pkey].dtype)
    for time_col_name in (args.time_col, args.end_time_col):
        if not time_col_name:
            continue
        metadata = table[time_col_name].metadata
        timestamp_unit = table[time_col_name].timestamp_unit
        if time_col_name == args.time_col:
            table.set_time_col(time_col_name, dtype=table[time_col_name].dtype)
            table.time_col.metadata = metadata
            table.time_col.timestamp_unit = timestamp_unit
        else:
            table.set_end_time_col(time_col_name,
                                   dtype=table[time_col_name].dtype)
            table.end_time_col.metadata = metadata
            table.end_time_col.timestamp_unit = timestamp_unit
    return table
