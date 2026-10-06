# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT

from dataclasses import field
from enum import Enum
from typing import List, Optional, Union

from pydantic import Field
from pydantic.dataclasses import dataclass

from kumoapi.common import ValidationError
from kumoapi.source_table import CustomSQLSourceTable, SourceTableType
from kumoapi.typing import (
    _FORBID_EXTRA_CONFIG,
    Dtype,
    MulticategoricalDelimiter,
    Stype,
    compatible_field_validator,
    validate_multicategorical_delimiter,
)


class TimestampUnit(Enum):
    r"""A timestamp unit for a column in a Kumo table."""

    #: Specified in seconds
    SECOND = 's'

    #: Specified in milliseconds
    MILLISECOND = 'ms'

    #: Specified in microseconds
    MICROSECOND = 'us'

    #: Specified in nanoseconds
    NANOSECOND = 'ns'


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class Column:
    r"""A column in a Kumo table.

    ``multicategorical_delimiter`` records how a scalar string
    multi-categorical column was split when its metadata was inferred. Kumo
    persists this value in the table definition so every later validation,
    snapshot, and training operation uses the same interpretation. It may be
    a comma (``,``), pipe (``|``), semicolon (``;``), or tab (``\t``).

    Resolved scalar string multi-categorical columns must carry this proof.
    Only the pre-inference :class:`ColumnMetadataRequest` may defer it. Native
    list-valued multi-categorical columns are already structurally separated
    and therefore do not use a delimiter. Columns with another semantic type
    cannot specify one either.
    """
    name: str
    stype: Stype
    dtype: Dtype

    # If the column represents a timestamp, the format that the timestmap
    # is represented in:
    timestamp_format: Optional[Union[TimestampUnit, str]] = None

    # The proven delimiter for a scalar string multi-categorical column.
    multicategorical_delimiter: Optional[MulticategoricalDelimiter] = None

    def _validate_multicategorical_delimiter(self) -> None:
        validate_multicategorical_delimiter(
            self.stype,
            self.dtype,
            self.multicategorical_delimiter,
        )

    def __post_init__(self) -> None:
        self._validate_multicategorical_delimiter()


@dataclass(config=_FORBID_EXTRA_CONFIG)
class TableDefinition:
    r"""A definition of a Kumo table."""
    # List of ALL columns selected from source table
    cols: List[Column]

    source_table: SourceTableType = Field(discriminator='data_source_type')

    # Name of the primary key column.
    pkey: Optional[str] = None

    # Name of the time column, required to have stype=SemanticType.timestamp
    time_col: Optional[str] = None

    # Name of the end time column:
    end_time_col: Optional[str] = None

    @compatible_field_validator("time_col", "end_time_col", "pkey")
    def empty_str_to_none(cls, v: Optional[str]) -> Optional[str]:
        if v == '':
            return None
        return v


@dataclass
class TableResource:
    # Table resource identifier in the format of "table-<16digit hex id>",
    # this is generated on the backend.
    id: str

    # Immutable metadata definition for a table.
    table: TableDefinition

    # Optional human-friendly name alias for the table. Tables can be created
    # with or without name alias. However, once a table resource sets the name
    # alias, it cannot be renamed. Name aliases must be unique, in other words,
    # 1:1 mapping is required between graph name aliases to TableResource.id
    # Effectively, name_alias is an optional, secondary identifier specified by
    # the client.  This means for graph with name alias, we can operate the
    # graph resource by either id or alias, e.g.:
    #   GET /tables/<TableResource.id>
    #   GET /tables/<name_alias>
    name_alias: Optional[str] = None


# Method: Infer Metadata ======================================================


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class ColumnMetadataRequest:
    r"""A request to infer metadata for a column in a Kumo table. This request
    can be incomplete in its stype, dtype, or timestamp format."""
    name: str
    stype: Optional[Stype] = None
    dtype: Optional[Dtype] = None

    # If the column represents a timestamp, the format that the timestmap
    # is represented in:
    timestamp_format: Optional[Union[TimestampUnit, str]] = None

    # A caller-provided delimiter hint. Types may remain absent in this
    # pre-inference shape; the resolved response is a fully validated Column.
    multicategorical_delimiter: Optional[MulticategoricalDelimiter] = None

    def _validate_multicategorical_delimiter(self) -> None:
        validate_multicategorical_delimiter(
            self.stype,
            self.dtype,
            self.multicategorical_delimiter,
            allow_incomplete_types=True,
        )

    def __post_init__(self) -> None:
        self._validate_multicategorical_delimiter()


@dataclass(config=_FORBID_EXTRA_CONFIG)
class TableMetadataRequest:
    r"""A request to infer Kumo table metadata."""
    cols: List[ColumnMetadataRequest]
    source_table: SourceTableType = Field(discriminator='data_source_type')
    pkey: Optional[str] = None
    time_col: Optional[str] = None
    end_time_col: Optional[str] = None


@dataclass
class TableMetadataResponse:
    r"""A response containing metadata for a Kumo table."""
    cols: List[Column]
    source_table: SourceTableType = Field(discriminator='data_source_type')
    pkey: Optional[str] = None
    time_col: Optional[str] = None
    end_time_col: Optional[str] = None


# Method: Resolve Custom SQL ================================================


@dataclass(config=_FORBID_EXTRA_CONFIG)
class SqlTableResolveRequest:
    r"""Resolve a Custom SQL source and optional column and role hints."""
    source_table: CustomSQLSourceTable
    # None discovers every query output; a list selects those outputs
    # explicitly.
    column_hints: Optional[List[ColumnMetadataRequest]] = None
    pkey: Optional[str] = None
    time_col: Optional[str] = None
    end_time_col: Optional[str] = None
    preview_rows: int = 10

    def _validate_column_hints(self) -> None:
        if self.column_hints == []:
            raise ValueError('column_hints must be omitted or non-empty')
        if not 1 <= self.preview_rows <= 1000:
            raise ValueError('preview_rows must be between 1 and 1000')

    def __post_init__(self) -> None:
        self._validate_column_hints()


@dataclass
class SqlTableResolveResponse:
    r"""A resolved table definition or validation errors, with a bounded
    preview."""
    table_definition: Optional[TableDefinition] = None
    errors: List[ValidationError] = field(default_factory=list)
    preview_rows: List[List[Optional[str]]] = field(default_factory=list)
    preview_note: Optional[str] = None

    def _validate_result(self) -> None:
        has_definition = self.table_definition is not None
        has_errors = bool(self.errors)
        if has_definition == has_errors:
            raise ValueError(
                'SQL resolution must return either a definition or errors')
        if has_errors and (self.preview_rows or self.preview_note is not None):
            raise ValueError('SQL resolution errors cannot include a preview')

    def __post_init__(self) -> None:
        self._validate_result()


# Method: Validate ============================================================


@dataclass(config=_FORBID_EXTRA_CONFIG)
class TableValidationRequest:
    # NOTE Response is of type common.ValidationResponse
    table_definition: TableDefinition
