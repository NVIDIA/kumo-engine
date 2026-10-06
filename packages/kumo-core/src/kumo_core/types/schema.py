# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable graph-schema identities used by native training."""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass as standard_dataclass
from dataclasses import field
from typing import Annotated, Literal, TypeAlias

from kumoapi.typing import Dtype, Stype
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic_core import core_schema

from ..utils.mixins import CastMixin

TS_FORMAT_METADATA = "TS_FORMAT"
_DATACLASS_CONFIG = ConfigDict(arbitrary_types_allowed=True)


@pydantic_dataclass(config=_DATACLASS_CONFIG, unsafe_hash=True)
class BaseColumnConfig(CastMixin):
    name: str
    stype: Stype | None = None
    dtype: Dtype | None = None
    na_values: list[str] | None = None
    pre_transform: object | None = field(default=None, hash=False)
    metadata: dict[str, object] = field(default_factory=dict, hash=False)
    timestamp_unit: str | None = None
    multicategorical_delimiter: str | None = None

    def update(self, obj: "BaseColumnConfig",
               override: bool = True) -> "BaseColumnConfig":
        def is_empty(key: str, value: object) -> bool:
            return value is None or key == "metadata" and value == {}

        for key in self.__dict__:
            if key.startswith("_"):
                continue
            value = getattr(obj, key, None)
            if not is_empty(key, value) and (override or is_empty(key, value)):
                setattr(self, key, value)
        return self


@pydantic_dataclass(config=_DATACLASS_CONFIG, order=True)
class ColumnArgs:
    name: str
    stype: Stype
    dtype: Dtype | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    timestamp_unit: str | None = None
    multicategorical_delimiter: str | None = None

    def __hash__(self) -> int:
        value = self.name + str(self.stype) + str(self.dtype)
        if self.multicategorical_delimiter is not None:
            value += ("multicategorical_delimiter=" +
                      self.multicategorical_delimiter)
        return hash(value)

    def as_base_column_config(self) -> BaseColumnConfig:
        return BaseColumnConfig(
            name=self.name,
            stype=self.stype,
            dtype=self.dtype,
            metadata=self.metadata,
            timestamp_unit=self.timestamp_unit,
            multicategorical_delimiter=self.multicategorical_delimiter,
        )


@standard_dataclass(frozen=True)
class VersionedSourceTableID:
    connector_name: str
    source_table_name: str
    version: str


@standard_dataclass(frozen=True)
class SourceTableID:
    connector_name: str
    source_table_name: str

    def make_versioned_id(self, version: str) -> VersionedSourceTableID:
        return VersionedSourceTableID(self.connector_name,
                                      self.source_table_name, version)


class FPID(str):
    def __new__(cls, value: str) -> "FPID":
        if len(value) == 0:
            raise ValueError("Please fill out this value")
        if not re.fullmatch(r"[a-f0-9]{32}", value):
            raise ValueError(
                f"Fingerprint ID must be 32-char hex digits, got {value}")
        return str.__new__(cls, value)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: object,
        handler: object,
    ) -> core_schema.CoreSchema:
        del source_type, handler
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.str_schema(),
        )

    @classmethod
    def is_valid(cls, value: str) -> bool:
        try:
            cls(value)
        except ValueError:
            return False
        return True


def source_name_path_token(source_name: str) -> str:
    return "src_" + hashlib.sha256(source_name.encode()).hexdigest()


@pydantic_dataclass(config=_DATACLASS_CONFIG)
class BaseTableFingerprint(ABC):
    connector_name: str
    source_name: str

    def fpid(self) -> str:
        return self._fpid()

    @abstractmethod
    def _fpid(self) -> str:
        raise NotImplementedError

    @property
    def version(self) -> int:
        return 3


def time_col_config_to_fp(time_col: BaseColumnConfig | None) -> str | None:
    if time_col is None:
        return None
    unit = time_col.timestamp_unit or ""
    format_metadata = time_col.metadata.get(TS_FORMAT_METADATA, "")
    return f"{time_col.name}:{unit}:{format_metadata}"


@pydantic_dataclass(config=_DATACLASS_CONFIG)
class IngestedTableFingerprint(BaseTableFingerprint):
    cols: dict[str, Dtype]
    pkey: str | None
    time_col: BaseColumnConfig | None
    end_time_col: BaseColumnConfig | None
    multicategorical_delimiters: dict[str, str] = field(default_factory=dict)
    _type: Literal["ingested"] = "ingested"

    def _fpid(self) -> str:
        values: dict[str, object] = {
            "connector_name": self.connector_name,
            "source_name": self.source_name,
            "cols": sorted(self.cols.items()),
            "time_col": time_col_config_to_fp(self.time_col),
            "end_time_col": time_col_config_to_fp(self.end_time_col),
        }
        if self.multicategorical_delimiters:
            values["multicategorical_delimiters"] = sorted(
                self.multicategorical_delimiters.items())
        payload = json.dumps(values, sort_keys=True, default=str)
        return (f"fpv{self.version}.{self.connector_name[:4]}."
                f"{source_name_path_token(self.source_name)[:12]}."
                f"{hashlib.md5(payload.encode()).hexdigest()}")

    @property
    def ingest_path_fpid(self) -> str:
        schema_payload = json.dumps(sorted(self.cols.items()), default=str)
        schema_fpid = hashlib.sha256(schema_payload.encode()).hexdigest()
        # Staged Parquet is projected and cast from ``cols``. Keep each output
        # schema immutable so graph snapshots cannot reuse incompatible bytes.
        return (f"fpv{self.version}.{self.connector_name}."
                f"{source_name_path_token(self.source_name)}."
                f"schema_{schema_fpid}")


@pydantic_dataclass(config=_DATACLASS_CONFIG)
class ViewFingerprint(BaseTableFingerprint):
    _type: Literal["view"] = "view"

    def _fpid(self) -> str:
        return f"{self.connector_name}.{source_name_path_token(self.source_name)}"

    @property
    def ingest_path_fpid(self) -> str:
        return self._fpid()


@pydantic_dataclass(config=_DATACLASS_CONFIG)
class SqlTableFingerprint(BaseTableFingerprint):
    cols: dict[str, Dtype]
    pkey: str | None
    time_col: BaseColumnConfig | None
    end_time_col: BaseColumnConfig | None
    data_version: str | None = None
    _type: Literal["sql_table"] = "sql_table"

    def _fpid(self) -> str:
        return (f"fpv{self.version}.sql.{self.connector_name}."
                f"{self.source_name}")

    @property
    def ingest_path_fpid(self) -> str:
        return self._fpid()


@pydantic_dataclass(config=_DATACLASS_CONFIG)
class PartialTableFingerprint(BaseTableFingerprint):
    cols: list[str]
    pkey: str | None
    time_col: BaseColumnConfig | None
    end_time_col: BaseColumnConfig | None
    _type: Literal["partial_ingested"] = "partial_ingested"

    def _fpid(self) -> str:
        values = {
            "connector_name": self.connector_name,
            "source_name": self.source_name,
            "cols": sorted(self.cols),
            "time_col": time_col_config_to_fp(self.time_col),
            "end_time_col": time_col_config_to_fp(self.end_time_col),
        }
        payload = json.dumps(values, sort_keys=True, default=str)
        return (f"fpv{self.version}.{self.connector_name[:4]}."
                f"{source_name_path_token(self.source_name)[:12]}."
                f"{hashlib.md5(payload.encode()).hexdigest()}")


@pydantic_dataclass(config=_DATACLASS_CONFIG)
class TrainingTableFingerprint(BaseTableFingerprint):
    _type: Literal["training"] = "training"

    def _fpid(self) -> str:
        return f"{self.connector_name}.{source_name_path_token(self.source_name)}"


TableFingerprint: TypeAlias = Annotated[
    IngestedTableFingerprint | PartialTableFingerprint | ViewFingerprint
    | SqlTableFingerprint | TrainingTableFingerprint,
    Field(discriminator="_type"),
]

__all__ = [
    "BaseColumnConfig",
    "BaseTableFingerprint",
    "ColumnArgs",
    "FPID",
    "IngestedTableFingerprint",
    "PartialTableFingerprint",
    "SourceTableID",
    "SqlTableFingerprint",
    "TS_FORMAT_METADATA",
    "TableFingerprint",
    "TrainingTableFingerprint",
    "VersionedSourceTableID",
    "ViewFingerprint",
    "source_name_path_token",
    "time_col_config_to_fp",
]
