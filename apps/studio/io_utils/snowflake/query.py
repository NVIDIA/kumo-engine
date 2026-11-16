# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TypeVar

from pydantic import BaseModel, Field, ValidationError, fields
from snowflake.connector.constants import FIELD_ID_TO_NAME, FIELD_NAME_TO_ID
from snowflake.connector.cursor import ResultMetadata, SnowflakeCursor

from io_utils import logging

logger = logging.getLogger(__name__)

DataclassType = TypeVar('DataclassType', bound=BaseModel)


class SnowflakeField:
    """
    Helper functions for constructing dataclass fields
    corresponding to snowflake datatypes
    """
    @staticmethod
    def _field(snowflake_type: str):
        type_code = FIELD_NAME_TO_ID[snowflake_type.upper()]
        # N.B. v1 passed `type_code` as a bare extra kwarg to Field(); v2 routes
        # arbitrary extras through json_schema_extra, read back via
        # model_fields[name].json_schema_extra['type_code'].
        return Field(json_schema_extra={'type_code': type_code})

    @staticmethod
    def text():
        return SnowflakeField._field('TEXT')

    @staticmethod
    def fixed():
        return SnowflakeField._field('FIXED')

    @staticmethod
    def boolean():
        return SnowflakeField._field('BOOLEAN')

    @staticmethod
    def timestamp_ntz():
        return SnowflakeField._field('TIMESTAMP_NTZ')


class SnowflakeQuery:
    """Utility functions for type-safe Snowflake queries

    Intended usage:
    - Create a subclass of this class
    - Declare result dataclasses
      - subclass of pydantic.BaseModel
      - all fields are SnowflakeField
    - Define a static method that invokes the desired query
      and wraps the result into the above dataclass, using
      the methods inherited from this class
    """
    @staticmethod
    def _check_result_schema(
        metadata: list[ResultMetadata],
        expected: dict[str, fields.FieldInfo],
    ) -> None:
        """Validate query result headers against dataclass fields"""
        if len(metadata) != len(expected):
            raise ValueError(f"expected {len(expected)} fields, "
                             f"got {len(metadata)} "
                             f"({metadata=} {expected=})")
        for res in metadata:
            if res.name not in expected:
                raise ValueError(f"unexpected field name {res.name} "
                                 f"({metadata=} {expected=})")

            # N.B. set by SnowflakeField. In v2 model_fields[name] IS the
            # FieldInfo and the type_code lives under json_schema_extra, which
            # pydantic types as JsonDict | callable | None; SnowflakeField only
            # ever sets the dict form, so narrow it back to int here.
            json_schema_extra = expected[res.name].json_schema_extra
            assert isinstance(json_schema_extra, dict)
            type_code = json_schema_extra['type_code']
            assert isinstance(type_code, int)

            if res.type_code != type_code:
                raise ValueError(f"expected field type "
                                 f"{FIELD_ID_TO_NAME[type_code]}, "
                                 f"got {FIELD_ID_TO_NAME[res.type_code]} "
                                 f"({metadata=} {expected=})")

    @staticmethod
    def _get_result(
        cur: SnowflakeCursor,
        dataclass: type[DataclassType],
    ) -> list[DataclassType]:
        # Use metadata from SnowflakeField to validate header schema
        SnowflakeQuery._check_result_schema(cur.description,
                                            dataclass.model_fields)

        # Convert each individual row to an instance of dataclass
        def _convert_one(description: list[ResultMetadata],
                         row) -> DataclassType:
            try:
                return dataclass(**dict([(m.name, v)
                                         for m, v in zip(description, row)]))
            except ValidationError:
                result_types = sorted([(m.name, type(v).__name__)
                                       for m, v in zip(description, row)])
                expected_types = sorted([
                    (n, f.annotation.__name__ if f.annotation else 'None')
                    for n, f in dataclass.model_fields.items()
                ])
                raise ValueError(f"datatype mismatch despite metadata match "
                                 f"{result_types=} {expected_types=}") from None

        return [_convert_one(cur.description, row) for row in cur.fetchall()]
