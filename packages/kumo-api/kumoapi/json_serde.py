# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
from collections import deque
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from typing import Any, Dict, Type, TypeVar, Union, cast
from uuid import UUID

from pydantic import BaseModel, SecretStr
from pydantic_core import to_jsonable_python as lib_encoder

# Immutable types that are safe to return as-is (no copying needed)
_IMMUTABLE_TYPES = (
    type(None),
    bool,
    int,
    float,
    str,
    bytes,
    datetime,
    date,
    time,
    timedelta,
    Decimal,
    UUID,
    Enum,
    PurePath,
    SecretStr,
    # Note: frozenset is immutable but may contain items needing conversion
)


def _convert_value(value: Any) -> Any:
    """Recursively convert a value, creating independent copies of containers.

    Handles:
        - Dataclasses: recursively converted to dicts
        - Pydantic BaseModel: converted via model_dump()/dict()
        - Dicts (including subclasses): keys and values recursively converted
        - NamedTuples: recreated with recursively converted values
        - Lists/Tuples (including subclasses): elements recursively converted
        - Immutable primitives: returned as-is (str, int, float, bool, None,
          bytes, datetime, date, time, timedelta, Decimal, UUID, Enum, Path)

    Raises:
        TypeError: If an unhandled type is encountered.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclass_to_dict(value)
    elif isinstance(value, dict):
        return type(value)(
            (_convert_value(k), _convert_value(v)) for k, v in value.items())
    elif isinstance(value, tuple) and hasattr(value, '_fields'):
        return type(value)(*[_convert_value(v) for v in value])
    elif isinstance(value, (list, tuple)):
        return type(value)(_convert_value(v) for v in value)
    elif isinstance(value, (set, frozenset)):
        return [_convert_value(v) for v in value]
    elif isinstance(value, deque):
        return [_convert_value(v) for v in value]
    elif isinstance(value, BaseModel):
        return _convert_value(value.model_dump())
    elif isinstance(value, _IMMUTABLE_TYPES):
        return value
    elif isinstance(value, bytearray):
        raise TypeError(
            f"dataclass_to_dict does not support {type(value).__name__}. "
            f"Convert to a supported type before serialization.")
    else:
        raise TypeError(
            f"dataclass_to_dict encountered unexpected type "
            f"{type(value).__name__}. Add it to _IMMUTABLE_TYPES if "
            f"immutable, or handle it explicitly.")


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    r"""Convert a dataclass to a dictionary.

    Defensive alternative to asdict() that works in distributed contexts
    (e.g., Databricks Serverless) where asdict() fails with nested Pydantic.

    Unlike asdict(), this function does NOT use deepcopy, but it does
    recursively process nested dicts, lists, and dataclasses to avoid
    sharing references with the original object.
    """
    result = {}
    for field in dataclasses.fields(obj):
        value = getattr(obj, field.name)
        result[field.name] = _convert_value(value)

    # Handle pydantic models with extra='allow' that store additional
    # fields in __pydantic_extra__ (e.g., HopConfig with edge overrides)
    if hasattr(obj, '__pydantic_extra__'):
        extra = obj.__pydantic_extra__
        if extra:
            for key, value in extra.items():
                result[key] = _convert_value(value)

    return result


T = TypeVar('T')


def trusted_encoder(obj: Any) -> Any:
    if isinstance(obj, SecretStr):
        return obj.get_secret_value()

    return lib_encoder(obj, context={'insecure': True})


def to_json(pydantic_obj: Any, insecure: bool = False) -> str:
    r"""Encodes a pydantic object into JSON.

    The `insecure` flag should only be used by trusted internal code where the
    output of the JSON is not accessible to any users and `SecretStr`s are
    hidden in some other fashion."""
    encoder = trusted_encoder if insecure else lib_encoder

    return json.dumps(
        pydantic_obj,
        default=encoder,
        allow_nan=True,
        indent=2,
    )


def to_json_dict(pydantic_obj: Any, insecure: bool = False) -> Dict[str, Any]:
    return json.loads(to_json(pydantic_obj, insecure=insecure))


def from_json(obj: Any, cls: Union[Type[T], object]) -> T:
    if isinstance(obj, str):
        obj = json.loads(obj)
    from pydantic import TypeAdapter
    adapter: TypeAdapter[T] = TypeAdapter(cast(Any, cls))
    return adapter.validate_python(obj)
