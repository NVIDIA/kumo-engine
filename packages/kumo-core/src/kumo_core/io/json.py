# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service-free JSON helpers for portable training artifacts."""

from __future__ import annotations

import json
import math
from typing import Any, TypeVar

from kumo_core.io.filesystem import exists, makedirs, open_file
from pydantic import TypeAdapter
from pydantic_core import to_jsonable_python

T = TypeVar("T")


def json_safe_float(value: Any) -> float | None:
    result = float(value)
    return None if math.isnan(result) or math.isinf(result) else result


def save_json(value: Any, path: str) -> None:
    payload = to_jsonable_python(value, serialize_unknown=True)
    parent = path.rsplit("/", maxsplit=1)[0] if "/" in path else ""
    if parent:
        makedirs(parent, exist_ok=True)
    with open_file(path, "wt", encoding="utf-8") as destination:
        json.dump(payload, destination, allow_nan=True, indent=2)


def load_json(
    path: str,
    cls: type[T],
    *,
    default: T | None = None,
    raise_on_missing: bool = True,
) -> T | None:
    if not exists(path):
        if raise_on_missing:
            raise FileNotFoundError(path)
        return default
    with open_file(path, "rt", encoding="utf-8") as source:
        return TypeAdapter(cls).validate_json(source.read())


def to_json_dict(value: Any) -> dict[str, Any]:
    payload = to_jsonable_python(value, serialize_unknown=True)
    if not isinstance(payload, dict):
        raise TypeError(f'expected object JSON, got {type(payload).__name__}')
    return payload


def from_json_dict(value: dict[str, Any], cls: type[T]) -> T:
    return TypeAdapter(cls).validate_python(value)
