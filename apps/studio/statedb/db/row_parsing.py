# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime


def row_field(row: object, name: str, index: int) -> object:
    if isinstance(row, dict):
        return row[name]
    try:
        return row[name]  # type: ignore[index]
    except (TypeError, KeyError, IndexError):
        values = tuple(row)  # type: ignore[arg-type]
        return values[index]


def row_str(row: object, name: str, index: int) -> str:
    return str(row_field(row, name, index))


def coerce_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError(f"Unsupported timestamp value: {value!r}")


def row_datetime(row: object, name: str, index: int) -> datetime:
    return coerce_datetime(row_field(row, name, index))
