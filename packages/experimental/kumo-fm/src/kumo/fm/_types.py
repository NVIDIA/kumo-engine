# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scalar Arrow types and occurrence validation for FM payloads."""
import math
from datetime import date, datetime, time, timezone

import pyarrow as pa

TYPES = {"int64": pa.int64(), "float64": pa.float64(), "bool": pa.bool_(),
         "string": pa.string(), "date": pa.date32(), "timestamp[us]": pa.timestamp("us", tz="UTC")}

UTC = timezone.utc
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def timestamp(value) -> datetime:
    if isinstance(value, date) and not isinstance(value, datetime):
        value = datetime.combine(value, time(), tzinfo=UTC)
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("Timestamps must include a UTC offset")
    return value.astimezone(UTC)


def validate_scalar(value, dtype: str, *, nullable: bool = False):
    if value is None:
        if nullable:
            return
        raise ValueError("Unexpected null value")
    valid = ((dtype == "int64" and type(value) is int and -(2**63) <= value < 2**63) or
             (dtype == "string" and isinstance(value, str)) or
             (dtype == "bool" and type(value) is bool) or
             (dtype == "float64" and type(value) in (int, float) and math.isfinite(value)))
    if dtype == "timestamp[us]":
        timestamp(value)
        return
    if not valid:
        raise ValueError(f"Value does not match {dtype}: {value!r}")

