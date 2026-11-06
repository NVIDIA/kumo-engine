# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode explicitly tagged fixture scalars without guessing user data types."""

from __future__ import annotations

import base64
from datetime import date, datetime
from decimal import Decimal
from typing import Any


def scalar(value: Any) -> Any:
    """JSON tags retain exact values that ordinary JSON cannot represent."""
    if not isinstance(value, dict):
        return value
    if len(value) != 1:
        raise ValueError("fixture scalar must use exactly one recognized type tag")
    tag, content = next(iter(value.items()))
    if tag == "$bytes" and isinstance(content, str):
        return base64.b64decode(content, validate=True)
    if tag == "$decimal" and isinstance(content, str):
        return Decimal(content)
    if tag == "$date" and isinstance(content, str):
        return date.fromisoformat(content)
    if tag == "$timestamp" and isinstance(content, str):
        return datetime.fromisoformat(content)
    if tag == "$float" and isinstance(content, str):
        if content in ("nan", "inf", "-inf"):
            return float(content)
        if content.startswith(("0x", "-0x")):
            return float.fromhex(content)
    raise ValueError(f"unknown or invalid fixture scalar tag {tag!r}")


def values(value: Any) -> Any:
    """Decode scalar tags recursively in expected arrays or structured records."""
    if isinstance(value, dict):
        if any(key.startswith("$") for key in value):
            return scalar(value)
        return {key: values(child) for key, child in value.items()}
    if isinstance(value, list):
        return [values(child) for child in value]
    return value
