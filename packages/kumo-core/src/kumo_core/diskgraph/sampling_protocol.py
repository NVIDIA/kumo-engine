# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode native Arrow sampling frames in Python Flight clients."""

from __future__ import annotations

from typing import Any

import orjson
import pyarrow as pa

SAMPLING_VERSION = 5
MAX_BATCH_BYTES = 64 * 1024**2
ENVELOPE_SCHEMA = pa.schema(
    [
        pa.field("metadata", pa.binary(), nullable=False),
        pa.field("components", pa.list_(pa.binary()), nullable=False),
    ]
)


def decode_frame(batch: pa.RecordBatch | None) -> tuple[dict[str, Any], list[bytes]]:
    """Validate and unpack a sampling envelope without decoding Arrow components.

    Args:
        batch: One-row Flight envelope.
    Returns:
        JSON metadata and opaque Arrow IPC components.
    Raises:
        ValueError: If the envelope or its size is invalid.
        TypeError: If metadata is not a JSON object.
    """
    if batch is None or batch.num_rows != 1 or not batch.schema.equals(ENVELOPE_SCHEMA):
        raise ValueError("Invalid task stream envelope")
    payload = batch.column(0)[0].as_py()
    components = batch.column(1)[0].as_py()
    if (
        not isinstance(payload, bytes)
        or not isinstance(components, list)
        or not all(isinstance(part, bytes) for part in components)
    ):
        raise ValueError("Invalid task frame components")
    if len(payload) + sum(map(len, components)) > MAX_BATCH_BYTES:
        raise ValueError("Sampling frame exceeds 64 MiB")
    metadata = orjson.loads(payload)
    if not isinstance(metadata, dict):
        raise TypeError("Invalid task frame metadata")
    return metadata, components
