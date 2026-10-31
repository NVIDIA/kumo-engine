# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Arrow sampling frames shared by Python Flight endpoints."""

from __future__ import annotations

from typing import Any

import orjson
import pyarrow as pa

PUSH_VERSION = 4
PUSH_WINDOW = 2
TASK_CHUNK_BYTES = 1024**2
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


def encode_frame(metadata: dict[str, Any], components: list[bytes]) -> pa.RecordBatch:
    """Build a protocol envelope containing metadata and Arrow IPC components.

    Args:
        metadata: Versioned frame metadata, including component references.
        components: Binary payloads; batch frames use one Arrow IPC stream per
            component and graph-filter registration uses a packed visibility mask.
    Returns:
        A one-row envelope using the native gateway's schema.
    Raises:
        ValueError: If the encoded frame exceeds the protocol's byte limit.
        TypeError: If metadata cannot be encoded as JSON.
    """
    payload = orjson.dumps(metadata)
    if len(payload) + sum(map(len, components)) > MAX_BATCH_BYTES:
        raise ValueError("Task batch exceeds 64 MiB; reduce batch size or fanout")
    return pa.RecordBatch.from_arrays(
        [
            pa.array([payload], pa.binary()),
            pa.array([components], pa.list_(pa.binary())),
        ],
        schema=ENVELOPE_SCHEMA,
    )
