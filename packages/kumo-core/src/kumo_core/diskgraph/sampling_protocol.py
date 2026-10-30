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


def encode_batch(batch: dict[str, Any], batch_id: int) -> pa.RecordBatch:
    """Serialize a complete native task batch without converting Arrow columns.

    Args:
        batch: Native task, topology and feature dictionary containing Arrow data.
        batch_id: Zero-based sequence number within the requested iterator.
    Returns:
        One protocol frame containing the complete logical batch.
    Raises:
        ValueError: If the IPC payload exceeds the protocol's byte limit.
        TypeError: If a non-Arrow value cannot be represented in metadata.
    """
    components: list[bytes] = []
    size = 0

    def encode(value: Any) -> Any:
        nonlocal size
        if isinstance(value, (pa.RecordBatch, pa.Array)):
            is_array = isinstance(value, pa.Array)
            record = (
                pa.RecordBatch.from_arrays([value], names=["value"])
                if is_array
                else value
            )
            with pa.BufferOutputStream() as sink:
                with pa.ipc.new_stream(sink, record.schema) as writer:
                    writer.write_batch(record)
                payload = sink.getvalue()
            size += payload.size
            if size > MAX_BATCH_BYTES:
                raise ValueError("Task batch exceeds 64 MiB")
            slot = len(components)
            components.append(payload.to_pybytes())
            return {"__arrow__": slot, "array": is_array}
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [encode(item) for item in value]
        return value

    value = encode(batch)
    return encode_frame(
        {"kind": "batch", "batch_id": batch_id, "value": value}, components
    )
