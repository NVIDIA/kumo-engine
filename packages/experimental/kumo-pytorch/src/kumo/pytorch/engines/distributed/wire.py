# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What crosses the feeder connection (P0: trusted network, pickled data, one blob per batch)."""

from __future__ import annotations

import io
import pickle
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import pyarrow as pa
import torch

from kumo.pytorch.contract.batch import Batch

BATCH = 0
ERROR = 1
STREAM_SCHEMA = pa.schema([("kind", pa.int8()), ("payload", pa.large_binary())])
LISTENING = "KUMO_FEEDER_LISTENING"

JSONValue = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True)
class EngineSpec:
    """An engine as data: its registry name and constructor keywords."""

    name: str
    config: Mapping[str, JSONValue]


@dataclass(frozen=True)
class Reply:
    """One control-action answer: a value, or the feeder's exception and traceback."""

    status: Literal["ok", "error"]
    value: object = None
    error: BaseException | None = None
    traceback: str = ""


def dumps(value: object) -> bytes:
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def loads(data: bytes) -> object:
    return pickle.loads(data)


def error_reply(error: BaseException) -> bytes:
    text = "".join(traceback.format_exception(error))
    try:
        portable: BaseException = pickle.loads(dumps(error))
    except Exception:
        portable = RuntimeError(f"{type(error).__name__}: {error}")
    return dumps(Reply("error", error=portable, traceback=text))


def raise_remote(reply: Reply) -> None:
    error = reply.error if reply.error is not None else RuntimeError("feeder failed")
    error.add_note(f"raised on the feeder:\n{reply.traceback}")
    raise error


def encode_batch(batch: Batch) -> bytes:
    buffer = io.BytesIO()
    torch.save(batch, buffer)
    return buffer.getvalue()


def decode_batch(data: bytes) -> Batch:
    value = torch.load(io.BytesIO(data), weights_only=False)
    if not isinstance(value, Batch):
        raise TypeError(f"feeder sent {type(value).__name__}, not a Batch")
    return value


def record(kind: int, payload: bytes) -> pa.RecordBatch:
    return pa.record_batch(
        [pa.array([kind], pa.int8()), pa.array([payload], pa.large_binary())],
        schema=STREAM_SCHEMA,
    )
