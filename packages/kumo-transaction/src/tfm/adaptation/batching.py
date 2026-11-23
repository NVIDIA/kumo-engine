# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Row slicing that works for every batch layout.

`TransactionBatch` and `KeyValueTimeBatch` are both dataclasses whose tensors
and id tuples share a leading row dimension. Slicing them field by field keeps
adaptation code independent of the layout, so one probe and one fine-tuning
loop serve every stack.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, TypeGuard

import torch
from torch import Tensor


def _rows(batch: Any) -> int:
    size = getattr(batch, "batch_size", None)
    if size is None:
        raise TypeError(f"{type(batch).__name__} has no batch_size.")
    return int(size)


def select_rows[BatchT](batch: BatchT, rows: Sequence[int]) -> BatchT:
    """The given rows of `batch`, in order, with every field kept aligned.

    Tensor fields and tuple fields whose length equals the batch size are
    indexed, and so is each tensor of a row-aligned dict field (per-event arrays); everything
    else (for example, an empty id tuple) is kept as is.
    """
    if not dataclasses.is_dataclass(batch) or isinstance(batch, type):
        raise TypeError(f"select_rows needs a dataclass batch, got {type(batch).__name__}.")
    n = _rows(batch)
    picked = list(rows)
    updates: dict[str, Any] = {}
    for field in dataclasses.fields(batch):
        value = getattr(batch, field.name)
        if isinstance(value, Tensor) and value.dim() > 0 and value.shape[0] == n:
            index = torch.as_tensor(picked, dtype=torch.long, device=value.device)
            updates[field.name] = value.index_select(0, index)
        elif isinstance(value, tuple) and len(value) == n:
            updates[field.name] = tuple(value[i] for i in picked)
        elif _array_rows(value, n):
            updates[field.name] = {
                name: array.index_select(
                    0, torch.as_tensor(picked, dtype=torch.long, device=array.device)
                )
                for name, array in value.items()
            }
    return dataclasses.replace(batch, **updates)


def concat_rows[BatchT](batches: Sequence[BatchT]) -> BatchT:
    """The rows of `batches`, in order, as one batch: `select_rows` inverted.

    Row-aligned tensors are concatenated and row-aligned tuples joined; every
    other field must be equal across `batches` and is kept once.
    """
    if not batches:
        raise ValueError("concat_rows needs at least one batch.")
    first = batches[0]
    if not dataclasses.is_dataclass(first) or isinstance(first, type):
        raise TypeError(f"concat_rows needs dataclass batches, got {type(first).__name__}.")
    sizes = [_rows(batch) for batch in batches]
    updates: dict[str, Any] = {}
    for field in dataclasses.fields(first):
        values = [getattr(batch, field.name) for batch in batches]
        arrays = [_array_rows(value, n) for value, n in zip(values, sizes, strict=True)]
        if any(arrays):
            if not all(arrays) or any(value.keys() != values[0].keys() for value in values):
                raise ValueError(f"concat_rows: field {field.name!r} differs across batches.")
            updates[field.name] = {
                name: torch.cat([value[name] for value in values]) for name in values[0]
            }
            continue
        aligned = [
            (isinstance(v, Tensor) and v.dim() > 0 and v.shape[0] == n)
            or (isinstance(v, tuple) and len(v) == n)
            for v, n in zip(values, sizes, strict=True)
        ]
        if all(aligned):
            if isinstance(values[0], Tensor):
                updates[field.name] = torch.cat(values)
            else:
                updates[field.name] = tuple(item for value in values for item in value)
        elif any(aligned) or any(
            value is not values[0] and not _same(value, values[0]) for value in values
        ):
            raise ValueError(f"concat_rows: field {field.name!r} differs across batches.")
    return dataclasses.replace(first, **updates)


def _array_rows(value: Any, rows: int) -> TypeGuard[dict[str, Tensor]]:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(
            isinstance(array, Tensor) and array.ndim > 0 and array.shape[0] == rows
            for array in value.values()
        )
    )


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        return left.shape == right.shape and bool(torch.equal(left, right))
    return bool(left == right)


def minibatches[BatchT](batch: BatchT, batch_size: int) -> list[BatchT]:
    """Consecutive row chunks of at most `batch_size`."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
    n = _rows(batch)
    return [
        select_rows(batch, range(start, min(start + batch_size, n)))
        for start in range(0, n, batch_size)
    ]


def shuffled[BatchT](batch: BatchT, seed: int) -> BatchT:
    """`batch` with its rows in a seeded random order."""
    generator = torch.Generator().manual_seed(seed)
    return select_rows(batch, torch.randperm(_rows(batch), generator=generator).tolist())
