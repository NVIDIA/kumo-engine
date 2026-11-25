# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-group helpers shared by the data-parallel backends (backend contracts).

`torchrun` sets the rendezvous environment but does not create the process
group. Without `join_distributed_group`, every rank reports `world_size` 1,
shards nothing, and trains on identical data.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import overload

import torch

from tfm.contracts.errors import UnsupportedCombinationError


def distributed_info() -> tuple[int, int]:
    """`(world_size, rank)` from `torch.distributed`, or `(1, 0)` single-process."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(), torch.distributed.get_rank()
    return 1, 0


def launcher_topology() -> tuple[int, int]:
    """`(world_size, local_rank)` from the launcher environment, without joining."""
    return int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("LOCAL_RANK", "0"))


def join_distributed_group(device_type: str = "cuda") -> tuple[int, int, int]:
    """Join the launcher's process group. Returns `(world_size, rank, local_rank)`.

    CUDA ranks use NCCL and are pinned to their local GPU; CPU ranks use Gloo.
    Single-process runs are left alone.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size, rank = distributed_info()
        return world_size, rank, int(os.environ.get("LOCAL_RANK", "0"))

    world_size, local_rank = launcher_topology()
    if world_size <= 1:
        return 1, 0, 0
    if not torch.distributed.is_available():
        raise UnsupportedCombinationError(
            f"WORLD_SIZE={world_size} but torch.distributed is unavailable. "
            "Run single-process or install a distributed-capable torch."
        )
    if device_type == "cuda":
        if local_rank >= torch.cuda.device_count():
            raise UnsupportedCombinationError(
                f"LOCAL_RANK={local_rank} exceeds the {torch.cuda.device_count()} "
                "visible CUDA devices."
            )
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
    else:
        torch.distributed.init_process_group(backend="gloo")
    return torch.distributed.get_world_size(), torch.distributed.get_rank(), local_rank


class _Strided[BatchT](Sequence[BatchT]):
    """`batches[i]` for `i` in `index`, read when indexed, so lazy batches stay lazy."""

    def __init__(self, batches: Sequence[BatchT], index: range) -> None:
        self._batches, self._index = batches, index

    def __len__(self) -> int:
        return len(self._index)

    @overload
    def __getitem__(self, i: int) -> BatchT: ...
    @overload
    def __getitem__(self, i: slice) -> Sequence[BatchT]: ...
    def __getitem__(self, i: int | slice) -> BatchT | Sequence[BatchT]:
        if isinstance(i, slice):
            return _Strided(self._batches, self._index[i])
        return self._batches[self._index[i]]


def shard_batches[BatchT](
    batches: Sequence[BatchT], *, world_size: int, rank: int
) -> Sequence[BatchT]:
    """This rank's share, with every rank getting an equal number of batches.

    Ranks must take the same number of optimizer steps: collectives run inside
    backward, so a rank that finishes early leaves the others waiting on a
    reduction that never comes. The remainder is dropped rather than padded, so
    no batch is ever seen twice in one pass.
    """
    if world_size <= 1:
        return batches
    usable = (len(batches) // world_size) * world_size
    if usable == 0:
        raise UnsupportedCombinationError(
            f"{len(batches)} batches cannot be split across {world_size} ranks. "
            "Lower training.batch_size or the world size."
        )
    return _Strided(batches, range(rank, usable, world_size))


def mean_across_ranks(value: float, *, world_size: int, device: torch.device) -> float:
    """Average a scalar over ranks so the reported loss is not one shard's view."""
    if world_size <= 1:
        return value
    tensor = torch.tensor([value], dtype=torch.float32, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return float(tensor.item() / world_size)


def leave_distributed_group(world_size: int) -> None:
    """Hold every rank until rank 0 has written, then leave the group."""
    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
