# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Continuous native iteration with one owned CPU lookahead batch.

The adapter consumes delivered tensors, not source-event IDs. The task remains
owned by its caller: close batches before closing the task or its index.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Protocol

import torch

from tfm.contracts.training import TrainableBatch


class NativeTask(Protocol):
    """The small native-task capability consumed without importing its backend."""

    def count(self, split: str | tuple[str, ...]) -> int: ...

    def batches(self, **kwargs: Any) -> Iterator[Any]: ...


class NativeBatches[BatchT: TrainableBatch]:
    """One native iterator per stage and epoch; native seed/epoch owns anchor ordering.

    ``adapter`` must return owned, validated CPU tensors. At most one adapted
    batch is prefetched in addition to the batch held by the consumer. Closing
    waits for the producer and closes its iterator before the caller may close
    the task. Concurrent consumers of one source are rejected.

    Example::

        with NativeBatches(task, tensor_adapter, batch_size=64, seed=42) as batches:
            trainer.fit(model, batches, shuffle=False, ...)

    A rank view uses the native task's rank/world-size selection, without
    materializing or shuffling a Python list of minibatches.
    """

    owns_batch_order = True

    def __init__(
        self,
        task: NativeTask,
        adapter: Callable[[Any], BatchT],
        *,
        split: str | tuple[str, ...] = "train",
        batch_size: int,
        seed: int,
        shuffle: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if batch_size < 1 or world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Native batches require positive sizes and a valid rank")
        self.task, self.adapter = task, adapter
        self.split, self.batch_size = split, batch_size
        self.seed, self.shuffle = seed, shuffle
        self.rank, self.world_size = rank, world_size
        count = task.count(split)
        self.owned_samples = max(0, (count - rank + world_size - 1) // world_size)
        # Native distributed iteration pads every rank to the same length.
        self.samples = (count + world_size - 1) // world_size
        self.epoch = 0
        self._active: Generator[BatchT, None, None] | None = None
        self._closed = False

    def __len__(self) -> int:
        return (self.samples + self.batch_size - 1) // self.batch_size

    def __enter__(self) -> NativeBatches[BatchT]:
        if self._closed:
            raise ValueError("Native batch source is closed")
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def for_rank(self, rank: int, world_size: int) -> NativeBatches[BatchT]:
        """An independently owned producer using the same fitted native task."""
        if self._closed or self._active is not None:
            raise ValueError("Create native rank views before iteration")
        return NativeBatches(
            self.task,
            self.adapter,
            split=self.split,
            batch_size=self.batch_size,
            seed=self.seed,
            shuffle=self.shuffle,
            rank=rank,
            world_size=world_size,
        )

    def __iter__(self) -> Iterator[BatchT]:
        if self._closed:
            raise ValueError("Native batch source is closed")
        if self._active is not None:
            raise ValueError("Native batch source already has an active iterator")
        epoch = self.epoch if self.shuffle else 0
        self.epoch += 1
        self._active = self._iterate(epoch)
        return self._active

    def _iterate(self, epoch: int) -> Generator[BatchT, None, None]:
        source: Iterator[Any] | None = None
        worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tfm-native-batch")
        pending: Future[BatchT | None] | None = None

        @torch.inference_mode(False)
        @torch.no_grad()
        def fetch() -> BatchT | None:
            nonlocal source
            if source is None:

                def stages() -> Iterator[Any]:
                    # Shared-graph stages have distinct cutoffs. Keep each
                    # stage's native iterator continuous without mixing a
                    # partial tail with a different temporal context.
                    splits = (self.split,) if isinstance(self.split, str) else self.split
                    for split in splits:
                        iterator = iter(
                            self.task.batches(
                                split=split,
                                batch_size=self.batch_size,
                                seed=self.seed,
                                epoch=epoch,
                                shuffle=self.shuffle,
                                drop_last=False,
                                rank=self.rank,
                                world_size=self.world_size,
                                output_format="model",
                            )
                        )
                        try:
                            yield from iterator
                        finally:
                            close = getattr(iterator, "close", None)
                            if callable(close):
                                close()

                source = stages()
            value = next(source, None)
            return None if value is None else self.adapter(value)

        def close_source() -> None:
            close = getattr(source, "close", None)
            if callable(close):
                close()

        emitted = False
        try:
            pending = worker.submit(fetch)
            while (batch := pending.result()) is not None:
                emitted = True
                pending = worker.submit(fetch)
                yield batch
            if self.samples and not emitted:
                raise ValueError("Native task returned an empty epoch for a nonempty split")
        finally:
            if pending is not None:
                pending.cancel()
            try:
                # Serialize closure after any in-flight read on the producer.
                worker.submit(close_source).result()
            finally:
                worker.shutdown(wait=True, cancel_futures=True)
                self._active = None

    def close(self) -> None:
        """Stop lookahead and release its tensors before task/index teardown."""
        self._closed = True
        active, self._active = self._active, None
        if active is not None:
            active.close()
