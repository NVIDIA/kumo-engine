# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded CPU batch production; the producer owns the underlying iterator."""

from __future__ import annotations

import contextvars
import threading
from collections import deque
from collections.abc import Callable, Iterator
from typing import TypeVar

T = TypeVar("T")


class PrefetchGroup:
    """The producers that must finish before their task releases its resources."""

    def __init__(self):
        self.closed = False
        self._active: set[PrefetchIterator] = set()
        self._lock = threading.Lock()

    def register(self, producer: PrefetchIterator) -> bool:
        with self._lock:
            if self.closed:
                return False
            self._active.add(producer)
            return True

    def discard(self, producer: PrefetchIterator) -> None:
        with self._lock:
            self._active.discard(producer)

    def close(self) -> None:
        with self._lock:
            self.closed = True
            active = self._active.copy()
        for producer in active:
            producer.close()


class PrefetchIterator(Iterator[T]):
    def __init__(
        self,
        produce: Callable[[], Iterator[T]],
        *,
        capacity: int,
        skip: int,
        group: PrefetchGroup | None = None,
    ):
        self._group = group
        self._ready: deque[T] = deque()
        self._condition = threading.Condition()
        self._stopped = False
        self._done = True
        self._error: BaseException | None = None

        def run():
            iterator = None
            try:
                with self._condition:
                    if self._stopped:
                        return
                iterator = produce()
                for _ in range(skip):
                    with self._condition:
                        if self._stopped:
                            return
                    next(iterator)
                while True:
                    with self._condition:
                        self._condition.wait_for(
                            lambda: self._stopped or len(self._ready) < capacity
                        )
                        if self._stopped:
                            return
                    # Reserve space before producing, bounding queued plus
                    # in-flight batches, not just the completed queue.
                    try:
                        value = next(iterator)
                    except StopIteration:
                        return
                    with self._condition:
                        if self._stopped:
                            return
                        self._ready.append(value)
                        del value
                        self._condition.notify_all()
            except BaseException as error:
                self._error = error
            finally:
                try:
                    if iterator is not None and callable(close := getattr(iterator, "close", None)):
                        close()
                except BaseException as error:
                    if self._error is None:
                        self._error = error
                finally:
                    with self._condition:
                        if self._group is not None:
                            self._group.discard(self)
                        self._done = True
                        self._condition.notify_all()

        context = contextvars.copy_context()
        self._thread = threading.Thread(
            target=context.run, args=(run,), name="kumo-batch-prefetch", daemon=True
        )

    def start(self) -> None:
        # The worker must acquire this condition before opening the source.
        # An interrupted start can therefore revoke access even after launch.
        with self._condition:
            try:
                if self._stopped or (self._group is not None and not self._group.register(self)):
                    self._stopped = True
                    return
                self._done = False
                self._thread.start()
            except BaseException:
                self._stopped = self._done = True
                if self._group is not None:
                    self._group.discard(self)
                self._condition.notify_all()
                raise

    def __next__(self) -> T:
        with self._condition:
            self._condition.wait_for(lambda: self._ready or self._done or self._stopped)
            if not self._stopped and self._ready:
                value = self._ready.popleft()
                self._condition.notify_all()
                return value
            error = None if self._stopped else self._error
        self.close()
        if error is not None:
            raise error
        raise StopIteration

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()

    def close(self) -> None:
        self.stop()
        with self._condition:
            if threading.current_thread() is not self._thread:
                self._condition.wait_for(lambda: self._done)
            self._ready.clear()
            self._error = None
