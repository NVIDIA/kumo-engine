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


class PrefetchIterator(Iterator[T]):
    def __init__(self, produce: Callable[[], Iterator[T]], *, capacity: int, skip: int):
        self._ready: deque[T] = deque()
        self._condition = threading.Condition()
        self._stopped = False
        self._done = False
        self._error: BaseException | None = None

        def run():
            iterator = None
            try:
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
                        self._done = True
                        self._condition.notify_all()

        context = contextvars.copy_context()
        self._thread = threading.Thread(
            target=context.run, args=(run,), name="kumo-batch-prefetch", daemon=True
        )
        self._thread.start()

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

    def close(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        if threading.current_thread() is not self._thread:
            self._thread.join()
            # An interrupted join can mark a thread stopped before it exits
            # on some Python versions. Confirm producer cleanup independently.
            with self._condition:
                self._condition.wait_for(lambda: self._done)
        with self._condition:
            self._ready.clear()
            self._error = None
