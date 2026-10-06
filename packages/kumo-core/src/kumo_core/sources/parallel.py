# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from concurrent.futures import (
    FIRST_EXCEPTION,
    Future,
    ThreadPoolExecutor,
    wait,
)
from typing import Callable, Sequence, TypeVar

Result = TypeVar("Result")


class WorkCancelled(Exception):
    """Never reaches a caller: the failure that set the stop signal is the
    one raised, and a cancelled item's result is never read."""


def run_in_parallel(
    work: Sequence[Callable[[], Result]],
    *,
    workers: int,
    stop: threading.Event,
) -> list[Result]:
    """Results keep the order of ``work``. ``stop`` is how work already
    running notices; what cannot be interrupted is left to finish rather than
    holding the failure back. Callers that permit side effects must fence each
    invocation: a callback already running can outlive this function.
    """
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(_unless_stopped, item, stop) for item in work]
        done, _ = wait(futures, return_when=FIRST_EXCEPTION)
        failure = _first_failure(futures, done)
        if failure is None:
            return [future.result() for future in futures]
        stop.set()
        for future in futures:
            future.cancel()
        raise failure
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _first_failure(futures: Sequence[Future],
                   done: set[Future]) -> BaseException | None:
    """Taken in submission order, so which of several failures is reported
    does not depend on the order the threads finished in."""
    for future in futures:
        failure = future.exception() if future in done else None
        if failure is not None:
            return failure
    return None


def _unless_stopped(item: Callable[[], Result],
                    stop: threading.Event) -> Result:
    if stop.is_set():
        raise WorkCancelled("abandoned after another item failed")
    return item()
