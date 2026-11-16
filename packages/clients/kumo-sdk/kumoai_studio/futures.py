# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import atexit
import concurrent.futures
import logging
import threading
from abc import ABC, abstractmethod
from asyncio.events import AbstractEventLoop
from typing import Any, Awaitable, Coroutine, Generic, TypeVar

logger = logging.getLogger(__name__)

CoroFuncType = Awaitable[Any]

# Kumo global event loop (our implementation of green threads for pollers and
# other interactions with the Kumo backend that require long-running tasks).
# Since the caller may have their own event loop that we do not want to
# mess with, _do not_ ever call `set_event_loop` here!! Instead, be extra
# cautious to pass this loop everywhere.
_KUMO_EVENT_LOOP: AbstractEventLoop = asyncio.new_event_loop()
_KUMO_EVENT_LOOP_THREAD: threading.Thread | None = None
_KUMO_EVENT_LOOP_LOCK = threading.Lock()


def initialize_event_loop() -> None:
    global _KUMO_EVENT_LOOP_THREAD

    def _run_background_loop(loop: AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.call_soon(started.set)
        loop.run_forever()

    with _KUMO_EVENT_LOOP_LOCK:
        if (_KUMO_EVENT_LOOP_THREAD is not None
                and _KUMO_EVENT_LOOP_THREAD.is_alive()):
            return
        started = threading.Event()
        _KUMO_EVENT_LOOP_THREAD = threading.Thread(
            target=_run_background_loop,
            args=(_KUMO_EVENT_LOOP, ),
            daemon=True,
            name='kumo-event-loop',
        )
        _KUMO_EVENT_LOOP_THREAD.start()

    # Importing the SDK must not leave a thread racing interpreter shutdown.
    # Waiting until the loop has processed one callback also guarantees it is
    # ready before create_future() can submit work from another thread.
    if not started.wait(timeout=5):
        raise RuntimeError('Kumo event loop failed to start.')


def _shutdown_event_loop() -> None:
    """Stop the SDK event-loop thread during interpreter shutdown."""
    with _KUMO_EVENT_LOOP_LOCK:
        thread = _KUMO_EVENT_LOOP_THREAD
        if thread is None or not thread.is_alive():
            return
        try:
            _KUMO_EVENT_LOOP.call_soon_threadsafe(_KUMO_EVENT_LOOP.stop)
        except RuntimeError:
            return

    if thread is not threading.current_thread():
        thread.join(timeout=5)


atexit.register(_shutdown_event_loop)


def create_future(coro: Coroutine[Any, Any, Any]) -> concurrent.futures.Future:
    r"""Creates a future to execute in the Kumo event loop."""
    # NOTE this function creates a future, chains it to the output of the
    # execution of `coro` in the Kumo event loop, and handles exceptions
    # before scheduling to run in the loop:
    return asyncio.run_coroutine_threadsafe(coro, _KUMO_EVENT_LOOP)


T = TypeVar("T")


class KumoFuture(ABC, Generic[T]):
    r"""Abstract base class for a Kumo future object."""

    # We cannot use Python future implementations (`asyncio.Future` or
    # `concurrent.futures.Future`) as they are native to the Python
    # implementation of asyncio and threading, and thus not easily extensible.
    # Python additionally recommends not exposing low-level Future objects in
    # user facing APIs.
    @abstractmethod
    def result(self) -> T:
        r"""Returns the resolved state of the future.

        Raises:
            Exception:
                If the future is complete but in a failed state due to an
                exception being raised, this method will raise the same
                exception.
        """
        raise NotImplementedError

    @abstractmethod
    def future(self) -> 'concurrent.futures.Future[T]':
        r"""Returns the :obj:`concurrent.futures.Future` object wrapped by
        this future. It is not recommended to access this object directly.
        """
        raise NotImplementedError

    def done(self) -> bool:
        r"""Returns :obj:`True` if this future has been resolved with
        ``result()``, or :obj:`False` if this future is still
        in-progress. Note that this method will return :obj:`True` if the
        future is complete, but in a failed state, and that this method will
        return :obj:`False` if the job is complete, but the future has not
        been awaited.
        """
        return self.future().done()


class KumoProgressFuture(KumoFuture[T]):
    @abstractmethod
    def _attach_internal(self, interval_s: float = 4.0) -> T:
        raise NotImplementedError

    def attach(self, interval_s: float = 4.0) -> T:
        r"""Allows a user to attach to a running job and view its progress.

        Args:
            interval_s (float): Time interval (seconds) between polls, minimum
                value allowed is 4 seconds.
        """
        try:
            return self._attach_internal(interval_s=interval_s)
        except Exception:
            logger.warning(
                "Detailed job tracking has become temporarily unavailable. "
                "The job is continuing to proceed on the Kumo server, "
                "and this call will complete when the job has finished.")
            return self.result()
