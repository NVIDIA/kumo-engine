# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import threading
from asyncio import CancelledError, Task
from contextlib import contextmanager
from contextvars import copy_context
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Generator, Optional, Type, TypeVar, overload

import temporalio.bridge.runtime
from kumo_core.utils.time import now_utc
from temporalio import activity

from io_utils import logging

log = logging.getLogger(__name__)

DEFAULT_HEARTBEAT_INTERVAL = timedelta(seconds=10)


class _ThreadExceptionRaiser:
    """
    Internal helper for raising exception in another thread. This class is
    used in heartbeater background thread to inject exception upon detecting
    cancellation into the main thread that runs synchrnous activity func.
    """
    def __init__(self, thread_id: int) -> None:
        self._lock = threading.Lock()
        self.thread_id: int = thread_id

    def raise_in_thread(self, exc_type: Type[BaseException]) -> None:
        with self._lock:
            log.error(f'Raising {exc_type} in thread id={self.thread_id}')
            temporalio.bridge.runtime.Runtime._raise_in_thread(
                self.thread_id, exc_type)


def _report_heartbeat(
    desired_interval: timedelta,
    stop_event: threading.Event,
    thread_ex_raiser: Optional[_ThreadExceptionRaiser],
    context: activity._Context,
    hearbeat_msg: Optional[str] = None,
):
    activity._Context.set(context)  # pylint: disable=protected-access
    last_heartbeat_time = now_utc()
    heartbeat_start_time = now_utc()
    heartbeat_msg = hearbeat_msg or 'Auto heartbeat from background'
    while True:
        try:
            activity.heartbeat(heartbeat_msg)
        except Exception as err:  # pylint: disable=broad-except
            log.exception('Heartbeat failed: %s', err)

        if activity.is_cancelled():
            log.warning('Activity is canceled, stop heartbeat: %s',
                        activity.info())
            if thread_ex_raiser is not None:
                log.warning('Attempt to raise exeception in main thread: '
                            f'{thread_ex_raiser.thread_id}')
                thread_ex_raiser.raise_in_thread(CancelledError)
            break

        log.debug('Heartbeat from activity %s__%s, workflow_id=%s, after %s',
                  activity.info().activity_type,
                  activity.info().activity_id,
                  activity.info().workflow_id,
                  now_utc() - heartbeat_start_time)
        elapsed_since_last_heartbeat = now_utc() - last_heartbeat_time
        last_heartbeat_time = now_utc()
        if elapsed_since_last_heartbeat > 2.0 * desired_interval:
            log.warning(
                'Delayed heatbeat, desired interval %s, actual delay: %s',
                desired_interval, elapsed_since_last_heartbeat)

        if stop_event.wait(desired_interval.total_seconds()):
            log.debug('Activity execution has completed, stop heartbeat.')
            break


def _start_heartbeat_thread(
    interval: timedelta,
    stop_event: threading.Event,
    thread_ex_raiser: Optional[_ThreadExceptionRaiser],
    heartbeat_msg: Optional[str] = None,
) -> threading.Thread:
    """
    Intended for long-running synchronous (non-cancellable) activity function
    to spawn a separate background thread to periodically report heartbeats.

    Caller is responsible for shutting down the thread by setting the stop
    event.

    Note that this is not a recommended usage pattern. Ideally, the main
    execution logic should periodically report heartbeat and check for
    cancellation signal.
    """
    if not activity.in_activity():
        raise RuntimeError('Heartbeat thread must be started within Activity.')
    if activity.info().is_local:
        raise RuntimeError('Local Activity does not support heartbeat.')
    heartbeat_timeout = activity.info().heartbeat_timeout
    if heartbeat_timeout and interval >= heartbeat_timeout:
        raise RuntimeError(
            f'Heartbeat interval {interval} should be smaller than heartbeat '
            f'timeout: {heartbeat_timeout}')

    heartbeat_thread_name = threading.currentThread().getName() + '-heartbeat'
    heartbeat_thread = threading.Thread(
        name=heartbeat_thread_name,
        target=_report_heartbeat,
        args=[
            interval,
            stop_event,
            thread_ex_raiser,
            activity._Context.current(),  # pylint: disable=protected-access
            heartbeat_msg,
        ],
    )
    heartbeat_thread.start()
    return heartbeat_thread


# Convenient context manager to be used as:
#
# with heartbeat_in_background(timedelta(seconds=10)):
#   ...perform long-running activity...
@contextmanager
def _heartbeat_in_background(
    *,
    interval: timedelta = DEFAULT_HEARTBEAT_INTERVAL,
    message: Optional[str] = None,
    reraise_cancelled_exception: bool = True,
) -> Generator[threading.Thread, None, None]:
    thread_id = threading.current_thread().ident
    assert thread_id is not None

    thread_ex_raiser = None
    if reraise_cancelled_exception:
        thread_ex_raiser = _ThreadExceptionRaiser(thread_id)

    stop_event = threading.Event()
    thread = _start_heartbeat_thread(interval, stop_event, thread_ex_raiser,
                                     message)
    try:
        yield thread
    finally:
        stop_event.set()
        thread.join()


@contextmanager
def detach_from_activity():
    """Detach from current acitivty context, useful for spawning async
    background task that's meant to run beyond the activity execution.
    By default, asyncio.create_task() will copy the current context, as result,
    the spawned task will always consider itself being executed under the
    activity context even if the activity execution has completed.

    Example usage:
    @activity.defn
    async def run_activity():
        with detach_from_activity():
            task = asyncio.create_task(run_in_background())
        ...
    """
    ctx = copy_context()
    if activity.in_activity():
        activity._current_context.set(None)  # type: ignore
    yield ctx


_AUTO_HEARTBEATER_ATTR = '__auto_heartbeater_enabled'


def _default_heartbeat_interval():
    heartbeat_timeout = (activity.info().heartbeat_timeout
                         or DEFAULT_HEARTBEAT_INTERVAL * 3)
    return min(heartbeat_timeout / 3, DEFAULT_HEARTBEAT_INTERVAL)


ActivityFn = TypeVar("ActivityFn", bound=Callable[..., Any])


@overload
def auto_heartbeater(fn: ActivityFn) -> ActivityFn:
    ...


@overload
def auto_heartbeater(
    *, heartbeat_interval: Optional[timedelta] = None
) -> Callable[[ActivityFn], ActivityFn]:
    ...


def auto_heartbeater(
    fn: Optional[ActivityFn] = None,
    *,
    heartbeat_interval: Optional[timedelta] = None,
):
    """
    Decorator for long-running activity to send periodic heartbeats if
    it's executed within a Temporal activity.
    The decorator is no-op if called from non-Temporal activity.

    This decorator works on both sync and async activity functions.
    Additionally, inside ActivityInboundInterceptor, we'll apply this
    `auto_heartbeater` to ALL long-running activties, if they are not decorated
    by @auto_heartbeater already.

    It's OK but not required for developers to explicitly apply
    @auto_heartbeater to activity functions expected to be long-running.
    """
    def _auto_heartbeat_decorator(fn: ActivityFn) -> ActivityFn:
        @wraps(fn)
        def _sync_heartbeat_wrapper(*args, **kwargs):
            if not activity.in_activity():
                return fn(*args, **kwargs)
            # Nested under an async activity that's already heartbeating —
            # the outer async caller stamped its loop on this thread via
            # ``run_with_async_caller_loop`` before invoking us. Skip our
            # thread-based heartbeat: it's redundant (outer's
            # ``heartbeat_every`` asyncio task covers liveness) AND broken
            # for async ``ctx.heartbeat`` (calling it from the spawned
            # thread ends at ``asyncio.create_task`` with "no running
            # event loop"). Same root cause as the polling-loop heartbeat
            # in ``_run_in_diskgraph_subprocess``; reuse the same signal.
            from temporalio_commons.process_pool_utils import (  # avoid cycle
                _get_async_caller_loop,
            )
            if _get_async_caller_loop() is not None:
                return fn(*args, **kwargs)
            with _heartbeat_in_background(interval=heartbeat_interval
                                          or _default_heartbeat_interval()):
                return fn(*args, **kwargs)

        # We want to ensure that the type hints from the original callable are
        # available via our wrapper, so we use the functools wraps decorator
        @wraps(fn)
        async def _async_heartbeat_wrapper(*args, **kwargs):
            if not activity.in_activity():
                return await fn(*args, **kwargs)

            # Heartbeat twice as often as the timeout
            main_task = asyncio.ensure_future(fn(*args, **kwargs))
            heartbeat_task = asyncio.create_task(
                heartbeat_every(
                    main_task,
                    (heartbeat_interval
                     or _default_heartbeat_interval()).total_seconds()))
            try:
                return await main_task
            finally:
                log.debug('Stopping auto heartbeater...')
                heartbeat_task.cancel()
                # Wait for heartbeat cancellation to complete
                await asyncio.wait([heartbeat_task])
                log.debug('Auto heartbeater has stopped.')

        if getattr(fn, _AUTO_HEARTBEATER_ATTR, False):
            # avoid multiple wraps of auto heartbeater
            log.debug('Auto heartbeater is already enabled for %s',
                      fn.__qualname__)
            return fn

        if inspect.iscoroutinefunction(fn):
            log.debug('Apply async auto_heartbeater to %s', fn.__qualname__)
            setattr(_async_heartbeat_wrapper, _AUTO_HEARTBEATER_ATTR, True)
            return _async_heartbeat_wrapper  # type: ignore
        else:
            log.debug('Apply sync auto_heartbeater to %s', fn.__qualname__)
            setattr(_sync_heartbeat_wrapper, _AUTO_HEARTBEATER_ATTR, True)
            return _sync_heartbeat_wrapper  # type: ignore

    if fn is not None:
        return _auto_heartbeat_decorator(fn)
    return _auto_heartbeat_decorator


async def heartbeat_every(main_task: Task, interval_sec: float) -> None:
    # Heartbeat every so often while not cancelled. Upon receiving cancellation
    # signal, cancel the main task.
    info = activity.info()
    activity_name = f'{info.workflow_type}.{info.activity_type}'
    activity_name += f'__{info.activity_id}'
    log.debug(f'Start auto-heartbeater asyncio task for {activity_name} '
              f'every {interval_sec:.2f} seconds.')
    while True:
        try:
            await asyncio.sleep(interval_sec)
        except CancelledError:
            main_task.cancel()
            status = 'canceled' if main_task.cancelled() else 'completed'
            log.debug('Auto heartbeater is stopping after %s is %s',
                      activity_name, status)
            raise

        log.debug("Heartbeating from async activity %s at %s", activity_name,
                  datetime.now())
        activity.heartbeat()
        if activity.is_cancelled():
            log.warning(
                'Auto heartbeater saw cancellation, aborting activity %s!',
                activity_name)
            main_task.cancel(msg='Received Activity Cancellation')
            raise CancelledError()
