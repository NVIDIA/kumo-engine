# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextvars
import functools
import inspect
import logging
import time
from typing import (
    Any,
    Awaitable,
    Callable,
    Generator,
    Generic,
    Optional,
    TypeVar,
)

from pyinstrument.stack_sampler import active_profiler_context_var
from typing_extensions import ParamSpec

from service_framework.deploy_stage import _DEPLOY_STAGE, DeployStage
from service_framework.env_var import EnvVar

log = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

# Tracks whether we are already inside a @measure_blocking_code wrapper on the
# current asyncio task. Used so a decorated child running inside a decorated
# parent has its blocking time attributed to the child only (not double-counted
# against the parent). See `measure_blocking_code` for details.
_IN_MEASURE_BLOCKING: 'contextvars.ContextVar[bool]' = contextvars.ContextVar(
    'in_measure_blocking', default=False)

_STRESS_TEST_MODE = EnvVar('STRESS_TEST_MODE', bool, default_value=False)
BLOCKING_WARN_THRESHOLD_SEC = 0.2
BLOCKING_ALERT_THRESHOLD_SEC = 5


def _stress_test_enabled() -> bool:
    deploy_stage = _DEPLOY_STAGE.parse_value()
    return (deploy_stage != DeployStage.PROD
            and _STRESS_TEST_MODE.parse_value())


# TODO(siyang): move this to a common util library
def _notify_kumo_if_prod(log_msg: str) -> None:
    """If running in prod, notify a Kumo engineer that we're blocking"""
    deploy_stage = _DEPLOY_STAGE.parse_value()
    if deploy_stage == DeployStage.PROD:
        from service_framework.deploy_env import _SLACK_URL, send_slack_message

        slack_url = _SLACK_URL.parse_value()
        msg = f"Critical error found in PROD: {log_msg}"
        asyncio.get_running_loop().run_in_executor(None, send_slack_message,
                                                   msg, slack_url)


class EventLoopBlockedError(RuntimeError):
    ...


def measure_blocking_code(
    f: Callable[P, Awaitable[R]],
    func_name: Optional[str] = None,
    disable_check: bool = False,
) -> Callable[P, Awaitable[R]]:
    """
    Decorator function to measure an async function execution time and
    warning if it executes blocking code too long.

    A few useful references for deep-dive understanding of how corotine/await
    works:
    https://stackoverflow.com/a/60118660
    http://www.dabeaz.com/coroutines/Coroutines.pdf

    Note: when an async function (parent) decorated with @measure_blocking_code
    awaits on another async function (child) that's also decorated with
    @measure_blocking_code, the child's blocking execution time will NOT be
    double-counted for the parent.  See `test_async_timer.py` for detailed
    example.
    """
    if not inspect.iscoroutinefunction(f):
        log.warning('Skip @measure_blocking_code for %s', f)
        return f  # This only works for coroutine function

    func_name = func_name or f.__qualname__

    @functools.wraps(f)
    async def _measure_blocking_code_wrapper(*args: P.args,
                                             **kwargs: P.kwargs) -> R:
        if active_profiler_context_var.get():
            # When profiling is enabled and active, we don't apply the trick of
            # measure_blocking_code which would obscure the profiling result.
            return await f(*args, **kwargs)

        total_blocking_time = 0.0
        num_futures_awaited = 0
        t_start = time.perf_counter()

        def _check_blocking(blocking_time: float) -> None:
            if disable_check or blocking_time < BLOCKING_WARN_THRESHOLD_SEC:
                return

            total_time = time.perf_counter() - t_start
            blocking_ratio = total_blocking_time / total_time
            log_msg = (
                f'Async func {func_name} has executed for total '
                f'{total_time} sec, awaited {num_futures_awaited} futures, '
                f'total blocking time: {total_blocking_time:.3f} sec'
                f'({blocking_ratio:.2%}), most recently blocked for '
                f'{blocking_time:.3f} sec')

            if blocking_time < BLOCKING_ALERT_THRESHOLD_SEC:
                log.warning(log_msg)
                return

            log.critical(log_msg)
            _notify_kumo_if_prod(log_msg)
            if _stress_test_enabled():
                raise EventLoopBlockedError(
                    'FATAL blocking detected on asyncio event loop:\n' +
                    log_msg)

        # The wrapped coroutine is driven by manually stepping it via .send() /
        # .throw() and measuring how long each synchronous segment runs (the
        # time between two awaits = the blocking time). The futures it yields are
        # awaited in the *driver*'s own running task, so task-scoped APIs such as
        # `asyncio.timeout()` / `asyncio.wait_for()` (which on Python 3.14 must
        # run inside a task) keep working and their cancellation -> TimeoutError
        # conversion is preserved. Driving via detached done-callbacks (the old
        # approach) ran the steps outside any task and broke on 3.14 with
        # "RuntimeError: Timeout should be used inside a task".
        coro_wrapper = f(*args, **kwargs).__await__()

        def _drive() -> Generator[Any, Any, R]:
            nonlocal total_blocking_time, num_futures_awaited
            send_value: Any = None
            throw_exc: Optional[BaseException] = None
            first_step = True
            while True:
                start_time = time.perf_counter()
                # Count a resume of the wrapped coroutine, i.e. every step after
                # the first one. This mirrors the original counter, which only
                # incremented on the chained (non-first) resume callbacks.
                if not first_step:
                    num_futures_awaited += 1
                first_step = False
                try:
                    if throw_exc is not None:
                        next_fut = coro_wrapper.throw(throw_exc)
                        throw_exc = None
                    else:
                        next_fut = coro_wrapper.send(send_value)
                except StopIteration as se:
                    blocking_time = time.perf_counter() - start_time
                    total_blocking_time += blocking_time
                    _check_blocking(blocking_time)
                    return se.value
                except asyncio.CancelledError:
                    # Task-scoped timeouts briefly use cancellation internally.
                    # Warn only when it escapes the wrapped coroutine.
                    if num_futures_awaited:
                        log.warning(
                            'Received asyncio cancellation after awaiting %d '
                            'futures, propagating into %s', num_futures_awaited,
                            func_name)
                    raise
                blocking_time = time.perf_counter() - start_time
                total_blocking_time += blocking_time
                _check_blocking(blocking_time)
                try:
                    # Yield the future up so it is awaited inside our running
                    # task; resume the wrapped coroutine with the result (or
                    # forward an exception, e.g. a timeout cancellation).
                    send_value = yield next_fut
                except BaseException as e:  # noqa: BLE001
                    throw_exc = e
                    send_value = None

        if _IN_MEASURE_BLOCKING.get():
            # Nested under another @measure_blocking_code: run the driver in its
            # own child task so its blocking time is attributed to this function
            # and shielded from the enclosing (parent) measurement. The child
            # task carries a fresh task identity, so the wrapped coroutine's own
            # task-scoped timeouts continue to work.
            async def _run_nested() -> R:
                token = _IN_MEASURE_BLOCKING.set(True)
                try:
                    return await _CoroutineDriver(_drive())
                finally:
                    _IN_MEASURE_BLOCKING.reset(token)

            return await asyncio.ensure_future(_run_nested())

        token = _IN_MEASURE_BLOCKING.set(True)
        try:
            return await _CoroutineDriver(_drive())
        finally:
            _IN_MEASURE_BLOCKING.reset(token)

    return _measure_blocking_code_wrapper


class _CoroutineDriver(Generic[R]):
    """Awaitable that pumps a driver generator inside the running task.

    The driver yields the futures the wrapped coroutine awaits; this awaitable
    forwards those yields out to the event loop (so they are awaited by the
    current task) and feeds results/exceptions back into the driver.
    """

    __slots__ = ('_driver', )

    def __init__(self, driver: Generator[Any, Any, R]) -> None:
        self._driver = driver

    def __await__(self) -> Generator[Any, Any, R]:
        send_value: Any = None
        throw_exc: Optional[BaseException] = None
        while True:
            try:
                if throw_exc is not None:
                    next_fut = self._driver.throw(throw_exc)
                    throw_exc = None
                else:
                    next_fut = self._driver.send(send_value)
            except StopIteration as se:
                return se.value
            try:
                send_value = yield next_fut
            except BaseException as e:  # noqa: BLE001
                throw_exc = e
                send_value = None
