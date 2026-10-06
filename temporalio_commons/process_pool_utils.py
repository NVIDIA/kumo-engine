# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import multiprocessing
import os
import signal
import threading
import time
import traceback
from asyncio import AbstractEventLoop, CancelledError
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from concurrent.futures.process import BrokenProcessPool, _RemoteTraceback
from contextvars import ContextVar
from datetime import timedelta
from typing import Any, Callable, Mapping, Optional, overload

from temporalio import activity
from temporalio.exceptions import ApplicationError, FailureError

from common_utils.error import PublicException
from common_utils.error_diagnostics import (
    exception_chain_contains_restricted_detail,
    exception_message_should_be_logged_at_error,
    safe_exception_message_for_log,
)
from common_utils.kumo_timing import kumo_timing_event
from io_utils import logging
from kumo.platform_errors import as_platform_public_exception
from service_framework.deploy_stage import _DEPLOY_STAGE, DeployStage
from service_framework.env_var import EnvVar
from temporalio_commons import error_utils
from temporalio_commons.heartbeat_utils import ActivityFn, auto_heartbeater

log = logging.getLogger(__name__)

_DISKGRAPH_SUBPROCESS_ENV = '_KUMO_DISKGRAPH_SUBPROCESS'
_TEST_DYNAMIC_ENV_VARS = ('DEPLOY_ENV_CONFIG_JSON', )
_PROCESS_MODULE_IMPORT_STARTED = time.monotonic()


def _future_exception_type(future: Any) -> Optional[str]:
    if future.cancelled():
        return None
    exception = future.exception()
    return None if exception is None else type(exception).__name__


# Best-effort upper bound on synchronous process-pool warm-up (see `warm_up`).
# Aligned with the Temporal worker health check `start_period` so a slow warm-up
# on a CPU-constrained host cannot delay the worker from polling past its first
# health probe.
_WARM_UP_TIMEOUT_SECONDS = EnvVar('PROCESS_POOL_WARM_UP_TIMEOUT_SECONDS',
                                  type=float, required=False,
                                  default_value=30.0)


def _process_pool_warmup_noop(delay_seconds: float = 0.05):
    time.sleep(delay_seconds)
    return os.getpid()


def _apply_child_env_vars(env_vars: Mapping[str, Optional[Any]]) -> None:
    """Apply ``env_vars`` to the spawned child's ``os.environ``.

    A non-``None`` value sets the variable; a ``None`` value means "ensure the
    variable is unset in the child". The pop uses a default so an
    already-absent variable is a no-op instead of raising ``KeyError`` -- a
    ``KeyError`` here runs inside the pool ``initializer``, which CPython's
    ``_process_worker`` catches and then returns from *before* consuming any
    work item, so the worker exits with a pending future and the executor
    reports the whole pool as ``BrokenProcessPool``.
    """
    for env_var_name, env_var_value in env_vars.items():
        if env_var_value is not None:
            os.environ[env_var_name] = str(env_var_value)
        else:
            os.environ.pop(env_var_name, None)


def _init_spawned_process(deploy_stage: DeployStage,
                          env_vars: Mapping[str, Optional[Any]]):
    init_started = time.monotonic()
    # Initialize logging based on deploy mode in spawned executor process.
    # TODO(siyang): show processId in logging
    if deploy_stage == DeployStage.PROD:
        logging.initialize_logging(mode=logging.LoggingMode.DEVELOPMENT,
                                   log_to_file=False, log_process_id=True)
    else:
        # TODO(siyang): Bump logging level for STAGING once things stabilize.
        logging.initialize_logging(logging.LoggingMode.DEBUG, log_to_file=False,
                                   log_process_id=True)

    # Temporal workers may not get snowflake credentials from the environment
    from kumo.graph.graph import Connector  # pants: no-infer-dep
    Connector.disable_interactive_input()

    # Set environ variables
    log.info('Set env variables in child worker process: %s', env_vars)
    _apply_child_env_vars(env_vars)
    kumo_timing_event(
        "process_pool.child_initialized",
        elapsed_since_module_import_seconds=round(
            time.monotonic() - _PROCESS_MODULE_IMPORT_STARTED, 6),
        initializer_seconds=round(time.monotonic() - init_started, 6),
        pid=os.getpid(),
    )

    # Log deploy config for debugging config propagation to child processes.
    from service_framework.deploy_env_config import deploy_env
    log.info('Child process deploy_env: %s', deploy_env())


PROCESS_POOL_COMPATIBLE_ATTR = '__process_pool_compatible_activity'

in_process_pool_activity: ContextVar[bool] = ContextVar(
    'in_process_pool_activity', default=False)


@overload
def process_pool_activity_defn(fn: ActivityFn) -> ActivityFn:
    ...


@overload
def process_pool_activity_defn(
    *, heartbeat_interval: Optional[timedelta] = None
) -> Callable[[ActivityFn], ActivityFn]:
    ...


def process_pool_activity_defn(
    fn: Optional[ActivityFn] = None,
    *,
    heartbeat_interval: Optional[timedelta] = None,
):
    """
    Generic wrapper for any sync activity executed by process pool to
    convert possibly non-pickable exception to ApplicationError (picklable).

    The converted ``ApplicationError`` is sanitized (#30109): its message
    names only the activity and the exception/cause type names, and it
    carries no raw exception text or traceback, because pre-built
    ``ApplicationError``s cross the Temporal activity boundary verbatim and
    land in workflow history, job-status messages, and centralized logs.
    Retryability classification is preserved via
    ``error_utils.is_error_non_retryable``.
    ``PublicException`` passes through untouched and is the designed channel
    for user-actionable error text; the one exception is a dead-cluster
    ``PublicException``, which is converted to a retryable
    ``ApplicationError`` that keeps its public-safe message.
    """
    def _decorator(fn: ActivityFn) -> ActivityFn:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            token = in_process_pool_activity.set(True)
            try:
                return fn(*args, **kwargs)
            except BaseException as ex:
                platform_exception = (None if isinstance(ex, PublicException)
                                      else as_platform_public_exception(ex))
                if platform_exception is not None:
                    raise platform_exception from None
                # TODO(siyang): Maybe allow caller to specify additional error
                # types that are known to be picklable, as an argument in
                # process_pool_activity_defn().
                if isinstance(ex, FailureError):
                    if ex.failure:
                        # failure is a protobuf that is not pickable
                        ex._failure = None
                    raise ex

                # PublicException is picklable and should be handled by the
                # activity interceptor, not wrapped here.
                # Exception: dead cluster errors must be converted to
                # ApplicationError with non_retryable=False to guarantee
                # Temporal retries the activity with a new cluster.
                if isinstance(ex, PublicException):
                    if error_utils.is_retryable_cluster_error(ex):
                        log.warning(
                            'Dead cluster error detected, converting to '
                            'retryable ApplicationError: %s', ex)
                        # The PublicException message is public-safe by
                        # contract; the raw traceback is not (its frames can
                        # carry non-public cause text), so it stays out of
                        # the ApplicationError.
                        raise ApplicationError(
                            str(ex),
                            type=type(ex).__name__,
                            non_retryable=False,
                        ) from None
                    raise

                allowed_picklable_errors = (CancelledError, )
                if isinstance(ex, allowed_picklable_errors):
                    raise

                log.warning(f'Convert {type(ex)} to ApplicationError.')
                activity_name = activity.info().activity_type
                cause = ex.__cause__
                cause_type = (type(cause).__name__ if cause is not None
                              and not isinstance(cause, _RemoteTraceback) else
                              'none')
                message_is_visible = exception_message_should_be_logged_at_error(
                    ex)
                has_restricted_detail = exception_chain_contains_restricted_detail(
                    ex)
                if message_is_visible:
                    if has_restricted_detail:
                        log.error(
                            'Process-pool activity %s failed before exception '
                            'conversion: %s (restricted cause_type=%s)',
                            activity_name,
                            ex,
                            cause_type,
                        )
                    else:
                        log.error(
                            'Process-pool activity %s failed before exception '
                            'conversion: %s',
                            activity_name,
                            ex,
                            exc_info=(type(ex), ex, ex.__traceback__),
                        )
                else:
                    safe_message = safe_exception_message_for_log(ex)
                    log.error(
                        'Process-pool activity %s failed before exception '
                        'conversion (type=%s, cause_type=%s%s)',
                        activity_name,
                        type(ex).__name__,
                        cause_type,
                        (f', detail={safe_message}'
                         if safe_message is not None else ''),
                    )
                if has_restricted_detail:
                    log.debug(
                        'Restricted raw process-pool failure before exception '
                        'conversion',
                        exc_info=(type(ex), ex, ex.__traceback__),
                    )
                # The message and details deliberately exclude str(ex) and
                # the traceback: arbitrary subprocess exception text can
                # carry request bodies, SQL, or credentials, and pre-built
                # ApplicationErrors cross the Temporal boundary verbatim
                # (see KumoActivityInboundInterceptor). Type names alone
                # survive for diagnosis; a boundary that wants user-facing
                # text must raise PublicException instead (#30109).
                message = (f'Activity {activity_name} failed due to an '
                           f'unexpected {type(ex).__name__} error.')
                # Name the cause type only; skip _RemoteTraceback
                # (multiprocessing internal wrapper).
                if cause and not isinstance(cause, _RemoteTraceback):
                    message += f' Caused by {type(cause).__name__}.'
                raise ApplicationError(
                    message,
                    type=type(ex).__name__,
                    non_retryable=error_utils.is_error_non_retryable(ex),
                ) from None  # Prevents __context__ chain serialization (avoids _RemoteTraceback leak)
            finally:
                in_process_pool_activity.reset(token)

        setattr(wrapper, PROCESS_POOL_COMPATIBLE_ATTR, True)
        # Make sure all process-pool executed activites use auto_heartbeater
        # to start a background thread to report heartbeat and handle
        # cancellation.
        with_auto_heartbeater = auto_heartbeater(
            heartbeat_interval=heartbeat_interval)(wrapper)
        return activity.defn(with_auto_heartbeater)  # type: ignore

    if fn is not None:
        return _decorator(fn)
    return _decorator


class HealingProcessPoolExecutor(ProcessPoolExecutor):
    # pylint: disable=super-init-not-called
    def __init__(
        self,
        max_workers: int,
        *,
        deploy_stage: DeployStage = _DEPLOY_STAGE.parse_value(),
        env_vars: Mapping[str, Optional[Any]] = {},
    ):
        self._max_workers = max_workers
        self._deploy_stage = deploy_stage
        self._env_vars = env_vars
        self._pool: ProcessPoolExecutor = self._create_pool()

    def _create_pool(self) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=self._max_workers,
            # This must be set to "spawn" for worker child processes.
            mp_context=multiprocessing.get_context('spawn'),
            initializer=_init_spawned_process,
            initargs=(self._deploy_stage, self._env_vars),
        )

    def _maybe_refresh_test_env_vars(self) -> None:
        if self._deploy_stage != DeployStage.TEST:
            return

        env_vars = dict(self._env_vars)
        changed = False
        for name in _TEST_DYNAMIC_ENV_VARS:
            current_value = os.environ.get(name)
            if env_vars.get(name) != current_value:
                env_vars[name] = current_value
                changed = True

        if not changed:
            return

        if self._pool._pending_work_items:
            log.warning(
                'Process pool env changed but %d work item(s) are still '
                'pending; keeping existing child processes.',
                len(self._pool._pending_work_items),
            )
            return

        log.info('Recreating test process pool with updated env vars: %s',
                 _TEST_DYNAMIC_ENV_VARS)
        self._pool.shutdown(wait=True, cancel_futures=False)
        self._env_vars = env_vars
        self._pool = self._create_pool()

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        return self._pool.map(fn, *iterables, timeout=timeout,
                              chunksize=chunksize)

    def shutdown(self, wait=True, *, cancel_futures=False):
        return self._pool.shutdown(wait=wait, cancel_futures=cancel_futures)

    def warm_up(self, timeout: float = _WARM_UP_TIMEOUT_SECONDS.parse_value()):
        # Pre-spawn child processes so the first real activity does not pay the
        # (multi-second) child import cost. This is a best-effort latency
        # optimization, NOT a correctness requirement: any child that has not
        # finished warming by `timeout` keeps initializing in the background and
        # warms lazily on first submit. The timeout exists because this runs
        # synchronously inside worker build, before the worker starts polling --
        # an unbounded wait here (e.g. many children serializing heavy imports on
        # a CPU-constrained host) would delay polling and trip the worker's
        # Temporal health check.
        started = time.monotonic()
        deadline = started + timeout
        futures = [
            self._pool.submit(_process_pool_warmup_noop)
            for _ in range(self._max_workers)
        ]
        pids = []
        timed_out = False
        for future in futures:
            remaining = deadline - time.monotonic()
            try:
                pids.append(future.result(timeout=max(remaining, 0.0)))
            except TimeoutError:  # concurrent.futures.TimeoutError (see imports)
                timed_out = True
                log.warning(
                    'Process pool warm_up exceeded %.1fs budget after warming '
                    '%d/%d child(ren); remaining children will initialize '
                    'lazily on first use.', timeout, len(pids),
                    self._max_workers)
                break
        kumo_timing_event(
            "process_pool.warm_up.done",
            elapsed_seconds=round(time.monotonic() - started, 6),
            max_workers=self._max_workers,
            child_pid_count=len(set(pids)),
            timed_out=timed_out,
        )

    def terminate(self):
        if self._pool._pending_work_items:
            child_pids = list(self._pool._processes.keys())
            for child_pid in child_pids:
                log.warning('Killing child process pid=%d', child_pid)
                os.kill(child_pid, signal.SIGKILL)
        self._pool.shutdown(wait=True)

    def submit(self, fn, /, *args, **kwargs):
        self._maybe_refresh_test_env_vars()
        if len(self._pool._pending_work_items) >= self._max_workers:
            log.warning(
                f'All {self._max_workers} worker processes are busy, '
                f'queue count: {self._pool._queue_count}, '
                f'pending work items: {len(self._pool._pending_work_items)}.')
            if activity.in_activity():
                log.warning(
                    'Activity will not heartbeat while waiting for the next '
                    f'available worker process: {activity.info()}')
        try:
            submit_started = time.monotonic()
            fn_name = getattr(fn, "__name__", fn.__class__.__name__)
            kumo_timing_event("process_pool.submit.start", fn_name=fn_name)
            future = self._pool.submit(fn, *args, **kwargs)  # type: ignore
            future.add_done_callback(lambda completed: kumo_timing_event(
                "process_pool.submit.done",
                elapsed_seconds=round(time.monotonic() - submit_started, 6),
                fn_name=fn_name,
                cancelled=completed.cancelled(),
                exception_type=_future_exception_type(completed),
            ))
            return future
        except BrokenProcessPool as ex:
            log.error("Process pool is unusable due to %s, restarting...\n%s",
                      ex, traceback.format_exc())
            # No wait for shutting down broken process pool. In certain case
            # of OOM killed child process, shutdown(wait=True) might hang
            # indefinitely and therefore freeze the entire worker.
            self._pool.shutdown(wait=False)
            self._pool = self._create_pool()
            log.warning("New process pool created, re-submitting task.")
            submit_started = time.monotonic()
            fn_name = getattr(fn, "__name__", fn.__class__.__name__)
            future = self._pool.submit(fn, *args, **kwargs)  # type: ignore
            future.add_done_callback(lambda completed: kumo_timing_event(
                "process_pool.submit.done",
                elapsed_seconds=round(time.monotonic() - submit_started, 6),
                fn_name=fn_name,
                restarted_pool=True,
                cancelled=completed.cancelled(),
                exception_type=_future_exception_type(completed),
            ))
            return future


# Per-worker-thread storage for the async caller's event loop.
#
# Why this exists: ``activity.heartbeat()`` is bound at activity-registration
# time. Sync activities get a thread-safe impl wrapped by Temporal in
# ``asyncio.run_coroutine_threadsafe`` (``temporalio/worker/_activity.py``
# ~line 716–727), so calling it from any thread Just Works. Async activities
# get the loop-coupled impl: ``self._worker._heartbeat`` → ``asyncio.create_task``
# (``_activity.py`` ~line 238), which requires the loop in the *calling* thread.
# When synchronous DiskGraph work is invoked via ``asyncio.to_thread`` from
# inside an async ``@activity.defn``, it runs in a thread-pool worker with no
# running loop. A direct async-activity heartbeat therefore raises
# ``RuntimeError: no running event loop``. This loop stamp is consumed both by
# the standalone ingest supervisor and by native operations that still use
# ``diskgraph_subprocess``.
#
# We use a ``threading.local`` rather than a ``ContextVar`` because the
# consumers are synchronous code running on the ``asyncio.to_thread`` worker
# thread; the caller stamps its loop there before invoking the sync body.
# Token-based ``ContextVar`` reset crashes with "was created in a different
# Context" when the wrapping ``with`` spans an ``await asyncio.to_thread``,
# because the awaited thread runs ``func`` in a *copied* context (see
# ``asyncio.to_thread`` impl) and the token's identity doesn't survive that
# copy. ``threading.local`` sidesteps that entirely — each worker thread
# has its own slot, set/cleared by the dispatch wrapper.
_thread_local = threading.local()


def _set_async_caller_loop(loop: Optional[AbstractEventLoop]) -> None:
    """Stamp the async caller's event loop onto the current (worker)
    thread so synchronous DiskGraph workers can route heartbeat callbacks
    back onto it. Set to ``None`` to clear after the call.
    """
    _thread_local.async_caller_loop = loop


def _get_async_caller_loop() -> Optional[AbstractEventLoop]:
    return getattr(_thread_local, 'async_caller_loop', None)


def run_with_async_caller_loop(loop: AbstractEventLoop, fn: Callable, *args:
                               Any, **kwargs: Any) -> Any:
    """Run ``fn(*args, **kwargs)`` with ``loop`` stamped on this thread's
    ``threading.local``, then clear the stamp on exit. Use as the target
    of ``asyncio.to_thread`` from an async ``@activity.defn`` to make
    synchronous heartbeat callbacks thread-safe-routable to ``loop``.
    """
    _set_async_caller_loop(loop)
    try:
        return fn(*args, **kwargs)
    finally:
        _set_async_caller_loop(None)


def _run_in_diskgraph_subprocess(fn, *args, **kwargs):
    """Run *fn* in a single-worker subprocess with SIGKILL cancellation.

    Polls ``fut.result(timeout=10)`` so the calling thread stays in Python
    bytecode where ``CancelledError`` (from ``auto_heartbeater``) can be
    delivered.  On cancellation the child process is SIGKILLed.

    Heartbeat dispatch on polling timeout has two paths:
    - Sync-activity caller (master path): ``activity.heartbeat()`` is
      thread-safe (Temporal's sync wrapper).  Call directly.
    - Async-activity caller via ``asyncio.to_thread`` (pool path): the
      caller stamped its loop with ``run_with_async_caller_loop``;
      route the heartbeat onto that loop with ``call_soon_threadsafe``,
      because the async-activity heartbeat impl needs the loop in the
      calling thread.  Fire-and-forget — heartbeat hiccups must not fail
      the ingest.
    """
    executor = HealingProcessPoolExecutor(
        1,
        env_vars={_DISKGRAPH_SUBPROCESS_ENV: '1'},
    )
    fut = executor.submit(fn, *args, **kwargs)
    try:
        while True:
            try:
                return fut.result(timeout=10.0)
            except TimeoutError:
                if activity.in_activity():
                    loop = _get_async_caller_loop()
                    if loop is not None:
                        loop.call_soon_threadsafe(activity.heartbeat)
                    else:
                        activity.heartbeat()
                    if activity.is_cancelled():
                        raise CancelledError(
                            f'{fn.__name__} cancelled via diskgraph_subprocess')
    finally:
        log.info('Terminating diskgraph subprocess for %s.', fn.__name__)
        executor.terminate()


def diskgraph_subprocess(fn):
    """Decorator that runs *fn* in a SIGKILL-able child process.

    Some table-generation and materialization operations still call Rust via
    PyO3. Python's thread-based cancellation cannot interrupt native code, so
    this decorator isolates those calls in a subprocess that can be SIGKILLed.
    Standalone graph ingest does not use this decorator.

    When the decorated function is called inside a child process (detected via
    the ``_KUMO_DISKGRAPH_SUBPROCESS`` env var), the original function is
    invoked directly — avoiding infinite recursion from pickle round-tripping.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if os.environ.get(_DISKGRAPH_SUBPROCESS_ENV):
            return fn(*args, **kwargs)
        # Only use subprocess isolation inside a Temporal activity — the
        # SIGKILL mechanism is pointless outside one, and arguments may not
        # be picklable in tests.
        if not activity.in_activity():
            return fn(*args, **kwargs)
        return _run_in_diskgraph_subprocess(wrapper, *args, **kwargs)

    return wrapper
