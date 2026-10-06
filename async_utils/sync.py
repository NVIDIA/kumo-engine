# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextvars
import functools
import inspect
import logging
import pickle
import traceback
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from enum import Enum
from hashlib import md5
from typing import Any, Awaitable, Callable, Generic, Optional, TypeVar, cast

from asgiref.sync import SyncToAsync
from typing_extensions import Concatenate, ParamSpec

from monitoring import tracing
from service_framework.deploy_stage import _DEPLOY_STAGE, DeployStage
from service_framework.env_var import EnvVar

from .caching import reuse_inflight_future
from .instrument import _notify_kumo_if_prod

log = logging.getLogger(__name__)

P = ParamSpec('P')
# pylint: disable=typevar-name-incorrect-variance
T = TypeVar('T', covariant=True)

SelfType = TypeVar("SelfType")

SyncMethod = Callable[Concatenate[SelfType, P], T]


class FutureDedupeOption(str, Enum):
    """
    Option to define caching(dedupe) behavior for concurrent && redundant
    invocation of a @blocking_method from multiple threads in threadpool
    executor.
    """
    # [Default] concurrent && redundant calls on the same instance method
    # would be deduped and executed once, with result shared by callers.
    BY_INSTANCE_ID = 'BY_INSTANCE_ID'

    # Suitable for read-only (no side effect) methods that depend on instance
    # state (pickled). This means concurrent && redundant method invocations on
    # difference instances with identical state will be deduped.
    BY_INSTANCE_STATE = 'BY_INSTANCE_STATE'

    # If the method invocation mutates instance state (the method is expected
    # to be either `@synchronized` or associated with single-thread executor),
    # we don't dedupe concurrent calls.
    DISABLED = 'DISABLED'


class RpcFuncSync(Generic[P, T]):
    def __init__(self, handler_func: Callable[P, T],
                 require_sync_call_from_thread: bool = False,
                 executor: Optional[ThreadPoolExecutor] = None,
                 tracing_options: Optional[tracing.TraceOptions] = None,
                 instance_hash: Optional[str] = None) -> None:
        self._tracing_options = tracing_options
        self._tracing_options_coro = deepcopy(tracing_options)
        if self._tracing_options_coro:
            threadpool_name = (executor._thread_name_prefix
                               if executor else 'DefaultThreadPool')
            self._tracing_options_coro.span_name_suffix = \
                f'-coroutine({threadpool_name})'
            # No need to track param or return values since this is just
            # sync func to async coroutine wrapper
            self._tracing_options_coro.trace_params = False
            self._tracing_options_coro.trace_return_value = False
        self._handler_func = (handler_func if not tracing_options else
                              tracing.trace(tracing_options)(handler_func))
        self._require_sync_call_from_thread = require_sync_call_from_thread
        self._executor = executor
        self._debug_mode = EnvVar('DEBUG_BLOCKING_IO', bool,
                                  default_value=False).parse_value()
        self._is_prod = (_DEPLOY_STAGE.parse_value() == DeployStage.PROD)
        self._instance_hash = instance_hash

    @property
    def sync(self) -> Callable[P, T]:
        if self._require_sync_call_from_thread:
            self._check_called_from_thread()
        return self._handler_func

    @property
    def coroutine(self) -> Callable[P, Awaitable[T]]:
        coro = SyncToAsync(self._handler_func, thread_sensitive=False,
                           executor=self._executor)
        if self._instance_hash != FutureDedupeOption.DISABLED:
            coro = reuse_inflight_future(
                instance_hash=self._instance_hash)(coro)

        if self._tracing_options_coro:
            coro = tracing.trace(self._tracing_options_coro)(coro)

        return coro  # type: ignore

    def __call__(self, *__args: P.args, **__kwargs: P.kwargs) -> T:
        if self._require_sync_call_from_thread:
            self._check_called_from_thread()
        return self._handler_func(*__args, **__kwargs)

    def _check_called_from_thread(self):
        try:
            loop = asyncio.get_event_loop()
            loop = cast(asyncio.BaseEventLoop, loop)
        except RuntimeError:
            # OK, no event loop in thread.
            return
        if loop.is_running():
            error_msg = (
                'Not allowed to call a blocking function: '
                f'{self._handler_func.__qualname__} directly on asyncio event '
                'loop, please change to await func.coroutine(params) instead')
            if self._debug_mode or self._is_prod:
                stack: traceback.StackSummary = traceback.extract_stack()
                error_msg = '{}\nCallstack:\n{}'.format(
                    error_msg, '\n'.join(stack.format()))
                log.critical(error_msg)
                _notify_kumo_if_prod(error_msg)
            else:
                raise RuntimeError(error_msg)


class RpcMethodSync(Generic[SelfType, P, T]):
    def __init__(
        self,
        handler_func: SyncMethod[SelfType, P, T],
        require_sync_call_from_thread: bool = False,
        executor: Optional[ThreadPoolExecutor] = None,
        tracing_options: Optional[tracing.TraceOptions] = None,
        fut_dedupe_option: FutureDedupeOption = FutureDedupeOption.
        BY_INSTANCE_ID,
    ) -> None:
        self._handler_func = handler_func
        self._require_sync_call_from_thread = require_sync_call_from_thread
        self._executor = executor
        self._tracing_options = tracing_options
        self._fut_dedupe_option = fut_dedupe_option

    def _get_instance_hash(self, instance: Any) -> str:
        if self._fut_dedupe_option == FutureDedupeOption.DISABLED:
            return FutureDedupeOption.DISABLED
        if self._fut_dedupe_option == FutureDedupeOption.BY_INSTANCE_ID:
            return str(id(instance))

        assert self._fut_dedupe_option == FutureDedupeOption.BY_INSTANCE_STATE
        try:
            serialized = pickle.dumps(instance)
            return md5(serialized).hexdigest()
        except BaseException:
            log.warning('Unable to pickle %s, using id() for caching key',
                        instance)
            return str(id(instance))

    def __get__(self, instance, owner):
        # `instance` is the class instance if present, and `owner` is
        # the class. `instance` may be `None` if the method is either a static
        # or a class method.
        if instance is not None:
            instance_hash = self._get_instance_hash(instance)
            return RpcFuncSync[P, T](self._handler_func.__get__(instance, None),
                                     self._require_sync_call_from_thread,
                                     self._executor, self._tracing_options,
                                     instance_hash)
        return self

    @property
    def sync(self) -> RpcFuncSync[P, T]:
        raise NotImplementedError()

    async def coroutine(self, *__args: P.args, **__kwargs: P.kwargs) -> T:
        raise NotImplementedError()

    def __call__(self, *__args: P.args, **__kwargs: P.kwargs) -> T:
        raise NotImplementedError()


def _is_instance_method(func):
    spec = inspect.signature(func)
    if len(spec.parameters) > 0:
        if list(spec.parameters.keys())[0] == 'self':
            return True
    return False


def _blocking_func(
    func: Callable[P, T],
    *,
    executor: Optional[ThreadPoolExecutor] = None,
    tracing_options: Optional[tracing.TraceOptions] = None,
) -> RpcFuncSync[P, T]:
    if _is_instance_method(func) or inspect.iscoroutinefunction(func):
        raise Exception(
            f'@blocking_func() cannot be applied to {func.__qualname__}')
    return RpcFuncSync[P, T](func, True, executor, tracing_options)


def blocking_func(
        executor: Optional[ThreadPoolExecutor] = None,
        tracing_options: Optional[
            tracing.TraceOptions] = tracing.TraceOptions(),
):
    """Decorator intended for any function (or static method) that performs
    blocking/IO operation.

    Args:
        executor (Optional[ThreadPoolExecutor], optional): A thread pool
        executor to run the decorated blocking func. Defaults to None, in which
        case the default executor of asyncio event loop will be used.
    """
    return functools.partial(_blocking_func, executor=executor,
                             tracing_options=tracing_options)


def _blocking_method(
    method: SyncMethod[SelfType, P, T],
    *,
    fut_dedupe_option: FutureDedupeOption,
    executor: Optional[ThreadPoolExecutor] = None,
    tracing_options: Optional[tracing.TraceOptions] = None,
) -> RpcMethodSync[SelfType, P, T]:
    if not _is_instance_method(method) or inspect.iscoroutinefunction(method):
        raise Exception(
            f'@blocking_method() cannot be applied to {method.__qualname__}')
    return RpcMethodSync[SelfType, P, T](method, True, executor,
                                         tracing_options, fut_dedupe_option)


def blocking_method(
    executor: Optional[ThreadPoolExecutor] = None,
    *,
    tracing_options: Optional[tracing.TraceOptions] = tracing.TraceOptions(),
    fut_dedupe_option: FutureDedupeOption = FutureDedupeOption.BY_INSTANCE_ID,
):
    """Decorator intended for any class instance method that performs
    blocking/IO operation.

    Args:
        executor (Optional[ThreadPoolExecutor], optional): A thread pool
        executor to run the decorated blocking func. Defaults to None, in which
        case the default executor of asyncio event loop will be used.
    """
    return functools.partial(
        _blocking_method,
        fut_dedupe_option=fut_dedupe_option,
        executor=executor,
        tracing_options=tracing_options,
    )


def sync_to_async(
    fn: Callable[P, T],
    executor: Optional[ThreadPoolExecutor],
    *,
    tracing_options: Optional[tracing.TraceOptions] = tracing.TraceOptions(),
    propagate_ctx: bool = True,
) -> Callable[P, Awaitable[T]]:
    """
    Convert a sync function to an async callable running in a thread.

    Unlike `asyncio.to_thread()` or `run_in_executor()`, this utility func
    provides a convenient and type-safe way to run sync(blocking) func as async
    coroutine with options to enable tracing, executor(threadpool), and context
    propagation.

    Args:
        fn: Synchronous function to wrap.
        executor: Optional thread pool to use.
        tracing_options: Optional tracing configuration.
        propagate_ctx: Whether to propagate contextvars to the thread.

    Returns:
        An async function returning the result of `fn`.
    """
    loop = asyncio.get_running_loop()

    if tracing_options:
        fn = tracing.trace(tracing_options)(fn)

    async def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        if propagate_ctx:
            ctx = contextvars.copy_context()
            closure = functools.partial(ctx.run, fn, *args, **kwargs)
        else:
            closure = functools.partial(fn, *args, **kwargs)
        return await loop.run_in_executor(executor, closure)

    return _wrapper
