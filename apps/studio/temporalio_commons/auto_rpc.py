# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AutoRPC helps expose a regular python class as a RPC service by simply adding
the @auto_rpc_service class decorator. Example:

@auto_rpc_service
class MyService:
    def __init__(self, ...):
        ...

    def foo_method(self, args) -> Y:
    def bar_method(self, arg1, arg2, /, arg3, arg4=x, arg5=y, *args) -> T:
    async def baz_async_method(self, ...):

To run the RPC Server:

    if __name__ == "__main__":
        my_service_impl = MyService(...)
        server_config = AutoRpcServerWorkerConfig(...)
        asyncio.run(run_server(my_service_impl, server_config))

To create and use a type-safe client stub of "MyService":
    client_config = AutoRpcClientConfig(...)
    my_service = create_client(MyService, client_config)
    ...
    foo_result: Y = my_service.foo_method(...)
    bar_result: T = my_service.bar_method(...)

"""

import asyncio
import copy
import dataclasses
import inspect
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from inspect import _ParameterKind
from typing import (
    Any,
    Awaitable,
    Callable,
    Concatenate,
    Dict,
    Generic,
    List,
    Optional,
    ParamSpec,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

from asgiref.sync import AsyncToSync
from temporalio import activity, workflow
from temporalio.api.common.v1 import Payload
from temporalio.client import WorkflowFailureError
from temporalio.common import RetryPolicy
from temporalio.exceptions import (
    ActivityError,
    ApplicationError,
    TemporalError,
    TimeoutError,
)
from temporalio.service import RPCError as TemporalRPCError
from temporalio.service import RPCStatusCode
from temporalio.worker import Worker
from temporalio.workflow import ActivityCancellationType

from async_utils.instrument import measure_blocking_code
from async_utils.sync import RpcFuncSync, RpcMethodSync, sync_to_async
from common_utils.error import PublicException
from common_utils.error_diagnostics import (
    exception_message_should_be_logged_at_error,
    safe_exception_message_for_log,
)
from io_utils import kumo_dataclass, logging
from monitoring import tracing
from service_framework.deploy_env_config import DeployEnvConfig
from temporalio_commons.auto_rpc_config import (
    DEFAULT_RETRY_POLICY,
    AutoRpcClientConfig,
    AutoRpcTimeoutConfig,
    parse_auto_rpc_server_name_prefix_from_env,
)
from temporalio_commons.auto_rpc_exception import (
    RpcError,
    _AutoRpcNonRetriableError,
    is_retryable,
    wf_failure_to_rpc_error,
)
from temporalio_commons.dataclass_converter import KumoPayloadConverter
from temporalio_commons.heartbeat_utils import auto_heartbeater
from temporalio_commons.temporal_client import TemporalClient
from temporalio_commons.worker_registry import WorkerEnv, WorkerOptions

log = logging.getLogger(__name__)
server_log = log.getChild('server')
client_log = log.getChild('client')

NO_RETRY_POLICY = RetryPolicy(maximum_attempts=1)

DEFAULT_TASK_TIMEOUT = timedelta(seconds=30)

P = ParamSpec('P')
# pylint: disable=typevar-name-incorrect-variance
T = TypeVar('T', covariant=True)

SelfType = TypeVar("SelfType")
SyncMethod = Callable[Concatenate[SelfType, P], T]
AsyncMethod = Callable[Concatenate[SelfType, P], Awaitable[T]]

_AUTO_RPC_DATA_CONVERTER = KumoPayloadConverter()
_AUTO_RPC_SERDE_THREADPOOL = ThreadPoolExecutor(
    max_workers=4, thread_name_prefix='auto-rpc-serde')


class RpcFuncAsync(Generic[P, T]):
    def __init__(self, handler_func: Callable[P, Awaitable[T]], *,
                 exclude_from_block_detection: bool = False) -> None:
        self._handler_func = measure_blocking_code(
            handler_func, disable_check=exclude_from_block_detection)

    @property
    def sync(self) -> Callable[P, T]:
        return AsyncToSync(self._handler_func)  # type: ignore

    async def coroutine(self, *__args: P.args, **__kwargs: P.kwargs) -> T:
        return await self(*__args, **__kwargs)

    async def __call__(
        self,
        *__args: P.args,
        **__kwargs: P.kwargs,
    ) -> T:
        return await self._handler_func(*__args, **__kwargs)


class RpcMethodAsync(Generic[SelfType, P, T]):
    def __init__(self, handler_func: AsyncMethod[SelfType, P, T]) -> None:
        self._handler_func = handler_func

    def __get__(self, instance, owner):
        if instance:
            return RpcFuncAsync[P,
                                T](self._handler_func.__get__(instance, None))
        return self

    @property
    def sync(self) -> RpcFuncSync[P, T]:
        raise NotImplementedError()

    async def coroutine(self, *__args: P.args, **__kwargs: P.kwargs) -> T:
        raise NotImplementedError()

    async def __call__(self, *__args: P.args, **__kwargs: P.kwargs) -> T:
        raise NotImplementedError()


# Decorator that wraps RPC method handler methods (instance methods) and
# provides RPC clients with a dual set of both synchronous and asynchronous
# (coroutine) callable interfaces.
#
# Caller can directly invoke the function as is, or use "func.sync(args)" or
# "func.coroutine(args)" to request either synchronous or asynchronous
# invocation of the same func. Example:
#
# class MyService:
#   @rpc_handler
#   def func_sync(self, arg1: Type1, arg2: Type2) -> ReturnType:
#     ...
#
#   @rpc_handler
#   async def func_async(self, ...) -> ReturnType:
#     ...
#
# my_service = MyService()
#
# ret1 = my_service.func_sync(arg1, arg2)
# assert type(ret1) == ReturnType
#
# ret2 = my_service.func_sync.sync(arg1, arg2)
# assert type(ret2) == ReturnType
#
# ret3 = my_service.func_sync.coroutine(arg1, arg2)
# assert type(ret3) == Awaitable[ReturnType]
#
# See "test_auto_rpc_handler.py" for more examples.
@overload
def rpc_handler(
        func: AsyncMethod[SelfType, P, T]) -> RpcMethodAsync[SelfType, P, T]:
    ...


@overload
def rpc_handler(
        func: SyncMethod[SelfType, P, T]) -> RpcMethodSync[SelfType, P, T]:
    ...


def rpc_handler(func: Union[AsyncMethod[SelfType, P, T], SyncMethod[SelfType,
                                                                    P, T]]):
    if inspect.iscoroutinefunction(func):
        return RpcMethodAsync[SelfType, P, T](func)  # type: ignore
    return RpcMethodSync[SelfType, P, T](func)  # type: ignore


# pylint: disable=too-few-public-methods
@workflow.defn
class _RpcMethodWorkflow:
    """A private, generic single-activity workflow that executes a rpc method
    (as activity func) by the rpc method name.
    """
    @workflow.run
    async def run(
        self,
        rpc_method_name: str,
        config: AutoRpcTimeoutConfig,
        *payloads: Payload,
    ) -> Payload:
        # server_log.debug('Executing %s with args: %s', rpc_method_name, args)
        # See https://docs.temporal.io/activities/#local-activity
        # Always execute "local activty" on AutoRPC server side.
        deadline = config.attempt_timeout or config.rpc_timeout
        rpc_start_time = workflow.time()
        try:
            return await workflow.execute_activity(
                activity=rpc_method_name,
                args=payloads,
                schedule_to_close_timeout=deadline,
                retry_policy=NO_RETRY_POLICY,
                cancellation_type=(
                    ActivityCancellationType.WAIT_CANCELLATION_COMPLETED),
            )
        except ActivityError as activity_err:
            cause = activity_err.cause
            if isinstance(cause, ApplicationError):
                raise cause
            raise
        except TimeoutError:
            now = workflow.time()
            server_log.exception(
                f'RPC {rpc_method_name} execution timed out '
                f'after {now-rpc_start_time:.3f} seconds, '
                f'deadline={deadline.total_seconds():.3f} sec')
            try:
                worker_info = await workflow.execute_local_activity(
                    activity='get_worker_info',
                    schedule_to_close_timeout=timedelta(seconds=1.0),
                    retry_policy=NO_RETRY_POLICY)
                elapsed = workflow.time() - now
                server_log.warning(
                    f'Fetched worker info in {elapsed:.3f} seconds:\n'
                    f'{worker_info}')
            except BaseException as e:
                server_log.exception(
                    f"Failed to examine worker info: most likely, the asyncio "
                    f"event loop is blocked. Exception: {e}")
            raise


def _is_public_instance_method(function: Callable) -> bool:
    # Whether the function is public instance method.
    if function.__name__.startswith('_'):
        return False

    # Verify it's a instance method, i.e., 1) must at least have the first
    # parameter being "self"; 2) must have __qualname__ as '<cls>.<func_name>'
    params = list(inspect.signature(function).parameters.values())
    if len(params) < 1 or params[0].name != 'self':
        return False
    names = function.__qualname__.split('.')
    return len(names) == 2 and names[1] == function.__name__


class AutoRpcFunctionDisqualified(Exception):
    def __init__(self, function: Callable, reason: str) -> None:
        super().__init__(reason)
        self._function = function
        self._reason = reason

    def __str__(self) -> str:
        return (
            f'{self._function.__qualname__} cannot serve as RPC method due ' +
            f'to: {self._reason}')


# TODO(siyang): Consider provide a function decorator to opt-in/out of AutoRPC.
def _qualify_as_rpc_method(function: Callable) -> bool:
    """Return whether the function is qualified to be exposed as RPC method."""

    # Disallow @overload method.
    if function.__name__ == '_overload_dummy':
        raise AutoRpcFunctionDisqualified(
            function, 'AutoRPC service class must not have overload function.')

    if not _is_public_instance_method(function):
        return False  # Skip private or non-instance method.

    if inspect.isasyncgenfunction(function):
        raise AutoRpcFunctionDisqualified(function,
                                          'func is async generator function')
    if inspect.isgeneratorfunction(function):
        raise AutoRpcFunctionDisqualified(function,
                                          'func is generator function')

    # We don't support keyward-only or kwargs.
    for param in inspect.signature(function).parameters.values():
        if param.kind in (_ParameterKind.KEYWORD_ONLY,
                          _ParameterKind.VAR_KEYWORD):
            raise AutoRpcFunctionDisqualified(
                function,
                f'func cannot have keyword-only or kwargs parameter: {param}.')

    # TODO(siyang): Implement more strict checks on parameter types and return
    # type to be serializable (primitives, dataclasses, list|set|dict)

    return True


# Internal attribute name to mark a method as rpc method.
_RPC_METHOD_ATTR = '__rpc_method__'

ServiceType = TypeVar('ServiceType')


# Predicate for a @rpc_handler function bound to a service impl instance.
def _auto_rpc_instance_member(m):
    return isinstance(m, (RpcFuncAsync, RpcFuncSync)) and hasattr(
        m._handler_func,  # pylint: disable=protected-access
        _RPC_METHOD_ATTR)


# Predicate for a @rpc_handler method bound to an @auto_rpc_service class.
def _auto_rpc_class_member(m):
    return isinstance(m, (RpcMethodAsync, RpcMethodSync)) and hasattr(
        m._handler_func,  # pylint: disable=protected-access
        _RPC_METHOD_ATTR)


def _report_retryable_exception(error: BaseException) -> None:
    if exception_message_should_be_logged_at_error(error):
        print(f'raise retriable ex: {error}')
        return

    safe_message = (safe_exception_message_for_log(error)
                    or 'Restricted integration operation failed')
    print(f'raise retriable ex: {type(error).__name__}: {safe_message}')
    server_log.debug(
        'Restricted raw AutoRPC retry failure (error_type=%s)',
        type(error).__name__,
        exc_info=(type(error), error, error.__traceback__),
    )


def autorpc_handler_wrapper(func):
    """
    Wrap the rpc handler function with generic framework-level logic to handle
    deserialization and non-retriable errors.
    """
    def _wrap_handler_sync(*payloads: Payload, **kwargs) -> Payload:
        for payload in payloads:
            assert isinstance(payload, Payload)
            assert payload.metadata.get('encoding',
                                        b'') != b'binary/passthrough'
        args = _AUTO_RPC_DATA_CONVERTER.from_payloads(payloads)
        try:
            ret = func(*args, **kwargs)
            return _AUTO_RPC_DATA_CONVERTER.to_payload(ret)
        except BaseException as ex:
            if isinstance(ex, (PublicException, ApplicationError)):
                raise
            if is_retryable(ex):
                _report_retryable_exception(ex)
                raise ex
            raise _AutoRpcNonRetriableError(type(ex), str(ex)) from ex

    async def _wrap_handler_async(*payloads: Payload, **kwargs) -> Payload:
        for payload in payloads:
            assert isinstance(payload, Payload)
            assert payload.metadata.get('encoding',
                                        b'') != b'binary/passthrough'
        args = await sync_to_async(_AUTO_RPC_DATA_CONVERTER.from_payloads,
                                   _AUTO_RPC_SERDE_THREADPOOL)(payloads)
        try:
            ret = await func(*args, **kwargs)
            return await sync_to_async(_AUTO_RPC_DATA_CONVERTER.to_payload,
                                       _AUTO_RPC_SERDE_THREADPOOL)(ret)
        except BaseException as ex:
            if isinstance(ex, (PublicException, ApplicationError)):
                raise
            if is_retryable(ex):
                _report_retryable_exception(ex)
                raise ex
            raise _AutoRpcNonRetriableError(type(ex), str(ex)) from ex

    # pylint: disable=protected-access
    activity_def = activity._Definition.from_callable(func)
    assert activity_def
    activity_name = activity_def.name
    if activity_def.is_async:
        return activity.defn(name=activity_name)(_wrap_handler_async)
    return activity.defn(name=activity_name)(_wrap_handler_sync)


# TODO: Add the correct type (see pyright warning)
def _export_rpc_method_activities(
        service_impl: ServiceType,  # type: ignore
) -> List[Callable]:
    """
    Scan the service implementation and exports a list of @rpc_method methods.
    """
    methods = inspect.getmembers(service_impl, _auto_rpc_instance_member)
    rpc_methods = []
    for _, method in methods:
        if isinstance(method, (RpcFuncAsync, RpcFuncSync)):
            rpc_methods.append(
                autorpc_handler_wrapper(
                    # Enable auto heartbeating for RPC handler execution so
                    # that it can receive cancellation signal (e.g., after
                    # timeout) and interrupt the exceution.
                    auto_heartbeater(method._handler_func)))
    return rpc_methods


class _ArgsFlattener:
    """
    Private helper class that matches function invocation arguments with
    declared parameters in the target function and produces a flattened list
    of arguments, that will be forwarded to the remote execution.

    Context: python function parameters are declared as:

    def func(pos1, pos2, pos3, /, pos_or_kwd, *-or-*args, kwd1, kwd2, **kwargs)
        * positional-only: pos1, pos2, pos3
        * positional-or-keyword: pos_or_kwd,
        * var-positional: args
        * keyword-only: kwd1, kwd2,
        * var-keyword: kwargs

    To keep it simple, we don't support keyword-only or var-keyword; only
    positional args are supported, or exception will be thrown.
    """
    def __init__(self, method) -> None:
        self._spec = inspect.getfullargspec(method)
        if self._spec.args[0] != 'self':
            raise RuntimeError(
                f'expect self as first arg, got: {self._spec.args[0]}')
        self._args = self._spec.args[1:]  # Drop the first 'self' arg
        # We pre-fill None for parameter without default value in the defaults
        # list, to make it the same length as self._args
        defaults = list(self._spec.defaults or [])
        assert len(self._args) >= len(defaults)
        num_args_no_default = len(self._args) - len(defaults)
        self._defaults = [None] * num_args_no_default + defaults

    def flatten_args(self, *args, **kwargs) -> list:
        flattened = list(args)
        if len(flattened) <= len(self._args):
            for i in range(len(flattened), len(self._args)):
                arg_name = self._args[i]
                if arg_name in kwargs:
                    flattened.append(kwargs.pop(arg_name))
                else:
                    flattened.append(self._defaults[i])
            # we have finished all positional args, and we don't support
            # more kwargs
            assert not kwargs
        else:
            # supplied args are longer, must be vararg
            assert self._spec.varargs is not None
        return flattened


def _task_queue_name(prefix: Optional[str], service_name: str):
    task_queue = f'{service_name}-task-queue'
    if prefix:
        task_queue = f'{prefix}-{task_queue}'
    return task_queue


def _rpc_workflow_name(service_name: str):
    return f'RpcWorkflow_{service_name}'


class RpcFuncAsyncToSync(Generic[P, T]):
    def __init__(
        self,
        handler_func: Callable[P, Awaitable[T]],
    ) -> None:
        self._handler_func = handler_func

    @property
    def sync(self) -> Callable[P, T]:
        return AsyncToSync(self._handler_func)  # type: ignore

    async def coroutine(self, *__args: P.args, **__kwargs: P.kwargs) -> T:
        return await self._handler_func(*__args, **__kwargs)

    def __call__(self, *__args: P.args, **__kwargs: P.kwargs) -> T:
        return self.sync(*__args, **__kwargs)


# This magic class serves as client stub to make RPC calls to 'ServiceType'.
# This is achieved via:
# 1. Scan the original 'ServiceType' class and find all public RPC methods;
# 2. For each rpc method, synthesize a proxy function of the same name via
# setattr(self, name, proxy_fn)
# 3. When caller invokes the proxy function, it executes a temporalio workflow
#    with the full RPC method name (ServiceName.method_name) and a list of
#    flattened positional args.
# 4. When target function is a coroutine (async def), we also synthesize an
#    async proxy func/coroutine.
# 5. We provide a generic create_client() helper function that makes the
#    client's side code type-safe.
class _ServiceClient(Generic[ServiceType]):
    def __init__(
        self,
        service_cls: Type[ServiceType],
        client: TemporalClient,
        config: AutoRpcClientConfig,
    ):
        self._service_name = service_cls.__name__
        self._workflow_name = _rpc_workflow_name(self._service_name)
        self._task_queue = _task_queue_name(config.server_name_prefix,
                                            self._service_name)
        self._client = client
        self._config = config
        self._num_calls = 0
        self._method_arg_flattener: Dict[str, _ArgsFlattener] = {}
        self._debug_tasks: Set[asyncio.Task] = set()
        # For testing purpose, record the last executed wf id per rpc method.
        self._wf_id_per_method: Dict[str, str] = {}

        for name, member in inspect.getmembers(service_cls,
                                               _auto_rpc_class_member):
            if isinstance(member, RpcMethodSync):
                self._method_arg_flattener[name] = _ArgsFlattener(
                    member._handler_func)
                setattr(self, name,
                        RpcFuncAsyncToSync(self._proxy_call_async(name)))

            if isinstance(member, RpcMethodAsync):
                self._method_arg_flattener[name] = _ArgsFlattener(
                    member._handler_func)
                setattr(
                    self,
                    name,
                    # Skip blocking-detection because we know the client side
                    # invocation is simply executing a workflow.
                    RpcFuncAsync(self._proxy_call_async(name),
                                 exclude_from_block_detection=True))

    def _get_workflow_id(self, method_name: str) -> str:
        wf_uuid = uuid.uuid4().hex[:8]
        wf_id = (f'{self._service_name}.{method_name}-{self._num_calls}-'
                 f'{wf_uuid}')
        self._wf_id_per_method[method_name] = wf_id
        return wf_id

    # Automatically adjust AutoRPC workflow execution timeout and retry
    # policy if the RPC call is made from a Temporal activity execution
    # which is much less latency sensitive (e.g., training or bp job).
    def _get_timeout_config(self, method_name: str) -> AutoRpcTimeoutConfig:
        raw_config = self._config.method_timeout_config.get(
            method_name, self._config.timeout_config)
        config = copy.deepcopy(raw_config)
        config.retry_policy = config.retry_policy or copy.deepcopy(
            DEFAULT_RETRY_POLICY)

        # Make sure non retriable errors include "_AutoRpcNonRetriableError"
        non_retriable_errors = config.retry_policy.non_retryable_error_types
        if not non_retriable_errors:
            non_retriable_errors = [_AutoRpcNonRetriableError.__name__]
        elif _AutoRpcNonRetriableError.__name__ not in non_retriable_errors:
            non_retriable_errors = list(non_retriable_errors) + [
                _AutoRpcNonRetriableError.__name__
            ]
        config.retry_policy.non_retryable_error_types = non_retriable_errors

        # Adjust timeout and retry policy if remaining time budget is known.
        time_budget = get_activity_remaining_time_budget()
        single_attempt_timeout = config.attempt_timeout or config.rpc_timeout
        if time_budget and time_budget > config.rpc_timeout:  # default is 30s
            config.rpc_timeout = time_budget
            # This RPC call is considered as "latency insensitive" if budget is
            # more than a minute. This provides us an opportunity to make more
            # retries with higher backoff interval to be more resilient to
            # transient failure.
            if time_budget > timedelta(minutes=1):
                config.retry_policy.maximum_attempts = 10
                config.retry_policy.maximum_interval = time_budget * 0.2

        if time_budget and time_budget > single_attempt_timeout:
            # If time budget is sufficient, try to increase timeout of
            # each single attempt to time_budget but not exceeding 1min cap.
            config.attempt_timeout = max(
                min(time_budget, timedelta(minutes=1)), single_attempt_timeout)

        return config

    async def _execute_workflow(self, method_name: str, *args) -> Any:
        r"""Execute a Temporal workflow. In case of Timeout Expired errors
        from the temporal client, this method re-creates the client and
        tries the workflow again after 120seconds of delay and retries for
        10 times before raising an RpcError.
        """
        retry_count = 0
        max_retries = 10
        # Retry delay starts with 10 seconds with a max of 120seconds
        # TODO configure this if needed.
        delay = 10
        max_delay = 120
        error_message = None
        temporal_error = None

        # Handle serialization of args in a threadpool to avoid blocking
        # the workflow worker event loop, which is necessary for large payload.
        payloads = await sync_to_async(_AUTO_RPC_DATA_CONVERTER.to_payloads,
                                       _AUTO_RPC_SERDE_THREADPOOL)(args)
        while retry_count < max_retries:
            try:
                ret_payload = await self._execute_workflow_without_retry(
                    method_name, retry_count > 0, *payloads)
                return await sync_to_async(
                    _AUTO_RPC_DATA_CONVERTER.from_payload,
                    _AUTO_RPC_SERDE_THREADPOOL)(ret_payload)
            except TemporalRPCError as error:
                # error is scoped only to the except block. assign it to the
                # temporal_error so that it can be referenced outside.
                temporal_error = error
                error_message = temporal_error.message
                if (temporal_error.status == RPCStatusCode.DEADLINE_EXCEEDED
                        or temporal_error.status == RPCStatusCode.CANCELLED):
                    client_log.debug(f"Timeout error: {temporal_error}")
                    client_log.debug(f"Retrying in {delay} seconds ..." +
                                     f"Attempt {retry_count+1} of " +
                                     f"{max_retries}")
                    retry_count += 1
                    # Wait before retrying as network errors can
                    # take time to resolve.
                    await asyncio.sleep(delay)
                    # Have an exponential backoff for the delay with a maximum
                    # delay of `max_delay`
                    delay = min(delay * 2, max_delay)
                else:
                    break
        # Raise an RpcError when the retries are exhausted.
        assert temporal_error is not None
        raise RpcError(
            self._service_name,
            method_name,
            error_type=type(temporal_error).__name__,
            message=error_message,
            stack_trace=None,
            non_retryable=False) from temporal_error

    async def _execute_workflow_without_retry(self, method_name: str,
                                              re_initialize: bool, *payloads:
                                              Payload) -> Payload:
        client_log.debug('Proxy call %s.%s', self._service_name, method_name)
        client = await self._client.get(re_initialize=re_initialize)
        timeout_config = self._get_timeout_config(method_name)

        # Note: This must be after the await self._client.get() call to prevent
        # a race condition:
        self._num_calls += 1
        workflow_id = self._get_workflow_id(method_name)

        rpc_method_name = f'{self._service_name}.{method_name}'
        try:
            return await client.execute_workflow(
                self._workflow_name,
                args=[rpc_method_name, timeout_config, *payloads],
                task_queue=self._task_queue,
                id=workflow_id,
                search_attributes={
                    'rpc_service': [self._service_name],
                    'rpc_method': [method_name],
                },
                execution_timeout=timeout_config.rpc_timeout,
                run_timeout=(timeout_config.attempt_timeout
                             or timeout_config.rpc_timeout),
                retry_policy=timeout_config.retry_policy,
                task_timeout=DEFAULT_TASK_TIMEOUT,
            )
        except TemporalError as temporal_error:
            workflow_url = self._client.format_workflow_web_url(workflow_id)
            client_log.exception(
                'AutoRPC call to %s.%s failed, workflow url: %s',
                self._service_name, method_name, workflow_url)
            if isinstance(temporal_error, WorkflowFailureError):
                if not isinstance(temporal_error.cause, ApplicationError):
                    self._start_debug_in_background(workflow_id)
                wf_failure_to_rpc_error(self._service_name, method_name,
                                        temporal_error)
            error_message = None
            if isinstance(temporal_error, TemporalRPCError):
                client_log.error(
                    "gRPC error when talking to Temporal service, "
                    "status code=%s, message=%s", temporal_error.status.name,
                    temporal_error.message)
                # raise temporal error in case of timeout expired
                if (temporal_error.status == RPCStatusCode.DEADLINE_EXCEEDED
                        or temporal_error.status == RPCStatusCode.CANCELLED):
                    raise temporal_error
                error_message = temporal_error.message

            raise RpcError(self._service_name, method_name,
                           error_type=type(temporal_error).__name__,
                           message=error_message, stack_trace=None,
                           non_retryable=False) from temporal_error

    def _start_debug_in_background(self, workflow_id: str):
        debug_task = asyncio.create_task(self._debug_failed_call(workflow_id))
        self._debug_tasks.add(debug_task)
        debug_task.add_done_callback(self._debug_tasks.discard)

    async def _debug_failed_call(self, workflow_id: str):
        client = await self._client.get()
        debug_start_time = time.perf_counter()
        try:
            worker_info = await client.execute_workflow(
                'WorkerInfoWorkflow', task_queue=self._task_queue,
                id=f'debug_{workflow_id}',
                execution_timeout=timedelta(seconds=10.0),
                run_timeout=timedelta(seconds=10.0),
                rpc_timeout=timedelta(seconds=1.0),
                retry_policy=NO_RETRY_POLICY)
            fetch_info_latency = time.perf_counter() - debug_start_time
            client_log.warning(
                f'Took {fetch_info_latency:.3f} sec to fetch '
                'Worker info fetched for failed workflow %s:\n%s', workflow_id,
                worker_info)
        except BaseException:
            fetch_info_latency = time.perf_counter() - debug_start_time
            client_log.exception(
                f'Failed to fetch worker info after {fetch_info_latency:.3f} '
                f'seconds for failed workflow {workflow_id}, likely because '
                'the worker is either unavailable or unresponsive.')

    def _proxy_call_sync(self, method_name: str):
        def wrapper(*args, **kwargs):
            args_flatterner = self._method_arg_flattener[method_name]
            flat_args = args_flatterner.flatten_args(*args, **kwargs)
            return AsyncToSync(self._execute_workflow)(method_name, *flat_args)

        return wrapper

    def _proxy_call_async(self, method_name: str):
        async def _autorpc_wrapper(*args, **kwargs):
            args_flatterner = self._method_arg_flattener[method_name]
            flat_args = args_flatterner.flatten_args(*args, **kwargs)
            return await self._execute_workflow(method_name, *flat_args)

        return tracing.trace(
            tracing.TraceOptions(
                span_name=f'{self._service_name}.{method_name}'))(
                    _autorpc_wrapper)


# Create a type-safe RPC client stub for the 'ServiceType'.
def create_client(
    cls: Type[ServiceType],
    config: AutoRpcClientConfig,
) -> ServiceType:

    # Set up temporal client.
    temporal_client = TemporalClient(config.temporal_config)
    return _ServiceClient(cls, temporal_client, config)  # type: ignore


class AutoRpcClassDisqualified(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


# Decorator to mark a "Service" class to automatically generate a "get_client"
# method.
def auto_rpc_service(cls: ServiceType) -> ServiceType:
    has_rpc_method = False
    for _, member in inspect.getmembers(
            cls, lambda m: isinstance(m, (RpcMethodAsync, RpcMethodSync)) or
            inspect.isfunction(m)):
        rpc_method_wrapper = None
        if isinstance(member, (RpcMethodSync, RpcMethodAsync)):
            rpc_method_wrapper = member
            # pylint: disable=protected-access
            member = rpc_method_wrapper._handler_func

        if inspect.isfunction(member) and _qualify_as_rpc_method(member):
            if rpc_method_wrapper is None:
                raise AutoRpcFunctionDisqualified(
                    member,
                    'RPC handler method must be decorated with @rpc_handler')
            if isinstance(member, RpcMethodAsync):
                # Automatically heartbeat for async activities:
                activity.defn(name=member.__qualname__)(
                    auto_heartbeater(member))
            else:
                activity.defn(name=member.__qualname__)(member)
            setattr(member, _RPC_METHOD_ATTR, True)
            has_rpc_method = True

    if not has_rpc_method:
        raise AutoRpcClassDisqualified(
            f'{cls} is marked as @auto_rpc_service without qualifying method.')

    return cls


@kumo_dataclass
class AutoRpcServerWorkerConfig:
    temporal_config: DeployEnvConfig
    # This is required if any RPC method is synchronous func.
    max_executor_threads: Optional[int] = None
    # A service may be configured to deploy at multiple "clusters", with each
    # cluster serving different clients or purposes. In that case, set the
    # server_name_prefix to distinguish.
    server_name_prefix: Optional[str] = None

    @staticmethod
    def parse_from_env(
        temporal_config: DeployEnvConfig,
        *,
        max_executor_threads: Optional[int] = None,
    ) -> 'AutoRpcServerWorkerConfig':
        return AutoRpcServerWorkerConfig(
            temporal_config,
            max_executor_threads=max_executor_threads,
            server_name_prefix=parse_auto_rpc_server_name_prefix_from_env(),
        )

    def to_worker_env(self, service_name: str) -> WorkerEnv:
        return WorkerEnv(
            f'{service_name}-worker',
            WorkerOptions(
                max_worker_threads=self.max_executor_threads,
                task_queue=_task_queue_name(self.server_name_prefix,
                                            service_name)),
            self.temporal_config)


# For testing use, build the server worker without running.
# TODO: Add the correct type (see pyright warning)
async def build_server(
        service_impl: ServiceType,  # type: ignore
        worker_env: WorkerEnv) -> Worker:
    service_name = service_impl.__class__.__name__
    activities = _export_rpc_method_activities(service_impl)
    if len(activities) == 0:
        raise RuntimeError(f'Cannot find activities from {service_name}')

    sync_activities = [
        fn for fn in activities if not inspect.iscoroutinefunction(fn)
    ]
    if sync_activities:
        if not worker_env.options.max_worker_threads:
            fn_names = ', '.join([fn.__name__ for fn in sync_activities])
            raise RuntimeError(
                'Expect server config to specify max_executor_threads '
                f'because [{fn_names}] functions are not coroutine.')

    # Here is a bit of magic: we dynamically copy the generic workflow type
    # "_RpcMethodWorkflow" into a new workflow type "RpcWorkflow_<SerivceName>"
    # which improves visibility in Temporal web UI as well as tracing.
    rpc_workflow_name = _rpc_workflow_name(service_name)
    rpc_workflow_cls = type(rpc_workflow_name, _RpcMethodWorkflow.__bases__,
                            dict(_RpcMethodWorkflow.__dict__))
    wf_def = workflow._Definition.must_from_class(_RpcMethodWorkflow)
    wf_def = dataclasses.replace(wf_def, name=rpc_workflow_name,
                                 cls=rpc_workflow_cls)
    setattr(rpc_workflow_cls, '__temporal_workflow_definition', wf_def)

    worker = await worker_env.build_worker(
        activities=activities,
        workflows=[rpc_workflow_cls],
        worker_name_suffix=f'-{service_name}-worker',
    )

    server_log.info('Starting AutoRPC server %s with config:\n%s\n',
                    service_name, worker.config())
    return worker


async def create_server_from_config(
    service_impl: ServiceType,  # type: ignore
    config: AutoRpcServerWorkerConfig) -> Tuple[WorkerEnv, Worker]:
    worker_env = config.to_worker_env(service_impl.__class__.__name__)
    worker = await build_server(service_impl, worker_env)
    return (worker_env, worker)


def get_activity_remaining_time_budget() -> Optional[timedelta]:
    if not activity.in_activity():
        return None

    info = activity.info()
    if info.schedule_to_close_timeout:
        deadline = info.scheduled_time + info.schedule_to_close_timeout
    elif info.start_to_close_timeout:
        deadline = info.started_time + info.start_to_close_timeout
    else:
        return None

    curr_time = datetime.now(deadline.tzinfo)
    if deadline > curr_time:
        time_budget = deadline - curr_time
        return time_budget
    return None
