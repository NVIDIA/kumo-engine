# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from datetime import timedelta
from threading import Event, Thread
from time import sleep
from typing import AsyncGenerator, List, Set, Tuple, cast
from unittest.mock import MagicMock

import pytest
from asgiref.sync import AsyncToSync
from temporalio.api.common.v1 import Payload
from temporalio.api.enums.v1 import EventType
from temporalio.common import RetryPolicy
from temporalio.exceptions import CancelledError as TemporalCancelledError
from temporalio.service import RPCError as TemporalRPCError
from temporalio.service import RPCStatusCode
from temporalio.worker import Worker

from io_utils import kumo_dataclass
from service_framework.deploy_env_config import DeployEnvConfig
from statedb.config.common import (
    PredictiveQueryID,
    TableIDv1,
    TableIDv2,
    to_table_id,
)
from temporalio_commons.auto_rpc import (
    NO_RETRY_POLICY,
    AutoRpcServerWorkerConfig,
    _ServiceClient,
    auto_rpc_service,
    create_client,
    create_server_from_config,
    rpc_handler,
)
from temporalio_commons.auto_rpc_config import (
    AutoRpcClientConfig,
    AutoRpcTimeoutConfig,
)
from temporalio_commons.auto_rpc_exception import RetriableError, RpcError
from temporalio_commons.temporal_client import TemporalClient
from temporalio_commons.worker_registry import WorkerEnv


@kumo_dataclass
class PydanticResult:
    ret: str


@auto_rpc_service
class MyService:
    # pylint: disable=disallowed-name
    def __init__(self, foo: str, bar: str) -> None:
        self._foo = foo
        self._bar = bar
        self._execution_count = 0
        self._exec_cancelled_event = Event()

    def _private_method(self):
        return f'{self._foo}-{self._bar}: '

    @rpc_handler
    async def bar(self, param: str) -> str:
        return self._private_method() + '-bar-' + param

    # pylint: disable=keyword-arg-before-vararg
    @rpc_handler
    async def multiparam(self, a: str, b: str = 'b', c: str = 'c',
                         *args) -> str:
        result = f'{a}-{b}-{c}'
        if args:
            result = result + '-' + ','.join(args)
        return result

    @rpc_handler
    async def set_param(self, ids: Set[str]) -> str:
        return ','.join(sorted(ids))

    @rpc_handler
    async def set_return(self, ids: str) -> Set[str]:
        return set(ids.split(','))

    @rpc_handler
    def foo_sync(self, param: str) -> str:
        return f'{param}-sync'

    @rpc_handler
    async def foo_async(self, param: str) -> PydanticResult:
        return PydanticResult(f'{param}-async')

    @rpc_handler
    def bar_sync(self, param: str) -> str:
        return f'bar-{param}'

    @rpc_handler
    async def bar_async(self, param: str) -> str:
        return f'bar-{param}'

    @rpc_handler
    async def raise_runtime_ex(self, param: str) -> str:
        self._execution_count += 1
        raise RuntimeError(f'failed: {param}')

    @rpc_handler
    async def raise_retriable_ex(self) -> str:
        self._execution_count += 1
        raise RetriableError()

    @rpc_handler
    async def raise_value_error(self) -> str:
        self._execution_count += 1
        raise ValueError('some value error')

    @rpc_handler
    def sleep_sync(self, seconds: float):
        self._exec_cancelled_event.clear()
        try:
            seconds_left = seconds
            while seconds_left > 0:
                sleep(min(1.0, seconds_left))
                seconds_left -= 1.0
        except (asyncio.CancelledError, TemporalCancelledError):
            self._exec_cancelled_event.set()

    @rpc_handler
    async def sleep_async(self, seconds: float):
        self._exec_cancelled_event.clear()
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            self._exec_cancelled_event.set()

    @rpc_handler
    async def sleep_async_blocking(self, seconds: float):
        sleep(seconds)  # blocking asyncio event loop

    @rpc_handler
    async def get_str_param_type(self, param: str) -> str:
        return f'{type(param)}'


@auto_rpc_service
class ForwardingService:
    """
    Another AutoRPC service that forwards rpc call to `MyService` below.
    This is for testing how timeout/retry policy is adjusted.
    """
    def __init__(self, my_service_client: MyService) -> None:
        self._my_service_client = my_service_client

    @rpc_handler
    async def foo_async(self, param: str) -> str:
        return (await self._my_service_client.foo_async(param)).ret

    @rpc_handler
    async def raise_non_retryable(self):
        return await self._my_service_client.raise_value_error()

    @rpc_handler
    async def raise_retryable(self):
        return await self._my_service_client.raise_runtime_ex('some error')


@pytest.fixture
def service_client(temporal_config: DeployEnvConfig) -> MyService:
    client_config = AutoRpcClientConfig(
        # No retry
        temporal_config,
        timeout_config=AutoRpcTimeoutConfig(retry_policy=RetryPolicy(
            maximum_attempts=1)))
    return create_client(MyService, client_config)


@asynccontextmanager
async def _start_service_worker(config: AutoRpcServerWorkerConfig,
                                *service_impl) -> AsyncGenerator[None, None]:
    workers: List[Tuple[WorkerEnv, Worker]] = [
        await create_server_from_config(s, config) for s in service_impl
    ]
    try:
        [asyncio.create_task(worker.run()) for _, worker in workers]
        yield
    finally:
        for worker_env, worker in workers:
            await worker.shutdown()
            worker_env.shutdown()


async def _wait_for_debug_tasks(service_client: MyService):
    service_client_impl = cast(_ServiceClient, service_client)
    debug_tasks = list(service_client_impl._debug_tasks)
    if debug_tasks:
        await asyncio.wait(debug_tasks)


def test_client_config_parsing(temporal_config: DeployEnvConfig,
                               monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('KUMO_AUTORPC_SERVER_NAME_PREFIX', raising=False)
    config = AutoRpcClientConfig.parse_from_env(MyService, temporal_config)
    assert len(config.method_timeout_config) == 0
    assert config.timeout_config.rpc_timeout.total_seconds() == 30
    assert config.server_name_prefix is None

    monkeypatch.setenv('RPC_TIMEOUT_DEFAULT', '60')
    config = AutoRpcClientConfig.parse_from_env(MyService, temporal_config)
    assert config.timeout_config.rpc_timeout.total_seconds() == 60

    monkeypatch.setenv('RPC_TIMEOUT_MYSERVICE', '100')
    config = AutoRpcClientConfig.parse_from_env(MyService, temporal_config)
    assert config.timeout_config.rpc_timeout.total_seconds() == 100

    monkeypatch.setenv('RPC_TIMEOUT_MYSERVICE__FOO_SYNC', '123')
    monkeypatch.setenv('RPC_TIMEOUT_MYSERVICE__FOO_ASYNC', '234')
    config = AutoRpcClientConfig.parse_from_env(MyService, temporal_config)
    assert config.timeout_config.rpc_timeout.total_seconds() == 100
    assert config.method_timeout_config['foo_sync'].rpc_timeout.total_seconds(
    ) == 123
    assert config.method_timeout_config['foo_async'].rpc_timeout.total_seconds(
    ) == 234

    monkeypatch.setenv('KUMO_AUTORPC_SERVER_NAME_PREFIX', 'phase1b-run')
    config = AutoRpcClientConfig.parse_from_env(MyService, temporal_config)
    assert config.server_name_prefix == 'phase1b-run'

    monkeypatch.setenv('KUMO_AUTORPC_SERVER_NAME_PREFIX', '   ')
    config = AutoRpcClientConfig.parse_from_env(MyService, temporal_config)
    assert config.server_name_prefix is None


def test_server_config_parsing(temporal_config: DeployEnvConfig,
                               monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('KUMO_AUTORPC_SERVER_NAME_PREFIX', raising=False)
    config = AutoRpcServerWorkerConfig.parse_from_env(temporal_config,
                                                      max_executor_threads=2)
    assert config.server_name_prefix is None
    assert config.max_executor_threads == 2

    monkeypatch.setenv('KUMO_AUTORPC_SERVER_NAME_PREFIX', 'phase1b-run')
    config = AutoRpcServerWorkerConfig.parse_from_env(temporal_config)
    assert config.server_name_prefix == 'phase1b-run'
    worker_env = config.to_worker_env(MyService.__name__)
    assert worker_env.options.task_queue == 'phase1b-run-MyService-task-queue'


@pytest.mark.asyncio
async def test_deadline_propagation(temporal_config: DeployEnvConfig,
                                    service_client: MyService):
    myservice = MyService('foo', 'bar')
    fwd_service = ForwardingService(service_client)
    server_config = AutoRpcServerWorkerConfig(temporal_config,
                                              max_executor_threads=4)
    fwd_client_config = AutoRpcClientConfig(
        temporal_config, AutoRpcTimeoutConfig(rpc_timeout=timedelta(minutes=5)))
    fwd_client = create_client(ForwardingService, fwd_client_config)
    rpc_call_params = f'testtesttest-{uuid.uuid4().hex}'
    async with _start_service_worker(server_config, myservice, fwd_service):
        result = await fwd_client.foo_async(rpc_call_params)
        assert result == (await myservice.foo_async(rpc_call_params)).ret
        assert result == f'{rpc_call_params}-async'

    assert isinstance(service_client, _ServiceClient)
    wf_id = service_client._wf_id_per_method['foo_async']

    temporal_client = TemporalClient(temporal_config)
    wf_handle = await temporal_client.get_workflow_handle(wf_id)
    client = await temporal_client.get()
    async for e in wf_handle.fetch_history_events():
        if e.event_type == EventType.EVENT_TYPE_WORKFLOW_EXECUTION_STARTED:
            attrs = e.workflow_execution_started_event_attributes
            in_payloads = attrs.input.payloads
            wf_inputs = await client.data_converter.decode(in_payloads)
            assert len(wf_inputs) == 3
            assert wf_inputs[0] == 'MyService.foo_async'
            assert isinstance(wf_inputs[2], Payload)
            assert wf_inputs[2].metadata['encoding'] == b'json/plain'
            assert in_payloads[2].metadata['encoding'] == b"binary/passthrough"
            assert in_payloads[2].metadata['original_encoding'] == b'json/plain'
            assert client.data_converter.payload_converter.from_payload(
                wf_inputs[2]) == rpc_call_params
            # Expect a single run/attempt's timeout to be increased from 30s
            # (default) to 1min (capped).
            run_timeout = attrs.workflow_run_timeout.ToTimedelta()
            assert run_timeout == timedelta(minutes=1)
            # Expect total execution timeout (including retries) to be
            # increased from 30s (default) to ~4min59s (remaining time budget).
            exec_timeout = attrs.workflow_execution_timeout.ToTimedelta()
            assert exec_timeout > timedelta(minutes=4)
            # Expect # of retries to be increased to 10 (capped).
            assert attrs.retry_policy.maximum_attempts == 10
            # Expect retry backoff interval to be increased to ~1min, i.e.
            # 1/5 of time budget.
            maximum_interval = attrs.retry_policy.maximum_interval
            assert maximum_interval.ToTimedelta() > timedelta(seconds=30)

        if e.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED:
            attrs = e.activity_task_scheduled_event_attributes
            timeout = attrs.schedule_to_close_timeout.ToTimedelta()
            # This should be the same as attempt/run timeout.
            assert timeout == timedelta(minutes=1)

        if e.event_type == EventType.EVENT_TYPE_WORKFLOW_EXECUTION_COMPLETED:
            attrs = e.workflow_execution_completed_event_attributes
            ret_payload = attrs.result.payloads[0]
            assert isinstance(ret_payload, Payload)
            assert ret_payload.metadata['encoding'] == b"binary/passthrough"
            assert ret_payload.metadata[
                'original_encoding'] == b'json/pydanticclass'
            original_payload = (await
                                client.data_converter.decode([ret_payload]))[0]
            assert isinstance(original_payload, Payload)
            assert original_payload.metadata[
                'encoding'] == b'json/pydanticclass'
            ret = client.data_converter.payload_converter.from_payload(
                original_payload)
            assert isinstance(ret, PydanticResult)
            assert ret == PydanticResult(result)


@pytest.mark.asyncio
async def test_auto_rpc_async(temporal_config: DeployEnvConfig,
                              service_client: MyService):
    myservice = MyService('foo', 'bar')
    server_config = AutoRpcServerWorkerConfig(temporal_config,
                                              max_executor_threads=4)

    async with _start_service_worker(server_config, myservice):
        print('Worker has started')
        result = await service_client.bar('test_bar_arg')
        assert result == 'foo-bar: -bar-test_bar_arg'

        result = await service_client.multiparam(a='a', b='b')
        assert result == 'a-b-c'
        result = await service_client.multiparam('a', b='b')
        assert result == 'a-b-c'
        result = await service_client.multiparam(b='a', a='b')
        assert result == 'b-a-c'
        result = await service_client.multiparam(c='a', b='b', a='c')
        assert result == 'c-b-a'
        result = await service_client.multiparam('a')
        assert result == 'a-b-c'
        result = await service_client.multiparam('a', 'b', 'c', 'd', 'e')
        assert result == 'a-b-c-d,e'
        result = await service_client.set_param(set(['a', 'b', 'cd']))
        assert result == 'a,b,cd'
        result = await service_client.set_return('e,d,c,b,a')
        assert result == set(['a', 'b', 'c', 'd', 'e'])

        # ========= Test decoding of call args to correct type =======
        assert await service_client.get_str_param_type(
            to_table_id('connector_' + '1' * 32 + '.my_table')
        ) == str(TableIDv1)
        assert await service_client.get_str_param_type(
            to_table_id('pquery_id.table_name-NA')) == str(TableIDv2)
        assert await service_client.get_str_param_type(
            PredictiveQueryID('pquery_id')) == str(PredictiveQueryID)

        # ========= Test @rpc_handler dual interfaces
        assert await service_client.bar_sync.coroutine('test') == 'bar-test'
        assert await service_client.bar_async('test') == 'bar-test'
        assert await service_client.bar_async.coroutine('test') == 'bar-test'

        # Example: run a sync function in corountine via executor
        def call_foo_sync():
            return service_client.foo_sync('foo')

        result = await asyncio.get_event_loop().run_in_executor(
            None, call_foo_sync)
        assert result == 'foo-sync'

        # Async rpc client cannot call the sync inteface!
        with pytest.raises(RuntimeError,
                           match='just await the async function directly'):
            service_client.foo_sync('test')
        with pytest.raises(RuntimeError,
                           match='just await the async function directly'):
            service_client.foo_sync.sync('test')
        with pytest.raises(RuntimeError,
                           match='just await the async function directly'):
            service_client.foo_async.sync('test')

        await _wait_for_debug_tasks(service_client)


@pytest.mark.asyncio
async def test_auto_rpc_retry(temporal_config: DeployEnvConfig):
    myservice = MyService('foo', 'bar')
    client_config = AutoRpcClientConfig(
        temporal_config,
        timeout_config=AutoRpcTimeoutConfig(retry_policy=RetryPolicy(
            initial_interval=timedelta(seconds=0.0),
            maximum_interval=timedelta(seconds=0.0),
            maximum_attempts=3,
        )))
    client = create_client(MyService, client_config)
    fwd_service = ForwardingService(client)
    server_config = AutoRpcServerWorkerConfig(temporal_config,
                                              max_executor_threads=4)
    fwd_client = create_client(ForwardingService, client_config)

    async with _start_service_worker(server_config, myservice, fwd_service):
        # ValueError will NOT be retried.
        myservice._execution_count = 0
        with pytest.raises(RpcError) as err:
            await client.raise_value_error()
        assert err.value.error_type == 'ValueError'
        assert myservice._execution_count == 1

        # RetriableError will be retried.
        myservice._execution_count = 0
        with pytest.raises(RpcError) as err:
            await client.raise_retriable_ex()
        assert err.value.error_type == 'RetriableError'
        assert myservice._execution_count == 3

        # RuntimeError will be retried.
        myservice._execution_count = 0
        with pytest.raises(RpcError) as err:
            await client.raise_runtime_ex('some error happened')
        assert err.value.error_type == 'RuntimeError'
        assert myservice._execution_count == 3

        # Test AutoRpc propagation/forwarding behavior
        # non-retryable
        myservice._execution_count = 0
        with pytest.raises(RpcError) as err:
            await fwd_client.raise_non_retryable()
        assert err.value.error_type == 'RpcError'
        assert myservice._execution_count == 1
        # retryable
        myservice._execution_count = 0
        with pytest.raises(RpcError) as err:
            await fwd_client.raise_retryable()
        assert err.value.error_type == 'RpcError'
        assert not err.value.non_retryable  # retryable
        # ForwardingService.raise_retryable() will be executed 3 times; each
        # time MyService.raise_runtime_ex() will be executed 3 times, so:
        assert myservice._execution_count == 3 * 3


def test_auto_rpc_sync(temporal_config: DeployEnvConfig,
                       service_client: MyService):
    myservice = MyService('foo', 'bar')
    server_config = AutoRpcServerWorkerConfig(temporal_config,
                                              max_executor_threads=4)

    # Run the server in a separate background thread.
    stop_event = Event()
    server_thread = Thread(target=_run_server_until_stop,
                           args=[myservice, server_config, stop_event])
    server_thread.start()

    # Sync code calls sync func
    assert service_client.foo_sync('foo') == 'foo-sync'
    # Sync code calls async func
    foo_async_ret = AsyncToSync(service_client.foo_async)('foo')
    assert isinstance(foo_async_ret, PydanticResult)
    assert foo_async_ret == PydanticResult('foo-async')

    # ========= Test @rpc_handler dual interfaces
    assert service_client.bar_sync('test') == 'bar-test'
    assert service_client.bar_sync.sync('test') == 'bar-test'
    assert service_client.bar_async.sync('test') == 'bar-test'

    # Test exception/failure scenario
    with pytest.raises(RpcError) as err:
        service_client.raise_runtime_ex.sync('test')
    assert err.value.error_type == 'RuntimeError'

    AsyncToSync(_wait_for_debug_tasks)(service_client)

    stop_event.set()
    server_thread.join()


def test_auto_rpc_sync_timeout(temporal_config: DeployEnvConfig):
    myservice = MyService('foo', 'bar')
    server_config = AutoRpcServerWorkerConfig(temporal_config,
                                              max_executor_threads=1)
    service_client = create_client(
        MyService,
        AutoRpcClientConfig(
            temporal_config,
            AutoRpcTimeoutConfig(rpc_timeout=timedelta(seconds=1.0),
                                 retry_policy=NO_RETRY_POLICY)))

    # Run the server in a separate background thread.
    stop_event = Event()
    server_thread = Thread(target=_run_server_until_stop,
                           args=[myservice, server_config, stop_event])
    server_thread.start()

    rpc_call_start = time.perf_counter()
    with pytest.raises(RpcError) as err:
        service_client.sleep_sync(2.0)
    rpc_call_latency = time.perf_counter() - rpc_call_start
    assert rpc_call_latency < 2.0  # deadline is 1.0 sec
    assert err.value.error_type == 'TimeoutError'

    # When timeout occurs, and execution is taking too long to complete, make
    # sure the sync execution receives cancellation signal.
    with pytest.raises(RpcError) as err:
        service_client.sleep_sync(3600.0)
    assert err.value.error_type == 'TimeoutError'
    assert myservice._exec_cancelled_event.wait(timeout=30)

    rpc_call_start = time.perf_counter()
    with pytest.raises(RpcError) as err:
        service_client.sleep_async_blocking.sync(2.0)
    rpc_call_latency = time.perf_counter() - rpc_call_start
    assert rpc_call_latency < 2.0  # deadline is 1.0 sec
    assert err.value.error_type == 'TimeoutError'

    rpc_call_start = time.perf_counter()
    with pytest.raises(RpcError) as err:
        service_client.sleep_async.sync(2.0)
    rpc_call_latency = time.perf_counter() - rpc_call_start
    assert rpc_call_latency < 2.0  # deadline is 1.0 sec
    assert err.value.error_type == 'TimeoutError'

    # When timeout occurs, and execution is taking too long to complete, make
    # sure the async execution receives cancellation signal.
    with pytest.raises(RpcError) as err:
        service_client.sleep_async.sync(3600.0)
    assert err.value.error_type == 'TimeoutError'
    assert myservice._exec_cancelled_event.wait(timeout=30)

    AsyncToSync(_wait_for_debug_tasks)(service_client)

    stop_event.set()
    server_thread.join()


def _run_server_until_stop(service_impl: MyService,
                           config: AutoRpcServerWorkerConfig,
                           stop_event: Event):
    async def run_server_wait_stop():
        async with _start_service_worker(config, service_impl):
            await asyncio.get_event_loop().run_in_executor(
                None,  # use the default executor thread pool
                stop_event.wait)

    asyncio.run(run_server_wait_stop())


@pytest.mark.asyncio
async def test_auto_rpc_retry_timeout(temporal_config: DeployEnvConfig,
                                      service_client: MyService):
    myservice = MyService('foo', 'bar')
    server_config = AutoRpcServerWorkerConfig(temporal_config,
                                              max_executor_threads=4)

    # Define a function to raise a timeout error only on the first workflow
    def raise_timeout_error_once(*args, **kwargs):
        raise_timeout_error_once.called = \
            getattr(raise_timeout_error_once, 'called', 0) + 1
        if raise_timeout_error_once.called == 1:
            raise TemporalRPCError(message="Timeout expired",
                                   status=RPCStatusCode.CANCELLED,
                                   raw_grpc_status=bytes('test', 'utf-8'))
        else:
            return "Normal behavior on subsequent calls"

    def raise_uncaught_error(*args, **kwargs):
        raise_uncaught_error.called = \
            getattr(raise_uncaught_error, 'called', 0) + 1
        raise TemporalRPCError(message="Timeout expired",
                               status=RPCStatusCode.ABORTED,
                               raw_grpc_status=bytes('test', 'utf-8'))

    async with _start_service_worker(server_config, myservice):
        client = await service_client._client.get()  # type: ignore
        client.execute_workflow = MagicMock(
            side_effect=raise_timeout_error_once)
        result = await service_client.bar('test_bar_arg')
        assert result == 'foo-bar: -bar-test_bar_arg'
        assert raise_timeout_error_once.called == 1

        client = await service_client._client.get()  # type: ignore
        client.execute_workflow = MagicMock(side_effect=raise_uncaught_error)
        with pytest.raises(RpcError):
            await service_client.bar('test_bar_arg')
        assert raise_uncaught_error.called == 1
