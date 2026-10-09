# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep
from typing import Dict

import pytest
from kumo_core.types.schema import BaseColumnConfig
from pytest import MonkeyPatch

import async_utils.instrument
from async_utils.caching import reuse_inflight_future
from async_utils.instrument import measure_blocking_code
from async_utils.sync import FutureDedupeOption, blocking_func, blocking_method
from rest.config.data_backend import ConnectorSpec
from rest.deps.authentication import RequestContext, UserAuthorization
from statedb.config.common import BatchPredictionJobID, ConnectorID


@measure_blocking_code
async def my_async_func(sync_time: float, async_time: float) -> float:
    sleep(sync_time)
    await asyncio.sleep(async_time)
    return sync_time + async_time


@pytest.mark.asyncio
async def test_async_timer(caplog: pytest.LogCaptureFixture,
                           monkeypatch: MonkeyPatch):
    # Disable stress test mode.
    monkeypatch.setenv('DEPLOY_STAGE', 'dev')
    monkeypatch.setenv('STRESS_TEST_MODE', '0')
    # Temporarily reduce the threshold for testing.
    monkeypatch.setattr(
        async_utils.instrument,
        'BLOCKING_ALERT_THRESHOLD_SEC',
        0.5,
    )

    with caplog.at_level(logging.WARNING):
        assert pytest.approx(0.11) == await my_async_func(
            sync_time=0.01, async_time=0.1)
    assert len(caplog.text) == 0

    with caplog.at_level(logging.WARNING):
        assert pytest.approx(0.3) == await my_async_func(
            sync_time=0.2, async_time=0.1)
    assert 'Async func my_async_func has executed for' in caplog.text

    with caplog.at_level(logging.CRITICAL):
        assert pytest.approx(0.7) == await my_async_func(
            sync_time=0.6, async_time=0.1)
    assert 'Async func my_async_func has executed for' in caplog.text

    # Enable stress test mode.
    monkeypatch.setenv('STRESS_TEST_MODE', '1')
    with pytest.raises(Exception, match='FATAL blocking detected'):
        await my_async_func(sync_time=5, async_time=0.1)


@pytest.mark.asyncio
async def test_async_timer_nested(caplog: pytest.LogCaptureFixture,
                                  monkeypatch: MonkeyPatch):
    async def _child(*sleep_times: float):
        for sleep_time in sleep_times:
            sleep(sleep_time)  # "blocking"
            await asyncio.sleep(sleep_time)  # "non-blocking"

    @measure_blocking_code
    async def _parent(child, *sleep_times: float):
        await child(*sleep_times)

    # Disable stress test mode.
    monkeypatch.setenv('DEPLOY_STAGE', 'dev')
    monkeypatch.setenv('STRESS_TEST_MODE', '0')

    with caplog.at_level(logging.WARNING):
        await _parent(_child, 0.1)
    assert len(caplog.text) == 0

    with caplog.at_level(logging.WARNING):
        await _parent(_child, 0.1, 0.1, 0.3)
    assert len(caplog.records) == 1
    assert '_parent has executed for' in caplog.text
    assert 'awaited 2 futures' in caplog.text

    # Now apply decorator to the child.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        await _parent(measure_blocking_code(_child), 0.3)
    assert len(caplog.records) == 1
    # Now we execpt the warning to come from the child, instead of parent
    assert '_child has executed for' in caplog.text


@pytest.mark.asyncio
async def test_async_timer_handled_timeout_does_not_log_cancellation(caplog):
    @measure_blocking_code
    async def _poll() -> str | None:
        await asyncio.sleep(0)
        try:
            await asyncio.wait_for(asyncio.sleep(1), timeout=0.01)
        except asyncio.TimeoutError:
            return 'continued'

    with caplog.at_level(logging.WARNING):
        assert await _poll() == 'continued'
    assert 'Received asyncio cancellation' not in caplog.text


@pytest.mark.asyncio
async def test_async_timer_real_cancellation_logs_and_propagates(caplog):
    started = asyncio.Event()

    @measure_blocking_code
    async def _wait_for_cancel() -> None:
        await asyncio.sleep(0)
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(_wait_for_cancel())
    await started.wait()
    with caplog.at_level(logging.WARNING):
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert 'Received asyncio cancellation' in caplog.text
    assert '_wait_for_cancel' in caplog.text


num_calls: int = 0


@reuse_inflight_future(param_keys=['param_a', 'param_b'])
async def dummy_async_func(param_a: int, param_b: str,
                           seconds_to_sleep: float = 1.0) -> str:
    global num_calls
    num_calls += 1
    print(f'Sleeping for {seconds_to_sleep:.2f} seconds...')
    sleep(seconds_to_sleep)
    print('Done sleeping.')
    return f'{param_a}--{param_b}'


@measure_blocking_code
async def dummy_create_bp_job(args: ConnectorSpec, cols: Dict[BaseColumnConfig,
                                                              str],
                              ctx: RequestContext) -> BatchPredictionJobID:
    @blocking_func()
    def _dummy_create_bp_sync():
        sleep(1.0)
        for col in cols.keys():
            assert col.name
        return BatchPredictionJobID.make_random_id()

    return await _dummy_create_bp_sync.coroutine()


@reuse_inflight_future()
async def async_func_raise(param: str):
    await asyncio.sleep(1.0)
    raise RuntimeError(f'dummy async exception: {param}-{uuid.uuid4().hex}')


@pytest.mark.asyncio
async def test_concurrent_task():
    global num_calls
    num_calls = 0
    task1 = asyncio.ensure_future(dummy_async_func(1, 'a', 1.0))
    task2 = asyncio.ensure_future(dummy_async_func(1, 'a', 2.0))
    ret1, ret2 = await asyncio.gather(task1, task2)
    assert ret1 == '1--a'
    assert ret2 == '1--a'
    assert num_calls == 1

    create_bp_args = ConnectorSpec(ConnectorID('connector_' + '1' * 32),
                                   connector_config=None)
    create_bp_task1 = asyncio.ensure_future(
        dummy_create_bp_job(
            create_bp_args, {BaseColumnConfig('colA'): 'abc'},
            RequestContext(UserAuthorization.disabled(), 'traceid1')))
    create_bp_task2 = asyncio.ensure_future(
        dummy_create_bp_job(
            create_bp_args, {BaseColumnConfig('colA'): 'abc'},
            RequestContext(UserAuthorization.disabled(), 'traceid2')))
    create_bp_task3 = asyncio.ensure_future(
        dummy_create_bp_job(
            ConnectorSpec(
                ConnectorID('connector_' + '2' * 32),
                connector_config=None,
            ), {BaseColumnConfig('colB'): 'abc'},
            RequestContext(UserAuthorization.disabled(), 'traceid1')))
    bp1, bp2, bp3 = await asyncio.gather(create_bp_task1, create_bp_task2,
                                         create_bp_task3)
    assert bp1 == bp2
    assert bp1 != bp3

    async_ex1 = asyncio.ensure_future(async_func_raise('call'))
    async_ex2 = asyncio.ensure_future(async_func_raise('call'))
    async_ex3 = asyncio.ensure_future(async_func_raise('call2'))
    ex1, ex2, ex3 = await asyncio.gather(async_ex1, async_ex2, async_ex3,
                                         return_exceptions=True)
    print(traceback.format_exception(ex1))
    print(traceback.format_exception(ex2))
    print(traceback.format_exception(ex3))
    assert ex1 == ex2
    assert ex1 != ex3


DUMMY_THREAD_POOL = ThreadPoolExecutor(thread_name_prefix='Dummy-Threadpool')


class DummyClass:
    def __init__(self, base: int):
        self.lock = Lock()
        self.base = base

    def __getstate__(self):
        return {'base': self.base}

    def _add_and_return(self, param_a: int) -> int:
        with self.lock:
            print('Sleeping for 1.0 seconds...')
            sleep(1.0)
            print('Done sleeping.')
            self.base += param_a
            return self.base

    @blocking_method(executor=DUMMY_THREAD_POOL,
                     fut_dedupe_option=FutureDedupeOption.BY_INSTANCE_ID)
    def add_and_return1(self, param_a: int) -> int:
        return self._add_and_return(param_a)

    @blocking_method(executor=DUMMY_THREAD_POOL,
                     fut_dedupe_option=FutureDedupeOption.BY_INSTANCE_STATE)
    def add_and_return2(self, param_a: int) -> int:
        return self._add_and_return(param_a)

    @blocking_method(executor=DUMMY_THREAD_POOL,
                     fut_dedupe_option=FutureDedupeOption.DISABLED)
    def add_and_return3(self, param_a: int) -> int:
        return self._add_and_return(param_a)


@pytest.mark.asyncio
async def test_blocking_method_dedupe_by_instance_id():
    obj1 = DummyClass(1)
    obj2 = DummyClass(2)
    obj3 = DummyClass(2)
    task1 = asyncio.ensure_future(obj1.add_and_return1.coroutine(1))
    task2 = asyncio.ensure_future(obj2.add_and_return1.coroutine(1))
    task3 = asyncio.ensure_future(obj3.add_and_return1.coroutine(1))
    ret1, ret2, ret3 = await asyncio.gather(task1, task2, task3)
    assert ret1 == 2
    assert ret2 == 3
    assert ret3 == 3
    assert obj1.base == ret1
    assert obj2.base == ret2
    assert obj3.base == ret3

    # Concurrent calls on same instance method should be deduped.
    task1 = asyncio.ensure_future(obj1.add_and_return1.coroutine(1))
    task2 = asyncio.ensure_future(obj1.add_and_return1.coroutine(1))
    task3 = asyncio.ensure_future(obj1.add_and_return1.coroutine(1))
    ret1, ret2, ret3 = await asyncio.gather(task1, task2, task3)
    assert ret1 == 3
    assert ret2 == 3
    assert ret3 == 3
    assert obj1.base == 3


@pytest.mark.asyncio
async def test_blocking_method_dedupe_by_instance_state():
    obj1 = DummyClass(1)
    obj2 = DummyClass(3)
    obj3 = DummyClass(3)  # obj2 & obj3 has identical state
    task1 = asyncio.ensure_future(obj1.add_and_return2.coroutine(1))
    task2 = asyncio.ensure_future(obj2.add_and_return2.coroutine(1))
    task3 = asyncio.ensure_future(obj3.add_and_return2.coroutine(1))
    ret1, ret2, ret3 = await asyncio.gather(task1, task2, task3)
    assert ret1 == 2
    assert ret2 == 4
    assert ret3 == 4
    assert obj1.base == 2
    # 1 invocation gets deduped.
    assert sorted([obj2.base, obj3.base]) == [3, 4]

    # Now obj1, obj2, obj3 are all different values: [2, 3, 4]
    task1 = asyncio.ensure_future(obj1.add_and_return2.coroutine(1))
    task2 = asyncio.ensure_future(obj2.add_and_return2.coroutine(1))
    task3 = asyncio.ensure_future(obj3.add_and_return2.coroutine(1))
    ret1, ret2, ret3 = await asyncio.gather(task1, task2, task3)
    assert sorted([ret1, ret2, ret3]) == [3, 4, 5]
    assert sorted([obj1.base, obj2.base, obj3.base]) == [3, 4, 5]


@pytest.mark.asyncio
async def test_blocking_method_dedupe_disabled():
    obj = DummyClass(1)
    task1 = asyncio.ensure_future(obj.add_and_return3.coroutine(1))
    task2 = asyncio.ensure_future(obj.add_and_return3.coroutine(1))
    task3 = asyncio.ensure_future(obj.add_and_return3.coroutine(1))
    ret1, ret2, ret3 = await asyncio.gather(task1, task2, task3)
    assert sorted([ret1, ret2, ret3]) == [2, 3, 4]
    assert obj.base == 4
