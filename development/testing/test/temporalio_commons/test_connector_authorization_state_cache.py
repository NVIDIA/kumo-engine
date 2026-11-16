# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import gc
import multiprocessing as mp
import os
import threading
import weakref
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rest.backends.data_backend import \
    connector_authorization_client as auth_client
from rest.config.data_backend import ConnectorExecutionRef
from service_framework._raw_deploy_config import (
    PostgresMetadataStorageConfig,
    SqliteMetadataStorageConfig,
)
from service_framework.deploy_stage import DeployStage
from statedb.config.common import ConnectorID, ProjectID, UserID
from statedb.db import StateDB, StateServices
from statedb.rest.connector_authorization import (
    ConnectorAuthorizationResourceKind,
    ConnectorCapabilityAction,
)

_PROJECT_ID = ProjectID('project_' + '1' * 32)


@pytest.fixture
def setup_rest_test_session() -> None:
    """Control-plane state tests do not require the Temporal harness."""
    return None


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield
    asyncio.run(auth_client.close_cached_authorization_states())


@dataclass(frozen=True)
class _Principal:
    user_id: UserID = UserID('user-1')
    user_name: str = 'User One'
    kumo_group_ids: tuple[str, ...] = ('group-1', )
    external_membership_unavailable: bool = False
    is_admin: bool = False


def _sqlite_config():
    return SimpleNamespace(
        metadata_storage=SqliteMetadataStorageConfig(),
        deploy_mode=DeployStage.DEV,
        instance_id='instance',
        client_id='rest',
        temporal_namespace='namespace',
    )


def test_sqlite_authorization_state_is_not_shared_across_separate_event_loops(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(auth_client, 'get_db_storage_root',
                        lambda: str(tmp_path))
    config = _sqlite_config()

    original_open = auth_client.open_state_db_from_deploy_config
    open_count = 0

    async def counted_open(*args, **kwargs):
        nonlocal open_count
        open_count += 1
        return await original_open(*args, **kwargs)

    monkeypatch.setattr(auth_client, 'open_state_db_from_deploy_config',
                        counted_open)

    async def open_and_close():
        result = await auth_client._open_authorization_state(config)
        await auth_client.close_cached_authorization_state_for_current_loop()
        return result

    first = asyncio.run(open_and_close())
    second = asyncio.run(open_and_close())

    assert first[0] is not second[0]
    assert first[1] is not second[1]
    assert open_count == 2


@pytest.mark.asyncio
async def test_n_way_bind_singleflights_sqlite_state_within_one_loop(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(auth_client, 'get_db_storage_root',
                        lambda: str(tmp_path))
    config = _sqlite_config()
    sqlite_path = auth_client.default_sqlite_path(str(tmp_path))
    token_count = 8
    state_db = await StateDB.open_sqlite(sqlite_path)
    try:
        service = StateServices(state_db).connector_authorizations
        tokens = []
        for index in range(token_count):
            _, token = await service.issue(
                principal=_Principal(),
                auth_backend='native_disabled',
                resource_kind=ConnectorAuthorizationResourceKind.CONNECTOR,
                resource_id=str(ConnectorID(f'connector_{index:032x}')),
                project_id=str(_PROJECT_ID),
                request_digest='',
                allowed_actions=(
                    ConnectorCapabilityAction.GET_CONNECTOR_SAMPLE_ROWS, ),
                deployment_id='instance:rest',
                temporal_namespace='namespace',
            )
            tokens.append(token)
    finally:
        await state_db.close()

    original_open = auth_client.open_state_db_from_deploy_config
    open_count = 0

    async def counted_open(*args, **kwargs):
        nonlocal open_count
        open_count += 1
        await asyncio.sleep(0.02)
        return await original_open(*args, **kwargs)

    monkeypatch.setattr(auth_client, 'open_state_db_from_deploy_config',
                        counted_open)

    async def bind(index: int):
        input = SimpleNamespace(
            args=[
                ConnectorExecutionRef(
                    project_id=_PROJECT_ID,
                    connector_id=ConnectorID(f'connector_{index:032x}'),
                    authorization_token=tokens[index],
                )
            ],
            id=f'workflow-{index}',
            execution_timeout=None,
            run_timeout=None,
        )
        return await auth_client.bind_connector_authorizations_for_start(
            input, config)

    bound_states = await asyncio.gather(*(bind(index)
                                          for index in range(token_count)))

    assert open_count == 1
    assert len({id(result[0]) for result in bound_states}) == 1
    assert all(len(result[2]) == 1 for result in bound_states)

    for state, services, newly_bound in bound_states:
        await auth_client.revoke_bound_connector_authorizations(
            state, services, newly_bound)
    await auth_client.close_cached_authorization_state_for_current_loop()


def test_concurrent_separate_loops_use_independent_sqlite_state(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(auth_client, 'get_db_storage_root',
                        lambda: str(tmp_path))
    config = _sqlite_config()
    sqlite_path = auth_client.default_sqlite_path(str(tmp_path))
    token_count = 8

    async def issue_tokens() -> list[str]:
        state_db = await StateDB.open_sqlite(sqlite_path)
        try:
            service = StateServices(state_db).connector_authorizations
            tokens = []
            for index in range(token_count):
                _, token = await service.issue(
                    principal=_Principal(),
                    auth_backend='native_disabled',
                    resource_kind=(
                        ConnectorAuthorizationResourceKind.CONNECTOR),
                    resource_id=str(ConnectorID(f'connector_{index:032x}')),
                    project_id=str(_PROJECT_ID),
                    request_digest='',
                    allowed_actions=(
                        ConnectorCapabilityAction.GET_CONNECTOR_SAMPLE_ROWS, ),
                    deployment_id='instance:rest',
                    temporal_namespace='namespace',
                )
                tokens.append(token)
            return tokens
        finally:
            await state_db.close()

    tokens = asyncio.run(issue_tokens())
    original_open = auth_client.open_state_db_from_deploy_config
    open_count = 0
    count_lock = threading.Lock()

    async def counted_open(*args, **kwargs):
        nonlocal open_count
        with count_lock:
            open_count += 1
        await asyncio.sleep(0.02)
        return await original_open(*args, **kwargs)

    monkeypatch.setattr(auth_client, 'open_state_db_from_deploy_config',
                        counted_open)
    barrier = threading.Barrier(token_count)
    bound_states = []
    errors: list[BaseException] = []

    def bind(index: int) -> None:
        async def run():
            input = SimpleNamespace(
                args=[
                    ConnectorExecutionRef(
                        project_id=_PROJECT_ID,
                        connector_id=ConnectorID(f'connector_{index:032x}'),
                        authorization_token=tokens[index],
                    )
                ],
                id=f'workflow-{index}',
                execution_timeout=None,
                run_timeout=None,
            )
            result = await auth_client.bind_connector_authorizations_for_start(
                input, config)
            await auth_client.revoke_bound_connector_authorizations(*result)
            await auth_client \
                .close_cached_authorization_state_for_current_loop()
            return result

        barrier.wait()
        try:
            bound_states.append(asyncio.run(run()))
        except BaseException as error:
            errors.append(error)

    threads = [
        threading.Thread(target=bind, args=(index, ))
        for index in range(token_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive(), 'concurrent capability bind thread hung'

    assert errors == []
    assert open_count == token_count
    assert len(bound_states) == token_count
    assert len({id(result[0]) for result in bound_states}) == token_count
    assert all(len(result[2]) == 1 for result in bound_states)


@pytest.mark.asyncio
async def test_authorization_state_cache_is_discarded_after_fork(
    monkeypatch, ) -> None:
    loop = asyncio.get_running_loop()
    cached = auth_client._CachedAuthorizationState(
        state_db=SimpleNamespace(),
        services=SimpleNamespace(),
    )
    opening = auth_client.concurrent.futures.Future()
    with auth_client._AUTHORIZATION_STATE_LOCK:
        auth_client._AUTHORIZATION_STATE_CACHE[loop] = {'path': cached}
        auth_client._AUTHORIZATION_STATE_OPENING[loop] = {'path': opening}

    child_pid = os.getpid() + 1
    monkeypatch.setattr(auth_client.os, 'getpid', lambda: child_pid)
    with auth_client._AUTHORIZATION_STATE_LOCK:
        auth_client._discard_inherited_authorization_state_locked()

    assert auth_client._AUTHORIZATION_STATE_PID == child_pid
    assert not auth_client._AUTHORIZATION_STATE_CACHE
    assert not auth_client._AUTHORIZATION_STATE_OPENING


def test_authorization_state_cache_does_not_retain_closed_loops() -> None:
    loop = asyncio.new_event_loop()
    loop_ref = weakref.ref(loop)
    with auth_client._AUTHORIZATION_STATE_LOCK:
        auth_client._AUTHORIZATION_STATE_CACHE[loop] = {}
    loop.close()
    del loop
    gc.collect()

    assert loop_ref() is None
    assert not auth_client._AUTHORIZATION_STATE_CACHE


def test_authorization_state_fork_replaces_a_locked_parent_guard() -> None:
    if 'fork' not in mp.get_all_start_methods():
        pytest.skip('fork start method is required')
    context = mp.get_context('fork')
    receive, send = context.Pipe(duplex=False)
    parent_guard = auth_client._AUTHORIZATION_STATE_LOCK
    parent_guard.acquire()

    def inspect_child_state() -> None:
        try:
            with auth_client._AUTHORIZATION_STATE_LOCK:
                auth_client._discard_inherited_authorization_state_locked()
            send.send((
                'ok',
                auth_client._AUTHORIZATION_STATE_PID,
                bool(auth_client._AUTHORIZATION_STATE_CACHE),
                bool(auth_client._AUTHORIZATION_STATE_OPENING),
            ))
        except BaseException as error:
            send.send(('error', type(error).__name__, False, False))
        finally:
            send.close()

    process = context.Process(target=inspect_child_state)
    try:
        process.start()
        process.join(timeout=3)
        if process.is_alive():
            process.terminate()
            process.join(timeout=3)
            pytest.fail('forked authorization child inherited a locked guard')
    finally:
        parent_guard.release()
        send.close()

    assert process.exitcode == 0
    assert receive.poll(1)
    status, child_pid, has_cache, has_opening = receive.recv()
    receive.close()
    assert status == 'ok'
    assert child_pid == process.pid
    assert has_cache is False
    assert has_opening is False


@pytest.mark.asyncio
async def test_postgres_authorization_state_is_not_process_cached(
    monkeypatch, ) -> None:
    config = SimpleNamespace(
        metadata_storage=PostgresMetadataStorageConfig(
            dsn='postgresql://state-db'), )
    first = SimpleNamespace(close=AsyncMock())
    second = SimpleNamespace(close=AsyncMock())
    open_state = AsyncMock(side_effect=[first, second])
    monkeypatch.setattr(auth_client, 'open_state_db_from_deploy_config',
                        open_state)

    first_state = await auth_client._open_authorization_state(config)
    second_state = await auth_client._open_authorization_state(config)

    assert first_state[0] is first
    assert second_state[0] is second
    assert open_state.await_count == 2
