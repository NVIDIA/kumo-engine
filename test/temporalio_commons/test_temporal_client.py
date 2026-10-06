# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, Mock, PropertyMock, call, patch

import pytest
import temporalio.api.namespace.v1.message_pb2 as nsv1
import temporalio.api.workflowservice.v1 as wsv1
from temporalio.api.common.v1 import Payload
from temporalio.service import RPCError, RPCStatusCode

from service_framework.deploy_env_config import DeployEnvConfig
from temporalio_commons import TemporalClient
from temporalio_commons.compression_codec import (
    BINARY_COMPRESSION_ENCODING,
    BINARY_OFFLOADED_ENCODING,
    CompressionCodec,
)
from temporalio_commons.temporal_client import (
    _NEW_NAMESPACE_WAIT_SEC,
    _REGISTER_NAMESPACE_POLL_INTERVAL_SEC,
    _TemporalServiceClientManager,
    _tmp_payloads_dir,
)


def test_tmp_payload_offload_requires_explicit_lifecycle_readiness(monkeypatch):
    makedirs = Mock()
    monkeypatch.setattr('temporalio_commons.temporal_client.fsutil.makedirs',
                        makedirs)
    config = DeployEnvConfig.for_test()
    storage = dataclasses.replace(
        config.storage,
        data_plane_storage_root='gs://dataplane/root',
        lifecycle_ready=False,
    )
    config = dataclasses.replace(config, storage=storage)
    assert _tmp_payloads_dir(config) is None
    makedirs.assert_not_called()

    config = dataclasses.replace(
        config,
        storage=dataclasses.replace(storage, lifecycle_ready=True),
    )
    assert _tmp_payloads_dir(config) == 'gs://dataplane/root/tmp_payloads'
    makedirs.assert_called_once_with('gs://dataplane/root/tmp_payloads',
                                     exist_ok=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('lifecycle_ready', [False, True])
async def test_lifecycle_readiness_controls_codec_offload_round_trip(
        monkeypatch, tmp_path: pathlib.Path, lifecycle_ready: bool):
    monkeypatch.setenv('CODEC_COMPRESS_SIZE_THRESHOLD', '1')
    monkeypatch.setenv('CODEC_OFFLOAD_SIZE_THRESHOLD', '1')
    loop = asyncio.get_running_loop()

    def run_immediately(_executor, func, *args):
        result = loop.create_future()
        try:
            result.set_result(func(*args))
        except Exception as error:
            result.set_exception(error)
        return result

    # The codec's executor behavior is owned by its existing tests. Keep this
    # lifecycle integration deterministic and focused on storage selection.
    monkeypatch.setattr(loop, 'run_in_executor', run_immediately)
    config = DeployEnvConfig.for_test()
    config = dataclasses.replace(
        config,
        storage=dataclasses.replace(
            config.storage,
            data_plane_storage_root=tmp_path.as_posix(),
            lifecycle_ready=lifecycle_ready,
        ),
    )
    payload = Payload(metadata={'encoding': b'binary'}, data=b'x' * 2048)

    codec = CompressionCodec(_tmp_payloads_dir(config))
    encoded = (await codec.encode([payload]))[0]

    if lifecycle_ready:
        assert encoded.metadata['encoding'] == BINARY_OFFLOADED_ENCODING
        assert encoded.metadata['offload_path'].decode().startswith(
            f'{tmp_path.as_posix()}/tmp_payloads/payload_')
    else:
        assert encoded.metadata['encoding'] == BINARY_COMPRESSION_ENCODING
        assert not (tmp_path / 'tmp_payloads').exists()
    assert await codec.decode([encoded]) == [payload]


def test_temporal_server_address(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('TEMPORAL_SERVER', 'cloud')
    env_config = DeployEnvConfig.for_test()
    expected_server_address = 'devtest.put0h.tmprl.cloud:7233'
    assert env_config.temporal_server_address == expected_server_address

    monkeypatch.setenv('DEPLOY_STAGE', 'prod')
    env_config_prod = DeployEnvConfig.parse_from_env()
    expected_server_address = 'prod.put0h.tmprl.cloud:7233'
    assert env_config_prod.temporal_server_address == expected_server_address

    monkeypatch.setenv('TEMPORAL_SERVER', 'localhost')
    monkeypatch.delenv('DEPLOY_STAGE', raising=False)
    env_config = DeployEnvConfig.for_test()
    assert env_config.temporal_server_address.startswith('127.0.0.1:')


@pytest.mark.asyncio
async def test_result_uses_exact_run_id() -> None:
    temporal_client = TemporalClient(DeployEnvConfig.for_test())
    handle = Mock()
    handle.result = AsyncMock(return_value='done')
    temporal_client.get_workflow_handle = AsyncMock(  # type: ignore[method-assign]
        return_value=handle)

    assert await temporal_client.result(
        'child-workflow',
        run_id='completed-run',
        disable_backoff=True,
    ) == 'done'
    temporal_client.get_workflow_handle.assert_awaited_once_with(
        'child-workflow', run_id='completed-run')


async def mock_describe_namespace(
        namespace: str,
        not_found: bool = False) -> wsv1.DescribeNamespaceResponse:
    if not_found:
        from temporalio.api.common.v1 import GrpcStatus

        raise RPCError(status=RPCStatusCode.NOT_FOUND, message='failed',
                       raw_grpc_status=GrpcStatus(code=5).SerializeToString())
    return wsv1.DescribeNamespaceResponse(namespace_info=nsv1.NamespaceInfo(
        name=namespace))


async def mock_register_namespace():
    return wsv1.RegisterNamespaceResponse()


async def mock_get_service(service_mock: Mock):
    return service_mock


def _mock_service_client(name: str) -> Mock:
    service_client = Mock(name=name)
    service_client._bridge_client = object()
    return service_client


@pytest.mark.asyncio
async def test_service_client_manager_reuses_within_one_loop() -> None:
    manager = _TemporalServiceClientManager()
    config = DeployEnvConfig.for_test()
    service_client = _mock_service_client('same-loop')
    connect_calls = 0

    async def connect(*_args, **_kwargs):
        nonlocal connect_calls
        connect_calls += 1
        await asyncio.sleep(0)
        return service_client

    manager._connect = connect  # type: ignore[method-assign]
    clients = await asyncio.gather(
        manager.get_service_client(config),
        manager.get_service_client(config),
    )

    assert clients == [service_client, service_client]
    assert connect_calls == 1
    await manager.close()
    assert service_client._bridge_client is None


def test_service_client_manager_isolates_two_thread_event_loops() -> None:
    manager = _TemporalServiceClientManager()
    config = DeployEnvConfig.for_test()
    start_barrier = threading.Barrier(2)
    calls_lock = threading.Lock()
    connected: list[Mock] = []

    async def connect(*_args, **_kwargs):
        with calls_lock:
            service_client = _mock_service_client(
                f'loop-client-{len(connected)}')
            connected.append(service_client)
        await asyncio.sleep(0.01)
        return service_client

    manager._connect = connect  # type: ignore[method-assign]

    def run_loop() -> tuple[int, Mock, Mock]:
        async def use_manager() -> tuple[int, Mock, Mock]:
            start_barrier.wait(timeout=5)
            first, second = await asyncio.gather(
                manager.get_service_client(config),
                manager.get_service_client(config),
            )
            loop_id = id(asyncio.get_running_loop())
            await manager.close()
            return loop_id, first, second

        return asyncio.run(use_manager())

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: run_loop(), range(2)))

    assert results[0][0] != results[1][0]
    assert results[0][1] is results[0][2]
    assert results[1][1] is results[1][2]
    assert results[0][1] is not results[1][1]
    assert len(connected) == 2
    assert all(client._bridge_client is None for client in connected)


def test_service_client_manager_all_loop_cleanup_requires_quiescence() -> None:
    manager = _TemporalServiceClientManager()
    config = DeployEnvConfig.for_test()
    service_client = _mock_service_client('all-loop-cleanup')

    async def connect(*_args, **_kwargs):
        return service_client

    manager._connect = connect  # type: ignore[method-assign]
    loop = asyncio.new_event_loop()

    async def initialize() -> None:
        await manager.get_service_client(config)
        with pytest.raises(RuntimeError, match='every owning event loop'):
            manager.close_all_loop_states()

    try:
        loop.run_until_complete(initialize())
        manager.close_all_loop_states()
        assert service_client._bridge_client is None
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_service_client_manager_reinitialize_and_close() -> None:
    manager = _TemporalServiceClientManager()
    config = DeployEnvConfig.for_test()
    first = _mock_service_client('first')
    second = _mock_service_client('second')
    clients = iter((first, second))

    async def connect(*_args, **_kwargs):
        return next(clients)

    manager._connect = connect  # type: ignore[method-assign]
    assert await manager.get_service_client(config) is first
    assert await manager.get_service_client(config) is first
    assert await manager.get_service_client(config,
                                            re_initialize=True) is second
    assert first._bridge_client is None

    await manager.close()
    assert second._bridge_client is None


@pytest.mark.asyncio
async def test_service_client_manager_failed_reinitialize_preserves_old(
) -> None:
    manager = _TemporalServiceClientManager()
    config = DeployEnvConfig.for_test()
    first = _mock_service_client('first')
    calls = 0

    async def connect(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return first
        raise RuntimeError('replacement failed')

    manager._connect = connect  # type: ignore[method-assign]
    assert await manager.get_service_client(config) is first
    with pytest.raises(RuntimeError, match='replacement failed'):
        await manager.get_service_client(config, re_initialize=True)

    assert first._bridge_client is not None
    assert await manager.get_service_client(config) is first
    await manager.close()
    assert first._bridge_client is None


@pytest.mark.asyncio
async def test_service_client_manager_lock_wait_is_bounded() -> None:
    manager = _TemporalServiceClientManager()
    config = DeployEnvConfig.for_test()
    loop = asyncio.get_running_loop()
    state = manager._loop_state(loop, create=True)
    assert state is not None
    await state.lock.acquire()
    try:
        with (
                patch(
                    'temporalio_commons.temporal_client.'
                    '_TEMPORAL_CLIENT_ACQUISITION_TIMEOUT_SEC', 0.01),
                pytest.raises(
                    TimeoutError,
                    match=r'loop_id=.*initializer_locked=True',
                ),
        ):
            await manager.get_service_client(config)
    finally:
        state.lock.release()
        await manager.close()


def test_temporal_client_wrapper_isolates_two_thread_event_loops() -> None:
    manager = _TemporalServiceClientManager()
    temporal_client = TemporalClient(DeployEnvConfig.for_test())
    temporal_client._client_manager = manager
    temporal_client._create_client = (  # type: ignore[method-assign]
        lambda service: service)
    start_barrier = threading.Barrier(2)
    calls_lock = threading.Lock()
    connected: list[Mock] = []

    async def connect(*_args, **_kwargs):
        with calls_lock:
            service_client = _mock_service_client(
                f'wrapper-client-{len(connected)}')
            connected.append(service_client)
        await asyncio.sleep(0.01)
        return service_client

    manager._connect = connect  # type: ignore[method-assign]

    def run_loop() -> Mock:
        async def use_client() -> Mock:
            start_barrier.wait(timeout=5)
            client = await temporal_client.get()
            assert await temporal_client.get() is client
            await temporal_client.close()
            return client

        return asyncio.run(use_client())

    with ThreadPoolExecutor(max_workers=2) as executor:
        clients = list(executor.map(lambda _index: run_loop(), range(2)))

    assert clients[0] is not clients[1]
    assert len(connected) == 2
    assert all(client._bridge_client is None for client in connected)
    assert len(temporal_client._loop_clients) == 0


def test_temporal_client_never_uses_stale_legacy_client_alias() -> None:
    manager = _TemporalServiceClientManager()
    temporal_client = TemporalClient(DeployEnvConfig.for_test())
    temporal_client._client_manager = manager
    stale_client = _mock_service_client('stale-alias')
    new_client = _mock_service_client('current-loop')
    temporal_client._client = stale_client
    temporal_client._create_client = (  # type: ignore[method-assign]
        lambda service: service)

    async def connect(*_args, **_kwargs):
        return new_client

    manager._connect = connect  # type: ignore[method-assign]

    async def get_and_close() -> Mock:
        client = await temporal_client.get()
        await temporal_client.close()
        return client

    assert asyncio.run(get_and_close()) is new_client
    assert new_client._bridge_client is None
    assert stale_client._bridge_client is not None


@patch('asyncio.sleep')
@pytest.mark.asyncio
async def test_temporal_existing_namespace(sleep):
    temporal_config = DeployEnvConfig.for_test()
    test_namespace = temporal_config.temporal_namespace

    temporal_client = TemporalClient(temporal_config)
    mock_client = Mock()
    mock_workflow_service = Mock()
    mock_workflow_service_property = PropertyMock(
        return_value=mock_workflow_service)
    # pylint: disable=protected-access
    temporal_client._set_client_for_loop(asyncio.get_running_loop(),
                                         mock_client)
    type(mock_client).workflow_service = mock_workflow_service_property

    # Get an existing namespace
    mock_workflow_service.describe_namespace.return_value = \
        mock_describe_namespace(test_namespace)
    await temporal_client.create_namespace()
    mock_workflow_service.describe_namespace.assert_called_once_with(
        wsv1.DescribeNamespaceRequest(namespace=test_namespace))


@patch('asyncio.sleep')
@pytest.mark.asyncio
async def test_temporal_new_namespace(sleep):
    temporal_config = DeployEnvConfig.for_test()
    test_namespace = temporal_config.temporal_namespace

    temporal_client = TemporalClient(temporal_config)
    mock_client = Mock()
    mock_workflow_service = Mock()
    mock_workflow_service_property = PropertyMock(
        return_value=mock_workflow_service)
    # pylint: disable=protected-access
    temporal_client._set_client_for_loop(asyncio.get_running_loop(),
                                         mock_client)
    type(mock_client).workflow_service = mock_workflow_service_property

    mock_workflow_service.describe_namespace.side_effect = [
        mock_describe_namespace(test_namespace, not_found=True),
        mock_describe_namespace(test_namespace, not_found=False),
        mock_describe_namespace(test_namespace, not_found=False),
    ]
    mock_workflow_service.register_namespace.return_value = \
        mock_register_namespace()

    await temporal_client.create_namespace()

    mock_workflow_service.describe_namespace.assert_called()
    assert mock_workflow_service.describe_namespace.call_count == 3
    mock_workflow_service.register_namespace.assert_called_once()
    assert sleep.call_count == 2
    sleep.assert_has_calls([
        call(_REGISTER_NAMESPACE_POLL_INTERVAL_SEC),
        call(_NEW_NAMESPACE_WAIT_SEC)
    ], any_order=False)
