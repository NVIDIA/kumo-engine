# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``pool.client``."""

import pytest

from pool.client import get_controller_client
from pool.service import PoolController
from service_framework.deploy_env_config import DeployEnvConfig
from service_framework.deploy_stage import DeployStage
from temporalio_commons.auto_rpc import (
    AutoRpcServerWorkerConfig,
    _task_queue_name,
)
from temporalio_commons.auto_rpc_config import AutoRpcClientConfig


def test_get_controller_client_uses_autorpc():
    from unittest.mock import MagicMock, patch
    config = MagicMock()
    sentinel = object()
    with patch('pool.client.create_client',
               return_value=sentinel) as mock_create:
        with patch('pool.client.AutoRpcClientConfig.parse_from_env',
                   return_value=MagicMock()):
            result = get_controller_client(config)
    assert result is sentinel
    assert mock_create.call_count == 1


def _pool_controller_queues_for_prefix(monkeypatch: pytest.MonkeyPatch,
                                       prefix: str) -> tuple[str, str]:
    """Resolve the effective client/server queues without contacting Temporal."""
    config = DeployEnvConfig(
        deploy_mode=DeployStage.TEST,
        instance_id='test-instance',
    )
    monkeypatch.setenv('KUMO_AUTORPC_SERVER_NAME_PREFIX', prefix)
    client_config = AutoRpcClientConfig.parse_from_env(PoolController, config)
    server_config = AutoRpcServerWorkerConfig.parse_from_env(config)
    server_worker_env = server_config.to_worker_env(PoolController.__name__)
    try:
        client_queue = config.task_queue(
            _task_queue_name(client_config.server_name_prefix,
                             PoolController.__name__))
        return client_queue, server_worker_env.task_queue
    finally:
        server_worker_env.shutdown()


def test_scoped_pool_controller_client_and_server_share_queue(monkeypatch):
    client_queue, server_queue = _pool_controller_queues_for_prefix(
        monkeypatch, 'train-a')

    assert client_queue == 'train-a-PoolController-task-queue-test-instance-test'
    assert server_queue == client_queue


def test_scoped_pool_controller_queues_are_distinct_per_child(monkeypatch):
    first_client, first_server = _pool_controller_queues_for_prefix(
        monkeypatch, 'train-a')
    second_client, second_server = _pool_controller_queues_for_prefix(
        monkeypatch, 'train-b')

    assert first_client == first_server
    assert second_client == second_server
    assert first_client != second_client
