# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Startup-order regression tests for :mod:`pool.controller_main`."""
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import pool.controller_main as controller_main


def test_startup_cleanup_completes_before_canary(monkeypatch, tmp_path):
    """A failing canary must not prevent leaked-worker startup cleanup."""
    events: list[str] = []
    controller = SimpleNamespace()

    async def create_controller(config, *, start_maintenance_loop):
        del config
        assert start_maintenance_loop is False
        events.append('startup-reconcile')
        return controller

    async def run_canary(config, canary_controller):
        del config
        assert canary_controller is controller
        events.append('canary')
        raise RuntimeError('canary could not schedule')

    monkeypatch.setattr(controller_main, 'init_logging_and_tracing',
                        lambda _: None)
    monkeypatch.setattr(controller_main, 'start_http_server',
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(controller_main, 'deploy_env', lambda: object())
    monkeypatch.setattr(controller_main, 'create_pool_controller',
                        create_controller)
    monkeypatch.setattr(controller_main, 'run_configured_init_canary',
                        run_canary)
    ready_path = tmp_path / 'ready'
    ready_path.touch()
    monkeypatch.setattr(controller_main, 'READY_PATH', ready_path)

    with pytest.raises(RuntimeError, match='could not schedule'):
        asyncio.run(controller_main.main())

    assert events == ['startup-reconcile', 'canary']
    assert not ready_path.exists()


def test_server_uses_autorpc_environment_configuration(monkeypatch, tmp_path):
    """The controller server must honor the client-side AutoRPC prefix."""
    deploy_config = object()
    controller = SimpleNamespace(_start_maintenance_loop=lambda: None)
    server_config = object()
    worker_env = MagicMock()

    async def create_controller(config, *, start_maintenance_loop):
        assert config is deploy_config
        assert start_maintenance_loop is False
        return controller

    ready_path = tmp_path / 'ready'

    async def run_worker():
        assert ready_path.exists()

    async def create_server(service, config):
        assert service is controller
        assert config is server_config
        return worker_env, SimpleNamespace(run=run_worker)

    async def run_canary(config, _canary_controller):
        assert config is deploy_config

    monkeypatch.setattr(controller_main, 'init_logging_and_tracing',
                        lambda _: None)
    monkeypatch.setattr(controller_main, 'start_http_server',
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(controller_main, 'deploy_env', lambda: deploy_config)
    monkeypatch.setattr(controller_main, 'create_pool_controller',
                        create_controller)
    monkeypatch.setattr(controller_main, 'run_configured_init_canary',
                        run_canary)
    monkeypatch.setattr(controller_main, 'create_server_from_config',
                        create_server)
    monkeypatch.setattr(controller_main, 'install_loop_guards', lambda: None)
    parse_from_env = MagicMock(return_value=server_config)
    monkeypatch.setattr(controller_main.AutoRpcServerWorkerConfig,
                        'parse_from_env', parse_from_env)
    monkeypatch.setattr(controller_main, 'READY_PATH', ready_path)

    asyncio.run(controller_main.main())

    parse_from_env.assert_called_once_with(deploy_config)
    worker_env.shutdown.assert_called_once_with()
    assert not ready_path.exists()
