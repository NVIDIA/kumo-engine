# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from temporalio_commons import wait_server_ready


def _local_config():
    return SimpleNamespace(
        orchestration=object(),
        temporal_namespace='devtest',
        temporal_server_address='127.0.0.1:7233',
    )


@pytest.mark.asyncio
async def test_readiness_probe_closes_each_successful_client():
    clients = []

    def make_client(_config):
        client = SimpleNamespace(
            init=AsyncMock(),
            create_namespace=AsyncMock(),
            close=AsyncMock(),
        )
        clients.append(client)
        return client

    with patch.object(wait_server_ready, 'deploy_env', _local_config), \
            patch.object(wait_server_ready, 'TemporalClient', make_client):
        for _ in range(3):
            await wait_server_ready.main(timedelta(seconds=1))

    assert len(clients) == 3
    for client in clients:
        client.init.assert_awaited_once()
        client.create_namespace.assert_awaited_once()
        client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_readiness_probe_closes_client_on_timeout():
    client = SimpleNamespace(
        init=AsyncMock(),
        create_namespace=AsyncMock(),
        close=AsyncMock(),
    )
    with patch.object(wait_server_ready, 'deploy_env', _local_config), \
            patch.object(wait_server_ready, 'TemporalClient',
                         return_value=client):
        with pytest.raises(RuntimeError, match='Temporal is not ready'):
            await wait_server_ready.main(timedelta(seconds=0))

    client.close.assert_awaited_once()
