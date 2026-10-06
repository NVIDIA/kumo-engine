# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from test.rest.integ import test_e2e_public_api
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from statedb.config.common import ProjectID


@pytest.mark.asyncio
async def test_public_e2e_setup_provisions_v1_with_public_client(
    monkeypatch,
) -> None:
    public_client = SimpleNamespace(name='public')
    fixture = SimpleNamespace(tables=[],
                              graphs=[],
                              queries=[],
                              predictions=None)
    randomized = SimpleNamespace(tables=['table'],
                                 graphs=['graph'],
                                 queries=['query'],
                                 predictions=['prediction'])
    project_id = ProjectID(f'project_{"a" * 32}')
    connector = SimpleNamespace()
    duplicate_paths = {'source': 'source-duplicate'}
    provision = AsyncMock(return_value=(project_id, [connector]))
    duplicate = AsyncMock(return_value=duplicate_paths)
    bind = MagicMock()
    monkeypatch.setattr(test_e2e_public_api, 'randomize_fixture_ids_v2',
                        MagicMock(return_value=randomized))
    monkeypatch.setattr(test_e2e_public_api,
                        'provision_fixture_connectors', provision)
    monkeypatch.setattr(test_e2e_public_api, 'duplicate_fixture_sources',
                        duplicate)
    monkeypatch.setattr(test_e2e_public_api, 'bind_test_client_project', bind)

    fixture_setup = test_e2e_public_api.setup_connector.__wrapped__(
        public_client,
        fixture,
    )
    await anext(fixture_setup)
    try:
        provision.assert_awaited_once_with(public_client,
                                           fixture,
                                           api_prefix='/v1')
        bind.assert_called_once_with(public_client, project_id)
        duplicate.assert_awaited_once_with(connector, fixture)
        assert public_client.fixture_duplicate_source_paths == duplicate_paths
        assert fixture.tables == randomized.tables
        assert fixture.graphs == randomized.graphs
        assert fixture.queries == randomized.queries
        assert fixture.predictions == randomized.predictions
    finally:
        await fixture_setup.aclose()
