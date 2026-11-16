# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import create_autospec

import kumoapi.graph as graph_api
import pytest
from kumo_core.types.schema import FPID
from kumoapi.data_source import DEFAULT_PROJECT_ID

from rest.config.graph_v2 import GraphDefinition
from rest.deps.authentication import Authentication
from rest.driver.graph_driver import GraphDriver
from rest.public.routers.graph import GraphRouter
from statedb.config.common import GraphIDv2, ProjectID


@pytest.fixture
def auto_setup_kumo_backend() -> bool:
    return False


@pytest.fixture
def _rest_install_dg_col_stats_fallback(
):  # pyright: ignore[reportUnusedFunction]
    yield


@pytest.fixture(scope='session')
def setup_rest_test_session():
    yield


@pytest.fixture
def auto_cleanup_per_func():
    yield


@pytest.fixture(scope='session')
def force_local_cache_dir_for_public_tests():
    yield


@pytest.fixture(scope='session')
def temporal_workers_ready():
    yield True


def _router() -> tuple[GraphRouter, Any]:
    driver = cast(Any, create_autospec(GraphDriver, instance=True))
    driver.create_full_graph_v2.return_value = _graph_id()
    auth = cast(Authentication, SimpleNamespace())
    return GraphRouter(lambda: None, driver, auth), driver


def _graph_id() -> GraphIDv2:
    return GraphIDv2.public(FPID('b' * 32))


@pytest.mark.parametrize(
    ('request_kwargs', 'driver_kwargs'),
    (
        (
            dict(name_alias='sdk_smoke_graph', force_rename=True,
                 skip_validation=True),
            dict(alias='sdk_smoke_graph', force_rename=True,
                 skip_validation=True),
        ),
        ({}, dict(alias=None, force_rename=False, skip_validation=False)),
        (dict(name_alias=''),
         dict(alias=None, force_rename=False, skip_validation=False)),
    ),
    ids=('named', 'unnamed', 'empty-alias'),
)
@pytest.mark.asyncio
async def test_create_graph_forwards_current_driver_signature(
    request_kwargs,
    driver_kwargs,
) -> None:
    router, driver = _router()
    request = graph_api.GraphDefinition(tables={}, col_groups=[])

    result = await router.create_graph(
        request,
        project_id=DEFAULT_PROJECT_ID,
        **request_kwargs,
    )

    driver.create_full_graph_v2.assert_awaited_once_with(
        GraphDefinition(**dataclasses.asdict(request)),
        project_id=ProjectID(DEFAULT_PROJECT_ID),
        **driver_kwargs,
    )
    assert result.id == _graph_id()
