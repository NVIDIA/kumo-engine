# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from hashlib import md5
from types import SimpleNamespace
from typing import Any, cast, get_args
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from kumoapi.typing import TimestampPrecision

from io_utils.json import from_json_dict
from kumo.diskgraph.service import DiskgraphService
from rest.backends.data_backend.activities import \
    data_backend_activities as activities
from rest.config.data_backend import GraphDataVersionInfo
from rest.config.graph_v2 import GraphDataComputeInput
from rest.public.routers import data_snapshot
from rest.public.routers.data_snapshot import GraphSnapshotRouter

CLOCKS = get_args(TimestampPrecision)


def _snapshot_version(monkeypatch, clock: str) -> GraphDataVersionInfo:
    monkeypatch.setattr(activities.DataBackendActivities,
                        'save_graph_data_version', mock.AsyncMock())
    monkeypatch.setattr(
        activities, 'data_backend_cache',
        lambda: SimpleNamespace(graph_cache=SimpleNamespace(
            save_latest_graph_version=SimpleNamespace(
                coroutine=mock.AsyncMock()))))
    graph_input = GraphDataComputeInput(
        'graph', 'graph-key', cast(Any, {'table': None}), {}, {},
        cast(Any, None), table_aliases=cast(Any, {'table': 'events'}),
        pinned_data_versions=cast(Any, {'table': 'v1'}),
        timestamp_precision=clock)
    return asyncio.run(
        activities.DataBackendActivities._refresh_graph_data_version(
            graph_input, {}, None))


def test_seconds_snapshot_keeps_the_existing_identity(monkeypatch):
    version = _snapshot_version(monkeypatch, 'seconds')
    assert version.graph_data_version == md5(b'events=v1').hexdigest()


def test_each_clock_has_its_own_snapshot_identity(monkeypatch):
    versions = [_snapshot_version(monkeypatch, clock) for clock in CLOCKS]
    assert [version.timestamp_precision for version in versions] == list(CLOCKS)
    assert len({version.graph_data_version for version in versions}) == 6


def test_snapshots_recorded_before_clocks_are_seconds():
    version = from_json_dict(
        {
            'graph_data_version': 'v1',
            'table_data_versions': {}
        }, GraphDataVersionInfo)
    assert version.timestamp_precision == 'seconds'


@pytest.fixture
def snapshot_api(monkeypatch):
    driver = mock.AsyncMock()
    driver.create_graph_snapshot.return_value = 'graph@v1'
    recorded = SimpleNamespace(timestamp_precision='seconds')
    monkeypatch.setattr(data_snapshot, 'get_graph_snapshot_info',
                        mock.AsyncMock(return_value=(None, recorded)))
    monkeypatch.setattr(data_snapshot.GraphIDv2, 'from_public',
                        lambda graph_id: SimpleNamespace(fpid='graph'))
    app = FastAPI()
    app.include_router(
        GraphSnapshotRouter(lambda: None, driver, mock.Mock()).router)
    return SimpleNamespace(client=TestClient(app), driver=driver,
                           recorded=recorded)


def _create(snapshot_api, **params):
    return snapshot_api.client.post('/graphsnapshots', params={
        'graph_id': 'graph',
        'project_id': 'default',
        **params
    })


def test_request_without_a_clock_builds_seconds(snapshot_api):
    response = _create(snapshot_api)
    kwargs = snapshot_api.driver.create_graph_snapshot.await_args.kwargs
    assert kwargs['timestamp_precision'] == 'seconds'
    assert response.headers['X-Kumo-Timestamp-Precision'] == 'seconds'


@pytest.mark.parametrize('clock', CLOCKS)
def test_snapshot_builds_the_requested_clock(snapshot_api, clock):
    snapshot_api.recorded.timestamp_precision = clock
    response = _create(snapshot_api, timestamp_precision=clock)
    kwargs = snapshot_api.driver.create_graph_snapshot.await_args.kwargs
    assert kwargs['timestamp_precision'] == clock
    assert response.headers['X-Kumo-Timestamp-Precision'] == clock


def test_snapshot_echoes_the_recorded_clock_not_the_request(snapshot_api):
    response = _create(snapshot_api, timestamp_precision='hours')
    assert response.headers['X-Kumo-Timestamp-Precision'] == 'seconds'


def test_unknown_clock_is_rejected(snapshot_api):
    assert _create(snapshot_api,
                   timestamp_precision='months').status_code == 422
    snapshot_api.driver.create_graph_snapshot.assert_not_awaited()


def test_prediction_statistics_use_the_snapshot_clock(monkeypatch):
    import diskgraph

    binding = mock.Mock(return_value=('{}', '{}'))
    monkeypatch.setattr(diskgraph, 'compute_col_stats_for_parquet', binding)
    DiskgraphService(cast(Any, None)).compute_col_stats_for_parquet(
        ['predictions.parquet'], [], 'v1', timestamp_precision='hours')
    assert binding.call_args.kwargs['timestamp_unit'] == 'hours'
