# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import get_args
from unittest import mock

import pytest
from kumoapi.data_snapshot import GraphSnapshotID
from kumoapi.typing import TimestampPrecision

from kumoai_studio.client.client import KumoClient
from kumoai_studio.graph import Graph

BASE_URL = 'https://kumo.ai'
SNAPSHOTS_URL = f'{BASE_URL}/v1/graphsnapshots'
CLOCKS = get_args(TimestampPrecision)


def _create_snapshot(requests_mock, echo, **kwargs):
    headers = {} if echo is None else {'X-Kumo-Timestamp-Precision': echo}
    requests_mock.post(SNAPSHOTS_URL, json={'id': 'graph@v1'}, headers=headers)
    return KumoClient(BASE_URL,
                      'api-key').graph_api.create_snapshot('graph', **kwargs)


@pytest.mark.parametrize('clock', CLOCKS)
def test_snapshot_in_a_clock_the_server_built(requests_mock, clock):
    _create_snapshot(requests_mock, clock, timestamp_precision=clock)
    sent = requests_mock.last_request.qs.get('timestamp_precision')
    assert sent == (None if clock == 'seconds' else [clock])


def test_server_without_clocks_still_builds_seconds(requests_mock):
    assert _create_snapshot(requests_mock, None) == 'graph@v1'


def test_server_without_clocks_cannot_build_other_clocks(requests_mock):
    with pytest.raises(RuntimeError, match='built a seconds one'):
        _create_snapshot(requests_mock, None,
                         timestamp_precision='milliseconds')


def test_server_building_a_different_clock_is_an_error(requests_mock):
    with pytest.raises(RuntimeError, match='built a seconds one'):
        _create_snapshot(requests_mock, 'seconds', timestamp_precision='hours')


def test_unknown_clock_is_rejected_before_any_request(requests_mock):
    with pytest.raises(ValueError, match='timestamp_precision'):
        _create_snapshot(requests_mock, 'months', timestamp_precision='months')
    assert not requests_mock.called


def _graph(monkeypatch, *snapshot_ids):
    import kumoai_studio.graph.graph as graph_module

    backend = mock.Mock()
    backend.graph.create_snapshot.side_effect = [
        GraphSnapshotID(snapshot_id) for snapshot_id in snapshot_ids
    ]
    backend.graph.get_snapshot.return_value = SimpleNamespace(table_ids={})
    monkeypatch.setattr(graph_module, 'cloud_backend', lambda: backend)
    graph = Graph(project_id='project-test')
    monkeypatch.setattr(graph, 'save', lambda: None)
    return graph, backend


def test_changing_the_clock_takes_a_new_snapshot(monkeypatch):
    graph, backend = _graph(monkeypatch, 'graph@seconds', 'graph@days')

    assert graph.snapshot(non_blocking=True) == 'graph@seconds'
    assert graph.snapshot(non_blocking=True) == 'graph@seconds'
    assert graph.snapshot(non_blocking=True,
                          timestamp_precision='days') == 'graph@days'
    clocks = [
        call.kwargs['timestamp_precision']
        for call in backend.graph.create_snapshot.call_args_list
    ]
    assert clocks == ['seconds', 'days']


def test_a_snapshot_without_a_clock_reuses_the_current_one(monkeypatch):
    graph, backend = _graph(monkeypatch, 'graph@milliseconds')

    graph.snapshot(non_blocking=True, timestamp_precision='milliseconds')

    assert graph.snapshot(non_blocking=True) == 'graph@milliseconds'
    assert backend.graph.create_snapshot.call_count == 1
