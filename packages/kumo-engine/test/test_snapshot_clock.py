# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from typing import Any, get_args

import pytest
from kumoapi.typing import TimestampPrecision

from kumo.engine._scratch import fingerprint
from kumo.engine.graph import Graph

CLOCKS = get_args(TimestampPrecision)


@pytest.fixture
def ingests(tmp_path, monkeypatch):
    import kumo_core.diskgraph.ingest as ingest
    import kumo_core.diskgraph.snapshot as snapshots

    import kumo.engine._scratch as scratch

    monkeypatch.setattr(scratch, '_root', tmp_path)
    monkeypatch.setattr(Graph, 'id', 'graph-key')
    monkeypatch.setattr(Graph, '_graph_config', {'tables': []})
    monkeypatch.setattr(Graph, 'validate', lambda self: None)
    monkeypatch.setattr(snapshots, 'check_diskgraph_output', lambda path: None)
    monkeypatch.setattr(snapshots.DiskgraphIngestResult, 'for_index',
                        lambda path: SimpleNamespace(to_json=lambda: '{}'))
    specs = []

    def execute(graph_config, spec):
        specs.append(spec)
        return SimpleNamespace(index_path=spec.output_dir)

    monkeypatch.setattr(ingest, 'execute_diskgraph_ingest', execute)
    return specs


def _graph():
    class Source:
        def prepare_ingest(self, connector, table_name, destination, **_):
            return SimpleNamespace(credentials=None, version='v1',
                                   source={'path': destination})

    table: Any = SimpleNamespace(
        table_name='events',
        definition=SimpleNamespace(cols=[], source_table=None),
        connector=SimpleNamespace(source_backend=Source(),
                                  validate_scratch_dir=lambda path: None))
    return Graph(tables=[table])


@pytest.mark.parametrize('clock', CLOCKS)
def test_snapshot_ingests_in_the_requested_clock(ingests, clock):
    _graph().snapshot(timestamp_precision=clock)
    assert [spec.timestamp_precision for spec in ingests] == [clock]


def test_seconds_snapshot_keeps_the_existing_cache_key(ingests):
    snapshot = _graph().snapshot()
    data_version = fingerprint(
        json.dumps({'events': 'v1'}, sort_keys=True).encode())
    assert snapshot.id == f'graph-key@{data_version[:16]}'


def test_each_clock_has_its_own_snapshot(ingests):
    ids = {_graph().snapshot(timestamp_precision=clock).id for clock in CLOCKS}
    assert len(ids) == len(CLOCKS)


def test_unknown_clock_is_rejected():
    with pytest.raises(ValueError, match='timestamp_precision'):
        _graph().snapshot(timestamp_precision='months')
