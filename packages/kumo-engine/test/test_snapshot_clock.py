# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast, get_args

import pytest
from kumo_core.sources.identity import SourceVersion
from kumoapi.typing import TimestampPrecision

from kumo.engine._scratch import fingerprint
from kumo.engine.graph import Graph, GraphSnapshot
from kumo.engine.trainer._spec import (
    SPEC_FILE,
    SPEC_VERSION,
    read_training_spec,
)
from kumo.engine.trainer.trainer import Trainer

CLOCKS = get_args(TimestampPrecision)


@pytest.fixture
def ingests(tmp_path, monkeypatch, completed_diskgraph_ingest):
    import kumo_core.diskgraph.ingest as ingest

    import kumo.engine._scratch as scratch

    monkeypatch.setattr(scratch, '_root', tmp_path)
    monkeypatch.setattr(Graph, 'id', 'graph-key')
    monkeypatch.setattr(Graph, '_graph_config', {'tables': []})
    monkeypatch.setattr(Graph, 'validate', lambda self: None)
    specs = []

    def execute(graph_config, spec):
        specs.append(spec)
        return completed_diskgraph_ingest(graph_config, spec)

    monkeypatch.setattr(ingest, 'execute_diskgraph_ingest', execute)
    return specs


def _graph():
    class Source:
        def prepare_ingest(self, connector, table_name, destination, **_):
            return SimpleNamespace(credentials=None,
                                   version=SourceVersion('v1', 'exact'),
                                   source={'path': destination})

    table: Any = SimpleNamespace(
        table_name='events',
        definition=SimpleNamespace(cols=[], source_table=None),
        connector=SimpleNamespace(source_backend=Source(),
                                  validate_scratch_dir=lambda path: None))
    return Graph(tables=[table])


@pytest.mark.parametrize('clock', CLOCKS)
def test_snapshot_ingests_in_the_requested_clock(ingests, clock):
    snapshot = _graph().snapshot(timestamp_precision=clock)
    assert [spec.timestamp_precision for spec in ingests] == [clock]
    metadata = json.loads((snapshot.index_path / 'index_metadata.json').read_text())
    assert snapshot.index_path.name == metadata['index_build_id']
    assert snapshot.index_path.parent.name == 'indexes'
    assert metadata['timestamp_unit'] == clock
    marker = json.loads((snapshot.index_path / '_SUCCESS').read_text())
    assert marker['index_path'] == str(snapshot.index_path)
    assert not Path(ingests[0].output_dir).exists()


def test_seconds_snapshot_keeps_the_existing_cache_key(ingests):
    snapshot = _graph().snapshot()
    data_version = fingerprint(
        json.dumps({'events': 'v1'}, sort_keys=True).encode())
    assert snapshot.id == f'graph-key@{data_version[:16]}'


def test_each_clock_has_its_own_snapshot(ingests):
    graph = _graph()
    snapshots = [graph.snapshot(timestamp_precision=clock) for clock in CLOCKS]
    assert len({snapshot.id for snapshot in snapshots}) == len(CLOCKS)
    assert len({snapshot.index_path for snapshot in snapshots}) == len(CLOCKS)
    for clock, previous in zip(CLOCKS, snapshots):
        reused = graph.snapshot(timestamp_precision=clock)
        assert reused.id == previous.id
        assert reused.index_path == previous.index_path
    assert len(ingests) == len(CLOCKS)


def test_unknown_clock_is_rejected():
    with pytest.raises(ValueError, match='timestamp_precision'):
        _graph().snapshot(timestamp_precision='months')


@pytest.mark.parametrize('recorded, read', [({}, 'seconds'), (
    {'timestamp_precision': 'milliseconds'}, 'milliseconds')])
def test_a_bundle_records_its_clock(tmp_path, recorded, read):
    (tmp_path / SPEC_FILE).write_text(json.dumps({
        'spec_version': SPEC_VERSION, 'predictive_query': 'q', 'model_plan': {},
        'snapshot_id': 'graph@v1', 'training_table_generation': {}, **recorded}))
    assert read_training_spec(tmp_path).timestamp_precision == read


def test_predict_refuses_a_snapshot_in_another_clock(tmp_path):
    (tmp_path / 'index_metadata.json').write_text(
        json.dumps({'timestamp_unit': 'seconds'}))
    trainer = Trainer(cast(Any, SimpleNamespace()))
    trainer._training_result = cast(Any, SimpleNamespace())
    trainer._timestamp_precision = 'milliseconds'
    with pytest.raises(ValueError, match='trained on a milliseconds snapshot'):
        trainer.predict(cast(Any, None),
                        snapshot=GraphSnapshot('graph@v1', tmp_path, {}))
