# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional cross-package qualification of core snapshots with the PyTorch loader."""

import hashlib
import json
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumoapi.graph import Edge
from kumoapi.typing import Stype

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir

api = pytest.importorskip('kumo.pytorch.api')


def _files(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob('*') if p.is_file()
    }


def test_core_snapshot_columnar_reuse_under_one_root(tmp_path, monkeypatch):
    data = tmp_path / 'sources'
    data.mkdir()
    pq.write_table(pa.table({
        'id': [-2, 7],
        'name': ['Alice', 'Bob']
    }), data / 'users.parquet')
    pq.write_table(
        pa.table({
            'id': [1, 2, 3, 4],
            'user': [-2, -2, -2, 7],
            'at': pa.array([10, 20, 30, 10], pa.timestamp('s')),
            'value': [2.0, 4.0, 100.0, None],
        }),
        data / 'events.parquet',
    )
    connector = kumo.FileConnector(root_dir=str(data))
    previous = get_scratch_dir()
    try:
        kumo.configure(scratch_dir=tmp_path / 'producer')
        graph = kumo.Graph(
            tables={
                'users':
                kumo.Table.from_source_table(connector['users.parquet'],
                                             primary_key='id'),
                'events':
                kumo.Table.from_source_table(connector['events.parquet'],
                                             primary_key='id',
                                             time_column='at').with_column(
                                                 'user', stype=Stype.ID),
            },
            edges=[Edge(src_table='events', fkey='user', dst_table='users')],
        )
        snapshot = graph.snapshot()
        assert not (snapshot.index_path / 'index_manifest.json').exists()
        build_id = json.loads(
            (snapshot.index_path /
             'index_metadata.json').read_text())['index_build_id']
        assert snapshot.index_path == tmp_path / 'producer' / 'indexes' / build_id
        anchor_path = tmp_path / 'anchors.parquet'
        pq.write_table(
            pa.table({
                'user': [-2, 7],
                'at': pa.array([20, 20], pa.timestamp('s'))
            }),
            anchor_path,
        )
        anchors = api.AnchorTable(api.Parquet(str(anchor_path)), 'user', 'at')

        def run(snapshot):
            before = _files(snapshot.index_path)
            engine = api.DiskGraphEngine.from_snapshot(snapshot, num_threads=1)
            assert engine.index == snapshot.index_path.resolve()
            relation = engine.schema.relations[0]
            contract = api.DataLoaderSpec(
                engine.schema,
                api.SampleSpec(
                    entity=api.EntitySpec('users'),
                    neighborhoods={
                        'history':
                        api.NeighborhoodSpec((api.GraphLayer((api.GraphSample(
                            relation.name,
                            fanout=2,
                            direction='reverse',
                            strategy='recent',
                        ), )), )),
                    },
                ),
                api.BatchSpec(
                    inputs={
                        'graph':
                        api.Input(
                            api.NeighborhoodSource('history'),
                            api.Graph(
                                node_fields={
                                    table.name:
                                    api.Fields(
                                        tuple(
                                            api.Encode(name, api.Raw())
                                            for name in table.columns))
                                    for table in engine.schema.tables
                                }),
                            None,
                        )
                    },
                    batching=api.Batching(max_samples=2),
                ),
            )
            with (
                    api.prepare(contract=contract, engine=engine) as plan,
                    plan.materialize(anchors) as task,
            ):
                task_root = tmp_path / 'producer' / 'tasks' / build_id
                assert any((path / 'index_metadata.json').is_file()
                           for path in task_root.iterdir()
                           if not path.name.startswith('scratch-'))
                batch = next(iter(task.batches(output_format='columnar')))
                out = batch.inputs['graph']
                events = out.nodes['events']
                assert set(events.ids.to_pylist()) == {1, 2, 4}
                assert set(
                    events.features['value'].to_pylist()) == {2.0, 4.0, None}
                assert relation.name in out.edges
                edge = out.edges[relation.name]
                pairs = {(out.nodes[edge.source_type].ids[left].as_py(),
                          out.nodes[edge.destination_type].ids[right].as_py())
                         for left, right in zip(
                             edge.index['source'].to_pylist(),
                             edge.index['destination'].to_pylist(), strict=True)
                         }
                expected_pairs = {(-2, 1), (-2, 2), (7, 4)}
                if edge.source_type == 'events':
                    expected_pairs = {(event, user)
                                      for user, event in expected_pairs}
                assert pairs == expected_pairs
                for edge in out.edges.values():
                    for left, right in zip(
                            edge.index['source'].to_pylist(),
                            edge.index['destination'].to_pylist(),
                            strict=True,
                    ):
                        assert (out.nodes[edge.source_type].owners[left] ==
                                out.nodes[edge.destination_type].owners[right])
                result = events.features.to_pylist()
            assert _files(snapshot.index_path) == before
            assert not list(task_root.iterdir())
            return result, engine._binding.identity

        expected, identity = run(snapshot)
        assert identity == build_id
        exports = kumo.FileConnector(root_dir=str(tmp_path / 'exports'))
        location = snapshot.export(kumo.SnapshotOutputConfig(exports, 'snapshot'))
        shutil.rmtree(data)
        loaded = kumo.GraphSnapshot.load_from_connector(exports, path=location)
        assert loaded.id == snapshot.id
        import diskgraph

        original = diskgraph.ingest

        def extension_only(*args, **kwargs):
            assert kwargs.get('parent_index_path') == str(snapshot.index_path)
            return original(*args, **kwargs)

        monkeypatch.setattr(diskgraph, 'ingest', extension_only)
        assert run(loaded) == (expected, identity)
    finally:
        kumo.configure(scratch_dir=previous)
