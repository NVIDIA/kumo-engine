# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq
import pytest
from kumo_core.diskgraph.snapshot import DiskgraphIngestResult
from kumo_core.sources.object_store_access import ObjectStoreAccess
from kumoapi.typing import Dtype, Stype
from pyarrow import fs

from kumo.engine import Graph, Table
from kumo.engine.connector.gcs import GCSConnector
from kumo.engine.connector.identity import GCSIdentity


@pytest.mark.parametrize('extension', ['csv', 'parquet'])
def test_source_table_inference_and_graph_acquisition(extension, tmp_path,
                                                      monkeypatch):
    objects = tmp_path / 'objects'
    dataset = objects / 'bucket' / 'events'
    dataset.mkdir(parents=True)
    expected = pa.table({
        'id':
        range(100),
        'value':
        pa.array([1, 2, 3, 4] + [None] * 96, type=pa.int64()),
    })
    source = dataset / f'part.{extension}'
    if extension == 'csv':
        csv.write_csv(expected, source)
    else:
        pq.write_table(expected, source)
    filesystem = fs.SubTreeFileSystem(str(objects), fs.LocalFileSystem())
    email = 'reader@test-project.iam.gserviceaccount.com'
    identity = GCSIdentity(workload_service_account_email=email)
    credential = object()
    bindings = []

    def session(principal):
        assert principal.config == identity._config
        return credential

    def access(*, credentials, project, requester_pays_project):
        assert credentials is credential
        bindings.append(requester_pays_project)
        return ObjectStoreAccess(filesystem)

    monkeypatch.setattr('kumo_gcs.backend._gcs_session_for', session)
    monkeypatch.setattr('kumo_gcs.source_access.source_access', access)
    monkeypatch.setattr('kumo.engine._scratch._root', tmp_path / 'scratch')
    connector = GCSConnector(root_dir='gs://bucket', identity=identity,
                             requester_pays_project='billing-project')
    table = Table.from_source_table(connector[f'events/part.{extension}'],
                                    primary_key='id')
    columns = {column.name: column for column in table.definition.cols}
    assert columns['id'].stype == Stype.ID
    assert columns['value'].dtype == Dtype.int64
    assert columns['value'].stype == Stype.categorical
    ingested = []

    def ingest(graph_config, spec):
        assert set(spec.source_paths) == {table.table_name}
        local = Path(spec.source_paths[table.table_name])
        assert local.is_relative_to(tmp_path / 'scratch')
        actual = pq.ParquetDataset(local).read()
        assert actual.to_pydict() == expected.to_pydict()
        assert actual.schema.field('value').type == pa.int64()
        ingested.append(graph_config)
        return DiskgraphIngestResult.for_index(spec.output_dir)

    monkeypatch.setattr('kumo_core.diskgraph.ingest.execute_diskgraph_ingest',
                        ingest)
    graph = Graph(tables=[table])
    snapshot = graph.snapshot()
    assert len(ingested) == 1
    assert snapshot.graph_config == ingested[0]
    assert snapshot.id.startswith(graph.id + '@')
    assert bindings and set(bindings) == {'billing-project'}


@pytest.mark.parametrize('suffix', ['', '.parquet'])
def test_graph_aliases_preserve_object_source_acquisition(
        suffix, tmp_path, monkeypatch):
    from kumo_s3.backend import S3SourceBackend
    from kumoapi.graph import Edge

    from kumo.engine.connector.s3 import S3Connector

    objects = tmp_path / 'objects' / 'bucket'
    objects.mkdir(parents=True)
    frames = {
        'markets': pa.table({'id': [1, 2]}),
        'events': pa.table({
            'id': [10, 11],
            'market_id': [1, 2]
        }),
    }
    for name, frame in frames.items():
        pq.write_table(frame, objects / f'{name}.parquet')
    access = ObjectStoreAccess(
        fs.SubTreeFileSystem(str(objects.parent), fs.LocalFileSystem()))
    monkeypatch.setattr(S3SourceBackend, '_source_access', lambda *args: access)
    monkeypatch.setattr('kumo.engine._scratch._root', tmp_path / 'scratch')
    connector = S3Connector(root_dir='s3://bucket')
    tables = {
        alias: Table.from_source_table(connector[alias + suffix],
                                       primary_key='id')
        for alias in frames
    }
    ingested = []

    def ingest(config, spec):
        assert set(spec.source_paths) == set(frames)
        for alias, frame in frames.items():
            assert pq.read_table(
                spec.source_paths[alias]).to_pydict() == frame.to_pydict()
        ingested.append(config)
        return DiskgraphIngestResult.for_index(spec.output_dir)

    monkeypatch.setattr('kumo_core.diskgraph.ingest.execute_diskgraph_ingest',
                        ingest)
    graph = Graph(tables, [Edge('events', 'market_id', 'markets')])
    graph.validate()
    assert {table.name for table in graph.native_graph.tables} == set(frames)
    snapshot = graph.snapshot()
    assert snapshot.graph_config == ingested[0]
    assert tables['markets'].table_name == 'markets' + suffix
    assert tables[
        'markets'].definition.source_table.source.path == 'markets' + suffix
    assert graph.id == Graph(dict(reversed(list(tables.items()))),
                             graph.edges).id
    assert graph.id != Graph({'other': tables['markets']}).id
