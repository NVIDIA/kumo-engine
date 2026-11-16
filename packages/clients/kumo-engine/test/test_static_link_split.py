# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumoapi.graph import Edge
from kumoapi.model_plan import TrainingTableGenerationPlan
from kumoapi.table import Column
from kumoapi.typing import Dtype, Stype

from kumo.engine import FileConnector, Graph, PredictiveQuery, Table
from kumo.engine.table import TableDefinition


@pytest.mark.parametrize('split', [
    'RandomSplit([0.6, 0.3, 0.1], seed=17)',
    'TemporalSplit([0.6, 0.3, 0.1])',
])
def test_public_static_link_generation_splits_links(tmp_path, monkeypatch,
                                                    split):
    monkeypatch.setattr('kumo.engine._scratch._root', tmp_path / 'scratch')
    source = tmp_path / 'inputs'
    source.mkdir()
    frames = {
        'users':
        pa.table({'id': list(range(20))}),
        'products':
        pa.table({'id': list(range(20))}),
        'orders':
        pa.table({
            'id': list(range(400)),
            'user_id': [i // 20 for i in range(400)],
            'product_id': [i % 20 for i in range(400)],
        }),
    }
    connector = FileConnector(root_dir=str(source))
    tables = []
    for name, frame in frames.items():
        pq.write_table(frame, source / f'{name}.parquet')
        tables.append(
            Table(
                connector, name,
                TableDefinition(
                    source_table=connector.source_table_for(name), pkey='id',
                    cols=[
                        Column(name=col, dtype=Dtype.int64, stype=Stype.ID)
                        for col in frame.column_names
                    ])))
    graph = Graph(tables, [
        Edge('orders', 'user_id', 'users'),
        Edge('orders', 'product_id', 'products')
    ])
    snapshot = graph.snapshot()
    query = PredictiveQuery(
        graph=graph, query='PREDICT LIST_DISTINCT(orders.product_id) '
        'RANK TOP 3 FOR EACH users.id')
    plan = TrainingTableGenerationPlan(split=split)
    table = query.generate_training_table(snapshot, plan=plan)
    assert plan.split == split
    manifest = json.loads(table.manifest_path.read_text())
    assert manifest['split_kind'] == 'RandomLinkSplit'
    stages = {}
    for stage in ['train', 'val', 'test']:
        frame = pq.read_table(table.staging_dir / f'{stage}_seeds')
        assert 'exclude_target_keys' in frame.column_names
        stages[stage] = {row['ENTITY1']: row for row in frame.to_pylist()}
    for entity in range(20):
        targets = [
            set(stages[stage].get(entity, {}).get('TARGET1', []))
            for stage in ['train', 'val', 'test']
        ]
        assert set.union(*targets) == set(range(20))
        assert sum(map(len, targets)) == 20
        for stage, expected in [('val', targets[0]),
                                ('test', targets[0] | targets[1])]:
            if entity in stages[stage]:
                assert set(
                    stages[stage][entity]['exclude_target_keys']) == expected
