# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.training.table import ingest_train_table
from kumoapi.graph import Edge
from kumoapi.model_plan import TrainingTableGenerationPlan
from kumoapi.train import TrainingTableSpec
from kumoapi.typing import Stype

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir

FREQUENCY = '__kumo_internal_frequency_column__'


def records(table):
    return Counter((row['ENTITY1'], row['TIMESTAMP'], row['SPLIT'],
                    tuple(sorted(zip(row['TARGET1'], row[FREQUENCY]))))
                   for row in table.to_pylist())


@pytest.fixture(scope='module')
def future_targets(tmp_path_factory):
    root = tmp_path_factory.mktemp('future-targets')
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=root / 'scratch')
    data = root / 'data'
    data.mkdir()
    pd.DataFrame({'ID': [0, 1]}).to_parquet(data / 'ents.parquet')
    pd.DataFrame({
        'IID': [10, 20, 30, 40],
        'CREATED':
        pd.to_datetime(['2024-01-01', '2024-02-25', '2024-03-03',
                        '2024-03-10']),
    }).to_parquet(data / 'items.parquet')
    anchors = pd.to_datetime(['2024-02-22', '2024-02-29', '2024-03-07'])
    rows = []
    for anchor, target in zip(anchors, (20, 30, 40)):
        for entity in (0, 1):
            for item, offset in ((10, 1), (10, 2), (target, 4)):
                rows.append((len(rows), entity, item,
                             anchor + pd.Timedelta(days=offset)))
    for entity in (0, 1):
        rows.append((len(rows), entity, 10, pd.Timestamp('2024-02-01')))
    rows.append((len(rows), 0, 10, pd.Timestamp('2024-03-15')))
    pd.DataFrame(rows, columns=['EID', 'ENT_ID', 'ITEM_ID',
                                'TS']).to_parquet(data / 'events.parquet')
    connector = kumo.FileConnector(root_dir=str(data))
    entities = kumo.Table.from_source_table(connector['ents.parquet'],
                                            primary_key='ID').infer_metadata()
    items = kumo.Table.from_source_table(
        connector['items.parquet'], primary_key='IID',
        time_column='CREATED').infer_metadata()
    events = kumo.Table.from_source_table(
        connector['events.parquet'],
        primary_key='EID', time_column='TS').infer_metadata().with_column(
            'ENT_ID', stype=Stype.ID).with_column('ITEM_ID', stype=Stype.ID)
    graph = kumo.Graph(
        tables={
            'ents': entities,
            'items': items,
            'events': events
        }, edges=[
            Edge(src_table='events', fkey='ENT_ID', dst_table='ents'),
            Edge(src_table='events', fkey='ITEM_ID', dst_table='items')
        ])
    snapshot = graph.snapshot()
    query = kumo.PredictiveQuery(
        graph, 'PREDICT LIST_DISTINCT(events.ITEM_ID, 0, 7, days) '
        'RANK TOP 2 FOR EACH ents.ID')
    plan = TrainingTableGenerationPlan(
        split="TimeRangeSplit([('2024-02-22', '2024-02-29'), "
        "('2024-02-29', '2024-03-07'), ('2024-03-07', '2024-03-14')], "
        'timeframe=7, flexible_boundaries=False)', timeframe_step=7)
    normal = query.generate_training_table(snapshot, plan=plan)
    output = root / 'export'
    output.mkdir()
    exported_connector = kumo.FileConnector(root_dir=str(output))
    normal.export(kumo.TrainingTableOutputConfig(exported_connector, 'normal'))
    exported = pq.read_table(output / 'normal')
    expected = Counter(
        (entity, anchor, stage, ((10, 2), (target, 1)))
        for stage, (anchor, target) in enumerate(zip(anchors, (20, 30, 40)))
        for entity in (0, 1))
    assert records(exported) == expected
    pq.write_table(exported.drop(['SPLIT']), output / 'source.parquet')
    yield query, snapshot, plan, exported_connector, output, exported
    kumo.configure(scratch_dir=previous)


def test_import_preserves_future_window_targets_when_enabled(future_targets):
    query, snapshot, plan, connector, output, normal = future_targets
    imported = kumo.TrainingTable.from_source_table(
        connector['source.parquet'], TrainingTableSpec(), pq=query,
        snapshot=snapshot, plan=plan, allow_new_target_entities=True)
    imported.export(kumo.TrainingTableOutputConfig(connector, 'preserved'))
    assert records(pq.read_table(output / 'preserved')) == records(normal)


def test_default_import_keeps_anchor_visibility_filter(future_targets):
    query, snapshot, plan, connector, output, normal = future_targets
    imported = kumo.TrainingTable.from_source_table(connector['source.parquet'],
                                                    TrainingTableSpec(),
                                                    pq=query, snapshot=snapshot,
                                                    plan=plan)
    imported.export(kumo.TrainingTableOutputConfig(connector, 'default'))
    expected = Counter((entity, anchor, stage, ((10, 2), ))
                       for entity, anchor, stage, _ in records(normal))
    assert records(pq.read_table(output / 'default')) == expected


@pytest.mark.parametrize('preserve', [False, True])
def test_connector_reload_respects_target_visibility_opt_in(
        future_targets, preserve):
    query, snapshot, _, connector, output, normal = future_targets
    options = {'allow_new_target_entities': True} if preserve else {}
    loaded = kumo.TrainingTable.load_from_connector(connector, path='normal',
                                                    pquery=query,
                                                    snapshot=snapshot,
                                                    **options)
    name = f'reloaded-{preserve}'
    loaded.export(kumo.TrainingTableOutputConfig(connector, name))
    expected = records(normal) if preserve else Counter(
        (entity, anchor, stage, ((10, 2), ))
        for entity, anchor, stage, _ in records(normal))
    assert records(pq.read_table(output / name)) == expected


@pytest.mark.parametrize('unknown_target', [False, True])
def test_opt_in_keeps_target_key_resolution_at_ingest(future_targets,
                                                      unknown_target):
    query, snapshot, plan, connector, output, normal = future_targets
    source = normal.drop(['SPLIT'])
    if unknown_target:
        for name, values in (('TARGET1', [9999]), (FREQUENCY, [1])):
            column = source[name]
            rows = column.to_pylist()
            rows[0] = values
            source = source.set_column(source.schema.get_field_index(name),
                                       name, pa.array(rows, column.type))
    name = f'ingest-{unknown_target}'
    pq.write_table(source, output / f'{name}.parquet')
    imported = kumo.TrainingTable.from_source_table(
        connector[f'{name}.parquet'], TrainingTableSpec(), pq=query,
        snapshot=snapshot, plan=plan, allow_new_target_entities=True)
    ingested = ingest_train_table(staging_dir=str(imported.staging_dir),
                                  index_path=str(snapshot.index_path),
                                  tt_id=name)
    manifest = json.loads(
        (Path(ingested.extension_path) / 'diskgraph_manifest.json').read_text())
    stages = manifest['task_dataset']['stages']
    assert manifest['task_dataset']['input_row_count'] == normal.num_rows
    assert sum(stage['row_count']
               for stage in stages.values()) == (normal.num_rows -
                                                 int(unknown_target))
    failures = manifest['key_failures']
    assert failures['parse_failures'] == 0
    assert (failures['lookup_failures'] +
            failures['membership_failures']) == int(unknown_target)
