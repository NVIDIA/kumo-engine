# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate tiny native tables with the four configured workload queries."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
from kumoapi.graph import Edge
from kumoapi.model_plan import TrainingTableGenerationPlan
from kumoapi.typing import Stype

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir

CONFIGURATION = Path(__file__).with_name('nightly') / 'inputs.json'


def tiny_graph(directory, workload):
    directory.mkdir()
    actor = 'viewers' if workload == 'streaming' else 'accounts'
    key = 'viewer_id' if workload == 'streaming' else 'account_id'
    clock = 'signup_time' if workload == 'streaming' else 'created_time'
    start = datetime(2023, 1, 1)
    frames = {
        actor:
        pd.DataFrame({
            key: range(12),
            clock: [start] * 12,
            'group': ['a', 'b'] * 6
        })
    }
    metadata = {actor: (key, clock, [])}
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='3D')
    if workload == 'streaming':
        frames['titles'] = pd.DataFrame(
            dict(title_id=range(80), added_time=[start] * 80,
                 category=['x', 'y'] * 40))
        metadata['titles'] = ('title_id', 'added_time', [])
        frames['sessions'] = pd.DataFrame([
            dict(session_id=i * 12 + a, viewer_id=a, title_id=(i + a) % 80,
                 start_time=date) for i, date in enumerate(dates)
            for a in range(12)
        ])
        metadata['sessions'] = ('session_id',
                                'start_time', [('viewer_id', 'viewers'),
                                               ('title_id', 'titles')])
    else:
        if workload in ('community', 'fraud'):
            frames['logins'] = pd.DataFrame([
                dict(account_id=a, login_time=date) for date in dates
                for a in range(12)
            ])
            metadata['logins'] = (None, 'login_time', [('account_id',
                                                        'accounts')])
        if workload == 'community':
            frames['messages'] = pd.DataFrame([
                dict(message_id=i * 12 + a, account_id=a, sent_time=date)
                for i, date in enumerate(dates) for a in range(12)
                if (i + a) % 2
            ])
            metadata['messages'] = ('message_id', 'sent_time', [('account_id',
                                                                 'accounts')])
        else:
            frames['account_flags'] = pd.DataFrame([
                dict(account_id=a,
                     flag_time=datetime(2024, 8, 1) + timedelta(days=a * 15))
                for a in range(8)
            ])
            metadata['account_flags'] = (None, 'flag_time', [('account_id',
                                                              'accounts')])
    connector = kumo.FileConnector(root_dir=str(directory))
    tables, edges = {}, []
    for name, frame in frames.items():
        frame.to_parquet(directory / (name + '.parquet'), index=False)
        primary_key, time_column, relations = metadata[name]
        table = kumo.Table.from_source_table(
            connector[name + '.parquet'], primary_key=primary_key,
            time_column=time_column).infer_metadata()
        for column, destination in relations:
            table = table.with_column(column, stype=Stype.ID)
            edges.append(
                Edge(src_table=name, fkey=column, dst_table=destination))
        tables[name] = table
    return kumo.Graph(tables, edges)


@pytest.mark.parametrize('workload,days,task_type', [
    ('streaming', 7, 'temporal_link_prediction'),
    ('community', 30, 'regression'),
    ('payments', 14, 'binary_classification'),
    ('fraud', 28, 'binary_classification'),
])
def test_configured_variant_query_generates_all_native_stages(
        tmp_path, workload, days, task_type):
    variants = json.loads(CONFIGURATION.read_text())['variants']
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=tmp_path / 'scratch')
    try:
        graph = tiny_graph(tmp_path / 'source', workload)
        graph.validate()
        query = kumo.PredictiveQuery(graph, variants[workload]['query'])
        query.validate()
        assert query.get_task_type().value == task_type
        end, horizon = datetime(2024, 12, 31), timedelta(days=days)
        ranges = [((datetime(2024, 1, 1) if offset == 3 else end -
                    offset * horizon).isoformat(),
                   (end - (offset - 1) * horizon).isoformat())
                  for offset in (3, 2, 1)]
        split = (f'TimeRangeSplit({ranges!r}, timeframe={days}, '
                 'flexible_boundaries=False)')
        plan = TrainingTableGenerationPlan(split=split, timeframe_step=days)
        table = query.generate_training_table(graph.snapshot(), plan=plan)
        destination = tmp_path / 'export'
        destination.mkdir()
        table.export(
            kumo.TrainingTableOutputConfig(
                kumo.FileConnector(root_dir=str(destination)), 'rows'))
        rows = pq.read_table(destination / 'rows').to_pandas()
        assert {'ENTITY1', 'TARGET1', 'SPLIT'} <= set(rows)
        assert set(rows.SPLIT.value_counts().index) == {0, 1, 2}
    finally:
        kumo.configure(scratch_dir=previous)
