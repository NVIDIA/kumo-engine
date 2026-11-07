# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
from kumoapi.graph import Edge
from kumoapi.typing import Stype

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir

START = pd.Timestamp('2024-01-01')
QUERY = ('PREDICT LIST_DISTINCT(events.ITEM_ID, 0, 7, days) '
         'RANK TOP 10 FOR EACH ents.ID')


def _graph(data):
    # Items 0-29 stay live, 30-39 end on day 30 and 40-59 appear on day 100.
    rng = np.random.default_rng(0)
    created = START + pd.to_timedelta(np.repeat([0, 100], [40, 20]), unit='D')
    ended = pd.Series(pd.NaT, index=range(60), dtype='datetime64[ns]')
    ended[30:40] = START + pd.Timedelta(days=30)
    item = rng.integers(0, 60, 4000)
    day = np.where(item < 40, rng.integers(0, 29, 4000),
                   rng.integers(100, 120, 4000))
    pd.DataFrame({'ID': np.arange(50)}).to_parquet(data / 'ents.parquet')
    pd.DataFrame({
        'IID': np.arange(60),
        'CREATED': created,
        'ENDED': ended,
    }).to_parquet(data / 'items.parquet')
    pd.DataFrame({
        'EID': np.arange(4000),
        'ENT_ID': rng.integers(0, 50, 4000),
        'ITEM_ID': item,
        'TS': START + pd.to_timedelta(day, unit='D'),
    }).to_parquet(data / 'events.parquet')

    connector = kumo.FileConnector(root_dir=str(data))
    ents = kumo.Table.from_source_table(connector['ents.parquet'],
                                        primary_key='ID').infer_metadata()
    items = kumo.Table.from_source_table(
        connector['items.parquet'], primary_key='IID', time_column='CREATED',
        end_time_column='ENDED').infer_metadata()
    events = kumo.Table.from_source_table(
        connector['events.parquet'],
        primary_key='EID', time_column='TS').infer_metadata().with_column(
            'ENT_ID', stype=Stype.ID).with_column('ITEM_ID', stype=Stype.ID)
    return kumo.Graph(
        tables={
            'ents': ents,
            'items': items,
            'events': events
        }, edges=[
            Edge(src_table='events', fkey='ENT_ID', dst_table='ents'),
            Edge(src_table='events', fkey='ITEM_ID', dst_table='items'),
        ])


def test_predict_ranks_only_items_valid_at_prediction_time(
        tmp_path, quick_plan):
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=tmp_path / 'scratch')
    data = tmp_path / 'data'
    data.mkdir()
    try:
        graph = _graph(data)
        snapshot = graph.snapshot()
        pquery = kumo.PredictiveQuery(graph=graph, query=QUERY)
        trainer = kumo.Trainer(model_plan=quick_plan)
        trainer.fit(pquery.generate_training_table(snapshot), snapshot=snapshot)
        table = pquery.generate_prediction_table(
            snapshot, prediction_time=START + pd.Timedelta(days=45))
        frame = trainer.predict(table, snapshot=snapshot,
                                num_classes_to_return=60).frame
    finally:
        kumo.configure(scratch_dir=previous)

    assert set(frame['CLASS'].astype(int)) == set(range(30))
