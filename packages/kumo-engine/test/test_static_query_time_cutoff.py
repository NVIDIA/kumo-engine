# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pandas as pd
import pytest
from kumoapi.model_plan import ModelPlan

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir
from kumo.engine.pquery.prediction_table import PredictionTable


@pytest.fixture(scope='module')
def fitted(tmp_path_factory):
    root = tmp_path_factory.mktemp('static')
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=root / 'scratch')
    rng = np.random.default_rng(0)
    data = root / 'data'
    data.mkdir()
    times = pd.Timestamp('2024-01-01') + pd.to_timedelta(
        rng.integers(0, 300, 400), unit='D')
    label = pd.array(rng.random(400) < 0.5, dtype='boolean')
    label[:40] = pd.NA
    pd.DataFrame({
        'ID': np.arange(400),
        'TS': times,
        'LABEL': label,
        'F1': rng.normal(size=400),
    }).to_parquet(data / 'ents.parquet')
    connector = kumo.FileConnector(root_dir=str(data))
    ents = kumo.Table.from_source_table(connector['ents.parquet'],
                                        primary_key='ID',
                                        time_column='TS').infer_metadata()
    graph = kumo.Graph(tables={'ents': ents}, edges=[])
    snapshot = graph.snapshot()
    pquery = kumo.PredictiveQuery(
        graph=graph, query='PREDICT ents.LABEL = True FOR EACH ents.ID')
    plan = ModelPlan()
    plan.training_job.disable_explain = True
    plan.optimization.max_epochs = 1
    plan.optimization.max_steps_per_epoch = 2
    trainer = kumo.Trainer(model_plan=plan)
    trainer.fit(pquery.generate_training_table(snapshot), snapshot=snapshot)
    [run] = trainer.scratch_dir.glob('runs/*-fit-*')
    yield run, pquery, snapshot, connector, data, times
    kumo.configure(scratch_dir=previous)


def test_static_query_on_timestamped_entity_is_temporal(fitted):
    run = fitted[0]
    metadata = json.loads((run / 'task_dataset' / 'metadata.json').read_text())
    model = json.loads(
        (run / 'published-model' / 'model' / 'config.json').read_text())

    assert metadata['is_temporal'] is True
    assert model['encoder']['exclude_cols_dict'] == {'ents': ['LABEL']}


def test_custom_prediction_rows_need_timestamp(fitted):
    _, pquery, snapshot, connector, data, times = fitted
    pd.DataFrame({'ENTITY1': [0, 1]}).to_parquet(data / 'bare.parquet')
    pd.DataFrame({
        'ENTITY1': [0, 1],
        'TIMESTAMP': times[:2],
    }).to_parquet(data / 'anchored.parquet')

    with pytest.raises(ValueError, match='TIMESTAMP'):
        PredictionTable.from_source_table(connector['bare.parquet'],
                                          pq=pquery, snapshot=snapshot)
    table = PredictionTable.from_source_table(connector['anchored.parquet'],
                                              pq=pquery, snapshot=snapshot)
    assert table.row_count == 2
