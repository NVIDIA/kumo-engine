# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time

import numpy as np
import pandas as pd
import pytest
from kumoapi.graph import Edge
from kumoapi.model_plan import ModelPlan
from kumoapi.typing import Stype

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir

mlflow = pytest.importorskip('mlflow')


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scratch, uri = get_scratch_dir(), mlflow.get_tracking_uri()
    kumo.configure(scratch_dir=tmp_path / 'scratch')
    mlflow.set_tracking_uri(f'sqlite:///{tmp_path / "mlflow.db"}')
    yield
    kumo.configure(scratch_dir=scratch)
    mlflow.set_tracking_uri(uri)


def test_a_prediction_on_a_newer_snapshot_keeps_the_models_lineage_apart(
        tmp_path, isolated):
    from kumo.engine.integrations.mlflow import MlflowTracking

    data = tmp_path / 'data'
    data.mkdir()
    rng = np.random.default_rng(0)

    def write(rows):
        pd.DataFrame({
            'EID': np.arange(rows), 'ENT_ID': rng.integers(0, 50, rows),
            'TS': pd.Timestamp('2024-01-01') +
            pd.to_timedelta(rng.integers(0, 120, rows), unit='D'),
        }).to_parquet(data / 'events.parquet')

    pd.DataFrame({'ID': np.arange(50)}).to_parquet(data / 'ents.parquet')
    write(3000)
    for path in data.iterdir():
        os.utime(path, (time.time() - 3600, ) * 2)
    connector = kumo.FileConnector(root_dir=str(data))
    graph = kumo.Graph(
        tables={
            'ents': kumo.Table.from_source_table(
                connector['ents.parquet'], primary_key='ID').infer_metadata(),
            'events': kumo.Table.from_source_table(
                connector['events.parquet'], primary_key='EID',
                time_column='TS').infer_metadata().with_column(
                    'ENT_ID', stype=Stype.ID),
        }, edges=[Edge(src_table='events', fkey='ENT_ID', dst_table='ents')])
    query = kumo.PredictiveQuery(
        graph=graph, query='PREDICT COUNT(events.*, 0, 7, days) > 0 '
        'FOR EACH ents.ID')
    plan = ModelPlan()
    plan.training_job.disable_explain = True
    plan.optimization.max_epochs = 1
    plan.optimization.max_steps_per_epoch = 2
    trainer = kumo.Trainer(plan)
    with kumo.mlops(kumo.MLOps(str(tmp_path / 'root'), tracking=MlflowTracking(
            'lineage', log_system_metrics=False))):
        trained = graph.snapshot()
        trainer.fit(query.generate_training_table(trained), snapshot=trained)
        write(3500)
        scored = graph.snapshot()
        trainer.predict(query.generate_prediction_table(scored),
                        snapshot=scored)

    table = trainer.result.manifest.parents.training_table
    ents, old, new = (
        s.manifest.parents.source_tables[name]
        for s, name in ((trained, 'ents'), (trained, 'events'),
                        (scored, 'events')))
    runs = {run.data.tags['kumo.kind']: run for run in mlflow.search_runs(
        experiment_names=['lineage'], output_format='list')}

    def inputs(kind):
        return {item.dataset.name: item.tags[0].value
                for item in runs[kind].inputs.dataset_inputs
                if item.tags[0].value != 'output'}

    def lineage(kind):
        card = runs[kind].data.tags['mlflow.note.content']
        return card.split('**Lineage**\n\n')[1].split('\n\n')[0].split('\n')

    assert inputs('fit') == {
        table.title: 'training examples',
        trained.manifest.title: 'graph context',
        'ents': 'source table', 'events': 'source table'}
    assert inputs('prediction') == {
        scored.manifest.title: 'scoring graph',
        'ents': 'scoring source table', 'events': 'scoring source table'}
    model = f'`{trainer.result.id[:8]}`'
    assert lineage('fit') == [
        f'- `ents` v{ents.version} (`{ents.location}`), `events` '
        f'v{old.version} (`{old.location}`) → {trained.manifest.title} → '
        f'{table.title} → model {model}']
    assert lineage('prediction') == [
        f'- Model: {model} ← {table.title} ← {trained.manifest.title} ← '
        f'`ents` v{ents.version}, `events` v{old.version}',
        f'- Scored: {scored.manifest.title} ← `ents` v{ents.version} '
        f'(`{ents.location}`), `events` v{new.version} (`{new.location}`)',
        '- Scored on a different snapshot than the model was trained on; '
        f'newer: `events` (v{old.version} → v{new.version}).']
