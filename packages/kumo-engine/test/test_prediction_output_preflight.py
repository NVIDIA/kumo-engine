# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pandas as pd
import pytest
from kumo_core.artifact_export import ArtifactDestinationConflict
from kumoapi.graph import Edge
from kumoapi.jobs import PredictionArtifactType
from kumoapi.model_plan import ModelPlan
from kumoapi.typing import Stype

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir
from kumo.engine.trainer.output_config import PredictionOutputConfig

_NON_ROOT = pytest.mark.skipif(os.geteuid() == 0,
                               reason='root bypasses mode permissions')


@pytest.fixture(scope='module')
def fitted(tmp_path_factory):
    root = tmp_path_factory.mktemp('preflight')
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=root / 'scratch')
    rng = np.random.default_rng(0)
    data = root / 'data'
    data.mkdir()
    pd.DataFrame({'ID': np.arange(50)}).to_parquet(data / 'ents.parquet')
    pd.DataFrame({
        'EID':
        np.arange(3000),
        'ENT_ID':
        rng.integers(0, 50, 3000),
        'TS':
        pd.Timestamp('2024-01-01') +
        pd.to_timedelta(rng.integers(0, 120, 3000), unit='D'),
    }).to_parquet(data / 'events.parquet')
    connector = kumo.FileConnector(root_dir=str(data))
    events = kumo.Table.from_source_table(
        connector['events.parquet'], primary_key='EID',
        time_column='TS').infer_metadata().with_column('ENT_ID', stype=Stype.ID)
    graph = kumo.Graph(
        tables={
            'ents':
            kumo.Table.from_source_table(connector['ents.parquet'],
                                         primary_key='ID').infer_metadata(),
            'events':
            events,
        }, edges=[Edge(src_table='events', fkey='ENT_ID', dst_table='ents')])
    snapshot = graph.snapshot()
    pquery = kumo.PredictiveQuery(
        graph=graph, query='PREDICT COUNT(events.*, 0, 7, days) > 0 '
        'FOR EACH ents.ID')
    plan = ModelPlan()
    plan.training_job.disable_explain = True
    plan.optimization.max_epochs = 1
    plan.optimization.max_steps_per_epoch = 2
    trainer = kumo.Trainer(model_plan=plan)
    trainer.fit(pquery.generate_training_table(snapshot), snapshot=snapshot)
    yield trainer, pquery.generate_prediction_table(snapshot), snapshot
    kumo.configure(scratch_dir=previous)


@_NON_ROOT
@pytest.mark.parametrize('outputs, locked, denied', [
    ({
        'artifact_type': PredictionArtifactType.PREDICTIONS,
        'table_name': 'results/'
    }, '.', 'results/'),
    ({
        'artifact_type': PredictionArtifactType.EMBEDDINGS,
        'table_name': 'results'
    }, '.', 'results'),
    ({
        'output_paths': {
            'predictions': 'predictions/',
            'embeddings': 'locked/embeddings/'
        }
    }, 'locked', 'locked/embeddings/'),
])
def test_unwritable_output_fails_before_prediction(fitted, tmp_path, outputs,
                                                   locked, denied):
    trainer, table, snapshot = fitted
    (tmp_path / 'locked').mkdir()
    output = PredictionOutputConfig(kumo.FileConnector(root_dir=str(tmp_path)),
                                    **outputs)
    runs = trainer.scratch_dir / 'runs'
    before = set(runs.glob('*-predict-*'))

    (tmp_path / locked).chmod(0o500)
    try:
        with pytest.raises(PermissionError) as error:
            trainer.predict(table, snapshot=snapshot, output_config=output,
                            binary_classification_threshold=.5)
    finally:
        (tmp_path / locked).chmod(0o700)
    assert error.value.filename == f'{tmp_path}/{denied}'
    assert set(runs.glob('*-predict-*')) == before
    assert [path.name for path in tmp_path.iterdir()] == ['locked']

    trainer.predict(table, snapshot=snapshot, output_config=output,
                    binary_classification_threshold=.5)
    for destination in output.destinations().values():
        assert list(tmp_path.glob(destination.rstrip('/') + '*'))


def test_occupied_output_directory_fails_before_prediction(fitted, tmp_path):
    trainer, table, snapshot = fitted
    (tmp_path / 'results').mkdir()
    (tmp_path / 'results' / 'old.txt').write_text('other')
    output = PredictionOutputConfig(
        kumo.FileConnector(root_dir=str(tmp_path)),
        artifact_type=PredictionArtifactType.PREDICTIONS, table_name='results/')
    runs = trainer.scratch_dir / 'runs'
    before = set(runs.glob('*-predict-*'))

    with pytest.raises(ArtifactDestinationConflict, match='another export'):
        trainer.predict(table, snapshot=snapshot, output_config=output,
                        binary_classification_threshold=.5)
    assert set(runs.glob('*-predict-*')) == before
    assert [p.name for p in (tmp_path / 'results').iterdir()] == ['old.txt']
