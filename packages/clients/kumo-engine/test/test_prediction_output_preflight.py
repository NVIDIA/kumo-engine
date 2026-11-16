# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pandas as pd
import pytest
from kumo_core.artifact_export import ArtifactDestinationConflict
from kumoapi.jobs import PredictionArtifactType, WriteMode

import kumo.engine as kumo
from kumo.engine.trainer.output_config import PredictionOutputConfig
from kumo.engine.trainer.trainer import PredictionResult

_NON_ROOT = pytest.mark.skipif(os.geteuid() == 0,
                               reason='root bypasses mode permissions')


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


def _predict(trainer, table, snapshot, output):
    return trainer.predict(table, snapshot=snapshot, output_config=output,
                           binary_classification_threshold=.5)


def test_mlops_keeps_every_run_on_the_artifact_root(fitted, tmp_path):
    trainer, table, snapshot = fitted
    output = PredictionOutputConfig(
        kumo.FileConnector(root_dir=str(tmp_path)),
        output_paths={'predictions': 'scores/'},
        write_mode=WriteMode.OVERWRITE)
    cfg = kumo.MLOps(str(tmp_path / 'root'))
    with kumo.mlops(cfg):
        runs = [_predict(trainer, table, snapshot, output) for _ in range(2)]

    assert json.loads((tmp_path / 'scores' / '_provenance.json'
                       ).read_text())['id'] == runs[-1].id
    for run in runs:
        loaded = PredictionResult.load_from_connector(
            cfg, path=f'predictions/{run.id}')
        assert (loaded.id, loaded.num_rows) == (run.id, run.num_rows)


def test_overwrite_replaces_the_destination(fitted, tmp_path):
    trainer, table, snapshot = fitted
    (tmp_path / 'results').mkdir()
    (tmp_path / 'results' / 'old.txt').write_text('other')
    output = PredictionOutputConfig(
        kumo.FileConnector(root_dir=str(tmp_path)),
        output_paths={'predictions': 'results/'},
        write_mode=WriteMode.OVERWRITE)

    for _ in range(2):
        run = _predict(trainer, table, snapshot, output)
        assert not (tmp_path / 'results' / 'old.txt').exists()
        assert run.manifest.outputs[0].destination.location == \
            f'{tmp_path}/results'
        assert json.loads((tmp_path / 'results' / '_provenance.json'
                           ).read_text())['id'] == run.id
        assert pd.read_parquet(tmp_path / 'results').shape[0] == run.num_rows


def test_occupied_output_directory_fails_before_scoring(fitted, tmp_path,
                                                        monkeypatch):
    trainer, table, snapshot = fitted
    taken = tmp_path / 'results'
    taken.mkdir()
    (taken / 'old.txt').write_text('other')

    def score(**_):
        raise AssertionError('scored before the destination was checked')

    monkeypatch.setattr('kumo_core.prediction.score_prediction', score)
    output = PredictionOutputConfig(
        kumo.FileConnector(root_dir=str(tmp_path)),
        artifact_type=PredictionArtifactType.PREDICTIONS, table_name='results/')
    with pytest.raises(ArtifactDestinationConflict,
                       match=f"'file://{taken}/'.*write_mode=WriteMode"):
        _predict(trainer, table, snapshot, output)
    assert [p.name for p in taken.iterdir()] == ['old.txt']
