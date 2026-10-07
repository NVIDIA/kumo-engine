# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from kumo_core.artifact_export import ArtifactDestinationConflict
from kumoapi.jobs import PredictionArtifactType

import kumo.engine as kumo
from kumo.engine.trainer.output_config import PredictionOutputConfig

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
