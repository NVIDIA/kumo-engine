# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused prediction artifact collection contract tests."""
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.types import PredictionArtifactType

from rest.backends.ml_backend.activities import parquet_io_activities as pio


def _job_input(*artifact_types):
    return SimpleNamespace(
        config=SimpleNamespace(outputs=[
            SimpleNamespace(artifact_type=artifact_type)
            for artifact_type in artifact_types
        ]), )


def _configure(monkeypatch, output_dir):
    monkeypatch.setattr(
        pio.PredictionConfig,
        'load',
        staticmethod(lambda _path: SimpleNamespace(
            output_dir=str(output_dir),
            diskgraph_index_path=None,
            model_dir='/tmp/model',
        )),
    )


def test_missing_requested_embeddings_raises(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    predictions = (tmp_path /
                   PredictionArtifactType.PREDICTIONS.output_file_name)
    predictions.mkdir()
    monkeypatch.setattr(pio.parquet_util, 'is_empty', lambda _path: False)

    with pytest.raises(pio.PredictionOutputFileMissingError):
        pio._collect_prediction_artifacts(
            _job_input(
                PredictionArtifactType.PREDICTIONS,
                PredictionArtifactType.EMBEDDINGS,
            ),
            '/tmp/job_config.json',
        )


def test_empty_requested_artifact_raises(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    monkeypatch.setattr(pio.fsutil, 'exists', lambda _path: True)
    monkeypatch.setattr(pio.parquet_util, 'is_empty', lambda _path: True)

    with pytest.raises(pio.PredictionOutputProcessingError,
                       match='empty predictions artifact'):
        pio._collect_prediction_artifacts(
            _job_input(PredictionArtifactType.PREDICTIONS),
            '/tmp/job_config.json',
        )


def test_duplicate_destinations_scan_artifact_once(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    existence_checks = []
    emptiness_checks = []
    monkeypatch.setattr(
        pio.fsutil,
        'exists',
        lambda path: existence_checks.append(path) or True,
    )
    monkeypatch.setattr(
        pio.parquet_util,
        'is_empty',
        lambda path: emptiness_checks.append(path) or False,
    )

    result = pio._collect_prediction_artifacts(
        _job_input(
            PredictionArtifactType.PREDICTIONS,
            PredictionArtifactType.PREDICTIONS,
        ),
        '/tmp/job_config.json',
    )

    assert set(result) == {PredictionArtifactType.PREDICTIONS}
    assert len(existence_checks) == 1
    assert len(emptiness_checks) == 1


def test_embeddings_only_multishard_collection_never_requires_predictions(
        monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    embedding_dir = (tmp_path /
                     PredictionArtifactType.EMBEDDINGS.output_file_name)
    embedding_dir.mkdir()
    for shard_id, entity_id in enumerate(['user_a', 'user_b']):
        pq.write_table(
            pa.table({
                'ID': [entity_id],
                'TYPE': ['users'],
                'EMBEDDING': [[1.0, 2.0]],
            }),
            embedding_dir / f'part-{shard_id:05d}-00000.parquet',
        )

    result = pio._collect_prediction_artifacts(
        _job_input(PredictionArtifactType.EMBEDDINGS),
        '/tmp/job_config.json',
    )

    assert result == {
        PredictionArtifactType.EMBEDDINGS: str(embedding_dir),
    }
    assert not (tmp_path /
                PredictionArtifactType.PREDICTIONS.output_file_name).exists()


def test_no_requested_outputs_fails_without_opening_predictions(
        monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    monkeypatch.setattr(
        pio.fsutil,
        'exists',
        lambda _path: pytest.fail('no-output must not open predictions'),
    )

    with pytest.raises(ValueError, match='at least one requested output'):
        pio._collect_prediction_artifacts(
            _job_input(),
            '/tmp/job_config.json',
        )
