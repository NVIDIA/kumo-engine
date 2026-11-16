# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prediction export to real BigQuery; skipped without application default
credentials."""

import os
import uuid

import pytest
from kumoapi.jobs import PredictionArtifactType, WriteMode

import kumo.engine as kumo
from kumo.engine.trainer.output_config import PredictionOutputConfig

pytest.importorskip('kumo_bigquery')
SCOPE = 'https://www.googleapis.com/auth/cloud-platform'
PROJECT = os.environ.get('KUMO_BIGQUERY_PROJECT')
STAGING_BUCKET = os.environ.get('KUMO_BIGQUERY_STAGING')
if not (PROJECT and STAGING_BUCKET):
    pytest.skip(
        'LIVE BIGQUERY OUTPUT NOT RUN: set KUMO_BIGQUERY_PROJECT and '
        'KUMO_BIGQUERY_STAGING', allow_module_level=True)
RUN = uuid.uuid4().hex[:8]
DATASET = f'kumo_conn_test_engine_{RUN}'
STAGING_URI = f'{STAGING_BUCKET}/engine/{RUN}'

try:
    import google.auth
    CREDENTIALS = google.auth.default(
        scopes=[SCOPE])[0].with_quota_project(PROJECT)
except Exception as error:
    pytest.skip(f'LIVE BIGQUERY OUTPUT NOT RUN: {type(error).__name__}',
                allow_module_level=True)

from google.cloud import bigquery, storage  # noqa: E402
from kumo_core.sources.errors import SourceNotFoundError  # noqa: E402


def staged_objects():
    bucket, _, prefix = STAGING_URI.removeprefix('gs://').partition('/')
    gcs = storage.Client(project=PROJECT, credentials=CREDENTIALS)
    return list(gcs.list_blobs(bucket, prefix=prefix))


@pytest.fixture(scope='module')
def client():
    bq = bigquery.Client(project=PROJECT, credentials=CREDENTIALS)
    bq.create_dataset(bigquery.Dataset(f'{PROJECT}.{DATASET}'))
    try:
        yield bq
    finally:
        bq.delete_dataset(f'{PROJECT}.{DATASET}', delete_contents=True,
                          not_found_ok=True)
        for blob in staged_objects():
            blob.delete()


@pytest.fixture
def cpu_fitted(fitted):
    # Two scorers without two GPUs.
    trainer, table, snapshot = fitted
    assert trainer._training_result is not None
    cpu_trainer = kumo.Trainer.load(
        trainer._training_result.bundle_dir, accelerator='cpu')
    return cpu_trainer, table, snapshot


def connector(staged, dataset=DATASET):
    return kumo.BigQueryConnector(project_id=PROJECT, dataset_id=dataset,
                                  identity=kumo.BigQueryIdentity(),
                                  staging_uri=STAGING_URI if staged else None, require_staged=staged)


def rows(client, name):
    # Table metadata lags rows committed through the Storage Write API.
    query = f'SELECT COUNT(*) AS n FROM `{PROJECT}.{DATASET}.{name}`'
    return next(iter(client.query(query).result())).n


@pytest.mark.parametrize('staged', [True, False])
def test_predict_and_export_to_bigquery(cpu_fitted, client, staged):
    trainer, table, snapshot = cpu_fitted
    name = f'preds_{uuid.uuid4().hex[:6]}'
    output = PredictionOutputConfig(
        connector(staged), artifact_type=PredictionArtifactType.PREDICTIONS,
        table_name=name)

    result = trainer.predict(table, snapshot=snapshot, output_config=output,
                             binary_classification_threshold=.5, num_workers=2)

    assert len(result.files) >= 2
    frame = result.frame
    written = client.get_table(f'{PROJECT}.{DATASET}.{name}')
    assert rows(client, name) == len(frame) == result.num_rows
    assert sorted(f.name for f in written.schema) == sorted(frame.columns)
    loaded = client.query(f'SELECT ENTITY FROM `{PROJECT}.{DATASET}.{name}`')
    assert sorted(row.ENTITY for row in loaded) == sorted(frame['ENTITY'])

    result.export(
        PredictionOutputConfig(connector(staged),
                               artifact_type=PredictionArtifactType.PREDICTIONS,
                               table_name=name, write_mode=WriteMode.APPEND))
    assert rows(client, name) == 2 * len(frame)

    result.export(output)
    assert rows(client, name) == len(frame)
    assert not staged_objects()


def test_embeddings_land_as_repeated_float(cpu_fitted, client):
    trainer, table, snapshot = cpu_fitted
    name = f'emb_{uuid.uuid4().hex[:6]}'
    result = trainer.predict(
        table, snapshot=snapshot, binary_classification_threshold=.5,
        output_config=PredictionOutputConfig(
            connector(False), artifact_type=PredictionArtifactType.EMBEDDINGS,
            table_name=name))

    written = client.get_table(f'{PROJECT}.{DATASET}.{name}')
    assert rows(client, name) == result.num_rows
    embedding = [f for f in written.schema if f.mode == 'REPEATED']
    assert [(f.field_type, f.mode)
            for f in embedding] == [('FLOAT', 'REPEATED')]


def test_missing_dataset_fails_before_scoring(cpu_fitted, client):
    trainer, table, snapshot = cpu_fitted
    runs = trainer.scratch_dir / 'runs'
    before = set(runs.glob('*-predict-*'))
    output = PredictionOutputConfig(
        connector(False, dataset=f'missing_{RUN}'),
        artifact_type=PredictionArtifactType.PREDICTIONS, table_name='preds')

    with pytest.raises(SourceNotFoundError, match='does not exist'):
        trainer.predict(table, snapshot=snapshot, output_config=output,
                        binary_classification_threshold=.5)
    assert set(runs.glob('*-predict-*')) == before
