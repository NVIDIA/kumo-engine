# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ADLS MLOps lifecycle; opt into Azure with KUMO_MLOPS_LIVE_ADLS_ROOT."""

import hashlib
import os
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from kumo_core.sources.object_store_access import ObjectStoreAccess
from kumoapi.model_plan import ModelPlan
from kumoapi.typing import Stype
from pyarrow.fs import FileType, LocalFileSystem, SubTreeFileSystem

import kumo.engine as kumo
from kumo.engine._deps import (
    Dependencies,
    FileAsset,
    asset_locations,
    connector_roots,
)
from kumo.engine._manifest import ModelManifest
from kumo.engine._scratch import get_scratch_dir
from kumo.engine.connector import connector_for_root, file_store


@pytest.fixture(params=['local', 'azure'])
def adls_store(request, monkeypatch, tmp_path):
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    if request.param == 'azure':
        base = os.environ.get('KUMO_MLOPS_LIVE_ADLS_ROOT')
        if not base:
            pytest.skip(
                'live Azure validation requires KUMO_MLOPS_LIVE_ADLS_ROOT '
                'and credentials; Azure was not exercised')
    else:
        base = 'abfss://artifacts@companylake.dfs.core.windows.net/tests'
        directory = tmp_path / 'storage'
        directory.mkdir()
        filesystem = SubTreeFileSystem(str(directory), LocalFileSystem())

        def identity(path):
            with filesystem.open_input_file(path) as stream:
                return (hashlib.sha256(stream.read()).hexdigest(), )

        monkeypatch.setattr(
            'kumo_adls.backend.ADLSSessions._source_access', lambda *args:
            ObjectStoreAccess(filesystem, object_identity=identity))
    uri = f'{base.rstrip("/")}/mlops-{uuid4().hex}'
    connector = connector_for_root(uri)
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=tmp_path / 'scratch')
    try:
        yield connector, uri
    finally:
        kumo.configure(scratch_dir=previous)
        cleanup_filesystem, cleanup_path = file_store(connector, uri)
        if cleanup_filesystem.get_file_info(
                cleanup_path).type == FileType.Directory:
            cleanup_filesystem.delete_dir(cleanup_path)


def test_adls_files_models_and_assets_roundtrip(adls_store, tmp_path):
    connector, uri = adls_store
    settings = kumo.MLOps((connector, uri), on_error='raise')
    vocabulary = tmp_path / 'vocabulary'
    vocabulary.mkdir()
    (vocabulary / 'words.txt').write_text('one two three')
    record = Dependencies(settings=settings).require(
        FileAsset('toy', 'words', vocabulary))
    manifest = ModelManifest(id='asset-check', dependencies=[record])
    with asset_locations(connector_roots(manifest, connector)):
        fetched = Dependencies.of(manifest).resolve('toy', 'words')
    assert (fetched / 'words.txt').read_text() == 'one two three'

    bundle = tmp_path / 'bundle'
    (bundle / 'nested').mkdir(parents=True)
    (bundle / 'nested' / 'weights.bin').write_bytes(b'weights')
    root = f'{uri}/model'
    models = connector_for_root(root)
    published = models.sink_backend.publish_model(models, str(bundle),
                                                  path=root)
    restored = tmp_path / 'fetched'
    models.sink_backend.fetch_model(models, str(restored), path=published.uri)
    assert (restored / 'nested' / 'weights.bin').read_bytes() == b'weights'


def test_adls_artifact_lifecycle(adls_store, tmp_path):
    pytest.importorskip('mlflow')
    from kumo.engine.integrations._mlflow_model import KumoModel

    connector, uri = adls_store
    sources = connector_for_root(f'{uri}/sources')
    rng = np.random.default_rng(0)
    sources.sink_backend.write_table(sources, 'users',
                                     pd.DataFrame({'ID': np.arange(50)}))
    events = pd.DataFrame({
        'EID':
        np.arange(2000),
        'USER_ID':
        rng.integers(0, 50, 2000),
        'TS':
        pd.Timestamp('2024-01-01') +
        pd.to_timedelta(rng.integers(0, 120, 2000), unit='D'),
    })
    sources.sink_backend.write_table(sources, 'events', events)
    graph = kumo.Graph(
        tables={
            'users':
            kumo.Table.from_source_table(sources['users.parquet'],
                                         primary_key='ID'),
            'events':
            kumo.Table.from_source_table(sources['events.parquet'],
                                         primary_key='EID',
                                         time_column='TS').with_column(
                                             'USER_ID', stype=Stype.ID),
        }, edges=[
            kumo.Edge(src_table='events', fkey='USER_ID', dst_table='users')
        ])
    artifacts_uri = f'{uri}/artifacts'
    settings = kumo.MLOps(artifacts_uri, on_error='raise')
    plan = ModelPlan()
    plan.training_job.disable_explain = True
    plan.optimization.max_epochs = 1
    plan.optimization.max_steps_per_epoch = 2
    trainer = kumo.Trainer(plan, accelerator='cpu')
    with kumo.mlops(settings):
        snapshot = graph.snapshot()
        query = kumo.PredictiveQuery(
            graph=graph,
            query='PREDICT COUNT(events.*, 0, 7, days) > 0 FOR EACH users.ID')
        training = query.generate_training_table(snapshot)
        fitted = trainer.fit(training, snapshot=snapshot)
        run = trainer.predict(
            query.generate_prediction_table(snapshot), snapshot=snapshot,
            binary_classification_threshold=0.5,
            output_config=kumo.PredictionOutputConfig(
                settings.root.connector, path='direct-predictions/'))
    assert all(source.connector == 'ADLS' and source.location.startswith(uri)
               for source in snapshot.manifest.parents.source_tables.values())
    assert run.num_rows > 0
    assert all(output.destination.connector == 'ADLS'
               and output.destination.location.startswith(artifacts_uri)
               for output in run.manifest.outputs)

    # The same directory also works as an exact-root connector/path pair.
    explicit = connector_for_root(artifacts_uri)
    settings = kumo.MLOps((explicit, artifacts_uri), on_error='raise')
    kumo.configure(scratch_dir=tmp_path / 'reload')
    restored = kumo.GraphSnapshot.load_from_connector(settings, id=snapshot.id)
    assert restored.id == snapshot.id
    assert restored.index_path.is_relative_to(tmp_path / 'reload')
    assert restored.index_path != snapshot.index_path
    loaded = kumo.Trainer.load_from_connector(settings,
                                              path=f'models/{fitted.id}',
                                              accelerator='cpu')
    stored = kumo.PredictionResult.load_from_connector(
        settings, path=f'predictions/{run.id}')
    pd.testing.assert_frame_equal(stored.frame, run.frame)
    assert loaded.score_drift(stored, snapshot=restored,
                              training_snapshot=restored).psi is not None

    # A newer source version supplies complete labels for the stored run.
    future = pd.DataFrame({
        'EID':
        np.arange(2000, 2051),
        'USER_ID': [*range(50), 0],
        'TS': [pd.Timestamp('2024-05-01')] * 50 + [pd.Timestamp('2024-05-10')],
    })
    sources.sink_backend.write_table(
        sources, 'events', pd.concat([events, future], ignore_index=True))
    labels = graph.snapshot()
    assert labels.id != snapshot.id
    evaluated = loaded.evaluate(stored, labels_snapshot=labels)
    assert evaluated.rows == run.num_rows
    assert evaluated.matured and evaluated.labelled_fraction == 1.0

    # Store an external asset and fetch it through the supplied ADLS connector.
    vocabulary = tmp_path / 'vocabulary'
    vocabulary.mkdir()
    (vocabulary / 'words.txt').write_text('one two three')
    record = Dependencies(settings=settings).require(
        FileAsset('toy', 'words', vocabulary))
    manifest = ModelManifest(id='asset-check', dependencies=[record])
    with asset_locations(connector_roots(manifest, explicit)):
        fetched = Dependencies.of(manifest).resolve('toy', 'words')
    assert (fetched / 'words.txt').read_text() == 'one two three'

    # A pyfunc opens the exported snapshot through assets_from, without retraining.
    kumo.configure(scratch_dir=tmp_path / 'serving')
    model = KumoModel()
    model.load_context(
        SimpleNamespace(
            artifacts={'bundle': str(fitted.bundle_dir)}, model_config={
                'columns': [],
                'assets_from': explicit,
                'snapshot': settings.snapshot_uri(snapshot)
            }))
    assert model._snapshots[settings.snapshot_uri(snapshot)].id == snapshot.id
    assert model._snapshots[settings.snapshot_uri(
        snapshot)].index_path.is_relative_to(tmp_path / 'serving')
