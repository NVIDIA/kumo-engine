# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ADLS MLOps lifecycle; opt into Azure with KUMO_MLOPS_LIVE_ADLS_ROOT."""

import hashlib
import os
import runpy
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from kumo_core.sources.object_store_access import ObjectStoreAccess
from kumoapi.model_plan import ModelPlan
from kumoapi.typing import Stype
from pyarrow.fs import (
    FileSelector,
    FileType,
    LocalFileSystem,
    SubTreeFileSystem,
)

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

lifecycle = SimpleNamespace(
    **runpy.run_path(str(Path(__file__).with_name('wheel_mlops_lifecycle.py'))))


def local_storage(monkeypatch, directory):
    filesystem = SubTreeFileSystem(str(directory), LocalFileSystem())

    def identity(path):
        with filesystem.open_input_file(path) as stream:
            return (hashlib.sha256(stream.read()).hexdigest(), )

    monkeypatch.setattr(
        'kumo_adls.backend.ADLSSessions._source_access',
        lambda *args: ObjectStoreAccess(filesystem, object_identity=identity))


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
        local_storage(monkeypatch, directory)
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

    from kumo_core.artifact_export import ArtifactDestinationConflict

    connector.sink_backend.preflight_write_files(connector, 'predictions/')
    connector.sink_backend.write_table(
        connector, 'predictions/', pd.DataFrame({'id': [1]}))
    with pytest.raises(ArtifactDestinationConflict):
        connector.sink_backend.preflight_write_files(connector, 'predictions/')


def test_adls_artifact_lifecycle(adls_store, tmp_path):
    pytest.importorskip('mlflow')
    _, uri = adls_store
    for phase in ('fit', 'reload'):
        command = [
            sys.executable, '-I',
            str(Path(__file__).resolve()), phase,
            str(tmp_path), uri,
            str(tmp_path / 'storage') if (tmp_path / 'storage').is_dir() else ''
        ]
        with (tmp_path / f'{phase}.log').open('w') as log:
            status = lifecycle.execute(command, tmp_path, log, timeout=600)
        assert status == 0, (tmp_path / f'{phase}.log').read_text()
        if phase == 'fit':
            shutil.rmtree(tmp_path / 'scratch-fit')
            shutil.rmtree(tmp_path / 'cache-fit')
    fit = lifecycle.read(tmp_path / 'fit.json')
    assert fit['pid'] != lifecycle.read(tmp_path / 'reload.json')['pid']


def events():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        'EID':
        np.arange(2000),
        'USER_ID':
        rng.integers(0, 50, 2000),
        'TS':
        pd.Timestamp('2024-01-01') +
        pd.to_timedelta(rng.integers(0, 120, 2000), unit='D'),
    })


def graph(sources):
    return kumo.Graph(
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


def prediction_frame(frame):
    frame = frame.sort_values('ENTITY').reset_index(drop=True)
    assert list(frame) == ['ENTITY', 'TARGET_PRED', 'False_PROB', 'True_PROB']
    assert frame.ENTITY.tolist() == list(range(50))
    assert np.isfinite(frame[['False_PROB', 'True_PROB']]).all().all()
    return frame


def check_output(connector, path, run):
    filesystem, directory = file_store(connector, path)
    paths = [
        item.path for item in filesystem.get_file_info(
            FileSelector(directory, recursive=True))
        if item.type == FileType.File and item.path.endswith('.parquet')
    ]
    assert paths
    exported = pq.read_table(paths, filesystem=filesystem).to_pandas()
    pd.testing.assert_frame_equal(prediction_frame(exported),
                                  prediction_frame(run.frame), check_exact=True)


def fit(uri, _root):
    sources = connector_for_root(f'{uri}/sources')
    sources.sink_backend.write_table(sources, 'users',
                                     pd.DataFrame({'ID': np.arange(50)}))
    sources.sink_backend.write_table(sources, 'events', events())
    source_graph = graph(sources)
    artifacts_uri = f'{uri}/artifacts'
    settings = kumo.MLOps(artifacts_uri, on_error='raise')
    plan = ModelPlan()
    plan.training_job.disable_explain = True
    plan.optimization.max_epochs = 1
    plan.optimization.max_steps_per_epoch = 2
    trainer = kumo.Trainer(plan, accelerator='cpu')
    with kumo.mlops(settings):
        snapshot = source_graph.snapshot()
        query = kumo.PredictiveQuery(
            graph=source_graph,
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
    prediction_frame(run.frame)
    check_output(settings.root.connector,
                 f'{artifacts_uri}/direct-predictions/', run)
    assert all(output.destination.connector == 'ADLS'
               and output.destination.location.startswith(artifacts_uri)
               for output in run.manifest.outputs)

    return dict(snapshot=snapshot.id, model=fitted.id, prediction=run.id,
                prediction_time=run.manifest.prediction_table.anchors[0])


def reload(uri, tmp_path):
    from kumo.engine.integrations._mlflow_model import KumoModel

    state = lifecycle.read(tmp_path / 'fit.json')
    assert not (tmp_path / 'scratch-fit').exists()
    assert not (tmp_path / 'cache-fit').exists()
    artifacts_uri = f'{uri}/artifacts'
    explicit = connector_for_root(artifacts_uri)
    settings = kumo.MLOps((explicit, artifacts_uri), on_error='raise')
    restored = kumo.GraphSnapshot.load_from_connector(settings,
                                                      id=state['snapshot'])
    assert restored.id == state['snapshot']
    assert restored.index_path.is_relative_to(tmp_path / 'scratch-reload')
    loaded = kumo.Trainer.load_from_connector(settings,
                                              path=f'models/{state["model"]}',
                                              accelerator='cpu')
    assert loaded.result.id == state['model']
    stored = kumo.PredictionResult.load_from_connector(
        settings, path=f'predictions/{state["prediction"]}')
    with kumo.mlops(settings):
        run = loaded.predict(
            snapshot=restored, prediction_time=pd.Timestamp(
                state['prediction_time']).tz_localize(None).to_pydatetime(),
            binary_classification_threshold=0.5,
            output_config=kumo.PredictionOutputConfig(
                explicit, path='reloaded-predictions/'))
    assert run.bundle_id == stored.bundle_id == state['model']
    before, after = prediction_frame(stored.frame), prediction_frame(run.frame)
    scores = ['False_PROB', 'True_PROB']
    comparison = lifecycle.gpu_helpers().compare_predictions(
        before[scores], after[scores], task='binary', atol=0.005,
        before_decisions=before.TARGET_PRED.to_numpy(),
        after_decisions=after.TARGET_PRED.to_numpy())
    check_output(explicit, f'{artifacts_uri}/reloaded-predictions/', run)
    assert loaded.score_drift(stored, snapshot=restored,
                              training_snapshot=restored).psi is not None
    sources = connector_for_root(f'{uri}/sources')
    future = pd.DataFrame({
        'EID':
        np.arange(2000, 2051),
        'USER_ID': [*range(50), 0],
        'TS': [pd.Timestamp('2024-05-01')] * 50 + [pd.Timestamp('2024-05-10')],
    })
    sources.sink_backend.write_table(
        sources, 'events', pd.concat([events(), future], ignore_index=True))
    labels = graph(sources).snapshot()
    assert labels.id != restored.id
    evaluated = loaded.evaluate(stored, labels_snapshot=labels)
    assert evaluated.rows == run.num_rows
    assert evaluated.matured and evaluated.labelled_fraction == 1.0

    vocabulary = tmp_path / 'vocabulary'
    vocabulary.mkdir()
    (vocabulary / 'words.txt').write_text('one two three')
    record = Dependencies(settings=settings).require(
        FileAsset('toy', 'words', vocabulary))
    manifest = ModelManifest(id='asset-check', dependencies=[record])
    with asset_locations(connector_roots(manifest, explicit)):
        fetched = Dependencies.of(manifest).resolve('toy', 'words')
    assert (fetched / 'words.txt').read_text() == 'one two three'

    kumo.configure(scratch_dir=tmp_path / 'serving')
    model = KumoModel()
    model.load_context(
        SimpleNamespace(
            artifacts={'bundle': str(loaded.result.bundle_dir)}, model_config={
                'columns': [],
                'assets_from': explicit,
                'snapshot': settings.snapshot_uri(restored)
            }))
    assert model._snapshots[settings.snapshot_uri(restored)].id == restored.id
    assert model._snapshots[settings.snapshot_uri(
        restored)].index_path.is_relative_to(tmp_path / 'serving')

    kumo.configure(scratch_dir=tmp_path / 'fallback-serving')
    fallback = KumoModel()
    snapshot_uri = settings.snapshot_uri(restored).replace('abfss:', 'ABFSS:')
    fallback.load_context(
        SimpleNamespace(
            artifacts={'bundle': str(loaded.result.bundle_dir)}, model_config={
                'columns': [],
                'assets_from': sources,
                'snapshot': snapshot_uri
            }))
    assert fallback._snapshots[snapshot_uri].id == restored.id
    assert fallback._snapshots[snapshot_uri].index_path.is_relative_to(
        tmp_path / 'fallback-serving')
    return dict(rows=run.num_rows, comparison=comparison)


def main():
    phase, root, uri, storage = sys.argv[1:]
    root = Path(root)
    import torch

    torch.set_num_threads(2)
    scratch = root / f'scratch-{phase}'
    cache = root / f'cache-{phase}'
    assert not scratch.exists() and not cache.exists()
    cache.mkdir()
    os.environ['XDG_CACHE_HOME'] = str(cache)
    kumo.configure(scratch_dir=scratch)
    with pytest.MonkeyPatch.context() as monkeypatch:
        if storage:
            local_storage(monkeypatch, Path(storage))
        result = {'fit': fit, 'reload': reload}[phase](uri, root)
    lifecycle.write(root / f'{phase}.json',
                    dict(pid=os.getpid(), uri=uri, **result))


if __name__ == '__main__':
    main()
