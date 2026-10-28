# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MLOps seams take any bundle, run and predictor, not only the
Trainer's."""

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.artifacts import (
    PROVENANCE_FILE,
    add_bundle_file,
    bundle_id_from_manifest,
)
from kumoapi.encoder import TextEmbedding
from kumoapi.plugin import ModelRegistryLocation, PublishedModel
from kumoapi.typing import Stype

import kumo.engine as kumo
from kumo.engine import _deps
from kumo.engine._deps import Dependencies, FileAsset, MissingDependencyError
from kumo.engine._manifest import ModelManifest, read_manifest, write_manifest
from kumo.engine._predictor import load_predictor
from kumo.engine._tracking import RunSpec, TrackingSession, run_record
from kumo.engine.tracking import Event
from kumo.engine.trainer._spec import MANIFEST_FILE, PROVENANCE_ROLE
from kumo.engine.trainer.trainer import _store_and_register


class ToyPredictor:
    def __init__(self, weight):
        self.weight = weight

    @classmethod
    def load(cls, bundle_dir):
        return cls(float((Path(bundle_dir) / 'weight.txt').read_text()))

    def predict(self, rows, snapshot, params):
        return pd.DataFrame({'ENTITY': rows['ENTITY'],
                             'SCORE': rows['ENTITY'] * self.weight})


def _toy_bundle(root, predictor=f'{__name__}:ToyPredictor'):
    root.mkdir(parents=True)
    (root / 'weight.txt').write_text('2.0')
    (root / MANIFEST_FILE).write_text(json.dumps({'artifacts': []}))
    add_bundle_file(root, 'weights', 'weight.txt')
    manifest = ModelManifest(id=bundle_id_from_manifest(root),
                             dependencies=[_deps.predictor(predictor)])
    write_manifest(root / PROVENANCE_FILE, manifest)
    add_bundle_file(root, PROVENANCE_ROLE, PROVENANCE_FILE)
    return root, manifest


class _Tracker:
    def __init__(self):
        self.calls = []

    def start(self, record):
        self.calls.append(('start', record))

    def on_trial(self, record, event):
        self.calls.append(('on_trial', record))

    def log_model(self, bundle_dir, manifest, snapshot, registry):
        self.calls.append(('log_model', (bundle_dir, manifest.id, snapshot)))
        return registry and PublishedModel(name=registry.model_name,
                                           version='2', uri='fake://TOY/2')

    def finish(self, record, error):
        self.calls.append(('finish', record))


def test_any_run_is_tracked_from_its_spec():
    tracker = _Tracker()
    session = TrackingSession(tracker)
    spec = RunSpec(kind='custom', params={'lr': 0.1},
                   inputs={'users': {'connector': 'S3'}})
    session.start(lambda: spec)
    session.on_event(Event('trial_started', 0.0, trial=1,
                           params={'attempt': 1}))
    session.finish(lambda: replace(spec, metrics={'auc': 0.9},
                                   parents={'bundle_id': 'b'}), None)
    session._worker.join(5)
    started, trial, finished = (record for _, record in tracker.calls)
    assert started == run_record(spec)
    assert trial.name == 'trial-1-attempt-1'
    assert trial.tags == {'kumo.kind': 'trial'}
    assert finished.metrics == {'auc': 0.9}
    assert finished.tags == {'kumo.kind': 'custom', 'kumo.bundle_id': 'b'}


class _RegistryTracker(_Tracker):
    log_model = None

    def __init__(self, connector):
        super().__init__()
        self.connector = connector


@pytest.mark.parametrize('logs', [False, True])
def test_any_bundle_is_stored_and_registered_by_its_tracker(tmp_path, logs):
    bundle, manifest = _toy_bundle(tmp_path / 'bundle')
    published = []
    sink = SimpleNamespace(
        publish_model=lambda connector, bundle_dir, **kw: published.append(
            (bundle_dir, kw)) or PublishedModel(name='TOY', version='1',
                                                uri='fake://TOY/1'))
    tracker = (_Tracker() if logs else _RegistryTracker(
        SimpleNamespace(sink_backend=sink)))
    settings = kumo.MLOps(str(tmp_path / 'root'), tracking=tracker,
                          registry=ModelRegistryLocation('TOY'))
    session = TrackingSession(tracker)
    session.start(lambda: RunSpec(kind='custom'))

    registered, error = _store_and_register(
        bundle, manifest, settings, session=session, snapshot='snap',
        metrics={'auc': 0.9})

    assert error is None
    stored = tmp_path / 'root' / 'models' / manifest.id
    assert (stored / 'weight.txt').read_text() == '2.0'
    if logs:
        assert registered.version == '2' and published == []
        assert ('log_model', (bundle, manifest.id, 'snap')) in tracker.calls
    else:
        assert registered.version == '1'
        assert published == [(str(bundle), {
            'path': None, 'location': ModelRegistryLocation('TOY'),
            'metadata': {'kumo_bundle_id': manifest.id},
            'metrics': {'auc': 0.9}
        })]
    session.finish(None, None)


def test_a_bundle_is_scored_by_the_predictor_it_names(tmp_path):
    pytest.importorskip('mlflow')
    from kumo.engine.integrations._mlflow_model import KumoModel

    bundle, _ = _toy_bundle(tmp_path / 'toy')
    rows = pd.DataFrame({'ENTITY': [1, 2]})
    assert load_predictor(bundle).predict(rows, None, {})['SCORE'].tolist(
    ) == [2.0, 4.0]
    model = KumoModel()
    model.load_context(SimpleNamespace(artifacts={'bundle': str(bundle)},
                                       model_config={'columns': []}))
    model._snapshots['s'] = 'snapshot'
    assert model.predict(None, rows, {'snapshot': 's'})['SCORE'].tolist() == [
        2.0, 4.0
    ]

    unknown, _ = _toy_bundle(tmp_path / 'unknown', 'nowhere.at_all:Model')
    with pytest.raises(ValueError, match="names predictor 'nowhere.at_all:"
                       "Model', which cannot be imported here"):
        load_predictor(unknown)


def test_a_file_asset_is_stored_once_and_resolved_by_its_hash(tmp_path):
    (tmp_path / 'vocab').mkdir()
    (tmp_path / 'vocab' / 'words.txt').write_text('a b c')
    asset = FileAsset('toy', 'vocab', tmp_path / 'vocab', version='1')
    settings = kumo.MLOps(str(tmp_path / 'root'))
    recorded = Dependencies(settings=settings)
    record = recorded.require(asset)
    stored = tmp_path / 'root' / 'assets' / 'toy' / asset.fingerprint()
    assert record.location == str(stored)
    assert read_manifest(stored / PROVENANCE_FILE, _deps.AssetManifest).name \
        == 'vocab'

    resolved = Dependencies(recorded.records).resolve('toy', 'vocab')
    assert (resolved / 'words.txt').read_text() == 'a b c'

    (stored / 'words.txt').write_text('changed')
    with pytest.raises(MissingDependencyError,
                       match=rf"needs toy 'vocab' \(sha256 {record.sha256}\)"):
        Dependencies(recorded.records).resolve('toy', 'vocab')


def test_a_manifest_without_dependencies_uses_the_trainer_predictor():
    old = ModelManifest.from_dict({'kind': 'model', 'schema_version': 1,
                                   'id': 'b'})
    assert old.dependencies is None
    assert Dependencies.of(old).records == []


WORDS = ['red', 'green', 'blue', 'cyan', 'gold', 'gray', 'pink']


def _vectors(root):
    root.mkdir()
    rng = np.random.default_rng(0)
    pd.DataFrame({'index': pd.Series(WORDS, dtype=object)}).to_parquet(
        root / 'glove_tiny_index.parquet')
    np.save(root / 'glove_tiny_8d_emb.npy',
            np.asfortranarray(rng.normal(size=(len(WORDS), 8)).astype(
                np.float32)))
    return root


def _users(root):
    rng = np.random.default_rng(1)
    root.mkdir()
    pq.write_table(
        pa.table({
            'id': pa.array(range(120), type=pa.int64()),
            'bio': [' '.join(rng.choice(WORDS, 6)) for _ in range(120)],
            'label': [bool(i % 2) if i < 100 else None for i in range(120)],
        }), root / 'users.parquet')
    return kumo.Table.from_source_table(
        kumo.FileConnector(root_dir=str(root))['users'],
        primary_key='id').with_column('bio', stype=Stype.text)


def test_a_model_serves_from_its_bundle_snapshot_and_assets_alone(
        tmp_path, monkeypatch):
    monkeypatch.setattr('kumo.engine._scratch._root', tmp_path / 'scratch')
    graph = kumo.Graph([_users(tmp_path / 'source')])
    snapshot = graph.snapshot()
    query = kumo.PredictiveQuery(graph,
                                 'PREDICT users.label FOR EACH users.id')
    plan = query.suggest_model_plan(snapshot)
    plan.training_job.num_experiments = 1
    plan.training_job.disable_explain = True
    plan.optimization.max_epochs = 1
    plan.column_processing.encoder_overrides = {
        'users.bio': TextEmbedding('colors')
    }
    store = kumo.EmbeddingModelStore()
    store.add('colors', kumo.GloVeVectors(_vectors(tmp_path / 'vectors')))
    trainer = kumo.Trainer(plan, embedding_models=store, accelerator='cpu')
    root = tmp_path / 'root'
    result = trainer.fit(query.generate_training_table(snapshot),
                         snapshot=snapshot, mlops=kumo.MLOps(str(root)))
    location = snapshot.export(
        kumo.SnapshotOutputConfig(
            kumo.FileConnector(root_dir=str(tmp_path / 'exports')), 'snap'))
    expected = trainer.predict(
        snapshot, binary_classification_threshold=.5).frame.sort_values(
            'ENTITY').reset_index(drop=True)
    expected.to_parquet(tmp_path / 'expected.parquet')
    dependency, = Dependencies.of(result.manifest).assets('text_encoder')
    assert dependency.name == 'colors' and dependency.packing == 'external'
    assert Path(dependency.location).parent == root / 'assets' / 'text_encoder'

    serve = tmp_path / 'serve'
    (serve / 'home').mkdir(parents=True)
    script = textwrap.dedent(f"""
        import pandas as pd
        import kumo.engine as kumo
        snapshot = kumo.GraphSnapshot.load_from_connector(
            kumo.FileConnector(root_dir={str(tmp_path / 'exports')!r}),
            path={location!r})
        trainer = kumo.Trainer.load({str(root / 'models' / result.id)!r},
                                    accelerator='cpu')
        scored = trainer.predict(
            snapshot, binary_classification_threshold=.5).frame.sort_values(
                'ENTITY').reset_index(drop=True)
        pd.testing.assert_frame_equal(
            scored, pd.read_parquet({str(tmp_path / 'expected.parquet')!r}))
    """)
    env = {**os.environ, 'HOME': str(serve / 'home'), 'CUDA_VISIBLE_DEVICES':
           '', 'HTTPS_PROXY': 'http://127.0.0.1:9',
           'HTTP_PROXY': 'http://127.0.0.1:9'}
    for name in ('XDG_CACHE_HOME', 'NO_PROXY', 'no_proxy'):
        env.pop(name, None)
    done = subprocess.run([sys.executable, '-c', script], cwd=serve, env=env,
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-3000:]

    (root / 'assets').rename(tmp_path / 'moved')
    done = subprocess.run([sys.executable, '-c', script], cwd=serve, env=env,
                          capture_output=True, text=True, timeout=600)
    assert "MissingDependencyError: The model needs text_encoder 'colors'" \
        in done.stderr
