# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
import shutil
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest
from kumo_core.artifacts import add_bundle_file
from kumo_core.types import TaskType
from kumoapi.model_plan import ModelPlan, TrainingTableGenerationPlan

import kumo.engine as kumo
from kumo.engine._manifest import (
    ModelManifest,
    ModelParents,
    read_manifest,
    write_manifest,
)
from kumo.engine.trainer._provenance import embedded_model
from kumo.engine.trainer._spec import MANIFEST_FILE
from kumo.engine.trainer.trainer import TrainingResult, _store_and_register


def test_fit_records_a_self_contained_manifest(fitted):
    trainer, _, snapshot = fitted
    result = trainer.result
    manifest = result.manifest
    assert manifest.kind == 'model'
    assert re.fullmatch('[0-9a-f]{64}', trainer.result.id)
    assert trainer.result.id == result.id == manifest.id
    assert trainer.result.manifest == manifest
    table = manifest.parents.training_table
    assert table.kind == 'training_table' and table.id.startswith('tt-')
    assert table.parents.snapshot == snapshot.manifest
    assert manifest.parents.warm_start is None
    assert manifest.fit_mode == 'refit_trainval'
    assert (result.bundle_dir / '_provenance.json').is_file()
    loaded = kumo.Trainer.load(result.bundle_dir)
    assert loaded.result.id == manifest.id
    assert loaded.result.manifest == manifest


def test_manifest_round_trips_unchanged(fitted, tmp_path):
    trainer, _, _ = fitted
    newer = json.loads(
        (trainer.result.bundle_dir / '_provenance.json').read_text())
    newer['future'] = True
    newer['fit']['gpu_seconds'] = 1.5
    source = tmp_path / 'newer.json'
    source.write_text(json.dumps(newer, indent=2, sort_keys=True) + '\n')

    manifest = read_manifest(source, ModelManifest)
    write_manifest(tmp_path / 'out.json', manifest)

    assert (tmp_path / 'out.json').read_bytes() == source.read_bytes()
    assert manifest.extra == {'future': True}
    assert manifest.fit.extra == {'gpu_seconds': 1.5}


def test_bundle_without_a_manifest_loads_and_predicts_identically(
        fitted, tmp_path):
    trainer, prediction_table, snapshot = fitted
    result = trainer.result
    old = tmp_path / 'old'
    shutil.copytree(result.bundle_dir, old)
    manifest = json.loads((old / MANIFEST_FILE).read_text())
    manifest['artifacts'] = [
        entry for entry in manifest['artifacts']
        if entry['role'] != 'provenance'
    ]
    (old / MANIFEST_FILE).write_text(json.dumps(manifest))
    (old / '_provenance.json').unlink()
    restored = kumo.Trainer.load(old)
    assert restored.result.id == result.bundle_id
    assert restored.result.manifest is None

    def predict(model):
        return model.predict(
            prediction_table, snapshot=snapshot,
            binary_classification_threshold=.5).frame.sort_values(
                'ENTITY').reset_index(drop=True)

    pd.testing.assert_frame_equal(predict(restored), predict(trainer))


def test_each_fit_returns_a_new_result(fitted):
    trainer, _, _ = fitted
    first = trainer.result
    second = trainer._finish_fit(
        TrainingResult({}, dict(first.artifact_paths), first.bundle_dir),
        predictive_query='PREDICT x', snapshot_id='snap', export_config=None,
        training_table_generation=TrainingTableGenerationPlan())
    assert second is not first and trainer.result is second


def _finish_stub_fit(bundle, trainer, warm_start=None, settings=None,
                     holdout_path=None):
    (bundle / 'model').mkdir(parents=True)
    (bundle / 'model' / 'config.json').write_text('{}')
    (bundle / MANIFEST_FILE).write_text(json.dumps({'artifacts': []}))
    add_bundle_file(bundle, 'trainer_config', 'model/config.json')
    table = SimpleNamespace(tt_id='tt-raw', id='tt-1', manifest=None)
    result = trainer._finish_fit(
        TrainingResult({}, {}, bundle), predictive_query='PREDICT x',
        snapshot_id='snap', export_config=None,
        training_table_generation=TrainingTableGenerationPlan(),
        provenance=dict(train_table=table, warm_start=warm_start,
                        started_at=datetime.now(timezone.utc), fit_mode=None,
                        task_type=TaskType.BINARY_CLASSIFICATION,
                        holdout_path=holdout_path, holdout_truncated=False))
    _store_and_register(result.bundle_dir, result.manifest, settings)
    return result


def test_missing_parents_are_stubs_and_retrain_chains_stay_flat(tmp_path):
    first = _finish_stub_fit(tmp_path / 'one', kumo.Trainer(ModelPlan()))
    assert first.manifest.parents.training_table.to_dict() == {
        'kind': 'training_table', 'schema_version': 1, 'id': 'tt-1',
        'created_at': None, 'environment': None, 'reconstructed': True,
        'parents': None, 'query': None, 'generation_plan': None,
        'task_type': None, 'split_seed': None}
    second = _finish_stub_fit(tmp_path / 'two', kumo.Trainer(ModelPlan()),
                              warm_start=first.manifest)
    third = _finish_stub_fit(tmp_path / 'three', kumo.Trainer(ModelPlan()),
                             warm_start=second.manifest)
    parent = third.manifest.parents.warm_start
    assert parent == embedded_model(second.manifest)
    assert parent.parents.warm_start == ModelManifest(id=first.id,
                                                      reconstructed=True)
    assert parent.parents == ModelParents(
        training_table=second.manifest.parents.training_table,
        warm_start=ModelManifest(id=first.id, reconstructed=True))


def test_a_mlops_block_stores_each_model_once(tmp_path):
    cfg = kumo.MLOps(artifact_root=str(tmp_path / 'artifacts'))
    result = _finish_stub_fit(tmp_path / 'bundle', kumo.Trainer(ModelPlan()),
                              settings=cfg)
    stored = tmp_path / 'artifacts' / 'models' / result.id
    assert kumo.Trainer.load(stored).result.id == result.id
    loaded = kumo.Trainer.load_from_connector(cfg, path=f'models/{result.id}')
    assert loaded.result.id == result.id
    assert loaded.result.snapshot_id == result.snapshot_id
    (stored / 'model' / 'config.json').write_text('{"damaged": 1}')
    again = _finish_stub_fit(tmp_path / 'again', kumo.Trainer(ModelPlan()),
                             settings=cfg)
    assert again.id == result.id
    assert (stored / 'model' / 'config.json').read_text() == '{}'
    off = kumo.MLOps(artifact_root=str(tmp_path / 'off'), store={'snapshot'})
    _finish_stub_fit(tmp_path / 'off-bundle', kumo.Trainer(ModelPlan()),
                     settings=off)
    assert not (tmp_path / 'off').exists()


def test_manifest_failure_warns_and_still_publishes(tmp_path, monkeypatch):
    from kumo.engine.trainer import _provenance

    def fail(*args, **kwargs):
        raise RuntimeError('unreadable manifest')

    monkeypatch.setattr(_provenance, 'write_manifest', fail)
    trainer = kumo.Trainer(ModelPlan())
    with pytest.warns(UserWarning, match='unreadable manifest'):
        result = _finish_stub_fit(tmp_path / 'bundle', trainer)
    assert result.manifest is None
    assert re.fullmatch('[0-9a-f]{64}', result.bundle_id)
    assert trainer.result is result
