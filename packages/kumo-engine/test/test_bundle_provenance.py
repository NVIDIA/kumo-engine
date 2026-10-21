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
from kumoapi.model_plan import ModelPlan, TrainingTableGenerationPlan

import kumo.engine as kumo
from kumo.engine.trainer._spec import MANIFEST_FILE
from kumo.engine.trainer.trainer import TrainingResult


def test_fit_records_provenance(fitted):
    trainer, _, snapshot = fitted
    result = trainer._training_result
    provenance = result.provenance
    parents = provenance['parents']
    assert (parents['snapshot'], parents['warm_start']) == (snapshot.id, None)
    assert parents['training_table']
    assert provenance['training_table'] == {
        'source': 'pql',
        'snapshot': snapshot.id
    }
    assert provenance['source_tables'] == (
        snapshot.manifest['parents']['source_tables'])
    assert provenance['fit_mode'] == 'refit_trainval'
    assert re.fullmatch('[0-9a-f]{64}', trainer.bundle_id)
    assert trainer.bundle_id == result.bundle_id


def test_bundle_without_provenance_loads_and_predicts_identically(
        fitted, tmp_path):
    trainer, prediction_table, snapshot = fitted
    result = trainer._training_result
    loaded = kumo.Trainer.load(result.bundle_dir)
    assert (loaded.bundle_id, loaded.provenance) == (result.bundle_id,
                                                     result.provenance)

    old = tmp_path / 'old'
    shutil.copytree(result.bundle_dir, old)
    manifest = json.loads((old / MANIFEST_FILE).read_text())
    manifest['artifacts'] = [
        entry for entry in manifest['artifacts']
        if entry['role'] != 'provenance'
    ]
    (old / MANIFEST_FILE).write_text(json.dumps(manifest))
    (old / 'provenance.json').unlink()
    restored = kumo.Trainer.load(old)
    assert restored.bundle_id == result.bundle_id
    assert restored.provenance is None

    def predict(model):
        return model.predict(
            prediction_table, snapshot=snapshot,
            binary_classification_threshold=.5).frame.sort_values(
                'ENTITY').reset_index(drop=True)

    pd.testing.assert_frame_equal(predict(restored), predict(trainer))


def _finish_stub_fit(bundle, trainer, manifest=None):
    (bundle / 'model').mkdir(parents=True)
    (bundle / 'model' / 'config.json').write_text('{}')
    (bundle / MANIFEST_FILE).write_text(json.dumps({'artifacts': []}))
    add_bundle_file(bundle, 'trainer_config', 'model/config.json')
    table = SimpleNamespace(tt_id='tt-1', manifest=manifest)
    return trainer._finish_fit(
        TrainingResult({}, {}, bundle), predictive_query='PREDICT x',
        snapshot_id='snap', export_config=None,
        training_table_generation=TrainingTableGenerationPlan(),
        provenance=dict(train_table=table, snapshot_id='snap',
                        warm_start_bundle_id=None,
                        started_at=datetime.now(timezone.utc),
                        fit_mode=None))


def test_provenance_records_custom_or_missing_table_source(tmp_path):
    source_table = {
        'connector': 'S3',
        'location': 's3://bucket/exports/tt',
        'version': '17'
    }
    manifest = {
        'source': 'custom',
        'parents': {
            'snapshot': 'snap',
            'source_table': source_table
        },
    }
    custom = _finish_stub_fit(tmp_path / 'custom', kumo.Trainer(ModelPlan()),
                              manifest)
    assert custom.provenance['training_table'] == {
        'source': 'custom',
        'snapshot': 'snap',
        **source_table,
    }
    older = _finish_stub_fit(tmp_path / 'older', kumo.Trainer(ModelPlan()))
    assert older.provenance['training_table'] == {
        'source': None,
        'snapshot': None
    }


def test_provenance_failure_warns_and_still_publishes(tmp_path, monkeypatch):
    from kumo.engine.trainer import _provenance

    def fail(*args):
        raise RuntimeError('unreadable manifest')

    monkeypatch.setattr(_provenance, '_training_table', fail)
    trainer = kumo.Trainer(ModelPlan())
    with pytest.warns(UserWarning, match='unreadable manifest'):
        result = _finish_stub_fit(tmp_path / 'bundle', trainer)
    assert result.provenance is None
    assert re.fullmatch('[0-9a-f]{64}', result.bundle_id)
    assert trainer._training_result is result
